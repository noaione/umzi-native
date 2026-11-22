#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <cuda.h>
#include <cudnn.h>

#define UMZI_BUILD
#include "moesr_images.h"
#include "../../common/cuda_module.h"
#include "../../common/cuda_plan_cache.h"
#include "../../runtime/binary.h"
#include "../../runtime/kernel.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <map>
#include <limits>
#include <memory>

namespace {
constexpr int channels = 64, scale = 4, full_hidden = 170, reduced_hidden = 96, threads = 256;
constexpr uint64_t default_workspace_mib = 512, max_workspace_mib = 4096;
// Bump when plan keys, descriptor semantics, or the choice format change.
constexpr uint32_t plan_schema = 1;
constexpr std::array<uint32_t, 10> configuration{3, 3, scale, 2, channels, 9, 4, 3, full_hidden, reduced_hidden};
using Clock = std::chrono::steady_clock;

void check(CUresult status, const char* operation) {
    if (status == CUDA_ERROR_OUT_OF_MEMORY)
        throw umzi::AllocationError(std::string(operation) + ": out of memory; try a smaller tile or batch");
    umzi::check_cuda(status, operation);
}

void check(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + cudnnGetErrorString(status));
}
#define CHECK(expr) check((expr), #expr)

struct ContextScope {
    explicit ContextScope(CUcontext context) { CHECK(cuCtxPushCurrent(context)); }
    ~ContextScope() {
        CUcontext previous;
        cuCtxPopCurrent(&previous);
    }
};

struct Shape {
    int n = 0, h = 0, w = 0;
    bool operator==(Shape other) const { return n == other.n && h == other.h && w == other.w; }
    Shape padded() const { return {n, (h + 1) & ~1, (w + 1) & ~1}; }
    int pixels() const { return n * h * w; }
    uint64_t rgb_bytes(bool fp16) const { return uint64_t(pixels()) * 3 * (fp16 ? 2 : 4); }
    UmziShape nchw() const { return {4, {uint32_t(n), 3, uint32_t(h), uint32_t(w)}}; }
};

Shape checked(const UmziShape& shape) {
    const auto n = shape.dimensions[0], h = shape.dimensions[2], w = shape.dimensions[3];
    if (shape.rank != 4 || shape.dimensions[1] != 3 || !n || !h || !w || n > 4096 || h > 65536 || w > 65536)
        throw std::runtime_error("MoESR requires positive RGB NCHW dimensions that fit native indexing");
    if (h == 1 || w == 1)
        throw std::runtime_error("MoESR reflection padding requires both spatial dimensions to be at least two");
    if (uint64_t(n) * ((h + 1) & ~1u) * ((w + 1) & ~1u) * (2 * full_hidden) > INT_MAX - threads)
        throw std::runtime_error("Requested MoESR shape exceeds native tensor indexing limits");
    return {int(n), int(h), int(w)};
}

// Offsets within section 3, matching the packer. Dense weights are FP16
// KRSC. Every other tensor keeps its FP32 checkpoint representation.
struct Filter { size_t weight = 0, bias = 0; };
struct Block {
    size_t norm_weight = 0, norm_bias = 0, gamma = 0;
    Filter expand, square, row, column, project;
};
struct Group {
    std::array<Block, configuration[6]> full;
    std::array<Block, configuration[7]> reduced;
    Filter down, up;
};

struct Network {
    Filter first, last;
    std::array<Group, configuration[5]> groups;
    size_t bytes = 0;

    Network() {
        first = dense(3, channels);
        for (auto& group : groups) {
            for (auto& block : group.full) block = gated(full_hidden);
            group.down = dense(channels, channels / 4);
            for (auto& block : group.reduced) block = gated(reduced_hidden);
            group.up = dense(channels, 4 * channels);
        }
        last = dense(channels, 3 * scale * scale);
    }

    size_t take(size_t size) {
        const size_t offset = (bytes + 255) & ~size_t(255);
        bytes = offset + size;
        return offset;
    }
    Filter dense(int cin, int cout) { return {take(size_t(cin) * cout * 9 * 2), take(size_t(cout) * 4)}; }
    Filter depthwise(int taps) { return {take(8 * taps * 4), take(8 * 4)}; }
    Block gated(int hidden) {
        Block block;
        block.norm_weight = take(channels * 4);
        block.norm_bias = take(channels * 4);
        block.expand = dense(channels, 2 * hidden);
        block.square = depthwise(9);
        block.row = depthwise(11);
        block.column = depthwise(11);
        block.project = dense(hidden, channels);
        block.gamma = take(channels * 4);
        return block;
    }
};

struct Convolution {
    cudnnTensorDescriptor_t input = nullptr, output = nullptr;
    cudnnFilterDescriptor_t filter = nullptr;
    cudnnConvolutionDescriptor_t convolution = nullptr;
    cudnnConvolutionFwdAlgo_t algorithm{};
    size_t workspace = 0;
    ~Convolution() {
        if (convolution) cudnnDestroyConvolutionDescriptor(convolution);
        if (filter) cudnnDestroyFilterDescriptor(filter);
        if (output) cudnnDestroyTensorDescriptor(output);
        if (input) cudnnDestroyTensorDescriptor(input);
    }
};

constexpr const char* kernel_names[] = {
    "pack_rgb_f32", "pack_rgb_f16", "input_bias", "norm_nhwc", "gate_nhwc", "gated_residual_nhwc",
    "reduce_nhwc", "expand_residual_nhwc", "add_nhwc", "output_nhwc_f32", "output_nhwc_f16",
};
enum Kernel {
    PackF32, PackF16, InputBias, Normalize, Gate, Residual, Reduce, Expand, Add, OutputF32, OutputF16, KernelCount
};
static_assert(std::size(kernel_names) == KernelCount);

class Session final : public umzi::KernelSession {
    Network model;
    bool fp16 = false, graph_enabled = false, pending = false;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUstream stream = nullptr;
    CUevent complete = nullptr;
    cudnnHandle_t cudnn = nullptr;
    std::array<CUfunction, KernelCount> functions{};
    CUdeviceptr weights = 0;

    Shape shape{}, padded{};
    CUdeviceptr stage_in = 0, stage_out = 0, scratch = 0;
    CUdeviceptr initial = 0, state = 0, reduced = 0, normalized = 0, projected = 0, gated = 0;
    CUgraph graph = nullptr;
    CUgraphExec executable = nullptr;
    std::map<umzi::PlanShape, std::unique_ptr<Convolution>> plans;
    umzi::CudaPlanCache<umzi::moesr_id, plan_schema> plan_cache;
    std::filesystem::path cache_dir;
    umzi::Digest cache_environment{}, plan_key{};
    std::string cache_environment_error;
    size_t activation_bytes = 0, workspace_bytes = 0, scratch_bytes = 0;
    std::array<double, 4> setup_ms{};
    std::array<CUevent, 2> timing{};
    int timed_plans = 0;
    uint64_t workspace_limit = default_workspace_mib << 20;

public:
    void open(const umzi::ModelPackage& package, const UmziSessionOptions& options) {
        if (package.schema != 1) throw std::runtime_error("Unsupported MoESR model payload version");
        umzi::BinaryReader architecture(package.section(1));
        for (auto value : configuration)
            if (architecture.number<uint32_t>() != value) throw std::runtime_error("Unsupported MoESR model architecture");
        if (architecture.size() || package.section(4).size() != 32)
            throw std::runtime_error("Invalid MoESR metadata");
        const auto data = package.section(3);
        if (!package.section(2).empty() || data.size() != model.bytes)
            throw std::runtime_error("Invalid MoESR weight layout");
        if (options.precision != UMZI_MIXED && options.precision != UMZI_FP16)
            throw std::runtime_error("Unsupported MoESR session precision");
        // workspace_mib (0 to 4096, default 512): cuDNN workspace that plan
        // candidates may use. The fast implicit-GEMM plans need about 45 MB per
        // 256x256 tile, so a small cap forces slow plans at larger batches.
        // Prepare keeps only the selected requirement allocated.
        for (uint32_t i = 0; i < options.kernel_option_count; ++i) {
            const auto& option = options.kernel_options[i];
            const std::string name = option.name ? option.name : "";
            if (name != "workspace_mib")
                throw std::runtime_error("Unknown MoESR option: " + name + " (expected workspace_mib)");
            if (i) throw std::runtime_error("Duplicate MoESR option: workspace_mib");
            if (option.value < 0 || uint64_t(option.value) > max_workspace_mib)
                throw std::runtime_error("MoESR option workspace_mib must be between 0 and 4096");
            workspace_limit = uint64_t(option.value) << 20;
        }
        fp16 = options.precision == UMZI_FP16;
        graph_enabled = (options.flags & UMZI_CUDA_GRAPH) != 0;
        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, options.device));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        ContextScope scope(context);
        const auto& image =
            umzi::load_kernel_module(device, module, moesr_images::embedded_cubins, moesr_images::embedded_ptx);
        for (int i = 0; i < KernelCount; ++i) CHECK(cuModuleGetFunction(&functions[i], module, kernel_names[i]));
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&complete, CU_EVENT_DISABLE_TIMING));
        for (auto& event : timing) CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        CHECK(cudnnCreate(&cudnn));
        CHECK(cuMemAlloc(&weights, data.size()));
        CHECK(cuMemcpyHtoD(weights, data.data(), data.size()));
        initialize_cache(options.cache_dir, package.digest, image);
    }

    void initialize_cache(const char* directory, const umzi::Digest& model_digest, const umzi::KernelImage& image) {
        if (!directory || !*directory) return;
        try {
            cache_dir = std::filesystem::u8path(directory);
            umzi::Sha256 hash;
            hash.add(umzi::plan_format_version);
            hash.add(umzi::moesr_id);
            hash.add(plan_schema);
            hash.add(sizeof(void*));
            hash.add(model_digest);
            hash.add(image.data, image.size);
            hash.add(static_cast<uint32_t>(fp16));
            hash.add(workspace_limit);
            CUuuid uuid{};
            CHECK(cuDeviceGetUuid(&uuid, device));
            hash.add(uuid);
            umzi::hash_driver_version(hash);
            hash.add(cudnnGetVersion());
            cache_environment = hash.finish();
        } catch (const std::exception& error) {
            cache_environment_error = error.what();
        }
    }

    void load_plan_cache() {
        try {
            if (!cache_environment_error.empty()) throw std::runtime_error(cache_environment_error);
            if (cache_dir.empty()) return;
            umzi::Sha256 hash;
            hash.add(cache_environment);
            hash.add(std::array<int, 3>{padded.n, padded.h, padded.w});
            plan_key = hash.finish();
            plan_cache.open(cache_dir, "moesr", plan_key);
        } catch (const std::exception& error) {
            plan_cache.unavailable(error);
        }
    }

    ~Session() override {
        if (!context) return;
        if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            if (pending) cuEventSynchronize(complete);
            if (stream) cuStreamSynchronize(stream);
            release_shape();
            if (cudnn) cudnnDestroy(cudnn);
            if (complete) cuEventDestroy(complete);
            for (auto event : timing)
                if (event) cuEventDestroy(event);
            if (stream) cuStreamDestroy(stream);
            if (weights) cuMemFree(weights);
            if (module) cuModuleUnload(module);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        cuDevicePrimaryCtxRelease(device);
    }

private:
    void release_shape() {
        if (executable) cuGraphExecDestroy(executable);
        if (graph) cuGraphDestroy(graph);
        executable = nullptr;
        graph = nullptr;
        plans.clear();
        plan_cache = {};
        for (auto* buffer : {&stage_in, &stage_out, &scratch, &initial, &state, &reduced, &normalized, &projected, &gated}) {
            if (*buffer) cuMemFree(*buffer);
            *buffer = 0;
        }
        shape = padded = {};
        activation_bytes = workspace_bytes = scratch_bytes = 0;
        setup_ms = {};
        timed_plans = 0;
    }

    void wait() {
        if (!pending) return;
        CHECK(cuEventSynchronize(complete));
        pending = false;
    }

    void activation(CUdeviceptr& buffer, size_t elements) {
        const auto bytes = elements * 2;
        CHECK(cuMemAlloc(&buffer, bytes));
        activation_bytes += bytes;
    }

    void make_plan(int h, int w, int cin, int cout) {
        auto plan = std::make_unique<Convolution>();
        auto& p = *plan;
        CHECK(cudnnCreateTensorDescriptor(&p.input));
        CHECK(cudnnCreateTensorDescriptor(&p.output));
        CHECK(cudnnCreateFilterDescriptor(&p.filter));
        CHECK(cudnnCreateConvolutionDescriptor(&p.convolution));
        CHECK(cudnnSetTensor4dDescriptor(p.input, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, padded.n, cin, h, w));
        CHECK(cudnnSetTensor4dDescriptor(p.output, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, padded.n, cout, h, w));
        CHECK(cudnnSetFilter4dDescriptor(p.filter, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, cout, cin, 3, 3));
        CHECK(cudnnSetConvolution2dDescriptor(p.convolution, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                             fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT));
        const umzi::PlanShape key{padded.n, h, w, cin, cout};
        if (auto saved = plan_cache.convolutions.find(key); saved != plan_cache.convolutions.end()) {
            const auto& choice = saved->second;
            const auto algorithm = static_cast<cudnnConvolutionFwdAlgo_t>(choice.algorithm);
            size_t required = 0;
            if (choice.algorithm < CUDNN_CONVOLUTION_FWD_ALGO_COUNT && choice.math <= CUDNN_FMA_MATH
                && choice.workspace <= workspace_limit
                && cudnnSetConvolutionMathType(p.convolution, static_cast<cudnnMathType_t>(choice.math))
                       == CUDNN_STATUS_SUCCESS
                && cudnnGetConvolutionForwardWorkspaceSize(cudnn, p.input, p.filter, p.convolution, p.output,
                                                           algorithm, &required) == CUDNN_STATUS_SUCCESS
                && required == choice.workspace) {
                p.algorithm = algorithm;
                p.workspace = required;
                // One run proves the saved plan executes before it is trusted.
                if (time_plan(p, 0.f) < std::numeric_limits<float>::infinity()) {
                    workspace_bytes = std::max(workspace_bytes, p.workspace);
                    plans.emplace(key, std::move(plan));
                    return;
                }
            }
        }
        plan_cache.record_tuning();
        CHECK(cudnnSetConvolutionMathType(p.convolution, CUDNN_TENSOR_OP_MATH));
        std::array<cudnnConvolutionFwdAlgoPerf_t, CUDNN_CONVOLUTION_FWD_ALGO_COUNT> choices{};
        int count = 0;
        CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, p.input, p.filter, p.convolution, p.output,
                                                    int(choices.size()), &count, choices.data()));
        // Heuristic order is not reliable for these shapes: at batch 2 the first
        // choice for one layer was a non-tensor-core kernel 30x slower. Time every
        // candidate that fits the workspace on the real buffers and keep the fastest.
        // gated holds at least cin features per pixel and projected at least cout.
        float best = std::numeric_limits<float>::infinity();
        const cudnnConvolutionFwdAlgoPerf_t* fastest = nullptr;
        for (int i = 0; i < count; ++i) {
            const auto& candidate = choices[i];
            if (candidate.status != CUDNN_STATUS_SUCCESS || candidate.memory > workspace_limit) continue;
            CHECK(cudnnSetConvolutionMathType(p.convolution, candidate.mathType));
            p.algorithm = candidate.algo;
            p.workspace = candidate.memory;
            const float elapsed = time_plan(p, best);
            if (elapsed < best) {
                best = elapsed;
                fastest = &candidate;
            }
        }
        if (!fastest)
            throw std::runtime_error("No cuDNN convolution for MoESR " + std::to_string(cin) + " to "
                                     + std::to_string(cout) + " channels within the workspace limit");
        CHECK(cudnnSetConvolutionMathType(p.convolution, fastest->mathType));
        p.algorithm = fastest->algo;
        p.workspace = fastest->memory;
        workspace_bytes = std::max(workspace_bytes, p.workspace);
        ++timed_plans;
        plan_cache.convolutions.insert_or_assign(
            key, umzi::ConvChoice{uint32_t(fastest->algo), uint32_t(fastest->mathType), p.workspace});
        plans.emplace(key, std::move(plan));
    }

    // Grow the tuning scratch to a candidate's requirement. Prepare later keeps
    // only the selected size. Returns false when the device has no room.
    bool ensure_scratch(size_t bytes) {
        if (bytes <= scratch_bytes) return true;
        CHECK(cuStreamSynchronize(stream));
        if (scratch) CHECK(cuMemFree(scratch));
        scratch = 0;
        scratch_bytes = 0;
        const auto status = cuMemAlloc(&scratch, bytes);
        if (status == CUDA_ERROR_OUT_OF_MEMORY) {
            scratch = 0;
            return false;
        }
        CHECK(status);
        scratch_bytes = bytes;
        return true;
    }

    // Time one run. Repeat three times only when it may beat the current
    // best, so very slow candidates cost one run. Infinity means failure.
    float time_plan(const Convolution& p, float best) {
        const float one = 1.f, zero = 0.f;
        const auto run = [&] {
            return cudnnConvolutionForward(cudnn, &one, p.input, reinterpret_cast<void*>(gated), p.filter,
                                           reinterpret_cast<void*>(weights), p.convolution, p.algorithm,
                                           reinterpret_cast<void*>(scratch), p.workspace, &zero, p.output,
                                           reinterpret_cast<void*>(projected)) == CUDNN_STATUS_SUCCESS;
        };
        const auto measure = [&](int repeats) {
            CHECK(cuEventRecord(timing[0], stream));
            for (int i = 0; i < repeats; ++i)
                if (!run()) {
                    CHECK(cuStreamSynchronize(stream));
                    return std::numeric_limits<float>::infinity();
                }
            CHECK(cuEventRecord(timing[1], stream));
            CHECK(cuEventSynchronize(timing[1]));
            float elapsed = 0;
            CHECK(cuEventElapsedTime(&elapsed, timing[0], timing[1]));
            return elapsed / repeats;
        };
        if (!ensure_scratch(p.workspace)) return std::numeric_limits<float>::infinity();
        const float first = measure(1);
        if (!(first < 2 * best)) return first;
        return measure(3);
    }

    void prepare(Shape requested) {
        wait();
        if (shape == requested) return;
        release_shape();
        try {
            auto start = Clock::now();
            shape = requested;
            padded = shape.padded();
            const size_t pixels = padded.pixels();
            activation(initial, pixels * channels);
            activation(state, pixels * channels);
            activation(reduced, pixels / 4 * channels);
            activation(normalized, pixels * channels);
            activation(projected, pixels * 2 * full_hidden);
            activation(gated, pixels * full_hidden);
            const auto input_bytes = shape.rgb_bytes(fp16);
            CHECK(cuMemAlloc(&stage_in, input_bytes));
            CHECK(cuMemAlloc(&stage_out, input_bytes * scale * scale));
            setup_ms[0] = elapsed(start);

            start = Clock::now();
            CHECK(cudnnSetStream(cudnn, stream));
            CHECK(cuMemsetD8Async(gated, 0, pixels * full_hidden * 2, stream));
            load_plan_cache();
            const int h = padded.h, w = padded.w;
            make_plan(h, w, 3, channels);
            make_plan(h, w, channels, 2 * full_hidden);
            make_plan(h, w, full_hidden, channels);
            make_plan(h, w, channels, channels / 4);
            make_plan(h / 2, w / 2, channels, 2 * reduced_hidden);
            make_plan(h / 2, w / 2, reduced_hidden, channels);
            make_plan(h / 2, w / 2, channels, 4 * channels);
            make_plan(h, w, channels, 3 * scale * scale);
            CHECK(cuStreamSynchronize(stream));
            if (scratch) CHECK(cuMemFree(scratch));
            scratch = 0;
            scratch_bytes = 0;
            if (!ensure_scratch(workspace_bytes)) CHECK(CUDA_ERROR_OUT_OF_MEMORY);
            plan_cache.save(plan_key);
            setup_ms[1] = elapsed(start);

            start = Clock::now();
            CHECK(cuMemsetD8Async(stage_in, 0, input_bytes, stream));
            forward(stage_in, stage_out, stream);
            CHECK(cuStreamSynchronize(stream));
            setup_ms[2] = elapsed(start);

            if (graph_enabled) {
                start = Clock::now();
                CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
                try {
                    forward(stage_in, stage_out, stream);
                } catch (...) {
                    cuStreamEndCapture(stream, &graph);
                    throw;
                }
                CHECK(cuStreamEndCapture(stream, &graph));
                CHECK(cuGraphInstantiate(&executable, graph, 0));
                CHECK(cuGraphUpload(executable, stream));
                CHECK(cuStreamSynchronize(stream));
                setup_ms[3] = elapsed(start);
            }
        } catch (...) {
            cuStreamSynchronize(stream);
            release_shape();
            throw;
        }
    }

    static double elapsed(Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    template<class... Args>
    void launch(CUstream target, Kernel kernel, int lanes, Args... args) {
        void* parameters[] = {&args...};
        CHECK(cuLaunchKernel(functions[kernel], (lanes + threads - 1) / threads, 1, 1, threads, 1, 1, 0,
                             target, parameters, nullptr));
    }

    template<class... Args>
    void elementwise(CUstream target, Kernel kernel, int count, Args... args) {
        launch(target, kernel, count, args..., count);
    }

    void conv(CUdeviceptr out, CUdeviceptr in, int h, int w, int cin, int cout, Filter filter) {
        const auto& p = *plans.at(umzi::PlanShape{padded.n, h, w, cin, cout});
        const float one = 1.f, zero = 0.f;
        CHECK(cudnnConvolutionForward(cudnn, &one, p.input, reinterpret_cast<void*>(in), p.filter,
            reinterpret_cast<void*>(weights + filter.weight), p.convolution, p.algorithm,
            reinterpret_cast<void*>(scratch), p.workspace, &zero, p.output, reinterpret_cast<void*>(out)));
    }

    void block(CUstream target, const Block& params, CUdeviceptr value, int h, int w, int hidden) {
        const int pixels = padded.n * h * w;
        launch(target, Normalize, pixels * 32, normalized, value, weights + params.norm_weight,
               weights + params.norm_bias, pixels);
        conv(projected, normalized, h, w, channels, 2 * hidden, params.expand);
        elementwise(target, Gate, pixels * hidden, gated, projected, weights + params.expand.bias,
                    weights + params.square.weight, weights + params.square.bias, weights + params.row.weight,
                    weights + params.row.bias, weights + params.column.weight, weights + params.column.bias,
                    hidden, h, w);
        conv(projected, gated, h, w, hidden, channels, params.project);
        elementwise(target, Residual, pixels * channels, value, projected, weights + params.project.bias,
                    weights + params.gamma);
    }

    void forward(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        const int pixels = padded.pixels(), h = padded.h, w = padded.w;
        elementwise(target, fp16 ? PackF16 : PackF32, pixels * 3, normalized, input, shape.h, shape.w, h, w);
        conv(projected, normalized, h, w, 3, channels, model.first);
        elementwise(target, InputBias, pixels * channels, state, initial, projected, weights + model.first.bias);
        for (const auto& group : model.groups) {
            for (const auto& layer : group.full) block(target, layer, state, h, w, full_hidden);
            conv(projected, state, h, w, channels, channels / 4, group.down);
            elementwise(target, Reduce, pixels / 4 * channels, reduced, projected, weights + group.down.bias, h / 2, w / 2);
            for (const auto& layer : group.reduced) block(target, layer, reduced, h / 2, w / 2, reduced_hidden);
            conv(projected, reduced, h / 2, w / 2, channels, 4 * channels, group.up);
            elementwise(target, Expand, pixels * channels, state, projected, weights + group.up.bias, h, w);
        }
        elementwise(target, Add, pixels * channels, state, initial);
        conv(projected, state, h, w, channels, 3 * scale * scale, model.last);
        elementwise(target, fp16 ? OutputF16 : OutputF32, shape.pixels() * 3 * scale * scale,
                    output, projected, weights + model.last.bias, scale * shape.h, scale * shape.w, h, w);
    }

    void execute(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        if (!executable) {
            CHECK(cudnnSetStream(cudnn, target));
            forward(input, output, target);
            return;
        }
        const auto input_bytes = shape.rgb_bytes(fp16);
        if (input != stage_in) CHECK(cuMemcpyDtoDAsync(stage_in, input, input_bytes, target));
        CHECK(cuGraphLaunch(executable, target));
        if (output != stage_out) CHECK(cuMemcpyDtoDAsync(output, stage_out, input_bytes * scale * scale, target));
    }

public:
    UmziModelInfo model_info() const override { return {sizeof(UmziModelInfo), "moesr", scale, 0}; }

    UmziIo tensors() const override {
        const auto dtype = uint32_t(fp16 ? UMZI_FLOAT16 : UMZI_FLOAT32);
        return {sizeof(UmziIo), {"input", {4, {0, 3, 0, 0}}, dtype, 0}, {"output", {4, {0, 3, 0, 0}}, dtype, 0}};
    }

    UmziIo describe(const UmziShape& request) const override {
        const auto concrete = checked(request);
        auto io = tensors();
        io.input.shape = request;
        io.input.bytes = concrete.rgb_bytes(fp16);
        io.output.shape = Shape{concrete.n, scale * concrete.h, scale * concrete.w}.nchw();
        io.output.bytes = io.input.bytes * scale * scale;
        return io;
    }

    void prepare_shape(const UmziShape& request) override {
        ContextScope scope(context);
        prepare(checked(request));
    }

    void run_host(const UmziShape& request, const void* input, void* output) override {
        ContextScope scope(context);
        prepare(checked(request));
        const auto input_bytes = shape.rgb_bytes(fp16);
        try {
            CHECK(cuMemcpyHtoDAsync(stage_in, input, input_bytes, stream));
            execute(stage_in, stage_out, stream);
            CHECK(cuMemcpyDtoHAsync(output, stage_out, input_bytes * scale * scale, stream));
            CHECK(cuStreamSynchronize(stream));
        } catch (...) {
            cuStreamSynchronize(stream);
            throw;
        }
    }

    void enqueue_device(const UmziShape& request, uint64_t input, uint64_t output, UmziStream stream_target) override {
        ContextScope scope(context);
        prepare(checked(request));
        auto target = reinterpret_cast<CUstream>(stream_target);
        try {
            execute(input, output, target);
            CHECK(cuEventRecord(complete, target));
            pending = true;
        } catch (...) {
            cuStreamSynchronize(target);
            throw;
        }
    }

    const char* cache_message() const override { return plan_cache.diagnostic.c_str(); }

    UmziSessionInfo session_info() const override {
        UmziSessionInfo info{};
        info.struct_size = sizeof(info);
        info.cache_status = plan_cache.status;
        info.buffer_bytes[0] = model.bytes;
        if (shape.n) {
            info.prepared_input = padded.nchw();
            info.buffer_bytes[1] = activation_bytes;
            info.buffer_bytes[2] = workspace_bytes;
            info.buffer_bytes[3] = shape.rgb_bytes(fp16) * (1 + scale * scale);
            std::copy(setup_ms.begin(), setup_ms.end(), info.shape_setup_ms);
        }
        return info;
    }

    uint32_t diagnostic_count() const override { return 2; }
    UmziDiagnostic diagnostic(uint32_t index) const override {
        if (index == 0) return {sizeof(UmziDiagnostic), "timed_plans", timed_plans};
        return {sizeof(UmziDiagnostic), "plan_workspace_bytes", int64_t(workspace_bytes)};
    }
};
} // namespace

std::unique_ptr<umzi::KernelSession> umzi::make_moesr(const ModelPackage& package, const UmziSessionOptions& options) {
    auto session = std::make_unique<Session>();
    session->open(package, options);
    return session;
}
