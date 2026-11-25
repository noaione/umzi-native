#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <cuda.h>
#include <cublasLt.h>
#include <cudnn.h>

#define UMZI_BUILD
#include "model_file.h"
#include "gaterv3_images.h"
#include "silu_plan.h"
#include "../../common/cuda_helpers.h"
#include "../../common/cuda_module.h"
#include "../../runtime/kernel.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
namespace fs = std::filesystem;
constexpr int tile = 128;
constexpr size_t default_workspace_bytes = 64 * 1024 * 1024;
constexpr int threads = 256;
constexpr int gate_pool_chunk = 64;
constexpr float bad_timing = 1e30f;
constexpr float one = 1, zero = 0;
constexpr uint16_t half_one = 0x3c00, half_zero = 0;
using gt::ShapeKey;
using gt::GemmChoice;
using umzi::AllocationError;
using umzi::Clock;
using umzi::ContextScope;

enum GateFlags : uint32_t {
    GT_CREATE_GRAPH = 1,
    GT_CREATE_FP16 = 2,
    GT_CREATE_FUSE_SILU = 4,
    GT_CREATE_TUNE_PROJECTIONS = 8,
    GT_CREATE_FUSE_GATE_POOL = 16,
    GT_CREATE_FP16_QKV = 32,
};

enum GateStat {
    GT_OPT_REPLAY_TUNED_SHAPES,
    GT_OPT_FUSED_GATE_POOL_SITES,
    GT_OPT_FP16_QKV_PROJECTIONS,
    GT_OPT_SPLIT_GATE_REGISTERS,
    GT_OPT_FUSED_GATE_REGISTERS,
    GT_OPT_STAT_COUNT,
};

#define CHECK(expr) umzi::check((expr), #expr)

void allocate(CUdeviceptr& pointer, size_t bytes, const char* name) {
    auto status = cuMemAlloc(&pointer, bytes);
    if (status == CUDA_ERROR_OUT_OF_MEMORY)
        throw AllocationError(std::string("Could not allocate ") + std::to_string(bytes)
            + " bytes for " + name + "; try a smaller tile or batch");
    umzi::check(status, name);
}

void allocate_host(void*& pointer, size_t bytes) {
    auto status = cuMemAllocHost(&pointer, bytes);
    if (status == CUDA_ERROR_OUT_OF_MEMORY)
        throw AllocationError("Could not allocate " + std::to_string(bytes) + " bytes for pinned RGB staging");
    umzi::check(status, "Allocate pinned RGB staging");
}

using namespace gt;

struct Convolution {
    cudnnTensorDescriptor_t input = nullptr, output = nullptr;
    cudnnFilterDescriptor_t weight = nullptr;
    cudnnConvolutionDescriptor_t conv = nullptr;
    cudnnConvolutionFwdAlgo_t algorithm{};
    size_t workspace = 0;

    ~Convolution() {
        if (input) cudnnDestroyTensorDescriptor(input);
        if (output) cudnnDestroyTensorDescriptor(output);
        if (weight) cudnnDestroyFilterDescriptor(weight);
        if (conv) cudnnDestroyConvolutionDescriptor(conv);
    }
};

struct Matmul {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t kernel = nullptr, input = nullptr, output = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace = 0;
    bool half_compute = false;
    int input_tensor = 0;
    size_t weight_offset = 0;
    bool accumulate = false;

    ~Matmul() {
        if (desc) cublasLtMatmulDescDestroy(desc);
        if (kernel) cublasLtMatrixLayoutDestroy(kernel);
        if (input) cublasLtMatrixLayoutDestroy(input);
        if (output) cublasLtMatrixLayoutDestroy(output);
    }
};

struct TimedGraph {
    std::vector<CUevent> events;
    CUgraph graph = nullptr;
    CUgraphExec executable = nullptr;
    explicit TimedGraph(size_t count) : events(count, nullptr) {}
    ~TimedGraph() {
        if (executable) cuGraphExecDestroy(executable);
        if (graph) cuGraphDestroy(graph);
        for (auto event : events) if (event) cuEventDestroy(event);
    }
};

struct TuningTimer {
    CUevent start = nullptr, stop = nullptr;
    ~TuningTimer() {
        if (start) cuEventDestroy(start);
        if (stop) cuEventDestroy(stop);
    }
    void initialize() {
        CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
        CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
    }
    template<class Launch>
    float measure(CUstream stream, Launch launch) {
        if (!launch()) return bad_timing;
        CHECK(cuEventRecord(start, stream));
        for (int i = 0; i < 5; ++i) {
            if (!launch()) return bad_timing;
        }
        CHECK(cuEventRecord(stop, stream));
        CHECK(cuEventSynchronize(stop));
        float elapsed;
        CHECK(cuEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }

    template<class Launch>
    float measure_replay(CUstream stream, Launch launch) {
        if (!launch()) return bad_timing;
        CHECK(cuStreamSynchronize(stream));
        TimedGraph graph(0);
        CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
        constexpr int launches = 8;
        bool captured = true;
        for (int i = 0; i < launches && captured; ++i) captured = launch();
        auto status = cuStreamEndCapture(stream, &graph.graph);
        if (!captured || status != CUDA_SUCCESS) return bad_timing;
        if (cuGraphInstantiate(&graph.executable, graph.graph, 0) != CUDA_SUCCESS)
            return bad_timing;
        CHECK(cuGraphLaunch(graph.executable, stream));
        CHECK(cuStreamSynchronize(stream));
        std::array<float, 3> elapsed{};
        for (auto& ms : elapsed) {
            CHECK(cuEventRecord(start, stream));
            CHECK(cuGraphLaunch(graph.executable, stream));
            CHECK(cuEventRecord(stop, stream));
            CHECK(cuEventSynchronize(stop));
            CHECK(cuEventElapsedTime(&ms, start, stop));
        }
        std::sort(elapsed.begin(), elapsed.end());
        return elapsed[1] / launches;
    }
};

} // namespace

class GtRunner {
public:
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUmodule module = nullptr;
    CUgraphExec executable = nullptr;
    CUdeviceptr input = 0, output = 0, weights = 0, arena = 0, scratch = 0;
    void* host_input = nullptr;
    void* host_output = nullptr;
    cudnnHandle_t cudnn = nullptr;
    cublasLtHandle_t blas = nullptr;
    int batch = 0, height = 0, width = 0;
    bool fp16 = false;
    bool fuse_silu = false;
    bool tune_projections = false;
    bool fuse_gate_pool = false;
    bool fp16_qkv = false;
    size_t io_bytes = 0;
    gt::PlanCache plan_cache;
    std::array<double, 4> setup_ms{};
    int fused_silu_count = 0;
    int timed_plans = 0;
    size_t peak_workspace_bytes = 0;
    std::string fusion_diagnostic;
    std::array<int, GT_OPT_STAT_COUNT> optimization_stats{};

    void buffer_sizes(uint64_t bytes[4]) const {
        bytes[0] = weight_bytes;
        bytes[1] = arena_bytes;
        bytes[2] = workspace_bytes;
        bytes[3] = (input ? io_bytes * batch : 0) + (output ? io_bytes * batch : 0);
    }

    size_t retained_workspace_bytes() const { return workspace_bytes; }

    void startup_memory(uint64_t bytes[4]) const {
        std::copy(startup_used.begin(), startup_used.end(), bytes);
    }

    void configure(Shape shape, uint32_t flags) {
        constexpr uint32_t supported = GT_CREATE_GRAPH | GT_CREATE_FP16 | GT_CREATE_FUSE_SILU
            | GT_CREATE_TUNE_PROJECTIONS | GT_CREATE_FUSE_GATE_POOL | GT_CREATE_FP16_QKV;
        if (flags & ~supported) throw std::runtime_error("Unsupported native runner flags");
        batch = shape.n;
        height = shape.h;
        width = shape.w;
        fp16 = (flags & GT_CREATE_FP16) != 0;
        fuse_silu = (flags & GT_CREATE_FUSE_SILU) != 0;
        tune_projections = (flags & GT_CREATE_TUNE_PROJECTIONS) != 0;
        fuse_gate_pool = (flags & GT_CREATE_FUSE_GATE_POOL) != 0;
        fp16_qkv = !fp16 && (flags & GT_CREATE_FP16_QKV) != 0;
        io_bytes = (shape.rgb_bytes(fp16) / shape.n);
    }

    void initialize_shape(Shape shape, uint32_t flags, const fs::path& cache_dir,
                          CUdevice selected_device, CUcontext shared_context, CUmodule shared_module,
                          CUdeviceptr shared_weights, cudnnHandle_t shared_cudnn, cublasLtHandle_t shared_blas,
                          const gt::Digest& build, const gt::ModelFile& model, gt::ResolvedModel resolved,
                          size_t workspace_limit = default_workspace_bytes) {
        auto stage_start = Clock::now();
        configure(shape, flags);
        scratch_limit_bytes = workspace_limit;
        device = selected_device;
        context = shared_context;
        module = shared_module;
        weights = shared_weights;
        cudnn = shared_cudnn;
        blas = shared_blas;
        weight_bytes = model.weight_bytes;
        tensors = std::move(resolved.tensors);
        alias_target = std::move(resolved.alias_target);
        conv_ld_in = std::move(resolved.conv_ld_in);
        fused_concat = std::move(resolved.fused_concat);
        inplace_inception = std::move(resolved.inplace_inception);
        skip_alloc = std::move(resolved.skip_alloc);
        ops = model.ops;
        arena_bytes = resolved.arena_bytes;
        ContextScope scope(context);
        record_memory(0);
        if (!cache_dir.empty()) {
            try {
                gt::Sha256 hash;
                hash.add(build);
                hash.add(model.content_digest);
                hash.add(std::array<int, 3>{batch, height, width});
                add_plan_environment(hash);
                plan_key = hash.finish();
            } catch (const std::exception& error) {
                plan_cache.unavailable(error);
            }
        }
        create_handles();
        allocate_shape();
        finish_setup(flags, cache_dir, stage_start);
    }

    void create_handles() {
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cudnnSetStream(cudnn, stream));
    }

    void finish_setup(uint32_t flags, const fs::path& cache_dir, Clock::time_point stage_start) {
        record_memory(1);
        setup_ms[0] = umzi::elapsed_ms(stage_start);
        stage_start = Clock::now();
        if (plan_cache.status != UMZI_CACHE_UNAVAILABLE) {
            try {
                plan_cache.open(cache_dir, "gaterv3", plan_key);
            } catch (const std::exception& error) {
                plan_cache.unavailable(error);
            }
        }
        prepare();
        if (fuse_silu) prepare_silu();
        CHECK(cuStreamSynchronize(stream));
        record_memory(2);
        setup_ms[1] = umzi::elapsed_ms(stage_start);
        stage_start = Clock::now();
        if (scratch) {
            CHECK(cuMemFree(scratch));
            scratch = 0;
        }
        if (workspace_bytes) {
            allocate(scratch, workspace_bytes, "inference workspace");
            peak_workspace_bytes = std::max(peak_workspace_bytes, workspace_bytes);
        }
        if (silu_plan) bind_silu();
        if (flags & GT_CREATE_GRAPH) {
            ensure_device_staging();
            CHECK(cuMemsetD8Async(input, 0, io_bytes * batch, stream));
            forward(stream);
            CHECK(cuStreamSynchronize(stream));
            setup_ms[2] = umzi::elapsed_ms(stage_start);
            stage_start = Clock::now();
            try {
                capture_graph();
            } catch (const std::exception& error) {
                if (!fused_silu_count) throw;
                if (executable) CHECK(cuGraphExecDestroy(executable));
                executable = nullptr;
                disable_fusion(error.what());
                capture_graph();
            }
            setup_ms[3] = umzi::elapsed_ms(stage_start);
        }
        record_memory(3);
        plan_cache.save(plan_key);
    }

    ~GtRunner() {
        if (!context) return;
        // Destruction also works if initialization failed partway through.
        if (cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
        if (stream) cuStreamSynchronize(stream);
        if (executable) cuGraphExecDestroy(executable);
        silu_bindings.clear();
        silu_plan.reset();
        matmuls.clear();
        convolutions.clear();
        for (auto ptr : {arena, scratch, input, output}) if (ptr) cuMemFree(ptr);
        if (host_input) cuMemFreeHost(host_input);
        if (host_output) cuMemFreeHost(host_output);
        if (stream) cuStreamDestroy(stream);
        CUcontext previous;
        cuCtxPopCurrent(&previous);
    }

    void enqueue(CUstream target) {
        if (executable) {
            CHECK(cuGraphLaunch(executable, target));
        } else {
            CHECK(cudnnSetStream(cudnn, target));
            forward(target);
        }
    }

    void run_host(const void* pixels, void* restored) {
        run_host({batch, height, width}, pixels, restored);
    }

    void ensure_device_staging() {
        if (!input) allocate(input, batch * io_bytes, "device RGB input");
        if (!output) allocate(output, batch * io_bytes, "device RGB output");
        if (!addresses.empty()) {
            addresses.front() = input;
            addresses.back() = output;
        }
    }

    void ensure_host_staging() {
        if (!host_input) allocate_host(host_input, io_bytes * batch);
        if (!host_output) allocate_host(host_output, io_bytes * batch);
    }

    void run_host(Shape shape, const void* pixels, void* restored) {
        ContextScope scope(context);
        ensure_host_staging();
        ensure_device_staging();
        if (shape.h == height && shape.w == width) {
            std::memcpy(host_input, pixels, batch * io_bytes);
        } else if (fp16) {
            pad_host<uint16_t>(shape, pixels);
        } else {
            pad_host<float>(shape, pixels);
        }
        CHECK(cuMemcpyHtoDAsync(input, host_input, batch * io_bytes, stream));
        enqueue(stream);
        CHECK(cuMemcpyDtoHAsync(host_output, output, batch * io_bytes, stream));
        CHECK(cuStreamSynchronize(stream));
        if (shape.h == height && shape.w == width) {
            std::memcpy(restored, host_output, batch * io_bytes);
        } else {
            size_t element = fp16 ? sizeof(uint16_t) : sizeof(float);
            auto* dst = static_cast<char*>(restored);
            const auto* src = static_cast<const char*>(host_output);
            for (int plane = 0; plane < batch * 3; ++plane) {
                for (int y = 0; y < shape.h; ++y) {
                    std::memcpy(dst, src + (size_t(plane) * height + y) * width * element, shape.w * element);
                    dst += shape.w * element;
                }
            }
        }
    }

    void run_device(Shape shape, CUdeviceptr pixels, CUdeviceptr restored, CUstream target) {
        if (shape.h == height && shape.w == width) {
            if (executable) {
                ensure_device_staging();
                CHECK(cuMemcpyDtoDAsync(input, pixels, batch * io_bytes, target));
                enqueue(target);
                CHECK(cuMemcpyDtoDAsync(restored, output, batch * io_bytes, target));
            } else {
                struct RestoreAddresses {
                    std::vector<CUdeviceptr>& addresses;
                    CUdeviceptr input, output;
                    ~RestoreAddresses() {
                        addresses.front() = input;
                        addresses.back() = output;
                    }
                } restore{addresses, input, output};
                addresses.front() = pixels;
                addresses.back() = restored;
                enqueue(target);
            }
            return;
        }
        ensure_device_staging();
        int padded_elements = batch * 3 * height * width;
        int cropped_elements = batch * 3 * shape.h * shape.w;
        launch(lookup(fp16 ? "pad_rgb16" : "pad_rgb"), target,
               (padded_elements + threads - 1) / threads, 1, threads,
               input, pixels, shape.h, shape.w, height, width, padded_elements);
        enqueue(target);
        launch(lookup(fp16 ? "crop_rgb16" : "crop_rgb"), target,
               (cropped_elements + threads - 1) / threads, 1, threads,
               restored, output, shape.h, shape.w, height, width, cropped_elements);
    }

    int operation_count() const { return int(ops.size()); }

    void profile(float* milliseconds, int count) {
        if (count != operation_count()) throw std::runtime_error("Invalid profile buffer size");
        ContextScope scope(context);
        TimedGraph timed(ops.size() + 1);
        for (auto& event : timed.events) CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        // Non-graph sessions have no staging buffers yet, and cuDNN cannot create plans inside stream capture.
        ensure_device_staging();
        CHECK(cuMemsetD8Async(input, 0, io_bytes * batch, stream));
        CHECK(cudnnSetStream(cudnn, stream));
        forward(stream);
        CHECK(cuStreamSynchronize(stream));
        CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
        try {
            forward(stream, timed.events.data());
        } catch (...) {
            cuStreamEndCapture(stream, &timed.graph);
            throw;
        }
        CHECK(cuStreamEndCapture(stream, &timed.graph));
        CHECK(cuGraphInstantiate(&timed.executable, timed.graph, 0));
        constexpr int iterations = 8;
        std::fill_n(milliseconds, count, 0.f);
        for (int run = 0; run <= iterations; ++run) {
            CHECK(cuGraphLaunch(timed.executable, stream));
            CHECK(cuStreamSynchronize(stream));
            if (!run) continue;
            for (int i = 0; i < count; ++i) {
                float elapsed;
                CHECK(cuEventElapsedTime(&elapsed, timed.events[i], timed.events[i + 1]));
                milliseconds[i] += elapsed / iterations;
            }
        }
    }

private:
    CUfunction dynamic_mean_project = nullptr;

    template<class T>
    void pad_host(Shape shape, const void* source) {
        auto* dst = static_cast<T*>(host_input);
        const auto* src = static_cast<const T*>(source);
        for (int plane = 0; plane < batch * 3; ++plane) {
            for (int y = 0; y < height; ++y) {
                int row = y < shape.h ? y : (height - shape.h < shape.h ? 2 * shape.h - 2 - y : shape.h - 1);
                const T* input_row = src + (size_t(plane) * shape.h + row) * shape.w;
                std::memcpy(dst, input_row, shape.w * sizeof(T));
                for (int x = shape.w; x < width; ++x) {
                    int column = width - shape.w < shape.w ? 2 * shape.w - 2 - x : shape.w - 1;
                    dst[x] = input_row[column];
                }
                dst += width;
            }
        }
    }

    void add_plan_environment(gt::Sha256& hash) const {
        auto add_u32 = [&](uint32_t value) {
            std::vector<char> bytes;
            umzi::append_number(bytes, value);
            hash.add(bytes.data(), bytes.size());
        };
        add_u32(umzi::plan_format_version);
        add_u32(umzi::gaterv3_id);
        add_u32(gt::plan_schema);
        add_u32(sizeof(void*));
        add_u32(sizeof(cublasLtMatmulAlgo_t));
        add_u32(static_cast<uint32_t>(scratch_limit_bytes));
        // The disk cache stores standalone convolutions and projections.
        // Gate/pool dispatch and the cuDNN SiLU graph are prepared separately.
        add_u32(static_cast<uint32_t>(fp16) | (tune_projections << 1) | (fp16_qkv << 2));
        CUuuid uuid{};
        CHECK(cuDeviceGetUuid(&uuid, device));
        hash.add(uuid);
        umzi::hash_driver_version(hash);
        hash.add(cudnnGetVersion());
        hash.add(cublasLtGetVersion());
    }

    void allocate_shape() {
        size_t free_bytes, capacity;
        CHECK(cuMemGetInfo(&free_bytes, &capacity));
        size_t required = arena_bytes;
        if (required > free_bytes)
            throw AllocationError("Shape buffers require " + std::to_string(required)
                + " bytes with " + std::to_string(free_bytes)
                + " bytes available; try a smaller tile or batch");
        allocate(arena, arena_bytes, "activation arena");
        addresses.resize(tensors.size());
        addresses.front() = input;
        addresses.back() = output;
        for (size_t i = 1; i + 1 < tensors.size(); ++i) {
            if (alias_target[i] >= 0) addresses[i] = addresses[alias_target[i]];
            else if (!skip_alloc[i]) addresses[i] = arena + size_t(tensors[i].offset) * batch;
        }
    }

    bool variable_pool() const { return height != tile || width != tile; }

    static bool variable_spatial(int channels, const Tensor& tensor) {
        int expected = tile * 32 / channels;
        return tensor.h != expected || tensor.w != expected;
    }

    std::vector<Tensor> tensors;
    std::vector<int> alias_target;
    std::vector<int> conv_ld_in;
    std::vector<bool> fused_concat;
    std::vector<bool> inplace_inception;
    std::vector<bool> skip_alloc;
    std::vector<Op> ops;
    std::vector<CUdeviceptr> addresses;
    std::vector<Convolution*> conv_by_op;
    std::vector<std::vector<std::unique_ptr<Matmul>>> matmuls;
    std::map<ShapeKey, GemmChoice> gemm_algorithms;
    std::map<ShapeKey, std::unique_ptr<Convolution>> convolutions;
    std::map<std::string, CUfunction> kernels;
    std::vector<CUfunction> kernel_by_op;
    CUfunction add_bias = nullptr;
    CUfunction mean_project = nullptr;
    CUfunction gate_pool_kernel = nullptr;
    CUfunction apply_channel_attention = nullptr;
    std::vector<bool> pool_fused;
    size_t arena_bytes = 0;
    size_t weight_bytes = 0;
    size_t workspace_bytes = 0;
    size_t scratch_limit_bytes = default_workspace_bytes;
    bool tuning_arena_cleared = false;
    std::array<uint64_t, 4> startup_used{};
    gt::Digest plan_key{};
    std::unique_ptr<gt::SiluPlan> silu_plan;
    std::vector<size_t> silu_sites;
    std::vector<std::unique_ptr<gt::BackendDescriptor>> silu_bindings;

    void ensure_tuning_workspace() {
        if (!tuning_arena_cleared) {
            CHECK(cuMemsetD8Async(arena, 0x28, arena_bytes, stream));
            tuning_arena_cleared = true;
        }
        if (scratch || !scratch_limit_bytes) return;
        allocate(scratch, scratch_limit_bytes, "tuning workspace");
        peak_workspace_bytes = std::max(peak_workspace_bytes, scratch_limit_bytes);
        record_memory(1);
    }

    void record_memory(size_t stage) {
        startup_used[stage] = umzi::device_used_bytes();
    }

    void capture_graph() {
        umzi::capture_cuda_graph(stream, executable, [&] { forward(stream); });
    }

    CUfunction lookup(std::string name) {
        auto [it, created] = kernels.try_emplace(std::move(name));
        if (created) CHECK(cuModuleGetFunction(&it->second, module, it->first.c_str()));
        return it->second;
    }

    CUdeviceptr weight(int offset) const { return offset < 0 ? 0 : weights + offset; }
    CUdeviceptr tensor(int id) const { return id < 0 ? 0 : addresses[id]; }
    int channels(int id) const { return id < 0 ? 0 : tensors[id].c; }
    bool fused_norm(size_t index) const {
        return index < ops.size() && ops[index].kind == Rms && ops[index].input[1] >= 0;
    }

    cublasStatus_t gemm(const Matmul& p, const Op& op, const cublasLtMatmulAlgo_t& algo,
                        size_t workspace, CUstream target) {
        auto y = pointer(tensor(op.out));
        const void* alpha_ptr = p.half_compute ? static_cast<const void*>(&half_one) : &one;
        const void* beta_ptr = p.half_compute
            ? static_cast<const void*>(p.accumulate ? &half_one : &half_zero)
            : (p.accumulate ? &one : &zero);
        return cublasLtMatmul(blas, p.desc, alpha_ptr,
            pointer(weight(op.weight[0]) + p.weight_offset), p.kernel,
            pointer(tensor(p.input_tensor) + 2 * op.y), p.input, beta_ptr,
            y, p.output, y, p.output, &algo, workspace ? pointer(scratch) : nullptr, workspace, target);
    }

    std::unique_ptr<Matmul> build_matmul(size_t index, int cin, int cout, int count, int ld_in, int ld_w,
                                         int bias_offset, int cache_tag, int input_tensor, size_t weight_offset,
                                         bool accumulate, bool selective, bool replay, TuningTimer& timer,
                                         const Op& op) {
        auto plan = std::make_unique<Matmul>();
        if (selective) ++optimization_stats[GT_OPT_FP16_QKV_PROJECTIONS];
        plan->half_compute = fp16 || selective;
        plan->input_tensor = input_tensor;
        plan->weight_offset = weight_offset;
        plan->accumulate = accumulate;
        CHECK(cublasLtMatmulDescCreate(&plan->desc,
            plan->half_compute ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
            plan->half_compute ? CUDA_R_16F : CUDA_R_32F));
        // NHWC pixels are columns, with channels contiguous within each column.
        // The packed weights have one contiguous row per output channel.
        cublasOperation_t transpose = CUBLAS_OP_T;
        CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                            &transpose, sizeof(transpose)));
        CHECK(cublasLtMatrixLayoutCreate(&plan->kernel, CUDA_R_16F, cin, cout, ld_w));
        CHECK(cublasLtMatrixLayoutCreate(&plan->input, CUDA_R_16F, cin, count, ld_in));
        CHECK(cublasLtMatrixLayoutCreate(&plan->output, CUDA_R_16F, cout, count, cout));
        if (bias_offset >= 0) {
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            CUdeviceptr ptr = weight(bias_offset);
            CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));
            CHECK(cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &ptr, sizeof(ptr)));
        }
        auto key = ShapeKey{cin, cout, count, ld_in, cache_tag};
        if (auto cached = gemm_algorithms.find(key); cached != gemm_algorithms.end()) {
            plan->algorithm = cached->second.algorithm;
            plan->workspace = cached->second.workspace;
            return plan;
        }
        if (auto saved = plan_cache.gemms.find(key); saved != plan_cache.gemms.end()) {
            const auto& choice = saved->second;
            cublasLtMatmulHeuristicResult_t checked{};
            if (choice.workspace <= scratch_limit_bytes
                && cublasLtMatmulAlgoCheck(blas, plan->desc, plan->kernel, plan->input, plan->output,
                                           plan->output, &choice.algorithm, &checked) == CUBLAS_STATUS_SUCCESS
                && checked.state == CUBLAS_STATUS_SUCCESS && checked.workspaceSize == choice.workspace) {
                plan->algorithm = choice.algorithm;
                plan->workspace = choice.workspace;
                workspace_bytes = std::max(workspace_bytes, plan->workspace);
                gemm_algorithms.emplace(key, choice);
                return plan;
            }
        }
        ensure_tuning_workspace();
        plan_cache.record_tuning();
        cublasLtMatmulPreference_t pref;
        CHECK(cublasLtMatmulPreferenceCreate(&pref));
        auto preference = std::unique_ptr<std::remove_pointer_t<cublasLtMatmulPreference_t>,
            decltype(&cublasLtMatmulPreferenceDestroy)>(pref, cublasLtMatmulPreferenceDestroy);
        uint64_t allotted = scratch_limit_bytes;
        CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &allotted, sizeof(allotted)));
        std::array<cublasLtMatmulHeuristicResult_t, 24> results{};
        constexpr int baseline_candidates = 8;
        int found = 0;
        CHECK(cublasLtMatmulAlgoGetHeuristic(blas, plan->desc, plan->kernel, plan->input,
                                            plan->output, plan->output, pref,
                                            replay ? int(results.size()) : baseline_candidates, results.data(), &found));
        float best = bad_timing;
        auto test_candidates = [&](int candidates, bool graph_replay) {
            for (int j = 0; j < candidates; ++j) {
                auto& candidate = results[j];
                if (candidate.state != CUBLAS_STATUS_SUCCESS || candidate.workspaceSize > scratch_limit_bytes) continue;
                auto run = [&] {
                    return gemm(*plan, op, candidate.algo, candidate.workspaceSize, stream) == CUBLAS_STATUS_SUCCESS;
                };
                float elapsed = graph_replay ? timer.measure_replay(stream, run) : timer.measure(stream, run);
                if (elapsed < best) {
                    best = elapsed;
                    plan->algorithm = candidate.algo;
                    plan->workspace = candidate.workspaceSize;
                }
            }
        };
        if (replay) {
            test_candidates(found, true);
            if (best != bad_timing) ++optimization_stats[GT_OPT_REPLAY_TUNED_SHAPES];
        }
        if (best == bad_timing) test_candidates(std::min(found, baseline_candidates), false);
        if (best == bad_timing) throw std::runtime_error("No working cuBLASLt plan for operation " + std::to_string(index));
        workspace_bytes = std::max(workspace_bytes, plan->workspace);
        ++timed_plans;
        auto selected = GemmChoice{plan->algorithm, plan->workspace};
        gemm_algorithms.emplace(key, selected);
        plan_cache.gemms.insert_or_assign(key, selected);
        return plan;
    }

    void prepare_matmul(size_t index, const Tensor& a, const Tensor& out, const Op& op, TuningTimer& timer) {
        bool selective = fp16_qkv && a.c == 512 && out.c == 2560;
        bool replay = tune_projections && (a.pixels() >= 4096 || a.c >= 512);
        int cin = a.c - op.y, count = out.pixels() * batch;
        int ld_in = conv_ld_in[index];
        matmuls[index].push_back(build_matmul(index, cin, out.c, count, ld_in, cin, op.weight[1],
            op.weight[1] >= 0 ? 1 : 0, op.input[0], 0, false, selective, replay, timer, op));
    }

    void prepare_concat_matmul(size_t index, const Op& cat, const Tensor& cat_out,
                               const Tensor& out, const Op& op, TuningTimer& timer) {
        bool selective = fp16_qkv && cat_out.c == 512 && out.c == 2560;
        bool replay = tune_projections && (cat_out.pixels() >= 4096 || cat_out.c >= 512);
        int count = out.pixels() * batch;
        int ld_w = cat_out.c;
        int ch_off = 0;
        for (int k = 0; k < 4 && cat.input[k] >= 0; ++k) {
            const auto& part = tensors[cat.input[k]];
            int bias_off = (k == 0) ? op.weight[1] : -1;
            int tag = (k == 0 ? (op.weight[1] >= 0 ? 2 : 6) : 4) | (ld_w << 3);
            matmuls[index].push_back(build_matmul(index, part.c, out.c, count, part.c, ld_w, bias_off,
                tag, cat.input[k], size_t(ch_off) * 2, k > 0, selective, replay, timer, op));
            ch_off += part.c;
        }
    }

    cudnnStatus_t convolution(const Convolution& p, const Op& op) {
        return cudnnConvolutionForward(cudnn, &one, p.input, pointer(tensor(op.input[0])),
            p.weight, pointer(weight(op.weight[0])), p.conv, p.algorithm, pointer(scratch),
            p.workspace, &zero, p.output, pointer(tensor(op.out)));
    }

    void prepare_conv(size_t index, const Tensor& in, const Tensor& out, const Op& op, TuningTimer& timer) {
        auto key = ShapeKey{in.h, in.w, in.c, out.c, op.x};
        auto [it, inserted] = convolutions.try_emplace(key);
        if (inserted) {
            it->second = std::make_unique<Convolution>();
            auto& p = *it->second;
            CHECK(cudnnCreateTensorDescriptor(&p.input));
            CHECK(cudnnCreateTensorDescriptor(&p.output));
            CHECK(cudnnCreateFilterDescriptor(&p.weight));
            CHECK(cudnnCreateConvolutionDescriptor(&p.conv));
            CHECK(cudnnSetTensor4dDescriptor(p.input, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF,
                                             batch, in.c, in.h, in.w));
            CHECK(cudnnSetTensor4dDescriptor(p.output, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF,
                                             batch, out.c, out.h, out.w));
            CHECK(cudnnSetFilter4dDescriptor(p.weight, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC,
                                            out.c, in.c, op.x, op.x));
            CHECK(cudnnSetConvolution2dDescriptor(p.conv, op.x / 2, op.x / 2,
                1, 1, 1, 1, CUDNN_CROSS_CORRELATION, fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT));
            if (auto saved = plan_cache.convolutions.find(key); saved != plan_cache.convolutions.end()) {
                const auto& choice = saved->second;
                size_t required = 0;
                auto algorithm = static_cast<cudnnConvolutionFwdAlgo_t>(choice.algorithm);
                auto math = static_cast<cudnnMathType_t>(choice.math);
                if (choice.algorithm < CUDNN_CONVOLUTION_FWD_ALGO_COUNT
                    && choice.math <= CUDNN_FMA_MATH && choice.workspace <= scratch_limit_bytes
                    && cudnnSetConvolutionMathType(p.conv, math) == CUDNN_STATUS_SUCCESS
                    && cudnnGetConvolutionForwardWorkspaceSize(cudnn, p.input, p.weight, p.conv,
                                                               p.output, algorithm, &required) == CUDNN_STATUS_SUCCESS
                    && required == choice.workspace) {
                    p.algorithm = algorithm;
                    p.workspace = required;
                    workspace_bytes = std::max(workspace_bytes, p.workspace);
                    conv_by_op[index] = &p;
                    return;
                }
            }
            ensure_tuning_workspace();
            plan_cache.record_tuning();
            CHECK(cudnnSetConvolutionMathType(p.conv, CUDNN_TENSOR_OP_MATH));
            std::array<cudnnConvolutionFwdAlgoPerf_t, 16> results;
            int count = 0;
            CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, p.input, p.weight, p.conv,
                                                        p.output, int(results.size()), &count, results.data()));
            const cudnnConvolutionFwdAlgoPerf_t* fastest = nullptr;
            float best = 1e30f;
            for (int j = 0; j < count; ++j) {
                const auto& candidate = results[j];
                if (candidate.status != CUDNN_STATUS_SUCCESS || candidate.memory > scratch_limit_bytes) continue;
                CHECK(cudnnSetConvolutionMathType(p.conv, candidate.mathType));
                p.algorithm = candidate.algo;
                p.workspace = candidate.memory;
                float elapsed = timer.measure(stream, [&] {
                    return convolution(p, op) == CUDNN_STATUS_SUCCESS;
                });
                if (elapsed < best) {
                    fastest = &candidate;
                    best = elapsed;
                }
            }
            if (!fastest)
                throw std::runtime_error("No cuDNN plan for operation " + std::to_string(index));
            p.algorithm = fastest->algo;
            p.workspace = fastest->memory;
            workspace_bytes = std::max(workspace_bytes, p.workspace);
            ++timed_plans;
            CHECK(cudnnSetConvolutionMathType(p.conv, fastest->mathType));
            plan_cache.convolutions.insert_or_assign(
                key, gt::ConvChoice{uint32_t(fastest->algo), uint32_t(fastest->mathType), p.workspace});
        }
        conv_by_op[index] = it->second.get();
    }

    void prepare() {
        conv_by_op.resize(ops.size());
        matmuls.resize(ops.size());
        kernel_by_op.resize(ops.size());
        pool_fused.resize(ops.size());
        TuningTimer timer;
        timer.initialize();
        add_bias = lookup(fp16 ? "bias16" : "bias");
        mean_project = lookup(fp16 ? "mean_project16" : "mean_project");
        if (fp16 && variable_pool()) dynamic_mean_project = lookup("mean_project16_dynamic");
        if (fuse_gate_pool) {
            const char* name = fp16 ? "gate_pool16_32" : "gate_pool32";
            gate_pool_kernel = lookup(variable_pool() ? std::string(name) + "_dynamic" : name);
        }
        constexpr const char* names[] = {
            "pack", nullptr, nullptr, nullptr, "mean_partial", "local_skip",
            nullptr, "global_skip", "shuffle", "concat", "silu", "span_gate",
            "depthwise", "attention", "gate", "add", "rgb", "shuffle_concat",
        };
        for (size_t i = 0; i < ops.size(); ++i) {
            if (fused_concat[i]) continue;
            const auto& op = ops[i];
            const auto& out = tensors[op.out];
            if (op.kind == MeanProject) {
                const auto& gated = tensors[op.input[0]];
                const auto pixels = gated.pixels();
                pool_fused[i] = fuse_gate_pool && i && ops[i - 1].kind == Local
                    && ops[i - 1].out == op.input[0] && gated.h == height && gated.w == width
                    && gated.c == 32 && out.c == 32;
                int chunk = pool_fused[i] ? gate_pool_chunk : 256;
                size_t partial_bytes = size_t((pixels + chunk - 1) / chunk) * batch * out.c
                    * (fp16 ? sizeof(uint16_t) : sizeof(float));
                workspace_bytes = std::max(workspace_bytes, partial_bytes);
                if (pool_fused[i]) {
                    ++optimization_stats[GT_OPT_FUSED_GATE_POOL_SITES];
                    CHECK(cuFuncGetAttribute(&optimization_stats[GT_OPT_SPLIT_GATE_REGISTERS],
                                              CU_FUNC_ATTRIBUTE_NUM_REGS, kernel_by_op[i - 1]));
                    CHECK(cuFuncGetAttribute(&optimization_stats[GT_OPT_FUSED_GATE_REGISTERS],
                                              CU_FUNC_ATTRIBUTE_NUM_REGS, gate_pool_kernel));
                }
            }
            if (op.kind == Conv) {
                const auto& in = tensors[op.input[0]];
                if (op.x == 1) {
                    if (i && fused_concat[i - 1]) {
                        const auto& cat = ops[i - 1];
                        if (in.c == 128 && out.c == 32 && cat.input[0] >= 0 && cat.input[1] >= 0
                            && cat.input[2] >= 0 && cat.input[3] >= 0 && tensors[cat.input[0]].c == 32
                            && tensors[cat.input[1]].c == 32 && tensors[cat.input[2]].c == 32
                            && tensors[cat.input[3]].c == 32) {
                            kernel_by_op[i] = lookup(fp16 ? "concat_conv1x1_32_16" : "concat_conv1x1_32");
                        } else {
                            prepare_concat_matmul(i, cat, in, out, op, timer);
                        }
                    } else if (conv_ld_in[i] == 96 && in.c == 48 && out.c == 32 && op.y == 0) {
                        kernel_by_op[i] = lookup(fp16 ? "strided_conv1x1_48_32_16" : "strided_conv1x1_48_32");
                    } else {
                        prepare_matmul(i, in, out, op, timer);
                    }
                } else {
                    prepare_conv(i, in, out, op, timer);
                }
            } else if (op.kind == Attention && op.weight[0] >= 0) {
                if (out.pixels() == 64) {
                    kernel_by_op[i] = lookup("attention_channels");
                } else {
                    kernel_by_op[i] = lookup("channel_attention_probabilities");
                    apply_channel_attention = lookup("channel_attention_apply");
                    const size_t probabilities_bytes = size_t(batch) * 16 * 32 * 32 * sizeof(uint16_t);
                    workspace_bytes = std::max(workspace_bytes, probabilities_bytes);
                }
            } else if (op.kind == Rms) {
                std::string name = fp16 ? "rms16_" : "rms";
                if (fused_norm(i)) {
                    if (!i || (ops[i - 1].kind != LocalSkip && ops[i - 1].kind != GlobalSkip))
                        throw std::runtime_error("Invalid fused RMS operation");
                    name = ops[i - 1].kind == LocalSkip ? "local_norm" : "global_norm";
                    if (fp16) name += "16_";
                }
                kernel_by_op[i] = lookup(name + std::to_string(out.c));
            } else if (op.kind == Local || op.kind == Inception) {
                int channels = op.kind == Local ? out.c : 2 * out.c / 3;
                const char* base = op.kind == Local ? (fp16 ? "local_gate16_" : "local_gate")
                                                    : (fp16 ? "inception16_" : "inception");
                auto name = base + std::to_string(channels);
                if (variable_spatial(channels, out)) name += "_dynamic";
                kernel_by_op[i] = lookup(name);
            } else {
                std::string name = names[op.kind];
                if (fp16 && op.kind != Shuffle && op.kind != Concat && op.kind != ShuffleConcat) name += "16";
                if ((op.kind == MeanProject && variable_pool())
                    || (op.kind == Depthwise && (out.h != 8 || out.w != 8))
                    || (op.kind == Attention && out.pixels() != 64))
                    name += "_dynamic";
                kernel_by_op[i] = lookup(name);
            }
        }
    }

    void disable_fusion(std::string diagnostic) {
        silu_bindings.clear();
        silu_plan.reset();
        fused_silu_count = 0;
        fusion_diagnostic = std::move(diagnostic);
    }

    void prepare_silu() {
        std::vector<int> uses(tensors.size());
        for (const auto& op : ops)
            for (int id : op.input)
                if (id >= 0) ++uses[id];
        for (size_t i = 0; i + 1 < ops.size(); ++i) {
            const auto& conv = ops[i];
            const auto& silu = ops[i + 1];
            if (conv.kind != Conv || conv.x != 3 || conv.weight[1] >= 0 || silu.kind != Silu
                || silu.input[0] != conv.out || uses[conv.out] != 1) continue;
            const auto& x = tensors[conv.input[0]];
            const auto& y = tensors[silu.out];
            if (x.h != height || x.w != width || x.c != 32 || y.h != height || y.w != width || y.c != 32)
                continue;
            const auto x_end = size_t(x.offset) + size_t(x.elements()) * 2;
            const auto y_end = size_t(y.offset) + size_t(y.elements()) * 2;
            if (std::max(size_t(x.offset), size_t(y.offset)) < std::min(x_end, y_end)) continue;
            silu_sites.push_back(i);
        }
        if (silu_sites.empty()) {
            fusion_diagnostic = "Schedule has no compatible non-aliasing convolution/SiLU sites";
            return;
        }
        try {
            auto candidate = std::make_unique<gt::SiluPlan>();
            candidate->initialize(cudnn, batch, height, width, fp16, scratch_limit_bytes);
            workspace_bytes = std::max(workspace_bytes, candidate->workspace);
            silu_plan = std::move(candidate);
        } catch (const std::exception& error) {
            disable_fusion(error.what());
        }
    }

    void bind_silu() {
        try {
            silu_bindings.resize(ops.size());
            for (auto i : silu_sites) {
                const auto& conv = ops[i];
                silu_bindings[i] = silu_plan->bind(pointer(tensor(conv.input[0])),
                    pointer(weight(conv.weight[0])), pointer(tensor(ops[i + 1].out)), pointer(scratch));
            }
            silu_plan->run(cudnn, *silu_bindings[silu_sites.front()]);
            CHECK(cuStreamSynchronize(stream));
            fused_silu_count = static_cast<int>(silu_sites.size());
        } catch (const std::exception& error) {
            disable_fusion(error.what());
        }
    }

    template<class... Args>
    void launch(CUfunction kernel, CUstream target, unsigned grid_x, unsigned grid_y,
                unsigned block, Args... args) {
        void* params[] = {static_cast<void*>(&args)...};
        CHECK(cuLaunchKernel(kernel, grid_x, grid_y, 1, block, 1, 1, 0, target, params, nullptr));
    }

    void forward(CUstream target, const CUevent* timing = nullptr) {
        for (size_t i = 0; i < ops.size(); ++i) {
            if (timing) CHECK(cuEventRecordWithFlags(timing[i], target, CU_EVENT_RECORD_EXTERNAL));
            if (fused_concat[i]) continue;
            if (fused_silu_count) {
                if (silu_bindings[i]) {
                    silu_plan->run(cudnn, *silu_bindings[i]);
                    continue;
                }
                if (i && silu_bindings[i - 1]) continue;
            }
            const auto& op = ops[i];
            const auto& o = tensors[op.out];
            const auto& in = tensors[op.input[0]];
            const auto a = tensor(op.input[0]), b = tensor(op.input[1]);
            const auto c = tensor(op.input[2]), d = tensor(op.input[3]);
            const auto w0 = weight(op.weight[0]), w1 = weight(op.weight[1]);
            const auto y = addresses[op.out];
            const int total = o.elements() * batch, pixels = o.pixels();
            const unsigned grid = (total + threads - 1) / threads;
            auto kernel = kernel_by_op[i];
            const auto run = [&](auto... args) { launch(kernel, target, grid, 1, threads, args...); };
            const auto pairs = [&](auto... args) {
                launch(kernel, target, (total / 2 + threads - 1) / threads, 1, threads, args..., total / 2);
            };
            if ((op.kind == LocalSkip || op.kind == GlobalSkip) && fused_norm(i + 1)) {
                const auto& next = ops[i + 1];
                int per_block = o.c == 32 ? 16 : 8;
                launch(kernel_by_op[i + 1], target, (pixels * batch + per_block - 1) / per_block,
                    1, threads, y, tensor(next.out), a, b, c, w0, weight(next.weight[0]),
                    weight(next.weight[1]), pixels, pixels * batch);
                continue;
            }
            switch (op.kind) {
            case Pack: run(y, a, pixels, total); break;
            case Conv:
                if (!matmuls[i].empty()) {
                    for (const auto& p : matmuls[i]) {
                        CHECK(gemm(*p, op, p->algorithm, p->workspace, target));
                    }
                } else if (kernel) {
                    const int px = o.pixels() * batch;
                    if (i && fused_concat[i - 1]) {
                        const auto& cat = ops[i - 1];
                        launch(kernel, target, unsigned((px + 127) / 128), 1, threads,
                               y, tensor(cat.input[0]), tensor(cat.input[1]), tensor(cat.input[2]),
                               tensor(cat.input[3]), w0, w1, px);
                    } else {
                        launch(kernel, target, unsigned((px + 127) / 128), 1, threads,
                               y, a, w0, w1, conv_ld_in[i], px);
                    }
                } else {
                    CHECK(convolution(*conv_by_op[i], op));
                    if (w1) launch(add_bias, target, grid, 1, threads, y, w1, o.c, total);
                }
                break;
            case Rms: {
                if (fused_norm(i)) break;
                int per_block = o.c == 32 ? 16 : 8;
                launch(kernel, target, (pixels * batch + per_block - 1) / per_block, 1, threads,
                       y, a, w0, w1, pixels * batch);
                break;
            }
            case Local:
                if (i + 1 < ops.size() && pool_fused[i + 1]) {
                    if (variable_pool()) {
                        int chunks = (pixels + gate_pool_chunk - 1) / gate_pool_chunk;
                        launch(gate_pool_kernel, target, batch * chunks, 1,
                               threads, y, scratch, a, w0, w1, o.h, o.w, chunks);
                    } else {
                        launch(gate_pool_kernel, target, pixels * batch / gate_pool_chunk, 1,
                               threads, y, scratch, a, w0, w1);
                    }
                } else if (variable_spatial(o.c, o)) {
                    pairs(y, a, w0, w1, o.h, o.w);
                } else {
                    pairs(y, a, w0, w1);
                }
                break;
            case MeanProject: {
                int chunk = pool_fused[i] ? gate_pool_chunk : 256;
                int chunks = (in.pixels() + chunk - 1) / chunk;
                if (!pool_fused[i]) {
                    unsigned sites = batch * chunks;
                    launch(kernel, target, (o.c / 32) * (variable_pool() ? sites : 1),
                           variable_pool() ? 1 : sites, threads, scratch, a, in.pixels(), o.c, chunks);
                }
                if (dynamic_mean_project) {
                    launch(dynamic_mean_project, target, batch, 1, threads, y, scratch, w0, w1,
                           in.pixels(), o.c, chunks, chunk);
                } else {
                    launch(mean_project, target, batch, 1, threads, y, scratch, w0, w1,
                           in.pixels(), o.c, chunks);
                }
                break;
            }
            case LocalSkip: pairs(y, a, b, c, w0, pixels, o.c / 2); break;
            case Inception: {
                int inplace = inplace_inception[i] ? 1 : 0;
                if (variable_spatial(2 * o.c / 3, o)) pairs(y, a, w0, w1, o.h, o.w, inplace);
                else pairs(y, a, w0, w1, inplace);
                break;
            }
            case GlobalSkip: pairs(y, a, b, w0, o.c / 2); break;
            case Shuffle: run(y, a, o.h, o.w, o.c, op.x, total); break;
            case Concat:
                launch(kernel, target, (total / 8 + threads - 1) / threads, 1, threads, y, a, b, c, d,
                       in.c / 8, channels(op.input[1]) / 8, channels(op.input[2]) / 8,
                       channels(op.input[3]) / 8, o.c / 8, total / 8);
                break;
            case Silu: pairs(y, a); break;
            case Span: pairs(y, a, b); break;
            case Depthwise:
                if (o.h == 8 && o.w == 8) run(y, a, w0, w1, in.c, total);
                else run(y, a, w0, w1, in.c, o.h, o.w, total);
                break;
            case Attention:
                if (w0 && pixels != 64) {
                    launch(kernel, target, batch * 16, 1, threads, scratch, a, w0, pixels);
                    launch(apply_channel_attention, target, batch * 16 * ((pixels + 63) / 64), 1, threads,
                           y, a, scratch, pixels);
                } else if (w0) launch(kernel, target, batch * 16, 1, threads, y, a, w0);
                else if (pixels == 64) launch(kernel, target, batch * 16, 1, threads, y, a);
                else launch(kernel, target, batch * 16 * ((pixels + 3) / 4), 1, 128, y, a, pixels);
                break;
            case Gate: run(y, a, b, in.c, total); break;
            case Add: run(y, a, b, total); break;
            case Rgb: run(y, a, b, fp16 ? w1 : w0, pixels, total); break;
            case ShuffleConcat:
                pairs(y, a, b, o.h, o.w, o.c / 2);
                break;
            }
        }
        if (timing) CHECK(cuEventRecordWithFlags(timing[ops.size()], target, CU_EVENT_RECORD_EXTERNAL));
    }

    static void* pointer(CUdeviceptr p) { return reinterpret_cast<void*>(p); }
};

namespace {
constexpr std::pair<const char*, uint32_t> gate_options[] = {
    {"fuse_silu", GT_CREATE_FUSE_SILU},
    {"tune_projections", GT_CREATE_TUNE_PROJECTIONS},
    {"fuse_gate_pool", GT_CREATE_FUSE_GATE_POOL},
    {"fp16_qkv", GT_CREATE_FP16_QKV},
};
constexpr const char* gate_stats[] = {
    "replay_tuned_shapes", "fused_gate_pool_sites", "fp16_qkv_projections",
    "split_gate_registers", "fused_gate_registers", "fused_silu_count",
    "timed_plans", "plan_workspace_bytes", "peak_plan_workspace_bytes",
};
static_assert(std::size(gate_stats) == GT_OPT_STAT_COUNT + 4);

class GateSession final : public umzi::KernelSession {
public:
    struct Shared {
        gt::ModelFile model;
        fs::path cache;
        uint32_t flags = GT_CREATE_GRAPH;
        size_t workspace_limit = default_workspace_bytes;
        CUdevice device = 0;
        CUcontext context = nullptr;
        CUmodule module = nullptr;
        CUdeviceptr weights = 0;
        cudnnHandle_t cudnn = nullptr;
        cublasLtHandle_t blas = nullptr;
        gt::Digest build{};

        ~Shared() {
            if (!context || cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
            if (cudnn) cudnnDestroy(cudnn);
            if (blas) cublasLtDestroy(blas);
            if (module) cuModuleUnload(module);
            if (weights) cuMemFree(weights);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
            cuDevicePrimaryCtxRelease(device);
        }
    };

    std::shared_ptr<Shared> shared = std::make_shared<Shared>();
    CUevent completed = nullptr;
    bool pending = false;
    std::unique_ptr<GtRunner> active;

    void open(const umzi::ModelPackage& package, const UmziSessionOptions& options) {
        shared->flags = (options.flags & UMZI_CUDA_GRAPH ? GT_CREATE_GRAPH : 0)
            | (options.precision == UMZI_FP16 ? GT_CREATE_FP16 : 0);
        shared->workspace_limit = default_workspace_bytes;
        if (options.cache_dir) shared->cache = fs::u8path(options.cache_dir);
        uint32_t seen = 0;
        bool seen_workspace = false;
        for (uint32_t i = 0; i < options.kernel_option_count; ++i) {
            const auto& option = options.kernel_options[i];
            if (option.name && std::strcmp(option.name, "workspace_mib") == 0) {
                if (seen_workspace || option.value < 0 || option.value > 2048)
                    throw std::runtime_error("Duplicate or invalid GateRV3 option: workspace_mib");
                seen_workspace = true;
                shared->workspace_limit = size_t(option.value) << 20;
                continue;
            }
            auto found = std::find_if(std::begin(gate_options), std::end(gate_options), [&](const auto& item) {
                return option.name && std::strcmp(item.first, option.name) == 0;
            });
            if (found == std::end(gate_options))
                throw std::runtime_error("Unknown GateRV3 option: " + std::string(option.name ? option.name : ""));
            if ((seen & found->second) || (option.value != 0 && option.value != 1))
                throw std::runtime_error("Duplicate or invalid GateRV3 option: " + std::string(option.name));
            seen |= found->second;
            if (option.value) shared->flags |= found->second;
        }
        GtRunner validator;
        validator.configure({1, tile, tile}, shared->flags);
        shared->model.parse(package);

        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&shared->device, options.device));
        CHECK(cuDevicePrimaryCtxRetain(&shared->context, shared->device));
        ContextScope scope(shared->context);
        const auto& image = umzi::load_kernel_module(shared->device, shared->module, gaterv3_images::embedded_cubins,
                                                     gaterv3_images::embedded_ptx);
        if (!shared->cache.empty()) {
            gt::Sha256 hash;
            const char build_id[] = __DATE__ " " __TIME__;
            hash.add(build_id);
            hash.add(package.schema);
            hash.add(image.data, image.size);
            shared->build = hash.finish();
        }
        CHECK(cudnnCreate(&shared->cudnn));
        CHECK(cublasLtCreate(&shared->blas));
        allocate(shared->weights, shared->model.weight_bytes, "model weights");
        CHECK(cuMemcpyHtoD(shared->weights, package.section(3).data(), shared->model.weight_bytes));
        CHECK(cuEventCreate(&completed, CU_EVENT_DISABLE_TIMING));
    }

    ~GateSession() override {
        if (!shared || !shared->context) return;
        if (cuCtxPushCurrent(shared->context) == CUDA_SUCCESS) {
            if (pending) cuEventSynchronize(completed);
            active.reset();
            if (completed) cuEventDestroy(completed);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        shared.reset();
    }

    std::unique_ptr<umzi::KernelSession> clone_shared() const override {
        auto copy = std::make_unique<GateSession>();
        copy->shared = shared;
        ContextScope scope(shared->context);
        CHECK(cuEventCreate(&copy->completed, CU_EVENT_DISABLE_TIMING));
        return copy;
    }

    bool fp16() const { return (shared->flags & GT_CREATE_FP16) != 0; }

    bool matches(Shape padded) const {
        return active && active->batch == padded.n && active->height == padded.h && active->width == padded.w;
    }

    uint64_t bytes_for(Shape shape) const {
        auto padded = shared->model.padded(shape);
        if (!matches(padded)) shared->model.check_shape(padded);
        return shape.rgb_bytes(fp16());
    }

    void wait() {
        if (pending) {
            CHECK(cuEventSynchronize(completed));
            pending = false;
        }
    }

    GtRunner& prepare(Shape shape) {
        ContextScope scope(shared->context);
        wait();
        auto padded = shared->model.padded(shape);
        if (!matches(padded)) {
            auto resolved = shared->model.resolve(padded);
            active.reset();
            auto plan = std::make_unique<GtRunner>();
            plan->initialize_shape(padded, shared->flags, shared->cache, shared->device, shared->context,
                                   shared->module, shared->weights, shared->cudnn, shared->blas,
                                   shared->build, shared->model, std::move(resolved), shared->workspace_limit);
            active = std::move(plan);
        }
        return *active;
    }

    void enqueue(Shape shape, CUdeviceptr input, CUdeviceptr output, CUstream target) {
        ContextScope scope(shared->context);
        auto& plan = prepare(shape);
        try {
            plan.run_device(shape, input, output, target);
            CHECK(cuEventRecord(completed, target));
            pending = true;
        } catch (...) {
            cuStreamSynchronize(target);
            throw;
        }
    }

    static Shape checked(const UmziShape& shape) {
        if (shape.rank != 4 || shape.dimensions[1] != 3)
            throw std::runtime_error("GateRV3 input requires RGB NCHW");
        return Shape::checked(shape.dimensions[0], shape.dimensions[2], shape.dimensions[3]);
    }

    UmziModelInfo model_info() const override { return {sizeof(UmziModelInfo), "gaterv3", 1, 0}; }
    umzi::TilingConstraints tiling_constraints() const override {
        const bool fixed_spatial = std::any_of(shared->model.ops.begin(), shared->model.ops.end(), [](const Op& op) {
            return op.kind == Attention && op.weight[0] >= 0;
        });
        return {shared->model.padded({1, 1, 1}).h, 1, 0, fixed_spatial ? tile : 0};
    }

    UmziIo tensors() const override {
        auto dtype = static_cast<uint32_t>(fp16() ? UMZI_FLOAT16 : UMZI_FLOAT32);
        return {sizeof(UmziIo), {"input", {4, {0, 3, 0, 0}}, dtype, 0},
            {"output", {4, {0, 3, 0, 0}}, dtype, 0}};
    }

    UmziIo describe(const UmziShape& shape) const override {
        auto bytes = bytes_for(checked(shape));
        auto io = tensors();
        io.input.shape = io.output.shape = shape;
        io.input.bytes = io.output.bytes = bytes;
        return io;
    }

    void prepare_shape(const UmziShape& shape) override { prepare(checked(shape)); }

    void run_host(const UmziShape& shape, const void* input, void* output) override {
        auto concrete = checked(shape);
        prepare(concrete).run_host(concrete, input, output);
    }

    void enqueue_device(const UmziShape& shape, uint64_t input, uint64_t output, UmziStream stream) override {
        enqueue(checked(shape), input, output, stream);
    }

    UmziSessionInfo session_info() const override {
        UmziSessionInfo result{};
        result.struct_size = sizeof(result);
        result.buffer_bytes[0] = shared->model.weight_bytes;
        if (active) {
            result.cache_status = active->plan_cache.status;
            result.prepared_input = {4, {static_cast<uint32_t>(active->batch), 3,
                static_cast<uint32_t>(active->height), static_cast<uint32_t>(active->width)}};
            active->buffer_sizes(result.buffer_bytes);
            active->startup_memory(result.startup_device_used_bytes);
            std::copy(active->setup_ms.begin(), active->setup_ms.end(), result.shape_setup_ms);
        }
        return result;
    }

    uint32_t diagnostic_count() const override { return std::size(gate_stats); }
    UmziDiagnostic diagnostic(uint32_t index) const override {
        if (index >= diagnostic_count()) throw std::runtime_error("Invalid GateRV3 diagnostic index");
        int64_t value = 0;
        if (active) {
            if (index < GT_OPT_STAT_COUNT) value = active->optimization_stats[index];
            else if (index == GT_OPT_STAT_COUNT) value = active->fused_silu_count;
            else if (index == GT_OPT_STAT_COUNT + 1) value = active->timed_plans;
            else if (index == GT_OPT_STAT_COUNT + 2) value = static_cast<int64_t>(active->retained_workspace_bytes());
            else value = static_cast<int64_t>(active->peak_workspace_bytes);
        }
        return {sizeof(UmziDiagnostic), gate_stats[index], value};
    }

    const char* cache_message() const override { return active ? active->plan_cache.diagnostic.c_str() : ""; }
    const char* kernel_message() const override { return active ? active->fusion_diagnostic.c_str() : ""; }

    void profile(float* milliseconds, uint32_t count) override {
        if (!active) throw std::runtime_error("Prepare a GateRV3 shape before profiling");
        ContextScope scope(shared->context);
        wait();
        active->profile(milliseconds, static_cast<int>(count));
    }

};
} // namespace

std::unique_ptr<umzi::KernelSession> umzi::make_gaterv3(const ModelPackage& package,
                                                      const UmziSessionOptions& options) {
    if (options.precision != UMZI_MIXED && options.precision != UMZI_FP16)
        throw std::runtime_error("Unsupported GateRV3 session precision");
    auto session = std::make_unique<GateSession>();
    session->open(package, options);
    return session;
}
