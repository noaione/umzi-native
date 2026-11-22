// GateRV2 kernel session. Features are NCHW FP16 in both precisions; the
// convolutions run through cuDNN and the family's CUDA kernels handle
// normalization, gating, layout and the linear attention. Attention
// normalization, contractions and the denominator always stay FP32, because
// an all-FP16 attention cancels to a zero denominator on flat tiles.
//   UMZI_MIXED: FP32 RGB I/O, FP32 convolution accumulation.
//   UMZI_FP16:  FP16 RGB I/O, FP16 convolution arithmetic.
// The FP32 parity reference is scripts/reference/gaterv2_native.py.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <cuda.h>
#include <cudnn.h>

#define UMZI_BUILD
#include "gaterv2_images.h"
#include "../../common/cuda_module.h"
#include "../../runtime/binary.h"
#include "../../runtime/kernel.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
using umzi::AllocationError;
using Clock = std::chrono::steady_clock;
constexpr int threads = 256;
constexpr size_t workspace_limit = size_t(256) << 20;
constexpr size_t architecture_words = 16;
constexpr uint32_t max_levels = 4;

double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check(CUresult status, const char* operation) {
    if (status == CUDA_ERROR_OUT_OF_MEMORY)
        throw AllocationError(std::string(operation) + ": out of memory; try a smaller tile or batch");
    umzi::check_cuda(status, operation);
}

void check(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + cudnnGetErrorString(status));
}
#define CHECK(expr) check((expr), #expr)

struct ContextScope {
    bool pushed = false;
    explicit ContextScope(CUcontext context) {
        CUcontext current = nullptr;
        CHECK(cuCtxGetCurrent(&current));
        if (current != context) {
            CHECK(cuCtxPushCurrent(context));
            pushed = true;
        }
    }
    ~ContextScope() {
        CUcontext previous;
        if (pushed) cuCtxPopCurrent(&previous);
    }
};

// Section 1: sixteen little-endian 32-bit values. See src/umzi_native/_model.py.
struct Architecture {
    uint32_t input = 0, output = 0, scale = 0, alignment = 0, base = 0, levels = 0, latent = 0;
    float eps = 0;
    std::array<uint32_t, max_levels> encoder{}, decoder{};

    int dim(uint32_t level) const { return int(base << level); }
    int latent_dim() const { return int(base << levels); }
    int keys() const { return latent_dim() / 16; }
};

struct Meta {
    int dim;
    size_t norm_scale, norm_offset, expand_w, expand_b, local_w, local_b, sca_w, sca_b, gamma0;
    size_t glob_scale, glob_offset, fc1_w, fc1_b, square_w, square_b, row_w, row_b, column_w, column_b;
    size_t fc2_w, fc2_b, gamma1;
};

struct Latent {
    size_t scale, offset, fc1_w, fc1_b, qkv_w, qkv_b, fc2_w, fc2_b;
};

struct Level {
    int dim;
    std::vector<Meta> blocks;
    size_t scale_w = 0, shor_w = 0, shor_b = 0;
};

// The parameter order is shared with src/umzi_native/exporter/_gaterv2.py
// and gaterv2_parameter_counts in src/umzi_native/_model.py.
struct Network {
    std::vector<uint64_t> counts;
    size_t in_w = 0, in_b = 0, out_w = 0, out_b = 0;
    std::vector<Level> encoder, decoder;
    std::vector<Latent> latent;

    size_t add(uint64_t count) {
        counts.push_back(count);
        return counts.size() - 1;
    }

    Meta meta(int d) {
        const uint64_t c = uint64_t(d), hidden = 3 * c / 2, branch = c / 8;
        Meta m{d};
        m.norm_scale = add(c);
        m.norm_offset = add(c);
        m.expand_w = add(2 * c * c);
        m.expand_b = add(2 * c);
        m.local_w = add(2 * c * 2 * 9);
        m.local_b = add(2 * c);
        m.sca_w = add(c * c);
        m.sca_b = add(c);
        m.gamma0 = add(c);
        m.glob_scale = add(c);
        m.glob_offset = add(c);
        m.fc1_w = add(2 * hidden * c);
        m.fc1_b = add(2 * hidden);
        m.square_w = add(branch * 9);
        m.square_b = add(branch);
        m.row_w = add(branch * 11);
        m.row_b = add(branch);
        m.column_w = add(branch * 11);
        m.column_b = add(branch);
        m.fc2_w = add(c * hidden);
        m.fc2_b = add(c);
        m.gamma1 = add(c);
        return m;
    }

    explicit Network(const Architecture& a) {
        const uint64_t base = a.base;
        in_w = add(base * a.input * 9);
        in_b = add(base);
        for (uint32_t level = 0; level < a.levels; ++level) {
            Level item{a.dim(level)};
            for (uint32_t i = 0; i < a.encoder[level]; ++i) item.blocks.push_back(meta(item.dim));
            item.scale_w = add(uint64_t(item.dim) / 2 * uint64_t(item.dim) * 9);
            encoder.push_back(std::move(item));
        }
        const uint64_t l = uint64_t(a.latent_dim()), k = uint64_t(a.keys());
        for (uint32_t i = 0; i < a.latent; ++i) {
            Latent block{};
            block.scale = add(l);
            block.offset = add(l);
            block.fc1_w = add(3 * l * l);
            block.fc1_b = add(3 * l);
            block.qkv_w = add((2 * k + l) * l);
            block.qkv_b = add(2 * k + l);
            block.fc2_w = add(l * 3 * l / 2);
            block.fc2_b = add(l);
            latent.push_back(block);
        }
        for (uint32_t i = 0; i < a.levels; ++i) {
            const uint64_t d = uint64_t(a.latent_dim()) >> i;
            Level item{int(d / 2)};
            item.scale_w = add(2 * d * d * 9);
            item.shor_w = add(d / 2 * d);
            item.shor_b = add(d / 2);
            for (uint32_t j = 0; j < a.decoder[i]; ++j) item.blocks.push_back(meta(int(d / 2)));
            decoder.push_back(std::move(item));
        }
        out_w = add(uint64_t(a.output) * base * 9);
        out_b = add(a.output);
    }
};

Architecture parse_architecture(std::string_view bytes) {
    if (bytes.size() != architecture_words * 4) throw std::runtime_error("Invalid GateRV2 model metadata size");
    umzi::BinaryReader reader(bytes);
    Architecture a;
    a.input = reader.number<uint32_t>();
    a.output = reader.number<uint32_t>();
    a.scale = reader.number<uint32_t>();
    a.alignment = reader.number<uint32_t>();
    a.base = reader.number<uint32_t>();
    a.levels = reader.number<uint32_t>();
    a.latent = reader.number<uint32_t>();
    auto eps = reader.number<uint32_t>();
    std::memcpy(&a.eps, &eps, sizeof(eps));
    for (auto& value : a.encoder) value = reader.number<uint32_t>();
    for (auto& value : a.decoder) value = reader.number<uint32_t>();
    bool valid = a.input == 3 && a.output == 3 && a.scale == 1 && a.levels >= 1 && a.levels <= max_levels
        && a.alignment == (1u << a.levels) && a.base >= 16 && a.base <= 64 && a.base % 16 == 0
        && a.latent >= 1 && a.latent <= 64 && a.eps > 0 && a.eps < 1;
    for (uint32_t i = 0; i < max_levels; ++i) {
        const bool used = i < a.levels;
        valid = valid && (used ? a.encoder[i] <= 32 && a.decoder[i] <= 32 : !a.encoder[i] && !a.decoder[i]);
    }
    if (!valid) throw std::runtime_error("Unsupported GateRV2 model architecture");
    return a;
}

struct Shape {
    int n = 0, h = 0, w = 0;
    bool operator==(const Shape& other) const { return n == other.n && h == other.h && w == other.w; }
};

struct Convolution {
    cudnnTensorDescriptor_t input = nullptr, output = nullptr;
    cudnnFilterDescriptor_t filter = nullptr;
    cudnnConvolutionDescriptor_t conv = nullptr;
    cudnnConvolutionFwdAlgo_t algorithm{};
    size_t workspace = 0;
    ~Convolution() {
        if (input) cudnnDestroyTensorDescriptor(input);
        if (output) cudnnDestroyTensorDescriptor(output);
        if (filter) cudnnDestroyFilterDescriptor(filter);
        if (conv) cudnnDestroyConvolutionDescriptor(conv);
    }
};

constexpr const char* kernel_names[] = {
    "pad_features_f32", "pad_features_f16", "residual_crop_f32", "residual_crop_f16", "weights_to_half",
    "rms_norm_f16", "local_gate_f16", "plane_sum_f16", "project", "local_skip_f16", "inception_f16",
    "gate_f16", "mish_skip_f16", "normalize_qk_f16", "key_value_f16", "attention_output_f16", "unshuffle_f16",
    "up_concat_f16", "add_bias_f16", "plane_sum_f32",
};
enum KernelId {
    PadF32, PadF16, CropF32, CropF16, WeightsToHalf, RmsNorm, LocalGate, PlaneSum, Project, LocalSkip, Inception,
    Gate, MishSkip, NormalizeQk, KeyValue, AttentionOutput, Unshuffle, UpConcat, AddBias, PlaneSumF32, KernelCount,
};
constexpr uint64_t feature_bytes = 2;
static_assert(std::size(kernel_names) == KernelCount);

class GateRV2Session final : public umzi::KernelSession {
public:
    Architecture arch;
    std::unique_ptr<Network> network;
    std::vector<uint64_t> offsets;
    size_t weight_bytes = 0;
    bool use_graph = false;
    bool fp16_io = false;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    std::array<CUfunction, KernelCount> functions{};
    CUdeviceptr weights = 0, half_weights = 0;
    CUstream stream = nullptr;
    CUevent completed = nullptr;
    cudnnHandle_t cudnn = nullptr;
    bool pending = false;

    // State for the prepared padded shape.
    Shape prepared{};
    std::map<std::tuple<int, int, int, int, int, int>, std::unique_ptr<Convolution>> convolutions;
    std::array<CUdeviceptr, 5> work{};
    std::vector<CUdeviceptr> skips;
    CUdeviceptr features_in = 0, stage_in = 0, stage_out = 0, mean = 0, sca = 0, key_sum = 0, value_sum = 0, matrix = 0, scratch = 0;
    size_t activation_bytes = 0, workspace_bytes = 0, io_bytes = 0;
    CUgraph graph = nullptr;
    CUgraphExec executable = nullptr;
    bool planning = false;
    CUstream target = nullptr;
    std::array<double, 4> setup_ms{};

    void open(const umzi::ModelPackage& package, const UmziSessionOptions& options) {
        if (package.schema != 1) throw std::runtime_error("Unsupported GateRV2 model payload version");
        if (options.precision != UMZI_MIXED && options.precision != UMZI_FP16)
            throw std::runtime_error("Unsupported GateRV2 session precision");
        fp16_io = options.precision == UMZI_FP16;
        if (options.kernel_option_count) throw std::runtime_error("GateRV2 sessions take no kernel options");
        use_graph = (options.flags & UMZI_CUDA_GRAPH) != 0;
        arch = parse_architecture(package.section(1));
        network = std::make_unique<Network>(arch);
        auto table = package.section(2);
        auto data = package.section(3);
        auto checkpoint = package.section(4);
        if (!checkpoint.empty() && checkpoint.size() != 32)
            throw std::runtime_error("Invalid GateRV2 model metadata size");
        const auto& counts = network->counts;
        if (table.size() != counts.size() * 16 || data.empty() || data.size() > (size_t(512) << 20))
            throw std::runtime_error("Invalid GateRV2 parameter table or weights");
        umzi::BinaryReader reader(table);
        uint64_t end = 0;
        for (auto count : counts) {
            auto offset = reader.number<uint64_t>();
            auto stored = reader.number<uint64_t>();
            if (stored != count || offset % 256 || offset < end || offset > data.size()
                || count > (data.size() - offset) / 4)
                throw std::runtime_error("Invalid GateRV2 parameter " + std::to_string(offsets.size()));
            offsets.push_back(offset);
            end = offset + count * 4;
        }
        if (end != data.size()) throw std::runtime_error("Unexpected trailing GateRV2 weight data");
        weight_bytes = data.size();

        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, options.device));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        ContextScope scope(context);
        umzi::load_kernel_module(device, module, gaterv2_images::embedded_cubins, gaterv2_images::embedded_ptx);
        for (int i = 0; i < KernelCount; ++i) CHECK(cuModuleGetFunction(&functions[i], module, kernel_names[i]));
        CHECK(cuMemAlloc(&weights, weight_bytes));
        CHECK(cuMemcpyHtoD(weights, data.data(), weight_bytes));
        // Convolution filters read an FP16 copy at half the FP32 byte offset.
        CHECK(cuMemAlloc(&half_weights, weight_bytes / 2));
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        target = stream;
        elementwise(WeightsToHalf, int(weight_bytes / 4), half_weights, weights);
        CHECK(cuStreamSynchronize(stream));
        CHECK(cuEventCreate(&completed, CU_EVENT_DISABLE_TIMING));
        CHECK(cudnnCreate(&cudnn));
    }

    ~GateRV2Session() override {
        if (!context || cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
        if (pending) cuEventSynchronize(completed);
        if (stream) cuStreamSynchronize(stream);
        release_shape();
        if (cudnn) cudnnDestroy(cudnn);
        if (completed) cuEventDestroy(completed);
        if (stream) cuStreamDestroy(stream);
        if (weights) cuMemFree(weights);
        if (half_weights) cuMemFree(half_weights);
        if (module) cuModuleUnload(module);
        CUcontext previous;
        cuCtxPopCurrent(&previous);
        cuDevicePrimaryCtxRelease(device);
    }

    CUdeviceptr param(size_t index) const { return weights + offsets[index]; }
    CUdeviceptr filter(size_t index) const { return half_weights + offsets[index] / 2; }
    uint64_t io_element() const { return fp16_io ? 2 : 4; }

    // ---- Shapes ----

    Shape checked(const UmziShape& shape) const {
        if (shape.rank != 4 || shape.dimensions[1] != 3) throw std::runtime_error("GateRV2 input requires RGB NCHW");
        const auto n = shape.dimensions[0], h = shape.dimensions[2], w = shape.dimensions[3];
        if (!n || !h || !w || n > 4096 || h > 65536 || w > 65536)
            throw std::runtime_error("Batch, height and width must be positive and fit native indexing");
        const uint32_t align = arch.alignment;
        const uint32_t pad_h = (align - h % align) % align, pad_w = (align - w % align) % align;
        if (pad_h >= h || pad_w >= w)
            throw std::runtime_error("GateRV2 reflection padding is invalid for " + std::to_string(h) + "x"
                + std::to_string(w));
        return {int(n), int(h), int(w)};
    }

    Shape padded(Shape shape) const {
        const int align = int(arch.alignment);
        return {shape.n, (shape.h + align - 1) / align * align, (shape.w + align - 1) / align * align};
    }

    // Elements of the largest feature buffer, which also bounds 32-bit kernel indexing.
    uint64_t total_elements(Shape p) const {
        const uint64_t pixels = uint64_t(p.h) * uint64_t(p.w);
        uint64_t largest = 3 * pixels;
        for (uint32_t level = 0; level <= arch.levels; ++level) {
            const uint64_t level_dim = uint64_t(arch.base) << level;
            largest = std::max(largest, 3 * level_dim * (pixels >> (2 * level)));
        }
        const uint64_t total = largest * uint64_t(p.n);
        if (total > uint64_t(INT_MAX) - 256)
            throw std::runtime_error("Requested shape exceeds native tensor indexing limits");
        return total;
    }

    void release_shape() {
        if (executable) cuGraphExecDestroy(executable);
        if (graph) cuGraphDestroy(graph);
        executable = nullptr;
        graph = nullptr;
        convolutions.clear();
        for (auto& buffer : work) {
            if (buffer) cuMemFree(buffer);
            buffer = 0;
        }
        for (auto buffer : skips)
            if (buffer) cuMemFree(buffer);
        skips.clear();
        for (auto* buffer : {&features_in, &stage_in, &stage_out, &mean, &sca, &key_sum, &value_sum, &matrix,
                             &scratch}) {
            if (*buffer) cuMemFree(*buffer);
            *buffer = 0;
        }
        prepared = {};
        activation_bytes = workspace_bytes = io_bytes = 0;
        setup_ms = {};
    }

    void allocate(CUdeviceptr& pointer, uint64_t bytes) {
        CHECK(cuMemAlloc(&pointer, std::max<uint64_t>(bytes, 256)));
        activation_bytes += bytes;
    }

    void wait() {
        if (pending) {
            CHECK(cuEventSynchronize(completed));
            pending = false;
        }
    }

    void prepare(Shape shape) {
        ContextScope scope(context);
        wait();
        const auto p = padded(shape);
        if (prepared == p) return;
        release_shape();
        try {
            auto start = Clock::now();
            const uint64_t elements = total_elements(p);
            for (auto& buffer : work) allocate(buffer, elements * feature_bytes);
            const uint64_t pixels = uint64_t(p.h) * uint64_t(p.w);
            for (uint32_t level = 0; level < arch.levels; ++level) {
                skips.push_back(0);
                allocate(skips.back(), uint64_t(p.n) * uint64_t(arch.dim(level)) * (pixels >> (2 * level))
                    * feature_bytes);
            }
            const uint64_t l = uint64_t(arch.latent_dim()), k = uint64_t(arch.keys()), n = uint64_t(p.n);
            allocate(mean, n * l * 4);
            allocate(sca, n * l * 4);
            allocate(key_sum, n * k * 4);
            allocate(value_sum, n * l * 4);
            allocate(matrix, n * k * l * 4);
            allocate(features_in, n * 3 * pixels * feature_bytes);
            // Host runs stage unpadded RGB in device memory; the padded size bounds every shape.
            io_bytes = n * 3 * pixels * io_element();
            CHECK(cuMemAlloc(&stage_in, io_bytes));
            CHECK(cuMemAlloc(&stage_out, io_bytes));
            prepared = p;
            setup_ms[0] = milliseconds_since(start);

            start = Clock::now();
            planning = true;
            forward(stream);
            planning = false;
            if (workspace_bytes) CHECK(cuMemAlloc(&scratch, workspace_bytes));
            setup_ms[1] = milliseconds_since(start);

            start = Clock::now();
            CHECK(cuMemsetD8Async(features_in, 0, n * 3 * pixels * feature_bytes, stream));
            CHECK(cudnnSetStream(cudnn, stream));
            forward(stream);
            CHECK(cuStreamSynchronize(stream));
            setup_ms[2] = milliseconds_since(start);

            if (use_graph) {
                start = Clock::now();
                CHECK(cudnnSetStream(cudnn, stream));
                CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
                try {
                    forward(stream);
                } catch (...) {
                    CUgraph failed = nullptr;
                    cuStreamEndCapture(stream, &failed);
                    if (failed) cuGraphDestroy(failed);
                    throw;
                }
                CHECK(cuStreamEndCapture(stream, &graph));
                CHECK(cuGraphInstantiate(&executable, graph, 0));
                CHECK(cuGraphUpload(executable, stream));
                CHECK(cuStreamSynchronize(stream));
                setup_ms[3] = milliseconds_since(start);
            }
        } catch (...) {
            planning = false;
            cuStreamSynchronize(stream);
            release_shape();
            throw;
        }
    }

    // ---- Operations ----

    template<class... Args>
    void launch_shaped(KernelId id, unsigned blocks, unsigned block_x, unsigned block_y, Args... args) {
        if (planning) return;
        void* parameters[] = {&args...};
        CHECK(cuLaunchKernel(functions[id], blocks, 1, 1, block_x, block_y, 1, 0, target, parameters, nullptr));
    }

    template<class... Args>
    void launch(KernelId id, unsigned blocks, Args... args) {
        launch_shaped(id, blocks, threads, 1, args...);
    }

    template<class... Args>
    void elementwise(KernelId id, int total, Args... args) {
        launch(id, unsigned((total + threads - 1) / threads), args..., total);
    }

    // A same-size convolution of cin channels from an NCHW slice whose batch
    // items are batch_planes planes apart. The bias is added separately, as in
    // the reference runner's cuDNN convolution followed by a bias addition.
    void conv(CUdeviceptr out, CUdeviceptr in, int cin, int batch_planes, int h, int w,
              size_t weight, int cout, int k, CUdeviceptr bias) {
        auto& plan = convolutions[std::make_tuple(cin, batch_planes, h, w, cout, k)];
        if (!plan) {
            plan = std::make_unique<Convolution>();
            auto& p = *plan;
            CHECK(cudnnCreateTensorDescriptor(&p.input));
            CHECK(cudnnCreateTensorDescriptor(&p.output));
            CHECK(cudnnCreateFilterDescriptor(&p.filter));
            CHECK(cudnnCreateConvolutionDescriptor(&p.conv));
            CHECK(cudnnSetTensor4dDescriptorEx(p.input, CUDNN_DATA_HALF, prepared.n, cin, h, w,
                                               batch_planes * h * w, h * w, w, 1));
            CHECK(cudnnSetTensor4dDescriptor(p.output, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, prepared.n, cout, h, w));
            CHECK(cudnnSetFilter4dDescriptor(p.filter, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, cout, cin, k, k));
            // Mixed precision accumulates in FP32; FP16 sessions use FP16 arithmetic.
            CHECK(cudnnSetConvolution2dDescriptor(p.conv, k / 2, k / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                                  fp16_io ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT));
            CHECK(cudnnSetConvolutionMathType(p.conv, CUDNN_TENSOR_OP_MATH));
            std::array<cudnnConvolutionFwdAlgoPerf_t, CUDNN_CONVOLUTION_FWD_ALGO_COUNT> results{};
            int count = 0;
            CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, p.input, p.filter, p.conv, p.output,
                                                        int(results.size()), &count, results.data()));
            const cudnnConvolutionFwdAlgoPerf_t* chosen = nullptr;
            for (int i = 0; i < count && !chosen; ++i) {
                const auto& candidate = results[i];
                if (candidate.status == CUDNN_STATUS_SUCCESS && candidate.memory <= workspace_limit)
                    chosen = &candidate;
            }
            if (!chosen) throw std::runtime_error("No cuDNN convolution algorithm for GateRV2");
            CHECK(cudnnSetConvolutionMathType(p.conv, chosen->mathType));
            p.algorithm = chosen->algo;
            p.workspace = chosen->memory;
            workspace_bytes = std::max(workspace_bytes, p.workspace);
        }
        if (planning) return;
        const float one = 1, zero = 0;
        CHECK(cudnnConvolutionForward(cudnn, &one, plan->input, reinterpret_cast<void*>(in), plan->filter,
                                      reinterpret_cast<void*>(filter(weight)), plan->conv, plan->algorithm,
                                      reinterpret_cast<void*>(scratch), plan->workspace, &zero, plan->output,
                                      reinterpret_cast<void*>(out)));
        if (bias) elementwise(AddBias, prepared.n * cout * h * w, out, bias, cout, h * w);
    }

    static CUdeviceptr planes(CUdeviceptr base, int count, int pixels) {
        return base + uint64_t(count) * uint64_t(pixels) * feature_bytes;
    }

    void rms(CUdeviceptr out, CUdeviceptr in, size_t scale, size_t offset, int channels, int pixels) {
        const int total = prepared.n * pixels;
        launch_shaped(RmsNorm, unsigned((total + 31) / 32), 32, 8, out, in, param(scale), param(offset), channels,
                      pixels, float(1.0 / std::sqrt(double(channels))), total);
    }

    void plane_sum(CUdeviceptr out, CUdeviceptr in, int pixels, int count, int batch_planes, int offset,
                   float add, int average, KernelId kernel = PlaneSum) {
        launch(kernel, unsigned(prepared.n * count), out, in, pixels, count, batch_planes, offset, add, average);
    }

    // MetaGated block from work[0] back to work[0].
    void meta(const Meta& m, int h, int w) {
        const auto [a, b, c, d, e] = work;
        const int n = prepared.n, dim = m.dim, pixels = h * w, hidden = 3 * dim / 2;
        rms(b, a, m.norm_scale, m.norm_offset, dim, pixels);
        // The 1x1 biases are added by the kernels that consume those convolutions.
        conv(c, b, dim, dim, h, w, m.expand_w, 2 * dim, 1, 0);
        elementwise(LocalGate, n * dim * pixels, b, c, param(m.expand_b), param(m.local_w), param(m.local_b), dim, h,
                    w);
        plane_sum(mean, b, pixels, dim, dim, 0, 0.f, 1);
        elementwise(Project, n * dim, sca, mean, param(m.sca_w), param(m.sca_b), dim, dim);
        elementwise(LocalSkip, n * dim * pixels, d, b, sca, a, param(m.gamma0), dim, pixels);
        rms(b, d, m.glob_scale, m.glob_offset, dim, pixels);
        conv(c, b, dim, dim, h, w, m.fc1_w, 2 * hidden, 1, 0);
        const CUdeviceptr fc1_bias = param(m.fc1_b);
        elementwise(Inception, n * dim * pixels, b, planes(c, 2 * hidden - dim, pixels),
                    fc1_bias + uint64_t(2 * hidden - dim) * 4, param(m.square_w), param(m.square_b), param(m.row_w), param(m.row_b),
                    param(m.column_w), param(m.column_b), dim, dim / 8, 2 * hidden, h, w);
        elementwise(Gate, n * hidden * pixels, e, c, fc1_bias, b, hidden, hidden - dim, pixels);
        conv(b, e, hidden, hidden, h, w, m.fc2_w, dim, 1, 0);
        elementwise(MishSkip, n * dim * pixels, a, b, param(m.fc2_b), d, param(m.gamma1), dim, pixels);
    }

    // Latent GatedCNNBlock with linear attention, from work[0] back to work[0].
    void latent(const Latent& block, int h, int w) {
        const auto [a, b, c, d, e] = work;
        const int n = prepared.n, dim = arch.latent_dim(), keys = arch.keys(), tokens = h * w;
        const int hidden = 3 * dim / 2, projected = 2 * keys + dim;
        rms(b, a, block.scale, block.offset, dim, tokens);
        conv(c, b, dim, dim, h, w, block.fc1_w, 2 * hidden, 1, param(block.fc1_b));
        conv(d, planes(c, 2 * hidden - dim, tokens), dim, 2 * hidden, h, w, block.qkv_w, projected, 1,
             param(block.qkv_b));
        elementwise(NormalizeQk, n * tokens, e, d, keys, projected, tokens);
        plane_sum(key_sum, e, tokens, keys, 2 * keys, keys, arch.eps, 0, PlaneSumF32);
        plane_sum(value_sum, d, tokens, dim, projected, 2 * keys, 0.f, 0);
        const int outputs = n * keys * dim;
        launch(KeyValue, unsigned((uint64_t(outputs) * 32 + threads - 1) / threads), matrix, e, d, keys, dim, projected,
               2 * keys, tokens, outputs);
        elementwise(AttentionOutput, n * dim * tokens, b, e, matrix, key_sum, value_sum, keys, dim, tokens);
        elementwise(Gate, n * hidden * tokens, e, c, CUdeviceptr(0), b, hidden, hidden - dim, tokens);
        conv(b, e, hidden, hidden, h, w, block.fc2_w, dim, 1, 0);
        elementwise(MishSkip, n * dim * tokens, a, b, param(block.fc2_b), CUdeviceptr(0), CUdeviceptr(0), dim, tokens);
    }

    // Padded FP16 RGB in features_in to the network output, before the RGB residual, in work[1].
    void forward(CUstream stream_target) {
        target = stream_target;
        const auto [a, b, c, d, e] = work;
        const int n = prepared.n, height = prepared.h, width = prepared.w;
        const auto& net = *network;
        conv(a, features_in, 3, 3, height, width, net.in_w, int(arch.base), 3, param(net.in_b));
        for (uint32_t level = 0; level < arch.levels; ++level) {
            const auto& item = net.encoder[level];
            const int h = height >> level, w = width >> level, dim = item.dim;
            for (const auto& block : item.blocks) meta(block, h, w);
            if (!planning)
                CHECK(cuMemcpyDtoDAsync(skips[level], a, uint64_t(n) * dim * h * w * feature_bytes, target));
            conv(b, a, dim, dim, h, w, item.scale_w, dim / 2, 3, 0);
            elementwise(Unshuffle, n * 2 * dim * (h / 2) * (w / 2), a, b, dim / 2, h / 2, w / 2);
        }
        const int low_h = height >> arch.levels, low_w = width >> arch.levels;
        for (const auto& block : net.latent) latent(block, low_h, low_w);
        for (uint32_t i = 0; i < arch.levels; ++i) {
            const auto& item = net.decoder[i];
            const int dim = 2 * item.dim, h = low_h << i, w = low_w << i;
            conv(b, a, dim, dim, h, w, item.scale_w, 2 * dim, 3, 0);
            elementwise(UpConcat, n * dim * 4 * h * w, c, b, skips[arch.levels - 1 - i], item.dim, 2 * h, 2 * w);
            conv(a, c, dim, dim, 2 * h, 2 * w, item.shor_w, item.dim, 1, param(item.shor_b));
            for (const auto& block : item.blocks) meta(block, 2 * h, 2 * w);
        }
        conv(b, a, int(arch.base), int(arch.base), height, width, net.out_w, 3, 3, param(net.out_b));
        (void)n;
    }

    void execute(CUstream stream_target) {
        if (executable) {
            CHECK(cuGraphLaunch(executable, stream_target));
        } else {
            CHECK(cudnnSetStream(cudnn, stream_target));
            forward(stream_target);
        }
    }

    // The input must stay valid until the stream completes: the crop adds it as the RGB residual.
    void enqueue_shape(Shape shape, CUdeviceptr input, CUdeviceptr output, CUstream stream_target) {
        const auto p = prepared;
        target = stream_target;
        elementwise(fp16_io ? PadF16 : PadF32, p.n * 3 * p.h * p.w, features_in, input, shape.h, shape.w, p.h, p.w);
        execute(stream_target);
        target = stream_target;
        elementwise(fp16_io ? CropF16 : CropF32, shape.n * 3 * shape.h * shape.w, output, work[1], input,
                    shape.h, shape.w, p.h, p.w);
    }

    // ---- KernelSession ----

    UmziModelInfo model_info() const override { return {sizeof(UmziModelInfo), "gaterv2", 1, 0}; }

    UmziIo tensors() const override {
        const auto dtype = static_cast<uint32_t>(fp16_io ? UMZI_FLOAT16 : UMZI_FLOAT32);
        return {sizeof(UmziIo), {"input", {4, {0, 3, 0, 0}}, dtype, 0}, {"output", {4, {0, 3, 0, 0}}, dtype, 0}};
    }

    UmziIo describe(const UmziShape& shape) const override {
        const auto concrete = checked(shape);
        total_elements(padded(concrete));
        auto io = tensors();
        io.input.shape = io.output.shape = shape;
        io.input.bytes = io.output.bytes
            = uint64_t(concrete.n) * 3 * uint64_t(concrete.h) * uint64_t(concrete.w) * io_element();
        return io;
    }

    void prepare_shape(const UmziShape& shape) override { prepare(checked(shape)); }

    void run_host(const UmziShape& shape, const void* input, void* output) override {
        const auto concrete = checked(shape);
        prepare(concrete);
        ContextScope scope(context);
        const uint64_t bytes = uint64_t(concrete.n) * 3 * uint64_t(concrete.h) * uint64_t(concrete.w) * io_element();
        CHECK(cuMemcpyHtoDAsync(stage_in, input, bytes, stream));
        enqueue_shape(concrete, stage_in, stage_out, stream);
        CHECK(cuMemcpyDtoHAsync(output, stage_out, bytes, stream));
        CHECK(cuStreamSynchronize(stream));
    }

    void enqueue_device(const UmziShape& shape, uint64_t input, uint64_t output, UmziStream stream_target) override {
        const auto concrete = checked(shape);
        prepare(concrete);
        ContextScope scope(context);
        auto selected = reinterpret_cast<CUstream>(stream_target);
        try {
            enqueue_shape(concrete, input, output, selected);
            CHECK(cuEventRecord(completed, selected));
            pending = true;
        } catch (...) {
            cuStreamSynchronize(selected);
            throw;
        }
    }

    UmziSessionInfo session_info() const override {
        UmziSessionInfo result{};
        result.struct_size = sizeof(result);
        result.cache_status = UMZI_CACHE_DISABLED;
        result.buffer_bytes[0] = weight_bytes + weight_bytes / 2;
        if (prepared.n) {
            result.prepared_input = {4, {uint32_t(prepared.n), 3, uint32_t(prepared.h), uint32_t(prepared.w)}};
            result.buffer_bytes[1] = activation_bytes;
            result.buffer_bytes[2] = workspace_bytes;
            result.buffer_bytes[3] = 2 * io_bytes;
            std::copy(setup_ms.begin(), setup_ms.end(), result.shape_setup_ms);
        }
        return result;
    }

    const char* kernel_message() const override {
        return fp16_io ? "GateRV2 FP16 with FP32 attention" : "GateRV2 mixed precision with FP32 attention";
    }
};
} // namespace

std::unique_ptr<umzi::KernelSession> umzi::make_gaterv2(const ModelPackage& package,
                                                      const UmziSessionOptions& options) {
    auto session = std::make_unique<GateRV2Session>();
    session->open(package, options);
    return session;
}
