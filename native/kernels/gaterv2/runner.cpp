// GateRV2 kernel session. Features are NHWC FP16 in both precisions:
//   UMZI_MIXED: FP32 RGB I/O, FP32 accumulation in cuBLASLt and cuDNN.
//   UMZI_FP16:  FP16 RGB I/O, FP16 arithmetic in cuBLASLt and cuDNN.
// The Q/K/V projection output, normalization, contractions, key sum and
// denominator of the linear attention are FP32 in both modes, because an
// all-FP16 attention cancels to a zero denominator on flat tiles.
// 1x1 convolutions are cuBLASLt GEMMs with a fused bias; the 3x3 down and up
// convolutions use cuDNN; the remaining operations are fused CUDA kernels.
// The FP32 parity reference is scripts/reference/gaterv2_native.py.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <cuda.h>
#include <cublasLt.h>
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
constexpr size_t scratch_bytes = size_t(64) << 20;
constexpr size_t architecture_words = 16;
constexpr uint32_t max_levels = 4;
constexpr float bad_timing = 1e30f;

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

void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLASLt status " + std::to_string(int(status)));
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
    // The kernels assume 8-channel vectors, at most 64 base channels and 64 keys,
    // and at most 256 channels in a MetaGated block.
    bool valid = a.input == 3 && a.output == 3 && a.scale == 1 && a.levels >= 1 && a.levels <= max_levels
        && a.alignment == (1u << a.levels) && a.base >= 16 && a.base <= 64 && a.base % 16 == 0
        && a.latent >= 1 && a.latent <= 64 && a.eps > 0 && a.eps < 1
        && (a.base << (a.levels - 1)) <= 256 && (a.base << a.levels) / 16 <= 64;
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

struct Gemm {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t weight = nullptr, input = nullptr, output = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace = 0;
    bool half_compute = false;
    ~Gemm() {
        if (desc) cublasLtMatmulDescDestroy(desc);
        if (weight) cublasLtMatrixLayoutDestroy(weight);
        if (input) cublasLtMatrixLayoutDestroy(input);
        if (output) cublasLtMatrixLayoutDestroy(output);
    }
};

struct Events {
    CUevent start = nullptr, stop = nullptr;
    Events() {
        CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
        CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
    }
    ~Events() {
        if (start) cuEventDestroy(start);
        if (stop) cuEventDestroy(stop);
    }
    template<class Launch>
    float measure(CUstream stream, Launch launch) {
        if (!launch()) return bad_timing;
        CHECK(cuEventRecord(start, stream));
        constexpr int repeats = 3;
        for (int i = 0; i < repeats; ++i)
            if (!launch()) return bad_timing;
        CHECK(cuEventRecord(stop, stream));
        CHECK(cuEventSynchronize(stop));
        float elapsed = 0;
        CHECK(cuEventElapsedTime(&elapsed, start, stop));
        return elapsed / repeats;
    }
};

constexpr const char* kernel_names[] = {
    "input_conv_f32", "input_conv_f16", "output_conv_f32", "output_conv_f16", "weights_to_half",
    "pixel_norm_0", "pixel_norm_1", "pixel_norm_2", "pixel_norm_3", "local_gate", "pool_project",
    "inception_gate", "attention_prepare", "attention_reduce", "attention_gate", "shuffle_concat",
};
enum KernelId {
    InputF32, InputF16, OutputF32, OutputF16, WeightsToHalf, NormCopy, NormUnshuffle, NormMish, NormScaled,
    LocalGate, PoolProject, InceptionGate, AttentionPrepare, AttentionReduce, AttentionGate, ShuffleConcat,
    KernelCount,
};
static_assert(std::size(kernel_names) == KernelCount);

class GateRV2Session final : public umzi::KernelSession {
public:
    Architecture arch;
    std::unique_ptr<Network> network;
    std::vector<uint64_t> offsets;
    size_t weight_bytes = 0;
    bool use_graph = false;
    bool fp16 = false;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    std::array<CUfunction, KernelCount> functions{};
    CUdeviceptr weights = 0, half_weights = 0;
    CUstream stream = nullptr;
    CUevent completed = nullptr;
    cudnnHandle_t cudnn = nullptr;
    cublasLtHandle_t blas = nullptr;
    bool pending = false;

    // State for the prepared padded shape.
    Shape prepared{};
    std::map<std::tuple<int, int, int, int, int>, std::unique_ptr<Convolution>> convolutions;
    std::map<std::tuple<int, int, int, int, int>, std::unique_ptr<Gemm>> gemms;
    std::array<CUdeviceptr, 4> work{};
    std::vector<CUdeviceptr> skips;
    CUdeviceptr latent_x = 0, decoder_x = 0, qkv = 0, key_sum = 0, value_sum = 0, matrix = 0;
    CUdeviceptr key_part = 0, value_part = 0, matrix_part = 0, local_part = 0, multiplier = 0;
    CUdeviceptr stage_in = 0, stage_out = 0, scratch = 0;
    size_t activation_bytes = 0, workspace_bytes = 0, io_bytes = 0;
    CUgraph graph = nullptr;
    CUgraphExec executable = nullptr;
    CUstream target = nullptr;
    CUdeviceptr input_rgb = 0, output_rgb = 0;
    Shape request{};
    std::array<double, 4> setup_ms{};
    int timed_plans = 0;

    void open(const umzi::ModelPackage& package, const UmziSessionOptions& options) {
        if (package.schema != 1) throw std::runtime_error("Unsupported GateRV2 model payload version");
        if (options.precision != UMZI_MIXED && options.precision != UMZI_FP16)
            throw std::runtime_error("Unsupported GateRV2 session precision");
        fp16 = options.precision == UMZI_FP16;
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
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&completed, CU_EVENT_DISABLE_TIMING));
        CHECK(cudnnCreate(&cudnn));
        CHECK(cublasLtCreate(&blas));
        CHECK(cuMemAlloc(&weights, weight_bytes));
        CHECK(cuMemcpyHtoD(weights, data.data(), weight_bytes));
        upload_half_weights(data);
    }

    // The FP16 copy keeps every parameter at half its FP32 byte offset. The 3x3
    // down and up filters become KRSC for NHWC cuDNN, and the up filter's
    // outputs are reordered subpixel-major for shuffle_concat.
    void upload_half_weights(std::string_view data) {
        std::vector<float> values(weight_bytes / 4);
        std::memcpy(values.data(), data.data(), weight_bytes);
        auto to_krsc = [&](size_t index, int outputs, int inputs, auto output_row) {
            float* filter = values.data() + offsets[index] / 4;
            std::vector<float> original(filter, filter + size_t(outputs) * inputs * 9);
            for (int o = 0; o < outputs; ++o)
                for (int i = 0; i < inputs; ++i)
                    for (int t = 0; t < 9; ++t)
                        filter[(size_t(output_row(o)) * 9 + t) * inputs + i] = original[(size_t(o) * inputs + i) * 9 + t];
        };
        for (const auto& level : network->encoder)
            to_krsc(level.scale_w, level.dim / 2, level.dim, [](int o) { return o; });
        for (const auto& level : network->decoder) {
            const int half = level.dim, dim = 2 * half;
            to_krsc(level.scale_w, 2 * dim, dim, [half](int o) { return (o % 4) * half + o / 4; });
        }
        CUdeviceptr staging = 0;
        CHECK(cuMemAlloc(&staging, weight_bytes));
        try {
            CHECK(cuMemcpyHtoD(staging, values.data(), weight_bytes));
            CHECK(cuMemAlloc(&half_weights, weight_bytes / 2));
            target = stream;
            elementwise(WeightsToHalf, int(weight_bytes / 4), half_weights, staging);
            CHECK(cuStreamSynchronize(stream));
        } catch (...) {
            cuMemFree(staging);
            throw;
        }
        CHECK(cuMemFree(staging));
    }

    ~GateRV2Session() override {
        if (!context || cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
        if (pending) cuEventSynchronize(completed);
        if (stream) cuStreamSynchronize(stream);
        release_shape();
        if (cudnn) cudnnDestroy(cudnn);
        if (blas) cublasLtDestroy(blas);
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
    uint64_t io_element() const { return fp16 ? 2 : 4; }

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

    // Elements of the largest work buffer: three base widths at full resolution.
    uint64_t total_elements(Shape p) const {
        const uint64_t total = uint64_t(p.n) * uint64_t(p.h) * uint64_t(p.w) * 3 * arch.base;
        if (total > uint64_t(INT_MAX) - 256)
            throw std::runtime_error("Requested shape exceeds native tensor indexing limits");
        return total;
    }

    int tokens() const { return (prepared.h >> arch.levels) * (prepared.w >> arch.levels); }
    int chunks() const { return (tokens() + 63) / 64; }

    void release_shape() {
        if (executable) cuGraphExecDestroy(executable);
        if (graph) cuGraphDestroy(graph);
        executable = nullptr;
        graph = nullptr;
        convolutions.clear();
        gemms.clear();
        for (auto& buffer : work) {
            if (buffer) cuMemFree(buffer);
            buffer = 0;
        }
        for (auto buffer : skips)
            if (buffer) cuMemFree(buffer);
        skips.clear();
        for (auto* buffer : {&latent_x, &decoder_x, &qkv, &key_sum, &value_sum, &matrix, &key_part, &value_part,
                             &matrix_part, &local_part, &multiplier, &stage_in, &stage_out, &scratch}) {
            if (*buffer) cuMemFree(*buffer);
            *buffer = 0;
        }
        prepared = {};
        activation_bytes = workspace_bytes = io_bytes = 0;
        setup_ms = {};
        timed_plans = 0;
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
            const uint64_t n = uint64_t(p.n), pixels = uint64_t(p.h) * uint64_t(p.w);
            const uint64_t elements = total_elements(p);
            for (auto& buffer : work) allocate(buffer, elements * 2);
            for (uint32_t level = 0; level < arch.levels; ++level) {
                skips.push_back(0);
                allocate(skips.back(), n * uint64_t(arch.dim(level)) * (pixels >> (2 * level)) * 2);
            }
            prepared = p;
            const uint64_t l = uint64_t(arch.latent_dim()), k = uint64_t(arch.keys()), t = uint64_t(tokens());
            const uint64_t c = uint64_t(chunks());
            allocate(latent_x, n * t * l * 2);
            allocate(decoder_x, n * pixels * arch.base * 2);
            allocate(qkv, n * t * (2 * k + l) * 4);
            allocate(key_sum, n * k * 4);
            allocate(value_sum, n * l * 4);
            allocate(matrix, n * k * l * 4);
            if (c > 1) {
                allocate(key_part, n * c * k * 4);
                allocate(value_part, n * c * l * 4);
                allocate(matrix_part, n * c * k * l * 4);
            }
            uint64_t parts = 0;
            for (uint32_t level = 0; level < arch.levels; ++level) {
                const int dim = arch.dim(level);
                const auto grid = gate_grid(dim, p.h >> level, p.w >> level, 1);
                parts = std::max(parts, uint64_t(grid[0]) * grid[1] * uint64_t(dim));
            }
            allocate(local_part, n * parts * 4);
            allocate(multiplier, n * 256 * 4);
            io_bytes = n * 3 * pixels * io_element();
            CHECK(cuMemAlloc(&stage_in, io_bytes));
            CHECK(cuMemAlloc(&stage_out, io_bytes));
            CHECK(cuMemAlloc(&scratch, scratch_bytes));
            for (auto buffer : work) CHECK(cuMemsetD8Async(buffer, 0, elements * 2, stream));
            CHECK(cuMemsetD8Async(stage_in, 0, io_bytes, stream));
            request = p;
            setup_ms[0] = milliseconds_since(start);

            // The first pass times and records every cuBLASLt and cuDNN plan.
            start = Clock::now();
            CHECK(cudnnSetStream(cudnn, stream));
            input_rgb = stage_in;
            output_rgb = stage_out;
            forward(stream);
            CHECK(cuStreamSynchronize(stream));
            setup_ms[1] = milliseconds_since(start);

            start = Clock::now();
            forward(stream);
            CHECK(cuStreamSynchronize(stream));
            setup_ms[2] = milliseconds_since(start);

            if (use_graph) {
                start = Clock::now();
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
            cuStreamSynchronize(stream);
            release_shape();
            throw;
        }
    }

    // ---- Operations ----

    template<class... Args>
    void launch_shared(KernelId id, std::array<unsigned, 3> grid, unsigned block, unsigned shared, Args... args) {
        void* parameters[] = {&args...};
        CHECK(cuLaunchKernel(functions[id], grid[0], grid[1], grid[2], block, 1, 1, shared, target, parameters,
                             nullptr));
    }

    template<class... Args>
    void launch3(KernelId id, std::array<unsigned, 3> grid, unsigned block, Args... args) {
        launch_shared(id, grid, block, 0, args...);
    }

    template<class... Args>
    void launch(KernelId id, unsigned grid_x, unsigned grid_y, unsigned block, Args... args) {
        launch3(id, {grid_x, grid_y, 1}, block, args...);
    }

    // local_gate covers 256 / (C / 2) columns and gate_rows (4) rows per block.
    static std::array<unsigned, 3> gate_grid(int channels, int h, int w, int n) {
        const int columns = threads / (channels / 2);
        return {unsigned((w + columns - 1) / columns), unsigned((h + 3) / 4), unsigned(n)};
    }

    template<class... Args>
    void elementwise(KernelId id, int total, Args... args) {
        launch(id, unsigned((total + threads - 1) / threads), 1, threads, args..., total);
    }

    // D[rows, cout] = A[rows, cin] W^T + bias. A rows are input_stride
    // elements apart. The QKV projection writes FP32 with an FP32 bias; all
    // other projections write FP16 with an FP16 bias.
    void gemm(CUdeviceptr out, CUdeviceptr in, int input_stride, int cin, int cout, int rows, size_t weight,
              size_t bias, bool fp32_output = false) {
        auto& plan = gemms[std::make_tuple(cin, cout, rows, input_stride, int(fp32_output))];
        const CUdeviceptr bias_pointer = fp32_output ? param(bias) : filter(bias);
        if (!plan) {
            plan = std::make_unique<Gemm>();
            auto& p = *plan;
            p.half_compute = fp16 && !fp32_output;
            CHECK(cublasLtMatmulDescCreate(&p.desc, p.half_compute ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
                                           p.half_compute ? CUDA_R_16F : CUDA_R_32F));
            cublasOperation_t transpose = CUBLAS_OP_T;
            CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
            cudaDataType_t bias_type = fp32_output ? CUDA_R_32F : CUDA_R_16F;
            CHECK(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                                 sizeof(bias_type)));
            CHECK(cublasLtMatrixLayoutCreate(&p.weight, CUDA_R_16F, cin, cout, cin));
            CHECK(cublasLtMatrixLayoutCreate(&p.input, CUDA_R_16F, cin, rows, input_stride));
            CHECK(cublasLtMatrixLayoutCreate(&p.output, fp32_output ? CUDA_R_32F : CUDA_R_16F, cout, rows, cout));
            cublasLtMatmulPreference_t preference = nullptr;
            CHECK(cublasLtMatmulPreferenceCreate(&preference));
            std::unique_ptr<std::remove_pointer_t<cublasLtMatmulPreference_t>,
                            decltype(&cublasLtMatmulPreferenceDestroy)> owner(preference,
                                                                              cublasLtMatmulPreferenceDestroy);
            uint64_t limit = scratch_bytes;
            CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit,
                                                       sizeof(limit)));
            std::array<cublasLtMatmulHeuristicResult_t, 8> results{};
            int found = 0;
            CHECK(cublasLtMatmulAlgoGetHeuristic(blas, p.desc, p.weight, p.input, p.output, p.output, preference,
                                                int(results.size()), results.data(), &found));
            if (!found) throw std::runtime_error("No cuBLASLt plan for a GateRV2 projection");
            Events events;
            float best = bad_timing;
            int chosen = -1;
            for (int i = 0; i < found; ++i) {
                if (results[i].state != CUBLAS_STATUS_SUCCESS) continue;
                p.algorithm = results[i].algo;
                p.workspace = results[i].workspaceSize;
                const float elapsed = events.measure(target, [&] {
                    return run_gemm(p, out, in, weight, bias_pointer) == CUBLAS_STATUS_SUCCESS;
                });
                if (elapsed < best) {
                    best = elapsed;
                    chosen = i;
                }
            }
            if (chosen < 0) throw std::runtime_error("No working cuBLASLt plan for a GateRV2 projection");
            p.algorithm = results[chosen].algo;
            p.workspace = results[chosen].workspaceSize;
            workspace_bytes = std::max(workspace_bytes, p.workspace);
            ++timed_plans;
        }
        CHECK(run_gemm(*plan, out, in, weight, bias_pointer));
    }

    // Plans are shared by projections of the same shape, so each call sets its own bias.
    cublasStatus_t run_gemm(const Gemm& p, CUdeviceptr out, CUdeviceptr in, size_t weight, CUdeviceptr bias) {
        const auto status = cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                                           sizeof(bias));
        if (status != CUBLAS_STATUS_SUCCESS) return status;
        const float one = 1, zero = 0;
        const uint16_t half_one = 0x3c00, half_zero = 0;
        const void* alpha = p.half_compute ? static_cast<const void*>(&half_one) : &one;
        const void* beta = p.half_compute ? static_cast<const void*>(&half_zero) : &zero;
        auto* y = reinterpret_cast<void*>(out);
        return cublasLtMatmul(blas, p.desc, alpha, reinterpret_cast<void*>(filter(weight)), p.weight,
                              reinterpret_cast<void*>(in), p.input, beta, y, p.output, y, p.output, &p.algorithm,
                              p.workspace ? reinterpret_cast<void*>(scratch) : nullptr, p.workspace, target);
    }

    // A 3x3 NHWC convolution without bias.
    void conv3x3(CUdeviceptr out, CUdeviceptr in, int cin, int cout, int h, int w, size_t weight) {
        auto& plan = convolutions[std::make_tuple(cin, cout, h, w, prepared.n)];
        const float one = 1, zero = 0;
        auto run = [&](const Convolution& p) {
            return cudnnConvolutionForward(cudnn, &one, p.input, reinterpret_cast<void*>(in), p.filter,
                                           reinterpret_cast<void*>(filter(weight)), p.conv, p.algorithm,
                                           reinterpret_cast<void*>(scratch), p.workspace, &zero, p.output,
                                           reinterpret_cast<void*>(out));
        };
        if (!plan) {
            plan = std::make_unique<Convolution>();
            auto& p = *plan;
            CHECK(cudnnCreateTensorDescriptor(&p.input));
            CHECK(cudnnCreateTensorDescriptor(&p.output));
            CHECK(cudnnCreateFilterDescriptor(&p.filter));
            CHECK(cudnnCreateConvolutionDescriptor(&p.conv));
            CHECK(cudnnSetTensor4dDescriptor(p.input, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, prepared.n, cin, h, w));
            CHECK(cudnnSetTensor4dDescriptor(p.output, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, prepared.n, cout, h, w));
            CHECK(cudnnSetFilter4dDescriptor(p.filter, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, cout, cin, 3, 3));
            CHECK(cudnnSetConvolution2dDescriptor(p.conv, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                                  fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT));
            CHECK(cudnnSetConvolutionMathType(p.conv, CUDNN_TENSOR_OP_MATH));
            std::array<cudnnConvolutionFwdAlgoPerf_t, CUDNN_CONVOLUTION_FWD_ALGO_COUNT> results{};
            int count = 0;
            CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, p.input, p.filter, p.conv, p.output,
                                                        int(results.size()), &count, results.data()));
            Events events;
            float best = bad_timing;
            cudnnConvolutionFwdAlgoPerf_t chosen{};
            for (int i = 0; i < count; ++i) {
                const auto& candidate = results[i];
                if (candidate.status != CUDNN_STATUS_SUCCESS || candidate.memory > scratch_bytes) continue;
                CHECK(cudnnSetConvolutionMathType(p.conv, candidate.mathType));
                p.algorithm = candidate.algo;
                p.workspace = candidate.memory;
                const float elapsed = events.measure(target, [&] { return run(p) == CUDNN_STATUS_SUCCESS; });
                if (elapsed < best) {
                    best = elapsed;
                    chosen = candidate;
                }
            }
            if (best >= bad_timing) throw std::runtime_error("No cuDNN convolution algorithm for GateRV2");
            CHECK(cudnnSetConvolutionMathType(p.conv, chosen.mathType));
            p.algorithm = chosen.algo;
            p.workspace = chosen.memory;
            workspace_bytes = std::max(workspace_bytes, p.workspace);
            ++timed_plans;
        }
        CHECK(run(*plan));
    }

    // Per-pixel channel work with an optional RMSNorm; see pixel_norm in kernels.cu.
    void pixel_norm(KernelId mode, CUdeviceptr out_x, CUdeviceptr out_norm, CUdeviceptr a, CUdeviceptr b,
                    CUdeviceptr vec, size_t scale, size_t offset, int channels, int pixels, int width) {
        const int lanes = std::max(1, channels / 32);
        const int total = prepared.n * pixels;
        const CUdeviceptr scale_pointer = out_norm ? param(scale) : 0, offset_pointer = out_norm ? param(offset) : 0;
        launch(mode, unsigned((uint64_t(total) * lanes + threads - 1) / threads), 1, threads, out_x, out_norm, a, b,
               vec, scale_pointer, offset_pointer, channels, lanes, pixels, width,
               float(1.0 / std::sqrt(double(channels))), total);
    }

    // The normalization parameters of the next consumer, or none.
    struct Next {
        size_t scale = 0, offset = 0;
        bool present = false;
    };

    // One MetaGated block on x, whose local norm is already in work[0]. It
    // leaves x updated and, when next is present, its norm in work[0].
    void meta(const Meta& m, CUdeviceptr x, int h, int w, Next next) {
        const auto [norm, expanded, local, last] = work;
        const int n = prepared.n, dim = m.dim, pixels = h * w, rows = n * pixels, hidden = 3 * dim / 2;
        gemm(expanded, norm, dim, dim, 2 * dim, rows, m.expand_w, m.expand_b);
        const auto grid = gate_grid(dim, h, w, n);
        const int tiles = int(grid[0] * grid[1]);
        launch3(LocalGate, grid, threads, local, local_part, expanded, param(m.local_w), param(m.local_b), dim, h, w);
        launch(PoolProject, unsigned(n), unsigned((dim + 7) / 8), threads, multiplier, local_part, param(m.sca_w),
               param(m.sca_b), param(m.gamma0), dim, tiles, pixels);
        pixel_norm(NormScaled, x, norm, local, x, multiplier, m.glob_scale, m.glob_offset, dim, pixels, w);
        gemm(expanded, norm, dim, dim, 2 * hidden, rows, m.fc1_w, m.fc1_b);
        elementwise(InceptionGate, rows * (hidden / 8), local, expanded, param(m.square_w), param(m.square_b),
                    param(m.row_w), param(m.row_b), param(m.column_w), param(m.column_b), dim, h, w);
        gemm(last, local, hidden, hidden, dim, rows, m.fc2_w, m.fc2_b);
        pixel_norm(NormMish, x, next.present ? norm : 0, last, x, param(m.gamma1), next.scale, next.offset, dim,
                   pixels, w);
    }

    void latent(const Latent& block, Next next) {
        const auto [norm, projected, gated, last] = work;
        const int n = prepared.n, dim = arch.latent_dim(), keys = arch.keys(), t = tokens();
        const int rows = n * t, hidden = 3 * dim / 2, stride = 2 * keys + dim;
        gemm(projected, norm, dim, dim, 2 * hidden, rows, block.fc1_w, block.fc1_b);
        gemm(qkv, projected + uint64_t(2 * hidden - dim) * 2, 2 * hidden, dim, stride, rows, block.qkv_w,
             block.qkv_b, true);
        const int parts = chunks();
        const unsigned slices = unsigned((dim + 63) / 64);
        if (parts == 1) {
            launch3(AttentionPrepare, {1, unsigned(n), slices}, threads, qkv, key_sum, value_sum, matrix, keys, dim,
                    t, arch.eps, 1);
        } else {
            launch3(AttentionPrepare, {unsigned(parts), unsigned(n), slices}, threads, qkv, key_part, value_part,
                    matrix_part, keys, dim, t, arch.eps, 0);
            launch(AttentionReduce, unsigned(n), 1, threads, key_sum, value_sum, matrix, key_part, value_part,
                   matrix_part, keys, dim, parts, arch.eps);
        }
        elementwise(AttentionGate, rows * (hidden / 8), gated, projected, qkv, key_sum, value_sum, matrix, keys, dim,
                    t);
        gemm(last, gated, hidden, hidden, dim, rows, block.fc2_w, block.fc2_b);
        pixel_norm(NormMish, latent_x, next.present ? norm : 0, last, CUdeviceptr(0), CUdeviceptr(0), next.scale,
                   next.offset, dim, t, 1);
    }

    Next first_norm(const Level& level) const {
        if (level.blocks.empty()) return {};
        return {level.blocks.front().norm_scale, level.blocks.front().norm_offset, true};
    }

    void blocks(const Level& level, CUdeviceptr x, int h, int w, Next after) {
        for (size_t i = 0; i < level.blocks.size(); ++i) {
            const Next next = i + 1 < level.blocks.size() ? first_norm_of(level.blocks[i + 1]) : after;
            meta(level.blocks[i], x, h, w, next);
        }
    }

    static Next first_norm_of(const Meta& m) { return {m.norm_scale, m.norm_offset, true}; }

    // Unpadded RGB in input_rgb to the cropped RGB output in output_rgb.
    void forward(CUstream stream_target) {
        target = stream_target;
        const auto& net = *network;
        const int n = prepared.n, height = prepared.h, width = prepared.w;
        const int h0 = request.h, w0 = request.w;
        const CUdeviceptr norm = work[0];
        const Next level0 = first_norm(net.encoder[0]);
        const int base = int(arch.base);
        elementwise(fp16 ? InputF16 : InputF32, n * height * width, skips[0], level0.present ? norm : 0,
                    input_rgb, param(net.in_w), param(net.in_b), level0.present ? param(level0.scale) : 0,
                    level0.present ? param(level0.offset) : 0, base, h0, w0, height, width,
                    float(1.0 / std::sqrt(double(base))));
        for (uint32_t level = 0; level < arch.levels; ++level) {
            const auto& item = net.encoder[level];
            const int h = height >> level, w = width >> level, dim = item.dim;
            blocks(item, skips[level], h, w, {});
            conv3x3(work[1], skips[level], dim, dim / 2, h, w, item.scale_w);
            const bool deepest = level + 1 == arch.levels;
            const Next next = deepest ? Next{net.latent[0].scale, net.latent[0].offset, true}
                                      : first_norm(net.encoder[level + 1]);
            pixel_norm(NormUnshuffle, deepest ? latent_x : skips[level + 1], next.present ? norm : 0, work[1],
                       CUdeviceptr(0), CUdeviceptr(0), next.scale, next.offset, 2 * dim, (h / 2) * (w / 2), w / 2);
        }
        for (size_t i = 0; i < net.latent.size(); ++i) {
            const Next next = i + 1 < net.latent.size() ? Next{net.latent[i + 1].scale, net.latent[i + 1].offset, true}
                                                         : Next{};
            latent(net.latent[i], next);
        }
        const int low_h = height >> arch.levels, low_w = width >> arch.levels;
        for (uint32_t i = 0; i < arch.levels; ++i) {
            const auto& item = net.decoder[i];
            const int half = item.dim, dim = 2 * half, h = low_h << i, w = low_w << i;
            const CUdeviceptr source = i ? decoder_x : latent_x;
            conv3x3(work[1], source, dim, 2 * dim, h, w, item.scale_w);
            const int pixels = 4 * h * w;
            elementwise(ShuffleConcat, n * pixels * (dim / 8), work[2], work[1], skips[arch.levels - 1 - i], half,
                        2 * h, 2 * w);
            gemm(decoder_x, work[2], dim, dim, half, n * pixels, item.shor_w, item.shor_b);
            const Next first = first_norm(item);
            if (first.present)
                pixel_norm(NormCopy, CUdeviceptr(0), norm, decoder_x, CUdeviceptr(0), CUdeviceptr(0), first.scale,
                           first.offset, half, pixels, 2 * w);
            blocks(item, decoder_x, 2 * h, 2 * w, {});
        }
        elementwise(fp16 ? OutputF16 : OutputF32, n * h0 * w0, output_rgb, decoder_x, param(net.out_w),
                    param(net.out_b), input_rgb, base, h0, w0, height, width);
    }

    // ---- KernelSession ----

    // Graph replays use the shape and buffers they were captured with, so other
    // shapes and buffers run directly.
    void enqueue_shape(Shape shape, CUdeviceptr input, CUdeviceptr output, CUstream stream_target) {
        const bool replay = executable && shape == request && input == stage_in && output == stage_out;
        if (replay) {
            CHECK(cuGraphLaunch(executable, stream_target));
            return;
        }
        if (executable && shape == request) {
            // Keep the graph's fixed buffers and copy through them.
            CHECK(cuMemcpyDtoDAsync(stage_in, input, io_bytes_for(shape), stream_target));
            CHECK(cuGraphLaunch(executable, stream_target));
            CHECK(cuMemcpyDtoDAsync(output, stage_out, io_bytes_for(shape), stream_target));
            return;
        }
        request = shape;
        input_rgb = input;
        output_rgb = output;
        CHECK(cudnnSetStream(cudnn, stream_target));
        forward(stream_target);
    }

    uint64_t io_bytes_for(Shape shape) const {
        return uint64_t(shape.n) * 3 * uint64_t(shape.h) * uint64_t(shape.w) * io_element();
    }

    UmziModelInfo model_info() const override { return {sizeof(UmziModelInfo), "gaterv2", 1, 0}; }

    UmziIo tensors() const override {
        const auto dtype = static_cast<uint32_t>(fp16 ? UMZI_FLOAT16 : UMZI_FLOAT32);
        return {sizeof(UmziIo), {"input", {4, {0, 3, 0, 0}}, dtype, 0}, {"output", {4, {0, 3, 0, 0}}, dtype, 0}};
    }

    UmziIo describe(const UmziShape& shape) const override {
        const auto concrete = checked(shape);
        total_elements(padded(concrete));
        auto io = tensors();
        io.input.shape = io.output.shape = shape;
        io.input.bytes = io.output.bytes = io_bytes_for(concrete);
        return io;
    }

    // Graphs are captured for the shape a session was prepared with; prepare
    // the exact unpadded shape so the captured crop matches it.
    void prepare_exact(Shape shape) {
        prepare(shape);
        if (!(request == shape) && executable) {
            ContextScope scope(context);
            cuGraphExecDestroy(executable);
            cuGraphDestroy(graph);
            executable = nullptr;
            graph = nullptr;
            request = shape;
            input_rgb = stage_in;
            output_rgb = stage_out;
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
        }
    }

    void prepare_shape(const UmziShape& shape) override { prepare_exact(checked(shape)); }

    void run_host(const UmziShape& shape, const void* input, void* output) override {
        const auto concrete = checked(shape);
        prepare_exact(concrete);
        ContextScope scope(context);
        const uint64_t bytes = io_bytes_for(concrete);
        CHECK(cuMemcpyHtoDAsync(stage_in, input, bytes, stream));
        enqueue_shape(concrete, stage_in, stage_out, stream);
        CHECK(cuMemcpyDtoHAsync(output, stage_out, bytes, stream));
        CHECK(cuStreamSynchronize(stream));
    }

    void enqueue_device(const UmziShape& shape, uint64_t input, uint64_t output, UmziStream stream_target) override {
        const auto concrete = checked(shape);
        prepare_exact(concrete);
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
            result.buffer_bytes[2] = scratch_bytes;
            result.buffer_bytes[3] = 2 * io_bytes;
            std::copy(setup_ms.begin(), setup_ms.end(), result.shape_setup_ms);
        }
        return result;
    }

    uint32_t diagnostic_count() const override { return 2; }
    UmziDiagnostic diagnostic(uint32_t index) const override {
        if (index == 0) return {sizeof(UmziDiagnostic), "timed_plans", timed_plans};
        if (index == 1) return {sizeof(UmziDiagnostic), "plan_workspace_bytes", int64_t(workspace_bytes)};
        throw std::runtime_error("Invalid GateRV2 diagnostic index");
    }

    const char* kernel_message() const override {
        return fp16 ? "GateRV2 FP16 with FP32 attention" : "GateRV2 mixed precision with FP32 attention";
    }
};
} // namespace

std::unique_ptr<umzi::KernelSession> umzi::make_gaterv2(const ModelPackage& package,
                                                      const UmziSessionOptions& options) {
    auto session = std::make_unique<GateRV2Session>();
    session->open(package, options);
    return session;
}
