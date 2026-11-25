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
#include "../../common/cuda_helpers.h"
#include "../../common/cuda_module.h"
#include "../../common/cuda_plan_cache.h"
#include "../../runtime/binary.h"
#include "../../runtime/kernel.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using umzi::AllocationError;
using umzi::Clock;
using umzi::ContextScope;
constexpr int threads = 256;
constexpr uint64_t default_workspace_mib = 64;
constexpr uint64_t missing = ~uint64_t(0);
constexpr size_t architecture_words = 16;
constexpr uint32_t max_levels = 4;
constexpr float bad_timing = 1e30f;
// Bump when the host changes plan keys, descriptor semantics, or the choice format.
constexpr uint32_t plan_schema = 1;

#define CHECK(expr) umzi::check((expr), #expr)

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
    "pixel_norm_0", "pixel_norm_1", "pixel_norm_2", "pixel_norm_3", "pixel_mish32_128", "pixel_norm_mish32_128",
    "pixel_norm_scaled32_128", "local_gate", "local_gate32_128", "pool_project",
    "inception_gate", "inception_gate_aligned", "inception_gate32_128", "attention_prepare", "attention_reduce",
    "attention_gate", "shuffle_concat",
};
enum KernelId {
    InputF32, InputF16, OutputF32, OutputF16, WeightsToHalf, NormCopy, NormUnshuffle, NormMish, NormScaled,
    Mish32, NormMish32, NormScaled32, LocalGate, LocalGate32, PoolProject, InceptionGate, InceptionAligned, InceptionGate32,
    AttentionPrepare, AttentionReduce, AttentionGate, ShuffleConcat,
    KernelCount,
};
static_assert(std::size(kernel_names) == KernelCount);

class GateRV2Session final : public umzi::KernelSession {
public:
    Architecture arch;
    std::unique_ptr<Network> network;
    // Packed device offsets of each parameter in the FP32 and FP16 copies, or
    // missing. Only small parameters keep FP32; projections and filters are FP16.
    std::vector<uint64_t> offsets32, offsets16;
    size_t weight_bytes = 0, weight32_bytes = 0, weight16_bytes = 0;
    bool tune_plans = true;
    uint64_t workspace_limit = default_workspace_mib << 20;
    bool use_graph = false;
    bool fp16 = false;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    std::array<CUfunction, KernelCount> functions{};
    struct Shared {
        CUdevice device = 0;
        CUcontext context = nullptr;
        CUmodule module = nullptr;
        CUdeviceptr weights = 0, half_weights = 0;
        cudnnHandle_t cudnn = nullptr;
        cublasLtHandle_t blas = nullptr;

        ~Shared() {
            if (!context || cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
            if (cudnn) cudnnDestroy(cudnn);
            if (blas) cublasLtDestroy(blas);
            if (weights) cuMemFree(weights);
            if (half_weights) cuMemFree(half_weights);
            if (module) cuModuleUnload(module);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
            cuDevicePrimaryCtxRelease(device);
        }
    };
    std::shared_ptr<Shared> shared = std::make_shared<Shared>();

    CUdeviceptr weights = 0, half_weights = 0;
    CUstream stream = nullptr;
    CUevent completed = nullptr;
    cudnnHandle_t cudnn = nullptr;
    cublasLtHandle_t blas = nullptr;
    bool pending = false;
    std::filesystem::path cache_dir;
    umzi::Digest cache_environment{}, plan_key{};
    std::string cache_environment_error;

    // State for the prepared padded shape.
    Shape prepared{};
    umzi::CudaPlanCache<umzi::gaterv2_id, plan_schema> plan_cache;
    std::map<umzi::PlanShape, std::unique_ptr<Convolution>> convolutions;
    std::map<umzi::PlanShape, std::unique_ptr<Gemm>> gemms;
    CUdeviceptr arena = 0, norm = 0, gated = 0;
    std::array<CUdeviceptr, max_levels> skips{}, downsampled{}, upsampled{};
    std::array<CUdeviceptr, max_levels + 1> projected{};
    CUdeviceptr latent_x = 0, qkv = 0, key_sum = 0, value_sum = 0, matrix = 0;
    CUdeviceptr key_part = 0, value_part = 0, matrix_part = 0, local_part = 0, multiplier = 0;
    CUdeviceptr stage_in = 0, stage_out = 0, scratch = 0;
    size_t activation_bytes = 0, workspace_bytes = 0, scratch_bytes = 0, peak_workspace_bytes = 0, io_bytes = 0;
    CUgraphExec executable = nullptr;
    CUstream target = nullptr;
    CUdeviceptr input_rgb = 0, output_rgb = 0;
    Shape request{};
    std::array<double, 4> setup_ms{};
    std::array<uint64_t, 4> startup_used{};
    int timed_plans = 0;

    void open(const umzi::ModelPackage& package, const UmziSessionOptions& options) {
        if (package.schema != 1) throw std::runtime_error("Unsupported GateRV2 model payload version");
        if (options.precision != UMZI_MIXED && options.precision != UMZI_FP16)
            throw std::runtime_error("Unsupported GateRV2 session precision");
        fp16 = options.precision == UMZI_FP16;
        parse_options(options);
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
                throw std::runtime_error("Invalid GateRV2 parameter " + std::to_string(offsets32.size()));
            offsets32.push_back(offset);
            end = offset + count * 4;
        }
        if (end != data.size()) throw std::runtime_error("Unexpected trailing GateRV2 weight data");
        weight_bytes = data.size();
        std::vector<uint64_t> package_offsets;
        package_offsets.swap(offsets32);

        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, options.device));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        shared->device = device;
        shared->context = context;
        ContextScope scope(context);
        const auto& image = umzi::load_kernel_module(
            device, module, gaterv2_images::embedded_cubins, gaterv2_images::embedded_ptx);
        shared->module = module;
        for (int i = 0; i < KernelCount; ++i) CHECK(cuModuleGetFunction(&functions[i], module, kernel_names[i]));
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&completed, CU_EVENT_DISABLE_TIMING));
        CHECK(cudnnCreate(&cudnn));
        shared->cudnn = cudnn;
        CHECK(cublasLtCreate(&blas));
        shared->blas = blas;
        upload_weights(data, package_offsets);
        initialize_cache(options.cache_dir, package.digest, image);
    }

    void initialize_cache(const char* directory, const umzi::Digest& model, const umzi::KernelImage& image) {
        if (!directory || !*directory) return;
        try {
            cache_dir = std::filesystem::u8path(directory);
            umzi::Sha256 hash;
            umzi::hash_plan_environment(hash, umzi::gaterv2_id, plan_schema, model, image, device);
            hash.add(sizeof(cublasLtMatmulAlgo_t));
            hash.add(cublasLtGetVersion());
            hash.add(static_cast<uint32_t>(fp16));
            hash.add(static_cast<uint32_t>(tune_plans));
            hash.add(workspace_limit);
            cache_environment = hash.finish();
        } catch (const std::exception& error) {
            cache_environment_error = error.what();
        }
    }

    void load_plan_cache(Shape shape) {
        try {
            if (!cache_environment_error.empty()) throw std::runtime_error(cache_environment_error);
            if (cache_dir.empty()) return;
            umzi::Sha256 hash;
            hash.add(cache_environment);
            hash.add(std::array<int, 3>{shape.n, shape.h, shape.w});
            plan_key = hash.finish();
            plan_cache.open(cache_dir, "gaterv2", plan_key);
        } catch (const std::exception& error) {
            plan_cache.unavailable(error);
        }
    }

    // Kernel options; each value is validated and duplicates are rejected.
    //   tune_plans (0 or 1, default 1): time cuBLASLt and cuDNN candidates when a
    //     shape is prepared, instead of taking the first heuristic choice.
    //   workspace_mib (0 to 1024, default 64): workspace the candidates may use.
    //     Prepare grows the workspace as needed, then retains the selected size.
    void parse_options(const UmziSessionOptions& options) {
        bool seen_tune = false, seen_workspace = false;
        for (uint32_t i = 0; i < options.kernel_option_count; ++i) {
            const auto& option = options.kernel_options[i];
            const std::string name = option.name ? option.name : "";
            if (name == "tune_plans" && !seen_tune) {
                if (option.value != 0 && option.value != 1)
                    throw std::runtime_error("GateRV2 option tune_plans must be 0 or 1");
                tune_plans = option.value != 0;
                seen_tune = true;
            } else if (name == "workspace_mib" && !seen_workspace) {
                if (option.value < 0 || option.value > 1024)
                    throw std::runtime_error("GateRV2 option workspace_mib must be between 0 and 1024");
                workspace_limit = uint64_t(option.value) << 20;
                seen_workspace = true;
            } else if (name == "tune_plans" || name == "workspace_mib") {
                throw std::runtime_error("Duplicate GateRV2 option: " + name);
            } else {
                throw std::runtime_error("Unknown GateRV2 option: " + name + " (expected tune_plans or workspace_mib)");
            }
        }
    }

    // Which copies each parameter needs: 1 = FP32 (read by the CUDA kernels),
    // 2 = FP16 (cuBLASLt projections with their biases, and cuDNN filters).
    std::vector<uint8_t> parameter_use() const {
        const auto& net = *network;
        std::vector<uint8_t> use(net.counts.size(), 0);
        auto fp32 = [&](std::initializer_list<size_t> items) { for (auto i : items) use[i] |= 1; };
        auto fp16 = [&](std::initializer_list<size_t> items) { for (auto i : items) use[i] |= 2; };
        fp32({net.in_w, net.in_b, net.out_w, net.out_b});
        auto meta = [&](const Meta& m) {
            fp32({m.norm_scale, m.norm_offset, m.local_w, m.local_b, m.sca_w, m.sca_b, m.gamma0, m.glob_scale,
                  m.glob_offset, m.square_w, m.square_b, m.row_w, m.row_b, m.column_w, m.column_b, m.gamma1});
            fp16({m.expand_w, m.expand_b, m.fc1_w, m.fc1_b, m.fc2_w, m.fc2_b});
        };
        for (const auto& level : net.encoder) {
            for (const auto& m : level.blocks) meta(m);
            fp16({level.scale_w});
        }
        for (const auto& block : net.latent) {
            fp32({block.scale, block.offset, block.qkv_b});
            fp16({block.fc1_w, block.fc1_b, block.qkv_w, block.fc2_w, block.fc2_b});
        }
        for (const auto& level : net.decoder) {
            fp16({level.scale_w, level.shor_w, level.shor_b});
            for (const auto& m : level.blocks) meta(m);
        }
        return use;
    }

    // Uploads the FP32 and FP16 parameter copies, each packed at 256-byte
    // offsets. The 3x3 down and up filters become KRSC for NHWC cuDNN, and the
    // up filter's outputs are reordered subpixel-major for shuffle_concat.
    void upload_weights(std::string_view data, const std::vector<uint64_t>& package_offsets) {
        const auto& counts = network->counts;
        std::vector<float> values(weight_bytes / 4);
        std::memcpy(values.data(), data.data(), weight_bytes);
        auto to_krsc = [&](size_t index, int outputs, int inputs, auto output_row) {
            float* filter_values = values.data() + package_offsets[index] / 4;
            std::vector<float> original(filter_values, filter_values + size_t(outputs) * inputs * 9);
            for (int o = 0; o < outputs; ++o)
                for (int i = 0; i < inputs; ++i)
                    for (int t = 0; t < 9; ++t)
                        filter_values[(size_t(output_row(o)) * 9 + t) * inputs + i]
                            = original[(size_t(o) * inputs + i) * 9 + t];
        };
        // Each local-gate thread can load the four weights for a channel pair
        // and one tap together. Store adjacent pairs next to one another.
        const auto pack_local_weights = [&](const Level& level) {
            const int channels = level.dim;
            for (const auto& meta : level.blocks) {
                float* filter_values = values.data() + package_offsets[meta.local_w] / 4;
                const std::vector<float> original(filter_values, filter_values + 2 * channels * 18);
                for (int tap = 0; tap < 18; ++tap) {
                    for (int pair = 0; pair < channels / 2; ++pair) {
                        float* packed = filter_values + tap * 2 * channels + 4 * pair;
                        for (int side = 0; side < 2; ++side)
                            for (int output = 0; output < 2; ++output)
                                packed[2 * side + output] = original[(side * channels + 2 * pair + output) * 18 + tap];
                    }
                }
            }
        };
        for (const auto& level : network->encoder) {
            to_krsc(level.scale_w, level.dim / 2, level.dim, [](int o) { return o; });
            pack_local_weights(level);
        }
        for (const auto& level : network->decoder) {
            const int half = level.dim, dim = 2 * half;
            to_krsc(level.scale_w, 2 * dim, dim, [half](int o) { return (o % 4) * half + o / 4; });
            pack_local_weights(level);
        }
        const auto use = parameter_use();
        offsets32.assign(counts.size(), missing);
        offsets16.assign(counts.size(), missing);
        std::vector<float> packed32;
        std::vector<uint16_t> packed16;
        for (size_t i = 0; i < counts.size(); ++i) {
            const float* source = values.data() + package_offsets[i] / 4;
            if (use[i] & 1) {
                packed32.resize((packed32.size() + 63) / 64 * 64);
                offsets32[i] = packed32.size() * 4;
                packed32.insert(packed32.end(), source, source + counts[i]);
            }
            if (use[i] & 2) {
                packed16.resize((packed16.size() + 127) / 128 * 128);
                offsets16[i] = packed16.size() * 2;
                for (uint64_t k = 0; k < counts[i]; ++k) packed16.push_back(umzi::float_to_half(source[k]));
            }
        }
        weight32_bytes = packed32.size() * 4;
        weight16_bytes = packed16.size() * 2;
        CHECK(cuMemAlloc(&weights, std::max<size_t>(weight32_bytes, 256)));
        shared->weights = weights;
        CHECK(cuMemAlloc(&half_weights, std::max<size_t>(weight16_bytes, 256)));
        shared->half_weights = half_weights;
        CHECK(cuMemcpyHtoD(weights, packed32.data(), weight32_bytes));
        CHECK(cuMemcpyHtoD(half_weights, packed16.data(), weight16_bytes));
    }

    ~GateRV2Session() override {
        if (context && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            if (pending) cuEventSynchronize(completed);
            if (stream) cuStreamSynchronize(stream);
            release_shape();
            if (completed) cuEventDestroy(completed);
            if (stream) cuStreamDestroy(stream);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        shared.reset();
    }

    std::unique_ptr<umzi::KernelSession> clone_shared() const override {
        auto copy = std::make_unique<GateRV2Session>();
        copy->shared = shared;
        copy->arch = arch;
        copy->network = std::make_unique<Network>(arch);
        copy->offsets32 = offsets32;
        copy->offsets16 = offsets16;
        copy->fp16 = fp16;
        copy->use_graph = use_graph;
        copy->tune_plans = tune_plans;
        copy->weight_bytes = weight_bytes;
        copy->weight32_bytes = weight32_bytes;
        copy->weight16_bytes = weight16_bytes;
        copy->workspace_limit = workspace_limit;
        copy->device = device;
        copy->context = context;
        copy->module = module;
        copy->functions = functions;
        copy->weights = weights;
        copy->half_weights = half_weights;
        copy->cudnn = cudnn;
        copy->blas = blas;
        copy->cache_dir = cache_dir;
        copy->cache_environment = cache_environment;
        copy->cache_environment_error = cache_environment_error;
        ContextScope scope(context);
        CHECK(cuStreamCreate(&copy->stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&copy->completed, CU_EVENT_DISABLE_TIMING));
        return copy;
    }

    CUdeviceptr param(size_t index) const {
        if (offsets32[index] == missing) throw std::runtime_error("GateRV2 parameter has no FP32 copy");
        return weights + offsets32[index];
    }
    CUdeviceptr filter(size_t index) const {
        if (offsets16[index] == missing) throw std::runtime_error("GateRV2 parameter has no FP16 copy");
        return half_weights + offsets16[index];
    }
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

    // Elements of the largest tensor, the full-resolution fc1 output of three
    // base widths, which bounds the kernels' 32-bit indexing.
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
        executable = nullptr;
        convolutions.clear();
        gemms.clear();
        plan_cache = {};
        for (auto* buffer : {&arena, &local_part, &multiplier, &stage_in, &stage_out, &scratch}) {
            if (*buffer) cuMemFree(*buffer);
            *buffer = 0;
        }
        prepared = {};
        activation_bytes = workspace_bytes = scratch_bytes = peak_workspace_bytes = io_bytes = 0;
        setup_ms = {};
        startup_used = {};
        timed_plans = 0;
    }

    void allocate(CUdeviceptr& pointer, uint64_t bytes) {
        CHECK(cuMemAlloc(&pointer, std::max<uint64_t>(bytes, 256)));
        activation_bytes += bytes;
    }

    void allocate_activations(Shape shape) {
        const uint64_t n = uint64_t(shape.n), pixels = uint64_t(shape.h) * shape.w;
        const uint32_t levels = arch.levels;
        // Keep each cuBLASLt/cuDNN tensor aligned like a standalone CUDA allocation.
        const auto aligned = [](uint64_t bytes) { return (bytes + 255) & ~uint64_t(255); };
        const auto features = [&](uint32_t level) {
            return n * uint64_t(arch.dim(level)) * (pixels >> (2 * level)) * 2;
        };
        uint64_t prefix = 0;
        const auto reserve = [&](uint64_t bytes) {
            const uint64_t offset = prefix;
            prefix += aligned(bytes);
            return offset;
        };
        std::array<uint64_t, max_levels + 1> skip_offsets{}, skip_ends{}, projection_sizes{};
        for (uint32_t level = 0; level <= levels; ++level) {
            skip_offsets[level] = reserve(features(level));
            skip_ends[level] = prefix;
        }
        // The latent attention temporaries live with the latent state. All of
        // them can occupy unused space below the smaller latent projection.
        const uint64_t l = uint64_t(arch.latent_dim()), k = uint64_t(arch.keys());
        const uint64_t t = pixels >> (2 * levels), c = (t + 63) / 64;
        const uint64_t qkv_offset = reserve(n * t * (2 * k + l) * 4);
        const uint64_t key_offset = reserve(n * k * 4);
        const uint64_t value_offset = reserve(n * l * 4);
        const uint64_t matrix_offset = reserve(n * k * l * 4);
        const uint64_t key_part_offset = reserve(c > 1 ? n * c * k * 4 : 0);
        const uint64_t value_part_offset = reserve(c > 1 ? n * c * l * 4 : 0);
        const uint64_t matrix_part_offset = reserve(c > 1 ? n * c * k * l * 4 : 0);
        projection_sizes[levels] = aligned(3 * features(levels));
        uint64_t projection_end = prefix + projection_sizes[levels];
        for (uint32_t level = 0; level < levels; ++level) {
            const uint64_t size = features(level);
            // MetaGated uses a 3C projection. The decoder can use the same
            // window for the upsample convolution (C) and concatenation (2C).
            projection_sizes[level] = std::max(aligned(3 * size), aligned(size) + aligned(2 * size));
            projection_end = std::max(projection_end, skip_ends[level] + projection_sizes[level]);
            // Transitions only use the top of the window, leaving the deeper
            // skip or latent state live until the up/down operation finishes.
            projection_end = std::max(projection_end, skip_ends[level + 1] + aligned(size));
        }
        // NormScaled can replace the local gate input in place. The 48-base
        // architecture keeps a distinct gated buffer since its older norm
        // lane mapping does not guarantee one writer per vector at all levels.
        const bool share_norm = arch.base != 48;
        const uint64_t gate_bytes = aligned(3 * features(0) / 2);
        const uint64_t gate_offset = projection_end + (share_norm ? 0 : aligned(features(0)));
        const uint64_t arena_bytes = gate_offset + gate_bytes;
        allocate(arena, arena_bytes);
        CHECK(cuMemsetD8Async(arena, 0, arena_bytes, stream));
        norm = arena + projection_end;
        gated = arena + gate_offset;
        for (uint32_t level = 0; level < levels; ++level) {
            skips[level] = arena + skip_offsets[level];
            projected[level] = norm - projection_sizes[level];
            downsampled[level] = norm - aligned(features(level + 1));
            upsampled[level] = norm - aligned(features(level));
        }
        projected[levels] = norm - projection_sizes[levels];
        latent_x = arena + skip_offsets[levels];
        qkv = arena + qkv_offset;
        key_sum = arena + key_offset;
        value_sum = arena + value_offset;
        matrix = arena + matrix_offset;
        key_part = c > 1 ? arena + key_part_offset : 0;
        value_part = c > 1 ? arena + value_part_offset : 0;
        matrix_part = c > 1 ? arena + matrix_part_offset : 0;
    }

    void resize_workspace(size_t bytes) {
        if (bytes == scratch_bytes) return;
        if (scratch) {
            CHECK(cuStreamSynchronize(stream));
            CHECK(cuMemFree(scratch));
            scratch = 0;
            scratch_bytes = 0;
        }
        if (bytes) CHECK(cuMemAlloc(&scratch, bytes));
        scratch_bytes = bytes;
        peak_workspace_bytes = std::max(peak_workspace_bytes, bytes);
        if (bytes) startup_used[1] = std::max(startup_used[1], umzi::device_used_bytes());
    }

    void ensure_workspace(size_t bytes) {
        if (bytes > scratch_bytes) resize_workspace(bytes);
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
            startup_used[0] = umzi::device_used_bytes();
            const uint64_t n = uint64_t(p.n), pixels = uint64_t(p.h) * uint64_t(p.w);
            total_elements(p);
            allocate_activations(p);
            prepared = p;
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
            CHECK(cuMemsetD8Async(stage_in, 0, io_bytes, stream));
            request = p;
            startup_used[1] = umzi::device_used_bytes();
            setup_ms[0] = umzi::elapsed_ms(start);

            // The first pass verifies cached plans and selects any missing plans.
            start = Clock::now();
            load_plan_cache(p);
            CHECK(cudnnSetStream(cudnn, stream));
            input_rgb = stage_in;
            output_rgb = stage_out;
            forward(stream);
            CHECK(cuStreamSynchronize(stream));
            startup_used[2] = umzi::device_used_bytes();
            plan_cache.save(plan_key);
            // Keep only the workspace the chosen plans use.
            resize_workspace(workspace_bytes);
            setup_ms[1] = umzi::elapsed_ms(start);

            start = Clock::now();
            forward(stream);
            CHECK(cuStreamSynchronize(stream));
            setup_ms[2] = umzi::elapsed_ms(start);

            start = Clock::now();
            if (use_graph) {
                umzi::capture_cuda_graph(stream, executable, [&] { forward(stream); });
            } else {
                for (auto* buffer : {&stage_in, &stage_out}) {
                    if (*buffer) cuMemFree(*buffer);
                    *buffer = 0;
                }
                input_rgb = output_rgb = 0;
            }
            setup_ms[3] = umzi::elapsed_ms(start);
            startup_used[3] = umzi::device_used_bytes();
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
        const umzi::PlanShape key{cin, cout, rows, input_stride, int(fp32_output)};
        auto& plan = gemms[key];
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
            if (auto saved = plan_cache.gemms.find(key); saved != plan_cache.gemms.end()) {
                const auto& choice = saved->second;
                CHECK(cublasLtMatmulDescSetAttribute(
                    p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer)));
                cublasLtMatmulHeuristicResult_t checked{};
                if (choice.workspace <= workspace_limit
                    && cublasLtMatmulAlgoCheck(blas, p.desc, p.weight, p.input, p.output, p.output,
                                               &choice.algorithm, &checked) == CUBLAS_STATUS_SUCCESS
                    && checked.state == CUBLAS_STATUS_SUCCESS && checked.workspaceSize == choice.workspace) {
                    p.algorithm = choice.algorithm;
                    p.workspace = checked.workspaceSize;
                    ensure_workspace(p.workspace);
                    if (run_gemm(p, out, in, weight, bias_pointer) == CUBLAS_STATUS_SUCCESS) {
                        workspace_bytes = std::max(workspace_bytes, p.workspace);
                        return;
                    }
                }
                const CUdeviceptr no_bias_pointer = 0;
                CHECK(cublasLtMatmulDescSetAttribute(
                    p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &no_bias_pointer, sizeof(no_bias_pointer)));
            }
            plan_cache.record_tuning();
            cublasLtMatmulPreference_t preference = nullptr;
            CHECK(cublasLtMatmulPreferenceCreate(&preference));
            std::unique_ptr<std::remove_pointer_t<cublasLtMatmulPreference_t>,
                            decltype(&cublasLtMatmulPreferenceDestroy)> owner(preference,
                                                                              cublasLtMatmulPreferenceDestroy);
            uint64_t limit = workspace_limit;
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
                if (results[i].state != CUBLAS_STATUS_SUCCESS || results[i].workspaceSize > workspace_limit) continue;
                if (!tune_plans) {
                    chosen = i;  // The heuristic lists its preferred plan first.
                    break;
                }
                p.algorithm = results[i].algo;
                p.workspace = results[i].workspaceSize;
                ensure_workspace(p.workspace);
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
            ensure_workspace(p.workspace);
            workspace_bytes = std::max(workspace_bytes, p.workspace);
            plan_cache.gemms.insert_or_assign(key, umzi::GemmChoice{p.algorithm, p.workspace});
            if (tune_plans) ++timed_plans;
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
        const umzi::PlanShape key{cin, cout, h, w, prepared.n};
        auto& plan = convolutions[key];
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
            if (auto saved = plan_cache.convolutions.find(key); saved != plan_cache.convolutions.end()) {
                const auto& choice = saved->second;
                const auto algo = static_cast<cudnnConvolutionFwdAlgo_t>(choice.algorithm);
                const auto math = static_cast<cudnnMathType_t>(choice.math);
                size_t required = 0;
                if (choice.algorithm < CUDNN_CONVOLUTION_FWD_ALGO_COUNT
                    && choice.math <= CUDNN_FMA_MATH && choice.workspace <= workspace_limit
                    && cudnnSetConvolutionMathType(p.conv, math) == CUDNN_STATUS_SUCCESS
                    && cudnnGetConvolutionForwardWorkspaceSize(cudnn, p.input, p.filter, p.conv, p.output,
                                                               algo, &required) == CUDNN_STATUS_SUCCESS
                    && required == choice.workspace) {
                    p.algorithm = algo;
                    p.workspace = required;
                    ensure_workspace(p.workspace);
                    if (run(p) == CUDNN_STATUS_SUCCESS) {
                        workspace_bytes = std::max(workspace_bytes, p.workspace);
                        return;
                    }
                }
            }
            plan_cache.record_tuning();
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
                if (candidate.status != CUDNN_STATUS_SUCCESS || candidate.memory > workspace_limit) continue;
                if (!tune_plans) {
                    chosen = candidate;  // cuDNN orders its heuristic results by expected speed.
                    best = 0;
                    break;
                }
                CHECK(cudnnSetConvolutionMathType(p.conv, candidate.mathType));
                p.algorithm = candidate.algo;
                p.workspace = candidate.memory;
                ensure_workspace(p.workspace);
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
            ensure_workspace(p.workspace);
            workspace_bytes = std::max(workspace_bytes, p.workspace);
            plan_cache.convolutions.insert_or_assign(
                key, umzi::ConvChoice{uint32_t(chosen.algo), uint32_t(chosen.mathType), p.workspace});
            if (tune_plans) ++timed_plans;
        }
        CHECK(run(*plan));
    }

    // Per-pixel channel work with an optional RMSNorm; see pixel_norm in kernels.cu.
    void pixel_norm(KernelId mode, CUdeviceptr out_x, CUdeviceptr out_norm, CUdeviceptr a, CUdeviceptr b,
                    CUdeviceptr vec, size_t scale, size_t offset, int channels, int pixels, int width) {
        // The kernel masks with lanes - 1 and keeps at most four 8-channel vectors per lane, so lanes must be a
        // power of two that divides the vector count (48-based widths give 6 * 2^k vectors, not a multiple of 4).
        const int vectors = channels / 8;
        int lanes = 1;
        while (vectors > 4 * lanes) lanes *= 2;
        if (channels % 8 || vectors % lanes || lanes > 32)
            throw std::runtime_error("GateRV2 cannot normalize " + std::to_string(channels) + " channels");
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

    // One MetaGated block consumes the current norm and leaves x updated,
    // plus the norm for the next consumer when requested.
    void meta(const Meta& m, CUdeviceptr x, CUdeviceptr expanded, int h, int w, Next next) {
        const int n = prepared.n, dim = m.dim, pixels = h * w, rows = n * pixels, hidden = 3 * dim / 2;
        const bool full_tile = dim == 32 && h == 128 && w == 128;
        gemm(expanded, norm, dim, dim, 2 * dim, rows, m.expand_w, m.expand_b);
        const auto grid = gate_grid(dim, h, w, n);
        const int tiles = int(grid[0] * grid[1]);
        launch3(full_tile ? LocalGate32 : LocalGate, grid, threads, gated, local_part, expanded,
                param(m.local_w), param(m.local_b), dim, h, w);
        launch(PoolProject, unsigned(n), unsigned((dim + 7) / 8), threads, multiplier, local_part, param(m.sca_w),
               param(m.sca_b), param(m.gamma0), dim, tiles, pixels);
        pixel_norm(full_tile ? NormScaled32 : NormScaled, x, norm, gated, x, multiplier, m.glob_scale, m.glob_offset,
                   dim, pixels, w);
        gemm(expanded, norm, dim, dim, 2 * hidden, rows, m.fc1_w, m.fc1_b);
        const int gate_vectors = rows * (hidden / 8);
        if (full_tile) {
            elementwise(InceptionGate32, gate_vectors, gated, expanded, param(m.square_w), param(m.square_b),
                        param(m.row_w), param(m.row_b), param(m.column_w), param(m.column_b));
        } else if (dim % 64 == 0) {
            const uint64_t branch_vectors = uint64_t(rows) * (dim / 64);
            const unsigned direct = unsigned((9 * branch_vectors + threads - 1) / threads);
            const unsigned branch = unsigned((branch_vectors + threads - 1) / threads);
            launch(InceptionAligned, direct + 3 * branch, 1, threads, gated, expanded, param(m.square_w),
                   param(m.square_b), param(m.row_w), param(m.row_b), param(m.column_w), param(m.column_b), dim,
                   h, w, rows, direct, branch);
        } else {
            elementwise(InceptionGate, gate_vectors, gated, expanded, param(m.square_w), param(m.square_b),
                        param(m.row_w), param(m.row_b), param(m.column_w), param(m.column_b), dim, h, w);
        }
        gemm(expanded, gated, hidden, hidden, dim, rows, m.fc2_w, m.fc2_b);
        const KernelId finish = full_tile ? (next.present ? NormMish32 : Mish32) : NormMish;
        pixel_norm(finish, x, next.present ? norm : 0, expanded, x, param(m.gamma1), next.scale, next.offset,
                   dim, pixels, w);
    }

    void latent(const Latent& block, Next next) {
        const CUdeviceptr expanded = projected[arch.levels];
        const int n = prepared.n, dim = arch.latent_dim(), keys = arch.keys(), t = tokens();
        const int rows = n * t, hidden = 3 * dim / 2, stride = 2 * keys + dim;
        gemm(expanded, norm, dim, dim, 2 * hidden, rows, block.fc1_w, block.fc1_b);
        gemm(qkv, expanded + uint64_t(2 * hidden - dim) * 2, 2 * hidden, dim, stride, rows, block.qkv_w,
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
        elementwise(AttentionGate, rows * (hidden / 8), gated, expanded, qkv, key_sum, value_sum, matrix, keys, dim,
                    t);
        gemm(expanded, gated, hidden, hidden, dim, rows, block.fc2_w, block.fc2_b);
        pixel_norm(NormMish, latent_x, next.present ? norm : 0, expanded, CUdeviceptr(0), CUdeviceptr(0), next.scale,
                   next.offset, dim, t, 1);
    }

    Next first_norm(const Level& level) const {
        if (level.blocks.empty()) return {};
        return {level.blocks.front().norm_scale, level.blocks.front().norm_offset, true};
    }

    void blocks(const Level& level, CUdeviceptr x, CUdeviceptr expanded, int h, int w, Next after) {
        for (size_t i = 0; i < level.blocks.size(); ++i) {
            const Next next = i + 1 < level.blocks.size() ? first_norm_of(level.blocks[i + 1]) : after;
            meta(level.blocks[i], x, expanded, h, w, next);
        }
    }

    static Next first_norm_of(const Meta& m) { return {m.norm_scale, m.norm_offset, true}; }

    // Unpadded RGB in input_rgb to the cropped RGB output in output_rgb.
    void forward(CUstream stream_target) {
        target = stream_target;
        const auto& net = *network;
        const int n = prepared.n, height = prepared.h, width = prepared.w;
        const int h0 = request.h, w0 = request.w;
        const Next level0 = first_norm(net.encoder[0]);
        const int base = int(arch.base);
        elementwise(fp16 ? InputF16 : InputF32, n * height * width, skips[0], level0.present ? norm : 0,
                    input_rgb, param(net.in_w), param(net.in_b), level0.present ? param(level0.scale) : 0,
                    level0.present ? param(level0.offset) : 0, base, h0, w0, height, width,
                    float(1.0 / std::sqrt(double(base))));
        for (uint32_t level = 0; level < arch.levels; ++level) {
            const auto& item = net.encoder[level];
            const int h = height >> level, w = width >> level, dim = item.dim;
            blocks(item, skips[level], projected[level], h, w, {});
            conv3x3(downsampled[level], skips[level], dim, dim / 2, h, w, item.scale_w);
            const bool deepest = level + 1 == arch.levels;
            const Next next = deepest ? Next{net.latent[0].scale, net.latent[0].offset, true}
                                      : first_norm(net.encoder[level + 1]);
            pixel_norm(NormUnshuffle, deepest ? latent_x : skips[level + 1], next.present ? norm : 0, downsampled[level],
                       CUdeviceptr(0), CUdeviceptr(0), next.scale, next.offset, 2 * dim, (h / 2) * (w / 2), w / 2);
        }
        for (size_t i = 0; i < net.latent.size(); ++i) {
            const Next next = i + 1 < net.latent.size() ? Next{net.latent[i + 1].scale, net.latent[i + 1].offset, true}
                                                         : Next{};
            latent(net.latent[i], next);
        }
        const int low_h = height >> arch.levels, low_w = width >> arch.levels;
        for (uint32_t i = 0; i < arch.levels; ++i) {
            const uint32_t level = arch.levels - 1 - i;
            const auto& item = net.decoder[i];
            const int half = item.dim, dim = 2 * half, h = low_h << i, w = low_w << i;
            // The decoder state replaces the skip it consumes, which has the same shape.
            const CUdeviceptr source = i ? skips[level + 1] : latent_x;
            const CUdeviceptr skip = skips[level], up = upsampled[level], concat = projected[level];
            conv3x3(up, source, dim, 2 * dim, h, w, item.scale_w);
            const int pixels = 4 * h * w;
            elementwise(ShuffleConcat, n * pixels * (dim / 8), concat, up, skip, half, 2 * h, 2 * w);
            gemm(skip, concat, dim, dim, half, n * pixels, item.shor_w, item.shor_b);
            const Next first = first_norm(item);
            if (first.present)
                pixel_norm(NormCopy, CUdeviceptr(0), norm, skip, CUdeviceptr(0), CUdeviceptr(0), first.scale,
                           first.offset, half, pixels, 2 * w);
            blocks(item, skip, concat, 2 * h, 2 * w, {});
        }
        elementwise(fp16 ? OutputF16 : OutputF32, n * h0 * w0, output_rgb, skips[0], param(net.out_w),
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
    umzi::TilingConstraints tiling_constraints() const override { return {int(arch.alignment), 1, 0}; }

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
            const auto start = Clock::now();
            cuGraphExecDestroy(executable);
            executable = nullptr;
            request = shape;
            input_rgb = stage_in;
            output_rgb = stage_out;
            CHECK(cudnnSetStream(cudnn, stream));
            umzi::capture_cuda_graph(stream, executable, [&] { forward(stream); });
            setup_ms[3] = umzi::elapsed_ms(start);
            startup_used[3] = umzi::device_used_bytes();
        }
    }

    void prepare_shape(const UmziShape& shape) override { prepare_exact(checked(shape)); }

    void ensure_staging() {
        if (stage_in && stage_out) return;
        // Allocate only what is missing: a failed second allocation must not leave a half-built pair that passes the early return.
        if (!stage_out) CHECK(cuMemAlloc(&stage_out, io_bytes));
        if (!stage_in) CHECK(cuMemAlloc(&stage_in, io_bytes));
    }

    void run_host(const UmziShape& shape, const void* input, void* output) override {
        const auto concrete = checked(shape);
        prepare_exact(concrete);
        ContextScope scope(context);
        ensure_staging();
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
        result.cache_status = plan_cache.status;
        result.buffer_bytes[0] = weight32_bytes + weight16_bytes;
        if (prepared.n) {
            result.prepared_input = {4, {uint32_t(prepared.n), 3, uint32_t(prepared.h), uint32_t(prepared.w)}};
            result.buffer_bytes[1] = activation_bytes;
            result.buffer_bytes[2] = workspace_bytes;
            result.buffer_bytes[3] = stage_in ? 2 * io_bytes : 0;
            std::copy(startup_used.begin(), startup_used.end(), result.startup_device_used_bytes);
            std::copy(setup_ms.begin(), setup_ms.end(), result.shape_setup_ms);
        }
        return result;
    }

    uint32_t diagnostic_count() const override { return 5; }
    UmziDiagnostic diagnostic(uint32_t index) const override {
        if (index == 0) return {sizeof(UmziDiagnostic), "timed_plans", timed_plans};
        if (index == 1) return {sizeof(UmziDiagnostic), "plan_workspace_bytes", int64_t(workspace_bytes)};
        if (index == 2) return {sizeof(UmziDiagnostic), "fp32_weight_bytes", int64_t(weight32_bytes)};
        if (index == 3) return {sizeof(UmziDiagnostic), "fp16_weight_bytes", int64_t(weight16_bytes)};
        if (index == 4) return {sizeof(UmziDiagnostic), "peak_plan_workspace_bytes", int64_t(peak_workspace_bytes)};
        throw std::runtime_error("Invalid GateRV2 diagnostic index");
    }

    const char* cache_message() const override { return plan_cache.diagnostic.c_str(); }

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
