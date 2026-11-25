#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <cuda.h>
#include <cudnn.h>

#include "gfisrv2_images.h"
#include "cufft_api.h"
#include "../../common/cuda_helpers.h"
#include "../../common/cuda_module.h"
#include "../../common/cuda_plan_cache.h"
#include "../../common/cudnn_conv_plan.h"
#include "../../runtime/binary.h"
#include "../../runtime/kernel.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr int channels = 48, blocks = 24, scale = 4, hidden = 128, fourier = 30, band = 6, threads = 256;
constexpr int rgb_lanes = 8, freq_lanes = 64;
constexpr int expand_shared_bytes = (3 * 180 * 16 + 2 * 3 * 128 * 16) * 2; // 41,856 B
constexpr int project_shared_bytes = (2 * 324 * 16 + 2 * 3 * 48 * 16) * 2; // 29,952 B
constexpr uint64_t default_workspace_mib = 512, max_workspace_mib = 4096;
constexpr uint32_t plan_schema = 4;
constexpr uint32_t saved_plan_marker = 0x4742;
constexpr std::array<uint32_t, 7> configuration{3, 3, scale, channels, blocks, hidden, fourier};
constexpr int r2c = 0x2a, c2r = 0x2c;
using umzi::Clock;
using umzi::ContextScope;

void fft_check(int status, const char* operation) {
    if (status == 2)
        throw umzi::AllocationError(std::string(operation) + ": cuFFT allocation failed; try a smaller tile or batch");
    if (status) throw std::runtime_error(std::string(operation) + ": cuFFT status " + std::to_string(status));
}
#define CHECK(expr) umzi::check((expr), #expr)
#define FFT_CHECK(expr) fft_check((expr), #expr)

struct Shape {
    int n = 0, h = 0, w = 0;
    bool operator==(Shape other) const { return n == other.n && h == other.h && w == other.w; }
    int pixels() const { return h * w; }
    uint64_t rgb_bytes(size_t element) const { return uint64_t(n) * pixels() * 3 * element; }
    UmziShape nchw() const { return {4, {uint32_t(n), 3, uint32_t(h), uint32_t(w)}}; }
};

Shape checked(const UmziShape& shape) {
    const auto n = shape.dimensions[0], h = shape.dimensions[2], w = shape.dimensions[3];
    if (shape.rank != 4 || shape.dimensions[1] != 3 || !n || !h || !w || n > 4096 || h > 65536 || w > 65536)
        throw std::runtime_error("GFISRv2 expects positive RGB NCHW dimensions within the native indexing limits");
    if (uint64_t(n) * h * w * 2 * hidden > INT_MAX - threads)
        throw std::runtime_error("Requested GFISRv2 shape exceeds native tensor indexing limits");
    return {int(n), int(h), int(w)};
}

struct Norm { size_t scale, offset; };
struct Filter { size_t weight, bias; };
struct Fourier { Norm pre; Filter position, project; Norm post; };
struct Block { Norm norm; Filter expand; Fourier fft; std::array<Filter, 3> bands; Filter project; size_t gamma; };

// Section 3 stores FP32 OIHW weights and affine parameters at 256-byte offsets.
struct Network {
    size_t bytes = 0;
    Filter first, tail_expand, tail_project, last;
    std::array<Block, blocks> body{};

    Network() {
        first = conv(3, channels);
        for (auto& block : body) {
            block.norm = norm(channels);
            block.expand = conv(channels, 2 * hidden);
            block.fft = {norm(2 * fourier), conv(1, 2 * fourier), conv(2 * fourier, 2 * fourier, 1), norm(fourier)};
            block.bands = {conv(1, band), conv(1, band, 11), conv(1, band, 11)};
            block.project = conv(hidden, channels);
            block.gamma = take(channels);
        }
        tail_expand = conv(channels, 2 * channels);
        tail_project = conv(2 * channels, channels);
        last = conv(channels, 3 * scale * scale);
    }

    size_t take(size_t count) {
        const size_t offset = (bytes + 255) & ~size_t(255);
        bytes = offset + count * sizeof(float);
        return offset;
    }
    Norm norm(int c) { return {take(c), take(c)}; }
    Filter conv(int cin, int cout, int taps = 9) { return {take(size_t(cin) * cout * taps), take(cout)}; }
};

struct DeviceNetwork {
    size_t bytes = 0;
    Filter first, tail_expand, tail_project, last;
    std::array<Block, blocks> body{};

    explicit DeviceNetwork(bool spectral_half) {
        first = dense_half(rgb_lanes, channels, 9);
        for (auto& block : body) {
            block.norm = norm(channels);
            block.expand = dense_half(channels, 2 * hidden, 9);
            block.fft = {
                norm(2 * fourier),
                {take_bytes(size_t(2 * fourier) * 9 * 4), take_bytes(size_t(2 * fourier) * 4)},
                spectral_half ? dense_half(freq_lanes, freq_lanes, 1) : dense_float(freq_lanes, freq_lanes, 1),
                norm(fourier),
            };
            block.bands = {{
                Filter{take_bytes(size_t(band) * 9 * 4), take_bytes(size_t(band) * 4)},
                Filter{take_bytes(size_t(band) * 11 * 4), take_bytes(size_t(band) * 4)},
                Filter{take_bytes(size_t(band) * 11 * 4), take_bytes(size_t(band) * 4)},
            }};
            block.project = dense_half(hidden, channels, 9);
            block.gamma = take_bytes(size_t(channels) * 4);
        }
        tail_expand = dense_half(channels, 2 * channels, 9);
        tail_project = dense_half(2 * channels, channels, 9);
        last = dense_half(channels, 3 * scale * scale, 9);
    }

    size_t take_bytes(size_t count) {
        const size_t offset = (bytes + 255) & ~size_t(255);
        bytes = offset + count;
        return offset;
    }
    Norm norm(int c) { return {take_bytes(size_t(c) * 4), take_bytes(size_t(c) * 4)}; }
    Filter dense_half(int cin, int cout, int taps) {
        return {take_bytes(size_t(cin) * cout * taps * 2), take_bytes(size_t(cout) * 4)};
    }
    Filter dense_float(int cin, int cout, int taps) {
        return {take_bytes(size_t(cin) * cout * taps * 4), take_bytes(size_t(cout) * 4)};
    }
};

std::array<int, hidden> block_canon(int block_index) {
    std::array<int, hidden> map{};
    for (int k = 0; k < hidden - channels; ++k) map[k] = k;
    int offset = 0;
    for (int slot = 0; slot < 4; ++slot) {
        const int kind = (slot + block_index) % 4;
        if (kind == 0) {
            for (int c = 0; c < fourier; ++c) map[80 + offset + c] = 80 + c;
            offset += fourier;
        } else {
            const int base = 80 + fourier + (kind - 1) * band;
            for (int c = 0; c < band; ++c) map[80 + offset + c] = base + c;
            offset += band;
        }
    }
    return map;
}

template<class OutSlot, class InSlot>
void pack_oihw_to_krsc_half(uint16_t* dst_w, float* dst_b, const char* in, Filter from, int cin, int cout, int to_cin,
                            int taps, OutSlot out_slot, InSlot in_slot) {
    const auto* src_w = reinterpret_cast<const float*>(in + from.weight);
    const auto* src_b = reinterpret_cast<const float*>(in + from.bias);
    for (int o = 0; o < cout; ++o) {
        const int target_o = out_slot(o);
        dst_b[target_o] = src_b[o];
        for (int i = 0; i < cin; ++i) {
            const int target_i = in_slot(i);
            for (int tap = 0; tap < taps; ++tap) {
                const float w = src_w[(size_t(o) * cin + i) * taps + tap];
                dst_w[(size_t(target_o) * taps + tap) * to_cin + target_i] = umzi::float_to_half(w);
            }
        }
    }
}

void pack_oihw_to_krsc_float(float* dst_w, float* dst_b, const char* in, Filter from, int cin, int cout, int to_cin,
                             int taps) {
    const auto* src_w = reinterpret_cast<const float*>(in + from.weight);
    const auto* src_b = reinterpret_cast<const float*>(in + from.bias);
    for (int o = 0; o < cout; ++o) {
        dst_b[o] = src_b[o];
        for (int i = 0; i < cin; ++i) {
            for (int tap = 0; tap < taps; ++tap) {
                dst_w[(size_t(o) * taps + tap) * to_cin + i] = src_w[(size_t(o) * cin + i) * taps + tap];
            }
        }
    }
}

// Pack logical [256, 9, 48] KRSC FP16 -> [2 passes][3 ks][3 dy][3 dx][128 rows][2 chunks][8 halves] swizzle16.
void pack_swizzle16_expand(char* dst, const uint16_t* logical) {
    constexpr int ntile = 128;
    auto* out = reinterpret_cast<uint16_t*>(dst);
    size_t stage = 0;
    for (int pass = 0; pass < 2; ++pass) {
        for (int ks = 0; ks < 3; ++ks) {
            for (int dy = 0; dy < 3; ++dy) {
                uint16_t* stage_dst = out + stage * (3 * ntile * 16);
                for (int dx = 0; dx < 3; ++dx) {
                    const int tap = dy * 3 + dx;
                    uint16_t* tap_dst = stage_dst + dx * (ntile * 16);
                    for (int row = 0; row < ntile; ++row) {
                        const int ch = pass * ntile + row;
                        const int mask = (row >> 2) & 1;
                        for (int chunk = 0; chunk < 2; ++chunk) {
                            const size_t d = (size_t(row) * 2 + (chunk ^ mask)) * 8;
                            const size_t s = (size_t(ch) * 9 + tap) * channels + ks * 16 + chunk * 8;
                            std::memcpy(tap_dst + d, logical + s, 16);
                        }
                    }
                }
                ++stage;
            }
        }
    }
}

// Pack logical [48, 9, 128] KRSC FP16 -> [8 ks][3 dy][3 dx][48 rows][2 chunks][8 halves] swizzle16.
void pack_swizzle16_project(char* dst, const uint16_t* logical) {
    auto* out = reinterpret_cast<uint16_t*>(dst);
    size_t stage = 0;
    for (int ks = 0; ks < hidden / 16; ++ks) {
        for (int dy = 0; dy < 3; ++dy) {
            uint16_t* stage_dst = out + stage * (3 * channels * 16);
            for (int dx = 0; dx < 3; ++dx) {
                const int tap = dy * 3 + dx;
                uint16_t* tap_dst = stage_dst + dx * (channels * 16);
                for (int row = 0; row < channels; ++row) {
                    const int mask = (row >> 2) & 1;
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        const size_t d = (size_t(row) * 2 + (chunk ^ mask)) * 8;
                        const size_t s = (size_t(row) * 9 + tap) * hidden + ks * 16 + chunk * 8;
                        std::memcpy(tap_dst + d, logical + s, 16);
                    }
                }
            }
            ++stage;
        }
    }
}

// Pack logical [64, 64] FP16 -> [4 ks][64 rows][2 chunks][8 halves] swizzle16.
void pack_swizzle16_spectral_f16(char* dst, const uint16_t* logical) {
    auto* out = reinterpret_cast<uint16_t*>(dst);
    for (int ks = 0; ks < 4; ++ks) {
        uint16_t* slice_dst = out + ks * (freq_lanes * 16);
        for (int row = 0; row < freq_lanes; ++row) {
            const int mask = (row >> 2) & 1;
            for (int chunk = 0; chunk < 2; ++chunk) {
                const size_t d = (size_t(row) * 2 + (chunk ^ mask)) * 8;
                const size_t s = size_t(row) * freq_lanes + ks * 16 + chunk * 8;
                std::memcpy(slice_dst + d, logical + s, 16);
            }
        }
    }
}

std::vector<char> build_device_weights(const Network& from, const DeviceNetwork& to, const char* in,
                                       bool spectral_half, bool fused_conv) {
    std::vector<char> out(to.bytes, 0);
    const auto copy_raw = [&](size_t dst, size_t src, size_t bytes) {
        std::memcpy(out.data() + dst, in + src, bytes);
    };
    const auto id = [](int c) { return c; };
    const auto w_half = [&](Filter f) { return reinterpret_cast<uint16_t*>(out.data() + f.weight); };
    const auto w_float = [&](Filter f) { return reinterpret_cast<float*>(out.data() + f.weight); };
    const auto b_float = [&](Filter f) { return reinterpret_cast<float*>(out.data() + f.bias); };

    pack_oihw_to_krsc_half(w_half(to.first), b_float(to.first), in, from.first, 3, channels, rgb_lanes, 9, id, id);
    for (int i = 0; i < blocks; ++i) {
        const auto& src = from.body[i];
        const auto& dst = to.body[i];
        const auto canon = block_canon(i);
        copy_raw(dst.norm.scale, src.norm.scale, channels * 4);
        copy_raw(dst.norm.offset, src.norm.offset, channels * 4);

        if (fused_conv) {
            // Uniform [gate x4 | val x4] grouping across all 32 groups (0..19 plain, 20..31 Fourier/branch).
            std::vector<uint16_t> logical(size_t(2 * hidden) * 9 * channels, 0);
            const auto fused_expand_slot = [&](int o) {
                const bool is_value = o >= hidden;
                const int k = is_value ? o - hidden : o;
                const int c = canon[k];
                return 8 * (c / 4) + (is_value ? 4 : 0) + (c % 4);
            };
            pack_oihw_to_krsc_half(logical.data(), b_float(dst.expand), in, src.expand, channels, 2 * hidden, channels,
                                   9, fused_expand_slot, id);
            pack_swizzle16_expand(out.data() + dst.expand.weight, logical.data());
        } else {
            const auto unfused_expand_slot = [&](int o) {
                const bool is_value = o >= hidden;
                const int k = is_value ? o - hidden : o;
                const int c = canon[k];
                if (c < 80) return 8 * (c / 4) + (is_value ? 4 : 0) + (c % 4);
                return (is_value ? 208 : 160) + (c - 80);
            };
            pack_oihw_to_krsc_half(w_half(dst.expand), b_float(dst.expand), in, src.expand, channels, 2 * hidden,
                                   channels, 9, unfused_expand_slot, id);
        }

        copy_raw(dst.fft.pre.scale, src.fft.pre.scale, 2 * fourier * 4);
        copy_raw(dst.fft.pre.offset, src.fft.pre.offset, 2 * fourier * 4);
        copy_raw(dst.fft.position.weight, src.fft.position.weight, size_t(2 * fourier) * 9 * 4);
        copy_raw(dst.fft.position.bias, src.fft.position.bias, size_t(2 * fourier) * 4);
        if (spectral_half) {
            std::vector<uint16_t> logical(size_t(freq_lanes) * freq_lanes, 0);
            pack_oihw_to_krsc_half(logical.data(), b_float(dst.fft.project), in, src.fft.project, 2 * fourier,
                                   2 * fourier, freq_lanes, 1, id, id);
            pack_swizzle16_spectral_f16(out.data() + dst.fft.project.weight, logical.data());
        } else {
            pack_oihw_to_krsc_float(w_float(dst.fft.project), b_float(dst.fft.project), in, src.fft.project,
                                    2 * fourier, 2 * fourier, freq_lanes, 1);
        }
        copy_raw(dst.fft.post.scale, src.fft.post.scale, fourier * 4);
        copy_raw(dst.fft.post.offset, src.fft.post.offset, fourier * 4);

        for (int b = 0; b < 3; ++b) {
            const int taps = b == 0 ? 9 : 11;
            copy_raw(dst.bands[b].weight, src.bands[b].weight, size_t(band) * taps * 4);
            copy_raw(dst.bands[b].bias, src.bands[b].bias, size_t(band) * 4);
        }

        if (fused_conv) {
            std::vector<uint16_t> logical(size_t(channels) * 9 * hidden, 0);
            pack_oihw_to_krsc_half(logical.data(), b_float(dst.project), in, src.project, hidden, channels, hidden, 9,
                                   id, [&](int k) { return canon[k]; });
            pack_swizzle16_project(out.data() + dst.project.weight, logical.data());
        } else {
            pack_oihw_to_krsc_half(w_half(dst.project), b_float(dst.project), in, src.project, hidden, channels, hidden,
                                   9, id, [&](int k) { return canon[k]; });
        }
        copy_raw(dst.gamma, src.gamma, channels * 4);
    }
    pack_oihw_to_krsc_half(w_half(to.tail_expand), b_float(to.tail_expand), in, from.tail_expand, channels,
                           2 * channels, channels, 9, id, id);
    pack_oihw_to_krsc_half(w_half(to.tail_project), b_float(to.tail_project), in, from.tail_project, 2 * channels,
                           channels, 2 * channels, 9, id, id);
    pack_oihw_to_krsc_half(w_half(to.last), b_float(to.last), in, from.last, channels, 3 * scale * scale, channels, 9,
                           id, id);
    return out;
}

struct LegacyConvolution {
    cudnnTensorDescriptor_t input = nullptr, output = nullptr;
    cudnnFilterDescriptor_t filter = nullptr;
    cudnnConvolutionDescriptor_t convolution = nullptr;
    cudnnConvolutionFwdAlgo_t algorithm{};
    size_t workspace = 0;
    ~LegacyConvolution() {
        if (convolution) cudnnDestroyConvolutionDescriptor(convolution);
        if (filter) cudnnDestroyFilterDescriptor(filter);
        if (output) cudnnDestroyTensorDescriptor(output);
        if (input) cudnnDestroyTensorDescriptor(input);
    }
};

struct FftPlan {
    int handle = 0;
    bool created = false;
    CUdeviceptr workspace = 0;
    size_t bytes = 0;

    void setup(const umzi::CufftApi& api, Shape shape, int transform) {
        FFT_CHECK(api.create(&handle));
        created = true;
        FFT_CHECK(api.auto_allocation(handle, 0));
        int sizes[] = {shape.h, shape.w};
        FFT_CHECK(api.make_plan(handle, 2, sizes, nullptr, 1, 0, nullptr, 1, 0, transform, shape.n * fourier, &bytes));
        if (bytes) {
            CHECK(cuMemAlloc(&workspace, bytes));
            FFT_CHECK(api.work_area(handle, reinterpret_cast<void*>(workspace)));
        }
    }

    void release(const umzi::CufftApi& api) {
        if (created) api.destroy(handle);
        if (workspace) cuMemFree(workspace);
        *this = {};
    }
};

constexpr const char* kernel_names[] = {
    // Legacy NCHW FP32 kernels (unchanged for bit-exact FP32).
    "rms_norm", "depthwise", "copy_channels", "pack_fft", "unpack_fft", "scale_fft", "add_bias",
    "gate", "residual", "apply_silu", "add_features", "shuffle_rgb",
    // Optimized NHWC kernels (FP16 and Mixed).
    "pack_rgb_nhwc_f16", "pack_rgb_nhwc_f32",
    "input_bias_rms_nhwc_f16", "input_bias_rms_nhwc_mixed",
    "residual_rms_nhwc_f16", "residual_nhwc_f16",
    "residual_rms_nhwc_mixed", "residual_to_half_nhwc_mixed",
    "gate_plain_nhwc", "extract_fourier_nhwc", "branch_dw_gate_nhwc",
    "pack_rms_fft_nhwc_f16", "pack_rms_fft_nhwc_f32",
    "spectral_dw_res_nhwc_f16", "spectral_dw_res_nhwc_f32",
    "scale_post_rms_gate_nhwc",
    "bias_silu_nhwc_96", "tail_add_nhwc_f16", "tail_add_nhwc_mixed",
    "bias_shuffle_rgb_nhwc_f16", "bias_shuffle_rgb_nhwc_f32",
    "fill_noise_f16", "fill_noise_f32",
    // Pass 2 fused tensor-core and compact kernels.
    "expand_gate_f16", "expand_gate_f32",
    "branch_dw_gate_compact", "scale_post_rms_gate_compact",
    "project_res_norm_f16", "project_res_f16",
    "project_res_norm_f32", "project_res_to_half_f32",
    "spectral_proj_gelu_unpack_f16", "spectral_proj_gelu_unpack_f32",
};
enum Kernel {
    Rms, Depthwise, CopyChannels, Pack, Unpack, ScaleFft, Bias, Gate, Residual, Silu, Add, Shuffle,
    PackRgbF16, PackRgbF32,
    InputBiasRmsF16, InputBiasRmsMixed,
    ResidualRmsF16, ResidualF16,
    ResidualRmsMixed, ResidualToHalfMixed,
    GatePlainNhwc, ExtractFourierNhwc, BranchDwGateNhwc,
    PackRmsFftF16, PackRmsFftF32,
    SpectralDwResF16, SpectralDwResF32,
    ScalePostRmsGateNhwc,
    BiasSilu96, TailAddF16, TailAddMixed,
    BiasShuffleRgbF16, BiasShuffleRgbF32,
    FillNoiseF16, FillNoiseF32,
    ExpandGateF16, ExpandGateF32,
    BranchDwGateCompact, ScalePostRmsGateCompact,
    ProjectResNormF16, ProjectResF16,
    ProjectResNormF32, ProjectResToHalfF32,
    SpectralProjGeluUnpackF16, SpectralProjGeluUnpackF32,
    KernelCount
};
static_assert(std::size(kernel_names) == KernelCount);

class Session final : public umzi::KernelSession {
    Network model;
    DeviceNetwork dev_model{false};
    bool graph_enabled = false, pending = false, unexact_erf = false, fused_conv = true;
    uint32_t precision = UMZI_FP32;
    uint64_t workspace_limit = default_workspace_mib << 20;
    struct Shared {
        CUdevice device = 0;
        CUcontext context = nullptr;
        CUmodule module = nullptr;
        cudnnHandle_t cudnn = nullptr;
        std::shared_ptr<umzi::CufftApi> fft;
        CUdeviceptr weights = 0;

        ~Shared() {
            if (!context) return;
            const umzi::PrimaryContextRelease release{device};
            if (cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
            if (cudnn) cudnnDestroy(cudnn);
            if (weights) cuMemFree(weights);
            if (module) cuModuleUnload(module);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
    };
    std::shared_ptr<Shared> shared = std::make_shared<Shared>();

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUstream stream = nullptr;
    CUevent complete = nullptr;
    std::array<CUevent, 2> timing{};
    cudnnHandle_t cudnn = nullptr;
    std::shared_ptr<umzi::CufftApi> fft;
    int fft_version = 0;
    std::array<CUfunction, KernelCount> functions{};
    CUdeviceptr weights = 0;
    size_t weight_bytes = 0;
    // Weights repacked without fused_conv after a fallback; shared with clones.
    std::shared_ptr<umzi::DeviceAllocation> unfused_weights;

    Shape shape{};
    int bins = 0;
    float fft_scale = 0;
    CUdeviceptr stage_in = 0, stage_out = 0, cudnn_scratch = 0;
    CUdeviceptr initial = 0, state = 0, norm = 0, projected = 0, gated = 0;
    CUdeviceptr fft_buf = 0, spectrum = 0, freq_a = 0, freq_b = 0;
    CUgraphExec executable = nullptr;
    FftPlan forward_fft, inverse_fft;

    using LegacyKey = std::array<int, 5>;
    std::map<LegacyKey, std::unique_ptr<LegacyConvolution>> legacy_plans;
    std::map<umzi::PlanShape, std::unique_ptr<umzi::CudnnBackendPlan>> backend_plans;
    std::map<std::array<CUdeviceptr, 4>, std::unique_ptr<umzi::CudnnDescriptor>> variants;
    umzi::CudaPlanCache<umzi::gfisrv2_id, plan_schema> plan_cache;
    std::filesystem::path cache_dir;
    umzi::Digest cache_environment{}, plan_key{};
    std::string cache_environment_error;

    size_t activation_bytes = 0, cudnn_workspace = 0, scratch_bytes = 0, peak_workspace_bytes = 0;
    int timed_plans = 0;
    std::array<double, 4> setup_ms{};
    std::array<uint64_t, 4> device_used{};

public:
    void open(const umzi::ModelPackage& package, const UmziSessionOptions& options) {
        if (package.schema != 1) throw std::runtime_error("Unsupported GFISRv2 model payload version");
        umzi::BinaryReader architecture(package.section(1));
        for (auto value : configuration)
            if (architecture.number<uint32_t>() != value) throw std::runtime_error("Unsupported GFISRv2 model architecture");
        if (architecture.size() || package.section(4).size() != 32)
            throw std::runtime_error("Invalid GFISRv2 metadata");
        const auto data = package.section(3);
        if (!package.section(2).empty() || data.size() != model.bytes)
            throw std::runtime_error("Invalid GFISRv2 weight layout");
        precision = options.precision;
        bool seen_erf = false, seen_ws = false, seen_fused = false;
        for (uint32_t i = 0; i < options.kernel_option_count; ++i) {
            const auto& option = options.kernel_options[i];
            const std::string_view name = option.name ? option.name : "";
            if (name == "unexact_erf") {
                if (seen_erf) throw std::runtime_error("Duplicate GFISRv2 option: unexact_erf");
                if (option.value != 0 && option.value != 1)
                    throw std::runtime_error("GFISRv2 unexact_erf must be 0 or 1");
                unexact_erf = option.value == 1;
                seen_erf = true;
            } else if (name == "workspace_mib") {
                if (seen_ws) throw std::runtime_error("Duplicate GFISRv2 option: workspace_mib");
                if (option.value < 0 || uint64_t(option.value) > max_workspace_mib)
                    throw std::runtime_error("GFISRv2 workspace_mib must be between 0 and 4096");
                workspace_limit = uint64_t(option.value) << 20;
                seen_ws = true;
            } else if (name == "fused_conv") {
                if (seen_fused) throw std::runtime_error("Duplicate GFISRv2 option: fused_conv");
                if (option.value != 0 && option.value != 1)
                    throw std::runtime_error("GFISRv2 fused_conv must be 0 or 1");
                fused_conv = option.value == 1;
                seen_fused = true;
            } else {
                throw std::runtime_error("Unknown GFISRv2 option: " + std::string(name));
            }
        }
        graph_enabled = (options.flags & UMZI_CUDA_GRAPH) != 0;
        dev_model = DeviceNetwork(half_io());

        fft = std::make_shared<umzi::CufftApi>();
        shared->fft = fft;
        FFT_CHECK(fft->version(&fft_version));
        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, options.device));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        shared->device = device;
        shared->context = context;
        ContextScope scope(context);

        const auto& image =
            umzi::load_kernel_module(device, module, gfisrv2_images::embedded_cubins, gfisrv2_images::embedded_ptx);
        shared->module = module;
        for (int i = 0; i < KernelCount; ++i) {
            const char* name = kernel_names[i];
            if (unexact_erf && i == Unpack) name = "unpack_fft_unexact";
            if (unexact_erf && i == SpectralProjGeluUnpackF16) name = "spectral_proj_gelu_unpack_f16_unexact";
            if (unexact_erf && i == SpectralProjGeluUnpackF32) name = "spectral_proj_gelu_unpack_f32_unexact";
            CHECK(cuModuleGetFunction(&functions[i], module, name));
        }
        for (auto [kernel, shared] : {std::pair{ExpandGateF16, expand_shared_bytes},
                                      std::pair{ExpandGateF32, expand_shared_bytes},
                                      std::pair{ProjectResNormF16, project_shared_bytes},
                                      std::pair{ProjectResF16, project_shared_bytes},
                                      std::pair{ProjectResNormF32, project_shared_bytes},
                                      std::pair{ProjectResToHalfF32, project_shared_bytes}})
            CHECK(cuFuncSetAttribute(functions[kernel], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));

        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&complete, CU_EVENT_DISABLE_TIMING));
        CHECK(cuEventCreate(&timing[0], CU_EVENT_DEFAULT));
        CHECK(cuEventCreate(&timing[1], CU_EVENT_DEFAULT));
        CHECK(cudnnCreate(&cudnn));
        shared->cudnn = cudnn;

        if (!reduced()) {
            weight_bytes = data.size();
            CHECK(cuMemAlloc(&weights, weight_bytes));
            shared->weights = weights;
            CHECK(cuMemcpyHtoD(weights, data.data(), weight_bytes));
        } else {
            const auto packed = build_device_weights(model, dev_model, data.data(), half_io(), fused_conv);
            weight_bytes = packed.size();
            CHECK(cuMemAlloc(&weights, weight_bytes));
            shared->weights = weights;
            CHECK(cuMemcpyHtoD(weights, packed.data(), weight_bytes));
            init_cache_environment(options.cache_dir, package.digest, image);
        }
    }

    ~Session() override {
        if (context && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            if (pending) cuEventSynchronize(complete);
            if (stream) cuStreamSynchronize(stream);
            release_shape();
            for (auto& ev : timing)
                if (ev) cuEventDestroy(ev);
            if (complete) cuEventDestroy(complete);
            if (stream) cuStreamDestroy(stream);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        fft.reset();
        unfused_weights.reset();
        shared.reset();
    }

    std::unique_ptr<umzi::KernelSession> clone_shared() const override {
        auto copy = std::make_unique<Session>();
        copy->copy_base_state(*this);
        copy->shared = shared;
        copy->unfused_weights = unfused_weights;
        copy->model = model;
        copy->dev_model = dev_model;
        copy->graph_enabled = graph_enabled;
        copy->unexact_erf = unexact_erf;
        copy->fused_conv = fused_conv;
        copy->precision = precision;
        copy->workspace_limit = workspace_limit;
        copy->device = device;
        copy->context = context;
        copy->module = module;
        copy->cudnn = cudnn;
        copy->fft = fft;
        copy->fft_version = fft_version;
        copy->functions = functions;
        copy->weights = weights;
        copy->weight_bytes = weight_bytes;
        copy->cache_dir = cache_dir;
        copy->cache_environment = cache_environment;
        copy->cache_environment_error = cache_environment_error;
        ContextScope scope(context);
        CHECK(cuStreamCreate(&copy->stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&copy->complete, CU_EVENT_DISABLE_TIMING));
        CHECK(cuEventCreate(&copy->timing[0], CU_EVENT_DEFAULT));
        CHECK(cuEventCreate(&copy->timing[1], CU_EVENT_DEFAULT));
        return copy;
    }

private:
    bool half_io() const { return precision == UMZI_FP16; }
    bool mixed() const { return precision == UMZI_MIXED; }
    bool reduced() const { return precision != UMZI_FP32; }
    size_t element(bool half) const { return half ? sizeof(uint16_t) : sizeof(float); }
    CUdeviceptr dev_ptr(size_t offset) const { return weights + offset; }

    void init_cache_environment(const char* directory, const umzi::Digest& model_digest,
                                const umzi::KernelImage& image) {
        if (!directory || !*directory) return;
        try {
            cache_dir = std::filesystem::u8path(directory);
            umzi::Sha256 hash;
            umzi::hash_plan_environment(hash, umzi::gfisrv2_id, plan_schema, model_digest, image, device);
            hash.add(precision);
            hash.add(workspace_limit);
            hash.add(static_cast<uint32_t>(unexact_erf));
            hash.add(static_cast<uint32_t>(fused_conv));
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
            hash.add(std::array<int, 3>{shape.n, shape.h, shape.w});
            plan_key = hash.finish();
            plan_cache.open(cache_dir, "gfisrv2", plan_key);
        } catch (const std::exception& error) {
            plan_cache.unavailable(error);
        }
    }

    void release_shape() {
        if (executable) cuGraphExecDestroy(executable);
        executable = nullptr;
        forward_fft.release(*fft);
        inverse_fft.release(*fft);
        variants.clear();
        backend_plans.clear();
        legacy_plans.clear();
        plan_cache = {};
        if (reduced()) freq_a = freq_b = 0;
        for (auto* buffer : {&stage_in, &stage_out, &cudnn_scratch, &initial, &state, &norm, &projected, &gated,
                             &fft_buf, &spectrum, &freq_a, &freq_b}) {
            if (*buffer) cuMemFree(*buffer);
            *buffer = 0;
        }
        shape = {};
        activation_bytes = cudnn_workspace = scratch_bytes = peak_workspace_bytes = 0;
        timed_plans = 0;
        setup_ms = {};
        device_used = {};
    }

    void wait() {
        if (!pending) return;
        CHECK(cuEventSynchronize(complete));
        pending = false;
    }

    void activation(CUdeviceptr& buffer, size_t bytes) {
        CHECK(cuMemAlloc(&buffer, bytes));
        activation_bytes += bytes;
    }

    void make_legacy_conv(int h, int w, int cin, int cout, int k) {
        auto plan = std::make_unique<LegacyConvolution>();
        auto& p = *plan;
        CHECK(cudnnCreateTensorDescriptor(&p.input));
        CHECK(cudnnCreateTensorDescriptor(&p.output));
        CHECK(cudnnCreateFilterDescriptor(&p.filter));
        CHECK(cudnnCreateConvolutionDescriptor(&p.convolution));
        CHECK(cudnnSetTensor4dDescriptor(p.input, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape.n, cin, h, w));
        CHECK(cudnnSetTensor4dDescriptor(p.output, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape.n, cout, h, w));
        CHECK(cudnnSetFilter4dDescriptor(p.filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, cout, cin, k, k));
        CHECK(cudnnSetConvolution2dDescriptor(p.convolution, k / 2, k / 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));
        CHECK(cudnnSetConvolutionMathType(p.convolution, CUDNN_FMA_MATH));
        const auto preferred = k == 1 || cin == 3 ? CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
                                                  : CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        const auto status = cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, p.input, p.filter, p.convolution, p.output, preferred, &p.workspace);
        p.algorithm = preferred;
        if (status != CUDNN_STATUS_SUCCESS) {
            if (status != CUDNN_STATUS_NOT_SUPPORTED) CHECK(status);
            p.algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                cudnn, p.input, p.filter, p.convolution, p.output, p.algorithm, &p.workspace));
        }
        cudnn_workspace = std::max(cudnn_workspace, p.workspace);
        legacy_plans.emplace(LegacyKey{h, w, cin, cout, k}, std::move(plan));
    }

    void make_backend_plan(int h, int w, int cin, int cout, int k, cudnnDataType_t io_type, bool fp16_compute,
                           CUdeviceptr in_buf, CUdeviceptr out_buf) {
        auto plan = std::make_unique<umzi::CudnnBackendPlan>();
        auto& p = *plan;
        p.graph = std::make_unique<umzi::CudnnConvGraph>(cudnn, shape.n, h, w, cin, cout, k, io_type, fp16_compute);
        auto candidates = p.graph->candidates(cudnn, workspace_limit);
        const umzi::PlanShape key{shape.n, h, w, cin, cout};
        auto time_candidate = [&](float best) {
            return umzi::time_cudnn_backend_plan(cudnn, stream, timing, p, in_buf, weights, out_buf, cudnn_scratch,
                                                 scratch_bytes, best, &peak_workspace_bytes, &device_used[1]);
        };
        if (auto saved = plan_cache.convolutions.find(key); saved != plan_cache.convolutions.end()) {
            const auto& choice = saved->second;
            if (choice.math == saved_plan_marker && choice.algorithm < candidates.size()
                && uint64_t(candidates[choice.algorithm].workspace) == choice.workspace) {
                auto& candidate = candidates[choice.algorithm];
                p.plan = std::move(candidate.plan);
                p.workspace = size_t(candidate.workspace);
                if (time_candidate(0.f) < std::numeric_limits<float>::infinity()) {
                    cudnn_workspace = std::max(cudnn_workspace, p.workspace);
                    backend_plans.emplace(key, std::move(plan));
                    return;
                }
                // Give the plan back so the tuning loop below still sees every candidate.
                candidate.plan = std::move(p.plan);
            }
        }
        plan_cache.record_tuning();
        float best = std::numeric_limits<float>::infinity();
        size_t fastest = candidates.size();
        for (size_t i = 0; i < candidates.size(); ++i) {
            p.plan = std::move(candidates[i].plan);
            p.workspace = size_t(candidates[i].workspace);
            const float elapsed = time_candidate(best);
            candidates[i].plan = std::move(p.plan);
            if (elapsed < best) {
                best = elapsed;
                fastest = i;
            }
        }
        if (fastest == candidates.size())
            throw std::runtime_error("No cuDNN convolution for GFISRv2 " + std::to_string(cin) + " to "
                                     + std::to_string(cout) + " channels within the workspace limit");
        p.plan = std::move(candidates[fastest].plan);
        p.workspace = size_t(candidates[fastest].workspace);
        cudnn_workspace = std::max(cudnn_workspace, p.workspace);
        ++timed_plans;
        plan_cache.convolutions.insert_or_assign(key,
                                                 umzi::ConvChoice{uint32_t(fastest), saved_plan_marker, p.workspace});
        backend_plans.emplace(key, std::move(plan));
    }

    void bind_stream(CUstream target) {
        CHECK(cudnnSetStream(cudnn, target));
        FFT_CHECK(fft->set_stream(forward_fft.handle, reinterpret_cast<void*>(target)));
        FFT_CHECK(fft->set_stream(inverse_fft.handle, reinterpret_cast<void*>(target)));
    }

    void prepare(Shape requested) {
        if (shape == requested) return;
        wait(); // Only a shape change frees buffers the previous batch may still use.
        release_shape();
        try {
            auto start = Clock::now();
            device_used[0] = umzi::device_used_bytes();
            shape = requested;
            const size_t pixels = size_t(shape.n) * shape.pixels();
            bins = shape.h * (shape.w / 2 + 1);
            fft_scale = float(1.0 / std::sqrt(double(shape.pixels())));

            if (!reduced()) {
                activation(initial, pixels * channels * sizeof(float));
                activation(state, pixels * channels * sizeof(float));
                activation(norm, pixels * channels * sizeof(float));
                activation(projected, pixels * 2 * hidden * sizeof(float));
                activation(gated, pixels * hidden * sizeof(float));
                const size_t frequency_bytes = size_t(shape.n) * bins * 2 * fourier * sizeof(float);
                activation(spectrum, frequency_bytes);
                activation(freq_a, frequency_bytes);
                activation(freq_b, frequency_bytes);
            } else {
                const size_t state_elem = element(half_io());
                activation(initial, pixels * channels * state_elem);
                activation(state, pixels * channels * state_elem);
                activation(norm, pixels * channels * sizeof(uint16_t));
                // With fused_conv, projected holds compact Fourier/branch buffers (80 ch)
                // during body blocks and tail_expand output (96 ch) during tail.
                const size_t proj_ch = fused_conv ? size_t(2 * channels) : size_t(2 * hidden);
                activation(projected, pixels * proj_ch * sizeof(uint16_t));
                activation(gated, pixels * hidden * sizeof(uint16_t));
                const size_t fft_bytes = pixels * fourier * sizeof(float);
                activation(spectrum, size_t(shape.n) * bins * 2 * fourier * sizeof(float));
                const size_t freq_elem = element(half_io());
                const size_t freq_bytes = size_t(shape.n) * bins * freq_lanes * freq_elem;
                const size_t freq_stride = (freq_bytes + 255) & ~size_t(255);
                activation(fft_buf, std::max(fft_bytes, freq_stride + freq_bytes));
                freq_a = fft_buf;
                freq_b = fft_buf + freq_stride;
            }
            device_used[1] = umzi::device_used_bytes();
            setup_ms[0] = umzi::elapsed_ms(start);

            start = Clock::now();
            const int h = shape.h, w = shape.w, w_freq = w / 2 + 1;
            forward_fft.setup(*fft, shape, r2c);
            inverse_fft.setup(*fft, shape, c2r);
            bind_stream(stream);

            if (!reduced()) {
                make_legacy_conv(h, w, 3, channels, 3);
                make_legacy_conv(h, w, channels, 2 * hidden, 3);
                make_legacy_conv(h, w, hidden, channels, 3);
                make_legacy_conv(h, w, channels, 2 * channels, 3);
                make_legacy_conv(h, w, 2 * channels, channels, 3);
                make_legacy_conv(h, w, channels, 3 * scale * scale, 3);
                make_legacy_conv(h, w_freq, 2 * fourier, 2 * fourier, 1);
                if (cudnn_workspace) {
                    if (!umzi::ensure_device_scratch(stream, cudnn_scratch, scratch_bytes, cudnn_workspace,
                                                     &peak_workspace_bytes))
                        CHECK(CUDA_ERROR_OUT_OF_MEMORY);
                    device_used[1] = std::max(device_used[1], umzi::device_used_bytes());
                }
                device_used[2] = umzi::device_used_bytes();
            } else {
                launch(stream, FillNoiseF16, int(pixels * hidden), gated);
                load_plan_cache();
                const bool fp16_acc = half_io();
                make_backend_plan(h, w, rgb_lanes, channels, 3, CUDNN_DATA_HALF, fp16_acc, gated, projected);
                if (!fused_conv) {
                    make_backend_plan(h, w, channels, 2 * hidden, 3, CUDNN_DATA_HALF, fp16_acc, gated, projected);
                    make_backend_plan(h, w, hidden, channels, 3, CUDNN_DATA_HALF, fp16_acc, gated, projected);
                }
                make_backend_plan(h, w, channels, 2 * channels, 3, CUDNN_DATA_HALF, fp16_acc, gated, projected);
                make_backend_plan(h, w, 2 * channels, channels, 3, CUDNN_DATA_HALF, fp16_acc, gated, projected);
                make_backend_plan(h, w, channels, 3 * scale * scale, 3, CUDNN_DATA_HALF, fp16_acc, gated, projected);
                CHECK(cuStreamSynchronize(stream));
                device_used[2] = umzi::device_used_bytes();
                if (cudnn_scratch) CHECK(cuMemFree(cudnn_scratch));
                cudnn_scratch = 0;
                scratch_bytes = 0;
                if (cudnn_workspace) {
                    if (!umzi::ensure_device_scratch(stream, cudnn_scratch, scratch_bytes, cudnn_workspace,
                                                     &peak_workspace_bytes))
                        CHECK(CUDA_ERROR_OUT_OF_MEMORY);
                }
                plan_cache.save(plan_key);
            }
            setup_ms[1] = umzi::elapsed_ms(start);

            if (graph_enabled) {
                start = Clock::now();
                ensure_staging();
                CHECK(cuMemsetD8Async(stage_in, 0, shape.rgb_bytes(element(half_io())), stream));
                forward(stage_in, stage_out, stream);
                CHECK(cuStreamSynchronize(stream));
                setup_ms[2] = umzi::elapsed_ms(start);
                start = Clock::now();
                umzi::capture_cuda_graph(stream, executable, [&] { forward(stage_in, stage_out, stream); });
                setup_ms[3] = umzi::elapsed_ms(start);
            }
            device_used[3] = umzi::device_used_bytes();
        } catch (const umzi::AllocationError&) {
            // Out of memory is for the tiler to handle, not a reason to drop fusion.
            cuStreamSynchronize(stream);
            release_shape();
            throw;
        } catch (const std::runtime_error& error) {
            // Only runtime failures may trigger the fallback; bad_alloc and logic errors fall through to the rethrow below.
            cuStreamSynchronize(stream);
            release_shape();
            if (!fused_conv || !reduced()) throw; // only the reduced-precision path has a fused layout
            const auto original = std::current_exception();
            try {
                disable_fused_conv(error.what());
            } catch (...) {
                std::rethrow_exception(original); // The fallback failed too; report the failure that started it.
            }
            prepare(requested);
        } catch (...) {
            cuStreamSynchronize(stream);
            release_shape();
            throw;
        }
    }

    // Repacks the weights for the unfused layout into a buffer this session (and its clones) own.
    void disable_fused_conv(const char* reason) {
        const auto package = reload_package();
        const auto packed = build_device_weights(model, dev_model, package->section(3).data(), half_io(), false);
        unfused_weights = std::make_shared<umzi::DeviceAllocation>(context, packed.data(), packed.size());
        weights = unfused_weights->pointer;
        weight_bytes = packed.size();
        fused_conv = false;
        if (cache_environment_error.empty() && !cache_dir.empty()) {
            umzi::Sha256 hash;
            hash.add(cache_environment);
            hash.add(uint32_t(0)); // fused_conv off; plans differ from the fused environment
            cache_environment = hash.finish();
        }
        warn_option(std::string("Disabled GFISRv2 fused_conv: ") + reason);
    }

    template<class... Args>
    void launch(CUstream target, Kernel id, int count, Args... args) {
        void* params[] = {&args..., &count};
        CHECK(cuLaunchKernel(functions[id], (count + threads - 1) / threads, 1, 1, threads, 1, 1, 0, target, params,
                             nullptr));
    }

    template<class... Args>
    void launch_grid(CUstream target, Kernel id, int grid_x, int block_x, unsigned shared_bytes, Args... args) {
        void* params[] = {&args...};
        CHECK(cuLaunchKernel(functions[id], grid_x, 1, 1, block_x, 1, 1, shared_bytes, target, params, nullptr));
    }

    CUdeviceptr channel_offset_fp32(CUdeviceptr base, int channel) const {
        return base + uint64_t(channel) * shape.pixels() * sizeof(float);
    }

    void conv_fp32(CUstream target, CUdeviceptr out, CUdeviceptr in, int h, int w, int cin, int cout, int k,
                   Filter filter) {
        const auto& p = *legacy_plans.at(LegacyKey{h, w, cin, cout, k});
        const float one = 1.f, zero = 0.f;
        CHECK(cudnnConvolutionForward(cudnn, &one, p.input, reinterpret_cast<void*>(in), p.filter,
            reinterpret_cast<void*>(weights + filter.weight), p.convolution, p.algorithm,
            reinterpret_cast<void*>(cudnn_scratch), p.workspace, &zero, p.output, reinterpret_cast<void*>(out)));
        launch(target, Bias, shape.n * h * w * cout, out, weights + filter.bias, cout, h * w);
    }

    void rms_fp32(CUstream target, CUdeviceptr out, CUdeviceptr in, Norm params, int c, int pixels, int out_stride) {
        launch(target, Rms, shape.n * pixels, out, in, weights + params.scale, weights + params.offset,
               c, c, out_stride, pixels, float(1.0 / std::sqrt(double(c))));
    }

    void fourier_unit_fp32(CUstream target, const Fourier& params, CUdeviceptr in, CUdeviceptr out) {
        const int pixels = shape.pixels(), spatial = shape.n * fourier * pixels, frequency = shape.n * fourier * bins;
        const int spectral_channels = 2 * fourier, width = shape.w / 2 + 1;
        launch(target, CopyChannels, spatial, gated, in, fourier, 2 * hidden, pixels);
        FFT_CHECK(fft->r2c(forward_fft.handle, reinterpret_cast<void*>(gated), reinterpret_cast<void*>(spectrum)));
        launch(target, Pack, frequency, freq_a, spectrum, fourier, bins, fft_scale);
        rms_fp32(target, freq_b, freq_a, params.pre, spectral_channels, bins, spectral_channels);
        launch(target, Depthwise, frequency * 2, freq_a, freq_b, weights + params.position.weight,
               weights + params.position.bias, spectral_channels, spectral_channels, spectral_channels,
               shape.h, width, 3, 3, 1);
        conv_fp32(target, freq_b, freq_a, shape.h, width, spectral_channels, spectral_channels, 1, params.project);
        launch(target, Unpack, frequency, spectrum, freq_b, bins);
        FFT_CHECK(fft->c2r(inverse_fft.handle, reinterpret_cast<void*>(spectrum), reinterpret_cast<void*>(gated)));
        launch(target, ScaleFft, spatial, gated, fft_scale);
        rms_fp32(target, out, gated, params.post, fourier, pixels, channels);
    }

    void forward_fp32(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        const int pixels = shape.pixels(), area = shape.n * pixels, h = shape.h, w = shape.w;
        conv_fp32(target, initial, input, h, w, 3, channels, 3, model.first);
        CHECK(cuMemcpyDtoDAsync(state, initial, uint64_t(area) * channels * sizeof(float), target));
        for (int i = 0; i < blocks; ++i) {
            const auto& block = model.body[i];
            rms_fp32(target, norm, state, block.norm, channels, pixels, channels);
            conv_fp32(target, projected, norm, h, w, channels, 2 * hidden, 3, block.expand);
            int offset = 0;
            for (int slot = 0; slot < 4; ++slot) {
                const int kind = (slot + i) % 4;
                const auto in = channel_offset_fp32(projected, 2 * hidden - channels + offset);
                const auto out = channel_offset_fp32(norm, offset);
                if (!kind) {
                    fourier_unit_fp32(target, block.fft, in, out);
                    offset += fourier;
                } else {
                    const auto& filter = block.bands[kind - 1];
                    const int kh = kind == 1 ? 3 : kind == 2 ? 1 : 11;
                    const int kw = kind == 1 ? 3 : kind == 2 ? 11 : 1;
                    launch(target, Depthwise, area * band, out, in, weights + filter.weight,
                           weights + filter.bias, band, 2 * hidden, channels, h, w, kh, kw, 0);
                    offset += band;
                }
            }
            launch(target, Gate, area * hidden, gated, projected, norm, hidden, channels, pixels);
            conv_fp32(target, projected, gated, h, w, hidden, channels, 3, block.project);
            launch(target, Residual, area * channels, state, projected, weights + block.gamma, channels, pixels);
        }
        conv_fp32(target, projected, state, h, w, channels, 2 * channels, 3, model.tail_expand);
        launch(target, Silu, area * 2 * channels, projected);
        conv_fp32(target, norm, projected, h, w, 2 * channels, channels, 3, model.tail_project);
        launch(target, Add, area * channels, state, norm, initial);
        conv_fp32(target, projected, state, h, w, channels, 3 * scale * scale, 3, model.last);
        launch(target, Shuffle, area * 3 * scale * scale, output, projected, h, w);
    }

    void conv_nhwc(CUdeviceptr out, CUdeviceptr in, int h, int w, int cin, int cout, Filter filter) {
        const auto& p = *backend_plans.at(umzi::PlanShape{shape.n, h, w, cin, cout});
        const CUdeviceptr wptr = dev_ptr(filter.weight);
        auto& variant = variants[std::array<CUdeviceptr, 4>{out, in, wptr, cudnn_scratch}];
        if (!variant)
            variant = umzi::bind_conv_variant(reinterpret_cast<void*>(in), reinterpret_cast<void*>(wptr),
                                              reinterpret_cast<void*>(out), reinterpret_cast<void*>(cudnn_scratch));
        CHECK(cudnnBackendExecute(cudnn, p.plan->get(), variant->get()));
    }

    void forward_nhwc(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        const int h = shape.h, w = shape.w, w_freq = w / 2 + 1;
        const int pixels_per_image = shape.pixels(), total_pixels = shape.n * pixels_per_image;
        const int pix_tiles_per_image = (pixels_per_image + 31) / 32;
        const int bin_tiles_32 = (bins + 31) / 32;
        const int bin_tiles_64 = (bins + 63) / 64;
        const int expand_tx = (w + 15) / 16, expand_ty = (h + 7) / 8;
        const int tile16_tx = (w + 15) / 16, tile16_ty = (h + 15) / 16;
        const int spec_tx = (w_freq + 7) / 8, spec_ty = (h + 7) / 8;
        const int rms48_blocks = (total_pixels * 8 + threads - 1) / threads;

        // Compact Fourier/branch sub-buffers aliased inside `projected` when fused_conv is active.
        const CUdeviceptr fourier_gate = projected;
        const CUdeviceptr branch_val = projected + uint64_t(total_pixels) * 32 * sizeof(uint16_t);
        const CUdeviceptr branch_gate = projected + uint64_t(total_pixels) * 56 * sizeof(uint16_t);

        launch(target, half_io() ? PackRgbF16 : PackRgbF32, total_pixels, gated, input, h, w);
        conv_nhwc(projected, gated, h, w, rgb_lanes, channels, dev_model.first);
        launch_grid(target, half_io() ? InputBiasRmsF16 : InputBiasRmsMixed, rms48_blocks, threads, 0,
                    state, initial, norm, projected, dev_ptr(dev_model.first.bias),
                    dev_ptr(dev_model.body[0].norm.scale), dev_ptr(dev_model.body[0].norm.offset), total_pixels);

        for (int i = 0; i < blocks; ++i) {
            const auto& block = dev_model.body[i];
            if (fused_conv) {
                launch_grid(target, half_io() ? ExpandGateF16 : ExpandGateF32, shape.n * expand_tx * expand_ty,
                            threads, expand_shared_bytes,
                            gated, fft_buf, fourier_gate, branch_val, branch_gate, norm,
                            dev_ptr(block.expand.weight), dev_ptr(block.expand.bias),
                            h, w, pixels_per_image, total_pixels, expand_tx, expand_ty);

                launch_grid(target, BranchDwGateCompact, shape.n * tile16_tx * tile16_ty, threads, 0,
                            gated, branch_val, branch_gate,
                            dev_ptr(block.bands[0].weight), dev_ptr(block.bands[0].bias),
                            dev_ptr(block.bands[1].weight), dev_ptr(block.bands[1].bias),
                            dev_ptr(block.bands[2].weight), dev_ptr(block.bands[2].bias),
                            h, w, total_pixels, tile16_tx, tile16_ty);
            } else {
                conv_nhwc(projected, norm, h, w, channels, 2 * hidden, block.expand);
                launch(target, GatePlainNhwc, total_pixels * 20, gated, projected, dev_ptr(block.expand.bias));
                launch_grid(target, BranchDwGateNhwc, shape.n * tile16_tx * tile16_ty, threads, 0,
                            gated, projected, dev_ptr(block.expand.bias),
                            dev_ptr(block.bands[0].weight), dev_ptr(block.bands[0].bias),
                            dev_ptr(block.bands[1].weight), dev_ptr(block.bands[1].bias),
                            dev_ptr(block.bands[2].weight), dev_ptr(block.bands[2].bias),
                            h, w, tile16_tx, tile16_ty);
                launch_grid(target, ExtractFourierNhwc, shape.n * pix_tiles_per_image, threads, 0,
                            fft_buf, projected, dev_ptr(block.expand.bias), pixels_per_image, pix_tiles_per_image);
            }

            // Fourier unit: cuFFT R2C -> fused pack+preRMS -> tiled 3x3 dw+res
            // -> fused 1x1+bias+GELU+unpack -> cuFFT C2R -> fused scale+postRMS+gate.
            FFT_CHECK(fft->r2c(forward_fft.handle, reinterpret_cast<void*>(fft_buf), reinterpret_cast<void*>(spectrum)));
            launch_grid(target, half_io() ? PackRmsFftF16 : PackRmsFftF32, shape.n * bin_tiles_32, threads, 0,
                        freq_a, spectrum, dev_ptr(block.fft.pre.scale), dev_ptr(block.fft.pre.offset),
                        bins, bin_tiles_32, fft_scale);
            launch_grid(target, half_io() ? SpectralDwResF16 : SpectralDwResF32, shape.n * spec_tx * spec_ty, threads,
                        0, freq_b, freq_a, dev_ptr(block.fft.position.weight), dev_ptr(block.fft.position.bias),
                        h, w_freq, spec_tx, spec_ty);
            launch_grid(target, half_io() ? SpectralProjGeluUnpackF16 : SpectralProjGeluUnpackF32,
                        shape.n * bin_tiles_64, 128, 0,
                        spectrum, freq_b, dev_ptr(block.fft.project.weight), dev_ptr(block.fft.project.bias),
                        bins, bin_tiles_64);
            FFT_CHECK(fft->c2r(inverse_fft.handle, reinterpret_cast<void*>(spectrum), reinterpret_cast<void*>(fft_buf)));

            if (fused_conv) {
                launch_grid(target, ScalePostRmsGateCompact, shape.n * pix_tiles_per_image, threads, 0,
                            gated, fft_buf, fourier_gate,
                            dev_ptr(block.fft.post.scale), dev_ptr(block.fft.post.offset),
                            pixels_per_image, pix_tiles_per_image, fft_scale);

                if (i + 1 < blocks) {
                    const auto& next_norm = dev_model.body[i + 1].norm;
                    launch_grid(target, half_io() ? ProjectResNormF16 : ProjectResNormF32,
                                shape.n * tile16_tx * tile16_ty, 128, project_shared_bytes,
                                state, norm, gated, dev_ptr(block.project.weight), dev_ptr(block.project.bias),
                                dev_ptr(block.gamma), dev_ptr(next_norm.scale), dev_ptr(next_norm.offset),
                                h, w, tile16_tx, tile16_ty);
                } else {
                    const CUdeviceptr null_ptr = 0;
                    launch_grid(target, half_io() ? ProjectResF16 : ProjectResToHalfF32,
                                shape.n * tile16_tx * tile16_ty, 128, project_shared_bytes,
                                state, norm, gated, dev_ptr(block.project.weight), dev_ptr(block.project.bias),
                                dev_ptr(block.gamma), null_ptr, null_ptr,
                                h, w, tile16_tx, tile16_ty);
                }
            } else {
                launch_grid(target, ScalePostRmsGateNhwc, shape.n * pix_tiles_per_image, threads, 0,
                            gated, fft_buf, projected, dev_ptr(block.expand.bias),
                            dev_ptr(block.fft.post.scale), dev_ptr(block.fft.post.offset),
                            pixels_per_image, pix_tiles_per_image, fft_scale);

                conv_nhwc(projected, gated, h, w, hidden, channels, block.project);
                if (i + 1 < blocks) {
                    const auto& next_norm = dev_model.body[i + 1].norm;
                    launch_grid(target, half_io() ? ResidualRmsF16 : ResidualRmsMixed, rms48_blocks, threads, 0,
                                state, norm, projected, dev_ptr(block.project.bias), dev_ptr(block.gamma),
                                dev_ptr(next_norm.scale), dev_ptr(next_norm.offset), total_pixels);
                } else {
                    const CUdeviceptr null_ptr = 0;
                    launch_grid(target, half_io() ? ResidualF16 : ResidualToHalfMixed, rms48_blocks, threads, 0,
                                state, norm, projected, dev_ptr(block.project.bias), dev_ptr(block.gamma),
                                null_ptr, null_ptr, total_pixels);
                }
            }
        }

        conv_nhwc(projected, half_io() ? state : norm, h, w, channels, 2 * channels, dev_model.tail_expand);
        launch(target, BiasSilu96, total_pixels * 12, projected, dev_ptr(dev_model.tail_expand.bias));
        conv_nhwc(norm, projected, h, w, 2 * channels, channels, dev_model.tail_project);
        launch(target, half_io() ? TailAddF16 : TailAddMixed, total_pixels * 6,
               gated, norm, dev_ptr(dev_model.tail_project.bias), initial);
        conv_nhwc(projected, gated, h, w, channels, 3 * scale * scale, dev_model.last);
        launch(target, half_io() ? BiasShuffleRgbF16 : BiasShuffleRgbF32, total_pixels * 3 * scale * scale,
               output, projected, dev_ptr(dev_model.last.bias), h, w);
    }

    void forward(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        if (!reduced()) forward_fp32(input, output, target);
        else forward_nhwc(input, output, target);
    }

    void execute(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        if (!executable) {
            bind_stream(target);
            forward(input, output, target);
            return;
        }
        const auto bytes = shape.rgb_bytes(element(half_io()));
        if (input != stage_in) CHECK(cuMemcpyDtoDAsync(stage_in, input, bytes, target));
        CHECK(cuGraphLaunch(executable, target));
        if (output != stage_out) CHECK(cuMemcpyDtoDAsync(output, stage_out, bytes * scale * scale, target));
    }

public:
    UmziModelInfo model_info() const override { return {sizeof(UmziModelInfo), "gfisrv2", scale, 0}; }
    UmziIo tensors() const override {
        const uint32_t dtype = half_io() ? UMZI_FLOAT16 : UMZI_FLOAT32;
        return {sizeof(UmziIo), {"input", {4, {0, 3, 0, 0}}, dtype, 0},
                                {"output", {4, {0, 3, 0, 0}}, dtype, 0}};
    }

    UmziIo describe(const UmziShape& request) const override {
        const auto concrete = checked(request);
        auto io = tensors();
        io.input.shape = request;
        io.input.bytes = concrete.rgb_bytes(element(half_io()));
        io.output.shape = Shape{concrete.n, scale * concrete.h, scale * concrete.w}.nchw();
        io.output.bytes = io.input.bytes * scale * scale;
        return io;
    }

    void prepare_shape(const UmziShape& request) override {
        ContextScope scope(context);
        prepare(checked(request));
    }

    void ensure_staging() {
        if (stage_in && stage_out) return;
        const auto bytes = shape.rgb_bytes(element(half_io()));
        // Allocate only what is missing: a failed second allocation must not leave a half-built pair that passes the early return.
        if (!stage_out) CHECK(cuMemAlloc(&stage_out, bytes * scale * scale));
        if (!stage_in) CHECK(cuMemAlloc(&stage_in, bytes));
    }

    void run_host(const UmziShape& request, const void* input, void* output) override {
        ContextScope scope(context);
        prepare(checked(request));
        wait();
        ensure_staging();
        const auto bytes = shape.rgb_bytes(element(half_io()));
        try {
            CHECK(cuMemcpyHtoDAsync(stage_in, input, bytes, stream));
            execute(stage_in, stage_out, stream);
            CHECK(cuMemcpyDtoHAsync(output, stage_out, bytes * scale * scale, stream));
            CHECK(cuStreamSynchronize(stream));
        } catch (...) {
            cuStreamSynchronize(stream);
            throw;
        }
    }

    void enqueue_device(const UmziShape& request, uint64_t input, uint64_t output, UmziStream target) override {
        ContextScope scope(context);
        prepare(checked(request));
        try {
            // Order this batch after the previous one on the device instead of blocking the host.
            if (pending) CHECK(cuStreamWaitEvent(target, complete, CU_EVENT_WAIT_DEFAULT));
            execute(input, output, target);
            pending = true; // Before the record, so teardown waits even if recording fails.
            CHECK(cuEventRecord(complete, target));
        } catch (...) {
            cuStreamSynchronize(target);
            throw;
        }
    }

    UmziSessionInfo session_info() const override {
        UmziSessionInfo info{};
        info.struct_size = sizeof(info);
        info.cache_status = reduced() ? plan_cache.status : UMZI_CACHE_DISABLED;
        info.buffer_bytes[0] = weight_bytes;
        if (shape.n) {
            info.prepared_input = shape.nchw();
            info.buffer_bytes[1] = activation_bytes;
            info.buffer_bytes[2] = cudnn_workspace + forward_fft.bytes + inverse_fft.bytes;
            info.buffer_bytes[3] = stage_in ? shape.rgb_bytes(element(half_io())) * (1 + scale * scale) : 0;
            std::copy(setup_ms.begin(), setup_ms.end(), info.shape_setup_ms);
            std::copy(device_used.begin(), device_used.end(), info.startup_device_used_bytes);
        }
        return info;
    }

    const char* cache_message() const override { return plan_cache.diagnostic.c_str(); }

    uint32_t diagnostic_count() const override { return 7; }
    UmziDiagnostic diagnostic(uint32_t index) const override {
        if (index == 0) return {sizeof(UmziDiagnostic), "cufft_version", fft_version};
        if (index == 1)
            return {sizeof(UmziDiagnostic), "cufft_workspace_bytes", int64_t(forward_fft.bytes + inverse_fft.bytes)};
        if (index == 2) return {sizeof(UmziDiagnostic), "unexact_erf", unexact_erf};
        if (index == 3) return {sizeof(UmziDiagnostic), "timed_plans", timed_plans};
        if (index == 4) return {sizeof(UmziDiagnostic), "plan_workspace_bytes", int64_t(cudnn_workspace)};
        if (index == 5) return {sizeof(UmziDiagnostic), "fused_conv", int64_t(fused_conv)};
        if (index == 6) return {sizeof(UmziDiagnostic), "peak_plan_workspace_bytes", int64_t(peak_workspace_bytes)};
        throw std::runtime_error("Invalid GFISRv2 diagnostic index");
    }
    const char* kernel_message() const override {
        if (mixed()) return fused_conv ? "GFISRv2 fused NHWC mixed precision with FP32 Fourier and residuals"
                                       : "GFISRv2 NHWC mixed precision with FP32 Fourier and residuals";
        if (half_io()) return fused_conv ? "GFISRv2 fused NHWC FP16 with FP32 cuFFT"
                                         : "GFISRv2 NHWC FP16 with FP32 cuFFT";
        return "GFISRv2 FP32 with cuFFT";
    }
};
} // namespace

std::unique_ptr<umzi::KernelSession> umzi::make_gfisrv2(const ModelPackage& package, const UmziSessionOptions& options) {
    auto session = std::make_unique<Session>();
    session->open(package, options);
    return session;
}
