#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <cuda.h>
#include <cudnn.h>

#define UMZI_BUILD
#include "mosr_images.h"
#include "../../common/cuda_helpers.h"
#include "../../common/cuda_module.h"
#include "../../common/cuda_plan_cache.h"
#include "../../common/cudnn_conv_plan.h"
#include "../../runtime/binary.h"
#include "../../runtime/kernel.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

namespace {
constexpr int rgb_lanes = 8, threads = 256;
constexpr int lanes(int count) { return (count + 7) & ~7; }
constexpr uint64_t default_workspace_mib = 512, max_workspace_mib = 4096;
constexpr uint32_t plan_schema = 2;
using umzi::Clock;
using umzi::ContextScope;
#define CHECK(expr) umzi::check((expr), #expr)

enum UpsamplerKind : uint32_t { UpsamplePS = 0, UpsampleGPS = 1, UpsampleDYS = 2 };

struct Architecture {
    uint32_t input_channels = 0;
    uint32_t output_channels = 0;
    uint32_t scale = 0;
    uint32_t blocks = 0;
    uint32_t dim = 0;
    uint32_t hidden = 0;
    uint32_t kernel_size = 0;
    uint32_t upsampler = 0;
    uint32_t dysample_groups = 0;
    uint32_t flags = 0;

    int d() const { return int(dim); }
    int hd() const { return int(hidden); }
    int k() const { return int(kernel_size); }
    int s() const { return int(scale); }
    int groups() const { return int(dysample_groups); }
    int offset_ch() const { return 2 * groups() * s() * s(); }
    int gps_ensemble() const { return upsampler == UpsampleGPS ? 8 : 1; }
};

Architecture parse_architecture(std::string_view section) {
    umzi::BinaryReader reader(section);
    if (section.size() != 40) throw std::runtime_error("Invalid MoSR model metadata size");
    Architecture a;
    a.input_channels = reader.number<uint32_t>();
    a.output_channels = reader.number<uint32_t>();
    a.scale = reader.number<uint32_t>();
    a.blocks = reader.number<uint32_t>();
    a.dim = reader.number<uint32_t>();
    a.hidden = reader.number<uint32_t>();
    a.kernel_size = reader.number<uint32_t>();
    a.upsampler = reader.number<uint32_t>();
    a.dysample_groups = reader.number<uint32_t>();
    a.flags = reader.number<uint32_t>();
    bool valid = a.input_channels == 3 && a.output_channels == 3 && a.scale >= 1 && a.scale <= 8 && a.blocks >= 1
                 && a.blocks <= 64 && a.dim >= 16 && a.dim <= 256 && (a.dim % 8) == 0 && a.hidden >= a.dim
                 && a.hidden <= 4 * a.dim && (a.kernel_size % 2) == 1 && a.kernel_size >= 1 && a.kernel_size <= 31
                 && a.upsampler <= UpsampleDYS && a.flags == 0;
    if (a.upsampler == UpsampleDYS) {
        valid = valid && a.scale >= 2 && a.dysample_groups > 0 && (a.dim % a.dysample_groups) == 0;
    } else {
        valid = valid && a.dysample_groups == 0;
    }
    if (!valid) throw std::runtime_error("Unsupported MoSR model architecture");
    return a;
}

struct Shape {
    int n = 0, h = 0, w = 0;
    bool operator==(Shape other) const { return n == other.n && h == other.h && w == other.w; }
    int pixels() const { return n * h * w; }
    uint64_t rgb_bytes(bool fp16) const { return uint64_t(pixels()) * 3 * (fp16 ? 2 : 4); }
    UmziShape nchw() const { return {4, {uint32_t(n), 3, uint32_t(h), uint32_t(w)}}; }
};

struct Filter {
    int cin = 0, cout = 0, k = 3, groups = 1;
    size_t weight = 0, bias = 0;
    bool has_bias = true;
};

struct GatedBlock {
    size_t norm_weight = 0, norm_bias = 0;
    Filter fc1, dw, fc2;
};

struct Network {
    struct Entry {
        size_t offset = 0, count = 0;
        bool is_conv = false;
        int cin = 0, cout = 0, k = 0;
    };
    std::vector<Entry> entries;
    Filter stem, tail0, tail1, tail2, sc0, sc1, sc11;
    Filter upsample_conv, dys_offset, dys_scope, dys_end;
    std::vector<GatedBlock> blocks;
    size_t bytes = 0;
    bool nhwc_half = false;

    Network() = default;
    Network(const Architecture& a, bool device_nhwc_half) : nhwc_half(device_nhwc_half) {
        const int d = a.d(), hd = a.hd(), k = a.k(), s = a.s();
        const int rgb_in = nhwc_half ? rgb_lanes : 3;
        const int fc1_out = nhwc_half ? lanes(2 * hd) : 2 * hd;
        const int fc2_in = nhwc_half ? lanes(hd) : hd;
        stem = conv(rgb_in, d, 3);
        blocks.resize(a.blocks);
        for (auto& b : blocks) {
            b.norm_weight = take_vec(d);
            b.norm_bias = take_vec(d);
            b.fc1 = conv(d, fc1_out, 3);
            b.dw = conv_dw(d, k);
            b.fc2 = conv(fc2_in, d, 3);
        }
        tail0 = conv(d, 2 * d, 3);
        tail1 = conv(2 * d, d, 3);
        tail2 = conv(d, d, 1);
        sc0 = conv(rgb_in, d, 3);
        sc1 = conv(d, d, 3);
        sc11 = conv(rgb_in, d, 1);
        if (a.upsampler == UpsampleDYS) {
            const int off_ch = a.offset_ch();
            const int off_lanes = nhwc_half ? lanes(off_ch) : off_ch;
            const int end_out = nhwc_half ? rgb_lanes : 3;
            dys_offset = conv(d, off_lanes, 1);
            dys_scope = conv(d, off_lanes, 1, false);
            dys_end = conv(d, end_out, 1);
        } else {
            const int real_out = 3 * s * s * a.gps_ensemble();
            const int up_lanes = nhwc_half ? lanes(real_out) : real_out;
            upsample_conv = conv(d, up_lanes, 3);
        }
    }

    size_t take(size_t size) {
        const size_t offset = (bytes + 255) & ~size_t(255);
        bytes = offset + size;
        return offset;
    }
    size_t take_vec(int count) {
        const size_t off = take(size_t(count) * 4);
        entries.push_back({off, size_t(count), false, 0, 0, 0});
        return off;
    }
    Filter conv(int cin, int cout, int k, bool bias = true) {
        const size_t elem = nhwc_half ? 2 : 4;
        const size_t w_count = size_t(cout) * cin * k * k;
        const size_t w_off = take(w_count * elem);
        entries.push_back({w_off, w_count, true, cin, cout, k});
        const size_t b_off = bias ? take_vec(cout) : 0;
        return {cin, cout, k, 1, w_off, b_off, bias};
    }
    Filter conv_dw(int dim, int k) {
        const size_t w_off = take_vec(size_t(dim) * k * k);
        const size_t b_off = take_vec(dim);
        return {dim, dim, k, dim, w_off, b_off, true};
    }
};

uint16_t checked_half(float val) {
    const uint16_t h = umzi::float_to_half(val);
    if ((h & 0x7fffu) >= 0x7c00u) throw std::runtime_error("MoSR weight exceeds the finite FP16 range");
    return h;
}

void pack_fused_fc1(uint16_t* dst_w, float* dst_b, const float* src_w, const float* src_b, int dim, int hidden) {
    const int plain = hidden - dim;
    for (int i = 0; i < plain; ++i) dst_b[i] = src_b[i];
    for (int i = 0; i < plain; ++i) dst_b[plain + i] = src_b[hidden + i];
    for (int i = 0; i < dim; ++i) dst_b[2 * plain + i] = src_b[hidden + plain + i];
    for (int i = 0; i < dim; ++i) dst_b[2 * plain + dim + i] = src_b[plain + i];

    auto orig_out = [&](int row) {
        if (row < plain) return row;
        if (row < 2 * plain) return hidden + (row - plain);
        if (row < 2 * plain + dim) return hidden + plain + (row - 2 * plain);
        return plain + (row - 2 * plain - dim);
    };

    const int cout = 2 * hidden;
    size_t stage = 0;
    for (int ks = 0; ks < dim / 16; ++ks) {
        for (int dy = 0; dy < 3; ++dy) {
            uint16_t* stage_dst = dst_w + stage * (3 * size_t(cout) * 16);
            for (int dx = 0; dx < 3; ++dx) {
                const int tap = dy * 3 + dx;
                uint16_t* tap_dst = stage_dst + dx * (size_t(cout) * 16);
                for (int row = 0; row < cout; ++row) {
                    const int co = orig_out(row);
                    const int mask = (row >> 2) & 1;
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        const size_t d = (size_t(row) * 2 + (chunk ^ mask)) * 8;
                        for (int hi = 0; hi < 8; ++hi) {
                            const int ci = ks * 16 + chunk * 8 + hi;
                            tap_dst[d + hi] = checked_half(src_w[(size_t(co) * dim + ci) * 9 + tap]);
                        }
                    }
                }
            }
            ++stage;
        }
    }
}

void pack_fused_fc2(uint16_t* dst_w, const float* src_w, int dim, int hidden) {
    size_t stage = 0;
    for (int ks = 0; ks < hidden / 16; ++ks) {
        for (int dy = 0; dy < 3; ++dy) {
            uint16_t* stage_dst = dst_w + stage * (3 * size_t(dim) * 16);
            for (int dx = 0; dx < 3; ++dx) {
                const int tap = dy * 3 + dx;
                uint16_t* tap_dst = stage_dst + dx * (size_t(dim) * 16);
                for (int row = 0; row < dim; ++row) {
                    const int mask = (row >> 2) & 1;
                    for (int chunk = 0; chunk < 2; ++chunk) {
                        const size_t d = (size_t(row) * 2 + (chunk ^ mask)) * 8;
                        for (int hi = 0; hi < 8; ++hi) {
                            const int ci = ks * 16 + chunk * 8 + hi;
                            tap_dst[d + hi] = checked_half(src_w[(size_t(row) * hidden + ci) * 9 + tap]);
                        }
                    }
                }
            }
            ++stage;
        }
    }
}

std::vector<char> pack_half_weights(const Network& src, const Network& dst, const char* raw, bool fused_gate, int dim,
                                    int hidden) {
    std::vector<char> out(dst.bytes, 0);
    std::vector<bool> skip_entry(src.entries.size(), false);
    std::vector<std::pair<size_t, size_t>> fc_blocks;
    if (fused_gate) {
        for (size_t b = 0; b < src.blocks.size(); ++b) {
            const auto& sb = src.blocks[b];
            const auto& db = dst.blocks[b];
            pack_fused_fc1(reinterpret_cast<uint16_t*>(out.data() + db.fc1.weight),
                           reinterpret_cast<float*>(out.data() + db.fc1.bias),
                           reinterpret_cast<const float*>(raw + sb.fc1.weight),
                           reinterpret_cast<const float*>(raw + sb.fc1.bias), dim, hidden);
            pack_fused_fc2(reinterpret_cast<uint16_t*>(out.data() + db.fc2.weight),
                           reinterpret_cast<const float*>(raw + sb.fc2.weight), dim, hidden);
            std::memcpy(out.data() + db.fc2.bias, raw + sb.fc2.bias, size_t(dim) * sizeof(float));
            fc_blocks.emplace_back(sb.fc1.weight, db.fc1.weight);
            fc_blocks.emplace_back(sb.fc1.bias, db.fc1.bias);
            fc_blocks.emplace_back(sb.fc2.weight, db.fc2.weight);
            fc_blocks.emplace_back(sb.fc2.bias, db.fc2.bias);
        }
        for (size_t idx = 0; idx < src.entries.size(); ++idx) {
            for (const auto& p : fc_blocks) {
                if (src.entries[idx].offset == p.first) {
                    skip_entry[idx] = true;
                    break;
                }
            }
        }
    }
    for (size_t idx = 0; idx < src.entries.size(); ++idx) {
        if (skip_entry[idx]) continue;
        const auto& s = src.entries[idx];
        const auto& d = dst.entries[idx];
        const float* in_f = reinterpret_cast<const float*>(raw + s.offset);
        if (!s.is_conv) {
            std::memcpy(out.data() + d.offset, in_f, s.count * sizeof(float));
            continue;
        }
        uint16_t* dst_h = reinterpret_cast<uint16_t*>(out.data() + d.offset);
        const int taps = s.k * s.k;
        for (int o = 0; o < s.cout; ++o) {
            for (int tap = 0; tap < taps; ++tap) {
                for (int i = 0; i < s.cin; ++i) {
                    const float val = in_f[((size_t(o) * s.cin + i) * taps) + tap];
                    dst_h[(size_t(o) * taps + tap) * d.cin + i] = checked_half(val);
                }
            }
        }
    }
    return out;
}

constexpr const char* kernel_names[] = {
    "copy_nchw_f32",
    "layer_norm_nchw_f32",
    "plain_gate_nchw_f32",
    "dw_gate_nchw_f32",
    "dw_gate_nchw_f32_k7",
    "dw_gate_nchw_f32_k17",
    "bias_norm_nchw_f32",
    "mish_residual_norm_nchw_f32",
    "mish_bias_residual_nchw_f32",
    "mish_bias_inplace_nchw_f32",
    "combine_body_shortcut_nchw_f32",
    "gps_shuffle_nchw_f32",
    "dysample_nchw_f32",
    "add_bias_nchw_f32",
    "pack_rgb_nhwc_f32",
    "pack_rgb_nhwc_f16",
    "layer_norm_nhwc",
    "plain_gate_nhwc",
    "plain_gate_nhwc_vec8",
    "dw_gate_nhwc",
    "dw_gate_nhwc_8ch_k7",
    "dw_gate_nhwc_8ch_k17",
    "bias_norm_nhwc",
    "mish_residual_norm_nhwc",
    "mish_bias_residual_nhwc",
    "mish_bias_inplace_nhwc",
    "add_bias_nhwc",
    "combine_body_shortcut_nhwc",
    "gps_shuffle_nhwc_f32",
    "gps_shuffle_nhwc_f16",
    "dysample_nhwc",
    "unpack_rgb_nhwc_f32",
    "unpack_rgb_nhwc_f16",
    "fill_noise_f16",
    "fill_noise_f32",
    "mosr_expand_gate_fp16",
    "mosr_expand_gate_mixed",
    "mosr_dw_gate_planar_k7",
    "mosr_dw_gate_planar_k17",
    "mosr_project_res_norm_fp16",
    "mosr_project_res_fp16",
    "mosr_project_res_norm_mixed",
    "mosr_project_res_mixed",
};

enum Kernel {
    CopyNchwF32,
    LayerNormNchwF32,
    PlainGateNchwF32,
    DwGateNchwF32,
    DwGateNchwF32K7,
    DwGateNchwF32K17,
    BiasNormNchwF32,
    MishResidualNormNchwF32,
    MishBiasResidualNchwF32,
    MishBiasInplaceNchwF32,
    CombineBodyShortcutNchwF32,
    GpsShuffleNchwF32,
    DysampleNchwF32,
    AddBiasNchwF32,
    PackRgbNhwcF32,
    PackRgbNhwcF16,
    LayerNormNhwc,
    PlainGateNhwc,
    PlainGateNhwcVec8,
    DwGateNhwc,
    DwGateNhwc8chK7,
    DwGateNhwc8chK17,
    BiasNormNhwc,
    MishResidualNormNhwc,
    MishBiasResidualNhwc,
    MishBiasInplaceNhwc,
    AddBiasNhwc,
    CombineBodyShortcutNhwc,
    GpsShuffleNhwcF32,
    GpsShuffleNhwcF16,
    DysampleNhwc,
    UnpackRgbNhwcF32,
    UnpackRgbNhwcF16,
    FillNoiseF16,
    FillNoiseF32,
    MosrExpandGateFp16,
    MosrExpandGateMixed,
    MosrDwGatePlanarK7,
    MosrDwGatePlanarK17,
    MosrProjectResNormFp16,
    MosrProjectResFp16,
    MosrProjectResNormMixed,
    MosrProjectResMixed,
    KernelCount,
};
static_assert(std::size(kernel_names) == KernelCount);

class Session final : public umzi::KernelSession {
    Architecture arch{};
    Network model, device_model;
    bool fp16 = false, fp32 = false, graph_enabled = false, pending = false;
    bool fused_gate_option = true, fused_gate = false;

    struct Shared {
        CUdevice device = 0;
        CUcontext context = nullptr;
        CUmodule module = nullptr;
        cudnnHandle_t cudnn = nullptr;
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
    cudnnHandle_t cudnn = nullptr;
    bool own_cudnn = false;
    std::array<CUfunction, KernelCount> functions{};
    CUdeviceptr weights = 0;

    Shape shape{};
    CUdeviceptr stage_in = 0, stage_out = 0, scratch = 0, arena = 0;
    CUdeviceptr rgb_buf = 0, state = 0, norm_out = 0, projected = 0, gated = 0;
    CUdeviceptr upsample_buf = 0, dys_off_buf = 0, dys_scp_buf = 0, dys_smp_buf = 0, dys_end_buf = 0;
    CUgraphExec executable = nullptr;
    std::map<umzi::PlanShape, std::unique_ptr<umzi::CudnnBackendPlan>> plans;
    std::map<std::array<CUdeviceptr, 4>, std::unique_ptr<umzi::CudnnDescriptor>> variants;
    static constexpr uint32_t saved_plan_marker = 0x4d4f;
    umzi::CudaPlanCache<umzi::mosr_id, plan_schema> plan_cache;
    std::filesystem::path cache_dir;
    umzi::Digest cache_environment{}, plan_key{};
    std::string cache_environment_error;
    size_t activation_bytes = 0, workspace_bytes = 0, scratch_bytes = 0, peak_workspace_bytes = 0;
    std::array<double, 4> setup_ms{};
    std::array<CUevent, 2> timing{};
    int timed_plans = 0, fp16_fallback_plans = 0;
    uint64_t workspace_limit = default_workspace_mib << 20;
    std::array<uint64_t, 4> device_used{};

    Shape checked(const UmziShape& req) const {
        const auto n = req.dimensions[0], h = req.dimensions[2], w = req.dimensions[3];
        if (req.rank != 4 || req.dimensions[1] != 3 || !n || !h || !w || n > 4096 || h > 65536 || w > 65536)
            throw std::runtime_error("MoSR requires positive RGB NCHW dimensions within native indexing limits");
        const int max_ch = std::max({2 * arch.hd(), 3 * arch.s() * arch.s() * arch.gps_ensemble(),
                                     arch.d() * arch.s() * arch.s()});
        if (uint64_t(n) * h * w * max_ch > INT_MAX - threads)
            throw std::runtime_error("Requested MoSR shape exceeds native tensor indexing limits");
        return {int(n), int(h), int(w)};
    }

    int arena_channels() const {
        const int d = arch.d(), hd = arch.hd(), s = arch.s();
        const int rgb_ch = fp32 ? 3 : rgb_lanes;
        const int proj_ch = fp32 ? 2 * hd : lanes(2 * hd);
        const int gated_ch = fp32 ? hd : lanes(hd);
        const int body_ch = rgb_ch + d + d + proj_ch + gated_ch;
        int up_ch = 0;
        if (arch.upsampler == UpsampleDYS) {
            const int off_ch = fp32 ? arch.offset_ch() : lanes(arch.offset_ch());
            const int end_ch = (fp32 ? 3 : rgb_lanes) * s * s;
            up_ch = d + std::max(2 * off_ch, end_ch) + d * s * s;
        } else {
            const int raw_up = 3 * s * s * arch.gps_ensemble();
            up_ch = d + (fp32 ? raw_up : lanes(raw_up));
        }
        return std::max(body_ch, up_ch);
    }

public:
    void open(const umzi::ModelPackage& package, const UmziSessionOptions& options) {
        if (package.schema != 1) throw std::runtime_error("Unsupported MoSR model payload version");
        arch = parse_architecture(package.section(1));
        if (package.section(4).size() != 32) throw std::runtime_error("Invalid MoSR metadata");
        model = Network(arch, false);
        const auto data = package.section(3);
        if (!package.section(2).empty() || data.size() != model.bytes)
            throw std::runtime_error("Invalid MoSR weight layout");
        if (options.precision != UMZI_FP32 && options.precision != UMZI_MIXED && options.precision != UMZI_FP16)
            throw std::runtime_error("Unsupported MoSR session precision");

        bool seen_workspace = false, seen_fused = false;
        for (uint32_t i = 0; i < options.kernel_option_count; ++i) {
            const auto& option = options.kernel_options[i];
            const std::string name = option.name ? option.name : "";
            if (name == "workspace_mib") {
                if (seen_workspace) throw std::runtime_error("Duplicate MoSR option: workspace_mib");
                if (option.value < 0 || uint64_t(option.value) > max_workspace_mib)
                    throw std::runtime_error("MoSR option workspace_mib must be between 0 and 4096");
                workspace_limit = uint64_t(option.value) << 20;
                seen_workspace = true;
            } else if (name == "fused_gate") {
                if (seen_fused) throw std::runtime_error("Duplicate MoSR option: fused_gate");
                if (option.value != 0 && option.value != 1)
                    throw std::runtime_error("MoSR option fused_gate must be 0 or 1");
                fused_gate_option = option.value != 0;
                seen_fused = true;
            } else {
                throw std::runtime_error("Unknown MoSR option: " + name + " (expected workspace_mib or fused_gate)");
            }
        }
        fp32 = options.precision == UMZI_FP32;
        fp16 = options.precision == UMZI_FP16;
        fused_gate = !fp32 && fused_gate_option && arch.d() == 64 && arch.hd() == 96 && (arch.k() == 7 || arch.k() == 17);
        device_model = Network(arch, !fp32);
        graph_enabled = (options.flags & UMZI_CUDA_GRAPH) != 0;

        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, options.device));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        shared->device = device;
        shared->context = context;
        ContextScope scope(context);
        const auto& image =
            umzi::load_kernel_module(device, module, mosr_images::embedded_cubins, mosr_images::embedded_ptx);
        shared->module = module;
        for (int i = 0; i < KernelCount; ++i) CHECK(cuModuleGetFunction(&functions[i], module, kernel_names[i]));
        for (Kernel k : {MosrExpandGateFp16, MosrExpandGateMixed, MosrProjectResNormFp16, MosrProjectResFp16,
                         MosrProjectResNormMixed, MosrProjectResMixed, MosrDwGatePlanarK7, MosrDwGatePlanarK17,
                         BiasNormNchwF32, MishResidualNormNchwF32}) {
            CHECK(cuFuncSetAttribute(functions[k], CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                     CU_SHAREDMEM_CARVEOUT_MAX_SHARED));
        }
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&complete, CU_EVENT_DISABLE_TIMING));
        for (auto& event : timing) CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        CHECK(cudnnCreate(&cudnn));
        shared->cudnn = cudnn;

        if (fp32) {
            CHECK(cuMemAlloc(&weights, data.size()));
            shared->weights = weights;
            CHECK(cuMemcpyHtoD(weights, data.data(), data.size()));
        } else {
            const auto packed = pack_half_weights(model, device_model, data.data(), fused_gate, arch.d(), arch.hd());
            CHECK(cuMemAlloc(&weights, packed.size()));
            shared->weights = weights;
            CHECK(cuMemcpyHtoD(weights, packed.data(), packed.size()));
        }
        initialize_cache(options.cache_dir, package.digest, image);
    }

    void initialize_cache(const char* directory, const umzi::Digest& model_digest, const umzi::KernelImage& image) {
        if (!directory || !*directory) return;
        try {
            cache_dir = std::filesystem::u8path(directory);
            umzi::Sha256 hash;
            umzi::hash_plan_environment(hash, umzi::mosr_id, plan_schema, model_digest, image, device);
            hash.add(static_cast<uint32_t>(fp32 ? 2 : (fp16 ? 1 : 0)));
            hash.add(static_cast<uint32_t>(fused_gate ? 1 : 0));
            hash.add(workspace_limit);
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
            plan_cache.open(cache_dir, "mosr", plan_key);
        } catch (const std::exception& error) {
            plan_cache.unavailable(error);
        }
    }

    ~Session() override {
        if (context && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            if (pending) cuEventSynchronize(complete);
            if (stream) cuStreamSynchronize(stream);
            release_shape();
            if (own_cudnn && cudnn) cudnnDestroy(cudnn);
            if (complete) cuEventDestroy(complete);
            for (auto event : timing)
                if (event) cuEventDestroy(event);
            if (stream) cuStreamDestroy(stream);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        shared.reset();
    }

    std::unique_ptr<umzi::KernelSession> clone_shared() const override {
        auto copy = std::make_unique<Session>();
        copy->copy_base_state(*this);
        copy->shared = shared;
        copy->arch = arch;
        copy->model = model;
        copy->device_model = device_model;
        copy->fp16 = fp16;
        copy->fp32 = fp32;
        copy->graph_enabled = graph_enabled;
        copy->device = device;
        copy->context = context;
        copy->module = module;
        copy->functions = functions;
        copy->weights = weights;
        copy->cache_dir = cache_dir;
        copy->cache_environment = cache_environment;
        copy->cache_environment_error = cache_environment_error;
        copy->workspace_limit = workspace_limit;
        ContextScope scope(context);
        copy->own_cudnn = true;
        CHECK(cudnnCreate(&copy->cudnn));
        CHECK(cuStreamCreate(&copy->stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&copy->complete, CU_EVENT_DISABLE_TIMING));
        for (auto& event : copy->timing) CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        return copy;
    }

private:
    void release_shape() {
        if (executable) cuGraphExecDestroy(executable);
        executable = nullptr;
        variants.clear();
        plans.clear();
        plan_cache = {};
        for (auto* buffer : {&stage_in, &stage_out, &scratch, &arena}) {
            if (*buffer) cuMemFree(*buffer);
            *buffer = 0;
        }
        rgb_buf = state = norm_out = projected = gated = 0;
        upsample_buf = dys_off_buf = dys_scp_buf = dys_smp_buf = dys_end_buf = 0;
        shape = {};
        activation_bytes = workspace_bytes = scratch_bytes = peak_workspace_bytes = 0;
        setup_ms = {};
        device_used = {};
        timed_plans = 0;
        fp16_fallback_plans = 0;
    }

    void wait() {
        if (!pending) return;
        CHECK(cuEventSynchronize(complete));
        pending = false;
    }

    static umzi::PlanShape shape_key(int h, int w, int cin, int cout, int k) {
        return {h, w, cin, cout, k};
    }

    void make_plan(int h, int w, int cin, int cout, int k, CUdeviceptr in_buf, CUdeviceptr out_buf) {
        const auto key = shape_key(h, w, cin, cout, k);
        if (plans.count(key)) return;
        auto plan = std::make_unique<umzi::CudnnBackendPlan>();
        auto& p = *plan;
        const auto io_type = fp32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
        const auto layout = fp32 ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
        bool half_compute = fp16;
        p.graph = std::make_unique<umzi::CudnnConvGraph>(cudnn, shape.n, h, w, cin, cout, k, io_type, half_compute,
                                                         layout);
        auto candidates = p.graph->candidates(cudnn, workspace_limit);
        if (candidates.empty() && half_compute) {
            half_compute = false;
            p.graph = std::make_unique<umzi::CudnnConvGraph>(cudnn, shape.n, h, w, cin, cout, k, io_type, false, layout);
            candidates = p.graph->candidates(cudnn, workspace_limit);
            ++fp16_fallback_plans;
        }
        auto time_candidate = [&](float best) {
            return umzi::time_cudnn_backend_plan(cudnn, stream, timing, p, in_buf, weights, out_buf, scratch,
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
                    workspace_bytes = std::max(workspace_bytes, p.workspace);
                    plans.emplace(key, std::move(plan));
                    return;
                }
                p.plan.reset();
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
            throw std::runtime_error("No cuDNN convolution for MoSR " + std::to_string(cin) + " to "
                                     + std::to_string(cout) + " (" + std::to_string(k) + "x" + std::to_string(k)
                                     + ") within the workspace limit");
        p.plan = std::move(candidates[fastest].plan);
        p.workspace = size_t(candidates[fastest].workspace);
        workspace_bytes = std::max(workspace_bytes, p.workspace);
        ++timed_plans;
        plan_cache.convolutions.insert_or_assign(key,
                                                 umzi::ConvChoice{uint32_t(fastest), saved_plan_marker, p.workspace});
        plans.emplace(key, std::move(plan));
    }

    void prepare(Shape requested) {
        if (shape == requested) return;
        wait();
        release_shape();
        try {
            auto start = Clock::now();
            device_used[0] = umzi::device_used_bytes();
            shape = requested;
            const size_t pixels = size_t(shape.pixels());
            const size_t elem = fp32 ? 4 : 2;
            const int d = arch.d(), hd = arch.hd(), s = arch.s();
            const int rgb_ch = fp32 ? 3 : rgb_lanes;
            const int proj_ch = fp32 ? 2 * hd : lanes(2 * hd);
            const int gated_ch = fp32 ? hd : lanes(hd);
            const int total_ch = arena_channels();
            activation_bytes = pixels * size_t(total_ch) * elem;
            const size_t stage_bytes = graph_enabled ? shape.rgb_bytes(fp16) * (1 + s * s) : 0;
            umzi::require_device_memory(activation_bytes + stage_bytes, "MoSR shape buffers", device);
            CHECK(cuMemAlloc(&arena, activation_bytes));

            state = arena;
            norm_out = state + pixels * size_t(d) * elem;
            projected = norm_out + pixels * size_t(d) * elem;
            gated = projected + pixels * size_t(proj_ch) * elem;
            rgb_buf = gated + pixels * size_t(gated_ch) * elem;
            upsample_buf = state + pixels * size_t(d) * elem;

            if (arch.upsampler == UpsampleDYS) {
                const int off_ch = fp32 ? arch.offset_ch() : lanes(arch.offset_ch());
                const int end_ch = (fp32 ? 3 : rgb_lanes) * s * s;
                const size_t off_bytes = pixels * size_t(off_ch) * elem;
                const size_t off_region = pixels * size_t(std::max(2 * off_ch, end_ch)) * elem;
                dys_off_buf = state + pixels * size_t(d) * elem;
                dys_scp_buf = dys_off_buf + off_bytes;
                dys_smp_buf = dys_off_buf + off_region;
                dys_end_buf = dys_off_buf;
            }
            device_used[1] = umzi::device_used_bytes();
            setup_ms[0] = umzi::elapsed_ms(start);

            start = Clock::now();
            CHECK(cudnnSetStream(cudnn, stream));
            const int noise_count = int(pixels * size_t(total_ch));
            if (fp32) elementwise(stream, FillNoiseF32, noise_count, arena);
            else elementwise(stream, FillNoiseF16, noise_count, arena);

            load_plan_cache();
            const int h = shape.h, w = shape.w;
            make_plan(h, w, rgb_ch, d, 3, rgb_buf, state);
            if (!fused_gate) {
                make_plan(h, w, d, proj_ch, 3, norm_out, projected);
                make_plan(h, w, gated_ch, d, 3, gated, norm_out);
            }
            make_plan(h, w, d, 2 * d, 3, state, projected);
            make_plan(h, w, 2 * d, d, 3, projected, gated);
            make_plan(h, w, d, d, 1, gated, state);
            make_plan(h, w, d, d, 3, norm_out, gated);
            make_plan(h, w, rgb_ch, d, 1, rgb_buf, projected);
            if (arch.upsampler == UpsampleDYS) {
                const int off_ch = fp32 ? arch.offset_ch() : lanes(arch.offset_ch());
                const int end_out = fp32 ? 3 : rgb_lanes;
                make_plan(h, w, d, off_ch, 1, state, dys_off_buf);
                make_plan(s * h, s * w, d, end_out, 1, dys_smp_buf, dys_end_buf);
            } else {
                const int up_raw = 3 * s * s * arch.gps_ensemble();
                const int up_ch = fp32 ? up_raw : lanes(up_raw);
                make_plan(h, w, d, up_ch, 3, state, upsample_buf);
            }
            CHECK(cuStreamSynchronize(stream));
            device_used[2] = umzi::device_used_bytes();
            if (scratch) CHECK(cuMemFree(scratch));
            scratch = 0;
            scratch_bytes = 0;
            if (!umzi::ensure_device_scratch(stream, scratch, scratch_bytes, workspace_bytes, &peak_workspace_bytes))
                CHECK(CUDA_ERROR_OUT_OF_MEMORY);
            plan_cache.save(plan_key);
            setup_ms[1] = umzi::elapsed_ms(start);

            if (graph_enabled) {
                start = Clock::now();
                ensure_staging();
                CHECK(cuMemsetD8Async(stage_in, 0, shape.rgb_bytes(fp16), stream));
                forward(stage_in, stage_out, stream);
                CHECK(cuStreamSynchronize(stream));
                setup_ms[2] = umzi::elapsed_ms(start);
                start = Clock::now();
                umzi::capture_cuda_graph(stream, executable, [&] { forward(stage_in, stage_out, stream); });
                setup_ms[3] = umzi::elapsed_ms(start);
            }
            device_used[3] = umzi::device_used_bytes();
        } catch (...) {
            cuStreamSynchronize(stream);
            release_shape();
            throw;
        }
    }

    template<class... Args>
    void launch(CUstream target, Kernel kernel, int count, Args... args) {
        void* parameters[] = {&args...};
        CHECK(cuLaunchKernel(functions[kernel], unsigned((count + threads - 1) / threads), 1, 1, threads, 1, 1, 0,
                             target, parameters, nullptr));
    }

    template<class... Args>
    void elementwise(CUstream target, Kernel kernel, int count, Args... args) {
        launch(target, kernel, count, args..., count);
    }

    void conv(CUdeviceptr out, CUdeviceptr in, int h, int w, const Filter& filter) {
        const auto& p = *plans.at(shape_key(h, w, filter.cin, filter.cout, filter.k));
        auto& variant = variants[std::array<CUdeviceptr, 4>{out, in, weights + filter.weight, scratch}];
        if (!variant)
            variant = umzi::bind_conv_variant(reinterpret_cast<void*>(in), reinterpret_cast<void*>(weights + filter.weight),
                                              reinterpret_cast<void*>(out), reinterpret_cast<void*>(scratch));
        CHECK(cudnnBackendExecute(cudnn, p.plan->get(), variant->get()));
    }

    void forward_fp32(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        const int h = shape.h, w = shape.w, hw = h * w, pixels = shape.pixels();
        const int d = arch.d(), hd = arch.hd(), k = arch.k(), s = arch.s();
        const auto& net = device_model;
        elementwise(target, CopyNchwF32, pixels * 3, rgb_buf, input);
        conv(state, rgb_buf, h, w, net.stem);
        const unsigned norm_f32_smem = (d <= 64) ? unsigned(128 * d) * sizeof(float) : 0u;
        const unsigned norm_f32_threads = (d <= 64) ? 128u : 256u;
        const unsigned norm_f32_grid = unsigned((pixels + int(norm_f32_threads) - 1) / int(norm_f32_threads));
        if (net.blocks.empty()) {
            elementwise(target, AddBiasNchwF32, pixels * d, state, state, weights + net.stem.bias, d, hw);
        } else {
            CUdeviceptr st_bias = weights + net.stem.bias, nw = weights + net.blocks[0].norm_weight,
                        nb = weights + net.blocks[0].norm_bias;
            int d_arg = d, hw_arg = hw, pix_arg = pixels;
            void* params[] = {&state, &norm_out, &st_bias, &nw, &nb, &d_arg, &hw_arg, &pix_arg};
            CHECK(cuLaunchKernel(functions[BiasNormNchwF32], norm_f32_grid, 1, 1, norm_f32_threads, 1, 1, norm_f32_smem,
                                 target, params, nullptr));
        }

        const int tiles_x = (w + 15) / 16, tiles_y = (h + 15) / 16;
        const int radius = k / 2;
        const unsigned smem_bytes = unsigned((16 + 2 * radius) * (16 + 2 * radius)) * sizeof(float);
        const int plain = hd - d;
        const Kernel dw_f32_kernel = k == 7 ? DwGateNchwF32K7 : (k == 17 ? DwGateNchwF32K17 : DwGateNchwF32);
        const unsigned dw_f32_threads = (k == 7 || k == 17) ? 128u : 256u;
        for (size_t idx = 0; idx < net.blocks.size(); ++idx) {
            const auto& blk = net.blocks[idx];
            conv(projected, norm_out, h, w, blk.fc1);
            if (plain > 0) {
                elementwise(target, PlainGateNchwF32, pixels * plain, gated, projected, weights + blk.fc1.bias, d, hd,
                            hw);
            }
            {
                CUdeviceptr g_out = gated, p_in = projected;
                CUdeviceptr fc1_b = weights + blk.fc1.bias, dw_w = weights + blk.dw.weight, dw_b = weights + blk.dw.bias;
                int h_arg = h, w_arg = w, d_arg = d, hd_arg = hd, k_arg = k, tx_arg = tiles_x;
                void* params[] = {&g_out, &p_in, &fc1_b, &dw_w, &dw_b, &d_arg, &hd_arg, &k_arg, &h_arg, &w_arg, &tx_arg};
                CHECK(cuLaunchKernel(functions[dw_f32_kernel], unsigned(tiles_x * tiles_y), unsigned(d), unsigned(shape.n),
                                     dw_f32_threads, 1, 1, smem_bytes, target, params, nullptr));
            }
            conv(norm_out, gated, h, w, blk.fc2);
            if (idx + 1 < net.blocks.size()) {
                const auto& next_blk = net.blocks[idx + 1];
                CUdeviceptr fc2_b = weights + blk.fc2.bias, nw = weights + next_blk.norm_weight,
                            nb = weights + next_blk.norm_bias;
                int d_arg = d, hw_arg = hw, pix_arg = pixels;
                void* params[] = {&state, &norm_out, &norm_out, &fc2_b, &nw, &nb, &d_arg, &hw_arg, &pix_arg};
                CHECK(cuLaunchKernel(functions[MishResidualNormNchwF32], norm_f32_grid, 1, 1, norm_f32_threads, 1, 1,
                                     norm_f32_smem, target, params, nullptr));
            } else {
                elementwise(target, MishBiasResidualNchwF32, pixels * d, state, norm_out, weights + blk.fc2.bias, d, hw);
            }
        }

        conv(projected, state, h, w, net.tail0);
        elementwise(target, MishBiasInplaceNchwF32, pixels * 2 * d, projected, weights + net.tail0.bias, 2 * d, hw);
        conv(gated, projected, h, w, net.tail1);
        elementwise(target, MishBiasInplaceNchwF32, pixels * d, gated, weights + net.tail1.bias, d, hw);
        conv(state, gated, h, w, net.tail2);

        conv(norm_out, rgb_buf, h, w, net.sc0);
        elementwise(target, MishBiasInplaceNchwF32, pixels * d, norm_out, weights + net.sc0.bias, d, hw);
        conv(gated, norm_out, h, w, net.sc1);
        elementwise(target, MishBiasInplaceNchwF32, pixels * d, gated, weights + net.sc1.bias, d, hw);
        conv(projected, rgb_buf, h, w, net.sc11);
        elementwise(target, CombineBodyShortcutNchwF32, pixels * d, state, weights + net.tail2.bias, gated, projected,
                    weights + net.sc11.bias, d, hw);

        if (arch.upsampler == UpsampleDYS) {
            conv(dys_off_buf, state, h, w, net.dys_offset);
            conv(dys_scp_buf, state, h, w, net.dys_scope);
            elementwise(target, DysampleNchwF32, pixels * d * s * s, dys_smp_buf, state, dys_off_buf,
                        weights + net.dys_offset.bias, dys_scp_buf, d, arch.groups(), s, h, w);
            conv(dys_end_buf, dys_smp_buf, s * h, s * w, net.dys_end);
            elementwise(target, AddBiasNchwF32, pixels * 3 * s * s, output, dys_end_buf, weights + net.dys_end.bias, 3,
                        hw * s * s);
        } else {
            conv(upsample_buf, state, h, w, net.upsample_conv);
            elementwise(target, GpsShuffleNchwF32, pixels * 3 * s * s, output, upsample_buf,
                        weights + net.upsample_conv.bias, 3, s, h, w, arch.gps_ensemble());
        }
    }

    void forward(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        if (fp32) {
            forward_fp32(input, output, target);
            return;
        }
        const int h = shape.h, w = shape.w, pixels = shape.pixels();
        const int d = arch.d(), hd = arch.hd(), k = arch.k(), s = arch.s();
        const int fc1_stride = lanes(2 * hd), gated_stride = lanes(hd);
        const auto& net = device_model;
        elementwise(target, fp16 ? PackRgbNhwcF16 : PackRgbNhwcF32, pixels * rgb_lanes, rgb_buf, input, h, w);
        conv(state, rgb_buf, h, w, net.stem);
        if (net.blocks.empty()) {
            launch(target, AddBiasNhwc, (pixels * d) / 2, state, weights + net.stem.bias, d, (pixels * d) / 2);
        } else {
            launch(target, BiasNormNhwc, pixels * 32, state, norm_out, weights + net.stem.bias,
                   weights + net.blocks[0].norm_weight, weights + net.blocks[0].norm_bias, d, pixels);
        }

        const int tiles_x = (w + 15) / 16, tiles_y = (h + 15) / 16;
        if (fused_gate) {
            const int exp_tx = (w + 15) / 16, exp_ty = (h + 7) / 8;
            const int dw_tile_w = (k == 7) ? 32 : 16;
            const int dw_tx = (w + dw_tile_w - 1) / dw_tile_w;
            const unsigned exp_grid = unsigned(exp_tx * exp_ty * shape.n);
            const unsigned dw_grid = unsigned(dw_tx * tiles_y);
            const unsigned proj_grid = unsigned(tiles_x * tiles_y * shape.n);
            const unsigned exp_smem = 48384u;
            const unsigned dw_threads = (k == 7) ? 256u : 128u;
            const unsigned dw_smem = unsigned((16 + k - 1) * (dw_tile_w + k - 1) * 16 + k * k * 32);
            const unsigned proj_smem = 33024u;
            const size_t elem = 2;
            CUdeviceptr plain_gated_buf = projected;
            CUdeviceptr dw_v_buf = plain_gated_buf + size_t(pixels) * 32 * elem;
            CUdeviceptr dw_g_buf = dw_v_buf + size_t(pixels) * 64 * elem;
            CUdeviceptr dw_gated_buf = dw_g_buf + size_t(pixels) * 64 * elem;
            const Kernel expand_k = fp16 ? MosrExpandGateFp16 : MosrExpandGateMixed;
            const Kernel dw_planar_k = k == 7 ? MosrDwGatePlanarK7 : MosrDwGatePlanarK17;
            const Kernel proj_norm_k = fp16 ? MosrProjectResNormFp16 : MosrProjectResNormMixed;
            const Kernel proj_last_k = fp16 ? MosrProjectResFp16 : MosrProjectResMixed;

            for (size_t idx = 0; idx < net.blocks.size(); ++idx) {
                const auto& blk = net.blocks[idx];
                {
                    CUdeviceptr fc1_w = weights + blk.fc1.weight, fc1_b = weights + blk.fc1.bias;
                    int h_arg = h, w_arg = w, ex_arg = exp_tx, ey_arg = exp_ty, pix_arg = pixels;
                    void* params[] = {&plain_gated_buf, &dw_v_buf, &dw_g_buf, &norm_out, &fc1_w, &fc1_b,
                                      &h_arg,           &w_arg,    &ex_arg,   &ey_arg,   &pix_arg};
                    CHECK(cuLaunchKernel(functions[expand_k], exp_grid, 1, 1, 384, 1, 1, exp_smem, target, params,
                                         nullptr));
                }
                {
                    CUdeviceptr dw_w = weights + blk.dw.weight, dw_b = weights + blk.dw.bias;
                    int h_arg = h, w_arg = w, tx_arg = dw_tx, pix_arg = pixels;
                    void* params[] = {&dw_gated_buf, &dw_v_buf, &dw_g_buf, &dw_w,   &dw_b,
                                      &h_arg,        &w_arg,    &tx_arg,   &pix_arg};
                    CHECK(cuLaunchKernel(functions[dw_planar_k], dw_grid, 8, unsigned(shape.n), dw_threads, 1, 1,
                                         dw_smem, target, params, nullptr));
                }
                {
                    const bool do_norm = idx + 1 < net.blocks.size();
                    const Kernel pk = do_norm ? proj_norm_k : proj_last_k;
                    CUdeviceptr fc2_w = weights + blk.fc2.weight, fc2_b = weights + blk.fc2.bias;
                    CUdeviceptr nw = do_norm ? (weights + net.blocks[idx + 1].norm_weight) : weights;
                    CUdeviceptr nb = do_norm ? (weights + net.blocks[idx + 1].norm_bias) : weights;
                    int h_arg = h, w_arg = w, tx_arg = tiles_x, ty_arg = tiles_y, pix_arg = pixels;
                    void* params[] = {&state,  &norm_out, &plain_gated_buf, &dw_gated_buf, &fc2_w, &fc2_b, &nw,
                                      &nb,     &h_arg,    &w_arg,           &tx_arg,       &ty_arg, &pix_arg};
                    CHECK(cuLaunchKernel(functions[pk], proj_grid, 1, 1, 256, 1, 1, proj_smem, target, params, nullptr));
                }
            }
        } else {
            const int radius = k / 2;
            const int plain = hd - d;
            const bool use_plain_vec8 =
                plain > 0 && (plain % 8 == 0) && (fc1_stride % 8 == 0) && (gated_stride % 8 == 0) && (gated_stride == hd);
            const bool use_dw_8ch = (d % 8 == 0) && ((2 * hd - d) % 8 == 0) && ((hd - d) % 8 == 0) &&
                                    (fc1_stride % 8 == 0) && (gated_stride % 8 == 0) && (k == 7 || k == 17);
            const unsigned smem_bytes =
                unsigned((16 + 2 * radius) * (16 + 2 * radius) * (use_dw_8ch ? 8 : 4)) * sizeof(float);
            for (size_t idx = 0; idx < net.blocks.size(); ++idx) {
                const auto& blk = net.blocks[idx];
                conv(projected, norm_out, h, w, blk.fc1);
                if (plain > 0) {
                    if (use_plain_vec8) {
                        elementwise(target, PlainGateNhwcVec8, pixels * (plain >> 3), gated, projected,
                                    weights + blk.fc1.bias, d, hd, fc1_stride, gated_stride);
                    } else {
                        elementwise(target, PlainGateNhwc, pixels * plain, gated, projected, weights + blk.fc1.bias, d, hd,
                                    fc1_stride, gated_stride);
                    }
                }
                {
                    CUdeviceptr g_out = gated, p_in = projected;
                    CUdeviceptr fc1_b = weights + blk.fc1.bias, dw_w = weights + blk.dw.weight, dw_b = weights + blk.dw.bias;
                    int d_arg = d, hd_arg = hd, fs_arg = fc1_stride, gs_arg = gated_stride, k_arg = k, h_arg = h, w_arg = w,
                        tx_arg = tiles_x;
                    if (use_dw_8ch) {
                        void* params[] = {&g_out,  &p_in,   &fc1_b,  &dw_w,  &dw_b,  &d_arg,
                                          &hd_arg, &fs_arg, &gs_arg, &h_arg, &w_arg, &tx_arg};
                        const Kernel dw_8ch = k == 7 ? DwGateNhwc8chK7 : DwGateNhwc8chK17;
                        CHECK(cuLaunchKernel(functions[dw_8ch], unsigned(tiles_x * tiles_y), unsigned(d / 8),
                                             unsigned(shape.n), 256, 1, 1, smem_bytes, target, params, nullptr));
                    } else {
                        void* params[] = {&g_out,  &p_in,   &fc1_b,  &dw_w,  &dw_b,  &d_arg,  &hd_arg,
                                          &fs_arg, &gs_arg, &k_arg,  &h_arg, &w_arg, &tx_arg};
                        CHECK(cuLaunchKernel(functions[DwGateNhwc], unsigned(tiles_x * tiles_y), unsigned(d / 4),
                                             unsigned(shape.n), 256, 1, 1, smem_bytes, target, params, nullptr));
                    }
                }
                conv(norm_out, gated, h, w, blk.fc2);
                if (idx + 1 < net.blocks.size()) {
                    const auto& next_blk = net.blocks[idx + 1];
                    launch(target, MishResidualNormNhwc, pixels * 32, state, norm_out, norm_out, weights + blk.fc2.bias,
                           weights + next_blk.norm_weight, weights + next_blk.norm_bias, d, pixels);
                } else {
                    launch(target, MishBiasResidualNhwc, (pixels * d) / 2, state, norm_out, weights + blk.fc2.bias, d,
                           (pixels * d) / 2);
                }
            }
        }

        conv(projected, state, h, w, net.tail0);
        launch(target, MishBiasInplaceNhwc, pixels * d, projected, weights + net.tail0.bias, 2 * d, pixels * d);
        conv(gated, projected, h, w, net.tail1);
        launch(target, MishBiasInplaceNhwc, (pixels * d) / 2, gated, weights + net.tail1.bias, d, (pixels * d) / 2);
        conv(state, gated, h, w, net.tail2);

        conv(norm_out, rgb_buf, h, w, net.sc0);
        launch(target, MishBiasInplaceNhwc, (pixels * d) / 2, norm_out, weights + net.sc0.bias, d, (pixels * d) / 2);
        conv(gated, norm_out, h, w, net.sc1);
        launch(target, MishBiasInplaceNhwc, (pixels * d) / 2, gated, weights + net.sc1.bias, d, (pixels * d) / 2);
        conv(projected, rgb_buf, h, w, net.sc11);
        launch(target, CombineBodyShortcutNhwc, (pixels * d) / 2, state, weights + net.tail2.bias, gated, projected,
               weights + net.sc11.bias, d, (pixels * d) / 2);

        if (arch.upsampler == UpsampleDYS) {
            const int off_stride = lanes(arch.offset_ch());
            conv(dys_off_buf, state, h, w, net.dys_offset);
            conv(dys_scp_buf, state, h, w, net.dys_scope);
            elementwise(target, DysampleNhwc, pixels * s * s * arch.groups(), dys_smp_buf, state, dys_off_buf,
                        weights + net.dys_offset.bias, dys_scp_buf, off_stride, d, arch.groups(), s, h, w);
            conv(dys_end_buf, dys_smp_buf, s * h, s * w, net.dys_end);
            elementwise(target, fp16 ? UnpackRgbNhwcF16 : UnpackRgbNhwcF32, pixels * 3 * s * s, output, dys_end_buf,
                        weights + net.dys_end.bias, 3, rgb_lanes, s * h, s * w);
        } else {
            const int up_stride = lanes(3 * s * s * arch.gps_ensemble());
            conv(upsample_buf, state, h, w, net.upsample_conv);
            elementwise(target, fp16 ? GpsShuffleNhwcF16 : GpsShuffleNhwcF32, pixels * 3 * s * s, output, upsample_buf,
                        weights + net.upsample_conv.bias, 3, s, h, w, arch.gps_ensemble(), up_stride);
        }
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
        if (output != stage_out) CHECK(cuMemcpyDtoDAsync(output, stage_out, input_bytes * arch.s() * arch.s(), target));
    }

public:
    UmziModelInfo model_info() const override { return {sizeof(UmziModelInfo), "mosr", arch.scale, 0}; }
    umzi::TilingConstraints tiling_constraints() const override { return {1, 1, 0}; }

    UmziIo tensors() const override {
        const auto dtype = uint32_t(fp16 ? UMZI_FLOAT16 : UMZI_FLOAT32);
        return {sizeof(UmziIo), {"input", {4, {0, 3, 0, 0}}, dtype, 0}, {"output", {4, {0, 3, 0, 0}}, dtype, 0}};
    }

    UmziIo describe(const UmziShape& request) const override {
        const auto concrete = checked(request);
        auto io = tensors();
        io.input.shape = request;
        io.input.bytes = concrete.rgb_bytes(fp16);
        io.output.shape = Shape{concrete.n, arch.s() * concrete.h, arch.s() * concrete.w}.nchw();
        io.output.bytes = io.input.bytes * arch.s() * arch.s();
        return io;
    }

    UmziMemoryInfo estimate_memory(const UmziShape& request) const override {
        const auto concrete = checked(request);
        ContextScope scope(context);
        const auto vram = umzi::device_vram_info(device);
        const size_t elem = fp32 ? 4 : 2;
        UmziMemoryInfo info{};
        info.struct_size = sizeof(info);
        info.buffer_bytes[0] = device_model.bytes;
        info.buffer_bytes[1] = uint64_t(concrete.pixels()) * arena_channels() * elem;
        info.buffer_bytes[2] = (shape == concrete) ? workspace_bytes : workspace_limit;
        info.buffer_bytes[3] = graph_enabled ? concrete.rgb_bytes(fp16) * (1 + arch.s() * arch.s()) : 0;
        info.device_free = vram.free_bytes;
        info.device_total = vram.total_bytes;
        return info;
    }

    void prepare_shape(const UmziShape& request) override {
        ContextScope scope(context);
        prepare(checked(request));
    }

    void ensure_staging() {
        if (stage_in && stage_out) return;
        const auto input_bytes = shape.rgb_bytes(fp16);
        const size_t out_bytes = size_t(input_bytes) * arch.s() * arch.s();
        const size_t needed = (stage_out ? 0 : out_bytes) + (stage_in ? 0 : size_t(input_bytes));
        umzi::require_device_memory(needed, "MoSR staging buffers", device);
        if (!stage_out) CHECK(cuMemAlloc(&stage_out, out_bytes));
        if (!stage_in) CHECK(cuMemAlloc(&stage_in, input_bytes));
    }

    void run_host(const UmziShape& request, const void* input, void* output) override {
        ContextScope scope(context);
        prepare(checked(request));
        wait();
        ensure_staging();
        const auto input_bytes = shape.rgb_bytes(fp16);
        try {
            CHECK(cuMemcpyHtoDAsync(stage_in, input, input_bytes, stream));
            execute(stage_in, stage_out, stream);
            CHECK(cuMemcpyDtoHAsync(output, stage_out, input_bytes * arch.s() * arch.s(), stream));
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
            if (pending) CHECK(cuStreamWaitEvent(target, complete, CU_EVENT_WAIT_DEFAULT));
            execute(input, output, target);
            pending = true;
            CHECK(cuEventRecord(complete, target));
        } catch (...) {
            if (const auto failure = umzi::drain_failure(target); !failure.empty()) {
                pending = true;
                warn_option(failure);
            }
            throw;
        }
    }

    const char* cache_message() const override { return plan_cache.diagnostic.c_str(); }
    const char* kernel_message() const override {
        if (fp32) return "MoSR NCHW FP32";
        if (fp16) return fused_gate ? "MoSR NHWC FP16 (fused tensor-core gate)" : "MoSR NHWC FP16";
        return fused_gate ? "MoSR NHWC mixed precision (fused tensor-core gate)" : "MoSR NHWC mixed precision";
    }

    UmziSessionInfo session_info() const override {
        UmziSessionInfo info{};
        info.struct_size = sizeof(info);
        info.cache_status = plan_cache.status;
        info.buffer_bytes[0] = device_model.bytes;
        if (shape.n) {
            info.prepared_input = shape.nchw();
            info.buffer_bytes[1] = activation_bytes;
            info.buffer_bytes[2] = workspace_bytes;
            info.buffer_bytes[3] = stage_in ? shape.rgb_bytes(fp16) * (1 + arch.s() * arch.s()) : 0;
            std::copy(device_used.begin(), device_used.end(), info.startup_device_used_bytes);
            std::copy(setup_ms.begin(), setup_ms.end(), info.shape_setup_ms);
        }
        return info;
    }

    uint32_t diagnostic_count() const override { return 4; }
    UmziDiagnostic diagnostic(uint32_t index) const override {
        if (index == 0) return {sizeof(UmziDiagnostic), "timed_plans", timed_plans};
        if (index == 1) return {sizeof(UmziDiagnostic), "plan_workspace_bytes", int64_t(workspace_bytes)};
        if (index == 2) return {sizeof(UmziDiagnostic), "peak_plan_workspace_bytes", int64_t(peak_workspace_bytes)};
        if (index == 3) return {sizeof(UmziDiagnostic), "fp16_fallback_plans", fp16_fallback_plans};
        throw std::runtime_error("Invalid MoSR diagnostic index");
    }
};
} // namespace

std::unique_ptr<umzi::KernelSession> umzi::make_mosr(const ModelPackage& package, const UmziSessionOptions& options) {
    auto session = std::make_unique<Session>();
    session->open(package, options);
    return session;
}
