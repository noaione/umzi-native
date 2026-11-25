#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <cuda.h>
#include <cudnn.h>

#define UMZI_BUILD
#include "moesr_images.h"
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
#include <map>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

namespace {
constexpr int channels = 64, scale = 4, full_hidden = 170, reduced_hidden = 96, threads = 256;
// Device layouts pad channel counts to multiples of 8 so cuDNN's tensor-core
// kernels need no separate padding pass: RGB to 8, the 170-wide hidden to 176,
// and the expansion to 8 channels per group of four hidden slots (344, 192).
constexpr int rgb_lanes = 8;
constexpr int lanes(int count) { return (count + 7) & ~7; }
constexpr int expand_width(int hidden) { return 8 * ((hidden + 3) / 4); }
// Shared memory for the 8x16 fused expansion+gate kernels (180-pixel halo +
// K=32 weight ring buffer) and 16x16 fused projection+residual+norm kernels
// (two 324-pixel K=16 halos + two 3-tap K=16 weight rows = 33,024 B).
constexpr int fused_rows = 8, fused_cols = 16, halo_halves = 180 * 64;
constexpr int fused_shared_full_f16 = (halo_halves + 2 * 192 * 32) * 2;
constexpr int fused_shared_full_f32 = (halo_halves + 3 * 88 * 32) * 2;
constexpr int fused_shared_reduced_f16 = (halo_halves + 2 * 192 * 32) * 2;
constexpr int fused_shared_reduced_f32 = (halo_halves + 3 * 96 * 32) * 2;
constexpr int project_shared_bytes = (2 * 324 * 16 + 2 * 3 * 64 * 16) * 2;
constexpr uint64_t default_workspace_mib = 512, max_workspace_mib = 4096;
// Bump when plan keys, descriptor semantics, or the choice format change.
constexpr uint32_t plan_schema = 6;
constexpr std::array<uint32_t, 10> configuration{3, 3, scale, 2, channels, 9, 4, 3, full_hidden, reduced_hidden};
using umzi::Clock;
using umzi::ContextScope;
#define CHECK(expr) umzi::check((expr), #expr)

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
    if (uint64_t(n) * ((h + 1) & ~1u) * ((w + 1) & ~1u) * (8 * ((full_hidden + 3) / 4)) > INT_MAX - threads)
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

// With padded = false the offsets match the packer's section 3. With padded =
// true they describe the device copy, whose first, expand and project filters
// use the padded channel counts of the device layout. When fused = true the
// full-resolution expansion weight is padded to 352 channels for swizzle32
// K=32 passes (192 + 160 for FP16, 4 x 88 for FP32).
struct Network {
    Filter first, last;
    std::array<Group, configuration[5]> groups;
    size_t bytes = 0;
    bool padded = false, fused = false;

    explicit Network(bool padded_layout = false, bool fused_layout = false)
        : padded(padded_layout), fused(fused_layout) {
        first = dense(padded ? rgb_lanes : 3, channels);
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
        const int width = padded ? lanes(hidden) : hidden;
        const int expand_cout = padded ? expand_width(hidden) : 2 * hidden;
        const int weight_cout = (padded && fused && hidden == full_hidden) ? 352 : expand_cout;
        Block block;
        block.norm_weight = take(channels * 4);
        block.norm_bias = take(channels * 4);
        block.expand = {take(size_t(channels) * weight_cout * 9 * 2), take(size_t(expand_cout) * 4)};
        block.square = depthwise(9);
        block.row = depthwise(11);
        block.column = depthwise(11);
        block.project = dense(width, channels);
        block.gamma = take(channels * 4);
        return block;
    }
};

// Copy a packed FP16 KRSC filter and its FP32 bias into a zero-filled device
// layout. Output channel o moves to out_slot(o) and input channel i to
// in_slot(i); to_cin is the device input stride. Unmapped slots stay zero.
template<class OutSlot, class InSlot>
void copy_dense(std::vector<char>& out, const char* in, Filter to, Filter from, int cin, int cout, int to_cin,
                OutSlot out_slot, InSlot in_slot) {
    for (int o = 0; o < cout; ++o) {
        const int target = out_slot(o);
        for (int tap = 0; tap < 9; ++tap)
            for (int i = 0; i < cin; ++i)
                std::memcpy(out.data() + to.weight + ((size_t(target) * 9 + tap) * to_cin + in_slot(i)) * 2,
                            in + from.weight + ((size_t(o) * 9 + tap) * cin + i) * 2, 2);
        std::memcpy(out.data() + to.bias + size_t(target) * 4, in + from.bias + size_t(o) * 4, 4);
    }
}

// Pack one N-pass [ch_start .. ch_start + ntile) of a logical [total_ch, 9, 64]
// FP16 weight into 18 contiguous swizzle32 K=32 stages (9 taps x 2 k-halves).
void pack_swizzle32_pass(char* dst, const uint16_t* logical, int total_ch, int ch_start, int ntile) {
    for (int step = 0; step < 18; ++step) {
        const int tap = step / 2, kh = step % 2;
        for (int row = 0; row < ntile; ++row) {
            const int ch = ch_start + row;
            if (ch >= total_ch) continue;
            const int mask = (row >> 1) & 3;
            for (int chunk = 0; chunk < 4; ++chunk) {
                const size_t dst_halves = (size_t(step * ntile + row) * 4 + (chunk ^ mask)) * 8;
                const size_t src_halves = (size_t(ch) * 9 + tap) * 64 + kh * 32 + chunk * 8;
                std::memcpy(dst + dst_halves * 2, logical + src_halves, 16);
            }
        }
    }
}

// Pack a logical [64, 9, width] FP16 projection weight into [width/16, 9, 64, 2, 8]
// swizzle16 K=16 stages (same total byte count as KRSC).
void pack_swizzle16_project(char* dst, const uint16_t* logical, int width) {
    const int slices = width / 16;
    for (int ks = 0; ks < slices; ++ks) {
        for (int tap = 0; tap < 9; ++tap) {
            for (int row = 0; row < channels; ++row) {
                const int mask = (row >> 2) & 1;
                for (int chunk = 0; chunk < 2; ++chunk) {
                    const size_t dst_halves = (((size_t(ks) * 9 + tap) * channels + row) * 2 + (chunk ^ mask)) * 8;
                    const size_t src_halves = (size_t(row) * 9 + tap) * width + ks * 16 + chunk * 8;
                    std::memcpy(dst + dst_halves * 2, logical + src_halves, 16);
                }
            }
        }
    }
}

// Slot of hidden channel k in a gated block's unfused device layout: the 24
// depthwise branch channels come first, then the plain ones.
int unfused_hidden_slot(int k, int hidden) {
    const int branch = hidden - 24;
    return k >= branch ? k - branch : 24 + k;
}

// Slot of hidden channel k in the fused projection input: plain channels come
// first at offset 0 (32-byte aligned for uint4 stores), followed by any zero
// padding (slots 148..151 at full resolution), then the 24 branch channels.
int fused_project_slot(int k, int hidden) {
    const int plain = hidden - 24;
    return k < plain ? k : lanes(hidden) - 24 + (k - plain);
}

std::vector<char> device_weights(const Network& from, const Network& to, const char* in, bool fused,
                                 bool accumulate_fp16) {
    std::vector<char> out(to.bytes, 0);
    const auto raw = [&](size_t target, size_t source, size_t size) {
        std::memcpy(out.data() + target, in + source, size);
    };
    const auto identity = [](int c) { return c; };
    const auto same = [&](Filter target, Filter source, int cin, int cout) {
        copy_dense(out, in, target, source, cin, cout, cin, identity, identity);
    };
    const auto block = [&](const Block& target, const Block& source, int hidden) {
        const int width = lanes(hidden);
        raw(target.norm_weight, source.norm_weight, channels * 4);
        raw(target.norm_bias, source.norm_bias, channels * 4);
        if (!fused) {
            const auto expand_slot = [&](int o) {
                const bool value = o >= hidden;
                const int slot = unfused_hidden_slot(value ? o - hidden : o, hidden);
                return 8 * (slot / 4) + (value ? 4 : 0) + slot % 4;
            };
            copy_dense(out, in, target.expand, source.expand, channels, 2 * hidden, channels, expand_slot, identity);
        } else {
            const int plain = hidden - 24, plain_groups = (plain + 3) / 4, total_ch = expand_width(hidden);
            std::vector<uint16_t> logical(size_t(total_ch) * 9 * channels, 0);
            for (int o = 0; o < 2 * hidden; ++o) {
                const bool value = o >= hidden;
                const int k = value ? o - hidden : o;
                const int g = k < plain ? k / 4 : plain_groups + (k - plain) / 4;
                const int sub = k < plain ? k % 4 : (k - plain) % 4;
                const int ch = 8 * g + (value ? 4 : 0) + sub;
                std::memcpy(logical.data() + size_t(ch) * 9 * channels,
                            in + source.expand.weight + size_t(o) * 9 * channels * 2, 9 * channels * 2);
                std::memcpy(out.data() + target.expand.bias + size_t(ch) * 4,
                            in + source.expand.bias + size_t(o) * 4, 4);
            }
            char* dst = out.data() + target.expand.weight;
            if (hidden == full_hidden) {
                if (accumulate_fp16) {
                    pack_swizzle32_pass(dst, logical.data(), total_ch, 0, 192);
                    pack_swizzle32_pass(dst + size_t(18) * 192 * 32 * 2, logical.data(), total_ch, 192, 160);
                } else {
                    for (int p = 0; p < 4; ++p)
                        pack_swizzle32_pass(dst + size_t(p) * 18 * 88 * 32 * 2, logical.data(), total_ch, p * 88, 88);
                }
            } else {
                if (accumulate_fp16) {
                    pack_swizzle32_pass(dst, logical.data(), total_ch, 0, 192);
                } else {
                    for (int p = 0; p < 2; ++p)
                        pack_swizzle32_pass(dst + size_t(p) * 18 * 96 * 32 * 2, logical.data(), total_ch, p * 96, 96);
                }
            }
        }
        for (auto [t, s, taps] : {std::tuple{target.square, source.square, 9}, {target.row, source.row, 11},
                                  {target.column, source.column, 11}}) {
            raw(t.weight, s.weight, 8 * taps * 4);
            raw(t.bias, s.bias, 8 * 4);
        }
        if (!fused) {
            copy_dense(out, in, target.project, source.project, hidden, channels, width, identity,
                       [&](int i) { return unfused_hidden_slot(i, hidden); });
        } else {
            std::vector<uint16_t> logical(size_t(channels) * 9 * width, 0);
            for (int o = 0; o < channels; ++o) {
                for (int tap = 0; tap < 9; ++tap)
                    for (int i = 0; i < hidden; ++i)
                        std::memcpy(logical.data() + (size_t(o) * 9 + tap) * width + fused_project_slot(i, hidden),
                                    in + source.project.weight + ((size_t(o) * 9 + tap) * hidden + i) * 2, 2);
                std::memcpy(out.data() + target.project.bias + size_t(o) * 4,
                            in + source.project.bias + size_t(o) * 4, 4);
            }
            pack_swizzle16_project(out.data() + target.project.weight, logical.data(), width);
        }
        raw(target.gamma, source.gamma, channels * 4);
    };
    copy_dense(out, in, to.first, from.first, 3, channels, rgb_lanes, identity, identity);
    for (size_t g = 0; g < from.groups.size(); ++g) {
        const auto &source = from.groups[g], &target = to.groups[g];
        for (size_t i = 0; i < source.full.size(); ++i) block(target.full[i], source.full[i], full_hidden);
        same(target.down, source.down, channels, channels / 4);
        for (size_t i = 0; i < source.reduced.size(); ++i)
            block(target.reduced[i], source.reduced[i], reduced_hidden);
        same(target.up, source.up, channels, 4 * channels);
    }
    same(to.last, from.last, channels, 3 * scale * scale);
    return out;
}

constexpr const char* kernel_names[] = {
    "pack_rgb_f32", "pack_rgb_f16", "input_bias_norm", "gate_full_nhwc", "gate_reduced_nhwc",
    "gated_residual_nhwc", "residual_norm_nhwc", "reduce_norm_nhwc", "expand_residual_norm_nhwc",
    "expand_residual_add_nhwc", "output_nhwc_f32", "output_nhwc_f16", "fill_noise",
    "gate_branch_tiled_full", "gate_branch_tiled_reduced",
    "expand_gate_full_f16", "expand_gate_full_f32", "expand_gate_reduced_f16", "expand_gate_reduced_f32",
    "project_full_norm_f16", "project_full_res_f16", "project_full_norm_f32", "project_full_res_f32",
    "project_reduced_norm_f16", "project_reduced_res_f16", "project_reduced_norm_f32", "project_reduced_res_f32",
};
enum Kernel {
    PackF32, PackF16, InputBiasNorm, GateFull, GateReduced, Residual, ResidualNorm, ReduceNorm, ExpandResidualNorm,
    ExpandResidualAdd, OutputF32, OutputF16, FillNoise, GateBranchFull, GateBranchReduced,
    ExpandGateFullF16, ExpandGateFullF32, ExpandGateReducedF16, ExpandGateReducedF32,
    ProjectFullNormF16, ProjectFullResF16, ProjectFullNormF32, ProjectFullResF32,
    ProjectReducedNormF16, ProjectReducedResF16, ProjectReducedNormF32, ProjectReducedResF32, KernelCount
};
static_assert(std::size(kernel_names) == KernelCount);

class Session final : public umzi::KernelSession {
    Network model, device_model{true};
    bool fp16 = false, graph_enabled = false, pending = false;
    struct Shared {
        CUdevice device = 0;
        CUcontext context = nullptr;
        CUmodule module = nullptr;
        cudnnHandle_t cudnn = nullptr;
        CUdeviceptr weights = 0;

        ~Shared() {
            if (!context || cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
            if (cudnn) cudnnDestroy(cudnn);
            if (weights) cuMemFree(weights);
            if (module) cuModuleUnload(module);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
            cuDevicePrimaryCtxRelease(device);
        }
    };
    std::shared_ptr<Shared> shared = std::make_shared<Shared>();

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
    CUgraphExec executable = nullptr;
    std::map<umzi::PlanShape, std::unique_ptr<umzi::CudnnBackendPlan>> plans;
    std::map<std::array<CUdeviceptr, 4>, std::unique_ptr<umzi::CudnnDescriptor>> variants;
    // ConvChoice.math value marking a backend-plan record (index into the
    // deduplicated candidate list) rather than a legacy algorithm.
    static constexpr uint32_t saved_plan_marker = 0x4d42;
    umzi::CudaPlanCache<umzi::moesr_id, plan_schema> plan_cache;
    std::filesystem::path cache_dir;
    umzi::Digest cache_environment{}, plan_key{};
    std::string cache_environment_error;
    size_t activation_bytes = 0, workspace_bytes = 0, scratch_bytes = 0, peak_workspace_bytes = 0;
    std::array<double, 4> setup_ms{};
    std::array<CUevent, 2> timing{};
    int timed_plans = 0;
    uint64_t workspace_limit = default_workspace_mib << 20;
    bool accumulate_fp16 = false, fused_gate = true;
    // Weights repacked without fused_gate after a fallback; shared with clones.
    std::shared_ptr<umzi::DeviceAllocation> unfused_weights;
    // Device memory in use (cuMemGetInfo) at shape setup, after workspace
    // allocation, after tuning, and after warmup/capture.
    std::array<uint64_t, 4> device_used{};

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
        // fused_gate (0 or 1, default 1): compute the full and reduced expansions
        // and plain Mish gates in custom tensor-core kernels (expand_gate_*),
        // writing the 48 depthwise-branch channels to projected for the tiled
        // depthwise kernel (gate_branch_tiled_*).
        bool seen_workspace = false, seen_fused = false;
        for (uint32_t i = 0; i < options.kernel_option_count; ++i) {
            const auto& option = options.kernel_options[i];
            const std::string name = option.name ? option.name : "";
            if (name == "workspace_mib") {
                if (seen_workspace) throw std::runtime_error("Duplicate MoESR option: workspace_mib");
                if (option.value < 0 || uint64_t(option.value) > max_workspace_mib)
                    throw std::runtime_error("MoESR option workspace_mib must be between 0 and 4096");
                workspace_limit = uint64_t(option.value) << 20;
                seen_workspace = true;
            } else if (name == "fused_gate") {
                if (seen_fused) throw std::runtime_error("Duplicate MoESR option: fused_gate");
                if (option.value != 0 && option.value != 1)
                    throw std::runtime_error("MoESR option fused_gate must be 0 or 1");
                fused_gate = option.value != 0;
                seen_fused = true;
            } else {
                throw std::runtime_error("Unknown MoESR option: " + name + " (expected workspace_mib or fused_gate)");
            }
        }
        fp16 = options.precision == UMZI_FP16;
        accumulate_fp16 = fp16;
        device_model = Network(true, fused_gate);
        graph_enabled = (options.flags & UMZI_CUDA_GRAPH) != 0;
        CHECK(cuInit(0));
        CHECK(cuDeviceGet(&device, options.device));
        CHECK(cuDevicePrimaryCtxRetain(&context, device));
        shared->device = device;
        shared->context = context;
        ContextScope scope(context);
        const auto& image =
            umzi::load_kernel_module(device, module, moesr_images::embedded_cubins, moesr_images::embedded_ptx);
        shared->module = module;
        for (int i = 0; i < KernelCount; ++i) CHECK(cuModuleGetFunction(&functions[i], module, kernel_names[i]));
        for (auto [kernel, shared_bytes] : {std::pair{ExpandGateFullF16, fused_shared_full_f16},
                                            std::pair{ExpandGateFullF32, fused_shared_full_f32},
                                            std::pair{ExpandGateReducedF16, fused_shared_reduced_f16},
                                            std::pair{ExpandGateReducedF32, fused_shared_reduced_f32},
                                            std::pair{ProjectFullNormF16, project_shared_bytes},
                                            std::pair{ProjectFullResF16, project_shared_bytes},
                                            std::pair{ProjectFullNormF32, project_shared_bytes},
                                            std::pair{ProjectFullResF32, project_shared_bytes},
                                            std::pair{ProjectReducedNormF16, project_shared_bytes},
                                            std::pair{ProjectReducedResF16, project_shared_bytes},
                                            std::pair{ProjectReducedNormF32, project_shared_bytes},
                                            std::pair{ProjectReducedResF32, project_shared_bytes}})
            CHECK(cuFuncSetAttribute(functions[kernel], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes));
        CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CHECK(cuEventCreate(&complete, CU_EVENT_DISABLE_TIMING));
        for (auto& event : timing) CHECK(cuEventCreate(&event, CU_EVENT_DEFAULT));
        CHECK(cudnnCreate(&cudnn));
        shared->cudnn = cudnn;
        const auto padded_weights = device_weights(model, device_model, data.data(), fused_gate, accumulate_fp16);
        CHECK(cuMemAlloc(&weights, padded_weights.size()));
        shared->weights = weights;
        CHECK(cuMemcpyHtoD(weights, padded_weights.data(), padded_weights.size()));
        initialize_cache(options.cache_dir, package.digest, image);
    }

    void initialize_cache(const char* directory, const umzi::Digest& model_digest, const umzi::KernelImage& image) {
        if (!directory || !*directory) return;
        try {
            cache_dir = std::filesystem::u8path(directory);
            umzi::Sha256 hash;
            umzi::hash_plan_environment(hash, umzi::moesr_id, plan_schema, model_digest, image, device);
            hash.add(static_cast<uint32_t>(fp16));
            hash.add(static_cast<uint32_t>(fused_gate));
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
            hash.add(std::array<int, 3>{padded.n, padded.h, padded.w});
            plan_key = hash.finish();
            plan_cache.open(cache_dir, "moesr", plan_key);
        } catch (const std::exception& error) {
            plan_cache.unavailable(error);
        }
    }

    ~Session() override {
        if (context && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            if (pending) cuEventSynchronize(complete);
            if (stream) cuStreamSynchronize(stream);
            release_shape();
            if (complete) cuEventDestroy(complete);
            for (auto event : timing)
                if (event) cuEventDestroy(event);
            if (stream) cuStreamDestroy(stream);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
        unfused_weights.reset();
        shared.reset();
    }

    std::unique_ptr<umzi::KernelSession> clone_shared() const override {
        auto copy = std::make_unique<Session>();
        copy->copy_base_state(*this);
        copy->shared = shared;
        copy->unfused_weights = unfused_weights;
        copy->model = model;
        copy->device_model = device_model;
        copy->fp16 = fp16;
        copy->graph_enabled = graph_enabled;
        copy->device = device;
        copy->context = context;
        copy->module = module;
        copy->cudnn = cudnn;
        copy->functions = functions;
        copy->weights = weights;
        copy->cache_dir = cache_dir;
        copy->cache_environment = cache_environment;
        copy->cache_environment_error = cache_environment_error;
        copy->workspace_limit = workspace_limit;
        copy->accumulate_fp16 = accumulate_fp16;
        copy->fused_gate = fused_gate;
        ContextScope scope(context);
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
        for (auto* buffer : {&stage_in, &stage_out, &scratch, &initial, &state, &reduced, &normalized, &projected, &gated}) {
            if (*buffer) cuMemFree(*buffer);
            *buffer = 0;
        }
        shape = padded = {};
        activation_bytes = workspace_bytes = scratch_bytes = peak_workspace_bytes = 0;
        setup_ms = {};
        device_used = {};
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
        auto plan = std::make_unique<umzi::CudnnBackendPlan>();
        auto& p = *plan;
        // Mixed accumulates in FP32. FP16 mode accumulates in FP16, which the
        // backend engines run about 1.4x faster on Ampere.
        p.graph = std::make_unique<umzi::CudnnConvGraph>(cudnn, padded.n, h, w, cin, cout, accumulate_fp16);
        auto candidates = p.graph->candidates(cudnn, workspace_limit);
        const umzi::PlanShape key{padded.n, h, w, cin, cout};
        auto time_candidate = [&](float best) {
            return umzi::time_cudnn_backend_plan(cudnn, stream, timing, p, gated, weights, projected, scratch,
                                                 scratch_bytes, best, &peak_workspace_bytes, &device_used[1]);
        };
        if (auto saved = plan_cache.convolutions.find(key); saved != plan_cache.convolutions.end()) {
            const auto& choice = saved->second;
            if (choice.math == saved_plan_marker && choice.algorithm < candidates.size()
                && uint64_t(candidates[choice.algorithm].workspace) == choice.workspace) {
                auto& candidate = candidates[choice.algorithm];
                p.plan = std::move(candidate.plan);
                p.workspace = size_t(candidate.workspace);
                // One run proves the saved plan executes before it is trusted.
                if (time_candidate(0.f) < std::numeric_limits<float>::infinity()) {
                    workspace_bytes = std::max(workspace_bytes, p.workspace);
                    plans.emplace(key, std::move(plan));
                    return;
                }
                p.plan.reset();
            }
        }
        plan_cache.record_tuning();
        // Heuristic order is not reliable for these shapes, so time every engine
        // configuration that fits the workspace on the real buffers. gated holds
        // at least cin features per pixel and projected at least cout.
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
            throw std::runtime_error("No cuDNN convolution for MoESR " + std::to_string(cin) + " to "
                                     + std::to_string(cout) + " channels within the workspace limit");
        p.plan = std::move(candidates[fastest].plan);
        p.workspace = size_t(candidates[fastest].workspace);
        workspace_bytes = std::max(workspace_bytes, p.workspace);
        ++timed_plans;
        plan_cache.convolutions.insert_or_assign(key,
                                                 umzi::ConvChoice{uint32_t(fastest), saved_plan_marker, p.workspace});
        plans.emplace(key, std::move(plan));
    }

    void prepare(Shape requested) {
        wait();
        if (shape == requested) return;
        release_shape();
        try {
            auto start = Clock::now();
            device_used[0] = umzi::device_used_bytes();
            shape = requested;
            padded = shape.padded();
            const size_t pixels = padded.pixels();
            activation(initial, pixels * channels);
            activation(state, pixels * channels);
            activation(reduced, pixels / 4 * channels);
            activation(normalized, pixels * channels);
            // With fused_gate no convolution writes the 344-channel expansion: the
            // widest remaining output is 64 channels per full-resolution pixel
            // (first, projection and up convolutions), so projected shrinks 5.4x.
            activation(projected, pixels * (fused_gate ? channels : expand_width(full_hidden)));
            activation(gated, pixels * lanes(full_hidden));
            device_used[1] = umzi::device_used_bytes();
            setup_ms[0] = umzi::elapsed_ms(start);

            start = Clock::now();
            CHECK(cudnnSetStream(cudnn, stream));
            // Time plans on random activations: zero inputs make some tensor-core
            // engines look faster than they run on real images.
            elementwise(stream, FillNoise, int(pixels * lanes(full_hidden)), gated);
            load_plan_cache();
            const int h = padded.h, w = padded.w;
            make_plan(h, w, rgb_lanes, channels);
            if (!fused_gate) {
                make_plan(h, w, channels, expand_width(full_hidden));
                make_plan(h, w, lanes(full_hidden), channels);
            }
            make_plan(h, w, channels, channels / 4);
            if (!fused_gate) {
                make_plan(h / 2, w / 2, channels, expand_width(reduced_hidden));
                make_plan(h / 2, w / 2, lanes(reduced_hidden), channels);
            }
            make_plan(h / 2, w / 2, channels, 4 * channels);
            make_plan(h, w, channels, 3 * scale * scale);
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
        } catch (const umzi::AllocationError&) {
            // Out of memory is for the tiler to handle, not a reason to drop fusion.
            cuStreamSynchronize(stream);
            release_shape();
            throw;
        } catch (const std::exception& error) {
            cuStreamSynchronize(stream);
            release_shape();
            if (!fused_gate) throw;
            disable_fused_gate(error.what());
            prepare(requested);
        } catch (...) {
            cuStreamSynchronize(stream);
            release_shape();
            throw;
        }
    }

    // Repacks the weights for the unfused layout into a buffer this session (and its clones) own.
    void disable_fused_gate(const char* reason) {
        auto layout = Network(true, false);
        const auto package = reload_package();
        const auto packed = device_weights(model, layout, package->section(3).data(), false, accumulate_fp16);
        unfused_weights = std::make_shared<umzi::DeviceAllocation>(context, packed.data(), packed.size());
        weights = unfused_weights->pointer;
        device_model = std::move(layout);
        fused_gate = false;
        if (cache_environment_error.empty() && !cache_dir.empty()) {
            umzi::Sha256 hash;
            hash.add(cache_environment);
            hash.add(uint32_t(0)); // fused_gate off; plans differ from the fused environment
            cache_environment = hash.finish();
        }
        warn_option(std::string("Disabled MoESR fused_gate: ") + reason);
    }

    template<class... Args>
    void launch(CUstream target, Kernel kernel, int lanes, Args... args) {
        // The gate kernels read the long depthwise windows from L1; small
        // blocks keep more of them resident (1.17 ms against 1.37 ms per
        // batch-2 full-resolution gate on the RTX 3050).
        const int block = kernel == GateFull || kernel == GateReduced ? 64 : threads;
        void* parameters[] = {&args...};
        CHECK(cuLaunchKernel(functions[kernel], (lanes + block - 1) / block, 1, 1, block, 1, 1, 0, target,
                             parameters, nullptr));
    }

    template<class... Args>
    void elementwise(CUstream target, Kernel kernel, int count, Args... args) {
        launch(target, kernel, count, args..., count);
    }

    // Variant packs fix the pointers of one call site. They are built on first use
    // after the shape is prepared (scratch is final by then) and reused, so CUDA
    // Graph capture and direct replays record the same backend calls.
    void conv(CUdeviceptr out, CUdeviceptr in, int h, int w, int cin, int cout, Filter filter) {
        const auto& p = *plans.at(umzi::PlanShape{padded.n, h, w, cin, cout});
        auto& variant = variants[std::array<CUdeviceptr, 4>{out, in, weights + filter.weight, scratch}];
        if (!variant)
            variant = umzi::bind_conv_variant(reinterpret_cast<void*>(in), reinterpret_cast<void*>(weights + filter.weight),
                                              reinterpret_cast<void*>(out), reinterpret_cast<void*>(scratch));
        CHECK(cudnnBackendExecute(cudnn, p.plan->get(), variant->get()));
    }

    // Runs a chain of gated blocks on value. The caller has already populated
    // `normalized` with the first block's LayerNorm (fused into input_bias_norm,
    // reduce_norm_nhwc, or expand_residual_norm_nhwc).
    template<size_t Count>
    void blocks(CUstream target, const std::array<Block, Count>& chain, CUdeviceptr value, int h, int w, int hidden) {
        const int pixels = padded.n * h * w;
        const bool full = hidden == full_hidden;
        for (size_t i = 0; i < Count; ++i) {
            const auto& params = chain[i];
            const Block* next = i + 1 < Count ? &chain[i + 1] : nullptr;
            if (fused_gate) {
                expand_gate(target, normalized, params.expand, h, w, pixels, full);
                gate_branch_tiled(target, params, h, w, pixels, full);
                project_fused(target, value, normalized, gated, params, next, h, w, full);
            } else {
                conv(projected, normalized, h, w, channels, expand_width(hidden), params.expand);
                // gate_groups uses one thread per group of four hidden slots, ordered by filter type.
                launch(target, full ? GateFull : GateReduced, pixels * ((hidden + 3) / 4), gated, projected,
                       weights + params.expand.bias, weights + params.square.weight, weights + params.square.bias,
                       weights + params.row.weight, weights + params.row.bias, weights + params.column.weight,
                       weights + params.column.bias, h, w, pixels);
                conv(projected, gated, h, w, lanes(hidden), channels, params.project);
                if (next) {
                    launch(target, ResidualNorm, pixels * 32, value, normalized, projected,
                           weights + params.project.bias, weights + params.gamma, weights + next->norm_weight,
                           weights + next->norm_bias, pixels);
                } else {
                    elementwise(target, Residual, pixels * channels, value, projected, weights + params.project.bias,
                                weights + params.gamma);
                }
            }
        }
    }

    // Fused projection + bias + Mish + gamma residual (+ optional next LayerNorm)
    // over 16 x 16 spatial tiles using pre-packed swizzle16 K=16 weights.
    void project_fused(CUstream target, CUdeviceptr value, CUdeviceptr norm_out, CUdeviceptr input, const Block& params,
                       const Block* next, int h, int w, bool full) {
        int tiles_x = (w + 15) / 16, tiles_y = (h + 15) / 16;
        CUdeviceptr weight = weights + params.project.weight, bias = weights + params.project.bias;
        CUdeviceptr gamma = weights + params.gamma;
        CUdeviceptr nw = next ? weights + next->norm_weight : 0, nb = next ? weights + next->norm_bias : 0;
        const Kernel kernel = full
            ? (accumulate_fp16 ? (next ? ProjectFullNormF16 : ProjectFullResF16)
                               : (next ? ProjectFullNormF32 : ProjectFullResF32))
            : (accumulate_fp16 ? (next ? ProjectReducedNormF16 : ProjectReducedResF16)
                               : (next ? ProjectReducedNormF32 : ProjectReducedResF32));
        void* parameters[] = {&value, &norm_out, &input, &weight, &bias, &gamma, &nw, &nb, &h, &w, &tiles_x, &tiles_y};
        CHECK(cuLaunchKernel(functions[kernel], unsigned(tiles_x * tiles_y * padded.n), 1, 1, 128, 1, 1,
                             project_shared_bytes, target, parameters, nullptr));
    }

    // Fused expansion + plain Mish gate over 8 x 16 spatial tiles. Writes plain
    // gated channels to `gated` and the 48 depthwise-branch channels in planar
    // form to `projected`.
    void expand_gate(CUstream target, CUdeviceptr input, Filter filter, int h, int w, int pixels, bool full) {
        int tiles_x = (w + fused_cols - 1) / fused_cols, tiles_y = (h + fused_rows - 1) / fused_rows;
        CUdeviceptr out = gated, branch = projected, weight = weights + filter.weight, bias = weights + filter.bias;
        const Kernel kernel = full ? (accumulate_fp16 ? ExpandGateFullF16 : ExpandGateFullF32)
                                   : (accumulate_fp16 ? ExpandGateReducedF16 : ExpandGateReducedF32);
        const int shared = full ? (accumulate_fp16 ? fused_shared_full_f16 : fused_shared_full_f32)
                                : (accumulate_fp16 ? fused_shared_reduced_f16 : fused_shared_reduced_f32);
        void* parameters[] = {&out, &branch, &input, &weight, &bias, &h, &w, &pixels, &tiles_x, &tiles_y};
        CHECK(cuLaunchKernel(functions[kernel], unsigned(tiles_x * tiles_y * padded.n), 1, 1, 256, 1, 1, shared,
                             target, parameters, nullptr));
    }

    // Tiled 16 x 16 depthwise branch + Mish gate reading the planar 48-channel
    // branch output from `projected` and writing branch slots into `gated`.
    void gate_branch_tiled(CUstream target, const Block& params, int h, int w, int pixels, bool full) {
        int tiles_x = (w + 15) / 16, tiles_y = (h + 15) / 16;
        CUdeviceptr out = gated, branch = projected, eb = weights + params.expand.bias;
        CUdeviceptr sq_w = weights + params.square.weight, sq_b = weights + params.square.bias;
        CUdeviceptr rw_w = weights + params.row.weight, rw_b = weights + params.row.bias;
        CUdeviceptr cl_w = weights + params.column.weight, cl_b = weights + params.column.bias;
        void* parameters[] = {&out, &branch, &eb, &sq_w, &sq_b, &rw_w, &rw_b, &cl_w, &cl_b,
                              &h,   &w,      &pixels, &tiles_x, &tiles_y};
        CHECK(cuLaunchKernel(functions[full ? GateBranchFull : GateBranchReduced],
                             unsigned(tiles_x * tiles_y * padded.n), 1, 1, 256, 1, 1, 0, target, parameters, nullptr));
    }

    void forward(CUdeviceptr input, CUdeviceptr output, CUstream target) {
        const int pixels = padded.pixels(), h = padded.h, w = padded.w;
        elementwise(target, fp16 ? PackF16 : PackF32, pixels * rgb_lanes, normalized, input, shape.h, shape.w, h, w);
        conv(projected, normalized, h, w, rgb_lanes, channels, device_model.first);
        const auto& first_block = device_model.groups[0].full[0];
        launch(target, InputBiasNorm, pixels * 32, state, initial, normalized, projected,
               weights + device_model.first.bias, weights + first_block.norm_weight, weights + first_block.norm_bias,
               pixels);
        for (size_t g = 0; g < device_model.groups.size(); ++g) {
            const auto& group = device_model.groups[g];
            blocks(target, group.full, state, h, w, full_hidden);
            conv(projected, state, h, w, channels, channels / 4, group.down);
            launch(target, ReduceNorm, (pixels / 4) * 32, reduced, normalized, projected, weights + group.down.bias,
                   weights + group.reduced[0].norm_weight, weights + group.reduced[0].norm_bias, h / 2, w / 2,
                   pixels / 4);
            blocks(target, group.reduced, reduced, h / 2, w / 2, reduced_hidden);
            conv(projected, reduced, h / 2, w / 2, channels, 4 * channels, group.up);
            if (g + 1 < device_model.groups.size()) {
                const auto& next_block = device_model.groups[g + 1].full[0];
                launch(target, ExpandResidualNorm, pixels * 32, state, normalized, projected, weights + group.up.bias,
                       weights + next_block.norm_weight, weights + next_block.norm_bias, h, w, pixels);
            } else {
                elementwise(target, ExpandResidualAdd, pixels * channels, state, initial, projected,
                            weights + group.up.bias, h, w);
            }
        }
        conv(projected, state, h, w, channels, 3 * scale * scale, device_model.last);
        elementwise(target, fp16 ? OutputF16 : OutputF32, shape.pixels() * 3 * scale * scale,
                    output, projected, weights + device_model.last.bias, scale * shape.h, scale * shape.w, h, w);
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
    umzi::TilingConstraints tiling_constraints() const override { return {2, 2, 0}; }

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

    void ensure_staging() {
        if (stage_in && stage_out) return;
        const auto input_bytes = shape.rgb_bytes(fp16);
        // Allocate only what is missing: a failed second allocation must not leave a half-built pair that passes the early return.
        if (!stage_out) CHECK(cuMemAlloc(&stage_out, input_bytes * scale * scale));
        if (!stage_in) CHECK(cuMemAlloc(&stage_in, input_bytes));
    }

    void run_host(const UmziShape& request, const void* input, void* output) override {
        ContextScope scope(context);
        prepare(checked(request));
        ensure_staging();
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
    const char* kernel_message() const override {
        if (fp16) return fused_gate ? "MoESR fused NHWC FP16" : "MoESR NHWC FP16";
        return fused_gate ? "MoESR fused NHWC mixed precision" : "MoESR NHWC mixed precision";
    }

    UmziSessionInfo session_info() const override {
        UmziSessionInfo info{};
        info.struct_size = sizeof(info);
        info.cache_status = plan_cache.status;
        info.buffer_bytes[0] = device_model.bytes;
        if (shape.n) {
            info.prepared_input = padded.nchw();
            info.buffer_bytes[1] = activation_bytes;
            info.buffer_bytes[2] = workspace_bytes;
            info.buffer_bytes[3] = stage_in ? shape.rgb_bytes(fp16) * (1 + scale * scale) : 0;
            std::copy(device_used.begin(), device_used.end(), info.startup_device_used_bytes);
            std::copy(setup_ms.begin(), setup_ms.end(), info.shape_setup_ms);
        }
        return info;
    }

    uint32_t diagnostic_count() const override { return 4; }
    UmziDiagnostic diagnostic(uint32_t index) const override {
        if (index == 0) return {sizeof(UmziDiagnostic), "timed_plans", timed_plans};
        if (index == 1) return {sizeof(UmziDiagnostic), "plan_workspace_bytes", int64_t(workspace_bytes)};
        if (index == 2) return {sizeof(UmziDiagnostic), "fused_gate", int64_t(fused_gate)};
        if (index == 3) return {sizeof(UmziDiagnostic), "peak_plan_workspace_bytes", int64_t(peak_workspace_bytes)};
        throw std::runtime_error("Invalid MoESR diagnostic index");
    }
};
} // namespace

std::unique_ptr<umzi::KernelSession> umzi::make_moesr(const ModelPackage& package, const UmziSessionOptions& options) {
    auto session = std::make_unique<Session>();
    session->open(package, options);
    return session;
}
