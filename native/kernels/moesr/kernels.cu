#include <cuda_fp16.h>
#include "../../common/resize_rgb.cuh"
#include "../../common/tensor_core.cuh"

extern "C" __global__ void pad_rgb(float* out, const float* in, int h, int w, int ph, int pw, int total) {
    resize_rgb<float, false>(out, in, h, w, ph, pw, total);
}

// One thread normalizes the channels of one NCHW pixel. Consecutive threads
// read consecutive pixels of each channel. Use population variance with
// epsilon inside the square root, as in the checkpoint's LayerNorm.
extern "C" __global__ void layer_norm(float* out, const float* in, const float* weight, const float* bias,
                                       int channels, int pixels, float eps, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const size_t first = size_t(index / pixels) * channels * pixels + index % pixels;
    float mean = 0.f;
    for (int c = 0; c < channels; ++c) mean += in[first + size_t(c) * pixels];
    mean /= float(channels);
    float variance = 0.f;
    for (int c = 0; c < channels; ++c) {
        const float centered = in[first + size_t(c) * pixels] - mean;
        variance = __fadd_rn(variance, __fmul_rn(centered, centered));
    }
    const float denominator = sqrtf(variance / float(channels) + eps);
    for (int c = 0; c < channels; ++c) {
        const size_t at = first + size_t(c) * pixels;
        const float normalized = (in[at] - mean) / denominator;
        out[at] = __fadd_rn(__fmul_rn(normalized, weight[c]), bias[c]);
    }
}

__device__ float mish(float value) {
    return value * tanhf(value > 20.f ? value : log1pf(expf(value)));
}

// projected is [gate | identity | inception input]. The first part of the
// inception input also passes through, followed by square, row and column.
extern "C" __global__ void mish_gate(float* out, const float* projected, const float* square,
                                      const float* row, const float* column, int hidden, int pixels,
                                      int branch, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int plane = index / pixels, n = plane / hidden, c = plane % hidden, pixel = index % pixels;
    const size_t gate = (size_t(n) * 2 * hidden + c) * pixels + pixel;
    const int convolved = c - (hidden - 3 * branch);
    float value;
    if (convolved < 0) {
        value = projected[gate + size_t(hidden) * pixels];
    } else {
        const float* values = convolved < branch ? square : convolved < 2 * branch ? row : column;
        value = values[(size_t(n) * branch + convolved % branch) * pixels + pixel];
    }
    out[index] = mish(projected[gate]) * value;
}

extern "C" __global__ void mish_residual(float* out, const float* projected, const float* shortcut,
                                          const float* gamma, int channels, int pixels, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int c = index / pixels % channels;
    out[index] = __fadd_rn(__fmul_rn(mish(projected[index]), gamma[c]), shortcut[index]);
}

__device__ float leaky(float value) {
    return value > 0.f ? value : value * 0.1f;
}

extern "C" __global__ void unshuffle2_leaky(float* out, const float* in, int h, int w, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int plane = index / (h * w), y = index / w % h, x = index % w;
    const int subpixel = plane % 4;
    const size_t source = (size_t(plane / 4) * (2 * h) + 2 * y + subpixel / 2) * (2 * w) + 2 * x + subpixel % 2;
    out[index] = leaky(in[source]);
}

template<int Scale>
__device__ float shuffled(const float* in, int index, int h, int w, int ph, int pw) {
    const int plane = index / (h * w), y = index / w % h, x = index % w;
    const int subpixel = (y % Scale) * Scale + x % Scale;
    const size_t source = ((size_t(plane) * Scale * Scale + subpixel) * ph + y / Scale) * pw + x / Scale;
    return in[source];
}

extern "C" __global__ void shuffle2_leaky_add(float* out, const float* in, const float* residual,
                                               int ph, int pw, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) out[index] = leaky(shuffled<2>(in, index, 2 * ph, 2 * pw, ph, pw)) + residual[index];
}

extern "C" __global__ void shuffle4_crop(float* out, const float* in, int h, int w, int ph, int pw, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) out[index] = shuffled<4>(in, index, h, w, ph, pw);
}

// Session features are NHWC FP16. Pointwise math and reductions use FP32.
constexpr int features = 64;
constexpr int branch_channels = 8;

__device__ int thread_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ float value_of(float value) { return value; }
__device__ float value_of(half value) { return __half2float(value); }
__device__ void write_value(float* out, int index, float value) { out[index] = value; }
__device__ void write_value(half* out, int index, float value) { out[index] = __float2half_rn(value); }

template<class Rgb>
__device__ void pack_rgb(half* out, const Rgb* in, int h, int w, int ph, int pw, int total) {
    const int index = thread_index();
    if (index >= total) return;
    // Eight channels per pixel: RGB, then zeros so cuDNN needs no padding pass.
    const int pixel = index / 8, c = index % 8, n = pixel / (ph * pw);
    if (c >= 3) {
        out[index] = __float2half_rn(0.f);
        return;
    }
    int y = pixel / pw % ph, x = pixel % pw;
    if (y >= h) y = h - 2;
    if (x >= w) x = w - 2;
    out[index] = __float2half_rn(value_of(in[((n * 3 + c) * h + y) * w + x]));
}

extern "C" __global__ void pack_rgb_f32(half* out, const float* in, int h, int w, int ph, int pw, int total) {
    pack_rgb(out, in, h, w, ph, pw, total);
}

extern "C" __global__ void pack_rgb_f16(half* out, const half* in, int h, int w, int ph, int pw, int total) {
    pack_rgb(out, in, h, w, ph, pw, total);
}

extern "C" __global__ void input_bias(half* state, half* initial, const half* projected, const float* bias, int total) {
    const int index = thread_index();
    if (index >= total) return;
    const half value = __float2half_rn(__half2float(projected[index]) + bias[index % features]);
    initial[index] = state[index] = value;
}

__device__ float warp_sum(float sum) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    return __shfl_sync(0xffffffff, sum, 0);
}

extern "C" __global__ void norm_nhwc(half* out, const half* in, const float* weight, const float* bias, int pixels) {
    const int index = thread_index(), pixel = index / 32, c = 2 * (index % 32);
    if (pixel >= pixels) return;
    const int at = pixel * features + c;
    const float2 pair = __half22float2(*reinterpret_cast<const half2*>(in + at));
    const float mean = warp_sum(pair.x + pair.y) / features;
    const float lo = pair.x - mean, hi = pair.y - mean;
    const float inverse = rsqrtf(warp_sum(lo * lo + hi * hi) / features + 1e-6f);
    *reinterpret_cast<half2*>(out + at) =
        __floats2half2_rn(fmaf(lo * inverse, weight[c], bias[c]), fmaf(hi * inverse, weight[c + 1], bias[c + 1]));
}

// x * tanh(softplus(x)) with one exponential: tanh(log(1 + e)) = n / (n + 2)
// where n = e * (e + 2).
__device__ float fast_mish(float value) {
    if (value > 20.f) return value;
    const float e = __expf(value), n = e * (e + 2.f);
    return value * __fdividef(n, n + 2.f);
}

// Device channel layout of a gated block (see device_weights in runner.cpp).
// Hidden channel k sits in slot k - (Hidden - 24) when it is one of the 24
// depthwise-branch channels (square 0-7, row 8-15, column 16-23), otherwise in
// slot 24 + k. Slots are grouped by four. The expansion writes group g as eight
// contiguous halves, [gate x4 | value x4], so `in` has 8 * Groups channels per
// pixel and a group is one 16-byte load. Padding slots have zero weights and
// bias, so they produce mish(0) * 0 = 0. `out` holds the gated slots in order,
// 4 * Groups channels, then zeros up to OutStride for the projection.
//
// One thread computes one group. Threads are ordered so that no warp mixes
// plain groups with depthwise groups: pixels * (Groups - 6) plain threads,
// then square, row and column regions of pixels * 2 threads each.
//
// BranchOnly launches only the 6 depthwise groups (pixels * 6 threads) and reads
// an input of InStride channels per pixel: the fused expansion path computes
// those groups with a separate 64->48 convolution.
template<int Groups, int OutStride, bool BranchOnly = false, int InStride = 8 * Groups>
__device__ void gate_groups(half* out, const half* in, const float* bias, const float* square,
                            const float* square_bias, const float* row, const float* row_bias,
                            const float* column, const float* column_bias, int h, int w, int pixels) {
    constexpr int branch_groups = 3 * branch_channels / 4, plain = BranchOnly ? 0 : Groups - branch_groups;
    constexpr int stride = InStride;
    static_assert(OutStride % 8 == 0 && OutStride >= 4 * Groups && branch_groups == 6);
    const int index = thread_index();
    int pixel, g, type = -1;
    if (!BranchOnly && index < pixels * plain) {
        pixel = index / (plain > 0 ? plain : 1);
        g = branch_groups + index % (plain > 0 ? plain : 1);
    } else {
        const int rest = index - pixels * plain;
        if (rest >= pixels * branch_groups) return;
        type = rest / (pixels * 2);
        const int within = rest % (pixels * 2);
        pixel = within / 2;
        g = 2 * type + within % 2;
    }
    const half* here = in + size_t(pixel) * stride + 8 * g;
    const uint4 packed = *reinterpret_cast<const uint4*>(here);
    const half2* pairs = reinterpret_cast<const half2*>(&packed);
    const float* b = bias + 8 * g;
    const float2 gate01 = __half22float2(pairs[0]), gate23 = __half22float2(pairs[1]);
    const float gate[4] = {gate01.x, gate01.y, gate23.x, gate23.y};
    float value[4];
    if (type < 0) {
        const float2 v01 = __half22float2(pairs[2]), v23 = __half22float2(pairs[3]);
        value[0] = v01.x + b[4], value[1] = v01.y + b[5], value[2] = v23.x + b[6], value[3] = v23.y + b[7];
    } else {
        const int channel = 4 * (g - 2 * type);
        const int x = pixel % w, y = pixel / w % h;
        const int radius_x = type == 0 ? 1 : type == 1 ? 5 : 0;
        const int radius_y = type == 0 ? 1 : type == 2 ? 5 : 0;
        const int width = 2 * radius_x + 1, taps = width * (2 * radius_y + 1);
        const float* filter = (type == 0 ? square : type == 1 ? row : column) + channel * taps;
        const float* offset = (type == 0 ? square_bias : type == 1 ? row_bias : column_bias) + channel;
        float accum[4] = {0.f, 0.f, 0.f, 0.f};
        for (int ky = -radius_y; ky <= radius_y; ++ky) {
            if (unsigned(y + ky) >= unsigned(h)) continue;
            for (int kx = -radius_x; kx <= radius_x; ++kx) {
                if (unsigned(x + kx) >= unsigned(w)) continue;
                const uint2 raw = *reinterpret_cast<const uint2*>(here + (ky * w + kx) * stride + 4);
                const half2* samples = reinterpret_cast<const half2*>(&raw);
                const float2 s01 = __half22float2(samples[0]), s23 = __half22float2(samples[1]);
                const int tap = (ky + radius_y) * width + kx + radius_x;
                accum[0] = fmaf(filter[tap], s01.x + b[4], accum[0]);
                accum[1] = fmaf(filter[taps + tap], s01.y + b[5], accum[1]);
                accum[2] = fmaf(filter[2 * taps + tap], s23.x + b[6], accum[2]);
                accum[3] = fmaf(filter[3 * taps + tap], s23.y + b[7], accum[3]);
            }
        }
        for (int j = 0; j < 4; ++j) value[j] = accum[j] + offset[j];
    }
    uint2 result;
    half2* gated = reinterpret_cast<half2*>(&result);
    gated[0] = __floats2half2_rn(fast_mish(gate[0] + b[0]) * value[0], fast_mish(gate[1] + b[1]) * value[1]);
    gated[1] = __floats2half2_rn(fast_mish(gate[2] + b[2]) * value[2], fast_mish(gate[3] + b[3]) * value[3]);
    half* target = out + size_t(pixel) * OutStride + 4 * g;
    *reinterpret_cast<uint2*>(target) = result;
    if constexpr (OutStride > 4 * Groups) {
        if (g == Groups - 1)
            for (int c = 4; c < OutStride - 4 * g; c += 4) *reinterpret_cast<uint2*>(target + c) = make_uint2(0, 0);
    }
}

#define GATE_ARGS                                                                                        \
    half *out, const half *in, const float *bias, const float *square, const float *square_bias,         \
        const float *row, const float *row_bias, const float *column, const float *column_bias, int h, int w, \
        int pixels
#define GATE_PASS out, in, bias, square, square_bias, row, row_bias, column, column_bias, h, w, pixels
// 170 hidden channels use 43 groups (172 slots) and a 176-channel projection
// input; 96 hidden channels use 24 groups exactly.
extern "C" __global__ void gate_full_nhwc(GATE_ARGS) { gate_groups<43, 176>(GATE_PASS); }
extern "C" __global__ void gate_reduced_nhwc(GATE_ARGS) { gate_groups<24, 96>(GATE_PASS); }

// ---------------------------------------------------------------------------
// Fused expansion + Mish gate (kernel option fused_gate, default on).
//
// Computes all expansion channels (344 full-res = 296 plain + 48 branch;
// 192 reduced-res = 144 plain + 48 branch) in one tensor-core implicit-GEMM
// kernel over an 8 x 16 spatial tile (10 x 18 halo kept resident in shared
// memory across all N-passes). Plain groups apply bias + Mish in the epilogue
// and write 16-byte coalesced vectors directly to `gated` (slots 0..147 full
// with zero pad 148..151; slots 0..71 reduced). The 48 depthwise-branch
// channels are written in planar-by-type layout to `branch_out` (three 8-ch
// value planes followed by a 24-ch gate plane), consumed by gate_branch_tiled_*.
//
// Weights are pre-packed on the host into swizzle32 K=32 stages per N-pass
// (see pack_expand_weights in runner.cpp) so cp.async loads are contiguous.
// 8 warps (256 threads) per block, <= 47.6 KB shared memory -> 2 active blocks
// per SM on sm_86 with zero local spills.

// Epilogue shared by all four expand_gate_* kernels. Every lane in a warp
// shares the same y, so `if (y >= h) continue;` is warp-uniform; `x < w` is
// checked only on stores so warp shuffles always run with full mask 0xffffffff.
template<bool HalfAccumulate, int MTiles, int N8Tiles, int PlainGroups, int TotalGroups, int OutStride>
__device__ __forceinline__ void write_expand_epilogue(
    Accumulator<HalfAccumulate> (&acc)[MTiles][N8Tiles], half* gated, half* branch_out, const float* bias, int image,
    int h, int w, int pixels, int y_start, int x_start, int base_group, int lane) {
    const int q = lane & 3;
    const bool all_plain = (base_group + N8Tiles <= PlainGroups);
    if (all_plain) {
#pragma unroll
        for (int mt = 0; mt < MTiles; ++mt) {
            const int y = y_start + mt;
            if (y >= h) continue;
            half* row_out = gated + (size_t(image) * h + y) * w * OutStride;
#pragma unroll
            for (int half_row = 0; half_row < 2; ++half_row) {
                const int x = x_start + (lane >> 2) + 8 * half_row;
                auto do_pair = [&](int nt_idx, unsigned& out_a, unsigned& out_b) {
                    const float2 o0 = acc[mt][nt_idx].pair(half_row);
                    const float2 o1 = acc[mt][nt_idx + 1].pair(half_row);
                    const float2 send = (q < 2) ? o1 : o0;
                    const float2 recv = make_float2(__shfl_xor_sync(0xffffffff, send.x, 2),
                                                    __shfl_xor_sync(0xffffffff, send.y, 2));
                    const float2 gv = (q < 2) ? o0 : recv;
                    const float2 vv = (q < 2) ? recv : o1;
                    const int my_g = base_group + nt_idx + (q >> 1);
                    const float* bp = bias + 8 * my_g + 2 * (q & 1);
                    const half2 r = __floats2half2_rn(fast_mish(gv.x + bp[0]) * (vv.x + bp[4]),
                                                      fast_mish(gv.y + bp[1]) * (vv.y + bp[5]));
                    const unsigned my_h2 = *reinterpret_cast<const unsigned*>(&r);
                    const unsigned other_h2 = __shfl_xor_sync(0xffffffff, my_h2, 2);
                    out_a = (q < 2) ? my_h2 : other_h2;
                    out_b = (q < 2) ? other_h2 : my_h2;
                };
#pragma unroll
                for (int blk = 0; blk + 4 <= N8Tiles; blk += 4) {
                    unsigned p0, p1, p2, p3;
                    do_pair(blk, p0, p1);
                    do_pair(blk + 2, p2, p3);
                    const unsigned s0 = __shfl_xor_sync(0xffffffff, q == 0 ? p2 : p0, 1);
                    const unsigned s1 = __shfl_xor_sync(0xffffffff, q == 0 ? p3 : p1, 1);
                    if (q < 2 && x < w) {
                        const int g_out = base_group + blk + 2 * q;
                        half* dst = row_out + size_t(x) * OutStride + 4 * g_out;
                        if ((g_out & 1) == 0) {
                            *reinterpret_cast<uint4*>(dst) =
                                (q == 0) ? make_uint4(p0, s0, p1, s1) : make_uint4(s0, p2, s1, p3);
                        } else {
                            *reinterpret_cast<uint2*>(dst) = (q == 0) ? make_uint2(p0, s0) : make_uint2(s0, p2);
                            *reinterpret_cast<uint2*>(dst + 4) = (q == 0) ? make_uint2(p1, s1) : make_uint2(s1, p3);
                        }
                    }
                }
                if constexpr (N8Tiles % 4 >= 2) {
                    constexpr int blk = (N8Tiles / 4) * 4;
                    unsigned p0, p1;
                    do_pair(blk, p0, p1);
                    const unsigned s0 = __shfl_xor_sync(0xffffffff, p0, 1);
                    const unsigned s1 = __shfl_xor_sync(0xffffffff, p1, 1);
                    if (q == 0 && x < w) {
                        const int g_out = base_group + blk;
                        half* dst = row_out + size_t(x) * OutStride + 4 * g_out;
                        if ((g_out & 1) == 0) {
                            *reinterpret_cast<uint4*>(dst) = make_uint4(p0, s0, p1, s1);
                        } else {
                            *reinterpret_cast<uint2*>(dst) = make_uint2(p0, s0);
                            *reinterpret_cast<uint2*>(dst + 4) = make_uint2(p1, s1);
                        }
                    }
                }
                if constexpr (N8Tiles % 2 == 1) {
                    constexpr int nt = N8Tiles - 1;
                    const float2 own = acc[mt][nt].pair(half_row);
                    const float2 partner = make_float2(__shfl_xor_sync(0xffffffff, own.x, 2),
                                                       __shfl_xor_sync(0xffffffff, own.y, 2));
                    const int g = base_group + nt;
                    if (q < 2 && x < w) {
                        const float* bp = bias + 8 * g + 2 * q;
                        const half2 r = __floats2half2_rn(fast_mish(own.x + bp[0]) * (partner.x + bp[4]),
                                                          fast_mish(own.y + bp[1]) * (partner.y + bp[5]));
                        *reinterpret_cast<half2*>(row_out + size_t(x) * OutStride + 4 * g + 2 * q) = r;
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int mt = 0; mt < MTiles; ++mt) {
            const int y = y_start + mt;
            if (y >= h) continue;
            const size_t row_pix = (size_t(image) * h + y) * w;
            half* row_gated = gated + row_pix * OutStride;
#pragma unroll
            for (int half_row = 0; half_row < 2; ++half_row) {
                const int x = x_start + (lane >> 2) + 8 * half_row;
                const size_t pix = row_pix + x;
#pragma unroll
                for (int nt = 0; nt < N8Tiles; ++nt) {
                    const int g = base_group + nt;
                    if (g < PlainGroups) {
                        const float2 own = acc[mt][nt].pair(half_row);
                        const float2 partner = make_float2(__shfl_xor_sync(0xffffffff, own.x, 2),
                                                           __shfl_xor_sync(0xffffffff, own.y, 2));
                        if (q < 2 && x < w) {
                            const float* bp = bias + 8 * g + 2 * q;
                            const half2 r = __floats2half2_rn(fast_mish(own.x + bp[0]) * (partner.x + bp[4]),
                                                              fast_mish(own.y + bp[1]) * (partner.y + bp[5]));
                            *reinterpret_cast<half2*>(row_gated + size_t(x) * OutStride + 4 * g + 2 * q) = r;
                            if constexpr (OutStride > 4 * TotalGroups) {
                                if (g == PlainGroups - 1 && q == 0)
                                    *reinterpret_cast<uint2*>(row_gated + size_t(x) * OutStride + 4 * PlainGroups) =
                                        make_uint2(0, 0);
                            }
                        }
                    } else if (g < TotalGroups) {
                        const unsigned raw = acc[mt][nt].packed_half2(half_row);
                        if (x < w) {
                            const int bg = g - PlainGroups, type = bg >> 1;
                            if (q < 2) {
                                *reinterpret_cast<unsigned*>(branch_out + size_t(pixels) * 24 + pix * 24 + 4 * bg
                                                             + 2 * q) = raw;
                            } else {
                                *reinterpret_cast<unsigned*>(branch_out + size_t(type) * pixels * 8 + pix * 8
                                                             + 4 * (bg & 1) + 2 * (q - 2)) = raw;
                            }
                        }
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void load_halo_8x16(half* a_tile, const half* in, int image, int h, int w, int y0, int x0,
                                               int thread) {
    constexpr int halo_cols = 18, halo_count = 180;
    for (int index = thread; index < halo_count * 8; index += 256) {
        const int p = index >> 3, chunk = index & 7;
        const int py = y0 - 1 + p / halo_cols, px = x0 - 1 + p % halo_cols;
        const bool valid = unsigned(py) < unsigned(h) && unsigned(px) < unsigned(w);
        const half* source = valid ? in + ((size_t(image) * h + py) * w + px) * 64 + chunk * 8 : in;
        copy_async(a_tile + swizzle64(p, chunk), source, valid);
    }
}

extern __shared__ __align__(16) half expand_shared[];

// Full-resolution FP16 expansion + gate: 2 passes (192 + 160 = 352 ch), 4x2
// warps, K=32 double-buffered ring (47,616 B shared memory, 2 blocks/SM).
#define EXPAND_FUSED_ARGS                                                                                         \
    half *gated, half *branch_out, const half *in, const half *weight, const float *bias, int h, int w, int pixels, \
        int tiles_x, int tiles_y
extern "C" __global__ void __launch_bounds__(256, 2) expand_gate_full_f16(EXPAND_FUSED_ARGS) {
    constexpr int halo_cols = 18, halo_count = 180, b0_halves = 192 * 32, b1_halves = 160 * 32;
    half* const a_tile = expand_shared;
    half* const b_ring = expand_shared + halo_count * 64;
    const int thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;
    const int warp_m = warp & 3, warp_n = warp >> 2;
    const int tile = blockIdx.x;
    const int tx = tile % tiles_x, ty = tile / tiles_x % tiles_y, image = tile / (tiles_x * tiles_y);
    const int y0 = ty * 8, x0 = tx * 16;

    load_halo_8x16(a_tile, in, image, h, w, y0, x0, thread);
    const half* const w0 = weight;
    const half* const w1 = weight + size_t(18) * b0_halves;
    auto load_b0 = [&](int s, half* dst) {
        const half* base = w0 + size_t(s) * b0_halves;
#pragma unroll
        for (int index = thread; index < b0_halves / 8; index += 256)
            copy_async_always(dst + index * 8, base + index * 8);
    };
    auto load_b1 = [&](int s, half* dst) {
        const half* base = w1 + size_t(s) * b1_halves;
#pragma unroll
        for (int index = thread; index < b1_halves / 8; index += 256)
            copy_async_always(dst + index * 8, base + index * 8);
    };
    load_b0(0, b_ring);
    asm volatile("cp.async.commit_group;\n");

    // Pass 0: 192 plain channels (groups 0..23).
    {
        Accumulator<true> acc[2][12];
        int step = 0;
        for (int tap = 0; tap < 9; ++tap) {
            const int dy = tap / 3, dx = tap % 3;
            for (int kh = 0; kh < 2; ++kh) {
                asm volatile("cp.async.wait_group 0;\n");
                __syncthreads();
                const int cb = step & 1;
                if (step + 1 < 18) load_b0(step + 1, b_ring + (cb ^ 1) * b0_halves);
                else load_b1(0, b_ring + (cb ^ 1) * b0_halves);
                asm volatile("cp.async.commit_group;\n");
                const half* b_tile = b_ring + cb * b0_halves;
                ++step;

                unsigned a_frag[2][2][4], b_frag[2][6][4];
                auto load_k16 = [&](int kk, int buf) {
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const int p = (warp_m * 2 + mt + dy) * halo_cols + (lane & 15) + dx;
                        load_matrix_x4(a_frag[buf][mt], a_tile + swizzle64(p, kh * 4 + 2 * kk + (lane >> 4)));
                    }
#pragma unroll
                    for (int j = 0; j < 6; ++j) {
                        const int row = 96 * warp_n + 16 * j + (lane >> 4) * 8 + (lane & 7);
                        load_matrix_x4(b_frag[buf][j], b_tile + swizzle32(row, 2 * kk + ((lane >> 3) & 1)));
                    }
                };
                load_k16(0, 0);
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    if (kk == 0) load_k16(1, 1);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                        for (int j = 0; j < 6; ++j) {
                            acc[mt][2 * j].mma(a_frag[kk][mt], b_frag[kk][j][0], b_frag[kk][j][1]);
                            acc[mt][2 * j + 1].mma(a_frag[kk][mt], b_frag[kk][j][2], b_frag[kk][j][3]);
                        }
                }
            }
        }
        write_expand_epilogue<true, 2, 12, 37, 43, 176>(acc, gated, branch_out, bias, image, h, w, pixels,
                                                        y0 + warp_m * 2, x0, 12 * warp_n, lane);
    }

    // Pass 1: 160 channels (13 plain groups 24..36 + 6 branch groups 37..42 + 1 pad).
    {
        Accumulator<true> acc[2][10];
        int step = 0;
        for (int tap = 0; tap < 9; ++tap) {
            const int dy = tap / 3, dx = tap % 3;
            for (int kh = 0; kh < 2; ++kh) {
                asm volatile("cp.async.wait_group 0;\n");
                __syncthreads();
                const int cb = step & 1;
                if (step + 1 < 18) {
                    load_b1(step + 1, b_ring + (cb ^ 1) * b0_halves);
                    asm volatile("cp.async.commit_group;\n");
                }
                const half* b_tile = b_ring + cb * b0_halves;
                ++step;

                unsigned a_frag[2][2][4], b_frag[2][5][4];
                auto load_k16 = [&](int kk, int buf) {
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const int p = (warp_m * 2 + mt + dy) * halo_cols + (lane & 15) + dx;
                        load_matrix_x4(a_frag[buf][mt], a_tile + swizzle64(p, kh * 4 + 2 * kk + (lane >> 4)));
                    }
#pragma unroll
                    for (int j = 0; j < 5; ++j) {
                        const int row = 80 * warp_n + 16 * j + (lane >> 4) * 8 + (lane & 7);
                        load_matrix_x4(b_frag[buf][j], b_tile + swizzle32(row, 2 * kk + ((lane >> 3) & 1)));
                    }
                };
                load_k16(0, 0);
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    if (kk == 0) load_k16(1, 1);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                        for (int j = 0; j < 5; ++j) {
                            acc[mt][2 * j].mma(a_frag[kk][mt], b_frag[kk][j][0], b_frag[kk][j][1]);
                            acc[mt][2 * j + 1].mma(a_frag[kk][mt], b_frag[kk][j][2], b_frag[kk][j][3]);
                        }
                }
            }
        }
        write_expand_epilogue<true, 2, 10, 37, 43, 176>(acc, gated, branch_out, bias, image, h, w, pixels,
                                                        y0 + warp_m * 2, x0, 24 + 10 * warp_n, lane);
    }
}

// Uniform-pass K=32 expansion + gate template used for full FP32 (4 x 88, 2-stage),
// reduced FP16 (1 x 192, 2-stage), and reduced FP32 (2 x 96, 3-stage).
template<bool HalfAccumulate, int NTile, int WarpsM, int WarpsN, int Stages, int NPasses, int PlainGroups,
         int TotalGroups, int OutStride>
__device__ void expand_gate_uniform(EXPAND_FUSED_ARGS) {
    constexpr int halo_cols = 18, halo_count = 180;
    constexpr int warp_rows = 8 / WarpsM, n_per_warp = NTile / WarpsN, n8_tiles = n_per_warp / 8;
    constexpr int nb4 = n8_tiles / 2;
    constexpr bool has_tail = (n8_tiles & 1) != 0;
    constexpr int b_halves = NTile * 32, all_steps = NPasses * 18;
    static_assert(WarpsM * WarpsN == 8 && (Stages == 2 || Stages == 3));

    half* const a_tile = expand_shared;
    half* const b_ring = expand_shared + halo_count * 64;
    const int thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;
    const int warp_m = warp % WarpsM, warp_n = warp / WarpsM;
    const int tile = blockIdx.x;
    const int tx = tile % tiles_x, ty = tile / tiles_x % tiles_y, image = tile / (tiles_x * tiles_y);
    const int y0 = ty * 8, x0 = tx * 16;

    load_halo_8x16(a_tile, in, image, h, w, y0, x0, thread);
    auto load_b = [&](int step_g, half* dst) {
        const half* base = weight + size_t(step_g) * b_halves;
#pragma unroll
        for (int index = thread; index < b_halves / 8; index += 256)
            copy_async_always(dst + index * 8, base + index * 8);
    };

    if constexpr (Stages == 2) {
        load_b(0, b_ring);
        asm volatile("cp.async.commit_group;\n");
    } else {
        asm volatile("cp.async.commit_group;\n");
        for (int s = 0; s < Stages - 1; ++s) {
            load_b(s, b_ring + s * b_halves);
            asm volatile("cp.async.commit_group;\n");
        }
        asm volatile("cp.async.wait_group 1;\n");
        __syncthreads();
    }

    int step_g = 0, ring_read = 0, ring_write = Stages - 1;
    for (int pass = 0; pass < NPasses; ++pass) {
        Accumulator<HalfAccumulate> acc[warp_rows][n8_tiles];
        for (int tap = 0; tap < 9; ++tap) {
            const int dy = tap / 3, dx = tap % 3;
            for (int kh = 0; kh < 2; ++kh) {
                const half* b_tile;
                int fut = 0;
                if constexpr (Stages == 2) {
                    asm volatile("cp.async.wait_group 0;\n");
                    __syncthreads();
                    const int cb = step_g & 1;
                    if (step_g + 1 < all_steps) {
                        load_b(step_g + 1, b_ring + (cb ^ 1) * b_halves);
                        asm volatile("cp.async.commit_group;\n");
                    }
                    b_tile = b_ring + cb * b_halves;
                    ++step_g;
                } else {
                    fut = step_g + (Stages - 1);
                    if (fut < all_steps) {
                        load_b(fut, b_ring + ring_write * b_halves);
                        ring_write = (ring_write + 1 == Stages) ? 0 : ring_write + 1;
                        asm volatile("cp.async.commit_group;\n");
                    }
                    b_tile = b_ring + ring_read * b_halves;
                    ring_read = (ring_read + 1 == Stages) ? 0 : ring_read + 1;
                    ++step_g;
                }

                unsigned a_frag[2][warp_rows][4], b_frag[2][nb4][4], b_tail[2][2];
                auto load_k16 = [&](int kk, int buf) {
#pragma unroll
                    for (int mt = 0; mt < warp_rows; ++mt) {
                        const int p = (warp_m * warp_rows + mt + dy) * halo_cols + (lane & 15) + dx;
                        load_matrix_x4(a_frag[buf][mt], a_tile + swizzle64(p, kh * 4 + 2 * kk + (lane >> 4)));
                    }
#pragma unroll
                    for (int j = 0; j < nb4; ++j) {
                        const int row = n_per_warp * warp_n + 16 * j + (lane >> 4) * 8 + (lane & 7);
                        load_matrix_x4(b_frag[buf][j], b_tile + swizzle32(row, 2 * kk + ((lane >> 3) & 1)));
                    }
                    if constexpr (has_tail) {
                        const int row = n_per_warp * warp_n + 16 * nb4 + (lane & 7);
                        load_matrix_x2(b_tail[buf], b_tile + swizzle32(row, 2 * kk + ((lane >> 3) & 1)));
                    }
                };
                load_k16(0, 0);
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    if (kk == 0) load_k16(1, 1);
#pragma unroll
                    for (int mt = 0; mt < warp_rows; ++mt) {
#pragma unroll
                        for (int j = 0; j < nb4; ++j) {
                            acc[mt][2 * j].mma(a_frag[kk][mt], b_frag[kk][j][0], b_frag[kk][j][1]);
                            acc[mt][2 * j + 1].mma(a_frag[kk][mt], b_frag[kk][j][2], b_frag[kk][j][3]);
                        }
                        if constexpr (has_tail)
                            acc[mt][n8_tiles - 1].mma(a_frag[kk][mt], b_tail[kk][0], b_tail[kk][1]);
                    }
                }

                if constexpr (Stages == 3) {
                    if (step_g < all_steps) {
                        if (fut < all_steps) asm volatile("cp.async.wait_group 1;\n");
                        else asm volatile("cp.async.wait_group 0;\n");
                        __syncthreads();
                    }
                }
            }
        }
        const int base_group = (pass * NTile + n_per_warp * warp_n) / 8;
        write_expand_epilogue<HalfAccumulate, warp_rows, n8_tiles, PlainGroups, TotalGroups, OutStride>(
            acc, gated, branch_out, bias, image, h, w, pixels, y0 + warp_m * warp_rows, x0, base_group, lane);
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) expand_gate_full_f32(EXPAND_FUSED_ARGS) {
    expand_gate_uniform<false, 88, 8, 1, 3, 4, 37, 43, 176>(gated, branch_out, in, weight, bias, h, w, pixels, tiles_x,
                                                            tiles_y);
}
extern "C" __global__ void __launch_bounds__(256, 2) expand_gate_reduced_f16(EXPAND_FUSED_ARGS) {
    expand_gate_uniform<true, 192, 4, 2, 2, 1, 18, 24, 96>(gated, branch_out, in, weight, bias, h, w, pixels, tiles_x,
                                                           tiles_y);
}
extern "C" __global__ void __launch_bounds__(256, 2) expand_gate_reduced_f32(EXPAND_FUSED_ARGS) {
    expand_gate_uniform<false, 96, 8, 1, 3, 2, 18, 24, 96>(gated, branch_out, in, weight, bias, h, w, pixels, tiles_x,
                                                           tiles_y);
}

// Unified single-block 16x16 tiled depthwise branch + Mish gate for the 24
// branch channels written in planar-by-type layout by expand_gate_*. Reuses
// one 13 KB float shared buffer across square (3x3), row (1x11), and column
// (11x1), converting half->float and adding expand_bias once per loaded pixel
// (zeros out of bounds -> branch-free inner loops), then writes all 24 gated
// channels (48 B contiguous) to `gated` in one shot.
template<int OutStride, int PlainGroups, int BranchSlotOffset>
__device__ void gate_branch_tiled(half* gated, const half* branch_out, const float* expand_bias, const float* square,
                                  const float* square_bias, const float* row, const float* row_bias,
                                  const float* column, const float* column_bias, int h, int w, int pixels, int tiles_x,
                                  int tiles_y) {
    __shared__ __align__(16) float tile_val[26 * 16 * 8];
    const int tid = threadIdx.x, lx = tid & 15, ly = tid >> 4;
    const int tile = blockIdx.x;
    const int tx = tile % tiles_x, ty = tile / tiles_x % tiles_y, image = tile / (tiles_x * tiles_y);
    const int x0 = tx * 16, y0 = ty * 16;
    const int y = y0 + ly, x = x0 + lx;
    const bool valid_out = (y < h && x < w);
    const size_t pix = (size_t(image) * h + (valid_out ? y : 0)) * w + (valid_out ? x : 0);

    auto load_plane_float = [&](int type, int rx, int ry) {
        const int sw = 16 + 2 * rx, sh = 16 + 2 * ry, total = sw * sh;
        const half* plane = branch_out + size_t(type) * pixels * 8;
        const float* eb = expand_bias + 8 * (PlainGroups + 2 * type);
        const float vb0 = eb[4], vb1 = eb[5], vb2 = eb[6], vb3 = eb[7];
        const float vb4 = eb[12], vb5 = eb[13], vb6 = eb[14], vb7 = eb[15];
        for (int idx = tid; idx < total; idx += 256) {
            const int sy = idx / sw, sx = idx % sw;
            const int py = y0 - ry + sy, px = x0 - rx + sx;
            float* dst = tile_val + idx * 8;
            if (unsigned(py) < unsigned(h) && unsigned(px) < unsigned(w)) {
                const size_t sp = (size_t(image) * h + py) * w + px;
                const uint4 raw = *reinterpret_cast<const uint4*>(plane + sp * 8);
                const half2* p = reinterpret_cast<const half2*>(&raw);
                const float2 f0 = __half22float2(p[0]), f1 = __half22float2(p[1]);
                const float2 f2 = __half22float2(p[2]), f3 = __half22float2(p[3]);
                *reinterpret_cast<float4*>(dst) = make_float4(f0.x + vb0, f0.y + vb1, f1.x + vb2, f1.y + vb3);
                *reinterpret_cast<float4*>(dst + 4) = make_float4(f2.x + vb4, f2.y + vb5, f3.x + vb6, f3.y + vb7);
            } else {
                *reinterpret_cast<float4*>(dst) = make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<float4*>(dst + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __syncthreads();
    };

    auto finish_type = [&](int type, const float (&accum)[8], const float* dw_bias) -> uint4 {
        const float* eb = expand_bias + 8 * (PlainGroups + 2 * type);
        const float gb[8] = {eb[0], eb[1], eb[2], eb[3], eb[8], eb[9], eb[10], eb[11]};
        const uint4 g_raw = *reinterpret_cast<const uint4*>(branch_out + size_t(pixels) * 24 + pix * 24 + 8 * type);
        const half2* gp = reinterpret_cast<const half2*>(&g_raw);
        uint4 res;
        half2* out_p = reinterpret_cast<half2*>(&res);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 gf = __half22float2(gp[pair]);
            out_p[pair] = __floats2half2_rn(
                fast_mish(gf.x + gb[2 * pair]) * (accum[2 * pair] + dw_bias[2 * pair]),
                fast_mish(gf.y + gb[2 * pair + 1]) * (accum[2 * pair + 1] + dw_bias[2 * pair + 1]));
        }
        return res;
    };

    uint4 r0, r1, r2;

    load_plane_float(0, 1, 1);
    if (valid_out) {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                const float* sp = tile_val + ((ly + ky) * 18 + (lx + kx)) * 8;
                const float4 v0 = *reinterpret_cast<const float4*>(sp);
                const float4 v1 = *reinterpret_cast<const float4*>(sp + 4);
                const int tap = ky * 3 + kx;
                acc[0] = fmaf(square[0 * 9 + tap], v0.x, acc[0]);
                acc[1] = fmaf(square[1 * 9 + tap], v0.y, acc[1]);
                acc[2] = fmaf(square[2 * 9 + tap], v0.z, acc[2]);
                acc[3] = fmaf(square[3 * 9 + tap], v0.w, acc[3]);
                acc[4] = fmaf(square[4 * 9 + tap], v1.x, acc[4]);
                acc[5] = fmaf(square[5 * 9 + tap], v1.y, acc[5]);
                acc[6] = fmaf(square[6 * 9 + tap], v1.z, acc[6]);
                acc[7] = fmaf(square[7 * 9 + tap], v1.w, acc[7]);
            }
        }
        r0 = finish_type(0, acc, square_bias);
    }
    __syncthreads();

    load_plane_float(1, 5, 0);
    if (valid_out) {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kx = 0; kx < 11; ++kx) {
            const float* sp = tile_val + (ly * 26 + (lx + kx)) * 8;
            const float4 v0 = *reinterpret_cast<const float4*>(sp);
            const float4 v1 = *reinterpret_cast<const float4*>(sp + 4);
            acc[0] = fmaf(row[0 * 11 + kx], v0.x, acc[0]);
            acc[1] = fmaf(row[1 * 11 + kx], v0.y, acc[1]);
            acc[2] = fmaf(row[2 * 11 + kx], v0.z, acc[2]);
            acc[3] = fmaf(row[3 * 11 + kx], v0.w, acc[3]);
            acc[4] = fmaf(row[4 * 11 + kx], v1.x, acc[4]);
            acc[5] = fmaf(row[5 * 11 + kx], v1.y, acc[5]);
            acc[6] = fmaf(row[6 * 11 + kx], v1.z, acc[6]);
            acc[7] = fmaf(row[7 * 11 + kx], v1.w, acc[7]);
        }
        r1 = finish_type(1, acc, row_bias);
    }
    __syncthreads();

    load_plane_float(2, 0, 5);
    if (valid_out) {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ky = 0; ky < 11; ++ky) {
            const float* sp = tile_val + ((ly + ky) * 16 + lx) * 8;
            const float4 v0 = *reinterpret_cast<const float4*>(sp);
            const float4 v1 = *reinterpret_cast<const float4*>(sp + 4);
            acc[0] = fmaf(column[0 * 11 + ky], v0.x, acc[0]);
            acc[1] = fmaf(column[1 * 11 + ky], v0.y, acc[1]);
            acc[2] = fmaf(column[2 * 11 + ky], v0.z, acc[2]);
            acc[3] = fmaf(column[3 * 11 + ky], v0.w, acc[3]);
            acc[4] = fmaf(column[4 * 11 + ky], v1.x, acc[4]);
            acc[5] = fmaf(column[5 * 11 + ky], v1.y, acc[5]);
            acc[6] = fmaf(column[6 * 11 + ky], v1.z, acc[6]);
            acc[7] = fmaf(column[7 * 11 + ky], v1.w, acc[7]);
        }
        r2 = finish_type(2, acc, column_bias);

        half* dst = gated + pix * OutStride + BranchSlotOffset;
        *reinterpret_cast<uint4*>(dst) = r0;
        *reinterpret_cast<uint4*>(dst + 8) = r1;
        *reinterpret_cast<uint4*>(dst + 16) = r2;
    }
}

#define BRANCH_TILED_ARGS                                                                                   \
    half *gated, const half *branch_out, const float *expand_bias, const float *square,                     \
        const float *square_bias, const float *row, const float *row_bias, const float *column,             \
        const float *column_bias, int h, int w, int pixels, int tiles_x, int tiles_y
#define BRANCH_TILED_PASS \
    gated, branch_out, expand_bias, square, square_bias, row, row_bias, column, column_bias, h, w, pixels, tiles_x, tiles_y
extern "C" __global__ void gate_branch_tiled_full(BRANCH_TILED_ARGS) {
    gate_branch_tiled<176, 37, 152>(BRANCH_TILED_PASS);
}
extern "C" __global__ void gate_branch_tiled_reduced(BRANCH_TILED_ARGS) {
    gate_branch_tiled<96, 18, 72>(BRANCH_TILED_PASS);
}

// ---------------------------------------------------------------------------
// Fused projection + bias + Mish + gamma residual (+ optional next LayerNorm).
//
// Replaces the 176->64 (full) and 96->64 (reduced) cuDNN convolutions as well
// as residual_norm_nhwc / gated_residual_nhwc. Each 4-warp (128-thread) block
// computes a 16x16 pixel tile (18x18 halo = 324 pixels) across all 64 output
// channels. Outer loop streams K=16 slices of the input halo and 3-tap rows of
// pre-packed swizzle16 weights (33,024 B shared memory -> 3 blocks/SM in FP16,
// 2 blocks/SM in FP32, zero local spills). Because each warp owns all 64 output
// channels of its rows, the 4 lanes holding a pixel compute LayerNorm with two
// quad-wide warp shuffles and no extra global round-trip.

template<bool HalfAccumulate, bool DoNorm, int Cin>
__device__ void project_k16_3tap(half* state, half* normalized, const half* in, const half* weight, const float* bias,
                                 const float* gamma, const float* norm_w, const float* norm_b, int h, int w,
                                 int tiles_x, int tiles_y) {
    constexpr int rows = 16, threads = 128, warp_rows = 4;
    constexpr int halo_cols = 18, halo_count = 18 * 18;
    constexpr int slices = Cin / 16, a_halves = halo_count * 16;
    constexpr int b_tap_halves = 64 * 16, b_3tap_halves = 3 * b_tap_halves;
    static_assert(Cin % 16 == 0);

    half* const a_bufs = expand_shared;
    half* const b_bufs = expand_shared + 2 * a_halves;
    const int thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;
    const int tile = blockIdx.x;
    const int tx = tile % tiles_x, ty = tile / tiles_x % tiles_y, image = tile / (tiles_x * tiles_y);
    const int y0 = ty * rows, x0 = tx * 16;

    auto load_a = [&](int ks, half* dst) {
        const int k_base = ks * 16;
        for (int index = thread; index < halo_count * 2; index += threads) {
            const int p = index >> 1, chunk = index & 1;
            const int py = y0 - 1 + p / halo_cols, px = x0 - 1 + p % halo_cols;
            const bool valid = unsigned(py) < unsigned(h) && unsigned(px) < unsigned(w);
            const half* src = valid ? in + ((size_t(image) * h + py) * w + px) * Cin + k_base + chunk * 8 : in;
            copy_async(dst + swizzle16(p, chunk), src, valid);
        }
    };
    auto load_b3 = [&](int step_g, half* dst) {
        const half* base = weight + size_t(step_g) * b_3tap_halves;
#pragma unroll
        for (int index = thread; index < b_3tap_halves / 8; index += threads)
            copy_async_always(dst + index * 8, base + index * 8);
    };

    Accumulator<HalfAccumulate> acc[warp_rows][8];
    load_a(0, a_bufs);
    load_b3(0, b_bufs);
    asm volatile("cp.async.commit_group;\n");

    int step_g = 0;
    constexpr int all_steps = slices * 3;
    for (int ks = 0; ks < slices; ++ks) {
        const half* a_tile = a_bufs + (ks & 1) * a_halves;
        for (int dy = 0; dy < 3; ++dy) {
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            const int cb = step_g & 1;
            if (step_g + 1 < all_steps) load_b3(step_g + 1, b_bufs + (cb ^ 1) * b_3tap_halves);
            if (dy == 0 && ks + 1 < slices) load_a(ks + 1, a_bufs + ((ks + 1) & 1) * a_halves);
            if (step_g + 1 < all_steps) asm volatile("cp.async.commit_group;\n");
            const half* b3 = b_bufs + cb * b_3tap_halves;
            ++step_g;

            unsigned a_frag[2][warp_rows][4], b_frag[2][4][4];
            auto load_dx = [&](int dx, int buf) {
                const half* b_tile = b3 + dx * b_tap_halves;
#pragma unroll
                for (int mt = 0; mt < warp_rows; ++mt) {
                    const int p = (warp * warp_rows + mt + dy) * halo_cols + (lane & 15) + dx;
                    load_matrix_x4(a_frag[buf][mt], a_tile + swizzle16(p, lane >> 4));
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int row = 16 * j + (lane >> 4) * 8 + (lane & 7);
                    load_matrix_x4(b_frag[buf][j], b_tile + swizzle16(row, (lane >> 3) & 1));
                }
            };
            load_dx(0, 0);
#pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                if (dx + 1 < 3) load_dx(dx + 1, (dx + 1) & 1);
                const int cur = dx & 1;
#pragma unroll
                for (int mt = 0; mt < warp_rows; ++mt)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[mt][2 * j].mma(a_frag[cur][mt], b_frag[cur][j][0], b_frag[cur][j][1]);
                        acc[mt][2 * j + 1].mma(a_frag[cur][mt], b_frag[cur][j][2], b_frag[cur][j][3]);
                    }
            }
        }
    }

    const int q = lane & 3;
#pragma unroll
    for (int mt = 0; mt < warp_rows; ++mt) {
        const int y = y0 + warp * warp_rows + mt;
        if (y >= h) continue;
        const size_t row_off = (size_t(image) * h + y) * w * 64;
#pragma unroll
        for (int half_row = 0; half_row < 2; ++half_row) {
            const int x = x0 + (lane >> 2) + 8 * half_row;
            const bool valid = x < w;
            half* st_pix = state + row_off + size_t(valid ? x : 0) * 64;
            float vals[16];
            float sum = 0.f;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int c = 8 * nt + 2 * q;
                const float2 acc_v = acc[mt][nt].pair(half_row);
                const half2 s_raw =
                    valid ? *reinterpret_cast<const half2*>(st_pix + c) : __floats2half2_rn(0.f, 0.f);
                const float2 s_v = __half22float2(s_raw);
                const half2 upd = __floats2half2_rn(fmaf(fast_mish(acc_v.x + bias[c]), gamma[c], s_v.x),
                                                    fmaf(fast_mish(acc_v.y + bias[c + 1]), gamma[c + 1], s_v.y));
                if (valid) *reinterpret_cast<half2*>(st_pix + c) = upd;
                if constexpr (DoNorm) {
                    const float2 uf = __half22float2(upd);
                    vals[2 * nt] = uf.x;
                    vals[2 * nt + 1] = uf.y;
                    sum += uf.x + uf.y;
                }
            }
            if constexpr (DoNorm) {
                const float mean = quad_sum(sum) * (1.f / 64.f);
                float var = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const float d = vals[i] - mean;
                    vals[i] = d;
                    var = fmaf(d, d, var);
                }
                const float inv = rsqrtf(quad_sum(var) * (1.f / 64.f) + 1e-6f);
                if (valid) {
                    half* nm_pix = normalized + row_off + size_t(x) * 64;
#pragma unroll
                    for (int nt = 0; nt < 8; ++nt) {
                        const int c = 8 * nt + 2 * q;
                        *reinterpret_cast<half2*>(nm_pix + c) = __floats2half2_rn(
                            fmaf(vals[2 * nt] * inv, norm_w[c], norm_b[c]),
                            fmaf(vals[2 * nt + 1] * inv, norm_w[c + 1], norm_b[c + 1]));
                    }
                }
            }
        }
    }
}

#define PROJECT_FUSED_ARGS                                                                                   \
    half *state, half *normalized, const half *in, const half *weight, const float *bias, const float *gamma, \
        const float *norm_w, const float *norm_b, int h, int w, int tiles_x, int tiles_y
#define PROJECT_FUSED_PASS state, normalized, in, weight, bias, gamma, norm_w, norm_b, h, w, tiles_x, tiles_y
extern "C" __global__ void __launch_bounds__(128, 3) project_full_norm_f16(PROJECT_FUSED_ARGS) {
    project_k16_3tap<true, true, 176>(PROJECT_FUSED_PASS);
}
extern "C" __global__ void __launch_bounds__(128, 3) project_full_res_f16(PROJECT_FUSED_ARGS) {
    project_k16_3tap<true, false, 176>(PROJECT_FUSED_PASS);
}
extern "C" __global__ void __launch_bounds__(128, 2) project_full_norm_f32(PROJECT_FUSED_ARGS) {
    project_k16_3tap<false, true, 176>(PROJECT_FUSED_PASS);
}
extern "C" __global__ void __launch_bounds__(128, 2) project_full_res_f32(PROJECT_FUSED_ARGS) {
    project_k16_3tap<false, false, 176>(PROJECT_FUSED_PASS);
}
extern "C" __global__ void __launch_bounds__(128, 3) project_reduced_norm_f16(PROJECT_FUSED_ARGS) {
    project_k16_3tap<true, true, 96>(PROJECT_FUSED_PASS);
}
extern "C" __global__ void __launch_bounds__(128, 3) project_reduced_res_f16(PROJECT_FUSED_ARGS) {
    project_k16_3tap<true, false, 96>(PROJECT_FUSED_PASS);
}
extern "C" __global__ void __launch_bounds__(128, 2) project_reduced_norm_f32(PROJECT_FUSED_ARGS) {
    project_k16_3tap<false, true, 96>(PROJECT_FUSED_PASS);
}
extern "C" __global__ void __launch_bounds__(128, 2) project_reduced_res_f32(PROJECT_FUSED_ARGS) {
    project_k16_3tap<false, false, 96>(PROJECT_FUSED_PASS);
}

extern "C" __global__ void gated_residual_nhwc(half* state, const half* projected, const float* bias,
                                               const float* gamma, int total) {
    const int index = thread_index();
    if (index >= total) return;
    const int c = index % features;
    state[index] = __float2half_rn(
        fmaf(fast_mish(__half2float(projected[index]) + bias[c]), gamma[c], __half2float(state[index])));
}

// gated_residual_nhwc followed by the next block's LayerNorm. One warp handles
// one pixel's 64 channels, so the updated state is read only once. The norm
// uses the FP16-rounded state, exactly as a separate norm_nhwc would.
extern "C" __global__ void residual_norm_nhwc(half* state, half* normalized, const half* projected,
                                              const float* bias, const float* gamma, const float* weight,
                                              const float* norm_bias, int pixels) {
    const int index = thread_index(), pixel = index / 32, c = 2 * (index % 32);
    if (pixel >= pixels) return;
    const int at = pixel * features + c;
    const float2 p = __half22float2(*reinterpret_cast<const half2*>(projected + at));
    const float2 s = __half22float2(*reinterpret_cast<const half2*>(state + at));
    const half2 updated = __floats2half2_rn(fmaf(fast_mish(p.x + bias[c]), gamma[c], s.x),
                                            fmaf(fast_mish(p.y + bias[c + 1]), gamma[c + 1], s.y));
    *reinterpret_cast<half2*>(state + at) = updated;
    const float2 pair = __half22float2(updated);
    const float mean = warp_sum(pair.x + pair.y) / features;
    const float lo = pair.x - mean, hi = pair.y - mean;
    const float inverse = rsqrtf(warp_sum(lo * lo + hi * hi) / features + 1e-6f);
    *reinterpret_cast<half2*>(normalized + at) = __floats2half2_rn(fmaf(lo * inverse, weight[c], norm_bias[c]),
                                                                   fmaf(hi * inverse, weight[c + 1], norm_bias[c + 1]));
}

// Fills a buffer with deterministic pseudo-random halves in [-1, 1) so that
// plan timing sees realistic switching activity rather than zeros.
extern "C" __global__ void fill_noise(half* out, int total) {
    const int index = thread_index();
    if (index < total) out[index] = __float2half_rn(hash_noise_float(index));
}
extern "C" __global__ void input_bias_norm(half* state, half* initial, half* normalized, const half* projected,
                                           const float* bias, const float* weight, const float* norm_bias, int pixels) {
    const int index = thread_index(), pixel = index / 32, c = 2 * (index % 32);
    if (pixel >= pixels) return;
    const int at = pixel * features + c;
    const float2 p = __half22float2(*reinterpret_cast<const half2*>(projected + at));
    const half2 updated = __floats2half2_rn(p.x + bias[c], p.y + bias[c + 1]);
    *reinterpret_cast<half2*>(initial + at) = updated;
    *reinterpret_cast<half2*>(state + at) = updated;
    const float2 pair = __half22float2(updated);
    const float mean = warp_sum(pair.x + pair.y) / features;
    const float lo = pair.x - mean, hi = pair.y - mean;
    const float inverse = rsqrtf(warp_sum(lo * lo + hi * hi) / features + 1e-6f);
    *reinterpret_cast<half2*>(normalized + at) =
        __floats2half2_rn(fmaf(lo * inverse, weight[c], norm_bias[c]), fmaf(hi * inverse, weight[c + 1], norm_bias[c + 1]));
}

extern "C" __global__ void reduce_norm_nhwc(half* out, half* normalized, const half* in, const float* bias,
                                            const float* weight, const float* norm_bias, int h, int w, int pixels) {
    const int index = thread_index(), pixel = index / 32, c = 2 * (index % 32);
    if (pixel >= pixels) return;
    const int n = pixel / (h * w), x0 = 2 * (pixel % w), y = 2 * (pixel / w % h) + (c / 2) % 2;
    const int channel = c / 4;
    const int source0 = ((n * (2 * h) + y) * (2 * w) + x0) * (features / 4) + channel;
    const int source1 = source0 + (features / 4);
    const half2 updated = __floats2half2_rn(leaky(__half2float(in[source0]) + bias[channel]),
                                            leaky(__half2float(in[source1]) + bias[channel]));
    const int at = pixel * features + c;
    *reinterpret_cast<half2*>(out + at) = updated;
    const float2 pair = __half22float2(updated);
    const float mean = warp_sum(pair.x + pair.y) / features;
    const float lo = pair.x - mean, hi = pair.y - mean;
    const float inverse = rsqrtf(warp_sum(lo * lo + hi * hi) / features + 1e-6f);
    *reinterpret_cast<half2*>(normalized + at) =
        __floats2half2_rn(fmaf(lo * inverse, weight[c], norm_bias[c]), fmaf(hi * inverse, weight[c + 1], norm_bias[c + 1]));
}

extern "C" __global__ void expand_residual_norm_nhwc(half* state, half* normalized, const half* in, const float* bias,
                                                     const float* weight, const float* norm_bias, int h, int w,
                                                     int pixels) {
    const int index = thread_index(), pixel = index / 32, c = 2 * (index % 32);
    if (pixel >= pixels) return;
    const int n = pixel / (h * w), x = pixel % w, y = pixel / w % h;
    const int sub = 2 * (y % 2) + (x % 2);
    const int base = ((n * (h / 2) + y / 2) * (w / 2) + x / 2) * (4 * features);
    const int ch0 = 4 * c + sub, ch1 = 4 * (c + 1) + sub;
    const int at = pixel * features + c;
    const float2 s = __half22float2(*reinterpret_cast<const half2*>(state + at));
    const half2 updated = __floats2half2_rn(leaky(__half2float(in[base + ch0]) + bias[ch0]) + s.x,
                                            leaky(__half2float(in[base + ch1]) + bias[ch1]) + s.y);
    *reinterpret_cast<half2*>(state + at) = updated;
    const float2 pair = __half22float2(updated);
    const float mean = warp_sum(pair.x + pair.y) / features;
    const float lo = pair.x - mean, hi = pair.y - mean;
    const float inverse = rsqrtf(warp_sum(lo * lo + hi * hi) / features + 1e-6f);
    *reinterpret_cast<half2*>(normalized + at) =
        __floats2half2_rn(fmaf(lo * inverse, weight[c], norm_bias[c]), fmaf(hi * inverse, weight[c + 1], norm_bias[c + 1]));
}

extern "C" __global__ void expand_residual_add_nhwc(half* state, const half* initial, const half* in,
                                                    const float* bias, int h, int w, int total) {
    const int index = thread_index();
    if (index >= total) return;
    const int c = index % features, pixel = index / features, n = pixel / (h * w);
    const int x = pixel % w, y = pixel / w % h;
    const int channel = 4 * c + 2 * (y % 2) + x % 2;
    const int source = ((n * (h / 2) + y / 2) * (w / 2) + x / 2) * (4 * features) + channel;
    const half updated = __float2half_rn(leaky(__half2float(in[source]) + bias[channel]) + __half2float(state[index]));
    state[index] = __hadd(updated, initial[index]);
}

extern "C" __global__ void reduce_nhwc(half* out, const half* in, const float* bias, int h, int w, int total) {
    const int index = thread_index();
    if (index >= total) return;
    const int c = index % features, pixel = index / features, n = pixel / (h * w);
    const int x = 2 * (pixel % w) + c % 2, y = 2 * (pixel / w % h) + c / 2 % 2;
    const int channel = c / 4;
    const int source = ((n * (2 * h) + y) * (2 * w) + x) * (features / 4) + channel;
    out[index] = __float2half_rn(leaky(__half2float(in[source]) + bias[channel]));
}

extern "C" __global__ void expand_residual_nhwc(half* state, const half* in, const float* bias, int h, int w, int total) {
    const int index = thread_index();
    if (index >= total) return;
    const int c = index % features, pixel = index / features, n = pixel / (h * w);
    const int x = pixel % w, y = pixel / w % h;
    const int channel = 4 * c + 2 * (y % 2) + x % 2;
    const int source = ((n * (h / 2) + y / 2) * (w / 2) + x / 2) * (4 * features) + channel;
    state[index] = __float2half_rn(leaky(__half2float(in[source]) + bias[channel]) + __half2float(state[index]));
}

extern "C" __global__ void add_nhwc(half* out, const half* initial, int total) {
    const int index = thread_index();
    if (index < total) out[index] = __hadd(out[index], initial[index]);
}

template<class Rgb>
__device__ void output_nhwc(Rgb* out, const half* in, const float* bias, int h, int w, int ph, int pw, int total) {
    const int index = thread_index();
    if (index >= total) return;
    const int plane = index / (h * w), y = index / w % h, x = index % w;
    const int c = 16 * (plane % 3) + 4 * (y % 4) + x % 4;
    const int source = (((plane / 3) * ph + y / 4) * pw + x / 4) * 48 + c;
    write_value(out, index, __half2float(in[source]) + bias[c]);
}

extern "C" __global__ void output_nhwc_f32(float* out, const half* in, const float* bias, int h, int w,
                                           int ph, int pw, int total) {
    output_nhwc(out, in, bias, h, w, ph, pw, total);
}

extern "C" __global__ void output_nhwc_f16(half* out, const half* in, const float* bias, int h, int w,
                                           int ph, int pw, int total) {
    output_nhwc(out, in, bias, h, w, ph, pw, total);
}
