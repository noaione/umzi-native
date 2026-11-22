#include <cuda_fp16.h>
#include "../../common/resize_rgb.cuh"

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

__device__ __forceinline__ unsigned shared_address(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

// 16-byte async copy; src_bytes 0 fills the destination with zeros.
__device__ __forceinline__ void copy_async(half* target, const half* source, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(shared_address(target)), "l"(source),
                 "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void copy_async_always(half* target, const half* source) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_address(target)), "l"(source));
}

__device__ __forceinline__ int swizzle64(int row, int chunk) { return (row * 8 + (chunk ^ (row & 7))) * 8; }
__device__ __forceinline__ int swizzle32(int row, int chunk) { return (row * 4 + (chunk ^ ((row >> 1) & 3))) * 8; }

__device__ __forceinline__ void load_matrix_x4(unsigned (&r)[4], const half* pointer) {
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(shared_address(pointer)));
}

__device__ __forceinline__ void load_matrix_x2(unsigned (&r)[2], const half* pointer) {
    asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(shared_address(pointer)));
}

template<bool HalfAccumulate> struct Accumulator;

template<> struct Accumulator<false> {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    __device__ __forceinline__ void mma(const unsigned (&a)[4], unsigned b0, unsigned b1) {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
            "{%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __device__ __forceinline__ float2 pair(int half_row) const {
        return make_float2(c[2 * half_row], c[2 * half_row + 1]);
    }
    __device__ __forceinline__ unsigned packed_half2(int half_row) const {
        const half2 h = __floats2half2_rn(c[2 * half_row], c[2 * half_row + 1]);
        return *reinterpret_cast<const unsigned*>(&h);
    }
};

template<> struct Accumulator<true> {
    unsigned c[2] = {0u, 0u};
    __device__ __forceinline__ void mma(const unsigned (&a)[4], unsigned b0, unsigned b1) {
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, "
            "{%0, %1};\n"
            : "+r"(c[0]), "+r"(c[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __device__ __forceinline__ float2 pair(int half_row) const {
        const unsigned value = c[half_row];
        return __half22float2(*reinterpret_cast<const half2*>(&value));
    }
    __device__ __forceinline__ unsigned packed_half2(int half_row) const { return c[half_row]; }
};

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

// Tiled 16x16 depthwise branch + Mish gate for the 24 branch channels written
// in planar-by-type layout by expand_gate_*. Grid: (tiles_x * tiles_y * n, 3).
template<int OutStride, int PlainGroups, int BranchSlotOffset>
__device__ void gate_branch_tiled(half* gated, const half* branch_out, const float* expand_bias, const float* square,
                                  const float* square_bias, const float* row, const float* row_bias,
                                  const float* column, const float* column_bias, int h, int w, int pixels, int tiles_x,
                                  int tiles_y) {
    __shared__ __align__(16) half tile_val[26 * 16 * 8];
    const int tid = threadIdx.x, lx = tid & 15, ly = tid >> 4;
    const int tile = blockIdx.x, type = blockIdx.y;
    const int tx = tile % tiles_x, ty = tile / tiles_x % tiles_y, image = tile / (tiles_x * tiles_y);
    const int x0 = tx * 16, y0 = ty * 16;

    const int rx = (type == 0) ? 1 : (type == 1 ? 5 : 0);
    const int ry = (type == 0) ? 1 : (type == 2 ? 5 : 0);
    const int sw = 16 + 2 * rx, sh = 16 + 2 * ry, total_tile = sw * sh;
    const half* plane = branch_out + size_t(type) * pixels * 8;
    for (int idx = tid; idx < total_tile; idx += 256) {
        const int sy = idx / sw, sx = idx % sw;
        const int py = y0 - ry + sy, px = x0 - rx + sx;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (unsigned(py) < unsigned(h) && unsigned(px) < unsigned(w)) {
            const size_t pix = (size_t(image) * h + py) * w + px;
            v = *reinterpret_cast<const uint4*>(plane + pix * 8);
        }
        *reinterpret_cast<uint4*>(tile_val + idx * 8) = v;
    }
    __syncthreads();

    const int y = y0 + ly, x = x0 + lx;
    if (y >= h || x >= w) return;

    const float* eb = expand_bias + 8 * (PlainGroups + 2 * type);
    const float vb[8] = {eb[4], eb[5], eb[6], eb[7], eb[12], eb[13], eb[14], eb[15]};
    const float gb[8] = {eb[0], eb[1], eb[2], eb[3], eb[8], eb[9], eb[10], eb[11]};
    const float* filter = (type == 0) ? square : (type == 1 ? row : column);
    const float* dw_bias = (type == 0) ? square_bias : (type == 1 ? row_bias : column_bias);

    float accum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (type == 0) {
#pragma unroll
        for (int ky = -1; ky <= 1; ++ky) {
            if (unsigned(y + ky) >= unsigned(h)) continue;
#pragma unroll
            for (int kx = -1; kx <= 1; ++kx) {
                if (unsigned(x + kx) >= unsigned(w)) continue;
                const uint4 raw = *reinterpret_cast<const uint4*>(tile_val + ((ly + 1 + ky) * 18 + (lx + 1 + kx)) * 8);
                const half2* p = reinterpret_cast<const half2*>(&raw);
                const int tap = (ky + 1) * 3 + (kx + 1);
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const float2 f = __half22float2(p[pair]);
                    accum[2 * pair] = fmaf(filter[(2 * pair) * 9 + tap], f.x + vb[2 * pair], accum[2 * pair]);
                    accum[2 * pair + 1] =
                        fmaf(filter[(2 * pair + 1) * 9 + tap], f.y + vb[2 * pair + 1], accum[2 * pair + 1]);
                }
            }
        }
    } else if (type == 1) {
#pragma unroll
        for (int kx = -5; kx <= 5; ++kx) {
            if (unsigned(x + kx) >= unsigned(w)) continue;
            const uint4 raw = *reinterpret_cast<const uint4*>(tile_val + (ly * 26 + (lx + 5 + kx)) * 8);
            const half2* p = reinterpret_cast<const half2*>(&raw);
            const int tap = kx + 5;
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const float2 f = __half22float2(p[pair]);
                accum[2 * pair] = fmaf(filter[(2 * pair) * 11 + tap], f.x + vb[2 * pair], accum[2 * pair]);
                accum[2 * pair + 1] =
                    fmaf(filter[(2 * pair + 1) * 11 + tap], f.y + vb[2 * pair + 1], accum[2 * pair + 1]);
            }
        }
    } else {
#pragma unroll
        for (int ky = -5; ky <= 5; ++ky) {
            if (unsigned(y + ky) >= unsigned(h)) continue;
            const uint4 raw = *reinterpret_cast<const uint4*>(tile_val + ((ly + 5 + ky) * 16 + lx) * 8);
            const half2* p = reinterpret_cast<const half2*>(&raw);
            const int tap = ky + 5;
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const float2 f = __half22float2(p[pair]);
                accum[2 * pair] = fmaf(filter[(2 * pair) * 11 + tap], f.x + vb[2 * pair], accum[2 * pair]);
                accum[2 * pair + 1] =
                    fmaf(filter[(2 * pair + 1) * 11 + tap], f.y + vb[2 * pair + 1], accum[2 * pair + 1]);
            }
        }
    }
    const size_t pix = (size_t(image) * h + y) * w + x;
    const uint4 g_raw = *reinterpret_cast<const uint4*>(branch_out + size_t(pixels) * 24 + pix * 24 + 8 * type);
    const half2* gp = reinterpret_cast<const half2*>(&g_raw);
    uint4 res;
    half2* out_p = reinterpret_cast<half2*>(&res);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 gf = __half22float2(gp[pair]);
        out_p[pair] = __floats2half2_rn(fast_mish(gf.x + gb[2 * pair]) * (accum[2 * pair] + dw_bias[2 * pair]),
                                        fast_mish(gf.y + gb[2 * pair + 1]) * (accum[2 * pair + 1] + dw_bias[2 * pair + 1]));
    }
    *reinterpret_cast<uint4*>(gated + pix * OutStride + BranchSlotOffset + 8 * type) = res;
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
    if (index >= total) return;
    unsigned x = unsigned(index) * 2654435761u + 0x9e3779b9u;
    x ^= x >> 15;
    x *= 0x2c1b3c6du;
    x ^= x >> 12;
    out[index] = __float2half_rn(float(x & 0xffff) / 32768.f - 1.f);
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
