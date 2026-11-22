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
template<int Groups, int OutStride>
__device__ void gate_groups(half* out, const half* in, const float* bias, const float* square,
                            const float* square_bias, const float* row, const float* row_bias,
                            const float* column, const float* column_bias, int h, int w, int pixels) {
    constexpr int branch_groups = 3 * branch_channels / 4, plain = Groups - branch_groups, stride = 8 * Groups;
    static_assert(OutStride % 8 == 0 && OutStride >= 4 * Groups && branch_groups == 6);
    const int index = thread_index();
    int pixel, g, type = -1;
    if (index < pixels * plain) {
        pixel = index / plain;
        g = branch_groups + index % plain;
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
