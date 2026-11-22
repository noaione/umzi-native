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
    const int pixel = index / 3, c = index % 3, n = pixel / (ph * pw);
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

extern "C" __global__ void gate_nhwc(half* out, const half* in, const float* bias, const float* square,
                                      const float* square_bias, const float* row, const float* row_bias,
                                      const float* column, const float* column_bias, int hidden, int h, int w,
                                      int total) {
    const int index = thread_index();
    if (index >= total) return;
    const int pixel = index / hidden, c = index % hidden, stride = 2 * hidden;
    const half* here = in + size_t(pixel) * stride;
    const int source_channel = hidden + c, branch = c - (hidden - 3 * branch_channels);
    float value;
    if (branch < 0) {
        value = __half2float(here[source_channel]) + bias[source_channel];
    } else {
        const int channel = branch % branch_channels, type = branch / branch_channels;
        const int x = pixel % w, y = pixel / w % h;
        const int radius_x = type == 0 ? 1 : type == 1 ? 5 : 0;
        const int radius_y = type == 0 ? 1 : type == 2 ? 5 : 0;
        const int width = 2 * radius_x + 1, taps = width * (2 * radius_y + 1);
        const float* filter = (type == 0 ? square : type == 1 ? row : column) + channel * taps;
        const float* offset = type == 0 ? square_bias : type == 1 ? row_bias : column_bias;
        float accum = 0.f;
        for (int ky = -radius_y; ky <= radius_y; ++ky) {
            if (unsigned(y + ky) >= unsigned(h)) continue;
            for (int kx = -radius_x; kx <= radius_x; ++kx) {
                if (unsigned(x + kx) >= unsigned(w)) continue;
                const half* neighbor = here + (ky * w + kx) * stride;
                const float sample = __half2float(neighbor[source_channel]) + bias[source_channel];
                accum = fmaf(filter[(ky + radius_y) * width + kx + radius_x], sample, accum);
            }
        }
        value = accum + offset[channel];
    }
    out[index] = __float2half_rn(mish(__half2float(here[c]) + bias[c]) * value);
}

extern "C" __global__ void gated_residual_nhwc(half* state, const half* projected, const float* bias,
                                               const float* gamma, int total) {
    const int index = thread_index();
    if (index >= total) return;
    const int c = index % features;
    state[index] = __float2half_rn(
        fmaf(mish(__half2float(projected[index]) + bias[c]), gamma[c], __half2float(state[index])));
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
