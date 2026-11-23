#include <cuda_fp16.h>

namespace {
constexpr int context = 2;
constexpr int features = 48;

__device__ int reflected(int coord, int extent) {
    if (coord < 0) return -coord;
    return coord < extent ? coord : 2 * extent - coord - 2;
}

__device__ float rounded(float value, bool fp16) {
    return fp16 ? __half2float(__float2half_rn(value)) : value;
}

template<class T>
__device__ void pack(half* output, const T* input, int h, int w, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int padded_w = w + 2 * context, padded_h = h + 2 * context;
    const int c = i % 8, row = i / (8 * padded_w);
    const int x = reflected(i / 8 % padded_w - context, w);
    const int y = reflected(row % padded_h - context, h);
    const int n = row / padded_h;
    output[i] = __float2half_rn(c < 3 ? float(input[((n * 3 + c) * h + y) * w + x]) : 0.f);
}

template<class T>
__device__ void shuffle(T* output, const half* input, const float* bias, int h, int w, int count, bool fp16) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int x = i % (2 * w), row = i / (2 * w), y = row % (2 * h);
    const int n = row / (6 * h), c = row / (2 * h) % 3;
    const int lane = c * 4 + (y % 2) * 2 + x % 2;
    const int pixel = (n * (h + 2 * context) + y / 2 + context) * (w + 2 * context) + x / 2 + context;
    output[i] = T(__half2float(input[pixel * 16 + lane]) + rounded(bias[lane], fp16));
}
}

extern "C" __global__ void pack_f32(half* output, const float* input, int h, int w, int count) {
    pack(output, input, h, w, count);
}

extern "C" __global__ void pack_f16(half* output, const half* input, int h, int w, int count) {
    pack(output, input, h, w, count);
}

extern "C" __global__ void silu(half* output, const float* bias, int count, bool fp16) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float x = rounded(__half2float(output[i]) + rounded(bias[i % features], fp16), fp16);
    output[i] = __float2half_rn(x / (1.f + expf(-x)));
}

extern "C" __global__ void gated_residual(half* output, const half* projection, const half* shortcut,
                                          const float* bias, const float* shortcut_bias, int count, bool fp16) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int c = i % features, src = i / features * 2 * features + c;
    const float value = rounded(__half2float(projection[src]) + rounded(bias[c], fp16), fp16);
    const float gate = rounded(__half2float(projection[src + features]) + rounded(bias[c + features], fp16), fp16);
    float skip = __half2float(shortcut[i]);
    if (shortcut_bias) skip = rounded(skip + rounded(shortcut_bias[c], fp16), fp16);
    output[i] = __float2half_rn(rounded(value + skip, fp16) * rounded(tanhf(gate), fp16));
}

extern "C" __global__ void residual(half* output, const half* skip, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __hadd(output[i], skip[i]);
}

extern "C" __global__ void concat(half* output, const half* rgb, const float* rgb_bias, const half* body,
                                  const float* body_bias, int count, bool fp16) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int pixel = i / 64, c = i % 64;
    float value = 0.f;
    if (c < 12) value = __half2float(rgb[pixel * 16 + c]) + rounded(rgb_bias[c], fp16);
    else if (c < 60) value = __half2float(body[pixel * features + c - 12]) + rounded(body_bias[c - 12], fp16);
    output[i] = __float2half_rn(value);
}

extern "C" __global__ void shuffle_f32(float* output, const half* input, const float* bias, int h, int w, int count) {
    shuffle(output, input, bias, h, w, count, false);
}

extern "C" __global__ void shuffle_f16(half* output, const half* input, const float* bias, int h, int w, int count) {
    shuffle(output, input, bias, h, w, count, true);
}
