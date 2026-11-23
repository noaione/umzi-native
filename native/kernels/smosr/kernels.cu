#include <cuda_fp16.h>

namespace {
constexpr int context = 2;
constexpr int features = 48;
constexpr int pairs = features / 2;
constexpr int half_values = 1 << 16;

__device__ int reflected(int coord, int extent) {
    if (coord < 0) return -coord;
    return coord < extent ? coord : 2 * extent - coord - 2;
}

__device__ float rounded(float value, bool fp16) {
    return fp16 ? __half2float(__float2half_rn(value)) : value;
}

__device__ float silu_value(float x) { return x / (1.f + expf(-x)); }

__device__ half2 lookup(half2 input, const half* table) {
    return __halves2half2(table[__half_as_ushort(__low2half(input))], table[__half_as_ushort(__high2half(input))]);
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

extern "C" __global__ void init_activations(half* table) {
    const auto i = static_cast<unsigned short>(blockIdx.x * blockDim.x + threadIdx.x);
    const float x = __half2float(__ushort_as_half(i));
    table[i] = __float2half_rn(silu_value(x));
    table[half_values + i] = __float2half_rn(tanhf(x));
}

extern "C" __global__ void silu(half2* output, const float2* bias, const half* table, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float2 b = bias[i % pairs];
    if (table) {
        output[i] = lookup(__hadd2(output[i], __float22half2_rn(b)), table);
    } else {
        const float2 x = __half22float2(output[i]);
        output[i] = __floats2half2_rn(silu_value(x.x + b.x), silu_value(x.y + b.y));
    }
}

extern "C" __global__ void gated_residual(half2* output, const half2* projection, const half2* shortcut,
                                          const float2* bias, const float2* shortcut_bias, const half2* middle_skip,
                                          const half* table, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int c = i % pairs, src = i / pairs * 2 * pairs + c;
    half2 result;
    if (table) {
        const half2 value = __hadd2(projection[src], __float22half2_rn(bias[c]));
        const half2 gate = __hadd2(projection[src + pairs], __float22half2_rn(bias[c + pairs]));
        half2 skip = shortcut[i];
        if (shortcut_bias) skip = __hadd2(skip, __float22half2_rn(shortcut_bias[c]));
        result = __hmul2_rn(__hadd2(value, skip), lookup(gate, table + half_values));
    } else {
        const float2 value = __half22float2(projection[src]), b = bias[c];
        const float2 gate = __half22float2(projection[src + pairs]), g = bias[c + pairs];
        float2 skip = __half22float2(shortcut[i]);
        if (shortcut_bias) {
            const float2 s = shortcut_bias[c];
            skip = {skip.x + s.x, skip.y + s.y};
        }
        result = __floats2half2_rn((value.x + b.x + skip.x) * tanhf(gate.x + g.x),
                                   (value.y + b.y + skip.y) * tanhf(gate.y + g.y));
    }
    output[i] = middle_skip ? __hadd2(result, middle_skip[i]) : result;
}

extern "C" __global__ void concat(half2* output, const half2* rgb, const float2* rgb_bias, const half2* body,
                                  const float2* body_bias, int count, bool fp16) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int pixel = i / 32, c = i % 32;
    if (c >= 30) {
        output[i] = __float2half2_rn(0.f);
        return;
    }
    const half2 value = c < 6 ? rgb[pixel * 8 + c] : body[pixel * pairs + c - 6];
    const float2 bias = c < 6 ? rgb_bias[c] : body_bias[c - 6];
    if (fp16) {
        output[i] = __hadd2(value, __float22half2_rn(bias));
    } else {
        const float2 x = __half22float2(value);
        output[i] = __floats2half2_rn(x.x + bias.x, x.y + bias.y);
    }
}

extern "C" __global__ void shuffle_f32(float* output, const half* input, const float* bias, int h, int w, int count) {
    shuffle(output, input, bias, h, w, count, false);
}

extern "C" __global__ void shuffle_f16(half* output, const half* input, const float* bias, int h, int w, int count) {
    shuffle(output, input, bias, h, w, count, true);
}
