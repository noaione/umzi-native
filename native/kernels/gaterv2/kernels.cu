#include <cuda_fp16.h>
#include "../../common/resize_rgb.cuh"

extern "C" __global__ void pad_rgb(float* out, const float* in, int h, int w, int ph, int pw, int total) {
    resize_rgb<float, false>(out, in, h, w, ph, pw, total);
}

extern "C" __global__ void crop_rgb(float* out, const float* in, int h, int w, int ph, int pw, int total) {
    resize_rgb<float, true>(out, in, h, w, ph, pw, total);
}

// Q and K are contiguous [batch, channels, tokens]. Each norm is over
// channels at one spatial token, independent across the batch.
extern "C" __global__ void normalize_qk(
    float* query_out, float* key_out, const float* query, const float* key,
    int channels, int tokens, int total_tokens
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_tokens) return;
    const int base = (index / tokens) * channels * tokens + index % tokens;
    float query_ss = 0;
    float key_ss = 0;
    for (int channel = 0; channel < channels; ++channel) {
        const int offset = base + channel * tokens;
        const float q = query[offset];
        const float k = key[offset];
        query_ss = fmaf(q, q, query_ss);
        key_ss = fmaf(k, k, key_ss);
    }
    const float inverse_q = 1.f / sqrtf(query_ss);
    const float inverse_k = 1.f / sqrtf(key_ss);
    for (int channel = 0; channel < channels; ++channel) {
        const int offset = base + channel * tokens;
        query_out[offset] = query[offset] * inverse_q;
        key_out[offset] = key[offset] * inverse_k;
    }
}

// The GEMM product is [batch, tokens, value_channels]. Write NCHW order
// for the following gate while applying the per-token denominator.
extern "C" __global__ void finish_attention(
    float* out, const float* product, const float* value_sum, const float* contraction,
    int channels, int tokens, int total
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int plane = index / tokens;
    const int batch = plane / channels;
    const int token = index % tokens;
    const int position = batch * tokens + token;
    const float numerator = value_sum[plane] + product[position * channels + plane % channels];
    const float reciprocal = 1.f / (float(tokens) + contraction[position]);
    out[index] = numerator * reciprocal;
}


// The kernels below execute the packaged session in NCHW order. Feature
// tensors are stored as T (half in both session precisions); register
// arithmetic, small parameters and attention intermediates are FP32. The build compiles this family
// without fast math, and explicit _rn intrinsics keep the separate roundings
// of the eager operations they replace.

__device__ __forceinline__ int grid_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ float load(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float load(const half* p, size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ void store(float* p, size_t i, float v) { p[i] = v; }
__device__ __forceinline__ void store(half* p, size_t i, float v) { p[i] = __float2half_rn(v); }

__device__ __forceinline__ float bias_at(const float* bias, int channel) { return bias ? bias[channel] : 0.f; }

__device__ __forceinline__ float warp_sum(float value) {
    for (int delta = 16; delta; delta /= 2) value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, delta));
    return value;
}

__device__ __forceinline__ float mish(float x) {
    return __fmul_rn(x, tanhf(log1pf(expf(x))));
}

// Reflection padding from the caller's RGB input into FP16 features.
template<class In>
__device__ void pad_features(half* out, const In* in, int h, int w, int ph, int pw, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int plane = index / (ph * pw);
    int y = index / pw % ph, x = index % pw;
    if (y >= h) y = ph - h < h ? 2 * h - 2 - y : h - 1;
    if (x >= w) x = pw - w < w ? 2 * w - 2 - x : w - 1;
    out[index] = __float2half_rn(load(in, (size_t(plane) * h + y) * w + x));
}

// Crop the padded network output and add the unpadded input. Inside the crop
// the reflection-padded input equals the original input.
template<class Io>
__device__ void residual_crop(Io* out, const half* network, const Io* in, int h, int w, int ph, int pw, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int plane = index / (h * w), y = index / w % h, x = index % w;
    store(out, index, __fadd_rn(__half2float(network[(size_t(plane) * ph + y) * pw + x]), load(in, index)));
}

extern "C" __global__ void pad_features_f32(half* o, const float* i, int h, int w, int ph, int pw, int t) {
    pad_features(o, i, h, w, ph, pw, t);
}
extern "C" __global__ void pad_features_f16(half* o, const half* i, int h, int w, int ph, int pw, int t) {
    pad_features(o, i, h, w, ph, pw, t);
}
extern "C" __global__ void residual_crop_f32(float* o, const half* n, const float* i, int h, int w, int ph, int pw,
                                             int t) {
    residual_crop(o, n, i, h, w, ph, pw, t);
}
extern "C" __global__ void residual_crop_f16(half* o, const half* n, const half* i, int h, int w, int ph, int pw,
                                             int t) {
    residual_crop(o, n, i, h, w, ph, pw, t);
}

extern "C" __global__ void weights_to_half(half* out, const float* in, int total) {
    const int index = grid_index();
    if (index < total) out[index] = __float2half_rn(in[index]);
}

// RMSNorm over channels at each pixel. A block covers 32 pixels; its eight
// rows split the channels and reduce through shared memory. total is the
// number of pixels over the batch.
template<class T>
__device__ void rms_norm(T* out, const T* in, const float* scale, const float* offset,
                         int channels, int pixels, float inverse_root, int total) {
    __shared__ float partial[8][33];
    const int index = blockIdx.x * 32 + threadIdx.x;
    const bool active = index < total;
    const size_t base = active ? size_t(index / pixels) * channels * pixels + index % pixels : 0;
    float sum = 0;
    if (active)
        for (int c = threadIdx.y; c < channels; c += 8) {
            const float value = load(in, base + size_t(c) * pixels);
            sum = fmaf(value, value, sum);
        }
    partial[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    if (!threadIdx.y) {
        for (int row = 1; row < 8; ++row) sum = __fadd_rn(sum, partial[row][threadIdx.x]);
        partial[0][threadIdx.x] = __fadd_rn(__fmul_rn(sqrtf(sum), inverse_root), 1e-6f);
    }
    __syncthreads();
    if (!active) return;
    const float denominator = partial[0][threadIdx.x];
    for (int c = threadIdx.y; c < channels; c += 8) {
        const size_t at = base + size_t(c) * pixels;
        store(out, at, __fadd_rn(__fmul_rn(scale[c], __fdiv_rn(load(in, at), denominator)), offset[c]));
    }
}

template<class T>
__device__ __forceinline__ float grouped3x3(
    const T* image, const float* weight, float bias, const float* input_bias, int channel, int height, int width,
    int y, int x
) {
    // Each group has two input and two output channels.
    const size_t source = size_t(channel & ~1) * height * width;
    const float* kernel = weight + channel * 18;
    float sum = 0;
    for (int i = 0; i < 2; ++i) {
        const float shift = input_bias[(channel & ~1) + i];
        for (int ky = 0; ky < 3; ++ky) {
            const int yy = y + ky - 1;
            if (yy < 0 || yy >= height) continue;
            for (int kx = 0; kx < 3; ++kx) {
                const int xx = x + kx - 1;
                if (xx < 0 || xx >= width) continue;
                const float value = __fadd_rn(load(image, source + (size_t(i) * height + yy) * width + xx), shift);
                sum = fmaf(kernel[(i * 3 + ky) * 3 + kx], value, sum);
            }
        }
    }
    return __fadd_rn(sum, bias);
}

// Grouped 3x3 convolution with 2 * channels outputs, followed by SimpleGate.
// input_bias is the bias of the preceding 1x1 expansion.
template<class T>
__device__ void local_gate(T* out, const T* in, const float* input_bias, const float* weight, const float* bias,
                           int channels, int height, int width, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int pixels = height * width;
    const int pixel = index % pixels;
    const int channel = index / pixels % channels;
    const int batch = index / pixels / channels;
    const T* image = in + size_t(batch) * 2 * channels * pixels;
    const int y = pixel / width, x = pixel % width;
    const int other = channel + channels;
    // Round each convolution output to FP16 storage, as a separate convolution would.
    const float first = __half2float(__float2half_rn(
        grouped3x3(image, weight, bias[channel], input_bias, channel, height, width, y, x)));
    const float second = __half2float(__float2half_rn(
        grouped3x3(image, weight, bias[other], input_bias, other, height, width, y, x)));
    store(out, index, __fmul_rn(first, second));
}

// One block per selected plane. The FP32 output is the plane sum plus add, or its mean.
template<class T>
__device__ void plane_sum(float* out, const T* in, int pixels, int planes, int batch_planes, int plane_offset,
                          float add, int average) {
    __shared__ float partial[256];
    const int plane = blockIdx.x;
    const size_t source = (size_t(plane / planes) * batch_planes + plane_offset + plane % planes) * pixels;
    float sum = 0;
    for (int i = threadIdx.x; i < pixels; i += blockDim.x) sum = __fadd_rn(sum, load(in, source + i));
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (!threadIdx.x) out[plane] = average ? __fdiv_rn(partial[0], float(pixels)) : __fadd_rn(partial[0], add);
}

// A 1x1 convolution of per-plane FP32 values: out[b, o] = sum_i w[o, i] * in[b, i] + bias[o].
extern "C" __global__ void project(
    float* out, const float* in, const float* weight, const float* bias, int inputs, int outputs, int total
) {
    const int index = grid_index();
    if (index >= total) return;
    const int output = index % outputs;
    const float* source = in + size_t(index / outputs) * inputs;
    const float* row = weight + size_t(output) * inputs;
    float sum = 0;
    for (int i = 0; i < inputs; ++i) sum = fmaf(row[i], source[i], sum);
    out[index] = __fadd_rn(sum, bias[output]);
}

// MetaGated local residual: local * attention * gamma0 + shortcut.
template<class T>
__device__ void local_skip(T* out, const T* local, const float* attention, const T* shortcut, const float* gamma,
                           int channels, int pixels, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int plane = index / pixels;
    const float value = __fmul_rn(__fmul_rn(load(local, index), attention[plane]), gamma[plane % channels]);
    store(out, index, __fadd_rn(value, load(shortcut, index)));
}

template<class T>
__device__ __forceinline__ float depthwise(
    const T* image, size_t source, const float* kernel, float bias, float shift, int height, int width, int y, int x,
    int kernel_height, int kernel_width
) {
    float sum = 0;
    for (int ky = 0; ky < kernel_height; ++ky) {
        const int yy = y + ky - kernel_height / 2;
        if (yy < 0 || yy >= height) continue;
        for (int kx = 0; kx < kernel_width; ++kx) {
            const int xx = x + kx - kernel_width / 2;
            if (xx < 0 || xx >= width) continue;
            sum = fmaf(kernel[ky * kernel_width + kx], __fadd_rn(load(image, source + size_t(yy) * width + xx), shift),
                       sum);
        }
    }
    return __fadd_rn(sum, bias);
}

// InceptionDWConv2d on a channel slice whose batches are batch_planes apart:
// identity, 3x3, 1x11 and 11x1 depthwise branches.
template<class T>
__device__ void inception(T* out, const T* in, const float* input_bias, const float* square, const float* square_bias,
                          const float* row, const float* row_bias, const float* column, const float* column_bias,
                          int channels, int branch, int batch_planes, int height, int width, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int pixels = height * width;
    const int pixel = index % pixels;
    const int channel = index / pixels % channels;
    const int batch = index / pixels / channels;
    const size_t source = (size_t(batch) * batch_planes + channel) * pixels;
    const int identity = channels - 3 * branch;
    const float shift = bias_at(input_bias, channel);
    if (channel < identity) {
        store(out, index, __fadd_rn(load(in, source + pixel), shift));
        return;
    }
    const int kind = (channel - identity) / branch;
    const int group = (channel - identity) % branch;
    const int y = pixel / width, x = pixel % width;
    float value;
    if (kind == 0)
        value = depthwise(in, source, square + group * 9, square_bias[group], shift, height, width, y, x, 3, 3);
    else if (kind == 1)
        value = depthwise(in, source, row + group * 11, row_bias[group], shift, height, width, y, x, 1, 11);
    else
        value = depthwise(in, source, column + group * 11, column_bias[group], shift, height, width, y, x, 11, 1);
    store(out, index, value);
}

// mish(g) * cat(i, mixed), where g and i are the first channels of the
// 2 * hidden fc1 output and mixed holds hidden - identity contiguous channels.
template<class T>
__device__ void gate(T* out, const T* projected, const float* projected_bias, const T* mixed, int hidden,
                     int identity, int pixels, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int pixel = index % pixels;
    const int channel = index / pixels % hidden;
    const int batch = index / pixels / hidden;
    const size_t source = size_t(batch) * 2 * hidden * pixels + pixel;
    float gated = mish(__fadd_rn(load(projected, source + size_t(channel) * pixels), bias_at(projected_bias, channel)));
    gated = __half2float(__float2half_rn(gated));  // Mish output is an FP16 tensor in eager.
    const float value = channel < identity
        ? __fadd_rn(load(projected, source + size_t(hidden + channel) * pixels), bias_at(projected_bias, hidden + channel))
        : load(mixed, (size_t(batch) * (hidden - identity) + channel - identity) * pixels + pixel);
    store(out, index, __fmul_rn(gated, value));
}

// mish(x), optionally followed by * gamma + shortcut.
template<class T>
__device__ void mish_skip(T* out, const T* in, const float* in_bias, const T* shortcut, const float* gamma,
                          int channels, int pixels, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int channel = index / pixels % channels;
    float value = mish(__fadd_rn(load(in, index), bias_at(in_bias, channel)));
    if (gamma) {
        value = __half2float(__float2half_rn(value));
        value = __fadd_rn(__fmul_rn(value, gamma[channel]), load(shortcut, index));
    }
    store(out, index, value);
}

// Normalize Q and K (the first 2 * channels planes of each batch) over
// channels at each token into a packed FP32 [batch, 2 * channels, tokens] buffer.
template<class T>
__device__ void normalize_qk_packed(float* out, const T* in, int channels, int batch_planes, int tokens,
                                    int total_tokens) {
    const int index = grid_index();
    if (index >= total_tokens) return;
    const int batch = index / tokens, token = index % tokens;
    for (int half_index = 0; half_index < 2; ++half_index) {
        const size_t source = (size_t(batch) * batch_planes + half_index * channels) * tokens + token;
        float* target = out + (size_t(batch) * 2 + half_index) * channels * tokens + token;
        float sum = 0;
        for (int c = 0; c < channels; ++c) {
            const float value = load(in, source + size_t(c) * tokens);
            sum = fmaf(value, value, sum);
        }
        const float inverse = __fdiv_rn(1.f, sqrtf(sum));
        for (int c = 0; c < channels; ++c)
            target[size_t(c) * tokens] = __fmul_rn(load(in, source + size_t(c) * tokens), inverse);
    }
}

// matrix[b, m, c] = sum_n K[b, m, n] * V[b, c, n]. K comes from the FP32
// normalized buffer and V from the projection output, value_offset planes in.
// One warp reduces each output over the tokens; total counts outputs.
template<class T>
__device__ void key_value(float* out, const float* normalized, const T* projected, int key_channels,
                          int value_channels, int batch_planes, int value_offset, int tokens, int total) {
    const int index = grid_index() / 32, lane = threadIdx.x % 32;
    if (index >= total) return;
    const int channel = index % value_channels;
    const int key = index / value_channels % key_channels;
    const int batch = index / value_channels / key_channels;
    const float* k = normalized + (size_t(batch) * 2 * key_channels + key_channels + key) * tokens;
    const size_t v = (size_t(batch) * batch_planes + value_offset + channel) * tokens;
    float sum = 0;
    for (int n = lane; n < tokens; n += 32) sum = fmaf(k[n], load(projected, v + n), sum);
    sum = warp_sum(sum);
    if (!lane) out[index] = sum;
}

// (value_sum + Q^T matrix) * 1 / (tokens + Q^T (sum K + eps)), in NCHW order.
template<class T>
__device__ void attention_output(T* out, const float* normalized, const float* matrix, const float* key_sum,
                                 const float* value_sum, int key_channels, int value_channels, int tokens,
                                 int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int token = index % tokens;
    const int channel = index / tokens % value_channels;
    const int batch = index / tokens / value_channels;
    const float* q = normalized + size_t(batch) * 2 * key_channels * tokens + token;
    const float* m = matrix + size_t(batch) * key_channels * value_channels + channel;
    const float* k = key_sum + size_t(batch) * key_channels;
    float contraction = 0, product = 0;
    for (int i = 0; i < key_channels; ++i) {
        const float query = q[size_t(i) * tokens];
        contraction = fmaf(query, k[i], contraction);
        product = fmaf(query, m[size_t(i) * value_channels], product);
    }
    const float numerator = __fadd_rn(value_sum[size_t(batch) * value_channels + channel], product);
    const float reciprocal = __fdiv_rn(1.f, __fadd_rn(float(tokens), contraction));
    store(out, index, __fmul_rn(numerator, reciprocal));
}

// PixelUnshuffle(2): out[b, 4c + 2i + j, y, x] = in[b, c, 2y + i, 2x + j].
template<class T>
__device__ void unshuffle(T* out, const T* in, int channels, int height, int width, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int x = index % width, y = index / width % height;
    const int plane = index / width / height;
    const int sub = plane % 4, channel = plane / 4 % channels, batch = plane / 4 / channels;
    out[index] = in[((size_t(batch) * channels + channel) * 2 * height + 2 * y + sub / 2) * 2 * width
        + 2 * x + sub % 2];
}

// cat(PixelShuffle(2)(up), skip): both halves have half_channels at height x width.
template<class T>
__device__ void up_concat(T* out, const T* up, const T* skip, int half_channels, int height, int width, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int x = index % width, y = index / width % height;
    const int channel = index / width / height % (2 * half_channels);
    const int batch = index / width / height / (2 * half_channels);
    if (channel < half_channels) {
        const int source = channel * 4 + (y % 2) * 2 + x % 2;
        out[index] = up[((size_t(batch) * 4 * half_channels + source) * (height / 2) + y / 2) * (width / 2) + x / 2];
    } else {
        out[index] = skip[((size_t(batch) * half_channels + channel - half_channels) * height + y) * width + x];
    }
}

template<class T>
__device__ void add_bias(T* out, const float* bias, int channels, int pixels, int total) {
    const int index = grid_index();
    if (index >= total) return;
    store(out, index, __fadd_rn(load(out, index), bias[index / pixels % channels]));
}

template<class T>
__device__ void add(T* out, const T* a, const T* b, int total) {
    const int index = grid_index();
    if (index >= total) return;
    store(out, index, __fadd_rn(load(a, index), load(b, index)));
}

#define UMZI_FEATURE_KERNELS(T, S) \
extern "C" __global__ void rms_norm_##S(T* o, const T* i, const float* s, const float* f, int c, int p, float r, int t) \
    { rms_norm(o, i, s, f, c, p, r, t); } \
extern "C" __global__ void local_gate_##S(T* o, const T* i, const float* ib, const float* w, const float* b, int c, \
    int h, int x, int t) { local_gate(o, i, ib, w, b, c, h, x, t); } \
extern "C" __global__ void plane_sum_##S(float* o, const T* i, int p, int n, int bp, int f, float a, int m) \
    { plane_sum(o, i, p, n, bp, f, a, m); } \
extern "C" __global__ void local_skip_##S(T* o, const T* l, const float* a, const T* s, const float* g, int c, int p, \
    int t) { local_skip(o, l, a, s, g, c, p, t); } \
extern "C" __global__ void inception_##S(T* o, const T* i, const float* ib, const float* a, const float* ab, \
    const float* b, const float* bb, const float* c, const float* cb, int ch, int br, int bp, int h, int w, int t) \
    { inception(o, i, ib, a, ab, b, bb, c, cb, ch, br, bp, h, w, t); } \
extern "C" __global__ void gate_##S(T* o, const T* p, const float* pb, const T* m, int h, int i, int x, int t) \
    { gate(o, p, pb, m, h, i, x, t); } \
extern "C" __global__ void mish_skip_##S(T* o, const T* i, const float* ib, const T* s, const float* g, int c, int p, \
    int t) { mish_skip(o, i, ib, s, g, c, p, t); } \
extern "C" __global__ void normalize_qk_##S(float* o, const T* i, int c, int bp, int n, int t) \
    { normalize_qk_packed(o, i, c, bp, n, t); } \
extern "C" __global__ void key_value_##S(float* o, const float* q, const T* p, int k, int v, int bp, int f, int n, \
    int t) { key_value(o, q, p, k, v, bp, f, n, t); } \
extern "C" __global__ void attention_output_##S(T* o, const float* q, const float* m, const float* k, \
    const float* v, int kc, int vc, int n, int t) { attention_output(o, q, m, k, v, kc, vc, n, t); } \
extern "C" __global__ void unshuffle_##S(T* o, const T* i, int c, int h, int w, int t) \
    { unshuffle(o, i, c, h, w, t); } \
extern "C" __global__ void up_concat_##S(T* o, const T* u, const T* s, int c, int h, int w, int t) \
    { up_concat(o, u, s, c, h, w, t); } \
extern "C" __global__ void add_bias_##S(T* o, const float* b, int c, int p, int t) { add_bias(o, b, c, p, t); } \
extern "C" __global__ void add_##S(T* o, const T* a, const T* b, int t) { add(o, a, b, t); }

UMZI_FEATURE_KERNELS(half, f16)

// The key sum reduces the FP32 normalized keys.
extern "C" __global__ void plane_sum_f32(float* o, const float* i, int p, int n, int bp, int f, float a, int m) {
    plane_sum(o, i, p, n, bp, f, a, m);
}
