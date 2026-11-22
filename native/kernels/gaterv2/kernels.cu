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

// The kernels below execute the packaged FP32 network in NCHW order. The build
// compiles this family without fast math, and explicit _rn intrinsics keep the
// separate roundings of the eager operations they replace.

__device__ __forceinline__ int grid_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ float mish(float x) {
    return __fmul_rn(x, tanhf(log1pf(expf(x))));
}

// RMSNorm over channels at one pixel: scale * x / (||x|| * C^-1/2 + eps) + offset.
extern "C" __global__ void rms_norm(
    float* out, const float* in, const float* scale, const float* offset,
    int channels, int pixels, float inverse_root, int total
) {
    const int index = grid_index();
    if (index >= total) return;
    const size_t base = size_t(index / pixels) * channels * pixels + index % pixels;
    float sum = 0;
    for (int c = 0; c < channels; ++c) {
        const float value = in[base + size_t(c) * pixels];
        sum = fmaf(value, value, sum);
    }
    const float denominator = __fadd_rn(__fmul_rn(sqrtf(sum), inverse_root), 1e-6f);
    for (int c = 0; c < channels; ++c) {
        const size_t at = base + size_t(c) * pixels;
        out[at] = __fadd_rn(__fmul_rn(scale[c], __fdiv_rn(in[at], denominator)), offset[c]);
    }
}

__device__ __forceinline__ float grouped3x3(
    const float* image, const float* weight, float bias, int channel, int height, int width, int y, int x
) {
    // Each group has two input and two output channels.
    const float* source = image + size_t(channel & ~1) * height * width;
    const float* kernel = weight + channel * 18;
    float sum = 0;
    for (int i = 0; i < 2; ++i)
        for (int ky = 0; ky < 3; ++ky) {
            const int yy = y + ky - 1;
            if (yy < 0 || yy >= height) continue;
            for (int kx = 0; kx < 3; ++kx) {
                const int xx = x + kx - 1;
                if (xx < 0 || xx >= width) continue;
                sum = fmaf(kernel[(i * 3 + ky) * 3 + kx], source[(size_t(i) * height + yy) * width + xx], sum);
            }
        }
    return __fadd_rn(sum, bias);
}

// Grouped 3x3 convolution with 2 * channels outputs, followed by SimpleGate.
extern "C" __global__ void local_gate(
    float* out, const float* in, const float* weight, const float* bias,
    int channels, int height, int width, int total
) {
    const int index = grid_index();
    if (index >= total) return;
    const int pixels = height * width;
    const int pixel = index % pixels;
    const int channel = index / pixels % channels;
    const int batch = index / pixels / channels;
    const float* image = in + size_t(batch) * 2 * channels * pixels;
    const int y = pixel / width, x = pixel % width;
    const float first = grouped3x3(image, weight, bias[channel], channel, height, width, y, x);
    const int other = channel + channels;
    const float second = grouped3x3(image, weight, bias[other], other, height, width, y, x);
    out[index] = __fmul_rn(first, second);
}

// One block per selected plane. The output is the plane sum plus add, or its mean.
extern "C" __global__ void plane_sum(
    float* out, const float* in, int pixels, int planes, int batch_planes, int plane_offset,
    float add, int average
) {
    __shared__ float partial[256];
    const int plane = blockIdx.x;
    const float* source = in + (size_t(plane / planes) * batch_planes + plane_offset + plane % planes) * pixels;
    float sum = 0;
    for (int i = threadIdx.x; i < pixels; i += blockDim.x) sum = __fadd_rn(sum, source[i]);
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (!threadIdx.x) out[plane] = average ? __fdiv_rn(partial[0], float(pixels)) : __fadd_rn(partial[0], add);
}

// A 1x1 convolution of per-plane values: out[b, o] = sum_i w[o, i] * in[b, i] + bias[o].
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
extern "C" __global__ void local_skip(
    float* out, const float* local, const float* attention, const float* shortcut, const float* gamma,
    int channels, int pixels, int total
) {
    const int index = grid_index();
    if (index >= total) return;
    const int plane = index / pixels;
    const float value = __fmul_rn(__fmul_rn(local[index], attention[plane]), gamma[plane % channels]);
    out[index] = __fadd_rn(value, shortcut[index]);
}

__device__ __forceinline__ float depthwise(
    const float* source, const float* kernel, float bias, int height, int width, int y, int x,
    int kernel_height, int kernel_width
) {
    float sum = 0;
    for (int ky = 0; ky < kernel_height; ++ky) {
        const int yy = y + ky - kernel_height / 2;
        if (yy < 0 || yy >= height) continue;
        for (int kx = 0; kx < kernel_width; ++kx) {
            const int xx = x + kx - kernel_width / 2;
            if (xx < 0 || xx >= width) continue;
            sum = fmaf(kernel[ky * kernel_width + kx], source[size_t(yy) * width + xx], sum);
        }
    }
    return __fadd_rn(sum, bias);
}

// InceptionDWConv2d on a channel slice whose batches are batch_planes apart:
// identity, 3x3, 1x11 and 11x1 depthwise branches.
extern "C" __global__ void inception(
    float* out, const float* in, const float* square, const float* square_bias,
    const float* row, const float* row_bias, const float* column, const float* column_bias,
    int channels, int branch, int batch_planes, int height, int width, int total
) {
    const int index = grid_index();
    if (index >= total) return;
    const int pixels = height * width;
    const int pixel = index % pixels;
    const int channel = index / pixels % channels;
    const int batch = index / pixels / channels;
    const float* source = in + (size_t(batch) * batch_planes + channel) * pixels;
    const int identity = channels - 3 * branch;
    if (channel < identity) {
        out[index] = source[pixel];
        return;
    }
    const int kind = (channel - identity) / branch;
    const int group = (channel - identity) % branch;
    const int y = pixel / width, x = pixel % width;
    if (kind == 0) out[index] = depthwise(source, square + group * 9, square_bias[group], height, width, y, x, 3, 3);
    else if (kind == 1) out[index] = depthwise(source, row + group * 11, row_bias[group], height, width, y, x, 1, 11);
    else out[index] = depthwise(source, column + group * 11, column_bias[group], height, width, y, x, 11, 1);
}

// mish(g) * cat(i, mixed), where g and i are the first channels of the
// 2 * hidden fc1 output and mixed holds hidden - identity contiguous channels.
extern "C" __global__ void gate(
    float* out, const float* projected, const float* mixed, int hidden, int identity, int pixels, int total
) {
    const int index = grid_index();
    if (index >= total) return;
    const int pixel = index % pixels;
    const int channel = index / pixels % hidden;
    const int batch = index / pixels / hidden;
    const float* source = projected + size_t(batch) * 2 * hidden * pixels + pixel;
    const float gated = mish(source[size_t(channel) * pixels]);
    const float value = channel < identity
        ? source[size_t(hidden + channel) * pixels]
        : mixed[(size_t(batch) * (hidden - identity) + channel - identity) * pixels + pixel];
    out[index] = __fmul_rn(gated, value);
}

// mish(x), optionally followed by * gamma + shortcut.
extern "C" __global__ void mish_skip(
    float* out, const float* in, const float* shortcut, const float* gamma, int channels, int pixels, int total
) {
    const int index = grid_index();
    if (index >= total) return;
    float value = mish(in[index]);
    if (gamma) value = __fadd_rn(__fmul_rn(value, gamma[index / pixels % channels]), shortcut[index]);
    out[index] = value;
}

// Normalize Q and K (the first 2 * channels planes of each batch) over
// channels at each token into a packed [batch, 2 * channels, tokens] buffer.
extern "C" __global__ void normalize_qk_packed(
    float* out, const float* in, int channels, int batch_planes, int tokens, int total_tokens
) {
    const int index = grid_index();
    if (index >= total_tokens) return;
    const int batch = index / tokens, token = index % tokens;
    for (int half = 0; half < 2; ++half) {
        const float* source = in + (size_t(batch) * batch_planes + half * channels) * tokens + token;
        float* target = out + (size_t(batch) * 2 + half) * channels * tokens + token;
        float sum = 0;
        for (int c = 0; c < channels; ++c) {
            const float value = source[size_t(c) * tokens];
            sum = fmaf(value, value, sum);
        }
        const float inverse = __fdiv_rn(1.f, sqrtf(sum));
        for (int c = 0; c < channels; ++c) target[size_t(c) * tokens] = __fmul_rn(source[size_t(c) * tokens], inverse);
    }
}

// matrix[b, m, c] = sum_n K[b, m, n] * V[b, c, n]. K comes from the normalized
// buffer and V from the projection output, value_offset planes into each batch.
extern "C" __global__ void key_value(
    float* out, const float* normalized, const float* projected, int key_channels, int value_channels,
    int batch_planes, int value_offset, int tokens, int total
) {
    const int index = grid_index();
    if (index >= total) return;
    const int channel = index % value_channels;
    const int key = index / value_channels % key_channels;
    const int batch = index / value_channels / key_channels;
    const float* k = normalized + (size_t(batch) * 2 * key_channels + key_channels + key) * tokens;
    const float* v = projected + (size_t(batch) * batch_planes + value_offset + channel) * tokens;
    float sum = 0;
    for (int n = 0; n < tokens; ++n) sum = fmaf(k[n], v[n], sum);
    out[index] = sum;
}

// (value_sum + Q^T matrix) * 1 / (tokens + Q^T (sum K + eps)), in NCHW order.
extern "C" __global__ void attention_output(
    float* out, const float* normalized, const float* matrix, const float* key_sum, const float* value_sum,
    int key_channels, int value_channels, int tokens, int total
) {
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
    out[index] = __fmul_rn(numerator, reciprocal);
}

// PixelUnshuffle(2): out[b, 4c + 2i + j, y, x] = in[b, c, 2y + i, 2x + j].
extern "C" __global__ void unshuffle(float* out, const float* in, int channels, int height, int width, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int x = index % width, y = index / width % height;
    const int plane = index / width / height;
    const int sub = plane % 4, channel = plane / 4 % channels, batch = plane / 4 / channels;
    out[index] = in[((size_t(batch) * channels + channel) * 2 * height + 2 * y + sub / 2) * 2 * width
        + 2 * x + sub % 2];
}

// cat(PixelShuffle(2)(up), skip): both halves have half channels at height x width.
extern "C" __global__ void up_concat(
    float* out, const float* up, const float* skip, int half, int height, int width, int total
) {
    const int index = grid_index();
    if (index >= total) return;
    const int x = index % width, y = index / width % height;
    const int channel = index / width / height % (2 * half);
    const int batch = index / width / height / (2 * half);
    if (channel < half) {
        const int source = channel * 4 + (y % 2) * 2 + x % 2;
        out[index] = up[((size_t(batch) * 4 * half + source) * (height / 2) + y / 2) * (width / 2) + x / 2];
    } else {
        out[index] = skip[((size_t(batch) * half + channel - half) * height + y) * width + x];
    }
}

extern "C" __global__ void add_bias(float* out, const float* bias, int channels, int pixels, int total) {
    const int index = grid_index();
    if (index >= total) return;
    out[index] = __fadd_rn(out[index], bias[index / pixels % channels]);
}

extern "C" __global__ void add(float* out, const float* a, const float* b, int total) {
    const int index = grid_index();
    if (index >= total) return;
    out[index] = __fadd_rn(a[index], b[index]);
}
