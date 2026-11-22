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



// ---------------------------------------------------------------------------
// Native session kernels. Features are NHWC FP16. Register arithmetic, small
// parameters, pooling and the linear attention are FP32. The build compiles
// this family without fast math.

__device__ __forceinline__ int grid_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

extern "C" __global__ void weights_to_half(half* out, const float* in, int total) {
    const int index = grid_index();
    if (index < total) out[index] = __float2half_rn(in[index]);
}

__device__ __forceinline__ float load(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float load(const half* p, size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ void store(float* p, size_t i, float v) { p[i] = v; }
__device__ __forceinline__ void store(half* p, size_t i, float v) { p[i] = __float2half_rn(v); }

__device__ __forceinline__ void load8(const half* p, float* out) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const half2* pairs = reinterpret_cast<const half2*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 value = __half22float2(pairs[i]);
        out[2 * i] = value.x;
        out[2 * i + 1] = value.y;
    }
}

__device__ __forceinline__ void store8(half* p, const float* in) {
    uint4 raw;
    half2* pairs = reinterpret_cast<half2*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) pairs[i] = __floats2half2_rn(in[2 * i], in[2 * i + 1]);
    *reinterpret_cast<uint4*>(p) = raw;
}

// x * tanh(softplus(x)) = x * n / (n + 2) with n = e^x (e^x + 2). Above 15 the
// ratio rounds to one in FP32, and the guard keeps n finite.
__device__ __forceinline__ float mish(float x) {
    if (x > 15.f) return x;
    const float e = expf(x);
    const float n = e * (e + 2.f);
    return x * n / (n + 2.f);
}

__device__ __forceinline__ float group_sum(float value, int lanes) {
    for (int delta = lanes / 2; delta; delta /= 2) value += __shfl_xor_sync(0xffffffffu, value, delta);
    return value;
}

__device__ __forceinline__ int reflect(int value, int size, int padded) {
    if (value < size) return value;
    return padded - size < size ? 2 * size - 2 - value : size - 1;
}

// The 3x3 input convolution on the reflection-padded RGB input, its bias, and
// the first RMSNorm. One thread computes all channels of one padded pixel.
template<class In>
__device__ void input_conv(half* out_x, half* out_norm, const In* in, const float* weight, const float* bias,
                           const float* scale, const float* offset, int channels, int h, int w, int ph, int pw,
                           float inverse_root, int total) {
    __shared__ float kernel[64 * 27];
    for (int i = threadIdx.x; i < channels * 27; i += blockDim.x) kernel[i] = weight[i];
    __syncthreads();
    const int index = grid_index();
    if (index >= total) return;
    const int x = index % pw, y = index / pw % ph, n = index / (pw * ph);
    float taps[27];
#pragma unroll
    for (int c = 0; c < 3; ++c)
#pragma unroll
        for (int ky = 0; ky < 3; ++ky)
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                const int py = y + ky - 1, px = x + kx - 1;
                float value = 0;
                if (py >= 0 && py < ph && px >= 0 && px < pw)
                    value = load(in, ((size_t(n) * 3 + c) * h + reflect(py, h, ph)) * w + reflect(px, w, pw));
                taps[(c * 3 + ky) * 3 + kx] = value;
            }
    float values[64];
    float sum = 0;
#pragma unroll
    for (int c = 0; c < 64; ++c) {
        if (c < channels) {
            float acc = 0;
#pragma unroll
            for (int t = 0; t < 27; ++t) acc = fmaf(kernel[c * 27 + t], taps[t], acc);
            values[c] = acc + bias[c];
            sum = fmaf(values[c], values[c], sum);
        }
    }
    const float inverse = 1.f / (sqrtf(sum) * inverse_root + 1e-6f);
    const size_t at = size_t(index) * channels;
#pragma unroll
    for (int c = 0; c < 64; c += 8) {
        if (c < channels) {
            float normalized[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) normalized[k] = fmaf(scale[c + k], values[c + k] * inverse, offset[c + k]);
            store8(out_x + at + c, values + c);
            if (out_norm) store8(out_norm + at + c, normalized);
        }
    }
}

// The 3x3 output convolution to RGB, its bias, and the RGB residual, written as
// the cropped NCHW output. One thread computes the three channels of one pixel.
template<class Io>
__device__ void output_conv(Io* out, const half* features, const float* weight, const float* bias, const Io* in,
                            int channels, int h, int w, int ph, int pw, int total) {
    __shared__ float kernel[3 * 64 * 9];
    for (int i = threadIdx.x; i < 3 * channels * 9; i += blockDim.x) kernel[i] = weight[i];
    __syncthreads();
    const int index = grid_index();
    if (index >= total) return;
    const int x = index % w, y = index / w % h, n = index / (w * h);
    float acc[3] = {0, 0, 0};
    for (int ky = 0; ky < 3; ++ky) {
        const int py = y + ky - 1;
        if (py < 0 || py >= ph) continue;
        for (int kx = 0; kx < 3; ++kx) {
            const int px = x + kx - 1;
            if (px < 0 || px >= pw) continue;
            const half* source = features + ((size_t(n) * ph + py) * pw + px) * channels;
            for (int c0 = 0; c0 < channels; c0 += 8) {
                float v[8];
                load8(source + c0, v);
#pragma unroll
                for (int k = 0; k < 8; ++k)
#pragma unroll
                    for (int o = 0; o < 3; ++o)
                        acc[o] = fmaf(kernel[(o * channels + c0 + k) * 9 + ky * 3 + kx], v[k], acc[o]);
            }
        }
    }
#pragma unroll
    for (int o = 0; o < 3; ++o) {
        const size_t at = ((size_t(n) * 3 + o) * h + y) * w + x;
        store(out, at, acc[o] + bias[o] + load(in, at));
    }
}

// Per-pixel channel work in NHWC, with an optional RMSNorm of the result.
// A group of lanes (a power of two up to 32) covers one pixel; each lane holds
// up to four vectors of eight channels. The source x is:
//   Mode 0: a (+ vec as a channel bias)
//   Mode 1: PixelUnshuffle(2) of a, which has channels / 4 channels at 2x size
//   Mode 2: mish(a), or mish(a) * vec + b
//   Mode 3: a * vec[image, channel] + b
template<int Mode>
__device__ void pixel_norm(half* out_x, half* out_norm, const half* a, const half* b, const float* vec,
                           const float* scale, const float* offset, int channels, int lanes, int pixels, int width,
                           float inverse_root, int total) {
    const int lane = threadIdx.x & (lanes - 1);
    const int pixel = grid_index() / lanes;
    const bool active = pixel < total;
    const int vectors = channels / (8 * lanes);
    float x[32];
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (j < vectors && active) {
            const int c = 8 * (lane + j * lanes);
            float* v = x + 8 * j;
            const size_t at = size_t(pixel) * channels + c;
            if (Mode == 0) {
                load8(a + at, v);
                if (vec)
#pragma unroll
                    for (int k = 0; k < 8; ++k) v[k] += vec[c + k];
            } else if (Mode == 1) {
                const int n = pixel / pixels, rest = pixel % pixels, y = rest / width, xx = rest % width;
                const int source_channels = channels / 4;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const size_t hi = (size_t(n) * 4 * pixels + size_t(2 * y + s / 2) * 2 * width + 2 * xx + s % 2)
                        * source_channels + c / 4;
                    const float2 pair = __half22float2(*reinterpret_cast<const half2*>(a + hi));
                    v[s] = pair.x;
                    v[4 + s] = pair.y;
                }
            } else if (Mode == 2) {
                load8(a + at, v);
                if (vec) {
                    float shortcut[8];
                    load8(b + at, shortcut);
#pragma unroll
                    for (int k = 0; k < 8; ++k) v[k] = fmaf(mish(v[k]), vec[c + k], shortcut[k]);
                } else {
#pragma unroll
                    for (int k = 0; k < 8; ++k) v[k] = mish(v[k]);
                }
            } else {
                float shortcut[8];
                load8(a + at, v);
                load8(b + at, shortcut);
                const float* multiplier = vec + size_t(pixel / pixels) * channels + c;
#pragma unroll
                for (int k = 0; k < 8; ++k) v[k] = fmaf(v[k], multiplier[k], shortcut[k]);
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) sum = fmaf(v[k], v[k], sum);
        }
    }
    sum = group_sum(sum, lanes);
    if (!active) return;
    const float inverse = 1.f / (sqrtf(sum) * inverse_root + 1e-6f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (j < vectors) {
            const int c = 8 * (lane + j * lanes);
            const size_t at = size_t(pixel) * channels + c;
            if (out_x) store8(out_x + at, x + 8 * j);
            if (out_norm) {
                float normalized[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) normalized[k] = fmaf(scale[c + k], x[8 * j + k] * inverse, offset[c + k]);
                store8(out_norm + at, normalized);
            }
        }
    }
}

// Grouped 3x3 convolution (two inputs and two outputs per group) on the 2C
// expansion, then SimpleGate. A thread owns one channel pair of both
// convolutions, keeps their 72 weights in registers, and computes a column of
// gate_rows pixels, reusing each loaded input row for up to three outputs. A
// block covers 256 / (C / 2) columns and writes its per-channel sums of the
// gated output, reduced in a fixed order, for the average pool.
constexpr int gate_rows = 4;

extern "C" __global__ void local_gate(half* out, float* partial, const half* in, const float* weight,
                                      const float* bias, int channels, int height, int width) {
    __shared__ float2 sums[256];
    const int pairs = channels / 2, columns = blockDim.x / pairs;
    const int p = threadIdx.x % pairs, column = threadIdx.x / pairs;
    const int x = blockIdx.x * columns + column, y0 = blockIdx.y * gate_rows, n = blockIdx.z;
    const int stride = 2 * channels;
    float w1[2][18], w2[2][18];
#pragma unroll
    for (int o = 0; o < 2; ++o)
#pragma unroll
        for (int t = 0; t < 18; ++t) {
            w1[o][t] = weight[(2 * p + o) * 18 + t];
            w2[o][t] = weight[(channels + 2 * p + o) * 18 + t];
        }
    float a1[gate_rows][2], a2[gate_rows][2];
#pragma unroll
    for (int r = 0; r < gate_rows; ++r) a1[r][0] = a1[r][1] = a2[r][0] = a2[r][1] = 0;
    const half* image = in + size_t(n) * height * width * stride;
    if (x < width) {
#pragma unroll
        for (int i = 0; i < gate_rows + 2; ++i) {
            const int yy = y0 + i - 1;
            if (yy < 0 || yy >= height) continue;
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                const int xx = x + kx - 1;
                if (xx < 0 || xx >= width) continue;
                const half* at = image + (size_t(yy) * width + xx) * stride;
                const float2 a = __half22float2(*reinterpret_cast<const half2*>(at + 2 * p));
                const float2 b = __half22float2(*reinterpret_cast<const half2*>(at + channels + 2 * p));
#pragma unroll
                for (int ky = 0; ky < 3; ++ky) {
                    const int r = i - ky;
                    if (r < 0 || r >= gate_rows) continue;
                    const int tap = ky * 3 + kx;
#pragma unroll
                    for (int o = 0; o < 2; ++o) {
                        a1[r][o] = fmaf(w1[o][9 + tap], a.y, fmaf(w1[o][tap], a.x, a1[r][o]));
                        a2[r][o] = fmaf(w2[o][9 + tap], b.y, fmaf(w2[o][tap], b.x, a2[r][o]));
                    }
                }
            }
        }
    }
    const float b10 = bias[2 * p], b11 = bias[2 * p + 1];
    const float b20 = bias[channels + 2 * p], b21 = bias[channels + 2 * p + 1];
    float2 total = make_float2(0.f, 0.f);
#pragma unroll
    for (int r = 0; r < gate_rows; ++r) {
        const int y = y0 + r;
        if (x >= width || y >= height) continue;
        const float g0 = (a1[r][0] + b10) * (a2[r][0] + b20);
        const float g1 = (a1[r][1] + b11) * (a2[r][1] + b21);
        const half2 value = __floats2half2_rn(g0, g1);
        *reinterpret_cast<half2*>(out + ((size_t(n) * height + y) * width + x) * channels + 2 * p) = value;
        const float2 rounded = __half22float2(value);
        total.x += rounded.x;
        total.y += rounded.y;
    }
    sums[threadIdx.x] = total;
    __syncthreads();
    if (threadIdx.x < pairs) {
        float2 s = make_float2(0.f, 0.f);
        for (int c = 0; c < columns; ++c) {
            s.x += sums[c * pairs + threadIdx.x].x;
            s.y += sums[c * pairs + threadIdx.x].y;
        }
        const size_t tile = size_t(blockIdx.y) * gridDim.x + blockIdx.x;
        const size_t tiles = size_t(gridDim.x) * gridDim.y;
        *reinterpret_cast<float2*>(partial + ((size_t(n) * tiles + tile) * channels + 2 * threadIdx.x)) = s;
    }
}

// Average pool from the local_gate partial sums, the channel-attention 1x1
// projection, and gamma0: multiplier[n, o] = (W mean + b)[o] * gamma0[o].
// Each block recomputes the means and produces eight outputs.
extern "C" __global__ void pool_project(float* multiplier, const float* partial, const float* weight,
                                        const float* bias, const float* gamma, int channels, int tiles,
                                        int pixels) {
    __shared__ float mean[256];
    const int n = blockIdx.x;
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        float s = 0;
        for (int t = 0; t < tiles; ++t) s += partial[(size_t(n) * tiles + t) * channels + c];
        mean[c] = s / float(pixels);
    }
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.y * (blockDim.x / 32) + warp;
    if (o >= channels) return;
    float acc = 0;
    for (int i = lane; i < channels; i += 32) acc = fmaf(weight[size_t(o) * channels + i], mean[i], acc);
    acc = group_sum(acc, 32);
    if (!lane) multiplier[size_t(n) * channels + o] = (acc + bias[o]) * gamma[o];
}

// e = mish(g) * cat(identity, mixed) from the 3C fc1 output
// [g (1.5C) | identity (0.5C) | conv (C)]. mixed keeps the first C - 3C/8 conv
// channels and applies 3x3, 1x11 and 11x1 depthwise kernels to the next three
// groups of C/8. One thread computes eight channels of one pixel. Lanes of a
// warp cover consecutive channel vectors of the same pixel, so their loads
// coalesce, and each thread loops only over its vector's single kernel kind.
extern "C" __global__ void inception_gate(half* out, const half* projected, const float* square,
                                          const float* square_bias, const float* row, const float* row_bias,
                                          const float* column, const float* column_bias, int channels, int height,
                                          int width, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int hidden = 3 * channels / 2, vectors = hidden / 8, stride = 2 * hidden;
    const int ch = 8 * (index % vectors);
    const int pixel = index / vectors;
    const half* here = projected + size_t(pixel) * stride;
    float g[8], value[8];
    load8(here + ch, g);
    const int identity_gate = hidden - channels;
    const int conv = 2 * hidden - channels;
    const int branch = channels / 8, identity = channels - 3 * branch;
    const int j0 = ch - identity_gate;
    if (ch < identity_gate) {
        load8(here + hidden + ch, value);
    } else if (j0 + 8 <= identity) {
        load8(here + conv + j0, value);
    } else {
        const int pixels = height * width, rest = pixel % pixels;
        const int y = rest / width, x = rest % width;
        const half* image = projected + size_t(pixel - rest) * stride + conv + j0;
        const int first = (j0 - identity) / branch, last = (j0 + 7 - identity) / branch;
        float t[8];
        if (j0 >= identity && first == last) {
            // All eight channels use one kernel kind.
            const int group = j0 - identity - first * branch;
            if (first == 0) {
#pragma unroll
                for (int k = 0; k < 8; ++k) value[k] = square_bias[group + k];
                for (int ky = 0; ky < 3; ++ky) {
                    const int yy = y + ky - 1;
                    if (yy < 0 || yy >= height) continue;
                    for (int kx = 0; kx < 3; ++kx) {
                        const int xx = x + kx - 1;
                        if (xx < 0 || xx >= width) continue;
                        load8(image + (size_t(yy) * width + xx) * stride, t);
#pragma unroll
                        for (int k = 0; k < 8; ++k)
                            value[k] = fmaf(square[(group + k) * 9 + ky * 3 + kx], t[k], value[k]);
                    }
                }
            } else {
                const float* kernel = first == 1 ? row : column;
                const float* bias = first == 1 ? row_bias : column_bias;
#pragma unroll
                for (int k = 0; k < 8; ++k) value[k] = bias[group + k];
                for (int s = -5; s <= 5; ++s) {
                    const int yy = first == 1 ? y : y + s, xx = first == 1 ? x + s : x;
                    if (yy < 0 || yy >= height || xx < 0 || xx >= width) continue;
                    load8(image + (size_t(yy) * width + xx) * stride, t);
#pragma unroll
                    for (int k = 0; k < 8; ++k) value[k] = fmaf(kernel[(group + k) * 11 + s + 5], t[k], value[k]);
                }
            }
        } else {
            // Narrow blocks: branches of four channels share a vector, possibly
            // with identity channels. Handle each channel's kind separately.
            load8(here + conv + j0, value);
            int kind[8], index[8], kinds = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int d = j0 + k - identity;
                kind[k] = d < 0 ? -1 : d / branch;
                index[k] = d < 0 ? 0 : d - kind[k] * branch;
                if (kind[k] >= 0) {
                    kinds |= 1 << kind[k];
                    value[k] = kind[k] == 0 ? square_bias[index[k]]
                        : kind[k] == 1 ? row_bias[index[k]] : column_bias[index[k]];
                }
            }
            if (kinds & 1) {
                for (int ky = 0; ky < 3; ++ky) {
                    const int yy = y + ky - 1;
                    if (yy < 0 || yy >= height) continue;
                    for (int kx = 0; kx < 3; ++kx) {
                        const int xx = x + kx - 1;
                        if (xx < 0 || xx >= width) continue;
                        load8(image + (size_t(yy) * width + xx) * stride, t);
#pragma unroll
                        for (int k = 0; k < 8; ++k)
                            if (kind[k] == 0) value[k] = fmaf(square[index[k] * 9 + ky * 3 + kx], t[k], value[k]);
                    }
                }
            }
            if (kinds & 6) {
                for (int s = -5; s <= 5; ++s) {
                    const bool in_row = x + s >= 0 && x + s < width, in_column = y + s >= 0 && y + s < height;
                    if ((kinds & 2) && in_row) {
                        load8(image + (size_t(y) * width + x + s) * stride, t);
#pragma unroll
                        for (int k = 0; k < 8; ++k)
                            if (kind[k] == 1) value[k] = fmaf(row[index[k] * 11 + s + 5], t[k], value[k]);
                    }
                    if ((kinds & 4) && in_column) {
                        load8(image + (size_t(y + s) * width + x) * stride, t);
#pragma unroll
                        for (int k = 0; k < 8; ++k)
                            if (kind[k] == 2) value[k] = fmaf(column[index[k] * 11 + s + 5], t[k], value[k]);
                    }
                }
            }
        }
    }
    float result[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) result[k] = mish(g[k]) * value[k];
    store8(out + size_t(pixel) * hidden + ch, result);
}

// Linear attention, part one. qkv is FP32 [image, token, 2K + V]. A block
// normalizes Q and K of up to 64 tokens of one image and accumulates K^T V for
// 64 value channels; four threads per value channel split the keys. Blocks of
// the first value slice also normalize Q in place and sum K. With one chunk
// per image the sums are final and include epsilon; otherwise
// attention_reduce adds the chunks.
extern "C" __global__ void attention_prepare(float* qkv, float* key_sum, float* value_sum, float* matrix,
                                             int keys, int values, int tokens, float eps, int final_sums) {
    __shared__ float normalized[64][65];
    __shared__ float value_tile[64][65];
    const int n = blockIdx.y, chunk = blockIdx.x, slice = blockIdx.z, t0 = chunk * 64;
    const int count = min(64, tokens - t0);
    const int stride = 2 * keys + values;
    float* base = qkv + (size_t(n) * tokens + t0) * stride;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int t = warp; t < 64; t += blockDim.x / 32) {
        if (t >= count) {
            normalized[t][lane] = normalized[t][lane + 32] = 0.f;
            continue;
        }
        float* row = base + size_t(t) * stride;
        const float ka = lane < keys ? row[keys + lane] : 0.f, kb = lane + 32 < keys ? row[keys + lane + 32] : 0.f;
        const float k_inverse = 1.f / sqrtf(group_sum(fmaf(kb, kb, ka * ka), 32));
        normalized[t][lane] = ka * k_inverse;
        normalized[t][lane + 32] = kb * k_inverse;
        if (!slice) {
            const float qa = lane < keys ? row[lane] : 0.f, qb = lane + 32 < keys ? row[lane + 32] : 0.f;
            const float q_inverse = 1.f / sqrtf(group_sum(fmaf(qb, qb, qa * qa), 32));
            if (lane < keys) row[lane] = qa * q_inverse;
            if (lane + 32 < keys) row[lane + 32] = qb * q_inverse;
        }
    }
    const int c0 = slice * 64;
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
        const int t = i / 64, c = i % 64;
        value_tile[t][c] = t < count && c0 + c < values ? base[size_t(t) * stride + 2 * keys + c0 + c] : 0.f;
    }
    __syncthreads();
    const size_t slot = final_sums ? size_t(n) : size_t(n) * gridDim.x + chunk;
    if (!slice && threadIdx.x < keys) {
        float s = 0;
        for (int t = 0; t < count; ++t) s += normalized[t][threadIdx.x];
        key_sum[slot * keys + threadIdx.x] = final_sums ? s + eps : s;
    }
    const int c = threadIdx.x % 64, quarter = threadIdx.x / 64;
    if (c0 + c >= values) return;
    if (!quarter) {
        float total = 0;
        for (int t = 0; t < count; ++t) total += value_tile[t][c];
        value_sum[slot * values + c0 + c] = total;
    }
    float acc[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) acc[m] = 0;
    for (int t = 0; t < count; ++t) {
        const float v = value_tile[t][c];
#pragma unroll
        for (int m = 0; m < 16; ++m) acc[m] = fmaf(normalized[t][quarter * 16 + m], v, acc[m]);
    }
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const int key = quarter * 16 + m;
        if (key < keys) matrix[(slot * keys + key) * values + c0 + c] = acc[m];
    }
}

// Linear attention, part two, for images with more than 64 tokens.
extern "C" __global__ void attention_reduce(float* key_sum, float* value_sum, float* matrix, const float* key_part,
                                            const float* value_part, const float* matrix_part, int keys, int values,
                                            int chunks, float eps) {
    const int n = blockIdx.x;
    for (int i = threadIdx.x; i < keys; i += blockDim.x) {
        float s = 0;
        for (int c = 0; c < chunks; ++c) s += key_part[(size_t(n) * chunks + c) * keys + i];
        key_sum[size_t(n) * keys + i] = s + eps;
    }
    for (int i = threadIdx.x; i < values; i += blockDim.x) {
        float s = 0;
        for (int c = 0; c < chunks; ++c) s += value_part[(size_t(n) * chunks + c) * values + i];
        value_sum[size_t(n) * values + i] = s;
    }
    const int size = keys * values;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float s = 0;
        for (int c = 0; c < chunks; ++c) s += matrix_part[(size_t(n) * chunks + c) * size + i];
        matrix[size_t(n) * size + i] = s;
    }
}

// Linear attention, part three, fused with the latent gate:
// e = mish(g) * cat(identity, (vsum + Q M) / (tokens + Q ksum)).
extern "C" __global__ void attention_gate(half* out, const half* projected, const float* qkv, const float* key_sum,
                                          const float* value_sum, const float* matrix, int keys, int values,
                                          int tokens, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int hidden = 3 * values / 2, vectors = hidden / 8;
    const int ch = 8 * (index % vectors);
    const int token = index / vectors;
    const half* here = projected + size_t(token) * 2 * hidden;
    float g[8], value[8];
    load8(here + ch, g);
    const int identity = hidden - values;
    if (ch < identity) {
        load8(here + hidden + ch, value);
    } else {
        const int c0 = ch - identity, n = token / tokens;
        const float* q = qkv + size_t(token) * (2 * keys + values);
        const float* k = key_sum + size_t(n) * keys;
        const float* m = matrix + size_t(n) * keys * values + c0;
        float contraction = 0, product[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < keys; ++i) {
            const float query = q[i];
            contraction = fmaf(query, k[i], contraction);
            const float4 lo = *reinterpret_cast<const float4*>(m + size_t(i) * values);
            const float4 hi = *reinterpret_cast<const float4*>(m + size_t(i) * values + 4);
            product[0] = fmaf(query, lo.x, product[0]);
            product[1] = fmaf(query, lo.y, product[1]);
            product[2] = fmaf(query, lo.z, product[2]);
            product[3] = fmaf(query, lo.w, product[3]);
            product[4] = fmaf(query, hi.x, product[4]);
            product[5] = fmaf(query, hi.y, product[5]);
            product[6] = fmaf(query, hi.z, product[6]);
            product[7] = fmaf(query, hi.w, product[7]);
        }
        const float reciprocal = 1.f / (float(tokens) + contraction);
        const float* sums = value_sum + size_t(n) * values + c0;
#pragma unroll
        for (int k8 = 0; k8 < 8; ++k8) value[k8] = (sums[k8] + product[k8]) * reciprocal;
    }
    float result[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) result[k] = mish(g[k]) * value[k];
    store8(out + size_t(token) * hidden + ch, result);
}

// cat(PixelShuffle(2)(up), skip) in NHWC at the higher resolution. The up
// convolution's output channels are stored subpixel-major, so each subpixel's
// channels are contiguous.
extern "C" __global__ void shuffle_concat(half* out, const half* up, const half* skip, int half_channels,
                                          int height, int width, int total) {
    const int index = grid_index();
    if (index >= total) return;
    const int vectors = half_channels / 4;
    const int c0 = 8 * (index % vectors);
    const int pixel = index / vectors;
    const int pixels = height * width, n = pixel / pixels, rest = pixel % pixels;
    const int y = rest / width, x = rest % width;
    const half* source;
    if (c0 < half_channels) {
        const int s = (y % 2) * 2 + x % 2;
        source = up + ((size_t(n) * (height / 2) + y / 2) * (width / 2) + x / 2) * 4 * half_channels
            + s * half_channels + c0;
    } else {
        source = skip + size_t(pixel) * half_channels + c0 - half_channels;
    }
    *reinterpret_cast<uint4*>(out + size_t(pixel) * 2 * half_channels + c0) = *reinterpret_cast<const uint4*>(source);
}

extern "C" __global__ void input_conv_f32(half* x, half* norm, const float* in, const float* w, const float* b,
                                          const float* s, const float* o, int c, int h, int width, int ph, int pw,
                                          float r, int t) {
    input_conv(x, norm, in, w, b, s, o, c, h, width, ph, pw, r, t);
}
extern "C" __global__ void input_conv_f16(half* x, half* norm, const half* in, const float* w, const float* b,
                                          const float* s, const float* o, int c, int h, int width, int ph, int pw,
                                          float r, int t) {
    input_conv(x, norm, in, w, b, s, o, c, h, width, ph, pw, r, t);
}
extern "C" __global__ void output_conv_f32(float* out, const half* f, const float* w, const float* b,
                                           const float* in, int c, int h, int width, int ph, int pw, int t) {
    output_conv(out, f, w, b, in, c, h, width, ph, pw, t);
}
extern "C" __global__ void output_conv_f16(half* out, const half* f, const float* w, const float* b, const half* in,
                                           int c, int h, int width, int ph, int pw, int t) {
    output_conv(out, f, w, b, in, c, h, width, ph, pw, t);
}

#define UMZI_PIXEL_NORM(M) \
extern "C" __global__ void pixel_norm_##M(half* x, half* norm, const half* a, const half* b, const float* v, \
    const float* s, const float* o, int c, int lanes, int p, int w, float r, int t) \
    { pixel_norm<M>(x, norm, a, b, v, s, o, c, lanes, p, w, r, t); }
UMZI_PIXEL_NORM(0)
UMZI_PIXEL_NORM(1)
UMZI_PIXEL_NORM(2)
UMZI_PIXEL_NORM(3)
