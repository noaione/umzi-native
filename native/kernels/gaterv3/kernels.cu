#include <cuda_fp16.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

__device__ float read(const half* values, int index) { return __half2float(values[index]); }

__device__ float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ float mish(float x) {
    // tanh(log(1+exp(x))) expressed in terms of a bounded exponential.
    float e = __expf(-fabsf(x)), square = e * e;
    float numerator = 2 * e + (x >= 0 ? 1.0f : square);
    return x * numerator / (numerator + (x >= 0 ? 2 * square : 2.0f));
}

__device__ float warp_sum(float v) {
    for (int delta = 16; delta; delta /= 2) v += __shfl_xor_sync(0xffffffff, v, delta);
    return v;
}

extern "C" __global__ void pack(half* out, const float* in, int pixels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int channel = i % 8, pixel = i / 8;
    int image = pixel / pixels, p = pixel % pixels;
    out[i] = channel < 3 ? in[(image * 3 + channel) * pixels + p] : 0.0f;
}

extern "C" __global__ void rgb(float* out, const half* residual, const float* in,
                                const float* gamma, int pixels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int image_channel = i / pixels, p = i % pixels;
    int c = image_channel % 3, image = image_channel / 3;
    out[i] = fmaf(gamma[c], in[i], read(residual, (image * pixels + p) * 8 + c));
}

template<int C, class ReadPair>
__device__ void normalize_from(half2* out, const half2* scale, const half2* offset,
                               int pixels, ReadPair read_pair) {
    constexpr int lanes = C == 32 ? 16 : 32;
    constexpr int pairs = C / 2, steps = pairs / lanes;
    int lane = threadIdx.x % lanes;
    int pixel = blockIdx.x * (blockDim.x / lanes) + threadIdx.x / lanes;
    if (pixel >= pixels) return;
    float2 values[steps];
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < steps; ++i) {
        values[i] = __half22float2(read_pair(pixel, i * lanes + lane));
        sum = fmaf(values[i].y, values[i].y, fmaf(values[i].x, values[i].x, sum));
    }
    #pragma unroll
    for (int delta = lanes / 2; delta; delta /= 2) sum += __shfl_xor_sync(0xffffffff, sum, delta);
    float inverse = 1.0f / (sqrtf(sum / C) + 1.0e-6f);
    #pragma unroll
    for (int i = 0; i < steps; ++i) {
        int c = i * lanes + lane;
        float2 s = __half22float2(scale[c]), b = __half22float2(offset[c]);
        out[pixel * pairs + c] = __floats2half2_rn(fmaf(values[i].x * inverse, s.x, b.x),
                                                   fmaf(values[i].y * inverse, s.y, b.y));
    }
}

#define NORMALIZE(C) \
extern "C" __global__ void rms##C(half2* y, const half2* x, const half2* s, \
                                  const half2* b, int pixels) { \
    normalize_from<C>(y, s, b, pixels, [=](int p, int c) { return x[p * (C / 2) + c]; }); \
}
NORMALIZE(32)
NORMALIZE(64)
NORMALIZE(128)
NORMALIZE(256)
NORMALIZE(512)

extern "C" __global__ void bias(half* out, const half* b, int channels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = read(out, i) + read(b, i % channels);
}

template<int channels, int square>
__device__ __forceinline__ half2 local_gate_value(const half2* in, const half2* weight,
                                                  const half2* bias, int pixel, int g, int h = square, int w = square) {
    constexpr int groups = channels / 2;
    int x = pixel % w, y = pixel / w % h;
    float2 a = __half22float2(bias[g]), b = __half22float2(bias[g + groups]);
    #pragma unroll
    for (int ky = -1; ky <= 1; ++ky) {
        #pragma unroll
        for (int kx = -1; kx <= 1; ++kx) {
            if (y + ky < 0 || y + ky >= h || x + kx < 0 || x + kx >= w) continue;
            int pos = (pixel + ky * w + kx) * channels;
            int k = ((ky + 1) * 3 + kx + 1) * 2 * channels;
            float2 left = __half22float2(in[pos + g]);
            float2 right = __half22float2(in[pos + g + groups]);
            float2 wa0 = __half22float2(weight[k + g]);
            float2 wa1 = __half22float2(weight[k + channels + g]);
            float2 wb0 = __half22float2(weight[k + g + groups]);
            float2 wb1 = __half22float2(weight[k + channels + g + groups]);
            a.x = fmaf(left.y, wa1.x, fmaf(left.x, wa0.x, a.x));
            a.y = fmaf(left.y, wa1.y, fmaf(left.x, wa0.y, a.y));
            b.x = fmaf(right.y, wb1.x, fmaf(right.x, wb0.x, b.x));
            b.y = fmaf(right.y, wb1.y, fmaf(right.x, wb0.y, b.y));
        }
    }
    return __floats2half2_rn(a.x * b.x, a.y * b.y);
}

template<int channels, int square>
__device__ void local_gate(half2* out, const half2* in, const half2* weight,
                           const half2* bias, int total, int h = square, int w = square) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        out[i] = local_gate_value<channels, square>(in, weight, bias, i / (channels / 2), i % (channels / 2), h, w);
}

template<bool Linear>
__device__ void mean_partial_spatial(float* out, const half* in, int pixels,
                                         int channels, int chunks) {
    // Each warp reads 32 consecutive channels. Warps partition the spatial sum.
    __shared__ float partial[8][32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int blocks = channels / 32;
    int row = Linear ? blockIdx.x / blocks : blockIdx.y;
    int c = (Linear ? blockIdx.x % blocks : blockIdx.x) * 32 + lane;
    int n = row / chunks;
    int start = row % chunks * 256;
    float acc = 0.0f;
    for (int p = start + warp; p < pixels && p < start + 256; p += 8)
        acc += read(in, (n * pixels + p) * channels + c);
    partial[warp][lane] = acc;
    __syncthreads();
    if (!warp) {
        #pragma unroll
        for (int i = 1; i < 8; ++i) acc += partial[i][lane];
        out[row * channels + c] = acc;
    }
}
extern "C" __global__ void mean_partial(float* y, const half* x, int pixels, int channels, int chunks) {
    mean_partial_spatial<false>(y, x, pixels, channels, chunks);
}
extern "C" __global__ void mean_partial_dynamic(float* y, const half* x, int pixels, int channels, int chunks) {
    mean_partial_spatial<true>(y, x, pixels, channels, chunks);
}

extern "C" __global__ void mean_project(half* out, const float* partial, const half* weight,
                                        const half* bias, int pixels, int channels, int chunks) {
    __shared__ half average[256];
    int c = threadIdx.x, lane = c % 32, warp = c / 32;
    int n = blockIdx.x;
    if (c < channels) {
        float sum = 0.0f;
        for (int j = 0; j < chunks; ++j) sum += partial[(n * chunks + j) * channels + c];
        average[c] = sum / pixels;
    }
    __syncthreads();
    for (int output = warp; output < channels; output += 8) {
        float dot = 0.0f;
        for (int i = lane; i < channels; i += 32)
            dot = fmaf(read(average, i), read(weight, output * channels + i), dot);
        dot = warp_sum(dot);
        if (lane == 0) out[n * channels + output] = dot + read(bias, output);
    }
}

extern "C" __global__ void local_skip(half2* out, const half2* gated, const half2* scale,
                                      const half2* skip, const half2* gamma,
                                      int pixels, int channels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float2 g = __half22float2(gated[i]), s = __half22float2(skip[i]);
    int c = i & (channels - 1);
    float2 a = __half22float2(scale[i / (pixels * channels) * channels + c]);
    float2 m = __half22float2(gamma[c]);
    out[i] = __floats2half2_rn(fmaf(g.x * a.x, m.x, s.x), fmaf(g.y * a.y, m.y, s.y));
}

template<int channels, int square>
__device__ void inception(half2* out, const half2* in, const half2* weight,
                          const half2* bias, int total, int h = square, int w = square) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int hidden = channels * 3 / 2, stride = channels * 3;
    int c = i % hidden, pixel = i / hidden, x = pixel % w, y = pixel / w % h;
    int band = channels / 8, branch = (c - (hidden - 3 * band)) / band;
    float2 v;
    if (c < hidden - 3 * band) {
        v = __half22float2(in[pixel * stride + hidden + c]);
    } else {
        int bc = c % band;
        v = __half22float2(bias[branch * band + bc]);
        if (branch == 0) {
            #pragma unroll
            for (int dy = -1; dy <= 1; ++dy) {
                #pragma unroll
                for (int dx = -1; dx <= 1; ++dx) {
                    if (y + dy < 0 || y + dy >= h || x + dx < 0 || x + dx >= w) continue;
                    int k = ((dy + 1) * 3 + dx + 1) * band + bc;
                    float2 a = __half22float2(in[(pixel + dy * w + dx) * stride + hidden + c]);
                    float2 b = __half22float2(weight[k]);
                    v.x = fmaf(a.x, b.x, v.x);
                    v.y = fmaf(a.y, b.y, v.y);
                }
            }
        } else {
            int offset = (branch == 1 ? 9 : 20) * band + bc;
            int coord = branch == 1 ? x : y;
            int step = branch == 1 ? 1 : w;
            #pragma unroll
            for (int j = -5; j <= 5; ++j) {
                if (coord + j < 0 || coord + j >= (branch == 1 ? w : h)) continue;
                float2 a = __half22float2(in[(pixel + j * step) * stride + hidden + c]);
                float2 b = __half22float2(weight[offset + (j + 5) * band]);
                v.x = fmaf(a.x, b.x, v.x);
                v.y = fmaf(a.y, b.y, v.y);
            }
        }
    }
    float2 g = __half22float2(in[pixel * stride + c]);
    out[i] = __floats2half2_rn(mish(g.x) * v.x, mish(g.y) * v.y);
}

#define SPATIAL(C, W) \
extern "C" __global__ void local_gate##C(half2* y, const half2* x, const half2* k, \
                                        const half2* b, int total) { local_gate<C, W>(y, x, k, b, total); } \
extern "C" __global__ void inception##C(half2* y, const half2* x, const half2* k, \
                                       const half2* b, int total) { inception<C / 2, W>(y, x, k, b, total); }
SPATIAL(32, 128)
SPATIAL(64, 64)
SPATIAL(128, 32)
SPATIAL(256, 16)

#define SPATIAL_DYNAMIC(C) \
extern "C" __global__ void local_gate##C##_dynamic(half2* y, const half2* x, const half2* k, \
    const half2* b, int h, int w, int total) { local_gate<C, 0>(y, x, k, b, total, h, w); } \
extern "C" __global__ void inception##C##_dynamic(half2* y, const half2* x, const half2* k, \
    const half2* b, int h, int w, int total) { inception<C / 2, 0>(y, x, k, b, total, h, w); }
SPATIAL_DYNAMIC(32)
SPATIAL_DYNAMIC(64)
SPATIAL_DYNAMIC(128)
SPATIAL_DYNAMIC(256)

extern "C" __global__ void global_skip(half2* out, const half2* in, const half2* skip,
                                       const half2* gamma, int channels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float2 a = __half22float2(in[i]);
    a.x = mish(a.x);
    a.y = mish(a.y);
    if (skip) {
        float2 g = __half22float2(gamma[i & (channels - 1)]), s = __half22float2(skip[i]);
        a.x = fmaf(a.x, g.x, s.x);
        a.y = fmaf(a.y, g.y, s.y);
    }
    out[i] = __float22half2_rn(a);
}

extern "C" __global__ void shuffle(half* out, const half* in, int h, int w, int c,
                                    int up, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int ch = i % c, pixel = i / c, x = pixel % w, y = pixel / w % h;
    int n = pixel / (w * h);
    if (up) {
        out[i] = in[((n * (h / 2) + y / 2) * (w / 2) + x / 2) * c * 4
                    + 4 * ch + (y % 2) * 2 + x % 2];
    } else {
        out[i] = in[((n * (2 * h) + y * 2 + ch % 4 / 2) * (2 * w) + x * 2 + ch % 2)
                    * (c / 4) + ch / 4];
    }
}

extern "C" __global__ void concat(uint4* out, const uint4* a, const uint4* b,
                                   const uint4* c, const uint4* d, int ca, int cb, int cc,
                                   int cd, int channels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int ch = i % channels, p = i / channels;
    if (ch < ca) out[i] = a[p * ca + ch];
    else if ((ch -= ca) < cb) out[i] = b[p * cb + ch];
    else if ((ch -= cb) < cc) out[i] = c[p * cc + ch];
    else out[i] = d[p * cd + ch - cc];
}

extern "C" __global__ void silu(half2* out, const half2* in, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float2 a = __half22float2(in[i]);
    out[i] = __floats2half2_rn(a.x * sigmoid(a.x), a.y * sigmoid(a.y));
}

extern "C" __global__ void span_gate(half2* out, const half2* in, const half2* skip, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float2 a = __half22float2(in[i]), b = __half22float2(skip[i]);
    out[i] = __floats2half2_rn((a.x + b.x) * (sigmoid(a.x) - 0.5f),
                              (a.y + b.y) * (sigmoid(a.y) - 0.5f));
}

template<int square>
__device__ void depthwise_spatial(half* out, const half* in, const half* weight,
                                     const half* bias, int stride, int total, int h = square, int w = square) {
    constexpr int channels = 1536;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = i % channels, pixel = i / channels;
    int x = pixel % w, y = pixel / w % h;
    float v = read(bias, c);
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (y + dy < 0 || y + dy >= h || x + dx < 0 || x + dx >= w) continue;
            v = fmaf(read(in, (pixel + dy * w + dx) * stride + stride - channels + c),
                     read(weight, ((dy + 1) * 3 + dx + 1) * channels + c), v);
        }
    }
    out[i] = v;
}
extern "C" __global__ void depthwise(half* y, const half* x, const half* k, const half* b,
                                  int stride, int total) {
    depthwise_spatial<8>(y, x, k, b, stride, total);
}
extern "C" __global__ void depthwise_dynamic(half* y, const half* x, const half* k, const half* b,
                                          int stride, int h, int w, int total) {
    depthwise_spatial<0>(y, x, k, b, stride, total, h, w);
}

__device__ void attention_softmax(half* p, const float* scores, int query, int lane) {
    float left = scores[query * 64 + lane] * 0.1767766953f;
    float right = scores[query * 64 + lane + 32] * 0.1767766953f;
    float max_value = fmaxf(left, right);
    for (int delta = 16; delta; delta /= 2)
        max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, delta));
    left = __expf(left - max_value);
    right = __expf(right - max_value);
    float sum = warp_sum(left + right);
    p[query * 64 + lane] = left / sum;
    p[query * 64 + lane + 32] = right / sum;
}

__device__ half warp_sum(half x) {
    for (int delta = 16; delta; delta /= 2) x = __hadd(x, __shfl_xor_sync(0xffffffff, x, delta));
    return x;
}

__device__ void attention_softmax(half* p, const half* scores, int query, int lane) {
    half scale = __float2half(0.1767766953f);
    half left = __hmul(scores[query * 64 + lane], scale);
    half right = __hmul(scores[query * 64 + lane + 32], scale);
    half max_value = __hmax(left, right);
    for (int delta = 16; delta; delta /= 2)
        max_value = __hmax(max_value, __shfl_xor_sync(0xffffffff, max_value, delta));
    half2 logits = __halves2half2(__hsub(left, max_value), __hsub(right, max_value));
    half2 exp = h2exp(logits);
    half sum = warp_sum(__hadd(__low2half(exp), __high2half(exp)));
    half2 probabilities = __hmul2(exp, __half2half2(hrcp(sum)));
    p[query * 64 + lane] = __low2half(probabilities);
    p[query * 64 + lane + 32] = __high2half(probabilities);
}

template<class Acc>
__device__ void attend(half* out, const half* in) {
    // A CTA owns one complete head: 64 spatial tokens, 32 channels per token.
    __shared__ __align__(32) half q[64 * 32], k[64 * 32], v[64 * 32], p[64 * 64];
    __shared__ __align__(32) Acc scores[64 * 64];
    int head = blockIdx.x % 16, n = blockIdx.x / 16;
    int t = threadIdx.x, warp = t / 32;
    for (int i = t; i < 64 * 32; i += 256) {
        int index = (n * 64 + i / 32) * 1536 + head * 32 + i % 32;
        q[i] = in[index];
        k[i] = in[index + 512];
        v[i] = in[index + 1024];
    }
    __syncthreads();

    int row = warp / 2 * 16, col = warp % 2 * 32;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, Acc> acc0, acc1;
    wmma::fill_fragment(acc0, Acc(0.0f));
    wmma::fill_fragment(acc1, Acc(0.0f));
    for (int i = 0; i < 32; i += 16) {
        wmma::load_matrix_sync(a, q + row * 32 + i, 32);
        wmma::load_matrix_sync(b, k + col * 32 + i, 32);
        wmma::mma_sync(acc0, a, b, acc0);
        wmma::load_matrix_sync(b, k + (col + 16) * 32 + i, 32);
        wmma::mma_sync(acc1, a, b, acc1);
    }
    wmma::store_matrix_sync(scores + row * 64 + col, acc0, 64, wmma::mem_row_major);
    wmma::store_matrix_sync(scores + row * 64 + col + 16, acc1, 64, wmma::mem_row_major);
    __syncthreads();

    int lane = t % 32;
    for (int query = warp; query < 64; query += 8)
        attention_softmax(p, scores, query, lane);
    __syncthreads();

    col = warp % 2 * 16;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> value;
    wmma::fill_fragment(acc0, Acc(0.0f));
    for (int i = 0; i < 64; i += 16) {
        wmma::load_matrix_sync(a, p + row * 64 + i, 64);
        wmma::load_matrix_sync(value, v + i * 32 + col, 32);
        wmma::mma_sync(acc0, a, value, acc0);
    }
    wmma::store_matrix_sync(scores + row * 64 + col, acc0, 64, wmma::mem_row_major);
    __syncthreads();
    for (int i = t; i < 64 * 32; i += 256)
        out[(n * 64 + i / 32) * 512 + head * 32 + i % 32] = scores[(i / 32) * 64 + i % 32];
}

extern "C" __global__ void attention(half* y, const half* x) { attend<float>(y, x); }
extern "C" __global__ void attention16(half* y, const half* x) { attend<half>(y, x); }

extern "C" __global__ void gate(half* out, const half* in, const half* attn,
                                 int stride, int total) {
    constexpr int c = 512;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int hidden = 3 * c / 2, pixel = i / hidden, ch = i % hidden;
    int base = pixel * stride;
    float v = ch < c / 2 ? read(in, base + hidden + ch)
                         : read(attn, pixel * c + ch - c / 2);
    out[i] = mish(read(in, base + ch)) * v;
}

extern "C" __global__ void add(half* out, const half* a, const half* b, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = read(a, i) + read(b, i);
}

extern "C" __global__ void shuffle_concat(half2* out, const half2* in, const half2* skip,
                                           int h, int w, int channels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int ch = i & (channels - 1), p = i / channels;
    int x = p % w, y = p / w % h, n = p / (w * h);
    if (ch >= channels / 2) {
        out[i] = skip[p * (channels / 2) + ch - channels / 2];
    } else {
        int phase = 2 * (y & 1) + (x & 1);
        int pixel = ((n * (h / 2) + y / 2) * (w / 2) + x / 2);
        const auto scalar = reinterpret_cast<const half*>(in);
        int index = (pixel * channels + 2 * ch) * 4 + phase;
        out[i] = __halves2half2(scalar[index], scalar[index + 4]);
    }
}

#include "fp16.cuh"

#include "../../common/resize_rgb.cuh"

#define RESIZE_RGB(NAME, T, CROP) \
extern "C" __global__ void NAME(T* y, const T* x, int h, int w, int ph, int pw, int n) { \
    resize_rgb<T, CROP>(y, x, h, w, ph, pw, n); \
}
RESIZE_RGB(pad_rgb, float, false)
RESIZE_RGB(crop_rgb, float, true)
RESIZE_RGB(pad_rgb16, half, false)
RESIZE_RGB(crop_rgb16, half, true)

__device__ float attn_mul(float a, float b) { return a * b; }
__device__ half attn_mul(half a, half b) { return __hmul(a, b); }
__device__ float attn_fma(float a, float b, float c) { return fmaf(a, b, c); }
__device__ half attn_fma(half a, half b, half c) { return __hfma(a, b, c); }
__device__ float attn_max(float a, float b) { return fmaxf(a, b); }
__device__ half attn_max(half a, half b) { return __hmax(a, b); }
__device__ float attn_exp_diff(float a, float b) { return __expf(a - b); }
__device__ half attn_exp_diff(half a, half b) { return hexp(__hsub(a, b)); }
__device__ float attn_div(float a, float b) { return a / b; }
__device__ half attn_div(half a, half b) { return __hdiv(a, b); }
template<class Acc> __device__ Acc attn_read(half x) { return Acc(x); }

template<class Acc>
__device__ void attend_dynamic(half* out, const half* in, int tokens) {
    // One warp per query. Combine 32-key softmax reductions without storing
    // the score matrix, keeping the original global attention semantics.
    constexpr int warps = 4, heads = 16, channels = 512, stride = 3 * channels;
    int groups = (tokens + warps - 1) / warps;
    int row = blockIdx.x % groups * warps + threadIdx.x / 32;
    if (row >= tokens) return;
    int head = blockIdx.x / groups % heads, image = blockIdx.x / groups / heads;
    int col = head * 32 + threadIdx.x % 32;
    const half* input = in + (image * tokens) * stride + col;
    Acc query = attn_read<Acc>(input[row * stride]);
    const float negative_infinity = __int_as_float(0xff800000);
    Acc maximum(negative_infinity), denominator(0.f), numerator(0.f);
    Acc norm_scale(fminf(1.f, 32.f / tokens)), logit_scale(0.1767766953f);
    for (int start = 0; start < tokens; start += 32) {
        Acc tile_max(negative_infinity), tile_sum(0.f), tile_value(0.f);
        int end = min(tokens, start + 32);
        for (int key = start; key < end; ++key) {
            const half* location = input + key * stride;
            Acc score = attn_mul(warp_sum(attn_mul(query, attn_read<Acc>(location[channels]))), logit_scale);
            Acc next = attn_max(score, tile_max);
            Acc a = attn_exp_diff(tile_max, next), b = attn_exp_diff(score, next);
            tile_sum = attn_fma(tile_sum, a, b);
            tile_value = attn_fma(attn_read<Acc>(location[2 * channels]), b, attn_mul(tile_value, a));
            tile_max = next;
        }
        Acc next = attn_max(tile_max, maximum);
        Acc a = attn_exp_diff(maximum, next), b = attn_mul(attn_exp_diff(tile_max, next), norm_scale);
        numerator = attn_fma(numerator, a, attn_mul(tile_value, b));
        denominator = attn_fma(denominator, a, attn_mul(tile_sum, b));
        maximum = next;
    }
    out[(image * tokens + row) * channels + col] = attn_div(numerator, denominator);
}

extern "C" __global__ void attention_dynamic(half* y, const half* x, int tokens) {
    attend_dynamic<float>(y, x, tokens);
}
extern "C" __global__ void attention16_dynamic(half* y, const half* x, int tokens) {
    attend_dynamic<half>(y, x, tokens);
}

// Each half warp owns one pixel and all 32 gated channels. The gate is
// materialized in FP16 because the later channel scaling also reads it.
template<bool Full, int Square>
__device__ void gate_pool(half2* out, void* partial, const half2* in,
                           const half2* weights, const half2* bias,
                           int h = Square, int w = Square, int chunks = 0) {
    constexpr int groups = 16, pixels_per_step = 16, chunk = 64;
    int g = threadIdx.x % groups, lane_pixel = threadIdx.x / groups;
    int base = blockIdx.x * chunk;
    int count = chunk;
    if constexpr (Square == 0) {
        // Lay out partials by image, including a final short chunk if needed.
        const int pixels = h * w;
        const int start = (blockIdx.x % chunks) * chunk;
        base = (blockIdx.x / chunks) * pixels + start;
        count = min(chunk, pixels - start);
    }
    if constexpr (Full) {
        __shared__ half2 sums[pixels_per_step][groups];
        half2 acc = splat16(0);
        for (int p = lane_pixel; p < count; p += pixels_per_step) {
            int pixel = base + p;
            half2 value = local16_value<32, Square>(in, weights, bias, pixel, g, h, w);
            out[pixel * groups + g] = value;
            acc = __hadd2(acc, value);
        }
        sums[lane_pixel][g] = __hmul2(acc, splat16(1.0f / chunk));
        __syncthreads();
        if (!lane_pixel) {
            acc = sums[0][g];
            for (int p = 1; p < pixels_per_step; ++p) acc = __hadd2(acc, sums[p][g]);
            static_cast<half2*>(partial)[blockIdx.x * groups + g] = acc;
        }
    } else {
        __shared__ float2 sums[pixels_per_step][groups];
        float2 acc{0.f, 0.f};
        for (int p = lane_pixel; p < count; p += pixels_per_step) {
            int pixel = base + p;
            half2 value = local_gate_value<32, Square>(in, weights, bias, pixel, g, h, w);
            out[pixel * groups + g] = value;
            float2 rounded = __half22float2(value);
            acc.x += rounded.x;
            acc.y += rounded.y;
        }
        sums[lane_pixel][g] = acc;
        __syncthreads();
        if (!lane_pixel) {
            for (int p = 1; p < pixels_per_step; ++p) {
                acc.x += sums[p][g].x;
                acc.y += sums[p][g].y;
            }
            static_cast<float2*>(partial)[blockIdx.x * groups + g] = acc;
        }
    }
}

extern "C" __global__ void gate_pool32(half2* y, float2* p, const half2* x,
                                        const half2* k, const half2* b) {
    gate_pool<false, 128>(y, p, x, k, b);
}
extern "C" __global__ void gate_pool16_32(half2* y, half2* p, const half2* x,
                                          const half2* k, const half2* b) {
    gate_pool<true, 128>(y, p, x, k, b);
}
extern "C" __global__ void gate_pool32_dynamic(half2* y, float2* p, const half2* x,
                                                const half2* k, const half2* b, int h, int w, int chunks) {
    gate_pool<false, 0>(y, p, x, k, b, h, w, chunks);
}
extern "C" __global__ void gate_pool16_32_dynamic(half2* y, half2* p, const half2* x,
                                                  const half2* k, const half2* b, int h, int w, int chunks) {
    gate_pool<true, 0>(y, p, x, k, b, h, w, chunks);
}

template<int C, bool Local, bool Full>
__device__ void residual_norm(half2* residual, half2* normalized, const half2* a,
                              const half2* b, const half2* skip, const half2* gamma,
                              const half2* scale, const half2* offset, int pixels, int total_pixels) {
    constexpr int pairs = C / 2;
    auto source = [=](int p, int c) {
        int index = p * pairs + c;
        half2 value;
        if constexpr (Full) {
            if constexpr (Local) {
                half2 gate = __hmul2(a[index], b[(p / pixels) * pairs + c]);
                value = __hfma2(gate, gamma[c], skip[index]);
            } else {
                half2 activation = mish16_pair(a[index]);
                value = b ? __hfma2(activation, gamma[c], b[index]) : activation;
            }
        } else {
            float2 x = __half22float2(a[index]);
            if constexpr (Local) {
                float2 gate = __half22float2(b[(p / pixels) * pairs + c]);
                float2 gain = __half22float2(gamma[c]), s = __half22float2(skip[index]);
                x.x = fmaf(x.x * gate.x, gain.x, s.x);
                x.y = fmaf(x.y * gate.y, gain.y, s.y);
            } else {
                x.x = mish(x.x);
                x.y = mish(x.y);
                if (b) {
                    float2 gain = __half22float2(gamma[c]), s = __half22float2(b[index]);
                    x.x = fmaf(x.x, gain.x, s.x);
                    x.y = fmaf(x.y, gain.y, s.y);
                }
            }
            value = __float22half2_rn(x);
        }
        residual[index] = value;
        return value;
    };
    if constexpr (Full) normalize16_from<C>(normalized, scale, offset, total_pixels, source);
    else normalize_from<C>(normalized, scale, offset, total_pixels, source);
}

#define RESIDUAL_NORM(NAME, C, LOCAL, FULL) \
extern "C" __global__ void NAME##C(half2* y, half2* z, const half2* a, const half2* b, \
                                    const half2* skip, const half2* gamma, const half2* scale, \
                                    const half2* offset, int p, int total) { \
    residual_norm<C, LOCAL, FULL>(y, z, a, b, skip, gamma, scale, offset, p, total); \
}
#define FUSED_NORM(C) \
    RESIDUAL_NORM(local_norm, C, true, false) \
    RESIDUAL_NORM(global_norm, C, false, false) \
    RESIDUAL_NORM(local_norm16_, C, true, true) \
    RESIDUAL_NORM(global_norm16_, C, false, true)
FUSED_NORM(32)
FUSED_NORM(64)
FUSED_NORM(128)
FUSED_NORM(256)
RESIDUAL_NORM(global_norm, 512, false, false)
RESIDUAL_NORM(global_norm16_, 512, false, true)

// One block owns a head. NHWC Q/K/V are transposed into [head_channel, HW].
extern "C" __global__ void attention_channels(half* out, const half* in, const float* temperatures) {
    using namespace nvcuda::wmma;
    constexpr int C = 32, P = 64, LD = P + 8;
    __shared__ __align__(32) half q[C][LD], k[C][LD], v[C][LD], probabilities[C][C];
    __shared__ __align__(32) float result[C][P];
    const int head = blockIdx.x % 16, batch = blockIdx.x / 16;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    in += batch * P * 1536 + head * C;
    out += batch * P * 512 + head * C;
    for (int i = tid; i < C * P; i += 256) {
        const int c = i % C, p = i / C, at = p * 1536 + c;
        q[c][p] = in[at];
        k[c][p] = in[at + 512];
        v[c][p] = in[at + 1024];
    }
    __syncthreads();

    for (int c = warp; c < C; c += 8) {
        const float qa = __half2float(q[c][lane]), qb = __half2float(q[c][lane+32]);
        const float ka = __half2float(k[c][lane]), kb = __half2float(k[c][lane+32]);
        const float qn = fmaxf(sqrtf(warp_sum(qa*qa + qb*qb)), 1e-12f);
        const float kn = fmaxf(sqrtf(warp_sum(ka*ka + kb*kb)), 1e-12f);
        q[c][lane] = __float2half_rn(qa / qn);
        q[c][lane+32] = __float2half_rn(qb / qn);
        k[c][lane] = __float2half_rn(ka / kn);
        k[c][lane+32] = __float2half_rn(kb / kn);
    }
    __syncthreads();

    if (warp < 4) {
        const int row = warp / 2 * 16, col = warp % 2 * 16;
        fragment<matrix_a, 16, 16, 16, half, row_major> a;
        fragment<matrix_b, 16, 16, 16, half, col_major> b;
        fragment<accumulator, 16, 16, 16, float> acc;
        fill_fragment(acc, 0.0f);
        #pragma unroll
        for (int s = 0; s < P; s += 16) {
            load_matrix_sync(a, &q[row][s], LD);
            load_matrix_sync(b, &k[col][s], LD);
            mma_sync(acc, a, b, acc);
        }
        store_matrix_sync(&result[row][col], acc, P, mem_row_major);
    }
    __syncthreads();

    for (int c = warp; c < C; c += 8) {
        float x = result[c][lane] * temperatures[head];
        float max_score = x;
        for (int d = 16; d; d /= 2) max_score = fmaxf(max_score, __shfl_xor_sync(0xffffffff, max_score, d));
        x = expf(x - max_score);
        probabilities[c][lane] = __float2half_rn(x / warp_sum(x));
    }
    __syncthreads();

    const int row = warp / 4 * 16, col = warp % 4 * 16;
    fragment<matrix_a, 16, 16, 16, half, row_major> a;
    fragment<matrix_b, 16, 16, 16, half, row_major> b;
    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.0f);
    #pragma unroll
    for (int s = 0; s < C; s += 16) {
        load_matrix_sync(a, &probabilities[row][s], C);
        load_matrix_sync(b, &v[s][col], LD);
        mma_sync(acc, a, b, acc);
    }
    store_matrix_sync(&result[row][col], acc, P, mem_row_major);
    __syncthreads();
    for (int i = tid; i < C * P; i += 256)
        out[i / C * 512 + i % C] = __float2half_rn(result[i % C][i / C]);
}
