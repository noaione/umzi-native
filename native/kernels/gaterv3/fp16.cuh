// Binary16 arithmetic for the full-FP16 execution mode.
__device__ half2 splat16(float x) { return __float2half2_rn(x); }

__device__ half2 select_positive(half2 x, half2 positive, half2 negative) {
    const half zero = __float2half(0);
    return __halves2half2(
        __hge(__low2half(x), zero) ? __low2half(positive) : __low2half(negative),
        __hge(__high2half(x), zero) ? __high2half(positive) : __high2half(negative));
}

__device__ half2 mish16_pair(half2 x) {
    half2 e = h2exp(__hneg2(__habs2(x)));
    half2 square = __hmul2(e, e), twice = __hadd2(e, e);
    half2 a = __hadd2(twice, select_positive(x, splat16(1), square));
    half2 b = __hadd2(a, select_positive(x, __hadd2(square, square), splat16(2)));
    return __hmul2(x, __hmul2(a, h2rcp(b)));
}

__device__ half2 sigmoid16_pair(half2 x) {
    half2 e = h2exp(__hneg2(__habs2(x)));
    half2 inv = h2rcp(__hadd2(splat16(1), e));
    return select_positive(x, inv, __hmul2(e, inv));
}

extern "C" __global__ void pack16(half* out, const half* in, int pixels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int channel = i % 8, pixel = i / 8;
    int image = pixel / pixels, p = pixel % pixels;
    out[i] = channel < 3 ? in[(image * 3 + channel) * pixels + p] : __float2half(0);
}

extern "C" __global__ void rgb16(half* out, const half* residual, const half* in,
                                  const half* gamma, int pixels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int image_channel = i / pixels, p = i % pixels;
    int c = image_channel % 3, image = image_channel / 3;
    out[i] = __hfma(gamma[c], in[i], residual[(image * pixels + p) * 8 + c]);
}

template<int C, class ReadPair>
__device__ void normalize16_from(half2* out, const half2* scale, const half2* offset,
                                 int pixels, ReadPair read_pair) {
    constexpr int lanes = C == 32 ? 16 : 32;
    constexpr int pairs = C / 2, steps = pairs / lanes;
    int lane = threadIdx.x % lanes;
    int pixel = blockIdx.x * (blockDim.x / lanes) + threadIdx.x / lanes;
    if (pixel >= pixels) return;
    half2 values[steps], maxima = splat16(0);
    #pragma unroll
    for (int i = 0; i < steps; ++i) {
        values[i] = read_pair(pixel, i * lanes + lane);
        maxima = __hmax2(maxima, __habs2(values[i]));
    }
    // Scale before squaring to avoid overflow and preserve small magnitudes.
    half max_value = __hmax(__float2half(0.0009765625f),
                            __hmax(__low2half(maxima), __high2half(maxima)));
    #pragma unroll
    for (int delta = lanes / 2; delta; delta /= 2)
        max_value = __hmax(max_value, __shfl_xor_sync(0xffffffff, max_value, delta));
    half inv_max = hrcp(max_value);
    half2 square = splat16(0);
    #pragma unroll
    for (int i = 0; i < steps; ++i) {
        values[i] = __hmul2(values[i], __half2half2(inv_max));
        square = __hfma2(values[i], values[i], square);
    }
    half sum = __hadd(__low2half(square), __high2half(square));
    #pragma unroll
    for (int delta = lanes / 2; delta; delta /= 2)
        sum = __hadd(sum, __shfl_xor_sync(0xffffffff, sum, delta));
    half norm = hsqrt(__hmul(sum, __float2half(1.0f / C)));
    half eps = __hmul(__float2half(1.0e-6f), inv_max);
    half2 inverse = __half2half2(hrcp(__hadd(norm, eps)));
    #pragma unroll
    for (int i = 0; i < steps; ++i) {
        int c = i * lanes + lane;
        out[pixel * pairs + c] = __hfma2(__hmul2(values[i], inverse), scale[c], offset[c]);
    }
}

#define NORMALIZE16(C) \
extern "C" __global__ void rms16_##C(half2* y, const half2* x, const half2* s, \
                                      const half2* b, int p) { \
    normalize16_from<C>(y, s, b, p, [=](int pixel, int c) { return x[pixel * (C / 2) + c]; }); \
}
NORMALIZE16(32)
NORMALIZE16(64)
NORMALIZE16(128)
NORMALIZE16(256)
NORMALIZE16(512)

extern "C" __global__ void bias16(half* out, const half* b, int channels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = __hadd(out[i], b[i % channels]);
}

template<int channels, int square>
__device__ __forceinline__ half2 local16_value(const half2* in, const half2* weight,
                                               const half2* bias, int pixel, int g, int h = square, int w = square) {
    constexpr int groups = channels / 2;
    int x = pixel % w, y = pixel / w % h;
    half2 a = bias[g], b = bias[g + groups];
    #pragma unroll
    for (int ky = -1; ky <= 1; ++ky) {
        #pragma unroll
        for (int kx = -1; kx <= 1; ++kx) {
            if (y + ky < 0 || y + ky >= h || x + kx < 0 || x + kx >= w) continue;
            int pos = (pixel + ky * w + kx) * channels;
            int k = ((ky + 1) * 3 + kx + 1) * 2 * channels + g;
            half2 left = in[pos + g], right = in[pos + g + groups];
            a = __hfma2(__low2half2(left), weight[k], a);
            a = __hfma2(__high2half2(left), weight[k + channels], a);
            b = __hfma2(__low2half2(right), weight[k + groups], b);
            b = __hfma2(__high2half2(right), weight[k + channels + groups], b);
        }
    }
    return __hmul2(a, b);
}

template<int channels, int square>
__device__ void local16(half2* out, const half2* in, const half2* weight,
                        const half2* bias, int total, int h = square, int w = square) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        out[i] = local16_value<channels, square>(in, weight, bias, i / (channels / 2), i % (channels / 2), h, w);
}

template<int channels, int square>
__device__ void inception16_pair(half2* out, const half2* in, const half2* weight,
                                  const half2* bias, int total, int h = square, int w = square) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    constexpr int hidden = channels * 3 / 2, stride = channels * 3, band = channels / 8;
    int c = i % hidden, pixel = i / hidden, x = pixel % w, y = pixel / w % h;
    half2 v;
    if (c < hidden - 3 * band) {
        v = in[pixel * stride + hidden + c];
    } else {
        int branch = (c - (hidden - 3 * band)) / band, bc = c % band;
        v = bias[branch * band + bc];
        if (branch == 0) {
            #pragma unroll
            for (int dy = -1; dy <= 1; ++dy) {
                #pragma unroll
                for (int dx = -1; dx <= 1; ++dx) {
                    if (y + dy < 0 || y + dy >= h || x + dx < 0 || x + dx >= w) continue;
                    int k = ((dy + 1) * 3 + dx + 1) * band + bc;
                    v = __hfma2(in[(pixel + dy * w + dx) * stride + hidden + c], weight[k], v);
                }
            }
        } else {
            int offset = (branch == 1 ? 9 : 20) * band + bc;
            int coord = branch == 1 ? x : y, step = branch == 1 ? 1 : w;
            #pragma unroll
            for (int j = -5; j <= 5; ++j) {
                if (coord + j < 0 || coord + j >= (branch == 1 ? w : h)) continue;
                v = __hfma2(in[(pixel + j * step) * stride + hidden + c],
                            weight[offset + (j + 5) * band], v);
            }
        }
    }
    out[i] = __hmul2(mish16_pair(in[pixel * stride + c]), v);
}

#define SPATIAL16(C, W) \
extern "C" __global__ void local_gate16_##C(half2* y, const half2* x, const half2* k, \
                                            const half2* b, int n) { local16<C, W>(y, x, k, b, n); } \
extern "C" __global__ void inception16_##C(half2* y, const half2* x, const half2* k, \
                                            const half2* b, int n) { inception16_pair<C / 2, W>(y, x, k, b, n); }
SPATIAL16(32, 128)
SPATIAL16(64, 64)
SPATIAL16(128, 32)
SPATIAL16(256, 16)

#define SPATIAL16_DYNAMIC(C) \
extern "C" __global__ void local_gate16_##C##_dynamic(half2* y, const half2* x, const half2* k, \
    const half2* b, int h, int w, int total) { local16<C, 0>(y, x, k, b, total, h, w); } \
extern "C" __global__ void inception16_##C##_dynamic(half2* y, const half2* x, const half2* k, \
    const half2* b, int h, int w, int total) { inception16_pair<C / 2, 0>(y, x, k, b, total, h, w); }
SPATIAL16_DYNAMIC(32)
SPATIAL16_DYNAMIC(64)
SPATIAL16_DYNAMIC(128)
SPATIAL16_DYNAMIC(256)

template<bool Linear>
__device__ void mean_partial16_spatial(half* out, const half* in, int pixels,
                                           int channels, int chunks) {
    __shared__ half partial[8][32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int blocks = channels / 32;
    int row = Linear ? blockIdx.x / blocks : blockIdx.y;
    int c = (Linear ? blockIdx.x % blocks : blockIdx.x) * 32 + lane;
    int n = row / chunks;
    int start = row % chunks * 256;
    half acc = __float2half(0);
    for (int p = start + warp; p < pixels && p < start + 256; p += 8)
        acc = __hadd(acc, in[(n * pixels + p) * channels + c]);
    partial[warp][lane] = __hmul(acc, __float2half(0.00390625f));
    __syncthreads();
    if (!warp) {
        acc = partial[0][lane];
        #pragma unroll
        for (int i = 1; i < 8; ++i) acc = __hadd(acc, partial[i][lane]);
        out[row * channels + c] = acc;
    }
}
extern "C" __global__ void mean_partial16(half* y, const half* x, int pixels, int channels, int chunks) {
    mean_partial16_spatial<false>(y, x, pixels, channels, chunks);
}
extern "C" __global__ void mean_partial16_dynamic(half* y, const half* x, int pixels, int channels, int chunks) {
    mean_partial16_spatial<true>(y, x, pixels, channels, chunks);
}

template<bool Dynamic>
__device__ void mean_project16_spatial(half* out, const half* partial, const half* weight,
                                          const half* bias, int pixels, int channels, int chunks, int chunk) {
    __shared__ half average[256];
    int c = threadIdx.x, lane = c % 32, warp = c / 32, n = blockIdx.x;
    if (c < channels) {
        half sum = __float2half(0);
        if constexpr (Dynamic) {
            // Partials all use the same divisor, even the final short chunk.
            // Scale before adding to keep the FP16 running sum in range.
            half scale = __float2half(float(chunk) / pixels);
            for (int j = 0; j < chunks; ++j)
                sum = __hfma(partial[(n * chunks + j) * channels + c], scale, sum);
            average[c] = sum;
        } else {
            for (int j = 0; j < chunks; ++j)
                sum = __hadd(sum, partial[(n * chunks + j) * channels + c]);
            average[c] = __hmul(sum, hrcp(__int2half_rn(chunks)));
        }
    }
    __syncthreads();
    for (int output = warp; output < channels; output += 8) {
        half dot = __float2half(0);
        for (int i = lane; i < channels; i += 32)
            dot = __hfma(average[i], weight[output * channels + i], dot);
        dot = warp_sum(dot);
        if (lane == 0) out[n * channels + output] = __hadd(dot, bias[output]);
    }
}
extern "C" __global__ void mean_project16(half* y, const half* p, const half* k, const half* b,
                                          int pixels, int channels, int chunks) {
    mean_project16_spatial<false>(y, p, k, b, pixels, channels, chunks, 256);
}
extern "C" __global__ void mean_project16_dynamic(half* y, const half* p, const half* k, const half* b,
                                                  int pixels, int channels, int chunks, int chunk) {
    mean_project16_spatial<true>(y, p, k, b, pixels, channels, chunks, chunk);
}

extern "C" __global__ void local_skip16(half2* out, const half2* gated, const half2* scale,
                                        const half2* skip, const half2* gamma,
                                        int pixels, int channels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = i & (channels - 1);
    half2 product = __hmul2(gated[i], scale[i / (pixels * channels) * channels + c]);
    out[i] = __hfma2(product, gamma[c], skip[i]);
}

extern "C" __global__ void global_skip16(half2* out, const half2* in, const half2* skip,
                                         const half2* gamma, int channels, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    half2 x = mish16_pair(in[i]);
    out[i] = skip ? __hfma2(x, gamma[i & (channels - 1)], skip[i]) : x;
}

extern "C" __global__ void silu16(half2* out, const half2* in, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = __hmul2(in[i], sigmoid16_pair(in[i]));
}

extern "C" __global__ void span_gate16(half2* out, const half2* in, const half2* skip, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    half2 x = in[i];
    out[i] = __hmul2(__hadd2(x, skip[i]), __hsub2(sigmoid16_pair(x), splat16(0.5f)));
}

template<int square>
__device__ void depthwise16_spatial(half* out, const half* in, const half* weight,
                                       const half* bias, int stride, int total, int h = square, int w = square) {
    constexpr int channels = 1536;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = i % channels, pixel = i / channels, x = pixel % w, y = pixel / w % h;
    half v = bias[c];
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (y + dy < 0 || y + dy >= h || x + dx < 0 || x + dx >= w) continue;
            v = __hfma(in[(pixel + dy * w + dx) * stride + stride - channels + c],
                       weight[((dy + 1) * 3 + dx + 1) * channels + c], v);
        }
    }
    out[i] = v;
}
extern "C" __global__ void depthwise16(half* y, const half* x, const half* k, const half* b,
                                  int stride, int total) {
    depthwise16_spatial<8>(y, x, k, b, stride, total);
}
extern "C" __global__ void depthwise16_dynamic(half* y, const half* x, const half* k, const half* b,
                                          int stride, int h, int w, int total) {
    depthwise16_spatial<0>(y, x, k, b, stride, total, h, w);
}

extern "C" __global__ void gate16(half* out, const half* in, const half* attn, int stride, int total) {
    constexpr int channels = 512, hidden = 3 * channels / 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int pixel = i / hidden, ch = i % hidden;
    int base = pixel * stride;
    half x = in[base + ch];
    half a = ch < channels / 2 ? in[base + hidden + ch] : attn[pixel * channels + ch - channels / 2];
    out[i] = __hmul(__low2half(mish16_pair(__half2half2(x))), a);
}

extern "C" __global__ void add16(half* out, const half* a, const half* b, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = __hadd(a[i], b[i]);
}
