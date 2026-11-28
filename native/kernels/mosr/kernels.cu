#include <cuda_fp16.h>
#include "../../common/tensor_core.cuh"

extern __shared__ uint4 mosr_smem[];

// ---------------------------------------------------------------------------
// Exact FP32 helpers (NCHW path)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float mish_exact(float x) {
    return x * tanhf(x > 20.f ? x : log1pf(expf(x)));
}

__device__ __forceinline__ float sigmoid_exact(float x) {
    return 1.f / (1.f + expf(-x));
}

extern "C" __global__ void copy_nchw_f32(float* out, const float* in, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) out[idx] = in[idx];
}

extern "C" __global__ void layer_norm_nchw_f32(float* out, const float* in, const float* weight, const float* bias,
                                               int dim, int hw, int pixels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixels) return;
    const size_t first = size_t(index / hw) * dim * hw + index % hw;
    float mean = 0.f;
    for (int c = 0; c < dim; ++c) mean += in[first + size_t(c) * hw];
    mean /= float(dim);
    float variance = 0.f;
    for (int c = 0; c < dim; ++c) {
        const float centered = in[first + size_t(c) * hw] - mean;
        variance = __fadd_rn(variance, __fmul_rn(centered, centered));
    }
    const float denominator = sqrtf(variance / float(dim) + 1e-6f);
    for (int c = 0; c < dim; ++c) {
        const size_t at = first + size_t(c) * hw;
        const float normalized = (in[at] - mean) / denominator;
        out[at] = __fadd_rn(__fmul_rn(normalized, weight[c]), bias[c]);
    }
}

extern "C" __global__ void plain_gate_nchw_f32(float* gated, const float* projected, const float* fc1_bias, int dim,
                                               int hidden, int hw, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int plain = hidden - dim;
    const int p = idx % hw, plane = idx / hw, n = plane / plain, c = plane % plain;
    const size_t g_idx = (size_t(n) * (2 * hidden) + c) * hw + p;
    const size_t i_idx = (size_t(n) * (2 * hidden) + hidden + c) * hw + p;
    const float g = projected[g_idx] + fc1_bias[c];
    const float i = projected[i_idx] + fc1_bias[hidden + c];
    gated[(size_t(n) * hidden + c) * hw + p] = mish_exact(g) * i;
}

template<int FixedK>
__device__ void dw_gate_nchw_impl(float* gated, const float* projected, const float* fc1_bias, const float* dw_weight,
                                  const float* dw_bias, int dim, int hidden, int ksize, int h, int w, int tiles_x) {
    float* smem = reinterpret_cast<float*>(mosr_smem);
    const int k = FixedK > 0 ? FixedK : ksize;
    const int tile_idx = blockIdx.x;
    const int tx_tile = tile_idx % tiles_x;
    const int ty_tile = tile_idx / tiles_x;
    const int ch = blockIdx.y;
    const int image = blockIdx.z;
    const int tid = threadIdx.x;
    const int radius = k >> 1;
    const int smem_w = 16 + 2 * radius;
    const int y0 = ty_tile * 16 - radius;
    const int x0 = tx_tile * 16 - radius;
    const size_t hw = size_t(h) * w;
    const int in_ch = 2 * hidden - dim + ch;
    const int g_ch = (hidden - dim) + ch;
    const float* c_plane = projected + (size_t(image) * (2 * hidden) + in_ch) * hw;
    const float bias_c = fc1_bias[in_ch];
    const int smem_count = smem_w * smem_w;
    constexpr int nthreads = (FixedK > 0) ? 128 : 256;
    for (int i = tid; i < smem_count; i += nthreads) {
        const int sy = i / smem_w, sx = i % smem_w;
        const int gy = y0 + sy, gx = x0 + sx;
        smem[i] = (unsigned(gy) < unsigned(h) && unsigned(gx) < unsigned(w)) ? (c_plane[size_t(gy) * w + gx] + bias_c)
                                                                             : 0.f;
    }
    __syncthreads();
    const float* w_ptr = dw_weight + size_t(ch) * k * k;
    const float dw_b = dw_bias[ch];
    const float fc1_gb = fc1_bias[g_ch];
    const float* g_plane = projected + (size_t(image) * (2 * hidden) + g_ch) * hw;
    float* out_plane = gated + (size_t(image) * hidden + g_ch) * hw;
    if constexpr (FixedK > 0) {
        const int lx = tid & 15, ly_pair = tid >> 4;
        const int py0 = ty_tile * 16 + ly_pair * 2;
        const int py1 = py0 + 1;
        const int px = tx_tile * 16 + lx;
        if (py0 >= h || px >= w) return;
        float acc0 = 0.f, acc1 = 0.f;
        for (int kx = 0; kx < FixedK; ++kx) {
            const float* col = smem + (ly_pair * 2) * smem_w + (lx + kx);
            float cur = col[0];
            if constexpr (FixedK <= 7) {
#pragma unroll
                for (int ky = 0; ky < FixedK; ++ky) {
                    const float nxt = col[(ky + 1) * smem_w];
                    const float wt = w_ptr[ky * FixedK + kx];
                    acc0 = fmaf(wt, cur, acc0);
                    acc1 = fmaf(wt, nxt, acc1);
                    cur = nxt;
                }
            } else {
                for (int ky = 0; ky < FixedK; ++ky) {
                    const float nxt = col[(ky + 1) * smem_w];
                    const float wt = w_ptr[ky * FixedK + kx];
                    acc0 = fmaf(wt, cur, acc0);
                    acc1 = fmaf(wt, nxt, acc1);
                    cur = nxt;
                }
            }
        }
        const size_t idx0 = size_t(py0) * w + px;
        out_plane[idx0] = mish_exact(g_plane[idx0] + fc1_gb) * (acc0 + dw_b);
        if (py1 < h) {
            const size_t idx1 = idx0 + w;
            out_plane[idx1] = mish_exact(g_plane[idx1] + fc1_gb) * (acc1 + dw_b);
        }
    } else {
        const int tx = tid & 15, ty = tid >> 4;
        const int y = ty_tile * 16 + ty, x = tx_tile * 16 + tx;
        if (y >= h || x >= w) return;
        float accum = 0.f;
        for (int ky = 0; ky < k; ++ky) {
            const float* row = smem + (ty + ky) * smem_w + tx;
            for (int kx = 0; kx < k; ++kx) accum = fmaf(w_ptr[ky * k + kx], row[kx], accum);
        }
        const size_t idx = size_t(y) * w + x;
        out_plane[idx] = mish_exact(g_plane[idx] + fc1_gb) * (accum + dw_b);
    }
}

extern "C" __global__ void dw_gate_nchw_f32(float* gated, const float* projected, const float* fc1_bias,
                                            const float* dw_weight, const float* dw_bias, int dim, int hidden,
                                            int ksize, int h, int w, int tiles_x) {
    dw_gate_nchw_impl<0>(gated, projected, fc1_bias, dw_weight, dw_bias, dim, hidden, ksize, h, w, tiles_x);
}

extern "C" __global__ void dw_gate_nchw_f32_k7(float* gated, const float* projected, const float* fc1_bias,
                                               const float* dw_weight, const float* dw_bias, int dim, int hidden,
                                               int ksize, int h, int w, int tiles_x) {
    dw_gate_nchw_impl<7>(gated, projected, fc1_bias, dw_weight, dw_bias, dim, hidden, ksize, h, w, tiles_x);
}

extern "C" __global__ void dw_gate_nchw_f32_k17(float* gated, const float* projected, const float* fc1_bias,
                                                const float* dw_weight, const float* dw_bias, int dim, int hidden,
                                                int ksize, int h, int w, int tiles_x) {
    dw_gate_nchw_impl<17>(gated, projected, fc1_bias, dw_weight, dw_bias, dim, hidden, ksize, h, w, tiles_x);
}

extern "C" __global__ void bias_norm_nchw_f32(float* state, float* norm_out, const float* stem_bias,
                                              const float* norm_weight, const float* norm_bias, int dim, int hw,
                                              int pixels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixels) return;
    const size_t first = size_t(index / hw) * dim * hw + index % hw;
    float mean = 0.f;
    if (dim <= 64) {
        float* s_col = reinterpret_cast<float*>(mosr_smem) + threadIdx.x;
        for (int c = 0; c < dim; ++c) {
            const size_t at = first + size_t(c) * hw;
            const float v = state[at] + stem_bias[c];
            state[at] = v;
            s_col[c * 128] = v;
            mean += v;
        }
        mean /= float(dim);
        float variance = 0.f;
        for (int c = 0; c < dim; ++c) {
            const float centered = s_col[c * 128] - mean;
            variance = __fadd_rn(variance, __fmul_rn(centered, centered));
        }
        const float denominator = sqrtf(variance / float(dim) + 1e-6f);
        for (int c = 0; c < dim; ++c) {
            const float normalized = (s_col[c * 128] - mean) / denominator;
            norm_out[first + size_t(c) * hw] = __fadd_rn(__fmul_rn(normalized, norm_weight[c]), norm_bias[c]);
        }
        return;
    }
    for (int c = 0; c < dim; ++c) {
        const size_t at = first + size_t(c) * hw;
        const float v = state[at] + stem_bias[c];
        state[at] = v;
        mean += v;
    }
    mean /= float(dim);
    float variance = 0.f;
    for (int c = 0; c < dim; ++c) {
        const float centered = state[first + size_t(c) * hw] - mean;
        variance = __fadd_rn(variance, __fmul_rn(centered, centered));
    }
    const float denominator = sqrtf(variance / float(dim) + 1e-6f);
    for (int c = 0; c < dim; ++c) {
        const size_t at = first + size_t(c) * hw;
        const float normalized = (state[at] - mean) / denominator;
        norm_out[at] = __fadd_rn(__fmul_rn(normalized, norm_weight[c]), norm_bias[c]);
    }
}

extern "C" __global__ void mish_residual_norm_nchw_f32(float* state, float* norm_out, const float* fc2_out,
                                                       const float* fc2_bias, const float* norm_weight,
                                                       const float* norm_bias, int dim, int hw, int pixels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixels) return;
    const size_t first = size_t(index / hw) * dim * hw + index % hw;
    float mean = 0.f;
    if (dim <= 64) {
        float* s_col = reinterpret_cast<float*>(mosr_smem) + threadIdx.x;
        for (int c = 0; c < dim; ++c) {
            const size_t at = first + size_t(c) * hw;
            const float v = mish_exact(fc2_out[at] + fc2_bias[c]) + (state[at] - 0.5f);
            state[at] = v;
            s_col[c * 128] = v;
            mean += v;
        }
        mean /= float(dim);
        float variance = 0.f;
        for (int c = 0; c < dim; ++c) {
            const float centered = s_col[c * 128] - mean;
            variance = __fadd_rn(variance, __fmul_rn(centered, centered));
        }
        const float denominator = sqrtf(variance / float(dim) + 1e-6f);
        for (int c = 0; c < dim; ++c) {
            const float normalized = (s_col[c * 128] - mean) / denominator;
            norm_out[first + size_t(c) * hw] = __fadd_rn(__fmul_rn(normalized, norm_weight[c]), norm_bias[c]);
        }
        return;
    }
    for (int c = 0; c < dim; ++c) {
        const size_t at = first + size_t(c) * hw;
        const float v = mish_exact(fc2_out[at] + fc2_bias[c]) + (state[at] - 0.5f);
        state[at] = v;
        mean += v;
    }
    mean /= float(dim);
    float variance = 0.f;
    for (int c = 0; c < dim; ++c) {
        const float centered = state[first + size_t(c) * hw] - mean;
        variance = __fadd_rn(variance, __fmul_rn(centered, centered));
    }
    const float denominator = sqrtf(variance / float(dim) + 1e-6f);
    for (int c = 0; c < dim; ++c) {
        const size_t at = first + size_t(c) * hw;
        const float normalized = (state[at] - mean) / denominator;
        norm_out[at] = __fadd_rn(__fmul_rn(normalized, norm_weight[c]), norm_bias[c]);
    }
}

extern "C" __global__ void mish_bias_residual_nchw_f32(float* state, const float* fc2_out, const float* fc2_bias,
                                                       int dim, int hw, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c = (idx / hw) % dim;
    state[idx] = mish_exact(fc2_out[idx] + fc2_bias[c]) + (state[idx] - 0.5f);
}

extern "C" __global__ void mish_bias_inplace_nchw_f32(float* buf, const float* bias, int channels, int hw, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c = (idx / hw) % channels;
    buf[idx] = mish_exact(buf[idx] + bias[c]);
}

extern "C" __global__ void combine_body_shortcut_nchw_f32(float* state, const float* tail2_bias, const float* sc_main,
                                                          const float* sc_11, const float* sc_11_bias, int dim, int hw,
                                                          int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c = (idx / hw) % dim;
    const float body = state[idx] + tail2_bias[c];
    const float sc = sc_main[idx] + (sc_11[idx] + sc_11_bias[c]);
    state[idx] = body + (sc - 0.5f);
}

extern "C" __global__ void gps_shuffle_nchw_f32(float* out, const float* in, const float* bias, int out_ch, int scale,
                                                int h, int w, int ensemble, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int oh = h * scale, ow = w * scale;
    const int ox = idx % ow, oy = (idx / ow) % oh, oc = (idx / (ow * oh)) % out_ch, n = idx / (ow * oh * out_ch);
    const int iy = oy / scale, ix = ox / scale;
    const int c_sub = (oc * scale + (oy % scale)) * scale + (ox % scale);
    const int plane = out_ch * scale * scale;
    const size_t hw = size_t(h) * w;
    float sum = 0.f;
    for (int k = 0; k < ensemble; ++k) {
        const int ch = k * plane + c_sub;
        sum += in[(size_t(n) * ensemble * plane + ch) * hw + size_t(iy) * w + ix] + bias[ch];
    }
    out[idx] = ensemble > 1 ? sum / float(ensemble) : sum;
}

extern "C" __global__ void dysample_nchw_f32(float* sampled, const float* state, const float* offset,
                                             const float* offset_bias, const float* scope, int dim, int groups,
                                             int scale, int h, int w, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int oh = h * scale, ow = w * scale;
    const int ox = idx % ow, oy = (idx / ow) % oh, c = (idx / (ow * oh)) % dim, n = idx / (ow * oh * dim);
    const int g = c / (dim / groups);
    const int iy = oy / scale, ix = ox / scale, sy = oy % scale, sx = ox % scale;
    const int sub = sy * scale + sx;
    const int scale2 = scale * scale;
    const int offset_ch = 2 * groups * scale2;
    const int x_ch = g * scale2 + sub;
    const int y_ch = (groups + g) * scale2 + sub;
    const size_t hw = size_t(h) * w;
    const size_t in_pix = size_t(iy) * w + ix;
    const float off_x = offset[(size_t(n) * offset_ch + x_ch) * hw + in_pix] + offset_bias[x_ch];
    const float scp_x = sigmoid_exact(scope[(size_t(n) * offset_ch + x_ch) * hw + in_pix]);
    const float off_y = offset[(size_t(n) * offset_ch + y_ch) * hw + in_pix] + offset_bias[y_ch];
    const float scp_y = sigmoid_exact(scope[(size_t(n) * offset_ch + y_ch) * hw + in_pix]);
    const float init_x = (float(sx) - 0.5f * float(scale - 1)) / float(scale);
    const float init_y = (float(sy) - 0.5f * float(scale - 1)) / float(scale);
    const float dx = off_x * scp_x * 0.5f + init_x;
    const float dy = off_y * scp_y * 0.5f + init_y;
    const float ux = fminf(fmaxf(float(ix) + dx, 0.f), float(w - 1));
    const float uy = fminf(fmaxf(float(iy) + dy, 0.f), float(h - 1));
    const int x0 = int(floorf(ux)), y0 = int(floorf(uy));
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float wx = ux - float(x0), wy = uy - float(y0);
    const float* plane = state + (size_t(n) * dim + c) * hw;
    const float v00 = plane[size_t(y0) * w + x0];
    const float v01 = plane[size_t(y0) * w + x1];
    const float v10 = plane[size_t(y1) * w + x0];
    const float v11 = plane[size_t(y1) * w + x1];
    const float top = v00 + wx * (v01 - v00);
    const float bot = v10 + wx * (v11 - v10);
    sampled[idx] = top + wy * (bot - top);
}

extern "C" __global__ void add_bias_nchw_f32(float* out, const float* in, const float* bias, int channels, int hw,
                                             int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c = (idx / hw) % channels;
    out[idx] = in[idx] + bias[c];
}

// ---------------------------------------------------------------------------
// Mixed & FP16 helpers (NHWC half activations, FP32 pointwise math)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float fast_mish(float value) {
    if (value > 20.f) return value;
    const float e = __expf(value), n = e * (e + 2.f);
    return value * __fdividef(n, n + 2.f);
}

__device__ __forceinline__ float fast_sigmoid(float value) {
    return 1.f / (1.f + __expf(-value));
}

__device__ __forceinline__ float warp_sum(float sum) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
    return __shfl_sync(0xffffffff, sum, 0);
}

__device__ __forceinline__ float value_of(float v) { return v; }
__device__ __forceinline__ float value_of(half v) { return __half2float(v); }
__device__ __forceinline__ void write_value(float* out, size_t idx, float v) { out[idx] = v; }
__device__ __forceinline__ void write_value(half* out, size_t idx, float v) { out[idx] = __float2half_rn(v); }

__device__ __forceinline__ float4 load_half4(const half* p) {
    if ((reinterpret_cast<unsigned long long>(p) & 7ull) == 0ull) {
        const uint2 raw = *reinterpret_cast<const uint2*>(p);
        const half2* h = reinterpret_cast<const half2*>(&raw);
        const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
        return make_float4(a.x, a.y, b.x, b.y);
    }
    return make_float4(__half2float(p[0]), __half2float(p[1]), __half2float(p[2]), __half2float(p[3]));
}

__device__ __forceinline__ void store_half4(half* p, float a, float b, float c, float d) {
    if ((reinterpret_cast<unsigned long long>(p) & 7ull) == 0ull) {
        uint2 raw;
        half2* h = reinterpret_cast<half2*>(&raw);
        h[0] = __floats2half2_rn(a, b);
        h[1] = __floats2half2_rn(c, d);
        *reinterpret_cast<uint2*>(p) = raw;
    } else {
        p[0] = __float2half_rn(a);
        p[1] = __float2half_rn(b);
        p[2] = __float2half_rn(c);
        p[3] = __float2half_rn(d);
    }
}

template<class Rgb>
__device__ void pack_rgb_nhwc(half* out, const Rgb* in, int h, int w, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c = idx & 7, pixel = idx >> 3;
    if (c >= 3) {
        out[idx] = __float2half_rn(0.f);
        return;
    }
    const int hw = h * w, n = pixel / hw, p = pixel % hw;
    out[idx] = __float2half_rn(value_of(in[(size_t(n) * 3 + c) * hw + p]));
}

extern "C" __global__ void pack_rgb_nhwc_f32(half* out, const float* in, int h, int w, int total) {
    pack_rgb_nhwc(out, in, h, w, total);
}

extern "C" __global__ void pack_rgb_nhwc_f16(half* out, const half* in, int h, int w, int total) {
    pack_rgb_nhwc(out, in, h, w, total);
}

extern "C" __global__ void layer_norm_nhwc(half* out, const half* in, const float* weight, const float* bias, int dim,
                                           int pixels) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= pixels) return;
    const half* row_in = in + size_t(warp) * dim;
    half* row_out = out + size_t(warp) * dim;
    float sum = 0.f;
    for (int c = 2 * lane; c < dim; c += 64) {
        const float2 p = __half22float2(*reinterpret_cast<const half2*>(row_in + c));
        sum += p.x + p.y;
    }
    const float mean = warp_sum(sum) / float(dim);
    float var = 0.f;
    for (int c = 2 * lane; c < dim; c += 64) {
        const float2 p = __half22float2(*reinterpret_cast<const half2*>(row_in + c));
        const float d0 = p.x - mean, d1 = p.y - mean;
        var = fmaf(d0, d0, fmaf(d1, d1, var));
    }
    const float inv = rsqrtf(warp_sum(var) / float(dim) + 1e-6f);
    for (int c = 2 * lane; c < dim; c += 64) {
        const float2 p = __half22float2(*reinterpret_cast<const half2*>(row_in + c));
        const float n0 = (p.x - mean) * inv;
        const float n1 = (p.y - mean) * inv;
        *reinterpret_cast<half2*>(row_out + c) =
            __floats2half2_rn(fmaf(n0, weight[c], bias[c]), fmaf(n1, weight[c + 1], bias[c + 1]));
    }
}

extern "C" __global__ void plain_gate_nhwc(half* gated, const half* projected, const float* fc1_bias, int dim,
                                           int hidden, int fc1_stride, int gated_stride, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int plain = hidden - dim;
    const int c = idx % plain, pixel = idx / plain;
    const half* row_in = projected + size_t(pixel) * fc1_stride;
    const float g = __half2float(row_in[c]) + fc1_bias[c];
    const float v = __half2float(row_in[hidden + c]) + fc1_bias[hidden + c];
    gated[size_t(pixel) * gated_stride + c] = __float2half_rn(fast_mish(g) * v);
    if (c == 0 && gated_stride > hidden) {
        for (int pad = hidden; pad < gated_stride; ++pad)
            gated[size_t(pixel) * gated_stride + pad] = __float2half_rn(0.f);
    }
}

extern "C" __global__ void plain_gate_nhwc_vec8(half* gated, const half* projected, const float* fc1_bias, int dim,
                                                int hidden, int fc1_stride, int gated_stride, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int groups = (hidden - dim) >> 3;
    const int g_idx = idx % groups, pixel = idx / groups;
    const int c = g_idx << 3;
    const half* row_in = projected + size_t(pixel) * fc1_stride;
    const uint4 g_u = *reinterpret_cast<const uint4*>(row_in + c);
    const uint4 v_u = *reinterpret_cast<const uint4*>(row_in + hidden + c);
    const half2* g_h = reinterpret_cast<const half2*>(&g_u);
    const half2* v_h = reinterpret_cast<const half2*>(&v_u);
    uint4 out_u;
    half2* out_h = reinterpret_cast<half2*>(&out_u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 gf = __half22float2(g_h[j]);
        const float2 vf = __half22float2(v_h[j]);
        const int cj = c + 2 * j;
        out_h[j] = __floats2half2_rn(fast_mish(gf.x + fc1_bias[cj]) * (vf.x + fc1_bias[hidden + cj]),
                                     fast_mish(gf.y + fc1_bias[cj + 1]) * (vf.y + fc1_bias[hidden + cj + 1]));
    }
    *reinterpret_cast<uint4*>(gated + size_t(pixel) * gated_stride + c) = out_u;
}

template<int FixedK>
__device__ void dw_gate_nhwc_8ch(half* gated, const half* projected, const float* fc1_bias, const float* dw_weight,
                                 const float* dw_bias, int dim, int hidden, int fc1_stride, int gated_stride, int h,
                                 int w, int tiles_x) {
    float* smem = reinterpret_cast<float*>(mosr_smem);
    const int tile_idx = blockIdx.x;
    const int tx_tile = tile_idx % tiles_x;
    const int ty_tile = tile_idx / tiles_x;
    const int ch0 = blockIdx.y << 3;
    const int image = blockIdx.z;
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    constexpr int radius = FixedK >> 1;
    constexpr int smem_w = 16 + 2 * radius;
    const int y0 = ty_tile * 16 - radius;
    const int x0 = tx_tile * 16 - radius;
    const int in_ch0 = 2 * hidden - dim + ch0;
    const int g_ch0 = (hidden - dim) + ch0;
    float b[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) b[j] = fc1_bias[in_ch0 + j];
    float4* s4 = reinterpret_cast<float4*>(smem);
    constexpr int smem_count = smem_w * smem_w;
    for (int i = tid; i < smem_count; i += 256) {
        const int sy = i / smem_w, sx = i % smem_w;
        const int gy = y0 + sy, gx = x0 + sx;
        float4 lo = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 hi = make_float4(0.f, 0.f, 0.f, 0.f);
        if (unsigned(gy) < unsigned(h) && unsigned(gx) < unsigned(w)) {
            const size_t pix = (size_t(image) * h + gy) * w + gx;
            const uint4 raw = *reinterpret_cast<const uint4*>(projected + pix * fc1_stride + in_ch0);
            const half2* hp = reinterpret_cast<const half2*>(&raw);
            const float2 f0 = __half22float2(hp[0]), f1 = __half22float2(hp[1]);
            const float2 f2 = __half22float2(hp[2]), f3 = __half22float2(hp[3]);
            lo = make_float4(f0.x + b[0], f0.y + b[1], f1.x + b[2], f1.y + b[3]);
            hi = make_float4(f2.x + b[4], f2.y + b[5], f3.x + b[6], f3.y + b[7]);
        }
        s4[2 * i] = lo;
        s4[2 * i + 1] = hi;
    }
    __syncthreads();
    const int y = ty_tile * 16 + ty, x = tx_tile * 16 + tx;
    if (y >= h || x >= w) return;
    constexpr int taps = FixedK * FixedK;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const float* wp[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) wp[j] = dw_weight + size_t(ch0 + j) * taps;
    for (int ky = 0; ky < FixedK; ++ky) {
        const float4* row = s4 + ((ty + ky) * smem_w + tx) * 2;
#pragma unroll
        for (int kx = 0; kx < FixedK; ++kx) {
            const float4 s0 = row[2 * kx];
            const float4 s1 = row[2 * kx + 1];
            const int tap = ky * FixedK + kx;
            acc[0] = fmaf(wp[0][tap], s0.x, acc[0]);
            acc[1] = fmaf(wp[1][tap], s0.y, acc[1]);
            acc[2] = fmaf(wp[2][tap], s0.z, acc[2]);
            acc[3] = fmaf(wp[3][tap], s0.w, acc[3]);
            acc[4] = fmaf(wp[4][tap], s1.x, acc[4]);
            acc[5] = fmaf(wp[5][tap], s1.y, acc[5]);
            acc[6] = fmaf(wp[6][tap], s1.z, acc[6]);
            acc[7] = fmaf(wp[7][tap], s1.w, acc[7]);
        }
    }
    const size_t pix = (size_t(image) * h + y) * w + x;
    const uint4 g_raw = *reinterpret_cast<const uint4*>(projected + pix * fc1_stride + g_ch0);
    const half2* g_h = reinterpret_cast<const half2*>(&g_raw);
    uint4 out_u;
    half2* out_h = reinterpret_cast<half2*>(&out_u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 gf = __half22float2(g_h[j]);
        const int c0 = 2 * j, c1 = 2 * j + 1;
        const float r0 = fast_mish(gf.x + fc1_bias[g_ch0 + c0]) * (acc[c0] + dw_bias[ch0 + c0]);
        const float r1 = fast_mish(gf.y + fc1_bias[g_ch0 + c1]) * (acc[c1] + dw_bias[ch0 + c1]);
        out_h[j] = __floats2half2_rn(r0, r1);
    }
    *reinterpret_cast<uint4*>(gated + pix * gated_stride + g_ch0) = out_u;
}

extern "C" __global__ void dw_gate_nhwc_8ch_k7(half* gated, const half* projected, const float* fc1_bias,
                                               const float* dw_weight, const float* dw_bias, int dim, int hidden,
                                               int fc1_stride, int gated_stride, int h, int w, int tiles_x) {
    dw_gate_nhwc_8ch<7>(gated, projected, fc1_bias, dw_weight, dw_bias, dim, hidden, fc1_stride, gated_stride, h, w,
                        tiles_x);
}

extern "C" __global__ void dw_gate_nhwc_8ch_k17(half* gated, const half* projected, const float* fc1_bias,
                                                const float* dw_weight, const float* dw_bias, int dim, int hidden,
                                                int fc1_stride, int gated_stride, int h, int w, int tiles_x) {
    dw_gate_nhwc_8ch<17>(gated, projected, fc1_bias, dw_weight, dw_bias, dim, hidden, fc1_stride, gated_stride, h, w,
                         tiles_x);
}

extern "C" __global__ void dw_gate_nhwc(half* gated, const half* projected, const float* fc1_bias,
                                        const float* dw_weight, const float* dw_bias, int dim, int hidden,
                                        int fc1_stride, int gated_stride, int ksize, int h, int w, int tiles_x) {
    float* smem = reinterpret_cast<float*>(mosr_smem);
    const int tile_idx = blockIdx.x;
    const int tx_tile = tile_idx % tiles_x;
    const int ty_tile = tile_idx / tiles_x;
    const int ch0 = blockIdx.y * 4;
    const int image = blockIdx.z;
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int radius = ksize >> 1;
    const int smem_w = 16 + 2 * radius;
    const int y0 = ty_tile * 16 - radius;
    const int x0 = tx_tile * 16 - radius;
    const int in_ch0 = 2 * hidden - dim + ch0;
    const int g_ch0 = (hidden - dim) + ch0;
    const float b0 = fc1_bias[in_ch0], b1 = fc1_bias[in_ch0 + 1], b2 = fc1_bias[in_ch0 + 2], b3 = fc1_bias[in_ch0 + 3];
    float4* s4 = reinterpret_cast<float4*>(smem);
    const int smem_count = smem_w * smem_w;
    for (int i = tid; i < smem_count; i += 256) {
        const int sy = i / smem_w, sx = i % smem_w;
        const int gy = y0 + sy, gx = x0 + sx;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (unsigned(gy) < unsigned(h) && unsigned(gx) < unsigned(w)) {
            const size_t pix = (size_t(image) * h + gy) * w + gx;
            const float4 raw = load_half4(projected + pix * fc1_stride + in_ch0);
            v = make_float4(raw.x + b0, raw.y + b1, raw.z + b2, raw.w + b3);
        }
        s4[i] = v;
    }
    __syncthreads();
    const int y = ty_tile * 16 + ty, x = tx_tile * 16 + tx;
    if (y >= h || x >= w) return;
    const int taps = ksize * ksize;
    const float* w0 = dw_weight + size_t(ch0 + 0) * taps;
    const float* w1 = dw_weight + size_t(ch0 + 1) * taps;
    const float* w2 = dw_weight + size_t(ch0 + 2) * taps;
    const float* w3 = dw_weight + size_t(ch0 + 3) * taps;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int ky = 0; ky < ksize; ++ky) {
        const float4* row = s4 + (ty + ky) * smem_w + tx;
        for (int kx = 0; kx < ksize; ++kx) {
            const float4 s = row[kx];
            const int tap = ky * ksize + kx;
            a0 = fmaf(w0[tap], s.x, a0);
            a1 = fmaf(w1[tap], s.y, a1);
            a2 = fmaf(w2[tap], s.z, a2);
            a3 = fmaf(w3[tap], s.w, a3);
        }
    }
    a0 += dw_bias[ch0 + 0];
    a1 += dw_bias[ch0 + 1];
    a2 += dw_bias[ch0 + 2];
    a3 += dw_bias[ch0 + 3];
    const size_t pix = (size_t(image) * h + y) * w + x;
    const float4 g_raw = load_half4(projected + pix * fc1_stride + g_ch0);
    store_half4(
        gated + pix * gated_stride + g_ch0,
        fast_mish(g_raw.x + fc1_bias[g_ch0 + 0]) * a0,
        fast_mish(g_raw.y + fc1_bias[g_ch0 + 1]) * a1,
        fast_mish(g_raw.z + fc1_bias[g_ch0 + 2]) * a2,
        fast_mish(g_raw.w + fc1_bias[g_ch0 + 3]) * a3);
}

extern "C" __global__ void mish_bias_residual_nhwc(half* state, const half* fc2_out, const float* fc2_bias, int dim,
                                                   int pairs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pairs) return;
    const int c = 2 * (idx % (dim / 2));
    const float2 f = __half22float2(reinterpret_cast<const half2*>(fc2_out)[idx]);
    const float2 s = __half22float2(reinterpret_cast<const half2*>(state)[idx]);
    reinterpret_cast<half2*>(state)[idx] =
        __floats2half2_rn(fast_mish(f.x + fc2_bias[c]) + (s.x - 0.5f),
                          fast_mish(f.y + fc2_bias[c + 1]) + (s.y - 0.5f));
}

extern "C" __global__ void mish_bias_inplace_nhwc(half* buf, const float* bias, int channels, int pairs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pairs) return;
    const int c = 2 * (idx % (channels / 2));
    const float2 v = __half22float2(reinterpret_cast<const half2*>(buf)[idx]);
    reinterpret_cast<half2*>(buf)[idx] =
        __floats2half2_rn(fast_mish(v.x + bias[c]), fast_mish(v.y + bias[c + 1]));
}

extern "C" __global__ void add_bias_nhwc(half* buf, const float* bias, int channels, int pairs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pairs) return;
    const int c = 2 * (idx % (channels / 2));
    const float2 v = __half22float2(reinterpret_cast<const half2*>(buf)[idx]);
    reinterpret_cast<half2*>(buf)[idx] = __floats2half2_rn(v.x + bias[c], v.y + bias[c + 1]);
}

extern "C" __global__ void bias_norm_nhwc(half* state, half* norm_out, const float* stem_bias,
                                          const float* norm_weight, const float* norm_bias, int dim, int pixels) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= pixels) return;
    half* row_state = state + size_t(warp) * dim;
    half* row_norm = norm_out + size_t(warp) * dim;
    float2 vals[4];
    float sum = 0.f;
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int c = 2 * lane + step * 64;
        if (c >= dim) break;
        const float2 s = __half22float2(*reinterpret_cast<const half2*>(row_state + c));
        const half2 h = __floats2half2_rn(s.x + stem_bias[c], s.y + stem_bias[c + 1]);
        *reinterpret_cast<half2*>(row_state + c) = h;
        vals[step] = __half22float2(h);
        sum += vals[step].x + vals[step].y;
    }
    const float mean = warp_sum(sum) / float(dim);
    float var = 0.f;
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int c = 2 * lane + step * 64;
        if (c >= dim) break;
        const float d0 = vals[step].x - mean, d1 = vals[step].y - mean;
        var = fmaf(d0, d0, fmaf(d1, d1, var));
    }
    const float inv = rsqrtf(warp_sum(var) / float(dim) + 1e-6f);
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int c = 2 * lane + step * 64;
        if (c >= dim) break;
        const float n0 = (vals[step].x - mean) * inv;
        const float n1 = (vals[step].y - mean) * inv;
        *reinterpret_cast<half2*>(row_norm + c) =
            __floats2half2_rn(fmaf(n0, norm_weight[c], norm_bias[c]), fmaf(n1, norm_weight[c + 1], norm_bias[c + 1]));
    }
}

extern "C" __global__ void mish_residual_norm_nhwc(half* state, half* norm_out, const half* fc2_out,
                                                   const float* fc2_bias, const float* norm_weight,
                                                   const float* norm_bias, int dim, int pixels) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= pixels) return;
    half* row_state = state + size_t(warp) * dim;
    const half* row_fc2 = fc2_out + size_t(warp) * dim;
    half* row_norm = norm_out + size_t(warp) * dim;
    float2 vals[4];
    float sum = 0.f;
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int c = 2 * lane + step * 64;
        if (c >= dim) break;
        const float2 f = __half22float2(*reinterpret_cast<const half2*>(row_fc2 + c));
        const float2 s = __half22float2(*reinterpret_cast<const half2*>(row_state + c));
        const half2 h = __floats2half2_rn(fast_mish(f.x + fc2_bias[c]) + (s.x - 0.5f),
                                          fast_mish(f.y + fc2_bias[c + 1]) + (s.y - 0.5f));
        *reinterpret_cast<half2*>(row_state + c) = h;
        vals[step] = __half22float2(h);
        sum += vals[step].x + vals[step].y;
    }
    const float mean = warp_sum(sum) / float(dim);
    float var = 0.f;
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int c = 2 * lane + step * 64;
        if (c >= dim) break;
        const float d0 = vals[step].x - mean, d1 = vals[step].y - mean;
        var = fmaf(d0, d0, fmaf(d1, d1, var));
    }
    const float inv = rsqrtf(warp_sum(var) / float(dim) + 1e-6f);
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int c = 2 * lane + step * 64;
        if (c >= dim) break;
        const float n0 = (vals[step].x - mean) * inv;
        const float n1 = (vals[step].y - mean) * inv;
        *reinterpret_cast<half2*>(row_norm + c) =
            __floats2half2_rn(fmaf(n0, norm_weight[c], norm_bias[c]), fmaf(n1, norm_weight[c + 1], norm_bias[c + 1]));
    }
}

extern "C" __global__ void combine_body_shortcut_nhwc(half* state, const float* tail2_bias, const half* sc_main,
                                                      const half* sc_11, const float* sc_11_bias, int dim, int pairs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pairs) return;
    const int c = 2 * (idx % (dim / 2));
    const float2 body = __half22float2(reinterpret_cast<const half2*>(state)[idx]);
    const float2 scm = __half22float2(reinterpret_cast<const half2*>(sc_main)[idx]);
    const float2 sc11 = __half22float2(reinterpret_cast<const half2*>(sc_11)[idx]);
    const float r0 = (body.x + tail2_bias[c]) + ((scm.x + (sc11.x + sc_11_bias[c])) - 0.5f);
    const float r1 = (body.y + tail2_bias[c + 1]) + ((scm.y + (sc11.y + sc_11_bias[c + 1])) - 0.5f);
    reinterpret_cast<half2*>(state)[idx] = __floats2half2_rn(r0, r1);
}

template<class Rgb>
__device__ void gps_shuffle_nhwc(Rgb* out, const half* in, const float* bias, int out_ch, int scale, int h, int w,
                                 int ensemble, int in_stride, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int oh = h * scale, ow = w * scale;
    const int ox = idx % ow, oy = (idx / ow) % oh, oc = (idx / (ow * oh)) % out_ch, n = idx / (ow * oh * out_ch);
    const int iy = oy / scale, ix = ox / scale;
    const int c_sub = (oc * scale + (oy % scale)) * scale + (ox % scale);
    const int plane = out_ch * scale * scale;
    const size_t pix = (size_t(n) * h + iy) * w + ix;
    const half* row = in + pix * in_stride;
    float sum = 0.f;
    for (int k = 0; k < ensemble; ++k) {
        const int ch = k * plane + c_sub;
        sum += __half2float(row[ch]) + bias[ch];
    }
    write_value(out, idx, ensemble > 1 ? sum / float(ensemble) : sum);
}

extern "C" __global__ void gps_shuffle_nhwc_f32(float* out, const half* in, const float* bias, int out_ch, int scale,
                                                int h, int w, int ensemble, int in_stride, int total) {
    gps_shuffle_nhwc(out, in, bias, out_ch, scale, h, w, ensemble, in_stride, total);
}

extern "C" __global__ void gps_shuffle_nhwc_f16(half* out, const half* in, const float* bias, int out_ch, int scale,
                                                int h, int w, int ensemble, int in_stride, int total) {
    gps_shuffle_nhwc(out, in, bias, out_ch, scale, h, w, ensemble, in_stride, total);
}

extern "C" __global__ void dysample_nhwc(half* sampled, const half* state, const half* offset, const float* offset_bias,
                                         const half* scope, int offset_stride, int dim, int groups, int scale, int h,
                                         int w, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int group_ch = dim / groups;
    const int oh = h * scale, ow = w * scale;
    const int g = idx % groups;
    const int out_pix = idx / groups;
    const int ox = out_pix % ow, oy = (out_pix / ow) % oh, n = out_pix / (ow * oh);
    const int iy = oy / scale, ix = ox / scale, sy = oy % scale, sx = ox % scale;
    const int sub = sy * scale + sx;
    const int scale2 = scale * scale;
    const int x_ch = g * scale2 + sub;
    const int y_ch = (groups + g) * scale2 + sub;
    const size_t in_pix = (size_t(n) * h + iy) * w + ix;
    const float off_x = __half2float(offset[in_pix * offset_stride + x_ch]) + offset_bias[x_ch];
    const float scp_x = fast_sigmoid(__half2float(scope[in_pix * offset_stride + x_ch]));
    const float off_y = __half2float(offset[in_pix * offset_stride + y_ch]) + offset_bias[y_ch];
    const float scp_y = fast_sigmoid(__half2float(scope[in_pix * offset_stride + y_ch]));
    const float init_x = (float(sx) - 0.5f * float(scale - 1)) / float(scale);
    const float init_y = (float(sy) - 0.5f * float(scale - 1)) / float(scale);
    const float dx = off_x * scp_x * 0.5f + init_x;
    const float dy = off_y * scp_y * 0.5f + init_y;
    const float ux = fminf(fmaxf(float(ix) + dx, 0.f), float(w - 1));
    const float uy = fminf(fmaxf(float(iy) + dy, 0.f), float(h - 1));
    const int x0 = int(floorf(ux)), y0 = int(floorf(uy));
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float wx = ux - float(x0), wy = uy - float(y0);
    const float w00 = (1.f - wy) * (1.f - wx);
    const float w01 = (1.f - wy) * wx;
    const float w10 = wy * (1.f - wx);
    const float w11 = wy * wx;
    const size_t base = size_t(n) * h * w;
    const half* p00 = state + (base + size_t(y0) * w + x0) * dim + g * group_ch;
    const half* p01 = state + (base + size_t(y0) * w + x1) * dim + g * group_ch;
    const half* p10 = state + (base + size_t(y1) * w + x0) * dim + g * group_ch;
    const half* p11 = state + (base + size_t(y1) * w + x1) * dim + g * group_ch;
    half* dst = sampled + size_t(out_pix) * dim + g * group_ch;
    for (int k = 0; k < group_ch; ++k) {
        const float val = w00 * __half2float(p00[k]) + w01 * __half2float(p01[k]) + w10 * __half2float(p10[k])
                          + w11 * __half2float(p11[k]);
        dst[k] = __float2half_rn(val);
    }
}

template<class Rgb>
__device__ void unpack_rgb_nhwc(Rgb* out, const half* in, const float* bias, int out_ch, int in_stride, int oh, int ow,
                                int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t hw = size_t(oh) * ow;
    const int p = idx % hw, c = (idx / hw) % out_ch, n = idx / (hw * out_ch);
    const size_t pix = size_t(n) * hw + p;
    write_value(out, idx, __half2float(in[pix * in_stride + c]) + bias[c]);
}

extern "C" __global__ void unpack_rgb_nhwc_f32(float* out, const half* in, const float* bias, int out_ch, int in_stride,
                                               int oh, int ow, int total) {
    unpack_rgb_nhwc(out, in, bias, out_ch, in_stride, oh, ow, total);
}

extern "C" __global__ void unpack_rgb_nhwc_f16(half* out, const half* in, const float* bias, int out_ch, int in_stride,
                                               int oh, int ow, int total) {
    unpack_rgb_nhwc(out, in, bias, out_ch, in_stride, oh, ow, total);
}

extern "C" __global__ void fill_noise_f16(half* out, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) out[idx] = __float2half_rn(hash_noise_float(idx));
}

extern "C" __global__ void fill_noise_f32(float* out, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) out[idx] = hash_noise_float(idx);
}

// ---------------------------------------------------------------------------
// Pass 2: Fused Tensor Core expand/project + coalesced planar depthwise gate
// (dim = 64, hidden = 96)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint4 quad_transpose4(unsigned v0, unsigned v1, unsigned v2, unsigned v3, int q) {
    const unsigned s1 = __shfl_xor_sync(0xffffffff, (q & 1) ? v0 : v1, 1);
    if (q & 1) v0 = s1;
    else v1 = s1;
    const unsigned s3 = __shfl_xor_sync(0xffffffff, (q & 1) ? v2 : v3, 1);
    if (q & 1) v2 = s3;
    else v3 = s3;
    const unsigned t0 = __shfl_xor_sync(0xffffffff, (q & 2) ? v0 : v2, 2);
    if (q & 2) v0 = t0;
    else v2 = t0;
    const unsigned t1 = __shfl_xor_sync(0xffffffff, (q & 2) ? v1 : v3, 2);
    if (q & 2) v1 = t1;
    else v3 = t1;
    return make_uint4(v0, v1, v2, v3);
}

template<bool HalfAccumulate>
__device__ void mosr_expand_gate_impl(half* plain_gated, half* dw_v, half* dw_g, const half* norm_out,
                                      const half* packed_weight, const float* packed_bias, int h, int w, int tiles_x,
                                      int tiles_y, int total_pixels) {
    constexpr int halo_cols = 18, halo_count = 180, a_slice_halves = halo_count * 16;
    constexpr int b_tap_halves = 192 * 16, b_3tap_halves = 3 * b_tap_halves;

    half* const a_bufs = reinterpret_cast<half*>(mosr_smem);
    half* const b_bufs = a_bufs + 2 * a_slice_halves;
    const int thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;
    const int warp_m = warp & 3, warp_n = warp >> 2;
    const int tile = blockIdx.x;
    const int tx = tile % tiles_x, ty = (tile / tiles_x) % tiles_y, image = tile / (tiles_x * tiles_y);
    const int y0 = ty * 8, x0 = tx * 16;

    auto load_a = [&](int ks, half* dst) {
        const int k_base = ks * 16;
        if (thread < halo_count * 2) {
            const int p = thread >> 1, chunk = thread & 1;
            const int py = y0 - 1 + p / halo_cols, px = x0 - 1 + p % halo_cols;
            const bool valid = unsigned(py) < unsigned(h) && unsigned(px) < unsigned(w);
            const half* src = valid ? norm_out + ((size_t(image) * h + py) * w + px) * 64 + k_base + chunk * 8 : norm_out;
            copy_async(dst + swizzle16(p, chunk), src, valid);
        }
    };
    auto load_b3 = [&](int step_g, half* dst) {
        const half* base = packed_weight + size_t(step_g) * b_3tap_halves;
#pragma unroll
        for (int idx = thread; idx < b_3tap_halves / 8; idx += 384)
            copy_async_always(dst + idx * 8, base + idx * 8);
    };

    load_a(0, a_bufs);
    load_b3(0, b_bufs);
    asm volatile("cp.async.commit_group;\n");

    const int q = lane & 3;
    int step = 0;

    Accumulator<HalfAccumulate> acc[2][8];
    for (int ks = 0; ks < 4; ++ks) {
        const half* a_slice = a_bufs + (ks & 1) * a_slice_halves;
        for (int dy = 0; dy < 3; ++dy) {
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            const int cb = step & 1;
            if (step + 1 < 12) load_b3(step + 1, b_bufs + (cb ^ 1) * b_3tap_halves);
            if (dy == 0 && ks + 1 < 4) load_a(ks + 1, a_bufs + ((ks + 1) & 1) * a_slice_halves);
            if (step + 1 < 12) asm volatile("cp.async.commit_group;\n");
            const half* b3 = b_bufs + cb * b_3tap_halves;
            ++step;

            if constexpr (HalfAccumulate) {
                unsigned a_frag[2][2][4], b_frag[2][4][4];
                auto load_dx = [&](int dx, int buf) {
                    const half* b_tile = b3 + dx * b_tap_halves;
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const int p = (warp_m * 2 + mt + dy) * halo_cols + (lane & 15) + dx;
                        load_matrix_x4(a_frag[buf][mt], a_slice + swizzle16(p, lane >> 4));
                    }
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int row = 64 * warp_n + 16 * j + (lane >> 4) * 8 + (lane & 7);
                        load_matrix_x4(b_frag[buf][j], b_tile + swizzle16(row, (lane >> 3) & 1));
                    }
                };
                load_dx(0, 0);
#pragma unroll
                for (int dx = 0; dx < 3; ++dx) {
                    if (dx + 1 < 3) load_dx(dx + 1, (dx + 1) & 1);
                    const int cur = dx & 1;
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            acc[mt][2 * j].mma(a_frag[cur][mt], b_frag[cur][j][0], b_frag[cur][j][1]);
                            acc[mt][2 * j + 1].mma(a_frag[cur][mt], b_frag[cur][j][2], b_frag[cur][j][3]);
                        }
                }
            } else {
                unsigned a_frag[2][2][4];
                auto load_a_dx = [&](int dx, int buf) {
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const int p = (warp_m * 2 + mt + dy) * halo_cols + (lane & 15) + dx;
                        load_matrix_x4(a_frag[buf][mt], a_slice + swizzle16(p, lane >> 4));
                    }
                };
                load_a_dx(0, 0);
#pragma unroll
                for (int dx = 0; dx < 3; ++dx) {
                    if (dx + 1 < 3) load_a_dx(dx + 1, (dx + 1) & 1);
                    const int cur = dx & 1;
                    const half* b_tile = b3 + dx * b_tap_halves;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        unsigned b_frag[4];
                        const int row = 64 * warp_n + 16 * j + (lane >> 4) * 8 + (lane & 7);
                        load_matrix_x4(b_frag, b_tile + swizzle16(row, (lane >> 3) & 1));
#pragma unroll
                        for (int mt = 0; mt < 2; ++mt) {
                            acc[mt][2 * j].mma(a_frag[cur][mt], b_frag[0], b_frag[1]);
                            acc[mt][2 * j + 1].mma(a_frag[cur][mt], b_frag[2], b_frag[3]);
                        }
                    }
                }
            }
        }
    }

    if (warp_n == 0) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int y = y0 + warp_m * 2 + mt;
            if (y >= h) continue;
#pragma unroll
            for (int half_row = 0; half_row < 2; ++half_row) {
                const int x = x0 + (lane >> 2) + 8 * half_row;
                unsigned r[4];
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const float2 gv = acc[mt][g].pair(half_row);
                    const float2 vv = acc[mt][4 + g].pair(half_row);
                    const int off = g * 8 + 2 * q;
                    const half2 h2 = __floats2half2_rn(
                        fast_mish(gv.x + packed_bias[off]) * (vv.x + packed_bias[32 + off]),
                        fast_mish(gv.y + packed_bias[off + 1]) * (vv.y + packed_bias[32 + off + 1]));
                    r[g] = *reinterpret_cast<const unsigned*>(&h2);
                }
                const uint4 u = quad_transpose4(r[0], r[1], r[2], r[3], q);
                if (x < w) {
                    const size_t pix = (size_t(image) * h + y) * w + x;
                    *reinterpret_cast<uint4*>(plain_gated + pix * 32 + q * 8) = u;
                }
            }
        }
    } else {
        half* const dst_planar = (warp_n == 1) ? dw_v : dw_g;
        const float* const b_base = packed_bias + warp_n * 64;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int y = y0 + warp_m * 2 + mt;
            if (y >= h) continue;
#pragma unroll
            for (int half_row = 0; half_row < 2; ++half_row) {
                const int x = x0 + (lane >> 2) + 8 * half_row;
                const size_t pix = (size_t(image) * h + y) * w + x;
#pragma unroll
                for (int hi = 0; hi < 2; ++hi) {
                    unsigned r[4];
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const int nt = hi * 4 + s;
                        const float2 f = acc[mt][nt].pair(half_row);
                        const int off = nt * 8 + 2 * q;
                        const half2 h2 = __floats2half2_rn(f.x + b_base[off], f.y + b_base[off + 1]);
                        r[s] = *reinterpret_cast<const unsigned*>(&h2);
                    }
                    const uint4 u = quad_transpose4(r[0], r[1], r[2], r[3], q);
                    if (x < w) {
                        const int slice_id = hi * 4 + q;
                        *reinterpret_cast<uint4*>(dst_planar + (size_t(slice_id) * total_pixels + pix) * 8) = u;
                    }
                }
            }
        }
    }
}

extern "C" __global__ void mosr_expand_gate_fp16(half* plain_gated, half* dw_v, half* dw_g, const half* norm_out,
                                                 const half* packed_weight, const float* packed_bias, int h, int w,
                                                 int tiles_x, int tiles_y, int total_pixels) {
    mosr_expand_gate_impl<true>(plain_gated, dw_v, dw_g, norm_out, packed_weight, packed_bias, h, w, tiles_x, tiles_y,
                                total_pixels);
}

extern "C" __global__ void mosr_expand_gate_mixed(half* plain_gated, half* dw_v, half* dw_g, const half* norm_out,
                                                  const half* packed_weight, const float* packed_bias, int h, int w,
                                                  int tiles_x, int tiles_y, int total_pixels) {
    mosr_expand_gate_impl<false>(plain_gated, dw_v, dw_g, norm_out, packed_weight, packed_bias, h, w, tiles_x, tiles_y,
                                 total_pixels);
}

template<int FixedK, int TileW>
__device__ void mosr_dw_gate_planar_impl(half* dw_gated, const half* dw_v, const half* dw_g, const float* dw_weight,
                                         const float* dw_bias, int h, int w, int tiles_x, int total_pixels) {
    uint4* s_u4 = mosr_smem;
    const int tile_idx = blockIdx.x;
    const int tx_tile = tile_idx % tiles_x;
    const int ty_tile = tile_idx / tiles_x;
    const int slice = blockIdx.y;
    const int ch0 = slice << 3;
    const int image = blockIdx.z;
    const int tid = threadIdx.x;
    constexpr int nthreads = (TileW == 32) ? 256 : 128;
    constexpr int radius = FixedK >> 1;
    constexpr int smem_w = TileW + 2 * radius;
    constexpr int smem_h = 16 + 2 * radius;
    constexpr int smem_count = smem_w * smem_h;
    constexpr int taps = FixedK * FixedK;
    const int y0_tile = ty_tile * 16;
    const int x0_tile = tx_tile * TileW;
    const half* slice_v = dw_v + size_t(slice) * total_pixels * 8;
    const uint4 zero_u = make_uint4(0u, 0u, 0u, 0u);
    for (int i = tid; i < smem_count; i += nthreads) {
        const int sy = i / smem_w, sx = i % smem_w;
        const int gy = y0_tile - radius + sy, gx = x0_tile - radius + sx;
        uint4 v = zero_u;
        if (unsigned(gy) < unsigned(h) && unsigned(gx) < unsigned(w)) {
            const size_t pix = (size_t(image) * h + gy) * w + gx;
            v = *reinterpret_cast<const uint4*>(slice_v + pix * 8);
        }
        s_u4[i] = v;
    }
    float4* w_s4 = reinterpret_cast<float4*>(s_u4 + smem_count);
    float* w_f = reinterpret_cast<float*>(w_s4);
    for (int i = tid; i < 8 * taps; i += nthreads) {
        const int lc = i / taps, tap = i % taps;
        w_f[tap * 8 + lc] = dw_weight[size_t(ch0 + lc) * taps + tap];
    }
    __syncthreads();

    const int lx = tid & (TileW - 1), ly_pair = tid / TileW;
    const int y0 = y0_tile + ly_pair * 2, y1 = y0 + 1, x = x0_tile + lx;
    if (y0 >= h || x >= w) return;

    float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    auto decode = [](uint4 u, float* out) {
        const half2* hp = reinterpret_cast<const half2*>(&u);
        const float2 f0 = __half22float2(hp[0]), f1 = __half22float2(hp[1]);
        const float2 f2 = __half22float2(hp[2]), f3 = __half22float2(hp[3]);
        out[0] = f0.x;
        out[1] = f0.y;
        out[2] = f1.x;
        out[3] = f1.y;
        out[4] = f2.x;
        out[5] = f2.y;
        out[6] = f3.x;
        out[7] = f3.y;
    };

    for (int kx = 0; kx < FixedK; ++kx) {
        const uint4* col = s_u4 + (ly_pair * 2) * smem_w + (lx + kx);
        float cur[8];
        decode(col[0], cur);
        if constexpr (FixedK <= 7) {
#pragma unroll
            for (int ky = 0; ky < FixedK; ++ky) {
                float nxt[8];
                decode(col[(ky + 1) * smem_w], nxt);
                const int tap = ky * FixedK + kx;
                const float4 w0 = w_s4[2 * tap];
                const float4 w1 = w_s4[2 * tap + 1];
                acc0[0] = fmaf(w0.x, cur[0], acc0[0]);
                acc0[1] = fmaf(w0.y, cur[1], acc0[1]);
                acc0[2] = fmaf(w0.z, cur[2], acc0[2]);
                acc0[3] = fmaf(w0.w, cur[3], acc0[3]);
                acc0[4] = fmaf(w1.x, cur[4], acc0[4]);
                acc0[5] = fmaf(w1.y, cur[5], acc0[5]);
                acc0[6] = fmaf(w1.z, cur[6], acc0[6]);
                acc0[7] = fmaf(w1.w, cur[7], acc0[7]);
                acc1[0] = fmaf(w0.x, nxt[0], acc1[0]);
                acc1[1] = fmaf(w0.y, nxt[1], acc1[1]);
                acc1[2] = fmaf(w0.z, nxt[2], acc1[2]);
                acc1[3] = fmaf(w0.w, nxt[3], acc1[3]);
                acc1[4] = fmaf(w1.x, nxt[4], acc1[4]);
                acc1[5] = fmaf(w1.y, nxt[5], acc1[5]);
                acc1[6] = fmaf(w1.z, nxt[6], acc1[6]);
                acc1[7] = fmaf(w1.w, nxt[7], acc1[7]);
#pragma unroll
                for (int j = 0; j < 8; ++j) cur[j] = nxt[j];
            }
        } else {
            for (int ky = 0; ky < FixedK; ++ky) {
                float nxt[8];
                decode(col[(ky + 1) * smem_w], nxt);
                const int tap = ky * FixedK + kx;
                const float4 w0 = w_s4[2 * tap];
                const float4 w1 = w_s4[2 * tap + 1];
                acc0[0] = fmaf(w0.x, cur[0], acc0[0]);
                acc0[1] = fmaf(w0.y, cur[1], acc0[1]);
                acc0[2] = fmaf(w0.z, cur[2], acc0[2]);
                acc0[3] = fmaf(w0.w, cur[3], acc0[3]);
                acc0[4] = fmaf(w1.x, cur[4], acc0[4]);
                acc0[5] = fmaf(w1.y, cur[5], acc0[5]);
                acc0[6] = fmaf(w1.z, cur[6], acc0[6]);
                acc0[7] = fmaf(w1.w, cur[7], acc0[7]);
                acc1[0] = fmaf(w0.x, nxt[0], acc1[0]);
                acc1[1] = fmaf(w0.y, nxt[1], acc1[1]);
                acc1[2] = fmaf(w0.z, nxt[2], acc1[2]);
                acc1[3] = fmaf(w0.w, nxt[3], acc1[3]);
                acc1[4] = fmaf(w1.x, nxt[4], acc1[4]);
                acc1[5] = fmaf(w1.y, nxt[5], acc1[5]);
                acc1[6] = fmaf(w1.z, nxt[6], acc1[6]);
                acc1[7] = fmaf(w1.w, nxt[7], acc1[7]);
#pragma unroll
                for (int j = 0; j < 8; ++j) cur[j] = nxt[j];
            }
        }
    }

    float db[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) db[j] = dw_bias[ch0 + j];

    const size_t slice_off = size_t(slice) * total_pixels;
    const size_t pix0 = (size_t(image) * h + y0) * w + x;
    {
        const uint4 g_raw = *reinterpret_cast<const uint4*>(dw_g + (slice_off + pix0) * 8);
        const half2* g_h = reinterpret_cast<const half2*>(&g_raw);
        uint4 out_u;
        half2* out_h = reinterpret_cast<half2*>(&out_u);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 gf = __half22float2(g_h[j]);
            const int c0 = 2 * j, c1 = 2 * j + 1;
            out_h[j] = __floats2half2_rn(fast_mish(gf.x) * (acc0[c0] + db[c0]),
                                         fast_mish(gf.y) * (acc0[c1] + db[c1]));
        }
        *reinterpret_cast<uint4*>(dw_gated + (slice_off + pix0) * 8) = out_u;
    }
    if (y1 < h) {
        const size_t pix1 = pix0 + w;
        const uint4 g_raw = *reinterpret_cast<const uint4*>(dw_g + (slice_off + pix1) * 8);
        const half2* g_h = reinterpret_cast<const half2*>(&g_raw);
        uint4 out_u;
        half2* out_h = reinterpret_cast<half2*>(&out_u);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 gf = __half22float2(g_h[j]);
            const int c0 = 2 * j, c1 = 2 * j + 1;
            out_h[j] = __floats2half2_rn(fast_mish(gf.x) * (acc1[c0] + db[c0]),
                                         fast_mish(gf.y) * (acc1[c1] + db[c1]));
        }
        *reinterpret_cast<uint4*>(dw_gated + (slice_off + pix1) * 8) = out_u;
    }
}

extern "C" __global__ void mosr_dw_gate_planar_k7(half* dw_gated, const half* dw_v, const half* dw_g,
                                                  const float* dw_weight, const float* dw_bias, int h, int w,
                                                  int tiles_x, int total_pixels) {
    mosr_dw_gate_planar_impl<7, 32>(dw_gated, dw_v, dw_g, dw_weight, dw_bias, h, w, tiles_x, total_pixels);
}

extern "C" __global__ void mosr_dw_gate_planar_k17(half* dw_gated, const half* dw_v, const half* dw_g,
                                                   const float* dw_weight, const float* dw_bias, int h, int w,
                                                   int tiles_x, int total_pixels) {
    mosr_dw_gate_planar_impl<17, 16>(dw_gated, dw_v, dw_g, dw_weight, dw_bias, h, w, tiles_x, total_pixels);
}

template<bool HalfAccumulate, bool DoNorm>
__device__ void mosr_project_res_norm_impl(half* state, half* norm_out, const half* plain_gated, const half* dw_gated,
                                           const half* packed_weight, const float* fc2_bias, const float* norm_weight,
                                           const float* norm_bias, int h, int w, int tiles_x, int tiles_y,
                                           int total_pixels) {
    constexpr int halo_cols = 18, halo_count = 324, a_slice_halves = halo_count * 16;
    constexpr int b_tap_halves = 64 * 16, b_3tap_halves = 3 * b_tap_halves;

    half* const a_bufs = reinterpret_cast<half*>(mosr_smem);
    half* const b_bufs = a_bufs + 2 * a_slice_halves;
    const int thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;
    const int tile = blockIdx.x;
    const int tx = tile % tiles_x, ty = (tile / tiles_x) % tiles_y, image = tile / (tiles_x * tiles_y);
    const int y0 = ty * 16, x0 = tx * 16;

    auto load_a = [&](int ks, half* dst) {
        for (int idx = thread; idx < halo_count * 2; idx += 256) {
            const int p = idx >> 1, chunk = idx & 1;
            const int py = y0 - 1 + p / halo_cols, px = x0 - 1 + p % halo_cols;
            const bool valid = unsigned(py) < unsigned(h) && unsigned(px) < unsigned(w);
            const size_t pix = (size_t(image) * h + py) * w + px;
            const half* src;
            if (!valid) {
                src = plain_gated;
            } else if (ks < 2) {
                src = plain_gated + pix * 32 + ks * 16 + chunk * 8;
            } else {
                const int slice_id = (ks - 2) * 2 + chunk;
                src = dw_gated + (size_t(slice_id) * total_pixels + pix) * 8;
            }
            copy_async(dst + swizzle16(p, chunk), src, valid);
        }
    };
    auto load_b3 = [&](int step_g, half* dst) {
        const half* base = packed_weight + size_t(step_g) * b_3tap_halves;
#pragma unroll
        for (int idx = thread; idx < b_3tap_halves / 8; idx += 256)
            copy_async_always(dst + idx * 8, base + idx * 8);
    };

    load_a(0, a_bufs);
    load_b3(0, b_bufs);
    asm volatile("cp.async.commit_group;\n");

    const int q = lane & 3;
    int step = 0;

    Accumulator<HalfAccumulate> acc[2][8];
    for (int ks = 0; ks < 6; ++ks) {
        const half* a_slice = a_bufs + (ks & 1) * a_slice_halves;
        for (int dy = 0; dy < 3; ++dy) {
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
            const int cb = step & 1;
            if (step + 1 < 18) load_b3(step + 1, b_bufs + (cb ^ 1) * b_3tap_halves);
            if (dy == 0 && ks + 1 < 6) load_a(ks + 1, a_bufs + ((ks + 1) & 1) * a_slice_halves);
            if (step + 1 < 18) asm volatile("cp.async.commit_group;\n");
            const half* b3 = b_bufs + cb * b_3tap_halves;
            ++step;

            if constexpr (HalfAccumulate) {
                unsigned a_frag[2][2][4], b_frag[2][4][4];
                auto load_dx = [&](int dx, int buf) {
                    const half* b_tile = b3 + dx * b_tap_halves;
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const int p = (warp * 2 + mt + dy) * halo_cols + (lane & 15) + dx;
                        load_matrix_x4(a_frag[buf][mt], a_slice + swizzle16(p, lane >> 4));
                    }
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int row = 16 * j + (lane >> 4) * 8 + (lane & 7);
                        load_matrix_x4(b_frag[buf][j], b_tile + swizzle16(row, (lane >> 3) & 1));
                    }
                };
                load_dx(0, 0);
#pragma unroll
                for (int dx = 0; dx < 3; ++dx) {
                    if (dx + 1 < 3) load_dx(dx + 1, (dx + 1) & 1);
                    const int cur = dx & 1;
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            acc[mt][2 * j].mma(a_frag[cur][mt], b_frag[cur][j][0], b_frag[cur][j][1]);
                            acc[mt][2 * j + 1].mma(a_frag[cur][mt], b_frag[cur][j][2], b_frag[cur][j][3]);
                        }
                }
            } else {
                unsigned a_frag[2][2][4];
                auto load_a_dx = [&](int dx, int buf) {
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const int p = (warp * 2 + mt + dy) * halo_cols + (lane & 15) + dx;
                        load_matrix_x4(a_frag[buf][mt], a_slice + swizzle16(p, lane >> 4));
                    }
                };
                load_a_dx(0, 0);
#pragma unroll
                for (int dx = 0; dx < 3; ++dx) {
                    if (dx + 1 < 3) load_a_dx(dx + 1, (dx + 1) & 1);
                    const int cur = dx & 1;
                    const half* b_tile = b3 + dx * b_tap_halves;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        unsigned b_frag[4];
                        const int row = 16 * j + (lane >> 4) * 8 + (lane & 7);
                        load_matrix_x4(b_frag, b_tile + swizzle16(row, (lane >> 3) & 1));
#pragma unroll
                        for (int mt = 0; mt < 2; ++mt) {
                            acc[mt][2 * j].mma(a_frag[cur][mt], b_frag[0], b_frag[1]);
                            acc[mt][2 * j + 1].mma(a_frag[cur][mt], b_frag[2], b_frag[3]);
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int y = y0 + warp * 2 + mt;
        if (y >= h) continue;
#pragma unroll
        for (int half_row = 0; half_row < 2; ++half_row) {
            const int x = x0 + (lane >> 2) + 8 * half_row;
            const bool valid_x = x < w;
            const size_t pix = (size_t(image) * h + y) * w + x;
            uint4 s_raw0 = make_uint4(0u, 0u, 0u, 0u);
            uint4 s_raw1 = make_uint4(0u, 0u, 0u, 0u);
            if (valid_x) {
                s_raw0 = *reinterpret_cast<const uint4*>(state + pix * 64 + q * 8);
                s_raw1 = *reinterpret_cast<const uint4*>(state + pix * 64 + 32 + q * 8);
            }
            const uint4 s_tr0 = quad_transpose4(s_raw0.x, s_raw0.y, s_raw0.z, s_raw0.w, q);
            const uint4 s_tr1 = quad_transpose4(s_raw1.x, s_raw1.y, s_raw1.z, s_raw1.w, q);
            const unsigned s_arr[8] = {s_tr0.x, s_tr0.y, s_tr0.z, s_tr0.w, s_tr1.x, s_tr1.y, s_tr1.z, s_tr1.w};

            unsigned st_u[8];
            float2 st_f[8];
            float sum = 0.f;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const float2 f = acc[mt][nt].pair(half_row);
                const unsigned su = s_arr[nt];
                const float2 sv = __half22float2(*reinterpret_cast<const half2*>(&su));
                const int c0 = nt * 8 + 2 * q;
                const half2 h2 = __floats2half2_rn(fast_mish(f.x + fc2_bias[c0]) + (sv.x - 0.5f),
                                                   fast_mish(f.y + fc2_bias[c0 + 1]) + (sv.y - 0.5f));
                st_u[nt] = *reinterpret_cast<const unsigned*>(&h2);
                if (DoNorm) {
                    st_f[nt] = __half22float2(h2);
                    sum += st_f[nt].x + st_f[nt].y;
                }
            }
            const uint4 st_out0 = quad_transpose4(st_u[0], st_u[1], st_u[2], st_u[3], q);
            const uint4 st_out1 = quad_transpose4(st_u[4], st_u[5], st_u[6], st_u[7], q);
            if (valid_x) {
                *reinterpret_cast<uint4*>(state + pix * 64 + q * 8) = st_out0;
                *reinterpret_cast<uint4*>(state + pix * 64 + 32 + q * 8) = st_out1;
            }
            if (DoNorm) {
                const float mean = quad_sum(sum) * (1.f / 64.f);
                float var = 0.f;
#pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    const float d0 = st_f[nt].x - mean, d1 = st_f[nt].y - mean;
                    var = fmaf(d0, d0, fmaf(d1, d1, var));
                }
                const float inv = rsqrtf(quad_sum(var) * (1.f / 64.f) + 1e-6f);
                unsigned nm_u[8];
#pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    const int c0 = nt * 8 + 2 * q;
                    const float n0 = (st_f[nt].x - mean) * inv;
                    const float n1 = (st_f[nt].y - mean) * inv;
                    const half2 h2 = __floats2half2_rn(fmaf(n0, norm_weight[c0], norm_bias[c0]),
                                                       fmaf(n1, norm_weight[c0 + 1], norm_bias[c0 + 1]));
                    nm_u[nt] = *reinterpret_cast<const unsigned*>(&h2);
                }
                const uint4 nm_out0 = quad_transpose4(nm_u[0], nm_u[1], nm_u[2], nm_u[3], q);
                const uint4 nm_out1 = quad_transpose4(nm_u[4], nm_u[5], nm_u[6], nm_u[7], q);
                if (valid_x) {
                    *reinterpret_cast<uint4*>(norm_out + pix * 64 + q * 8) = nm_out0;
                    *reinterpret_cast<uint4*>(norm_out + pix * 64 + 32 + q * 8) = nm_out1;
                }
            }
        }
    }
}

extern "C" __global__ void mosr_project_res_norm_fp16(half* state, half* norm_out, const half* plain_gated,
                                                      const half* dw_gated, const half* packed_weight,
                                                      const float* fc2_bias, const float* norm_weight,
                                                      const float* norm_bias, int h, int w, int tiles_x, int tiles_y,
                                                      int total_pixels) {
    mosr_project_res_norm_impl<true, true>(state, norm_out, plain_gated, dw_gated, packed_weight, fc2_bias, norm_weight,
                                           norm_bias, h, w, tiles_x, tiles_y, total_pixels);
}

extern "C" __global__ void mosr_project_res_fp16(half* state, half* norm_out, const half* plain_gated,
                                                 const half* dw_gated, const half* packed_weight, const float* fc2_bias,
                                                 const float* norm_weight, const float* norm_bias, int h, int w,
                                                 int tiles_x, int tiles_y, int total_pixels) {
    mosr_project_res_norm_impl<true, false>(state, norm_out, plain_gated, dw_gated, packed_weight, fc2_bias,
                                            norm_weight, norm_bias, h, w, tiles_x, tiles_y, total_pixels);
}

extern "C" __global__ void mosr_project_res_norm_mixed(half* state, half* norm_out, const half* plain_gated,
                                                       const half* dw_gated, const half* packed_weight,
                                                       const float* fc2_bias, const float* norm_weight,
                                                       const float* norm_bias, int h, int w, int tiles_x, int tiles_y,
                                                       int total_pixels) {
    mosr_project_res_norm_impl<false, true>(state, norm_out, plain_gated, dw_gated, packed_weight, fc2_bias,
                                            norm_weight, norm_bias, h, w, tiles_x, tiles_y, total_pixels);
}

extern "C" __global__ void mosr_project_res_mixed(half* state, half* norm_out, const half* plain_gated,
                                                  const half* dw_gated, const half* packed_weight,
                                                  const float* fc2_bias, const float* norm_weight,
                                                  const float* norm_bias, int h, int w, int tiles_x, int tiles_y,
                                                  int total_pixels) {
    mosr_project_res_norm_impl<false, false>(state, norm_out, plain_gated, dw_gated, packed_weight, fc2_bias,
                                             norm_weight, norm_bias, h, w, tiles_x, tiles_y, total_pixels);
}
