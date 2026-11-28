#include <cuda_fp16.h>
#include "../../common/tensor_core.cuh"

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

extern "C" __global__ void dw_gate_nchw_f32(float* gated, const float* projected, const float* fc1_bias,
                                            const float* dw_weight, const float* dw_bias, int dim, int hidden,
                                            int ksize, int h, int w, int tiles_x) {
    extern __shared__ float smem[];
    const int tile_idx = blockIdx.x;
    const int tx_tile = tile_idx % tiles_x;
    const int ty_tile = tile_idx / tiles_x;
    const int ch = blockIdx.y;
    const int image = blockIdx.z;
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int radius = ksize >> 1;
    const int smem_w = 16 + 2 * radius;
    const int y0 = ty_tile * 16 - radius;
    const int x0 = tx_tile * 16 - radius;
    const size_t hw = size_t(h) * w;
    const int in_ch = 2 * hidden - dim + ch;
    const int g_ch = (hidden - dim) + ch;
    const float* c_plane = projected + (size_t(image) * (2 * hidden) + in_ch) * hw;
    const float bias_c = fc1_bias[in_ch];
    const int smem_count = smem_w * smem_w;
    for (int i = tid; i < smem_count; i += 256) {
        const int sy = i / smem_w, sx = i % smem_w;
        const int gy = y0 + sy, gx = x0 + sx;
        smem[i] = (unsigned(gy) < unsigned(h) && unsigned(gx) < unsigned(w)) ? (c_plane[size_t(gy) * w + gx] + bias_c)
                                                                             : 0.f;
    }
    __syncthreads();
    const int y = ty_tile * 16 + ty, x = tx_tile * 16 + tx;
    if (y >= h || x >= w) return;
    const float* w_ptr = dw_weight + size_t(ch) * ksize * ksize;
    float accum = 0.f;
    for (int ky = 0; ky < ksize; ++ky) {
        const float* row = smem + (ty + ky) * smem_w + tx;
        for (int kx = 0; kx < ksize; ++kx) accum = fmaf(w_ptr[ky * ksize + kx], row[kx], accum);
    }
    accum += dw_bias[ch];
    const float g = projected[(size_t(image) * (2 * hidden) + g_ch) * hw + size_t(y) * w + x] + fc1_bias[g_ch];
    gated[(size_t(image) * hidden + g_ch) * hw + size_t(y) * w + x] = mish_exact(g) * accum;
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

extern "C" __global__ void dw_gate_nhwc(half* gated, const half* projected, const float* fc1_bias,
                                        const float* dw_weight, const float* dw_bias, int dim, int hidden,
                                        int fc1_stride, int gated_stride, int ksize, int h, int w, int tiles_x) {
    extern __shared__ float smem[];
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
