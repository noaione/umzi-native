#include <cuda_fp16.h>

// NCHW operations with explicit FP16 storage and FP32 arithmetic.
__device__ int index_1d() { return blockIdx.x * blockDim.x + threadIdx.x; }

using Half = unsigned short;
__device__ float load(float x) { return x; }
__device__ float load(Half x) {
    float y;
    asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(x));
    return y;
}
template<class T> __device__ T store(float x);
template<> __device__ float store<float>(float x) { return x; }
template<> __device__ Half store<Half>(float x) {
    Half y;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(y) : "f"(x));
    return y;
}
template<class T> __device__ float rounded(float x) { return load(store<T>(x)); }

__device__ float silu(float value) {
    return value / (1.f + expf(-value));
}

// Match the FP32 erf rounding of the CUDA 13.2 Torch reference independently
// of libdevice. CUDA 13.0 erff differs by one ULP at some inputs, which the
// Fourier residual and normalization can amplify across the network.
__device__ float reference_erf(float x) {
    const float magnitude = fabsf(x);
    const bool wide = magnitude >= 0x1.00f650p+0f;
    const float t = wide ? magnitude : x * x;
    float p = wide ? 0x1.1c936ep-14f : 0x1.1b1c12p-14f;
    p = fmaf(p, t, wide ? -0x1.e78e46p-11f : -0x1.90f4fap-11f);
    p = fmaf(p, t, wide ? 0x1.c7c088p-8f : 0x1.510d82p-8f);
    p = fmaf(p, t, wide ? -0x1.336deep-5f : -0x1.b7899ep-6f);
    p = fmaf(p, t, wide ? 0x1.40e556p-3f : 0x1.ce2576p-4f);
    p = fmaf(p, t, wide ? 0x1.d431e6p-1f : -0x1.812710p-2f);
    p = fmaf(p, t, wide ? 0x1.41e7d6p-1f : 0x1.06eba8p-3f);
    float result = fmaf(p, x, x);
    if (wide) {
        float exponential;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(-fabsf(result)));
        result = copysignf(1.f - exponential, x);
    }
    return result;
}

template<bool unexact>
__device__ float gelu(float value) {
    const float argument = value * 0.70710678118654752440f;
    const float error_function = unexact ? erff(argument) : reference_erf(argument);
    return (value * 0.5f) * (1.f + error_function);
}

template<class T, class U>
__device__ void normalize(U* out, const T* in, const T* scale, const T* offset,
                                      int channels, int in_stride, int out_stride, int pixels, float factor,
                                      int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int pixel = i % pixels, image = i / pixels;
    const T* source = in + (image * in_stride) * pixels + pixel;
    float a = 0.f, b = 0.f, c = 0.f, d = 0.f;
    int channel = 0;
    for (; channel + 3 < channels; channel += 4) {
        const float v0 = load(source[channel * pixels]), v1 = load(source[(channel + 1) * pixels]);
        const float v2 = load(source[(channel + 2) * pixels]), v3 = load(source[(channel + 3) * pixels]);
        a = fmaf(v0, v0, a);
        b = fmaf(v1, v1, b);
        c = fmaf(v2, v2, c);
        d = fmaf(v3, v3, d);
    }
    if (channel < channels) {
        const float v0 = load(source[channel * pixels]);
        a = fmaf(v0, v0, a);
        if (channel + 1 < channels) {
            const float v1 = load(source[(channel + 1) * pixels]);
            b = fmaf(v1, v1, b);
        }
    }
    const float sum = __fadd_rn(__fadd_rn(__fadd_rn(a, b), c), d);
    const float rms = rounded<T>(__fmul_rn(rounded<T>(sqrtf(sum)), factor));
    const float denominator = rounded<T>(__fadd_rn(rms, 1e-6f));
    U* target = out + (image * out_stride) * pixels + pixel;
    for (int j = 0; j < channels; ++j) {
        const float normalized = rounded<T>(load(source[j * pixels]) / denominator);
        const float scaled = rounded<T>(__fmul_rn(load(scale[j]), normalized));
        target[j * pixels] = store<U>(__fadd_rn(scaled, load(offset[j])));
    }
}

#define RMS_KERNEL(name, input_type, output_type) \
extern "C" __global__ void name(output_type* out, const input_type* in, const input_type* scale, \
    const input_type* offset, int channels, int in_stride, int out_stride, int pixels, float factor, int total) { \
    normalize(out, in, scale, offset, channels, in_stride, out_stride, pixels, factor, total); \
}
RMS_KERNEL(rms_norm, float, float)
RMS_KERNEL(rms_norm_half, Half, Half)
RMS_KERNEL(rms_norm_to_half, float, Half)
#undef RMS_KERNEL

template<class T>
__device__ void depthwise_impl(T* out, const T* in, const T* weight, const T* bias,
                                      int channels, int in_stride, int out_stride, int h, int w,
                                      int kh, int kw, int residual, int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int pixels = h * w, channel = i / pixels % channels, image = i / pixels / channels;
    const int y = i / w % h, x = i % w;
    const T* source = in + (image * in_stride + channel) * pixels;
    const T* kernel = weight + channel * kh * kw;
    float value = load(bias[channel]);
    for (int ky = 0; ky < kh; ++ky) {
        const int iy = y + ky - kh / 2;
        if (unsigned(iy) >= unsigned(h)) continue;
        for (int kx = 0; kx < kw; ++kx) {
            const int ix = x + kx - kw / 2;
            if (unsigned(ix) < unsigned(w)) value = fmaf(load(kernel[ky * kw + kx]), load(source[iy * w + ix]), value);
        }
    }
    const int pixel = y * w + x;
    out[(image * out_stride + channel) * pixels + pixel] = store<T>(residual ? value + load(source[pixel]) : value);
}

#define DEPTHWISE_KERNEL(name, type) \
extern "C" __global__ void name(type* out, const type* in, const type* weight, const type* bias, \
    int channels, int in_stride, int out_stride, int h, int w, int kh, int kw, int residual, int total) { \
    depthwise_impl(out, in, weight, bias, channels, in_stride, out_stride, h, w, kh, kw, residual, total); \
}
DEPTHWISE_KERNEL(depthwise, float)
DEPTHWISE_KERNEL(depthwise_half, Half)
#undef DEPTHWISE_KERNEL

extern "C" __global__ void copy_channels(float* out, const float* in, int channels, int stride, int pixels, int total) {
    const int i = index_1d();
    if (i < total) out[i] = in[i + (i / (channels * pixels)) * (stride - channels) * pixels];
}

extern "C" __global__ void copy_channels_to_float(float* out, const Half* in, int channels, int stride, int pixels, int total) {
    const int i = index_1d();
    if (i < total) out[i] = load(in[i + (i / (channels * pixels)) * (stride - channels) * pixels]);
}

template<class T>
__device__ void pack_fft_impl(T* out, const float2* in, int channels, int bins, float factor, int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int offset = i % (channels * bins), image = i / (channels * bins);
    const int dst = image * 2 * channels * bins + offset;
    const float2 pair = in[i];
    out[dst] = store<T>(pair.x * factor);
    out[dst + channels * bins] = store<T>(pair.y * factor);
}

extern "C" __global__ void pack_fft(float* out, const float2* in, int channels, int bins, float factor, int total) {
    pack_fft_impl(out, in, channels, bins, factor, total);
}
extern "C" __global__ void pack_fft_half(Half* out, const float2* in, int channels, int bins, float factor, int total) {
    pack_fft_impl(out, in, channels, bins, factor, total);
}

template<bool unexact, class T>
__device__ void unpack_fft_gelu(float2* out, const T* in, int bins, int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int base = (2 * (i / bins)) * bins + i % bins;
    out[i] = make_float2(rounded<T>(gelu<unexact>(load(in[base]))),
                         rounded<T>(gelu<unexact>(load(in[base + bins]))));
}

extern "C" __global__ void unpack_fft(float2* out, const float* in, int bins, int total) {
    unpack_fft_gelu<false>(out, in, bins, total);
}

extern "C" __global__ void unpack_fft_unexact(float2* out, const float* in, int bins, int total) {
    unpack_fft_gelu<true>(out, in, bins, total);
}
extern "C" __global__ void unpack_fft_half(float2* out, const Half* in, int bins, int total) {
    unpack_fft_gelu<false>(out, in, bins, total);
}
extern "C" __global__ void unpack_fft_half_unexact(float2* out, const Half* in, int bins, int total) {
    unpack_fft_gelu<true>(out, in, bins, total);
}

extern "C" __global__ void scale_fft(float* values, float factor, int total) {
    const int i = index_1d();
    if (i < total) values[i] *= factor;
}
extern "C" __global__ void scale_fft_to_half(Half* out, const float* values, float factor, int total) {
    const int i = index_1d();
    if (i < total) out[i] = store<Half>(values[i] * factor);
}

extern "C" __global__ void add_bias(float* out, const float* bias, int channels, int pixels, int total) {
    const int i = index_1d();
    if (i < total) out[i] += bias[i / pixels % channels];
}
extern "C" __global__ void add_bias_half(Half* out, const Half* bias, int channels, int pixels, int total) {
    const int i = index_1d();
    if (i < total) out[i] = store<Half>(load(out[i]) + load(bias[i / pixels % channels]));
}

template<class T, bool half_rounding>
__device__ void gate_impl(T* out, const T* projected, const T* branches,
                                 int hidden, int mixed, int pixels, int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int image = i / (hidden * pixels), channel = i / pixels % hidden, pixel = i % pixels;
    const T* values = projected + image * 2 * hidden * pixels;
    const int identity = hidden - mixed;
    const float value = load(channel < identity ? values[(hidden + channel) * pixels + pixel]
        : branches[(image * mixed + channel - identity) * pixels + pixel]);
    const float activation = silu(load(values[channel * pixels + pixel]));
    out[i] = store<T>((half_rounding ? rounded<T>(activation) : activation) * value);
}

#define GATE_KERNEL(name, type, round) \
extern "C" __global__ void name(type* out, const type* projected, const type* branches, \
    int hidden, int mixed, int pixels, int total) { \
    gate_impl<type, round>(out, projected, branches, hidden, mixed, pixels, total); \
}
GATE_KERNEL(gate, float, false)
GATE_KERNEL(gate_half, Half, true)
GATE_KERNEL(gate_mixed, Half, false)
#undef GATE_KERNEL

extern "C" __global__ void residual(float* state, const float* projected, const float* gamma,
                                     int channels, int pixels, int total) {
    const int i = index_1d();
    if (i < total) state[i] = __fadd_rn(__fmul_rn(silu(projected[i]), gamma[i / pixels % channels]), state[i]);
}
extern "C" __global__ void residual_half(Half* state, const Half* projected, const Half* gamma,
                                          int channels, int pixels, int total) {
    const int i = index_1d();
    if (i >= total) return;
    const float activated = rounded<Half>(silu(load(projected[i])));
    const float scaled = rounded<Half>(activated * load(gamma[i / pixels % channels]));
    state[i] = store<Half>(scaled + load(state[i]));
}
extern "C" __global__ void residual_mixed(float* state, const Half* projected, const float* gamma,
                                           int channels, int pixels, int total) {
    const int i = index_1d();
    if (i < total) state[i] = __fadd_rn(__fmul_rn(silu(load(projected[i])), gamma[i / pixels % channels]), state[i]);
}

extern "C" __global__ void apply_silu(float* values, int total) {
    const int i = index_1d();
    if (i < total) values[i] = silu(values[i]);
}
extern "C" __global__ void apply_silu_half(Half* values, int total) {
    const int i = index_1d();
    if (i < total) values[i] = store<Half>(silu(load(values[i])));
}

extern "C" __global__ void add_features(float* out, const float* first, const float* second, int total) {
    const int i = index_1d();
    if (i < total) out[i] = first[i] + second[i];
}
extern "C" __global__ void add_features_half(Half* out, const Half* first, const Half* second, int total) {
    const int i = index_1d();
    if (i < total) out[i] = store<Half>(load(first[i]) + load(second[i]));
}
extern "C" __global__ void add_features_mixed(float* out, const Half* first, const float* second, int total) {
    const int i = index_1d();
    if (i < total) out[i] = load(first[i]) + second[i];
}

extern "C" __global__ void cast_to_half(Half* out, const float* in, int total) {
    const int i = index_1d();
    if (i < total) out[i] = store<Half>(in[i]);
}
extern "C" __global__ void cast_to_float(float* out, const Half* in, int total) {
    const int i = index_1d();
    if (i < total) out[i] = load(in[i]);
}
extern "C" __global__ void add_half_residual(Half* values, const Half* skip, int total) {
    const int i = index_1d();
    if (i < total) values[i] = store<Half>(load(values[i]) + load(skip[i]));
}

template<class T, class U>
__device__ void shuffle(U* out, const T* in, int h, int w, int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int width = 4 * w, height = 4 * h;
    const int x = i % width, y = i / width % height, plane = i / (height * width);
    const int packed = 16 * plane + 4 * (y % 4) + x % 4;
    out[i] = store<U>(load(in[(packed * h + y / 4) * w + x / 4]));
}

#define SHUFFLE_KERNEL(name, input_type, output_type) \
extern "C" __global__ void name(output_type* out, const input_type* in, int h, int w, int total) { \
    shuffle(out, in, h, w, total); \
}
SHUFFLE_KERNEL(shuffle_rgb, float, float)
SHUFFLE_KERNEL(shuffle_rgb_half, Half, Half)
SHUFFLE_KERNEL(shuffle_rgb_mixed, Half, float)
#undef SHUFFLE_KERNEL

// ===========================================================================
// Optimized NHWC kernels for reduced precision (FP16 and Mixed).
// ===========================================================================

__device__ __forceinline__ float fast_silu(float value) {
    return __fdividef(value, 1.f + __expf(-value));
}

__device__ __forceinline__ float oct_sum(float value) {
    value += __shfl_xor_sync(0xffffffff, value, 1);
    value += __shfl_xor_sync(0xffffffff, value, 2);
    value += __shfl_xor_sync(0xffffffff, value, 4);
    return value;
}

__device__ __forceinline__ float read_elem(const half* p) { return __half2float(*p); }
__device__ __forceinline__ float read_elem(const float* p) { return *p; }
__device__ __forceinline__ void write_elem(half* p, float v) { *p = __float2half_rn(v); }
__device__ __forceinline__ void write_elem(float* p, float v) { *p = v; }

// Pack NCHW [N, 3, H, W] -> NHWC [N, H, W, 8] half (ch 3..7 zero-padded).
template<class Rgb>
__device__ void pack_rgb_nhwc_impl(half* out, const Rgb* in, int h, int w, int pixels) {
    const int pix = index_1d();
    if (pix >= pixels) return;
    const int hw = h * w, n = pix / hw, rem = pix % hw;
    const size_t base = size_t(n) * 3 * hw + rem;
    const float r = read_elem(in + base);
    const float g = read_elem(in + base + hw);
    const float b = read_elem(in + base + 2 * hw);
    uint4 packed;
    half2* h2 = reinterpret_cast<half2*>(&packed);
    h2[0] = __floats2half2_rn(r, g);
    h2[1] = __floats2half2_rn(b, 0.f);
    h2[2] = __floats2half2_rn(0.f, 0.f);
    h2[3] = __floats2half2_rn(0.f, 0.f);
    *reinterpret_cast<uint4*>(out + size_t(pix) * 8) = packed;
}

extern "C" __global__ void pack_rgb_nhwc_f16(half* out, const half* in, int h, int w, int pixels) {
    pack_rgb_nhwc_impl(out, in, h, w, pixels);
}
extern "C" __global__ void pack_rgb_nhwc_f32(half* out, const float* in, int h, int w, int pixels) {
    pack_rgb_nhwc_impl(out, in, h, w, pixels);
}

// 48-channel fused bias + initial/state copy + block-0 RMSNorm.
// 8 threads per pixel (4 pixels per warp): each thread owns 3 half2/float2
// pairs at channel offsets 2*lane8, 16+2*lane8, 32+2*lane8 -> 100% coalesced.
template<class StateT>
__device__ void input_bias_rms_nhwc_impl(StateT* state, StateT* initial, half* norm, const half* projected,
                                         const float* bias, const float* scale, const float* offset, int pixels) {
    const int tid = index_1d();
    const int pix = tid >> 3, l8 = tid & 7;
    if (pix >= pixels) return;
    const size_t base = size_t(pix) * 48;
    float v[6];
    float sq = 0.f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int c = k * 16 + 2 * l8;
        const float2 p = __half22float2(*reinterpret_cast<const half2*>(projected + base + c));
        const float x0 = p.x + bias[c], x1 = p.y + bias[c + 1];
        if constexpr (sizeof(StateT) == 2) {
            const half2 h = __floats2half2_rn(x0, x1);
            *reinterpret_cast<half2*>(initial + base + c) = h;
            *reinterpret_cast<half2*>(state + base + c) = h;
            const float2 r = __half22float2(h);
            v[2 * k] = r.x;
            v[2 * k + 1] = r.y;
        } else {
            const float2 f = make_float2(x0, x1);
            *reinterpret_cast<float2*>(initial + base + c) = f;
            *reinterpret_cast<float2*>(state + base + c) = f;
            v[2 * k] = x0;
            v[2 * k + 1] = x1;
        }
        sq = fmaf(v[2 * k], v[2 * k], sq);
        sq = fmaf(v[2 * k + 1], v[2 * k + 1], sq);
    }
    const float inv = __fdividef(1.f, sqrtf(oct_sum(sq)) * 0.14433756729740644f + 1e-6f);
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int c = k * 16 + 2 * l8;
        *reinterpret_cast<half2*>(norm + base + c) = __floats2half2_rn(
            fmaf(v[2 * k] * inv, scale[c], offset[c]), fmaf(v[2 * k + 1] * inv, scale[c + 1], offset[c + 1]));
    }
}

extern "C" __global__ void input_bias_rms_nhwc_f16(half* state, half* initial, half* norm, const half* projected,
                                                   const float* bias, const float* scale, const float* offset,
                                                   int pixels) {
    input_bias_rms_nhwc_impl(state, initial, norm, projected, bias, scale, offset, pixels);
}
extern "C" __global__ void input_bias_rms_nhwc_mixed(float* state, float* initial, half* norm, const half* projected,
                                                     const float* bias, const float* scale, const float* offset,
                                                     int pixels) {
    input_bias_rms_nhwc_impl(state, initial, norm, projected, bias, scale, offset, pixels);
}

// 48-channel fused project bias + SiLU + gamma residual (+ optional next RMSNorm
// or tail half cast). 8 threads per pixel.
template<class StateT, bool DoNorm, bool WriteHalfState>
__device__ void residual_rms_nhwc_impl(StateT* state, half* norm, const half* projected, const float* bias,
                                       const float* gamma, const float* scale, const float* offset, int pixels) {
    const int tid = index_1d();
    const int pix = tid >> 3, l8 = tid & 7;
    if (pix >= pixels) return;
    const size_t base = size_t(pix) * 48;
    float v[6];
    float sq = 0.f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int c = k * 16 + 2 * l8;
        const float2 p = __half22float2(*reinterpret_cast<const half2*>(projected + base + c));
        float x0, x1;
        if constexpr (sizeof(StateT) == 2) {
            const float2 s = __half22float2(*reinterpret_cast<const half2*>(state + base + c));
            const half2 h = __floats2half2_rn(fmaf(fast_silu(p.x + bias[c]), gamma[c], s.x),
                                              fmaf(fast_silu(p.y + bias[c + 1]), gamma[c + 1], s.y));
            *reinterpret_cast<half2*>(state + base + c) = h;
            const float2 r = __half22float2(h);
            x0 = r.x;
            x1 = r.y;
        } else {
            const float2 s = *reinterpret_cast<const float2*>(state + base + c);
            x0 = fmaf(fast_silu(p.x + bias[c]), gamma[c], s.x);
            x1 = fmaf(fast_silu(p.y + bias[c + 1]), gamma[c + 1], s.y);
            *reinterpret_cast<float2*>(state + base + c) = make_float2(x0, x1);
            if constexpr (WriteHalfState) {
                *reinterpret_cast<half2*>(norm + base + c) = __floats2half2_rn(x0, x1);
            }
        }
        v[2 * k] = x0;
        v[2 * k + 1] = x1;
        if constexpr (DoNorm) {
            sq = fmaf(x0, x0, sq);
            sq = fmaf(x1, x1, sq);
        }
    }
    if constexpr (DoNorm) {
        const float inv = __fdividef(1.f, sqrtf(oct_sum(sq)) * 0.14433756729740644f + 1e-6f);
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int c = k * 16 + 2 * l8;
            *reinterpret_cast<half2*>(norm + base + c) = __floats2half2_rn(
                fmaf(v[2 * k] * inv, scale[c], offset[c]), fmaf(v[2 * k + 1] * inv, scale[c + 1], offset[c + 1]));
        }
    }
}

extern "C" __global__ void residual_rms_nhwc_f16(half* state, half* norm, const half* projected, const float* bias,
                                                 const float* gamma, const float* scale, const float* offset,
                                                 int pixels) {
    residual_rms_nhwc_impl<half, true, false>(state, norm, projected, bias, gamma, scale, offset, pixels);
}
extern "C" __global__ void residual_nhwc_f16(half* state, half* norm, const half* projected, const float* bias,
                                             const float* gamma, const float* scale, const float* offset, int pixels) {
    residual_rms_nhwc_impl<half, false, false>(state, norm, projected, bias, gamma, scale, offset, pixels);
}
extern "C" __global__ void residual_rms_nhwc_mixed(float* state, half* norm, const half* projected, const float* bias,
                                                   const float* gamma, const float* scale, const float* offset,
                                                   int pixels) {
    residual_rms_nhwc_impl<float, true, false>(state, norm, projected, bias, gamma, scale, offset, pixels);
}
extern "C" __global__ void residual_to_half_nhwc_mixed(float* state, half* norm, const half* projected,
                                                       const float* bias, const float* gamma, const float* scale,
                                                       const float* offset, int pixels) {
    residual_rms_nhwc_impl<float, false, true>(state, norm, projected, bias, gamma, scale, offset, pixels);
}

// Plain 80-channel gate (20 groups of [gate x4 | val x4] at channels 0..159 of projected).
extern "C" __global__ void gate_plain_nhwc(half* gated, const half* projected, const float* expand_bias, int total) {
    const int idx = index_1d();
    if (idx >= total) return;
    const int pix = idx / 20, g = idx % 20;
    const uint4 raw = *reinterpret_cast<const uint4*>(projected + size_t(pix) * 256 + 8 * g);
    const half2* p = reinterpret_cast<const half2*>(&raw);
    const float* b = expand_bias + 8 * g;
    const float2 g01 = __half22float2(p[0]), g23 = __half22float2(p[1]);
    const float2 v01 = __half22float2(p[2]), v23 = __half22float2(p[3]);
    uint2 out;
    half2* o = reinterpret_cast<half2*>(&out);
    o[0] = __floats2half2_rn(fast_silu(g01.x + b[0]) * (v01.x + b[4]), fast_silu(g01.y + b[1]) * (v01.y + b[5]));
    o[1] = __floats2half2_rn(fast_silu(g23.x + b[2]) * (v23.x + b[6]), fast_silu(g23.y + b[3]) * (v23.y + b[7]));
    *reinterpret_cast<uint2*>(gated + size_t(pix) * 128 + 4 * g) = out;
}

// Extract 30 Fourier value channels [208..237] from projected [N, H, W, 256] + bias
// into FP32 planar [N, 30, pixels] for cuFFT R2C.
// Block = 256 threads (8 warps), 32 pixels of one image per block.
extern "C" __global__ void extract_fourier_nhwc(float* fft_in, const half* projected, const float* expand_bias,
                                                int pixels_per_image, int tiles_per_image) {
    __shared__ float smem[30][33];
    const int tile = blockIdx.x;
    const int image = tile / tiles_per_image, t_local = tile % tiles_per_image;
    const int pix0 = t_local * 32;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float bias_c = lane < 30 ? expand_bias[208 + lane] : 0.f;
    for (int p = warp; p < 32; p += 8) {
        const int pix = pix0 + p;
        if (pix < pixels_per_image && lane < 30) {
            const size_t src = (size_t(image) * pixels_per_image + pix) * 256 + 208 + lane;
            smem[lane][p] = __half2float(projected[src]) + bias_c;
        }
    }
    __syncthreads();
    const int pix = pix0 + lane;
    if (pix < pixels_per_image) {
        for (int c = warp; c < 30; c += 8) {
            fft_in[(size_t(image) * 30 + c) * pixels_per_image + pix] = smem[c][lane];
        }
    }
}

// Unified 16x16 tiled depthwise branch + SiLU gate for the 3 x 6 spatial bands
// (square 3x3, row 1x11, column 11x1). Reads values from projected[238..255] and
// gates from projected[190..207], writes 18 gated channels to gated[110..127].
extern "C" __global__ void branch_dw_gate_nhwc(half* gated, const half* projected, const float* expand_bias,
                                               const float* square, const float* square_bias, const float* row,
                                               const float* row_bias, const float* col, const float* col_bias, int h,
                                               int w, int tiles_x, int tiles_y) {
    __shared__ float tile_val[26 * 16 * 6];
    const int tid = threadIdx.x, lx = tid & 15, ly = tid >> 4;
    const int tile = blockIdx.x;
    const int tx = tile % tiles_x, ty = tile / tiles_x % tiles_y, image = tile / (tiles_x * tiles_y);
    const int x0 = tx * 16, y0 = ty * 16;
    const int y = y0 + ly, x = x0 + lx;
    const bool valid_out = (y < h && x < w);
    const size_t out_pix = (size_t(image) * h + (valid_out ? y : 0)) * w + (valid_out ? x : 0);

    auto load_band = [&](int type, int rx, int ry) {
        const int sw = 16 + 2 * rx, sh = 16 + 2 * ry, total = sw * sh;
        const int ch_off = 238 + 6 * type;
        const float b0 = expand_bias[ch_off + 0], b1 = expand_bias[ch_off + 1], b2 = expand_bias[ch_off + 2];
        const float b3 = expand_bias[ch_off + 3], b4 = expand_bias[ch_off + 4], b5 = expand_bias[ch_off + 5];
        for (int idx = tid; idx < total; idx += 256) {
            const int sy = idx / sw, sx = idx % sw;
            const int py = y0 - ry + sy, px = x0 - rx + sx;
            float* dst = tile_val + idx * 6;
            if (unsigned(py) < unsigned(h) && unsigned(px) < unsigned(w)) {
                const half* src = projected + ((size_t(image) * h + py) * w + px) * 256 + ch_off;
                const float2 p0 = __half22float2(*reinterpret_cast<const half2*>(src));
                const float2 p1 = __half22float2(*reinterpret_cast<const half2*>(src + 2));
                const float2 p2 = __half22float2(*reinterpret_cast<const half2*>(src + 4));
                dst[0] = p0.x + b0;
                dst[1] = p0.y + b1;
                dst[2] = p1.x + b2;
                dst[3] = p1.y + b3;
                dst[4] = p2.x + b4;
                dst[5] = p2.y + b5;
            } else {
                dst[0] = dst[1] = dst[2] = dst[3] = dst[4] = dst[5] = 0.f;
            }
        }
        __syncthreads();
    };

    auto apply_gate = [&](int type, const float (&acc)[6], const float* dw_bias, half2 (&out_h)[3]) {
        const int g_off = 190 + 6 * type;
        const half* gp = projected + out_pix * 256 + g_off;
        const float* gb = expand_bias + g_off;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const float2 gf = __half22float2(*reinterpret_cast<const half2*>(gp + 2 * k));
            out_h[k] = __floats2half2_rn(fast_silu(gf.x + gb[2 * k]) * (acc[2 * k] + dw_bias[2 * k]),
                                         fast_silu(gf.y + gb[2 * k + 1]) * (acc[2 * k + 1] + dw_bias[2 * k + 1]));
        }
    };

    half2 r0[3], r1[3], r2[3];

    load_band(0, 1, 1);
    if (valid_out) {
        float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                const float* sp = tile_val + ((ly + ky) * 18 + (lx + kx)) * 6;
                const int tap = ky * 3 + kx;
#pragma unroll
                for (int c = 0; c < 6; ++c) acc[c] = fmaf(square[c * 9 + tap], sp[c], acc[c]);
            }
        }
        apply_gate(0, acc, square_bias, r0);
    }
    __syncthreads();

    load_band(1, 5, 0);
    if (valid_out) {
        float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kx = 0; kx < 11; ++kx) {
            const float* sp = tile_val + (ly * 26 + (lx + kx)) * 6;
#pragma unroll
            for (int c = 0; c < 6; ++c) acc[c] = fmaf(row[c * 11 + kx], sp[c], acc[c]);
        }
        apply_gate(1, acc, row_bias, r1);
    }
    __syncthreads();

    load_band(2, 0, 5);
    if (valid_out) {
        float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ky = 0; ky < 11; ++ky) {
            const float* sp = tile_val + ((ly + ky) * 16 + lx) * 6;
#pragma unroll
            for (int c = 0; c < 6; ++c) acc[c] = fmaf(col[c * 11 + ky], sp[c], acc[c]);
        }
        apply_gate(2, acc, col_bias, r2);

        half* dst = gated + out_pix * 128 + 110;
        *reinterpret_cast<half2*>(dst + 0) = r0[0];
        *reinterpret_cast<half2*>(dst + 2) = r0[1];
        *reinterpret_cast<half2*>(dst + 4) = r0[2];
        *reinterpret_cast<half2*>(dst + 6) = r1[0];
        *reinterpret_cast<half2*>(dst + 8) = r1[1];
        *reinterpret_cast<half2*>(dst + 10) = r1[2];
        *reinterpret_cast<half2*>(dst + 12) = r2[0];
        *reinterpret_cast<half2*>(dst + 14) = r2[1];
        *reinterpret_cast<half2*>(dst + 16) = r2[2];
    }
}

// Fused pack_fft + 60-channel pre-RMSNorm -> NHWC [N, bins, 64] (ch 60..63 = 0).
// Block = 256 threads, 32 bins of one image per block.
template<class OutT>
__device__ void pack_rms_fft_nhwc_impl(OutT* out, const float2* spectrum, const float* scale, const float* offset,
                                       int bins, int tiles_per_image, float fft_scale) {
    __shared__ float smem[32][65];
    const int tile = blockIdx.x;
    const int image = tile / tiles_per_image, t_local = tile % tiles_per_image;
    const int bin0 = t_local * 32;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int bin_r = bin0 + lane;
    if (bin_r < bins) {
        for (int c = warp; c < 30; c += 8) {
            const float2 pair = spectrum[(size_t(image) * 30 + c) * bins + bin_r];
            smem[lane][c] = pair.x * fft_scale;
            smem[lane][30 + c] = pair.y * fft_scale;
        }
    }
    __syncthreads();

    // 32 bins x 8 threads per bin = 256 threads; each thread owns 8 contiguous channels.
    const int p = tid >> 3, l8 = tid & 7;
    const int bin_w = bin0 + p;
    if (bin_w >= bins) return;
    const int c0 = 8 * l8;
    float vals[8];
    float sq = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int c = c0 + j;
        if (c < 60) {
            const float v = smem[p][c];
            vals[j] = v;
            sq = fmaf(v, v, sq);
        } else {
            vals[j] = 0.f;
        }
    }
    const float inv = __fdividef(1.f, sqrtf(oct_sum(sq)) * 0.12909944487358055f + 1e-6f);
    OutT* dst = out + (size_t(image) * bins + bin_w) * 64 + c0;
    if constexpr (sizeof(OutT) == 2) {
        uint4 packed;
        half2* h2 = reinterpret_cast<half2*>(&packed);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int ca = c0 + 2 * j, cb = ca + 1;
            const float ra = ca < 60 ? fmaf(vals[2 * j] * inv, scale[ca], offset[ca]) : 0.f;
            const float rb = cb < 60 ? fmaf(vals[2 * j + 1] * inv, scale[cb], offset[cb]) : 0.f;
            h2[j] = __floats2half2_rn(ra, rb);
        }
        *reinterpret_cast<uint4*>(dst) = packed;
    } else {
        float4 f0, f1;
        float out_v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int c = c0 + j;
            out_v[j] = c < 60 ? fmaf(vals[j] * inv, scale[c], offset[c]) : 0.f;
        }
        f0 = make_float4(out_v[0], out_v[1], out_v[2], out_v[3]);
        f1 = make_float4(out_v[4], out_v[5], out_v[6], out_v[7]);
        *reinterpret_cast<float4*>(dst) = f0;
        *reinterpret_cast<float4*>(dst + 4) = f1;
    }
}

extern "C" __global__ void pack_rms_fft_nhwc_f16(half* out, const float2* spectrum, const float* scale,
                                                 const float* offset, int bins, int tiles_per_image, float fft_scale) {
    pack_rms_fft_nhwc_impl(out, spectrum, scale, offset, bins, tiles_per_image, fft_scale);
}
extern "C" __global__ void pack_rms_fft_nhwc_f32(float* out, const float2* spectrum, const float* scale,
                                                 const float* offset, int bins, int tiles_per_image, float fft_scale) {
    pack_rms_fft_nhwc_impl(out, spectrum, scale, offset, bins, tiles_per_image, fft_scale);
}

// Tiled 8x8 NHWC 60-channel 3x3 spectral depthwise + bias + residual.
// Input/output: [N, H, W_freq, 64] (ch 60..63 = 0). Block = 256 threads (4 threads/pixel x 64 pixels).
// Weights (64x9) and biases (64) are cached in shared memory with channels 60..63 zeroed.
template<class T>
__device__ void spectral_dw_res_nhwc_impl(T* out, const T* in, const float* weight, const float* bias, int h, int w,
                                          int tiles_x, int tiles_y) {
    __shared__ __align__(16) T tile_in[10 * 10 * 64];
    __shared__ __align__(16) float w_smem[9][64];
    __shared__ __align__(16) float b_smem[64];
    const int tid = threadIdx.x;
    const int tile = blockIdx.x;
    const int tx = tile % tiles_x, ty = tile / tiles_x % tiles_y, image = tile / (tiles_x * tiles_y);
    const int x0 = tx * 8, y0 = ty * 8;

    // Load 60x9 weights transposed to [tap][channel] in shared memory (zeroing ch 60..63).
    for (int idx = tid; idx < 9 * 64; idx += 256) {
        const int tap = idx >> 6, c = idx & 63;
        w_smem[tap][c] = c < 60 ? weight[c * 9 + tap] : 0.f;
    }
    if (tid < 64) b_smem[tid] = tid < 60 ? bias[tid] : 0.f;

    if constexpr (sizeof(T) == 2) {
        for (int idx = tid; idx < 100 * 8; idx += 256) {
            const int p = idx >> 3, chunk = idx & 7;
            const int py = y0 - 1 + p / 10, px = x0 - 1 + p % 10;
            uint4* dst = reinterpret_cast<uint4*>(tile_in + idx * 8);
            if (unsigned(py) < unsigned(h) && unsigned(px) < unsigned(w)) {
                *dst = *reinterpret_cast<const uint4*>(in + ((size_t(image) * h + py) * w + px) * 64 + chunk * 8);
            } else {
                *dst = make_uint4(0, 0, 0, 0);
            }
        }
    } else {
        for (int idx = tid; idx < 100 * 16; idx += 256) {
            const int p = idx >> 4, chunk = idx & 15;
            const int py = y0 - 1 + p / 10, px = x0 - 1 + p % 10;
            float4* dst = reinterpret_cast<float4*>(tile_in + idx * 4);
            if (unsigned(py) < unsigned(h) && unsigned(px) < unsigned(w)) {
                *dst = *reinterpret_cast<const float4*>(in + ((size_t(image) * h + py) * w + px) * 64 + chunk * 4);
            } else {
                *dst = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
    }
    __syncthreads();

    const int pix_local = tid >> 2, q = tid & 3;
    const int ly = pix_local >> 3, lx = pix_local & 7;
    const int y = y0 + ly, x = x0 + lx;
    if (y >= h || x >= w) return;
    const int c0 = q * 16;
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = b_smem[c0 + j];

    if constexpr (sizeof(T) == 2) {
#pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                const int tap = ky * 3 + kx;
                const T* sp = tile_in + ((ly + ky) * 10 + (lx + kx)) * 64 + c0;
                const uint4 u0 = *reinterpret_cast<const uint4*>(sp);
                const uint4 u1 = *reinterpret_cast<const uint4*>(sp + 8);
                const half2* p0 = reinterpret_cast<const half2*>(&u0);
                const half2* p1 = reinterpret_cast<const half2*>(&u1);
                const float* wt = w_smem[tap] + c0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __half22float2(p0[j]);
                    acc[2 * j] = fmaf(wt[2 * j], f.x, acc[2 * j]);
                    acc[2 * j + 1] = fmaf(wt[2 * j + 1], f.y, acc[2 * j + 1]);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __half22float2(p1[j]);
                    acc[8 + 2 * j] = fmaf(wt[8 + 2 * j], f.x, acc[8 + 2 * j]);
                    acc[8 + 2 * j + 1] = fmaf(wt[8 + 2 * j + 1], f.y, acc[8 + 2 * j + 1]);
                }
            }
        }
        const T* center = tile_in + ((ly + 1) * 10 + (lx + 1)) * 64 + c0;
        const uint4 c_u0 = *reinterpret_cast<const uint4*>(center);
        const uint4 c_u1 = *reinterpret_cast<const uint4*>(center + 8);
        const half2* cp0 = reinterpret_cast<const half2*>(&c_u0);
        const half2* cp1 = reinterpret_cast<const half2*>(&c_u1);
        uint4 o0, o1;
        half2* h0 = reinterpret_cast<half2*>(&o0);
        half2* h1 = reinterpret_cast<half2*>(&o1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 cf = __half22float2(cp0[j]);
            h0[j] = __floats2half2_rn(acc[2 * j] + cf.x, acc[2 * j + 1] + cf.y);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 cf = __half22float2(cp1[j]);
            const int ca = c0 + 8 + 2 * j;
            if (ca < 60) {
                h1[j] = __floats2half2_rn(acc[8 + 2 * j] + cf.x, acc[8 + 2 * j + 1] + cf.y);
            } else {
                h1[j] = __floats2half2_rn(0.f, 0.f);
            }
        }
        T* dst = out + ((size_t(image) * h + y) * w + x) * 64 + c0;
        *reinterpret_cast<uint4*>(dst) = o0;
        *reinterpret_cast<uint4*>(dst + 8) = o1;
    } else {
#pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
#pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                const int tap = ky * 3 + kx;
                const float* sp = tile_in + ((ly + ky) * 10 + (lx + kx)) * 64 + c0;
                const float* wt = w_smem[tap] + c0;
#pragma unroll
                for (int grp = 0; grp < 4; ++grp) {
                    const float4 s = *reinterpret_cast<const float4*>(sp + grp * 4);
                    const float4 wv = *reinterpret_cast<const float4*>(wt + grp * 4);
                    acc[grp * 4 + 0] = fmaf(wv.x, s.x, acc[grp * 4 + 0]);
                    acc[grp * 4 + 1] = fmaf(wv.y, s.y, acc[grp * 4 + 1]);
                    acc[grp * 4 + 2] = fmaf(wv.z, s.z, acc[grp * 4 + 2]);
                    acc[grp * 4 + 3] = fmaf(wv.w, s.w, acc[grp * 4 + 3]);
                }
            }
        }
        const float* center = tile_in + ((ly + 1) * 10 + (lx + 1)) * 64 + c0;
        float* dst = out + ((size_t(image) * h + y) * w + x) * 64 + c0;
#pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            if (c0 + grp * 4 < 60) {
                const float4 cf = *reinterpret_cast<const float4*>(center + grp * 4);
                *reinterpret_cast<float4*>(dst + grp * 4) =
                    make_float4(acc[grp * 4 + 0] + cf.x, acc[grp * 4 + 1] + cf.y,
                                acc[grp * 4 + 2] + cf.z, acc[grp * 4 + 3] + cf.w);
            } else {
                *reinterpret_cast<float4*>(dst + grp * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
    }
}

extern "C" __global__ void spectral_dw_res_nhwc_f16(half* out, const half* in, const float* weight, const float* bias,
                                                    int h, int w, int tiles_x, int tiles_y) {
    spectral_dw_res_nhwc_impl(out, in, weight, bias, h, w, tiles_x, tiles_y);
}
extern "C" __global__ void spectral_dw_res_nhwc_f32(float* out, const float* in, const float* weight,
                                                    const float* bias, int h, int w, int tiles_x, int tiles_y) {
    spectral_dw_res_nhwc_impl(out, in, weight, bias, h, w, tiles_x, tiles_y);
}

// Fused spectral projection bias + GELU + adjacent-pair unpack [2c, 2c+1] -> float2 spectrum.
// Block = 256 threads, 32 bins of one image per block.
template<bool unexact, class InT>
__device__ void unpack_bias_gelu_nhwc_impl(float2* spectrum, const InT* in, const float* bias, int bins,
                                           int tiles_per_image) {
    __shared__ float2 smem[30][33];
    const int tile = blockIdx.x;
    const int image = tile / tiles_per_image, t_local = tile % tiles_per_image;
    const int bin0 = t_local * 32;
    const int tid = threadIdx.x;

    // 32 bins x 8 threads/bin = 256 threads; each thread handles 4 complex channels (8 real channels).
    const int p = tid >> 3, l8 = tid & 7;
    const int bin_r = bin0 + p;
    if (bin_r < bins) {
        const int c0 = 8 * l8;
        const InT* src = in + (size_t(image) * bins + bin_r) * 64 + c0;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int k = 4 * l8 + pair;
            if (k < 30) {
                const int c = 2 * k;
                const float re = gelu<unexact>(read_elem(src + 2 * pair) + bias[c]);
                const float im = gelu<unexact>(read_elem(src + 2 * pair + 1) + bias[c + 1]);
                smem[k][p] = make_float2(re, im);
            }
        }
    }
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31;
    const int bin_w = bin0 + lane;
    if (bin_w < bins) {
        for (int k = warp; k < 30; k += 8) {
            spectrum[(size_t(image) * 30 + k) * bins + bin_w] = smem[k][lane];
        }
    }
}

extern "C" __global__ void unpack_bias_gelu_nhwc_f16(float2* spectrum, const half* in, const float* bias, int bins,
                                                     int tiles_per_image) {
    unpack_bias_gelu_nhwc_impl<false>(spectrum, in, bias, bins, tiles_per_image);
}
extern "C" __global__ void unpack_bias_gelu_nhwc_f16_unexact(float2* spectrum, const half* in, const float* bias,
                                                             int bins, int tiles_per_image) {
    unpack_bias_gelu_nhwc_impl<true>(spectrum, in, bias, bins, tiles_per_image);
}
extern "C" __global__ void unpack_bias_gelu_nhwc_f32(float2* spectrum, const float* in, const float* bias, int bins,
                                                     int tiles_per_image) {
    unpack_bias_gelu_nhwc_impl<false>(spectrum, in, bias, bins, tiles_per_image);
}
extern "C" __global__ void unpack_bias_gelu_nhwc_f32_unexact(float2* spectrum, const float* in, const float* bias,
                                                             int bins, int tiles_per_image) {
    unpack_bias_gelu_nhwc_impl<true>(spectrum, in, bias, bins, tiles_per_image);
}

// Fused scale_fft + 30-channel post-RMSNorm + Fourier SiLU gate -> gated[80..109].
// Block = 256 threads, 32 pixels of one image per block.
extern "C" __global__ void scale_post_rms_gate_nhwc(half* gated, const float* fft_out, const half* projected,
                                                    const float* expand_bias, const float* scale, const float* offset,
                                                    int pixels_per_image, int tiles_per_image, float fft_scale) {
    __shared__ float smem[32][33];
    const int tile = blockIdx.x;
    const int image = tile / tiles_per_image, t_local = tile % tiles_per_image;
    const int pix0 = t_local * 32;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int pix_r = pix0 + lane;
    if (pix_r < pixels_per_image) {
        for (int c = warp; c < 30; c += 8) {
            smem[lane][c] = fft_out[(size_t(image) * 30 + c) * pixels_per_image + pix_r] * fft_scale;
        }
    }
    __syncthreads();

    // 32 pixels x 8 threads/pixel = 256 threads; each thread owns 4 channels (30 total).
    const int p = tid >> 3, l8 = tid & 7;
    const int pix_local = pix0 + p;
    if (pix_local >= pixels_per_image) return;
    const int c0 = 4 * l8;
    float vals[4];
    float sq = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c = c0 + j;
        if (c < 30) {
            const float v = smem[p][c];
            vals[j] = v;
            sq = fmaf(v, v, sq);
        } else {
            vals[j] = 0.f;
        }
    }
    const float inv = __fdividef(1.f, sqrtf(oct_sum(sq)) * 0.18257418583505536f + 1e-6f);
    const size_t global_pix = size_t(image) * pixels_per_image + pix_local;
    const half* gp = projected + global_pix * 256 + 160 + c0;
    const float* gb = expand_bias + 160 + c0;
    half* dst = gated + global_pix * 128 + 80 + c0;
    if (c0 + 4 <= 30) {
        const float2 g01 = __half22float2(*reinterpret_cast<const half2*>(gp));
        const float2 g23 = __half22float2(*reinterpret_cast<const half2*>(gp + 2));
        uint2 packed;
        half2* h2 = reinterpret_cast<half2*>(&packed);
        h2[0] = __floats2half2_rn(fast_silu(g01.x + gb[0]) * fmaf(vals[0] * inv, scale[c0 + 0], offset[c0 + 0]),
                                  fast_silu(g01.y + gb[1]) * fmaf(vals[1] * inv, scale[c0 + 1], offset[c0 + 1]));
        h2[1] = __floats2half2_rn(fast_silu(g23.x + gb[2]) * fmaf(vals[2] * inv, scale[c0 + 2], offset[c0 + 2]),
                                  fast_silu(g23.y + gb[3]) * fmaf(vals[3] * inv, scale[c0 + 3], offset[c0 + 3]));
        *reinterpret_cast<uint2*>(dst) = packed;
    } else if (c0 < 30) {
        // l8 == 7 -> c0 == 28, two remaining channels (28, 29).
        const float2 g01 = __half22float2(*reinterpret_cast<const half2*>(gp));
        *reinterpret_cast<half2*>(dst) =
            __floats2half2_rn(fast_silu(g01.x + gb[0]) * fmaf(vals[0] * inv, scale[c0 + 0], offset[c0 + 0]),
                              fast_silu(g01.y + gb[1]) * fmaf(vals[1] * inv, scale[c0 + 1], offset[c0 + 1]));
    }
}

// Fused tail_expand bias + SiLU on NHWC [N, H, W, 96] (12 uint4 per pixel).
extern "C" __global__ void bias_silu_nhwc_96(half* values, const float* bias, int total_vec8) {
    const int idx = index_1d();
    if (idx >= total_vec8) return;
    const int c0 = (idx % 12) * 8;
    uint4 raw = *reinterpret_cast<const uint4*>(values + size_t(idx) * 8);
    half2* h2 = reinterpret_cast<half2*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(h2[j]);
        h2[j] = __floats2half2_rn(fast_silu(f.x + bias[c0 + 2 * j]), fast_silu(f.y + bias[c0 + 2 * j + 1]));
    }
    *reinterpret_cast<uint4*>(values + size_t(idx) * 8) = raw;
}

// Fused tail_project bias + global residual add (state = tail_proj + bias + initial).
// Writes half output to `out_half` for the final conv.
template<class StateT>
__device__ void tail_add_nhwc_impl(half* out_half, const half* tail_proj, const float* bias, const StateT* initial,
                                   int total_vec8) {
    const int idx = index_1d();
    if (idx >= total_vec8) return;
    const int c0 = (idx % 6) * 8;
    uint4 raw = *reinterpret_cast<const uint4*>(tail_proj + size_t(idx) * 8);
    half2* h2 = reinterpret_cast<half2*>(&raw);
    if constexpr (sizeof(StateT) == 2) {
        uint4 init_raw = *reinterpret_cast<const uint4*>(initial + size_t(idx) * 8);
        const half2* ih = reinterpret_cast<const half2*>(&init_raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 t = __half22float2(h2[j]), i = __half22float2(ih[j]);
            h2[j] = __floats2half2_rn(t.x + bias[c0 + 2 * j] + i.x, t.y + bias[c0 + 2 * j + 1] + i.y);
        }
    } else {
        const float4 i0 = *reinterpret_cast<const float4*>(initial + size_t(idx) * 8);
        const float4 i1 = *reinterpret_cast<const float4*>(initial + size_t(idx) * 8 + 4);
        const float2 f0 = __half22float2(h2[0]), f1 = __half22float2(h2[1]);
        const float2 f2 = __half22float2(h2[2]), f3 = __half22float2(h2[3]);
        h2[0] = __floats2half2_rn(f0.x + bias[c0 + 0] + i0.x, f0.y + bias[c0 + 1] + i0.y);
        h2[1] = __floats2half2_rn(f1.x + bias[c0 + 2] + i0.z, f1.y + bias[c0 + 3] + i0.w);
        h2[2] = __floats2half2_rn(f2.x + bias[c0 + 4] + i1.x, f2.y + bias[c0 + 5] + i1.y);
        h2[3] = __floats2half2_rn(f3.x + bias[c0 + 6] + i1.z, f3.y + bias[c0 + 7] + i1.w);
    }
    *reinterpret_cast<uint4*>(out_half + size_t(idx) * 8) = raw;
}

extern "C" __global__ void tail_add_nhwc_f16(half* out_half, const half* tail_proj, const float* bias,
                                             const half* initial, int total_vec8) {
    tail_add_nhwc_impl(out_half, tail_proj, bias, initial, total_vec8);
}
extern "C" __global__ void tail_add_nhwc_mixed(half* out_half, const half* tail_proj, const float* bias,
                                               const float* initial, int total_vec8) {
    tail_add_nhwc_impl(out_half, tail_proj, bias, initial, total_vec8);
}

// Fused last bias + 4x pixel shuffle from NHWC [N, H, W, 48] half -> NCHW [N, 3, 4H, 4W].
template<class OutT>
__device__ void bias_shuffle_rgb_nhwc_impl(OutT* out, const half* in, const float* bias, int h, int w, int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int width = 4 * w, height = 4 * h;
    const int x = i % width, y = i / width % height, plane = i / (height * width);
    const int image = plane / 3, rgb = plane % 3;
    const int c = 16 * rgb + 4 * (y & 3) + (x & 3);
    const size_t pix = (size_t(image) * h + (y >> 2)) * w + (x >> 2);
    write_elem(out + i, __half2float(in[pix * 48 + c]) + bias[c]);
}

extern "C" __global__ void bias_shuffle_rgb_nhwc_f16(half* out, const half* in, const float* bias, int h, int w,
                                                     int total) {
    bias_shuffle_rgb_nhwc_impl(out, in, bias, h, w, total);
}
extern "C" __global__ void bias_shuffle_rgb_nhwc_f32(float* out, const half* in, const float* bias, int h, int w,
                                                     int total) {
    bias_shuffle_rgb_nhwc_impl(out, in, bias, h, w, total);
}

// Deterministic pseudo-random fill for realistic cuDNN plan timing.
extern "C" __global__ void fill_noise_f16(half* out, int total) {
    const int i = index_1d();
    if (i >= total) return;
    unsigned x = unsigned(i) * 2654435761u + 0x9e3779b9u;
    x ^= x >> 15;
    x *= 0x2c1b3c6du;
    x ^= x >> 12;
    out[i] = __float2half_rn(float(x & 0xffff) / 32768.f - 1.f);
}
extern "C" __global__ void fill_noise_f32(float* out, int total) {
    const int i = index_1d();
    if (i >= total) return;
    unsigned x = unsigned(i) * 2654435761u + 0x9e3779b9u;
    x ^= x >> 15;
    x *= 0x2c1b3c6du;
    x ^= x >> 12;
    out[i] = float(x & 0xffff) / 32768.f - 1.f;
}
