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
