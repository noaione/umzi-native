// FP32 NCHW operations. Preserve the checkpoint's intermediate rounding.
__device__ int index_1d() { return blockIdx.x * blockDim.x + threadIdx.x; }

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

extern "C" __global__ void rms_norm(float* out, const float* in, const float* scale, const float* offset,
                                      int channels, int in_stride, int out_stride, int pixels, float factor,
                                      int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int pixel = i % pixels, image = i / pixels;
    const float* source = in + (image * in_stride) * pixels + pixel;
    float a = 0.f, b = 0.f, c = 0.f, d = 0.f;
    int channel = 0;
    for (; channel + 3 < channels; channel += 4) {
        const float v0 = source[channel * pixels], v1 = source[(channel + 1) * pixels];
        const float v2 = source[(channel + 2) * pixels], v3 = source[(channel + 3) * pixels];
        a = fmaf(v0, v0, a);
        b = fmaf(v1, v1, b);
        c = fmaf(v2, v2, c);
        d = fmaf(v3, v3, d);
    }
    if (channel < channels) {
        const float v0 = source[channel * pixels];
        a = fmaf(v0, v0, a);
        if (channel + 1 < channels) {
            const float v1 = source[(channel + 1) * pixels];
            b = fmaf(v1, v1, b);
        }
    }
    const float sum = __fadd_rn(__fadd_rn(__fadd_rn(a, b), c), d);
    const float denominator = __fadd_rn(__fmul_rn(sqrtf(sum), factor), 1e-6f);
    float* target = out + (image * out_stride) * pixels + pixel;
    for (int j = 0; j < channels; ++j)
        target[j * pixels] = __fadd_rn(__fmul_rn(scale[j], source[j * pixels] / denominator), offset[j]);
}

extern "C" __global__ void depthwise(float* out, const float* in, const float* weight, const float* bias,
                                      int channels, int in_stride, int out_stride, int h, int w,
                                      int kh, int kw, int residual, int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int pixels = h * w, channel = i / pixels % channels, image = i / pixels / channels;
    const int y = i / w % h, x = i % w;
    const float* source = in + (image * in_stride + channel) * pixels;
    const float* kernel = weight + channel * kh * kw;
    float value = bias[channel];
    for (int ky = 0; ky < kh; ++ky) {
        const int iy = y + ky - kh / 2;
        if (unsigned(iy) >= unsigned(h)) continue;
        for (int kx = 0; kx < kw; ++kx) {
            const int ix = x + kx - kw / 2;
            if (unsigned(ix) < unsigned(w)) value = fmaf(kernel[ky * kw + kx], source[iy * w + ix], value);
        }
    }
    const int pixel = y * w + x;
    out[(image * out_stride + channel) * pixels + pixel] = residual ? value + source[pixel] : value;
}

extern "C" __global__ void copy_channels(float* out, const float* in, int channels, int stride, int pixels, int total) {
    const int i = index_1d();
    if (i < total) out[i] = in[i + (i / (channels * pixels)) * (stride - channels) * pixels];
}

extern "C" __global__ void pack_fft(float* out, const float2* in, int channels, int bins, float factor, int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int offset = i % (channels * bins), image = i / (channels * bins);
    const int dst = image * 2 * channels * bins + offset;
    const float2 pair = in[i];
    out[dst] = pair.x * factor;
    out[dst + channels * bins] = pair.y * factor;
}

template<bool unexact>
__device__ void unpack_fft_gelu(float2* out, const float* in, int bins, int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int base = (2 * (i / bins)) * bins + i % bins;
    out[i] = make_float2(gelu<unexact>(in[base]), gelu<unexact>(in[base + bins]));
}

extern "C" __global__ void unpack_fft(float2* out, const float* in, int bins, int total) {
    unpack_fft_gelu<false>(out, in, bins, total);
}

extern "C" __global__ void unpack_fft_unexact(float2* out, const float* in, int bins, int total) {
    unpack_fft_gelu<true>(out, in, bins, total);
}

extern "C" __global__ void scale_fft(float* values, float factor, int total) {
    const int i = index_1d();
    if (i < total) values[i] *= factor;
}

extern "C" __global__ void add_bias(float* out, const float* bias, int channels, int pixels, int total) {
    const int i = index_1d();
    if (i < total) out[i] += bias[i / pixels % channels];
}

extern "C" __global__ void gate(float* out, const float* projected, const float* branches,
                                 int hidden, int mixed, int pixels, int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int image = i / (hidden * pixels), channel = i / pixels % hidden, pixel = i % pixels;
    const float* values = projected + image * 2 * hidden * pixels;
    const int identity = hidden - mixed;
    const float value = channel < identity ? values[(hidden + channel) * pixels + pixel]
        : branches[(image * mixed + channel - identity) * pixels + pixel];
    out[i] = silu(values[channel * pixels + pixel]) * value;
}

extern "C" __global__ void residual(float* state, const float* projected, const float* gamma,
                                     int channels, int pixels, int total) {
    const int i = index_1d();
    if (i < total) state[i] = __fadd_rn(__fmul_rn(silu(projected[i]), gamma[i / pixels % channels]), state[i]);
}

extern "C" __global__ void apply_silu(float* values, int total) {
    const int i = index_1d();
    if (i < total) values[i] = silu(values[i]);
}

extern "C" __global__ void add_features(float* out, const float* first, const float* second, int total) {
    const int i = index_1d();
    if (i < total) out[i] = first[i] + second[i];
}

extern "C" __global__ void shuffle_rgb(float* out, const float* in, int h, int w, int total) {
    const int i = index_1d();
    if (i >= total) return;
    const int width = 4 * w, height = 4 * h;
    const int x = i % width, y = i / width % height, plane = i / (height * width);
    const int packed = 16 * plane + 4 * (y % 4) + x % 4;
    out[i] = in[(packed * h + y / 4) * w + x / 4];
}
