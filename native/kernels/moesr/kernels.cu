#include "../../common/resize_rgb.cuh"

extern "C" __global__ void pad_rgb(float* out, const float* in, int h, int w, int ph, int pw, int total) {
    resize_rgb<float, false>(out, in, h, w, ph, pw, total);
}

// One thread normalizes the channels of one NCHW pixel. Consecutive threads
// read consecutive pixels of each channel. Use population variance with
// epsilon inside the square root, as in the checkpoint's LayerNorm.
extern "C" __global__ void layer_norm(float* out, const float* in, const float* weight, const float* bias,
                                       int channels, int pixels, float eps, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const size_t first = size_t(index / pixels) * channels * pixels + index % pixels;
    float mean = 0.f;
    for (int c = 0; c < channels; ++c) mean += in[first + size_t(c) * pixels];
    mean /= float(channels);
    float variance = 0.f;
    for (int c = 0; c < channels; ++c) {
        const float centered = in[first + size_t(c) * pixels] - mean;
        variance = __fadd_rn(variance, __fmul_rn(centered, centered));
    }
    const float denominator = sqrtf(variance / float(channels) + eps);
    for (int c = 0; c < channels; ++c) {
        const size_t at = first + size_t(c) * pixels;
        const float normalized = (in[at] - mean) / denominator;
        out[at] = __fadd_rn(__fmul_rn(normalized, weight[c]), bias[c]);
    }
}

__device__ float mish(float value) {
    return value * tanhf(value > 20.f ? value : log1pf(expf(value)));
}

// projected is [gate | identity | inception input]. The first part of the
// inception input also passes through, followed by square, row and column.
extern "C" __global__ void mish_gate(float* out, const float* projected, const float* square,
                                      const float* row, const float* column, int hidden, int pixels,
                                      int branch, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int plane = index / pixels, n = plane / hidden, c = plane % hidden, pixel = index % pixels;
    const size_t gate = (size_t(n) * 2 * hidden + c) * pixels + pixel;
    const int convolved = c - (hidden - 3 * branch);
    float value;
    if (convolved < 0) {
        value = projected[gate + size_t(hidden) * pixels];
    } else {
        const float* values = convolved < branch ? square : convolved < 2 * branch ? row : column;
        value = values[(size_t(n) * branch + convolved % branch) * pixels + pixel];
    }
    out[index] = mish(projected[gate]) * value;
}

extern "C" __global__ void mish_residual(float* out, const float* projected, const float* shortcut,
                                          const float* gamma, int channels, int pixels, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int c = index / pixels % channels;
    out[index] = __fadd_rn(__fmul_rn(mish(projected[index]), gamma[c]), shortcut[index]);
}

__device__ float leaky(float value) {
    return value > 0.f ? value : value * 0.1f;
}

extern "C" __global__ void unshuffle2_leaky(float* out, const float* in, int h, int w, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int plane = index / (h * w), y = index / w % h, x = index % w;
    const int subpixel = plane % 4;
    const size_t source = (size_t(plane / 4) * (2 * h) + 2 * y + subpixel / 2) * (2 * w) + 2 * x + subpixel % 2;
    out[index] = leaky(in[source]);
}

template<int Scale>
__device__ float shuffled(const float* in, int index, int h, int w, int ph, int pw) {
    const int plane = index / (h * w), y = index / w % h, x = index % w;
    const int subpixel = (y % Scale) * Scale + x % Scale;
    const size_t source = ((size_t(plane) * Scale * Scale + subpixel) * ph + y / Scale) * pw + x / Scale;
    return in[source];
}

extern "C" __global__ void shuffle2_leaky_add(float* out, const float* in, const float* residual,
                                               int ph, int pw, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) out[index] = leaky(shuffled<2>(in, index, 2 * ph, 2 * pw, ph, pw)) + residual[index];
}

extern "C" __global__ void shuffle4_crop(float* out, const float* in, int h, int w, int ph, int pw, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total) out[index] = shuffled<4>(in, index, h, w, ph, pw);
}
