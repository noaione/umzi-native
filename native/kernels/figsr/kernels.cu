#include <cuda_fp16.h>

namespace {
constexpr int border = 4, scale = 2, fourier = 8, hidden = 80, expansion = 2 * hidden;

__device__ float round_half(float value, bool fp16) {
    return fp16 ? __half2float(__float2half_rn(value)) : value;
}

__device__ float read_scalar(const void* source, int index, bool fp32) {
    return fp32 ? static_cast<const float*>(source)[index] : __half2float(static_cast<const half*>(source)[index]);
}

__device__ void write_scalar(void* dst, int index, float value, bool fp32) {
    if (fp32) static_cast<float*>(dst)[index] = value;
    else static_cast<half*>(dst)[index] = __float2half_rn(value);
}

__device__ int reflected(int coordinate, int extent) {
    if (coordinate < 0) return -coordinate;
    return coordinate < extent ? coordinate : 2 * extent - coordinate - 2;
}

__device__ float normalize(float x, const float* parameters, int channels, bool fp16) {
    const int lane = threadIdx.x % 32;
    float squared = x * x;
    for (int offset = 16; offset; offset /= 2) squared += __shfl_xor_sync(0xffffffff, squared, offset);
    const float norm = round_half(sqrtf(squared), fp16);
    const float denominator = round_half(fmaf(norm, round_half(parameters[2 * channels + 1], fp16),
                                               round_half(parameters[2 * channels], fp16)), fp16);
    if (lane >= channels) return 0.f;
    const float divided = round_half(x / denominator, fp16);
    return fmaf(divided, round_half(parameters[lane], fp16), round_half(parameters[channels + lane], fp16));
}
}

extern "C" __global__ void pack_rgb(void* dst, const void* source, const float* statistics,
                                     int h, int w, int ph, int pw, int count, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int c = i % 8, pixel = i / 8;
    float normalized = 0.f;
    if (c < 3) {
        const int x = reflected(pixel % pw - border, w);
        const int y = reflected(pixel / pw % ph - border, h);
        const int n = pixel / (ph * pw);
        const int j = ((n * 3 + c) * h + y) * w + x;
        const float value = read_scalar(source, j, !fp16);
        const float shifted = round_half(value - round_half(statistics[c], fp16), fp16);
        normalized = shifted / round_half(statistics[c + 3], fp16);
    }
    write_scalar(dst, i, normalized, fp32);
}

extern "C" __global__ void bias_residual(void* dst, const void* convolution, const float* bias,
                                         const void* residual, int channels, int count, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float value = read_scalar(convolution, i, fp32) + round_half(bias[i % channels], fp16);
    if (residual) value = round_half(value, fp16) + read_scalar(residual, i, fp32);
    write_scalar(dst, i, value, fp32);
}

extern "C" __global__ void rms_norm(void* dst, const void* source, const float* parameters,
                                   int channels, int pixels, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel = i / 32, lane = i % 32;
    if (pixel >= pixels) return;
    const int j = pixel * channels + lane;
    const float value = lane < channels ? read_scalar(source, j, fp32) : 0.f;
    const float normalized = normalize(value, parameters, channels, fp16);
    if (lane < channels) write_scalar(dst, j, normalized, fp32);
}

extern "C" __global__ void copy_channels(void* dst, const void* source, int dst_stride, int source_stride,
                                        int dst_offset, int source_offset, int width, int count, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int pixel = i / width, channel = i % width;
    const int from = pixel * source_stride + source_offset + channel;
    const int to = pixel * dst_stride + dst_offset + channel;
    if (fp32) static_cast<float*>(dst)[to] = static_cast<const float*>(source)[from];
    else static_cast<half*>(dst)[to] = static_cast<const half*>(source)[from];
}

extern "C" __global__ void extract_fourier(float* dst, const void* expanded, int area, int count, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int c = i / area % fourier, p = i / (area * fourier) * area + i % area;
    dst[i] = read_scalar(expanded, p * expansion + expansion - 4 * fourier + c, fp32);
}

extern "C" __global__ void spectrum_norm(void* dst, const float2* source, const float* parameters,
                                         int bins, int pixels, float ortho, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = i / 32, lane = i % 32;
    if (p >= pixels) return;
    float value = 0.f;
    if (lane < 2 * fourier) {
        // Resselt packs all real channels before all imaginary channels.
        const float2 bin = source[(p / bins * fourier + lane % fourier) * bins + p % bins];
        value = round_half((lane < fourier ? bin.x : bin.y) * ortho, fp16);
    }
    const float normalized = normalize(value, parameters, 2 * fourier, fp16);
    if (lane < 2 * fourier) write_scalar(dst, p * 2 * fourier + lane, normalized, fp32);
}

extern "C" __global__ void spectrum_gelu(float2* dst, const void* source, const float* bias,
                                         int bins, int count, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int c = i / bins % fourier, p = i / (bins * fourier) * bins + i % bins;
    const int real = 2 * c, offset = p * 2 * fourier + real;
    const auto gelu = [&](int index) {
        const float x = round_half(read_scalar(source, offset + index, fp32) + round_half(bias[real + index], fp16), fp16);
        return round_half(x * 0.5f * (1.f + erff(x * 0.7071067811865475f)), fp16);
    };
    // The inverse uses adjacent real/imaginary pairs from the learned projection.
    dst[i] = make_float2(gelu(0), gelu(1));
}

extern "C" __global__ void inverse_norm(void* dst, const float* source, const float* parameters,
                                        int area, int pixels, float ortho, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = i / 32, lane = i % 32;
    if (p >= pixels) return;
    float value = 0.f;
    if (lane < fourier) {
        value = round_half(source[(p / area * fourier + lane) * area + p % area] * ortho, fp16);
    }
    const float normalized = normalize(value, parameters, fourier, fp16);
    if (lane < fourier) write_scalar(dst, p * hidden + hidden - 4 * fourier + lane, normalized, fp32);
}

extern "C" __global__ void apply_gate(void* value, const void* expanded, int count, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = read_scalar(expanded, i / hidden * expansion + i % hidden, fp32);
    const float silu = round_half(g / (1.f + expf(-g)), fp16);
    write_scalar(value, i, read_scalar(value, i, fp32) * silu, fp32);
}

extern "C" __global__ void shuffle_rgb(void* dst, const void* source, const float* bias, const float* statistics,
                                        int h, int w, int ph, int pw, int count, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int x = i % (w * scale), row = i / (w * scale);
    const int y = row % (h * scale), c = row / (h * scale) % 3, n = row / (3 * h * scale);
    const int lane = c * scale * scale + y % scale * scale + x % scale;
    const int p = (n * ph + y / scale + border) * pw + x / scale + border;
    const float value = round_half(read_scalar(source, p * 16 + lane, fp32) + round_half(bias[lane], fp16), fp16);
    const float scaled = round_half(__fmul_rn(value, round_half(statistics[c + 3], fp16)), fp16);
    const float shifted = scaled + round_half(statistics[c], fp16);
    write_scalar(dst, i, shifted, !fp16);
}
