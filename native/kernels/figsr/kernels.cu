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

template<int channels>
__device__ float normalize(float x, const float* parameters, bool fp16) {
    const int lane = threadIdx.x % channels;
    float squared = x * x;
    const unsigned active = __activemask();
    for (int offset = channels / 2; offset; offset /= 2)
        squared += __shfl_xor_sync(active, squared, offset, channels);
    const float norm = round_half(sqrtf(squared), fp16);
    const float denominator = round_half(fmaf(norm, round_half(parameters[2 * channels + 1], fp16),
                                               round_half(parameters[2 * channels], fp16)), fp16);
    const float divided = round_half(x / denominator, fp16);
    return fmaf(divided, round_half(parameters[lane], fp16), round_half(parameters[channels + lane], fp16));
}

__device__ float biased_value(const void* ptr, int i, float bias, bool fp16, bool fp32) {
    return round_half(read_scalar(ptr, i, fp32) + round_half(bias, fp16), !fp32);
}

__device__ int planar_index(int pixel, int channel, int channels, int area) {
    return (pixel / area * channels + channel) * area + pixel % area;
}

__device__ float gate_value(const void* expanded, const float* bias, int pixel, int lane, bool fp16, bool fp32) {
    const float g = biased_value(expanded, pixel * expansion + lane, bias[lane], fp16, fp32);
    return round_half(g / (1.f + expf(-g)), fp16);
}

__device__ float gate_planar(const float* expanded, const float* bias, int pixel, int lane, int area) {
    const float g = expanded[planar_index(pixel, lane, expansion, area)] + bias[lane];
    return g / (1.f + expf(-g));
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
                                   int pixels, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i / 32 >= pixels) return;
    write_scalar(dst, i, normalize<32>(read_scalar(source, i, fp32), parameters, fp16), fp32);
}

extern "C" __global__ void residual_norm(void* state, void* projected, const float* bias,
                                         const float* parameters, int pixels, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i / 32 >= pixels) return;
    const float projection = round_half(read_scalar(projected, i, fp32) + round_half(bias[i % 32], fp16), fp16);
    const float value = round_half(projection + read_scalar(state, i, fp32), !fp32);
    write_scalar(state, i, value, fp32);
    write_scalar(projected, i, normalize<32>(value, parameters, fp16), fp32);
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

extern "C" __global__ void extract_fourier(float* dst, const void* expanded, const float* bias,
                                           int area, int count, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int c = i / area % fourier, p = i / (area * fourier) * area + i % area;
    const int lane = expansion - 4 * fourier + c;
    dst[i] = biased_value(expanded, p * expansion + lane, bias[lane], fp16, fp32);
}

extern "C" __global__ void split_spatial(void* dst, const void* expanded, const float* bias,
                                         int offset, int count, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int lane = hidden + offset + i % fourier;
    write_scalar(dst, i, biased_value(expanded, i / fourier * expansion + lane, bias[lane], fp16, fp32), fp32);
}

extern "C" __global__ void gate_spatial(void* dst, const void* source, const float* bias,
                                        const void* expanded, const float* expanded_bias,
                                        int offset, int count, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int p = i / fourier, lane = offset + i % fourier;
    const float value = biased_value(source, i, bias[i % fourier], fp16, fp32);
    write_scalar(dst, p * hidden + lane, value * gate_value(expanded, expanded_bias, p, lane, fp16, fp32), fp32);
}

extern "C" __global__ void spectrum_norm(void* dst, const float2* source, const float* parameters,
                                         int bins, int pixels, float ortho, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = i / (2 * fourier), lane = i % (2 * fourier);
    if (p >= pixels) return;
    // Resselt packs all real channels before all imaginary channels.
    const float2 bin = source[(p / bins * fourier + lane % fourier) * bins + p % bins];
    const float value = round_half((lane < fourier ? bin.x : bin.y) * ortho, fp16);
    write_scalar(dst, i, normalize<2 * fourier>(value, parameters, fp16), fp32);
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
                                        const void* expanded, const float* expanded_bias,
                                        int area, int pixels, float ortho, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = i / fourier, lane = i % fourier;
    if (p >= pixels) return;
    const float value = round_half(source[(p / area * fourier + lane) * area + p % area] * ortho, fp16);
    const float normalized = round_half(normalize<fourier>(value, parameters, fp16), !fp32);
    const int channel = hidden - 4 * fourier + lane;
    write_scalar(dst, p * hidden + channel, normalized * gate_value(expanded, expanded_bias, p, channel, fp16, fp32), fp32);
}

extern "C" __global__ void gate_identity(void* dst, const void* expanded, const float* bias,
                                         int count, bool fp16, bool fp32) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    constexpr int channels = hidden - 4 * fourier;
    const int p = i / channels, lane = i % channels;
    const float value = biased_value(expanded, p * expansion + hidden + lane, bias[hidden + lane], fp16, fp32);
    write_scalar(dst, p * hidden + lane, value * gate_value(expanded, bias, p, lane, fp16, fp32), fp32);
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

extern "C" __global__ void rms_norm_nchw(float* dst, const float* source, const float* parameters, int pixels, int area) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i / 32 >= pixels) return;
    dst[planar_index(i / 32, i % 32, 32, area)] = normalize<32>(source[i], parameters, false);
}

extern "C" __global__ void residual_norm_nchw(float* state, float* projected, const float* bias,
                                              const float* parameters, int pixels, int area) {
    // Transpose 32 pixels at a time for coalesced planar I/O and warp-per-pixel reduction.
    __shared__ float tile[32][33];
    const int base = blockIdx.x * 32, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int pixel = base + lane;
    for (int channel = warp; channel < 32; channel += 8) {
        if (pixel < pixels) tile[channel][lane] = projected[planar_index(pixel, channel, 32, area)];
    }
    __syncthreads();
    for (int column = warp; column < 32; column += 8) {
        const int p = base + column;
        if (p < pixels) {
            const float value = (tile[lane][column] + bias[lane]) + state[p * 32 + lane];
            state[p * 32 + lane] = value;
            if (parameters) tile[lane][column] = normalize<32>(value, parameters, false);
        }
    }
    if (!parameters) return;
    __syncthreads();
    for (int channel = warp; channel < 32; channel += 8) {
        if (pixel < pixels) projected[planar_index(pixel, channel, 32, area)] = tile[channel][lane];
    }
}

extern "C" __global__ void extract_fourier_nchw(float* dst, const float* expanded, const float* bias,
                                                int area, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int lane = expansion - 4 * fourier + i / area % fourier;
    const int batch = i / (area * fourier);
    dst[i] = expanded[(batch * expansion + lane) * area + i % area] + bias[lane];
}

extern "C" __global__ void split_spatial_nchw(float* dst, const float* expanded, const float* bias,
                                              int offset, int count, int area) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int lane = hidden + offset + i % fourier;
    dst[i] = expanded[planar_index(i / fourier, lane, expansion, area)] + bias[lane];
}

extern "C" __global__ void gate_spatial_nchw(float* dst, const float* source, const float* bias,
                                             const float* expanded, const float* expanded_bias,
                                             int offset, int count, int area) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int p = i / fourier, lane = offset + i % fourier;
    dst[planar_index(p, lane, hidden, area)] = (source[i] + bias[i % fourier]) *
        gate_planar(expanded, expanded_bias, p, lane, area);
}

extern "C" __global__ void inverse_norm_nchw(float* dst, const float* source, const float* parameters,
                                             const float* expanded, const float* expanded_bias,
                                             int area, int pixels, float ortho) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = i / fourier, lane = i % fourier;
    if (p >= pixels) return;
    const float value = source[planar_index(p, lane, fourier, area)] * ortho;
    const int channel = hidden - 4 * fourier + lane;
    dst[planar_index(p, channel, hidden, area)] = normalize<fourier>(value, parameters, false) *
        gate_planar(expanded, expanded_bias, p, channel, area);
}

extern "C" __global__ void gate_identity_nchw(float* dst, const float* expanded, const float* bias, int count, int area) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    constexpr int channels = hidden - 4 * fourier;
    const int p = i / (channels * area) * area + i % area, lane = i / area % channels;
    const float value = expanded[planar_index(p, hidden + lane, expansion, area)] + bias[hidden + lane];
    dst[planar_index(p, lane, hidden, area)] = value * gate_planar(expanded, bias, p, lane, area);
}
