#include "../../common/resize_rgb.cuh"

extern "C" __global__ void pad_rgb(float* out, const float* in, int h, int w, int ph, int pw, int total) {
    resize_rgb<float, false>(out, in, h, w, ph, pw, total);
}

extern "C" __global__ void crop_rgb(float* out, const float* in, int h, int w, int ph, int pw, int total) {
    resize_rgb<float, true>(out, in, h, w, ph, pw, total);
}

// Q and K are contiguous [batch, channels, tokens]. Each norm is over
// channels at one spatial token, independent across the batch.
extern "C" __global__ void normalize_qk(
    float* query_out, float* key_out, const float* query, const float* key,
    int channels, int tokens, int total_tokens
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_tokens) return;
    const int base = (index / tokens) * channels * tokens + index % tokens;
    float query_ss = 0;
    float key_ss = 0;
    for (int channel = 0; channel < channels; ++channel) {
        const int offset = base + channel * tokens;
        const float q = query[offset];
        const float k = key[offset];
        query_ss = fmaf(q, q, query_ss);
        key_ss = fmaf(k, k, key_ss);
    }
    const float inverse_q = 1.f / sqrtf(query_ss);
    const float inverse_k = 1.f / sqrtf(key_ss);
    for (int channel = 0; channel < channels; ++channel) {
        const int offset = base + channel * tokens;
        query_out[offset] = query[offset] * inverse_q;
        key_out[offset] = key[offset] * inverse_k;
    }
}

// The GEMM product is [batch, tokens, value_channels]. Write NCHW order
// for the following gate while applying the per-token denominator.
extern "C" __global__ void finish_attention(
    float* out, const float* product, const float* value_sum, const float* contraction,
    int channels, int tokens, int total
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int plane = index / tokens;
    const int batch = plane / channels;
    const int token = index % tokens;
    const int position = batch * tokens + token;
    const float numerator = value_sum[plane] + product[position * channels + plane % channels];
    const float reciprocal = 1.f / (float(tokens) + contraction[position]);
    out[index] = numerator * reciprocal;
}
