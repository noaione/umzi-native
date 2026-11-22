#pragma once

template<class T, bool Crop>
__device__ void resize_rgb(T* out, const T* in, int h, int w, int ph, int pw, int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    if constexpr (Crop) {
        const int plane = index / (h * w), y = index / w % h, x = index % w;
        out[index] = in[(plane * ph + y) * pw + x];
    } else {
        const int plane = index / (ph * pw);
        int y = index / pw % ph, x = index % pw;
        if (y >= h) y = ph - h < h ? 2 * h - 2 - y : h - 1;
        if (x >= w) x = pw - w < w ? 2 * w - 2 - x : w - 1;
        out[index] = in[(plane * h + y) * w + x];
    }
}
