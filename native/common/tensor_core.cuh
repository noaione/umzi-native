#pragma once

#include <cuda_fp16.h>

// Shared Ampere+ (sm_86+) PTX helpers for cp.async, ldmatrix, shared-memory
// XOR swizzling, m16n8k16 tensor-core MMA accumulators, and deterministic noise.

__device__ __forceinline__ unsigned shared_address(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

// 16-byte async copy; src_bytes 0 fills the destination with zeros.
__device__ __forceinline__ void copy_async(half* target, const half* source, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(shared_address(target)), "l"(source),
                 "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void copy_async_always(half* target, const half* source) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_address(target)), "l"(source));
}

__device__ __forceinline__ int swizzle64(int row, int chunk) { return (row * 8 + (chunk ^ (row & 7))) * 8; }
__device__ __forceinline__ int swizzle32(int row, int chunk) { return (row * 4 + (chunk ^ ((row >> 1) & 3))) * 8; }
__device__ __forceinline__ int swizzle16(int row, int chunk) { return (row * 2 + (chunk ^ ((row >> 2) & 1))) * 8; }

__device__ __forceinline__ void load_matrix_x4(unsigned (&r)[4], const half* pointer) {
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(shared_address(pointer)));
}

__device__ __forceinline__ void load_matrix_x2(unsigned (&r)[2], const half* pointer) {
    asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(shared_address(pointer)));
}

template<bool HalfAccumulate> struct Accumulator;

template<> struct Accumulator<false> {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    __device__ __forceinline__ void mma(const unsigned (&a)[4], unsigned b0, unsigned b1) {
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
            "{%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __device__ __forceinline__ float2 pair(int half_row) const {
        return make_float2(c[2 * half_row], c[2 * half_row + 1]);
    }
    __device__ __forceinline__ unsigned packed_half2(int half_row) const {
        const half2 h = __floats2half2_rn(c[2 * half_row], c[2 * half_row + 1]);
        return *reinterpret_cast<const unsigned*>(&h);
    }
};

template<> struct Accumulator<true> {
    unsigned c[2] = {0u, 0u};
    __device__ __forceinline__ void mma(const unsigned (&a)[4], unsigned b0, unsigned b1) {
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, "
            "{%0, %1};\n"
            : "+r"(c[0]), "+r"(c[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
    __device__ __forceinline__ float2 pair(int half_row) const {
        const unsigned value = c[half_row];
        return __half22float2(*reinterpret_cast<const half2*>(&value));
    }
    __device__ __forceinline__ unsigned packed_half2(int half_row) const { return c[half_row]; }
};

__device__ __forceinline__ float quad_sum(float value) {
    value += __shfl_xor_sync(0xffffffff, value, 1);
    value += __shfl_xor_sync(0xffffffff, value, 2);
    return value;
}

__device__ __forceinline__ float hash_noise_float(int index) {
    unsigned x = unsigned(index) * 2654435761u + 0x9e3779b9u;
    x ^= x >> 15;
    x *= 0x2c1b3c6du;
    x ^= x >> 12;
    return float(x & 0xffffu) / 32768.f - 1.f;
}
