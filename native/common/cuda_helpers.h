#pragma once

#include <cuda.h>
#include <cudnn.h>

#include "cuda_module.h"
#include "platform.h"
#include "../runtime/kernel.h"
#include "../runtime/plan_cache_file.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace umzi {
using Clock = std::chrono::steady_clock;

inline double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

inline void check(CUresult status, const char* operation) {
    if (status == CUDA_ERROR_OUT_OF_MEMORY)
        throw AllocationError(std::string(operation) + ": out of memory; try a smaller tile or batch");
    check_cuda(status, operation);
}

inline void check(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + cudnnGetErrorString(status));
}

#if defined(CUBLAS_API_H_) || defined(CUBLASLT_H_)
inline void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLASLt status " + std::to_string(int(status)));
}
#endif

struct ContextScope {
    bool pushed = false;

    explicit ContextScope(CUcontext context) {
        CUcontext current = nullptr;
        check(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
        if (current != context) {
            check(cuCtxPushCurrent(context), "cuCtxPushCurrent");
            pushed = true;
        }
    }
    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;
    ~ContextScope() {
        if (pushed) {
            CUcontext previous = nullptr;
            cuCtxPopCurrent(&previous);
        }
    }
};

// Balances cuDevicePrimaryCtxRetain at scope exit, even when the context can no longer be made current.
struct PrimaryContextRelease {
    CUdevice device;
    ~PrimaryContextRelease() { cuDevicePrimaryCtxRelease(device); }
};

// Device copy of weights repacked for the unfused path after a fusion fallback.
// Shared with clones, so it outlives the session that created it.
struct DeviceAllocation {
    CUcontext context;
    CUdeviceptr pointer = 0;

    DeviceAllocation(CUcontext owner, const void* data, size_t bytes) : context(owner) {
        ContextScope scope(owner); // Allocate in the context the destructor frees from.
        check(cuMemAlloc(&pointer, bytes), "cuMemAlloc");
        try {
            check(cuMemcpyHtoD(pointer, data, bytes), "cuMemcpyHtoD");
        } catch (...) {
            cuMemFree(pointer);
            throw;
        }
    }
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
    ~DeviceAllocation() {
        if (!pointer || cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
        cuMemFree(pointer);
        CUcontext previous = nullptr;
        cuCtxPopCurrent(&previous);
    }
};

inline uint64_t device_used_bytes() {
    size_t available = 0, total = 0;
    check(cuMemGetInfo(&available, &total), "cuMemGetInfo");
    return static_cast<uint64_t>(total - available);
}

// IEEE binary16 with round-to-nearest-even, matching __float2half_rn.
inline uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) return uint16_t(sign | 0x7c00u | (magnitude > 0x7f800000u ? 0x200u : 0u));
    if (magnitude >= 0x477ff000u) return uint16_t(sign | 0x7c00u);
    if (magnitude < 0x38800000u) {
        if (magnitude < 0x33000000u) return uint16_t(sign);
        const uint32_t exponent = magnitude >> 23;
        const uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
        const uint32_t shift = 126 - exponent;
        uint32_t result = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        if (rest > halfway || (rest == halfway && (result & 1u))) ++result;
        return uint16_t(sign | result);
    }
    uint32_t result = (magnitude - 0x38000000u) >> 13;
    const uint32_t rest = magnitude & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (result & 1u))) ++result;
    return uint16_t(sign | result);
}

template<class Forward>
inline void capture_cuda_graph(CUstream stream, CUgraphExec& executable, Forward&& forward) {
    CUgraph graph = nullptr;
    check(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), "cuStreamBeginCapture");
    try {
        forward();
    } catch (...) {
        cuStreamEndCapture(stream, &graph);
        if (graph) cuGraphDestroy(graph);
        throw;
    }
    check(cuStreamEndCapture(stream, &graph), "cuStreamEndCapture");
    try {
        check(cuGraphInstantiate(&executable, graph, 0), "cuGraphInstantiate");
        check(cuGraphUpload(executable, stream), "cuGraphUpload");
        check(cuStreamSynchronize(stream), "cuStreamSynchronize");
    } catch (...) {
        if (executable) {
            cuGraphExecDestroy(executable);
            executable = nullptr;
        }
        cuGraphDestroy(graph);
        throw;
    }
    check(cuGraphDestroy(graph), "cuGraphDestroy");
}

inline void hash_plan_environment(Sha256& hash, uint32_t family, uint32_t schema, const Digest& model,
                                  const KernelImage& image, CUdevice device) {
    hash.add(plan_format_version);
    hash.add(family);
    hash.add(schema);
    hash.add(sizeof(void*));
    hash.add(model);
    hash.add(image.data, image.size);
    CUuuid uuid{};
    check(cuDeviceGetUuid(&uuid, device), "cuDeviceGetUuid");
    hash.add(uuid);
    hash_driver_version(hash);
    hash.add(cudnnGetVersion());
}
} // namespace umzi
