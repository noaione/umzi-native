#pragma once

#include <cuda.h>
#include <cudnn.h>

#include "cuda_module.h"
#include "half.h"
#include "platform.h"
#include "../runtime/kernel.h"
#include "../runtime/plan_cache_file.h"

#include <chrono>
#include <cstdint>
#include <cstring>
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

inline uint64_t device_used_bytes() {
    size_t available = 0, total = 0;
    check(cuMemGetInfo(&available, &total), "cuMemGetInfo");
    return static_cast<uint64_t>(total - available);
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
