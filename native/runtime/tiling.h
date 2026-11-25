#pragma once

#include "kernel.h"
#include <cuda.h>
#include <functional>

namespace umzi {
using OpenTileRunner = std::function<std::unique_ptr<KernelSession>()>;
struct PageTiler {
    CUdevice device{};
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUstream stream = nullptr;
    CUfunction extract = nullptr, copy = nullptr, blend = nullptr, finish = nullptr;
    void* staging = nullptr;
    size_t staging_bytes = 0;
    struct EdgeRunner { UmziShape shape; std::unique_ptr<KernelSession> kernel; };
    std::vector<EdgeRunner> edges;

    explicit PageTiler(int32_t device_id);
    PageTiler(const PageTiler&) = delete;
    PageTiler& operator=(const PageTiler&) = delete;
    ~PageTiler();
    void* host_staging(size_t bytes);
    KernelSession& edge_runner(const UmziShape& shape, const OpenTileRunner& open);
};

void run_tiled(KernelSession& kernel, std::unique_ptr<PageTiler>& tiler, int32_t device_id,
               uint32_t height, uint32_t width, const UmziTilingOptions& options,
               const uint8_t* input, uint64_t input_bytes, void* output, uint64_t output_bytes,
               const OpenTileRunner& open_edge = {});
}
