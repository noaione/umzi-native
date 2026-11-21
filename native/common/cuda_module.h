#pragma once

#include <cuda.h>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace umzi {
struct KernelImage {
    int capability;
    const unsigned char* data;
    std::size_t size;
};

inline void check_cuda(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) return;
    const char* detail = nullptr;
    cuGetErrorString(status, &detail);
    throw std::runtime_error(std::string(operation) + ": " + (detail ? detail : "CUDA error"));
}

template<std::size_t Count>
const KernelImage& load_kernel_module(CUdevice device, CUmodule& module, const KernelImage (&cubins)[Count],
                                      const KernelImage& ptx) {
    int major = 0, minor = 0;
    check_cuda(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "CUDA capability");
    check_cuda(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "CUDA capability");
    auto capability = 10 * major + minor;
    if (capability < ptx.capability)
        throw std::runtime_error("Kernel requires CUDA compute capability "
            + std::to_string(ptx.capability / 10) + "." + std::to_string(ptx.capability % 10) + " or higher");
    const auto* image = &ptx;
    for (const auto& cubin : cubins)
        if (cubin.capability / 10 == major && cubin.capability <= capability
            && (image == &ptx || cubin.capability > image->capability))
            image = &cubin;

    auto status = cuModuleLoadData(&module, image->data);
    if (status == CUDA_ERROR_NO_BINARY_FOR_GPU && image != &ptx) {
        image = &ptx;
        status = cuModuleLoadData(&module, image->data);
    }
    if (image == &ptx && (status == CUDA_ERROR_UNSUPPORTED_PTX_VERSION || status == CUDA_ERROR_JIT_COMPILER_NOT_FOUND))
        check_cuda(status, "Could not JIT the PTX fallback; install a CUDA 13-capable NVIDIA driver with PTX JIT support");
    check_cuda(status, "Load Umzi CUDA kernels");
    return *image;
}
}
