#pragma once

#include "umzi_native.h"

#ifdef _WIN32
#  ifdef UMZI_BUILD
#    define UMZI_VULKAN_API __declspec(dllexport)
#  else
#    define UMZI_VULKAN_API __declspec(dllimport)
#  endif
#else
#  define UMZI_VULKAN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct UmziVulkanDevice {
    uint32_t struct_size;
    uint32_t api_version, driver_version, vendor_id, device_id, device_type, subgroup_size;
    uint32_t shader_float16, storage_buffer16, cooperative_matrix, max_storage_buffer_range;
    uint32_t max_workgroup_invocations, max_shared_memory, max_workgroup_count[3];
    char device_name[256], driver_name[256], driver_info[256];
} UmziVulkanDevice;

// Vulkan ABI v1 uses the host session functions from umzi_native.h.
// Calls on the same session must be serialized.
UMZI_VULKAN_API uint32_t umzi_vulkan_abi_version(void);
UMZI_VULKAN_API int umzi_vulkan_device_count(uint32_t* count);
UMZI_VULKAN_API int umzi_vulkan_device(uint32_t index, UmziVulkanDevice* info);

#ifdef __cplusplus
}
#endif

#undef UMZI_VULKAN_API
