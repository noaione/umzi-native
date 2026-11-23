#include "context.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace umzi::vk {
namespace {
constexpr auto portability_subset = "VK_KHR_portability_subset";

std::vector<VkExtensionProperties> extensions(VkPhysicalDevice device = {}) {
    auto get = [device](uint32_t* count, VkExtensionProperties* entries) {
        return device ? vkEnumerateDeviceExtensionProperties(device, nullptr, count, entries)
                      : vkEnumerateInstanceExtensionProperties(nullptr, count, entries);
    };
    uint32_t count = 0;
    check(get(&count, nullptr), "Enumerate Vulkan extensions");
    std::vector<VkExtensionProperties> entries(count);
    if (count) check(get(&count, entries.data()), "Read Vulkan extensions");
    entries.resize(count);
    return entries;
}

bool has_extension(const std::vector<VkExtensionProperties>& entries, const char* name) {
    return std::any_of(entries.begin(), entries.end(),
                       [name](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
}
}

void check(VkResult result, const char* operation) {
    if (result == VK_SUCCESS) return;
    auto message = std::string(operation) + " failed with VkResult " + std::to_string(result);
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY)
        throw AllocationError(message);
    throw std::runtime_error(message);
}

void Context::open_instance() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Umzi Vulkan";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    const char* portability = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    if (has_extension(extensions(), portability)) {
        info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        info.enabledExtensionCount = 1;
        info.ppEnabledExtensionNames = &portability;
    }
    const char* validation = "VK_LAYER_KHRONOS_validation";
    const char* enabled = std::getenv("UMZI_VULKAN_VALIDATION");
    if (enabled && std::strcmp(enabled, "1") == 0) {
        info.enabledLayerCount = 1;
        info.ppEnabledLayerNames = &validation;
    }
    check(vkCreateInstance(&info, nullptr, &instance), "Create Vulkan 1.2 instance");
}

std::vector<Adapter> Context::adapters() const {
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "Enumerate Vulkan devices");
    std::vector<VkPhysicalDevice> physical(count);
    if (count) check(vkEnumeratePhysicalDevices(instance, &count, physical.data()), "Read Vulkan devices");
    physical.resize(count);
    std::vector<Adapter> supported;
    for (auto handle : physical) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(handle, &properties);
        if (properties.apiVersion < VK_API_VERSION_1_2 ||
            (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
             properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
             properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)) continue;
        uint32_t queues = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(handle, &queues, nullptr);
        std::vector<VkQueueFamilyProperties> families(queues);
        vkGetPhysicalDeviceQueueFamilyProperties(handle, &queues, families.data());
        auto family = std::find_if(families.begin(), families.end(), [](const auto& q) {
            return q.queueCount && (q.queueFlags & VK_QUEUE_COMPUTE_BIT);
        });
        if (family == families.end()) continue;

        VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &driver};
        VkPhysicalDeviceProperties2 extended{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup};
        vkGetPhysicalDeviceProperties2(handle, &extended);
        VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &features12};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &features11};
        vkGetPhysicalDeviceFeatures2(handle, &features);
        const bool subgroup32 = subgroup.subgroupSize == 32
            && (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)
            && (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
        Adapter adapter{handle, static_cast<uint32_t>(family - families.begin()), family->timestampValidBits,
                        subgroup32, {}};
        auto& info = adapter.info;
        info.struct_size = sizeof(info);
        info.api_version = properties.apiVersion;
        info.driver_version = properties.driverVersion;
        info.vendor_id = properties.vendorID;
        info.device_id = properties.deviceID;
        info.device_type = properties.deviceType;
        info.subgroup_size = subgroup.subgroupSize;
        info.shader_float16 = features12.shaderFloat16;
        info.storage_buffer16 = features11.storageBuffer16BitAccess;
        info.cooperative_matrix = has_extension(extensions(handle), VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        info.max_storage_buffer_range = properties.limits.maxStorageBufferRange;
        info.max_workgroup_invocations = properties.limits.maxComputeWorkGroupInvocations;
        info.max_shared_memory = properties.limits.maxComputeSharedMemorySize;
        std::copy_n(properties.limits.maxComputeWorkGroupCount, 3, info.max_workgroup_count);
        std::copy_n(properties.deviceName, sizeof(info.device_name), info.device_name);
        std::copy_n(driver.driverName, sizeof(info.driver_name), info.driver_name);
        std::copy_n(driver.driverInfo, sizeof(info.driver_info), info.driver_info);
        supported.push_back(adapter);
    }
    return supported;
}

void Context::open_device(const Adapter& adapter, bool storage16, bool arithmetic16) {
    vkGetPhysicalDeviceMemoryProperties(adapter.handle, &memory);
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(adapter.handle, &props);
    limits = props.limits;

    const float priority = 1;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = adapter.family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &features12};
    if (storage16 || arithmetic16) {
        features12.shaderFloat16 = arithmetic16;
        features11.storageBuffer16BitAccess = storage16;
        info.pNext = &features11;
    }
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue_info;
    const char* subset = portability_subset;
    if (has_extension(extensions(adapter.handle), subset)) {
        info.enabledExtensionCount = 1;
        info.ppEnabledExtensionNames = &subset;
    }
    check(vkCreateDevice(adapter.handle, &info, nullptr, &device), "Create Vulkan device");
    vkGetDeviceQueue(device, adapter.family, 0, &queue);
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = adapter.family;
    check(vkCreateCommandPool(device, &pool_info, nullptr, &pool), "Create Vulkan command pool");
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(device, &fence_info, nullptr, &fence), "Create Vulkan fence");
}

VkCommandBuffer Context::allocate_command() const {
    VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    info.commandPool = pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    VkCommandBuffer command{};
    check(vkAllocateCommandBuffers(device, &info, &command), "Allocate Vulkan command buffer");
    return command;
}

void Context::submit(VkCommandBuffer command) const {
    check(vkResetFences(device, 1, &fence), "Reset Vulkan fence");
    VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.commandBufferCount = 1;
    info.pCommandBuffers = &command;
    check(vkQueueSubmit(queue, 1, &info, fence), "Submit Vulkan command buffer");
    check(vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max()),
          "Wait for Vulkan inference");
}

Context::~Context() {
    if (device) {
        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, pool, nullptr);
        vkDestroyDevice(device, nullptr);
    }
    if (instance) vkDestroyInstance(instance, nullptr);
}

void Buffer::allocate(const Context& context, VkDeviceSize size, VkBufferUsageFlags usage, bool host_visible) {
    owner = context.device;
    bytes = size;
    VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    create.size = size;
    create.usage = usage;
    check(vkCreateBuffer(owner, &create, nullptr, &handle), "Create Vulkan buffer");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(owner, handle, &req);
    auto required = host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t index = context.memory.memoryTypeCount;
    for (uint32_t i = 0; i < context.memory.memoryTypeCount; ++i) {
        auto flags = context.memory.memoryTypes[i].propertyFlags;
        if (!(req.memoryTypeBits & (1u << i)) || !(flags & required)) continue;
        if (index == context.memory.memoryTypeCount) index = i;
        if (!host_visible || (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            index = i;
            break;
        }
    }
    if (index == context.memory.memoryTypeCount)
        throw AllocationError("Vulkan device has no compatible memory type for the requested buffer");
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = req.size;
    allocate.memoryTypeIndex = index;
    check(vkAllocateMemory(owner, &allocate, nullptr, &memory), "Allocate Vulkan buffer memory");
    check(vkBindBufferMemory(owner, handle, memory, 0), "Bind Vulkan buffer memory");
    coherent = context.memory.memoryTypes[index].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (host_visible)
        check(vkMapMemory(owner, memory, 0, VK_WHOLE_SIZE, 0, &mapping), "Map Vulkan staging memory");
}

void Buffer::flush() const {
    if (coherent) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory, 0, VK_WHOLE_SIZE};
    check(vkFlushMappedMemoryRanges(owner, 1, &range), "Flush Vulkan staging memory");
}

void Buffer::invalidate() const {
    if (coherent) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory, 0, VK_WHOLE_SIZE};
    check(vkInvalidateMappedMemoryRanges(owner, 1, &range), "Invalidate Vulkan staging memory");
}

Buffer::~Buffer() {
    if (mapping) vkUnmapMemory(owner, memory);
    if (handle) vkDestroyBuffer(owner, handle, nullptr);
    if (memory) vkFreeMemory(owner, memory, nullptr);
}
}
