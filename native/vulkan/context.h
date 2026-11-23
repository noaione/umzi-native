#pragma once

#include "../include/umzi_native_vulkan.h"
#include <vulkan/vulkan.h>

#include <stdexcept>
#include <vector>

namespace umzi::vk {
struct AllocationError : std::runtime_error { using std::runtime_error::runtime_error; };

void check(VkResult result, const char* operation);

struct Adapter {
    VkPhysicalDevice handle;
    uint32_t family;
    UmziVulkanDevice info;
};

class Context {
public:
    VkInstance instance{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool pool{};
    VkFence fence{};
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceLimits limits{};

    Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context();

    void open_instance();
    std::vector<Adapter> adapters() const;
    void open_device(const Adapter& adapter);
    VkCommandBuffer allocate_command() const;
    void submit(VkCommandBuffer command) const;
};

class Buffer {
    VkDevice owner{};
    VkDeviceMemory memory{};
    bool coherent{};

public:
    VkBuffer handle{};
    VkDeviceSize bytes{};
    void* mapping{};

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer();

    void allocate(const Context& context, VkDeviceSize size, VkBufferUsageFlags usage, bool host_visible);
    void flush() const;
    void invalidate() const;
};
}
