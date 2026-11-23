#pragma once

#include "context.h"

#include <span>

namespace umzi::vk {
void begin(VkCommandBuffer command, VkCommandBufferUsageFlags flags = 0);
void barrier(VkCommandBuffer command, VkPipelineStageFlags src, VkPipelineStageFlags dst,
             VkAccessFlags read_from, VkAccessFlags write_to);
void dispatch(const Context& context, VkCommandBuffer command, uint32_t groups);
VkPipeline create_pipeline(VkDevice device, VkPipelineLayout layout, VkPipelineCache cache,
                           std::span<const uint32_t> code, const VkSpecializationInfo* specialization = nullptr);
}
