#include "compute.h"

#include <algorithm>

namespace umzi::vk {
void begin(VkCommandBuffer command, VkCommandBufferUsageFlags flags) {
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = flags;
    check(vkBeginCommandBuffer(command, &info), "Begin Vulkan command buffer");
}

void barrier(VkCommandBuffer command, VkPipelineStageFlags src, VkPipelineStageFlags dst,
             VkAccessFlags read_from, VkAccessFlags write_to) {
    VkMemoryBarrier dependency{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, read_from, write_to};
    vkCmdPipelineBarrier(command, src, dst, 0, 1, &dependency, 0, nullptr, 0, nullptr);
}

void dispatch(const Context& context, VkCommandBuffer command, uint32_t groups) {
    if (!groups) throw std::runtime_error("A Vulkan dispatch requires at least one workgroup");
    const auto width = std::min<uint32_t>(groups, context.limits.maxComputeWorkGroupCount[0]);
    const auto height = (groups - 1) / width + 1;
    if (height > context.limits.maxComputeWorkGroupCount[1])
        throw std::runtime_error("Vulkan dispatch exceeds Vulkan workgroup count limits");
    vkCmdDispatch(command, width, height, 1);
}

VkPipeline create_pipeline(VkDevice device, VkPipelineLayout layout, VkPipelineCache cache,
                           std::span<const uint32_t> code, const VkSpecializationInfo* specialization) {
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.codeSize = code.size_bytes();
    shader.pCode = code.data();
    VkShaderModule module{};
    check(vkCreateShaderModule(device, &shader, nullptr, &module), "Create Vulkan shader module");
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.stage.pSpecializationInfo = specialization;
    info.layout = layout;
    VkPipeline pipeline{};
    auto result = vkCreateComputePipelines(device, cache, 1, &info, nullptr, &pipeline);
    vkDestroyShaderModule(device, module, nullptr);
    if (result != VK_SUCCESS && pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    check(result, "Create Vulkan compute pipeline");
    return pipeline;
}
}
