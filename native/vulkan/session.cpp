#include "session.h"

#include <chrono>
#include <tuple>

namespace {
using umzi::vk::Buffer;
using umzi::vk::check;
using umzi::vk::Context;

constexpr uint32_t conv_code[]{
#include "conv_spv.h"
};
constexpr uint32_t element_code[]{
#include "element_spv.h"
};
constexpr uint32_t reduce_code[]{
#include "reduce_spv.h"
};
constexpr uint32_t attention_code[]{
#include "attention_spv.h"
};

struct Push {
    std::array<int, 4> offsets{}, channels{}, weights{};
    int output_offset, output_channels, h, w, batch, kind, input_h, input_w;
    int kernel_size, channel_offset, mixed, scratch;
};
static_assert(sizeof(Push) == 96);

enum PipelineIndex { convolution, elementwise, reduction, attention };
constexpr int mean_projection = 18;
constexpr auto storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
constexpr auto src_transfer = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
constexpr auto dst_transfer = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool equal(gt::Shape a, gt::Shape b) {
    return a.n == b.n && a.h == b.h && a.w == b.w;
}

void begin(VkCommandBuffer command, VkCommandBufferUsageFlags flags = 0) {
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = flags;
    check(vkBeginCommandBuffer(command, &info), "Begin Vulkan command buffer");
}

void barrier(VkCommandBuffer command, VkPipelineStageFlags src, VkPipelineStageFlags dst,
             VkAccessFlags read_from, VkAccessFlags write_to) {
    VkMemoryBarrier dependency{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, read_from, write_to};
    vkCmdPipelineBarrier(command, src, dst, 0, 1, &dependency, 0, nullptr, 0, nullptr);
}

int ceil_div(int value, int divisor) {
    return (value - 1) / divisor + 1;
}

void dispatch(const Context& context, VkCommandBuffer command, int groups) {
    const auto width = std::min<uint32_t>(groups, context.limits.maxComputeWorkGroupCount[0]);
    const auto height = (groups - 1) / width + 1;
    if (height > context.limits.maxComputeWorkGroupCount[1])
        throw std::runtime_error("GateRV3 dispatch exceeds Vulkan workgroup count limits");
    vkCmdDispatch(command, width, height, 1);
}

VkPipeline create_pipeline(VkDevice device, VkPipelineLayout layout, VkPipelineCache cache,
                           const uint32_t* code, size_t bytes) {
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.codeSize = bytes;
    shader.pCode = code;
    VkShaderModule module{};
    check(vkCreateShaderModule(device, &shader, nullptr, &module), "Create GateRV3 shader module");
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = layout;
    VkPipeline pipeline{};
    auto result = vkCreateComputePipelines(device, cache, 1, &info, nullptr, &pipeline);
    vkDestroyShaderModule(device, module, nullptr);
    if (result != VK_SUCCESS && pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    check(result, "Create GateRV3 Vulkan pipeline");
    return pipeline;
}

int padding_index(int i, int original, int padded) {
    if (i < original) return i;
    return padded - original < original ? 2 * original - 2 - i : original - 1;
}
}

struct UmziSession::Prepared {
    const Context& context;
    gt::Shape shape;
    Buffer arena, staging;
    VkDescriptorPool descriptor_pool{};
    VkCommandBuffer command{};
    size_t rgb_bytes{};
    double allocation_ms{}, recording_ms{};
    int64_t dispatch_count{};

    Prepared(const Context& context, gt::Shape shape) : context(context), shape(shape) {}
    ~Prepared() {
        if (command) vkFreeCommandBuffers(context.device, context.pool, 1, &command);
        if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
    }
};

UmziSession::UmziSession() = default;
UmziSession::~UmziSession() {
    prepared.reset();
    if (context.device) {
        for (auto pipeline : pipelines) vkDestroyPipeline(context.device, pipeline, nullptr);
        vkDestroyPipelineCache(context.device, pipeline_cache, nullptr);
        vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(context.device, descriptor_layout, nullptr);
    }
}

void UmziSession::open(const char* path, const UmziSessionOptions& options) {
    const auto started = Clock::now();
    const std::string_view utf8_path(path);
    const umzi::ModelPackage package(std::filesystem::path(std::u8string(utf8_path.begin(), utf8_path.end())));
    if (package.family != umzi::gaterv3_id)
        throw std::runtime_error("The Vulkan backend currently supports only GateRV3");
    model.parse(package);
    precision = options.precision;

    context.open_instance();
    const auto adapters = context.adapters();
    if (options.device < 0 || size_t(options.device) >= adapters.size())
        throw std::runtime_error("Vulkan device index is out of range; enumerate Vulkan GPU devices first");
    const auto& selected = adapters[options.device];
    context.open_device(selected);
    const auto& device = selected.info;
    device_description = std::string(device.device_name) + " / " + device.driver_name + " " + device.driver_info;
    const auto& limits = context.limits;
    if (limits.maxComputeWorkGroupInvocations < 64 || limits.maxComputeWorkGroupSize[0] < 64
        || limits.maxComputeWorkGroupSize[1] < 8 || limits.maxComputeSharedMemorySize < 4096
        || limits.maxPushConstantsSize < sizeof(Push) || limits.maxPerStageDescriptorStorageBuffers < 2)
        throw std::runtime_error("Vulkan GPU does not meet GateRV3 compute limits");

    const size_t bytes = (model.weight_bytes + 3) & ~size_t(3);
    if (bytes > limits.maxStorageBufferRange)
        throw umzi::vk::AllocationError("GateRV3 weights exceed the Vulkan storage buffer limit");
    weights.allocate(context, bytes, storage | dst_transfer, false);
    {
        Buffer upload;
        upload.allocate(context, bytes, src_transfer, true);
        std::memcpy(upload.mapping, package.section(3).data(), model.weight_bytes);
        std::memset(static_cast<char*>(upload.mapping) + model.weight_bytes, 0, bytes - model.weight_bytes);
        upload.flush();
        auto command = context.allocate_command();
        begin(command, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        VkBufferCopy copy{0, 0, bytes};
        vkCmdCopyBuffer(command, upload.handle, weights.handle, 1, &copy);
        barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        check(vkEndCommandBuffer(command), "Finish Vulkan weight upload commands");
        context.submit(command);
        vkFreeCommandBuffers(context.device, context.pool, 1, &command);
    }

    const VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo descriptor{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor.bindingCount = 2;
    descriptor.pBindings = bindings;
    check(vkCreateDescriptorSetLayout(context.device, &descriptor, nullptr, &descriptor_layout),
          "Create Vulkan descriptor layout");

    VkPushConstantRange constants{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo pipeline{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline.setLayoutCount = 1;
    pipeline.pSetLayouts = &descriptor_layout;
    pipeline.pushConstantRangeCount = 1;
    pipeline.pPushConstantRanges = &constants;
    check(vkCreatePipelineLayout(context.device, &pipeline, nullptr, &pipeline_layout),
          "Create Vulkan pipeline layout");
    VkPipelineCacheCreateInfo cache{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    check(vkCreatePipelineCache(context.device, &cache, nullptr, &pipeline_cache),
          "Create Vulkan in-memory pipeline cache");

    for (auto [index, code, bytes] : {
         std::tuple{convolution, conv_code, sizeof(conv_code)},
         std::tuple{elementwise, element_code, sizeof(element_code)},
         std::tuple{reduction, reduce_code, sizeof(reduce_code)},
         std::tuple{attention, attention_code, sizeof(attention_code)}})
        pipelines[index] = create_pipeline(context.device, pipeline_layout, pipeline_cache, code, bytes);
    model_ms = elapsed(started);
}

void UmziSession::prepare(gt::Shape shape) {
    const auto padded = model.padded(shape);
    if (prepared && equal(prepared->shape, padded)) return;
    const auto started = Clock::now();
    const auto resolved = model.resolve(padded, sizeof(float));
    auto plan = std::make_unique<Prepared>(context, padded);
    const size_t rgb_bytes = padded.rgb_bytes(false);
    const size_t scratch_bytes = size_t(padded.n) * 256 * sizeof(float);
    const size_t bytes = rgb_bytes * 2 + resolved.arena_bytes + scratch_bytes;
    if (bytes > context.limits.maxStorageBufferRange || bytes > uint64_t(INT_MAX))
        throw umzi::vk::AllocationError("GateRV3 activations exceed the Vulkan storage buffer limit; "
                                        "try a smaller tile or batch");
    plan->rgb_bytes = rgb_bytes;
    plan->arena.allocate(context, bytes, storage | src_transfer | dst_transfer, false);
    plan->staging.allocate(context, rgb_bytes * 2, src_transfer | dst_transfer, true);

    VkDescriptorPoolSize storage_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = 1;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &storage_size;
    check(vkCreateDescriptorPool(context.device, &pool, nullptr, &plan->descriptor_pool),
          "Create Vulkan descriptor pool");
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = plan->descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &descriptor_layout;
    VkDescriptorSet set{};
    check(vkAllocateDescriptorSets(context.device, &set_info, &set), "Allocate Vulkan descriptor set");
    VkDescriptorBufferInfo buffers[]{
        {plan->arena.handle, 0, bytes},
        {weights.handle, 0, weights.bytes},
    };
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffers[i];
    }
    vkUpdateDescriptorSets(context.device, 2, writes, 0, nullptr);
    plan->allocation_ms = elapsed(started);

    const auto recording = Clock::now();
    auto command = plan->command = context.allocate_command();
    begin(command);
    const auto arena = plan->arena.handle;
    const auto staging = plan->staging.handle;
    const VkBufferCopy input{0, 0, rgb_bytes};
    vkCmdCopyBuffer(command, staging, arena, 1, &input);
    barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);

    const size_t output_offset = rgb_bytes + resolved.arena_bytes;
    const size_t scratch_offset = output_offset + rgb_bytes;
    auto offset = [&](int id) -> int {
        if (id <= 0) return id;
        if (size_t(id) == resolved.tensors.size() - 1) return int(output_offset / sizeof(float));
        return int((rgb_bytes + size_t(resolved.tensors[id].offset) * padded.n) / sizeof(float));
    };
    auto record = [&](PipelineIndex pipeline, const Push& op, int groups) {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[pipeline]);
        vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(op), &op);
        dispatch(context, command, groups);
        ++plan->dispatch_count;
        // The arena reuses dead tensor ranges, so order reads as well as writes before dispatching again.
        barrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    };
    for (const auto& op : model.ops) {
        const auto& in = resolved.tensors[op.input[0]];
        const auto& out = resolved.tensors[op.out];
        Push push{};
        for (int j = 0; j < 4; ++j) {
            const int id = op.input[j];
            push.offsets[j] = offset(id);
            push.channels[j] = id >= 0 ? resolved.tensors[id].c : 0;
            if (j < 3) push.weights[j] = op.weight[j];
        }
        push.output_offset = offset(op.out);
        push.output_channels = out.c;
        push.h = out.h;
        push.w = out.w;
        push.batch = padded.n;
        push.kind = op.kind;
        push.input_h = in.h;
        push.input_w = in.w;
        push.kernel_size = op.x;
        push.channel_offset = op.y;
        push.mixed = precision == UMZI_MIXED;
        push.scratch = int(scratch_offset / sizeof(float));
        const int pixels = padded.n * out.pixels();
        switch (op.kind) {
        case gt::Conv:
            record(convolution, push, ceil_div(pixels, 32) * ceil_div(out.c, 32));
            break;
        case gt::Rms:
            record(reduction, push, pixels);
            break;
        case gt::MeanProject:
            record(reduction, push, padded.n * (out.c / 8));
            push.kind = mean_projection;
            record(reduction, push, ceil_div(padded.n * out.c, 64));
            break;
        case gt::Attention:
            record(attention, push, pixels * 16);
            break;
        default:
            record(elementwise, push, ceil_div(padded.n * out.elements(), 64));
            break;
        }
    }

    barrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    const VkBufferCopy output{output_offset, rgb_bytes, rgb_bytes};
    vkCmdCopyBuffer(command, arena, staging, 1, &output);
    barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    check(vkEndCommandBuffer(command), "Finish GateRV3 Vulkan command buffer");
    plan->recording_ms = elapsed(recording);
    prepared = std::move(plan);
}

void UmziSession::run(gt::Shape shape, const float* input, float* output) {
    prepare(shape);
    const auto& plan = *prepared;
    const auto padded = plan.shape;
    auto* staging = static_cast<float*>(plan.staging.mapping);
    if (equal(shape, padded)) {
        std::memcpy(staging, input, plan.rgb_bytes);
    } else {
        for (int plane = 0; plane < shape.n * 3; ++plane) {
            for (int y = 0; y < padded.h; ++y) {
                const int row = padding_index(y, shape.h, padded.h);
                const float* src = input + (size_t(plane) * shape.h + row) * shape.w;
                std::memcpy(staging, src, size_t(shape.w) * sizeof(float));
                for (int x = shape.w; x < padded.w; ++x)
                    staging[x] = src[padding_index(x, shape.w, padded.w)];
                staging += padded.w;
            }
        }
    }
    plan.staging.flush();
    context.submit(plan.command);
    plan.staging.invalidate();
    auto* result = static_cast<const float*>(plan.staging.mapping) + plan.rgb_bytes / sizeof(float);
    if (equal(shape, padded)) {
        std::memcpy(output, result, plan.rgb_bytes);
    } else {
        for (int plane = 0; plane < shape.n * 3; ++plane) {
            for (int y = 0; y < shape.h; ++y) {
                const float* src = result + (size_t(plane) * padded.h + y) * padded.w;
                std::memcpy(output, src, size_t(shape.w) * sizeof(float));
                output += shape.w;
            }
        }
    }
}

void UmziSession::info(UmziSessionInfo& result) const {
    result = {sizeof(result), UMZI_CACHE_DISABLED};
    result.model_load_ms = model_ms;
    result.buffer_bytes[0] = weights.bytes;
    if (!prepared) return;
    const auto& plan = *prepared;
    const auto [n, h, w] = plan.shape;
    const size_t scratch = size_t(n) * 256 * sizeof(float);
    result.prepared_input = {4, {uint32_t(n), 3, uint32_t(h), uint32_t(w)}};
    result.buffer_bytes[1] = plan.arena.bytes - 2 * plan.rgb_bytes - scratch;
    result.buffer_bytes[2] = scratch;
    result.buffer_bytes[3] = 2 * plan.rgb_bytes + plan.staging.bytes;
    result.shape_setup_ms[0] = plan.allocation_ms;
    result.shape_setup_ms[3] = plan.recording_ms;
}

int64_t UmziSession::dispatches() const { return prepared ? prepared->dispatch_count : 0; }
