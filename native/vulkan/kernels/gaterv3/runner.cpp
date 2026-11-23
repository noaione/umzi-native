#include "../../common/compute.h"
#include "../../runtime/kernel.h"
#include "../../../kernels/gaterv3/model_file.h"
#include "winograd.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <span>
#include <string_view>

namespace {
using umzi::vk::barrier;
using umzi::vk::begin;
using umzi::vk::Buffer;
using umzi::vk::create_pipeline;
using umzi::vk::dispatch;
using umzi::vk::check;
using umzi::vk::Context;

constexpr uint32_t conv_code[]{
#include "gaterv3_conv_spv.h"
};
constexpr uint32_t conv_half_code[]{
#include "gaterv3_conv_half_spv.h"
};
constexpr uint32_t element_code[]{
#include "gaterv3_element_spv.h"
};
constexpr uint32_t reduce_code[]{
#include "gaterv3_reduce_spv.h"
};
constexpr uint32_t attention_code[]{
#include "gaterv3_attention_spv.h"
};
constexpr uint32_t conv_fast_code[]{
#include "gaterv3_conv_fast_spv.h"
};
constexpr uint32_t conv_fast_half_code[]{
#include "gaterv3_conv_fast_half_spv.h"
};
constexpr uint32_t conv_native_half_code[]{
#include "gaterv3_conv_native_half_spv.h"
};
constexpr uint32_t conv_native_half_fp32_storage_code[]{
#include "gaterv3_conv_native_half_fp32_storage_spv.h"
};
constexpr uint32_t conv_half2_code[]{
#include "gaterv3_conv_half2_spv.h"
};
constexpr uint32_t projection_code[]{
#include "gaterv3_projection_spv.h"
};
constexpr uint32_t conv_winograd_code[]{
#include "gaterv3_conv_winograd_spv.h"
};
constexpr uint32_t conv_winograd_reuse_code[]{
#include "gaterv3_conv_winograd_reuse_spv.h"
};
constexpr uint32_t element_half_code[]{
#include "gaterv3_element_half_spv.h"
};
constexpr uint32_t reduce_half_code[]{
#include "gaterv3_reduce_half_spv.h"
};
constexpr uint32_t attention_half_code[]{
#include "gaterv3_attention_half_spv.h"
};
constexpr uint32_t mean_fast_code[]{
#include "gaterv3_mean_fast_spv.h"
};
constexpr uint32_t mean_fast_half_code[]{
#include "gaterv3_mean_fast_half_spv.h"
};
constexpr uint32_t norm_fast_code[]{
#include "gaterv3_norm_fast_spv.h"
};
constexpr uint32_t norm_fast_half_code[]{
#include "gaterv3_norm_fast_half_spv.h"
};
constexpr uint32_t gates_code[]{
#include "gaterv3_gates_spv.h"
};
constexpr uint32_t gates_half_code[]{
#include "gaterv3_gates_half_spv.h"
};

struct Push {
    std::array<int, 4> offsets{}, channels{}, weights{};
    int output_offset, output_channels, h, w, batch, kind, input_h, input_w;
    int kernel_size, channel_offset, mixed, scratch;
};
static_assert(sizeof(Push) == 96);

enum PipelineIndex {
    convolution, convolution_spatial, convolution_silu, elementwise, reduction, attention,
    spatial_mean, norm, local_gate, inception_gate, winograd, winograd_silu, projection, pipeline_count,
};

struct KernelOptions {
    bool reference = false;
    bool profile = false;
    bool fast_convolution = true;
    bool fp16_storage = true;
    bool fuse_silu = true;
    bool subgroup_norm = true;
    bool fuse_residual_norm = true;
    bool chunked_mean = true;
    bool paired_gates = true;
    bool half2_convolution = true;
    bool tiled_projection = true;
    bool winograd_convolution = false;
    bool winograd_input_reuse = true;
};

struct KernelOption {
    const char* name;
    bool KernelOptions::*field;
};
constexpr KernelOption kernel_options[] = {
    {"reference", &KernelOptions::reference},
    {"profile", &KernelOptions::profile},
    {"fast_convolution", &KernelOptions::fast_convolution},
    {"fp16_storage", &KernelOptions::fp16_storage},
    {"fuse_silu", &KernelOptions::fuse_silu},
    {"subgroup_norm", &KernelOptions::subgroup_norm},
    {"fuse_residual_norm", &KernelOptions::fuse_residual_norm},
    {"chunked_mean", &KernelOptions::chunked_mean},
    {"paired_gates", &KernelOptions::paired_gates},
    {"half2_convolution", &KernelOptions::half2_convolution},
    {"tiled_projection", &KernelOptions::tiled_projection},
    {"winograd_convolution", &KernelOptions::winograd_convolution},
    {"winograd_input_reuse", &KernelOptions::winograd_input_reuse},
};

KernelOptions read_options(const UmziSessionOptions& config) {
    KernelOptions options;
    uint32_t seen = 0;
    for (uint32_t i = 0; i < config.kernel_option_count; ++i) {
        const auto& option = config.kernel_options[i];
        const std::string_view name(option.name ? option.name : "");
        const auto match = std::find_if(std::begin(kernel_options), std::end(kernel_options),
                                        [&](const auto& item) { return name == item.name; });
        if (match == std::end(kernel_options))
            throw std::runtime_error("Unknown Vulkan GateRV3 option: " + std::string(name));
        const uint32_t bit = 1u << (match - std::begin(kernel_options));
        if ((seen & bit) || (option.value != 0 && option.value != 1))
            throw std::runtime_error("Duplicate or invalid Vulkan GateRV3 option: " + std::string(name));
        seen |= bit;
        options.*match->field = option.value != 0;
    }
    if (options.reference) {
        for (const auto& option : kernel_options) {
            if (option.field != &KernelOptions::reference && option.field != &KernelOptions::profile)
                options.*option.field = false;
        }
    }
    return options;
}

constexpr int mean_projection = 18;
constexpr int conv3_stage = 19;
constexpr int fused_norm_stage = 20;
constexpr int mean_chunk = 256;
constexpr int projection_tile = 32;
constexpr int projection_depth = 16;
constexpr int winograd_tiles_per_group = 8;
constexpr int winograd_reuse_tiles_per_group = 4;
constexpr uint32_t winograd_shared_bytes = 16 * 4 * (9 + 17) * (4 * sizeof(uint16_t));
constexpr uint32_t winograd_reuse_shared_bytes = 4 * 4 * 32 * (4 * sizeof(uint16_t));
constexpr size_t winograd_filter_bytes = 32 * 32 * 16 * sizeof(uint16_t);
constexpr std::array profile_names{
    "gpu_pack_ns", "gpu_conv1_ns", "gpu_rms_ns", "gpu_local_ns", "gpu_mean_ns",
    "gpu_local_skip_ns", "gpu_inception_ns", "gpu_global_skip_ns", "gpu_shuffle_ns",
    "gpu_concat_ns", "gpu_silu_ns", "gpu_span_ns", "gpu_depthwise_ns", "gpu_attention_ns",
    "gpu_gate_ns", "gpu_add_ns", "gpu_rgb_ns", "gpu_shuffle_concat_ns", "gpu_mean_project_ns",
    "gpu_conv3_ns", "gpu_fused_residual_norm_ns",
};
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

int ceil_div(int value, int divisor) {
    return (value - 1) / divisor + 1;
}

int padding_index(int i, int original, int padded) {
    if (i < original) return i;
    return padded - original < original ? 2 * original - 2 - i : original - 1;
}

class GateRV3Session final : public umzi::vk::KernelSession {
public:
    GateRV3Session();
    ~GateRV3Session() override;

    void open(const umzi::ModelPackage& package, const UmziSessionOptions& config);
    UmziModelInfo model_info() const override;
    UmziIo tensors() const override;
    UmziIo describe(const UmziShape& shape) const override;
    void prepare_shape(const UmziShape& shape) override;
    void run_host(const UmziShape& shape, const void* input, void* output) override;
    UmziSessionInfo session_info() const override;
    uint32_t diagnostic_count() const override;
    UmziDiagnostic diagnostic(uint32_t index) const override;
    const char* kernel_message() const override { return device_description.c_str(); }

private:
    struct Prepared;
    Context context;
    gt::ModelFile model;
    Buffer weights;
    VkDescriptorSetLayout descriptor_layout{};
    VkPipelineLayout pipeline_layout{};
    VkPipelineCache pipeline_cache{};
    std::array<VkPipeline, pipeline_count> pipelines{};
    std::unique_ptr<Prepared> prepared;
    std::string device_description;
    uint32_t precision{};
    uint32_t timestamp_bits{};
    KernelOptions options;
    std::vector<int> winograd_weights;

    void prepare(gt::Shape shape);
    void run(gt::Shape shape, const float* input, float* output);
};

struct GateRV3Session::Prepared {
    const Context& context;
    gt::Shape shape;
    Buffer arena, staging;
    VkDescriptorPool descriptor_pool{};
    VkCommandBuffer command{};
    VkQueryPool queries{};
    std::vector<int> dispatch_stages;
    std::vector<uint64_t> timestamps;
    std::array<int64_t, profile_names.size()> durations{};
    bool durations_valid{};
    size_t rgb_bytes{}, activation_bytes{}, scratch_bytes{};
    double allocation_ms{}, recording_ms{};
    int64_t dispatch_count{};

    Prepared(const Context& context, gt::Shape shape) : context(context), shape(shape) {}
    ~Prepared() {
        if (command) vkFreeCommandBuffers(context.device, context.pool, 1, &command);
        if (queries) vkDestroyQueryPool(context.device, queries, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
    }
};

GateRV3Session::GateRV3Session() = default;
GateRV3Session::~GateRV3Session() {
    prepared.reset();
    if (context.device) {
        for (auto pipeline : pipelines) vkDestroyPipeline(context.device, pipeline, nullptr);
        vkDestroyPipelineCache(context.device, pipeline_cache, nullptr);
        vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(context.device, descriptor_layout, nullptr);
    }
}

void GateRV3Session::open(const umzi::ModelPackage& package, const UmziSessionOptions& config) {
    model.parse(package);
    precision = config.precision;
    options = read_options(config);
    const bool fp16_math = precision == UMZI_FP16;
    if (fp16_math && !options.fast_convolution)
        throw std::runtime_error("Vulkan FP16 arithmetic requires the fast_convolution option");

    context.open_instance();
    const auto adapters = context.adapters();
    if (config.device < 0 || size_t(config.device) >= adapters.size())
        throw std::runtime_error("Vulkan device index is out of range; enumerate Vulkan GPU devices first");
    const auto& selected = adapters[config.device];
    timestamp_bits = selected.timestamp_bits;
    if (options.profile && !timestamp_bits) throw std::runtime_error("Selected Vulkan queue does not support timestamps");
    const auto& device = selected.info;
    options.fp16_storage &= precision != UMZI_FP32 && device.shader_float16 && device.storage_buffer16;
    options.half2_convolution &= fp16_math && options.fp16_storage;
    options.tiled_projection &= options.half2_convolution;
    options.subgroup_norm &= selected.subgroup32_arithmetic;
    options.fuse_residual_norm &= options.subgroup_norm;
    options.fuse_silu &= options.fast_convolution;
    if (fp16_math && !device.shader_float16)
        throw std::runtime_error("Vulkan FP16 arithmetic requires shaderFloat16");
    context.open_device(selected, options.fp16_storage, fp16_math || options.fp16_storage);
    device_description = std::string(device.device_name) + " / " + device.driver_name + " " + device.driver_info;
    const auto& limits = context.limits;
    if (limits.maxComputeWorkGroupInvocations < 64 || limits.maxComputeWorkGroupSize[0] < 64
        || limits.maxComputeWorkGroupSize[1] < 8
        || limits.maxComputeSharedMemorySize < (options.fast_convolution ? 4224u : 4096u)
        || limits.maxPushConstantsSize < sizeof(Push)
        || limits.maxPerStageDescriptorStorageBuffers < (options.fp16_storage ? 3u : 2u))
        throw std::runtime_error("Vulkan GPU does not meet GateRV3 compute limits");

    const bool winograd_base_ok = fp16_math && options.fp16_storage;
    const bool winograd_reuse_ok = winograd_base_ok && options.winograd_input_reuse
        && limits.maxComputeWorkGroupInvocations >= 32
        && limits.maxComputeSharedMemorySize >= winograd_reuse_shared_bytes;
    const bool winograd_legacy_ok = winograd_base_ok
        && limits.maxComputeWorkGroupInvocations >= 128
        && limits.maxComputeSharedMemorySize >= winograd_shared_bytes;
    options.winograd_convolution &= winograd_reuse_ok || winograd_legacy_ok;
    options.winograd_input_reuse = options.winograd_convolution && winograd_reuse_ok;
    const size_t packed_bytes = (model.weight_bytes + 7) & ~size_t(7);
    size_t bytes = packed_bytes;
    if (options.winograd_convolution) {
        winograd_weights.resize(model.ops.size(), -1);
        const auto limit = std::min<size_t>(limits.maxStorageBufferRange, INT_MAX);
        for (size_t i = 0; i < model.ops.size(); ++i) {
            const auto& op = model.ops[i];
            if (op.kind != gt::Conv || op.x != 3 || model.channels(op.input[0]) != 32 || model.channels(op.out) != 32)
                continue;
            if (bytes + winograd_filter_bytes > limit) break;
            winograd_weights[i] = int(bytes);
            bytes += winograd_filter_bytes;
        }
        options.winograd_convolution = bytes > packed_bytes;
        options.winograd_input_reuse &= options.winograd_convolution;
    }
    if (bytes > limits.maxStorageBufferRange)
        throw umzi::vk::AllocationError("GateRV3 weights exceed the Vulkan storage buffer limit");
    weights.allocate(context, bytes, storage | dst_transfer, false);
    {
        Buffer upload;
        upload.allocate(context, bytes, src_transfer, true);
        const auto* source = package.section(3).data();
        auto* destination = static_cast<char*>(upload.mapping);
        std::memcpy(destination, source, model.weight_bytes);
        std::memset(destination + model.weight_bytes, 0, packed_bytes - model.weight_bytes);
        if (options.winograd_convolution) {
            for (size_t i = 0; i < winograd_weights.size(); ++i) {
                if (winograd_weights[i] >= 0)
                    umzi::vk::transform_winograd_filters(source + model.ops[i].weight[0],
                                                         destination + winograd_weights[i],
                                                         options.winograd_input_reuse);
            }
        }
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
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo descriptor{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor.bindingCount = options.fp16_storage ? 3 : 2;
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

    using Code = std::span<const uint32_t>;
    auto build = [&](PipelineIndex index, Code code, const VkSpecializationInfo* specialize = nullptr) {
        pipelines[index] = create_pipeline(context.device, pipeline_layout, pipeline_cache, code, specialize);
    };
    const bool half_storage = options.fp16_storage;
    if (!options.fast_convolution) {
        build(convolution, half_storage ? Code(conv_half_code) : Code(conv_code));
    } else {
        Code code = half_storage ? Code(conv_fast_half_code) : Code(conv_fast_code);
        if (fp16_math) code = half_storage ? Code(conv_native_half_code) : Code(conv_native_half_fp32_storage_code);
        if (options.half2_convolution) code = Code(conv_half2_code);
        const VkSpecializationMapEntry entries[]{
            {0, 0, sizeof(VkBool32)}, {1, sizeof(VkBool32), sizeof(VkBool32)},
        };
        const VkBool32 spatial[]{VK_TRUE, VK_FALSE};
        const VkBool32 silu[]{VK_TRUE, VK_TRUE};
        const VkSpecializationInfo specialize{2, entries, sizeof(spatial), spatial};
        const VkSpecializationInfo specialize_silu{2, entries, sizeof(silu), silu};
        build(convolution, code);
        build(convolution_spatial, code, &specialize);
        if (options.fuse_silu) build(convolution_silu, code, &specialize_silu);
        if (options.tiled_projection) build(projection, projection_code);
        if (options.winograd_convolution) {
            const Code winograd_code = options.winograd_input_reuse
                ? Code(conv_winograd_reuse_code) : Code(conv_winograd_code);
            build(winograd, winograd_code);
            const VkSpecializationInfo epilogue{1, entries + 1, sizeof(silu), silu};
            if (options.fuse_silu) build(winograd_silu, winograd_code, &epilogue);
        }
    }
    if (options.chunked_mean) build(spatial_mean, half_storage ? Code(mean_fast_half_code) : Code(mean_fast_code));
    if (options.paired_gates) {
        const Code gates = half_storage ? Code(gates_half_code) : Code(gates_code);
        const VkSpecializationMapEntry entry{0, 0, sizeof(VkBool32)};
        const VkBool32 inception = VK_TRUE;
        const VkSpecializationInfo specialize_inception{1, &entry, sizeof(inception), &inception};
        build(local_gate, gates);
        build(inception_gate, gates, &specialize_inception);
    }
    if (options.subgroup_norm) build(norm, half_storage ? Code(norm_fast_half_code) : Code(norm_fast_code));
    build(elementwise, half_storage ? Code(element_half_code) : Code(element_code));
    build(reduction, half_storage ? Code(reduce_half_code) : Code(reduce_code));
    build(attention, half_storage ? Code(attention_half_code) : Code(attention_code));
}

gt::Shape image_shape(const UmziShape& shape) {
    if (shape.rank != 4 || shape.dimensions[1] != 3) throw std::runtime_error("GateRV3 expects RGB NCHW input");
    return gt::Shape::checked(shape.dimensions[0], shape.dimensions[2], shape.dimensions[3]);
}

UmziIo rgb_tensors(const UmziShape& shape, uint64_t bytes) {
    return {sizeof(UmziIo), {"input", shape, UMZI_FLOAT32, bytes}, {"output", shape, UMZI_FLOAT32, bytes}};
}

UmziModelInfo GateRV3Session::model_info() const { return {sizeof(UmziModelInfo), "gaterv3", 1, 0}; }

UmziIo GateRV3Session::tensors() const { return rgb_tensors({4, {0, 3, 0, 0}}, 0); }

UmziIo GateRV3Session::describe(const UmziShape& shape) const {
    const auto dims = image_shape(shape);
    model.check_shape(model.padded(dims));
    return rgb_tensors(shape, dims.rgb_bytes(false));
}

void GateRV3Session::prepare_shape(const UmziShape& shape) { prepare(image_shape(shape)); }

void GateRV3Session::run_host(const UmziShape& shape, const void* input, void* output) {
    run(image_shape(shape), static_cast<const float*>(input), static_cast<float*>(output));
}

void GateRV3Session::prepare(gt::Shape shape) {
    const auto padded = model.padded(shape);
    if (prepared && equal(prepared->shape, padded)) return;
    const auto started = Clock::now();
    const size_t element_bytes = options.fp16_storage ? sizeof(uint16_t) : sizeof(float);
    const auto resolved = model.resolve(padded, element_bytes);
    auto plan = std::make_unique<Prepared>(context, padded);
    const size_t rgb_bytes = padded.rgb_bytes(false);
    size_t scratch_bytes = options.chunked_mean ? 0 : size_t(padded.n) * 256 * element_bytes;
    std::vector<int> uses(resolved.tensors.size());
    for (const auto& op : model.ops) {
        if (options.chunked_mean && op.kind == gt::MeanProject) {
            const auto& in = resolved.tensors[op.input[0]];
            scratch_bytes = std::max(scratch_bytes,
                size_t(padded.n) * ceil_div(in.pixels(), mean_chunk) * in.c * sizeof(float));
        }
        for (int id : op.input)
            if (id >= 0) ++uses[id];
    }
    const size_t bytes = rgb_bytes * 2 + resolved.arena_bytes + scratch_bytes;
    if (bytes > context.limits.maxStorageBufferRange || bytes > uint64_t(INT_MAX))
        throw umzi::vk::AllocationError("GateRV3 activations exceed the Vulkan storage buffer limit; "
                                        "try a smaller tile or batch");
    plan->rgb_bytes = rgb_bytes;
    plan->scratch_bytes = scratch_bytes;
    plan->activation_bytes = resolved.arena_bytes;
    plan->arena.allocate(context, bytes, storage | src_transfer | dst_transfer, false);
    plan->staging.allocate(context, rgb_bytes * 2, src_transfer | dst_transfer, true);

    const uint32_t descriptor_count = options.fp16_storage ? 3 : 2;
    VkDescriptorPoolSize storage_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptor_count};
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
        {plan->arena.handle, 0, bytes},
    };
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < descriptor_count; ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffers[i];
    }
    vkUpdateDescriptorSets(context.device, descriptor_count, writes, 0, nullptr);
    plan->allocation_ms = elapsed(started);

    const auto recording = Clock::now();
    auto command = plan->command = context.allocate_command();
    begin(command);
    if (options.profile) {
        const auto count = uint32_t(model.ops.size() * 2 + 1);
        VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query.queryCount = count;
        check(vkCreateQueryPool(context.device, &query, nullptr, &plan->queries), "Create Vulkan timestamp pool");
        vkCmdResetQueryPool(command, plan->queries, 0, count);
    }
    const auto arena = plan->arena.handle;
    const auto staging = plan->staging.handle;
    const VkBufferCopy input{0, 0, rgb_bytes};
    vkCmdCopyBuffer(command, staging, arena, 1, &input);
    barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
    if (plan->queries) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, plan->queries, 0);

    const size_t output_offset = rgb_bytes + resolved.arena_bytes;
    const size_t scratch_offset = output_offset + rgb_bytes;
    auto offset = [&](int id) -> int {
        if (id <= 0) return id;
        if (size_t(id) == resolved.tensors.size() - 1) return int(output_offset / sizeof(float));
        return int((rgb_bytes + size_t(resolved.tensors[id].offset) * padded.n) / element_bytes);
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
        if (plan->queries) {
            int stage = op.kind;
            if (op.kind == gt::Conv && op.kernel_size == 3) stage = conv3_stage;
            if (pipeline == norm && op.kind != gt::Rms) stage = fused_norm_stage;
            plan->dispatch_stages.push_back(stage);
            vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, plan->queries,
                                uint32_t(plan->dispatch_count));
        }
    };
    for (size_t index = 0; index < model.ops.size(); ++index) {
        const auto& op = model.ops[index];
        const auto* next = index + 1 < model.ops.size() ? &model.ops[index + 1] : nullptr;
        if (next && next->input[0] != op.out) next = nullptr;
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
        push.mixed = precision != UMZI_FP32;
        push.scratch = int(scratch_offset / (options.chunked_mean ? sizeof(float) : element_bytes));
        const int pixels = padded.n * out.pixels();
        switch (op.kind) {
        case gt::Conv: {
            const int cin = in.c - op.y;
            if (options.tiled_projection && op.x == 1 && pixels % projection_tile == 0
                && out.c % projection_tile == 0 && cin % projection_depth == 0) {
                record(projection, push, (pixels / projection_tile) * (out.c / projection_tile));
                break;
            }
            const int transformed = options.winograd_convolution ? winograd_weights[index] : -1;
            auto pipeline = convolution;
            if (op.x == 3 && options.fast_convolution) {
                pipeline = convolution_spatial;
                if (options.fuse_silu && next && next->kind == gt::Silu && uses[op.out] == 1) {
                    pipeline = convolution_silu;
                    push.output_offset = offset(next->out);
                    ++index;
                }
            }
            if (transformed >= 0) {
                push.weights[0] = transformed;
                const int tiles = padded.n * ceil_div(out.h, 2) * ceil_div(out.w, 2);
                const int groups = options.winograd_input_reuse
                    ? ceil_div(tiles, winograd_reuse_tiles_per_group)
                    : 2 * ceil_div(tiles, winograd_tiles_per_group);
                record(pipeline == convolution_silu ? winograd_silu : winograd, push, groups);
            } else {
                record(pipeline, push, ceil_div(pixels, 32) * ceil_div(out.c, 32));
            }
            break;
        }
        case gt::Rms:
            if (options.subgroup_norm) record(norm, push, ceil_div(pixels, 2));
            else record(reduction, push, pixels);
            break;
        case gt::MeanProject: {
            const auto pipeline = options.chunked_mean ? spatial_mean : reduction;
            const int chunks = options.chunked_mean ? ceil_div(in.pixels(), mean_chunk) : 1;
            record(pipeline, push, padded.n * (out.c / 8) * chunks);
            push.kind = mean_projection;
            record(pipeline, push, options.chunked_mean ? padded.n : ceil_div(padded.n * out.c, 64));
            break;
        }
        case gt::Attention:
            record(attention, push, pixels * 16);
            break;
        case gt::Local:
        case gt::Inception:
            if (options.paired_gates) {
                record(op.kind == gt::Local ? local_gate : inception_gate, push,
                       ceil_div(padded.n * out.elements() / 2, 64));
                break;
            }
            record(elementwise, push, ceil_div(padded.n * out.elements(), 64));
            break;
        case gt::LocalSkip:
        case gt::GlobalSkip:
            if (options.fuse_residual_norm && next && next->kind == gt::Rms && next->input[1] >= 0) {
                push.offsets[3] = push.output_offset;
                push.output_offset = offset(next->out);
                push.weights[1] = next->weight[0];
                push.weights[2] = next->weight[1];
                record(norm, push, ceil_div(pixels, 2));
                ++index;
                break;
            }
            [[fallthrough]];
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
    if (plan->queries) plan->timestamps.resize(plan->dispatch_count + 1);
    plan->recording_ms = elapsed(recording);
    prepared = std::move(plan);
}

void GateRV3Session::run(gt::Shape shape, const float* input, float* output) {
    prepare(shape);
    auto& plan = *prepared;
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
    const auto submitted = Clock::now();
    context.submit(plan.command);
    if (plan.queries) {
        auto& ticks = plan.timestamps;
        check(vkGetQueryPoolResults(context.device, plan.queries, 0, uint32_t(ticks.size()), ticks.size() * sizeof(uint64_t),
                                    ticks.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
              "Read Vulkan inference timestamps");
        const uint64_t mask = UINT64_MAX >> (64 - timestamp_bits);
        // A few drivers report out-of-order samples for per-dispatch queries.
        // Reject intervals larger than the entire host submission window.
        const double limit_ns = elapsed(submitted) * 1e6;
        plan.durations.fill(0);
        plan.durations_valid = true;
        double total_ns = 0;
        for (size_t i = 0; i < plan.dispatch_stages.size(); ++i) {
            const double nanoseconds = double((ticks[i + 1] - ticks[i]) & mask) * context.limits.timestampPeriod;
            total_ns += nanoseconds;
            if (!(total_ns <= limit_ns)) {
                plan.durations_valid = false;
                plan.durations.fill(0);
                break;
            }
            plan.durations[plan.dispatch_stages[i]] += int64_t(nanoseconds);
        }
    }
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

UmziSessionInfo GateRV3Session::session_info() const {
    UmziSessionInfo result{sizeof(result), UMZI_CACHE_DISABLED};
    result.buffer_bytes[0] = weights.bytes;
    if (!prepared) return result;
    const auto& plan = *prepared;
    const auto [n, h, w] = plan.shape;
    result.prepared_input = {4, {uint32_t(n), 3, uint32_t(h), uint32_t(w)}};
    result.buffer_bytes[1] = plan.activation_bytes;
    result.buffer_bytes[2] = plan.scratch_bytes;
    result.buffer_bytes[3] = 2 * plan.rgb_bytes + plan.staging.bytes;
    result.shape_setup_ms[0] = plan.allocation_ms;
    result.shape_setup_ms[3] = plan.recording_ms;
    return result;
}

uint32_t GateRV3Session::diagnostic_count() const {
    return 1 + uint32_t(std::size(kernel_options)) + (options.profile ? 1 + uint32_t(profile_names.size()) : 0);
}

UmziDiagnostic GateRV3Session::diagnostic(uint32_t index) const {
    if (index == 0) return {sizeof(UmziDiagnostic), "dispatches", prepared ? prepared->dispatch_count : 0};
    --index;
    if (options.profile) {
        if (index == 0) return {sizeof(UmziDiagnostic), "gpu_timestamps_valid", prepared && prepared->durations_valid};
        --index;
        if (index < profile_names.size())
            return {sizeof(UmziDiagnostic), profile_names[index], prepared ? prepared->durations[index] : 0};
        index -= uint32_t(profile_names.size());
    }
    const auto& option = kernel_options[index];
    return {sizeof(UmziDiagnostic), option.name, options.*option.field};
}
} // namespace

std::unique_ptr<umzi::vk::KernelSession> umzi::vk::make_gaterv3(
    const umzi::ModelPackage& package, const UmziSessionOptions& options) {
    auto kernel = std::make_unique<GateRV3Session>();
    kernel->open(package, options);
    return kernel;
}
