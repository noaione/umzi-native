#pragma once

#include "context.h"
#include "../kernels/gaterv3/model_file.h"

#include <memory>

struct UmziSession {
    struct Prepared;
    umzi::vk::Context context;
    gt::ModelFile model;
    umzi::vk::Buffer weights;
    VkDescriptorSetLayout descriptor_layout{};
    VkPipelineLayout pipeline_layout{};
    VkPipelineCache pipeline_cache{};
    std::array<VkPipeline, 4> pipelines{};
    std::unique_ptr<Prepared> prepared;
    std::string device_description;
    uint32_t precision{};
    double model_ms{};

    UmziSession();
    ~UmziSession();
    void open(const char* path, const UmziSessionOptions& options);
    void prepare(gt::Shape shape);
    void run(gt::Shape shape, const float* input, float* output);
    void info(UmziSessionInfo& result) const;
    int64_t dispatches() const;
};
