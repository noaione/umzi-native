#pragma once

// cuDNN backend (graph API) plans for MoESR's NHWC FP16 3x3 convolutions.
//
// The legacy cudnnConvolutionForward API exposes only a few algorithms, and on
// Ampere its best FP16-accumulation choice for the 64->344 expansion ran 2.2x
// slower than the best backend engine. A backend plan enumerates every engine
// configuration (heuristics A and fallback), so the session can time them and
// keep the fastest. The plan cache stores the chosen candidate's position in
// the deduplicated list with its workspace size; the key includes the device,
// driver and cuDNN version, and a reused choice must still finalize with the
// same workspace and run before it is trusted. Variant packs bind the input,
// filter, output and workspace pointers of each call site.

#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace moesr {
class Descriptor {
    cudnnBackendDescriptor_t handle = nullptr;

public:
    explicit Descriptor(cudnnBackendDescriptorType_t type) {
        if (cudnnBackendCreateDescriptor(type, &handle) != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error("cudnnBackendCreateDescriptor failed");
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    ~Descriptor() {
        if (handle) cudnnBackendDestroyDescriptor(handle);
    }
    cudnnBackendDescriptor_t get() const { return handle; }
    void set(cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type, const void* values,
             int64_t count = 1) {
        const auto status = cudnnBackendSetAttribute(handle, name, type, count, values);
        if (status != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error(std::string("cudnnBackendSetAttribute: ") + cudnnGetErrorString(status));
    }
    void ref(cudnnBackendAttributeName_t name, const Descriptor& value) {
        auto descriptor = value.get();
        set(name, CUDNN_TYPE_BACKEND_DESCRIPTOR, &descriptor);
    }
    cudnnStatus_t try_finalize() { return cudnnBackendFinalize(handle); }
    void finalize() {
        const auto status = try_finalize();
        if (status != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error(std::string("cudnnBackendFinalize: ") + cudnnGetErrorString(status));
    }
};

// One candidate engine configuration, finalized into an execution plan.
struct Candidate {
    std::unique_ptr<Descriptor> plan;
    int64_t workspace = 0;
    std::string name;
};

class ConvGraph {
    std::vector<std::unique_ptr<Descriptor>> descriptors;

    Descriptor& make(cudnnBackendDescriptorType_t type) {
        return *descriptors.emplace_back(std::make_unique<Descriptor>(type));
    }

    Descriptor& tensor(int64_t uid, int64_t n, int64_t c, int64_t h, int64_t w) {
        auto& desc = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        constexpr int64_t alignment = 16;
        const auto type = CUDNN_DATA_HALF;
        const int64_t dimensions[] = {n, c, h, w};
        const int64_t strides[] = {h * w * c, 1, w * c, c};
        const bool is_virtual = false;
        desc.set(CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, &uid);
        desc.set(CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, &type);
        desc.set(CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, dimensions, 4);
        desc.set(CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, strides, 4);
        desc.set(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, &alignment);
        desc.set(CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, &is_virtual);
        desc.finalize();
        return desc;
    }

public:
    Descriptor* graph = nullptr;

    // Builds the operation graph for an NHWC 3x3 convolution with unit padding.
    ConvGraph(cudnnHandle_t handle, int n, int h, int w, int cin, int cout, bool fp16_compute) {
        auto& x = tensor(1, n, cin, h, w);
        auto& filter = tensor(2, cout, cin, 3, 3);
        auto& y = tensor(3, n, cout, h, w);
        const auto compute = fp16_compute ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
        auto& conv = make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        constexpr int64_t spatial = 2, unit[] = {1, 1};
        const auto mode = CUDNN_CROSS_CORRELATION;
        conv.set(CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, &compute);
        conv.set(CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, &mode);
        conv.set(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, &spatial);
        conv.set(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, unit, 2);
        conv.set(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, unit, 2);
        conv.set(CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, unit, 2);
        conv.set(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, unit, 2);
        conv.finalize();

        auto& op = make(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
        constexpr float one = 1, zero = 0;
        op.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, &one);
        op.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, &zero);
        op.ref(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, conv);
        op.ref(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, x);
        op.ref(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, filter);
        op.ref(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, y);
        op.finalize();

        graph = &make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        const auto operation = op.get();
        graph->set(CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, &handle);
        graph->set(CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, &operation, 1);
        graph->finalize();
    }

    // Every engine configuration from heuristics A and fallback that finalizes
    // into a plan within the workspace limit, in heuristic order, deduplicated.
    std::vector<Candidate> candidates(cudnnHandle_t handle, uint64_t limit) {
        std::vector<Candidate> result;
        for (auto mode : {CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK}) {
            Descriptor heuristic(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
            heuristic.set(CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, &mode);
            heuristic.ref(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, *graph);
            if (heuristic.try_finalize() != CUDNN_STATUS_SUCCESS) continue;
            int64_t found = 0;
            if (cudnnBackendGetAttribute(heuristic.get(), CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &found, nullptr) != CUDNN_STATUS_SUCCESS
                || found <= 0)
                continue;
            found = std::min<int64_t>(found, 64);
            std::vector<std::unique_ptr<Descriptor>> configs;
            std::vector<cudnnBackendDescriptor_t> raw;
            for (int64_t i = 0; i < found; ++i) {
                configs.push_back(std::make_unique<Descriptor>(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
                raw.push_back(configs.back()->get());
            }
            if (cudnnBackendGetAttribute(heuristic.get(), CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR, found, &found, raw.data())
                != CUDNN_STATUS_SUCCESS)
                continue;
            for (int64_t i = 0; i < found; ++i) {
                auto plan = std::make_unique<Descriptor>(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
                plan->set(CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, &handle);
                plan->set(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, &raw[i]);
                if (plan->try_finalize() != CUDNN_STATUS_SUCCESS) continue;
                int64_t bytes = 0, count = 0;
                if (cudnnBackendGetAttribute(plan->get(), CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                             CUDNN_TYPE_INT64, 1, &count, &bytes) != CUDNN_STATUS_SUCCESS
                    || bytes < 0 || uint64_t(bytes) > limit)
                    continue;
                std::string name = plan_json(*plan);
                if (name.empty()) name = std::to_string(result.size()) + "/" + std::to_string(i);
                const bool duplicate = std::any_of(result.begin(), result.end(),
                                                   [&](const Candidate& seen) { return seen.name == name; });
                if (duplicate) continue;
                result.push_back({std::move(plan), bytes, std::move(name)});
            }
        }
        return result;
    }

    // A stable identity for deduplicating configurations that both heuristics return.
    static std::string plan_json(const Descriptor& plan) {
        int64_t size = 0;
        if (cudnnBackendGetAttribute(plan.get(), CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, CUDNN_TYPE_CHAR,
                                     0, &size, nullptr)
                != CUDNN_STATUS_SUCCESS
            || size <= 0)
            return {};
        std::string json(size_t(size), '\0');
        if (cudnnBackendGetAttribute(plan.get(), CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, CUDNN_TYPE_CHAR,
                                     size, &size, json.data())
            != CUDNN_STATUS_SUCCESS)
            return {};
        json.resize(std::strlen(json.c_str()));
        return json;
    }
};

// Binds the per-call pointers of one convolution site.
inline std::unique_ptr<Descriptor> bind(void* x, void* filter, void* y, void* workspace) {
    auto variant = std::make_unique<Descriptor>(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    const int64_t ids[] = {1, 2, 3};
    void* data[] = {x, filter, y};
    variant->set(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, ids, 3);
    variant->set(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, data, 3);
    variant->set(CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, &workspace);
    variant->finalize();
    return variant;
}
} // namespace moesr
