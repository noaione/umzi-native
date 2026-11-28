#pragma once

// Shared cuDNN backend (graph API) helpers for convolutions and fused graphs.
//
// Used by MoESR, GFISRv2, and GateRV3 SiLU fusion. Enumerates candidate engine
// configurations from heuristics A and fallback, deduplicates them by JSON
// representation, binds per-call variant packs, and times candidates against a
// bounded device scratch buffer.

#include <cuda.h>
#include <cudnn.h>

#include "cuda_helpers.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace umzi {
class CudnnDescriptor {
    cudnnBackendDescriptor_t handle = nullptr;

public:
    explicit CudnnDescriptor(cudnnBackendDescriptorType_t type) {
        if (cudnnBackendCreateDescriptor(type, &handle) != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error("cudnnBackendCreateDescriptor failed");
    }
    CudnnDescriptor(const CudnnDescriptor&) = delete;
    CudnnDescriptor& operator=(const CudnnDescriptor&) = delete;
    ~CudnnDescriptor() {
        if (handle) cudnnBackendDestroyDescriptor(handle);
    }
    cudnnBackendDescriptor_t get() const { return handle; }
    void set(cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type, const void* values,
             int64_t count = 1) {
        const auto status = cudnnBackendSetAttribute(handle, name, type, count, values);
        if (status != CUDNN_STATUS_SUCCESS)
            throw std::runtime_error(std::string("cudnnBackendSetAttribute: ") + cudnnGetErrorString(status));
    }
    void ref(cudnnBackendAttributeName_t name, const CudnnDescriptor& value) {
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

struct CudnnCandidate {
    std::unique_ptr<CudnnDescriptor> plan;
    int64_t workspace = 0;
    std::string name;
};

class CudnnConvGraph {
    std::vector<std::unique_ptr<CudnnDescriptor>> descriptors;
    const bool strict_float;
    const cudnnTensorFormat_t format;

    static bool preserves_float(cudnnBackendDescriptor_t config) {
        CudnnDescriptor engine(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        auto descriptor = engine.get();
        int64_t count = 0;
        if (cudnnBackendGetAttribute(config, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                     1, &count, &descriptor) != CUDNN_STATUS_SUCCESS || count != 1)
            return false;
        std::array<cudnnBackendNumericalNote_t, CUDNN_NUMERICAL_NOTE_TYPE_COUNT> notes{};
        if (cudnnBackendGetAttribute(engine.get(), CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE,
                                     notes.size(), &count, notes.data()) != CUDNN_STATUS_SUCCESS)
            return false;
        for (int64_t i = 0; i < count; ++i)
            if (notes[i] == CUDNN_NUMERICAL_NOTE_TENSOR_CORE || notes[i] == CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS
                || notes[i] == CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION)
                return false;
        return true;
    }

    CudnnDescriptor& make(cudnnBackendDescriptorType_t type) {
        return *descriptors.emplace_back(std::make_unique<CudnnDescriptor>(type));
    }

    CudnnDescriptor& tensor(int64_t uid, int64_t n, int64_t c, int64_t h, int64_t w, cudnnDataType_t type,
                            bool is_virtual = false) {
        auto& desc = make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        constexpr int64_t alignment = 16;
        const int64_t dimensions[] = {n, c, h, w};
        const auto strides = format == CUDNN_TENSOR_NCHW
            ? std::array<int64_t, 4>{c * h * w, h * w, w, 1}
            : std::array<int64_t, 4>{h * w * c, 1, w * c, c};
        desc.set(CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, &uid);
        desc.set(CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, &type);
        desc.set(CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, dimensions, 4);
        desc.set(CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, strides.data(), 4);
        desc.set(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, &alignment);
        desc.set(CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, &is_virtual);
        desc.finalize();
        return desc;
    }

    CudnnDescriptor& pointwise(cudnnPointwiseMode_t mode, CudnnDescriptor& x, CudnnDescriptor& y,
                              CudnnDescriptor* bias = nullptr) {
        auto& desc = make(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        const auto math = CUDNN_DATA_FLOAT;
        desc.set(CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, &mode);
        desc.set(CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, &math);
        if (mode == CUDNN_POINTWISE_SWISH_FWD) {
            constexpr float beta = 1.f;
            desc.set(CUDNN_ATTR_POINTWISE_SWISH_BETA, CUDNN_TYPE_FLOAT, &beta);
        }
        desc.finalize();
        auto& op = make(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        op.ref(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, desc);
        op.ref(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, x);
        op.ref(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, y);
        if (bias) op.ref(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, *bias);
        op.finalize();
        return op;
    }

public:
    CudnnDescriptor* graph = nullptr;

    CudnnConvGraph(cudnnHandle_t handle, int n, int h, int w, int cin, int cout, int kh, int kw, int groups,
                   cudnnDataType_t io_type, bool fp16_compute, cudnnTensorFormat_t layout = CUDNN_TENSOR_NHWC,
                   bool fused_silu = false, bool float_silu_bias = false)
        : strict_float(io_type == CUDNN_DATA_FLOAT), format(layout) {
        auto& x = tensor(1, n, cin, h, w, io_type);
        auto& filter = tensor(2, cout, cin / groups, kh, kw, io_type);
        auto& y = tensor(3, n, cout, h, w, io_type, fused_silu);
        const auto compute = (io_type == CUDNN_DATA_HALF && fp16_compute) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
        auto& conv = make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        constexpr int64_t spatial = 2, unit[] = {1, 1};
        const int64_t pad[] = {kh / 2, kw / 2};
        const auto mode = CUDNN_CROSS_CORRELATION;
        conv.set(CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, &compute);
        conv.set(CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, &mode);
        conv.set(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, &spatial);
        conv.set(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, pad, 2);
        conv.set(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, pad, 2);
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
        std::array<cudnnBackendDescriptor_t, 3> operations{op.get()};
        if (fused_silu) {
            // Mixed preserves the half convolution output and float bias sum before SiLU.
            const auto bias_type = float_silu_bias ? CUDNN_DATA_FLOAT : io_type;
            auto& biased = tensor(4, n, cout, h, w, bias_type, true);
            auto& bias = tensor(5, 1, cout, 1, 1, bias_type);
            auto& result = tensor(6, n, cout, h, w, io_type);
            operations[1] = pointwise(CUDNN_POINTWISE_ADD, y, biased, &bias).get();
            operations[2] = pointwise(CUDNN_POINTWISE_SWISH_FWD, biased, result).get();
        }
        graph->set(CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, &handle);
        graph->set(CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, operations.data(), fused_silu ? 3 : 1);
        const auto status = graph->try_finalize();
        if (fused_silu && (status == CUDNN_STATUS_NOT_SUPPORTED || status == CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN
            || status == CUDNN_STATUS_NOT_SUPPORTED_SHAPE || status == CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE
            || status == CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH)) {
            graph = nullptr;
        } else {
            check(status, "cudnnBackendFinalize");
        }
    }

    CudnnConvGraph(cudnnHandle_t handle, int n, int h, int w, int cin, int cout, int k, cudnnDataType_t io_type,
                   bool fp16_compute, cudnnTensorFormat_t layout = CUDNN_TENSOR_NHWC, bool fused_silu = false,
                   bool float_silu_bias = false)
        : CudnnConvGraph(handle, n, h, w, cin, cout, k, k, 1, io_type, fp16_compute, layout, fused_silu,
                         float_silu_bias) {}

    CudnnConvGraph(cudnnHandle_t handle, int n, int h, int w, int cin, int cout, bool fp16_compute)
        : CudnnConvGraph(handle, n, h, w, cin, cout, 3, CUDNN_DATA_HALF, fp16_compute) {}

    // Every engine configuration from heuristics A and fallback that finalizes
    // into a plan within the workspace limit, in heuristic order, deduplicated.
    std::vector<CudnnCandidate> candidates(cudnnHandle_t handle, uint64_t limit, bool first_only = false) {
        std::vector<CudnnCandidate> result;
        if (!graph) return result;
        for (auto mode : {CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK}) {
            CudnnDescriptor heuristic(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
            heuristic.set(CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, &mode);
            heuristic.ref(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, *graph);
            if (heuristic.try_finalize() != CUDNN_STATUS_SUCCESS) continue;
            int64_t found = 0;
            if (cudnnBackendGetAttribute(heuristic.get(), CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &found, nullptr) != CUDNN_STATUS_SUCCESS
                || found <= 0)
                continue;
            found = std::min<int64_t>(found, 64);
            std::vector<std::unique_ptr<CudnnDescriptor>> configs;
            std::vector<cudnnBackendDescriptor_t> raw;
            for (int64_t i = 0; i < found; ++i) {
                configs.push_back(std::make_unique<CudnnDescriptor>(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
                raw.push_back(configs.back()->get());
            }
            if (cudnnBackendGetAttribute(heuristic.get(), CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR, found, &found, raw.data())
                != CUDNN_STATUS_SUCCESS)
                continue;
            for (int64_t i = 0; i < found; ++i) {
                if (strict_float && !preserves_float(raw[i])) continue;
                auto plan = std::make_unique<CudnnDescriptor>(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
                plan->set(CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, &handle);
                plan->set(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, &raw[i]);
                if (plan->try_finalize() != CUDNN_STATUS_SUCCESS) continue;
                int64_t bytes = 0, count = 0;
                if (cudnnBackendGetAttribute(plan->get(), CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                             CUDNN_TYPE_INT64, 1, &count, &bytes) != CUDNN_STATUS_SUCCESS
                    || bytes < 0 || uint64_t(bytes) > limit)
                    continue;
                if (first_only) {
                    result.push_back({std::move(plan), bytes, {}});
                    return result;
                }
                std::string name = plan_json(*plan);
                if (name.empty()) name = std::to_string(result.size()) + "/" + std::to_string(i);
                const bool duplicate = std::any_of(result.begin(), result.end(),
                                                   [&](const CudnnCandidate& seen) { return seen.name == name; });
                if (duplicate) continue;
                result.push_back({std::move(plan), bytes, std::move(name)});
            }
        }
        return result;
    }

    static std::string plan_json(const CudnnDescriptor& plan) {
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

inline std::unique_ptr<CudnnDescriptor> bind_conv_variant(void* x, void* filter, void* y, void* workspace,
                                                         void* bias = nullptr) {
    auto variant = std::make_unique<CudnnDescriptor>(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    const int64_t ids[] = {1, 2, bias ? 6 : 3, 5};
    void* data[] = {x, filter, y, bias};
    const auto count = bias ? 4 : 3;
    variant->set(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, ids, count);
    variant->set(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, data, count);
    variant->set(CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, &workspace);
    variant->finalize();
    return variant;
}

struct CudnnBackendPlan {
    std::unique_ptr<CudnnConvGraph> graph;
    std::unique_ptr<CudnnDescriptor> plan;
    size_t workspace = 0;
};

// Grow the tuning scratch to a candidate's requirement. Returns false when the
// device has no room so candidate search can skip oversized plans cleanly.
inline bool ensure_device_scratch(CUstream stream, CUdeviceptr& scratch, size_t& scratch_bytes, size_t required,
                                  size_t* peak_bytes = nullptr) {
    if (required <= scratch_bytes) return true;
    // Allocate in the stream's context, not whichever one the caller left current.
    CUcontext owner = nullptr;
    check(cuStreamGetCtx(stream, &owner), "cuStreamGetCtx");
    ContextScope scope(owner);
    check(cuStreamSynchronize(stream), "cuStreamSynchronize");
    if (scratch) check(cuMemFree(scratch), "cuMemFree");
    scratch = 0;
    scratch_bytes = 0;
    if (required > device_vram_info().free_bytes) return false;
    const auto status = cuMemAlloc(&scratch, required);
    if (status == CUDA_ERROR_OUT_OF_MEMORY) {
        scratch = 0;
        return false;
    }
    check(status, "cuMemAlloc");
    scratch_bytes = required;
    if (peak_bytes) *peak_bytes = std::max(*peak_bytes, required);
    return true;
}

// Time one candidate execution plan. Repeats 5 times only when the first run
// is within 2x of the current best. Returns infinity on failure.
inline float time_cudnn_backend_plan(cudnnHandle_t cudnn, CUstream stream, const std::array<CUevent, 2>& timing,
                                     const CudnnBackendPlan& p, CUdeviceptr in_buf, CUdeviceptr weight_buf,
                                     CUdeviceptr out_buf, CUdeviceptr& scratch, size_t& scratch_bytes, float best,
                                     size_t* peak_bytes = nullptr, uint64_t* peak_device_used = nullptr,
                                     CUdeviceptr bias = 0) {
    const size_t before = scratch_bytes;
    if (!ensure_device_scratch(stream, scratch, scratch_bytes, p.workspace, peak_bytes))
        return std::numeric_limits<float>::infinity();
    if (peak_device_used && scratch_bytes > before) *peak_device_used = std::max(*peak_device_used, device_used_bytes());
    const auto variant = bind_conv_variant(reinterpret_cast<void*>(in_buf), reinterpret_cast<void*>(weight_buf),
                                           reinterpret_cast<void*>(out_buf), reinterpret_cast<void*>(scratch),
                                           reinterpret_cast<void*>(bias));
    const auto run = [&] { return cudnnBackendExecute(cudnn, p.plan->get(), variant->get()) == CUDNN_STATUS_SUCCESS; };
    const auto measure = [&](int repeats) {
        check(cuEventRecord(timing[0], stream), "cuEventRecord");
        for (int i = 0; i < repeats; ++i)
            if (!run()) {
                check(cuStreamSynchronize(stream), "cuStreamSynchronize");
                return std::numeric_limits<float>::infinity();
            }
        check(cuEventRecord(timing[1], stream), "cuEventRecord");
        check(cuEventSynchronize(timing[1]), "cuEventSynchronize");
        float elapsed = 0;
        check(cuEventElapsedTime(&elapsed, timing[0], timing[1]), "cuEventElapsedTime");
        return elapsed / repeats;
    };
    if (!run()) {
        check(cuStreamSynchronize(stream), "cuStreamSynchronize");
        return std::numeric_limits<float>::infinity();
    }
    const float first = measure(1);
    if (!(first < 2.f * best)) return first;
    return measure(5);
}
} // namespace umzi
