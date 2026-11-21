#pragma once

namespace gt {
class BackendDescriptor {
    cudnnBackendDescriptor_t handle = nullptr;

public:
    explicit BackendDescriptor(cudnnBackendDescriptorType_t type) {
        CHECK(cudnnBackendCreateDescriptor(type, &handle));
    }
    BackendDescriptor(const BackendDescriptor&) = delete;
    BackendDescriptor& operator=(const BackendDescriptor&) = delete;
    ~BackendDescriptor() { cudnnBackendDestroyDescriptor(handle); }

    cudnnBackendDescriptor_t get() const { return handle; }
    void set(cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type,
             const void* values, int64_t count = 1) {
        CHECK(cudnnBackendSetAttribute(handle, name, type, count, values));
    }
    void ref(cudnnBackendAttributeName_t name, const BackendDescriptor& value) {
        auto descriptor = value.get();
        set(name, CUDNN_TYPE_BACKEND_DESCRIPTOR, &descriptor);
    }
    void finalize() { CHECK(cudnnBackendFinalize(handle)); }
};

// Fuses bias-free C32 convolutions that feed SiLU at the prepared tile size.
class SiluPlan {
    std::vector<std::unique_ptr<BackendDescriptor>> descriptors;
    std::unique_ptr<BackendDescriptor> plan;

    BackendDescriptor& descriptor(cudnnBackendDescriptorType_t type) {
        return *descriptors.emplace_back(std::make_unique<BackendDescriptor>(type));
    }
    BackendDescriptor& tensor(int64_t uid, int64_t n, int64_t h, int64_t w, bool is_virtual) {
        auto& desc = descriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        constexpr int64_t c = 32, alignment = 16;
        auto type = CUDNN_DATA_HALF;
        const int64_t dimensions[] = {n, c, h, w};
        const int64_t strides[] = {h * w * c, 1, w * c, c};
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
    size_t workspace = 0;

    void initialize(cudnnHandle_t handle, int batch, int height, int width, bool fp16, size_t limit) {
        auto& x = tensor(1, batch, height, width, false);
        auto& weights = tensor(2, 32, 3, 3, false);
        auto& tmp = tensor(3, batch, height, width, true);
        auto& output = tensor(4, batch, height, width, false);

        auto type = fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
        auto& conv_desc = descriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        constexpr int64_t spatial = 2, unit[] = {1, 1};
        auto mode = CUDNN_CROSS_CORRELATION;
        conv_desc.set(CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, &type);
        conv_desc.set(CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, &mode);
        conv_desc.set(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, &spatial);
        conv_desc.set(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, unit, 2);
        conv_desc.set(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, unit, 2);
        conv_desc.set(CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, unit, 2);
        conv_desc.set(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, unit, 2);
        conv_desc.finalize();

        auto& conv = descriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
        constexpr float one = 1, zero = 0;
        conv.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, &one);
        conv.set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, &zero);
        conv.ref(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, conv_desc);
        conv.ref(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, x);
        conv.ref(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, weights);
        conv.ref(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, tmp);
        conv.finalize();

        auto& pointwise_desc = descriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        auto swish = CUDNN_POINTWISE_SWISH_FWD;
        pointwise_desc.set(CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, &swish);
        pointwise_desc.set(CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, &type);
        pointwise_desc.set(CUDNN_ATTR_POINTWISE_SWISH_BETA, CUDNN_TYPE_FLOAT, &one);
        pointwise_desc.finalize();

        auto& silu = descriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        silu.ref(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pointwise_desc);
        silu.ref(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, tmp);
        silu.ref(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, output);
        silu.finalize();

        auto& graph = descriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        const auto ops = std::array{conv.get(), silu.get()};
        graph.set(CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, &handle);
        graph.set(CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, ops.data(), ops.size());
        graph.finalize();

        for (auto heur_mode : {CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK}) {
            auto& heuristic = descriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
            heuristic.set(CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, &heur_mode);
            heuristic.ref(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, graph);
            if (cudnnBackendFinalize(heuristic.get()) != CUDNN_STATUS_SUCCESS) continue;
            constexpr int64_t candidates = 12;
            std::array<cudnnBackendDescriptor_t, candidates> configs{};
            int64_t found = 0;
            CHECK(cudnnBackendGetAttribute(heuristic.get(), CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &found, nullptr));
            if (found <= 0) continue;
            found = std::min(found, candidates);
            for (int64_t i = 0; i < found; ++i)
                configs[i] = descriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR).get();
            CHECK(cudnnBackendGetAttribute(heuristic.get(), CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR, found, &found, configs.data()));
            for (int64_t i = 0; i < found; ++i) {
                auto candidate = std::make_unique<BackendDescriptor>(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
                candidate->set(CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, &handle);
                candidate->set(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, &configs[i]);
                if (cudnnBackendFinalize(candidate->get()) != CUDNN_STATUS_SUCCESS) continue;
                int64_t bytes = 0, count = 0;
                CHECK(cudnnBackendGetAttribute(candidate->get(), CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                               CUDNN_TYPE_INT64, 1, &count, &bytes));
                if (bytes < 0 || static_cast<size_t>(bytes) > limit) continue;
                workspace = static_cast<size_t>(bytes);
                plan = std::move(candidate);
                return;
            }
        }
        throw std::runtime_error("No supported cuDNN convolution/SiLU graph plan");
    }

    std::unique_ptr<BackendDescriptor> bind(void* x, void* weights, void* y, void* scratch) const {
        auto variant = std::make_unique<BackendDescriptor>(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        const int64_t ids[] = {1, 2, 4};
        void* data[] = {x, weights, y};
        variant->set(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, ids, 3);
        variant->set(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, data, 3);
        variant->set(CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, &scratch);
        variant->finalize();
        return variant;
    }

    void run(cudnnHandle_t handle, const BackendDescriptor& bindings) const {
        CHECK(cudnnBackendExecute(handle, plan->get(), bindings.get()));
    }
};
} // namespace gt
