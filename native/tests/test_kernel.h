#pragma once

#include "../runtime/kernel.h"

namespace umzi {
class TestKernel final : public KernelSession {
public:
    UmziModelInfo model_info() const override { return {sizeof(UmziModelInfo), "test_kernel", 2, 0}; }

    UmziIo tensors() const override {
        return {sizeof(UmziIo), {"source", {4, {0, 1, 0, 0}}, UMZI_FLOAT32, 0},
            {"enlarged", {4, {0, 1, 0, 0}}, UMZI_FLOAT32, 0}};
    }

    UmziIo describe(const UmziShape& shape) const override {
        if (shape.rank != 4 || shape.dimensions[1] != 1 || shape.dimensions[0] > 8
            || shape.dimensions[2] > 128 || shape.dimensions[3] > 128)
            throw std::runtime_error("Invalid test kernel input shape");
        auto io = tensors();
        io.input.shape = io.output.shape = shape;
        io.input.bytes = uint64_t{shape.dimensions[0]} * shape.dimensions[2] * shape.dimensions[3] * sizeof(float);
        io.output.shape.dimensions[2] *= 2;
        io.output.shape.dimensions[3] *= 2;
        io.output.bytes = io.input.bytes * 4;
        return io;
    }

    void prepare_shape(const UmziShape& shape) override { prepared = describe(shape).input.shape; }

    void run_host(const UmziShape& shape, const void* input, void* output) override {
        prepare_shape(shape);
        auto* src = static_cast<const float*>(input);
        auto* dst = static_cast<float*>(output);
        auto rows = shape.dimensions[2], cols = shape.dimensions[3];
        for (uint32_t batch = 0; batch < shape.dimensions[0]; ++batch)
            for (uint32_t y = 0; y < 2 * rows; ++y)
                for (uint32_t x = 0; x < 2 * cols; ++x)
                    *dst++ = src[(batch * rows + y / 2) * cols + x / 2];
    }

    void enqueue_device(const UmziShape&, uint64_t, uint64_t, UmziStream) override {
        throw std::runtime_error("Test kernel supports host inference");
    }

    UmziSessionInfo session_info() const override {
        UmziSessionInfo info{};
        info.prepared_input = prepared;
        return info;
    }

private:
    UmziShape prepared{};
};

inline std::unique_ptr<KernelSession> make_test_kernel(const ModelPackage& package, const UmziSessionOptions& options) {
    if (package.schema != 1 || package.section(1) != "test" || options.precision != UMZI_MIXED
        || options.kernel_option_count)
        throw std::runtime_error("Invalid test kernel model or options");
    return std::make_unique<TestKernel>();
}
}
