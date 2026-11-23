#pragma once

#include "../../include/umzi_native.h"
#include "../../runtime/model_package.h"

#include <memory>

namespace umzi::vk {
class KernelSession {
public:
    virtual ~KernelSession() = default;
    virtual UmziModelInfo model_info() const = 0;
    virtual UmziIo tensors() const = 0;
    virtual UmziIo describe(const UmziShape& shape) const = 0;
    virtual void prepare_shape(const UmziShape& shape) = 0;
    virtual void run_host(const UmziShape& shape, const void* input, void* output) = 0;
    virtual UmziSessionInfo session_info() const = 0;
    virtual uint32_t diagnostic_count() const = 0;
    virtual UmziDiagnostic diagnostic(uint32_t index) const = 0;
    virtual const char* kernel_message() const = 0;
};

using KernelFactory = std::unique_ptr<KernelSession>(*)(const ModelPackage&, const UmziSessionOptions&);
std::unique_ptr<KernelSession> make_gaterv3(const ModelPackage& package, const UmziSessionOptions& options);
KernelFactory kernel_factory(uint32_t family);
}
