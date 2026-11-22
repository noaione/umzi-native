#pragma once

#include "../include/umzi_native.h"
#include "model_package.h"

#include <memory>

namespace umzi {
struct AllocationError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class KernelSession {
public:
    virtual ~KernelSession() = default;
    virtual UmziModelInfo model_info() const = 0;
    virtual UmziIo tensors() const = 0;
    virtual UmziIo describe(const UmziShape& shape) const = 0;
    virtual void prepare_shape(const UmziShape& shape) = 0;
    virtual void run_host(const UmziShape& shape, const void* input, void* output) = 0;
    virtual void enqueue_device(const UmziShape& shape, uint64_t input, uint64_t output, UmziStream stream) = 0;
    virtual UmziSessionInfo session_info() const = 0;
    virtual uint32_t diagnostic_count() const { return 0; }
    virtual UmziDiagnostic diagnostic(uint32_t) const { throw std::runtime_error("Invalid kernel diagnostic index"); }
    virtual const char* cache_message() const { return ""; }
    virtual const char* kernel_message() const { return ""; }
    virtual void profile(float*, uint32_t) { throw std::runtime_error("Kernel profiling is unavailable"); }
};

using KernelFactory = std::unique_ptr<KernelSession>(*)(const ModelPackage&, const UmziSessionOptions&);
std::unique_ptr<KernelSession> make_gaterv3(const ModelPackage& package, const UmziSessionOptions& options);
std::unique_ptr<KernelSession> make_gfisrv2(const ModelPackage& package, const UmziSessionOptions& options);
std::unique_ptr<KernelSession> make_gaterv2(const ModelPackage& package, const UmziSessionOptions& options);
std::unique_ptr<KernelSession> make_moesr(const ModelPackage& package, const UmziSessionOptions& options);
KernelFactory kernel_factory(uint32_t family);
}
