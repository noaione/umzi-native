#pragma once

#include "../include/umzi_native.h"
#include "model_package.h"

#include <memory>

namespace umzi {
struct AllocationError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct TilingConstraints {
    int alignment = 1, minimum = 1, internal_border = 0;
    int fixed_spatial_through = 0;
};

class KernelSession {
public:
    virtual ~KernelSession() = default;
    virtual UmziModelInfo model_info() const = 0;
    virtual TilingConstraints tiling_constraints() const { return {}; }
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
    virtual std::unique_ptr<KernelSession> clone_shared() const { return nullptr; }

    // kernel_message() plus any ignored-option warnings, for the C API.
    const char* kernel_report() const {
        report_ = kernel_message();
        if (!warnings_.empty()) report_ += (report_.empty() ? "" : "; ") + warnings_;
        return report_.c_str();
    }

    // Where this session's package came from; set by the API so a fallback can re-read the weights.
    void set_package_source(std::filesystem::path path, const Digest& digest) {
        package_path_ = std::move(path);
        package_digest_ = digest;
    }

protected:
    void copy_base_state(const KernelSession& source) {
        warnings_ = source.warnings_;
        package_path_ = source.package_path_;
        package_digest_ = source.package_digest_;
    }
    // Re-reads the package this session was opened from, refusing a file that changed since.
    ModelPackage reload_package() const {
        if (package_path_.empty()) throw std::runtime_error("The model package path is unknown");
        ModelPackage package(package_path_);
        if (package.digest != package_digest_)
            throw std::runtime_error("The Umzi model package changed while the session was open");
        return package;
    }
    // Ignore a bad kernel option instead of failing the session; surfaced via kernel_report().
    void warn_option(const std::string& text) { warnings_ += (warnings_.empty() ? "" : "; ") + text; }

private:
    std::string warnings_;
    std::filesystem::path package_path_;
    Digest package_digest_{};
    mutable std::string report_;
};

using KernelFactory = std::unique_ptr<KernelSession>(*)(const ModelPackage&, const UmziSessionOptions&);
std::unique_ptr<KernelSession> make_gaterv3(const ModelPackage& package, const UmziSessionOptions& options);
std::unique_ptr<KernelSession> make_gfisrv2(const ModelPackage& package, const UmziSessionOptions& options);
std::unique_ptr<KernelSession> make_gaterv2(const ModelPackage& package, const UmziSessionOptions& options);
std::unique_ptr<KernelSession> make_moesr(const ModelPackage& package, const UmziSessionOptions& options);
std::unique_ptr<KernelSession> make_smosr(const ModelPackage& package, const UmziSessionOptions& options);
std::unique_ptr<KernelSession> make_figsr(const ModelPackage& package, const UmziSessionOptions& options);
KernelFactory kernel_factory(uint32_t family);
}
