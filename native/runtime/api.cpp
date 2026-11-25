#define UMZI_BUILD
#include "kernel.h"
#include "tiling.h"

#include <chrono>
#include <limits>
#include <string>

struct UmziSession {
    std::unique_ptr<umzi::KernelSession> kernel;
    std::unique_ptr<umzi::PageTiler> tiler;
    double model_load_ms = 0;
    std::filesystem::path path;
    umzi::Digest digest{};
    UmziSessionOptions options{};
    std::string cache;
    std::vector<std::pair<std::string, int64_t>> kernel_options;

    std::unique_ptr<umzi::KernelSession> open_edge() const {
        const umzi::ModelPackage package(path);
        if (package.digest != digest)
            throw std::runtime_error("The Umzi model package changed while the variable tiler session was open");
        std::vector<UmziOption> settings;
        for (const auto& option : kernel_options) settings.push_back({option.first.c_str(), option.second});
        auto view = options;
        view.cache_dir = cache.empty() ? nullptr : cache.c_str();
        view.kernel_options = settings.data();
        return umzi::kernel_factory(package.family)(package, view);
    }
};

namespace {
thread_local std::string error_text;
thread_local int error_code = UMZI_OK;

template<class Action>
int report(Action action) noexcept {
    try {
        action();
        return error_code = UMZI_OK;
    } catch (const umzi::AllocationError& error) {
        error_text = error.what();
        return error_code = UMZI_OUT_OF_MEMORY;
    } catch (const std::bad_alloc& error) {
        error_text = error.what();
        return error_code = UMZI_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        error_text = error.what();
        return error_code = UMZI_ERROR;
    }
}

umzi::KernelSession& kernel(const UmziSession* session) {
    if (!session) throw std::runtime_error("An Umzi session is required");
    return *session->kernel;
}

template<class T>
void write(T* out, T value) {
    if (!out || out->struct_size < sizeof(T)) throw std::runtime_error("Invalid Umzi ABI output struct size");
    value.struct_size = out->struct_size;
    *out = value;
}

const UmziShape& checked_shape(const UmziShape* shape) {
    if (!shape || !shape->rank || shape->rank > 4) throw std::runtime_error("Invalid Umzi input rank");
    for (uint32_t i = 0; i < shape->rank; ++i)
        if (!shape->dimensions[i]) throw std::runtime_error("Umzi input dimensions must be positive");
    return *shape;
}

void check_buffers(const UmziIo& io, bool pointers, uint64_t input, uint64_t output) {
    if (!pointers || input < io.input.bytes || output < io.output.bytes)
        throw std::runtime_error("Contiguous input and output require " + std::to_string(io.input.bytes)
            + " and " + std::to_string(io.output.bytes) + " bytes, respectively");
}
}

uint32_t umzi_abi_version() { return 1; }
const char* umzi_error() { return error_text.c_str(); }
int umzi_error_code() { return error_code; }

UmziSession* umzi_session_open(const char* path, const UmziSessionOptions* options) {
    UmziSession* result = nullptr;
    report([&] {
        auto start = std::chrono::steady_clock::now();
        const UmziSessionOptions defaults{sizeof(UmziSessionOptions), 0, UMZI_MIXED, UMZI_CUDA_GRAPH};
        if (!path || !*path) throw std::runtime_error("An Umzi model path is required");
        if (!options) options = &defaults;
        if (options->struct_size < sizeof(*options) || options->reserved || options->device < 0
            || options->precision > UMZI_FP32 || (options->flags & ~UMZI_CUDA_GRAPH)
            || options->kernel_option_count > 64 || (options->kernel_option_count && !options->kernel_options))
            throw std::runtime_error("Invalid Umzi session options");
        auto session = std::make_unique<UmziSession>();
        session->path = std::filesystem::absolute(std::filesystem::u8path(path));
        const umzi::ModelPackage package(session->path);
        session->kernel = umzi::kernel_factory(package.family)(package, *options);
        session->digest = package.digest;
        session->options = *options;
        session->options.cache_dir = nullptr;
        session->options.kernel_options = nullptr;
        if (options->cache_dir) session->cache = options->cache_dir;
        for (uint32_t i = 0; i < options->kernel_option_count; ++i) {
            const auto& option = options->kernel_options[i];
            if (!option.name) throw std::runtime_error("Invalid Umzi kernel option name");
            session->kernel_options.emplace_back(option.name, option.value);
        }
        session->model_load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        result = session.release();
    });
    return result;
}

void umzi_session_close(UmziSession* session) { delete session; }

int umzi_session_model(const UmziSession* session, UmziModelInfo* info) {
    return report([&] { write(info, kernel(session).model_info()); });
}

int umzi_session_tensors(const UmziSession* session, UmziIo* io) {
    return report([&] { write(io, kernel(session).tensors()); });
}

int umzi_session_describe(const UmziSession* session, const UmziShape* shape, UmziIo* io) {
    return report([&] { write(io, kernel(session).describe(checked_shape(shape))); });
}

int umzi_session_prepare(UmziSession* session, const UmziShape* shape) {
    return report([&] { kernel(session).prepare_shape(checked_shape(shape)); });
}

int umzi_session_run(UmziSession* session, const UmziShape* shape,
                     const void* input, uint64_t input_bytes, void* output, uint64_t output_bytes) {
    return report([&] {
        auto& implementation = kernel(session);
        const auto& concrete = checked_shape(shape);
        check_buffers(implementation.describe(concrete), input && output, input_bytes, output_bytes);
        implementation.run_host(concrete, input, output);
    });
}

int umzi_session_enqueue(UmziSession* session, const UmziShape* shape,
                         uint64_t input, uint64_t input_bytes, uint64_t output, uint64_t output_bytes, UmziStream stream) {
    return report([&] {
        auto& implementation = kernel(session);
        const auto& concrete = checked_shape(shape);
        check_buffers(implementation.describe(concrete), input && output, input_bytes, output_bytes);
        implementation.enqueue_device(concrete, input, output, stream);
    });
}

int umzi_session_run_tiled(UmziSession* session, uint32_t height, uint32_t width,
                           const UmziTilingOptions* options, const uint8_t* input,
                           uint64_t input_bytes, void* output, uint64_t output_bytes) {
    return report([&] {
        auto& implementation = kernel(session);
        if (!options || options->struct_size < sizeof(*options))
            throw std::runtime_error("Invalid Umzi tiling options");
        umzi::run_tiled(implementation, session->tiler, session->options.device, height, width,
                        *options, input, input_bytes, output, output_bytes);
    });
}

int umzi_session_run_tiled_variable(UmziSession* session, uint32_t height, uint32_t width,
                                    const UmziTilingOptions* options, const uint8_t* input,
                                    uint64_t input_bytes, void* output, uint64_t output_bytes) {
    return report([&] {
        auto& implementation = kernel(session);
        if (!options || options->struct_size < sizeof(*options))
            throw std::runtime_error("Invalid Umzi tiling options");
        umzi::run_tiled(implementation, session->tiler, session->options.device, height, width,
                        *options, input, input_bytes, output, output_bytes, [&] { return session->open_edge(); });
    });
}

int umzi_session_info(const UmziSession* session, UmziSessionInfo* info) {
    return report([&] {
        auto value = kernel(session).session_info();
        value.model_load_ms = session->model_load_ms;
        write(info, value);
    });
}

uint32_t umzi_session_diagnostic_count(const UmziSession* session) {
    return session ? session->kernel->diagnostic_count() : 0;
}

int umzi_session_diagnostic(const UmziSession* session, uint32_t index, UmziDiagnostic* diagnostic) {
    return report([&] { write(diagnostic, kernel(session).diagnostic(index)); });
}

const char* umzi_session_cache_message(const UmziSession* session) {
    return session ? session->kernel->cache_message() : "";
}

const char* umzi_session_kernel_message(const UmziSession* session) {
    return session ? session->kernel->kernel_message() : "";
}

int umzi_session_profile(UmziSession* session, float* milliseconds, uint32_t count) {
    return report([&] {
        if (!milliseconds || count > std::numeric_limits<int>::max())
            throw std::runtime_error("Invalid Umzi profile buffer");
        kernel(session).profile(milliseconds, count);
    });
}
