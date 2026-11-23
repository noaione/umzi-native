#include "kernel.h"
#include "../common/context.h"

#include <chrono>

struct UmziSession {
    std::unique_ptr<umzi::vk::KernelSession> kernel;
    double model_load_ms = 0;
};

namespace {
thread_local std::string last_error;
thread_local int last_status = UMZI_OK;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
int protect(Function invoke) noexcept {
    try {
        invoke();
        last_error.clear();
        last_status = UMZI_OK;
    } catch (const std::bad_alloc&) {
        last_error = "Vulkan host allocation failed";
        last_status = UMZI_OUT_OF_MEMORY;
    } catch (const umzi::vk::AllocationError& error) {
        last_error = error.what();
        last_status = UMZI_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        last_error = error.what();
        last_status = UMZI_ERROR;
    }
    return last_status;
}

umzi::vk::KernelSession& kernel(const UmziSession* session) {
    require(session, "Missing Vulkan session");
    return *session->kernel;
}

template<class T>
void write(T* target, T value, const char* error) {
    require(target && target->struct_size == sizeof(T), error);
    *target = value;
}

const UmziShape& checked_shape(const UmziShape* shape) {
    require(shape, "Missing Vulkan input shape");
    return *shape;
}
}

uint32_t umzi_vulkan_abi_version() { return 1; }
uint32_t umzi_abi_version() { return 1; }
const char* umzi_error() { return last_error.c_str(); }
int umzi_error_code() { return last_status; }

int umzi_vulkan_device_count(uint32_t* count) {
    return protect([&] {
        require(count, "Missing Vulkan device count destination");
        umzi::vk::Context context;
        context.open_instance();
        *count = uint32_t(context.adapters().size());
    });
}

int umzi_vulkan_device(uint32_t index, UmziVulkanDevice* info) {
    return protect([&] {
        require(info && info->struct_size == sizeof(*info), "Invalid Vulkan device information structure");
        umzi::vk::Context context;
        context.open_instance();
        auto devices = context.adapters();
        require(index < devices.size(), "Vulkan device index is out of range");
        *info = devices[index].info;
    });
}

UmziSession* umzi_session_open(const char* path, const UmziSessionOptions* options) {
    UmziSession* handle = nullptr;
    protect([&] {
        require(path && *path, "Missing Umzi model package path");
        require(options && options->struct_size == sizeof(*options), "Invalid Umzi session options");
        require(options->precision <= UMZI_FP32, "Vulkan precision must be 'mixed', 'fp16', or 'fp32'");
        require(options->flags == 0 && options->reserved == 0 && !options->cache_dir,
                "Vulkan sessions do not accept CUDA graphs or disk caches");
        require(!options->kernel_option_count || options->kernel_options, "Missing Vulkan kernel options");
        const auto started = std::chrono::steady_clock::now();
        const std::string_view utf8_path(path);
        const umzi::ModelPackage package(std::filesystem::path(std::u8string(utf8_path.begin(), utf8_path.end())));
        auto session = std::make_unique<UmziSession>();
        session->kernel = umzi::vk::kernel_factory(package.family)(package, *options);
        session->model_load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        handle = session.release();
    });
    return handle;
}

void umzi_session_close(UmziSession* session) { delete session; }

int umzi_session_model(const UmziSession* session, UmziModelInfo* info) {
    return protect([&] { write(info, kernel(session).model_info(), "Invalid Umzi model metadata request"); });
}

int umzi_session_tensors(const UmziSession* session, UmziIo* io) {
    return protect([&] { write(io, kernel(session).tensors(), "Invalid Umzi I/O metadata structure"); });
}

int umzi_session_describe(const UmziSession* session, const UmziShape* shape, UmziIo* io) {
    return protect([&] {
        write(io, kernel(session).describe(checked_shape(shape)), "Invalid Umzi I/O metadata structure");
    });
}

int umzi_session_prepare(UmziSession* session, const UmziShape* shape) {
    return protect([&] { kernel(session).prepare_shape(checked_shape(shape)); });
}

int umzi_session_run(UmziSession* session, const UmziShape* shape, const void* input, uint64_t input_bytes,
                     void* output, uint64_t output_bytes) {
    return protect([&] {
        require(input && output, "Missing Vulkan host buffer");
        auto& implementation = kernel(session);
        const auto& dimensions = checked_shape(shape);
        const auto io = implementation.describe(dimensions);
        require(input_bytes == io.input.bytes && output_bytes == io.output.bytes,
                "Invalid Vulkan input or output byte count");
        implementation.run_host(dimensions, input, output);
    });
}

int umzi_session_info(const UmziSession* session, UmziSessionInfo* info) {
    return protect([&] {
        auto value = kernel(session).session_info();
        value.model_load_ms = session->model_load_ms;
        write(info, value, "Invalid Umzi session information request");
    });
}

uint32_t umzi_session_diagnostic_count(const UmziSession* session) {
    return session ? session->kernel->diagnostic_count() : 0;
}

int umzi_session_diagnostic(const UmziSession* session, uint32_t index, UmziDiagnostic* diagnostic) {
    return protect([&] {
        const auto& implementation = kernel(session);
        require(index < implementation.diagnostic_count(), "Invalid Vulkan diagnostic index");
        write(diagnostic, implementation.diagnostic(index), "Invalid Vulkan diagnostic request");
    });
}

const char* umzi_session_cache_message(const UmziSession*) { return "In-memory Vulkan pipeline cache per session"; }
const char* umzi_session_kernel_message(const UmziSession* session) {
    return session ? session->kernel->kernel_message() : "";
}
