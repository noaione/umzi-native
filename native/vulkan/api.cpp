#include "session.h"

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

gt::Shape image_shape(const UmziShape* shape) {
    require(shape && shape->rank == 4 && shape->dimensions[1] == 3, "GateRV3 expects RGB NCHW input");
    return gt::Shape::checked(shape->dimensions[0], shape->dimensions[2], shape->dimensions[3]);
}

void fill_io(UmziIo* io, const UmziShape& shape, uint64_t bytes) {
    require(io && io->struct_size == sizeof(*io), "Invalid Umzi I/O metadata structure");
    io->input = {"input", shape, UMZI_FLOAT32, bytes};
    io->output = {"output", shape, UMZI_FLOAT32, bytes};
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
        auto session = std::make_unique<UmziSession>();
        session->open(path, *options);
        handle = session.release();
    });
    return handle;
}

void umzi_session_close(UmziSession* session) { delete session; }

int umzi_session_model(const UmziSession* session, UmziModelInfo* info) {
    return protect([&] {
        require(session && info && info->struct_size == sizeof(*info), "Invalid Umzi model metadata request");
        *info = {sizeof(*info), "gaterv3", 1, 0};
    });
}

int umzi_session_tensors(const UmziSession* session, UmziIo* io) {
    return protect([&] {
        require(session, "Missing Vulkan session");
        fill_io(io, {4, {0, 3, 0, 0}}, 0);
    });
}

int umzi_session_describe(const UmziSession* session, const UmziShape* shape, UmziIo* io) {
    return protect([&] {
        require(session, "Missing Vulkan session");
        const auto dims = image_shape(shape);
        session->model.check_shape(session->model.padded(dims));
        fill_io(io, *shape, dims.rgb_bytes(false));
    });
}

int umzi_session_prepare(UmziSession* session, const UmziShape* shape) {
    return protect([&] {
        require(session, "Missing Vulkan session");
        session->prepare(image_shape(shape));
    });
}

int umzi_session_run(UmziSession* session, const UmziShape* shape, const void* input, uint64_t input_bytes,
                     void* output, uint64_t output_bytes) {
    return protect([&] {
        require(session && input && output, "Missing Vulkan session or host buffer");
        const auto dims = image_shape(shape);
        session->model.check_shape(session->model.padded(dims));
        const auto bytes = dims.rgb_bytes(false);
        require(input_bytes == bytes && output_bytes == bytes, "Invalid GateRV3 input or output byte count");
        session->run(dims, static_cast<const float*>(input), static_cast<float*>(output));
    });
}

int umzi_session_info(const UmziSession* session, UmziSessionInfo* info) {
    return protect([&] {
        require(session && info && info->struct_size == sizeof(*info), "Invalid Umzi session information request");
        session->info(*info);
    });
}

uint32_t umzi_session_diagnostic_count(const UmziSession* session) {
    return session ? session->diagnostic_count() : 0;
}

int umzi_session_diagnostic(const UmziSession* session, uint32_t index, UmziDiagnostic* diagnostic) {
    return protect([&] {
        require(session && index < session->diagnostic_count() && diagnostic
                && diagnostic->struct_size == sizeof(*diagnostic),
                "Invalid Vulkan diagnostic request");
        *diagnostic = session->diagnostic(index);
    });
}

const char* umzi_session_cache_message(const UmziSession*) { return "In-memory Vulkan pipeline cache per session"; }
const char* umzi_session_kernel_message(const UmziSession* session) {
    return session ? session->device_description.c_str() : "";
}
