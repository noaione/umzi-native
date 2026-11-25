#include "tiling.h"
#include "../common/cuda_helpers.h"
#include "../tiling/grid.h"
#include "tiling_images.h"

#include <algorithm>
#include <climits>
#include <limits>

namespace umzi {
namespace {
struct DeviceBuffer {
    CUdeviceptr ptr = 0;
    explicit DeviceBuffer(size_t bytes) {
        if (bytes) check(cuMemAlloc(&ptr, bytes), "Allocate GPU tiling buffer");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cuMemFree(ptr); }
};

struct HostBuffer {
    void* ptr = nullptr;
    explicit HostBuffer(size_t bytes) {
        if (bytes) check(cuMemAllocHost(&ptr, bytes), "Allocate pinned page buffer");
    }
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    ~HostBuffer() { if (ptr) cuMemFreeHost(ptr); }
};

size_t multiply(size_t value, size_t factor) {
    if (factor && value > std::numeric_limits<size_t>::max() / factor)
        throw std::runtime_error("Umzi tiling buffer size overflow");
    return value * factor;
}

int tile_count(int length, int overlap, int step) {
    return length <= overlap ? 1 : 1 + (length - overlap - 1) / step;
}

void launch(CUfunction function, CUstream stream, size_t work, void** args, unsigned tiles = 1) {
    constexpr unsigned threads = 256;
    const auto blocks = static_cast<unsigned>(std::min<size_t>(1 + (work - 1) / threads, 65535));
    check(cuLaunchKernel(function, blocks, std::min(tiles, 65535u), 1, threads, 1, 1, 0, stream, args, nullptr),
          "Launch GPU tiling kernel");
}

size_t element_size(uint32_t dtype) {
    if (dtype == UMZI_FLOAT16) return 2;
    if (dtype == UMZI_FLOAT32) return 4;
    throw std::runtime_error("GPU tiling requires FP16 or FP32 model I/O");
}
}

PageTiler::PageTiler(int32_t device_id) {
    check(cuDeviceGet(&device, device_id), "Get GPU tiling device");
    check(cuDevicePrimaryCtxRetain(&context, device), "Retain GPU tiling context");
    try {
        ContextScope scope(context);
        load_kernel_module(device, module, tiling_images::embedded_cubins, tiling_images::embedded_ptx);
        check(cuModuleGetFunction(&extract, module, "extract_tiles"), "Load tile extraction kernel");
        check(cuModuleGetFunction(&copy, module, "copy_tiles"), "Load tile copy kernel");
        check(cuModuleGetFunction(&blend, module, "blend_tiles"), "Load tile blending kernel");
        check(cuModuleGetFunction(&finish, module, "finish_page"), "Load page finalization kernel");
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "Create GPU tiling stream");
    } catch (...) {
        if (module) {
            try { ContextScope scope(context); cuModuleUnload(module); } catch (...) {}
        }
        cuDevicePrimaryCtxRelease(device);
        throw;
    }
}

PageTiler::~PageTiler() {
    try {
        ContextScope scope(context);
        if (staging) cuMemFreeHost(staging);
        cuStreamDestroy(stream);
        cuModuleUnload(module);
    } catch (...) {}
    cuDevicePrimaryCtxRelease(device);
}

void* PageTiler::host_staging(size_t bytes) {
    if (bytes > staging_bytes) {
        if (staging) check(cuMemFreeHost(staging), "Release pinned page buffer");
        staging = nullptr;
        staging_bytes = 0;
        check(cuMemAllocHost(&staging, bytes), "Allocate pinned page buffer");
        staging_bytes = bytes;
    }
    return staging;
}

void run_tiled(KernelSession& kernel, std::unique_ptr<PageTiler>& tiler, int32_t device_id,
               uint32_t height, uint32_t width, const UmziTilingOptions& options,
               const uint8_t* input, uint64_t input_bytes, void* output, uint64_t output_bytes) {
    const auto model = kernel.model_info();
    if (!model.scale || !height || !width || !options.tile_size || !options.batch_size
        || std::max({height, width, options.tile_size}) > INT_MAX / model.scale || options.batch_size > INT_MAX
        || model.context > (options.tile_size - 1) / 2 || (model.context && model.scale != 1)
        || options.overlap >= options.tile_size - 2 * model.context
        || (options.background != 0 && options.background != 255) || options.output_format > UMZI_PAGE_FLOAT32)
        throw std::runtime_error("Invalid GPU tiling dimensions, overlap, background, or output format");

    const auto channels = kernel.tensors().input.shape.dimensions[1];
    const UmziShape shape{4, {options.batch_size, channels, options.tile_size, options.tile_size}};
    const auto io = kernel.describe(shape);
    const auto output_channels = io.output.shape.dimensions[1];
    const auto core = options.tile_size - 2 * model.context;
    const auto output_core = core * model.scale;
    const auto input_element = element_size(io.input.dtype), output_element = element_size(io.output.dtype);
    if (!channels || channels > INT_MAX || !output_channels || output_channels > INT_MAX || io.output.shape.rank != 4
        || io.output.shape.dimensions[0] != options.batch_size || io.output.shape.dimensions[2] != output_core
        || io.output.shape.dimensions[3] != output_core)
        throw std::runtime_error("GPU tiling requires a fixed NCHW batch and the model's declared scale and context");
    const auto input_plane = multiply(options.tile_size, options.tile_size);
    const auto output_plane = multiply(output_core, output_core);
    const auto input_samples = multiply(input_plane, options.batch_size);
    const auto output_samples = multiply(output_plane, options.batch_size);
    if (io.input.bytes != multiply(multiply(input_samples, channels), input_element)
        || io.output.bytes != multiply(multiply(output_samples, output_channels), output_element))
        throw std::runtime_error("GPU tiling requires contiguous model I/O");

    const auto out_width = int(width * model.scale), out_height = int(height * model.scale);
    const auto num_pixels = multiply(out_width, out_height);
    const auto num_values = multiply(num_pixels, output_channels);
    const bool raw = options.output_format == UMZI_PAGE_FLOAT32;
    const auto source_size = multiply(multiply(width, height), channels);
    const auto result_size = multiply(num_values, raw ? sizeof(float) : sizeof(uint8_t));
    if (!input || !output || input_bytes < source_size || output_bytes < result_size)
        throw std::runtime_error("Invalid GPU tiling page buffers");

    const int step = int(core - options.overlap);
    TileGrid grid{int(width), int(height), int(channels), int(output_channels), int(options.tile_size),
                  int(options.batch_size), int(options.overlap), int(model.context), int(model.scale), step,
                  tile_count(int(width), int(options.overlap), step), tile_count(int(height), int(options.overlap), step),
                  int(options.background), input_element == 2, output_element == 2, raw};
    if (!tiler) tiler = std::make_unique<PageTiler>(device_id);
    ContextScope scope(tiler->context);
    kernel.prepare_shape(shape);
    const auto stream = tiler->stream;
    const auto sum_size = grid.overlap ? multiply(num_values, sizeof(float)) : 0;
    const auto weight_size = grid.overlap ? multiply(num_pixels, sizeof(float)) : 0;
    const bool reuse_sums = raw && grid.overlap;
    const size_t host_size = std::max(source_size, result_size);
    HostBuffer raw_staging(raw ? host_size : 0);
    void* host = raw ? raw_staging.ptr : tiler->host_staging(host_size);

    size_t arena_bytes = 0;
    const auto reserve = [&](size_t bytes) {
        constexpr size_t alignment = 256;
        const auto remaining = std::numeric_limits<size_t>::max() - arena_bytes;
        if (bytes > remaining - (alignment - 1)) throw std::runtime_error("GPU tiling arena size overflow");
        const auto offset = arena_bytes;
        arena_bytes += (bytes + alignment - 1) & ~(alignment - 1);
        return offset;
    };
    const auto page_offset = reserve(source_size);
    const auto input_offset = reserve(size_t(io.input.bytes));
    const auto output_offset = reserve(size_t(io.output.bytes));
    const auto sums_offset = reserve(sum_size);
    const auto weights_offset = reserve(weight_size);
    const auto result_offset = reuse_sums ? sums_offset : reserve(result_size);
    DeviceBuffer arena(arena_bytes);
    auto page = arena.ptr + page_offset, tile_input = arena.ptr + input_offset, tile_output = arena.ptr + output_offset;
    auto sums = arena.ptr + sums_offset, weights = arena.ptr + weights_offset, destination = arena.ptr + result_offset;

    std::memcpy(host, input, source_size);
    try {
        check(cuMemcpyHtoDAsync(page, host, source_size, stream), "Upload input page");
        if (grid.overlap) {
            check(cuMemsetD32Async(sums, 0, num_values, stream), "Clear tile sums");
            check(cuMemsetD32Async(weights, 0, num_pixels, stream), "Clear tile weights");
        }
        const auto total = uint64_t(grid.columns) * grid.rows;
        const int output_step = step * grid.scale;
        for (uint64_t first = 0; first < total; first += grid.batch) {
            int real = int(std::min<uint64_t>(grid.batch, total - first));
            int first_column = int(first % grid.columns), first_row = int(first / grid.columns);
            void* extract_args[]{&grid, &page, &tile_input, &first_column, &first_row, &real};
            launch(tiler->extract, stream, input_plane, extract_args, grid.batch);
            kernel.enqueue_device(shape, tile_input, tile_output, stream);

            if (!grid.overlap) {
                void* copy_args[]{&grid, &tile_output, &destination, &first_column, &first_row, &real};
                launch(tiler->copy, stream, output_plane, copy_args, real);
                continue;
            }

            const auto last = first + real - 1;
            const int last_row = int(last / grid.columns);
            int left = first_row == last_row ? first_column * output_step : 0;
            int top = first_row * output_step;
            const int right_start = int(last % grid.columns) * output_step;
            const int bottom_start = int(last_row) * output_step;
            int region_width = first_row == last_row
                ? right_start - left + std::min(int(output_core), out_width - right_start) : out_width;
            int region_height = bottom_start - top + std::min(int(output_core), out_height - bottom_start);
            void* blend_args[]{&grid, &tile_output, &sums, &weights, &first, &real, &left, &top,
                               &region_width, &region_height};
            launch(tiler->blend, stream, multiply(region_width, region_height), blend_args);
        }
        if (grid.overlap) {
            void* finish_args[]{&grid, &sums, &weights, &destination};
            launch(tiler->finish, stream, num_pixels, finish_args);
        }
        check(cuMemcpyDtoHAsync(host, destination, result_size, stream), "Download result page");
        check(cuStreamSynchronize(stream), "Wait for GPU tiling");
    } catch (...) {
        cuStreamSynchronize(stream);
        throw;
    }
    std::memcpy(output, host, result_size);
}
}
