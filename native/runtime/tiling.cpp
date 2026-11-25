#include "tiling.h"
#include "tile_plan.h"
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
    DeviceBuffer() = default;
    void allocate(size_t bytes) { check(cuMemAlloc(&ptr, bytes), "Allocate GPU tiling buffer"); }
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

void submit_group(KernelSession& model, PageTiler& tiler, const TileGroup& group, TileGrid grid,
                  CUdeviceptr page, CUdeviceptr tile_input, CUdeviceptr tile_output,
                  CUdeviceptr sums, CUdeviceptr weights, CUdeviceptr destination) {
    const int out_width = grid.width * grid.scale, out_height = grid.height * grid.scale;
    const int output_w = (grid.tile_w - 2 * grid.context) * grid.scale;
    const int output_h = (grid.tile_h - 2 * grid.context) * grid.scale;
    const int output_step = grid.step * grid.scale;
    const auto input_plane = multiply(grid.tile_h, grid.tile_w), output_plane = multiply(output_h, output_w);
    const auto total = group.tiles ? group.first + group.tiles : uint64_t(grid.columns) * grid.rows;
    const auto stream = tiler.stream;
    for (uint64_t first = group.first; first < total; first += grid.batch) {
        int real = int(std::min<uint64_t>(grid.batch, total - first));
        int first_column = int(first % grid.columns), first_row = int(first / grid.columns);
        void* extract_args[]{&grid, &page, &tile_input, &first_column, &first_row, &real};
        launch(tiler.extract, stream, input_plane, extract_args, grid.batch);
        model.enqueue_device(group.shape, tile_input, tile_output, stream);

        if (!grid.overlap) {
            void* copy_args[]{&grid, &tile_output, &destination, &first_column, &first_row, &real};
            launch(tiler.copy, stream, output_plane, copy_args, real);
            continue;
        }

        const auto last = first + real - 1;
        const int last_row = int(last / grid.columns);
        const bool one_row = first_row == last_row;
        int left = (grid.column + (one_row ? first_column : 0)) * output_step;
        int top = (grid.row + first_row) * output_step;
        const int right_start = (grid.column + (one_row ? int(last % grid.columns) : grid.columns - 1)) * output_step;
        const int bottom_start = (grid.row + last_row) * output_step;
        int region_width = right_start - left + std::min(output_w, out_width - right_start);
        int region_height = bottom_start - top + std::min(output_h, out_height - bottom_start);
        void* blend_args[]{&grid, &tile_output, &sums, &weights, &first, &real, &left, &top,
                           &region_width, &region_height};
        launch(tiler.blend, stream, multiply(region_width, region_height), blend_args);
    }
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

KernelSession& PageTiler::edge_runner(const UmziShape& shape, const OpenTileRunner& open) {
    for (auto it = edges.begin(); it != edges.end(); ++it) {
        if (std::equal(std::begin(shape.dimensions), std::end(shape.dimensions), it->shape.dimensions)) {
            std::rotate(it, it + 1, edges.end());
            return *edges.back().kernel;
        }
    }
    std::unique_ptr<KernelSession> runner;
    if (edges.size() == 3) {
        runner = std::move(edges.front().kernel);
        edges.erase(edges.begin());
    }
    const auto prepare = [&] {
        if (!runner) runner = open();
        runner->prepare_shape(shape);
    };
    try {
        prepare();
    } catch (const AllocationError&) {
        if (edges.empty()) throw;
        edges.clear();
        prepare();
    }
    edges.push_back({shape, std::move(runner)});
    return *edges.back().kernel;
}

void run_tiled(KernelSession& kernel, std::unique_ptr<PageTiler>& tiler, int32_t device_id,
               uint32_t height, uint32_t width, const UmziTilingOptions& options,
               const uint8_t* input, uint64_t input_bytes, void* output, uint64_t output_bytes,
               const OpenTileRunner& open_edge) {
    const auto model = kernel.model_info();
    if (!model.scale || !height || !width || !options.tile_size || !options.batch_size
        || std::max({height, width, options.tile_size}) > INT_MAX / model.scale || options.batch_size > INT_MAX
        || model.context > (options.tile_size - 1) / 2 || (model.context && model.scale != 1)
        || options.overlap >= options.tile_size - 2 * model.context
        || (options.background != 0 && options.background != 255) || options.output_format > UMZI_PAGE_FLOAT32)
        throw std::runtime_error("Invalid GPU tiling dimensions, overlap, background, or output format");

    const auto channels = kernel.tensors().input.shape.dimensions[1];
    const auto plan = plan_tiles(kernel, int(height), int(width), int(options.tile_size), int(options.batch_size),
                                 int(options.overlap), int(model.context), int(model.scale), bool(open_edge));
    const auto io = plan.groups[0].io;
    const auto output_channels = io.output.shape.dimensions[1];
    const auto core = options.tile_size - 2 * model.context;
    const auto input_element = element_size(io.input.dtype), output_element = element_size(io.output.dtype);
    if (!channels || channels > INT_MAX || !output_channels || output_channels > INT_MAX)
        throw std::runtime_error("Invalid GPU tiling channel count");
    size_t max_input = 0, max_output = 0;
    for (int i = 0; i < plan.size; ++i) {
        const auto& group = plan.groups[i];
        const auto* dims = group.shape.dimensions;
        const auto& output_tensor = group.io.output;
        const auto output_h = (dims[2] - 2 * model.context) * model.scale;
        const auto output_w = (dims[3] - 2 * model.context) * model.scale;
        if (output_tensor.shape.rank != 4 || output_tensor.shape.dimensions[0] != dims[0]
            || output_tensor.shape.dimensions[1] != output_channels || output_tensor.shape.dimensions[2] != output_h
            || output_tensor.shape.dimensions[3] != output_w || group.io.input.dtype != io.input.dtype
            || output_tensor.dtype != io.output.dtype)
            throw std::runtime_error("GPU tiling requires NCHW and the model's declared scale and context");
        const auto input_size = multiply(multiply(multiply(dims[2], dims[3]), dims[0]), multiply(channels, input_element));
        const auto output_size = multiply(multiply(multiply(output_h, output_w), dims[0]),
                                          multiply(output_channels, output_element));
        if (group.io.input.bytes != input_size || output_tensor.bytes != output_size)
            throw std::runtime_error("GPU tiling requires contiguous model I/O");
        max_input = std::max(max_input, input_size);
        max_output = std::max(max_output, output_size);
    }

    const auto out_width = int(width * model.scale), out_height = int(height * model.scale);
    const auto num_pixels = multiply(out_width, out_height);
    const auto num_values = multiply(num_pixels, output_channels);
    const bool raw = options.output_format == UMZI_PAGE_FLOAT32;
    const auto source_size = multiply(multiply(width, height), channels);
    const auto result_size = multiply(num_values, raw ? sizeof(float) : sizeof(uint8_t));
    if (!input || !output || input_bytes < source_size || output_bytes < result_size)
        throw std::runtime_error("Invalid GPU tiling page buffers");

    const int step = int(core - options.overlap);
    TileGrid grid{int(width), int(height), int(channels), int(output_channels), 0, 0, 0,
                  int(options.overlap), int(model.context), int(model.scale), step, 0, 0,
                  int(options.background), input_element == 2, output_element == 2, raw, 0, 0,
                  plan.rows, plan.columns, plan.replicate};
    if (!tiler) tiler = std::make_unique<PageTiler>(device_id);
    ContextScope scope(tiler->context);
    try {
        kernel.prepare_shape(plan.groups[0].shape);
    } catch (const AllocationError&) {
        if (tiler->edges.empty()) throw;
        tiler->edges.clear();
        kernel.prepare_shape(plan.groups[0].shape);
    }
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
    const auto input_offset = reserve(max_input);
    const auto output_offset = reserve(max_output);
    const auto sums_offset = reserve(sum_size);
    const auto weights_offset = reserve(weight_size);
    const auto result_offset = reuse_sums ? sums_offset : reserve(result_size);
    DeviceBuffer arena;
    try {
        arena.allocate(arena_bytes);
    } catch (const AllocationError&) {
        if (tiler->edges.empty()) throw;
        tiler->edges.clear();
        arena.allocate(arena_bytes);
    }
    auto page = arena.ptr + page_offset, tile_input = arena.ptr + input_offset, tile_output = arena.ptr + output_offset;
    auto sums = arena.ptr + sums_offset, weights = arena.ptr + weights_offset, destination = arena.ptr + result_offset;

    std::memcpy(host, input, source_size);
    try {
        check(cuMemcpyHtoDAsync(page, host, source_size, stream), "Upload input page");
        if (grid.overlap) {
            check(cuMemsetD32Async(sums, 0, num_values, stream), "Clear tile sums");
            check(cuMemsetD32Async(weights, 0, num_pixels, stream), "Clear tile weights");
        }
        for (int i = 0; i < plan.size; ++i) {
            const auto& group = plan.groups[i];
            grid.row = group.row;
            grid.column = group.column;
            grid.rows = group.rows;
            grid.columns = group.columns;
            grid.batch = int(group.shape.dimensions[0]);
            grid.tile_h = int(group.shape.dimensions[2]);
            grid.tile_w = int(group.shape.dimensions[3]);
            auto& runner = i ? tiler->edge_runner(group.shape, open_edge) : kernel;
            submit_group(runner, *tiler, group, grid, page, tile_input, tile_output, sums, weights, destination);
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
