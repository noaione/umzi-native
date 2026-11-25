#include <cuda_fp16.h>
#include "grid.h"

using Index = unsigned long long;

__device__ Index thread_index() { return Index(blockIdx.x) * blockDim.x + threadIdx.x; }
__device__ Index thread_stride() { return Index(gridDim.x) * blockDim.x; }

extern "C" __global__ void extract_tiles(TileGrid grid, const unsigned char* page, void* tiles,
                                         int first_column, int first_row, int real) {
    const Index plane = Index(grid.tile_h) * grid.tile_w;
    const bool edge_padding = grid.overlap || grid.context || grid.replicate;
    for (Index slot = blockIdx.y; slot < grid.batch; slot += gridDim.y) {
        const unsigned tile = unsigned(first_column) + min(int(slot), real - 1);
        const int left = (grid.column + int(tile % grid.columns)) * grid.step - grid.context;
        const int top = (grid.row + first_row + int(tile / grid.columns)) * grid.step - grid.context;
        for (Index pixel = thread_index(); pixel < plane; pixel += thread_stride()) {
            const long long x = (long long)left + int(pixel % grid.tile_w);
            const long long y = (long long)top + int(pixel / grid.tile_w);
            const int sx = int(max(0LL, min(x, (long long)grid.width - 1)));
            const int sy = int(max(0LL, min(y, (long long)grid.height - 1)));
            const bool read_page = edge_padding || (sx == x && sy == y);
            const Index source = (Index(sy) * grid.width + sx) * grid.input_channels;
            const Index target = slot * grid.input_channels * plane + pixel;
            for (int channel = 0; channel < grid.input_channels; ++channel) {
                const unsigned char value = read_page ? page[source + channel] : grid.background;
                const float normalized = __fdiv_rn(float(value), 255.0f);
                const Index index = target + channel * plane;
                if (grid.input_half) static_cast<__half*>(tiles)[index] = __float2half_rn(normalized);
                else static_cast<float*>(tiles)[index] = normalized;
            }
        }
    }
}

__device__ float feather(int offset, int length, int overlap, bool previous, bool next) {
    const int ramp = min(overlap, length);
    float value = 1.0f;
    if (previous && offset < ramp) value = __fdiv_rn(float(offset + 1), float(ramp + 1));
    if (next && offset >= length - ramp) value *= __fdiv_rn(float(length - offset), float(ramp + 1));
    return value;
}

__device__ void store_pixel(void* output, Index index, float value, bool raw) {
    if (raw) static_cast<float*>(output)[index] = value;
    else static_cast<unsigned char*>(output)[index] = __float2uint_rn(__saturatef(value) * 255.0f);
}

__device__ float load_tile(const void* tiles, Index index, bool half) {
    return half ? __half2float(static_cast<const __half*>(tiles)[index]) : static_cast<const float*>(tiles)[index];
}

extern "C" __global__ void copy_tiles(TileGrid grid, const void* tiles, void* output,
                                      int first_column, int first_row, int real) {
    const int width = grid.width * grid.scale, height = grid.height * grid.scale;
    const int core_w = (grid.tile_w - 2 * grid.context) * grid.scale;
    const int core_h = (grid.tile_h - 2 * grid.context) * grid.scale;
    const int step = grid.step * grid.scale;
    const Index plane = Index(core_w) * core_h;
    for (Index slot = blockIdx.y; slot < real; slot += gridDim.y) {
        const unsigned tile = unsigned(first_column) + unsigned(slot);
        const int left = (grid.column + int(tile % grid.columns)) * step;
        const int top = (grid.row + first_row + int(tile / grid.columns)) * step;
        const int valid_width = min(core_w, width - left), valid_height = min(core_h, height - top);
        for (Index pixel = thread_index(); pixel < plane; pixel += thread_stride()) {
            const int x = pixel % core_w, y = pixel / core_w;
            if (x >= valid_width || y >= valid_height) continue;
            const Index source = slot * grid.output_channels * plane + pixel;
            const Index target = (Index(top + y) * width + left + x) * grid.output_channels;
            for (int channel = 0; channel < grid.output_channels; ++channel) {
                const float value = load_tile(tiles, source + channel * plane, grid.output_half);
                store_pixel(output, target + channel, value, grid.raw);
            }
        }
    }
}

extern "C" __global__ void blend_tiles(TileGrid grid, const void* tiles, float* sums, float* weights,
                                      Index first, int real, int left, int top,
                                      int region_width, int region_height) {
    const int width = grid.width * grid.scale, height = grid.height * grid.scale;
    const int core_w = (grid.tile_w - 2 * grid.context) * grid.scale;
    const int core_h = (grid.tile_h - 2 * grid.context) * grid.scale;
    const int overlap = grid.overlap * grid.scale, step = grid.step * grid.scale;
    const Index plane = Index(core_w) * core_h;
    const Index last = first + real - 1;
    const int x0 = left + int(blockIdx.x) * blockDim.x + threadIdx.x;
    const int x_end = left + region_width;
    const int x_stride = int(gridDim.x) * blockDim.x;
    for (int dy_reg = int(blockIdx.y); dy_reg < region_height; dy_reg += int(gridDim.y)) {
        const int y = top + dy_reg;
        const int local_y = y - grid.row * step;
        const int y_min = max(local_y < core_h ? 0 : (local_y - core_h) / step + 1, int(first / grid.columns));
        const int y_max = min(min(local_y / step, grid.rows - 1), int(last / grid.columns));
        const int ring_y = grid.strip_height ? (y % grid.strip_height) : y;
        const Index ring_row = Index(ring_y) * width;
        for (int x = x0; x < x_end; x += x_stride) {
            const int local_x = x - grid.column * step;
            const int x_min = local_x < core_w ? 0 : (local_x - core_w) / step + 1;
            const int x_max = min(local_x / step, grid.columns - 1);
            const Index pixel = ring_row + x;
            for (int row = y_min; row <= y_max; ++row) {
                const Index row_start = Index(row) * grid.columns;
                const int start = row_start < first ? max(x_min, int(first - row_start)) : x_min;
                const int end = min(x_max, int(min(last - row_start, Index(grid.columns - 1))));
                const int dy = local_y - row * step, page_row = grid.row + row;
                const float wy = feather(dy, min(core_h, height - page_row * step), overlap,
                                         page_row > 0, page_row + 1 < grid.page_rows);
                for (int column = start; column <= end; ++column) {
                    const int dx = local_x - column * step, page_column = grid.column + column;
                    const Index sample = row_start + column - first;
                    const Index source = sample * grid.output_channels * plane + Index(dy) * core_w + dx;
                    const float weight = wy * feather(dx, min(core_w, width - page_column * step), overlap,
                                                      page_column > 0, page_column + 1 < grid.page_columns);
                    for (int channel = 0; channel < grid.output_channels; ++channel) {
                        const Index from = source + plane * channel;
                        const Index to = pixel * grid.output_channels + channel;
                        const float value = load_tile(tiles, from, grid.output_half);
                        sums[to] = __fadd_rn(sums[to], __fmul_rn(value, weight));
                    }
                    weights[pixel] += weight;
                }
            }
        }
    }
}

extern "C" __global__ void finish_page(TileGrid grid, const float* sums, const float* weights, void* output) {
    const Index pixels = Index(grid.width) * grid.height * grid.scale * grid.scale;
    for (Index pixel = thread_index(); pixel < pixels; pixel += thread_stride()) {
        const float weight = weights[pixel];
        for (int channel = 0; channel < grid.output_channels; ++channel) {
            const Index index = pixel * grid.output_channels + channel;
            store_pixel(output, index, weight > 0.f ? sums[index] / weight : 0.f, grid.raw); // uncovered pixel: 0, not 0/0
        }
    }
}

extern "C" __global__ void finish_strip(TileGrid grid, float* sums, float* weights, void* output,
                                        int y_start, int rows) {
    const int width = grid.width * grid.scale;
    const int x0 = int(blockIdx.x) * blockDim.x + threadIdx.x;
    const int x_stride = int(gridDim.x) * blockDim.x;
    for (int local_y = int(blockIdx.y); local_y < rows; local_y += int(gridDim.y)) {
        const int y = y_start + local_y;
        const int ring_y = grid.strip_height ? (y % grid.strip_height) : y;
        const Index ring_row = Index(ring_y) * width;
        const Index out_row = Index(y) * width;
        for (int x = x0; x < width; x += x_stride) {
            const Index ring_pixel = ring_row + x;
            const Index out_pixel = out_row + x;
            const float weight = weights[ring_pixel];
            weights[ring_pixel] = 0.f;
            for (int channel = 0; channel < grid.output_channels; ++channel) {
                const Index from = ring_pixel * grid.output_channels + channel;
                store_pixel(output, out_pixel * grid.output_channels + channel, weight > 0.f ? sums[from] / weight : 0.f, grid.raw);
                sums[from] = 0.f;
            }
        }
    }
}
