#include <cuda_fp16.h>
#include "grid.h"

using Index = unsigned long long;

__device__ Index thread_index() { return Index(blockIdx.x) * blockDim.x + threadIdx.x; }
__device__ Index thread_stride() { return Index(gridDim.x) * blockDim.x; }

extern "C" __global__ void extract_tiles(TileGrid grid, const unsigned char* page, void* tiles,
                                         int first_column, int first_row, int real) {
    const Index plane = Index(grid.tile) * grid.tile;
    const bool edge_padding = grid.overlap || grid.context;
    for (Index slot = blockIdx.y; slot < grid.batch; slot += gridDim.y) {
        const unsigned tile = unsigned(first_column) + min(int(slot), real - 1);
        const int left = int(tile % grid.columns) * grid.step - grid.context;
        const int top = (first_row + int(tile / grid.columns)) * grid.step - grid.context;
        for (Index pixel = thread_index(); pixel < plane; pixel += thread_stride()) {
            const long long x = (long long)left + int(pixel % grid.tile);
            const long long y = (long long)top + int(pixel / grid.tile);
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
    const int core = (grid.tile - 2 * grid.context) * grid.scale;
    const Index plane = Index(core) * core;
    for (Index slot = blockIdx.y; slot < real; slot += gridDim.y) {
        const unsigned tile = unsigned(first_column) + unsigned(slot);
        const int left = int(tile % grid.columns) * core;
        const int top = (first_row + int(tile / grid.columns)) * core;
        const int valid_width = min(core, width - left), valid_height = min(core, height - top);
        for (Index pixel = thread_index(); pixel < plane; pixel += thread_stride()) {
            const int x = pixel % core, y = pixel / core;
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
    const int core = (grid.tile - 2 * grid.context) * grid.scale;
    const int overlap = grid.overlap * grid.scale, step = grid.step * grid.scale;
    const Index plane = Index(core) * core;
    const Index last = first + real - 1;
    for (Index index = thread_index(); index < Index(region_width) * region_height; index += thread_stride()) {
        const int x = left + index % region_width, y = top + index / region_width;
        const int x_min = x < core ? 0 : (x - core) / step + 1;
        const int y_min = max(y < core ? 0 : (y - core) / step + 1, int(first / grid.columns));
        const int x_max = min(x / step, grid.columns - 1);
        const int y_max = min(min(y / step, grid.rows - 1), int(last / grid.columns));
        const Index pixel = Index(y) * width + x;
        for (int row = y_min; row <= y_max; ++row) {
            const Index row_start = Index(row) * grid.columns;
            const int start = row_start < first ? max(x_min, int(first - row_start)) : x_min;
            const int end = min(x_max, int(min(last - row_start, Index(grid.columns - 1))));
            const int dy = y - row * step;
            const float wy = feather(dy, min(core, height - row * step), overlap, row > 0, row + 1 < grid.rows);
            for (int column = start; column <= end; ++column) {
                const int dx = x - column * step;
                const Index sample = row_start + column - first;
                const Index source = sample * grid.output_channels * plane + Index(dy) * core + dx;
                const float weight = wy * feather(dx, min(core, width - column * step), overlap,
                                                  column > 0, column + 1 < grid.columns);
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

extern "C" __global__ void finish_page(TileGrid grid, const float* sums, const float* weights, void* output) {
    const Index pixels = Index(grid.width) * grid.height * grid.scale * grid.scale;
    for (Index pixel = thread_index(); pixel < pixels; pixel += thread_stride()) {
        const float weight = weights[pixel];
        for (int channel = 0; channel < grid.output_channels; ++channel) {
            const Index index = pixel * grid.output_channels + channel;
            store_pixel(output, index, sums[index] / weight, grid.raw);
        }
    }
}
