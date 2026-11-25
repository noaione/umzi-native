#pragma once

struct TileGrid {
    int width, height, input_channels, output_channels;
    int tile_h, tile_w, batch, overlap, context, scale, step, columns, rows;
    int background, input_half, output_half, raw;
    int row, column, page_rows, page_columns, replicate, strip_height, centered, flush_edges;
};

#ifdef __CUDACC__
__host__ __device__
#endif
inline int tile_origin(int index, int count, int length, int core, int step, bool flush_edges) {
    if (flush_edges && index == count - 1) return length > core ? length - core : 0;
    return index * step;
}
