#pragma once

struct TileGrid {
    int width, height, input_channels, output_channels;
    int tile_h, tile_w, batch, overlap, context, scale, step, columns, rows;
    int background, input_half, output_half, raw;
    int row, column, page_rows, page_columns, replicate;
};
