#pragma once

struct TileGrid {
    int width, height, input_channels, output_channels;
    int tile, batch, overlap, context, scale, step, columns, rows;
    int background, input_half, output_half, raw;
};
