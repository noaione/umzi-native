#pragma once

#include "kernel.h"

#include <algorithm>
#include <array>
#include <climits>
#include <tuple>

namespace umzi {
struct TileGroup {
    int row, column, rows, columns;
    UmziShape shape;
    UmziIo io;
    uint64_t first = 0, tiles = 0;
};

inline int tile_count(int length, int overlap, int step) {
    return length <= overlap ? 1 : 1 + (length - overlap - 1) / step;
}

inline int64_t align_tile(int64_t length, int alignment) {
    return ((length + alignment - 1) / alignment) * alignment;
}

struct TilePlan {
    std::array<TileGroup, 4> groups{};
    int size = 0, rows = 0, columns = 0;
    long double pixels = 0;
    uint64_t calls = 0;
    bool replicate = false;
};

inline TilePlan plan_tiles(const KernelSession& model, int height, int width, int tile, int batch,
                           int overlap, int context, int scale, bool variable) {
    const int core = tile - 2 * context, step = core - overlap;
    const int columns = tile_count(width, overlap, step), rows = tile_count(height, overlap, step);
    const uint32_t channels = model.tensors().input.shape.dimensions[1];
    const auto fixed = [&] {
        TilePlan plan;
        const UmziShape shape{4, {uint32_t(batch), channels, uint32_t(tile), uint32_t(tile)}};
        plan.groups[0] = {0, 0, rows, columns, shape, model.describe(shape)};
        plan.size = 1;
        plan.rows = rows;
        plan.columns = columns;
        return plan;
    };
    if (!variable) return fixed();

    const auto constraints = model.tiling_constraints();
    if (tile <= constraints.fixed_spatial_through) {
        auto plan = fixed();
        const uint64_t useful = uint64_t(rows) * columns;
        const auto tail = uint32_t(useful % batch);
        if (tail) {
            auto last = plan.groups[0];
            last.shape.dimensions[0] = tail;
            last.io = model.describe(last.shape);
            if (useful == tail) {
                plan.groups[0] = last;
            } else {
                plan.groups[0].tiles = useful - tail;
                last.first = useful - tail;
                last.tiles = tail;
                plan.groups[plan.size++] = last;
            }
        }
        return plan;
    }
    if (tile < constraints.minimum) throw std::runtime_error("Tile is smaller than the model's minimum spatial size");
    const int bucket = std::max(16, constraints.alignment);
    const int tolerance = std::min(64, tile / 8);
    const auto may_absorb = [&](int length, int count) {
        return count > 1 && int64_t(length) - (int64_t(count - 2) * step + core) <= tolerance;
    };
    struct Span { int first, count; int64_t tile; };
    const auto spans = [&](int length, int count) {
        const int64_t needed = int64_t(length) - int64_t(count - 1) * step + 2 * context;
        int64_t tail = align_tile(std::max<int64_t>(needed, constraints.minimum), bucket);
        if (needed <= tile) tail = std::min<int64_t>(tail, tile);
        if (tail == tile) return std::array<Span, 2>{{{0, count, tile}, {0, 0, 0}}};
        return std::array<Span, 2>{{{0, count - 1, tile}, {count - 1, 1, tail}}};
    };

    TilePlan best;
    try {
        best = fixed();
        const uint64_t useful = uint64_t(rows) * columns;
        best.calls = 1 + (useful - 1) / batch;
        const auto padded = align_tile(int64_t(tile) + constraints.internal_border, constraints.alignment);
        best.pixels = static_cast<long double>(best.calls * batch) * padded * padded;
    } catch (const std::runtime_error&) {
        // A smaller variable batch can fit even if the full nominal shape does not.
    }
    const int max_dy = may_absorb(height, rows), max_dx = may_absorb(width, columns);
    for (int dy = 0; dy <= max_dy; ++dy) {
        for (int dx = 0; dx <= max_dx; ++dx) {
            TilePlan candidate;
            candidate.rows = rows - dy;
            candidate.columns = columns - dx;
            bool valid = true;
            for (const auto& y : spans(height, candidate.rows)) {
                for (const auto& x : spans(width, candidate.columns)) {
                    if (!x.count || !y.count) continue;
                    if (std::max(y.tile, x.tile) > INT_MAX / scale) { valid = false; continue; }
                    const uint64_t useful = uint64_t(y.count) * x.count;
                    const auto n = uint32_t(std::min<uint64_t>(useful, batch));
                    const UmziShape shape{4, {n, channels, uint32_t(y.tile), uint32_t(x.tile)}};
                    try {
                        candidate.groups[candidate.size++] = {y.first, x.first, y.count, x.count,
                                                               shape, model.describe(shape)};
                    } catch (const std::runtime_error&) {
                        valid = false;
                        continue;
                    }
                    const auto calls = 1 + (useful - 1) / n;
                    const auto padded_h = align_tile(y.tile + constraints.internal_border, constraints.alignment);
                    const auto padded_w = align_tile(x.tile + constraints.internal_border, constraints.alignment);
                    candidate.pixels += static_cast<long double>(calls * n) * padded_h * padded_w;
                    candidate.calls += calls;
                }
            }
            if (valid && (!best.size || std::tie(candidate.pixels, candidate.calls, candidate.size)
                                         < std::tie(best.pixels, best.calls, best.size)))
                best = candidate;
        }
    }
    if (!best.size) throw std::runtime_error("No variable tile layout fits the model's input limits");
    best.replicate = true;
    return best;
}
}
