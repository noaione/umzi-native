#pragma once

#include "plan_cache.h"

#include <algorithm>
#include <climits>
#include <limits>

namespace gt {
enum Kind {
    Pack, Conv, Rms, Local, MeanProject, LocalSkip, Inception, GlobalSkip, Shuffle,
    Concat, Silu, Span, Depthwise, Attention, Gate, Add, Rgb, ShuffleConcat,
};

struct Tensor {
    int h, w, c, offset;
    int pixels() const { return h * w; }
    int elements() const { return pixels() * c; }
};

struct Op {
    int kind, out;
    std::array<int, 4> input;
    std::array<int, 3> weight;
    int x, y;
};
static_assert(sizeof(Op) == 44 && sizeof(Tensor) == 16);

struct Shape {
    int n, h, w;

    static Shape checked(uint32_t n, uint32_t h, uint32_t w) {
        if (!n || !h || !w || n > INT_MAX || h > INT_MAX - 15 || w > INT_MAX - 15)
            throw std::runtime_error("Batch, height and width must be positive and fit native indexing");
        return {static_cast<int>(n), static_cast<int>(h), static_cast<int>(w)};
    }
    Shape padded(int alignment) const {
        const int round_up = alignment - 1;
        return {n, (h + round_up) / alignment * alignment, (w + round_up) / alignment * alignment};
    }
    bool operator==(const Shape& other) const { return n == other.n && h == other.h && w == other.w; }
    uint64_t rgb_bytes(bool fp16) const {
        return uint64_t(n) * h * w * 3 * (fp16 ? 2 : 4);
    }
};

struct ResolvedModel {
    std::vector<Tensor> tensors;
    size_t arena_bytes = 0;
    std::vector<int> alias_target;
    std::vector<int> conv_ld_in;
    std::vector<bool> fused_concat;
    std::vector<bool> inplace_inception;
    std::vector<bool> skip_alloc;
};

class ModelFile {
    struct TensorSpec { int level, channels; };
    static_assert(sizeof(TensorSpec) == 8);

    std::vector<TensorSpec> specs;
    std::vector<size_t> last_use;
    int padding_alignment = 16;

    void validate(const std::vector<Tensor>& tensors) {
        last_use.resize(tensors.size());
        for (size_t i = 0; i < ops.size(); ++i) {
            const auto& op = ops[i];
            auto require = [&](bool condition) {
                if (!condition) throw std::runtime_error("Invalid native model operation " + std::to_string(i));
            };
            require(op.out == i + 1 && op.kind >= Pack && op.kind <= ShuffleConcat && op.input[0] >= 0);
            bool ended = false;
            int count = 0;
            for (int id : op.input) {
                require(id >= -1 && id <= int(i) && (!ended || id == -1));
                ended |= id < 0;
                if (id < 0) continue;
                last_use[id] = i;
                ++count;
            }
            last_use[op.out] = i;
            for (int id : op.weight)
                require(id == -1 || (id >= 0 && id % 256 == 0 && size_t(id) < weight_bytes));

            const auto& a = tensors[op.input[0]];
            const auto& y = tensors[op.out];
            const auto& b = tensors[count > 1 ? op.input[1] : op.input[0]];
            auto spatial = [](const Tensor& x, const Tensor& y) { return x.h == y.h && x.w == y.w; };
            auto same = [&](const Tensor& x, const Tensor& y) { return spatial(x, y) && x.c == y.c; };
            auto weight = [&](int slot, size_t size, bool optional = false) {
                int offset = op.weight[slot];
                require(offset < 0 ? optional : size <= weight_bytes - size_t(offset));
            };
            auto local_channels = [](int c) { return c == 32 || c == 64 || c == 128 || c == 256; };
            require(i == 0 ? op.kind == Pack : op.kind != Pack);
            require(i + 1 == ops.size() ? op.kind == Rgb : op.kind != Rgb);
            switch (op.kind) {
            case Pack:
                require(count == 1 && a.c == 3 && y.c == 8 && spatial(a, y));
                break;
            case Conv:
                require(count == 1 && spatial(a, y) && (op.x == 1 || op.x == 3)
                    && op.y >= 0 && op.y < a.c && (op.x == 1 || !op.y)
                    && (a.c - op.y) % 8 == 0 && op.y % 8 == 0 && y.c % 8 == 0);
                weight(0, size_t(a.c - op.y) * y.c * op.x * op.x * 2);
                weight(1, y.c * 2, true);
                break;
            case Rms:
                require(same(a, y) && (local_channels(y.c) || y.c == 512));
                if (count > 1) {
                    require(i && op.input[0] == i && (ops[i - 1].kind == LocalSkip || ops[i - 1].kind == GlobalSkip));
                    const auto& previous = ops[i - 1].input;
                    for (int j = 1; j < 4; ++j) require(op.input[j] == previous[j - 1]);
                } else {
                    require(!i || (ops[i - 1].kind != LocalSkip && ops[i - 1].kind != GlobalSkip));
                }
                weight(0, y.c * 2);
                weight(1, y.c * 2);
                break;
            case Local:
                require(count == 1 && spatial(a, y) && local_channels(y.c) && a.c == 2 * y.c);
                weight(0, y.c * 72);
                weight(1, y.c * 4);
                break;
            case MeanProject:
                require(count == 1 && y.h == 1 && y.w == 1 && local_channels(y.c) && a.c == y.c);
                weight(0, y.c * y.c * 2);
                weight(1, y.c * 2);
                break;
            case LocalSkip:
                require(count == 3 && local_channels(y.c) && same(a, y) && same(tensors[op.input[2]], y)
                    && b.h == 1 && b.w == 1 && b.c == y.c);
                weight(0, y.c * 2);
                break;
            case Inception: {
                int c = 2 * y.c / 3;
                require(count == 1 && spatial(a, y) && local_channels(c) && y.c == 3 * c / 2 && a.c == 2 * y.c);
                weight(0, 31 * (c / 8) * 2);
                weight(1, 3 * (c / 8) * 2);
                break;
            }
            case GlobalSkip:
                require((count == 1 || count == 2) && same(a, y) && (local_channels(y.c) || y.c == 512));
                if (count == 2) {
                    require(same(b, y));
                    weight(0, y.c * 2);
                }
                break;
            case Shuffle:
                require(count == 1 && op.x == 0 && a.h == y.h * 2 && a.w == y.w * 2 && y.c == a.c * 4);
                break;
            case Concat: {
                int channels = 0;
                for (int j = 0; j < count; ++j) {
                    const auto& input = tensors[op.input[j]];
                    require(spatial(input, y) && input.c % 8 == 0);
                    channels += input.c;
                }
                require(channels == y.c && count >= 2);
                break;
            }
            case Silu:
                require(count == 1 && same(a, y) && y.c % 2 == 0);
                break;
            case Span:
            case Add:
                require(count == 2 && same(a, y) && same(b, y) && y.c % 2 == 0);
                break;
            case Depthwise:
                require(count == 1 && spatial(a, y) && a.c == 2560 && y.c == 1536);
                weight(0, 9 * 1536 * 2);
                weight(1, 1536 * 2);
                break;
            case Attention:
                require(count == 1 && spatial(a, y) && a.c == 1536 && y.c == 512);
                weight(0, 16 * sizeof(float), true);
                break;
            case Gate:
                require(count == 2 && spatial(a, y) && spatial(b, y) && a.c == 2560 && b.c == 512 && y.c == 768);
                break;
            case Rgb:
                require(count == 2 && op.input[1] == 0 && spatial(a, y) && same(b, y) && a.c == 8 && y.c == 3);
                weight(0, 4);
                weight(1, 2);
                break;
            case ShuffleConcat:
                require(count == 2 && a.h * 2 == y.h && a.w * 2 == y.w && spatial(b, y)
                    && local_channels(y.c / 2) && y.c == 2 * b.c && a.c == 2 * y.c);
                break;
            }
        }
        for (size_t i = 0; i + 1 < ops.size(); ++i) {
            if (ops[i].kind == Conv && ops[i].x == 3 && ops[i + 1].kind == Silu) {
                int source = ops[i].input[0];
                last_use[source] = std::max(last_use[source], i + 1);
            }
        }
    }

    std::vector<Tensor> tensor_shapes(const Shape& shape) const {
        std::vector<Tensor> tensors;
        tensors.reserve(specs.size());
        for (auto spec : specs) {
            int h = spec.level < 0 ? 1 : shape.h >> spec.level;
            int w = spec.level < 0 ? 1 : shape.w >> spec.level;
            // Check each factor before multiplying by the next one.
            int limit = (INT_MAX - 256) / spec.channels / shape.n;
            if (!h || !w || h > limit / w)
                throw std::runtime_error("Requested shape exceeds native tensor indexing limits");
            tensors.push_back({h, w, spec.channels, -1});
        }
        return tensors;
    }

public:
    Digest content_digest{};
    size_t weight_bytes = 0;
    std::vector<Op> ops;

    Shape padded(Shape shape) const { return shape.padded(padding_alignment); }
    void check_shape(const Shape& shape) const { tensor_shapes(shape); }

    void parse(const umzi::ModelPackage& package) {
        if (package.schema != 1) throw std::runtime_error("Unsupported GateRV3 model payload version");
        auto architecture = package.section(1);
        auto schedule = package.section(2);
        auto weights = package.section(3);
        auto checkpoint = package.section(4);
        if (architecture.size() != 32 || (!checkpoint.empty() && checkpoint.size() != 32))
            throw std::runtime_error("Invalid GateRV3 model metadata size");
        umzi::BinaryReader reader(architecture);
        auto input_channels = reader.number<uint32_t>();
        auto scale = reader.number<uint32_t>();
        auto height_alignment = reader.number<uint32_t>();
        auto width_alignment = reader.number<uint32_t>();
        auto base_channels = reader.number<uint32_t>();
        auto output_channels = reader.number<uint32_t>();
        if (input_channels != 3 || output_channels != 3 || scale != 1 || base_channels != 32
            || (height_alignment != 8 && height_alignment != 16) || height_alignment != width_alignment)
            throw std::runtime_error("Unsupported GateRV3 model architecture or padding alignment");
        padding_alignment = static_cast<int>(height_alignment);
        auto tensors = reader.number<uint32_t>();
        auto operations = reader.number<uint32_t>();
        if (operations < 2 || operations > 2000 || tensors != operations + 1
            || schedule.size() != tensors * sizeof(TensorSpec) + operations * sizeof(Op)
            || weights.empty() || weights.size() > (256u << 20))
            throw std::runtime_error("Invalid GateRV3 execution data or weights");
        content_digest = package.digest;
        specs.resize(tensors);
        ops.resize(operations);
        // The existing GateRV3 payload is a little-endian table of int32 values.
        auto spec_bytes = specs.size() * sizeof(TensorSpec);
        std::memcpy(specs.data(), schedule.data(), spec_bytes);
        std::memcpy(ops.data(), schedule.data() + spec_bytes, ops.size() * sizeof(Op));
        weight_bytes = weights.size();
        const int deepest_level = padding_alignment == 8 ? 3 : 4;
        int max_level = 0;
        for (const auto& spec : specs) {
            if (spec.level < -1 || spec.level > deepest_level || spec.channels < 1 || spec.channels > 2560)
                throw std::runtime_error("Invalid GateRV3 tensor description");
            max_level = std::max(max_level, spec.level);
        }
        if (max_level != deepest_level)
            throw std::runtime_error("GateRV3 padding alignment does not match the schedule depth");
        for (auto spec : {specs.front(), specs.back()})
            if (spec.level || spec.channels != 3) throw std::runtime_error("Invalid GateRV3 RGB shape");
        validate(tensor_shapes({1, 128, 128}));
    }

    ResolvedModel resolve(const Shape& shape) const {
        ResolvedModel plan{tensor_shapes(shape)};
        const size_t num_tensors = plan.tensors.size();
        std::vector<int> uses(num_tensors, 0);
        for (const auto& op : ops) {
            for (int id : op.input) {
                if (id >= 0) ++uses[id];
            }
        }
        std::vector<size_t> effective_last = last_use;
        plan.alias_target.assign(num_tensors, -1);
        plan.skip_alloc.assign(num_tensors, false);
        plan.fused_concat.assign(ops.size(), false);
        plan.inplace_inception.assign(ops.size(), false);
        plan.conv_ld_in.assign(ops.size(), 0);
        for (size_t i = 0; i < ops.size(); ++i) {
            plan.conv_ld_in[i] = plan.tensors[ops[i].input[0]].c;
        }
        for (size_t i = 0; i + 1 < ops.size(); ++i) {
            const auto& cat = ops[i];
            const auto& next = ops[i + 1];
            if (cat.kind != Concat || next.kind != Conv || next.x != 1 || next.y != 0
                || next.input[0] != cat.out || uses[cat.out] != 1)
                continue;
            bool internal_inputs = true;
            for (int id : cat.input) {
                if (id == 0 || id >= int(num_tensors - 1)) internal_inputs = false;
            }
            if (!internal_inputs) continue;
            plan.fused_concat[i] = true;
            plan.skip_alloc[cat.out] = true;
            for (int id : cat.input) {
                if (id >= 0) effective_last[id] = std::max(effective_last[id], i + 1);
            }
        }
        for (size_t i = 0; i < ops.size(); ++i) {
            const auto& op = ops[i];
            if (op.kind != Inception || op.input[1] >= 0 || specs[op.out].level != 0) continue;
            int in_id = op.input[0];
            int out_id = op.out;
            if (in_id <= 0 || in_id + 1 >= int(num_tensors) || uses[in_id] != 1 || effective_last[in_id] != i)
                continue;
            bool conv_consumers_only = true;
            for (size_t c = i + 1; c < ops.size(); ++c) {
                const auto& consumer = ops[c];
                for (int slot = 0; slot < 4; ++slot) {
                    if (consumer.input[slot] != out_id) continue;
                    if (slot != 0 || consumer.kind != Conv || consumer.x != 1 || consumer.y != 0
                        || consumer.input[1] >= 0) {
                        conv_consumers_only = false;
                    }
                }
            }
            if (!conv_consumers_only) continue;
            plan.inplace_inception[i] = true;
            plan.skip_alloc[out_id] = true;
            plan.alias_target[out_id] = in_id;
            effective_last[in_id] = std::max(effective_last[in_id], effective_last[out_id]);
            for (size_t c = i + 1; c < ops.size(); ++c) {
                if (ops[c].input[0] == out_id) plan.conv_ld_in[c] = plan.tensors[in_id].c;
            }
        }

        struct Lifetime { size_t id, first, last, bytes; };
        std::vector<Lifetime> tensors;
        for (size_t i = 1; i + 1 < num_tensors; ++i) {
            if (plan.skip_alloc[i]) continue;
            size_t first = i - 1;
            if (ops[first].kind == Rms && ops[first].input[1] >= 0) --first;
            size_t bytes = (size_t(plan.tensors[i].elements()) * 2 + 255) & ~size_t(255);
            tensors.push_back({i, first, effective_last[i], bytes});
        }
        auto pack_classic = [&](std::vector<int>& offsets, size_t& extent) {
            auto ordered = tensors;
            std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
                return a.bytes != b.bytes ? a.bytes > b.bytes : a.first < b.first;
            });
            offsets.assign(num_tensors, 0);
            extent = 0;
            for (auto cursor = ordered.begin(); cursor != ordered.end(); ++cursor) {
                std::vector<std::pair<size_t, size_t>> conflicts;
                for (auto other = ordered.begin(); other != cursor; ++other) {
                    if (cursor->first <= other->last && other->first <= cursor->last)
                        conflicts.emplace_back(size_t(offsets[other->id]), other->bytes);
                }
                std::sort(conflicts.begin(), conflicts.end());
                size_t offset = 0;
                for (const auto& [start, bytes] : conflicts) {
                    if (offset + cursor->bytes <= start) break;
                    offset = std::max(offset, start + bytes);
                }
                offsets[cursor->id] = offset > INT_MAX ? -1 : static_cast<int>(offset);
                extent = std::max(extent, offset + cursor->bytes);
            }
        };
        auto pack_bounded = [&](std::vector<int>& offsets, size_t& extent) {
            size_t peak_live = 0;
            for (size_t step = 0; step < ops.size(); ++step) {
                size_t live = 0;
                for (const auto& t : tensors) {
                    if (t.first <= step && step <= t.last) live += t.bytes;
                }
                peak_live = std::max(peak_live, live);
            }
            auto ordered = tensors;
            std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
                if (a.bytes != b.bytes) return a.bytes > b.bytes;
                size_t span_a = a.last - a.first, span_b = b.last - b.first;
                return span_a != span_b ? span_a > span_b : a.first < b.first;
            });
            offsets.assign(num_tensors, 0);
            extent = 0;
            for (auto cursor = ordered.begin(); cursor != ordered.end(); ++cursor) {
                std::vector<std::pair<size_t, size_t>> conflicts;
                for (auto other = ordered.begin(); other != cursor; ++other) {
                    if (cursor->first <= other->last && other->first <= cursor->last)
                        conflicts.emplace_back(size_t(offsets[other->id]), other->bytes);
                }
                std::sort(conflicts.begin(), conflicts.end());
                std::vector<std::pair<size_t, size_t>> gaps;
                size_t off = 0;
                for (const auto& [start, bytes] : conflicts) {
                    if (off + cursor->bytes <= start) gaps.emplace_back(off, start);
                    off = std::max(off, start + bytes);
                }
                gaps.emplace_back(off, peak_live);
                size_t chosen = off;
                bool placed_top = false;
                if (cursor->bytes >= peak_live / 2) {
                    for (auto it = gaps.rbegin(); it != gaps.rend(); ++it) {
                        size_t hi = std::min(it->second, peak_live);
                        if (hi >= it->first + cursor->bytes) {
                            chosen = hi - cursor->bytes;
                            placed_top = true;
                            break;
                        }
                    }
                }
                if (!placed_top) {
                    for (const auto& [lo, hi] : gaps) {
                        if (hi >= lo + cursor->bytes) {
                            chosen = lo;
                            break;
                        }
                    }
                }
                offsets[cursor->id] = chosen > INT_MAX ? -1 : static_cast<int>(chosen);
                extent = std::max(extent, chosen + cursor->bytes);
            }
        };
        std::vector<int> classic_offsets, bounded_offsets;
        size_t classic_extent = 0, bounded_extent = 0;
        pack_classic(classic_offsets, classic_extent);
        pack_bounded(bounded_offsets, bounded_extent);
        const auto& chosen_offsets = bounded_extent < classic_extent ? bounded_offsets : classic_offsets;
        size_t extent = std::min(classic_extent, bounded_extent);
        for (const auto& t : tensors) {
            if (chosen_offsets[t.id] < 0)
                throw std::runtime_error("Requested shape exceeds native arena offset limits");
            plan.tensors[t.id].offset = chosen_offsets[t.id];
        }
        for (size_t i = 1; i + 1 < num_tensors; ++i) {
            if (plan.alias_target[i] >= 0) plan.tensors[i].offset = plan.tensors[plan.alias_target[i]].offset;
        }
        plan.arena_bytes = extent * shape.n;
        return plan;
    }
};
} // namespace gt
