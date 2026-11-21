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
    Shape padded() const { return {n, (h + 15) / 16 * 16, (w + 15) / 16 * 16}; }
    bool operator==(const Shape& other) const { return n == other.n && h == other.h && w == other.w; }
    uint64_t rgb_bytes(bool fp16) const {
        return uint64_t(n) * h * w * 3 * (fp16 ? 2 : 4);
    }
};

struct ResolvedModel {
    std::vector<Tensor> tensors;
    size_t arena_bytes = 0;
};

class ModelFile {
    static constexpr char magic[] = "GTR3PCK";
    static constexpr uint32_t format_version = 1;

    struct TensorSpec { int level, channels; };
    struct Section { uint32_t kind, reserved; uint64_t offset, size; };
    struct Header { std::array<char, 8> magic; uint32_t version, sections; Digest digest; };
    static_assert(sizeof(Header) == 48 && sizeof(Section) == 24 && sizeof(TensorSpec) == 8);

    std::vector<TensorSpec> specs;
    std::vector<size_t> last_use;

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
    Digest content_digest{}, checkpoint{};
    size_t weight_offset = 0, weight_bytes = 0;
    std::vector<Op> ops;

    void check_shape(const Shape& shape) const { tensor_shapes(shape); }

    void parse(const std::vector<char>& bytes) {
        auto copy = [&](size_t position, void* target, size_t count) {
            if (position > bytes.size() || count > bytes.size() - position)
                throw std::runtime_error("Truncated GateRV3 model");
            std::memcpy(target, bytes.data() + position, count);
        };
        Header header{};
        copy(0, &header, sizeof(header));
        if (std::memcmp(header.magic.data(), magic, sizeof(magic))
            || header.version != format_version || header.sections != 4)
            throw std::runtime_error("Unsupported GateRV3 model format");
        std::array<Section, 4> sections{};
        copy(sizeof(header), sections.data(), sizeof(sections));
        size_t position = sizeof(header) + sizeof(sections);
        for (size_t i = 0; i < sections.size(); ++i) {
            const auto& section = sections[i];
            if (section.kind != i + 1 || section.reserved || section.offset != position
                || section.size > bytes.size() - position)
                throw std::runtime_error("Invalid GateRV3 model section table");
            position += static_cast<size_t>(section.size);
        }
        if (position != bytes.size()) throw std::runtime_error("Unexpected trailing GateRV3 model data");
        Sha256 hash;
        hash.add(bytes.data() + sizeof(header), bytes.size() - sizeof(header));
        content_digest = hash.finish();
        if (content_digest != header.digest) throw std::runtime_error("GateRV3 model checksum mismatch");

        std::array<uint32_t, 8> metadata{};
        if (sections[0].size != sizeof(metadata) || sections[3].size != sizeof(checkpoint))
            throw std::runtime_error("Invalid GateRV3 model metadata size");
        copy(sections[0].offset, metadata.data(), sizeof(metadata));
        constexpr std::array<uint32_t, 6> required{3, 1, 16, 16, 32, 3};
        uint32_t tensors = metadata[6], operations = metadata[7];
        if (!std::equal(required.begin(), required.end(), metadata.begin())
            || operations < 2 || operations > 2000 || tensors != operations + 1
            || sections[1].size != tensors * sizeof(TensorSpec) + operations * sizeof(Op)
            || !sections[2].size)
            throw std::runtime_error("Unsupported GateRV3 model architecture");
        specs.resize(tensors);
        ops.resize(operations);
        copy(sections[1].offset, specs.data(), specs.size() * sizeof(TensorSpec));
        copy(sections[1].offset + specs.size() * sizeof(TensorSpec), ops.data(), ops.size() * sizeof(Op));
        copy(sections[3].offset, checkpoint.data(), sizeof(checkpoint));
        weight_offset = sections[2].offset;
        weight_bytes = sections[2].size;
        for (const auto& spec : specs) {
            if (spec.level < -1 || spec.level > 4 || spec.channels < 1 || spec.channels > 2560)
                throw std::runtime_error("Invalid GateRV3 tensor description");
        }
        for (auto spec : {specs.front(), specs.back()})
            if (spec.level || spec.channels != 3) throw std::runtime_error("Invalid GateRV3 RGB shape");
        validate(tensor_shapes({1, 128, 128}));
    }

    ResolvedModel resolve(const Shape& shape) const {
        ResolvedModel plan{tensor_shapes(shape)};
        struct Lifetime { size_t id, first, last, bytes; };
        std::vector<Lifetime> tensors;
        for (size_t i = 1; i + 1 < plan.tensors.size(); ++i) {
            size_t first = i - 1;
            if (ops[first].kind == Rms && ops[first].input[1] >= 0) --first;
            size_t bytes = (size_t(plan.tensors[i].elements()) * 2 + 255) & ~size_t(255);
            tensors.push_back({i, first, last_use[i], bytes});
        }
        std::stable_sort(tensors.begin(), tensors.end(), [](const auto& a, const auto& b) {
            return a.bytes != b.bytes ? a.bytes > b.bytes : a.first < b.first;
        });
        size_t extent = 0;
        for (auto cursor = tensors.begin(); cursor != tensors.end(); ++cursor) {
            std::vector<std::pair<size_t, size_t>> conflicts;
            for (auto other = tensors.begin(); other != cursor; ++other) {
                if (cursor->first <= other->last && other->first <= cursor->last)
                    conflicts.emplace_back(plan.tensors[other->id].offset, other->bytes);
            }
            std::sort(conflicts.begin(), conflicts.end());
            size_t offset = 0;
            for (const auto& [start, bytes] : conflicts) {
                if (offset + cursor->bytes <= start) break;
                offset = std::max(offset, start + bytes);
            }
            if (offset > INT_MAX) throw std::runtime_error("Requested shape exceeds native arena offset limits");
            plan.tensors[cursor->id].offset = static_cast<int>(offset);
            extent = std::max(extent, offset + cursor->bytes);
        }
        plan.arena_bytes = extent * shape.n;
        return plan;
    }
};
} // namespace gt
