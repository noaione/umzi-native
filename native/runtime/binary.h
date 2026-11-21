#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace umzi {
class BinaryReader {
    std::string_view remaining;
public:
    explicit BinaryReader(std::string_view bytes) : remaining(bytes) {}
    std::string_view take(size_t size) {
        if (size > remaining.size()) throw std::runtime_error("Truncated binary data");
        auto value = remaining.substr(0, size);
        remaining.remove_prefix(size);
        return value;
    }
    template<class UInt>
    UInt number() {
        UInt value = 0;
        auto bytes = take(sizeof(UInt));
        for (size_t i = 0; i < bytes.size(); ++i)
            value |= static_cast<UInt>(static_cast<unsigned char>(bytes[i])) << (8 * i);
        return value;
    }
    size_t size() const { return remaining.size(); }
};

template<class UInt>
inline void append_number(std::vector<char>& out, UInt value) {
    for (size_t i = 0; i < sizeof(value); ++i)
        out.push_back(static_cast<char>((value >> (8 * i)) & 255));
}

template<class Buffer>
inline void append_bytes(std::vector<char>& out, const Buffer& value) {
    const auto* bytes = reinterpret_cast<const char*>(value.data());
    out.insert(out.end(), bytes, bytes + value.size() * sizeof(*value.data()));
}
}
