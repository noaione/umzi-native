#pragma once

#include "../common/platform.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace gt {
using ShapeKey = std::array<int, 5>;

struct GemmChoice {
    cublasLtMatmulAlgo_t algorithm{};
    uint64_t workspace = 0;
};

struct ConvChoice {
    uint32_t algorithm = 0;
    uint32_t math = 0;
    uint64_t workspace = 0;
};

class PlanCache {
    static constexpr uint32_t format = 0x31545047; // GPT1
    static constexpr size_t file_limit = 256 * 1024;
    std::filesystem::path path;
    bool dirty = false;

    static std::string hex(const Digest& key) {
        constexpr char digits[] = "0123456789abcdef";
        std::string name;
        name.reserve(key.size() * 2);
        for (auto byte : key) {
            name += digits[byte >> 4];
            name += digits[byte & 15];
        }
        return name;
    }

    template<class T>
    static T consume(const std::vector<char>& bytes, size_t& position) {
        if (position > bytes.size() || sizeof(T) > bytes.size() - position)
            throw std::runtime_error("Truncated cache entry");
        T value{};
        std::memcpy(&value, bytes.data() + position, sizeof(T));
        position += sizeof(T);
        return value;
    }

    template<class T>
    static void append(std::vector<char>& bytes, const T& value) {
        const auto* data = reinterpret_cast<const char*>(&value);
        bytes.insert(bytes.end(), data, data + sizeof(T));
    }

    template<class Choice>
    static auto read_choices(const std::vector<char>& bytes, size_t& position) {
        std::map<ShapeKey, Choice> choices;
        auto count = consume<uint32_t>(bytes, position);
        if (count > 2000) throw std::runtime_error("Invalid cache entry count");
        for (uint32_t i = 0; i < count; ++i) {
            auto key = consume<ShapeKey>(bytes, position);
            auto choice = consume<Choice>(bytes, position);
            if (!choices.emplace(key, choice).second) throw std::runtime_error("Repeated cache plan");
        }
        return choices;
    }

    template<class Choice>
    static void write_choices(std::vector<char>& bytes, const std::map<ShapeKey, Choice>& choices) {
        append(bytes, static_cast<uint32_t>(choices.size()));
        for (const auto& [key, choice] : choices) {
            append(bytes, key);
            append(bytes, choice);
        }
    }

public:
    GtCacheStatus status = GT_CACHE_DISABLED;
    std::string diagnostic;
    std::map<ShapeKey, GemmChoice> gemms;
    std::map<ShapeKey, ConvChoice> convolutions;

    void unavailable(const std::exception& error) {
        status = GT_CACHE_UNAVAILABLE;
        diagnostic = error.what();
    }

    void open(const std::filesystem::path& directory, const Digest& key) {
        if (directory.empty()) return;
        auto base = std::filesystem::absolute(directory);
        std::filesystem::create_directories(base);
        path = base / (hex(key) + ".gtplan");
        if (!std::filesystem::exists(path)) {
            status = GT_CACHE_MISS;
            return;
        }
        try {
            const auto length = std::filesystem::file_size(path);
            if (length <= sizeof(Digest) || length > file_limit)
                throw std::runtime_error("Invalid cache file size");
            std::ifstream file(path, std::ios::binary);
            std::vector<char> bytes(static_cast<size_t>(length) - sizeof(Digest));
            Digest expected{};
            if (!file.read(reinterpret_cast<char*>(expected.data()), expected.size())
                || !file.read(bytes.data(), bytes.size()))
                throw std::runtime_error("Could not read cache entry");
            Sha256 hash;
            hash.add(bytes.data(), bytes.size());
            if (hash.finish() != expected) throw std::runtime_error("Cache checksum mismatch");
            size_t position = 0;
            if (consume<uint32_t>(bytes, position) != format || consume<Digest>(bytes, position) != key)
                throw std::runtime_error("Incompatible cache entry");
            gemms = read_choices<GemmChoice>(bytes, position);
            convolutions = read_choices<ConvChoice>(bytes, position);
            if (position != bytes.size()) throw std::runtime_error("Unexpected cache payload");
            status = GT_CACHE_HIT;
        } catch (const std::exception& error) {
            gemms.clear();
            convolutions.clear();
            status = GT_CACHE_REBUILT;
            diagnostic = error.what();
        }
    }

    void record_tuning() {
        dirty = true;
        if (status == GT_CACHE_HIT) {
            status = GT_CACHE_REBUILT;
            diagnostic = "A cached plan was missing or did not pass backend validation";
        }
    }

    void save(const Digest& key) {
        if (!dirty || path.empty() || status == GT_CACHE_UNAVAILABLE) return;
        std::filesystem::path temp;
        try {
            std::vector<char> bytes;
            append(bytes, format);
            append(bytes, key);
            write_choices(bytes, gemms);
            write_choices(bytes, convolutions);
            Sha256 hash;
            hash.add(bytes.data(), bytes.size());
            auto digest = hash.finish();
            temp = temporary_cache_path(path.parent_path());
            std::ofstream file(temp, std::ios::binary | std::ios::trunc);
            file.write(reinterpret_cast<const char*>(digest.data()), digest.size());
            file.write(bytes.data(), bytes.size());
            file.close();
            if (!file) throw std::runtime_error("Could not write the cache entry");
            publish_cache(temp, path);
        } catch (const std::exception& error) {
            std::error_code ignored;
            if (!temp.empty()) std::filesystem::remove(temp, ignored);
            unavailable(error);
        }
    }
};
} // namespace gt
