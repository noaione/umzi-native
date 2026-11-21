#pragma once

#include <windows.h>
#include <bcrypt.h>

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
using Digest = std::array<unsigned char, 32>;
using ShapeKey = std::array<int, 5>;

class Sha256 {
    BCRYPT_ALG_HANDLE provider = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;

    static void check(NTSTATUS status) {
        if (status < 0) throw std::runtime_error("SHA-256 operation failed");
    }

public:
    Sha256() {
        check(BCryptOpenAlgorithmProvider(&provider, BCRYPT_SHA256_ALGORITHM, nullptr, 0));
        try {
            check(BCryptCreateHash(provider, &hash, nullptr, 0, nullptr, 0, 0));
        } catch (...) {
            BCryptCloseAlgorithmProvider(provider, 0);
            throw;
        }
    }
    Sha256(const Sha256&) = delete;
    Sha256& operator=(const Sha256&) = delete;
    ~Sha256() {
        BCryptDestroyHash(hash);
        BCryptCloseAlgorithmProvider(provider, 0);
    }
    void add(const void* data, size_t size) {
        // The model loader bounds its inputs well below the BCrypt length limit.
        check(BCryptHashData(hash, static_cast<unsigned char*>(const_cast<void*>(data)),
                            static_cast<ULONG>(size), 0));
    }
    template<class T>
    void add(const T& value) { add(&value, sizeof(value)); }

    Digest finish() {
        Digest digest{};
        check(BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0));
        return digest;
    }
};

inline std::array<DWORD, 2> driver_file_version() {
    std::array<wchar_t, 32768> path{};
    auto module = GetModuleHandleW(L"nvcuda.dll");
    auto length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (!module || !length || length >= path.size()) throw std::runtime_error("Could not identify CUDA driver");
    DWORD unused = 0;
    auto size = GetFileVersionInfoSizeW(path.data(), &unused);
    std::vector<char> version(size);
    if (!size || !GetFileVersionInfoW(path.data(), 0, size, version.data()))
        throw std::runtime_error("Could not read CUDA driver version");
    void* value = nullptr;
    UINT bytes = 0;
    if (!VerQueryValueW(version.data(), L"\\", &value, &bytes) || bytes < sizeof(VS_FIXEDFILEINFO))
        throw std::runtime_error("Could not read CUDA driver version");
    const auto* info = static_cast<const VS_FIXEDFILEINFO*>(value);
    return {info->dwFileVersionMS, info->dwFileVersionLS};
}

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
            std::array<wchar_t, MAX_PATH> name{};
            if (!GetTempFileNameW(path.parent_path().c_str(), L"gtp", 0, name.data()))
                throw std::runtime_error("Could not create a temporary cache file");
            temp = name.data();
            std::ofstream file(temp, std::ios::binary | std::ios::trunc);
            file.write(reinterpret_cast<const char*>(digest.data()), digest.size());
            file.write(bytes.data(), bytes.size());
            file.close();
            if (!file || !MoveFileExW(temp.c_str(), path.c_str(),
                                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
                throw std::runtime_error("Could not publish the cache entry");
        } catch (const std::exception& error) {
            std::error_code ignored;
            if (!temp.empty()) std::filesystem::remove(temp, ignored);
            unavailable(error);
        }
    }
};
} // namespace gt
