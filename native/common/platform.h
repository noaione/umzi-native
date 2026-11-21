#pragma once

#include <cuda.h>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#else
#include <dlfcn.h>
#include <openssl/evp.h>
#include <unistd.h>
#endif

namespace gt {
using Digest = std::array<unsigned char, 32>;

class Sha256 {
#ifdef _WIN32
    BCRYPT_ALG_HANDLE provider = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
#else
    EVP_MD_CTX* hash = nullptr;
#endif

    static void require(bool ok) {
        if (!ok) throw std::runtime_error("SHA-256 operation failed");
    }

public:
    Sha256() {
#ifdef _WIN32
        require(BCryptOpenAlgorithmProvider(&provider, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0);
        if (BCryptCreateHash(provider, &hash, nullptr, 0, nullptr, 0, 0) < 0) {
            BCryptCloseAlgorithmProvider(provider, 0);
            require(false);
        }
#else
        hash = EVP_MD_CTX_new();
        if (!hash || EVP_DigestInit_ex(hash, EVP_sha256(), nullptr) != 1) {
            EVP_MD_CTX_free(hash);
            require(false);
        }
#endif
    }
    Sha256(const Sha256&) = delete;
    Sha256& operator=(const Sha256&) = delete;
    ~Sha256() {
#ifdef _WIN32
        BCryptDestroyHash(hash);
        BCryptCloseAlgorithmProvider(provider, 0);
#else
        EVP_MD_CTX_free(hash);
#endif
    }
    void add(const void* data, size_t size) {
#ifdef _WIN32
        require(BCryptHashData(hash, static_cast<unsigned char*>(const_cast<void*>(data)),
                               static_cast<ULONG>(size), 0) >= 0);
#else
        require(EVP_DigestUpdate(hash, data, size) == 1);
#endif
    }
    template<class T>
    void add(const T& value) { add(&value, sizeof(value)); }

    Digest finish() {
        Digest digest{};
#ifdef _WIN32
        require(BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0) >= 0);
#else
        unsigned size = 0;
        require(EVP_DigestFinal_ex(hash, digest.data(), &size) == 1 && size == digest.size());
#endif
        return digest;
    }
};

inline void hash_driver_version(Sha256& hash) {
    int api = 0;
    if (cuDriverGetVersion(&api) != CUDA_SUCCESS)
        throw std::runtime_error("Could not query the CUDA driver version");
    hash.add(api);
#ifdef _WIN32
    std::array<wchar_t, 32768> path{};
    auto module = GetModuleHandleW(L"nvcuda.dll");
    if (!module) throw std::runtime_error("Could not identify the loaded CUDA driver");
    auto length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (!length || length >= path.size()) throw std::runtime_error("Could not identify the loaded CUDA driver");
    DWORD unused = 0;
    auto size = GetFileVersionInfoSizeW(path.data(), &unused);
    std::vector<char> version(size);
    if (!size || !GetFileVersionInfoW(path.data(), 0, size, version.data()))
        throw std::runtime_error("Could not read the CUDA driver file version");
    void* value = nullptr;
    UINT bytes = 0;
    if (!VerQueryValueW(version.data(), L"\\", &value, &bytes) || bytes < sizeof(VS_FIXEDFILEINFO))
        throw std::runtime_error("Could not read the CUDA driver file version");
    const auto* info = static_cast<const VS_FIXEDFILEINFO*>(value);
    hash.add(std::array<DWORD, 2>{info->dwFileVersionMS, info->dwFileVersionLS});
#else
    Dl_info info{};
    if (!dladdr(reinterpret_cast<const void*>(&cuDriverGetVersion), &info) || !info.dli_fname)
        throw std::runtime_error("Could not identify the loaded CUDA driver");
    auto path = std::filesystem::canonical(info.dli_fname);
    hash.add(path.native().data(), path.native().size());
    hash.add(std::filesystem::file_size(path));
    hash.add(std::filesystem::last_write_time(path).time_since_epoch().count());
#endif
}

inline std::filesystem::path temporary_cache_path(const std::filesystem::path& directory) {
#ifdef _WIN32
    std::array<wchar_t, MAX_PATH> name{};
    if (!GetTempFileNameW(directory.c_str(), L"gtp", 0, name.data()))
        throw std::runtime_error("Could not create a temporary cache file");
    return name.data();
#else
    auto name = (directory / ".gtp-XXXXXX").string();
    int file = mkstemp(name.data());
    if (file < 0) throw std::runtime_error("Could not create a temporary cache file");
    close(file);
    return name;
#endif
}

inline void publish_cache(const std::filesystem::path& temporary,
                          const std::filesystem::path& target) {
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Could not publish the cache entry");
#else
    std::filesystem::rename(temporary, target);
#endif
}
} // namespace gt
