#pragma once

#include <cuda.h>
#include "sha256.h"
#include <array>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace umzi {
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
} // namespace umzi
