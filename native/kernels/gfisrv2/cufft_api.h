#pragma once

#include "../../common/platform.h"

namespace gfi {
// Only the cuFFT 12 entry points used by this family. Resolve them at session
// open so loading the Umzi library has no link-time cuFFT dependency.
#ifdef _WIN32
#define GFI_FFT_CALL WINAPI
#else
#define GFI_FFT_CALL
#endif

struct CufftApi {
#ifdef _WIN32
    HMODULE library = nullptr;
#else
    void* library = nullptr;
#endif
    using Simple = int (GFI_FFT_CALL*)(int);
    using Create = int (GFI_FFT_CALL*)(int*);
    using Flag = int (GFI_FFT_CALL*)(int, int);
    using Plan = int (GFI_FFT_CALL*)(int, int, int*, int*, int, int, int*, int, int, int, int, size_t*);
    using Pointer = int (GFI_FFT_CALL*)(int, void*);
    using Exec = int (GFI_FFT_CALL*)(int, void*, void*);

    Create create = nullptr, version = nullptr;
    Simple destroy = nullptr;
    Flag auto_allocation = nullptr;
    Plan make_plan = nullptr;
    Pointer work_area = nullptr, set_stream = nullptr;
    Exec r2c = nullptr, c2r = nullptr;

    CufftApi() {
#ifdef _WIN32
        library = LoadLibraryW(L"cufft64_12.dll");
#else
        library = dlopen("libcufft.so.12", RTLD_NOW | RTLD_LOCAL);
#endif
        if (!library)
            throw std::runtime_error("GFISRv2 requires cuFFT 12 (cufft64_12.dll or libcufft.so.12) on the library search path");
        try {
            create = symbol<Create>("cufftCreate");
            version = symbol<Create>("cufftGetVersion");
            destroy = symbol<Simple>("cufftDestroy");
            auto_allocation = symbol<Flag>("cufftSetAutoAllocation");
            make_plan = symbol<Plan>("cufftMakePlanMany");
            work_area = symbol<Pointer>("cufftSetWorkArea");
            set_stream = symbol<Pointer>("cufftSetStream");
            r2c = symbol<Exec>("cufftExecR2C");
            c2r = symbol<Exec>("cufftExecC2R");
        } catch (...) {
            close();
            throw;
        }
    }
    CufftApi(const CufftApi&) = delete;
    CufftApi& operator=(const CufftApi&) = delete;
    ~CufftApi() { close(); }

private:
    template<class T>
    T symbol(const char* name) const {
#ifdef _WIN32
        auto entry = GetProcAddress(library, name);
#else
        auto entry = dlsym(library, name);
#endif
        if (!entry) throw std::runtime_error(std::string("GFISRv2 could not resolve cuFFT entry point ") + name);
        return reinterpret_cast<T>(entry);
    }

    void close() {
#ifdef _WIN32
        FreeLibrary(library);
#else
        dlclose(library);
#endif
    }
};
#undef GFI_FFT_CALL
} // namespace gfi
