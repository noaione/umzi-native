#pragma once

#include <array>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#else
#include <openssl/evp.h>
#endif

namespace umzi {
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

} // namespace umzi
