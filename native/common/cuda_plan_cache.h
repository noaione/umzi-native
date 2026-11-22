#pragma once

#include "../include/umzi_native.h"
#include "../runtime/binary.h"
#include "../runtime/plan_cache_file.h"

#include <cublasLt.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace umzi {
using PlanShape = std::array<int, 5>;

struct GemmChoice {
    cublasLtMatmulAlgo_t algorithm{};
    uint64_t workspace = 0;
};

struct ConvChoice {
    uint32_t algorithm = 0;
    uint32_t math = 0;
    uint64_t workspace = 0;
};

template<uint32_t Family, uint32_t Schema>
class CudaPlanCache {
    static constexpr size_t file_limit = 256 * 1024;
    static constexpr uint32_t record_limit = 2000;
    std::filesystem::path path;
    bool dirty = false;

    template<class Choice>
    static auto read_choices(umzi::BinaryReader& reader, uint32_t count) {
        std::map<PlanShape, Choice> choices;
        PlanShape previous{};
        for (uint32_t i = 0; i < count; ++i) {
            PlanShape key{};
            for (auto& dimension : key) {
                auto value = reader.number<uint32_t>();
                if (value > std::numeric_limits<int>::max())
                    throw std::runtime_error("Invalid .umplan dimension");
                dimension = static_cast<int>(value);
            }
            if (i && key <= previous) throw std::runtime_error("Unordered or repeated .umplan key");
            Choice choice{};
            if constexpr (std::is_same_v<Choice, GemmChoice>) {
                choice.workspace = reader.number<uint64_t>();
                auto data = reader.take(sizeof(choice.algorithm));
                std::memcpy(&choice.algorithm, data.data(), data.size());
            } else {
                choice.algorithm = reader.number<uint32_t>();
                choice.math = reader.number<uint32_t>();
                choice.workspace = reader.number<uint64_t>();
            }
            choices.emplace(key, choice);
            previous = key;
        }
        return choices;
    }

    template<class Choice>
    static void write_choices(std::vector<char>& bytes, const std::map<PlanShape, Choice>& choices) {
        for (const auto& [key, choice] : choices) {
            for (auto dimension : key) umzi::append_number(bytes, static_cast<uint32_t>(dimension));
            if constexpr (std::is_same_v<Choice, GemmChoice>) {
                umzi::append_number(bytes, choice.workspace);
                auto* data = reinterpret_cast<const char*>(&choice.algorithm);
                bytes.insert(bytes.end(), data, data + sizeof(choice.algorithm));
            } else {
                umzi::append_number(bytes, choice.algorithm);
                umzi::append_number(bytes, choice.math);
                umzi::append_number(bytes, choice.workspace);
            }
        }
    }

public:
    UmziCacheStatus status = UMZI_CACHE_DISABLED;
    std::string diagnostic;
    std::map<PlanShape, GemmChoice> gemms;
    std::map<PlanShape, ConvChoice> convolutions;

    void unavailable(const std::exception& error) {
        status = UMZI_CACHE_UNAVAILABLE;
        diagnostic = error.what();
    }

    void open(const std::filesystem::path& directory, std::string_view kernel, const Digest& key) {
        if (directory.empty()) return;
        auto base = std::filesystem::absolute(directory);
        std::filesystem::create_directories(base);
        path = umzi::plan_path(base, kernel, key);
        if (!std::filesystem::exists(path)) {
            status = UMZI_CACHE_MISS;
            return;
        }
        try {
            auto bytes = umzi::read_plan(path, Family, Schema, key, file_limit);
            umzi::BinaryReader reader({bytes.data(), bytes.size()});
            auto gemm_count = reader.number<uint32_t>();
            auto conv_count = reader.number<uint32_t>();
            auto algo_size = reader.number<uint32_t>();
            if (gemm_count > record_limit || conv_count > record_limit || algo_size != sizeof(cublasLtMatmulAlgo_t))
                throw std::runtime_error("Invalid CUDA .umplan payload header");
            gemms = read_choices<GemmChoice>(reader, gemm_count);
            convolutions = read_choices<ConvChoice>(reader, conv_count);
            if (reader.size()) throw std::runtime_error("Unexpected .umplan payload data");
            status = UMZI_CACHE_HIT;
        } catch (const std::filesystem::filesystem_error& error) {
            unavailable(error);
        } catch (const std::exception& error) {
            gemms.clear();
            convolutions.clear();
            status = UMZI_CACHE_REBUILT;
            diagnostic = error.what();
        }
    }

    void record_tuning() {
        dirty = true;
        if (status == UMZI_CACHE_HIT) {
            status = UMZI_CACHE_REBUILT;
            diagnostic = "A cached plan was missing or did not pass backend validation";
        }
    }

    void save(const Digest& key) {
        if (!dirty || path.empty() || status == UMZI_CACHE_UNAVAILABLE) return;
        try {
            if (gemms.size() > record_limit || convolutions.size() > record_limit)
                throw std::runtime_error("CUDA .umplan record limit exceeded");
            std::vector<char> payload;
            umzi::append_number(payload, static_cast<uint32_t>(gemms.size()));
            umzi::append_number(payload, static_cast<uint32_t>(convolutions.size()));
            umzi::append_number(payload, static_cast<uint32_t>(sizeof(cublasLtMatmulAlgo_t)));
            write_choices(payload, gemms);
            write_choices(payload, convolutions);
            umzi::write_plan(path, Family, Schema, key, payload, file_limit);
        } catch (const std::exception& error) {
            unavailable(error);
        }
    }
};
}
