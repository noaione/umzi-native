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

struct JsonPlanChoice {
    uint32_t math = 0;
    uint64_t workspace = 0;
    std::string json;
};

template<uint32_t Family, uint32_t Schema, bool StoreJson = false>
class CudaPlanCache {
    static constexpr size_t file_limit = StoreJson ? 1024 * 1024 : 256 * 1024;
    static constexpr size_t max_json_bytes = 256 * 1024;
    static constexpr uint32_t record_limit = 2000;
    std::filesystem::path path;
    bool dirty = false;

    static PlanShape read_shape_key(umzi::BinaryReader& reader, uint32_t index, const PlanShape& previous) {
        PlanShape key{};
        for (auto& dimension : key) {
            auto value = reader.number<uint32_t>();
            if (value > std::numeric_limits<int>::max()) throw std::runtime_error("Invalid .umplan dimension");
            dimension = static_cast<int>(value);
        }
        if (index && key <= previous) throw std::runtime_error("Unordered or repeated .umplan key");
        return key;
    }

    static void write_shape_key(std::vector<char>& bytes, const PlanShape& key) {
        for (auto dimension : key) umzi::append_number(bytes, static_cast<uint32_t>(dimension));
    }

    template<class Choice>
    static auto read_choices(umzi::BinaryReader& reader, uint32_t count) {
        std::map<PlanShape, Choice> choices;
        PlanShape previous{};
        for (uint32_t i = 0; i < count; ++i) {
            PlanShape key = read_shape_key(reader, i, previous);
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

    static std::map<PlanShape, JsonPlanChoice> read_json_plans(umzi::BinaryReader& reader, uint32_t count) {
        std::map<PlanShape, JsonPlanChoice> plans;
        PlanShape previous{};
        for (uint32_t i = 0; i < count; ++i) {
            PlanShape key = read_shape_key(reader, i, previous);
            JsonPlanChoice choice{};
            choice.math = reader.number<uint32_t>();
            choice.workspace = reader.number<uint64_t>();
            const auto length = reader.number<uint32_t>();
            if (!length || length > max_json_bytes) throw std::runtime_error("Invalid .umplan JSON size");
            auto text = reader.take(length);
            choice.json.assign(text.data(), text.size());
            plans.emplace(key, std::move(choice));
            previous = key;
        }
        return plans;
    }

    template<class Choice>
    static void write_choices(std::vector<char>& bytes, const std::map<PlanShape, Choice>& choices) {
        for (const auto& [key, choice] : choices) {
            write_shape_key(bytes, key);
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

    static void write_json_plans(std::vector<char>& bytes, const std::map<PlanShape, JsonPlanChoice>& plans) {
        for (const auto& [key, choice] : plans) {
            if (choice.json.empty() || choice.json.size() > max_json_bytes)
                throw std::runtime_error("Invalid .umplan JSON size");
            write_shape_key(bytes, key);
            umzi::append_number(bytes, choice.math);
            umzi::append_number(bytes, choice.workspace);
            umzi::append_number(bytes, static_cast<uint32_t>(choice.json.size()));
            bytes.insert(bytes.end(), choice.json.begin(), choice.json.end());
        }
    }

public:
    UmziCacheStatus status = UMZI_CACHE_DISABLED;
    std::string diagnostic;
    std::map<PlanShape, GemmChoice> gemms;
    std::map<PlanShape, ConvChoice> convolutions;
    std::map<PlanShape, JsonPlanChoice> json_plans;

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
            const uint32_t json_count = StoreJson ? reader.number<uint32_t>() : 0;
            if (json_count > record_limit) throw std::runtime_error("Invalid CUDA .umplan JSON record count");
            gemms = read_choices<GemmChoice>(reader, gemm_count);
            convolutions = read_choices<ConvChoice>(reader, conv_count);
            if constexpr (StoreJson) json_plans = read_json_plans(reader, json_count);
            if (reader.size()) throw std::runtime_error("Unexpected .umplan payload data");
            status = UMZI_CACHE_HIT;
        } catch (const std::filesystem::filesystem_error& error) {
            unavailable(error);
        } catch (const std::exception& error) {
            gemms.clear();
            convolutions.clear();
            json_plans.clear();
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
            if (gemms.size() > record_limit || convolutions.size() > record_limit || json_plans.size() > record_limit)
                throw std::runtime_error("CUDA .umplan record limit exceeded");
            std::vector<char> payload;
            umzi::append_number(payload, static_cast<uint32_t>(gemms.size()));
            umzi::append_number(payload, static_cast<uint32_t>(convolutions.size()));
            umzi::append_number(payload, static_cast<uint32_t>(sizeof(cublasLtMatmulAlgo_t)));
            if constexpr (StoreJson) umzi::append_number(payload, static_cast<uint32_t>(json_plans.size()));
            write_choices(payload, gemms);
            write_choices(payload, convolutions);
            if constexpr (StoreJson) write_json_plans(payload, json_plans);
            umzi::write_plan(path, Family, Schema, key, payload, file_limit);
        } catch (const std::exception& error) {
            unavailable(error);
        }
    }
};
}
