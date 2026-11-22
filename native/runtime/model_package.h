#pragma once

#include "../common/platform.h"

#include <map>
#include <string_view>

namespace umzi {
inline constexpr uint32_t gaterv3_id = 1;
inline constexpr uint32_t gfisrv2_id = 2;
inline constexpr uint32_t gaterv2_id = 3;
inline constexpr uint32_t moesr_id = 4;

class ModelPackage {
    std::vector<char> bytes;
    std::map<uint32_t, std::string_view> sections;
public:
    uint32_t family = 0;
    uint32_t schema = 0;
    Digest digest{};

    explicit ModelPackage(const std::filesystem::path& path);
    ModelPackage(const ModelPackage&) = delete;
    ModelPackage& operator=(const ModelPackage&) = delete;

    std::string_view section(uint32_t id) const;
};
}
