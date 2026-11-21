#pragma once

#include "../common/platform.h"

#include <string_view>

namespace umzi {
inline constexpr uint32_t plan_format_version = 1;

std::filesystem::path plan_path(const std::filesystem::path& directory, std::string_view kernel,
                                const Digest& key);
std::vector<char> read_plan(const std::filesystem::path& path, uint32_t family, uint32_t schema,
                            const Digest& key, size_t limit);
void write_plan(const std::filesystem::path& path, uint32_t family, uint32_t schema,
                const Digest& key, const std::vector<char>& payload, size_t limit);
}
