#include "plan_cache_file.h"
#include "binary.h"

#include <cstring>
#include <fstream>

namespace umzi {
namespace {
constexpr std::string_view magic = "UMZIPLAN";
constexpr size_t checksum_offset = 64;
constexpr size_t header_size = 96;

Digest checksum(std::string_view prefix, std::string_view payload) {
    Sha256 hash;
    hash.add(prefix.data(), prefix.size());
    hash.add(payload.data(), payload.size());
    return hash.finish();
}
}

std::filesystem::path plan_path(const std::filesystem::path& directory, std::string_view kernel,
                                const Digest& key) {
    constexpr char digits[] = "0123456789abcdef";
    std::string name(kernel);
    name += '_';
    for (auto byte : key) {
        name += digits[byte >> 4];
        name += digits[byte & 15];
    }
    return directory / (name + ".umplan");
}

std::vector<char> read_plan(const std::filesystem::path& path, uint32_t family, uint32_t schema,
                            const Digest& key, size_t limit) {
    const auto size = std::filesystem::file_size(path);
    if (size < header_size || size > limit) throw std::runtime_error("Invalid .umplan file size");
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::filesystem::filesystem_error("Could not open .umplan", path,
        std::make_error_code(std::errc::io_error));
    std::array<char, header_size> header{};
    std::vector<char> payload(static_cast<size_t>(size) - header_size);
    if (!file.read(header.data(), header.size()) || !file.read(payload.data(), payload.size())
        || file.peek() != EOF)
        throw std::runtime_error("Incomplete .umplan file");
    const std::string_view view(header.data(), header.size());
    BinaryReader reader(view);
    if (reader.take(8) != magic || reader.number<uint32_t>() != plan_format_version
        || reader.number<uint32_t>() != family || reader.number<uint32_t>() != schema
        || reader.number<uint32_t>() || reader.number<uint64_t>() != size
        || std::memcmp(reader.take(key.size()).data(), key.data(), key.size()))
        throw std::runtime_error("Incompatible .umplan header or key");
    auto expected = checksum(view.substr(0, checksum_offset), {payload.data(), payload.size()});
    if (std::memcmp(reader.take(expected.size()).data(), expected.data(), expected.size()))
        throw std::runtime_error(".umplan checksum mismatch");
    return payload;
}

void write_plan(const std::filesystem::path& path, uint32_t family, uint32_t schema,
                const Digest& key, const std::vector<char>& payload, size_t limit) {
    if (payload.size() + header_size > limit) throw std::runtime_error(".umplan file size limit exceeded");
    std::vector<char> header;
    header.reserve(header_size);
    append_bytes(header, magic);
    append_number(header, plan_format_version);
    append_number(header, family);
    append_number(header, schema);
    append_number(header, uint32_t{0});
    append_number(header, uint64_t{header_size + payload.size()});
    append_bytes(header, key);
    append_bytes(header, checksum({header.data(), header.size()}, {payload.data(), payload.size()}));

    auto temporary = temporary_cache_path(path.parent_path());
    try {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(header.data(), header.size());
        file.write(payload.data(), payload.size());
        file.close();
        if (!file) throw std::runtime_error("Could not write .umplan");
        publish_cache(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}
}
