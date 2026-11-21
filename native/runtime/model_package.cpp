#include "model_package.h"
#include "binary.h"

#include <cstring>
#include <fstream>

namespace umzi {
ModelPackage::ModelPackage(const std::filesystem::path& path) {
    constexpr size_t header_size = 64;
    constexpr size_t entry_size = 24;
    const auto size = std::filesystem::file_size(path);
    if (size < header_size || size > (1ull << 30))
        throw std::runtime_error("Invalid Umzi model package size");
    bytes.resize(static_cast<size_t>(size));
    std::ifstream file(path, std::ios::binary);
    if (!file.read(bytes.data(), bytes.size()) || file.peek() != EOF)
        throw std::runtime_error("Could not read the complete Umzi model package");
    std::string_view view(bytes.data(), bytes.size());
    BinaryReader reader(view);
    if (reader.take(8) != "UMZINPKG") throw std::runtime_error("Invalid Umzi model package magic");
    if (reader.number<uint32_t>() != 1) throw std::runtime_error("Unsupported Umzi model package version");
    family = reader.number<uint32_t>();
    schema = reader.number<uint32_t>();
    auto count = reader.number<uint32_t>();
    if (!family || !schema || !count || count > 32 || reader.number<uint64_t>() != size)
        throw std::runtime_error("Invalid Umzi model package header");
    auto checksum = reader.take(digest.size());
    Sha256 hash;
    hash.add(view.data(), 32);
    hash.add(view.data() + header_size, view.size() - header_size);
    digest = hash.finish();
    if (std::memcmp(checksum.data(), digest.data(), digest.size()))
        throw std::runtime_error("Umzi model package checksum mismatch");

    uint64_t position = header_size + count * entry_size;
    if (position > size) throw std::runtime_error("Truncated Umzi model section table");
    uint32_t previous = 0;
    for (uint32_t i = 0; i < count; ++i) {
        auto id = reader.number<uint32_t>();
        auto reserved = reader.number<uint32_t>();
        auto offset = reader.number<uint64_t>();
        auto length = reader.number<uint64_t>();
        if (id <= previous || reserved || offset != position || !length || length > size - position)
            throw std::runtime_error("Invalid Umzi model section table");
        sections.emplace(id, view.substr(static_cast<size_t>(offset), static_cast<size_t>(length)));
        position += length;
        previous = id;
    }
    if (position != size) throw std::runtime_error("Unexpected trailing Umzi model data");
}

std::string_view ModelPackage::section(uint32_t id) const {
    auto found = sections.find(id);
    return found == sections.end() ? std::string_view{} : found->second;
}
}
