#pragma once

#include <cstddef>
#include <filesystem>
#include <span>

namespace murmur {

// A read-only view of a whole file through the OS page cache (memory
// mapping). Move-only: exactly one object owns the mapping.
class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    std::span<const std::byte> Bytes() const;

private:
    void Release() noexcept;

    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace murmur
