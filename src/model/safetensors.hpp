#pragma once

#include "model/mapped_file.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace murmur {

enum class DType {
    kF32,
    kF16,
    kBF16,
    kI32,
    kI64,
};

std::size_t DTypeSize(DType dtype);
const char* DTypeName(DType dtype);

struct TensorInfo {
    std::string name;
    DType dtype;
    std::vector<std::int64_t> shape;
    std::span<const std::byte> bytes;   // points into the mapped file

    std::int64_t ElementCount() const;
};

// A .safetensors file opened through a memory mapping: tensor data is read
// straight from the page cache, never copied up front.
class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::filesystem::path& path);

    const TensorInfo& Get(std::string_view name) const;
    const TensorInfo* Find(std::string_view name) const;
    std::span<const TensorInfo> Tensors() const;

private:
    MappedFile file_;
    std::vector<TensorInfo> tensors_;   // sorted by name
};

}  // namespace murmur
