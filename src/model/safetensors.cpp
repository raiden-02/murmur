#include "model/safetensors.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

namespace murmur {

namespace {

using Json = nlohmann::json;

static_assert(std::endian::native == std::endian::little, "safetensors stores the header size little-endian");

// The format's own limit; also stops a corrupt size from requesting a huge parse.
constexpr std::uint64_t kMaxHeaderBytes = 100ull << 20;
constexpr std::size_t kHeaderSizeBytes = sizeof(std::uint64_t);

[[noreturn]] void Fail(const std::filesystem::path& path, const std::string& message) {
    throw std::runtime_error(path.string() + ": invalid safetensors file: " + message);
}

std::optional<DType> ParseDType(std::string_view name) {
    if (name == "F32") return DType::kF32;
    if (name == "F16") return DType::kF16;
    if (name == "BF16") return DType::kBF16;
    if (name == "I32") return DType::kI32;
    if (name == "I64") return DType::kI64;
    return std::nullopt;
}

// Product of the dimensions, or nullopt if a dimension is negative or the
// product overflows.
std::optional<std::uint64_t> CheckedElementCount(const std::vector<std::int64_t>& shape) {
    std::uint64_t count = 1;
    for (const std::int64_t dim : shape) {
        if (dim < 0) {
            return std::nullopt;
        }
        const auto d = static_cast<std::uint64_t>(dim);
        if (d != 0 && count > std::numeric_limits<std::uint64_t>::max() / d) {
            return std::nullopt;
        }
        count *= d;
    }
    return count;
}

}  // namespace

std::size_t DTypeSize(DType dtype) {
    switch (dtype) {
    case DType::kF32: return 4;
    case DType::kF16: return 2;
    case DType::kBF16: return 2;
    case DType::kI32: return 4;
    case DType::kI64: return 8;
    }
    throw std::invalid_argument("DTypeSize: unknown dtype");
}

const char* DTypeName(DType dtype) {
    switch (dtype) {
    case DType::kF32: return "F32";
    case DType::kF16: return "F16";
    case DType::kBF16: return "BF16";
    case DType::kI32: return "I32";
    case DType::kI64: return "I64";
    }
    return "unknown";
}

std::int64_t TensorInfo::ElementCount() const {
    // Validated when the file was opened, so this cannot overflow.
    return static_cast<std::int64_t>(*CheckedElementCount(shape));
}

// Opens and validates a .safetensors file. Layout: an 8-byte little-endian
// header size N, N bytes of JSON describing each tensor (dtype, shape, and
// [begin, end) byte offsets into the data that follows), then the data.
//
// The file is untrusted: before any tensor is exposed, every offset is
// checked to lie inside the file, every byte length to equal shape x dtype
// size (with overflow checks), and tensors are checked not to overlap.
SafetensorsFile::SafetensorsFile(const std::filesystem::path& path) : file_(path) {
    const std::span<const std::byte> bytes = file_.Bytes();
    if (bytes.size() < kHeaderSizeBytes) {
        Fail(path, "shorter than the 8-byte header size");
    }
    std::uint64_t header_size = 0;
    std::memcpy(&header_size, bytes.data(), kHeaderSizeBytes);
    if (header_size > kMaxHeaderBytes || header_size > bytes.size() - kHeaderSizeBytes) {
        Fail(path, "header size " + std::to_string(header_size) + " is out of range");
    }
    const auto* header_begin = reinterpret_cast<const char*>(bytes.data() + kHeaderSizeBytes);
    const std::span<const std::byte> data = bytes.subspan(kHeaderSizeBytes + header_size);

    Json header;
    try {
        header = Json::parse(header_begin, header_begin + header_size);
    } catch (const Json::exception& error) {
        Fail(path, std::string("header is not valid JSON: ") + error.what());
    }
    if (!header.is_object()) {
        Fail(path, "header is not a JSON object");
    }

    for (const auto& [name, entry] : header.items()) {
        if (name == "__metadata__") {
            continue;
        }
        try {
            const auto dtype = ParseDType(entry.at("dtype").get<std::string>());
            if (!dtype) {
                Fail(path, name + ": unsupported dtype");
            }
            auto shape = entry.at("shape").get<std::vector<std::int64_t>>();
            const auto offsets = entry.at("data_offsets").get<std::vector<std::uint64_t>>();
            if (offsets.size() != 2 || offsets[0] > offsets[1] || offsets[1] > data.size()) {
                Fail(path, name + ": data offsets outside the file");
            }
            const std::optional<std::uint64_t> count = CheckedElementCount(shape);
            const std::uint64_t length = offsets[1] - offsets[0];
            if (!count || *count > length / DTypeSize(*dtype) || *count * DTypeSize(*dtype) != length) {
                Fail(path, name + ": byte length doesn't match shape and dtype");
            }
            tensors_.push_back({name, *dtype, std::move(shape), data.subspan(offsets[0], length)});
        } catch (const Json::exception& error) {
            Fail(path, name + ": " + error.what());
        }
    }

    std::ranges::sort(tensors_, {}, [](const TensorInfo& t) { return t.bytes.data(); });
    for (std::size_t i = 1; i < tensors_.size(); ++i) {
        const TensorInfo& prev = tensors_[i - 1];
        if (tensors_[i].bytes.data() < prev.bytes.data() + prev.bytes.size()) {
            Fail(path, tensors_[i].name + " overlaps " + prev.name);
        }
    }
    std::ranges::sort(tensors_, {}, &TensorInfo::name);
}

// The tensor called `name`. Throws std::out_of_range if there is none.
const TensorInfo& SafetensorsFile::Get(std::string_view name) const {
    if (const TensorInfo* tensor = Find(name)) {
        return *tensor;
    }
    throw std::out_of_range("safetensors: no tensor named " + std::string(name));
}

// The tensor called `name`, or nullptr.
const TensorInfo* SafetensorsFile::Find(std::string_view name) const {
    const auto it = std::ranges::lower_bound(tensors_, name, {}, &TensorInfo::name);
    return it != tensors_.end() && it->name == name ? &*it : nullptr;
}

// All tensors, sorted by name.
std::span<const TensorInfo> SafetensorsFile::Tensors() const {
    return tensors_;
}

}  // namespace murmur
