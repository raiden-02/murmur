#include "engine/kv_cache.hpp"

#include <stdexcept>
#include <vector>

namespace murmur {

namespace {

int CheckedPositive(int value, const char* message) {
    if (value <= 0) {
        throw std::invalid_argument(message);
    }
    return value;
}

// The first layers' keys are 10-30x larger than the rest's (Qwen2.5's
// layer 0 reaches 318) and their attention is nearly one-hot, so E4M3's 3
// mantissa bits move those scores enough to change which token is attended
// to (measured: mean KL 1.3 against the FP32 cache when every layer is FP8).
// In FP8 mode these layers keep BF16.
constexpr int kBf16LayersInFp8Mode = 2;

// Bytes one token takes in one layer: keys and values, and FP8's scales.
std::size_t BytesPerToken(const ModelConfig& config, KvFormat format) {
    std::size_t bytes = 0;
    switch (format) {
    case KvFormat::kFp32:
        bytes = 4;
        break;
    case KvFormat::kBf16:
        bytes = 2;
        break;
    case KvFormat::kFp8:
        bytes = 1;
        break;
    }
    const std::size_t scales = format == KvFormat::kFp8 ? config.num_kv_heads * sizeof(float) : 0;
    return 2 * (static_cast<std::size_t>(config.KvWidth()) * bytes + scales);
}

std::size_t BytesPerValue(KvFormat format) {
    switch (format) {
    case KvFormat::kFp32:
        return 4;
    case KvFormat::kBf16:
        return 2;
    case KvFormat::kFp8:
        return 1;
    }
    return 0;
}

}  // namespace

std::string_view KvFormatName(KvFormat format) {
    switch (format) {
    case KvFormat::kFp32:
        return "fp32";
    case KvFormat::kBf16:
        return "bf16";
    case KvFormat::kFp8:
        return "fp8";
    }
    return "?";
}

// The format `layer` actually stores in a cache of `format`.
KvFormat LayerKvFormat(KvFormat format, int layer) {
    return format == KvFormat::kFp8 && layer < kBf16LayersInFp8Mode ? KvFormat::kBf16 : format;
}

std::optional<KvFormat> ParseKvFormat(std::string_view name) {
    for (const KvFormat format : {KvFormat::kFp32, KvFormat::kBf16, KvFormat::kFp8}) {
        if (name == KvFormatName(format)) {
            return format;
        }
    }
    return std::nullopt;
}

// Allocates every page up front, in one block: keys then values for each
// layer, back to back (and, for FP8 layers, their scales in a second block).
// Nothing is allocated while generating.
PagedKvCache::PagedKvCache(const ModelConfig& config, int page_count, int page_size, KvFormat format)
    : page_count_(CheckedPositive(page_count, "PagedKvCache: page_count must be positive")),
      page_size_(CheckedPositive(page_size, "PagedKvCache: page_size must be positive")),
      format_(format) {
    const std::size_t rows = static_cast<std::size_t>(page_count_) * page_size_;
    std::size_t bytes = 0;
    std::size_t scales = 0;
    for (int layer = 0; layer < config.num_layers; ++layer) {
        const KvFormat layer_format = LayerKvFormat(format, layer);
        LayerPlace place{.format = layer_format,
                         .offset = bytes,
                         .bytes = rows * config.KvWidth() * BytesPerValue(layer_format),
                         .scale_offset = scales,
                         .scales = layer_format == KvFormat::kFp8 ? rows * config.num_kv_heads : 0};
        bytes += 2 * place.bytes;
        scales += 2 * place.scales;
        layers_.push_back(place);
    }
    storage_ = DeviceBuffer<std::uint8_t>(bytes);
    scales_ = DeviceBuffer<float>(scales);
}

// GPU memory one page takes across all layers, keys and values (and scales).
std::size_t PagedKvCache::BytesPerPage(const ModelConfig& config, int page_size, KvFormat format) {
    std::size_t per_token = 0;
    for (int layer = 0; layer < config.num_layers; ++layer) {
        per_token += BytesPerToken(config, LayerKvFormat(format, layer));
    }
    return per_token * page_size;
}

// Where `layer`'s keys and values (and their scales) start.
KvLayer PagedKvCache::Layer(int layer) {
    if (layer < 0 || layer >= static_cast<int>(layers_.size())) {
        throw std::out_of_range("PagedKvCache: layer out of range");
    }
    const LayerPlace& place = layers_[layer];
    float* scales = place.scales > 0 ? scales_.Data() + place.scale_offset : nullptr;
    return {.format = place.format,
            .keys = storage_.Data() + place.offset,
            .values = storage_.Data() + place.offset + place.bytes,
            .key_scales = scales,
            .value_scales = scales != nullptr ? scales + place.scales : nullptr};
}

}  // namespace murmur
