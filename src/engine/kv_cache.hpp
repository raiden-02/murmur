#pragma once

#include "common/device_buffer.hpp"
#include "engine/kernels/paged_kv.cuh"
#include "model/model_config.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace murmur {

std::string_view KvFormatName(KvFormat format);
std::optional<KvFormat> ParseKvFormat(std::string_view name);
KvFormat LayerKvFormat(KvFormat format, int layer);

// The keys and values of every cached token, for every layer, in GPU
// memory, cut into fixed-size pages that any sequence can own (see
// KvCacheManager for who owns which). Each layer holds page_count *
// page_size rows of kv_heads * head_dim values for keys and the same for
// values, in the layer's format (LayerKvFormat); row page * page_size +
// slot is one token's.
class PagedKvCache {
public:
    PagedKvCache(const ModelConfig& config, int page_count, int page_size, KvFormat format);

    KvLayer Layer(int layer);
    KvFormat Format() const { return format_; }
    int PageCount() const { return page_count_; }
    int PageSize() const { return page_size_; }
    std::size_t SizeBytes() const { return storage_.SizeBytes() + scales_.SizeBytes(); }
    static std::size_t BytesPerPage(const ModelConfig& config, int page_size, KvFormat format);

private:
    struct LayerPlace {
        KvFormat format;
        std::size_t offset;          // bytes into storage_ of the keys; the values follow
        std::size_t bytes;           // keys (or values) of the layer
        std::size_t scale_offset;    // floats into scales_ of the key scales; the value scales follow
        std::size_t scales;          // key (or value) scales of the layer: 0 unless FP8
    };

    int page_count_;
    int page_size_;
    KvFormat format_;
    std::vector<LayerPlace> layers_;
    DeviceBuffer<std::uint8_t> storage_;
    DeviceBuffer<float> scales_;
};

}  // namespace murmur
