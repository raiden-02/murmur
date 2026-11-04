#pragma once

#include "common/device_buffer.hpp"
#include "engine/kernels/weight_view.cuh"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace murmur {

std::string_view WeightFormatName(WeightFormat format);
std::optional<WeightFormat> ParseWeightFormat(std::string_view name);

// A weight matrix in GPU memory, in one of the WeightFormats: the packed
// values plus the scales that format needs. Move-only, like its buffers.
class WeightMatrix {
public:
    WeightMatrix() = default;

    static WeightMatrix FromBf16(DeviceBuffer<std::uint8_t> bf16, int out_features, int in_features,
                                 WeightFormat format, std::span<const float> column_importance = {});

    WeightView View() const;
    WeightFormat Format() const { return format_; }
    int OutFeatures() const { return out_features_; }
    int InFeatures() const { return in_features_; }
    std::size_t ElementCount() const { return static_cast<std::size_t>(out_features_) * in_features_; }
    std::size_t SizeBytes() const;

private:
    WeightFormat format_ = WeightFormat::kBf16;
    int out_features_ = 0;
    int in_features_ = 0;
    DeviceBuffer<std::uint8_t> data_;
    DeviceBuffer<float> row_scales_;              // kFp8
    DeviceBuffer<std::uint32_t> group_ranges_;    // kInt4
};

}  // namespace murmur
