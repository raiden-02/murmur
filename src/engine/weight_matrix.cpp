#include "engine/weight_matrix.hpp"

#include "common/cuda_check.hpp"
#include "engine/kernels/quantize.cuh"

#include <stdexcept>
#include <utility>

namespace murmur {

std::string_view WeightFormatName(WeightFormat format) {
    switch (format) {
    case WeightFormat::kBf16:
        return "bf16";
    case WeightFormat::kFp8:
        return "fp8";
    case WeightFormat::kInt4:
        return "int4";
    }
    return "?";
}

std::optional<WeightFormat> ParseWeightFormat(std::string_view name) {
    for (const WeightFormat format : {WeightFormat::kBf16, WeightFormat::kFp8, WeightFormat::kInt4}) {
        if (name == WeightFormatName(format)) {
            return format;
        }
    }
    return std::nullopt;
}

// Builds a matrix in `format` from out_features x in_features BF16 weights
// already in GPU memory (raw bytes, row-major). BF16 keeps the buffer as is;
// FP8 and INT4 quantize it on the GPU and free it. INT4 needs in_features
// to be a multiple of its group size, and uses column_importance (one value
// per input channel, see ChannelImportance) when given. Blocks until the
// GPU is done.
WeightMatrix WeightMatrix::FromBf16(DeviceBuffer<std::uint8_t> bf16, int out_features, int in_features,
                                    WeightFormat format, std::span<const float> column_importance) {
    const std::size_t elements = static_cast<std::size_t>(out_features) * in_features;
    if (out_features <= 0 || in_features <= 0 || bf16.Size() != elements * sizeof(Bf16Bits)) {
        throw std::invalid_argument("WeightMatrix: the BF16 buffer doesn't match the shape");
    }
    if (!column_importance.empty() && column_importance.size() != static_cast<std::size_t>(in_features)) {
        throw std::invalid_argument("WeightMatrix: one importance value per input channel");
    }
    WeightMatrix m;
    m.format_ = format;
    m.out_features_ = out_features;
    m.in_features_ = in_features;
    const auto* source = reinterpret_cast<const Bf16Bits*>(bf16.Data());
    switch (format) {
    case WeightFormat::kBf16:
        m.data_ = std::move(bf16);
        return m;
    case WeightFormat::kFp8:
        m.data_ = DeviceBuffer<std::uint8_t>(elements);
        m.row_scales_ = DeviceBuffer<float>(out_features);
        QuantizeToFp8(source, out_features, in_features, m.data_.Data(), m.row_scales_.Data(), /*stream=*/nullptr);
        break;
    case WeightFormat::kInt4: {
        if (in_features % kInt4GroupSize != 0) {
            throw std::invalid_argument("WeightMatrix: INT4 needs in_features to be a multiple of the group size");
        }
        m.data_ = DeviceBuffer<std::uint8_t>(elements / 2);
        m.group_ranges_ = DeviceBuffer<std::uint32_t>(elements / kInt4GroupSize);
        DeviceBuffer<float> importance(column_importance.size());
        importance.CopyFromHost(column_importance);
        QuantizeToInt4(source, out_features, in_features, importance.Size() > 0 ? importance.Data() : nullptr,
                       m.data_.Data(), m.group_ranges_.Data(), /*stream=*/nullptr);
        CudaCheck(cudaDeviceSynchronize());   // the BF16 source and the importance are freed on return
        return m;
    }
    }
    CudaCheck(cudaDeviceSynchronize());   // the BF16 source is freed on return
    return m;
}

WeightView WeightMatrix::View() const {
    return {.format = format_,
            .out_features = out_features_,
            .in_features = in_features_,
            .data = data_.Data(),
            .row_scales = row_scales_.Data(),
            .group_ranges = group_ranges_.Data()};
}

// GPU memory the matrix takes, scales included: also the bytes a decode
// step reads from it.
std::size_t WeightMatrix::SizeBytes() const {
    return data_.SizeBytes() + row_scales_.SizeBytes() + group_ranges_.SizeBytes();
}

}  // namespace murmur
