#pragma once

#include "common/device_buffer.hpp"

namespace murmur {

enum class TransposeVersion {
    kNaive,
    kTiled,
    kTiledPadded,
};

const char* TransposeVersionName(TransposeVersion version);

void Transpose(const DeviceBuffer<float>& input, DeviceBuffer<float>& output, int rows, int cols,
               TransposeVersion version);

}  // namespace murmur
