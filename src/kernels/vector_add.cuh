#pragma once
#include "common/device_buffer.hpp"

namespace murmur {

// c = a + b, element by element. All three must have the same size.
void VectorAdd(const DeviceBuffer<float>& a, const DeviceBuffer<float>& b, DeviceBuffer<float>& c);

}

