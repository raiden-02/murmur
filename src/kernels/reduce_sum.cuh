#pragma once

#include "common/device_buffer.hpp"

#include <cstddef>

namespace murmur {

class Reducer {
public:
    explicit Reducer(std::size_t max_count);

    float Sum(const DeviceBuffer<float>& input);
    std::size_t MaxCount() const;

private:
    std::size_t max_count_;
    DeviceBuffer<float> scratch_a_;
    DeviceBuffer<float> scratch_b_;
};

}  // namespace murmur
