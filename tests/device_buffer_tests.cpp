#include "common/device_buffer.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(DeviceBufferTest, RoundTrip) {
    std::vector<int> host_data = {1, 2, 3, 4, 5};
    murmur::DeviceBuffer<int> device_buffer(host_data.size());
    device_buffer.CopyFromHost(host_data);
    std::vector<int> result(host_data.size());
    device_buffer.CopyToHost(result);
    EXPECT_EQ(result, host_data);
}

TEST(DeviceBufferTest, Move) {
    std::vector<int> host_data = {1, 2, 3, 4, 5};
    murmur::DeviceBuffer<int> device_buffer(host_data.size());
    device_buffer.CopyFromHost(host_data);
    murmur::DeviceBuffer<int> moved_buffer(std::move(device_buffer));

    // The source no longer owns anything.
    EXPECT_EQ(device_buffer.Data(), nullptr);
    EXPECT_EQ(device_buffer.Size(), 0u);

    // The destination owns the original memory and its contents.
    EXPECT_EQ(moved_buffer.Size(), 5u);
    std::vector<int> result(host_data.size());
    moved_buffer.CopyToHost(result);
    EXPECT_EQ(result, host_data);
}

TEST(DeviceBufferTest, SizeMismatch) {
    std::vector<int> host_data = {1, 2, 3, 4, 5};
    murmur::DeviceBuffer<int> device_buffer(host_data.size());
    device_buffer.CopyFromHost(host_data);
    std::vector<int> result(6);
    EXPECT_THROW(device_buffer.CopyToHost(result), std::invalid_argument);
}