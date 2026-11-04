#include "kernels/vector_add.cuh"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

// Runs VectorAdd on n elements and compares every result with the CPU.
void ExpectMatchesCpu(std::size_t n) {
    std::vector<float> host_a(n);
    std::vector<float> host_b(n);
    for (std::size_t i = 0; i < n; ++i) {
        host_a[i] = static_cast<float>(i);
        host_b[i] = 2.0f * static_cast<float>(i) + 1.0f;
    }

    murmur::DeviceBuffer<float> a(n);
    murmur::DeviceBuffer<float> b(n);
    murmur::DeviceBuffer<float> c(n);
    a.CopyFromHost(host_a);
    b.CopyFromHost(host_b);

    murmur::VectorAdd(a, b, c);

    std::vector<float> host_c(n);
    c.CopyToHost(host_c);
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(host_c[i], host_a[i] + host_b[i]) << "at index " << i;
    }
}

}  // namespace

TEST(VectorAddTest, OneElement) {
    ExpectMatchesCpu(1);
}

// 1000 is not a multiple of 256 threads per block, so the last block has
// threads with no element: the bounds check must stop them.
TEST(VectorAddTest, PartialLastBlock) {
    ExpectMatchesCpu(1000);
}

TEST(VectorAddTest, ManyBlocks) {
    ExpectMatchesCpu(std::size_t{ 1 } << 20);
}

TEST(VectorAddTest, EmptyIsANoOp) {
    murmur::DeviceBuffer<float> a;
    murmur::DeviceBuffer<float> b;
    murmur::DeviceBuffer<float> c;
    EXPECT_NO_THROW(murmur::VectorAdd(a, b, c));
}

TEST(VectorAddTest, SizeMismatchThrows) {
    murmur::DeviceBuffer<float> a(4);
    murmur::DeviceBuffer<float> b(4);
    murmur::DeviceBuffer<float> c(5);
    EXPECT_THROW(murmur::VectorAdd(a, b, c), std::invalid_argument);
}
