#include "kernels/reduce_sum.cuh"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

murmur::DeviceBuffer<float> Upload(const std::vector<float>& values) {
    murmur::DeviceBuffer<float> buffer(values.size());
    buffer.CopyFromHost(values);
    return buffer;
}

float Sum(const murmur::DeviceBuffer<float>& input) {
    murmur::Reducer reducer(input.Size());
    return reducer.Sum(input);
}

// Sums n ones. Every partial sum is a whole number below 2^24, so float
// addition is exact and the answer must be exactly n.
void ExpectSumOfOnes(std::size_t n) {
    const auto input = Upload(std::vector<float>(n, 1.0f));
    EXPECT_EQ(Sum(input), static_cast<float>(n)) << "n = " << n;
}

}  // namespace

TEST(ReduceSumTest, EmptyIsZero) {
    EXPECT_EQ(Sum(murmur::DeviceBuffer<float>{}), 0.0f);
}

TEST(ReduceSumTest, OneElement) {
    EXPECT_EQ(Sum(Upload({ 42.5f })), 42.5f);
}

// One block of 256 threads, each with a single element.
TEST(ReduceSumTest, ExactlyOneBlockOfThreads) {
    ExpectSumOfOnes(256);
}

// 1000 isn't a multiple of 256: threads with no element must contribute 0.
TEST(ReduceSumTest, PartialBlock) {
    ExpectSumOfOnes(1000);
}

// Needs three passes: 1,048,576 -> 256 -> 1 (each block covers 4,096 values).
TEST(ReduceSumTest, MultiplePasses) {
    ExpectSumOfOnes(std::size_t{ 1 } << 20);
}

// Random values: the GPU adds in a different order than the CPU, so float
// rounding differs slightly. Compare with a double-precision CPU sum. A tree
// of float additions has rounding error on the order of
// (float epsilon ~1.2e-7) x (number of levels) x (sum of |values|); the
// tolerance below allows that with margin.
TEST(ReduceSumTest, RandomValuesMatchCpuWithinTolerance) {
    constexpr std::size_t kCount = 3'000'001;
    std::mt19937 random(7);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> values(kCount);
    double expected = 0.0;
    double sum_of_magnitudes = 0.0;
    for (float& value : values) {
        value = distribution(random);
        expected += value;
        sum_of_magnitudes += std::abs(value);
    }

    EXPECT_NEAR(Sum(Upload(values)), expected, 1e-6 * sum_of_magnitudes);
}

// One reducer, reused for inputs of different sizes up to its maximum: the
// scratch buffers left over from a bigger sum must not leak into a smaller one.
TEST(ReduceSumTest, ReusedReducerHandlesDifferentSizes) {
    murmur::Reducer reducer(std::size_t{ 1 } << 20);
    EXPECT_EQ(reducer.Sum(Upload(std::vector<float>(std::size_t{ 1 } << 20, 1.0f))), 1048576.0f);
    EXPECT_EQ(reducer.Sum(Upload(std::vector<float>(1000, 2.0f))), 2000.0f);
    EXPECT_EQ(reducer.Sum(Upload({ 3.0f })), 3.0f);
}

TEST(ReduceSumTest, InputLargerThanMaxThrows) {
    murmur::Reducer reducer(10);
    EXPECT_THROW(reducer.Sum(Upload(std::vector<float>(11, 1.0f))), std::length_error);
}
