#pragma once

// Compares a forward pass with the Hugging Face reference dumps
// (reference/dump_reference.py), checkpoint by checkpoint.

#include "common/cuda_check.hpp"
#include "engine/transformer.hpp"
#include "model/safetensors.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace murmur::test {

// Largest absolute difference, relative to the largest reference magnitude:
// scale-free, so one threshold works for every checkpoint.
inline double RelativeError(std::span<const float> actual, std::span<const float> expected) {
    double max_diff = 0.0;
    double max_ref = 1e-30;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(static_cast<double>(actual[i]) - expected[i]));
        max_ref = std::max(max_ref, std::abs(static_cast<double>(expected[i])));
    }
    return max_diff / max_ref;
}

inline std::vector<float> ReadFloats(const TensorInfo& tensor) {
    std::vector<float> values(tensor.bytes.size() / sizeof(float));
    std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
    return values;
}

// Compares every checkpoint of a forward pass with the reference dump and
// records the relative error of each.
class ReferenceComparer : public ForwardObserver {
public:
    explicit ReferenceComparer(const SafetensorsFile& reference) : reference_(reference) {}

    void OnCheckpoint(std::string_view name, const float* device_data, int rows, int cols,
                      int row_stride) override {
        const TensorInfo& expected_tensor = reference_.Get(name);
        ASSERT_EQ(expected_tensor.dtype, DType::kF32);
        ASSERT_EQ(expected_tensor.ElementCount(), static_cast<std::int64_t>(rows) * cols) << name;
        std::vector<float> actual(static_cast<std::size_t>(rows) * cols);
        CudaCheck(cudaMemcpy2D(actual.data(), cols * sizeof(float), device_data, row_stride * sizeof(float),
                               cols * sizeof(float), rows, cudaMemcpyDeviceToHost));
        errors_.push_back({std::string(name), RelativeError(actual, ReadFloats(expected_tensor))});
    }

    struct Error {
        std::string name;
        double value;
    };
    const std::vector<Error>& Errors() const { return errors_; }

    // Saves every checkpoint's error in the test report (--gtest_output),
    // keyed "<prefix><checkpoint>", for docs/diagrams/make_reference_diagrams.py.
    void RecordErrors(std::string_view prefix) const {
        for (const Error& error : errors_) {
            ::testing::Test::RecordProperty(std::string(prefix) + error.name, error.value);
        }
    }

private:
    const SafetensorsFile& reference_;
    std::vector<Error> errors_;
};

}  // namespace murmur::test
