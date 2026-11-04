#include "kernels/transpose.cuh"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using murmur::TransposeVersion;

constexpr TransposeVersion kAllVersions[] = {
    TransposeVersion::kNaive,
    TransposeVersion::kTiled,
    TransposeVersion::kTiledPadded,
};

// Transposes a rows x cols matrix whose every element is unique on the GPU and
// checks each output element against the CPU definition out[c][r] = in[r][c].
// Transposing only moves values, so the comparison is exact.
void ExpectMatchesCpu(int rows, int cols, TransposeVersion version) {
    const std::size_t count = static_cast<std::size_t>(rows) * cols;
    std::vector<float> host_input(count);
    for (std::size_t i = 0; i < count; ++i) {
        host_input[i] = static_cast<float>(i);
    }

    murmur::DeviceBuffer<float> input(count);
    murmur::DeviceBuffer<float> output(count);
    input.CopyFromHost(host_input);
    murmur::Transpose(input, output, rows, cols, version);

    std::vector<float> host_output(count);
    output.CopyToHost(host_output);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            ASSERT_EQ(host_output[static_cast<std::size_t>(c) * rows + r],
                      host_input[static_cast<std::size_t>(r) * cols + c])
                << murmur::TransposeVersionName(version) << ", " << rows << " x " << cols
                << ", at row " << r << " col " << c;
        }
    }
}

}  // namespace

TEST(TransposeTest, SingleElement) {
    for (TransposeVersion version : kAllVersions) {
        ExpectMatchesCpu(1, 1, version);
    }
}

// Exactly one 32 x 32 tile.
TEST(TransposeTest, OneFullTile) {
    for (TransposeVersion version : kAllVersions) {
        ExpectMatchesCpu(32, 32, version);
    }
}

// Sizes that aren't multiples of 32: the edge tiles are partly outside the
// matrix and their missing elements must be neither read nor written.
TEST(TransposeTest, PartialEdgeTiles) {
    for (TransposeVersion version : kAllVersions) {
        ExpectMatchesCpu(33, 65, version);
        ExpectMatchesCpu(1000, 777, version);
    }
}

// Very wide and very tall: rows and cols must not be mixed up.
TEST(TransposeTest, NonSquareShapes) {
    for (TransposeVersion version : kAllVersions) {
        ExpectMatchesCpu(8, 4096, version);
        ExpectMatchesCpu(4096, 8, version);
    }
}

TEST(TransposeTest, EmptyIsANoOp) {
    murmur::DeviceBuffer<float> input;
    murmur::DeviceBuffer<float> output;
    for (TransposeVersion version : kAllVersions) {
        EXPECT_NO_THROW(murmur::Transpose(input, output, 0, 0, version));
    }
}

TEST(TransposeTest, SizeMismatchThrows) {
    murmur::DeviceBuffer<float> input(12);
    murmur::DeviceBuffer<float> output(12);
    EXPECT_THROW(murmur::Transpose(input, output, 3, 5, TransposeVersion::kNaive),
                 std::invalid_argument);
}
