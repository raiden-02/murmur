#include "kernels/transpose.cuh"

#include "common/cuda_check.hpp"

#include <cstddef>
#include <stdexcept>

namespace murmur {

namespace {

// Every block handles one 32 x 32 tile of the matrix. A block has 32 x 8
// threads, so each thread handles 4 elements of its column of the tile
// (rows threadIdx.y, +8, +16, +24).
constexpr int kTile = 32;
constexpr int kBlockRows = 8;

// Naive: each thread reads input[row][col] and writes output[col][row]
// straight from/to global memory.
//   reads:  a warp's 32 threads read 32 neighbouring floats of one row  -> fast
//   writes: they write 32 floats that are `rows` apart (a column)       -> slow
__global__ void TransposeNaiveKernel(const float* input, float* output, int rows, int cols) {
    const int col = blockIdx.x * kTile + threadIdx.x;
    const int first_row = blockIdx.y * kTile + threadIdx.y;
    for (int offset = 0; offset < kTile; offset += kBlockRows) {
        const int row = first_row + offset;
        if (col < cols && row < rows) {
            output[static_cast<std::size_t>(col) * rows + row] =
                input[static_cast<std::size_t>(row) * cols + col];
        }
    }
}

// Tiled: copy the tile into shared memory with row-wise (fast) reads, then
// write it out row-wise too, reading the shared tile column-wise instead.
// Shared memory is on-chip, so the column-wise access is cheap there.
//
// kPadding = 1 adds one unused float per tile row. Without it, a column of
// the 32 x 32 tile falls in a single shared-memory bank and the 32 reads
// queue up one after another (a bank conflict); the extra float shifts each
// row by one bank so a column spreads across all 32 banks.
template <int kPadding>
__global__ void TransposeTiledKernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[kTile][kTile + kPadding];

    // Phase 1: read the input tile, one row of the tile per warp step.
    int col = blockIdx.x * kTile + threadIdx.x;
    int row = blockIdx.y * kTile + threadIdx.y;
    for (int offset = 0; offset < kTile; offset += kBlockRows) {
        if (col < cols && row + offset < rows) {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[static_cast<std::size_t>(row + offset) * cols + col];
        }
    }

    // Every element of the tile must be loaded before anyone reads it.
    __syncthreads();

    // Phase 2: write the transposed tile. This block's tile lands at the
    // mirrored position: block (x, y) of the input becomes block (y, x) of
    // the output, so x and y swap roles here.
    col = blockIdx.y * kTile + threadIdx.x;   // column in the output = row in the input
    row = blockIdx.x * kTile + threadIdx.y;   // row in the output = column in the input
    for (int offset = 0; offset < kTile; offset += kBlockRows) {
        if (col < rows && row + offset < cols) {
            output[static_cast<std::size_t>(row + offset) * rows + col] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

}  // namespace

const char* TransposeVersionName(TransposeVersion version) {
    switch (version) {
    case TransposeVersion::kNaive: return "naive";
    case TransposeVersion::kTiled: return "tiled";
    case TransposeVersion::kTiledPadded: return "tiled + padding";
    }
    return "unknown";
}

// Writes the transpose of `input` (rows x cols, row-major) into `output`
// (cols x rows, row-major): output[c][r] = input[r][c].
// Launches a 2D grid: blockIdx.x walks tile columns, blockIdx.y tile rows.
void Transpose(const DeviceBuffer<float>& input, DeviceBuffer<float>& output, int rows, int cols,
               TransposeVersion version) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("Transpose: negative dimensions");
    }
    const std::size_t count = static_cast<std::size_t>(rows) * cols;
    if (input.Size() != count || output.Size() != count) {
        throw std::invalid_argument("Transpose: buffer sizes don't match rows x cols");
    }
    if (count == 0) {
        return;
    }

    const dim3 block(kTile, kBlockRows);
    const dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
    if (grid.y > 65535) {
        throw std::length_error("Transpose: too many rows for one launch");
    }

    switch (version) {
    case TransposeVersion::kNaive:
        TransposeNaiveKernel<<<grid, block>>>(input.Data(), output.Data(), rows, cols);
        break;
    case TransposeVersion::kTiled:
        TransposeTiledKernel<0><<<grid, block>>>(input.Data(), output.Data(), rows, cols);
        break;
    case TransposeVersion::kTiledPadded:
        TransposeTiledKernel<1><<<grid, block>>>(input.Data(), output.Data(), rows, cols);
        break;
    }
    CheckKernelLaunch();
}

}  // namespace murmur
