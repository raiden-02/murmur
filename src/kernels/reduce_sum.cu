#include "kernels/reduce_sum.cuh"

#include "common/cuda_check.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace murmur {

namespace {

constexpr unsigned int kThreadsPerBlock = 256;

// How many elements each thread adds up on its own before the block's tree
// ("coins per worker"). More per thread = fewer blocks and fewer trees.
constexpr std::size_t kElementsPerThread = 16;
constexpr std::size_t kElementsPerBlock = kThreadsPerBlock * kElementsPerThread;

// Each block produces one partial sum in block_sums[blockIdx.x].
// Each thread first sums its own elements in a register, taking every
// total_threads-th element so that neighbouring threads read neighbouring
// addresses at the same time; then the block adds its threads' sums with a
// tree in shared memory.
//
// Must be launched with exactly kThreadsPerBlock threads per block.
__global__ void BlockSumKernel(const float* input, float* block_sums, std::size_t n) {
    // The block's scratchpad in fast on-chip shared memory, indexed by the
    // local threadIdx.x (0..255). Each block has its own copy.
    __shared__ float shared_memory[kThreadsPerBlock];

    // Load: each thread starts at its global index and jumps ahead by the
    // total number of threads, adding into its own register. Threads with no
    // elements left skip the loop and contribute 0.
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_threads = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    float thread_sum = 0.0f;
    for (std::size_t j = i; j < n; j += total_threads) {
        thread_sum += input[j];
    }
    shared_memory[threadIdx.x] = thread_sum;
    __syncthreads();

    // Tree: each round, the lower half adds in the upper half. The barrier is
    // outside the `if` so every thread of the block reaches it.
    for (unsigned int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_memory[threadIdx.x] += shared_memory[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // Write: one writer per block, its local thread 0, into the block's slot.
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = shared_memory[0];
    }
}

// Number of blocks (= partial sums produced) for one pass over `count` values.
std::size_t BlocksFor(std::size_t count) {
    return (count + kElementsPerBlock - 1) / kElementsPerBlock;
}

}  // namespace

// Allocates the scratch space once, sized for the largest input this reducer
// will ever see. Sum() then never allocates: GPU allocation is slow (it
// measured ~40% of a reduction's time when done per call).
//   scratch A holds pass 1's partial sums (and passes 3, 5, ...),
//   scratch B holds pass 2's (and 4, 6, ...); each later pass is smaller.
Reducer::Reducer(std::size_t max_count)
    : max_count_(max_count),
      scratch_a_(BlocksFor(max_count)),
      scratch_b_(BlocksFor(BlocksFor(max_count))) {
    if (BlocksFor(max_count) > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("Reducer: too many elements for one launch");
    }
}

// Sums every element of `input` on the GPU and returns the total.
//
// Each pass turns n values into one partial sum per block, where a block
// covers kElementsPerBlock (256 threads x 16) values, so passes repeat until
// a single value is left:
//   64 Mi values -> 16,384 -> 4 -> 1
// The two scratch buffers alternate as each pass's output and the next
// pass's input, so a pass never overwrites what it is reading. Throws
// std::length_error if `input` is larger than the reducer was built for.
// An empty input sums to 0.
float Reducer::Sum(const DeviceBuffer<float>& input) {
    std::size_t n = input.Size();
    if (n > max_count_) {
        throw std::length_error("Reducer::Sum: input larger than max_count");
    }
    if (n == 0) {
        return 0.0f;
    }

    const float* pass_input = input.Data();
    float* pass_output = scratch_a_.Data();
    float* spare = scratch_b_.Data();
    while (true) {
        const std::size_t blocks = BlocksFor(n);
        BlockSumKernel<<<static_cast<unsigned int>(blocks), kThreadsPerBlock>>>(
            pass_input, pass_output, n);
        CheckKernelLaunch();
        if (blocks == 1) {
            break;
        }
        n = blocks;
        pass_input = pass_output;
        std::swap(pass_output, spare);
    }

    float total = 0.0f;
    CudaCheck(cudaMemcpy(&total, pass_output, sizeof(float), cudaMemcpyDeviceToHost));
    return total;
}

std::size_t Reducer::MaxCount() const {
    return max_count_;
}

}  // namespace murmur
