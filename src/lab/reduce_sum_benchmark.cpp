#include "lab/lab.hpp"

#include "common/device_buffer.hpp"
#include "common/gpu_timer.hpp"
#include "kernels/reduce_sum.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <print>
#include <vector>

namespace murmur::lab {

namespace {

// 64 Mi floats = 256 MiB, far larger than the 64 MiB L2 cache.
constexpr std::size_t kElementCount = std::size_t{ 1 } << 26;
constexpr int kWarmupRuns = 3;
constexpr int kTimedRuns = 20;

}  // namespace

// Times Reducer::Sum over 64 Mi floats and reports achieved bandwidth. Each
// element is read once; the partial sums of later passes are tiny (<0.4% of
// the input) and not counted. Values are all 1, so the exact answer is known.
int RunReduceSumBenchmark(Args args) {
    if (const std::optional<int> exit_code = ArgParser("murmur-lab reduce").Parse(args)) {
        return *exit_code;
    }
    DeviceBuffer<float> input(kElementCount);
    input.CopyFromHost(std::vector<float>(kElementCount, 1.0f));

    // Scratch space is allocated once, outside the timed runs.
    Reducer reducer(kElementCount);

    float total = 0.0f;
    for (int run = 0; run < kWarmupRuns; ++run) {
        total = reducer.Sum(input);
    }

    // 2^26 ones: every partial sum is a power of two times a whole number of
    // 256s, so float addition stays exact.
    if (total != static_cast<float>(kElementCount)) {
        std::println(stderr, "wrong sum: {} (expected {})", total, kElementCount);
        return EXIT_FAILURE;
    }

    GpuTimer timer;
    std::vector<float> milliseconds;
    for (int run = 0; run < kTimedRuns; ++run) {
        timer.Start();
        reducer.Sum(input);
        milliseconds.push_back(timer.StopMilliseconds());
    }

    std::ranges::sort(milliseconds);
    const float best_ms = milliseconds.front();
    const float median_ms = milliseconds[milliseconds.size() / 2];
    const double bytes_read = static_cast<double>(kElementCount) * sizeof(float);
    const auto gb_per_s = [&](float ms) { return bytes_read / (ms * 1e-3) / 1e9; };

    std::println("reduce sum: {} elements, {:.0f} MB read per run, {} runs", kElementCount,
                 bytes_read / 1e6, kTimedRuns);
    std::println("  best   {:.3f} ms  {:.0f} GB/s", best_ms, gb_per_s(best_ms));
    std::println("  median {:.3f} ms  {:.0f} GB/s", median_ms, gb_per_s(median_ms));
    std::println("  sum checked: OK");
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
