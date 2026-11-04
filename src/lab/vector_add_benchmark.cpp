#include "lab/lab.hpp"

#include "common/device_buffer.hpp"
#include "common/gpu_timer.hpp"
#include "kernels/vector_add.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <print>
#include <vector>

namespace murmur::lab {

namespace {

// 64 Mi floats = 256 MiB per array, far larger than the 64 MiB L2 cache, so
// every byte really comes from (and goes to) global memory.
constexpr std::size_t kElementCount = std::size_t{ 1 } << 26;
constexpr int kWarmupRuns = 3;
constexpr int kTimedRuns = 20;

}  // namespace

// Times VectorAdd on large arrays and reports achieved memory bandwidth:
// each element reads a and b and writes c, so 3 * 4 bytes move per element.
// Checks the result against a CPU computation before reporting.
int RunVectorAddBenchmark(Args args) {
    if (const std::optional<int> exit_code = ArgParser("murmur-lab vector-add").Parse(args)) {
        return *exit_code;
    }
    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>(i % 1000);
        host_b[i] = 0.5f * static_cast<float>(i % 7);
    }

    DeviceBuffer<float> a(kElementCount);
    DeviceBuffer<float> b(kElementCount);
    DeviceBuffer<float> c(kElementCount);
    a.CopyFromHost(host_a);
    b.CopyFromHost(host_b);

    // Warm-up: the first launches pay one-time costs (context, module load).
    for (int run = 0; run < kWarmupRuns; ++run) {
        VectorAdd(a, b, c);
    }

    GpuTimer timer;
    std::vector<float> milliseconds;
    for (int run = 0; run < kTimedRuns; ++run) {
        timer.Start();
        VectorAdd(a, b, c);
        milliseconds.push_back(timer.StopMilliseconds());
    }

    std::vector<float> host_c(kElementCount);
    c.CopyToHost(host_c);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        if (host_c[i] != host_a[i] + host_b[i]) {
            std::println(stderr, "wrong result at {}: {} != {}", i, host_c[i], host_a[i] + host_b[i]);
            return EXIT_FAILURE;
        }
    }

    std::ranges::sort(milliseconds);
    const float best_ms = milliseconds.front();
    const float median_ms = milliseconds[milliseconds.size() / 2];
    const double bytes_moved = 3.0 * kElementCount * sizeof(float);
    const auto gb_per_s = [&](float ms) { return bytes_moved / (ms * 1e-3) / 1e9; };

    std::println("vector add: {} elements, {:.0f} MB moved per run, {} runs", kElementCount,
                 bytes_moved / 1e6, kTimedRuns);
    std::println("  best   {:.3f} ms  {:.0f} GB/s", best_ms, gb_per_s(best_ms));
    std::println("  median {:.3f} ms  {:.0f} GB/s", median_ms, gb_per_s(median_ms));
    std::println("  result checked against the CPU: OK");
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
