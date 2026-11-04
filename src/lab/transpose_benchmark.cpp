#include "lab/lab.hpp"

#include "common/device_buffer.hpp"
#include "common/gpu_timer.hpp"
#include "kernels/transpose.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <print>
#include <vector>

namespace murmur::lab {

namespace {

// 8192 x 8192 floats = 256 MiB per matrix, far larger than the 64 MiB L2.
constexpr int kSize = 8192;
constexpr int kWarmupRuns = 3;
constexpr int kTimedRuns = 20;

constexpr TransposeVersion kVersions[] = {
    TransposeVersion::kNaive,
    TransposeVersion::kTiled,
    TransposeVersion::kTiledPadded,
};

}  // namespace

// Times each transpose version on an 8192 x 8192 matrix and reports achieved
// bandwidth: every element is read once and written once, 2 x 4 bytes.
// Each version's output is checked against the CPU before it is reported.
int RunTransposeBenchmark(Args args) {
    if (const std::optional<int> exit_code = ArgParser("murmur-lab transpose").Parse(args)) {
        return *exit_code;
    }
    const std::size_t count = static_cast<std::size_t>(kSize) * kSize;
    std::vector<float> host_input(count);
    for (std::size_t i = 0; i < count; ++i) {
        host_input[i] = static_cast<float>(i % 65536);
    }

    DeviceBuffer<float> input(count);
    DeviceBuffer<float> output(count);
    input.CopyFromHost(host_input);
    std::vector<float> host_output(count);

    const double bytes_moved = 2.0 * count * sizeof(float);
    std::println("transpose: {} x {} floats, {:.0f} MB moved per run (read + write), {} runs",
                 kSize, kSize, bytes_moved / 1e6, kTimedRuns);

    GpuTimer timer;
    for (TransposeVersion version : kVersions) {
        for (int run = 0; run < kWarmupRuns; ++run) {
            Transpose(input, output, kSize, kSize, version);
        }

        output.CopyToHost(host_output);
        for (std::size_t r = 0; r < kSize; ++r) {
            for (std::size_t c = 0; c < kSize; ++c) {
                if (host_output[c * kSize + r] != host_input[r * kSize + c]) {
                    std::println(stderr, "{}: wrong value at row {} col {}",
                                 TransposeVersionName(version), r, c);
                    return EXIT_FAILURE;
                }
            }
        }

        std::vector<float> milliseconds;
        for (int run = 0; run < kTimedRuns; ++run) {
            timer.Start();
            Transpose(input, output, kSize, kSize, version);
            milliseconds.push_back(timer.StopMilliseconds());
        }
        std::ranges::sort(milliseconds);
        const float best_ms = milliseconds.front();
        const float median_ms = milliseconds[milliseconds.size() / 2];
        const auto gb_per_s = [&](float ms) { return bytes_moved / (ms * 1e-3) / 1e9; };
        std::println("  {:<16} best {:.3f} ms {:>4.0f} GB/s   median {:.3f} ms {:>4.0f} GB/s",
                     TransposeVersionName(version), best_ms, gb_per_s(best_ms), median_ms,
                     gb_per_s(median_ms));
    }
    std::println("  all versions checked against the CPU: OK");
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
