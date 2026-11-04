#include "lab/lab.hpp"

#include "common/cuda_check.hpp"
#include "common/device_buffer.hpp"
#include "common/gpu_timer.hpp"
#include "engine/kernels/decode_attention.cuh"
#include "engine/kv_cache.hpp"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <print>
#include <random>
#include <vector>

namespace murmur::lab {

namespace {

constexpr double kPeakBandwidthGBs = 736.0;
constexpr int kHeads = 12;   // Qwen2.5-1.5B
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 128;
constexpr int kLayers = 28;   // one cache per layer, as in the model: together far larger than L2
constexpr int kPageSize = 16;
constexpr int kWarmupCalls = 200;
constexpr int kBatches = 10;

}  // namespace

// Decode attention alone, as one step of the model runs it: `batch`
// conversations of `context` cached tokens each, keys and values on
// shuffled pages, one call per layer over 28 layers' caches (so nothing
// stays in L2). Reports the time per call and the bandwidth against the
// bytes it must read (every cached key and value once).
int RunAttentionBenchmark(Args args) {
    int batch = 64;
    int context = 512;
    KvFormat format = KvFormat::kBf16;
    ArgParser parser("murmur-lab attention");
    parser.PositiveInt("--batch", "<N>", batch, "conversations decoding together (default 64)")
        .PositiveInt("--context", "<L>", context, "cached tokens per conversation (default 512)")
        .Choice("--kv", "fp32|bf16|fp8", format, ParseKvFormat, "the KV cache's format (default bf16)");
    if (const std::optional<int> exit_code = parser.Parse(args)) {
        return *exit_code;
    }
    const int pages_per_sequence = (context + kPageSize - 1) / kPageSize;
    const int pages = batch * pages_per_sequence;
    const std::size_t rows = static_cast<std::size_t>(pages) * kPageSize;
    const int kv_width = kKvHeads * kHeadDim;
    const std::size_t value_bytes = format == KvFormat::kFp8 ? 1 : format == KvFormat::kBf16 ? 2 : 4;

    struct Layer {
        DeviceBuffer<std::uint8_t> keys, values;
        DeviceBuffer<float> key_scales, value_scales;
    };
    std::vector<Layer> layers;
    for (int l = 0; l < kLayers; ++l) {
        Layer layer{DeviceBuffer<std::uint8_t>(rows * kv_width * value_bytes),
                    DeviceBuffer<std::uint8_t>(rows * kv_width * value_bytes), DeviceBuffer<float>(rows * kKvHeads),
                    DeviceBuffer<float>(rows * kKvHeads)};
        CudaCheck(cudaMemset(layer.keys.Data(), 0x3C, layer.keys.SizeBytes()));
        CudaCheck(cudaMemset(layer.values.Data(), 0x3C, layer.values.SizeBytes()));
        CudaCheck(cudaMemset(layer.key_scales.Data(), 0, layer.key_scales.SizeBytes()));
        CudaCheck(cudaMemset(layer.value_scales.Data(), 0, layer.value_scales.SizeBytes()));
        layers.push_back(std::move(layer));
    }
    std::vector<int> page_ids(pages);
    std::iota(page_ids.begin(), page_ids.end(), 0);
    std::shuffle(page_ids.begin(), page_ids.end(), std::mt19937(7));
    DeviceBuffer<int> tables(page_ids.size());
    tables.CopyFromHost(page_ids);
    DeviceBuffer<int> contexts(batch);
    contexts.CopyFromHost(std::vector<int>(batch, context));
    DeviceBuffer<float> q(static_cast<std::size_t>(batch) * kHeads * kHeadDim);
    DeviceBuffer<float> out(q.Size());
    CudaCheck(cudaMemset(q.Data(), 0, q.SizeBytes()));
    DecodeAttentionWorkspace workspace(
        {.num_heads = kHeads, .num_kv_heads = kKvHeads, .head_dim = kHeadDim, .max_context = 4096, .max_batch = batch});
    const PagedKv kv{.page_tables = tables.Data(), .page_table_stride = pages_per_sequence, .page_shift = 4};

    const auto call = [&](int i) {
        Layer& layer = layers[i % kLayers];
        const KvLayer cache{.format = format,
                            .keys = layer.keys.Data(),
                            .values = layer.values.Data(),
                            .key_scales = layer.key_scales.Data(),
                            .value_scales = layer.value_scales.Data()};
        DecodeAttention(q.Data(), cache, kv, contexts.Data(), batch, out.Data(), workspace, nullptr);
    };
    for (int i = 0; i < kWarmupCalls; ++i) {
        call(i);
    }
    GpuTimer timer;
    double best_us = 1e30;
    for (int b = 0; b < kBatches; ++b) {
        timer.Start();
        for (int i = 0; i < kLayers; ++i) {
            call(i);
        }
        best_us = std::min(best_us, 1e3 * timer.StopMilliseconds() / kLayers);
    }
    const double bytes = 2.0 * batch * context * kv_width * value_bytes;
    const double floor_us = bytes / (kPeakBandwidthGBs * 1e3);
    std::println("{} cache, {} conversations x {} tokens: {:.1f} us per layer, {:.0f} GB/s, {:.0f}% of peak "
                 "(floor {:.1f} us)",
                 KvFormatName(format), batch, context, best_us, bytes / (best_us * 1e3),
                 100.0 * floor_us / best_us, floor_us);
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
