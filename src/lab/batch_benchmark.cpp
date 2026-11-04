#include "lab/lab.hpp"

#include "engine/model_loader.hpp"
#include "engine/weight_matrix.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <print>
#include <vector>

namespace murmur::lab {

namespace {

// RTX 4080 SUPER peak memory bandwidth (see `murmur-lab devices`).
constexpr double kPeakBandwidthBytesPerSecond = 736e9;
constexpr int kWarmupSteps = 4;
constexpr int kTimedSteps = 16;
// Best of several runs: other GPU work can only slow a run down.
constexpr int kRuns = 3;
constexpr int kBatchSizes[] = {1, 2, 4, 8, 16, 32, 64};
constexpr int kDefaultContext = 512;
constexpr TokenId kFillerToken = 9707;

}  // namespace

// Throughput versus latency of batched decode: B conversations, each with
// `context` tokens already cached, generate one token per step together.
// The step time is every conversation's time per token (latency); B / step
// time is the engine's total tokens per second (throughput). A step reads
// every weight once however large B is, plus each conversation's keys and
// values, so the bandwidth ceiling for B tokens is
//   B * peak bandwidth / (weight bytes + B * context * KV bytes per token).
//
// Before the table it times one prompt chunk (prefill) of `context` tokens.
int RunBatchBenchmark(Args args) {
    TransformerOptions options;
    options.kv_cache_bytes = std::size_t{4} << 30;   // room for 64 conversations
    int context = kDefaultContext;
    int only_batch = 0;   // 0: every batch size
    WeightFormat format = WeightFormat::kBf16;
    bool exact = false;
    bool no_graphs = false;
    ArgParser parser("murmur-lab batch");
    parser.PositiveInt("--context", "<N>", context, "tokens cached per conversation (default 512)")
        .Choice("--weights", "bf16|fp8|int4", format, ParseWeightFormat, "the model's weight format (default bf16)")
        .Choice("--math", "exact|gemm|cublas|fp8|fast", options.multi_row_math, ParseMultiRowMath,
                "how several rows are multiplied")
        .Switch("--exact", exact, "short for --math exact")
        .PositiveInt("--batch", "<N>", only_batch, "only this batch size (under a profiler, say)")
        .Switch("--no-graphs", no_graphs, "launch every kernel instead of replaying a CUDA graph");
    if (const std::optional<int> exit_code = parser.Parse(args)) {
        return *exit_code;
    }
    if (exact) {
        options.multi_row_math = MultiRowMath::kExactFp32;
    }
    options.use_cuda_graphs = !no_graphs;
    if (context + kWarmupSteps + kRuns * kTimedSteps > options.max_context) {
        std::println(stderr, "batch: context too long (max {})", options.max_context - kWarmupSteps - kRuns * kTimedSteps);
        return EXIT_FAILURE;
    }

    LoadedModel model = LoadModel(DefaultModelDirectory(), options, format);
    Transformer& transformer = *model.transformer;
    KvCacheManager& kv = transformer.KvCache();
    std::println("{} weights, {:.2f} GB; multi-row math {}; CUDA graphs {}; {} tokens of context per conversation",
                 WeightFormatName(format), transformer.WeightBytes() / 1e9, MultiRowMathName(options.multi_row_math),
                 options.use_cuda_graphs ? "on" : "off", context);

    const std::vector<TokenId> prompt(context, kFillerToken);
    {
        // One prompt chunk of `context` tokens (at most max_chunk): prefill speed.
        KvSequence sequence;
        const int count = std::min(context, options.max_chunk);
        if (!kv.Reserve(sequence, count)) {
            std::println(stderr, "batch: the KV cache is full");
            return EXIT_FAILURE;
        }
        const StepEntry entry{.sequence = &sequence, .tokens = std::span(prompt).first(count), .logits = 1};
        std::vector<float> sync(1);
        double best = 1e30;
        for (int run = 0; run < kRuns + 1; ++run) {   // the first run warms up
            transformer.CopyLogits(0, sync);
            const auto start = std::chrono::steady_clock::now();
            transformer.Step(std::span(&entry, 1));
            transformer.CopyLogits(0, sync);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            best = run > 0 ? std::min(best, seconds) : best;
        }
        kv.Release(sequence);
        std::println("prefill: {} tokens in {:.1f} ms ({:.0f} tokens/s)", count, 1e3 * best, count / best);
    }
    std::println("best of {} runs of {} decode steps\n", kRuns, kTimedSteps);
    std::println("{:>6} {:>12} {:>13} {:>15} {:>12} {:>10}", "batch", "ms/step", "tokens/s", "per conversation",
                 "ceiling t/s", "% of peak");

    const TokenId next = kFillerToken;
    for (const int batch : kBatchSizes) {
        if (only_batch != 0 && batch != only_batch) {
            continue;
        }
        if (batch > options.max_batch) {
            break;
        }
        std::vector<KvSequence> sequences(batch);
        for (KvSequence& sequence : sequences) {
            if (!kv.Reserve(sequence, context + kWarmupSteps + kRuns * kTimedSteps)) {
                std::println(stderr, "batch: the KV cache is full");
                return EXIT_FAILURE;
            }
            for (int begin = 0; begin < context; begin += options.max_chunk) {
                const int count = std::min(options.max_chunk, context - begin);
                const StepEntry entry{.sequence = &sequence,
                                      .tokens = std::span(prompt).subspan(begin, count),
                                      .logits = 0};
                transformer.Step(std::span(&entry, 1));
                sequence.length += count;
            }
        }
        std::vector<StepEntry> entries;
        for (KvSequence& sequence : sequences) {
            entries.push_back({.sequence = &sequence, .tokens = std::span(&next, 1)});
        }
        const auto step = [&] {
            transformer.Step(entries);
            for (KvSequence& sequence : sequences) {
                ++sequence.length;
            }
        };
        for (int i = 0; i < kWarmupSteps; ++i) {
            step();
        }
        std::vector<float> sync(1);
        double seconds = 1e30;
        for (int run = 0; run < kRuns; ++run) {
            transformer.CopyLogits(0, sync);   // waits for the GPU
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < kTimedSteps; ++i) {
                step();
            }
            transformer.CopyLogits(0, sync);
            seconds = std::min(seconds, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
        for (KvSequence& sequence : sequences) {
            kv.Release(sequence);
        }

        const double ms_per_step = 1e3 * seconds / kTimedSteps;
        const int average_context = context + kWarmupSteps + kRuns * kTimedSteps / 2;
        const double bytes_per_step = static_cast<double>(transformer.WeightBytes()) +
                                      static_cast<double>(batch) * average_context * transformer.KvBytesPerToken();
        const double ceiling = batch * kPeakBandwidthBytesPerSecond / bytes_per_step;
        const double tokens_per_second = batch * kTimedSteps / seconds;
        std::println("{:>6} {:>12.2f} {:>13.0f} {:>15.1f} {:>12.0f} {:>9.0f}%", batch, ms_per_step, tokens_per_second,
                     1e3 / ms_per_step, ceiling, 100.0 * tokens_per_second / ceiling);
    }
    kv.CheckInvariants();
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
