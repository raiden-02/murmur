#include "lab/lab.hpp"

#include "engine/model_loader.hpp"
#include "engine/weight_matrix.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <optional>
#include <print>
#include <string_view>
#include <vector>

namespace murmur::lab {

namespace {

// RTX 4080 SUPER peak memory bandwidth (see `murmur-lab devices`).
constexpr double kPeakBandwidthBytesPerSecond = 736e9;
constexpr int kWarmupTokens = 8;
constexpr int kTimedTokens = 32;
// Best of several runs: other GPU work (a desktop compositor, an open game
// editor) can only slow a run down, so the fastest is the engine's own speed.
constexpr int kRuns = 4;
constexpr int kDefaultContexts[] = {128, 1024, 3800};

}  // namespace

// Single-stream decode speed at several context lengths, against the memory
// bandwidth ceiling. Each step must read every weight and the whole KV cache
// once, so the best possible time per token is (weight bytes + KV bytes) /
// peak bandwidth. Greedy decoding keeps the run deterministic.
int RunDecodeBenchmark(Args args) {
    TransformerOptions options;
    std::vector<int> contexts;
    WeightFormat format = WeightFormat::kBf16;
    bool no_graphs = false;
    ArgParser parser("murmur-lab decode");
    parser.Choice("--weights", "bf16|fp8|int4", format, ParseWeightFormat, "the model's weight format (default bf16)")
        .Switch("--no-graphs", no_graphs, "launch every kernel instead of replaying a CUDA graph")
        .Positionals("<context>", "context lengths to time (default 128 1024 3800)", [&contexts](std::string_view arg) {
            const std::optional<int> context = ParsePositiveInt(arg);
            if (context) {
                contexts.push_back(*context);
            }
            return context.has_value();
        });
    if (const std::optional<int> exit_code = parser.Parse(args)) {
        return *exit_code;
    }
    options.use_cuda_graphs = !no_graphs;
    if (contexts.empty()) {
        contexts.assign(std::begin(kDefaultContexts), std::end(kDefaultContexts));
    }

    LoadedModel model = LoadModel(DefaultModelDirectory(), options, format);
    Transformer& transformer = *model.transformer;
    const int vocab = model.tokenizer.VocabSize();
    std::println("{} weights, {:.2f} GB; CUDA graphs {}; best of {} runs of {} decode steps\n",
                 WeightFormatName(format), transformer.WeightBytes() / 1e9, options.use_cuda_graphs ? "on" : "off",
                 kRuns, kTimedTokens);
    std::println("{:>8} {:>10} {:>10} {:>12} {:>9}", "context", "ms/token", "tokens/s", "ceiling t/s", "% of peak");

    const int max_context = transformer.Options().max_context - kWarmupTokens - kRuns * kTimedTokens;
    for (const int context : contexts) {
        if (context > max_context) {
            std::println(stderr, "decode: context {} leaves no room for {} steps (max {})", context,
                         kWarmupTokens + kRuns * kTimedTokens, max_context);
            return EXIT_FAILURE;
        }
        // A prompt of `context` tokens (the content doesn't matter for speed);
        // the timed steps follow it.
        const std::vector<TokenId> prompt(context, 9707);
        transformer.Forward(prompt, 0);
        int position = static_cast<int>(prompt.size());
        TokenId next = transformer.ArgmaxLastLogits(vocab);
        const auto step = [&] {
            transformer.Forward(std::span(&next, 1), position++);
            next = transformer.ArgmaxLastLogits(vocab);
        };
        for (int i = 0; i < kWarmupTokens; ++i) {
            step();
        }
        double seconds = 1e30;
        for (int run = 0; run < kRuns; ++run) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < kTimedTokens; ++i) {
                step();
            }
            seconds = std::min(seconds, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }

        const double ms_per_token = 1e3 * seconds / kTimedTokens;
        const int average_context = context + kWarmupTokens + kRuns * kTimedTokens / 2;
        const double bytes_per_token = static_cast<double>(transformer.WeightBytes()) +
                                       static_cast<double>(average_context) * transformer.KvBytesPerToken();
        const double ceiling = kPeakBandwidthBytesPerSecond / bytes_per_token;
        const double tokens_per_second = kTimedTokens / seconds;
        std::println("{:>8} {:>10.2f} {:>10.1f} {:>12.1f} {:>8.0f}%", context, ms_per_token, tokens_per_second,
                     ceiling, 100.0 * tokens_per_second / ceiling);
    }
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
