#include "lab/lab.hpp"

#include "engine/engine.hpp"
#include "engine/model_loader.hpp"
#include "engine/weight_matrix.hpp"
#include "game/personas.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <vector>

namespace murmur::lab {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kRequests = 64;
constexpr int kMaxTokens = 256;
constexpr int kRepeats = 5;
constexpr int kBatchSizes[] = {1, 8, 32, 64};

// The shared workload: what every engine in the comparison is asked, word
// for word. Each request starts with its own tag, so no engine can reuse
// another request's cached prompt: the comparison measures computing, not
// caching.
struct Request {
    std::string system;
    std::string user;
};

nlohmann::json MakeWorkload() {
    nlohmann::json requests = nlohmann::json::array();
    for (int i = 0; i < kRequests; ++i) {
        const personas::Persona& persona = personas::kMarketCast[i % personas::kMarketCast.size()];
        requests.push_back(
            {{"system", std::format("[Request {}] ", i) + std::string(personas::kCityLore)},
             {"user", std::format("Tell me a long, detailed story about {}, who {}. Make it at least 1,500 words.",
                                  persona.name, persona.role)}});
    }
    return {{"max_tokens", kMaxTokens},
            {"sampling", "greedy"},
            {"note", "each request is a system prompt (a unique tag, then the market's city lore) and one user "
                     "message; engines generate until the end of the turn or max_tokens"},
            {"requests", requests}};
}

std::vector<Request> ReadWorkload(const nlohmann::json& workload) {
    std::vector<Request> requests;
    for (const auto& r : workload.at("requests")) {
        requests.push_back({r.at("system").get<std::string>(), r.at("user").get<std::string>()});
    }
    return requests;
}

double Seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double Median(std::vector<double> values) {
    std::ranges::sort(values);
    return values[values.size() / 2];
}

struct Batch {
    double seconds = 0.0;
    int tokens = 0;
    int prompt_tokens = 0;
    std::vector<ReplyStats> replies;
    std::string first_text;   // the reply to the first request, whole
};

// Sends `count` requests (from `first` on) at once and runs the engine until
// every reply is done.
Batch RunBatch(Transformer& model, const Tokenizer& tokenizer, std::span<const Request> requests, int first, int count,
               int max_tokens) {
    model.KvCache().ForgetCachedPrefixes();   // nothing carried over from the previous run
    Engine engine(model, tokenizer, {.max_prefill_tokens = model.Options().max_chunk});
    const SamplingParams greedy{.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .repetition_penalty = 1.0f};
    for (int i = 0; i < count; ++i) {
        const Request& r = requests[(first + i) % requests.size()];
        engine.Open(i, {.system_prompt = r.system, .sampling = greedy, .max_reply_tokens = max_tokens});
    }
    Batch batch;
    const Clock::time_point start = Clock::now();
    for (int i = 0; i < count; ++i) {
        engine.Send(i, requests[(first + i) % requests.size()].user);
    }
    while (engine.HasWork()) {
        for (const EngineEvent& event : engine.Step()) {
            if (event.kind == EngineEventKind::kText && event.conversation == 0) {
                batch.first_text += event.text;
            } else if (event.kind == EngineEventKind::kReplyFinished) {
                batch.replies.push_back(event.stats);
                batch.tokens += event.stats.reply_tokens;
                batch.prompt_tokens += event.stats.prompt_tokens;
            }
        }
    }
    batch.seconds = Seconds(start);
    return batch;
}

}  // namespace

// The same workload as bench/compare's vLLM and llama.cpp runners, through
// Murmur's engine end to end (scheduling, sampling, the KV cache):
//   - time to first token: one request at a time, one token (median of 5);
//   - decode: one request at a time, 256 tokens; (time - time to first token)
//     / (tokens - 1) (median of 5);
//   - throughput: 1, 8, 32 and 64 requests at once, all generated tokens
//     over the wall time from the first send to the last token.
// Every request starts with its own tag, so nothing is reused from a cache.
int RunCompareBenchmark(Args args) {
    WeightFormat weights = WeightFormat::kBf16;
    std::filesystem::path workload_path = "bench/compare/workload.json";
    std::string json_path;
    int only_batch = 0;   // 0: every throughput run
    ArgParser parser("murmur-lab compare");
    parser.Choice("--weights", "bf16|fp8|int4", weights, ParseWeightFormat, "the model's weight format (default bf16)")
        .Value("--workload", "<path>", "the shared workload (default bench/compare/workload.json; written if missing)",
               [&workload_path](std::string_view path) {
                   workload_path = path;
                   return true;
               })
        .PositiveInt("--batch", "<N>", only_batch, "only the throughput run with N requests (under a profiler, say)")
        .Text("--json", "<path>", json_path, "also write the results there");
    if (const std::optional<int> exit_code = parser.Parse(args)) {
        return *exit_code;
    }
    if (!std::filesystem::exists(workload_path)) {
        std::filesystem::create_directories(workload_path.parent_path());
        std::ofstream(workload_path) << MakeWorkload().dump(1) << '\n';
        std::println("wrote the workload to {}", workload_path.string());
    }
    const nlohmann::json workload = nlohmann::json::parse(std::ifstream(workload_path));
    const std::vector<Request> requests = ReadWorkload(workload);
    const int max_tokens = workload.at("max_tokens").get<int>();

    TransformerOptions options;
    options.kv_cache_bytes = std::size_t{4} << 30;   // 64 requests of ~700 tokens, nothing evicted
    LoadedModel model = LoadModel(DefaultModelDirectory(), options, weights);
    Transformer& transformer = *model.transformer;
    std::println("Murmur, {} weights, {} requests of up to {} tokens, greedy\n", WeightFormatName(weights),
                 requests.size(), max_tokens);

    RunBatch(transformer, model.tokenizer, requests, 0, 8, 32);   // warm-up: kernels, allocations
    transformer.RecordDecodeGraphs();   // as vLLM does at startup
    if (only_batch > 0) {
        const Batch batch = RunBatch(transformer, model.tokenizer, requests, 0, only_batch, max_tokens);
        std::println("{} requests: {:.0f} tokens/s", only_batch, batch.tokens / batch.seconds);
        return EXIT_SUCCESS;
    }
    std::vector<double> ttft;
    std::vector<double> decode;
    int prompt_tokens = 0;
    std::string first_reply;
    for (int r = 0; r < kRepeats; ++r) {
        const Batch first = RunBatch(transformer, model.tokenizer, requests, r, 1, 1);
        ttft.push_back(first.replies.front().prefill_ms);
        prompt_tokens = first.replies.front().prompt_tokens;
        const Batch full = RunBatch(transformer, model.tokenizer, requests, kRepeats + r, 1, max_tokens);
        const ReplyStats& reply = full.replies.front();
        if (r == 0) {
            first_reply = full.first_text;
        }
        decode.push_back((reply.reply_tokens - 1) / (reply.decode_ms / 1e3));
    }
    std::println("time to first token {:.1f} ms ({} prompt tokens), decode {:.0f} tokens/s", Median(ttft),
                 prompt_tokens, Median(decode));

    nlohmann::json throughput = nlohmann::json::array();
    std::println("\n{:>9} {:>12} {:>9} {:>14}", "requests", "tokens/s", "seconds", "tokens out");
    for (const int n : kBatchSizes) {
        const Batch batch = RunBatch(transformer, model.tokenizer, requests, 0, n, max_tokens);
        const double rate = batch.tokens / batch.seconds;
        std::println("{:>9} {:>12.0f} {:>9.2f} {:>14}", n, rate, batch.seconds, batch.tokens);
        throughput.push_back({{"requests", n},
                              {"tokens_per_second", rate},
                              {"seconds", batch.seconds},
                              {"tokens", batch.tokens},
                              {"prompt_tokens", batch.prompt_tokens}});
    }
    if (!json_path.empty()) {
        const nlohmann::json json = {{"engine", "Murmur"},
                                     {"format", WeightFormatName(weights)},
                                     {"weight_gb", transformer.WeightBytes() / 1e9},
                                     {"ttft_ms", Median(ttft)},
                                     {"prompt_tokens", prompt_tokens},
                                     {"decode_tokens_per_second", Median(decode)},
                                     {"first_reply", first_reply},
                                     {"throughput", throughput}};
        std::filesystem::create_directories(std::filesystem::path(json_path).parent_path());
        std::ofstream(json_path) << json.dump(1) << '\n';
    }
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
