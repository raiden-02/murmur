#include "lab/lab.hpp"

#include "engine/engine.hpp"
#include "engine/model_loader.hpp"
#include "engine/speculator.hpp"
#include "engine/weight_matrix.hpp"
#include "game/personas.hpp"
#include "model/chat_template.hpp"
#include "model/model_config.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <vector>

namespace murmur::lab {

namespace {

constexpr int kConversations = 8;
constexpr const char* kQuestions[] = {
    "What's good tonight?", "Have you heard any rumours about the docks?", "How much for that?",
    "Is the market safe after midnight?", "Who should I talk to about a cheap implant?",
};

struct Run {
    double draft_ms = 0.0;   // summed over steps
    double step_ms = 0.0;
    int steps = 0;
    int tokens = 0;
    double decode_ms = 0.0;
    int drafted = 0;
    int accepted = 0;
    std::vector<std::string> replies;
    nlohmann::json first_reply = nlohmann::json::array();   // conversation 0's pieces: [text, guessed]
};

// One reply from each of kConversations market NPCs, one conversation at a
// time (the latency of a single conversation is what speculation buys).
Run Converse(Transformer& model, const Tokenizer& tokenizer, Speculator* speculator, int draft_tokens,
             const SamplingParams& sampling) {
    Run run;
    Engine engine(model, tokenizer, {.draft_tokens = draft_tokens}, speculator);
    for (int i = 0; i < kConversations; ++i) {
        ConversationOptions options{.system_prompt = personas::MarketSystemPrompt(personas::kMarketCast[i]),
                                    .sampling = sampling,
                                    .max_reply_tokens = 96,
                                    .speculative = speculator != nullptr};
        options.sampling.seed = static_cast<std::uint64_t>(i) + 1;
        engine.Open(i, options);
        engine.Send(i, kQuestions[i % std::size(kQuestions)]);
        std::string text;
        while (engine.HasWork()) {
            const std::vector<EngineEvent> events = engine.Step();
            const EngineStats& stats = engine.Stats();
            if (stats.last_prefill_rows == 0) {   // decode steps only
                run.draft_ms += stats.last_draft_ms;
                run.step_ms += stats.last_step_ms;
                ++run.steps;
            }
            for (const EngineEvent& event : events) {
                if (event.kind == EngineEventKind::kText) {
                    text += event.text;
                    if (i == 0) {
                        run.first_reply.push_back({event.text, event.drafted});
                    }
                } else {
                    // The first token comes from the prompt; time the rest.
                    run.tokens += event.stats.reply_tokens - 1;
                    run.decode_ms += event.stats.decode_ms;
                    run.drafted += event.stats.drafted_tokens;
                    run.accepted += event.stats.accepted_tokens;
                }
            }
        }
        engine.Close(i);
        run.replies.push_back(text);
    }
    return run;
}

}  // namespace

// Speculative decoding for one conversation at a time: the 0.5B draft model
// guesses k tokens, the main model checks them in one pass. For each k,
// reports decode speed against plain decoding, how many guesses were kept,
// and whether the replies came out identical (greedy decoding must match
// exactly). Eight market NPCs answer one question each.
int RunSpeculativeBenchmark(Args args) {
    WeightFormat weights = WeightFormat::kFp8;
    WeightFormat draft_weights = WeightFormat::kInt4;
    bool sample = false;
    std::string json_path;
    ArgParser parser("murmur-lab speculative");
    parser.Choice("--weights", "bf16|fp8|int4", weights, ParseWeightFormat, "the model's weight format (default fp8)")
        .Choice("--draft-weights", "bf16|fp8|int4", draft_weights, ParseWeightFormat,
                "the draft model's weight format (default int4)")
        .Switch("--sample", sample, "the NPCs' sampling settings instead of greedy")
        .Text("--json", "<path>", json_path, "also write the results there");
    if (const std::optional<int> exit_code = parser.Parse(args)) {
        return *exit_code;
    }

    LoadedModel model = LoadModel(DefaultModelDirectory(), {}, weights);
    const std::unique_ptr<Speculator> speculator = LoadSpeculator(DraftModelDirectory(), draft_weights, model.tokenizer);
    const SamplingParams sampling =
        sample ? SamplingParams{} : SamplingParams{.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .repetition_penalty = 1.0f};

    std::println("{} main weights, {} draft weights, {}; {} conversations, one at a time\n", WeightFormatName(weights),
                 WeightFormatName(draft_weights), sample ? "sampled" : "greedy", kConversations);
    Converse(*model.transformer, model.tokenizer, nullptr, 0, sampling);   // warm-up
    const Run plain = Converse(*model.transformer, model.tokenizer, nullptr, 0, sampling);
    const double plain_speed = plain.tokens / (plain.decode_ms / 1e3);
    std::println("{:>8} {:>12} {:>9} {:>10} {:>16} {:>14} {:>14}", "guesses", "tokens/s", "speed-up", "kept",
                 "identical replies", "draft ms/step", "check ms/step");
    std::println("{:>8} {:>12.0f} {:>8.2f}x {:>10} {:>16} {:>14} {:>14.2f}", "none", plain_speed, 1.0, "", "", "",
                 plain.step_ms / plain.steps);
    nlohmann::json json = {{"weights", WeightFormatName(weights)},
                           {"draft_weights", WeightFormatName(draft_weights)},
                           {"sampled", sample},
                           {"plain_tokens_per_second", plain_speed},
                           {"plain_step_ms", plain.step_ms / plain.steps},
                           {"runs", nlohmann::json::array()}};
    for (const int k : {1, 2, 3, 4, 6, 8}) {
        const Run run = Converse(*model.transformer, model.tokenizer, speculator.get(), k, sampling);
        const double speed = run.tokens / (run.decode_ms / 1e3);
        int identical = 0;
        for (int i = 0; i < kConversations; ++i) {
            identical += run.replies[i] == plain.replies[i] ? 1 : 0;
        }
        const double kept = 100.0 * run.accepted / std::max(1, run.drafted);
        std::println("{:>8} {:>12.0f} {:>8.2f}x {:>9.0f}% {:>13}/{} {:>14.2f} {:>14.2f}", k, speed, speed / plain_speed,
                     kept, identical, kConversations, run.draft_ms / run.steps, (run.step_ms - run.draft_ms) / run.steps);
        json["runs"].push_back({{"draft_tokens", k},
                                {"tokens_per_second", speed},
                                {"kept_percent", kept},
                                {"draft_ms_per_step", run.draft_ms / run.steps},
                                {"check_ms_per_step", (run.step_ms - run.draft_ms) / run.steps},
                                {"identical", identical},
                                {"first_reply", run.first_reply}});
    }
    if (!json_path.empty()) {
        std::ofstream(json_path) << json.dump(1) << '\n';
    }
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
