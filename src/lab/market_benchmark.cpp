#include "lab/lab.hpp"

#include "engine/engine.hpp"
#include "engine/model_loader.hpp"
#include "engine/weight_matrix.hpp"
#include "game/market_tuning.hpp"
#include "game/personas.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <print>
#include <string>
#include <vector>

namespace murmur::lab {

namespace {

constexpr int kPlayer = 1000;   // the player's conversation id
constexpr const char* kNpcCue = "(Someone slows down near your stall. Call out to them in two short sentences.)";
constexpr const char* kPlayerQuestions[] = {
    "What's good tonight?", "Have you heard any rumours about the docks?", "How much for that?",
    "Is the market safe after midnight?", "Who should I talk to about a cheap implant?",
};

using Clock = std::chrono::steady_clock;

double Seconds(Clock::duration d) {
    return std::chrono::duration<double>(d).count();
}

double Percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::ranges::sort(values);
    const std::size_t index = static_cast<std::size_t>(fraction * (values.size() - 1) + 0.5);
    return values[std::min(index, values.size() - 1)];
}

}  // namespace

// The market under load, without the game: `npcs` conversations (market
// personas sharing the city lore) talk non-stop, each starting a new reply
// as soon as the last one ends, while the player asks a question every
// `player-every` seconds at higher priority. Reports the player's time to
// first token (p50/p99/max), throughput, step times and evictions, then
// closes every conversation and checks that every page of the KV cache came
// back: the allocator's books must balance after any amount of play.
int RunMarketBenchmark(Args args) {
    int npc_count = 30;
    double seconds = 60;
    double player_every = 2;
    int kv_mb = 1024;
    WeightFormat format = WeightFormat::kBf16;
    TransformerOptions options;
    ArgParser parser("murmur-lab market");
    parser.PositiveInt("--npcs", "<N>", npc_count, "NPCs talking (default 30, at most the whole cast)")
        .PositiveNumber("--seconds", "<S>", seconds, "how long the market runs (default 60)")
        .PositiveNumber("--player-every", "<S>", player_every, "seconds between the player's questions (default 2)")
        .PositiveInt("--kv-mb", "<MB>", kv_mb, "the KV cache's size (default 1024)")
        .Choice("--weights", "bf16|fp8|int4", format, ParseWeightFormat, "the model's weight format (default bf16)")
        .Choice("--kv", "fp32|bf16|fp8", options.kv_format, ParseKvFormat, "the KV cache's format (default bf16)")
        .Choice("--math", "exact|gemm|cublas|fp8|fast", options.multi_row_math, ParseMultiRowMath,
                "how several rows are multiplied");
    if (const std::optional<int> exit_code = parser.Parse(args)) {
        return *exit_code;
    }
    const int npcs = std::min(npc_count, static_cast<int>(personas::kMarketCast.size()));

    options.kv_cache_bytes = static_cast<std::size_t>(kv_mb) << 20;
    LoadedModel model = LoadModel(DefaultModelDirectory(), options, format);
    Engine engine(*model.transformer, model.tokenizer);
    for (int id = 0; id < npcs; ++id) {
        ConversationOptions conversation;
        conversation.system_prompt = personas::MarketSystemPrompt(personas::kMarketCast[id]);
        conversation.sampling.seed = static_cast<std::uint64_t>(id) + 1;
        conversation.max_reply_tokens = 64;
        conversation.memory_tokens = game::kNpcMemoryTokens;
        engine.Open(id, conversation);
        engine.Send(id, kNpcCue);
    }
    ConversationOptions player;
    player.system_prompt = personas::MarketSystemPrompt(personas::kMarketCast[0]);
    player.priority = game::kPlayerPriority;
    player.max_reply_tokens = 48;
    player.memory_tokens = game::kNpcMemoryTokens;   // in the game the player talks to an NPC, in its conversation
    engine.Open(kPlayer, player);

    std::println("{} NPCs talking non-stop for {:.0f} s; the player asks every {:.1f} s; {} weights, {} math; "
                 "{} KV cache of {} pages ({} MB)\n",
                 npcs, seconds, player_every, WeightFormatName(format), MultiRowMathName(options.multi_row_math),
                 KvFormatName(options.kv_format), engine.Cache().PageCount(), kv_mb);
    std::vector<double> player_first_token_ms;
    std::vector<double> step_ms;
    std::vector<int> npcs_in_batch;
    std::int64_t npc_tokens = 0;
    int question = 0;
    bool player_replying = false;
    const Clock::time_point start = Clock::now();
    Clock::time_point next_question = start + std::chrono::milliseconds(500);
    while (Seconds(Clock::now() - start) < seconds) {
        if (!player_replying && Clock::now() >= next_question) {
            engine.Send(kPlayer, kPlayerQuestions[question++ % std::size(kPlayerQuestions)]);
            player_replying = true;
            next_question = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                               std::chrono::duration<double>(player_every));
        }
        for (const EngineEvent& event : engine.Step()) {
            if (event.kind != EngineEventKind::kReplyFinished) {
                continue;
            }
            if (event.conversation == kPlayer) {
                player_first_token_ms.push_back(event.stats.prefill_ms);
                player_replying = false;
            } else {
                npc_tokens += event.stats.reply_tokens;
                engine.Send(event.conversation, kNpcCue);   // and straight on to the next line
            }
        }
        const EngineStats stats = engine.Stats();
        step_ms.push_back(stats.last_step_ms);
        npcs_in_batch.push_back(static_cast<int>(stats.last_batch.size()));
    }
    const double elapsed = Seconds(Clock::now() - start);
    const EngineStats stats = engine.Stats();

    std::println("steps {}   generated {} tokens = {:.0f} tokens/s   evictions {}   forgotten turns {}   "
                 "prompt tokens reused {}",
                 stats.steps, stats.generated_tokens, stats.generated_tokens / elapsed, stats.evictions,
                 stats.forgotten_turns, stats.reused_tokens);
    std::println("conversations per step: median {:.0f}, max {:.0f}",
                 Percentile(std::vector<double>(npcs_in_batch.begin(), npcs_in_batch.end()), 0.5),
                 Percentile(std::vector<double>(npcs_in_batch.begin(), npcs_in_batch.end()), 1.0));
    std::println("step time: p50 {:.1f} ms, p99 {:.1f} ms", Percentile(step_ms, 0.5), Percentile(step_ms, 0.99));
    std::println("player time to first token over {} questions: p50 {:.1f} ms, p99 {:.1f} ms, max {:.1f} ms",
                 player_first_token_ms.size(), Percentile(player_first_token_ms, 0.5),
                 Percentile(player_first_token_ms, 0.99), Percentile(player_first_token_ms, 1.0));

    // Every conversation closes; with the prefix cache emptied, every page
    // must be free again.
    for (int id = 0; id < npcs; ++id) {
        engine.Close(id);
    }
    engine.Close(kPlayer);
    KvCacheManager& cache = engine.Cache();
    while (cache.EvictCachedPage()) {
    }
    cache.CheckInvariants();
    const bool balanced = cache.FreePages() == cache.PageCount();
    std::println("after closing everything: {} / {} pages free ({})", cache.FreePages(), cache.PageCount(),
                 balanced ? "no leaks" : "LEAK");
    return balanced ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace murmur::lab
