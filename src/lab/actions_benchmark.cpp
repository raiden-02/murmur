#include "lab/lab.hpp"

#include "engine/engine.hpp"
#include "engine/grammar.hpp"
#include "engine/model_loader.hpp"
#include "engine/weight_matrix.hpp"
#include "game/action_language.hpp"
#include "game/market_tuning.hpp"
#include "game/personas.hpp"
#include "model/chat_template.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <vector>

namespace murmur::lab {

namespace {

using Clock = std::chrono::steady_clock;

// What the player says, chosen so that most actions have a reason to come up.
constexpr std::array<std::string_view, 8> kRequests{
    "Can I buy something from you?",
    "Do you have a job for me? I can carry things.",
    "I'm broke and hungry. Can you spare anything?",
    "Go and see what's happening at the docks.",
    "Can you hack a sign for me? Make it say something rude about Tessera.",
    "My phone is cracked. Get someone over here who can fix it.",
    "What's good tonight?",
    "Tell me a rumour.",
};

// Ordinary printable text, 1 to 64 characters: the same framing ("ACTION:"
// after the reply, then the end of the turn) with no grammar to follow.
constexpr std::string_view kFreeText = "free ::= [ -~]{1,64}\n";

std::string_view Trim(std::string_view text) {
    while (!text.empty() && text.front() == ' ') {
        text.remove_prefix(1);
    }
    while (!text.empty() && text.back() == ' ') {
        text.remove_suffix(1);
    }
    return text;
}

// The action with what the model chose in [brackets]; the rest the grammar wrote.
std::string MarkChoices(std::string_view text, std::string_view forced) {
    std::string marked;
    bool open = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const bool chosen = i >= forced.size() || forced[i] == 0;
        if (chosen != open) {
            marked += chosen ? '[' : ']';
            open = chosen;
        }
        marked += text[i];
    }
    return open ? marked + ']' : marked;
}

double Milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Run {
    int actions = 0;
    int parsed = 0;
    std::map<std::string, int> verbs;
    std::vector<std::string> unparsed;
    std::map<std::string, std::string> examples;   // per verb, one action with the model's picks in [brackets]
    std::int64_t sampled = 0;   // action tokens the model chose
    std::int64_t forced = 0;    // ...and the grammar wrote
};

// Every market NPC answers every request, all at once (as in the game), and
// ends with an action written under `grammar`.
Run Act(Transformer& model, const Tokenizer& tokenizer, const TokenGrammar& grammar, bool jump_forward) {
    Engine engine(model, tokenizer, {.jump_forward = jump_forward});
    engine.SetActionLanguage({.grammar = std::shared_ptr<const TokenGrammar>(&grammar, [](const TokenGrammar*) {}),
                              .lead_in = tokenizer.Encode("\nACTION:", SpecialTokens::kAsText)});
    const std::string rules = actions::Rules();
    const int npcs = static_cast<int>(personas::kMarketCast.size());
    for (int i = 0; i < npcs; ++i) {
        ConversationOptions options{.system_prompt = personas::MarketSystemPrompt(personas::kMarketCast[i], rules),
                                    .max_reply_tokens = 96,
                                    .memory_tokens = game::kNpcMemoryTokens,
                                    .acts = true};
        options.sampling.seed = static_cast<std::uint64_t>(i) + 1;
        engine.Open(i, options);
    }
    Run run;
    for (const std::string_view request : kRequests) {
        for (int i = 0; i < npcs; ++i) {
            engine.Send(i, request);
        }
        while (engine.HasWork()) {
            for (const EngineEvent& event : engine.Step()) {
                if (event.kind != EngineEventKind::kAction) {
                    continue;
                }
                ++run.actions;
                // Free text often starts with a space (" none"): forgive the spacing, judge the action.
                const std::string_view text = Trim(event.text);
                if (const std::optional<actions::Action> action = actions::Parse(text)) {
                    ++run.parsed;
                    const std::string verb(actions::VerbName(action->verb));
                    ++run.verbs[verb];
                    if (!run.examples.contains(verb)) {
                        run.examples[verb] = MarkChoices(event.text, event.forced);
                    }
                } else if (run.unparsed.size() < 8 && std::ranges::find(run.unparsed, text) == run.unparsed.end()) {
                    run.unparsed.emplace_back(text);
                }
            }
        }
    }
    run.sampled = engine.Stats().action_tokens;
    run.forced = engine.Stats().forced_tokens;
    return run;
}

nlohmann::json ToJson(const Run& run) {
    return {{"actions", run.actions},          {"parsed", run.parsed},          {"verbs", run.verbs},
            {"unparsed", run.unparsed},        {"sampled_tokens", run.sampled}, {"forced_tokens", run.forced},
            {"examples", run.examples}};
}

// One row: how many actions parse, and per action how many tokens the model
// chose (each one a step of the model, one after another) and how many the
// grammar wrote for it.
void Print(std::string_view name, const Run& run) {
    std::println("{:<26} {:>4}/{:<4} {:>6.1f}%  {:>12.2f} {:>14.2f}", name, run.parsed, run.actions,
                 100.0 * run.parsed / std::max(1, run.actions),
                 static_cast<double>(run.sampled) / std::max(1, run.actions),
                 static_cast<double>(run.forced) / std::max(1, run.actions));
}

void PrintMix(std::string_view name, const Run& run) {
    std::print("{:<26}", name);
    for (const auto& [verb, count] : run.verbs) {
        std::print("  {} {}", verb, count);
    }
    std::println("");
}

}  // namespace

// NPC actions under the market's grammar. Compiles the grammar (reports the
// automaton's size and how long each stage takes), then has all 32 NPCs
// answer eight requests three ways: held to the grammar with jump-forward,
// held to it without, and free (any short printable text after the same
// "ACTION:" lead-in). Reports how many actions parse, how many tokens per
// action the model chose (one model step each) and the grammar wrote, and
// the mix of actions.
int RunActionsBenchmark(Args args) {
    WeightFormat weights = WeightFormat::kFp8;
    std::string json_path;
    ArgParser parser("murmur-lab actions");
    parser.Choice("--weights", "bf16|fp8|int4", weights, ParseWeightFormat, "the model's weight format (default fp8)")
        .Text("--json", "<path>", json_path, "also write the results there");
    if (const std::optional<int> exit_code = parser.Parse(args)) {
        return *exit_code;
    }

    LoadedModel model = LoadModel(DefaultModelDirectory(), {}, weights);
    const TokenId end = ChatTemplate(model.tokenizer).EndOfTurn();
    const std::string text = actions::Grammar();
    Clock::time_point start = Clock::now();
    const ByteAutomaton automaton = ByteAutomaton::Compile(text);
    const double compile_ms = Milliseconds(start);
    start = Clock::now();
    const TokenGrammar grammar(automaton, model.tokenizer, end);
    const double lift_ms = Milliseconds(start);
    std::size_t allowed = 0;
    std::size_t forced_states = 0;
    for (int s = 0; s < grammar.StateCount(); ++s) {
        allowed += grammar.AllowedCount(s);
        forced_states += grammar.Forced(s).empty() ? 0 : 1;
    }
    std::println("grammar: {} bytes of text -> {} automaton states in {:.1f} ms -> token masks in {:.0f} ms",
                 text.size(), automaton.StateCount(), compile_ms, lift_ms);
    std::println("  allowed tokens per state: {:.1f} on average, {} at the start (of {}); {} of {} states fix their "
                 "next bytes\n",
                 static_cast<double>(allowed) / grammar.StateCount(), grammar.AllowedCount(grammar.Start()),
                 grammar.VocabSize(), forced_states, grammar.StateCount());
    const TokenGrammar free_text(ByteAutomaton::Compile(kFreeText), model.tokenizer, end);

    Act(*model.transformer, model.tokenizer, grammar, true);   // warm-up
    std::println("{:<26} {:>9} {:>7}  {:>12} {:>14}", "", "parsed", "", "model steps", "grammar wrote");
    const Run constrained = Act(*model.transformer, model.tokenizer, grammar, true);
    Print("grammar + jump-forward", constrained);
    const Run masked = Act(*model.transformer, model.tokenizer, grammar, false);
    Print("grammar, no jump-forward", masked);
    const Run free = Act(*model.transformer, model.tokenizer, free_text, true);
    Print("free text", free);
    std::println("(per action: tokens the model chose, one step each, and tokens the grammar wrote)\n");
    PrintMix("mix under the grammar:", constrained);
    PrintMix("mix of free text parsed:", free);
    std::println("\nwho wrote what ([the model's picks], the rest by the grammar):");
    for (const auto& [verb, example] : constrained.examples) {
        std::println("  {}", example);
    }
    std::println("\nfree text that doesn't parse, for example:");
    for (const std::string& sentence : free.unparsed) {
        std::println("  {}", sentence);
    }
    if (!json_path.empty()) {
        const nlohmann::json json = {{"weights", WeightFormatName(weights)},
                                     {"grammar", text},
                                     {"system_prompt_mei", personas::MarketSystemPrompt(personas::kMarketCast[0],
                                                                                         actions::Rules())},
                                     {"grammar_bytes", text.size()},
                                     {"automaton_states", automaton.StateCount()},
                                     {"compile_ms", compile_ms},
                                     {"token_masks_ms", lift_ms},
                                     {"start_allowed", grammar.AllowedCount(grammar.Start())},
                                     {"average_allowed", static_cast<double>(allowed) / grammar.StateCount()},
                                     {"forced_states", forced_states},
                                     {"vocab", grammar.VocabSize()},
                                     {"constrained", ToJson(constrained)},
                                     {"no_jump_forward", ToJson(masked)},
                                     {"free", ToJson(free)}};
        std::ofstream(json_path) << json.dump(1) << '\n';
    }
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
