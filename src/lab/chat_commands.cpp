#include "lab/lab.hpp"

#include "engine/chat_session.hpp"
#include "engine/model_loader.hpp"
#include "game/personas.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <print>
#include <string>
#include <string_view>

namespace murmur::lab {

namespace {

constexpr int kMaxReplyTokens = 200;

LoadedModel LoadWithProgress() {
    const auto dir = DefaultModelDirectory();
    std::println("loading {} ...", dir.string());
    const auto start = std::chrono::steady_clock::now();
    LoadedModel model = LoadModel(dir);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::println("loaded {:.2f} GB of weights in {:.1f} s\n", model.transformer->WeightBytes() / 1e9, seconds);
    return model;
}

void PrintStreamed(std::string_view text) {
    std::print("{}", text);
    std::fflush(stdout);
}

// Decode reads every weight and the KV cache once per token, so its speed
// is bounded by those bytes / memory bandwidth; this reports how close it gets.
void PrintStats(const ReplyStats& stats) {
    std::println("\n\nprompt {} tokens, time to first token {:.1f} ms", stats.prompt_tokens, stats.prefill_ms);
    std::println("reply {} tokens at {:.1f} tokens/s = {:.0f} GB/s ({:.0f}% of {:.0f} GB/s peak)", stats.reply_tokens,
                 stats.DecodeTokensPerSecond(), stats.DecodeBandwidth() / 1e9, 100.0 * stats.DecodeFractionOfPeak(),
                 stats.peak_bandwidth / 1e9);
}

}  // namespace

// One question to the noodle vendor, answered greedily (reproducible) or with
// the model's sampling settings (--sample), followed by speed figures.
int RunGenerate(Args args) {
    bool sample = false;
    std::string message;
    ArgParser parser("murmur-lab generate");
    parser.Switch("--sample", sample, "sample like the game does, instead of greedy")
        .Positionals("<word>", "the question, in one or more words (default \"What's good tonight?\")",
                     [&message](std::string_view word) {
                         message += message.empty() ? "" : " ";
                         message += word;
                         return true;
                     });
    if (const std::optional<int> exit_code = parser.Parse(args)) {
        return *exit_code;
    }
    if (message.empty()) {
        message = "What's good tonight?";
    }

    LoadedModel model = LoadWithProgress();
    SamplingParams sampling;
    if (!sample) {
        sampling.temperature = 0.0f;
        sampling.repetition_penalty = 1.0f;
    }
    ChatSession session(*model.transformer, model.tokenizer, std::string(personas::kNoodleVendor), sampling,
                        kMaxReplyTokens);
    std::println("> {}", message);
    ReplyStats stats;
    session.Reply(message, PrintStreamed, {}, &stats);
    PrintStats(stats);
    return EXIT_SUCCESS;
}

// A conversation in the terminal; the session keeps earlier turns in the
// KV cache, so every reply knows what was said before.
int RunChat(Args args) {
    if (const std::optional<int> exit_code = ArgParser("murmur-lab chat").Parse(args)) {
        return *exit_code;
    }
    LoadedModel model = LoadWithProgress();
    SamplingParams sampling;
    sampling.seed = static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    ChatSession session(*model.transformer, model.tokenizer, std::string(personas::kNoodleVendor), sampling,
                        kMaxReplyTokens);
    std::println("You're at Mei's noodle stall. Say something (an empty line leaves).");
    std::string line;
    while (true) {
        std::print("\n> ");
        if (!std::getline(std::cin, line) || line.empty()) {
            break;
        }
        std::print("Mei: ");
        session.Reply(line, PrintStreamed);
        std::println("");
    }
    return EXIT_SUCCESS;
}

}  // namespace murmur::lab
