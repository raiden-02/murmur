#include "engine/engine.hpp"
#include "engine/model_loader.hpp"
#include "engine/speculator.hpp"
#include "engine/transformer.hpp"
#include "model/chat_template.hpp"
#include "model/model_config.hpp"
#include "model/tokenizer.hpp"

#include "test_files.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace murmur {
namespace {

constexpr const char* kMei =
    "You are Mei, a noodle vendor in a neon-lit night market. Answer in one or two short sentences.";

SamplingParams Greedy() {
    return {.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .repetition_penalty = 1.0f};
}

// The main model with exact multi-row math (a pass over several rows then
// does the same arithmetic as one row at a time, so checking guesses can't
// change the text) and the draft model in INT4, its fastest format.
class SpeculativeTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!std::filesystem::exists(test::ModelDir() / "model.safetensors") ||
            !std::filesystem::exists(test::DraftModelDir() / "model.safetensors")) {
            return;
        }
        main_.emplace(LoadModel(test::ModelDir(), {.multi_row_math = MultiRowMath::kExactFp32}));
        speculator_ = LoadSpeculator(test::DraftModelDir(), WeightFormat::kInt4, main_->tokenizer);
    }
    static void TearDownTestSuite() {
        speculator_.reset();
        main_.reset();
    }
    void SetUp() override {
        if (!main_) {
            GTEST_SKIP() << "model or draft model not present";
        }
    }

    struct Reply {
        std::string text;
        std::string drafted_text;
        ReplyStats stats;
    };

    // Asks every question in turn in one conversation and collects the replies.
    static std::vector<Reply> Converse(ConversationOptions options, const std::vector<std::string>& questions,
                                       bool speculate, int draft_tokens = 4) {
        options.speculative = speculate;
        Engine engine(*main_->transformer, main_->tokenizer, {.draft_tokens = draft_tokens},
                      speculate ? speculator_.get() : nullptr);
        engine.Open(1, options);
        std::vector<Reply> replies;
        for (const std::string& question : questions) {
            engine.Send(1, question);
            Reply reply;
            for (int step = 0; step < 2000 && engine.HasWork(); ++step) {
                for (const EngineEvent& event : engine.Step()) {
                    if (event.kind == EngineEventKind::kText) {
                        reply.text += event.text;
                        if (event.drafted) {
                            reply.drafted_text += event.text;
                        }
                    } else {
                        reply.stats = event.stats;
                    }
                }
            }
            replies.push_back(reply);
        }
        engine.Close(1);
        return replies;
    }

    static inline std::optional<LoadedModel> main_;
    static inline std::unique_ptr<Speculator> speculator_;
};

TEST_F(SpeculativeTest, GreedyRepliesAreIdentical) {
    const std::vector<std::string> questions = {"What's good tonight?", "And what would you not recommend?"};
    const ConversationOptions mei{.system_prompt = kMei, .sampling = Greedy()};
    const std::vector<Reply> plain = Converse(mei, questions, false);
    const std::vector<Reply> speculative = Converse(mei, questions, true);
    for (std::size_t i = 0; i < questions.size(); ++i) {
        EXPECT_EQ(speculative[i].text, plain[i].text) << "turn " << i;
        EXPECT_EQ(speculative[i].stats.reply_tokens, plain[i].stats.reply_tokens);
        EXPECT_GT(speculative[i].stats.drafted_tokens, 0);
        EXPECT_GT(speculative[i].stats.accepted_tokens, 0);
        EXPECT_FALSE(speculative[i].drafted_text.empty());
    }
}

TEST_F(SpeculativeTest, SampledRepliesFollowTheSameDraws) {
    // Guesses are checked with the very draws decoding one token at a time
    // would use, so sampling gives the same text too, up to floating point:
    // a pass over several rows sums in another order than a pass over one
    // (logits differ by ~1e-4 relative), which flips a draw that lands on a
    // boundary between two tokens now and then. Misaligned draws would
    // instead change the text at the first guess of almost every reply.
    const std::vector<std::string> questions = {"Tell me about the market.", "Why is it so busy?",
                                                "Who runs the stall next to yours?"};
    int identical = 0;
    int replies = 0;
    for (const std::uint64_t seed : {1u, 7u, 42u}) {
        for (const std::string& question : questions) {
            SCOPED_TRACE("seed " + std::to_string(seed) + ": " + question);
            ConversationOptions mei{.system_prompt = kMei};
            mei.sampling.seed = seed;
            const std::string plain = Converse(mei, {question}, false)[0].text;
            const std::string speculative = Converse(mei, {question}, true, 6)[0].text;
            const auto [a, b] = std::ranges::mismatch(plain, speculative);
            EXPECT_GE(a - plain.begin(), 20) << plain << "\n" << speculative;
            identical += plain == speculative ? 1 : 0;
            ++replies;
        }
    }
    EXPECT_GE(3 * identical, 2 * replies);
}

TEST_F(SpeculativeTest, GuessesStopAtTheReplyLimit) {
    // Never more guesses than the reply has room for: a truncated reply ends
    // at the same token either way.
    const ConversationOptions mei{.system_prompt = kMei, .sampling = Greedy(), .max_reply_tokens = 7};
    const std::vector<Reply> plain = Converse(mei, {"Describe your noodles in detail."}, false);
    const std::vector<Reply> speculative = Converse(mei, {"Describe your noodles in detail."}, true, 8);
    EXPECT_EQ(speculative[0].text, plain[0].text);
    EXPECT_TRUE(speculative[0].stats.truncated);
    EXPECT_EQ(speculative[0].stats.reply_tokens, 7);
    // Every page came back: guesses that weren't kept were dropped from the cache.
    KvCacheManager& cache = main_->transformer->KvCache();
    while (cache.EvictCachedPage()) {
    }
    EXPECT_EQ(cache.FreePages(), cache.PageCount());
}

}  // namespace
}  // namespace murmur
