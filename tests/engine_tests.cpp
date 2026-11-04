#include "engine/engine.hpp"
#include "engine/model_weights.hpp"
#include "engine/transformer.hpp"
#include "model/safetensors.hpp"
#include "model/tokenizer.hpp"

#include "test_files.hpp"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace murmur {
namespace {

constexpr const char* kReferenceSystemPrompt =
    "You are Mei, a noodle vendor in a neon-lit night market. Answer in one or two short sentences.";

SamplingParams Greedy() {
    return {.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .repetition_penalty = 1.0f};
}

ConversationOptions GreedyConversation(std::string system_prompt = kReferenceSystemPrompt, int priority = 0) {
    return {.system_prompt = std::move(system_prompt), .sampling = Greedy(), .priority = priority};
}

// What the engine produced for each conversation.
struct Transcript {
    std::map<ConversationId, std::string> text;
    std::map<ConversationId, ReplyStats> stats;
    std::vector<ConversationId> finish_order;
    std::vector<ConversationId> first_token_order;
    int most_decode_rows = 0;
    int most_prefill_rows = 0;

    void Add(const std::vector<EngineEvent>& events) {
        for (const EngineEvent& event : events) {
            if (event.kind == EngineEventKind::kText) {
                if (!text.contains(event.conversation)) {
                    first_token_order.push_back(event.conversation);
                }
                text[event.conversation] += event.text;
            } else {
                stats[event.conversation] = event.stats;
                finish_order.push_back(event.conversation);
            }
        }
    }
};

void StepUntilIdle(Engine& engine, Transcript& transcript, int max_steps = 2000) {
    for (int i = 0; i < max_steps && engine.HasWork(); ++i) {
        transcript.Add(engine.Step());
        const EngineStats stats = engine.Stats();
        transcript.most_decode_rows = std::max(transcript.most_decode_rows, stats.last_decode_rows);
        transcript.most_prefill_rows = std::max(transcript.most_prefill_rows, stats.last_prefill_rows);
    }
    ASSERT_FALSE(engine.HasWork()) << "replies did not finish";
}

std::string ReferenceReply() {
    std::string expected = test::ReadJson(test::ReferenceDir() / "noodle_vendor.json").at("greedy_text");
    expected.resize(expected.size() - std::string("<|im_end|>").size());
    return expected;
}

// Loads the model once per suite; each suite picks its KV-cache size.
template <std::size_t kKvCacheBytes, int kMaxContext>
class ModelSuite : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const auto dir = test::ModelDir();
        if (!std::filesystem::exists(dir / "model.safetensors") || !std::filesystem::exists(test::ReferenceDir())) {
            return;
        }
        const ModelConfig config = ModelConfig::FromFile(dir / "config.json");
        const SafetensorsFile weights_file(dir / "model.safetensors");
        model_ = std::make_unique<Transformer>(config, ModelWeights::Upload(weights_file, config),
                                               TransformerOptions{.max_context = kMaxContext,
                                                                  .kv_cache_bytes = kKvCacheBytes,
                                                                  .multi_row_math = MultiRowMath::kExactFp32});
        tokenizer_.emplace(Tokenizer::FromFile(dir / "tokenizer.json"));
    }
    static void TearDownTestSuite() {
        model_.reset();
        tokenizer_.reset();
    }
    void SetUp() override {
        if (!model_) {
            GTEST_SKIP() << "model weights or reference dumps not present";
        }
    }
    void TearDown() override {
        if (model_) {
            // Every test must leave no page behind except what the prefix cache holds.
            KvCacheManager& cache = model_->KvCache();
            while (cache.EvictCachedPage()) {
            }
            EXPECT_EQ(cache.FreePages(), cache.PageCount());
            cache.CheckInvariants();
        }
    }

    static inline std::unique_ptr<Transformer> model_;
    static inline std::optional<Tokenizer> tokenizer_;
};

using EngineTest = ModelSuite<std::size_t{1} << 30, 4096>;

TEST_F(EngineTest, OneConversationReproducesTheReferenceReply) {
    Engine engine(*model_, *tokenizer_);
    engine.Open(7, GreedyConversation());
    engine.Send(7, "What's good tonight?");
    Transcript transcript;
    StepUntilIdle(engine, transcript);
    EXPECT_EQ(transcript.text[7], ReferenceReply());
    EXPECT_FALSE(transcript.stats[7].truncated);
    EXPECT_EQ(transcript.stats[7].prompt_tokens,
              static_cast<int>(test::ReadJson(test::ReferenceDir() / "noodle_vendor.json").at("input_ids").size()));
    engine.Close(7);
}

TEST_F(EngineTest, ConversationsJoinAndLeaveTheBatchAtAnyStep) {
    Engine engine(*model_, *tokenizer_);
    const char* questions[] = {"What's good tonight?", "Do you know the man at the pawn shop?",
                               "Is it always raining here?"};
    for (int id = 0; id < 3; ++id) {
        engine.Open(id, GreedyConversation());
    }
    engine.Send(0, questions[0]);
    Transcript transcript;
    for (int i = 0; i < 4; ++i) {
        transcript.Add(engine.Step());   // conversation 0 is decoding alone...
    }
    engine.Send(1, questions[1]);   // ...when the others join
    engine.Send(2, questions[2]);
    StepUntilIdle(engine, transcript);
    for (int id = 0; id < 3; ++id) {
        EXPECT_FALSE(transcript.text[id].empty()) << id;
        EXPECT_TRUE(transcript.stats.contains(id)) << id;
        engine.Close(id);
    }
    EXPECT_EQ(transcript.most_decode_rows, 3);   // all three were generating in the same steps
}

TEST_F(EngineTest, PromptsRunInChunksWhileOthersKeepDecoding) {
    Engine engine(*model_, *tokenizer_, {.max_prefill_tokens = 16});
    engine.Open(0, GreedyConversation());
    engine.Open(1, GreedyConversation(std::string(kReferenceSystemPrompt) +
                                      " Your stall has been in the market for thirty years, and you remember every "
                                      "storm, every blackout and every corporate raid it has survived."));
    engine.Send(0, "What's good tonight?");
    Transcript transcript;
    transcript.Add(engine.Step());
    transcript.Add(engine.Step());   // conversation 0's prompt is processed 16 tokens at a time
    engine.Send(1, "Tell me about the market.");
    StepUntilIdle(engine, transcript);
    EXPECT_LE(transcript.most_prefill_rows, 16);
    EXPECT_EQ(transcript.text[0], ReferenceReply());   // chunking doesn't change the result
    EXPECT_FALSE(transcript.text[1].empty());
    engine.Close(0);
    engine.Close(1);
}

TEST_F(EngineTest, TheHigherPriorityConversationGetsItsFirstTokenFirst) {
    Engine engine(*model_, *tokenizer_, {.max_prefill_tokens = 16});
    engine.Open(0, GreedyConversation());                            // an NPC
    engine.Open(1, GreedyConversation(kReferenceSystemPrompt, 1));   // the player
    engine.Send(0, "Tell me everything about the noodles you sell, one by one.");
    engine.Send(1, "Hi!");   // sent after, but goes first
    Transcript transcript;
    StepUntilIdle(engine, transcript);
    ASSERT_EQ(transcript.first_token_order.size(), 2u);
    EXPECT_EQ(transcript.first_token_order.front(), 1);
    engine.Close(0);
    engine.Close(1);
}

TEST_F(EngineTest, ASingleMessageCanCarryAHigherPriority) {
    Engine engine(*model_, *tokenizer_, {.max_prefill_tokens = 16});
    engine.Open(0, GreedyConversation());
    engine.Open(1, GreedyConversation());
    engine.Send(0, "Tell me everything about the noodles you sell, one by one.");
    engine.Send(1, "Hi!", 1);   // the player talking to an NPC whose conversation is ordinary
    Transcript transcript;
    StepUntilIdle(engine, transcript);
    ASSERT_EQ(transcript.first_token_order.size(), 2u);
    EXPECT_EQ(transcript.first_token_order.front(), 1);
    engine.Close(0);
    engine.Close(1);
}

TEST_F(EngineTest, ASharedSystemPromptIsComputedOnce) {
    Engine engine(*model_, *tokenizer_);
    engine.Open(0, GreedyConversation());
    engine.Open(1, GreedyConversation());
    engine.Send(0, "What's good tonight?");
    Transcript transcript;
    StepUntilIdle(engine, transcript);
    const std::int64_t before = engine.Stats().reused_tokens;
    engine.Send(1, "What's good tonight?");   // same system prompt and question
    StepUntilIdle(engine, transcript);
    EXPECT_GE(engine.Stats().reused_tokens - before, 2 * engine.Cache().PageSize());
    EXPECT_EQ(transcript.text[1], transcript.text[0]);   // same answer from the shared pages
    engine.Close(0);
    engine.Close(1);
}

TEST_F(EngineTest, AConversationForgetsItsOldestTurnsToStayWithinItsBudget) {
    Engine engine(*model_, *tokenizer_);
    ConversationOptions options = GreedyConversation();
    options.max_reply_tokens = 16;
    options.memory_tokens = 160;   // the system prompt is ~30 tokens; a turn is ~40
    engine.Open(0, options);
    Transcript transcript;
    bool forgot = false;
    std::string long_message = "Tell me";
    for (int i = 0; i < 70; ++i) {
        long_message += " very";
    }
    long_message += " much about it.";   // needs more than half of what the system prompt leaves
    for (const std::string& message : {std::string("Hi!"), std::string("What do you sell?"),
                                       std::string("How much is a bowl?"), std::string("Do you take credit?"),
                                       std::string("What time do you close?"), long_message,
                                       std::string("Any rumours tonight?")}) {
        engine.Send(0, message);
        StepUntilIdle(engine, transcript);
        forgot = forgot || transcript.stats[0].context_reset;
        EXPECT_GT(transcript.stats[0].reply_tokens, 0) << message;
        // What the reply started from, plus the reply, stays within the budget.
        EXPECT_LE(transcript.stats[0].prompt_tokens + options.max_reply_tokens, options.memory_tokens) << message;
    }
    EXPECT_TRUE(forgot);
    EXPECT_GT(engine.Stats().forgotten_turns, 0);
    EXPECT_THROW(engine.Send(0, std::string(2000, 'x')), std::length_error);
    engine.Close(0);
    // A system prompt that leaves no room for a reply is refused up front.
    ConversationOptions crowded = options;
    for (int i = 0; i < 200; ++i) {
        crowded.system_prompt += " noodle";   // one token each
    }
    EXPECT_THROW(engine.Open(1, crowded), std::invalid_argument);
}

TEST_F(EngineTest, SwitchingFormatsMidReplyKeepsEveryConversationGoing) {
    // Weights and KV cache change between two steps while two replies are in
    // progress: both finish, on the new weights and in the new cache. Back
    // on the original weights, a fresh conversation reproduces the reference.
    const auto dir = test::ModelDir();
    const ModelConfig config = ModelConfig::FromFile(dir / "config.json");
    const SafetensorsFile weights_file(dir / "model.safetensors");
    const std::size_t original_kv_bytes = model_->Options().kv_cache_bytes;
    Engine engine(*model_, *tokenizer_);
    engine.Open(1, GreedyConversation());
    engine.Open(2, GreedyConversation());
    engine.Send(1, "What's good tonight?");
    engine.Send(2, "Is it always raining here?");
    Transcript transcript;
    for (int i = 0; i < 6; ++i) {
        transcript.Add(engine.Step());
    }

    engine.Reconfigure(ModelWeights::Upload(weights_file, config, WeightFormat::kFp8), std::size_t{256} << 20,
                       KvFormat::kFp8);
    const EngineStats switched = engine.Stats();
    EXPECT_EQ(switched.weight_format, WeightFormat::kFp8);
    EXPECT_EQ(switched.kv_format, KvFormat::kFp8);
    EXPECT_EQ(switched.kv_bytes, std::size_t{256} << 20);
    EXPECT_EQ(switched.pages_free, static_cast<int>(switched.page_owner.size()));   // everyone was evicted
    StepUntilIdle(engine, transcript);
    EXPECT_EQ(transcript.finish_order.size(), 2u);
    EXPECT_FALSE(transcript.text[1].empty());
    EXPECT_FALSE(transcript.text[2].empty());
    engine.Close(1);
    engine.Close(2);

    engine.Reconfigure(ModelWeights::Upload(weights_file, config), original_kv_bytes, KvFormat::kBf16);
    engine.Open(3, GreedyConversation());
    engine.Send(3, "What's good tonight?");
    StepUntilIdle(engine, transcript);
    EXPECT_EQ(transcript.text[3], ReferenceReply());
    engine.Close(3);
}

// The X-ray draws its page map, batch strip and speculation hint from these.
TEST_F(EngineTest, StatsReportTheEnginesLimits) {
    const Engine engine(*model_, *tokenizer_, {.max_speculative_batch = 3});
    const EngineStats stats = engine.Stats();
    EXPECT_EQ(stats.page_tokens, model_->Options().page_size);
    EXPECT_EQ(stats.max_batch, model_->Options().max_batch);
    EXPECT_EQ(stats.max_speculative_batch, 3);
}

TEST_F(EngineTest, AnEngineGivesBackItsConversationsPagesWhenItGoes) {
    {
        Engine engine(*model_, *tokenizer_);
        for (int id = 0; id < 3; ++id) {
            engine.Open(id, GreedyConversation());
            engine.Send(id, "What's good tonight?");
        }
        Transcript transcript;
        StepUntilIdle(engine, transcript);
    }   // conversations still open
    // Every page is free again or only held by the prefix cache (reclaimable).
    EXPECT_EQ(model_->KvCache().FreePages() + model_->KvCache().CachedPages(), model_->KvCache().PageCount());
}

TEST_F(EngineTest, CancelEndsAReplyAndRejectsBadRequests) {
    Engine engine(*model_, *tokenizer_);
    engine.Open(0, GreedyConversation());
    engine.Send(0, "Tell me the whole history of the night market, in detail.");
    Transcript transcript;
    transcript.Add(engine.Step());
    transcript.Add(engine.Step());
    EXPECT_THROW(engine.Send(0, "Wait"), std::logic_error);   // still replying
    engine.Cancel(0);
    StepUntilIdle(engine, transcript);
    EXPECT_TRUE(transcript.stats[0].truncated);
    EXPECT_LE(transcript.stats[0].reply_tokens, 2);
    EXPECT_THROW(engine.Send(42, "Hello?"), std::invalid_argument);
    EXPECT_THROW(engine.Open(0, GreedyConversation()), std::invalid_argument);
    EXPECT_THROW(engine.Open(1, {.sampling = {.temperature = 0.7f, .top_k = 0}}), std::invalid_argument);
    engine.Close(0);
}

// A KV budget of 48 pages (768 tokens) for conversations of a few hundred
// tokens each: the cache can't hold them all, so idle conversations are
// evicted and recompute their history when they speak again.
using TightCacheEngineTest = ModelSuite<std::size_t{48} * 16 * 2 * 28 * 256 * 2, 512>;   // BF16 keys and values

TEST_F(TightCacheEngineTest, EvictsConversationsWhenTheCacheIsFullAndStillAnswersEveryone) {
    Engine engine(*model_, *tokenizer_);
    std::string story;
    for (int i = 0; i < 12; ++i) {
        story += " The market never sleeps, and neither do you.";
    }
    for (int id = 0; id < 4; ++id) {
        // Each prompt differs from its first token, so no pages are shared.
        ConversationOptions options =
            GreedyConversation("Vendor " + std::to_string(id) + " here. " + kReferenceSystemPrompt + story);
        options.max_reply_tokens = 24;
        engine.Open(id, options);
    }
    Transcript transcript;
    for (int round = 0; round < 2; ++round) {
        for (int id = 0; id < 4; ++id) {
            engine.Send(id, "What do you sell?");
            StepUntilIdle(engine, transcript);
        }
    }
    EXPECT_GT(engine.Stats().evictions, 0);
    for (int id = 0; id < 4; ++id) {
        EXPECT_FALSE(transcript.text[id].empty()) << id;
        engine.Close(id);
    }
}

// Everyone speaks again at once: each conversation takes back its history
// from the prefix cache as it is planned, so together they hold the whole
// cache before any has run its new message. The oldest must push the newer
// ones out (they recompute later), or nobody could go on.
TEST_F(TightCacheEngineTest, ConversationsThatFillTheCacheTogetherAllFinish) {
    Engine engine(*model_, *tokenizer_);
    std::string story;
    for (int i = 0; i < 12; ++i) {
        story += " The market never sleeps, and neither do you.";
    }
    for (int id = 0; id < 4; ++id) {
        ConversationOptions options =
            GreedyConversation("Vendor " + std::to_string(id) + " here. " + kReferenceSystemPrompt + story);
        options.max_reply_tokens = 24;
        engine.Open(id, options);
    }
    Transcript transcript;
    for (int id = 0; id < 4; ++id) {
        engine.Send(id, "What do you sell?");
        StepUntilIdle(engine, transcript);
    }
    for (int id = 0; id < 4; ++id) {
        engine.Send(id, "And what does it cost?");
    }
    StepUntilIdle(engine, transcript);
    EXPECT_EQ(transcript.finish_order.size(), 8u);
    EXPECT_GT(engine.Stats().evictions, 0);
    for (int id = 0; id < 4; ++id) {
        engine.Close(id);
    }
}

}  // namespace
}  // namespace murmur
