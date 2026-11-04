#include "engine/chat_session.hpp"
#include "engine/transformer.hpp"
#include "model/safetensors.hpp"
#include "model/tokenizer.hpp"

#include "common/cuda_check.hpp"
#include "reference_comparer.hpp"
#include "test_files.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace murmur {
namespace {

using test::ReferenceComparer;
using test::RelativeError;

// Uploading 6 GB of weights takes seconds, so the suite shares one model.
class TransformerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const auto dir = test::ModelDir();
        if (!std::filesystem::exists(dir / "model.safetensors") || !std::filesystem::exists(test::ReferenceDir())) {
            return;
        }
        const ModelConfig config = ModelConfig::FromFile(dir / "config.json");
        const SafetensorsFile weights_file(dir / "model.safetensors");
        // The reference comparisons need exact arithmetic: FP32 cached keys and
        // values, exact multi-row products. The tensor-core paths have their
        // own, looser tests below.
        model_ = std::make_unique<Transformer>(config, ModelWeights::Upload(weights_file, config),
                                               TransformerOptions{.kv_format = KvFormat::kFp32,
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

    static std::vector<TokenId> PromptIds(const std::string& name) {
        return test::ReadJson(test::ReferenceDir() / (name + ".json")).at("input_ids").get<std::vector<TokenId>>();
    }

    static std::unique_ptr<Transformer> model_;
    static std::optional<Tokenizer> tokenizer_;
};

std::unique_ptr<Transformer> TransformerTest::model_;
std::optional<Tokenizer> TransformerTest::tokenizer_;

class TransformerReferenceTest : public TransformerTest, public ::testing::WithParamInterface<const char*> {
protected:
    void ExpectCloseToReference(MultiRowMath tensor_cores);
};

// Measured worst case is ~3.6e-5 (layer 0's attention mix for the chat
// prompt); FP32 sums in a different order than PyTorch's drift by about that
// much. 1e-3 leaves a ~28x margin while still catching any real bug, which
// shows up as errors of order 1.
constexpr double kCheckpointTolerance = 1e-3;
// Rounding activations to BF16 before each multi-row product drifts much
// more: the measured worst case is ~4.6e-2 (layer 26's output, chat
// prompt), from the few very large activations this model carries. 1e-1
// leaves a ~2x margin; the predicted next token must not change at all.
constexpr double kTensorCoreTolerance = 1e-1;

TEST_P(TransformerReferenceTest, EveryCheckpointMatchesTheReference) {
    const std::string name = GetParam();
    const SafetensorsFile reference(test::ReferenceDir() / (name + ".safetensors"));
    ReferenceComparer comparer(reference);
    model_->Forward(PromptIds(name), 0, LogitsRows::kAll, &comparer);

    // Fused kernels skip some of the reference's tensors; these are always
    // reported: the embedding, logits, and per layer q, k, v, the attention
    // output, the MLP activation and the layer output.
    const std::size_t per_layer = 6;
    ASSERT_EQ(comparer.Errors().size(), 2 + per_layer * model_->Config().num_layers);
    comparer.RecordErrors("fp32.");
    for (const auto& error : comparer.Errors()) {
        EXPECT_LT(error.value, kCheckpointTolerance) << error.name;
    }
}

TEST_P(TransformerReferenceTest, GreedyDecodingReproducesTheReference) {
    const std::string name = GetParam();
    const auto expected = test::ReadJson(test::ReferenceDir() / (name + ".json"))
                              .at("greedy_ids").get<std::vector<TokenId>>();
    const std::vector<TokenId> prompt = PromptIds(name);
    model_->Forward(prompt, 0);
    int position = static_cast<int>(prompt.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const TokenId next = model_->ArgmaxLastLogits(tokenizer_->VocabSize());
        ASSERT_EQ(next, expected[i]) << "first difference at generated token " << i;
        model_->Forward(std::span(&next, 1), position++);
    }
}

INSTANTIATE_TEST_SUITE_P(Prompts, TransformerReferenceTest, ::testing::Values("plain", "noodle_vendor"));

// Switches the shared model to a tensor-core multi-row math for one test.
class TensorCoreMath {
public:
    TensorCoreMath(Transformer& model, MultiRowMath math) : model_(model) { model_.SetMultiRowMath(math); }
    ~TensorCoreMath() { model_.SetMultiRowMath(MultiRowMath::kExactFp32); }
    TensorCoreMath(const TensorCoreMath&) = delete;
    TensorCoreMath& operator=(const TensorCoreMath&) = delete;

private:
    Transformer& model_;
};

TEST_P(TransformerReferenceTest, TensorCorePathsStayCloseToTheReference) {
    // Prompt rows multiplied on tensor cores round every activation to BF16
    // (cuBLAS) or FP16 (GemmF16) before each matrix product: the logits drift
    // from the FP32 reference, but stay close, and the predicted next token
    // is unchanged.
    for (const MultiRowMath tensor_cores : {MultiRowMath::kCublasBf16, MultiRowMath::kTensorCores}) {
        SCOPED_TRACE(tensor_cores == MultiRowMath::kCublasBf16 ? "cuBLAS BF16" : "GemmF16");
        ExpectCloseToReference(tensor_cores);
    }
}

void TransformerReferenceTest::ExpectCloseToReference(MultiRowMath tensor_cores) {
    const TensorCoreMath math(*model_, tensor_cores);
    const std::string name = GetParam();
    const SafetensorsFile reference(test::ReferenceDir() / (name + ".safetensors"));
    ReferenceComparer comparer(reference);
    model_->Forward(PromptIds(name), 0, LogitsRows::kAll, &comparer);
    double worst = 0.0;
    std::string worst_name;
    for (const auto& error : comparer.Errors()) {
        if (error.value > worst) {
            worst = error.value;
            worst_name = error.name;
        }
    }
    EXPECT_LT(worst, kTensorCoreTolerance) << worst_name;
    comparer.RecordErrors(tensor_cores == MultiRowMath::kCublasBf16 ? "bf16." : "f16.");
    const auto expected = test::ReadJson(test::ReferenceDir() / (name + ".json")).at("greedy_ids").get<std::vector<TokenId>>();
    model_->Forward(PromptIds(name), 0);
    EXPECT_EQ(model_->ArgmaxLastLogits(tokenizer_->VocabSize()), expected.front());
}

TEST_F(TransformerTest, PrefillInPiecesMatchesOnePass) {
    // Feeding the prompt in two calls makes the second call read the first
    // call's keys and values from the cache: the same contract decode uses.
    const std::vector<TokenId> prompt = PromptIds("noodle_vendor");
    const int vocab = model_->Config().vocab_size;
    std::vector<float> one_pass(vocab);
    model_->Forward(prompt, 0);
    model_->CopyLastLogits(one_pass);

    constexpr std::size_t kSplit = 17;
    std::vector<float> two_passes(vocab);
    model_->Forward(std::span(prompt).first(kSplit), 0);
    model_->Forward(std::span(prompt).subspan(kSplit), kSplit);
    model_->CopyLastLogits(two_passes);
    EXPECT_LT(RelativeError(two_passes, one_pass), 1e-4);
}

TEST_F(TransformerTest, GraphReplayMatchesLaunchingEachKernel) {
    // A decode step normally replays a recorded CUDA graph; asking for every
    // row's logits takes the directly launched path. Same kernels, same
    // inputs, so the logits must be bit-for-bit identical.
    const std::vector<TokenId> prompt = PromptIds("noodle_vendor");
    const int vocab = model_->Config().vocab_size;
    model_->Forward(prompt, 0);
    const int position = static_cast<int>(prompt.size());
    for (const TokenId token : {TokenId{1}, TokenId{9707}, TokenId{151644}}) {
        std::vector<float> replayed(vocab);
        std::vector<float> launched(vocab);
        model_->Forward(std::span(&token, 1), position, LogitsRows::kLast);
        model_->CopyLastLogits(replayed);
        model_->Forward(std::span(&token, 1), position, LogitsRows::kAll);
        model_->CopyLastLogits(launched);
        EXPECT_EQ(replayed, launched) << "token " << token;
    }
}

TEST_F(TransformerTest, RejectsInvalidInput) {
    const int vocab = model_->Config().vocab_size;
    const std::vector<TokenId> bad_id = {1, vocab};
    EXPECT_THROW(model_->Forward(bad_id, 0), std::invalid_argument);
    EXPECT_THROW(model_->Forward({}, 0), std::invalid_argument);
    const std::vector<TokenId> one = {1};
    EXPECT_THROW(model_->Forward(one, model_->Options().max_context), std::invalid_argument);
    EXPECT_THROW(model_->Forward(one, -1), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(model_->ArgmaxLastLogits(vocab + 1)), std::invalid_argument);
}

SamplingParams PureGreedy() {
    return {.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .repetition_penalty = 1.0f};
}

// The reference's system prompt, so a greedy session must reproduce its reply.
constexpr const char* kReferenceSystemPrompt =
    "You are Mei, a noodle vendor in a neon-lit night market. Answer in one or two short sentences.";

TEST_F(TransformerTest, ChatSessionReproducesTheReferenceReply) {
    const nlohmann::json reference = test::ReadJson(test::ReferenceDir() / "noodle_vendor.json");
    std::string expected = reference.at("greedy_text").get<std::string>();
    const std::string end_of_turn = "<|im_end|>";
    ASSERT_TRUE(expected.ends_with(end_of_turn));
    expected.resize(expected.size() - end_of_turn.size());

    ChatSession session(*model_, *tokenizer_, kReferenceSystemPrompt, PureGreedy());
    std::string streamed;
    ReplyStats stats;
    const std::string reply =
        session.Reply("What's good tonight?", [&](std::string_view text) { streamed += text; }, {}, &stats);
    EXPECT_EQ(reply, expected);
    EXPECT_EQ(streamed, reply);
    EXPECT_FALSE(stats.truncated);
    EXPECT_EQ(stats.prompt_tokens, static_cast<int>(reference.at("input_ids").size()));
    EXPECT_EQ(session.TokensInContext(), stats.prompt_tokens + stats.reply_tokens);
}

TEST_F(TransformerTest, ChatSessionOnlyPrefillsTheNewTurn) {
    ChatSession session(*model_, *tokenizer_, kReferenceSystemPrompt, PureGreedy(), 16);
    session.Reply("Hi!", [](std::string_view) {});
    const int before = session.TokensInContext();

    const ChatTemplate chat(*tokenizer_);
    const std::size_t new_turn = 1 + chat.EncodeAfterReply().size() +   // closing the previous reply
                                 chat.EncodeMessage({ChatRole::kUser, "And to drink?"}).size() +
                                 chat.EncodeGenerationPrompt().size();
    ReplyStats stats;
    session.Reply("And to drink?", [](std::string_view) {}, {}, &stats);
    EXPECT_EQ(stats.prompt_tokens, static_cast<int>(new_turn));
    EXPECT_EQ(session.TokensInContext(), before + stats.prompt_tokens + stats.reply_tokens);
}

TEST_F(TransformerTest, ChatSessionStopsAtTheLengthLimitAndOnRequest) {
    ChatSession session(*model_, *tokenizer_, kReferenceSystemPrompt, PureGreedy(), 3);
    ReplyStats stats;
    session.Reply("Tell me a long story.", [](std::string_view) {}, {}, &stats);
    EXPECT_TRUE(stats.truncated);
    EXPECT_EQ(stats.reply_tokens, 3);

    std::stop_source stop;
    stop.request_stop();
    const std::string reply = session.Reply("Go on.", [](std::string_view) {}, stop.get_token(), &stats);
    EXPECT_TRUE(reply.empty());
    EXPECT_TRUE(stats.truncated);
}

TEST_F(TransformerTest, ChatSessionRestartsWhenTheContextWouldOverflow) {
    // Size the reply budget so the first turn fills the context window
    // exactly: the second message then can't fit, and the session must drop
    // the first turn and start again from the system prompt.
    const ChatTemplate chat(*tokenizer_);
    const std::vector<ChatMessage> first_turn = {{ChatRole::kSystem, kReferenceSystemPrompt},
                                                 {ChatRole::kUser, "What's good tonight?"}};
    const int first_prompt = static_cast<int>(chat.EncodeConversation(first_turn, true).size());
    const int max_reply = model_->Options().max_context - first_prompt;
    ChatSession session(*model_, *tokenizer_, kReferenceSystemPrompt, PureGreedy(), max_reply);
    ReplyStats stats;
    session.Reply("What's good tonight?", [](std::string_view) {}, {}, &stats);
    EXPECT_FALSE(stats.context_reset);
    const std::string reply = session.Reply("What's good tonight?", [](std::string_view) {}, {}, &stats);
    EXPECT_TRUE(stats.context_reset);
    EXPECT_FALSE(reply.empty());
}

// Batched steps add up partial sums in a different order than a single
// sequence's (a matrix product instead of a matrix-vector product), so they
// agree to rounding, not bit for bit.
constexpr double kBatchTolerance = 2e-4;

std::vector<float> RowLogits(Transformer& model, int row) {
    std::vector<float> logits(model.Config().vocab_size);
    model.CopyLogits(row, logits);
    return logits;
}

// The next-token logits after `prompt`, then after each of `continuation`,
// computed on the transformer's private sequence one at a time.
std::vector<std::vector<float>> SoloLogits(Transformer& model, const std::vector<TokenId>& prompt,
                                           const std::vector<TokenId>& continuation) {
    std::vector<std::vector<float>> logits;
    model.Forward(prompt, 0);
    logits.push_back(RowLogits(model, 0));
    int position = static_cast<int>(prompt.size());
    for (const TokenId token : continuation) {
        model.Forward(std::span(&token, 1), position++);
        logits.push_back(RowLogits(model, 0));
    }
    return logits;
}

TEST_F(TransformerTest, GraphsRecordedAheadMatchGraphsRecordedOnDemand) {
    // RecordDecodeGraphs records each batch size from its shape alone, before
    // any step of that size has been staged: the same decode step must give
    // bit-for-bit the logits of a graph recorded from the real step.
    const std::vector<TokenId> noodle = PromptIds("noodle_vendor");
    const std::vector<std::vector<TokenId>> prompts = {PromptIds("plain"), noodle};
    KvCacheManager& kv = model_->KvCache();
    std::vector<KvSequence> sequences(prompts.size());
    std::vector<StepEntry> entries;
    for (std::size_t s = 0; s < prompts.size(); ++s) {
        ASSERT_TRUE(kv.Reserve(sequences[s], static_cast<int>(prompts[s].size()) + 1));
        entries.push_back({.sequence = &sequences[s], .tokens = prompts[s]});
    }
    model_->Step(entries);
    const TokenId next[] = {TokenId{9707}, TokenId{1}};
    for (std::size_t s = 0; s < prompts.size(); ++s) {
        sequences[s].length += static_cast<int>(prompts[s].size());
        entries[s].tokens = std::span(&next[s], 1);
    }
    const auto decode = [&] {
        model_->Step(entries);
        return std::vector{RowLogits(*model_, 0), RowLogits(*model_, 1)};
    };
    model_->SetMultiRowMath(MultiRowMath::kExactFp32);   // drops every recorded graph
    const auto on_demand = decode();
    model_->SetMultiRowMath(MultiRowMath::kExactFp32);
    model_->RecordDecodeGraphs();
    EXPECT_EQ(decode(), on_demand);
    for (KvSequence& sequence : sequences) {
        kv.Release(sequence);
    }
    kv.CheckInvariants();
}

TEST_F(TransformerTest, BatchedStepsMatchEachSequenceRunAlone) {
    // Three prompts prefilled in one step, then decoded together for a few
    // greedy tokens: every sequence must get the logits it gets alone.
    const std::vector<TokenId> noodle = PromptIds("noodle_vendor");
    const std::vector<std::vector<TokenId>> prompts = {
        PromptIds("plain"), noodle, std::vector<TokenId>(noodle.begin(), noodle.begin() + 20)};
    const int vocab = tokenizer_->VocabSize();
    KvCacheManager& kv = model_->KvCache();
    std::vector<KvSequence> sequences(prompts.size());
    std::vector<StepEntry> entries;
    for (std::size_t s = 0; s < prompts.size(); ++s) {
        ASSERT_TRUE(kv.Reserve(sequences[s], static_cast<int>(prompts[s].size()) + 8));
        entries.push_back({.sequence = &sequences[s], .tokens = prompts[s]});
    }
    std::vector<std::vector<TokenId>> generated(prompts.size());
    std::vector<std::vector<std::vector<float>>> batched(prompts.size());
    model_->Step(entries);
    for (int step = 0; step < 5; ++step) {
        for (std::size_t s = 0; s < prompts.size(); ++s) {
            sequences[s].length += static_cast<int>(entries[s].tokens.size());
            batched[s].push_back(RowLogits(*model_, static_cast<int>(s)));
            generated[s].push_back(model_->ArgmaxLogits(static_cast<int>(s), vocab));
            entries[s].tokens = std::span(&generated[s].back(), 1);
        }
        if (step < 4) {
            model_->Step(entries);   // a decode step for all three: replayed from a recorded graph
        }
    }

    for (std::size_t s = 0; s < prompts.size(); ++s) {
        std::vector<TokenId> fed(generated[s].begin(), generated[s].end() - 1);
        const auto solo = SoloLogits(*model_, prompts[s], fed);
        for (std::size_t step = 0; step < solo.size(); ++step) {
            SCOPED_TRACE("sequence " + std::to_string(s) + ", step " + std::to_string(step));
            EXPECT_LT(RelativeError(batched[s][step], solo[step]), kBatchTolerance);
        }
        kv.Release(sequences[s]);
    }
    kv.CheckInvariants();
}

TEST_F(TransformerTest, DecodeAndPromptChunksMixInOneStep) {
    // While one sequence decodes, another's prompt runs in two chunks in the
    // same steps (chunked prefill): both must be unaffected by the other.
    const std::vector<TokenId> a_prompt = PromptIds("plain");
    const std::vector<TokenId> b_prompt = PromptIds("noodle_vendor");
    KvCacheManager& kv = model_->KvCache();
    KvSequence a;
    KvSequence b;
    ASSERT_TRUE(kv.Reserve(a, static_cast<int>(a_prompt.size()) + 2));
    ASSERT_TRUE(kv.Reserve(b, static_cast<int>(b_prompt.size())));
    StepEntry first{.sequence = &a, .tokens = a_prompt};
    model_->Step(std::span(&first, 1));
    a.length = static_cast<int>(a_prompt.size());
    const TokenId a_next = model_->ArgmaxLogits(0, tokenizer_->VocabSize());

    const std::size_t half = b_prompt.size() / 2;
    const StepEntry mixed1[] = {{.sequence = &b, .tokens = std::span(b_prompt).first(half), .logits = 0},
                                {.sequence = &a, .tokens = std::span(&a_next, 1)}};
    model_->Step(mixed1);
    const std::vector<float> a_logits = RowLogits(*model_, 0);
    a.length += 1;
    b.length = static_cast<int>(half);
    const StepEntry mixed2[] = {{.sequence = &b, .tokens = std::span(b_prompt).subspan(half)}};
    model_->Step(mixed2);
    const std::vector<float> b_logits = RowLogits(*model_, 0);

    EXPECT_LT(RelativeError(a_logits, SoloLogits(*model_, a_prompt, {a_next}).back()), kBatchTolerance);
    EXPECT_LT(RelativeError(b_logits, SoloLogits(*model_, b_prompt, {}).back()), kBatchTolerance);
    kv.Release(a);
    kv.Release(b);
}

TEST_F(TransformerTest, ASharedPrefixGivesTheSameResultAsComputingIt) {
    // The first sequence publishes its prompt's full pages; the second starts
    // on those pages and only computes the rest of its own, longer prompt.
    const std::vector<TokenId> prompt = PromptIds("noodle_vendor");
    std::vector<TokenId> longer = prompt;
    longer.insert(longer.end(), {9707, 11, 1246});
    KvCacheManager& kv = model_->KvCache();
    KvSequence first;
    ASSERT_TRUE(kv.Reserve(first, static_cast<int>(prompt.size())));
    StepEntry entry{.sequence = &first, .tokens = prompt};
    model_->Step(std::span(&entry, 1));
    first.length = static_cast<int>(prompt.size());
    kv.Publish(first, prompt);

    KvSequence second;
    const int reused = kv.AttachPrefix(second, longer);
    EXPECT_EQ(reused, static_cast<int>(prompt.size()) / kv.PageSize() * kv.PageSize());
    EXPECT_EQ(second.pages.front(), first.pages.front());
    ASSERT_TRUE(kv.Reserve(second, static_cast<int>(longer.size())));
    entry = {.sequence = &second, .tokens = std::span(longer).subspan(reused)};
    model_->Step(std::span(&entry, 1));
    const std::vector<float> shared = RowLogits(*model_, 0);
    EXPECT_LT(RelativeError(shared, SoloLogits(*model_, longer, {}).back()), kBatchTolerance);
    kv.Release(first);
    kv.Release(second);
}

TEST_F(TransformerTest, StepRejectsInvalidBatches) {
    KvCacheManager& kv = model_->KvCache();
    KvSequence sequence;
    ASSERT_TRUE(kv.Reserve(sequence, 4));
    const std::vector<TokenId> tokens = {1, 2, 3, 4};
    const std::vector<TokenId> many(model_->Options().max_chunk + 1, 1);
    KvSequence big;
    ASSERT_TRUE(kv.Reserve(big, static_cast<int>(many.size())));
    const StepEntry entry{.sequence = &sequence, .tokens = tokens};
    const StepEntry twice[] = {entry, entry};   // one sequence listed twice
    EXPECT_THROW(model_->Step(twice), std::invalid_argument);
    EXPECT_THROW(model_->Step({}), std::invalid_argument);
    const StepEntry too_many_rows{.sequence = &big, .tokens = many};
    EXPECT_THROW(model_->Step(std::span(&too_many_rows, 1)), std::invalid_argument);
    KvSequence no_pages;
    const StepEntry without_pages{.sequence = &no_pages, .tokens = std::span(tokens).first(1)};
    EXPECT_THROW(model_->Step(std::span(&without_pages, 1)), std::invalid_argument);
    kv.Release(sequence);
    kv.Release(big);
}

}  // namespace
}  // namespace murmur
