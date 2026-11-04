#include "engine/inference_service.hpp"

#include "test_files.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace murmur {
namespace {

using namespace std::chrono_literals;

// Drains events until one of kind `until` arrives for `conversation` (that
// event is returned last) or the timeout passes. Mirrors what the game does
// once per frame.
std::vector<InferenceEvent> WaitFor(InferenceService& service, InferenceEventKind until,
                                    ConversationId conversation = kNoConversation,
                                    std::chrono::seconds timeout = 120s) {
    std::vector<InferenceEvent> seen;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        for (InferenceEvent& event : service.TakeEvents()) {
            const bool done = event.kind == until && event.conversation == conversation;
            seen.push_back(std::move(event));
            if (done) {
                return seen;
            }
        }
        std::this_thread::sleep_for(5ms);
    }
    ADD_FAILURE() << "timed out waiting for an event";
    return seen;
}

std::string TextOf(const std::vector<InferenceEvent>& events, ConversationId conversation) {
    std::string text;
    for (const InferenceEvent& event : events) {
        EXPECT_NE(event.kind, InferenceEventKind::kError) << event.text;
        if (event.kind == InferenceEventKind::kText && event.conversation == conversation) {
            text += event.text;
        }
    }
    return text;
}

// Exact multi-row arithmetic, so a greedy reply can be compared with the reference.
InferenceServiceOptions ExactOptions() {
    return {.model_dir = test::ModelDir(), .transformer = {.multi_row_math = MultiRowMath::kExactFp32}};
}

ConversationOptions GreedyMei() {
    return {.system_prompt = "You are Mei, a noodle vendor in a neon-lit night market. "
                             "Answer in one or two short sentences.",
            .sampling = {.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .repetition_penalty = 1.0f}};
}

TEST(InferenceService, ReportsAModelThatCannotBeLoaded) {
    InferenceService service({.model_dir = "no/such/model"});
    const std::vector<InferenceEvent> events = WaitFor(service, InferenceEventKind::kStopped);
    ASSERT_FALSE(events.empty());
    EXPECT_EQ(events.back().kind, InferenceEventKind::kStopped);
}

TEST(InferenceService, AnswersOnItsOwnThreadLikeTheReference) {
    MURMUR_SKIP_IF_MISSING(test::ModelDir() / "model.safetensors");
    MURMUR_SKIP_IF_MISSING(test::ReferenceDir() / "noodle_vendor.json");
    std::string expected = test::ReadJson(test::ReferenceDir() / "noodle_vendor.json").at("greedy_text");
    expected.resize(expected.size() - std::string("<|im_end|>").size());

    InferenceService service(ExactOptions());
    const ConversationId mei = service.Open(GreedyMei());   // usable before the model has loaded
    WaitFor(service, InferenceEventKind::kReady);
    service.Ask(mei, "What's good tonight?");
    EXPECT_EQ(TextOf(WaitFor(service, InferenceEventKind::kReplyFinished, mei), mei), expected);

    // Cancelling right after the first piece of text ends the reply early.
    service.Ask(mei, "Tell me the whole history of the night market, in detail.");
    std::vector<InferenceEvent> first = WaitFor(service, InferenceEventKind::kText, mei);
    service.Cancel(mei);
    const std::vector<InferenceEvent> rest = WaitFor(service, InferenceEventKind::kReplyFinished, mei);
    int text_events = 0;
    for (const auto& events : {first, rest}) {
        for (const InferenceEvent& event : events) {
            text_events += event.kind == InferenceEventKind::kText;
        }
    }
    EXPECT_LT(text_events, 20);

    // A second message while a reply runs is refused, and the running
    // reply still ends once, with its own text.
    service.Ask(mei, "Tell me about the market.");
    service.Ask(mei, "And another thing!");
    const std::vector<InferenceEvent> overlapping = WaitFor(service, InferenceEventKind::kReplyFinished, mei);
    int refused = 0;
    int texts = 0;
    for (const InferenceEvent& event : overlapping) {
        refused += event.kind == InferenceEventKind::kError;
        texts += event.kind == InferenceEventKind::kText;
    }
    EXPECT_EQ(refused, 1);
    EXPECT_GT(texts, 0) << "the refusal must not end the running reply";

    // A request the engine can't take is reported for its conversation.
    service.Ask(12345, "Anyone there?");
    const std::vector<InferenceEvent> errors = WaitFor(service, InferenceEventKind::kReplyFinished, 12345);
    EXPECT_EQ(errors.front().kind, InferenceEventKind::kError);
    service.Close(mei);
}

TEST(InferenceService, ConversationsReplyAtTheSameTime) {
    MURMUR_SKIP_IF_MISSING(test::ModelDir() / "model.safetensors");
    InferenceService service({.model_dir = test::ModelDir()});
    std::vector<ConversationId> ids;
    for (const char* name : {"Mei", "Kenji", "Aiko"}) {
        ConversationOptions options = GreedyMei();
        options.system_prompt = std::string("You are ") + name + ", a vendor in a neon-lit night market.";
        ids.push_back(service.Open(options));
    }
    WaitFor(service, InferenceEventKind::kReady);
    for (const ConversationId id : ids) {
        service.Ask(id, "Who are you?");
    }
    std::map<ConversationId, std::string> replies;
    std::map<ConversationId, bool> finished;
    const auto deadline = std::chrono::steady_clock::now() + 120s;
    int most_in_batch = 0;
    while (finished.size() < ids.size() && std::chrono::steady_clock::now() < deadline) {
        for (const InferenceEvent& event : service.TakeEvents()) {
            ASSERT_NE(event.kind, InferenceEventKind::kError) << event.text;
            if (event.kind == InferenceEventKind::kText) {
                replies[event.conversation] += event.text;
            } else if (event.kind == InferenceEventKind::kReplyFinished) {
                finished[event.conversation] = true;
            }
        }
        most_in_batch = std::max(most_in_batch, service.Stats().last_decode_rows);
        std::this_thread::sleep_for(1ms);
    }
    ASSERT_EQ(finished.size(), ids.size());
    for (const ConversationId id : ids) {
        EXPECT_FALSE(replies[id].empty());
    }
    EXPECT_EQ(most_in_batch, 3);
}

TEST(InferenceService, SwitchesFormatsWithinItsMemoryBudget) {
    // With a memory budget, smaller weights leave a bigger KV cache: INT4
    // weights (a quarter of BF16's bytes) leave room for many more pages.
    MURMUR_SKIP_IF_MISSING(test::ModelDir() / "model.safetensors");
    InferenceService service({.model_dir = test::ModelDir(), .memory_budget = std::size_t{3584} << 20});
    const ConversationId mei = service.Open(GreedyMei());
    WaitFor(service, InferenceEventKind::kReady);
    const EngineStats bf16 = service.Stats();
    EXPECT_EQ(bf16.weight_format, WeightFormat::kBf16);
    EXPECT_EQ(bf16.weight_bytes + bf16.kv_bytes, std::size_t{3584} << 20);

    service.Ask(mei, "What's good tonight?");
    service.SetFormats(WeightFormat::kInt4, KvFormat::kFp8);   // mid-reply
    WaitFor(service, InferenceEventKind::kReconfigured);
    const EngineStats int4 = service.Stats();
    EXPECT_EQ(int4.weight_format, WeightFormat::kInt4);
    EXPECT_EQ(int4.kv_format, KvFormat::kFp8);
    EXPECT_EQ(int4.weight_bytes + int4.kv_bytes, std::size_t{3584} << 20);
    EXPECT_GT(int4.page_owner.size(), 4 * bf16.page_owner.size());
    EXPECT_FALSE(TextOf(WaitFor(service, InferenceEventKind::kReplyFinished, mei), mei).empty());
    service.Close(mei);
}

}  // namespace
}  // namespace murmur
