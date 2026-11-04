#include "engine/engine.hpp"
#include "engine/grammar.hpp"
#include "engine/model_loader.hpp"
#include "model/chat_template.hpp"
#include "model/tokenizer.hpp"

#include "test_files.hpp"
#include "tiny_tokenizer.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace murmur {
namespace {

// A small action language in the shape the market uses.
constexpr const char* kActions = R"g(# what an NPC can do
action  ::= "none" | walk | sell | hack
walk    ::= "walk(" place ")"
sell    ::= "sell(" item ", " price ")"
hack    ::= "hack(" place ", \"" message "\")"
place   ::= "DOCKS" | "RAMEN" | "NET CAFE"
item    ::= "noodles" | "data shard"
price   ::= [1-9] [0-9]{0,2}
message ::= [A-Z] [A-Z0-9 !?]{0,11}
)g";

bool Accepts(const ByteAutomaton& automaton, std::string_view text) {
    const int state = automaton.Walk(automaton.Start(), text);
    return state != ByteAutomaton::kDead && automaton.Accepting(state);
}

TEST(GrammarTest, CompilesARegularGrammarToAnAutomaton) {
    const ByteAutomaton actions = ByteAutomaton::Compile(kActions);
    for (const char* good : {"none", "walk(DOCKS)", "walk(NET CAFE)", "sell(noodles, 12)", "sell(data shard, 999)",
                             "hack(RAMEN, \"FREE CHIPS!\")", "hack(DOCKS, \"X\")"}) {
        EXPECT_TRUE(Accepts(actions, good)) << good;
    }
    for (const char* bad : {"", "nonee", "walk(PIER)", "walk(DOCKS", "sell(noodles, 0)", "sell(noodles, 1000)",
                            "hack(RAMEN, \"free\")", "hack(RAMEN, \"WAY TOO LONG MESSAGE\")", "WALK(DOCKS)"}) {
        EXPECT_FALSE(Accepts(actions, bad)) << bad;
    }
    // A prefix of a sentence is alive but not finished.
    const int partial = actions.Walk(actions.Start(), "sell(noo");
    EXPECT_NE(partial, ByteAutomaton::kDead);
    EXPECT_FALSE(actions.Accepting(partial));
}

TEST(GrammarTest, SupportsEveryRepetitionAndEscape) {
    const ByteAutomaton g = ByteAutomaton::Compile(R"g(s ::= "a"? "b"* "c"+ "d"{2} "e"{1,2} "\"" [x-z\]] "\n")g");
    EXPECT_TRUE(Accepts(g, "cdde\"x\n"));
    EXPECT_TRUE(Accepts(g, "abbbcccddee\"]\n"));
    EXPECT_FALSE(Accepts(g, "aacdde\"x\n"));
    EXPECT_FALSE(Accepts(g, "dde\"x\n"));
    EXPECT_FALSE(Accepts(g, "cddeee\"x\n"));
    EXPECT_FALSE(Accepts(g, "cdde\"w\n"));
}

TEST(GrammarTest, RejectsWhatItCannotCompile) {
    EXPECT_THROW(ByteAutomaton::Compile(""), std::invalid_argument);
    EXPECT_THROW(ByteAutomaton::Compile("a ::= b\nb ::= \"x\" a"), std::invalid_argument);   // recursive
    EXPECT_THROW(ByteAutomaton::Compile("a ::= missing"), std::invalid_argument);
    EXPECT_THROW(ByteAutomaton::Compile("a ::= \"unterminated"), std::invalid_argument);
    EXPECT_THROW(ByteAutomaton::Compile("a ::= (\"x\""), std::invalid_argument);
    EXPECT_THROW(ByteAutomaton::Compile("a = \"x\""), std::invalid_argument);
    EXPECT_THROW(ByteAutomaton::Compile("a ::= [z-a]"), std::invalid_argument);
    EXPECT_THROW(ByteAutomaton::Compile("a ::= \"x\"{3,1}"), std::invalid_argument);
    EXPECT_THROW(ByteAutomaton::Compile("a ::= \"x\"{2,"), std::invalid_argument);   // ends mid-repeat
    // Too big to be an action: nested repeats (the NFA) or a DFA that has
    // to remember the last 13 letters (2^13 states).
    EXPECT_THROW(ByteAutomaton::Compile("a ::= ((\"x\"{64}){64}){64}"), std::invalid_argument);
    EXPECT_THROW(ByteAutomaton::Compile("a ::= [ab]* \"a\" [ab]{12}"), std::invalid_argument);
    EXPECT_THROW(ByteAutomaton::Compile("a ::= \"x\"\na ::= \"y\""), std::invalid_argument);
}

// Token grammars are logic over a vocabulary, so these run on the tiny
// tokenizer (tiny_tokenizer.hpp) and need no download; ActingEngineTest
// below covers the model's own vocabulary.
class TokenGrammarTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const std::filesystem::path path = test::WriteTinyTokenizer();
        tokenizer_.emplace(Tokenizer::FromFile(path));
        std::filesystem::remove(path);
    }
    static void TearDownTestSuite() { tokenizer_.reset(); }
    static TokenId EndOfTurn() { return ChatTemplate(*tokenizer_).EndOfTurn(); }

    static inline std::optional<Tokenizer> tokenizer_;
};

TEST_F(TokenGrammarTest, EveryAllowedTokenKeepsTheTextInTheLanguage) {
    const ByteAutomaton automaton = ByteAutomaton::Compile(kActions);
    const TokenGrammar grammar(automaton, *tokenizer_, EndOfTurn());
    ASSERT_EQ(grammar.StateCount(), automaton.StateCount());
    // At the start the model may begin any action, in many tokenizations.
    EXPECT_GT(grammar.AllowedCount(grammar.Start()), 5u);
    EXPECT_EQ(grammar.Next(grammar.Start(), tokenizer_->Encode("walk", SpecialTokens::kAsText).front()) ==
                  ByteAutomaton::kDead,
              false);
    EXPECT_EQ(grammar.Next(grammar.Start(), tokenizer_->Encode("Hello", SpecialTokens::kAsText).front()),
              ByteAutomaton::kDead);
    // Writing a whole sentence token by token ends in a state where only the end token is left.
    int state = grammar.Start();
    for (const TokenId id : tokenizer_->Encode("hack(DOCKS, \"FREE NOODLES\")", SpecialTokens::kAsText)) {
        state = grammar.Next(state, id);
        ASSERT_NE(state, ByteAutomaton::kDead);
    }
    EXPECT_TRUE(grammar.Accepting(state));
    ASSERT_EQ(grammar.Forced(state).size(), 1u);
    EXPECT_EQ(grammar.Forced(state).front(), EndOfTurn());
    // Where the grammar fixes the next bytes, they come as tokens: after "sell(noodles", the ", ".
    const int after_item = automaton.Walk(automaton.Start(), "sell(noodles");
    std::string fixed;
    for (const TokenId id : grammar.Forced(after_item)) {
        fixed += tokenizer_->TokenBytes(id);
    }
    EXPECT_EQ(fixed, ", ");
    EXPECT_TRUE(grammar.Forced(grammar.Start()).empty()) << "the start is a choice";
    EXPECT_EQ(grammar.Next(state, EndOfTurn()), TokenGrammar::kFinished);
    // Special tokens never match by their spelling.
    EXPECT_EQ(grammar.Next(grammar.Start(), tokenizer_->SpecialTokenId("<|im_start|>")), ByteAutomaton::kDead);
}

// A fixed space before a choice is left to the model, which writes it
// merged with the word that follows (" walk" is one token).
TEST_F(TokenGrammarTest, AFixedSpaceIsLeftToMergeWithTheNextWord) {
    const TokenGrammar grammar(ByteAutomaton::Compile(R"g(s ::= " " ("walk" | "none"))g"), *tokenizer_, EndOfTurn());
    EXPECT_TRUE(grammar.Forced(grammar.Start()).empty());
    const TokenId walk = tokenizer_->Encode(" walk", SpecialTokens::kAsText).front();
    ASSERT_EQ(tokenizer_->TokenBytes(walk), " walk");
    EXPECT_NE(grammar.Next(grammar.Start(), walk), ByteAutomaton::kDead);
}

TEST_F(TokenGrammarTest, ALanguageWithOneSentenceIsWrittenWithoutChoices) {
    const TokenGrammar grammar(ByteAutomaton::Compile("s ::= \"walk(DOCKS)\""), *tokenizer_, EndOfTurn());
    int state = grammar.Start();
    std::string written;
    for (int steps = 0; steps < 20 && state != TokenGrammar::kFinished && !grammar.Forced(state).empty(); ++steps) {
        for (const TokenId id : grammar.Forced(state)) {
            state = grammar.Next(state, id);
            ASSERT_NE(state, ByteAutomaton::kDead);
            if (state == TokenGrammar::kFinished) {
                break;
            }
            written += tokenizer_->TokenBytes(id);
        }
    }
    EXPECT_EQ(state, TokenGrammar::kFinished);
    EXPECT_EQ(written, "walk(DOCKS)");
}

// End to end: NPC replies through the engine, every action parses.
class ActingEngineTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (std::filesystem::exists(test::ModelDir() / "model.safetensors")) {
            model_.emplace(LoadModel(test::ModelDir()));
        }
    }
    static void TearDownTestSuite() { model_.reset(); }
    void SetUp() override {
        if (!model_) {
            GTEST_SKIP() << "model not present";
        }
    }

    static inline std::optional<LoadedModel> model_;
};

// Has Mei answer four questions with an action each; checks every action
// parses and stays out of her words. Returns the engine's statistics.
EngineStats ActFourTimes(const LoadedModel& model, bool jump_forward) {
    const ByteAutomaton automaton = ByteAutomaton::Compile(kActions);
    const ChatTemplate chat(model.tokenizer);
    Engine engine(*model.transformer, model.tokenizer, {.jump_forward = jump_forward});
    engine.SetActionLanguage({.grammar = std::make_shared<TokenGrammar>(automaton, model.tokenizer, chat.EndOfTurn()),
                              .lead_in = model.tokenizer.Encode("\nACTION: ", SpecialTokens::kAsText)});
    const std::string system =
        "You are Mei, a noodle vendor in a neon-lit night market. Answer in one sentence. After you speak, the "
        "game asks for one action: none, walk(PLACE), sell(ITEM, PRICE) or hack(PLACE, \"MESSAGE\"). Places: "
        "DOCKS, RAMEN, NET CAFE. Items: noodles, data shard.";
    const char* questions[] = {"What's good tonight?", "Can I buy some noodles?", "Go check the docks for me.",
                               "Hack the sign across the street!"};
    int id = 0;
    for (const char* question : questions) {
        ConversationOptions options{.system_prompt = system, .max_reply_tokens = 60, .acts = true};
        options.sampling.seed = static_cast<std::uint64_t>(id) + 3;
        engine.Open(id, options);
        engine.Send(id, question);
        std::string text;
        std::vector<std::string> actions;
        ReplyStats stats;
        for (int step = 0; step < 500 && engine.HasWork(); ++step) {
            for (const EngineEvent& event : engine.Step()) {
                if (event.kind == EngineEventKind::kText) {
                    text += event.text;
                } else if (event.kind == EngineEventKind::kAction) {
                    actions.push_back(event.text);
                } else {
                    stats = event.stats;
                }
            }
        }
        SCOPED_TRACE(std::string(question) + " -> " + text);
        EXPECT_FALSE(engine.HasWork());
        EXPECT_EQ(actions.size(), 1u);
        EXPECT_TRUE(!actions.empty() && Accepts(automaton, actions.front())) << (actions.empty() ? "" : actions.front());
        EXPECT_EQ(text.find("ACTION"), std::string::npos) << "the action is not part of the words";
        EXPECT_GT(stats.action_tokens, 0);
        engine.Close(id++);
    }
    return engine.Stats();
}

TEST_F(ActingEngineTest, EveryReplyEndsWithAnActionThatParses) {
    const EngineStats stats = ActFourTimes(*model_, true);
    EXPECT_EQ(stats.actions, 4);
    EXPECT_GT(stats.forced_tokens, 0) << "punctuation like \")\" is written without the model";
}

TEST_F(ActingEngineTest, WithoutJumpForwardTheModelWritesEveryToken) {
    const EngineStats stats = ActFourTimes(*model_, false);
    EXPECT_EQ(stats.actions, 4);
    EXPECT_EQ(stats.forced_tokens, 0);
}

// A grammar may allow actions longer than the room a reply keeps for one;
// such an action is abandoned and the reply still ends (here the only
// sentence is ~100 tokens, all written by jump-forward).
TEST_F(ActingEngineTest, AnActionLongerThanItsRoomIsAbandoned) {
    std::string sentence = "say(";
    for (int i = 0; i < 50; ++i) {
        sentence += "lantern " + std::to_string(i) + " ";
    }
    const ByteAutomaton automaton = ByteAutomaton::Compile("a ::= \"" + sentence + ")\"");
    const ChatTemplate chat(model_->tokenizer);
    Engine engine(*model_->transformer, model_->tokenizer);
    engine.SetActionLanguage({.grammar = std::make_shared<TokenGrammar>(automaton, model_->tokenizer, chat.EndOfTurn()),
                              .lead_in = model_->tokenizer.Encode("\nACTION: ", SpecialTokens::kAsText)});
    engine.Open(0, {.system_prompt = "You are Mei.", .max_reply_tokens = 8, .acts = true});
    for (int reply = 0; reply < 2; ++reply) {   // and the conversation goes on
        engine.Send(0, "Say something.");
        int actions = 0;
        ReplyStats stats;
        for (int step = 0; step < 200 && engine.HasWork(); ++step) {
            for (const EngineEvent& event : engine.Step()) {
                actions += event.kind == EngineEventKind::kAction;
                if (event.kind == EngineEventKind::kReplyFinished) {
                    stats = event.stats;
                }
            }
        }
        EXPECT_FALSE(engine.HasWork());
        EXPECT_EQ(actions, 0);
        EXPECT_TRUE(stats.truncated);
    }
    EXPECT_EQ(engine.Stats().actions, 0);
    engine.Close(0);
}

}  // namespace
}  // namespace murmur
