#include "engine/grammar.hpp"
#include "game/action_language.hpp"
#include "model/chat_template.hpp"
#include "model/tokenizer.hpp"

#include "test_files.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>

namespace murmur {
namespace {

using actions::Verb;

// One sentence per verb: the grammar accepts it (after the space every
// action starts with) and Parse reads back every field, with or without
// that space.
TEST(ActionLanguageTest, ParsesEverySentenceOfTheGrammar) {
    const ByteAutomaton grammar = ByteAutomaton::Compile(actions::Grammar());
    struct Case {
        const char* sentence;
        actions::Action expected;
    };
    const Case cases[] = {
        {"none", {.verb = Verb::kNone}},
        {"walk(NET CAFE)", {.verb = Verb::kWalk, .place = "NET CAFE"}},
        {"give(tea)", {.verb = Verb::kGive, .item = "tea"}},
        {"sell(data shard, 40)", {.verb = Verb::kSell, .item = "data shard", .price = 40}},
        {"hack(CLUB, \"FREE ENTRY!\")", {.verb = Verb::kHack, .place = "CLUB", .message = "FREE ENTRY!"}},
        {"quest(steamed buns, Old Lin)", {.verb = Verb::kQuest, .item = "steamed buns", .npc = "Old Lin"}},
        {"call(Hana)", {.verb = Verb::kCall, .npc = "Hana"}},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.sentence);
        const std::string written = std::string(" ") + c.sentence;   // as the grammar writes it
        const int state = grammar.Walk(grammar.Start(), written);
        ASSERT_NE(state, ByteAutomaton::kDead);
        EXPECT_TRUE(grammar.Accepting(state));
        for (const std::string& sentence : {std::string(c.sentence), written}) {
            const std::optional<actions::Action> action = actions::Parse(sentence);
            ASSERT_TRUE(action.has_value());
            EXPECT_EQ(action->verb, c.expected.verb);
            EXPECT_EQ(action->place, c.expected.place);
            EXPECT_EQ(action->item, c.expected.item);
            EXPECT_EQ(action->npc, c.expected.npc);
            EXPECT_EQ(action->message, c.expected.message);
            EXPECT_EQ(action->price, c.expected.price);
        }
    }
}

// Every verb's name is the one the grammar spells.
TEST(ActionLanguageTest, NamesEveryVerbAsTheGrammarSpellsIt) {
    const std::string grammar = actions::Grammar();
    for (std::size_t verb = 0; verb < actions::kVerbCount; ++verb) {
        const std::string name(actions::VerbName(static_cast<Verb>(verb)));
        EXPECT_NE(grammar.find("\"" + name), std::string::npos) << name;
    }
    EXPECT_EQ(actions::VerbName(Verb::kQuest), "quest");
}

// The parser is the measure of an unconstrained model's actions, so it has
// to reject exactly what the grammar rejects.
TEST(ActionLanguageTest, RejectsWhatTheGrammarRejects) {
    const ByteAutomaton grammar = ByteAutomaton::Compile(actions::Grammar());
    for (const char* sentence : {"", "None", "walk(the docks)", "walk(DOCKS) ", "give(sword)", "sell(tea)",
                                 "sell(tea, 0)", "sell(tea, 1000)", "sell(tea, 05)", "hack(CLUB, FREE)",
                                 "hack(CLUB, \"free\")", "hack(CLUB, \" FREE\")", "hack(CLUB, \"THIS MESSAGE IS TOO LONG\")",
                                 "quest(tea, Nobody)", "call(Mei", "ACTION: none", "walk(DOCKS)\n"}) {
        SCOPED_TRACE(sentence);
        const int state = grammar.Walk(grammar.Start(), std::string(" ") + sentence);
        EXPECT_FALSE(state != ByteAutomaton::kDead && grammar.Accepting(state));
        EXPECT_FALSE(actions::Parse(sentence).has_value());
    }
}

TEST(ActionLanguageTest, TellsTheModelEveryPlaceItemAndName) {
    const std::string rules = actions::Rules();
    for (const std::string_view place : actions::kPlaces) {
        EXPECT_NE(rules.find(place), std::string::npos) << place;
    }
    for (const std::string_view item : actions::kItems) {
        EXPECT_NE(rules.find(item), std::string::npos) << item;
    }
    EXPECT_NE(rules.find("Old Lin"), std::string::npos);
}

// The game builds the token masks while the model loads, so they must be
// quick. The time is always recorded; it passes or fails the test only with
// MURMUR_TIMING_TESTS=1 (see test_files.hpp).
TEST(ActionLanguageTest, LiftsToTokensQuickly) {
    MURMUR_SKIP_IF_MISSING(test::ModelDir() / "tokenizer.json");
    const Tokenizer tokenizer = Tokenizer::FromFile(test::ModelDir() / "tokenizer.json");
    const ByteAutomaton automaton = ByteAutomaton::Compile(actions::Grammar());
    const auto start = std::chrono::steady_clock::now();
    const TokenGrammar grammar(automaton, tokenizer, ChatTemplate(tokenizer).EndOfTurn());
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    EXPECT_EQ(grammar.StateCount(), automaton.StateCount());
    RecordProperty("lift_ms", static_cast<int>(seconds * 1000.0));
    if (test::TimingChecksEnabled()) {
        EXPECT_LT(seconds, 2.0) << grammar.StateCount() << " states";
    }
}

}  // namespace
}  // namespace murmur
