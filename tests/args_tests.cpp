#include "lab/args.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace murmur::lab {
namespace {

enum class Color { kRed, kBlue };

std::optional<Color> ParseColor(std::string_view name) {
    if (name == "red") {
        return Color::kRed;
    }
    if (name == "blue") {
        return Color::kBlue;
    }
    return std::nullopt;
}

// One of every kind of flag, with the defaults a tool would start from.
struct Flags {
    bool verbose = false;
    std::string path;
    int count = 1;
    double seconds = 2.0;
    Color color = Color::kRed;
    std::vector<std::string> rest;

    ArgParser Parser() {
        ArgParser parser("tool", "does a thing");
        parser.Switch("--verbose", verbose, "say more")
            .Text("--json", "<path>", path, "write here")
            .PositiveInt("--count", "<N>", count, "how many")
            .PositiveNumber("--seconds", "<S>", seconds, "how long")
            .Choice("--color", "red|blue", color, ParseColor, "which colour");
        return parser;
    }
};

ArgResult Read(ArgParser parser, std::vector<std::string_view> args) {
    return parser.Read(args);
}

TEST(ArgParser, ReadsEveryKindOfFlag) {
    Flags flags;
    const ArgResult result = Read(flags.Parser(), {"--verbose", "--json", "out.json", "--count", "8", "--seconds",
                                                   "0.5", "--color", "blue"});
    EXPECT_EQ(result.outcome, ArgOutcome::kRun) << result.error;
    EXPECT_TRUE(flags.verbose);
    EXPECT_EQ(flags.path, "out.json");
    EXPECT_EQ(flags.count, 8);
    EXPECT_DOUBLE_EQ(flags.seconds, 0.5);
    EXPECT_EQ(flags.color, Color::kBlue);
}

TEST(ArgParser, LeavesDefaultsAloneWithoutArguments) {
    Flags flags;
    EXPECT_EQ(Read(flags.Parser(), {}).outcome, ArgOutcome::kRun);
    EXPECT_FALSE(flags.verbose);
    EXPECT_EQ(flags.count, 1);
    EXPECT_EQ(flags.color, Color::kRed);
}

TEST(ArgParser, AnswersHelpWithEveryFlag) {
    Flags flags;
    EXPECT_EQ(Read(flags.Parser(), {"--count", "3", "--help"}).outcome, ArgOutcome::kHelp);
    EXPECT_EQ(Read(flags.Parser(), {"-h"}).outcome, ArgOutcome::kHelp);
    const std::string usage = flags.Parser().Usage();
    EXPECT_TRUE(usage.starts_with("usage: tool [flags]\ndoes a thing\n")) << usage;
    for (const char* row : {"--verbose", "--json <path>", "--count <N>", "--seconds <S>", "--color red|blue",
                            "-h, --help"}) {
        EXPECT_NE(usage.find(row), std::string::npos) << row;
    }
}

// Every mistake is an error that names the argument, never silently ignored.
TEST(ArgParser, RejectsWhatItDoesNotKnow) {
    struct Case {
        std::vector<std::string_view> args;
        std::string_view error;
    };
    const Case cases[] = {
        {{"--colour", "red"}, "unknown flag --colour"},
        {{"--json"}, "--json needs a value: <path>"},
        {{"--count", "0"}, "bad value \"0\" for --count (expected <N>)"},
        {{"--count", "3x"}, "bad value \"3x\" for --count (expected <N>)"},
        {{"--seconds", "-1"}, "bad value \"-1\" for --seconds (expected <S>)"},
        {{"--seconds", "inf"}, "bad value \"inf\" for --seconds (expected <S>)"},
        {{"--color", "green"}, "bad value \"green\" for --color (expected red|blue)"},
        {{"stray"}, "unexpected argument \"stray\""},
    };
    for (const Case& c : cases) {
        Flags flags;
        const ArgResult result = Read(flags.Parser(), c.args);
        EXPECT_EQ(result.outcome, ArgOutcome::kError) << c.error;
        EXPECT_EQ(result.error, c.error);
    }
}

TEST(ArgParser, HandsPositionalArgumentsToTheirReader) {
    std::vector<int> lengths;
    bool verbose = false;
    ArgParser parser("decode", "time decoding");
    parser.Switch("--verbose", verbose, "say more").Positionals("<length>", "context lengths", [&](std::string_view arg) {
        lengths.push_back(static_cast<int>(arg.size()));
        return arg != "bad";
    });
    EXPECT_EQ(Read(parser, {"a", "--verbose", "bbb"}).outcome, ArgOutcome::kRun);
    EXPECT_TRUE(verbose);
    EXPECT_EQ(lengths, (std::vector<int>{1, 3}));
    EXPECT_EQ(Read(parser, {"bad"}).error, "bad argument \"bad\" (expected <length>)");
    EXPECT_NE(parser.Usage().find("usage: decode [flags] [<length>...]"), std::string::npos);
}

}  // namespace
}  // namespace murmur::lab
