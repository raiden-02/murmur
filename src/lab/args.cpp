#include "lab/args.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <format>
#include <print>
#include <system_error>
#include <utility>

namespace murmur::lab {

namespace {

constexpr std::string_view kHelpFlags[] = {"--help", "-h"};

bool IsHelp(std::string_view arg) {
    return std::ranges::find(kHelpFlags, arg) != std::end(kHelpFlags);
}

// Flags start with "--"; "-h" is the one short spelling. Anything else is a
// positional argument (a negative number included).
bool LooksLikeFlag(std::string_view arg) {
    return arg.starts_with("--") || IsHelp(arg);
}

// The whole of `text` as a number of type T, or nullopt: "12abc" is not 12.
template <typename T>
std::optional<T> ParseWhole(std::string_view text) {
    T value{};
    const char* end = text.data() + text.size();
    const auto [stop, error] = std::from_chars(text.data(), end, value);
    if (text.empty() || error != std::errc() || stop != end) {
        return std::nullopt;
    }
    return value;
}

}  // namespace

// A whole number of at least 1 (a count, a length), all of `text`.
std::optional<int> ParsePositiveInt(std::string_view text) {
    const std::optional<int> value = ParseWhole<int>(text);
    return value && *value >= 1 ? value : std::nullopt;
}

// A finite number above 0 (seconds, megabytes), all of `text`.
std::optional<double> ParsePositiveNumber(std::string_view text) {
    const std::optional<double> value = ParseWhole<double>(text);
    return value && std::isfinite(*value) && *value > 0.0 ? value : std::nullopt;
}

// `program` is what the user types to run it ("murmur-lab market"), shown
// in the usage line; `summary`, if any, is one line on what it does (a lab
// command's is in `murmur-lab --help`).
ArgParser::ArgParser(std::string program, std::string summary)
    : program_(std::move(program)), summary_(std::move(summary)) {}

// --flag on its own: sets `target` to true.
ArgParser& ArgParser::Switch(std::string_view flag, bool& target, std::string_view help) {
    flags_.push_back({std::string(flag), "", std::string(help), [&target](std::string_view) {
                          target = true;
                          return true;
                      }});
    return *this;
}

// --flag <value>: any text, a path for example.
ArgParser& ArgParser::Text(std::string_view flag, std::string_view value, std::string& target,
                           std::string_view help) {
    return Value(flag, value, help, [&target](std::string_view text) {
        target = std::string(text);
        return true;
    });
}

// --flag <N>: see ParsePositiveInt.
ArgParser& ArgParser::PositiveInt(std::string_view flag, std::string_view value, int& target,
                                  std::string_view help) {
    return Choice(flag, value, target, ParsePositiveInt, help);
}

// --flag <X>: see ParsePositiveNumber.
ArgParser& ArgParser::PositiveNumber(std::string_view flag, std::string_view value, double& target,
                                     std::string_view help) {
    return Choice(flag, value, target, ParsePositiveNumber, help);
}

// --flag <value>, read by `read`, which rejects a bad value by returning false.
ArgParser& ArgParser::Value(std::string_view flag, std::string_view value, std::string_view help, Reader read) {
    flags_.push_back({std::string(flag), std::string(value), std::string(help), std::move(read)});
    return *this;
}

// Arguments that aren't flags, each handed to `read` in order. Without this,
// any such argument is an error.
ArgParser& ArgParser::Positionals(std::string_view value, std::string_view help, Reader read) {
    positionals_ = Flag{"", std::string(value), std::string(help), std::move(read)};
    return *this;
}

const ArgParser::Flag* ArgParser::Find(std::string_view name) const {
    const auto it = std::ranges::find(flags_, name, &Flag::name);
    return it == flags_.end() ? nullptr : &*it;
}

// Reads `args` into the targets, without printing anything. Stops at the
// first --help or the first bad argument; targets read before it keep their
// new values.
ArgResult ArgParser::Read(Args args) const {
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (IsHelp(arg)) {
            return {ArgOutcome::kHelp, ""};
        }
        if (!LooksLikeFlag(arg)) {
            if (!positionals_) {
                return {ArgOutcome::kError, std::format("unexpected argument \"{}\"", arg)};
            }
            if (!positionals_->read(arg)) {
                return {ArgOutcome::kError, std::format("bad argument \"{}\" (expected {})", arg, positionals_->value)};
            }
            continue;
        }
        const Flag* flag = Find(arg);
        if (flag == nullptr) {
            return {ArgOutcome::kError, std::format("unknown flag {}", arg)};
        }
        if (flag->value.empty()) {
            flag->read(arg);
            continue;
        }
        if (i + 1 == args.size()) {
            return {ArgOutcome::kError, std::format("{} needs a value: {}", arg, flag->value)};
        }
        const std::string_view value = args[++i];
        if (!flag->read(value)) {
            return {ArgOutcome::kError, std::format("bad value \"{}\" for {} (expected {})", value, arg, flag->value)};
        }
    }
    return {};
}

// Read, then: after --help the usage goes to stdout and the program should
// exit successfully; after a bad argument the error and the usage go to
// stderr and it should fail. Returns that exit code, or nullopt to run.
std::optional<int> ArgParser::Parse(Args args) const {
    const ArgResult result = Read(args);
    switch (result.outcome) {
    case ArgOutcome::kRun:
        return std::nullopt;
    case ArgOutcome::kHelp:
        std::print("{}", Usage());
        return EXIT_SUCCESS;
    case ArgOutcome::kError:
        return Fail(result.error);
    }
    return EXIT_FAILURE;
}

// Reports a mistake Read can't see (flags that only make sense together, say)
// the way Parse reports its own: the error and the usage on stderr. Returns
// the exit code to fail with.
int ArgParser::Fail(std::string_view error) const {
    std::print(stderr, "{}: {}\n\n{}", program_, error, Usage());
    return EXIT_FAILURE;
}

// The usage line, the summary and one row per flag:
//   usage: murmur-lab attention [flags]
//   decode attention alone: time per layer and bandwidth
//
//     --batch <N>      conversations decoding together (default 64)
//     -h, --help       show this help
std::string ArgParser::Usage() const {
    std::vector<std::pair<std::string, std::string>> rows;
    if (positionals_) {
        rows.emplace_back(positionals_->value + "...", positionals_->help);
    }
    for (const Flag& flag : flags_) {
        rows.emplace_back(flag.value.empty() ? flag.name : flag.name + " " + flag.value, flag.help);
    }
    rows.emplace_back("-h, --help", "show this help");
    std::size_t width = 0;
    for (const auto& [left, help] : rows) {
        width = std::max(width, left.size());
    }
    std::string usage = std::format("usage: {} [flags]{}\n", program_,
                                    positionals_ ? " [" + positionals_->value + "...]" : "");
    usage += summary_.empty() ? "\n" : summary_ + "\n\n";
    for (const auto& [left, help] : rows) {
        usage += std::format("  {:<{}}  {}\n", left, width, help);
    }
    return usage;
}

}  // namespace murmur::lab
