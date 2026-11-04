#pragma once

// Command-line flags for murmur-lab's commands and the game. A program lists
// the flags it takes; Parse fills them in, answers --help with a usage
// table, and rejects a flag it doesn't know instead of quietly ignoring it.

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace murmur::lab {

using Args = std::span<const std::string_view>;

enum class ArgOutcome { kRun, kHelp, kError };

struct ArgResult {
    ArgOutcome outcome = ArgOutcome::kRun;
    std::string error;   // kError: what was wrong, naming the argument
};

std::optional<int> ParsePositiveInt(std::string_view text);
std::optional<double> ParsePositiveNumber(std::string_view text);

class ArgParser {
public:
    // Reads one argument; false if it is not a valid value.
    using Reader = std::function<bool(std::string_view)>;

    explicit ArgParser(std::string program, std::string summary = {});

    ArgParser& Switch(std::string_view flag, bool& target, std::string_view help);
    ArgParser& Text(std::string_view flag, std::string_view value, std::string& target, std::string_view help);
    ArgParser& PositiveInt(std::string_view flag, std::string_view value, int& target, std::string_view help);
    ArgParser& PositiveNumber(std::string_view flag, std::string_view value, double& target, std::string_view help);
    ArgParser& Value(std::string_view flag, std::string_view value, std::string_view help, Reader read);
    ArgParser& Positionals(std::string_view value, std::string_view help, Reader read);

    // A flag whose value is one of the names `parse` knows (a weight format,
    // say); `value` lists them for the usage table.
    template <typename T>
    ArgParser& Choice(std::string_view flag, std::string_view value, T& target,
                      std::optional<T> (*parse)(std::string_view), std::string_view help) {
        return Value(flag, value, help, [&target, parse](std::string_view text) {
            const std::optional<T> parsed = parse(text);
            if (parsed) {
                target = *parsed;
            }
            return parsed.has_value();
        });
    }

    ArgResult Read(Args args) const;
    std::optional<int> Parse(Args args) const;
    int Fail(std::string_view error) const;
    std::string Usage() const;

private:
    struct Flag {
        std::string name;
        std::string value;   // what follows the flag, for the usage table; empty for a switch
        std::string help;
        Reader read;
    };

    const Flag* Find(std::string_view name) const;

    std::string program_;
    std::string summary_;
    std::vector<Flag> flags_;
    std::optional<Flag> positionals_;   // arguments that aren't flags; its name stays empty
};

}  // namespace murmur::lab
