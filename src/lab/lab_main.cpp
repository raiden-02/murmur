#include "lab/lab.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <print>
#include <string_view>
#include <vector>

namespace {

struct Command {
    std::string_view name;
    std::string_view description;
    int (*run)(murmur::lab::Args);
};

constexpr Command kCommands[] = {
    { "devices", "print GPU properties and theoretical bandwidth", murmur::lab::RunDeviceQuery },
    { "vector-add", "time c = a + b and report achieved bandwidth", murmur::lab::RunVectorAddBenchmark },
    { "reduce", "time the sum of 64 Mi floats and report achieved bandwidth", murmur::lab::RunReduceSumBenchmark },
    { "transpose", "time three transpose kernels on an 8192 x 8192 matrix", murmur::lab::RunTransposeBenchmark },
    { "matmul", "time 4096 x 4096 matrix multiply: naive, tiled and cuBLAS", murmur::lab::RunMatmulBenchmark },
    { "generate", "ask the noodle vendor one question and report speed", murmur::lab::RunGenerate },
    { "chat", "talk to the noodle vendor in the terminal (empty line quits)", murmur::lab::RunChat },
    { "decode", "time single-stream decode against the memory-bandwidth ceiling", murmur::lab::RunDecodeBenchmark },
    { "matvec", "bandwidth of the BF16 matrix-vector kernel on every model shape", murmur::lab::RunMatVecBenchmark },
    { "batch", "batched decode: throughput vs latency for 1 to 64 conversations", murmur::lab::RunBatchBenchmark },
    { "market", "30 NPCs talking non-stop: player time to first token, throughput, leak check", murmur::lab::RunMarketBenchmark },
    { "schedule", "continuous batching step by step for five conversations", murmur::lab::RunScheduleTrace },
    { "quality", "quality loss of each weight format against the original model", murmur::lab::RunQuality },
    { "calibrate", "measure channel importance for INT4 weights (saved next to the model)", murmur::lab::RunCalibrate },
    { "speculative", "speculative decoding: speed-up, kept guesses, identical replies", murmur::lab::RunSpeculativeBenchmark },
    { "actions", "NPC actions under a grammar: parse rate vs free text, jump-forward, action mix", murmur::lab::RunActionsBenchmark },
    { "compare", "the shared workload of bench/compare: time to first token, decode, throughput at 1-64 requests", murmur::lab::RunCompareBenchmark },
    { "attention", "decode attention alone: time per layer and bandwidth", murmur::lab::RunAttentionBenchmark },
};

void PrintUsage(std::FILE* stream) {
    std::println(stream, "usage: murmur-lab <command> [flags]   (murmur-lab <command> --help lists its flags)\n");
    for (const Command& command : kCommands) {
        std::println(stream, "  {:<12} {}", command.name, command.description);
    }
}

}  // namespace

// Runs the lab command named by the first argument. Any exception (for
// example a failed CUDA call) is printed with its file and line.
int main(int argc, char** argv) try {
    if (argc < 2) {
        PrintUsage(stderr);
        return EXIT_FAILURE;
    }
    const std::string_view requested = argv[1];
    if (requested == "--help" || requested == "-h") {
        PrintUsage(stdout);
        return EXIT_SUCCESS;
    }
    const std::vector<std::string_view> args(argv + 2, argv + argc);
    for (const Command& command : kCommands) {
        if (command.name == requested) {
            return command.run(args);
        }
    }
    std::println(stderr, "unknown command: {}\n", requested);
    PrintUsage(stderr);
    return EXIT_FAILURE;
}
catch (const std::exception& error) {
    std::println(stderr, "{}", error.what());
    return EXIT_FAILURE;
}
