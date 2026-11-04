#pragma once

#include "lab/args.hpp"

// Lab commands, run as `murmur-lab <command> [arguments]`. Each receives the
// arguments after the command name (reading them with an ArgParser, so every
// command answers --help), returns a process exit code and may throw;
// lab_main.cpp reports exceptions.
namespace murmur::lab {

int RunDeviceQuery(Args args);
int RunVectorAddBenchmark(Args args);
int RunReduceSumBenchmark(Args args);
int RunTransposeBenchmark(Args args);
int RunMatmulBenchmark(Args args);
int RunGenerate(Args args);
int RunChat(Args args);
int RunDecodeBenchmark(Args args);
int RunMatVecBenchmark(Args args);
int RunBatchBenchmark(Args args);
int RunMarketBenchmark(Args args);
int RunScheduleTrace(Args args);
int RunQuality(Args args);
int RunCalibrate(Args args);
int RunSpeculativeBenchmark(Args args);
int RunActionsBenchmark(Args args);
int RunCompareBenchmark(Args args);
int RunAttentionBenchmark(Args args);

}  // namespace murmur::lab
