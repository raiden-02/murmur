// Murmur: a rainy night market whose 32 stallholders talk, live, through
// Murmur's own GPU inference engine. They chatter among themselves while you
// walk; stop at any stall and talk to whoever runs it. Every reply ends with
// an action the model writes under a grammar (walk somewhere, hack a sign,
// call someone over, give or sell the player something, hand them a
// delivery), and the market carries it out. Tab shows the engine at work; Q
// and K change how the model's weights and memory are stored; G lets a small
// draft model guess the next words (speculative decoding).
//
// `murmur --help` lists the command line: the starting formats (FP8 weights
// and a BF16 cache by default), the X-ray, and the hands-free demo
// (demo.hpp).

#include "engine/inference_service.hpp"
#include "engine/model_loader.hpp"
#include "game/action_language.hpp"
#include "game/demo.hpp"
#include "game/dialogue.hpp"
#include "game/hud.hpp"
#include "game/market.hpp"
#include "game/post_fx.hpp"
#include "game/street.hpp"
#include "game/text_layout.hpp"
#include "lab/args.hpp"

#include "raylib.h"

#include <cstddef>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace murmur;
using namespace murmur::game;

constexpr int kFontBaseSize = 20;
constexpr int kDraftTokens = 4;   // guesses per step with speculation on (G)

// The GPU memory the game reserves for the model, weights and KV cache
// together, the way a game budgets VRAM. Whatever the weights don't take
// holds cached tokens: with BF16 weights about 1,450 pages, fewer than 32
// NPCs' memories need, so the weight format visibly changes how many fit.
constexpr std::size_t kModelMemoryBudget = std::size_t{3584} << 20;

// What the command line asks for.
struct LaunchOptions {
    WeightFormat weights = WeightFormat::kFp8;
    KvFormat kv = KvFormat::kBf16;
    bool xray = false;
    std::optional<DemoOptions> demo;   // set by --demo
};

// Reads the command line into `options`. Returns the exit code instead when
// the game shouldn't start: 0 after --help, 1 after a bad argument (both
// print the usage).
std::optional<int> ReadCommandLine(int argc, char** argv, LaunchOptions& options) {
    std::string message;
    double warmup_seconds = 0.0;
    std::string screenshot_path;
    bool has_message = false;
    lab::ArgParser parser("murmur", "the night market; every NPC talks through the engine");
    parser.Choice("--weights", "bf16|fp8|int4", options.weights, ParseWeightFormat, "starting weight format (default fp8)")
        .Choice("--kv", "fp32|bf16|fp8", options.kv, ParseKvFormat, "starting KV cache format (default bf16)")
        .Switch("--xray", options.xray, "start with the X-ray open (Tab)")
        .Value("--demo", "<message>", "hands-free: after the warm-up, say this to Mei, report and exit",
               [&](std::string_view text) {
                   message = text;
                   has_message = true;
                   return true;
               })
        .PositiveNumber("--warmup", "<seconds>", warmup_seconds, "with --demo: chatter to wait through first")
        .Text("--screenshot", "<file.png>", screenshot_path, "with --demo: save the last frame here");
    const std::vector<std::string_view> args(argv + 1, argv + argc);
    if (const std::optional<int> exit_code = parser.Parse(args)) {
        return exit_code;
    }
    if (!has_message && (warmup_seconds > 0.0 || !screenshot_path.empty())) {
        return parser.Fail("--warmup and --screenshot only go with --demo");
    }
    if (has_message) {
        options.demo = DemoOptions{.message = std::move(message),
                                   .screenshot_path = std::move(screenshot_path),
                                   .warmup_seconds = static_cast<float>(warmup_seconds)};
    }
    return std::nullopt;
}

WeightFormat NextWeights(WeightFormat format) {
    switch (format) {
    case WeightFormat::kBf16:
        return WeightFormat::kFp8;
    case WeightFormat::kFp8:
        return WeightFormat::kInt4;
    case WeightFormat::kInt4:
        return WeightFormat::kBf16;
    }
    return WeightFormat::kFp8;
}

// Demo runs open on a second monitor when there is one, out of the way.
void MoveToSecondMonitor() {
    if (GetMonitorCount() > 1) {
        const Vector2 origin = GetMonitorPosition(1);
        SetWindowPosition(static_cast<int>(origin.x) + 40, static_cast<int>(origin.y) + 40);
    }
}

}  // namespace

int main(int argc, char** argv) {
    LaunchOptions launch;
    if (const std::optional<int> exit_code = ReadCommandLine(argc, argv, launch)) {
        return *exit_code;
    }
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(kScreenWidth, kScreenHeight, "Murmur");
    SetTargetFPS(kTargetFps);
    SetExitKey(KEY_NULL);   // Esc closes the chat box; the window's close button quits
    int exit_code = 0;

    {
        // The model loads on the service's own thread while the street is
        // already running; the frame loop never waits for the GPU. Without
        // the model the street still runs, under a panel saying what to
        // download (the service then stops at once).
        const std::filesystem::path model_dir = DefaultModelDirectory();
        const std::vector<std::string> missing_files = MissingModelFiles(model_dir);
        InferenceServiceOptions engine_options{ .model_dir = model_dir,
            .weights = launch.weights,
            .memory_budget = kModelMemoryBudget,
            .action_grammar = actions::Grammar() };
        engine_options.transformer.kv_format = launch.kv;
        // The draft model (Qwen2.5-0.5B, INT4) can guess the next words while
        // the street is quiet; it starts off (G), since with the NPCs' sampled
        // words it keeps too few guesses to pay for itself.
        const std::filesystem::path draft_dir = DraftModelDirectory();
        const bool has_draft_model = MissingModelFiles(draft_dir).empty();
        if (has_draft_model) {
            engine_options.draft_model_dir = draft_dir;
        }
        engine_options.engine.draft_tokens = 0;
        InferenceService engine(std::move(engine_options));
        const UiFont font(kFontBaseSize);
        Market market(engine);
        Dialogue dialogue(market, font.Get());
        Hud hud(font.Get());
        hud.SetDraftModelInstalled(has_draft_model);
        if (!missing_files.empty()) {
            hud.ShowMissingModel(model_dir, missing_files);
        }
        Rain rain;
        PostFx post_fx(kScreenWidth, kScreenHeight, kStreetTop, kStreetBottom);
        Vector2 player{ kScreenWidth / 2.0f, kScreenHeight / 2.0f };
        hud.ShowXRay(launch.xray);
        std::optional<Demo> demo;
        if (launch.demo) {
            demo.emplace(*launch.demo);
            MoveToSecondMonitor();
        }

        while (!WindowShouldClose()) {
            const float delta_time = GetFrameTime();
            for (const InferenceEvent& event : engine.TakeEvents()) {
                hud.OnEvent(event);
                market.OnEvent(event);
            }
            const EngineStats stats = engine.Stats();
            hud.Update(stats, delta_time);
            if (demo) {
                if (const std::optional<int> result = demo->Update(dialogue, market, hud, player, delta_time)) {
                    exit_code = *result;
                    break;
                }
            }
            if (!dialogue.CapturesKeyboard()) {
                if (IsKeyPressed(KEY_TAB)) {
                    hud.ToggleXRay();
                }
                // Swapping formats reloads the weights (a second or two) while
                // everyone keeps talking afterwards.
                if (hud.EngineReady() && !hud.Switching()) {
                    if (IsKeyPressed(KEY_Q)) {
                        hud.OnFormatsRequested();
                        engine.SetFormats(NextWeights(stats.weight_format), stats.kv_format);
                    } else if (IsKeyPressed(KEY_G) && has_draft_model) {
                        engine.SetDraftTokens(stats.draft_tokens > 0 ? 0 : kDraftTokens);
                    } else if (IsKeyPressed(KEY_G)) {
                        hud.Notify(std::format("G needs the draft model: put Qwen2.5-0.5B-Instruct in {}",
                                               std::filesystem::absolute(draft_dir).string()));
                    } else if (IsKeyPressed(KEY_K)) {
                        hud.OnFormatsRequested();
                        engine.SetFormats(stats.weight_format,
                            stats.kv_format == KvFormat::kFp8 ? KvFormat::kBf16 : KvFormat::kFp8);
                    }
                }
                player = MovePlayer(player, delta_time);
            }
            dialogue.Update(player, delta_time, hud.EngineReady());
            market.Update(delta_time, hud.EngineReady(), dialogue.Partner());
            rain.Update(delta_time);

            BeginDrawing();
            post_fx.BeginScene();   // the lit street: bloom and reflections
            market.DrawStreet();
            rain.Draw();
            post_fx.EndScene(static_cast<float>(GetTime()));
            market.Draw(font.Get(), dialogue.Partner());
            dialogue.Draw(player);
            hud.Draw(market, stats);
            DrawFPS(kScreenWidth - 90, static_cast<int>(kStreetBottom) + 4);
            EndDrawing();
        }
    }

    CloseWindow();
    return exit_code;
}
