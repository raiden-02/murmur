#pragma once

// `murmur --demo "<message>" [--warmup <seconds>] [--xray] [--screenshot <file.png>]`
// (read in main.cpp): lets the market chatter for a while, then says one
// message to Mei without any input, measures frame times and her time to
// first word while the market keeps talking, saves a screenshot and exits.
// Used to verify the game and to record the demo.

#include "game/dialogue.hpp"
#include "game/hud.hpp"
#include "game/market.hpp"

#include "raylib.h"

#include <optional>
#include <string>

namespace murmur::game {

struct DemoOptions {
    std::string message;
    std::string screenshot_path;   // empty: no screenshot
    float warmup_seconds = 0.0f;
};

class Demo {
public:
    explicit Demo(DemoOptions options);

    const DemoOptions& Options() const { return options_; }

    // Advances the script by one frame (it may move the player); returns the
    // process exit code once the demo is over.
    std::optional<int> Update(Dialogue& dialogue, Market& market, const Hud& hud, Vector2& player, float frame_seconds);

private:
    enum class Stage { kWaitingForEngine, kWarmingUp, kReplying, kSettling };

    int Finish(const Market& market);

    DemoOptions options_;
    Stage stage_ = Stage::kWaitingForEngine;
    float elapsed_ = 0.0f;
    float stage_time_ = 0.0f;
    int reply_frames_ = 0;
    double reply_seconds_ = 0.0;
    float worst_frame_ = 0.0f;
    int most_talking_ = 0;
    std::string reply_;   // Mei's words, kept the moment she finished
};

}  // namespace murmur::game
