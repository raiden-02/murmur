#include "game/demo.hpp"

#include "game/personas.hpp"

#include <algorithm>
#include <cstdlib>
#include <print>
#include <string_view>
#include <utility>

namespace murmur::game {

namespace {

constexpr float kTimeoutSeconds = 180.0f;
constexpr float kSettleSeconds = 0.5f;   // let the finished bubble render before the screenshot
constexpr std::string_view kPartner = "Mei";

static_assert(std::ranges::find(personas::kMarketCast, kPartner, &personas::Persona::name) !=
                  personas::kMarketCast.end(),
              "the demo talks to someone in the market's cast");

}  // namespace

Demo::Demo(DemoOptions options) : options_(std::move(options)) {}

std::optional<int> Demo::Update(Dialogue& dialogue, Market& market, const Hud& hud, Vector2& player,
    float frame_seconds) {
    elapsed_ += frame_seconds;
    stage_time_ += frame_seconds;
    if (hud.EngineFailed()) {
        std::println(stderr, "demo: the engine failed to start: {}", hud.Error());
        return EXIT_FAILURE;
    }
    if (elapsed_ > kTimeoutSeconds + options_.warmup_seconds) {
        std::println(stderr, "demo: timed out");
        return EXIT_FAILURE;
    }
    const int talking = static_cast<int>(std::ranges::count_if(market.Npcs(), &Npc::replying));
    switch (stage_) {
    case Stage::kWaitingForEngine:
        if (hud.EngineReady()) {
            stage_ = Stage::kWarmingUp;
            stage_time_ = 0.0f;
        }
        break;
    case Stage::kWarmingUp:
        if (stage_time_ >= options_.warmup_seconds) {
            // Stand right in front of Mei's stall.
            Npc& mei = *market.FindByName(kPartner);
            player = { mei.position.x, mei.position.y + 40.0f };
            dialogue.Say(mei, options_.message);
            stage_ = Stage::kReplying;
            stage_time_ = 0.0f;
        }
        break;
    case Stage::kReplying:
        ++reply_frames_;
        reply_seconds_ += frame_seconds;
        worst_frame_ = std::max(worst_frame_, frame_seconds);
        most_talking_ = std::max(most_talking_, talking);
        if (!dialogue.Replying()) {
            // Kept now: once she is done, Mei may walk off on an errand and
            // the dialogue turn to whoever stands nearest.
            reply_ = dialogue.Reply();
            stage_ = Stage::kSettling;
            stage_time_ = 0.0f;
        }
        break;
    case Stage::kSettling:
        if (stage_time_ >= kSettleSeconds) {
            return Finish(market);
        }
        break;
    }
    return std::nullopt;
}

// Saves the screenshot and reports how smooth the frame loop stayed and how
// quickly Mei answered while the rest of the market kept talking.
int Demo::Finish(const Market& market) {
    if (!options_.screenshot_path.empty()) {
        // ExportImage takes any path; TakeScreenshot would force it under the working directory.
        const Image screen = LoadImageFromScreen();
        if (!ExportImage(screen, options_.screenshot_path.c_str())) {
            std::println(stderr, "demo: could not save {}", options_.screenshot_path);
        }
        UnloadImage(screen);
    }
    std::println("> {}", options_.message);
    std::println("{}: {}", kPartner, reply_);
    const double average_fps = reply_seconds_ > 0.0 ? reply_frames_ / reply_seconds_ : 0.0;
    std::println("while Mei replied: {} frames in {:.2f} s, average {:.1f} fps, slowest frame {:.1f} ms",
        reply_frames_, reply_seconds_, average_fps, worst_frame_ * 1000.0f);
    std::println("first word after {:.0f} ms, with up to {} other NPCs talking", market.LastPlayerFirstWordMs(),
        std::max(0, most_talking_ - 1));
    return EXIT_SUCCESS;
}

}  // namespace murmur::game
