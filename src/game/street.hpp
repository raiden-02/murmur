#pragma once

// The night-market street: layout, neon storefront signs, rain and walking.

#include "raylib.h"

#include <array>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>

namespace murmur::game {

constexpr int kScreenWidth = 1280;
constexpr int kScreenHeight = 720;
constexpr int kTargetFps = 60;
constexpr float kPlayerRadius = 10.0f;

// Storefront rows at the top and bottom, walkable street between.
constexpr float kStorefrontDepth = 120.0f;
constexpr float kStreetTop = kStorefrontDepth;
constexpr float kStreetBottom = kScreenHeight - kStorefrontDepth;

struct RainDrop {
    Vector2 position;
    float speed;
};

class Rain {
public:
    Rain();
    void Update(float delta_time);
    void Draw() const;

private:
    static constexpr int kDropCount = 220;

    RainDrop Spawn(float y);
    float Uniform(float low, float high);

    std::mt19937 random_{ 1234 };
    std::array<RainDrop, kDropCount> drops_{};
};

struct Storefront {
    Rectangle sign;
    std::string_view label;   // one of actions::kPlaces
    Color fill;               // the neon tube's colour
    Color text;
};

// A sign someone has taken over: it shows their message, in their colour,
// glitching, until the time runs out.
struct SignHack {
    std::string message;
    Color color{};
    float seconds_left = 0.0f;
};

std::span<const Storefront> Storefronts();
std::optional<std::size_t> FindStorefront(std::string_view label);
Vector2 Doorstep(std::size_t storefront);
void DrawStreet(std::span<const SignHack> hacks, float time);
Vector2 MovePlayer(Vector2 position, float delta_time);

}  // namespace murmur::game
