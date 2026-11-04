#include "game/street.hpp"

#include "game/action_language.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace murmur::game {

namespace {

constexpr float kPlayerSpeed = 220.0f;

constexpr Color kNight{ 21, 19, 25, 255 };
constexpr Color kStreet{ 30, 27, 66, 255 };
constexpr Color kCurb{ 60, 52, 137, 255 };
constexpr Color kBuilding{ 24, 23, 28, 255 };
constexpr Color kSignBoard{ 12, 10, 18, 255 };
constexpr Color kRainColor{ 24, 95, 165, 255 };

constexpr std::array kStorefronts{
    Storefront{ { 40, 20, 200, 70 }, "RAMEN", { 212, 83, 126, 255 }, { 75, 21, 40, 255 } },
    Storefront{ { 280, 20, 200, 70 }, "CHROME", { 93, 202, 165, 255 }, { 4, 52, 44, 255 } },
    Storefront{ { 520, 20, 180, 70 }, "PAWN", { 239, 159, 39, 255 }, { 65, 36, 2, 255 } },
    Storefront{ { 740, 20, 220, 70 }, "NET CAFE", { 133, 183, 235, 255 }, { 4, 44, 83, 255 } },
    Storefront{ { 1000, 20, 240, 70 }, "CLINIC", { 237, 147, 177, 255 }, { 75, 21, 40, 255 } },
    Storefront{ { 60, 630, 200, 70 }, "ARMS", { 240, 149, 149, 255 }, { 80, 19, 19, 255 } },
    Storefront{ { 300, 630, 220, 70 }, "ORACLE", { 175, 169, 236, 255 }, { 38, 33, 92, 255 } },
    Storefront{ { 560, 630, 180, 70 }, "CLUB", { 151, 196, 89, 255 }, { 23, 52, 4, 255 } },
    Storefront{ { 780, 630, 200, 70 }, "DOCKS", { 250, 199, 117, 255 }, { 65, 36, 2, 255 } },
    Storefront{ { 1020, 630, 200, 70 }, "BAR", { 93, 202, 165, 255 }, { 4, 52, 44, 255 } },
};

// The model names storefronts by these labels (actions::kPlaces).
constexpr bool LabelsMatchPlaces() {
    for (std::size_t i = 0; i < kStorefronts.size(); ++i) {
        if (kStorefronts[i].label != actions::kPlaces[i]) {
            return false;
        }
    }
    return kStorefronts.size() == actions::kPlaces.size();
}
static_assert(LabelsMatchPlaces());

constexpr int kSignFont = 28;

// `label` centred on the sign, as large as kSignFont, smaller if that's
// what it takes to fit (a hacked sign can say up to 18 characters).
void DrawLabel(const Rectangle& area, std::string_view label, Color color, float dx = 0.0f) {
    const std::string text(label);   // raylib wants a null-terminated string
    int size = kSignFont;
    while (size > 12 && MeasureText(text.c_str(), size) > area.width - 16) {
        size -= 2;
    }
    const int text_width = MeasureText(text.c_str(), size);
    DrawText(text.c_str(), static_cast<int>(area.x + (area.width - text_width) / 2 + dx),
        static_cast<int>(area.y + (area.height - size) / 2), size, color);
}

Color Lit(Color color, float level) {
    return { static_cast<unsigned char>(color.r * level), static_cast<unsigned char>(color.g * level),
        static_cast<unsigned char>(color.b * level), color.a };
}

// How lit sign `index` is: a steady faint hum, and every few seconds (at its
// own moment) a brief stutter, the way failing neon does.
float NeonLevel(std::size_t index, float time) {
    const float i = static_cast<float>(index);
    const float period = 6.0f + static_cast<float>(index % 4) * 1.7f;
    const float t = std::fmod(time + i * 2.3f, period);
    if (t < 0.3f) {
        return std::sin(t * 70.0f) > 0.0f ? 0.25f : 1.0f;
    }
    return 0.93f + 0.07f * std::sin(time * 5.0f + i);
}

// A neon sign: a dark board, a tube of the sign's colour around it and the
// letters in a hotter shade of it (the bloom makes them glow).
void DrawNeonSign(const Storefront& storefront, float level) {
    const Rectangle& area = storefront.sign;
    const Color tube = storefront.fill;
    const Color hot{ static_cast<unsigned char>(tube.r + (255 - tube.r) * 0.35f),
        static_cast<unsigned char>(tube.g + (255 - tube.g) * 0.35f),
        static_cast<unsigned char>(tube.b + (255 - tube.b) * 0.35f), 255 };
    DrawRectangleRounded(area, 0.15f, 6, kSignBoard);
    DrawRectangleRoundedLinesEx(area, 0.15f, 6, 3.0f, Lit(tube, level));
    DrawLabel(area, storefront.label, Lit(hot, level));
}

// A hacked sign: dark, with the hacker's message in their colour, a little
// horizontal jitter, a colour-split ghost and the odd scanline, flickering
// back to the real sign for a frame now and then.
void DrawHackedSign(const Storefront& storefront, const SignHack& hack, float time) {
    const Rectangle& area = storefront.sign;
    const float phase = time * 17.0f + area.x;
    if (std::fmod(phase, 23.0f) < 0.6f) {
        DrawNeonSign(storefront, 1.0f);
        return;
    }
    DrawRectangleRounded(area, 0.1f, 4, kSignBoard);
    DrawRectangleRoundedLinesEx(area, 0.1f, 4, 2.0f, hack.color);
    const float jitter = std::sin(phase) > 0.92f ? 4.0f * std::sin(phase * 3.1f) : 0.0f;
    DrawLabel(area, hack.message, Fade({ 255, 40, 120, 255 }, 0.6f), jitter - 2.0f);
    DrawLabel(area, hack.message, hack.color, jitter);
    const float scan = area.y + std::fmod(time * 60.0f, area.height);
    DrawRectangleRec({ area.x + 4, scan, area.width - 8, 2 }, Fade(hack.color, 0.35f));
}

}  // namespace

Rain::Rain() {
    for (RainDrop& drop : drops_) {
        drop = Spawn(Uniform(0.0f, static_cast<float>(kScreenHeight)));
    }
}

void Rain::Update(float delta_time) {
    for (RainDrop& drop : drops_) {
        drop.position.y += drop.speed * delta_time;
        drop.position.x -= 0.15f * drop.speed * delta_time;
        if (drop.position.y > kScreenHeight) {
            drop = Spawn(-10.0f);
        }
    }
}

void Rain::Draw() const {
    for (const RainDrop& drop : drops_) {
        DrawLineV(
            drop.position,
            Vector2{ drop.position.x - 3.0f, drop.position.y + 14.0f },
            kRainColor);
    }
}

RainDrop Rain::Spawn(float y) {
    return RainDrop{
        { Uniform(0.0f, kScreenWidth + 100.0f), y },
        Uniform(500.0f, 900.0f)
    };
}

float Rain::Uniform(float low, float high) {
    return std::uniform_real_distribution<float>(low, high)(random_);
}

std::span<const Storefront> Storefronts() {
    return kStorefronts;
}

std::optional<std::size_t> FindStorefront(std::string_view label) {
    const auto it = std::ranges::find(kStorefronts, label, &Storefront::label);
    return it == kStorefronts.end() ? std::nullopt : std::optional(static_cast<std::size_t>(it - kStorefronts.begin()));
}

// Where an NPC stands to visit a storefront: on the street, just in front.
Vector2 Doorstep(std::size_t storefront) {
    const Rectangle& sign = kStorefronts[storefront].sign;
    const bool top = sign.y < kScreenHeight / 2;
    return { sign.x + sign.width / 2, top ? kStreetTop + 18 : kStreetBottom - 18 };
}

// `hacks` has one entry per storefront; a sign with time left shows its hack.
void DrawStreet(std::span<const SignHack> hacks, float time) {
    ClearBackground(kNight);
    DrawRectangle(0, 0, kScreenWidth, static_cast<int>(kStorefrontDepth), kBuilding);
    DrawRectangle(0, static_cast<int>(kStreetBottom), kScreenWidth,
        static_cast<int>(kStorefrontDepth), kBuilding);
    DrawRectangle(0, static_cast<int>(kStreetTop) + 60, kScreenWidth,
        static_cast<int>(kStreetBottom - kStreetTop) - 120, kStreet);
    DrawRectangle(0, static_cast<int>(kStreetTop) + 58, kScreenWidth, 3, kCurb);
    DrawRectangle(0, static_cast<int>(kStreetBottom) - 61, kScreenWidth, 3, kCurb);

    for (std::size_t i = 0; i < kStorefronts.size(); ++i) {
        const Storefront& storefront = kStorefronts[i];
        if (i < hacks.size() && hacks[i].seconds_left > 0.0f) {
            DrawHackedSign(storefront, hacks[i], time);
            continue;
        }
        DrawNeonSign(storefront, NeonLevel(i, time));
    }
}

Vector2 MovePlayer(Vector2 position, float delta_time) {
    Vector2 direction{ 0.0f, 0.0f };
    if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) direction.y -= 1.0f;
    if (IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN)) direction.y += 1.0f;
    if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) direction.x -= 1.0f;
    if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) direction.x += 1.0f;

    position.x += direction.x * kPlayerSpeed * delta_time;
    position.y += direction.y * kPlayerSpeed * delta_time;
    position.x = std::clamp(position.x, kPlayerRadius, kScreenWidth - kPlayerRadius);
    position.y = std::clamp(position.y, kStreetTop + kPlayerRadius,
        kStreetBottom - kPlayerRadius);
    return position;
}

}  // namespace murmur::game
