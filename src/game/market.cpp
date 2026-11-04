#include "game/market.hpp"

#include "game/text_layout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <string_view>
#include <utility>

namespace murmur::game {

namespace {

constexpr int kNpcsPerRow = 16;
constexpr float kRowInset = 40.0f;      // distance from the storefronts to the stalls
constexpr float kFirstStallX = 50.0f;
constexpr float kStallSpacing = 78.0f;
constexpr int kMaxReplyTokens = 120;
constexpr float kChatterMin = 2.5f;
constexpr float kChatterMax = 9.0f;
constexpr float kLineSeconds = 6.0f;    // a finished line stays up this long
constexpr float kBubbleFont = 15.0f;
constexpr float kBubbleWidth = 190.0f;
constexpr float kBubbleLine = 18.0f;
constexpr int kBubbleLines = 3;

constexpr float kWalkSpeed = 120.0f;
constexpr float kPlayerErrandDelay = 4.0f;   // the player gets to read the reply before the NPC walks off
constexpr float kChatterErrandDelay = 0.6f;
constexpr float kVisitSeconds = 5.0f;        // at a storefront
constexpr float kCallSeconds = 8.0f;         // beside whoever called them over
constexpr float kHackSeconds = 25.0f;
constexpr float kRestSeconds = 20.0f;        // between errands of its own, so stalls aren't always empty
constexpr float kActionLabelSeconds = 5.0f;
constexpr int kDeliveryReward = 25;

constexpr Color kStall{ 92, 60, 40, 255 };
constexpr Color kBubbleFill{ 245, 240, 255, 215 };
constexpr Color kBubbleText{ 38, 33, 92, 255 };

// What happens around an NPC when it speaks up on its own. Short stage
// directions, sent as the "user" turn of its conversation; some invite an
// action.
constexpr std::array<std::string_view, 9> kCues{
    "(Someone slows down near your stall. Call out to them in one short sentence.)",
    "(A security drone hums overhead. Mutter something about it in one sentence.)",
    "(A rival's sign across the street glows brighter than yours. Say one sentence about it.)",
    "(The rain gets heavier. Say one sentence to the stall next to you.)",
    "(You need something from another regular. Say whom, and why, in one sentence.)",
    "(A customer haggles over your prices. Answer them in one sentence.)",
    "(You are restless and want to stretch your legs. Say where you're going, in one sentence.)",
    "(The lights flicker like a blackout is coming. React in one sentence.)",
    "(A regular walks by. Greet them by name in one sentence.)",
};

Color NpcColor(int index) {
    const float hue = std::fmod(index * 137.508f, 360.0f);   // the golden angle spreads hues evenly
    return ColorFromHSV(hue, 0.55f, 0.95f);
}

// raylib takes null-terminated strings, so a view is copied first.
float TextWidth(const Font& font, std::string_view text, float size) {
    return MeasureTextEx(font, std::string(text).c_str(), size, 1.0f).x;
}

Vector2 Toward(Vector2 from, Vector2 to, float step, bool& arrived) {
    const float dx = to.x - from.x;
    const float dy = to.y - from.y;
    const float distance = std::hypot(dx, dy);
    arrived = distance <= step;
    return arrived ? to : Vector2{ from.x + dx / distance * step, from.y + dy / distance * step };
}

bool TopRow(Vector2 p) {
    return p.y < (kStreetTop + kStreetBottom) / 2;
}

// The NPC's latest action above its head, fading out: the bytes the model
// chose in its colour, the ones the grammar wrote for it (jump-forward) dim.
void DrawActionLabel(const Font& font, const Npc& npc, Vector2 p) {
    constexpr float kSize = 13.0f;
    const float fade = std::clamp(2.0f - npc.action_age / kActionLabelSeconds * 2.0f, 0.0f, 1.0f);
    const float width = TextWidth(font, npc.action, kSize);
    const float x0 = std::clamp(p.x - width / 2, 4.0f, kScreenWidth - width - 4);
    DrawRectangleRounded({ x0 - 4, p.y - 36, width + 8, 17 }, 0.3f, 4, Fade({ 10, 8, 16, 255 }, 0.8f * fade));
    const std::string& forced = npc.action_forced;
    std::size_t run = 0;
    for (std::size_t i = 1; i <= npc.action.size(); ++i) {
        const bool same = i < npc.action.size() && i < forced.size() && forced[i] == forced[run];
        if (same) {
            continue;
        }
        const bool by_grammar = run < forced.size() && forced[run] != 0;
        const float x = x0 + TextWidth(font, npc.action.substr(0, run), kSize);
        DrawTextEx(font, npc.action.substr(run, i - run).c_str(), { x, p.y - 34 }, kSize, 1.0f,
            Fade(by_grammar ? Color{ 150, 146, 160, 255 } : npc.color, fade));
        run = i;
    }
}

}  // namespace

// Opens one conversation per NPC (they start talking once the model has
// loaded) and places them at stalls along both sides of the street. Every
// conversation acts, and speculates when the engine has a draft model (the
// engine only does so while few conversations are decoding).
Market::Market(InferenceService& service) : service_(&service), hacks_(Storefronts().size()) {
    const auto& cast = personas::kMarketCast;
    const std::string rules = actions::Rules();
    npcs_.reserve(cast.size());
    for (std::size_t i = 0; i < cast.size(); ++i) {
        const bool top_row = i < kNpcsPerRow;
        const int column = static_cast<int>(i % kNpcsPerRow);
        Npc npc;
        npc.persona = &cast[i];
        npc.home = { kFirstStallX + column * kStallSpacing, top_row ? kStreetTop + kRowInset : kStreetBottom - kRowInset };
        npc.position = npc.home;
        npc.color = NpcColor(static_cast<int>(i));
        ConversationOptions options;
        options.system_prompt = personas::MarketSystemPrompt(cast[i], rules);
        options.sampling.seed = 1000 + i;
        options.max_reply_tokens = kMaxReplyTokens;
        options.memory_tokens = kNpcMemoryTokens;
        options.acts = true;
        options.speculative = true;
        npc.conversation = service.Open(std::move(options));
        npc.next_chatter = std::uniform_real_distribution<float>(1.0f, kChatterMin + 3.0f)(random_);
        npcs_.push_back(std::move(npc));
    }
}

// Once per frame: ages lines, actions and hacked signs, walks NPCs on their
// errands, and starts idle chatter for NPCs whose timer ran out (not the one
// talking to the player).
void Market::Update(float delta_time, bool engine_ready, const Npc* talking_to_player) {
    time_ += delta_time;
    for (SignHack& hack : hacks_) {
        hack.seconds_left = std::max(0.0f, hack.seconds_left - delta_time);
    }
    for (Npc& npc : npcs_) {
        npc.line_age += npc.replying ? 0.0f : delta_time;
        npc.action_age += delta_time;
        npc.rested += npc.errand ? 0.0f : delta_time;
        Move(npc, delta_time);
        if (!engine_ready || !chatter_ || npc.replying || &npc == talking_to_player) {
            continue;
        }
        npc.next_chatter -= delta_time;
        if (npc.next_chatter <= 0.0f) {
            Chatter(npc);
        }
    }
}

// Walks an NPC towards its errand (once the delay is up), keeps it there for
// the errand's stay, then walks it home. An NPC talking to the player stands
// still.
void Market::Move(Npc& npc, float delta_time) {
    if (npc.TalkingToPlayer()) {
        return;
    }
    Vector2 target = npc.home;
    if (npc.errand) {
        if (npc.errand_delay > 0.0f) {
            npc.errand_delay -= delta_time;
            return;
        }
        target = *npc.errand;
    }
    bool arrived = false;
    npc.position = Toward(npc.position, target, kWalkSpeed * delta_time, arrived);
    if (arrived && npc.errand) {
        npc.errand_stay -= delta_time;
        if (npc.errand_stay <= 0.0f) {
            npc.errand.reset();
            npc.rested = 0.0f;
        }
    }
}

void Market::Chatter(Npc& npc) {
    npc.next_chatter = NextChatterDelay();
    Ask(npc, std::string(kCues[cue_++ % kCues.size()]), false);
}

// Starts the NPC's next reply; `for_player` records whom it answers, for
// every event of it that follows.
void Market::Ask(Npc& npc, std::string message, bool for_player) {
    npc.replying = true;
    npc.reply_for_player = for_player;
    npc.line.clear();
    npc.drafted.clear();
    if (offer_ && offer_->seller == &npc) {
        offer_.reset();
    }
    if (for_player) {
        service_->Ask(npc.conversation, std::move(message), kPlayerPriority);
    } else {
        service_->Ask(npc.conversation, std::move(message));
    }
}

float Market::NextChatterDelay() {
    return std::uniform_real_distribution<float>(kChatterMin, kChatterMax)(random_);
}

// The player's message to `npc`, sent ahead of all chatter. Anything the
// NPC should know first (a delivery the player brings, a purchase) goes in
// front of it, as a stage direction. If the NPC is in the middle of a line,
// that line is cut short and the message follows as soon as it has stopped;
// until then the line's own events (its action included) are still the
// line's, not the player's.
void Market::PlayerSays(Npc& npc, std::string message) {
    for (auto it = player_.deliveries.begin(); it != player_.deliveries.end();) {
        const auto item = std::ranges::find(player_.items, it->item);
        if (it->to != npc.persona->name || item == player_.items.end()) {
            ++it;
            continue;
        }
        npc.note += std::format("(The player hands you the {} from {}.) ", it->item, it->from);
        player_.items.erase(item);
        player_.credits += kDeliveryReward;
        ++player_.delivered;
        it = player_.deliveries.erase(it);
    }
    message = std::exchange(npc.note, {}) + message;
    if (npc.replying) {
        service_->Cancel(npc.conversation);
        pending_ = PendingMessage{ &npc, std::move(message) };
        npc.player_waiting = true;
        return;
    }
    npc.player_waiting = false;
    Ask(npc, std::move(message), true);
}

// Walking away mid-reply cuts the reply short, and the offer lapses.
void Market::PlayerLeaves(Npc& npc) {
    if (pending_ && pending_->npc == &npc) {
        pending_.reset();
        npc.player_waiting = false;
    }
    if (npc.replying && npc.reply_for_player) {
        service_->Cancel(npc.conversation);
    }
    if (offer_ && offer_->seller == &npc) {
        offer_.reset();
    }
}

// Takes the current offer if the player can pay; the seller hears about it
// with the player's next message. An offer the player can't afford stands.
BuyResult Market::Buy() {
    if (!offer_) {
        return BuyResult::kNoOffer;
    }
    if (player_.credits < offer_->price) {
        return BuyResult::kTooFewCredits;
    }
    Npc* seller = Find(offer_->seller->conversation);
    player_.credits -= offer_->price;
    player_.items.push_back(offer_->item);
    seller->note += std::format("(The player paid you {} credits for the {}.) ", offer_->price, offer_->item);
    offer_.reset();
    return BuyResult::kBought;
}

// Carries out the action that ended `npc`'s reply; returns whether anything
// happened. Things only change hands when the reply answered the player,
// and an NPC only goes on an errand of its own after a rest (the player's
// requests don't wait).
bool Market::Act(Npc& npc, const actions::Action& action) {
    const auto free_to_go = [](const Npc& n) {
        return !n.errand && (n.reply_for_player || n.rested >= kRestSeconds);
    };
    ++action_counts_[static_cast<std::size_t>(action.verb)];
    switch (action.verb) {
    case actions::Verb::kNone:
    case actions::Verb::kCount:   // never parsed: Parse only makes real verbs
        return false;
    case actions::Verb::kWalk:
        if (const auto storefront = FindStorefront(action.place); storefront && free_to_go(npc)) {
            const float spread = static_cast<float>(static_cast<int>(&npc - npcs_.data()) % 5 - 2) * 14.0f;
            const Vector2 door = Doorstep(*storefront);
            npc.errand = Vector2{ door.x + spread, door.y };
            npc.errand_delay = npc.reply_for_player ? kPlayerErrandDelay : kChatterErrandDelay;
            npc.errand_stay = kVisitSeconds;
            return true;
        }
        return false;
    case actions::Verb::kHack:
        if (const auto storefront = FindStorefront(action.place)) {
            hacks_[*storefront] = { action.message, npc.color, kHackSeconds };
            return true;
        }
        return false;
    case actions::Verb::kCall:
        if (Npc* other = FindByName(action.npc); other != nullptr && other != &npc && !other->TalkingToPlayer() &&
            free_to_go(*other)) {
            const float side = npc.home.x < kScreenWidth / 2 ? 26.0f : -26.0f;
            other->errand = Vector2{ npc.home.x + side, npc.home.y + (TopRow(npc.home) ? 22.0f : -22.0f) };
            other->errand_delay = 0.3f;
            other->errand_stay = kCallSeconds;
            if (!other->replying) {
                other->next_chatter = NextChatterDelay();
                Ask(*other, std::format("({} waves you over. Say one sentence to them.)", npc.persona->name), false);
            }
            return true;
        }
        return false;
    case actions::Verb::kGive:
        if (npc.reply_for_player) {
            player_.items.push_back(action.item);
        }
        return npc.reply_for_player;
    case actions::Verb::kSell:
        if (npc.reply_for_player) {
            offer_ = Offer{ &npc, action.item, action.price };
        }
        return npc.reply_for_player;
    case actions::Verb::kQuest:
        if (npc.reply_for_player && action.npc != npc.persona->name) {
            player_.items.push_back(action.item);
            player_.deliveries.push_back({ action.item, std::string(npc.persona->name), action.npc });
            return true;
        }
        return false;
    }
    return false;
}

void Market::OnEvent(const InferenceEvent& event) {
    Npc* npc = Find(event.conversation);
    if (npc == nullptr) {
        return;
    }
    switch (event.kind) {
    case InferenceEventKind::kText:
        npc->line += event.text;
        npc->drafted.append(event.text.size(), event.drafted ? 1 : 0);
        break;
    case InferenceEventKind::kAction:
        if (const std::optional<actions::Action> action = actions::Parse(event.text); action && Act(*npc, *action)) {
            const std::size_t space = event.text.starts_with(' ') ? 1 : 0;   // every action starts with one
            npc->action = event.text.substr(space);
            npc->action_forced = event.forced.substr(std::min(space, event.forced.size()));
            npc->action_age = 0.0f;
        }
        break;
    case InferenceEventKind::kReplyFinished:
        if (npc->reply_for_player && event.stats.reply_tokens > 0) {
            last_player_first_word_ms_ = event.stats.prefill_ms;
        }
        npc->replying = false;
        npc->reply_for_player = false;
        npc->line_age = 0.0f;
        npc->last_reply = event.stats;
        if (pending_ && pending_->npc == npc) {
            PendingMessage message = std::move(*pending_);
            pending_.reset();
            PlayerSays(*npc, std::move(message.text));
        }
        break;
    case InferenceEventKind::kError:
        npc->line = "(lost in thought)";
        npc->drafted.assign(npc->line.size(), 0);
        break;
    case InferenceEventKind::kReady:
    case InferenceEventKind::kReconfigured:
    case InferenceEventKind::kStopped:
        break;   // engine-wide: the HUD's news
    }
}

Npc* Market::NearestTo(Vector2 point, float range) {
    Npc* nearest = nullptr;
    float best = range;
    for (Npc& npc : npcs_) {
        const float distance = std::hypot(npc.position.x - point.x, npc.position.y - point.y);
        if (distance <= best) {
            best = distance;
            nearest = &npc;
        }
    }
    return nearest;
}

const Npc* Market::FindByConversation(ConversationId conversation) const {
    const auto it = std::ranges::find(npcs_, conversation, &Npc::conversation);
    return it == npcs_.end() ? nullptr : &*it;
}

Npc* Market::Find(ConversationId conversation) {
    const auto it = std::ranges::find(npcs_, conversation, &Npc::conversation);
    return it == npcs_.end() ? nullptr : &*it;
}

Npc* Market::FindByName(std::string_view name) {
    const auto it = std::ranges::find(npcs_, name, [](const Npc& npc) { return npc.persona->name; });
    return it == npcs_.end() ? nullptr : &*it;
}

void Market::DrawStreet() const {
    game::DrawStreet(hacks_, time_);
}

// Stalls, NPCs, names, their latest action, and chatter bubbles. Bubbles
// open towards the middle of the street, alternating between two distances
// so neighbours' lines overlap less, and the freshest lines are drawn last,
// on top. The NPC talking to the player gets the big bubble instead
// (Dialogue draws it).
void Market::Draw(const Font& font, const Npc* talking_to_player) const {
    std::vector<std::size_t> speaking;
    for (std::size_t i = 0; i < npcs_.size(); ++i) {
        const Npc& npc = npcs_[i];
        const Vector2 home = npc.home;
        DrawRectangleRounded({ home.x - 30, home.y + 12, 60, 12 }, 0.4f, 4, kStall);
        const Vector2 p = npc.position;
        DrawCircleV(p, kPlayerRadius + 1, npc.color);
        if (npc.replying) {
            DrawCircleLinesV(p, kPlayerRadius + 5, npc.color);
        }
        const std::string name(npc.persona->name);
        DrawTextEx(font, name.c_str(), { p.x - TextWidth(font, name, 13.0f) / 2, p.y + 26 }, 13.0f, 1.0f, npc.color);
        if (npc.action_age < kActionLabelSeconds) {
            DrawActionLabel(font, npc, p);
        }
        const bool showing = npc.replying ? !npc.line.empty() : npc.line_age < kLineSeconds && !npc.line.empty();
        if (showing && &npc != talking_to_player && !npc.TalkingToPlayer()) {
            speaking.push_back(i);
        }
    }
    std::ranges::sort(speaking, [&](std::size_t a, std::size_t b) {
        const float age_a = npcs_[a].replying ? -1.0f : npcs_[a].line_age;
        const float age_b = npcs_[b].replying ? -1.0f : npcs_[b].line_age;
        return age_a > age_b;
    });
    for (const std::size_t i : speaking) {
        const Npc& npc = npcs_[i];
        const Vector2 p = npc.position;
        std::vector<std::string_view> lines = WrapText(font, npc.line, kBubbleFont, kBubbleWidth);
        if (lines.size() > kBubbleLines) {
            lines.erase(lines.begin(), lines.end() - kBubbleLines);
        }
        float width = 0.0f;
        for (const std::string_view line : lines) {
            width = std::max(width, TextWidth(font, line, kBubbleFont));
        }
        const float box_w = width + 12;
        const float box_h = lines.size() * kBubbleLine + 8;
        const float lane = (i % 2 == 0) ? 46.0f : 46.0f + 3 * kBubbleLine + 16;
        const float x = std::clamp(p.x - box_w / 2, 4.0f, kScreenWidth - box_w - 4);
        const float y = TopRow(p) ? p.y + lane : p.y - lane - box_h;
        const float fade = npc.replying ? 1.0f : std::clamp(1.5f - npc.line_age / kLineSeconds * 1.5f, 0.0f, 1.0f);
        DrawRectangleRounded({ x, y, box_w, box_h }, 0.2f, 4, Fade(kBubbleFill, 0.85f * fade + 0.15f));
        DrawRectangleLinesEx({ x, y, 3, box_h }, 3, npc.color);
        for (std::size_t l = 0; l < lines.size(); ++l) {
            DrawTextEx(font, std::string(lines[l]).c_str(), { x + 7, y + 4 + l * kBubbleLine }, kBubbleFont, 1.0f,
                kBubbleText);
        }
    }
}

}  // namespace murmur::game
