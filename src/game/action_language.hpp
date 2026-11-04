#pragma once

// What the market's NPCs can do. After its words, every NPC writes one
// action, and the engine holds the model to this grammar while it does
// (constrained decoding), so every action is one the game can carry out:
//   walk(DOCKS)                 go to a storefront and come back
//   give(tea)                   give the player something
//   sell(data shard, 40)        offer the player something for credits
//   hack(CLUB, "FREE ENTRY")    take over a storefront's sign for a while
//   quest(noodles, Kenji)       give the player something to deliver
//   call(Hana)                  wave another regular over
//   none

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace murmur::actions {

// The storefronts along the street, left to right, top row then bottom.
inline constexpr std::array<std::string_view, 10> kPlaces{
    "RAMEN", "CHROME", "PAWN", "NET CAFE", "CLINIC", "ARMS", "ORACLE", "CLUB", "DOCKS", "BAR",
};

// What changes hands: the things the stalls sell.
inline constexpr std::array<std::string_view, 22> kItems{
    "noodles", "skewers",  "tea",    "dumplings", "steamed buns", "coffee",    "spices",    "lantern",
    "charm",   "umbrella", "data shard", "phone", "optic implant", "battery", "transit permit", "tunnel map",
    "knife",   "jacket",   "seeds",  "ID badge",  "newspaper",    "neon sign",
};

// kCount is not a verb: it counts them, to size tables indexed by verb.
enum class Verb { kNone, kWalk, kGive, kSell, kHack, kQuest, kCall, kCount };
inline constexpr std::size_t kVerbCount = static_cast<std::size_t>(Verb::kCount);

struct Action {
    Verb verb = Verb::kNone;
    std::string place;     // walk, hack
    std::string item;      // give, sell, quest
    std::string npc;       // quest (whom to deliver to), call
    std::string message;   // hack
    int price = 0;         // sell, in credits
};

std::string Grammar();
std::string Rules();
std::optional<Action> Parse(std::string_view sentence);
std::string_view VerbName(Verb verb);

}  // namespace murmur::actions
