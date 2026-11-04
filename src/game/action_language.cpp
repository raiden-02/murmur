#include "game/action_language.hpp"

#include "game/personas.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <iterator>

namespace murmur::actions {

namespace {

// How the grammar spells each verb, in Verb's order.
constexpr std::string_view kVerbNames[] = {"none", "walk", "give", "sell", "hack", "quest", "call"};
static_assert(std::size(kVerbNames) == kVerbCount, "every verb needs a name, and every name a verb");

// `"a" | "b" | ...`: a grammar rule's alternatives, one literal per name.
template <typename Names>
std::string Alternatives(const Names& names) {
    std::string rule;
    for (const auto& name : names) {
        rule += rule.empty() ? "\"" : " | \"";
        rule += name;
        rule += '"';
    }
    return rule;
}

std::array<std::string_view, personas::kMarketCast.size()> NpcNames() {
    std::array<std::string_view, personas::kMarketCast.size()> names;
    std::ranges::transform(personas::kMarketCast, names.begin(), &personas::Persona::name);
    return names;
}

template <typename Names>
std::string Join(const Names& names) {
    std::string joined;
    for (const auto& name : names) {
        joined += joined.empty() ? "" : ", ";
        joined += name;
    }
    return joined;
}

template <typename Names>
bool Contains(const Names& names, std::string_view name) {
    return std::ranges::find(names, name) != names.end();
}

bool IsNpc(std::string_view name) {
    return std::ranges::find(personas::kMarketCast, name, &personas::Persona::name) != personas::kMarketCast.end();
}

// Splits "x, y" in two at the first ", ".
std::optional<std::pair<std::string_view, std::string_view>> SplitPair(std::string_view text) {
    const std::size_t comma = text.find(", ");
    if (comma == std::string_view::npos) {
        return std::nullopt;
    }
    return std::pair(text.substr(0, comma), text.substr(comma + 2));
}

bool IsSignText(std::string_view message) {
    return !message.empty() && message.size() <= 18 && std::isalnum(static_cast<unsigned char>(message.front())) &&
           std::ranges::all_of(message, [](char c) {
               return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || std::string_view(" !?.'").contains(c);
           });
}

}  // namespace

// The action grammar, built from the game's own lists, so a new storefront,
// item or NPC is something the model can name at once. Sign messages are
// short capitals that fit a storefront; prices are 1 to 999 credits. An
// action starts with a space: the lead-in ends at "ACTION:", and the model
// writes a word after a colon as a token with its space (" walk"). A lead-in
// ending in the space would leave it only tokens it barely considers.
std::string Grammar() {
    return "# what an NPC does after it speaks\n"
           "action  ::= \" \" (\"none\" | walk | give | sell | hack | quest | call)\n"
           "walk    ::= \"walk(\" place \")\"\n"
           "give    ::= \"give(\" item \")\"\n"
           "sell    ::= \"sell(\" item \", \" price \")\"\n"
           "hack    ::= \"hack(\" place \", \\\"\" message \"\\\")\"\n"
           "quest   ::= \"quest(\" item \", \" npc \")\"\n"
           "call    ::= \"call(\" npc \")\"\n"
           "place   ::= " + Alternatives(kPlaces) + "\n"
           "item    ::= " + Alternatives(kItems) + "\n"
           "npc     ::= " + Alternatives(NpcNames()) + "\n"
           "price   ::= [1-9] [0-9]{0,2}\n"
           "message ::= [A-Z0-9] [A-Z0-9 !?.']{0,17}\n";
}

// What every NPC is told about actions, after the city lore (the same text
// for everyone, so it is part of the shared, cached prompt prefix).
std::string Rules() {
    return "After you speak, the game asks for your action, and you write exactly one:\n"
           "none: you only talk (the usual case).\n"
           "walk(PLACE): you go over to a storefront and come back.\n"
           "give(ITEM): you hand the player something, free.\n"
           "sell(ITEM, PRICE): you offer the player something for a price in credits.\n"
           "hack(PLACE, \"MESSAGE\"): you take over a storefront's sign with a short message in capitals.\n"
           "quest(ITEM, NAME): you give the player something to deliver to another regular.\n"
           "call(NAME): you wave another regular over.\n"
           "Places: " + Join(kPlaces) + ".\n"
           "Items: " + Join(kItems) + ".\n"
           "Regulars: " + Join(NpcNames()) + ".\n"
           "The action must match what you just said, but never say it or write it in your words: the game asks "
           "for it after you finish speaking. When nobody is talking to you directly, only none, walk, hack or "
           "call make sense.\n";
}

// Reads one sentence of the action grammar back into an Action; nullopt if
// it isn't one (the engine's constrained actions always are, but the same
// check measures how often an unconstrained model gets it right). Leading
// spaces are skipped.
std::optional<Action> Parse(std::string_view sentence) {
    while (sentence.starts_with(' ')) {
        sentence.remove_prefix(1);
    }
    if (sentence == "none") {
        return Action{};
    }
    const std::size_t open = sentence.find('(');
    if (open == std::string_view::npos || !sentence.ends_with(')')) {
        return std::nullopt;
    }
    const std::string_view verb = sentence.substr(0, open);
    const std::string_view inside = sentence.substr(open + 1, sentence.size() - open - 2);
    Action action;
    if (verb == "walk" && Contains(kPlaces, inside)) {
        action.verb = Verb::kWalk;
        action.place = inside;
    } else if (verb == "give" && Contains(kItems, inside)) {
        action.verb = Verb::kGive;
        action.item = inside;
    } else if (verb == "call" && IsNpc(inside)) {
        action.verb = Verb::kCall;
        action.npc = inside;
    } else if (const auto pair = SplitPair(inside); pair && verb == "sell" && Contains(kItems, pair->first)) {
        int price = 0;
        const auto [end, error] = std::from_chars(pair->second.data(), pair->second.data() + pair->second.size(), price);
        if (error != std::errc{} || end != pair->second.data() + pair->second.size() || price < 1 || price > 999 ||
            pair->second.front() == '0') {
            return std::nullopt;
        }
        action.verb = Verb::kSell;
        action.item = pair->first;
        action.price = price;
    } else if (pair && verb == "quest" && Contains(kItems, pair->first) && IsNpc(pair->second)) {
        action.verb = Verb::kQuest;
        action.item = pair->first;
        action.npc = pair->second;
    } else if (pair && verb == "hack" && Contains(kPlaces, pair->first) && pair->second.size() >= 2 &&
               pair->second.front() == '"' && pair->second.back() == '"' &&
               IsSignText(pair->second.substr(1, pair->second.size() - 2))) {
        action.verb = Verb::kHack;
        action.place = pair->first;
        action.message = pair->second.substr(1, pair->second.size() - 2);
    } else {
        return std::nullopt;
    }
    return action;
}

// How the grammar spells `verb`.
std::string_view VerbName(Verb verb) {
    const auto index = static_cast<std::size_t>(verb);
    return index < kVerbCount ? kVerbNames[index] : "?";
}

}  // namespace murmur::actions
