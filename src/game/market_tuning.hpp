#pragma once

// How the market drives the engine, shared by the game and `murmur-lab
// market`, so the load test measures the conversations the player walks
// among.

namespace murmur::game {

// What one NPC remembers: the shared lore, the action rules and its persona,
// then as many of the latest turns as fit.
inline constexpr int kNpcMemoryTokens = 1280;

// The player's messages go before NPC chatter.
inline constexpr int kPlayerPriority = 1;

}  // namespace murmur::game
