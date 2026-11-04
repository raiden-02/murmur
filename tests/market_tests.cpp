#include "engine/inference_service.hpp"
#include "game/market.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace murmur::game {
namespace {

// The market's bookkeeping on its own: the engine never loads (no model
// there), and the test plays the engine's part by handing the market the
// events a reply would produce.
class MarketTest : public ::testing::Test {
protected:
    MarketTest() : service_({.model_dir = "no/such/model"}), market_(service_) {}

    Npc& Find(std::string_view name) {
        Npc* npc = market_.FindByName(name);
        EXPECT_NE(npc, nullptr) << name;
        return *npc;
    }
    void Act(const Npc& npc, const std::string& action) {
        market_.OnEvent({.kind = InferenceEventKind::kAction,
                         .conversation = npc.conversation,
                         .text = " " + action,
                         .forced = std::string(action.size() + 1, '\0')});
    }
    void Finish(const Npc& npc) {
        market_.OnEvent({.kind = InferenceEventKind::kReplyFinished, .conversation = npc.conversation});
    }

    InferenceService service_;
    Market market_;
};

// The player speaks up while Mei is calling out to passers-by. Her chatter
// still ends with its own action, and that one wasn't meant for the player:
// it mustn't hand them anything. The reply to the player follows it.
TEST_F(MarketTest, ChatterCutShortByThePlayerGivesThemNothing) {
    market_.Update(100.0f, true, nullptr);   // every NPC's chatter timer runs out
    Npc& mei = Find("Mei");
    ASSERT_TRUE(mei.replying);
    EXPECT_FALSE(mei.reply_for_player);

    market_.PlayerSays(mei, "Hello");
    EXPECT_TRUE(mei.player_waiting);
    EXPECT_TRUE(mei.TalkingToPlayer());
    Act(mei, "give(tea)");
    EXPECT_TRUE(market_.Player().items.empty());

    Finish(mei);   // the chatter stops; the player's message goes out
    EXPECT_TRUE(mei.replying);
    EXPECT_TRUE(mei.reply_for_player);
    EXPECT_FALSE(mei.player_waiting);
    Act(mei, "give(tea)");
    EXPECT_EQ(market_.Player().items, std::vector<std::string>{"tea"});

    Finish(mei);
    EXPECT_FALSE(mei.TalkingToPlayer());
}

TEST_F(MarketTest, SellsOnlyWhatThePlayerCanAfford) {
    Npc& kenji = Find("Kenji");
    const int credits = market_.Player().credits;
    market_.PlayerSays(kenji, "What have you got?");
    Act(kenji, "sell(phone, 999)");
    Finish(kenji);
    ASSERT_TRUE(market_.CurrentOffer().has_value());
    EXPECT_EQ(market_.Buy(), BuyResult::kTooFewCredits);
    EXPECT_EQ(market_.Player().credits, credits);
    EXPECT_TRUE(market_.CurrentOffer().has_value()) << "the offer stands";

    market_.PlayerSays(kenji, "Anything cheaper?");
    Act(kenji, "sell(battery, 20)");
    Finish(kenji);
    EXPECT_EQ(market_.Buy(), BuyResult::kBought);
    EXPECT_EQ(market_.Player().credits, credits - 20);
    EXPECT_EQ(market_.Player().items, std::vector<std::string>{"battery"});
    EXPECT_EQ(market_.Buy(), BuyResult::kNoOffer);
}

// An NPC busy with the player isn't called away by a neighbour's chatter.
TEST_F(MarketTest, ACallDoesNotPullAnNpcAwayFromThePlayer) {
    Npc& hana = Find("Hana");
    Npc& mei = Find("Mei");
    market_.PlayerSays(hana, "Any rumours?");
    market_.Update(100.0f, true, &hana);   // everyone else starts chattering
    ASSERT_TRUE(mei.replying);
    Act(mei, "call(Hana)");
    EXPECT_FALSE(hana.errand.has_value());
}

}  // namespace
}  // namespace murmur::game
