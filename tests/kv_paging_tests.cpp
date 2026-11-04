#include "engine/kv_cache_manager.hpp"
#include "engine/page_pool.hpp"
#include "engine/prefix_cache.hpp"

#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace murmur {
namespace {

constexpr int kPageSize = 4;

std::vector<TokenId> Tokens(int count, TokenId first = 100) {
    std::vector<TokenId> tokens(count);
    std::iota(tokens.begin(), tokens.end(), first);
    return tokens;
}

TEST(PagePool, HandsOutEveryPageOnceThenRunsOut) {
    PagePool pool(3);
    EXPECT_EQ(pool.Allocate(), 0);
    EXPECT_EQ(pool.Allocate(), 1);
    EXPECT_EQ(pool.Allocate(), 2);
    EXPECT_FALSE(pool.Allocate().has_value());
    EXPECT_EQ(pool.FreeCount(), 0);
    pool.Release(1);
    EXPECT_EQ(pool.Allocate(), 1);
    pool.CheckInvariants();
}

TEST(PagePool, SharedPagesAreFreedByTheirLastOwner) {
    PagePool pool(2);
    const PageId page = *pool.Allocate();
    pool.Retain(page);
    EXPECT_EQ(pool.RefCount(page), 2);
    pool.Release(page);
    EXPECT_EQ(pool.FreeCount(), 1);
    pool.Release(page);
    EXPECT_EQ(pool.FreeCount(), 2);
    pool.CheckInvariants();
}

TEST(PagePool, RejectsDoubleFreesAndBadIds) {
    PagePool pool(2);
    const PageId page = *pool.Allocate();
    pool.Release(page);
    EXPECT_THROW(pool.Release(page), std::logic_error);
    EXPECT_THROW(pool.Retain(page), std::logic_error);
    EXPECT_THROW(pool.Release(7), std::out_of_range);
    EXPECT_THROW(PagePool(0), std::invalid_argument);
}

class PrefixCacheTest : public ::testing::Test {
protected:
    // A sequence's pages, filled with `tokens` and published.
    std::vector<PageId> CachedSequence(const std::vector<TokenId>& tokens) {
        std::vector<PageId> pages;
        for (std::size_t i = 0; i < (tokens.size() + kPageSize - 1) / kPageSize; ++i) {
            pages.push_back(*pool_.Allocate());
        }
        cache_.Insert(tokens, pages);
        return pages;
    }

    PagePool pool_{16};
    PrefixCache cache_{pool_, kPageSize};
};

TEST_F(PrefixCacheTest, MatchesTheLongestCachedRunOfFullPages) {
    const std::vector<TokenId> lore = Tokens(10);   // 2 full pages + 2 tokens
    const std::vector<PageId> pages = CachedSequence(lore);
    EXPECT_EQ(cache_.Size(), 2);   // the partial third page is never cached

    std::vector<TokenId> prompt = Tokens(8);
    prompt.push_back(7);   // diverges inside the third page
    const std::vector<PageId> matched = cache_.Match(prompt, 10);
    ASSERT_EQ(matched.size(), 2u);
    EXPECT_EQ(matched[0], pages[0]);
    EXPECT_EQ(matched[1], pages[1]);
    EXPECT_EQ(pool_.RefCount(pages[0]), 3);   // sequence + cache + the new match
}

TEST_F(PrefixCacheTest, SameTokensAtADifferentPlaceDontMatch) {
    CachedSequence(Tokens(8));
    // The cached second page's tokens, after a different first page.
    std::vector<TokenId> shifted = Tokens(4, 900);
    const std::vector<TokenId> second_page = Tokens(4, 104);
    shifted.insert(shifted.end(), second_page.begin(), second_page.end());
    EXPECT_TRUE(cache_.Match(shifted, 10).empty());
    EXPECT_EQ(cache_.Match(Tokens(8), 1).size(), 1u);   // max_pages caps the match
}

TEST_F(PrefixCacheTest, EvictsOnlyPagesNobodyUsesLeastRecentFirst) {
    const std::vector<PageId> a = CachedSequence(Tokens(4, 0));
    const std::vector<PageId> b = CachedSequence(Tokens(4, 50));
    pool_.Release(a[0]);   // the sequences finish; only the cache holds their pages
    pool_.Release(b[0]);
    static_cast<void>(cache_.Match(Tokens(4, 0), 1));   // a is used again (and is shared now)

    ASSERT_TRUE(cache_.EvictOne());   // b: unused and least recent
    EXPECT_FALSE(cache_.Contains(b[0]));
    EXPECT_EQ(pool_.RefCount(b[0]), 0);
    EXPECT_FALSE(cache_.EvictOne());   // a is in use by the match above
    pool_.Release(a[0]);
    EXPECT_TRUE(cache_.EvictOne());
    EXPECT_EQ(pool_.FreeCount(), 16);
    pool_.CheckInvariants();
}

TEST_F(PrefixCacheTest, KeepsTheFirstPageForADuplicatePrefix) {
    const std::vector<PageId> first = CachedSequence(Tokens(4));
    const std::vector<PageId> second = CachedSequence(Tokens(4));   // computed again independently
    EXPECT_EQ(cache_.Size(), 1);
    EXPECT_EQ(pool_.RefCount(first[0]), 2);
    EXPECT_EQ(pool_.RefCount(second[0]), 1);   // stays private to its sequence
}

TEST(KvCacheManager, ReservesWholePagesAndRollsBackWhenFull) {
    KvCacheManager manager(3, kPageSize);
    KvSequence a;
    ASSERT_TRUE(manager.Reserve(a, 5));   // 2 pages
    EXPECT_EQ(a.pages.size(), 2u);
    KvSequence b;
    EXPECT_FALSE(manager.Reserve(b, 9));   // needs 3, only 1 left
    EXPECT_TRUE(b.pages.empty());
    EXPECT_EQ(manager.FreePages(), 1);
    manager.Release(a);
    EXPECT_EQ(manager.FreePages(), 3);
    manager.CheckInvariants();
}

TEST(KvCacheManager, ReclaimsUnusedCachedPagesWhenFull) {
    KvCacheManager manager(2, kPageSize);
    const std::vector<TokenId> tokens = Tokens(8);
    KvSequence a;
    ASSERT_TRUE(manager.Reserve(a, 8));
    a.length = 8;
    manager.Publish(a, tokens);
    manager.Release(a);   // its pages stay cached
    EXPECT_EQ(manager.CachedPages(), 2);
    EXPECT_EQ(manager.FreePages(), 0);

    KvSequence b;
    EXPECT_TRUE(manager.Reserve(b, 8));   // evicts both cached pages
    EXPECT_EQ(manager.CachedPages(), 0);
    manager.Release(b);
    manager.CheckInvariants();
}

TEST(KvCacheManager, ForgettingCachedPrefixesFreesOnlyPagesNobodyUses) {
    KvCacheManager manager(4, kPageSize);
    const std::vector<TokenId> tokens = Tokens(8);
    KvSequence a;
    ASSERT_TRUE(manager.Reserve(a, 8));
    a.length = 8;
    manager.Publish(a, tokens);
    KvSequence b;
    ASSERT_TRUE(manager.Reserve(b, 8));
    b.length = 8;
    manager.Publish(b, Tokens(8, 500));
    manager.Release(b);   // b's pages: only the cache holds them now

    manager.ForgetCachedPrefixes();
    EXPECT_EQ(manager.CachedPages(), 0);
    EXPECT_EQ(manager.FreePages(), 2);   // b's; a still uses its own
    KvSequence c;
    EXPECT_EQ(manager.AttachPrefix(c, tokens), 0) << "nothing is found any more";
    manager.Release(a);
    EXPECT_EQ(manager.FreePages(), 4);
    manager.CheckInvariants();
}

TEST(KvCacheManager, AttachesACachedPrefixButLeavesATokenToCompute) {
    KvCacheManager manager(8, kPageSize);
    const std::vector<TokenId> prompt = Tokens(8);
    KvSequence first;
    ASSERT_TRUE(manager.Reserve(first, 8));
    first.length = 8;
    manager.Publish(first, prompt);

    KvSequence second;
    EXPECT_EQ(manager.AttachPrefix(second, prompt), 4);   // the whole prompt is cached, but its last token must run
    EXPECT_EQ(second.pages.front(), first.pages.front());
    std::vector<TokenId> longer = prompt;
    longer.push_back(1);
    KvSequence third;
    EXPECT_EQ(manager.AttachPrefix(third, longer), 8);
    EXPECT_EQ(manager.RefCount(first.pages[0]), 4);   // first, cache, second, third
    EXPECT_THROW(manager.AttachPrefix(third, longer), std::logic_error);   // not empty
}

TEST(KvCacheManager, TruncatesPrivatePagesButNeverRewritesSharedOnes) {
    KvCacheManager manager(8, kPageSize);
    KvSequence sequence;
    ASSERT_TRUE(manager.Reserve(sequence, 10));
    sequence.length = 10;
    manager.Truncate(sequence, 5);
    EXPECT_EQ(sequence.pages.size(), 2u);
    EXPECT_EQ(sequence.length, 5);

    manager.Publish(sequence, Tokens(4));   // page 0 is now shared with the cache
    EXPECT_THROW(manager.Truncate(sequence, 3), std::logic_error);
    manager.Truncate(sequence, 4);   // page 0 is full and kept as is
    manager.Release(sequence);
    manager.CheckInvariants();
}

// Many sequences grow, share prompt prefixes, publish, truncate and finish at
// random; the pool's books must balance after every step, and once everything
// is released and the cache cleared every page must be free again.
TEST(KvCacheManager, RandomWorkloadNeverLeaksOrDoubleFreesPages) {
    constexpr int kPages = 64;
    KvCacheManager manager(kPages, kPageSize);
    std::mt19937 rng(7);
    struct Live {
        KvSequence kv;
        std::vector<TokenId> tokens;
    };
    std::vector<Live> sequences(12);
    const std::vector<TokenId> lore = Tokens(9, 5000);   // a shared prompt start

    for (int step = 0; step < 20000; ++step) {
        Live& s = sequences[rng() % sequences.size()];
        switch (rng() % 5) {
        case 0:   // start: maybe on the shared lore
            if (s.kv.pages.empty()) {
                s.tokens = rng() % 2 ? lore : Tokens(3, static_cast<TokenId>(rng() % 50));
                s.kv.length = 0;
                manager.AttachPrefix(s.kv, s.tokens);
            }
            break;
        case 1:
        case 2: {   // run a few more tokens
            const int target = static_cast<int>(s.tokens.size()) + 1 + static_cast<int>(rng() % 6);
            if (!s.tokens.empty() && manager.Reserve(s.kv, target)) {
                while (static_cast<int>(s.tokens.size()) < target) {
                    s.tokens.push_back(static_cast<TokenId>(rng() % 1000));
                }
                s.kv.length = target;
                manager.Publish(s.kv, s.tokens);
            }
            break;
        }
        case 3:   // finish
            manager.Release(s.kv);
            s.tokens.clear();
            break;
        case 4:   // free memory the way the scheduler does under pressure
            static_cast<void>(manager.EvictCachedPage());
            break;
        }
        manager.CheckInvariants();
        ASSERT_LE(manager.FreePages(), kPages);
    }
    for (Live& s : sequences) {
        manager.Release(s.kv);
    }
    while (manager.EvictCachedPage()) {
    }
    EXPECT_EQ(manager.FreePages(), kPages);
    EXPECT_EQ(manager.CachedPages(), 0);
}

}  // namespace
}  // namespace murmur
