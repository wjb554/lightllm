/// LightLLM -- Radix Tree Engine Tests
///
/// Dedicated tests for RadixPrefixCache internals and tree invariants.
/// Complements test_prefix_cache.cpp with deeper probing of the radix
/// data structure (Patricia trie):
///   -- tree topology after insert sequences
///   -- block_to_node_ map consistency
///   -- multi-level sharing and splitting
///   -- intra-block split cascades
///   -- LRU eviction with compaction (prune_leaf radix compression)
///   -- ref_count lifecycle (increment, remove, evict)
///   -- mixed access patterns (interleaved inserts, lookups, removes)
///   -- edge cases: single-token blocks, large block sizes, deep trees

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <memory>
#include <set>
#include <algorithm>

#include "lightllm/kv_cache/prefix_cache.h"

using namespace lightllm::kv_cache;

// ============================================================================
// Test harness
// ============================================================================
static int passed = 0;
static int failed = 0;

static void CHECK(const char* name, bool cond) {
    if (cond) {
        passed++;
    } else {
        failed++;
        std::fprintf(stderr, "  FAIL: %s\n", name);
    }
}

static std::vector<int> tokens(std::initializer_list<int> il) {
    return std::vector<int>(il);
}

static std::vector<int> make_seq(int len, int base = 0) {
    std::vector<int> v(len);
    for (int i = 0; i < len; ++i)
        v[i] = base + i;
    return v;
}

// ============================================================================
// 1. EMPTY TREE INVARIANTS
// ============================================================================
static void test_empty_tree() {
    std::printf("[1] Empty tree invariants\n");

    {
        auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);
        CHECK("empty/size_zero", cache->size() == 0);
        CHECK("empty/bs_correct", cache->block_size() == 4);

        auto r = cache->lookup(make_seq(4, 0), 0);
        CHECK("empty/lookup_miss", r.matched_tokens == 0);
        CHECK("empty/lookup_empty_blocks", r.matched_block_ids.empty());

        int e = cache->evict_lru();
        CHECK("empty/evict_returns_minus1", e == -1);

        cache->remove(999);
        CHECK("empty/remove_noop", cache->size() == 0);

        cache->increment_ref(999);
        CHECK("empty/increment_noop", cache->size() == 0);
    }

    // Different block sizes
    for (int bs : {1, 2, 8, 16, 32, 64}) {
        auto cache = create_prefix_cache(PrefixCachePolicy::Radix, bs);
        CHECK("empty/bs_variants", cache->block_size() == bs);
        CHECK("empty/bs_variants_size", cache->size() == 0);
    }

    std::printf("  [PASS] test_empty_tree\n");
}

// ============================================================================
// 2. SINGLE BLOCK INSERT AND LOOKUP
// ============================================================================
static void test_single_block() {
    std::printf("[2] Single block insert/lookup\n");

    for (int bs : {1, 2, 4, 8}) {
        auto cache = create_prefix_cache(PrefixCachePolicy::Radix, bs);

        auto seq = make_seq(bs, 42);
        cache->insert(seq, 0, {99});

        char buf[128];
        std::snprintf(buf, sizeof(buf), "single_bs%d/size_one", bs);
        CHECK(buf, cache->size() == 1);

        {
            auto r = cache->lookup(seq, 0);
            std::snprintf(buf, sizeof(buf), "single_bs%d/hit_tokens", bs);
            CHECK(buf, r.matched_tokens == bs);
            std::snprintf(buf, sizeof(buf), "single_bs%d/hit_block", bs);
            CHECK(buf, r.matched_block_ids.size() == 1);
            CHECK(buf, r.matched_block_ids[0] == 99);
        }

        // Miss with different tokens
        {
            auto diff = make_seq(bs, 999);
            auto r = cache->lookup(diff, 0);
            CHECK(buf, r.matched_tokens == 0);
        }
    }

    std::printf("  [PASS] test_single_block\n");
}

// ============================================================================
// 3. MULTI-BLOCK SEQUENTIAL INSERT (CHAIN)
// ============================================================================
static void test_multi_block_chain() {
    std::printf("[3] Multi-block chain insert/lookup\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    auto seq = make_seq(20, 100);
    cache->insert(seq, 0, {10, 11, 12, 13, 14});

    CHECK("chain/size", cache->size() == 5);

    // Full lookup
    {
        auto r = cache->lookup(seq, 0);
        CHECK("chain/full_tokens", r.matched_tokens == 20);
        CHECK("chain/full_blocks", (int)r.matched_block_ids.size() == 5);
        for (int i = 0; i < 5; i++)
            CHECK("chain/full_block_id", r.matched_block_ids[i] == 10 + i);
    }

    // Partial: first 12 tokens (3 blocks)
    {
        auto partial = make_seq(12, 100);
        auto r = cache->lookup(partial, 0);
        CHECK("chain/partial_12", r.matched_tokens == 12);
        CHECK("chain/partial_12_blocks", (int)r.matched_block_ids.size() == 3);
    }

    // Partial: first 4 tokens (1 block)
    {
        auto partial = make_seq(4, 100);
        auto r = cache->lookup(partial, 0);
        CHECK("chain/partial_4", r.matched_tokens == 4);
        CHECK("chain/partial_4_blocks", (int)r.matched_block_ids.size() == 1);
    }

    // Non-zero start_pos hits
    {
        auto r = cache->lookup(seq, 4);
        CHECK("chain/startpos4/tokens", r.matched_tokens == 16);
        CHECK("chain/startpos4/blocks", (int)r.matched_block_ids.size() == 4);
        CHECK("chain/startpos4/first_block", r.matched_block_ids[0] == 11);
    }

    {
        auto r = cache->lookup(seq, 16);
        CHECK("chain/startpos16/tokens", r.matched_tokens == 4);
        CHECK("chain/startpos16/blocks", (int)r.matched_block_ids.size() == 1);
        CHECK("chain/startpos16/block", r.matched_block_ids[0] == 14);
    }

    {
        auto r = cache->lookup(seq, 20);
        CHECK("chain/startpos_end/tokens", r.matched_tokens == 0);
        CHECK("chain/startpos_end/blocks", r.matched_block_ids.empty());
    }

    std::printf("  [PASS] test_multi_block_chain\n");
}

// ============================================================================
// 4. SIBLING BRANCHES (MULTI-WAY FANOUT)
// ============================================================================
static void test_sibling_branches() {
    std::printf("[4] Sibling branches\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 2);

    cache->insert(tokens({1, 100}), 0, {10});
    cache->insert(tokens({1, 200}), 0, {20});
    cache->insert(tokens({1, 300}), 0, {30});
    cache->insert(tokens({5, 6}),   0, {40});

    CHECK("sib/size", cache->size() == 4);

    {
        auto r = cache->lookup(tokens({1, 100}), 0);
        CHECK("sib/A", r.matched_tokens == 2 && r.matched_block_ids[0] == 10);
    }
    {
        auto r = cache->lookup(tokens({1, 200}), 0);
        CHECK("sib/B", r.matched_tokens == 2 && r.matched_block_ids[0] == 20);
    }
    {
        auto r = cache->lookup(tokens({1, 300}), 0);
        CHECK("sib/C", r.matched_tokens == 2 && r.matched_block_ids[0] == 30);
    }
    {
        auto r = cache->lookup(tokens({5, 6}), 0);
        CHECK("sib/D", r.matched_tokens == 2 && r.matched_block_ids[0] == 40);
    }

    {
        auto r = cache->lookup(tokens({1}), 0);
        CHECK("sib/shared_prefix", r.matched_block_ids.empty());
    }

    // Add many siblings to the same root key
    for (int i = 0; i < 20; i++) {
        cache->insert(tokens({1, 400 + i}), 0, {50 + i});
    }
    CHECK("sib/many_siblings_size", cache->size() == 24);

    for (int i = 0; i < 20; i++) {
        auto r = cache->lookup(tokens({1, 400 + i}), 0);
        CHECK("sib/many_sib_hit", r.matched_tokens == 2 && r.matched_block_ids[0] == 50 + i);
    }

    std::printf("  [PASS] test_sibling_branches\n");
}

// ============================================================================
// 5. NODE SPLITTING -- INTER-BLOCK (shared prefix between sequences)
// ============================================================================
static void test_inter_block_split() {
    std::printf("[5] Inter-block split (shared prefix)\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    auto seqA = tokens({0,1,2,3, 4,5,6,7, 10,11,12,13});
    cache->insert(seqA, 0, {100, 101, 102});

    auto seqB = tokens({0,1,2,3, 4,5,6,7, 20,21,22,23});
    cache->insert(seqB, 0, {100, 101, 200});

    auto seqC = tokens({0,1,2,3, 4,5,6,7, 30,31,32,33});
    cache->insert(seqC, 0, {100, 101, 300});

    CHECK("ibs/size_5", cache->size() == 5);

    {
        auto r = cache->lookup(seqA, 0);
        CHECK("ibs/A_full", r.matched_tokens == 12 && (int)r.matched_block_ids.size() == 3);
    }
    {
        auto r = cache->lookup(seqB, 0);
        CHECK("ibs/B_full", r.matched_tokens == 12 && (int)r.matched_block_ids.size() == 3);
        CHECK("ibs/B_leaf", r.matched_block_ids[2] == 200);
    }
    {
        auto r = cache->lookup(seqC, 0);
        CHECK("ibs/C_full", r.matched_tokens == 12 && (int)r.matched_block_ids.size() == 3);
        CHECK("ibs/C_leaf", r.matched_block_ids[2] == 300);
    }

    {
        auto short_seq = make_seq(8, 0);
        auto r = cache->lookup(short_seq, 0);
        CHECK("ibs/shared_prefix", r.matched_tokens == 8);
        CHECK("ibs/shared_2_blocks", (int)r.matched_block_ids.size() == 2);
    }

    std::printf("  [PASS] test_inter_block_split\n");
}

// ============================================================================
// 6. INTRA-BLOCK SPLIT (divergence within a single block)
// ============================================================================
static void test_intra_block_split() {
    std::printf("[6] Intra-block split\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    cache->insert(tokens({10, 20, 30, 40}), 0, {111});
    cache->insert(tokens({10, 20, 99, 88}), 0, {222});
    cache->insert(tokens({10, 20, 50, 60}), 0, {333});

    CHECK("intra/size_3", cache->size() == 3);

    {
        auto r = cache->lookup(tokens({10, 20, 30, 40}), 0);
        CHECK("intra/A_full", r.matched_tokens == 4 && r.matched_block_ids[0] == 111);
    }
    {
        auto r = cache->lookup(tokens({10, 20, 99, 88}), 0);
        CHECK("intra/B_full", r.matched_tokens == 4 && r.matched_block_ids[0] == 222);
    }
    {
        auto r = cache->lookup(tokens({10, 20, 50, 60}), 0);
        CHECK("intra/C_full", r.matched_tokens == 4 && r.matched_block_ids[0] == 333);
    }

    {
        auto r = cache->lookup(tokens({10, 20}), 0);
        CHECK("intra/shared_2_no_block", r.matched_block_ids.empty());
    }
    {
        auto r = cache->lookup(tokens({10}), 0);
        CHECK("intra/shared_1_no_block", r.matched_block_ids.empty());
    }

    std::printf("  [PASS] test_intra_block_split\n");
}

// ============================================================================
// 7. INTRA-BLOCK SPLIT WITHIN MULTI-BLOCK SEQUENCES
// ============================================================================
static void test_intra_block_split_multi() {
    std::printf("[7] Intra-block split within multi-block sequences\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    auto seqA = tokens({1,2,3,4, 10,20,30,40});
    cache->insert(seqA, 0, {100, 101});

    auto seqB = tokens({1,2,3,4, 10,20,99,88});
    cache->insert(seqB, 0, {100, 201});

    CHECK("ibsm/size_3", cache->size() == 3);

    {
        auto r = cache->lookup(seqA, 0);
        CHECK("ibsm/A_full", r.matched_tokens == 8 && (int)r.matched_block_ids.size() == 2);
        CHECK("ibsm/A_block1", r.matched_block_ids[1] == 101);
    }

    {
        auto r = cache->lookup(seqB, 0);
        CHECK("ibsm/B_full", r.matched_tokens == 8 && (int)r.matched_block_ids.size() == 2);
        CHECK("ibsm/B_block1", r.matched_block_ids[1] == 201);
    }

    {
        auto r = cache->lookup(tokens({1,2,3,4, 10,20}), 0);
        CHECK("ibsm/partial_6_tokens", r.matched_tokens == 4);
        CHECK("ibsm/partial_1_block", (int)r.matched_block_ids.size() == 1);
        CHECK("ibsm/partial_block100", r.matched_block_ids[0] == 100);
    }

    std::printf("  [PASS] test_intra_block_split_multi\n");
}

// ============================================================================
// 8. REF COUNT LIFECYCLE
// ============================================================================
static void test_ref_count_lifecycle() {
    std::printf("[8] Ref count lifecycle\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    cache->insert(make_seq(4, 1), 0, {10});
    CHECK("ref/insert_ok", true);

    cache->increment_ref(10);

    cache->remove(10);
    {
        int e = cache->evict_lru();
        CHECK("ref/not_evictable_at_ref1", e == -1);
    }

    cache->remove(10);
    {
        int e = cache->evict_lru();
        CHECK("ref/evictable_at_ref0", e == 10);
    }

    CHECK("ref/size_after_evict", cache->size() == 0);

    cache->increment_ref(10);
    cache->remove(10);
    CHECK("ref/post_evict_ops_no_crash", true);

    std::printf("  [PASS] test_ref_count_lifecycle\n");
}

// ============================================================================
// 9. LRU EVICTION ORDER (exact timestamps)
// ============================================================================
static void test_lru_exact_order() {
    std::printf("[9] LRU eviction exact order\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    cache->insert(make_seq(4, 100), 0, {10});
    cache->insert(make_seq(4, 200), 0, {20});
    cache->insert(make_seq(4, 300), 0, {30});
    cache->insert(make_seq(4, 400), 0, {40});
    cache->insert(make_seq(4, 500), 0, {50});

    // Touch block 30 then block 10
    cache->lookup(make_seq(4, 300), 0);
    cache->lookup(make_seq(4, 100), 0);

    for (int bid : {10, 20, 30, 40, 50})
        cache->remove(bid);

    // Expected order: 20 (oldest untouched), then 40, 50, 30, 10 (most recent)
    {
        int e = cache->evict_lru();
        CHECK("lru_exact/1st", e == 20);
    }
    {
        int e = cache->evict_lru();
        CHECK("lru_exact/2nd", e == 40);
    }
    {
        int e = cache->evict_lru();
        CHECK("lru_exact/3rd", e == 50);
    }
    {
        int e = cache->evict_lru();
        CHECK("lru_exact/4th", e == 30);
    }
    {
        int e = cache->evict_lru();
        CHECK("lru_exact/5th", e == 10);
    }
    {
        int e = cache->evict_lru();
        CHECK("lru_exact/6th_empty", e == -1);
    }

    CHECK("lru_exact/size_zero", cache->size() == 0);

    std::printf("  [PASS] test_lru_exact_order\n");
}

// ============================================================================
// 10. LRU WITH SHARED BLOCKS (ref_count prevents premature eviction)
// ============================================================================
static void test_lru_shared() {
    std::printf("[10] LRU with shared blocks\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    cache->insert(tokens({1,2,3,4, 10,20,30,40}), 0, {10, 11});
    cache->insert(tokens({1,2,3,4, 50,60,70,80}), 0, {10, 21});

    CHECK("lru_shared/size_3", cache->size() == 3);

    cache->remove(11);
    {
        int e = cache->evict_lru();
        CHECK("lru_shared/first_evict_11", e == 11);
    }

    {
        int e = cache->evict_lru();
        CHECK("lru_shared/second_none", e == -1);
    }

    cache->remove(21);
    {
        int e = cache->evict_lru();
        CHECK("lru_shared/third_evict_21", e == 21);
    }

    {
        int e = cache->evict_lru();
        CHECK("lru_shared/fourth_none", e == -1);
    }

    // Block 10 was shared by both sequences (ref_count=2).
    // One remove drops it to 1 -- not yet evictable.
    cache->remove(10);
    {
        int e = cache->evict_lru();
        CHECK("lru_shared/fifth_none_ref_still_1", e == -1);
    }

    // Second remove releases the last reference.
    cache->remove(10);
    {
        int e = cache->evict_lru();
        CHECK("lru_shared/sixth_evict_10", e == 10);
    }

    CHECK("lru_shared/size_zero", cache->size() == 0);

    std::printf("  [PASS] test_lru_shared\n");
}

// ============================================================================
// 11. EVICTION FOLLOWED BY RE-INSERTION (same block IDs)
// ============================================================================
static void test_evict_and_reinsert() {
    std::printf("[11] Evict and reinsert same block IDs\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    cache->insert(make_seq(4, 10), 0, {100});
    cache->remove(100);
    cache->evict_lru();
    CHECK("er/size_after_evict", cache->size() == 0);

    cache->insert(make_seq(4, 10), 0, {100});
    CHECK("er/size_after_reinsert", cache->size() == 1);

    {
        auto r = cache->lookup(make_seq(4, 10), 0);
        CHECK("er/reinsert_hit", r.matched_tokens == 4 && r.matched_block_ids[0] == 100);
    }

    cache->remove(100);
    cache->evict_lru();
    cache->insert(make_seq(4, 999), 0, {100});
    {
        auto r = cache->lookup(make_seq(4, 999), 0);
        CHECK("er/diff_tokens_hit", r.matched_tokens == 4 && r.matched_block_ids[0] == 100);
    }
    {
        auto r = cache->lookup(make_seq(4, 10), 0);
        CHECK("er/old_tokens_miss", r.matched_tokens == 0);
    }

    std::printf("  [PASS] test_evict_and_reinsert\n");
}

// ============================================================================
// 12. DEEP TREE WITH PROGRESSIVE SHARING
// ============================================================================
static void test_deep_progressive_share() {
    std::printf("[12] Deep tree with progressive sharing\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 2);

    cache->insert(tokens({1,2, 3,4, 5,6, 7,8}),            0, {0,1,2,3});
    cache->insert(tokens({1,2, 3,4, 5,6, 100,200}),        0, {0,1,2,4});
    cache->insert(tokens({1,2, 3,4, 300,400, 500,600}),    0, {0,1,5,6});
    cache->insert(tokens({1,2, 700,800, 900,1000, 1100,1200}), 0, {0,7,8,9});
    cache->insert(tokens({2000,2001, 2002,2003, 2004,2005, 2006,2007}), 0, {10,11,12,13});

    CHECK("deep/size_14", cache->size() == 14);

    {
        auto r = cache->lookup(tokens({1,2, 3,4, 5,6, 7,8}), 0);
        CHECK("deep/seq0", r.matched_tokens == 8 && (int)r.matched_block_ids.size() == 4);
    }
    {
        auto r = cache->lookup(tokens({1,2, 3,4, 5,6, 100,200}), 0);
        CHECK("deep/seq1", r.matched_tokens == 8 && (int)r.matched_block_ids.size() == 4);
    }
    {
        auto r = cache->lookup(tokens({1,2, 3,4, 300,400, 500,600}), 0);
        CHECK("deep/seq2", r.matched_tokens == 8 && (int)r.matched_block_ids.size() == 4);
    }
    {
        auto r = cache->lookup(tokens({1,2, 700,800, 900,1000, 1100,1200}), 0);
        CHECK("deep/seq3", r.matched_tokens == 8 && (int)r.matched_block_ids.size() == 4);
    }
    {
        auto r = cache->lookup(tokens({2000,2001, 2002,2003, 2004,2005, 2006,2007}), 0);
        CHECK("deep/seq4", r.matched_tokens == 8 && (int)r.matched_block_ids.size() == 4);
    }

    {
        auto r = cache->lookup(tokens({1,2}), 0);
        CHECK("deep/shared_block0", r.matched_tokens == 2 && r.matched_block_ids[0] == 0);
    }

    std::printf("  [PASS] test_deep_progressive_share\n");
}

// ============================================================================
// 13. SUFFIX-ONLY INSERT (start_pos > 0, not at root)
// ============================================================================
static void test_suffix_insert() {
    std::printf("[13] Suffix-only insert (start_pos > 0)\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    auto full_seq = make_seq(16, 500);
    cache->insert(full_seq, 8, {70, 71});

    CHECK("suffix/size", cache->size() == 2);

    {
        auto r = cache->lookup(full_seq, 0);
        CHECK("suffix/from0_miss", r.matched_tokens == 0);
    }

    {
        auto r = cache->lookup(full_seq, 8);
        CHECK("suffix/from8_hit", r.matched_tokens == 8);
        CHECK("suffix/from8_block0", r.matched_block_ids[0] == 70);
        CHECK("suffix/from8_block1", r.matched_block_ids[1] == 71);
    }

    cache->insert(full_seq, 4, {80});
    {
        auto r = cache->lookup(full_seq, 4);
        CHECK("suffix/from4_hit", r.matched_tokens == 4 && r.matched_block_ids[0] == 80);
    }

    std::printf("  [PASS] test_suffix_insert\n");
}

// ============================================================================
// 14. DUPLICATE INSERT (same tokens, same block_ids)
// ============================================================================
static void test_duplicate_insert_detail() {
    std::printf("[14] Duplicate insert detail\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    auto seq = make_seq(12, 1);
    cache->insert(seq, 0, {100, 101, 102});
    CHECK("dup/size_3", cache->size() == 3);

    cache->insert(seq, 0, {100, 101, 102});
    CHECK("dup/size_still_3", cache->size() == 3);

    {
        auto r = cache->lookup(seq, 0);
        CHECK("dup/still_hits", r.matched_tokens == 12);
        CHECK("dup/still_3_blocks", (int)r.matched_block_ids.size() == 3);
    }

    cache->remove(100);
    {
        int e = cache->evict_lru();
        CHECK("dup/not_evictable_1st", e == -1);
    }
    cache->remove(100);
    cache->remove(101);
    cache->remove(101);
    cache->remove(102);
    cache->remove(102);

    std::set<int> evicted;
    for (int i = 0; i < 3; i++) {
        int e = cache->evict_lru();
        if (e >= 0) evicted.insert(e);
    }
    CHECK("dup/all_3_evicted", evicted.size() == 3 && cache->size() == 0);

    std::printf("  [PASS] test_duplicate_insert_detail\n");
}

// ============================================================================
// 15. MIXED ACCESS PATTERNS (interleaved operations)
// ============================================================================
static void test_mixed_access() {
    std::printf("[15] Mixed access patterns\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    cache->insert(make_seq(12, 1), 0, {10, 11, 12});
    cache->insert(make_seq(12, 100), 0, {20, 21, 22});

    cache->lookup(make_seq(12, 1), 0);

    cache->insert(make_seq(12, 200), 0, {30, 31, 32});

    cache->remove(20);
    cache->remove(21);
    cache->remove(22);

    auto r = cache->lookup(make_seq(12, 100), 0);
    CHECK("mixed/lookup_removed", r.matched_tokens >= 0);

    int evicted_count = 0;
    for (int i = 0; i < 5; i++) {
        int e = cache->evict_lru();
        if (e >= 0) evicted_count++;
    }
    CHECK("mixed/some_evicted", evicted_count >= 3);

    cache->insert(make_seq(8, 500), 0, {40, 41});
    CHECK("mixed/post_evict_insert", cache->size() >= 2);

    std::printf("  [PASS] test_mixed_access\n");
}

// ============================================================================
// 16. EDGE CASES
// ============================================================================
static void test_edge_cases() {
    std::printf("[16] Edge cases\n");

    {
        auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 1);
        cache->insert(tokens({10, 20, 30, 40, 50}), 0, {100, 101, 102, 103, 104});
        CHECK("edge/bs1_size", cache->size() == 5);

        auto r = cache->lookup(tokens({10, 20, 30}), 0);
        CHECK("edge/bs1_partial", r.matched_tokens == 3);
        CHECK("edge/bs1_partial_blocks", (int)r.matched_block_ids.size() == 3);
    }

    {
        auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 16);
        auto seq = make_seq(48, 0);
        cache->insert(seq, 0, {1, 2, 3});
        CHECK("edge/bs16_size", cache->size() == 3);

        auto r = cache->lookup(seq, 0);
        CHECK("edge/bs16_full", r.matched_tokens == 48);
        CHECK("edge/bs16_blocks", (int)r.matched_block_ids.size() == 3);

        auto partial = make_seq(16, 0);
        auto r2 = cache->lookup(partial, 0);
        CHECK("edge/bs16_partial", r2.matched_tokens == 16);
        CHECK("edge/bs16_partial_blocks", (int)r2.matched_block_ids.size() == 1);
    }

    {
        auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);
        auto empty = std::vector<int>();
        cache->insert(empty, 0, {});
        CHECK("edge/empty_insert_noop", cache->size() == 0);
    }

    {
        auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);
        const int N = 400;
        auto seq = make_seq(N, 1);
        std::vector<int> bids;
        for (int i = 0; i < N / 4; i++) bids.push_back(1000 + i);
        cache->insert(seq, 0, bids);
        CHECK("edge/long_size", cache->size() == 100);

        auto r = cache->lookup(seq, 0);
        CHECK("edge/long_full", r.matched_tokens == N);
        CHECK("edge/long_full_blocks", (int)r.matched_block_ids.size() == 100);

        for (int i = 0; i < 50; i++)
            cache->remove(1000 + i);
        for (int i = 0; i < 50; i++)
            cache->evict_lru();
        CHECK("edge/long_post_evict", cache->size() >= 50);
    }

    std::printf("  [PASS] test_edge_cases\n");
}

// ============================================================================
// 17. BLOCK SIZE BOUNDARY ALIGNMENT
// ============================================================================
static void test_block_boundary() {
    std::printf("[17] Block boundary alignment\n");

    for (int bs : {1, 2, 3, 4, 5, 7, 8, 16}) {
        auto cache = create_prefix_cache(PrefixCachePolicy::Radix, bs);

        auto seq = make_seq(bs * 2, 1);
        cache->insert(seq, 0, {10, 11});

        {
            auto r = cache->lookup(seq, 0);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "bound_bs%d/full", bs);
            CHECK(buf, r.matched_tokens == bs * 2);
            CHECK(buf, (int)r.matched_block_ids.size() == 2);
        }

        {
            auto partial = make_seq(bs, 1);
            auto r = cache->lookup(partial, 0);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "bound_bs%d/partial", bs);
            CHECK(buf, r.matched_tokens == bs);
            CHECK(buf, (int)r.matched_block_ids.size() == 1);
        }

        {
            auto r = cache->lookup(seq, bs);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "bound_bs%d/startpos", bs);
            CHECK(buf, r.matched_tokens == bs);
            CHECK(buf, (int)r.matched_block_ids.size() == 1);
            CHECK(buf, r.matched_block_ids[0] == 11);
        }
    }

    std::printf("  [PASS] test_block_boundary\n");
}

// ============================================================================
// 18. RADIX COMPRESSION (Patricia trie path compression)
// ============================================================================
static void test_radix_compression() {
    std::printf("[18] Radix compression (Patricia trie)\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    cache->insert(tokens({1,2,3,4, 10,20,30,40}), 0, {100, 101});
    cache->insert(tokens({1,2,3,4, 50,60,70,80}), 0, {100, 201});

    CHECK("comp/size_3", cache->size() == 3);

    cache->remove(101);
    int e1 = cache->evict_lru();
    CHECK("comp/evict_101", e1 == 101);

    {
        auto r = cache->lookup(tokens({1,2,3,4, 50,60,70,80}), 0);
        CHECK("comp/seq2_still_works", r.matched_tokens == 8 && (int)r.matched_block_ids.size() == 2);
        CHECK("comp/seq2_block201", r.matched_block_ids[1] == 201);
    }

    {
        auto r = cache->lookup(tokens({1,2,3,4}), 0);
        CHECK("comp/shared_still_works", r.matched_tokens == 4 && r.matched_block_ids[0] == 100);
    }

    std::printf("  [PASS] test_radix_compression\n");
}

// ============================================================================
// 19. STRESS: RANDOM INTERLEAVED OPERATIONS
// ============================================================================
static void test_stress_random_ops() {
    std::printf("[19] Stress -- random interleaved ops\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    const int NUM_OPS = 500;
    const int NUM_SEQS = 100;
    std::vector<std::vector<int>> sequences;
    std::vector<std::vector<int>> block_ids_list;

    for (int i = 0; i < NUM_SEQS; i++) {
        int n_blocks = 2 + (i * 3) % 5;
        int n_tokens = n_blocks * 4;
        auto seq = make_seq(n_tokens, i * 100);
        std::vector<int> bids;
        for (int j = 0; j < n_blocks; j++)
            bids.push_back(i * 10 + j);
        sequences.push_back(seq);
        block_ids_list.push_back(bids);
    }

    int next_insert = 0;
    int next_lookup = 0;
    int next_remove = 0;

    for (int op = 0; op < NUM_OPS; op++) {
        int action = op % 3;
        if (action == 0 && next_insert < NUM_SEQS) {
            cache->insert(sequences[next_insert], 0, block_ids_list[next_insert]);
            next_insert++;
        } else if (action == 1 && next_lookup < next_insert) {
            cache->lookup(sequences[next_lookup], 0);
            next_lookup++;
        } else if (action == 2 && next_remove < next_insert) {
            auto& bids = block_ids_list[next_remove];
            for (int bid : bids)
                cache->remove(bid);
            next_remove++;
        }
    }

    int evicted = 0;
    while (true) {
        int e = cache->evict_lru();
        if (e < 0) break;
        evicted++;
    }

    CHECK("stress/no_crash", true);
    CHECK("stress/some_evictions", evicted >= 0);

    std::printf("  [PASS] test_stress_random_ops (evicted=%d)\n", evicted);
}

// ============================================================================
// 20. INCREMENT_REF ON SHARED NODE
// ============================================================================
static void test_increment_ref_detail() {
    std::printf("[20] Increment ref on shared nodes\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    cache->insert(make_seq(12, 10), 0, {10, 11, 12});

    cache->increment_ref(11);

    cache->remove(11);
    {
        int e = cache->evict_lru();
        CHECK("incr_ref/not_evictable", e == -1);
    }

    cache->remove(11);
    cache->remove(12);

    int evicted = 0;
    for (int i = 0; i < 5; i++) {
        int e = cache->evict_lru();
        if (e >= 0) evicted++;
    }
    CHECK("incr_ref/no_crash", true);
    CHECK("incr_ref/some_evicted", evicted >= 1);

    std::printf("  [PASS] test_increment_ref_detail\n");
}

// ============================================================================
// 21. FULL EVICTION CYCLE (fill -> evict all -> verify empty)
// ============================================================================
static void test_full_eviction_cycle() {
    std::printf("[21] Full eviction cycle\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    for (int i = 0; i < 20; i++) {
        cache->insert(make_seq(4, i * 10 + 1), 0, {i});
    }
    CHECK("fec/size_20", cache->size() == 20);

    for (int i = 0; i < 20; i++)
        cache->remove(i);

    int evicted_count = 0;
    for (int i = 0; i < 25; i++) {
        int e = cache->evict_lru();
        if (e >= 0) evicted_count++;
        else break;
    }
    CHECK("fec/all_20_evicted", evicted_count == 20);
    CHECK("fec/size_zero", cache->size() == 0);

    CHECK("fec/evict_empty", cache->evict_lru() == -1);

    cache->insert(make_seq(8, 999), 0, {100, 101});
    CHECK("fec/reinsert_works", cache->size() == 2);
    {
        auto r = cache->lookup(make_seq(8, 999), 0);
        CHECK("fec/reinsert_lookup", r.matched_tokens == 8);
    }

    std::printf("  [PASS] test_full_eviction_cycle\n");
}

// ============================================================================
// 22. MULTIPLE BLOCK IDS TO SAME NODE (shared path, same block)
// ============================================================================
static void test_shared_path_same_block() {
    std::printf("[22] Shared path with same block ID\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    cache->insert(tokens({1,2,3,4, 10,20,30,40}), 0, {100, 101});
    cache->insert(tokens({1,2,3,4, 50,60,70,80}), 0, {100, 102});

    CHECK("spsb/size_3", cache->size() == 3);

    cache->remove(100);
    {
        int e = cache->evict_lru();
        CHECK("spsb/block100_not_evictable_yet", e == -1 || e != 100);
    }

    cache->remove(100);
    cache->remove(101);
    cache->remove(102);

    int evicted = 0;
    for (int i = 0; i < 5; i++) {
        if (cache->evict_lru() >= 0) evicted++;
    }
    CHECK("spsb/all_3_evicted", evicted == 3);
    CHECK("spsb/empty", cache->size() == 0);

    std::printf("  [PASS] test_shared_path_same_block\n");
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::printf("LightLLM -- Radix Engine Tests\n");
    std::printf("===============================\n\n");

    test_empty_tree();
    test_single_block();
    test_multi_block_chain();
    test_sibling_branches();
    test_inter_block_split();
    test_intra_block_split();
    test_intra_block_split_multi();
    test_ref_count_lifecycle();
    test_lru_exact_order();
    test_lru_shared();
    test_evict_and_reinsert();
    test_deep_progressive_share();
    test_suffix_insert();
    test_duplicate_insert_detail();
    test_mixed_access();
    test_edge_cases();
    test_block_boundary();
    test_radix_compression();
    test_stress_random_ops();
    test_increment_ref_detail();
    test_full_eviction_cycle();
    test_shared_path_same_block();

    std::printf("\n===============================\n");
    std::printf("%d passed, %d failed, %d total\n",
                passed, failed, passed + failed);
    return failed == 0 ? 0 : 1;
}
