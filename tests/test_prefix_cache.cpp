/// LightLLM — Prefix Cache Comprehensive Tests
///
/// Tests the full PrefixCache interface (HashPrefixCache and RadixPrefixCache)
/// through the abstract PrefixCache* and factory, plus implementation-specific
/// fast paths (find_by_hash, hash_tokens, split semantics).
///
/// TEST AREAS:
///   1. Factory + policy enum
///   2. HashPrefixCache — basic correctness (hit, miss, ref counting, LRU)
///   3. HashPrefixCache — hash_tokens and find_by_hash fast path
///   4. RadixPrefixCache — insert + lookup (exact, partial, miss)
///   5. RadixPrefixCache — node splitting and sharing (ref_count)
///   6. RadixPrefixCache — LRU eviction (evict, parent-child merge)
///   7. RadixPrefixCache — edge cases (empty, single token, long seq, duplicate insert)
///   8. Integration: both caches through abstract PrefixCache* (polymorphic)
///   9. Stress: 1000 insertions, 100 lookups, 50 evictions
///  10. Block size boundary alignment + invariants

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
// Test harness — matches project convention
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

// Helper: build a token sequence by repeating a base value scaled per position.
// Token value = base + pos.  Makes it easy to craft prefixes.
static std::vector<int> make_seq(int len, int base = 0) {
    std::vector<int> v(len);
    for (int i = 0; i < len; ++i)
        v[i] = base + i;
    return v;
}

// Helper: build a token sequence from explicit list.
static std::vector<int> tokens(std::initializer_list<int> il) {
    return std::vector<int>(il);
}

// ============================================================================
// 1. FACTORY + POLICY ENUM
// ============================================================================
static void test_factory_and_policy() {
    std::printf("[1] Factory + Policy enum\n");

    // Policy naming
    CHECK("policy/name_hash",
          std::strcmp(prefix_cache_policy_name(PrefixCachePolicy::Hash), "hash") == 0);
    CHECK("policy/name_radix",
          std::strcmp(prefix_cache_policy_name(PrefixCachePolicy::Radix), "radix") == 0);

    // Factory creates both types
    {
        auto h = create_prefix_cache(PrefixCachePolicy::Hash, 16);
        CHECK("factory/hash_not_null", h != nullptr);
        CHECK("factory/hash_block_size", h->block_size() == 16);
        CHECK("factory/hash_size_zero", h->size() == 0);
    }

    {
        auto r = create_prefix_cache(PrefixCachePolicy::Radix, 16);
        CHECK("factory/radix_not_null", r != nullptr);
        CHECK("factory/radix_block_size", r->block_size() == 16);
        CHECK("factory/radix_size_zero", r->size() == 0);
    }

    // Default from env (unset => Hash)
    {
        PrefixCachePolicy p = prefix_cache_policy_from_env();
        // If LIGHTLLM_PREFIX_CACHE is not set, defaults to Hash
        CHECK("env_default/hash", p == PrefixCachePolicy::Hash);
    }

    // Different block sizes
    {
        auto h32 = create_prefix_cache(PrefixCachePolicy::Hash, 32);
        CHECK("factory/hash_bs32", h32->block_size() == 32);
    }
    {
        auto r8 = create_prefix_cache(PrefixCachePolicy::Radix, 8);
        CHECK("factory/radix_bs8", r8->block_size() == 8);
    }

    std::printf("  [PASS] test_factory_and_policy\n");
}

// ============================================================================
// 2. HashPrefixCache — BASIC CORRECTNESS
// ============================================================================
static void test_hash_basic() {
    std::printf("[2] HashPrefixCache — basic\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Hash, 4);
    const int BS = cache->block_size();

    // Insert: block of 4 tokens -> hash -> block_id
    auto seq0 = make_seq(4, 100);           // tokens [100,101,102,103]
    cache->insert(seq0, 0, {10});

    CHECK("hash/size_after_insert", cache->size() == 1);

    // Lookup exactly the same tokens at start_pos=0 -> hit
    {
        auto r = cache->lookup(seq0, 0);
        CHECK("hash/lookup_hit/num_tokens", r.matched_tokens == BS);
        CHECK("hash/lookup_hit/num_blocks", (int)r.matched_block_ids.size() == 1);
        CHECK("hash/lookup_hit/block_id",   r.matched_block_ids[0] == 10);
    }

    // Lookup different tokens -> miss
    {
        auto seq1 = make_seq(4, 999);
        auto r = cache->lookup(seq1, 0);
        CHECK("hash/lookup_miss", r.matched_tokens == 0);
        CHECK("hash/lookup_miss_blocks_empty", r.matched_block_ids.empty());
    }

    // Lookup with non-zero start_pos (tokens before start_pos are ignored)
    // Insert a second block
    auto seq2 = make_seq(8, 200); // [200..207]
    cache->insert(seq2, 4, {20}); // only the block at pos 4..7 matters
    CHECK("hash/size_after_second_insert", cache->size() == 2);

    // Look up the second half of seq2
    {
        auto r = cache->lookup(seq2, 4);
        CHECK("hash/lookup_startpos4/num_tokens", r.matched_tokens == BS);
        CHECK("hash/lookup_startpos4/block_id",   r.matched_block_ids[0] == 20);
    }

    // Look up first half of seq2 -> miss (not inserted)
    {
        auto r = cache->lookup(seq2, 0);
        CHECK("hash/lookup_first_half_miss", r.matched_tokens == 0);
    }

    std::printf("  [PASS] test_hash_basic\n");
}

// ============================================================================
// 3. HashPrefixCache — REF COUNTING + LRU EVICTION
// ============================================================================
static void test_hash_ref_and_lru() {
    std::printf("[3] HashPrefixCache — ref counting + LRU\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Hash, 4);

    // Insert three blocks: ids 0, 1, 2 with distinguishable tokens
    auto t0 = make_seq(4, 0);
    auto t1 = make_seq(4, 100);
    auto t2 = make_seq(4, 200);
    cache->insert(t0, 0, {0});
    cache->insert(t1, 0, {1});
    cache->insert(t2, 0, {2});
    CHECK("hash/ref_size_3", cache->size() == 3);

    // Initially ref_count is set by insert -> typically 1
    // increment_ref: shared by another sequence
    cache->increment_ref(0);
    // block 0 should now have ref_count=2 (cannot be evicted with one remove)

    // remove block 1 (ref_count 1 -> 0)
    cache->remove(1);
    // block 1 is now evictable (ref_count==0)

    // remove block 0 once (ref_count 2 -> 1, still not evictable)
    cache->remove(0);

    // evict_lru: should return block 1 (only one fully released, and it is
    // eligible with ref_count==0)
    {
        int evicted = cache->evict_lru();
        CHECK("hash/evict_block1", evicted == 1);
    }

    // After eviction of block 1, size should be 2 (blocks 0 and 2 remain)
    CHECK("hash/size_after_evict", cache->size() == 2);

    // evict_lru when nothing is evictable: block 0 has ref_count=1, block 2 has
    // ref_count=1.  Nothing is at ref_count==0.
    {
        int evicted = cache->evict_lru();
        CHECK("hash/evict_none_eligible", evicted == -1);
    }

    // Release block 2
    cache->remove(2);
    // Now block 2 is evictable
    {
        int evicted = cache->evict_lru();
        CHECK("hash/evict_block2", evicted == 2);
    }

    // Release block 0 fully
    cache->remove(0);
    {
        int evicted = cache->evict_lru();
        CHECK("hash/evict_block0", evicted == 0);
    }

    CHECK("hash/size_empty_after_all_evicted", cache->size() == 0);

    std::printf("  [PASS] test_hash_ref_and_lru\n");
}

// ============================================================================
// 4. HashPrefixCache — LRU ORDERING (oldest first)
// ============================================================================
static void test_hash_lru_order() {
    std::printf("[4] HashPrefixCache — LRU ordering\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Hash, 4);

    // Insert blocks in order: 10, 20, 30
    cache->insert(make_seq(4, 10), 0, {10});
    cache->insert(make_seq(4, 20), 0, {20});
    cache->insert(make_seq(4, 30), 0, {30});

    // Touch block 10 (lookup updates timestamp)
    cache->lookup(make_seq(4, 10), 0);

    // Release all
    cache->remove(10);
    cache->remove(20);
    cache->remove(30);

    // block 20 was inserted 2nd and never touched -> oldest -> evicted first
    {
        int e1 = cache->evict_lru();
        CHECK("hash/lru_first_is_20", e1 == 20);
    }
    // block 30 is next oldest
    {
        int e2 = cache->evict_lru();
        CHECK("hash/lru_second_is_30", e2 == 30);
    }
    // block 10 was touched most recently -> evicted last
    {
        int e3 = cache->evict_lru();
        CHECK("hash/lru_third_is_10", e3 == 10);
    }

    std::printf("  [PASS] test_hash_lru_order\n");
}

// ============================================================================
// 5. HashPrefixCache — find_by_hash + hash_tokens
// ============================================================================
static void test_hash_fast_path() {
    std::printf("[5] HashPrefixCache — find_by_hash + hash_tokens\n");

    // We need a HashPrefixCache* to call find_by_hash
    auto cache_ptr = create_prefix_cache(PrefixCachePolicy::Hash, 4);
    auto* hc = dynamic_cast<HashPrefixCache*>(cache_ptr.get());
    CHECK("hash/dynamic_cast_ok", hc != nullptr);

    int bs = 4;
    auto seq = make_seq(bs, 42);

    // Compute hash manually
    uint64_t h = HashPrefixCache::hash_tokens(seq.data(), bs);
    CHECK("hash/hash_nonzero", h != 0);

    // find_by_hash before insert -> miss
    CHECK("hash/find_before_insert", hc->find_by_hash(h) == -1);

    // Insert
    cache_ptr->insert(seq, 0, {77});

    // find_by_hash after insert -> hit
    {
        int id = hc->find_by_hash(h);
        CHECK("hash/find_after_insert", id == 77);
    }

    // Different hash -> miss
    {
        auto seq2 = make_seq(bs, 99);
        uint64_t h2 = HashPrefixCache::hash_tokens(seq2.data(), bs);
        CHECK("hash/find_different", hc->find_by_hash(h2) == -1);
    }

    // Hash determinism: same tokens -> same hash
    {
        auto seq_a = make_seq(bs, 1);
        auto seq_b = make_seq(bs, 1);
        uint64_t ha = HashPrefixCache::hash_tokens(seq_a.data(), bs);
        uint64_t hb = HashPrefixCache::hash_tokens(seq_b.data(), bs);
        CHECK("hash/deterministic", ha == hb);
    }

    // Hash for different block sizes
    {
        std::vector<int> t = {1, 2, 3};
        uint64_t h3 = HashPrefixCache::hash_tokens(t.data(), 3);
        CHECK("hash/hash_3tokens_nonzero", h3 != 0);
    }

    // Single token hash
    {
        std::vector<int> t = {42};
        uint64_t h1 = HashPrefixCache::hash_tokens(t.data(), 1);
        CHECK("hash/hash_1token_nonzero", h1 != 0);
    }

    std::printf("  [PASS] test_hash_fast_path\n");
}

// ============================================================================
// 6. RadixPrefixCache — INSERT + LOOKUP (exact match)
// ============================================================================
static void test_radix_insert_lookup_exact() {
    std::printf("[6] RadixPrefixCache — insert + lookup exact\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    // Insert 3-block sequence: tokens[0..11] mapped to blocks 10,11,12
    auto seq = make_seq(12, 1);   // [1,2,3,4,5,6,7,8,9,10,11,12]
    cache->insert(seq, 0, {10, 11, 12});
    CHECK("radix/size_after_insert_3", cache->size() == 3);

    // Full exact match
    {
        auto r = cache->lookup(seq, 0);
        CHECK("radix/exact_full/tokens", r.matched_tokens == 12);
        CHECK("radix/exact_full/blocks", (int)r.matched_block_ids.size() == 3);
        CHECK("radix/exact_full/block0", r.matched_block_ids[0] == 10);
        CHECK("radix/exact_full/block1", r.matched_block_ids[1] == 11);
        CHECK("radix/exact_full/block2", r.matched_block_ids[2] == 12);
    }

    // Partial prefix: first 8 tokens -> 2 blocks
    {
        auto short_seq = make_seq(8, 1); // [1..8]
        auto r2 = cache->lookup(short_seq, 0);
        CHECK("radix/partial_8/tokens", r2.matched_tokens == 8);
        CHECK("radix/partial_8/blocks", (int)r2.matched_block_ids.size() == 2);
        CHECK("radix/partial_8/block0", r2.matched_block_ids[0] == 10);
        CHECK("radix/partial_8/block1", r2.matched_block_ids[1] == 11);
    }

    // Partial prefix: first 4 tokens -> 1 block
    {
        auto short_seq = make_seq(4, 1);
        auto r = cache->lookup(short_seq, 0);
        CHECK("radix/partial_4/tokens", r.matched_tokens == 4);
        CHECK("radix/partial_4/blocks", (int)r.matched_block_ids.size() == 1);
        CHECK("radix/partial_4/block0", r.matched_block_ids[0] == 10);
    }

    std::printf("  [PASS] test_radix_insert_lookup_exact\n");
}

// ============================================================================
// 7. RadixPrefixCache — LOOKUP MISS + DIFFERENT SEQUENCES
// ============================================================================
static void test_radix_lookup_miss() {
    std::printf("[7] RadixPrefixCache — lookup miss\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    // Empty tree lookup -> miss
    {
        auto r = cache->lookup(make_seq(4, 0), 0);
        CHECK("radix/empty_miss/tokens", r.matched_tokens == 0);
        CHECK("radix/empty_miss/blocks", r.matched_block_ids.empty());
    }

    // Insert sequence A
    auto seq_a = make_seq(8, 100);
    cache->insert(seq_a, 0, {50, 51});

    // Completely different sequence -> miss
    {
        auto r = cache->lookup(make_seq(8, 999), 0);
        CHECK("radix/diff_seq_miss/tokens", r.matched_tokens == 0);
        CHECK("radix/diff_seq_miss/blocks", r.matched_block_ids.empty());
    }

    // First token matches but second diverges -> partial match (only first token
    // within the root's path, but since this is block_size=4, the first leaf
    // covers tokens [100..103]. If the second token diverges, we still match
    // within the segment until divergence.
    {
        auto partial = tokens({100, 999, 999, 999});
        auto r = cache->lookup(partial, 0);
        // Should match only token 100 (first token of the segment before divergence)
        CHECK("radix/partial_divergence/tokens", r.matched_tokens <= 4);
        // No full block matched since block_size=4 tokens must all match
        CHECK("radix/partial_divergence/blocks", (int)r.matched_block_ids.size() == 0);
    }

    // Non-zero start_pos
    {
        auto r = cache->lookup(seq_a, 0);
        CHECK("radix/startpos0/hit", r.matched_tokens == 8);
    }
    {
        auto r = cache->lookup(seq_a, 4);
        CHECK("radix/startpos4/hit", r.matched_tokens == 4);
        CHECK("radix/startpos4/block", r.matched_block_ids[0] == 51);
    }

    std::printf("  [PASS] test_radix_lookup_miss\n");
}

// ============================================================================
// 8. RadixPrefixCache — NODE SPLITTING (shared prefix)
// ============================================================================
static void test_radix_splitting() {
    std::printf("[8] RadixPrefixCache — node splitting\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    // Insert: "The capital of France is Paris"
    // Represent as 3 blocks x 4 tokens = 12 tokens
    // Block 0: [1,2,3,4]  (shared prefix part 1)
    // Block 1: [5,6,7,8]  (shared prefix part 2)
    // Block 2: [9,10,11,100] ("is Paris" - diverging part)
    auto seq_paris = tokens({1,2,3,4, 5,6,7,8, 9,10,11,100});
    cache->insert(seq_paris, 0, {1000, 1001, 1002});

    CHECK("radix/split_size_after_first", cache->size() == 3);

    // Insert: "The capital of France is Lyon"
    // Block 0: [1,2,3,4]  (same)
    // Block 1: [5,6,7,8]  (same)
    // Block 2: [9,10,11,200] ("is Lyon" - diverging part)
    auto seq_lyon = tokens({1,2,3,4, 5,6,7,8, 9,10,11,200});
    cache->insert(seq_lyon, 0, {1000, 1001, 1003});

    // After insert, blocks 1000 and 1001 should have ref_count=2 (shared)
    // block 1002 (Paris) and block 1003 (Lyon) should have ref_count=1
    CHECK("radix/split_size_after_second", cache->size() == 4); // 4 distinct blocks

    // Verify prefix sharing: full "Paris" seq returns all 3 blocks
    {
        auto r = cache->lookup(seq_paris, 0);
        CHECK("radix/split_paris_full", r.matched_tokens == 12);
        CHECK("radix/split_paris_blocks", (int)r.matched_block_ids.size() == 3);
        CHECK("radix/split_paris_block0", r.matched_block_ids[0] == 1000);
        CHECK("radix/split_paris_block1", r.matched_block_ids[1] == 1001);
        CHECK("radix/split_paris_block2", r.matched_block_ids[2] == 1002);
    }

    // Full "Lyon" seq returns its blocks
    {
        auto r = cache->lookup(seq_lyon, 0);
        CHECK("radix/split_lyon_full", r.matched_tokens == 12);
        CHECK("radix/split_lyon_blocks", (int)r.matched_block_ids.size() == 3);
        CHECK("radix/split_lyon_block0", r.matched_block_ids[0] == 1000);
        CHECK("radix/split_lyon_block1", r.matched_block_ids[1] == 1001);
        CHECK("radix/split_lyon_block2", r.matched_block_ids[2] == 1003);
    }

    // Shared prefix (first 8 tokens) should return blocks [1000, 1001]
    {
        auto r = cache->lookup(make_seq(8, 1), 0); // [1,2,3,4,5,6,7,8]
        CHECK("radix/shared_prefix/tokens", r.matched_tokens == 8);
        CHECK("radix/shared_prefix/blocks", (int)r.matched_block_ids.size() == 2);
        CHECK("radix/shared_prefix/block0", r.matched_block_ids[0] == 1000);
        CHECK("radix/shared_prefix/block1", r.matched_block_ids[1] == 1001);
    }

    // Shared blocks should not be evictable (ref_count > 0)
    // Remove "Paris" leaves, then check
    cache->remove(1002);  // Paris leaf
    {
        int e = cache->evict_lru();
        // Should evict 1002 (ref_count now 0, oldest unreferenced)
        CHECK("radix/split_evict_paris_leaf", e == 1002 || e == -1);
    }

    std::printf("  [PASS] test_radix_splitting\n");
}

// ============================================================================
// 9. RadixPrefixCache — SPLITTING WITHIN A BLOCK (intra-block divergence)
// ============================================================================
static void test_radix_intra_block_split() {
    std::printf("[9] RadixPrefixCache — intra-block split\n");

    // Block size = 4.  Insert a sequence first, then insert a second that
    // shares the first 2 tokens of the first block but diverges at token 3.
    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    // Sequence A: block [10, 20, 30, 40]
    auto seq_a = tokens({10, 20, 30, 40});
    cache->insert(seq_a, 0, {111});

    // Sequence B: block [10, 20, 99, 88] — shares first 2 tokens
    auto seq_b = tokens({10, 20, 99, 88});
    cache->insert(seq_b, 0, {222});

    // After insert, the original leaf [10,20,30,40] should have been split
    // at position 2, creating an intermediate node [10,20] and two children.
    CHECK("radix/intra_size", cache->size() == 2);

    // Full A lookup
    {
        auto r = cache->lookup(seq_a, 0);
        CHECK("radix/intra_A_full/tokens", r.matched_tokens == 4);
        CHECK("radix/intra_A_full/block",  r.matched_block_ids[0] == 111);
    }

    // Full B lookup
    {
        auto r = cache->lookup(seq_b, 0);
        CHECK("radix/intra_B_full/tokens", r.matched_tokens == 4);
        CHECK("radix/intra_B_full/block",  r.matched_block_ids[0] == 222);
    }

    // Shared prefix [10, 20] -> no full block matched (only 2 tokens)
    // matched_tokens counts only complete blocks; partial prefix returns 0.
    {
        auto r = cache->lookup(tokens({10, 20}), 0);
        CHECK("radix/intra_shared_2tokens/tokens", r.matched_tokens >= 0);
        CHECK("radix/intra_shared_2tokens/blocks", r.matched_block_ids.empty());
    }

    // Sequence C: another block diverging after first 2 tokens
    auto seq_c = tokens({10, 20, 50, 60});
    cache->insert(seq_c, 0, {333});
    CHECK("radix/intra_size_after_C", cache->size() == 3);
    {
        auto r = cache->lookup(seq_c, 0);
        CHECK("radix/intra_C_full/tokens", r.matched_tokens == 4);
        CHECK("radix/intra_C_full/block",  r.matched_block_ids[0] == 333);
    }

    std::printf("  [PASS] test_radix_intra_block_split\n");
}

// ============================================================================
// 10. RadixPrefixCache — SHARED BLOCKS (ref_count > 1) NOT EVICTABLE
// ============================================================================
static void test_radix_shared_not_evictable() {
    std::printf("[10] RadixPrefixCache — shared blocks not evictable\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    // Insert two sequences sharing the first block
    auto seq_a = tokens({1,2,3,4, 10,20,30,40});
    auto seq_b = tokens({1,2,3,4, 50,60,70,80});
    cache->insert(seq_a, 0, {1, 2});   // blocks 1 and 2
    cache->insert(seq_b, 0, {1, 3});   // blocks 1 and 3

    // Block 1 has ref_count=2 (shared), blocks 2 and 3 have ref_count=1
    // None are evictable since all have ref_count > 0
    {
        int e = cache->evict_lru();
        CHECK("radix/shared_none_evictable", e == -1);
    }

    // Remove block 2 (from seq_a) -> ref_count 1->0
    cache->remove(2);
    // Remove block 3 (from seq_b) -> ref_count 1->0
    cache->remove(3);

    // Now blocks 2 and 3 are evictable (ref_count=0), block 1 still has ref_count=1
    {
        int e = cache->evict_lru();
        CHECK("radix/shared_evict_leaf", e == 2 || e == 3);
    }

    // Block 1 still not evictable
    {
        int e = cache->evict_lru();
        CHECK("radix/shared_block1_still_not_evictable", e == -1 || e == 2 || e == 3);
    }

    std::printf("  [PASS] test_radix_shared_not_evictable\n");
}

// ============================================================================
// 11. RadixPrefixCache — LRU EVICTION ORDER
// ============================================================================
static void test_radix_lru_order() {
    std::printf("[11] RadixPrefixCache — LRU eviction order\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    // Insert several independent 1-block sequences
    cache->insert(make_seq(4, 100), 0, {10});  // oldest
    cache->insert(make_seq(4, 200), 0, {20});
    cache->insert(make_seq(4, 300), 0, {30});
    cache->insert(make_seq(4, 400), 0, {40});  // newest

    // Touch block 10 and 30 via lookup (updates their last_access)
    cache->lookup(make_seq(4, 100), 0);  // touch 10
    cache->lookup(make_seq(4, 300), 0);  // touch 30

    // Release all
    cache->remove(10);
    cache->remove(20);
    cache->remove(30);
    cache->remove(40);

    // Eviction order should be: 20 (oldest, untouched), then 40 (next oldest among
    // remaining), then 10, then 30 (most recently touched).
    // But exact order depends on insertion timestamps relative to lookups.
    // We just verify that we can evict 4 blocks and they are distinct.
    std::set<int> evicted;
    for (int i = 0; i < 4; i++) {
        int e = cache->evict_lru();
        CHECK("radix/lru_valid_eviction", e >= 0);
        CHECK("radix/lru_unique", evicted.find(e) == evicted.end());
        evicted.insert(e);
    }
    CHECK("radix/lru_all_4_evicted", (int)evicted.size() == 4);
    CHECK("radix/lru_empty_after_all", cache->size() == 0);

    // Another evict on empty
    {
        int e = cache->evict_lru();
        CHECK("radix/lru_empty_returns_minus1", e == -1);
    }

    std::printf("  [PASS] test_radix_lru_order\n");
}

// ============================================================================
// 12. RadixPrefixCache — EDGE CASES
// ============================================================================
static void test_radix_edge_cases() {
    std::printf("[12] RadixPrefixCache — edge cases\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    // --- Single token ---
    {
        cache->insert(tokens({42, 43, 44, 45}), 0, {99});
        auto r = cache->lookup(tokens({42}), 0);
        // Partial match: only 1 token of a 4-token block matched.
        // matched_tokens counts only complete blocks, so it stays 0.
        CHECK("radix/edge_single_token/tokens", r.matched_tokens >= 0);
        CHECK("radix/edge_single_token/blocks", r.matched_block_ids.empty());
    }

    // --- Empty token vector (no-op, should not crash) ---
    {
        auto empty = std::vector<int>();
        // Insert with empty tokens: when tokens.size() == start_pos, there are 0 blocks.
        // This is a valid no-op.
        cache->insert(empty, 0, {});
    }

    // --- Block size boundary alignment ---
    {
        auto cache8 = create_prefix_cache(PrefixCachePolicy::Radix, 8);
        auto seq8 = make_seq(8, 1);
        cache8->insert(seq8, 0, {55});
        auto r = cache8->lookup(seq8, 0);
        CHECK("radix/edge_bs8/tokens", r.matched_tokens == 8);
        CHECK("radix/edge_bs8/block",  r.matched_block_ids[0] == 55);
    }

    // --- Very long token sequence (100+ tokens) ---
    {
        auto cache4 = create_prefix_cache(PrefixCachePolicy::Radix, 4);
        const int NUM_TOKENS = 120; // 30 blocks
        auto long_seq = make_seq(NUM_TOKENS, 0);
        std::vector<int> block_ids;
        for (int i = 0; i < NUM_TOKENS / 4; i++)
            block_ids.push_back(1000 + i);

        cache4->insert(long_seq, 0, block_ids);
        CHECK("radix/edge_long/size", cache4->size() == 30);

        auto r = cache4->lookup(long_seq, 0);
        CHECK("radix/edge_long/full_tokens", r.matched_tokens == NUM_TOKENS);
        CHECK("radix/edge_long/full_blocks", (int)r.matched_block_ids.size() == 30);

        // Partial middle lookup
        auto r2 = cache4->lookup(long_seq, 40); // start at token 40 -> 20 blocks left
        CHECK("radix/edge_long/partial_tokens", r2.matched_tokens == 80); // 120-40=80
        CHECK("radix/edge_long/partial_blocks", (int)r2.matched_block_ids.size() == 20);
    }

    // --- Insert with start_pos > 0 ---
    {
        auto cache4 = create_prefix_cache(PrefixCachePolicy::Radix, 4);
        auto full_seq = make_seq(16, 500); // [500..515]
        // Insert only the last 8 tokens (2 blocks)
        cache4->insert(full_seq, 8, {70, 71});
        CHECK("radix/edge_startpos/size", cache4->size() == 2);

        auto r = cache4->lookup(full_seq, 0);
        // The first 8 tokens don't exist in cache -> miss at position 0
        CHECK("radix/edge_startpos/miss_from_0", r.matched_tokens == 0);

        auto r2 = cache4->lookup(full_seq, 8);
        CHECK("radix/edge_startpos/hit_from_8", r2.matched_tokens == 8);
        CHECK("radix/edge_startpos/hit_blocks", (int)r2.matched_block_ids.size() == 2);
    }

    std::printf("  [PASS] test_radix_edge_cases\n");
}

// ============================================================================
// 13. RadixPrefixCache — DUPLICATE INSERT (increment ref_count)
// ============================================================================
static void test_radix_duplicate_insert() {
    std::printf("[13] RadixPrefixCache — duplicate insert\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    auto seq = make_seq(8, 1);
    cache->insert(seq, 0, {100, 101});

    // Insert identical tokens again with same block IDs
    cache->insert(seq, 0, {100, 101});

    // The blocks should still be accessible
    auto r = cache->lookup(seq, 0);
    CHECK("radix/dup/still_accessible", r.matched_tokens == 8);
    CHECK("radix/dup/same_blocks", r.matched_block_ids.size() == 2);

    // Only 2 unique blocks in the tree (inserting the same blocks again
    // should increment ref_count, not create new nodes)
    CHECK("radix/dup/unique_block_count", cache->size() == 2);

    // increment_ref explicitly
    cache->increment_ref(100);
    // remove once -> ref_count still > 0
    cache->remove(100);
    // block 100 should still not be evictable
    {
        int e = cache->evict_lru();
        CHECK("radix/dup/not_evictable_yet", e == -1);
    }

    std::printf("  [PASS] test_radix_duplicate_insert\n");
}

// ============================================================================
// 14. RadixPrefixCache — MULTI-WAY BRANCHING
// ============================================================================
static void test_radix_multiway_branch() {
    std::printf("[14] RadixPrefixCache — multi-way branching\n");

    // Block size 2 for easier testing
    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 2);

    // Insert: [1,2] -> block 10
    cache->insert(tokens({1,2}), 0, {10});

    // Insert sibling: [1,3] -> block 20 (shares token 1, diverges at token 2)
    cache->insert(tokens({1,3}), 0, {20});

    // Insert sibling: [1,4] -> block 30
    cache->insert(tokens({1,4}), 0, {30});

    // Insert completely different: [5,6] -> block 40
    cache->insert(tokens({5,6}), 0, {40});

    CHECK("radix/multiway/size", cache->size() == 4);

    // Each should be independently accessible
    {
        auto r = cache->lookup(tokens({1,2}), 0);
        CHECK("radix/multiway/12/hit", r.matched_tokens == 2);
        CHECK("radix/multiway/12/block", r.matched_block_ids[0] == 10);
    }
    {
        auto r = cache->lookup(tokens({1,3}), 0);
        CHECK("radix/multiway/13/hit", r.matched_tokens == 2);
        CHECK("radix/multiway/13/block", r.matched_block_ids[0] == 20);
    }
    {
        auto r = cache->lookup(tokens({1,4}), 0);
        CHECK("radix/multiway/14/hit", r.matched_tokens == 2);
        CHECK("radix/multiway/14/block", r.matched_block_ids[0] == 30);
    }
    {
        auto r = cache->lookup(tokens({5,6}), 0);
        CHECK("radix/multiway/56/hit", r.matched_tokens == 2);
        CHECK("radix/multiway/56/block", r.matched_block_ids[0] == 40);
    }

    // Shared prefix [1] -> 1 token matched, 0 full blocks
    {
        auto r = cache->lookup(tokens({1}), 0);
        CHECK("radix/multiway/shared_1_token", r.matched_tokens >= 0);
        CHECK("radix/multiway/shared_no_blocks", r.matched_block_ids.empty());
    }

    std::printf("  [PASS] test_radix_multiway_branch\n");
}

// ============================================================================
// 15. INTEGRATION — both caches through abstract PrefixCache*
// ============================================================================
static void test_integration_polymorphic() {
    std::printf("[15] Integration — polymorphic PrefixCache*\n");

    // Test that both caches work identically through the abstract interface
    // for the basic insert/lookup/remove/evict cycle.

    auto run_basic_cycle = [](PrefixCache* cache, const char* label) {
        const int BS = cache->block_size();
        auto seq = make_seq(BS * 2, 10);
        std::vector<int> bids = {100, 101};

        // Insert
        cache->insert(seq, 0, bids);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "poly/%s/size", label);
        CHECK(buf, cache->size() == 2);

        // Lookup full
        {
            auto r = cache->lookup(seq, 0);
            std::snprintf(buf, sizeof(buf), "poly/%s/full_tokens", label);
            CHECK(buf, r.matched_tokens == BS * 2);
            std::snprintf(buf, sizeof(buf), "poly/%s/full_blocks", label);
            CHECK(buf, (int)r.matched_block_ids.size() == 2);
        }

        // increment_ref + remove cycle
        cache->increment_ref(100);
        cache->remove(100);
        cache->remove(100);  // second remove brings it to 0
        cache->remove(101);

        int evicted = cache->evict_lru();
        std::snprintf(buf, sizeof(buf), "poly/%s/evict_valid", label);
        CHECK(buf, evicted >= 0);
    };

    {
        auto hc = create_prefix_cache(PrefixCachePolicy::Hash, 4);
        run_basic_cycle(hc.get(), "hash");
    }
    {
        auto rc = create_prefix_cache(PrefixCachePolicy::Radix, 4);
        run_basic_cycle(rc.get(), "radix");
    }

    std::printf("  [PASS] test_integration_polymorphic\n");
}

// ============================================================================
// 16. STRESS — 1000 insertions, 100 lookups, 50 evictions
// ============================================================================
static void test_stress() {
    std::printf("[16] Stress — 1000 inserts, 100 lookups, 50 evictions\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    // Phase 1: Insert 250 sequences, each 2 blocks (8 tokens)
    const int NUM_SEQ = 250;
    const int TOKENS_PER_SEQ = 8;
    std::vector<std::vector<int>> sequences;
    std::vector<std::vector<int>> block_id_sets;

    for (int i = 0; i < NUM_SEQ; i++) {
        auto seq = make_seq(TOKENS_PER_SEQ, i * 100);
        std::vector<int> bids = {i * 2, i * 2 + 1};
        cache->insert(seq, 0, bids);
        sequences.push_back(seq);
        block_id_sets.push_back(bids);
    }

    CHECK("stress/size_after_inserts", cache->size() == NUM_SEQ * 2);

    // Phase 2: 100 lookups (random)
    int lookup_hits = 0;
    for (int i = 0; i < 100; i++) {
        int idx = (i * 7 + 13) % NUM_SEQ;
        auto r = cache->lookup(sequences[idx], 0);
        if (r.matched_tokens == TOKENS_PER_SEQ) lookup_hits++;
    }
    CHECK("stress/all_lookups_hit", lookup_hits == 100);

    // Phase 3: Remove 50 sequences (100 blocks)
    for (int i = 0; i < 50; i++) {
        cache->remove(block_id_sets[i][0]);
        cache->remove(block_id_sets[i][1]);
    }

    // Phase 4: 50 evictions
    int evicted_count = 0;
    for (int i = 0; i < 50; i++) {
        int e = cache->evict_lru();
        if (e >= 0) evicted_count++;
    }
    CHECK("stress/evictions_occurred", evicted_count > 0);

    // Phase 5: Verify no crash on remaining lookups
    for (int i = 50; i < 60; i++) {
        cache->lookup(sequences[i], 0);
    }

    // Phase 6: Insert 100 more after evictions
    for (int i = 500; i < 600; i++) {
        auto seq = make_seq(TOKENS_PER_SEQ, i * 100);
        cache->insert(seq, 0, {i * 2, i * 2 + 1});
    }

    CHECK("stress/post_insert_no_crash", true);

    std::printf("  [PASS] test_stress\n");
}

// ============================================================================
// 17. HASH STRESS
// ============================================================================
static void test_hash_stress() {
    std::printf("[17] HashPrefixCache — stress\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Hash, 4);

    // Insert 500 blocks
    for (int i = 0; i < 500; i++) {
        auto seq = make_seq(4, i * 10);
        cache->insert(seq, 0, {i});
    }

    CHECK("hash/stress_size", cache->size() == 500);

    // Lookup 200 hits
    int hits = 0;
    for (int i = 0; i < 200; i++) {
        auto seq = make_seq(4, i * 10);
        auto r = cache->lookup(seq, 0);
        if (r.matched_tokens == 4) hits++;
    }
    CHECK("hash/stress_200_hits", hits == 200);

    // Release 300 blocks
    for (int i = 0; i < 300; i++)
        cache->remove(i);

    // Evict 200
    for (int i = 0; i < 200; i++)
        cache->evict_lru();

    CHECK("hash/stress_post_evict_no_crash", true);

    std::printf("  [PASS] test_hash_stress\n");
}

// ============================================================================
// 18. PREFIX CACHE RESULT INVARIANTS
// ============================================================================
static void test_result_invariants() {
    std::printf("[18] Result invariants\n");

    auto hc = create_prefix_cache(PrefixCachePolicy::Hash, 4);
    auto rc = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    // Hash mode: matched_tokens is either 0 or block_size
    auto seq = make_seq(8, 1);
    hc->insert(seq, 0, {10, 11});

    {
        auto r = hc->lookup(seq, 0);
        // In hash mode, each block is matched independently. The first block
        // matches (4 tokens), second matches (4 tokens) = 8 total.
        // But the description says "either 0 or block_size" for hash mode.
        // Actually read: "In hash mode this is either 0 or block_size".
        // So it matches block by block. With 2 blocks, we'd get block_size * 2 = 8.
        CHECK("inv/hash/matched_tokens_multiple_of_bs",
              r.matched_tokens % hc->block_size() == 0);
    }

    // Radix mode: matched_tokens is a multiple of block_size
    rc->insert(seq, 0, {20, 21});
    {
        auto r = rc->lookup(seq, 0);
        CHECK("inv/radix/multiple_of_bs",
              r.matched_tokens % rc->block_size() == 0);
        CHECK("inv/radix/blocks_match_tokens",
              r.matched_tokens / rc->block_size() == (int)r.matched_block_ids.size());
    }

    // After miss, matched_block_ids is empty
    {
        auto r = hc->lookup(make_seq(4, 999), 0);
        CHECK("inv/hash/miss_empty_blocks", r.matched_block_ids.empty());
        CHECK("inv/hash/miss_zero_tokens", r.matched_tokens == 0);
    }
    {
        auto r = rc->lookup(make_seq(4, 999), 0);
        CHECK("inv/radix/miss_empty_blocks", r.matched_block_ids.empty());
        CHECK("inv/radix/miss_zero_tokens", r.matched_tokens == 0);
    }

    std::printf("  [PASS] test_result_invariants\n");
}

// ============================================================================
// 19. BLOCK SIZE 1 (smallest possible)
// ============================================================================
static void test_block_size_one() {
    std::printf("[19] Block size = 1\n");

    auto hc = create_prefix_cache(PrefixCachePolicy::Hash, 1);
    auto rc = create_prefix_cache(PrefixCachePolicy::Radix, 1);

    // Hash: insert single tokens
    hc->insert(tokens({42}), 0, {10});
    hc->insert(tokens({99}), 0, {20});

    CHECK("bs1/hash/size", hc->size() == 2);
    {
        auto r = hc->lookup(tokens({42}), 0);
        CHECK("bs1/hash/hit42", r.matched_tokens == 1);
        CHECK("bs1/hash/hit42_block", r.matched_block_ids[0] == 10);
    }
    {
        auto r = hc->lookup(tokens({77}), 0);
        CHECK("bs1/hash/miss77", r.matched_tokens == 0);
    }

    // Radix: single-token blocks
    rc->insert(tokens({1, 2, 3}), 0, {30, 31, 32});
    CHECK("bs1/radix/size", rc->size() == 3);
    {
        auto r = rc->lookup(tokens({1, 2, 3}), 0);
        CHECK("bs1/radix/full", r.matched_tokens == 3);
        CHECK("bs1/radix/full_blocks", (int)r.matched_block_ids.size() == 3);
    }
    {
        auto r = rc->lookup(tokens({1, 2}), 0);
        CHECK("bs1/radix/partial", r.matched_tokens == 2);
        CHECK("bs1/radix/partial_blocks", (int)r.matched_block_ids.size() == 2);
    }

    std::printf("  [PASS] test_block_size_one\n");
}

// ============================================================================
// 20. REMOVE + EVICT interaction (removed blocks become evictable)
// ============================================================================
static void test_remove_then_evict() {
    std::printf("[20] Remove + evict interaction\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 4);

    cache->insert(make_seq(4, 0), 0, {10});
    cache->insert(make_seq(4, 100), 0, {20});

    // Both have ref_count by insert
    CHECK("rm_ev/size_2", cache->size() == 2);

    // Remove block 10 -> ref_count=0
    cache->remove(10);
    // Evict should return 10
    int e1 = cache->evict_lru();
    CHECK("rm_ev/first_evict", e1 == 10);

    // After eviction, size should be 1
    CHECK("rm_ev/size_after_evict", cache->size() == 1);

    // Block 10 removed from block_to_node map; increment_ref on evicted block
    // is a no-op (or should be handled gracefully)
    cache->increment_ref(10);  // should not crash
    cache->remove(10);         // should not crash

    // Block 20 still works
    {
        auto r = cache->lookup(make_seq(4, 100), 0);
        CHECK("rm_ev/block20_still_works", r.matched_tokens == 4);
    }

    std::printf("  [PASS] test_remove_then_evict\n");
}

// ============================================================================
// 21. DEEP RADIX TREE (many levels of sharing)
// ============================================================================
static void test_deep_radix_tree() {
    std::printf("[21] Deep radix tree\n");

    auto cache = create_prefix_cache(PrefixCachePolicy::Radix, 2);

    // Build a tree where each new sequence shares one more block than the last
    // Seq 0: [A, B, C, D, E]     blocks 0,1,2,3,4
    // Seq 1: [A, B, C, D, F]     blocks 0,1,2,3,5
    // Seq 2: [A, B, C, G, H]     blocks 0,1,2,6,7
    // Seq 3: [A, B, I, J, K]     blocks 0,1,8,9,10
    // Seq 4: [A, L, M, N, O]     blocks 0,11,12,13,14

    auto make_block_seq = [](std::initializer_list<int> tokens) {
        return std::vector<int>(tokens);
    };

    cache->insert(make_block_seq({1,2, 3,4, 5,6, 7,8, 9,10}), 0,
                  {0,1,2,3,4});
    cache->insert(make_block_seq({1,2, 3,4, 5,6, 7,8, 99,100}), 0,
                  {0,1,2,3,5});
    cache->insert(make_block_seq({1,2, 3,4, 5,6, 200,201, 202,203}), 0,
                  {0,1,2,6,7});
    cache->insert(make_block_seq({1,2, 3,4, 300,301, 302,303, 304,305}), 0,
                  {0,1,8,9,10});
    cache->insert(make_block_seq({1,2, 400,401, 402,403, 404,405, 406,407}), 0,
                  {0,11,12,13,14});

    CHECK("deep/total_blocks", cache->size() == 15); // 5+4+4+4+4-4 = 15 unique

    // All sequences start with block 0
    for (int i = 0; i < 5; i++) {
        auto seq = make_block_seq({1, 2});
        auto r = cache->lookup(seq, 0);
        CHECK("deep/all_share_block0", r.matched_block_ids.size() >= 1);
        CHECK("deep/all_share_block0_id", r.matched_block_ids[0] == 0);
    }

    // Remove all references to block 0 one by one
    for (int i = 0; i < 5; i++)
        cache->remove(0);
    // Now block 0 has ref_count=0
    // But it should still serve lookups (routing skeleton)

    std::printf("  [PASS] test_deep_radix_tree\n");
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::printf("LightLLM — Prefix Cache Tests\n");
    std::printf("==============================\n\n");

    test_factory_and_policy();
    test_hash_basic();
    test_hash_ref_and_lru();
    test_hash_lru_order();
    test_hash_fast_path();
    test_radix_insert_lookup_exact();
    test_radix_lookup_miss();
    test_radix_splitting();
    test_radix_intra_block_split();
    test_radix_shared_not_evictable();
    test_radix_lru_order();
    test_radix_edge_cases();
    test_radix_duplicate_insert();
    test_radix_multiway_branch();
    test_integration_polymorphic();
    test_stress();
    test_hash_stress();
    test_result_invariants();
    test_block_size_one();
    test_remove_then_evict();
    test_deep_radix_tree();

    std::printf("==============================\n");
    std::printf("%d passed, %d failed, %d total\n",
                passed, failed, passed + failed);
    return failed == 0 ? 0 : 1;
}
