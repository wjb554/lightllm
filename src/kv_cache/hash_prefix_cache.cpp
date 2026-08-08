/// HashPrefixCache implementation.
///
/// Each block is hashed independently (FNV-1a 64-bit over block_size tokens).
/// No cross-block or partial-block prefix matching.
///
/// LRU is tracked through a vector of (block_id, last_access) entries.
/// For typical pool sizes (hundreds of blocks) a linear scan on evict_lru()
/// is fast enough.

#include "lightllm/kv_cache/prefix_cache.h"

#include <cstdlib>   // std::getenv
#include <cstring>   // std::strcmp
#include <stdexcept>

namespace lightllm {
namespace kv_cache {

// ============================================================================
// Hash function — FNV-1a 64-bit
// ============================================================================

uint64_t HashPrefixCache::hash_tokens(const int* tokens, int n) {
    uint64_t h = 14695981039346656037ULL;  // FNV offset basis
    for (int i = 0; i < n; i++) {
        h ^= static_cast<uint64_t>(static_cast<unsigned int>(tokens[i]));
        h *= 1099511628211ULL;             // FNV prime
    }
    return h;
}

// ============================================================================
// Construction
// ============================================================================

HashPrefixCache::HashPrefixCache(int block_size)
    : block_size_(block_size) {}

// ============================================================================
// lookup()
// ============================================================================

PrefixCacheResult HashPrefixCache::lookup(const std::vector<int>& tokens,
                                           int start_pos) {
    PrefixCacheResult result;

    int n_tokens = static_cast<int>(tokens.size());
    int pos = start_pos;

    // Walk forward block-by-block.  Hash mode only matches at exact block
    // boundaries; stop on the first miss.
    while (pos + block_size_ <= n_tokens) {
        uint64_t h = hash_tokens(tokens.data() + pos, block_size_);
        auto it = cache_.find(h);
        if (it == cache_.end()) {
            break;  // first miss terminates the match
        }

        int block_id = it->second;
        result.matched_block_ids.push_back(block_id);
        result.matched_tokens += block_size_;
        pos += block_size_;

        // Touch LRU timestamp so frequently-matched blocks resist eviction.
        auto li = lru_index_.find(block_id);
        if (li != lru_index_.end()) {
            lru_list_[li->second].last_access = ++access_counter_;
        }
    }

    return result;
}

// ============================================================================
// insert()
// ============================================================================

void HashPrefixCache::insert(const std::vector<int>& tokens,
                              int start_pos,
                              const std::vector<int>& block_ids) {
    int n_tokens = static_cast<int>(tokens.size());
    int num_blocks = static_cast<int>(block_ids.size());

    // When prompts are split across prefill chunks, some blocks may not
    // yet be fully filled (e.g., 20-token prompt with chunk_size=16:
    // block 0 is full, block 1 is pending).  Only insert the blocks the
    // caller actually provides; skip validation against total token span.
    (void)n_tokens;  // unused after relaxing the check

    for (int i = 0; i < num_blocks; i++) {
        int pos = start_pos + i * block_size_;
        // Guard: skip blocks that extend past the available token array.
        // This can happen when the last block is partially filled and the
        // caller deliberately passes only the fully-filled blocks.
        if (pos + block_size_ > n_tokens) continue;
        uint64_t h = hash_tokens(tokens.data() + pos, block_size_);
        int block_id = block_ids[i];

        // If this block was previously cached under a different hash,
        // clean up the stale hash mapping.
        auto old_bh = block_to_hash_.find(block_id);
        if (old_bh != block_to_hash_.end() && old_bh->second != h) {
            auto stale = cache_.find(old_bh->second);
            if (stale != cache_.end() && stale->second == block_id) {
                cache_.erase(stale);
            }
        }

        // Store forward and reverse mappings.
        cache_[h] = block_id;
        block_to_hash_[block_id] = h;
        ref_counts_[block_id] = 1;  // one sequence references this block

        // Insert or update the LRU entry.
        auto li = lru_index_.find(block_id);
        if (li != lru_index_.end()) {
            lru_list_[li->second].last_access = ++access_counter_;
        } else {
            lru_index_[block_id] = lru_list_.size();
            lru_list_.push_back({block_id, ++access_counter_});
        }
    }
}

// ============================================================================
// remove() — mark a block as unreferenced (ref_count = 0)
// ============================================================================

void HashPrefixCache::remove(int block_id) {
    auto rc = ref_counts_.find(block_id);
    if (rc == ref_counts_.end()) {
        return;  // not in cache, nothing to do
    }

    // Decrement.  Once it reaches 0 the entry is evictable.
    if (rc->second > 0) {
        rc->second--;
    }

    // NOTE: deliberately do NOT touch LRU timestamp here.
    // EViction order must reflect when blocks were last USEFULLY accessed
    // (via lookup / increment_ref), not when they were released.  Touching
    // on remove() would make eviction order depend on release order rather
    // than usage order, defeating the purpose of LRU.
}

// ============================================================================
// evict_lru() — physically remove the oldest evictable entry
// ============================================================================

int HashPrefixCache::evict_lru() {
    int best_block_id = -1;
    uint64_t best_ts = UINT64_MAX;
    size_t best_idx = 0;
    bool found = false;

    // Linear scan for the entry with ref_count == 0 and smallest last_access.
    // For pool sizes of hundreds of blocks this is fast enough.
    for (size_t i = 0; i < lru_list_.size(); i++) {
        int bid = lru_list_[i].block_id;
        auto rc = ref_counts_.find(bid);
        if (rc == ref_counts_.end() || rc->second > 0) {
            continue;  // still referenced, or already removed
        }

        if (lru_list_[i].last_access < best_ts) {
            best_ts = lru_list_[i].last_access;
            best_block_id = bid;
            best_idx = i;
            found = true;
        }
    }

    if (!found) {
        return -1;  // no evictable entries
    }

    // ---- Erase the entry from all data structures ----

    // Remove from hash cache (only if it still maps to this block).
    auto bh = block_to_hash_.find(best_block_id);
    if (bh != block_to_hash_.end()) {
        auto ci = cache_.find(bh->second);
        if (ci != cache_.end() && ci->second == best_block_id) {
            cache_.erase(ci);
        }
        block_to_hash_.erase(bh);
    }

    // Remove ref_count tracking.
    ref_counts_.erase(best_block_id);

    // Remove from LRU list (swap-with-back + pop).
    lru_index_.erase(best_block_id);
    if (best_idx != lru_list_.size() - 1) {
        lru_list_[best_idx] = lru_list_.back();
        lru_index_[lru_list_[best_idx].block_id] = best_idx;
    }
    lru_list_.pop_back();

    return best_block_id;
}

// ============================================================================
// size()
// ============================================================================

size_t HashPrefixCache::size() const {
    return cache_.size();
}

// ============================================================================
// increment_ref() — bump ref_count when a new sequence shares a cached block
// ============================================================================

void HashPrefixCache::increment_ref(int block_id) {
    auto rc = ref_counts_.find(block_id);
    if (rc != ref_counts_.end()) {
        rc->second++;
    }

    // Touch LRU.
    auto li = lru_index_.find(block_id);
    if (li != lru_index_.end()) {
        lru_list_[li->second].last_access = ++access_counter_;
    }
}

// ============================================================================
// find_by_hash() — direct single-block lookup (backward-compatible fast path)
// ============================================================================

int HashPrefixCache::find_by_hash(uint64_t token_hash) const {
    auto it = cache_.find(token_hash);
    return (it != cache_.end()) ? it->second : -1;
}

// ============================================================================
// Free functions — policy parsing and factory
// ============================================================================

PrefixCachePolicy prefix_cache_policy_from_env() {
    const char* val = std::getenv("LIGHTLLM_PREFIX_CACHE");
    if (val == nullptr) {
        return PrefixCachePolicy::Hash;
    }
    if (std::strcmp(val, "radix") == 0 || std::strcmp(val, "Radix") == 0) {
        return PrefixCachePolicy::Radix;
    }
    // Default to Hash for any unrecognised value.
    return PrefixCachePolicy::Hash;
}

const char* prefix_cache_policy_name(PrefixCachePolicy policy) {
    switch (policy) {
        case PrefixCachePolicy::Hash:  return "hash";
        case PrefixCachePolicy::Radix: return "radix";
        default:                       return "unknown";
    }
}

}  // namespace kv_cache
}  // namespace lightllm
