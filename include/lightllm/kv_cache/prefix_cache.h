// ============================================================================
// File: include/lightllm/kv_cache/prefix_cache.h
// ============================================================================

#pragma once
/// LightLLM Configurable Prefix Cache System
///
/// Two implementations:
///   HashPrefixCache  — single-level hash (current behavior, default)
///   RadixPrefixCache — SGLang-style radix tree (longest-prefix matching)
///
/// Selected at construction time via PrefixCachePolicy enum, configurable
/// through the LIGHTLLM_PREFIX_CACHE environment variable ("hash" / "radix").

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lightllm {
namespace kv_cache {

// ============================================================================
// PrefixCacheResult — returned by lookup()
// ============================================================================
struct PrefixCacheResult {
    /// Number of tokens matched.  In hash mode this is either 0 or block_size;
    /// in radix mode it is a multiple of block_size up to the full sequence.
    int matched_tokens = 0;

    /// Physical block IDs corresponding to the matched prefix, in order.
    /// matched_tokens / block_size == matched_block_ids.size().
    std::vector<int> matched_block_ids;
};

// ============================================================================
// PrefixCachePolicy
// ============================================================================
enum class PrefixCachePolicy {
    Hash,   // single-level hash map (current behavior, default)
    Radix   // SGLang-style radix tree with Patricia-trie compression
};

/// Parse policy from environment variable LIGHTLLM_PREFIX_CACHE.
/// Defaults to Hash if unset or unrecognised.
PrefixCachePolicy prefix_cache_policy_from_env();

/// Return the string name for a policy (for logging).
const char* prefix_cache_policy_name(PrefixCachePolicy policy);

// ============================================================================
// PrefixCache — abstract interface
// ============================================================================
class PrefixCache {
public:
    virtual ~PrefixCache() = default;

    // ------------------------------------------------------------------
    // Core operations
    // ------------------------------------------------------------------

    /// Look up the longest prefix match for tokens[start_pos..].
    ///
    /// Walks the cache structure following the token sequence.  Returns the
    /// number of consecutive matching tokens and the block IDs that hold the
    /// KV data for those matched blocks.
    ///
    /// The caller uses this result to determine how many leading blocks can
    /// be shared with an already-cached sequence.
    virtual PrefixCacheResult lookup(const std::vector<int>& tokens,
                                     int start_pos) = 0;

    /// Insert a contiguous run of full blocks into the cache.
    ///
    /// @param tokens    Full token sequence for the request so far.
    /// @param start_pos Starting position within `tokens` (must be a block
    ///                  boundary, i.e. start_pos % block_size == 0).
    /// @param block_ids Physical block IDs to associate, one per block.
    ///                  block_ids.size() == (tokens.size() - start_pos) / block_size
    ///
    /// Precondition: the blocks have already been filled with K/V data.
    /// After this call the blocks become visible to future lookup() calls.
    virtual void insert(const std::vector<int>& tokens,
                        int start_pos,
                        const std::vector<int>& block_ids) = 0;

    /// Remove a specific block from the cache metadata.
    ///
    /// Called when a block's ref_count reaches 0 and it is freed back to the
    /// pool.  In the radix tree this decrements the node's ref_count and may
    /// mark it eligible for eviction (ref_count==0 leaves).
    virtual void remove(int block_id) = 0;

    /// Find and evict the least-recently-used evictable leaf/entry.
    ///
    /// A leaf/entry is evictable when its ref_count == 0.  Among evictable
    /// candidates the one with the oldest last_access timestamp is chosen.
    ///
    /// @return The evicted block_id, or -1 if no block is evictable.
    ///         The caller is responsible for releasing the block's GPU memory.
    virtual int evict_lru() = 0;

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

    /// Number of cached blocks (entries / leaves).
    virtual size_t size() const = 0;

    /// Increment the reference count for a cached block (called when a
    /// sequence adopts an already-cached block).
    virtual void increment_ref(int block_id) = 0;

    /// Block size in tokens (set at construction time).
    virtual int block_size() const = 0;
};

// ============================================================================
// HashPrefixCache — current behavior wrapped behind the PrefixCache interface
// ============================================================================
// Each block is hashed independently (token_hash = hash of block_size tokens).
// No cross-block or partial-block prefix matching.  LRU is tracked through a
// simple sorted vector of (block_id, last_access) entries.
//
// Thread safety: not thread-safe.  The caller (BlockAllocator) serialises
// access within a single step() call.

class HashPrefixCache : public PrefixCache {
public:
    explicit HashPrefixCache(int block_size);

    PrefixCacheResult lookup(const std::vector<int>& tokens,
                             int start_pos) override;
    void insert(const std::vector<int>& tokens,
                int start_pos,
                const std::vector<int>& block_ids) override;
    void remove(int block_id) override;
    int evict_lru() override;
    size_t size() const override;
    void increment_ref(int block_id) override;
    int block_size() const override { return block_size_; }

    /// Direct hash-based single-block lookup (fast path for callers that
    /// already have the hash and only need one block).
    int find_by_hash(uint64_t token_hash) const;

    /// Compute the hash for a block-size chunk of tokens.
    static uint64_t hash_tokens(const int* tokens, int n);

private:
    int block_size_;
    std::unordered_map<uint64_t, int> cache_;  // token_hash -> block_id

    // LRU bookkeeping — sorted vector searched linearly on evict_lru().
    // For typical pool sizes (hundreds of blocks) this is fast enough.
    struct LRUEntry {
        int block_id;
        uint64_t last_access;
    };
    std::vector<LRUEntry> lru_list_;
    uint64_t access_counter_ = 0;

    // Map block_id -> index into lru_list_ for O(1) timestamp updates.
    std::unordered_map<int, size_t> lru_index_;

    // Reverse mapping from block_id back to its token_hash for O(1) cache_
    // erasure during evict_lru().
    std::unordered_map<int, uint64_t> block_to_hash_;

    // Reference count per cached block.  A block is evictable when its
    // ref_count reaches 0 (set by remove()).  increment_ref() bumps it
    // when a new sequence adopts a shared block.
    std::unordered_map<int, int> ref_counts_;
};

// ============================================================================
// RadixNode — Patricia trie node for RadixPrefixCache
// ============================================================================
// The radix tree stores token sequences as variable-length segments.
//
//   * A leaf node   has block_id >= 0.  Its token_segment is exactly
//     block_size tokens and represents one fully-filled KV block.
//   * An intermediate node has block_id == -1.  Its token_segment can be
//     any length (1 .. block_size).  Intermediates arise from splitting a
//     leaf that had ref_count == 1.
//
// Key invariant: children are keyed by the FIRST token of the child's
// token_segment, guaranteeing O(log fanout) child lookup.

struct RadixNode {
    /// Variable-length token segment (Patricia-trie property: this path
    /// edge carries multiple tokens without intermediate nodes).
    std::vector<int> token_segment;

    /// Physical block index, or -1 for intermediate routing nodes.
    int block_id = -1;

    /// Number of in-flight sequences that reference this node's block.
    /// Only meaningful when block_id >= 0.  When ref_count drops to 0
    /// the node becomes eligible for LRU eviction.
    int ref_count = 0;

    /// Monotonic access timestamp for LRU.  Updated on every lookup()
    /// or insert() that traverses this node.
    uint64_t last_access = 0;

    /// Child nodes keyed by the first token of the child's token_segment.
    std::map<int, std::unique_ptr<RadixNode>> children;

    bool is_leaf() const { return block_id >= 0; }
};

// ============================================================================
// RadixPrefixCache — SGLang-style radix tree prefix cache
// ============================================================================
// Algorithm summary
// -----------------
// LOOKUP
//   Walk from root using walk(tokens, start_pos).  At each step match as
//   many tokens as possible against the current node's token_segment.  If a
//   leaf is fully matched, append its block_id and continue to its children.
//   Stop at the first token mismatch.  Return accumulated block_ids and
//   total matched token count.
//
// INSERT
//   Walk from root.  For each new block:
//     - Follow matching path as far as possible.
//     - If a leaf is partially matched AND ref_count == 1: SPLIT the leaf
//       at the divergence point, creating an intermediate + two children.
//     - If a leaf is partially matched AND ref_count > 1: the leaf cannot
//       be split (other sequences depend on the full block).  Fall through
//       to allocate a fresh path.
//     - Create new leaf nodes for unmatched tokens (block_size per leaf).
//   After insert the tree has one leaf per inserted block, linked as a chain
//   or as siblings to existing nodes.
//
// EVICT (evict_lru)
//   Post-order DFS.  At each leaf with ref_count == 0, compare last_access
//   against the best candidate.  Return the block_id of the globally oldest
//   unreferenced leaf.  The caller frees the block and calls remove().
//
// REMOVE
//   Look up the RadixNode by block_id (O(1) via block_to_node_ map).
//   Set ref_count = 0.  Do NOT delete the node — it stays as a routing
//   skeleton for future lookups.  It will be physically evicted later by
//   evict_lru().

class RadixPrefixCache : public PrefixCache {
public:
    explicit RadixPrefixCache(int block_size);

    PrefixCacheResult lookup(const std::vector<int>& tokens,
                             int start_pos) override;
    void insert(const std::vector<int>& tokens,
                int start_pos,
                const std::vector<int>& block_ids) override;
    void remove(int block_id) override;
    int evict_lru() override;
    size_t size() const override;
    void increment_ref(int block_id) override;
    int block_size() const override { return block_size_; }

private:
    int block_size_;
    std::unique_ptr<RadixNode> root_;
    uint64_t access_counter_ = 0;

    /// O(1) mapping from block_id to its RadixNode (set on insert, cleared
    /// on physical eviction).
    std::unordered_map<int, RadixNode*> block_to_node_;

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Result of one walk step through the tree.
    struct WalkResult {
        RadixNode* node = nullptr;    // node we arrived at (never null if ok)
        RadixNode* parent = nullptr;  // parent of `node` (nullptr for root)
        int matched_tokens = 0;       // total tokens in `tokens` matched from start_pos
        int matched_in_segment = 0;   // tokens matched within node->token_segment
    };

    /// Walk the tree from root, consuming tokens[start_pos..] token by token.
    ///
    /// Stops at the deepest node where the token path still matches.
    /// Returns the final node, how many tokens were matched total, and how
    /// many were matched within that final node's segment.
    WalkResult walk(const std::vector<int>& tokens, int start_pos);

    /// Split `node` at position `pos` (0 < pos < node->token_segment.size()).
    ///
    /// Creates a new intermediate node with token_segment[0..pos-1] and
    /// block_id = -1.  `node` becomes a child with token_segment[pos..end].
    ///
    /// Returns a pointer to the new intermediate node (which replaces `node`
    /// in the tree — the caller must wire it into the parent).
    ///
    /// Precondition: if node->is_leaf(), node->ref_count must be 1.
    RadixNode* split_node(RadixNode* node, int pos);

    /// Post-order DFS helper: find the leaf with ref_count==0 and the
    /// smallest last_access value among the subtree rooted at `node`.
    void find_lru_leaf(RadixNode* node,
                       RadixNode*& best_node,
                       uint64_t& best_ts);

    /// Count leaves in subtree (for size() reporting).
    int count_leaves(const RadixNode* node) const;

    /// Prune a ref_count==0 leaf: remove from parent's children and from
    /// block_to_node_.  If the parent becomes a single-child intermediate
    /// after removal, optionally merge parent and sibling (radix compression).
    void prune_leaf(RadixNode* leaf, RadixNode* parent);
};

// ============================================================================
// Factory
// ============================================================================

/// Create a PrefixCache of the given policy.
std::unique_ptr<PrefixCache> create_prefix_cache(PrefixCachePolicy policy,
                                                  int block_size);

}  // namespace kv_cache
}  // namespace lightllm
