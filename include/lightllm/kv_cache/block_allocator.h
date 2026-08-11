#pragma once
/// PagedAttention: Block-based KV cache allocator with configurable prefix caching.
///
/// Physical blocks are fixed-size: [block_size, num_kv_heads, head_dim].
/// Multiple sequences can share blocks when their token prefixes match.
///
/// Two prefix-cache policies are available (select via constructor or the
/// LIGHTLLM_PREFIX_CACHE environment variable):
///
///   Hash  — single-level token-hash map (current behavior, default)
///   Radix — SGLang-style radix tree with longest-prefix matching
///
/// The prefix cache is transparent to most callers.  The key difference is
/// that the radix tree enables cross-block prefix detection: two requests
/// that share N leading blocks will automatically share those blocks without
/// the engine having to explicitly hash and probe block-by-block.

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lightllm/kv_cache/prefix_cache.h"
#include "lightllm/tensor.h"

namespace lightllm {
namespace kv_cache {

struct Block {
    int block_id;            // physical index in the pool
    int ref_count = 0;       // how many sequences reference this block
    uint64_t token_hash = 0; // hash of the token chunk for prefix caching
};

class BlockAllocator {
public:
    /// @param num_blocks    total physical blocks
    /// @param block_size    tokens per block (typically 16)
    /// @param num_kv_heads  KV heads (2 for Qwen2.5-0.5B GQA)
    /// @param head_dim      dimension per head (64)
    /// @param policy        prefix-cache policy (Hash / Radix)
    /// @param kv_dtype      data type for K/V storage (F32 or F16)
    BlockAllocator(int num_blocks, int block_size,
                   int num_kv_heads, int head_dim,
                   PrefixCachePolicy policy = PrefixCachePolicy::Hash,
                   DType kv_dtype = DType::F32);

    // ------------------------------------------------------------------
    // Block allocation
    // ------------------------------------------------------------------

    /// Allocate one free block, optionally querying the prefix cache.
    ///
    /// In hash mode: if token_hash != 0, probe the hash cache first.
    /// In radix mode: token_hash is ignored; use allocate_with_tokens()
    ///                for prefix-aware allocation.
    ///
    /// @return physical block_id, or -1 if pool is exhausted.
    int allocate(uint64_t token_hash = 0);

    /// Allocate blocks for a sequence of tokens, using the prefix cache to
    /// find and share already-cached blocks.
    ///
    /// This is the primary allocation path for prefill.  It:
    ///   1. Calls prefix_cache_->lookup(tokens, start_pos) to find how many
    ///      leading blocks are already cached.
    ///   2. Increments ref_count on shared blocks.
    ///   3. Allocates fresh blocks for the remaining tokens.
    ///   4. Calls prefix_cache_->insert() to register the new blocks.
    ///
    /// @param tokens       Full token sequence for the request so far.
    /// @param start_pos    Token index where new blocks are needed (0 for
    ///                     a fresh request, or >0 when extending).
    /// @param out_blocks   [out] Appended: physical block_ids for ALL blocks
    ///                     covering tokens[0 .. tokens.size()-1], both shared
    ///                     and newly allocated.
    /// @return Number of newly allocated blocks (0 if all were shared).
    ///         On pool exhaustion, out_blocks is incomplete and the caller
    ///         must check out_blocks.size() against expected.
    int allocate_tokens(const int* tokens, int n_tokens, int start_pos,
                        std::vector<int>& out_blocks);

    /// Decrement ref_count; free if it reaches 0.
    void release(int block_id);

    // ------------------------------------------------------------------
    // Prefix cache access
    // ------------------------------------------------------------------

    /// Direct hash lookup (backward-compatible fast path for hash mode).
    int find_by_hash(uint64_t token_hash) const;

    /// Look up longest prefix match for tokens starting at start_pos.
    PrefixCacheResult find_cached_prefix(const std::vector<int>& tokens,
                                         int start_pos) const;

    /// Register blocks that have been filled with K/V data.
    void insert_cached_blocks(const std::vector<int>& tokens,
                              int start_pos,
                              const std::vector<int>& block_ids);

    /// Evict the least-recently-used evictable block.  If a block is
    /// evicted its GPU memory is freed (returned to the free list).
    /// @return evicted block_id, or -1 if nothing was evictable.
    int evict_lru();

    /// Increment ref_count on an already-cached block (used when a new
    /// sequence adopts a shared block found by find_cached_prefix).
    void increment_ref(int block_id);

    /// Full reset: rebuild free list, clear all block metadata,
    /// replace prefix cache with a fresh one, zero GPU storage.
    /// Used between benchmark runs to ensure a clean KV cache state.
    void reset();

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Access underlying K/V storage for a given block.
    void* k_data(int block_id);
    void* v_data(int block_id);

    /// Raw pointer to entire K/V storage (for CUDA kernels).
    const void* k_storage_raw() const { return k_storage_.raw(); }
    const void* v_storage_raw() const { return v_storage_.raw(); }
    size_t k_storage_bytes() const { return k_storage_.nbytes(); }
    size_t v_storage_bytes() const { return v_storage_.nbytes(); }

    int num_blocks()  const { return num_blocks_; }
    int block_size()  const { return block_size_; }
    int num_free()    const { return static_cast<int>(free_list_.size()); }
    int num_kv_heads() const { return num_kv_heads_; }
    int head_dim()    const { return head_dim_; }
    DType kv_dtype()  const { return kv_dtype_; }

    PrefixCachePolicy policy() const { return policy_; }
    const PrefixCache* prefix_cache() const { return prefix_cache_.get(); }

private:
    int num_blocks_, block_size_, num_kv_heads_, head_dim_;
    DType kv_dtype_ = DType::F32;
    PrefixCachePolicy policy_;
    std::vector<Block> blocks_;
    std::deque<int> free_list_;

    // Full K/V storage: [num_blocks, block_size, num_kv_heads, head_dim]
    Tensor k_storage_, v_storage_;

    // Configurable prefix cache (HashPrefixCache or RadixPrefixCache).
    std::unique_ptr<PrefixCache> prefix_cache_;

    // ---- Internal helpers ----

    /// Allocate a fresh block from the free list (no cache interaction).
    int allocate_free_block();

    /// Free a block back to the pool (zero metadata, push to free list).
    void free_block_physical(int block_id);
};

// ---------------------------------------------------------------------------
// BlockTable: per-sequence logical-to-physical mapping.
// ---------------------------------------------------------------------------
class BlockTable {
public:
    explicit BlockTable(int max_blocks);

    /// Append a physical block to this sequence.
    void append(int physical_id);

    /// Get the physical block at logical position `pos`.
    int operator[](int pos) const;

    /// Number of blocks in this sequence.
    int size() const { return static_cast<int>(blocks_.size()); }

    /// Raw pointer to the block ID array (for passing to CUDA kernel).
    const int* data() const { return blocks_.data(); }

private:
    int max_blocks_;
    std::vector<int> blocks_;  // logical -> physical
};

}  // namespace kv_cache
}  // namespace lightllm
