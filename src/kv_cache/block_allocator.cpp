/// BlockAllocator + BlockTable implementation.

#include "lightllm/kv_cache/block_allocator.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace lightllm {
namespace kv_cache {

// ===== BlockAllocator =======================================================

BlockAllocator::BlockAllocator(int num_blocks, int block_size,
                               int num_kv_heads, int head_dim,
                               PrefixCachePolicy policy)
    : num_blocks_(num_blocks), block_size_(block_size),
      num_kv_heads_(num_kv_heads), head_dim_(head_dim),
      policy_(policy),
      prefix_cache_(create_prefix_cache(policy, block_size)),
      k_storage_({num_blocks, block_size, num_kv_heads, head_dim},
                  DType::F32, Device::CUDA),
      v_storage_({num_blocks, block_size, num_kv_heads, head_dim},
                  DType::F32, Device::CUDA)
{
    blocks_.resize(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
        blocks_[i].block_id = i;
        free_list_.push_back(i);
    }
}

// ---- Internal helpers ------------------------------------------------------

int BlockAllocator::allocate_free_block() {
    if (free_list_.empty()) return -1;
    int id = free_list_.front();
    free_list_.pop_front();
    blocks_[id].ref_count = 1;
    blocks_[id].token_hash = 0;
    return id;
}

void BlockAllocator::free_block_physical(int block_id) {
    if (block_id < 0 || block_id >= num_blocks_) return;
    blocks_[block_id].token_hash = 0;
    blocks_[block_id].ref_count = 0;
    free_list_.push_back(block_id);
}

// ---- Block allocation ------------------------------------------------------

int BlockAllocator::allocate(uint64_t token_hash) {
    // Hash mode fast path: probe the hash cache for an existing block.
    if (token_hash != 0 && policy_ == PrefixCachePolicy::Hash) {
        int cached_id = find_by_hash(token_hash);
        if (cached_id >= 0) {
            blocks_[cached_id].ref_count++;
            return cached_id;
        }
    }

    // Allocate a fresh block.
    int id = allocate_free_block();
    if (id < 0) {
        // Pool exhausted — try to evict an LRU entry and retry.
        int evicted = evict_lru();
        if (evicted >= 0) {
            id = allocate_free_block();
        }
        if (id < 0) return -1;
    }

    blocks_[id].token_hash = token_hash;
    return id;
}

int BlockAllocator::allocate_tokens(const int* tokens, int n_tokens,
                                    int start_pos,
                                    std::vector<int>& out_blocks) {
    // 1. Build a vector view for the prefix cache lookup.
    std::vector<int> token_vec(tokens, tokens + n_tokens);

    // 2. Look up how many leading blocks are already cached.
    PrefixCacheResult result = prefix_cache_->lookup(token_vec, start_pos);

    // 3. Adopt shared blocks (increment ref_count for each).
    for (int bid : result.matched_block_ids) {
        blocks_[bid].ref_count++;
        out_blocks.push_back(bid);
    }

    // 4. Allocate fresh blocks for the remaining tokens.
    int matched = result.matched_tokens;
    int remaining_start = start_pos + matched;
    int n_new = 0;

    while (remaining_start < n_tokens) {
        int id = allocate_free_block();
        if (id < 0) {
            int evicted = evict_lru();
            if (evicted >= 0) {
                id = allocate_free_block();
            }
            if (id < 0) return n_new; // pool exhausted
        }
        out_blocks.push_back(id);
        n_new++;
        remaining_start += block_size_;
    }

    // 5. Register newly allocated blocks in the prefix cache.
    if (n_new > 0) {
        std::vector<int> new_blocks(out_blocks.end() - n_new,
                                    out_blocks.end());
        int insert_start = start_pos + matched;
        prefix_cache_->insert(token_vec, insert_start, new_blocks);

        // Update token_hash metadata for hash-mode backward compat.
        if (policy_ == PrefixCachePolicy::Hash) {
            for (int i = 0; i < n_new; i++) {
                int bid = new_blocks[static_cast<size_t>(i)];
                int tok_offset = insert_start + i * block_size_;
                if (tok_offset + block_size_ <= n_tokens) {
                    blocks_[bid].token_hash = HashPrefixCache::hash_tokens(
                        tokens + tok_offset, block_size_);
                }
            }
        }
    }

    return n_new;
}

void BlockAllocator::release(int block_id) {
    if (block_id < 0 || block_id >= num_blocks_) return;
    auto& b = blocks_[block_id];
    if (b.ref_count <= 0) return;
    b.ref_count--;
    if (b.ref_count == 0) {
        prefix_cache_->remove(block_id);
        free_block_physical(block_id);
    }
}

// ---- Prefix cache access ---------------------------------------------------

int BlockAllocator::find_by_hash(uint64_t token_hash) const {
    if (policy_ == PrefixCachePolicy::Hash) {
        auto* hash_cache =
            static_cast<const HashPrefixCache*>(prefix_cache_.get());
        return hash_cache->find_by_hash(token_hash);
    }
    return -1;
}

PrefixCacheResult BlockAllocator::find_cached_prefix(
        const std::vector<int>& tokens, int start_pos) const {
    return prefix_cache_->lookup(tokens, start_pos);
}

void BlockAllocator::insert_cached_blocks(
        const std::vector<int>& tokens, int start_pos,
        const std::vector<int>& block_ids) {
    prefix_cache_->insert(tokens, start_pos, block_ids);

    // Update token_hash metadata on each block for hash-mode backward compat
    // (so release() can still check token_hash if needed, and callers that
    // read Block::token_hash see a consistent value).
    if (policy_ == PrefixCachePolicy::Hash) {
        for (size_t i = 0; i < block_ids.size(); i++) {
            int bid = block_ids[i];
            int tok_offset = start_pos + static_cast<int>(i) * block_size_;
            if (tok_offset + block_size_ <= static_cast<int>(tokens.size())) {
                blocks_[bid].token_hash = HashPrefixCache::hash_tokens(
                    tokens.data() + tok_offset, block_size_);
            }
        }
    }
}

int BlockAllocator::evict_lru() {
    int evicted = prefix_cache_->evict_lru();
    if (evicted >= 0) {
        free_block_physical(evicted);
    }
    return evicted;
}

void BlockAllocator::increment_ref(int block_id) {
    if (block_id < 0 || block_id >= num_blocks_) return;
    blocks_[block_id].ref_count++;
    prefix_cache_->increment_ref(block_id);
}

// ---- Accessors -------------------------------------------------------------

void* BlockAllocator::k_data(int block_id) {
    size_t offset = static_cast<size_t>(block_id) * block_size_
                    * num_kv_heads_ * head_dim_ * sizeof(float);
    return static_cast<char*>(k_storage_.raw()) + offset;
}

void* BlockAllocator::v_data(int block_id) {
    size_t offset = static_cast<size_t>(block_id) * block_size_
                    * num_kv_heads_ * head_dim_ * sizeof(float);
    return static_cast<char*>(v_storage_.raw()) + offset;
}

// ===== BlockTable ===========================================================

BlockTable::BlockTable(int max_blocks) : max_blocks_(max_blocks) {
    blocks_.reserve(max_blocks);
}

void BlockTable::append(int physical_id) {
    if (static_cast<int>(blocks_.size()) >= max_blocks_)
        throw std::runtime_error("BlockTable: exceeded max_blocks");
    blocks_.push_back(physical_id);
}

int BlockTable::operator[](int pos) const {
    return blocks_[pos];
}

}  // namespace kv_cache
}  // namespace lightllm
