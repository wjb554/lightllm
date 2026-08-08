/// RadixPrefixCache — SGLang-style radix tree (Patricia trie) implementation.
///
/// See include/lightllm/kv_cache/prefix_cache.h for the full design document
/// and algorithm descriptions.
///
/// Thread safety: not thread-safe.  The caller (BlockAllocator) serialises
/// access within a single step() call.

#include "lightllm/kv_cache/prefix_cache.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <utility>

namespace lightllm {
namespace kv_cache {

// Forward declaration for file-local helpers.
namespace {
RadixNode* find_parent_impl(RadixNode* current, RadixNode* target);
}

static RadixNode* find_parent(RadixNode* root, RadixNode* target);

// ============================================================================
// RadixPrefixCache — constructor
// ============================================================================

RadixPrefixCache::RadixPrefixCache(int block_size)
    : block_size_(block_size)
    , root_(std::make_unique<RadixNode>())
{
    root_->block_id = -1;
}

// ============================================================================
// lookup — find longest prefix of full-block matches
// ============================================================================
// The radix tree encodes absolute token position through its structure:
// the path from root determines where in a sequence each node's segment
// begins.  Therefore the walk must ALWAYS start from token position 0.
//
// Phase 1: walk tokens[0 .. start_pos)   — consume, do not record
// Phase 2: walk tokens[start_pos .. n)   — record leaf block_ids
//
// A leaf is recorded only when its entire segment (block_size tokens) was
// matched AND the leaf's position in the sequence is >= start_pos.
// matched_tokens reflects the delta from start_pos.

PrefixCacheResult RadixPrefixCache::lookup(
    const std::vector<int>& tokens, int start_pos)
{
    PrefixCacheResult result;
    int n = static_cast<int>(tokens.size());
    uint64_t now = ++access_counter_;

    // ------------------------------------------------------------------
    // Phase 1: walk from position 0 toward start_pos.
    // The radix tree ideally encodes the full token path from the root.
    // However, insert() may have been called with start_pos > 0 (e.g. a
    // suffix-only insertion), in which case the tree has no path for the
    // skipped prefix.  If we cannot reach start_pos we fall back to direct
    // matching from start_pos at the root level.
    // ------------------------------------------------------------------
    RadixNode* node = root_.get();
    int pos = 0;
    bool reached_start = (start_pos == 0);

    if (start_pos > 0) {
        while (pos < start_pos) {
            auto it = node->children.find(tokens[pos]);
            if (it == node->children.end()) break;

            RadixNode* child = it->second.get();
            int seg_len  = static_cast<int>(child->token_segment.size());
            int remaining = start_pos - pos;
            int max_cmp   = std::min(seg_len, remaining);
            int match_len = 0;
            while (match_len < max_cmp &&
                   tokens[pos + match_len] == child->token_segment[match_len])
                match_len++;

            if (match_len < max_cmp) break;   // cannot reach start_pos

            pos  += match_len;
            node  = child;
            node->last_access = now;

            if (match_len < seg_len) break;   // partial segment at boundary
        }
        reached_start = (pos == start_pos);
    }

    // If we did NOT reach start_pos through the tree, try matching directly
    // from start_pos at the root level (handles suffix-only insertions).
    if (!reached_start) {
        node = root_.get();
        pos  = start_pos;
    }

    // ------------------------------------------------------------------
    // Phase 2: walk from current position (at or past start_pos) recording
    // fully-matched leaf blocks.
    // ------------------------------------------------------------------
    while (pos < n) {
        auto it = node->children.find(tokens[pos]);
        if (it == node->children.end()) break;

        RadixNode* child = it->second.get();
        int seg_len = static_cast<int>(child->token_segment.size());
        int max_cmp = std::min(seg_len, n - pos);
        int match_len = 0;
        while (match_len < max_cmp &&
               tokens[pos + match_len] == child->token_segment[match_len])
            match_len++;

        if (match_len < seg_len) {
            child->last_access = now;
            break;
        }

        pos  += match_len;
        node  = child;
        node->last_access = now;

        if (node->is_leaf() && pos > start_pos) {
            result.matched_tokens = pos - start_pos;
            result.matched_block_ids.push_back(node->block_id);
        }
    }

    return result;
}

// ============================================================================
// insert — register newly-filled blocks into the radix tree
// ============================================================================
// For each block in block_ids:
//   1. Walk the tree matching as many tokens as possible.
//   2. At a partial match, SPLIT the existing node (creating an intermediate
//      routing node with the common prefix).
//   3. Create new leaf nodes for any unmatched token ranges.
//
// Splitting is always allowed (even when the existing leaf has ref_count > 1)
// because the cumulative token path from root to any extant leaf is preserved
// — splitting only adds an intermediate node above it, which does not alter
// the path seen by other sequences that reference the same block_id.

void RadixPrefixCache::insert(const std::vector<int>& tokens,
                               int start_pos,
                               const std::vector<int>& block_ids)
{
    uint64_t now = ++access_counter_;

    // ------------------------------------------------------------------
    // Pre-walk: advance `node` through the tree from position 0 up to
    // start_pos.  The tree must encode the full token path from the root,
    // so we walk tokens[0..start_pos) following the tree structure to
    // position ourselves at the correct insertion point.
    // ------------------------------------------------------------------
    RadixNode* node = root_.get();
    int pos = 0;

    while (pos < start_pos) {
        auto it = node->children.find(tokens[pos]);
        if (it == node->children.end()) {
            // Tree does not have the full prefix — insertion point is
            // unreachable.  Fall back to direct insertion from start_pos
            // at the root level (handles suffix-only, non-chained inserts).
            node = root_.get();
            pos  = start_pos;
            break;
        }
        RadixNode* child = it->second.get();
        int seg_len  = static_cast<int>(child->token_segment.size());
        int remaining = start_pos - pos;
        int max_cmp   = std::min(seg_len, remaining);
        int match_len = 0;
        while (match_len < max_cmp &&
               tokens[pos + match_len] == child->token_segment[match_len])
            match_len++;
        if (match_len < max_cmp) {
            // Cannot reach start_pos; fall back.
            node = root_.get();
            pos  = start_pos;
            break;
        }
        pos  += match_len;
        node  = child;
        child->last_access = now;
        if (match_len < seg_len) {
            // We are partway through a node at start_pos — can't cleanly
            // insert from here.  Fall back to root-level.
            node = root_.get();
            pos  = start_pos;
            break;
        }
    }

    // ------------------------------------------------------------------
    // Now insert new blocks starting from the reached tree position.
    // ------------------------------------------------------------------

    for (size_t j = 0; j < block_ids.size(); j++) {
        int block_id = block_ids[j];
        int remaining = block_size_;

        while (remaining > 0) {
            int first_token = tokens[pos];
            auto it = node->children.find(first_token);

            // ----------------------------------------------------------
            // Case A: No existing child — create a fresh leaf
            // ----------------------------------------------------------
            if (it == node->children.end()) {
                auto leaf = std::make_unique<RadixNode>();
                leaf->token_segment.assign(
                    tokens.begin() + pos,
                    tokens.begin() + pos + remaining);
                leaf->block_id    = block_id;
                leaf->ref_count   = 1;
                leaf->last_access = now;

                block_to_node_[block_id] = leaf.get();

                // Snapshot parent and the leaf's map key BEFORE updating
                // `node` so we insert the leaf under the right parent.
                RadixNode* parent = node;
                int key = leaf->token_segment[0];
                RadixNode* leaf_ptr = leaf.get();

                pos       += remaining;
                remaining  = 0;
                node       = leaf_ptr;

                parent->children[key] = std::move(leaf);
                break;
            }

            RadixNode* child   = it->second.get();
            int seg_len        = static_cast<int>(child->token_segment.size());

            // Compute common prefix length
            int match_len = 0;
            int max_cmp   = std::min(remaining, seg_len);
            while (match_len < max_cmp &&
                   tokens[pos + match_len] == child->token_segment[match_len]) {
                match_len++;
            }

            // match_len == 0 should be impossible (keyed by first token),
            // but guard against corrupt trees.
            if (match_len == 0) {
                auto leaf = std::make_unique<RadixNode>();
                leaf->token_segment.assign(
                    tokens.begin() + pos,
                    tokens.begin() + pos + remaining);
                leaf->block_id    = block_id;
                leaf->ref_count   = 1;
                leaf->last_access = now;
                block_to_node_[block_id] = leaf.get();

                // Snapshot parent before updating `node`.
                RadixNode* parent = node;
                int key = leaf->token_segment[0];
                RadixNode* leaf_ptr = leaf.get();

                pos       += remaining;
                remaining  = 0;
                node       = leaf_ptr;

                parent->children[key] = std::move(leaf);
                break;
            }

            // ----------------------------------------------------------
            // Case B: Full segment match
            // ----------------------------------------------------------
            if (match_len == seg_len) {
                pos       += match_len;
                remaining -= match_len;
                node       = child;
                node->last_access = now;

                if (child->is_leaf()) {
                    // Existing leaf fully matches our prefix — bump ref.
                    child->ref_count++;

                    if (remaining == 0) {
                        // Our block matches this existing leaf exactly.
                        block_to_node_[block_id] = child;
                        break;
                    }
                }
                // remaining > 0: continue traversing into children.
                continue;
            }

            // ----------------------------------------------------------
            // Case C: Partial match (match_len < seg_len) — SPLIT
            // ----------------------------------------------------------
            // Detach the existing child from parent (take ownership).
            auto child_owned = std::move(it->second);

            // Create an intermediate routing node for the common prefix.
            auto intermediate = std::make_unique<RadixNode>();
            RadixNode* inter_ptr = intermediate.get();

            intermediate->token_segment.assign(
                child_owned->token_segment.begin(),
                child_owned->token_segment.begin() + match_len);
            intermediate->block_id    = -1;
            intermediate->ref_count   = 0;
            intermediate->last_access = child_owned->last_access;

            // Move the existing child's children to the intermediate.
            intermediate->children = std::move(child_owned->children);

            // Trim the existing child's segment to the suffix.
            child_owned->token_segment.erase(
                child_owned->token_segment.begin(),
                child_owned->token_segment.begin() + match_len);

            // Re-attach the trimmed child under the intermediate.
            int child_key = child_owned->token_segment[0];
            RadixNode* child_raw = child_owned.get();
            intermediate->children[child_key] = std::move(child_owned);

            // Update block_to_node_ if the trimmed node is still a leaf.
            if (child_raw->is_leaf()) {
                block_to_node_[child_raw->block_id] = child_raw;
            }

            // Advance our cursor past the common prefix.
            pos       += match_len;
            remaining -= match_len;

            // If our remaining tokens match exactly the intermediate's
            // prefix (our block ends at the split point), the intermediate
            // ITSELF holds our block's value — serving as both a routing
            // node and a value node.
            if (remaining == 0) {
                intermediate->block_id  = block_id;
                intermediate->ref_count = 1;
                intermediate->last_access = now;
                block_to_node_[block_id] = inter_ptr;
                node->children[first_token] = std::move(intermediate);
                node = inter_ptr;
                break;
            }

            // Insert the intermediate back into the parent.
            node->children[first_token] = std::move(intermediate);

            // Continue the while loop from the intermediate node for
            // the unmatched suffix of our block.
            node = inter_ptr;
        }
    }
}

// ============================================================================
// remove — mark a cached block as no longer referenced
// ============================================================================
// Called when a block's ref_count reaches 0 in BlockAllocator.  We set
// ref_count = 0 here too, which makes the node eligible for eviction.
// The node is NOT removed from the tree — it remains as a routing skeleton
// until evict_lru() reclaims it.

void RadixPrefixCache::remove(int block_id)
{
    auto it = block_to_node_.find(block_id);
    if (it == block_to_node_.end()) return;

    RadixNode* node = it->second;
    if (node->ref_count > 0) {
        node->ref_count--;
    }
    // Node stays in tree.  If ref_count reaches 0 it becomes evictable.
}

// ============================================================================
// increment_ref — bump reference count on an already-cached block
// ============================================================================
// Used by the caller when a new sequence adopts a shared block found via
// lookup().  Must match each remove() with a prior increment_ref().

void RadixPrefixCache::increment_ref(int block_id)
{
    auto it = block_to_node_.find(block_id);
    if (it != block_to_node_.end()) {
        it->second->ref_count++;
    }
}

// ============================================================================
// evict_lru — find and physically remove the oldest evictable leaf
// ============================================================================
// Post-order DFS locates the leaf with ref_count == 0 and children.empty()
// whose last_access is smallest.  The leaf is removed from the tree and
// from block_to_node_.  If after removal the parent becomes a single-child
// intermediate (block_id == -1), parent and remaining child are merged
// (radix compression).
//
// Returns the evicted block_id, or -1 if nothing is evictable.

int RadixPrefixCache::evict_lru()
{
    RadixNode* best_node = nullptr;
    uint64_t   best_ts   = UINT64_MAX;

    find_lru_leaf(root_.get(), best_node, best_ts);

    if (best_node == nullptr) return -1;

    int block_id = best_node->block_id;

    // Find the parent of the eviction candidate.
    RadixNode* parent = find_parent(root_.get(), best_node);
    if (parent != nullptr) {
        prune_leaf(best_node, parent);
    } else {
        // best_node is the root itself.  Clear its value so it becomes a pure
        // routing node again, otherwise the next evict_lru() call will find it
        // again and return the same block_id forever.
        best_node->block_id = -1;
        best_node->ref_count = 0;
        best_node->token_segment.clear();
    }

    block_to_node_.erase(block_id);
    return block_id;
}

// ============================================================================
// size — number of currently-cached blocks (leaves with value)
// ============================================================================

size_t RadixPrefixCache::size() const
{
    return block_to_node_.size();
}

// ============================================================================
// walk — traverse the tree consuming tokens from start_pos
// ============================================================================
// Walks from root, following matching token_segments as far as possible.
// Returns the final node reached, its parent, the total matched token count,
// and how many tokens were matched within the final node's segment.
//
// Used by insert() to locate the insertion point; also useful for debugging.

RadixPrefixCache::WalkResult RadixPrefixCache::walk(
    const std::vector<int>& tokens, int start_pos)
{
    WalkResult wr;
    wr.node   = root_.get();
    wr.parent = nullptr;

    RadixNode* node = root_.get();
    int pos = start_pos;
    int n = static_cast<int>(tokens.size());

    while (pos < n) {
        auto it = node->children.find(tokens[pos]);
        if (it == node->children.end()) break;

        RadixNode* child = it->second.get();
        int seg_len = static_cast<int>(child->token_segment.size());
        int max_cmp = std::min(seg_len, n - pos);
        int match_len = 0;
        while (match_len < max_cmp &&
               tokens[pos + match_len] == child->token_segment[match_len]) {
            match_len++;
        }

        if (match_len < seg_len) {
            // Partial match — stop at this child
            wr.node              = child;
            wr.parent            = node;
            wr.matched_tokens    = pos - start_pos + match_len;
            wr.matched_in_segment = match_len;
            break;
        }

        // Full match — advance
        pos += match_len;
        node = child;
    }

    // If we exhausted tokens or hit the end of the tree, the final node is
    // `node` (might be root or an intermediate).
    if (wr.node == root_.get() && wr.parent == nullptr) {
        wr.node              = node;
        wr.matched_tokens    = pos - start_pos;
        wr.matched_in_segment = (node == root_.get())
            ? 0
            : static_cast<int>(node->token_segment.size());
    }

    return wr;
}

// ============================================================================
// split_node — split a node's token_segment at position pos
// ============================================================================
// Creates a new intermediate node with token_segment[0..pos-1] and
// block_id == -1.  The original `node` becomes a child of the intermediate
// with token_segment[pos..end], retaining its original block_id and ref_count.
//
// PRECONDITION: `node` has been detached from its parent (ownership
// transferred to the caller via unique_ptr::release()).  This function
// takes ownership of the raw pointer and returns a raw pointer to the
// new intermediate, which the caller must wrap back into a unique_ptr
// and wire into the parent.
//
// Returns the new intermediate node (never null; 0 < pos < segment size).

RadixNode* RadixPrefixCache::split_node(RadixNode* node, int pos)
{
    // Take ownership of the detached node immediately.
    std::unique_ptr<RadixNode> owned_node(node);

    auto intermediate  = std::make_unique<RadixNode>();
    RadixNode* inter_raw = intermediate.get();

    // Common prefix goes to the intermediate.
    intermediate->token_segment.assign(
        owned_node->token_segment.begin(),
        owned_node->token_segment.begin() + pos);
    intermediate->block_id    = -1;
    intermediate->ref_count   = 0;
    intermediate->last_access = owned_node->last_access;

    // Move the original node's children to the intermediate.
    intermediate->children = std::move(owned_node->children);

    // Trim the original node's segment to the suffix.
    owned_node->token_segment.erase(
        owned_node->token_segment.begin(),
        owned_node->token_segment.begin() + pos);

    // Attach the original node as a child of the intermediate.
    int child_key = owned_node->token_segment[0];
    intermediate->children[child_key] = std::move(owned_node);

    // Return raw pointer; caller takes ownership.
    return intermediate.release();
}

// ============================================================================
// find_lru_leaf — post-order DFS for the evictable leaf with oldest timestamp
// ============================================================================
// Only considers nodes where:
//   - block_id >= 0  (has a value / is a leaf)
//   - ref_count == 0 (not referenced by any sequence)
//   - children.empty() OR the node is ready for eviction
//
// Nodes with children are NOT evicted (they serve as routing nodes);
// they can be converted to pure intermediates by clear_value() if needed.

void RadixPrefixCache::find_lru_leaf(
    RadixNode* node, RadixNode*& best_node, uint64_t& best_ts)
{
    if (node == nullptr) return;

    // Check this node if it carries a value and is unreferenced.
    if (node->is_leaf() && node->ref_count == 0) {
        // Only consider nodes without children for physical eviction.
        // Nodes WITH children that carry a value are "branch value nodes"
        // — evicting them would orphan their children.  They can safely
        // have their value cleared (converted to intermediate) but that
        // is a separate compaction pass.
        if (node->children.empty() && node->last_access < best_ts) {
            best_node = node;
            best_ts   = node->last_access;
        }
    }

    // Recurse into children (post-order ensures leaves are visited first).
    for (auto& kv : node->children) {
        find_lru_leaf(kv.second.get(), best_node, best_ts);
    }
}

// ============================================================================
// count_leaves — recursive count of nodes with block_id >= 0 in subtree
// ============================================================================

int RadixPrefixCache::count_leaves(const RadixNode* node) const
{
    if (node == nullptr) return 0;

    int cnt = node->is_leaf() ? 1 : 0;
    for (const auto& kv : node->children) {
        cnt += count_leaves(kv.second.get());
    }
    return cnt;
}

// ============================================================================
// prune_leaf — remove a leaf from its parent and apply radix compression
// ============================================================================
// 1. Erases the leaf from parent->children (keyed by leaf's first token).
// 2. If parent is now an intermediate (block_id == -1) with exactly one
//    remaining child, merges parent and that child:
//      parent->token_segment += child->token_segment
//      parent adopts child's block_id, ref_count, and children
//    This maintains the Patricia-trie invariant — a path that can be
//    collapsed without ambiguity is collapsed.

void RadixPrefixCache::prune_leaf(RadixNode* leaf, RadixNode* parent)
{
    // Find and erase the leaf from parent's children.
    int key = leaf->token_segment[0];
    auto it = parent->children.find(key);
    if (it == parent->children.end()) return;

    parent->children.erase(it);

    // Radix compression: if parent is a pure intermediate (block_id == -1)
    // and now has exactly one child, merge them.
    // Never merge into the root — root must stay as a routing node so that
    // lookup() can always start from root and walk down via children.
    if (parent->block_id == -1 && parent->children.size() == 1
        && parent != root_.get()) {

        auto& only_child_ref = parent->children.begin()->second;

        // CRITICAL: Extract ALL data from the child BEFORE we reassign
        // parent->children.  std::map::operator=(map&&) will destroy the
        // old map entries first, which destroys the unique_ptr holding
        // only_child.  After that point, `only_child_ref` is dangling.
        // We must copy/move everything we need OUT of the child NodeBEFORE
        // that destruction.
        std::vector<int> child_segment = only_child_ref->token_segment;
        int               child_bid     = only_child_ref->block_id;
        int               child_rc      = only_child_ref->ref_count;
        uint64_t          child_ts      = only_child_ref->last_access;
        std::map<int, std::unique_ptr<RadixNode>> child_kids =
            std::move(only_child_ref->children);

        // Now perform the merge into parent.
        parent->token_segment.insert(
            parent->token_segment.end(),
            child_segment.begin(),
            child_segment.end());

        parent->block_id    = child_bid;
        parent->ref_count   = child_rc;
        parent->last_access = child_ts;

        // Remap block_to_node_ to point to `parent` now.
        if (parent->is_leaf()) {
            block_to_node_[parent->block_id] = parent;
        }

        // Replace children map.  This destroys only_child_ref (which now
        // has an empty children map and stale token_segment — harmless).
        parent->children = std::move(child_kids);
    }

    // Note: if parent's children are now empty and parent is intermediate
    // (block_id == -1), parent itself can be pruned.  We do NOT cascade
    // upward here — a future eviction pass will handle the orphan.
}

// ============================================================================
// find_parent — DFS to locate the parent of a target node
// ============================================================================

namespace {

RadixNode* find_parent_impl(RadixNode* current, RadixNode* target)
{
    if (current == nullptr || target == nullptr) return nullptr;

    for (auto& kv : current->children) {
        if (kv.second.get() == target) {
            return current;
        }
        RadixNode* found = find_parent_impl(kv.second.get(), target);
        if (found != nullptr) return found;
    }
    return nullptr;
}

}  // anonymous namespace

// Exposed as a free function in the .cpp (not in the header).
// Used internally by evict_lru().
static RadixNode* find_parent(RadixNode* root, RadixNode* target)
{
    return find_parent_impl(root, target);
}

// ============================================================================
// Factory — create_prefix_cache
// ============================================================================

std::unique_ptr<PrefixCache> create_prefix_cache(PrefixCachePolicy policy,
                                                  int block_size)
{
    switch (policy) {
    case PrefixCachePolicy::Hash:
        return std::make_unique<HashPrefixCache>(block_size);
    case PrefixCachePolicy::Radix:
        return std::make_unique<RadixPrefixCache>(block_size);
    }
    // Unreachable; silence compiler warning.
    return std::make_unique<HashPrefixCache>(block_size);
}

}  // namespace kv_cache
}  // namespace lightllm
