#pragma once
/// Unified Request Scheduler — Chunked Prefill + Interleaved Scheduling.
///
/// Request State Machine:
///   WAITING → (prefill chunks) → DECODING → FINISHED
///
/// Each ScheduleStep describes a heterogeneous batch:
///   - Some requests contribute prefill chunks (up to CHUNK_SIZE tokens)
///   - Some requests contribute decode tokens (1 token each)
///   - Total batch: sum(num_tokens) × hidden_dim

#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace lightllm {
namespace engine {

// ============================================================================
// Request — unified state for prefill and decode phases
// ============================================================================
struct Request {
    int id;
    std::vector<int> prompt_tokens;   // original prompt
    std::vector<int> output_tokens;   // generated tokens
    int num_computed = 0;             // how many prompt tokens have been prefilled
    int max_new_tokens = 64;
    int eos_token_id = 151643;

    /// JSON Schema for structured output (empty = unconstrained).
    std::string json_schema;
    /// Regex pattern for structured output (empty = unconstrained).
    std::string regex;

    enum State { WAITING, DECODING, FINISHED };
    State state = WAITING;

    // How many tokens are left to prefill (0 if done)
    int prefill_remaining() const { return (int)prompt_tokens.size() - num_computed; }
    bool needs_prefill() const { return state == WAITING; }
    int total_len() const { return (int)prompt_tokens.size() + (int)output_tokens.size(); }
};

// ============================================================================
// ScheduleStep — one batch to execute
// ============================================================================
struct ScheduleEntry {
    int request_idx;     // index into scheduler's request list
    int num_tokens;      // tokens to process this step (1 for decode, <=CHUNK for prefill)
    int start_pos;       // absolute position of first token (for RoPE)
    bool is_prefill;     // true = prefill chunk, false = decode
};

struct ScheduleStep {
    std::vector<ScheduleEntry> entries;
    int total_tokens() const {
        int n = 0;
        for (auto& e : entries) n += e.num_tokens;
        return n;
    }
    bool empty() const { return entries.empty(); }
};

// ============================================================================
// SchedulerPolicy — tag for dispatch
// ============================================================================
enum class SchedulerPolicy { FCFS, PrefillFirst, DecodeFirst, Interleave };

// ============================================================================
// Scheduler — abstract base
// ============================================================================
class Scheduler {
public:
    virtual ~Scheduler() = default;

    /// Submit a new request (enters WAITING).
    virtual void add_request(Request req) = 0;

    /// Produce the next batch to execute.
    /// Returns empty step when no work remains.
    virtual ScheduleStep step() = 0;

    /// Notify the scheduler that a request finished (reached EOS or max_tokens).
    virtual void finish_request(int req_id) = 0;

    /// Number of pending + running requests.
    virtual int num_active() const = 0;

    /// Factory: create scheduler by policy.
    static std::unique_ptr<Scheduler> create(SchedulerPolicy policy, int chunk_size = 16);
};

// ============================================================================
// FCFS Scheduler — one request at a time, prefill all chunks then decode
// ============================================================================
class FCFSScheduler : public Scheduler {
public:
    FCFSScheduler(int chunk_size = 16);
    void add_request(Request req) override;
    ScheduleStep step() override;
    void finish_request(int req_id) override;
    int num_active() const override { return (int)requests_.size(); }

private:
    int chunk_size_;
    std::deque<Request> requests_;      // waiting queue
    Request* current_ = nullptr;         // request being processed
};

// ============================================================================
// PrefillFirst Scheduler — prefill gets priority, decode tokens batched
// ============================================================================
class PrefillFirstScheduler : public Scheduler {
public:
    PrefillFirstScheduler(int chunk_size = 16);
    void add_request(Request req) override;
    ScheduleStep step() override;
    void finish_request(int req_id) override;
    int num_active() const override { return (int)waiting_.size() + (int)decoding_.size(); }

private:
    int chunk_size_;
    std::deque<Request> waiting_;        // WAITING queue
    std::vector<Request> decoding_;      // DECODING queue

    bool prefill_one_chunk(ScheduleStep& step, Request& req);
    void collect_decode_tokens(ScheduleStep& step);
};

// ============================================================================
// DecodeFirst Scheduler — decode has priority, prefill uses leftover budget
// ============================================================================
class DecodeFirstScheduler : public Scheduler {
public:
    DecodeFirstScheduler(int chunk_size=16, int max_batch_tokens=256,
                         int max_prefill_entries=8);
    void add_request(Request req) override;
    ScheduleStep step() override;
    void finish_request(int req_id) override;
    int num_active() const override { return (int)waiting_.size()+(int)decoding_.size(); }
private:
    int chunk_size_, max_batch_tokens_, max_prefill_entries_;
    std::deque<Request> waiting_;
    std::vector<Request> decoding_;
};

}  // namespace engine
}  // namespace lightllm
