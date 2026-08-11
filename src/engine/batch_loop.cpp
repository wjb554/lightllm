/// BatchMainLoop implementation — orchestrates Scheduler + EngineServer.
///
/// LIFECYCLE (per step):
///   1. scheduler_->step()  -> ScheduleStep (heterogeneous batch)
///   2. engine_.step(step, states_) -> vector<SampledToken>
///   3. For each SampledToken: record TTFT, update metrics, check termination
///   4. For finished requests: engine_.release_request(),
///      scheduler_->finish_request()
///
/// NOTES:
///   - EngineServer::step() already appends tokens to state.generated_tokens,
///     updates state.seq_len, sets state.next_hidden_state, and sets
///     state.finished on termination.  The batch loop must NOT duplicate
///     these mutations.
///   - With speculative decoding, EngineServer::speculative_step() appends
///     0..K additional tokens to state.generated_tokens IN THE SAME step
///     call (after the normal decode token is returned).  These speculative
///     tokens count toward output_len but do NOT increment total_steps_.
///   - state.finished can be set by either the normal decode EOS (st.is_eos)
///     or by a speculative EOS token (where st.is_eos is false).  The
///     termination check must handle both.

#include "lightllm/engine/batch_loop.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace lightllm {
namespace engine {

// ============================================================================
// Constructor
// ============================================================================

BatchMainLoop::BatchMainLoop(EngineServer& engine,
                             SchedulerPolicy policy,
                             int chunk_size,
                             int max_batch_tokens,
                             int max_prefill_entries)
    : engine_(engine)
    , chunk_size_(chunk_size)
    , max_batch_tokens_(max_batch_tokens)
    , epoch_(std::chrono::steady_clock::now())
{
    // For DecodeFirst we bypass the factory so we can pass the
    // user-specified max_batch_tokens (the factory hardcodes 256).
    // For all other policies the factory signature matches.
    if (policy == SchedulerPolicy::DecodeFirst) {
        scheduler_ = std::make_unique<DecodeFirstScheduler>(chunk_size_,
                                                            max_batch_tokens_,
                                                            max_prefill_entries);
    } else {
        scheduler_ = Scheduler::create(policy, chunk_size_);
    }

    printf("[BatchMainLoop] policy=%d chunk_size=%d max_batch_tokens=%d max_prefill_entries=%d\n",
           static_cast<int>(policy), chunk_size_, max_batch_tokens_, max_prefill_entries);
}

// ============================================================================
// Speculative decoding
// ============================================================================

void BatchMainLoop::enable_speculative_decode(int num_draft_tokens,
                                              const std::string& draft_model_dir,
                                              int draft_layers,
                                              bool verify_batch,
                                              bool probability_acceptance) {
    SpeculativeDecodeConfig cfg;
    cfg.enabled                = true;
    cfg.num_draft_tokens       = num_draft_tokens;
    cfg.draft_model_dir        = draft_model_dir;
    cfg.draft_layers           = draft_layers;
    cfg.verify_batch           = verify_batch;
    cfg.probability_acceptance = probability_acceptance;
    engine_.enable_speculative_decode(cfg);
}

BatchMainLoop::~BatchMainLoop() {
    fprintf(stderr, "[BatchMainLoop] destructor: %zu states\n", states_.size());
    fflush(stderr);
    for (auto& [id, state] : states_) {
        if (state && !state->block_tables.empty()) {
            engine_.release_request(*state);
        }
    }
    fprintf(stderr, "[BatchMainLoop] destructor done\n");
    fflush(stderr);
}

// ============================================================================
// wall_clock
// ============================================================================

double BatchMainLoop::wall_clock() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - epoch_).count();
}

// ============================================================================
// submit
// ============================================================================

int BatchMainLoop::submit(const std::vector<int>& prompt_tokens,
                          int max_new_tokens,
                          int eos_token_id,
                          const std::string& json_schema,
                          const std::string& regex,
                          float temperature)
{
    int id = next_id_++;

    // 1. Create ONE scheduler-side Request.  We need it for both
    //    create_request_state() (by const ref) and add_request() (by move).
    Request sreq;
    sreq.id              = id;
    sreq.prompt_tokens   = prompt_tokens;
    sreq.max_new_tokens  = max_new_tokens;
    sreq.eos_token_id    = eos_token_id;
    sreq.json_schema     = json_schema;
    sreq.regex           = regex;
    sreq.temperature     = temperature;
    sreq.state           = Request::WAITING;
    sreq.num_computed    = 0;

    // 2. Create engine-side RequestState BEFORE moving sreq into the
    //    scheduler.  create_request_state takes a const reference so it
    //    reads from sreq without consuming it.
    auto state = engine_.create_request_state(sreq, engine_.hidden_dim());
    states_[id] = std::move(state);

    // 3. Enqueue in scheduler (consumes sreq via move).
    scheduler_->add_request(std::move(sreq));

    // 4. Create metrics record.
    BatchMetrics m;
    m.request_id        = id;
    m.prompt_len        = static_cast<int>(prompt_tokens.size());
    m.output_len        = 0;
    m.arrival_time_sec  = wall_clock();
    m.finished          = false;
    m.first_token_recorded = false;

    size_t idx = metrics_.size();
    metrics_.push_back(m);
    metrics_index_[id] = idx;

    return id;
}

// ============================================================================
// step
// ============================================================================

bool BatchMainLoop::step() {
    // ---- 1. SCHEDULE ----
    ScheduleStep batch = scheduler_->step();

    if (batch.empty()) {
        return false;
    }

    total_steps_++;

    // ---- 2. RECORD STEP METRICS (before engine execution) ----
    BatchStepMetrics sm;
    sm.step_number      = total_steps_;
    sm.step_time_sec    = wall_clock();
    sm.tokens_processed = batch.total_tokens();
    sm.num_requests     = static_cast<int>(batch.entries.size());

    for (const auto& entry : batch.entries) {
        if (entry.is_prefill) {
            sm.prefill_tokens += entry.num_tokens;
            sm.num_prefill_entries++;
        } else {
            sm.decode_tokens += entry.num_tokens;
            sm.num_decode_entries++;
        }
    }

    // ---- 3. EXECUTE (GPU forward pass) ----
    std::vector<SampledToken> output_tokens;
    // Decode requests handled ENTIRELY by speculate() (verify-as-decode).
    std::vector<int> spec_decode_ids;

    if (engine_.is_speculative()) {
        // verify-as-decode: prefill entries go to step() (which also keeps the
        // pre-sample producing the first token); ALL decode entries go to
        // speculate(), which produces the base + drafts in ONE target forward
        // per request — eliminating step()'s separate decode pass.
        std::vector<ScheduleEntry> step_entries, spec_entries;
        for (const auto& e : batch.entries) {
            if (e.is_prefill) step_entries.push_back(e);
            else              spec_entries.push_back(e);
        }
        ScheduleStep step_batch = batch; step_batch.entries = step_entries;
        ScheduleStep spec_batch = batch; spec_batch.entries = spec_entries;

        output_tokens = engine_.step(step_batch, states_);
        engine_.speculate(spec_batch, states_);
        for (const auto& e : spec_entries)
            spec_decode_ids.push_back(e.request_idx);
    } else {
        output_tokens = engine_.step(batch, states_);
    }

    // ---- 4. PROCESS RESULTS ----
    double now = wall_clock();

    // Non-spec decode tokens (returned by step()).
    for (const SampledToken& st : output_tokens) {
        int req_id = st.request_id;

        // Look up state (must exist — engine references it).
        auto state_it = states_.find(req_id);
        if (state_it == states_.end()) {
            fprintf(stderr,
                    "WARNING: BatchMainLoop::step got token for unknown "
                    "request %d\n", req_id);
            continue;
        }
        RequestState& state = *state_it->second;

        // Look up metrics.
        auto idx_it = metrics_index_.find(req_id);
        if (idx_it == metrics_index_.end()) {
            fprintf(stderr,
                    "WARNING: BatchMainLoop::step no metrics for request %d\n",
                    req_id);
            continue;
        }
        BatchMetrics& m = metrics_[idx_it->second];

        // --- Record TTFT on first decode token ---
        if (!m.first_token_recorded) {
            m.first_token_time_sec = now;
            m.first_token_recorded = true;
        }

        // --- Update token count ---
        m.output_len = static_cast<int>(state.generated_tokens.size());

        // --- Check termination ---
        if (st.is_eos || state.finished) {
            m.finished        = true;
            m.finish_time_sec = now;

            // Notify scheduler to purge from its decoding queue.
            scheduler_->finish_request(req_id);

            // Release ALL KV cache blocks across every layer.
            engine_.release_request(state);
        }
    }

    // Spec decode requests: state was mutated by speculate() (no SampledToken).
    for (int req_id : spec_decode_ids) {
        auto state_it = states_.find(req_id);
        if (state_it == states_.end()) continue;
        RequestState& state = *state_it->second;
        auto idx_it = metrics_index_.find(req_id);
        if (idx_it == metrics_index_.end()) continue;
        BatchMetrics& m = metrics_[idx_it->second];

        // TTFT on first emitted token (the pre-sampled first token or the
        // first speculate-accepted token).
        if (!m.first_token_recorded && !state.generated_tokens.empty()) {
            m.first_token_time_sec = now;
            m.first_token_recorded = true;
        }
        m.output_len = static_cast<int>(state.generated_tokens.size());

        if (state.finished) {
            m.finished        = true;
            m.finish_time_sec = now;
            scheduler_->finish_request(req_id);
            engine_.release_request(state);
        }
    }

    // Record queue depth AFTER processing (for throughput monitoring).
    sm.active_after = scheduler_->num_active();
    step_metrics_.push_back(sm);

    return true;
}

// ============================================================================
// run
// ============================================================================

void BatchMainLoop::run() {
    while (has_active()) {
        bool did_work = step();
        if (!did_work) {
            // Idle — avoid busy-waiting.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

// ============================================================================
// has_active / active_count
// ============================================================================

bool BatchMainLoop::has_active() const {
    return scheduler_->num_active() > 0;
}

const std::vector<int>& BatchMainLoop::generated_tokens(int request_id) const {
    static const std::vector<int> empty;
    auto it = states_.find(request_id);
    if (it == states_.end()) return empty;
    return it->second->generated_tokens;
}

int BatchMainLoop::spec_accepted(int request_id) const {
    auto it = states_.find(request_id);
    return (it != states_.end()) ? it->second->spec_accepted : 0;
}

int BatchMainLoop::spec_drafted(int request_id) const {
    auto it = states_.find(request_id);
    return (it != states_.end()) ? it->second->spec_drafted : 0;
}

SpeculativeStats BatchMainLoop::spec_stats(int request_id) const {
    auto it = states_.find(request_id);
    if (it == states_.end()) return {};
    return it->second->spec_stats;
}

int BatchMainLoop::active_count() const {
    return scheduler_->num_active();
}

// ============================================================================
// metrics (single request lookup)
// ============================================================================

const BatchMetrics& BatchMainLoop::metrics(int request_id) const {
    auto it = metrics_index_.find(request_id);
    if (it == metrics_index_.end())
        throw std::runtime_error("Unknown request ID: "
                                 + std::to_string(request_id));
    return metrics_[it->second];
}

// ============================================================================
// total_tokens_generated
// ============================================================================

int BatchMainLoop::total_tokens_generated() const {
    int total = 0;
    for (const auto& m : metrics_)
        total += m.output_len;
    return total;
}

// ============================================================================
// Percentile helper
// ============================================================================

double BatchMainLoop::percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());

    // Linear-interpolation percentile (same as numpy/pandas default).
    double idx = (p / 100.0) * static_cast<double>(values.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = lo + 1;
    if (hi >= values.size()) return values.back();

    double frac = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

// ============================================================================
// Aggregate stat helpers
// ============================================================================

/// Collect a metric from all FINISHED requests.  In-flight requests are
/// excluded so they don't skew latency percentiles.
static std::vector<double> collect_finished(
    const std::vector<BatchMetrics>& metrics,
    double (BatchMetrics::*fn)() const)
{
    std::vector<double> out;
    for (const auto& m : metrics) {
        if (m.finished) {
            double v = (m.*fn)();
            out.push_back(v);
        }
    }
    return out;
}

double BatchMainLoop::avg_ttft_ms() const {
    auto vals = collect_finished(metrics_, &BatchMetrics::ttft_ms);
    if (vals.empty()) return 0.0;
    return std::accumulate(vals.begin(), vals.end(), 0.0)
           / static_cast<double>(vals.size());
}

double BatchMainLoop::avg_tpot_ms() const {
    auto vals = collect_finished(metrics_, &BatchMetrics::tpot_ms);
    if (vals.empty()) return 0.0;
    return std::accumulate(vals.begin(), vals.end(), 0.0)
           / static_cast<double>(vals.size());
}

double BatchMainLoop::avg_latency_ms() const {
    auto vals = collect_finished(metrics_, &BatchMetrics::latency_ms);
    if (vals.empty()) return 0.0;
    return std::accumulate(vals.begin(), vals.end(), 0.0)
           / static_cast<double>(vals.size());
}

double BatchMainLoop::p50_ttft_ms() const {
    return percentile(collect_finished(metrics_, &BatchMetrics::ttft_ms), 50.0);
}
double BatchMainLoop::p50_tpot_ms() const {
    return percentile(collect_finished(metrics_, &BatchMetrics::tpot_ms), 50.0);
}
double BatchMainLoop::p50_latency_ms() const {
    return percentile(collect_finished(metrics_, &BatchMetrics::latency_ms), 50.0);
}

double BatchMainLoop::p90_ttft_ms() const {
    return percentile(collect_finished(metrics_, &BatchMetrics::ttft_ms), 90.0);
}
double BatchMainLoop::p90_tpot_ms() const {
    return percentile(collect_finished(metrics_, &BatchMetrics::tpot_ms), 90.0);
}
double BatchMainLoop::p90_latency_ms() const {
    return percentile(collect_finished(metrics_, &BatchMetrics::latency_ms), 90.0);
}

double BatchMainLoop::p95_ttft_ms() const {
    return percentile(collect_finished(metrics_, &BatchMetrics::ttft_ms), 95.0);
}
double BatchMainLoop::p95_tpot_ms() const {
    return percentile(collect_finished(metrics_, &BatchMetrics::tpot_ms), 95.0);
}
double BatchMainLoop::p95_latency_ms() const {
    return percentile(collect_finished(metrics_, &BatchMetrics::latency_ms), 95.0);
}

double BatchMainLoop::p99_ttft_ms() const {
    return percentile(collect_finished(metrics_, &BatchMetrics::ttft_ms), 99.0);
}
double BatchMainLoop::p99_tpot_ms() const {
    return percentile(collect_finished(metrics_, &BatchMetrics::tpot_ms), 99.0);
}
double BatchMainLoop::p99_latency_ms() const {
    return percentile(collect_finished(metrics_, &BatchMetrics::latency_ms), 99.0);
}

double BatchMainLoop::throughput_tok_s() const {
    double elapsed = wall_clock();
    if (elapsed <= 0.0) return 0.0;
    return static_cast<double>(total_tokens_generated()) / elapsed;
}

}  // namespace engine
}  // namespace lightllm
