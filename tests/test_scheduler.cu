/// Scheduler comparison: FCFS vs PrefillFirst
/// Demonstrates chunked prefill + interleaved scheduling.
#include <cstdio>
#include <vector>
#include <string>
#include <chrono>
#include "lightllm/engine/scheduler.h"

using namespace lightllm::engine;

static void run_simulation(const char* name, SchedulerPolicy policy, int chunk_size,
                            const std::vector<std::vector<int>>& prompts, int max_new) {
    printf("\n=== %s (chunk=%d) ===\n", name, chunk_size);

    auto sched = Scheduler::create(policy, chunk_size);
    for (size_t i = 0; i < prompts.size(); i++) {
        Request req;
        req.id = (int)i;
        req.prompt_tokens = prompts[i];
        req.max_new_tokens = max_new;
        sched->add_request(std::move(req));
    }

    int total_prefill_chunks = 0, total_decode_steps = 0, total_tokens = 0;
    printf("%-6s %-8s %-7s %-6s %s\n", "Step", "Type", "ReqID", "Tokens", "Pos");

    int step_no = 0;
    while (true) {
        auto step = sched->step();
        if (step.empty()) break;
        step_no++;

        for (auto& e : step.entries) {
            const char* type = e.is_prefill ? "PREFILL" : "DECODE";
            printf("%-6d %-8s %-6d %-7d %d\n", step_no, type, e.request_idx,
                   e.num_tokens, e.start_pos);
            if (e.is_prefill) total_prefill_chunks++;
            else total_decode_steps++;
            total_tokens += e.num_tokens;
        }

        // Simulate: if a decode entry produced EOS, finish the request
        for (auto& e : step.entries) {
            if (!e.is_prefill) {
                static int counter = 0;
                if (++counter % (max_new + 1) == 0)  // simulate EOS after max_new tokens
                    sched->finish_request(e.request_idx);
            }
        }
    }

    printf("Result: %d prefill chunks, %d decode steps, %d total tokens, %d steps\n",
           total_prefill_chunks, total_decode_steps, total_tokens, step_no);
}

int main() {
    // 3 users with different prompt lengths
    std::vector<std::vector<int>> prompts = {
        {576, 8319, 315, 13466, 374},          // 5 tokens
        {15837, 467, 315, 1605},               // 4 tokens
        {10585, 374, 279, 18865, 1234, 5678},  // 6 tokens
    };
    int max_new = 3; // generate 3 tokens per user

    // Compare all schedulers
    run_simulation("FCFS (chunk=32)", SchedulerPolicy::FCFS, 32, prompts, max_new);
    run_simulation("PrefillFirst (chunk=32)", SchedulerPolicy::PrefillFirst, 32, prompts, max_new);
    run_simulation("DecodeFirst (chunk=32)", SchedulerPolicy::DecodeFirst, 32, prompts, max_new);
    run_simulation("DecodeFirst (chunk=4)", SchedulerPolicy::DecodeFirst, 4, prompts, max_new);

    printf("\n--- Comparison ---\n");
    printf("FCFS:        sequential, lowest concurrency, simplest\n");
    printf("PrefillFirst: prefills get priority, decode batched. Good for throughput\n");
    printf("DecodeFirst:  decode NEVER stalls, prefill uses leftover budget.\n");
    printf("              Best for interactive scenarios (chatbots, copilots)\n");
    printf("              TPOT stable even under high arrival rate\n");
    return 0;
}
