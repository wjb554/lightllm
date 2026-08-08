/// Find max concurrent users. Submit N, run until idle. Binary search.
#include <cstdio>
#include "lightllm/engine/engine.h"
#include "lightllm/engine/batch_loop.h"
using namespace lightllm::engine;

int test_N(EngineServer& engine, int N) {
    BatchMainLoop batch(engine, SchedulerPolicy::DecodeFirst, 16, 256, 999);
    std::vector<int> prompt = {576, 8319, 315, 13466, 374};
    for (int i=0; i<N; i++) batch.submit(prompt, 4, 151643);

    int steps=0, max_steps=N*20;
    while (batch.has_active() && steps<max_steps) {
        batch.step(); steps++;
    }
    if (!batch.has_active()) {
        int tok=0;
        for (auto& m:batch.all_metrics()) if(m.finished) tok+=m.output_len;
        return tok;
    }
    return -1; // didn't finish = crash
}

int main() {
    printf("=== Max Concurrent Users ===\n");
    printf("RTX 2060 6GB | 0.5B fp32 | prompt=5 tok | gen=4 tok\n\n");

    EngineServer engine("models/qwen2.5-0.5b", 0, 256);
    printf("KV blocks: %d | KV pool: 771MB | Model: ~2GB\n\n", engine.num_blocks());
    fflush(stdout);

    int N_vals[] = {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
    int max_ok = 0;
    int n_vals_count = sizeof(N_vals)/sizeof(N_vals[0]);

    printf("  N     |  Status  |  Tokens |  Blocks\n");
    printf("  ------|----------|---------|--------\n");
    fflush(stdout);

    for (int ni=0; ni < n_vals_count; ni++) {
        int N = N_vals[ni];
        int tok = test_N(engine, N);
        int free = engine.free_blocks();

        if (tok > 0) {
            printf("  %4d  |  OK      | %7d | %6d free\n", N, tok, free);
            max_ok = N;
        } else {
            printf("  %4d  |  FAIL    |       - |      -\n", N);
            break;
        }
        fflush(stdout);
    }

    printf("\n  Max concurrent stable: %d users\n", max_ok);
    printf("  Theoretical max (KV blocks): %d users (30tok/req)\n",
           engine.num_blocks() / 2);
    return 0;
}
