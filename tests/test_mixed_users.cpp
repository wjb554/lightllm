/// LightLLM — multi-user mixed-length test.
/// One BatchMainLoop, all requests submitted, scheduler handles them all.

#include <cstdio>
#include <map>
#include <vector>
#include <algorithm>
#include <chrono>
#include "lightllm/engine/engine.h"
#include "lightllm/engine/batch_loop.h"

using namespace lightllm::engine;

static int passed=0, failed=0;
static void CHECK(const char* n, bool c) {
    if(c){passed++;}else{failed++;fprintf(stderr,"  FAIL: %s\n",n);}
}
static double pct(std::vector<double>& v, double p) {
    if(v.empty())return 0; std::sort(v.begin(),v.end());
    return v[std::min((int)(p/100.0*v.size()),(int)v.size()-1)];
}

int main() {
    printf("=============================================================\n");
    printf("  LightLLM — Multi-User Mixed-Length Test\n");
    printf("  16 requests, 1-20 token prompts, one scheduler handles all\n");
    printf("=============================================================\n\n");

    printf("Loading model...\n"); fflush(stdout);
    EngineServer engine("models/qwen2.5-0.5b", 0, 256);
    BatchMainLoop batch(engine, SchedulerPolicy::DecodeFirst, 16, 256);
    printf("Ready.\n\n"); fflush(stdout);

    struct Req { std::vector<int> prompt; int max_new; int cat; };
    std::vector<Req> reqs = {
        // 4 x 1-token (cat 0)
        {{576}, 6, 0}, {{576}, 6, 0},
        {{1000}, 6, 0}, {{2000}, 6, 0},
        // 6 x 3-5 token (cat 1)
        {{576,8319,315}, 8, 1},
        {{576,8319,315,13466,374}, 10, 1},
        {{576,8319,315,13466,374}, 10, 1},
        {{1001,1002,1003}, 8, 1},
        {{2001,2002,2003,2004}, 9, 1},
        {{3001,3002,3003,3004,3005}, 10, 1},
        // 6 x 10-20 token (cat 2)
        {{100,101,102,103,104,105,106,107,108,109}, 10, 2},
        {{200,201,202,203,204,205,206,207,208,209}, 10, 2},
        {{300,301,302,303,304,305,306,307,308,309,
          310,311,312,313,314,315,316,317,318,319}, 10, 2},
        {{400,401,402,403,404,405,406,407,408,409,
          410,411,412,413,414}, 10, 2},
        {{500,501,502,503,504,505,506,507,508,509,
          510,511,512}, 10, 2},
        {{600,601,602,603,604,605,606,607,608,609,
          610,611,612,613,614,615,616,617,618,619}, 10, 2},
    };

    // Submit ALL requests — scheduler manages them all
    printf("Submitting %zu requests...\n", reqs.size()); fflush(stdout);
    for(auto& r:reqs) batch.submit(r.prompt, r.max_new, 151643);

    // Run until all done
    printf("Running (Continuous Batching, DecodeFirst)...\n"); fflush(stdout);
    auto t0=std::chrono::steady_clock::now();
    int steps=0;
    while(batch.has_active()&&steps<1000){batch.step();steps++;}
    auto t1=std::chrono::steady_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    // Results
    int finished=0, total_out=0;
    std::map<int,std::vector<double>> ttft_by_cat, tpot_by_cat, lat_by_cat;
    for(auto& m:batch.all_metrics()){
        if(!m.finished) continue;
        finished++; total_out+=m.output_len;
        int cat=-1;
        for(auto& r:reqs) if(r.prompt==std::vector<int>(m.prompt_len,0)) continue;
        if(m.prompt_len==1) cat=0;
        else if(m.prompt_len<=5) cat=1;
        else cat=2;
        ttft_by_cat[cat].push_back(m.ttft_ms());
        tpot_by_cat[cat].push_back(m.tpot_ms());
        lat_by_cat[cat].push_back(m.latency_ms());
    }

    printf("\n=============================================================\n");
    printf("  16 requests: %d finished, %d output tokens, %.0f ms\n",
           finished, total_out, ms);
    printf("  Throughput: %.1f tok/s | %d steps\n\n", total_out*1000.0/ms, steps);

    printf("  %-16s | %5s | %7s | %7s | %7s\n",
           "Prompt len","Count","P50 TTFT","P50 TPOT","P50 Total");
    printf("  -----------------|-------|---------|---------|--------\n");
    const char* labels[]={"1 tok","3-5 tok","10-20 tok"};
    for(int c=0;c<3;c++){
        if(!ttft_by_cat[c].empty())
            printf("  %-16s | %5zu | %6.0fms | %6.0fms | %6.0fms\n",
                   labels[c], ttft_by_cat[c].size(),
                   pct(ttft_by_cat[c],50), pct(tpot_by_cat[c],50), pct(lat_by_cat[c],50));
    }

    printf("\n=============================================================\n");
    CHECK("all_finished", finished==16);
    CHECK("tokens_generated", total_out>40);
    CHECK("throughput_reasonable", total_out*1000.0/ms > 5.0);
    printf("  RESULTS: %d passed, %d failed\n", passed, failed);
    printf("=============================================================\n");
    return failed>0?1:0;
}
