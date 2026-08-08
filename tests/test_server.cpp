/// LightLLM HTTP Inference Server — loads model, serves requests.
/// POST /v1/chat/completions  {"prompt":[ids],"max_tokens":N} → {"tokens":[...]}
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include "lightllm/engine/engine.h"
#include "lightllm/server/http_server.h"

using namespace lightllm::engine;
using namespace lightllm::server;

// Minimal JSON builder
static std::string json_obj(const std::string& content){
    return "{" + content + "}";
}
static std::string json_key(const std::string& k, const std::string& v){
    return "\"" + k + "\": " + v;
}
static std::string json_array(const std::vector<int>& ids){
    std::string s="[";
    for(size_t i=0;i<ids.size();i++){s+=std::to_string(ids[i]);if(i+1<ids.size())s+=",";}
    return s+"]";
}
// Minimal JSON parser — extracts value for key
static int json_int(const std::string& body, const std::string& key){
    auto p=body.find("\""+key+"\"");
    if(p==std::string::npos)return 0;
    p=body.find(':',p);
    while(p<body.size()&&(body[p]<'0'||body[p]>'9'))p++;
    return std::atoi(body.c_str()+p);
}
static std::vector<int> json_int_array(const std::string& body,const std::string& key){
    std::vector<int> v;
    auto p=body.find("\""+key+"\"");
    if(p==std::string::npos)return v;
    p=body.find('[',p);if(p==std::string::npos)return v;
    size_t e=body.find(']',p);
    for(size_t i=p+1;i<e;){while(i<e&&(body[i]<'0'||body[i]>'9'))i++;if(i>=e)break;size_t j=i;while(j<e&&body[j]>='0'&&body[j]<='9')j++;v.push_back(std::atoi(body.c_str()+i));i=j+1;}
    return v;
}

int main(){
    printf("=== LightLLM HTTP Server ===\nLoading model...\n");
    InferenceEngine engine("models/qwen2.5-0.5b");

    run_server(8080, [&](const HttpRequest& req) -> HttpResponse {
        HttpResponse resp;
        if(req.path=="/health"){resp.body="{\"status\":\"ok\"}";return resp;}

        auto prompt=json_int_array(req.body,"prompt");
        int max_tok=json_int(req.body,"max_tokens");
        if(max_tok==0)max_tok=16;
        if(prompt.empty())prompt={1}; // BOS token

        printf("Request: %zu prompt tokens, max_tokens=%d\n",prompt.size(),max_tok);

        GenerateParams params;params.max_new_tokens=max_tok;
        auto t0=std::chrono::steady_clock::now();
        auto result=engine.generate(prompt,params);
        auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count();

        int new_tokens=(int)result.token_ids.size()-(int)prompt.size();
        printf("  Generated %d tokens in %lld ms\n",new_tokens,ms);

        auto output=json_array(result.token_ids);
        resp.body=json_obj(
            json_key("tokens",output)+","+
            json_key("new_tokens",std::to_string(new_tokens))+","+
            json_key("time_ms",std::to_string(ms))
        );
        return resp;
    });
    return 0;
}
