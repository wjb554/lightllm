/// Tokenizer test — verify vocab load + decode.
#include <cstdio>
#include "lightllm/tokenizer/tokenizer.h"
using namespace lightllm::tokenizer;

int main(){
    printf("=== Tokenizer Test ===\n");
    Tokenizer tok("models/qwen2.5-0.5b/tokenizer.json");
    printf("Vocab size: %d\n", tok.vocab_size());

    // Test known tokens
    printf("id=576   → '%s'\n", tok.id_to_str(576).c_str());
    printf("id=374   → '%s'\n", tok.id_to_str(374).c_str());
    printf("id=151643 → '%s'\n", tok.id_to_str(151643).c_str());

    // Decode our test prompt
    auto text = tok.decode({576, 8319, 315, 13466, 374});
    printf("Decode: '%s'\n", text.c_str());

    printf("ALL PASSED\n");
    return 0;
}
