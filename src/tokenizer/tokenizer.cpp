/// Tokenizer — HuggingFace tokenizer.json loader (GPT-2 BPE).
/// Supports decode (token IDs → text) and encode (text → token IDs via BPE).
#include "lightllm/tokenizer/tokenizer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <set>

namespace lightllm {
namespace tokenizer {

// ---- JSON helpers ----
static std::string extract_json_str(const std::string& json, size_t& pos) {
    while (pos < json.size() && json[pos] != '"') pos++;
    if (pos >= json.size()) return "";
    pos++;
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                case 'u': { // \uXXXX
                    int cp = 0;
                    for (int i = 0; i < 4 && pos+1+i < json.size(); i++)
                        cp = cp * 16 + (json[pos+1+i] <= '9' ? json[pos+1+i]-'0' : (json[pos+1+i]|32)-'a'+10);
                    pos += 4;
                    if (cp < 0x80) result += (char)cp;
                    else if (cp < 0x800) { result += (char)(0xC0|cp>>6); result += (char)(0x80|cp&0x3F); }
                    else { result += (char)(0xE0|cp>>12); result += (char)(0x80|(cp>>6)&0x3F); result += (char)(0x80|cp&0x3F); }
                    break;
                }
                default: result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        pos++;
    }
    if (pos < json.size()) pos++;
    return result;
}

// ---- Byte encoder (GPT-2 style) ----
static const char* byte_to_unicode() {
    // GPT-2 maps bytes 33-126 directly, remaps 0-32 and 127-255 to avoid control chars
    static char map[256][4] = {};
    static bool init = false;
    if (init) return (const char*)map;
    init = true;
    // Printable range
    for (int b = 33; b <= 126; b++) { map[b][0] = (char)b; map[b][1] = 0; }
    // Remap others to unicode
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (b >= 33 && b <= 126) continue;
        int cp = 256 + n++;
        if (cp < 0x80) { map[b][0] = (char)cp; map[b][1] = 0; }
        else if (cp < 0x800) { map[b][0] = (char)(0xC0|cp>>6); map[b][1] = (char)(0x80|cp&0x3F); map[b][2] = 0; }
        else { map[b][0] = (char)(0xE0|cp>>12); map[b][1] = (char)(0x80|(cp>>6)&0x3F); map[b][2] = (char)(0x80|cp&0x3F); map[b][3] = 0; }
    }
    return (const char*)map;
}

// ---- Constructor ----
Tokenizer::Tokenizer(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open tokenizer: " + path);
    std::stringstream ss; ss << f.rdbuf();
    std::string json = ss.str();

    // ---- Parse vocab ----
    size_t pos = json.find("\"vocab\"");
    if (pos == std::string::npos) throw std::runtime_error("tokenizer.json: no 'vocab'");
    pos = json.find('{', pos);
    if (pos == std::string::npos) throw std::runtime_error("tokenizer.json: malformed vocab");
    pos++;
    while (pos < json.size()) {
        while (pos < json.size() && ((unsigned char)json[pos] <= ' ' || json[pos] == ',')) pos++;
        if (pos >= json.size() || json[pos] == '}') break;
        std::string tok = extract_json_str(json, pos);
        while (pos < json.size() && json[pos] != ':') pos++;
        pos++; while (pos < json.size() && (unsigned char)json[pos] <= ' ') pos++;
        int id = 0;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') { id = id*10+(json[pos]-'0'); pos++; }
        if (!tok.empty()) {
            if (id >= (int)id_to_token_.size()) id_to_token_.resize(id+1);
            id_to_token_[id] = tok;
            token_to_id_[tok] = id;
        }
        if (pos > json.size()-2) break;
    }

    // ---- Parse merges (for BPE encoding) ----
    pos = json.find("\"merges\"");
    if (pos != std::string::npos) {
        pos = json.find('[', pos);
        if (pos != std::string::npos) {
            pos++;
            while (pos < json.size() && json[pos] != ']') {
                while (pos < json.size() && json[pos] != '"') { if (json[pos]==']') goto done_merges; pos++; }
                if (pos >= json.size()) break;
                pos++; // skip opening "
                size_t start = pos;
                while (pos < json.size() && json[pos] != '"') pos++;
                std::string merge = json.substr(start, pos-start);
                if (pos < json.size()) pos++;
                // Merge format: "token_a token_b" (space-separated)
                size_t sp = merge.find(' ');
                if (sp != std::string::npos) {
                    std::string a = merge.substr(0, sp), b = merge.substr(sp+1);
                    merges_.push_back({a, b});
                }
            }
            done_merges:;
        }
    }

    if (id_to_token_.empty()) throw std::runtime_error("tokenizer.json: empty vocab");

    // ---- Build byte decoder ----
    const char* b2u = byte_to_unicode();
    for (int b = 0; b < 256; b++) {
        byte_decoder_[std::string(b2u + b*4)] = (unsigned char)b;
    }
}

// ---- Decode ----
const std::string& Tokenizer::id_to_str(int id) const {
    static const std::string unknown = "<unk>";
    if (id < 0 || id >= (int)id_to_token_.size()) return unknown;
    return id_to_token_[id];
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (int id : token_ids) {
        const auto& s = id_to_str(id);
        // Replace byte-level encoded chars with actual bytes
        size_t i = 0;
        while (i < s.size()) {
            // Check for multi-byte UTF-8 sequences
            int len = 1;
            if ((unsigned char)s[i] >= 0xC0) len = 2;
            if ((unsigned char)s[i] >= 0xE0) len = 3;
            if ((unsigned char)s[i] >= 0xF0) len = 4;
            std::string ch = s.substr(i, len);
            auto it = byte_decoder_.find(ch);
            if (it != byte_decoder_.end()) {
                result += (char)it->second;
            } else {
                result += ch;
            }
            i += len;
        }
    }
    return result;
}

// ---- Encode (BPE) ----
std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (merges_.empty()) {
        // Fallback: word-level lookup
        std::vector<int> ids;
        std::string word;
        for (char c : text) {
            if (c == ' ') {
                if (!word.empty()) {
                    auto it = token_to_id_.find("Ġ" + word);
                    if (it != token_to_id_.end()) ids.push_back(it->second);
                    word.clear();
                }
            } else { word += c; }
        }
        if (!word.empty()) {
            auto it = token_to_id_.find(word);
            if (it != token_to_id_.end()) ids.push_back(it->second);
        }
        return ids;
    }

    // Full BPE encoding
    const char* b2u = byte_to_unicode();

    // Step 1: Convert text to BPE tokens (one per byte initially)
    std::vector<std::string> tokens;
    for (char c : text) {
        unsigned char b = (unsigned char)c;
        tokens.push_back(std::string(b2u + b*4));
    }

    // Step 2: Apply BPE merges in order
    for (auto& merge : merges_) {
        std::vector<std::string> new_tokens;
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i + 1 < tokens.size() && tokens[i] == merge.first && tokens[i+1] == merge.second) {
                new_tokens.push_back(merge.first + merge.second);
                i++; // skip next token
            } else {
                new_tokens.push_back(tokens[i]);
            }
        }
        tokens = std::move(new_tokens);
    }

    // Step 3: Map tokens to IDs
    std::vector<int> ids;
    for (auto& tok : tokens) {
        auto it = token_to_id_.find(tok);
        if (it != token_to_id_.end()) ids.push_back(it->second);
    }
    return ids;
}

}  // namespace tokenizer
}  // namespace lightllm
