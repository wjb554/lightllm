#pragma once
/// Minimal HTTP server for LightLLM inference.
/// Single-threaded, blocking, handles one request at a time.
/// Binds to localhost:8080, returns JSON.

#include <functional>
#include <string>

namespace lightllm {
namespace server {

struct HttpRequest {
    std::string method;   // "POST"
    std::string path;     // "/v1/chat/completions"
    std::string body;     // JSON payload
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
};

using Handler = std::function<HttpResponse(const HttpRequest&)>;

/// Start a blocking HTTP server on the given port.
/// Calls `handler` for each request. Returns on error or signal.
void run_server(int port, Handler handler);

}  // namespace server
}  // namespace lightllm
