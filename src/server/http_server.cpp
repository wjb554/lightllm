/// Minimal HTTP server using Winsock2.
/// Single-threaded, blocking, JSON-in/JSON-out.

#include "lightllm/server/http_server.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#define closesocket close
#endif

#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>

namespace lightllm {
namespace server {

// Parse HTTP request from raw bytes
static HttpRequest parse_request(const char* data, int len) {
    HttpRequest req;
    std::string raw(data, len);

    // Parse first line: "POST /path HTTP/1.1"
    size_t nl = raw.find("\r\n");
    if (nl == std::string::npos) return req;
    std::string first = raw.substr(0, nl);

    size_t sp1 = first.find(' ');
    size_t sp2 = first.find(' ', sp1 + 1);
    if (sp1 != std::string::npos) req.method = first.substr(0, sp1);
    if (sp1 != std::string::npos && sp2 != std::string::npos)
        req.path = first.substr(sp1 + 1, sp2 - sp1 - 1);

    // Find body (after \r\n\r\n)
    size_t body_start = raw.find("\r\n\r\n");
    if (body_start != std::string::npos)
        req.body = raw.substr(body_start + 4);

    return req;
}

// Build HTTP response string
static std::string build_response(const HttpResponse& resp) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << resp.status << " OK\r\n";
    oss << "Content-Type: " << resp.content_type << "\r\n";
    oss << "Content-Length: " << resp.body.size() << "\r\n";
    oss << "Access-Control-Allow-Origin: *\r\n";
    oss << "Connection: close\r\n";
    oss << "\r\n";
    oss << resp.body;
    return oss.str();
}

void run_server(int port, Handler handler) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);
#endif

    int sock = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { fprintf(stderr, "socket failed\n"); return; }

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((u_short)port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "bind failed (port %d in use?)\n", port);
        closesocket(sock);
        return;
    }
    listen(sock, 4);
    printf("LightLLM HTTP server listening on http://localhost:%d\n", port);
    fflush(stdout);

    char buf[65536];
    while (true) {
        struct sockaddr_in client;
        socklen_t client_len = sizeof(client);
        int client_sock = (int)accept(sock, (struct sockaddr*)&client, &client_len);
        if (client_sock < 0) continue;

        int n = recv(client_sock, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = 0;
            auto req = parse_request(buf, n);
            auto resp = handler(req);
            auto response_str = build_response(resp);
            send(client_sock, response_str.c_str(), (int)response_str.size(), 0);
        }
        closesocket(client_sock);
    }
    closesocket(sock);
}

}  // namespace server
}  // namespace lightllm
