#!/usr/bin/env python3
"""Small deterministic HTTP backend for the protected AppConfig smoke test."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


REQUESTS = 0
REQUESTS_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format_string, *args):
        print("backend " + (format_string % args), flush=True)

    def _send(self, status, body, content_type="application/json"):
        encoded = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, "ok\n", "text/plain; charset=utf-8")
            return
        if self.path == "/metrics":
            self._send(200, "", "text/plain; version=0.0.4")
            return
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            self._send(404, json.dumps({"error": "not found"}))
            return

        length = int(self.headers.get("Content-Length", "0"))
        request_body = b""
        if length:
            request_body = self.rfile.read(length)

        try:
            request_document = json.loads(request_body or b"{}")
        except json.JSONDecodeError:
            request_document = {}
        sleep_ms = int(
            request_document.get("smoke_sleep_ms", os.environ.get("SMOKE_SLEEP_MS", "0"))
        )
        if sleep_ms:
            time.sleep(sleep_ms / 1000.0)

        global REQUESTS
        with REQUESTS_LOCK:
            REQUESTS += 1
            request_number = REQUESTS

        if self.path == "/v1/completions":
            body = {
                "id": f"cmpl-smoke-{request_number}",
                "object": "text_completion",
                "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
                "model": "appconfig-smoke-model",
            }
        else:
            body = {
                "id": f"chatcmpl-smoke-{request_number}",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "model": "appconfig-smoke-model",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        self._send(200, json.dumps(body, separators=(",", ":")))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"backend listening on {port}", flush=True)
    server.serve_forever()
