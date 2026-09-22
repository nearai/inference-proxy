"""Deterministic OpenAI-compatible engine used by the real-Agent smoke test."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class EngineState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.hold = False
        self.release = False
        self.active = 0
        self.completed = 0
        self.total_requests = 0

    def stats(self) -> dict[str, int | bool]:
        with self.condition:
            return {
                "active": self.active,
                "completed": self.completed,
                "total_requests": self.total_requests,
                "hold": self.hold,
                "release": self.release,
            }

    def reset(self) -> None:
        with self.condition:
            self.hold = False
            self.release = False
            self.active = 0
            self.completed = 0
            self.total_requests = 0
            self.condition.notify_all()

    def set_hold(self) -> None:
        with self.condition:
            self.hold = True
            self.release = False

    def set_release(self) -> None:
        with self.condition:
            self.release = True
            self.hold = False
            self.condition.notify_all()


STATE = EngineState()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        # Keep the engine output stable and useful if the test fails.
        print("engine:", format % args, flush=True)

    def send_json(self, status: int, payload: object) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "test-model",
                            "object": "model",
                            "owned_by": "deterministic-mock-engine",
                        }
                    ],
                },
            )
            return
        if self.path == "/control/stats":
            self.send_json(200, STATE.stats())
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/control/reset":
            STATE.reset()
            self.send_json(200, STATE.stats())
            return
        if self.path == "/control/hold":
            STATE.set_hold()
            self.send_json(200, STATE.stats())
            return
        if self.path == "/control/release":
            STATE.set_release()
            self.send_json(200, STATE.stats())
            return
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        with STATE.condition:
            STATE.total_requests += 1
            STATE.active += 1
            while STATE.hold and not STATE.release:
                STATE.condition.wait()

        payload = {
            "id": "chatcmpl-deterministic",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        self.send_json(200, payload)
        with STATE.condition:
            STATE.active -= 1
            STATE.completed += 1
            STATE.condition.notify_all()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8001), Handler)
    print("mock engine listening on :8001", flush=True)
    server.serve_forever()
