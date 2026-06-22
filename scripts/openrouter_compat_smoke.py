#!/usr/bin/env python3
"""Smoke-test OpenRouter-compatible Fusion payloads against a real proxy process.

The script starts a local mock direct-completions endpoint, launches the
inference-proxy binary with Fusion enabled, then sends:

  1. OpenRouter server-tool Fusion with nested `parameters`.
  2. OpenRouter plugin Fusion with `plugins: [{"id": "fusion", ...}]`.

No external services or API tokens are used.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


CLIENT_TOKEN = "local-client-token"
INTERNAL_TOKEN = "local-internal-token"
MODEL = "test-model"
PANEL_MODEL = "panel-a"
JUDGE_MODEL = "judge-a"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class MockDirectEndpoint(BaseHTTPRequestHandler):
    endpoint_port: int
    calls: list[str] = []

    def log_message(self, _fmt: str, *_args: Any) -> None:
        return

    def write_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/endpoints":
            self.write_json(
                200,
                {
                    "endpoints": [
                        {
                            "domain": f"http://127.0.0.1:{self.endpoint_port}",
                            "models": [PANEL_MODEL, JUDGE_MODEL, MODEL],
                        }
                    ]
                },
            )
            return
        if self.path == "/v1/attestation/report":
            self.write_json(200, {"ok": True})
            return
        self.write_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.write_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        serialized = json.dumps(payload)

        if "one member of a private multi-model panel" in serialized:
            self.calls.append("panel")
            self.write_json(
                200,
                {
                    "id": "chatcmpl-panel",
                    "object": "chat.completion",
                    "model": PANEL_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Panel recommends keeping the compatibility layer small.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                    },
                },
            )
            return

        if "strict JSON only" in serialized:
            self.calls.append(f"judge:{payload.get('model')}")
            self.write_json(
                200,
                {
                    "id": "chatcmpl-judge",
                    "object": "chat.completion",
                    "model": JUDGE_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "consensus": "compatibility works",
                                        "disagreements": [],
                                        "strengths": ["OpenRouter request shape accepted"],
                                        "risks": [],
                                        "synthesis_guidance": "Answer that compatibility works.",
                                    }
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                        "total_tokens": 7,
                    },
                },
            )
            return

        if "final synthesis model" in serialized:
            self.calls.append("synthesis")
            self.write_json(
                200,
                {
                    "id": "chatcmpl-final",
                    "object": "chat.completion",
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "OpenRouter Fusion compatibility works.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 6,
                        "total_tokens": 11,
                    },
                },
            )
            return

        self.write_json(500, {"error": "unexpected mock request", "payload": payload})


@contextlib.contextmanager
def run_server(handler: type[BaseHTTPRequestHandler], port: int):
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def wait_for_proxy(port: int, proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"proxy exited early with {proc.returncode}\n{output}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1).close()
            return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError("proxy did not become ready")


@contextlib.contextmanager
def run_proxy(binary: Path, mock_port: int, proxy_port: int):
    env = {
        **os.environ,
        "MODEL_NAME": MODEL,
        "TOKEN": CLIENT_TOKEN,
        "VLLM_BASE_URL": f"http://127.0.0.1:{mock_port}",
        "LISTEN_PORT": str(proxy_port),
        "DEV": "true",
        "FUSION_ENABLED": "true",
        "FUSION_ENDPOINTS_URL": f"http://127.0.0.1:{mock_port}/endpoints",
        "FUSION_INTERNAL_BEARER_TOKEN": INTERNAL_TOKEN,
        "FUSION_INTERNAL_MAX_ATTEMPTS": "1",
        "FUSION_PANEL_TIMEOUT_SECS": "10",
        "RUST_LOG": "warn",
    }
    proc = subprocess.Popen(
        [str(binary)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_proxy(proxy_port, proc)
        yield
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def post_json(port: int, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {CLIENT_TOKEN}",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def assert_fusion_response(name: str, response: dict[str, Any]) -> None:
    assert response["choices"][0]["message"]["content"] == "OpenRouter Fusion compatibility works."
    expected_usage = {
        "prompt_tokens": 9,
        "completion_tokens": 12,
        "total_tokens": 21,
    }
    assert response["usage"] == expected_usage, response
    metadata = response["nearai_fusion"]
    assert metadata["status"] == "invoked"
    assert metadata["judge"]["status"] == "ok"
    assert metadata["aggregate_usage"] == response["usage"]
    assert metadata["panel"][0]["model"] == PANEL_MODEL
    print(f"ok: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target/release/vllm-proxy-rs"),
        help="Path to the inference-proxy binary",
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.exists():
        raise SystemExit(f"binary not found: {binary}")

    mock_port = free_port()
    proxy_port = free_port()
    MockDirectEndpoint.endpoint_port = mock_port
    MockDirectEndpoint.calls = []

    with run_server(MockDirectEndpoint, mock_port), run_proxy(binary, mock_port, proxy_port):
        tool_response = post_json(
            proxy_port,
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Does compatibility work?"}],
                "tools": [
                    {
                        "type": "openrouter:fusion",
                        "parameters": {
                            "analysis_models": [f"~{PANEL_MODEL}"],
                            "model": JUDGE_MODEL,
                            "max_completion_tokens": 32,
                            "temperature": 0,
                        },
                    }
                ],
                "tool_choice": "required",
            },
        )
        assert_fusion_response("server tool parameters", tool_response)

        plugin_response = post_json(
            proxy_port,
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Does plugin compatibility work?"}],
                "plugins": [
                    {
                        "id": "fusion",
                        "analysis_models": [PANEL_MODEL],
                        "max_completion_tokens": 32,
                        "temperature": 0,
                    }
                ],
                "tool_choice": "required",
            },
        )
        assert_fusion_response("plugin", plugin_response)

    expected_calls = [
        "panel",
        f"judge:{JUDGE_MODEL}",
        "synthesis",
        "panel",
        f"judge:{MODEL}",
        "synthesis",
    ]
    assert MockDirectEndpoint.calls == expected_calls, MockDirectEndpoint.calls
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        raise
