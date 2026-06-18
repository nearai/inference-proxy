#!/usr/bin/env python3
"""Run a local Fusion E2E smoke test against live direct model proxies.

This starts:
  - a local backend shim for final synthesis calls, forwarding to a live direct
    completions proxy with FUSION_INTERNAL_BEARER_TOKEN;
  - optionally a local Brave LLM Context fixture, unless --real-brave is used;
  - the local inference-proxy binary with FUSION_ENABLED=true.

No production deployment is performed. The script intentionally does not print
tokens or environment values.
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
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_DIRECT_CHAT_URL = "https://gemma-4-31b.completions.near.ai/v1/chat/completions"
DEFAULT_BRAVE_URL = "https://api.search.brave.com/res/v1/llm/context"
LOCAL_TOKEN = "local-fusion-e2e-token"
FIXTURE_PHRASE = "fusion local e2e ok"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def require_any_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise SystemExit(f"one of {', '.join(names)} is required")


def find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def post_json(url: str, body: dict[str, Any], token: str, timeout: int = 240) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def get_json(url: str, token: str, timeout: int = 60) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_for_proxy(port: int, proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    url = f"http://127.0.0.1:{port}/"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"inference-proxy exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("inference-proxy did not become ready")


class BackendShim(BaseHTTPRequestHandler):
    direct_chat_url: str
    model: str
    token: str

    def log_message(self, _fmt: str, *_args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        payload["model"] = self.model
        req = urllib.request.Request(
            self.direct_chat_url,
            data=json.dumps(payload).encode(),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.token}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                status = response.status
                content_type = response.headers.get("content-type", "application/json")
                body = response.read()
        except urllib.error.HTTPError as err:
            status = err.code
            content_type = err.headers.get("content-type", "application/json")
            body = err.read()

        self.send_response(status)
        self.send_header("content-type", content_type)
        self.end_headers()
        self.wfile.write(body)


class BraveFixture(BaseHTTPRequestHandler):
    def log_message(self, _fmt: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/res/v1/llm/context":
            self.send_response(404)
            self.end_headers()
            return
        body = {
            "grounding": {
                "generic": [
                    {
                        "url": "https://near.ai/fusion-local-e2e",
                        "title": "Fusion Local E2E Fixture",
                        "snippets": [
                            f"{FIXTURE_PHRASE} from local Brave-compatible fixture"
                        ],
                    }
                ]
            },
            "sources": {},
        }
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@contextlib.contextmanager
def run_server(handler: type[BaseHTTPRequestHandler], port: int):
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextlib.contextmanager
def run_proxy(repo_root: Path, env: dict[str, str], port: int):
    binary = repo_root / "target" / "debug" / "vllm-proxy-rs"
    if not binary.exists():
        subprocess.run(["cargo", "build"], cwd=repo_root, check=True)
    proc = subprocess.Popen(
        [str(binary)],
        cwd=repo_root,
        env={**os.environ, **env},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_proxy(port, proc)
        yield proc
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def parse_sse(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


def run_request(proxy_port: int, stream: bool, expect_fixture_phrase: bool) -> tuple[str, dict[str, Any]]:
    body = {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use web search first. If you see the exact phrase "
                    f"`{FIXTURE_PHRASE}`, answer with exactly that phrase. "
                    "Otherwise answer concisely from the web context."
                ),
            }
        ],
        "tools": [
            {
                "type": "nearai:fusion" if stream else "openrouter:fusion",
                "analysis_models": [DEFAULT_MODEL],
                "model": DEFAULT_MODEL,
                "max_tool_calls": 1,
                "max_completion_tokens": 96,
                "temperature": 0,
            },
            {"type": "web_context_search"},
        ],
        "tool_choice": "required",
        "stream": stream,
    }
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    if not stream:
        response = post_json(url, body, LOCAL_TOKEN)
        content = response["choices"][0]["message"]["content"]
        metadata = response["nearai_fusion"]
        chat_id = response["id"]
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {LOCAL_TOKEN}",
            },
        )
        with urllib.request.urlopen(req, timeout=240) as http_response:
            raw = http_response.read().decode()
        if not raw.endswith("data: [DONE]\n\n"):
            raise AssertionError("stream did not end with [DONE]")
        events = parse_sse(raw)
        content = "".join(
            choice.get("delta", {}).get("content", "")
            for event in events
            for choice in event.get("choices", [])
        )
        usage_event = next(event for event in events if event.get("usage"))
        metadata = usage_event["nearai_fusion"]
        chat_id = events[0]["id"]

    if metadata["status"] != "invoked":
        raise AssertionError(f"Fusion was not invoked: {metadata}")
    if metadata["panel"][0]["web_tool_calls"] != 1:
        raise AssertionError(f"expected one web tool call: {metadata['panel'][0]}")
    if expect_fixture_phrase and FIXTURE_PHRASE not in content:
        raise AssertionError(f"fixture phrase missing from final content: {content!r}")

    signature = get_json(
        f"http://127.0.0.1:{proxy_port}/v1/signature/{chat_id}?signing_algo=ecdsa",
        LOCAL_TOKEN,
    )
    if len(signature.get("text", "").split(":")) != 3:
        raise AssertionError("signature text is not model:request_hash:response_hash")
    return content, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-brave", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--direct-chat-url", default=DEFAULT_DIRECT_CHAT_URL)
    args = parser.parse_args()

    if args.model != DEFAULT_MODEL:
        raise SystemExit("custom --model is parsed for future use but not yet wired into prompts")

    token = require_env("FUSION_INTERNAL_BEARER_TOKEN")
    repo_root = Path(__file__).resolve().parents[1]
    backend_port = find_free_port()
    brave_port = find_free_port()
    proxy_port = find_free_port()

    BackendShim.direct_chat_url = args.direct_chat_url
    BackendShim.model = args.model
    BackendShim.token = token

    if args.real_brave:
        brave_url = os.environ.get("WEB_CONTEXT_SEARCH_URL", DEFAULT_BRAVE_URL)
        brave_key = require_any_env(
            ("WEB_CONTEXT_SEARCH_API_KEY", "BRAVE_LLM_CONTEXT_API_KEY")
        )
        brave_context = contextlib.nullcontext()
        expect_fixture_phrase = False
    else:
        brave_url = f"http://127.0.0.1:{brave_port}/res/v1/llm/context"
        brave_key = "local-fixture-key"
        brave_context = run_server(BraveFixture, brave_port)
        expect_fixture_phrase = True

    proxy_env = {
        "MODEL_NAME": args.model,
        "TOKEN": LOCAL_TOKEN,
        "DEV": "true",
        "GPU_NO_HW_MODE": "true",
        "VLLM_BASE_URL": f"http://127.0.0.1:{backend_port}",
        "LISTEN_PORT": str(proxy_port),
        "FUSION_ENABLED": "true",
        "FUSION_INTERNAL_BEARER_TOKEN": token,
        "FUSION_ENDPOINTS_URL": "https://completions.near.ai/endpoints",
        "FUSION_DEFAULT_ANALYSIS_MODELS": args.model,
        "WEB_CONTEXT_SEARCH_URL": brave_url,
        "WEB_CONTEXT_SEARCH_API_KEY": brave_key,
        "VLLM_PROXY_TIMEOUT_SECS": "240",
    }

    with run_server(BackendShim, backend_port):
        with brave_context:
            with run_proxy(repo_root, proxy_env, proxy_port):
                non_stream_content, non_stream_meta = run_request(
                    proxy_port, stream=False, expect_fixture_phrase=expect_fixture_phrase
                )
                stream_content, stream_meta = run_request(
                    proxy_port, stream=True, expect_fixture_phrase=expect_fixture_phrase
                )

    print("fusion_e2e ok")
    print("non_stream_content:", non_stream_content)
    print("stream_content:", stream_content)
    print("non_stream_usage:", non_stream_meta["aggregate_usage"])
    print("stream_usage:", stream_meta["aggregate_usage"])
    print("web_mode:", "real_brave" if args.real_brave else "fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
