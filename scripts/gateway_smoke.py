#!/usr/bin/env python3
"""End-to-end smoke test for a gateway-mode inference-proxy (see docs/gateway-mode.md).

Runs a handful of real requests against a running gateway and checks the
behaviors an API aggregator depends on:

  * models listing and key validation (401 for a bogus key)
  * prior-assistant ``reasoning_content`` survives across tool calls
    (the model must answer with the secret word it "thought" of earlier)
  * unsupported ``video_url`` input is a 400, never a 502
  * streaming works, carries running usage, and a client cancel mid-stream is
    billed only for the tokens observed at cancellation (gateway metrics)
  * conversation affinity keeps sequential turns on one backend
  * stateful/undeclared routes are 404

Reads ``GATEWAY_URL`` (default http://127.0.0.1:31700) and ``NEARAI_API_KEY``
(a customer ``sk-`` key) from the environment. Prints a JSON summary; never
prints keys, and only prints model output where the test is about that output.
Every request uses tiny prompts and small ``max_tokens`` to keep GPU load
negligible.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:31700").rstrip("/")
API_KEY = os.environ.get("NEARAI_API_KEY", "")
MODEL = os.environ.get("GATEWAY_MODEL", "z-ai/glm-5.3-flash")


def request(method: str, path: str, body=None, key: str | None = API_KEY, stream=False, timeout=120,
            request_id: str | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(GATEWAY + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    if request_id:
        req.add_header("X-Request-Id", request_id)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        if stream:
            return resp.status, resp
        return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def metrics() -> dict[str, float]:
    status, body = request("GET", "/metrics", key=None)
    out: dict[str, float] = {}
    if status != 200:
        return out
    for line in body.decode().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        try:
            out[name] = float(value)
        except ValueError:
            pass
    return out


def metric_sum(m: dict[str, float], prefix: str, **labels) -> float:
    total = 0.0
    for name, value in m.items():
        if not name.startswith(prefix):
            continue
        if all(f'{k}="{v}"' in name for k, v in labels.items()):
            total += value
    return total


REASONING_REPRO = {
    "model": MODEL,
    "temperature": 0,
    "max_tokens": 64,
    "tools": [{"type": "function", "function": {"name": "get_time", "description": "Get the current time",
                                                 "parameters": {"type": "object", "properties": {}}}}],
    "messages": [
        {"role": "user", "content": "Call the get_time tool. While thinking, choose a secret word. After the tool "
                                    "returns, reply with exactly \"SECRET: <word>\" using the secret word from your "
                                    "thinking. If you cannot see your earlier thinking, reply \"SECRET: unknown\"."},
        {"role": "assistant", "content": None,
         "reasoning_content": "The secret word is \"xylophone\". After the get_time tool returns, I must reply with "
                              "exactly: SECRET: xylophone",
         "tool_calls": [{"id": "call_get_time_1", "type": "function",
                         "function": {"name": "get_time", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_get_time_1", "content": "12:00"},
    ],
}


def sse_events(resp):
    """Yield (comment_or_data, payload) from an SSE response."""
    buf = b""
    while True:
        chunk = resp.read(1)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            for line in event.decode(errors="replace").splitlines():
                if line.startswith(":"):
                    yield "comment", line
                elif line.startswith("data:"):
                    yield "data", line[5:].strip()


def main() -> int:
    results: dict[str, object] = {"gateway": GATEWAY, "model": MODEL}
    ok = True

    def check(name: str, passed: bool, **detail):
        nonlocal ok
        ok &= passed
        results[name] = {"pass": passed, **detail}

    # 1. Health / models (no key).
    status, body = request("GET", "/healthz", key=None)
    check("healthz", status == 200, status=status, body=json.loads(body or b"{}"))
    status, body = request("GET", "/v1/models", key=None)
    ids = [m.get("id") for m in json.loads(body or b"{}").get("data", [])] if status == 200 else []
    check("models", status == 200 and MODEL in ids, status=status, ids=ids)

    # 2. Auth is enforced by cloud-api.
    status, _ = request("POST", "/v1/chat/completions",
                        {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                        key="sk-0000000000000000000000000000dead")
    check("bogus_key_401", status == 401, status=status)
    if not API_KEY:
        results["error"] = "NEARAI_API_KEY not set; skipping authenticated checks"
        print(json.dumps(results, indent=2))
        return 1

    # 3. Stateless surface.
    status, _ = request("POST", "/v1/responses", {"model": MODEL, "input": "hi"})
    check("responses_api_404", status == 404, status=status)

    # 4. Unsupported video -> 400 before dispatch.
    m0 = metrics()
    video = {"model": MODEL, "max_tokens": 1, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this video."},
        {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}]}]}
    status, body = request("POST", "/v1/chat/completions", video)
    err = json.loads(body or b"{}").get("error", {})
    m1 = metrics()
    check("video_400", status == 400 and "video_url" in err.get("message", ""), status=status, error=err,
          upstream_requests_delta=metric_sum(m1, "upstream_request_duration_seconds_count")
          - metric_sum(m0, "upstream_request_duration_seconds_count"))

    # 5. reasoning_content preserved across tool calls (non-streaming and streaming).
    status, body = request("POST", "/v1/chat/completions", REASONING_REPRO)
    content = ""
    usage = {}
    if status == 200:
        j = json.loads(body)
        content = (j["choices"][0]["message"].get("content") or "").strip()
        usage = j.get("usage", {})
    check("reasoning_content_json", status == 200 and "xylophone" in content.lower(), status=status,
          content=content, usage=usage)

    status, resp = request("POST", "/v1/chat/completions", {**REASONING_REPRO, "stream": True}, stream=True)
    text, usage_events, finish, done, comments = "", 0, None, False, 0
    if status == 200:
        for kind, payload in sse_events(resp):
            if kind == "comment":
                comments += 1
                continue
            if payload == "[DONE]":
                done = True
                break
            j = json.loads(payload)
            if j.get("usage"):
                usage_events += 1
                usage = j["usage"]
            for c in j.get("choices", []):
                text += c.get("delta", {}).get("content") or ""
                finish = c.get("finish_reason") or finish
    check("reasoning_content_stream", status == 200 and done and "xylophone" in text.lower(), status=status,
          content=text.strip(), finish_reason=finish, running_usage_events=usage_events, final_usage=usage,
          keepalive_comments=comments)

    # 6. Cancel mid-stream: only the tokens observed so far are reported. The
    # model may spend its budget in hidden reasoning, so count reasoning deltas
    # as progress too and cancel early. The request id is printed so the
    # gateway journal ("Reported usage for interrupted stream") and the CVM
    # proxy logs ("Client disconnected") can be correlated.
    time.sleep(2.0)  # let earlier usage reports settle before taking the baseline
    m0 = metrics()
    cancel_request_id = str(uuid.uuid4())  # the proxy only keeps inbound ids that are UUIDs
    long_req = {"model": MODEL, "stream": True, "max_tokens": 600, "temperature": 0,
                "messages": [{"role": "user", "content": "Count from 1 to 400, one number per line."}]}
    status, resp = request("POST", "/v1/chat/completions", long_req, stream=True, request_id=cancel_request_id)
    seen_deltas, last_usage, done = 0, {}, False
    t0 = time.time()
    if status == 200:
        for kind, payload in sse_events(resp):
            if kind != "data":
                continue
            if payload == "[DONE]":
                done = True
                break
            j = json.loads(payload)
            if j.get("usage"):
                last_usage = j["usage"]
            for c in j.get("choices", []):
                d = c.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    seen_deltas += 1
            if seen_deltas >= 25:
                break
        resp.close()  # client cancel
    cancel_at = time.time() - t0
    time.sleep(3.0)  # let the usage report land
    m1 = metrics()
    reports = metric_sum(m1, "inference_proxy_usage_reports_total", outcome="accepted") - \
        metric_sum(m0, "inference_proxy_usage_reports_total", outcome="accepted")
    disconnects = metric_sum(m1, "stream_client_disconnects_total") - metric_sum(m0, "stream_client_disconnects_total")
    check("cancel_mid_stream",
          status == 200 and not done and disconnects >= 1 and reports >= 1
          and 0 < last_usage.get("completion_tokens", 0) < 600,
          status=status, request_id=cancel_request_id, stream_finished_before_cancel=done,
          client_saw_deltas=seen_deltas, cancelled_after_s=round(cancel_at, 2),
          usage_seen_by_client_at_cancel=last_usage, usage_reports_accepted_delta=reports,
          client_disconnects_delta=disconnects)

    # 7. Affinity: three sequential turns of one conversation pin to one backend.
    m0 = metrics()
    convo = [{"role": "user", "content": "Reply with the single word OK."}]
    pinned_ok = True
    for turn in range(3):
        status, body = request("POST", "/v1/chat/completions",
                               {"model": MODEL, "max_tokens": 4, "temperature": 0, "messages": convo})
        if status != 200:
            pinned_ok = False
            break
        answer = json.loads(body)["choices"][0]["message"].get("content") or "OK"
        convo += [{"role": "assistant", "content": answer}, {"role": "user", "content": f"Again, turn {turn + 2}."}]
    m1 = metrics()
    pinned = metric_sum(m1, "backend_affinity_selections_total", outcome="pinned") - \
        metric_sum(m0, "backend_affinity_selections_total", outcome="pinned")
    new = metric_sum(m1, "backend_affinity_selections_total", outcome="new") - \
        metric_sum(m0, "backend_affinity_selections_total", outcome="new")
    check("conversation_affinity", pinned_ok and new >= 1 and pinned >= 2, pinned_delta=pinned, new_delta=new,
          pool_size=m1.get("backend_pool_size"), pool_healthy=m1.get("backend_pool_healthy"))

    results["all_pass"] = ok
    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
