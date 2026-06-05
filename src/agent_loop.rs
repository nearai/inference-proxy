//! Server-side agent loop for `/v1/chat/completions`.
//!
//! Opt-in via a single namespaced tool: `{"type":"web_context_search"}`. When
//! that exact tool is present (no other tools, streaming on, Brave creds
//! configured), this module drives a tool-call loop entirely inside the CVM:
//! the model emits a tool call, we call the Brave LLM Context API, splice the
//! result back into the conversation as a synthetic SSE chunk, and re-issue
//! the upstream request until the model stops asking for tools.
//!
//! Anything that doesn't match the trigger falls through to the existing
//! pass-through path in `routes::chat`. Nothing about non-agent-loop requests
//! changes.
//!
//! Privacy: the only thing that egresses the CVM is the search query, going
//! directly to Brave under TLS. Tool args and results live in process memory
//! for the lifetime of the request and are dropped on completion. No state is
//! kept in `AppState`.

use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::Response;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};

use crate::auth::RequireAuth;
use crate::encryption::ChunkTransform;
use crate::error::AppError;
use crate::proxy::{make_usage_reporter, normalize_chat_chunk, spawn_usage_report, StreamingGuard};
use crate::{AppState, TracingIds};

pub const WEB_CONTEXT_SEARCH_TOOL_NAME: &str = "web_context_search";

const NEARAI_TOOL_RESULT_KEY: &str = "nearai_tool_result";
const NEARAI_LOOP_TERMINATED_KEY: &str = "nearai_loop_terminated";

/// Hard cap on bytes read from Brave's response body. The defaults we send
/// (`maximum_number_of_tokens=8192`, `maximum_number_of_urls=20`) should
/// produce well under this; the cap is a backstop against a misconfigured
/// or malicious search endpoint that returns an unbounded body.
const BRAVE_MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

/// Hard cap on the formatted tool output that we emit downstream and feed
/// back to the model on the next iteration. Independent of Brave's input
/// caps so we don't depend on the upstream respecting them. Beyond this,
/// the output is truncated with a marker.
const MAX_FORMATTED_OUTPUT_BYTES: usize = 32 * 1024;

/// True iff the request's `tools` field is exactly one entry of type
/// `web_context_search`. Mixed tool types or multiple entries return false
/// and let the request flow through the existing pass-through path.
pub fn is_web_context_search_request(request_json: &Value) -> bool {
    let Some(tools) = request_json.get("tools").and_then(|v| v.as_array()) else {
        return false;
    };
    tools.len() == 1
        && tools[0].get("type").and_then(|v| v.as_str()) == Some(WEB_CONTEXT_SEARCH_TOOL_NAME)
}

/// Drive an agent loop for a `/v1/chat/completions` request.
///
/// Preconditions (checked by the caller in `routes::chat`):
/// - `request_json` has been decrypted in place (if E2EE is active)
/// - `request_json.tools == [{"type":"web_context_search"}]`
/// - `request_json.stream == true`
/// - `state.config.web_context_search_url` and `..._api_key` are set
///
/// `request_hash` is the SHA-256 of the original wire body, used unchanged in
/// the signed text — signature semantics match the non-loop path.
pub async fn run_chat_completion(
    state: AppState,
    auth: RequireAuth,
    tracing_ids: TracingIds,
    request_hash: String,
    mut request_json: Value,
    chunk_transform: Option<ChunkTransform>,
) -> Result<Response, AppError> {
    // Rewrite our namespaced tool into a standard OpenAI `function` so the
    // upstream engine (vLLM/SGLang) emits tool_calls in its usual shape.
    rewrite_tool_for_upstream(&mut request_json);

    // Force stream + usage so we can splice tool results between iterations
    // and report aggregated token usage at the end.
    request_json["stream"] = true.into();
    let mut stream_opts = request_json
        .get("stream_options")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    stream_opts.insert("include_usage".into(), true.into());
    request_json["stream_options"] = Value::Object(stream_opts);

    // Pick one backend; reuse it across every iteration so the engine can
    // prefix-cache across rounds (each iteration's prompt = prior iteration +
    // a few new messages, so the cache hit rate should be very high).
    let (upstream_url, backend_guard) = state.backend_pool.select_url("/v1/chat/completions");

    // Send the first upstream request synchronously so its HTTP status
    // propagates to the caller. Without this, a 400/429/503 from upstream
    // on iteration 0 would arrive as a 200 text/event-stream followed by a
    // broken / empty body, hiding the real failure from the client. Only
    // when this first call succeeds do we return 200 + spawn the loop.
    let first_request_body =
        serde_json::to_vec(&request_json).map_err(|e| AppError::Internal(e.into()))?;
    let first_response = send_upstream(
        &state.http_client,
        &upstream_url,
        first_request_body,
        &tracing_ids,
    )
    .await?;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    let http_client = state.http_client.clone();
    let signing = state.signing.clone();
    let cache = state.cache.clone();
    let model_name = state.config.model_name.clone();
    let max_iterations = state.config.agent_loop_max_iterations;
    let brave_url = state
        .config
        .web_context_search_url
        .clone()
        .expect("caller verified web_context_search_url is set");
    let brave_key = state
        .config
        .web_context_search_api_key
        .clone()
        .expect("caller verified web_context_search_api_key is set");
    let tool_timeout = Duration::from_secs(state.config.web_context_search_timeout_secs);
    let usage_reporter = make_usage_reporter(&auth, &state);

    tokio::spawn(async move {
        // Keep the streaming_connections gauge accurate for the full loop
        // duration, the same way `proxy_streaming_request` does.
        let _streaming_guard = StreamingGuard::new();
        let _backend_guard = backend_guard;
        let outcome = drive_loop(
            LoopCtx {
                client: &http_client,
                upstream_url: &upstream_url,
                request_json: &mut request_json,
                tx: &tx,
                chunk_transform: chunk_transform.as_ref(),
                max_iterations,
                brave_url: &brave_url,
                brave_key: &brave_key,
                tool_timeout,
                tracing_ids: &tracing_ids,
            },
            first_response,
        )
        .await;

        match outcome {
            Ok(result) => {
                info!(
                    request_id = %tracing_ids.request_id,
                    org_id = %tracing_ids.org_id_or_empty(),
                    workspace_id = %tracing_ids.workspace_id_or_empty(),
                    model = %model_name,
                    chat_id = %result.chat_id.as_deref().unwrap_or(""),
                    iterations = result.iterations,
                    terminated_by = %result.terminated_by,
                    input_tokens = result.input_tokens,
                    output_tokens = result.output_tokens,
                    completed_cleanly = result.completed_cleanly,
                    "agent loop completed"
                );
                // Bill for the tokens already produced, even when the loop did
                // NOT finish cleanly (client disconnect or mid-loop error). The
                // loop accumulates usage across iterations, so result.{input,output}_tokens
                // hold the partial total at the point of interruption. This must not
                // be gated on a clean [DONE] (nearai/infra#98) — mirrors the same fix
                // in `proxy_streaming_request`. The reporter only exists for direct
                // sk- requests (RequireAuth.cloud_api_key); cloud-api's own
                // InterceptStream is not in that path, so this is the sole biller —
                // no double-billing. Reported at most once per loop (one chat id).
                if let (Some(reporter), Some(chat_id)) =
                    (usage_reporter.as_ref(), result.chat_id.as_deref())
                {
                    if result.input_tokens > 0 || result.output_tokens > 0 {
                        let body = json!({
                            "type": "chat_completion",
                            "model": reporter.model_name,
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                            "id": chat_id,
                        });
                        spawn_usage_report(reporter, body);
                        if !result.completed_cleanly {
                            info!(
                                request_id = %tracing_ids.request_id,
                                org_id = %tracing_ids.org_id_or_empty(),
                                workspace_id = %tracing_ids.workspace_id_or_empty(),
                                chat_id = %chat_id,
                                input_tokens = result.input_tokens,
                                output_tokens = result.output_tokens,
                                "Reported usage for interrupted agent loop"
                            );
                        }
                    }
                }

                // Only sign / cache when the stream closed cleanly (final `[DONE]`
                // was sent downstream): a partial response cannot be verified.
                if !result.completed_cleanly {
                    return;
                }
                let Some(chat_id) = result.chat_id.as_deref() else {
                    warn!("Agent loop completed without observing a chat id; skipping signature");
                    return;
                };
                let response_sha256 = hex::encode(result.hasher.finalize());
                let text = format!("{model_name}:{request_hash}:{response_sha256}");
                match signing.sign_chat(&text) {
                    Ok(signed) => {
                        if let Ok(signed_json) = serde_json::to_string(&signed) {
                            cache.set_chat(chat_id, &signed_json);
                        }
                    }
                    Err(e) => error!(error = %e, "Signing failed for agent loop response"),
                }
            }
            Err(e) => {
                error!(error = %e, "Agent loop failed");
                let _ = tx
                    .send(Err(std::io::Error::other(format!("agent loop: {e}"))))
                    .await;
            }
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream);
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(body)
        .expect("response builder"))
}

/// Replace `{"type":"web_context_search"}` with a standard function tool so
/// the upstream engine knows how to advertise it to the model. The function
/// surface is intentionally narrow: only `query`. Brave-specific tuning
/// (count, max_urls, threshold, …) is fixed on our side; clients don't get
/// to influence it via the tool schema, which keeps the prompt surface small
/// and the result shape predictable.
///
/// Also forces `parallel_tool_calls: false`. Phase 1 caps execution at one
/// tool call per iteration; instructing the model not to emit more than one
/// keeps the upstream from producing tool calls we'd then drop.
fn rewrite_tool_for_upstream(request_json: &mut Value) {
    request_json["tools"] = json!([{
        "type": "function",
        "function": {
            "name": WEB_CONTEXT_SEARCH_TOOL_NAME,
            "description": "Search the web for source-grounded context to answer the user's question. Use this when up-to-date or citable information is needed. The query may include Brave search operators (quoted phrases, `site:`, `filetype:`, `-` to exclude).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string."
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }
        }
    }]);
    // Phase 1: serialize tool calls — at most one per iteration.
    request_json["parallel_tool_calls"] = false.into();
}

struct LoopCtx<'a> {
    client: &'a reqwest::Client,
    upstream_url: &'a str,
    request_json: &'a mut Value,
    tx: &'a tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
    chunk_transform: Option<&'a ChunkTransform>,
    max_iterations: u32,
    brave_url: &'a str,
    brave_key: &'a str,
    tool_timeout: Duration,
    tracing_ids: &'a TracingIds,
}

struct LoopResult {
    chat_id: Option<String>,
    hasher: Sha256,
    iterations: u32,
    input_tokens: u64,
    output_tokens: u64,
    terminated_by: &'static str,
    /// True only if the loop closed cleanly (model stop / max_iterations
    /// terminator) and the final `[DONE]` was sent downstream. False on
    /// client disconnect or upstream stream ending without a `[DONE]`.
    completed_cleanly: bool,
}

async fn drive_loop(
    ctx: LoopCtx<'_>,
    first_response: reqwest::Response,
) -> Result<LoopResult, AppError> {
    let mut hasher = Sha256::new();
    let mut chat_id: Option<String> = None;
    let mut model_echo: Option<String> = None;
    let mut created: Option<i64> = None;
    let mut total_input_tokens: u64 = 0;
    let mut total_output_tokens: u64 = 0;
    let mut iterations: u32 = 0;
    let terminated_by;
    let mut completed_cleanly = false;
    // The iter-0 upstream Response is already in hand (see
    // `run_chat_completion`); subsequent iterations build their own via
    // `send_upstream`.
    let mut next_response: Option<reqwest::Response> = Some(first_response);

    loop {
        // Bail out early if the client has gone away — avoids issuing another
        // upstream LLM request or a Brave call we'd only throw away on send.
        if ctx.tx.is_closed() {
            terminated_by = "client_disconnect";
            break;
        }

        if iterations >= ctx.max_iterations {
            emit_max_iterations_terminator(
                ctx.tx,
                ctx.chunk_transform,
                &mut hasher,
                chat_id.as_deref(),
                model_echo.as_deref(),
                created,
            )
            .await?;
            forward_done(ctx.tx, &mut hasher).await?;
            terminated_by = "max_iterations";
            completed_cleanly = true;
            break;
        }

        // Acquire the upstream Response for this iteration. Iter 0 reuses
        // the synchronous first call from `run_chat_completion`; iter 1+
        // sends now, racing the send against client disconnect.
        let response = match next_response.take() {
            Some(r) => r,
            None => {
                let body = serde_json::to_vec(ctx.request_json)
                    .map_err(|e| AppError::Internal(e.into()))?;
                let send_outcome = tokio::select! {
                    r = send_upstream(ctx.client, ctx.upstream_url, body, ctx.tracing_ids) => Some(r),
                    _ = ctx.tx.closed() => None,
                };
                let Some(send_outcome) = send_outcome else {
                    terminated_by = "client_disconnect";
                    break;
                };
                match send_outcome {
                    Ok(r) => r,
                    Err(AppError::Upstream { status, body }) => {
                        warn!(
                            status = %status,
                            "upstream returned non-2xx mid-loop"
                        );
                        // Surface the failure to the client as an SSE error
                        // chunk so they don't see a silent stall. No `[DONE]`
                        // and no signature — this is not a successful
                        // completion. The `body` from upstream is dropped
                        // (not forwarded) because it can contain
                        // provider-side internals or user data.
                        let _ = body;
                        let status_code = status.as_u16();
                        let code_str = status_code.to_string();
                        emit_synthetic_error_chunk(
                            ctx.tx,
                            ctx.chunk_transform,
                            &mut hasher,
                            chat_id.as_deref(),
                            model_echo.as_deref(),
                            created,
                            &format!("upstream returned HTTP {status_code} on a follow-up tool-loop iteration"),
                            "upstream_error",
                            Some(&code_str),
                        )
                        .await?;
                        terminated_by = "upstream_error";
                        break;
                    }
                    Err(other) => return Err(other),
                }
            }
        };

        iterations += 1;
        let iter_started = Instant::now();

        let iter_outcome = run_iteration(
            IterCtx {
                tx: ctx.tx,
                hasher: &mut hasher,
                chunk_transform: ctx.chunk_transform,
                rewrite_id_to: if iterations > 1 {
                    chat_id.as_deref()
                } else {
                    None
                },
            },
            response,
        )
        .await?;

        if chat_id.is_none() {
            chat_id = iter_outcome.chat_id.clone();
        }
        if model_echo.is_none() {
            model_echo = iter_outcome.model.clone();
        }
        if created.is_none() {
            created = iter_outcome.created;
        }
        total_input_tokens = total_input_tokens.saturating_add(iter_outcome.input_tokens);
        total_output_tokens = total_output_tokens.saturating_add(iter_outcome.output_tokens);

        debug!(
            iteration = iterations,
            finish_reason = iter_outcome.finish_reason.as_deref().unwrap_or(""),
            tool_calls = iter_outcome.tool_calls.len(),
            saw_done = iter_outcome.saw_done,
            upstream_ms = iter_started.elapsed().as_millis() as u64,
            "agent loop iteration"
        );

        // Client disconnected mid-iteration.
        if iter_outcome.client_disconnected {
            terminated_by = "client_disconnect";
            break;
        }

        // Upstream surfaced an inline error chunk (e.g. SGLang abort).
        // The error chunk itself is already on the wire to the client;
        // we just refuse to forward `[DONE]` or sign over an aborted
        // generation.
        if iter_outcome.upstream_error.is_some() {
            terminated_by = "upstream_error";
            break;
        }

        // Phase 1: cap at exactly one tool call per iteration. We force
        // `parallel_tool_calls: false` upstream; if a model still emits
        // multiple (some engines ignore the flag), fall through to the
        // non-loop path so we never silently drop a tool call.
        let is_tool_call_iteration = iter_outcome.saw_done
            && iter_outcome.finish_reason.as_deref() == Some("tool_calls")
            && iter_outcome.tool_calls.len() == 1
            && all_calls_are_web_context_search(&iter_outcome.tool_calls);

        if !is_tool_call_iteration {
            // If the upstream stream didn't carry a `[DONE]` (transport
            // failure, abrupt EOF, mid-stream error chunk), do NOT synthesize
            // one — that would mislead the client and produce a cached
            // signature over an incomplete response. Drop the stream and let
            // the spawn task surface an error.
            if !iter_outcome.saw_done {
                terminated_by = "upstream_eof";
                break;
            }
            forward_done(ctx.tx, &mut hasher).await?;
            terminated_by = match iter_outcome.finish_reason.as_deref() {
                Some(r) => match r {
                    "stop" => "model_stop",
                    "length" => "model_length",
                    "tool_calls" => "client_side_tool",
                    other => str_to_static(other),
                },
                None => "upstream_eof",
            };
            completed_cleanly = true;
            break;
        }

        // Execute the tool call (Phase 1: exactly one). Emit one synthetic
        // chunk with the result, then append the assistant + tool messages
        // to the conversation for the next iteration.
        let mut tool_messages: Vec<Value> = Vec::with_capacity(iter_outcome.tool_calls.len());
        let mut disconnected_in_tool = false;
        for tc in &iter_outcome.tool_calls {
            let tool_call_id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args_str = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let query = parse_query(args_str);

            let (status, output) = match query {
                Some(q) if !q.is_empty() => {
                    let started = Instant::now();
                    // Race the Brave call against client disconnect: if the
                    // client goes away while we're waiting on Brave, abort
                    // promptly rather than holding the request open until
                    // the search finishes.
                    let brave_result = tokio::select! {
                        r = brave_llm_context_search(
                            ctx.client,
                            ctx.brave_url,
                            ctx.brave_key,
                            &q,
                            ctx.tool_timeout,
                            ctx.tracing_ids,
                        ) => Some(r),
                        _ = ctx.tx.closed() => None,
                    };
                    let Some(brave_result) = brave_result else {
                        info!(
                            tool = WEB_CONTEXT_SEARCH_TOOL_NAME,
                            tool_call_id = %tool_call_id,
                            "client disconnected during tool execution"
                        );
                        disconnected_in_tool = true;
                        break;
                    };
                    match brave_result {
                        Ok(text) => {
                            info!(
                                tool = WEB_CONTEXT_SEARCH_TOOL_NAME,
                                tool_call_id = %tool_call_id,
                                elapsed_ms = started.elapsed().as_millis() as u64,
                                output_chars = text.len(),
                                "tool ok"
                            );
                            ("ok", text)
                        }
                        Err(BraveError::Timeout) => {
                            warn!(
                                tool = WEB_CONTEXT_SEARCH_TOOL_NAME,
                                tool_call_id = %tool_call_id,
                                "tool timeout"
                            );
                            (
                                "timeout",
                                "Web search timed out. Please answer using your existing knowledge or ask the user for more detail.".to_string(),
                            )
                        }
                        Err(BraveError::Other(msg)) => {
                            warn!(
                                tool = WEB_CONTEXT_SEARCH_TOOL_NAME,
                                tool_call_id = %tool_call_id,
                                error = %msg,
                                "tool error"
                            );
                            (
                                "error",
                                "Web search failed. Please answer using your existing knowledge or ask the user for more detail.".to_string(),
                            )
                        }
                    }
                }
                _ => {
                    warn!(
                        tool = WEB_CONTEXT_SEARCH_TOOL_NAME,
                        tool_call_id = %tool_call_id,
                        "tool arguments missing or invalid; skipping search"
                    );
                    (
                        "error",
                        "Tool arguments were missing or invalid; no search performed.".to_string(),
                    )
                }
            };

            emit_tool_result_chunk(
                ctx.tx,
                ctx.chunk_transform,
                &mut hasher,
                chat_id.as_deref(),
                model_echo.as_deref(),
                created,
                &tool_call_id,
                status,
                &output,
            )
            .await?;

            tool_messages.push(json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": output,
            }));
        }

        if disconnected_in_tool {
            terminated_by = "client_disconnect";
            break;
        }

        // Append the assistant tool_calls message + tool result messages
        // to the conversation for the next iteration.
        let messages = ctx
            .request_json
            .get_mut("messages")
            .and_then(|m| m.as_array_mut())
            .ok_or_else(|| AppError::Internal(anyhow::anyhow!("messages array missing")))?;
        messages.push(json!({
            "role": "assistant",
            "tool_calls": iter_outcome.tool_calls,
        }));
        for m in tool_messages {
            messages.push(m);
        }
    }

    Ok(LoopResult {
        chat_id,
        hasher,
        iterations,
        input_tokens: total_input_tokens,
        output_tokens: total_output_tokens,
        terminated_by,
        completed_cleanly,
    })
}

/// Bridge to give us `&'static str` for less-common finish_reason values
/// without leaking the original. We only care about a few well-known
/// reasons; anything else collapses to a single bucket for metrics.
fn str_to_static(reason: &str) -> &'static str {
    match reason {
        "content_filter" => "content_filter",
        _ => "other",
    }
}

fn all_calls_are_web_context_search(tool_calls: &[Value]) -> bool {
    !tool_calls.is_empty()
        && tool_calls.iter().all(|tc| {
            tc.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                == Some(WEB_CONTEXT_SEARCH_TOOL_NAME)
        })
}

fn parse_query(arguments: &str) -> Option<String> {
    let parsed: Value = serde_json::from_str(arguments).ok()?;
    parsed
        .get("query")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
}

// ── single iteration ────────────────────────────────────────────────

struct IterCtx<'a> {
    tx: &'a tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
    hasher: &'a mut Sha256,
    chunk_transform: Option<&'a ChunkTransform>,
    /// When set, rewrite each forwarded chunk's `id` to this value so the
    /// client sees one logical completion across loop iterations.
    rewrite_id_to: Option<&'a str>,
}

struct IterOutcome {
    chat_id: Option<String>,
    model: Option<String>,
    created: Option<i64>,
    finish_reason: Option<String>,
    tool_calls: Vec<Value>,
    input_tokens: u64,
    output_tokens: u64,
    /// True iff the upstream stream terminated with a `data: [DONE]` line.
    /// Drive_loop only forwards a downstream `[DONE]` and signs the response
    /// when this is true; an abrupt EOF or a transport error must NOT be
    /// presented to the client as a clean completion.
    saw_done: bool,
    /// True iff the downstream channel closed mid-iteration (client went
    /// away). Drive_loop short-circuits and skips signing in that case.
    client_disconnected: bool,
    /// First top-level `{"error": {...}}` SSE chunk seen mid-stream. SGLang
    /// surfaces aborts (queue overflow, priority disabled, waiting timeout)
    /// this way while keeping the response otherwise well-formed; treating
    /// these as upstream failure prevents signing a "successful" response
    /// over an aborted generation.
    upstream_error: Option<Value>,
}

async fn run_iteration(
    ctx: IterCtx<'_>,
    response: reqwest::Response,
) -> Result<IterOutcome, AppError> {
    // The HTTP status was already checked by `send_upstream` before we
    // reached this iteration. Here we only consume the response body.
    let mut byte_stream = std::pin::pin!(response.bytes_stream());
    // Raw byte buffer — `from_utf8_lossy` on per-chunk slices would corrupt
    // multi-byte UTF-8 characters split across chunk boundaries (replacing
    // partial bytes with U+FFFD permanently). We hold raw bytes until a full
    // `\n`-terminated line is in hand and only then decode.
    let mut byte_buf: Vec<u8> = Vec::with_capacity(8 * 1024);
    let mut outcome = IterOutcome {
        chat_id: None,
        model: None,
        created: None,
        finish_reason: None,
        tool_calls: Vec::new(),
        input_tokens: 0,
        output_tokens: 0,
        saw_done: false,
        client_disconnected: false,
        upstream_error: None,
    };

    // SSE line loop. We hold off forwarding `[DONE]` until the caller decides
    // whether to continue looping; everything else is forwarded as it arrives.
    // `select!` on `tx.closed()` lets us abort promptly when the client drops
    // mid-stream, matching `proxy_streaming_request`.
    'outer: loop {
        tokio::select! {
            chunk = byte_stream.next() => {
                match chunk {
                    Some(Ok(chunk)) => {
                        byte_buf.extend_from_slice(&chunk);

                        while let Some(nl) = byte_buf.iter().position(|b| *b == b'\n') {
                            let line_bytes: Vec<u8> = byte_buf.drain(..=nl).collect();
                            // Skip if the line isn't valid UTF-8 — SSE lines
                            // are required to be UTF-8 by spec; we don't try
                            // to recover from malformed upstream data.
                            let Ok(line) = std::str::from_utf8(&line_bytes) else {
                                warn!("agent loop: skipping non-UTF-8 SSE line");
                                continue;
                            };
                            let trimmed = line.trim_end_matches(['\n', '\r']);
                            let data = trimmed
                                .strip_prefix("data: ")
                                .or_else(|| trimmed.strip_prefix("data:"));

                            if let Some(data) = data {
                                let data = data.trim();
                                if data == "[DONE]" {
                                    outcome.saw_done = true;
                                    break 'outer;
                                }
                                if data.is_empty() {
                                    continue;
                                }
                                let mut parsed: Value = match serde_json::from_str(data) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        warn!(error = %e, "agent loop: failed to parse upstream SSE line");
                                        continue;
                                    }
                                };
                                ingest_chunk_metadata(&parsed, &mut outcome);

                                // SGLang and friends emit top-level
                                // `{"error": {...}}` chunks on aborts.
                                // Do NOT forward the upstream chunk
                                // verbatim — `error.message` is outside
                                // what the chunk transform encrypts, and
                                // backends may put validation
                                // input/request details (which under E2EE
                                // is data we decrypted inside the CVM)
                                // into it. Replace with a sanitized
                                // synthetic error chunk and abort the
                                // iteration.
                                if outcome.upstream_error.is_some() {
                                    emit_synthetic_error_chunk(
                                        ctx.tx,
                                        ctx.chunk_transform,
                                        ctx.hasher,
                                        outcome.chat_id.as_deref(),
                                        outcome.model.as_deref(),
                                        outcome.created,
                                        "upstream emitted an error chunk; response was aborted",
                                        "upstream_error",
                                        None,
                                    )
                                    .await?;
                                    break 'outer;
                                }

                                if let Some(new_id) = ctx.rewrite_id_to {
                                    if parsed.get("id").is_some() {
                                        parsed["id"] = Value::String(new_id.to_string());
                                    }
                                }

                                // Match the non-loop path's behavior:
                                // upstream reasoning parsers that emit
                                // `delta.reasoning` get normalized to
                                // `delta.reasoning_content` so clients see
                                // a consistent field name across both paths.
                                normalize_chat_chunk(&mut parsed);

                                if let Some(transform) = ctx.chunk_transform {
                                    transform(&mut parsed)?;
                                }

                                let serialized = serde_json::to_string(&parsed)
                                    .map_err(|e| AppError::Internal(e.into()))?;
                                let mut emit = String::with_capacity(serialized.len() + 8);
                                emit.push_str("data: ");
                                emit.push_str(&serialized);
                                emit.push_str("\n\n");
                                let bytes = emit.into_bytes();
                                ctx.hasher.update(&bytes);
                                if ctx.tx.send(Ok(Bytes::from(bytes))).await.is_err() {
                                    outcome.client_disconnected = true;
                                    break 'outer;
                                }
                            } else if !trimmed.is_empty() {
                                // Non-data line (comment, event:, id:, retry:). Forward verbatim.
                                let bytes = line_bytes.clone();
                                ctx.hasher.update(&bytes);
                                if ctx.tx.send(Ok(Bytes::from(bytes))).await.is_err() {
                                    outcome.client_disconnected = true;
                                    break 'outer;
                                }
                            }
                            // blank line: drop (we re-emit `\n\n` after each data line)
                        }
                    }
                    Some(Err(e)) => return Err(AppError::Internal(e.into())),
                    None => break, // upstream closed before [DONE]
                }
            }
            _ = ctx.tx.closed() => {
                // Client disconnected while we were waiting for upstream
                // bytes — abort promptly so we don't keep GPU work going
                // and don't fire another Brave call.
                outcome.client_disconnected = true;
                break 'outer;
            }
        }
    }

    Ok(outcome)
}

fn ingest_chunk_metadata(event: &Value, outcome: &mut IterOutcome) {
    // SGLang and friends surface mid-stream aborts as a top-level
    // `{"error": {...}}` chunk while otherwise emitting a well-formed SSE
    // (including `[DONE]`). Capturing this lets the loop refuse to sign or
    // forward `[DONE]` over an aborted generation.
    if outcome.upstream_error.is_none() {
        if let Some(err) = event.get("error").filter(|v| v.is_object()) {
            outcome.upstream_error = Some(err.clone());
        }
    }

    if outcome.chat_id.is_none() {
        outcome.chat_id = event.get("id").and_then(|v| v.as_str()).map(String::from);
    }
    if outcome.model.is_none() {
        outcome.model = event
            .get("model")
            .and_then(|v| v.as_str())
            .map(String::from);
    }
    if outcome.created.is_none() {
        outcome.created = event.get("created").and_then(|v| v.as_i64());
    }

    if let Some(usage) = event.get("usage").filter(|v| v.is_object()) {
        if let Some(p) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
            outcome.input_tokens = outcome.input_tokens.saturating_add(p);
        }
        if let Some(c) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
            outcome.output_tokens = outcome.output_tokens.saturating_add(c);
        }
    }

    if let Some(choices) = event.get("choices").and_then(|v| v.as_array()) {
        for choice in choices {
            if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                outcome.finish_reason = Some(fr.to_string());
            }
            if let Some(delta) = choice.get("delta").filter(|v| v.is_object()) {
                if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                    merge_tool_call_deltas(&mut outcome.tool_calls, tcs);
                }
            }
        }
    }
}

/// Merge incremental tool_call deltas into the cumulative list, indexed by
/// the upstream `index` field. Mirrors `ChoiceAssembler::merge_tool_calls`
/// in `proxy.rs`; duplicated here so this module stays self-contained.
fn merge_tool_call_deltas(acc: &mut Vec<Value>, deltas: &[Value]) {
    for delta in deltas {
        let idx = delta.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        while acc.len() <= idx {
            acc.push(json!({
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""}
            }));
        }
        let existing = &mut acc[idx];
        if let Some(id) = delta.get("id").and_then(|v| v.as_str()) {
            existing["id"] = id.into();
        }
        if let Some(t) = delta.get("type").and_then(|v| v.as_str()) {
            existing["type"] = t.into();
        }
        if let Some(func) = delta.get("function").filter(|v| v.is_object()) {
            if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                if !name.is_empty() {
                    existing["function"]["name"] = name.into();
                }
            }
            if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                let prev = existing["function"]["arguments"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                existing["function"]["arguments"] = Value::String(prev + args);
            }
        }
    }
}

// ── upstream send ───────────────────────────────────────────────────

/// Issue one upstream chat-completions request and return the streaming
/// response. Maps non-2xx into `AppError::Upstream` so the caller can decide
/// whether to surface the failure as an HTTP error (iter 0, before
/// returning to the client) or as an inline SSE error chunk (iter 1+).
async fn send_upstream(
    client: &reqwest::Client,
    upstream_url: &str,
    body: Vec<u8>,
    tracing_ids: &TracingIds,
) -> Result<reqwest::Response, AppError> {
    let mut req = client
        .post(upstream_url)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream");
    for (k, v) in tracing_ids.upstream_headers() {
        req = req.header(k, v);
    }
    let response = req
        .body(body)
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_default();
        return Err(AppError::Upstream {
            status: StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            body,
        });
    }
    Ok(response)
}

// ── synthetic chunks ────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn emit_tool_result_chunk(
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
    chunk_transform: Option<&ChunkTransform>,
    hasher: &mut Sha256,
    chat_id: Option<&str>,
    model: Option<&str>,
    created: Option<i64>,
    tool_call_id: &str,
    status: &str,
    output: &str,
) -> Result<(), AppError> {
    let mut chunk = json!({
        "id": chat_id.unwrap_or(""),
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                NEARAI_TOOL_RESULT_KEY: {
                    "tool_call_id": tool_call_id,
                    "name": WEB_CONTEXT_SEARCH_TOOL_NAME,
                    "status": status,
                    "output": output,
                }
            }
        }]
    });
    if let Some(m) = model {
        chunk["model"] = m.into();
    }
    if let Some(c) = created {
        chunk["created"] = c.into();
    }
    if let Some(transform) = chunk_transform {
        transform(&mut chunk)?;
    }
    let serialized = serde_json::to_string(&chunk).map_err(|e| AppError::Internal(e.into()))?;
    let bytes = format!("data: {serialized}\n\n").into_bytes();
    hasher.update(&bytes);
    tx.send(Ok(Bytes::from(bytes)))
        .await
        .map_err(|_| AppError::Internal(anyhow::anyhow!("client disconnected")))?;
    Ok(())
}

/// Emit a sanitized OpenAI-shaped error chunk to the client when an
/// upstream failure happens after `200 text/event-stream` headers have
/// already been sent. The message text is controlled by us — we don't
/// pass through upstream-provided strings, which under E2EE could
/// contain prompt fragments or other user data the upstream backend
/// echoed back. Closing without `[DONE]` keeps the response unsigned.
#[allow(clippy::too_many_arguments)]
async fn emit_synthetic_error_chunk(
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
    chunk_transform: Option<&ChunkTransform>,
    hasher: &mut Sha256,
    chat_id: Option<&str>,
    model: Option<&str>,
    created: Option<i64>,
    message: &str,
    error_type: &str,
    code: Option<&str>,
) -> Result<(), AppError> {
    let mut error = json!({
        "message": message,
        "type": error_type,
    });
    if let Some(c) = code {
        error["code"] = c.into();
    }
    let mut chunk = json!({
        "id": chat_id.unwrap_or(""),
        "object": "chat.completion.chunk",
        "choices": [],
        "error": error,
    });
    if let Some(m) = model {
        chunk["model"] = m.into();
    }
    if let Some(c) = created {
        chunk["created"] = c.into();
    }
    // Pass through the chunk transform for shape parity with normal
    // chunks. The `error` object has no encryptable string fields under
    // the current transform — but the message text we put here is
    // controlled by us, so this is safe regardless.
    if let Some(transform) = chunk_transform {
        transform(&mut chunk)?;
    }
    let serialized = serde_json::to_string(&chunk).map_err(|e| AppError::Internal(e.into()))?;
    let bytes = format!("data: {serialized}\n\n").into_bytes();
    hasher.update(&bytes);
    tx.send(Ok(Bytes::from(bytes)))
        .await
        .map_err(|_| AppError::Internal(anyhow::anyhow!("client disconnected")))?;
    Ok(())
}

async fn emit_max_iterations_terminator(
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
    chunk_transform: Option<&ChunkTransform>,
    hasher: &mut Sha256,
    chat_id: Option<&str>,
    model: Option<&str>,
    created: Option<i64>,
) -> Result<(), AppError> {
    let mut chunk = json!({
        "id": chat_id.unwrap_or(""),
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
            NEARAI_LOOP_TERMINATED_KEY: "max_iterations"
        }]
    });
    if let Some(m) = model {
        chunk["model"] = m.into();
    }
    if let Some(c) = created {
        chunk["created"] = c.into();
    }
    if let Some(transform) = chunk_transform {
        transform(&mut chunk)?;
    }
    let serialized = serde_json::to_string(&chunk).map_err(|e| AppError::Internal(e.into()))?;
    let bytes = format!("data: {serialized}\n\n").into_bytes();
    hasher.update(&bytes);
    tx.send(Ok(Bytes::from(bytes)))
        .await
        .map_err(|_| AppError::Internal(anyhow::anyhow!("client disconnected")))?;
    Ok(())
}

async fn forward_done(
    tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
    hasher: &mut Sha256,
) -> Result<(), AppError> {
    let bytes: Bytes = Bytes::from_static(b"data: [DONE]\n\n");
    hasher.update(&bytes);
    tx.send(Ok(bytes))
        .await
        .map_err(|_| AppError::Internal(anyhow::anyhow!("client disconnected")))
}

// ── Brave LLM Context API ───────────────────────────────────────────

enum BraveError {
    Timeout,
    Other(String),
}

/// Hit Brave's LLM Context endpoint and return a model-ready text block.
///
/// Defaults match cloud-api's `WebContextSearchToolExecutor` so behavior is
/// consistent between `/v1/responses` (in cloud-api) and the in-CVM loop.
async fn brave_llm_context_search(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    query: &str,
    timeout: Duration,
    tracing_ids: &TracingIds,
) -> Result<String, BraveError> {
    // Cloud-api defaults (see crates/services/src/responses/tools/web_context_search.rs).
    let query_params: [(&str, &str); 9] = [
        ("q", query),
        ("count", "20"),
        ("maximum_number_of_urls", "20"),
        ("maximum_number_of_tokens", "8192"),
        ("maximum_number_of_snippets", "50"),
        ("maximum_number_of_tokens_per_url", "4096"),
        ("maximum_number_of_snippets_per_url", "50"),
        ("context_threshold_mode", "balanced"),
        ("spellcheck", "true"),
    ];

    let mut req = client
        .get(url)
        .header("X-Subscription-Token", api_key)
        .header("Accept", "application/json")
        .timeout(timeout)
        .query(&query_params);
    // Propagate request_id only if it was inbound; org/workspace IDs are not
    // forwarded to third-party APIs.
    if tracing_ids.request_id_inbound {
        req = req.header("x-request-id", tracing_ids.request_id.as_str());
    }

    let response = req.send().await.map_err(|e| {
        if e.is_timeout() {
            BraveError::Timeout
        } else {
            BraveError::Other(format!("brave request failed: {}", error_category(&e)))
        }
    })?;

    let status = response.status();
    if !status.is_success() {
        return Err(BraveError::Other(format!("brave HTTP {}", status.as_u16())));
    }

    // Read the body in chunks so we can enforce a hard size cap regardless
    // of `Content-Length`. The defaults we send keep responses well under
    // `BRAVE_MAX_RESPONSE_BYTES`; this cap is a backstop in case the search
    // endpoint is misconfigured or compromised.
    let mut body_bytes: Vec<u8> = Vec::with_capacity(16 * 1024);
    let mut body_stream = response.bytes_stream();
    while let Some(chunk) = body_stream.next().await {
        let chunk = chunk.map_err(|e| {
            BraveError::Other(format!("brave body read failed: {}", error_category(&e)))
        })?;
        if body_bytes.len().saturating_add(chunk.len()) > BRAVE_MAX_RESPONSE_BYTES {
            return Err(BraveError::Other(format!(
                "brave response exceeded {BRAVE_MAX_RESPONSE_BYTES}-byte cap"
            )));
        }
        body_bytes.extend_from_slice(&chunk);
    }

    let parsed: BraveContextResponse = serde_json::from_slice(&body_bytes)
        .map_err(|e| BraveError::Other(format!("brave JSON parse failed: {e}")))?;

    Ok(format_context_response(&parsed))
}

fn error_category(e: &reqwest::Error) -> &'static str {
    if e.is_timeout() {
        "timeout"
    } else if e.is_connect() {
        "connect"
    } else if e.is_request() {
        "request"
    } else if e.is_body() {
        "body"
    } else if e.is_decode() {
        "decode"
    } else {
        "unknown"
    }
}

/// Brave LLM Context response shape (subset). Mirror of cloud-api's
/// `BraveContextResponse` — only the fields we use.
#[derive(serde::Deserialize)]
struct BraveContextResponse {
    #[serde(default)]
    grounding: BraveContextGrounding,
    #[serde(default)]
    sources: std::collections::HashMap<String, BraveContextSource>,
}

#[derive(serde::Deserialize, Default)]
struct BraveContextGrounding {
    #[serde(default)]
    generic: Vec<BraveContextResult>,
}

#[derive(serde::Deserialize)]
struct BraveContextResult {
    url: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    snippets: Vec<String>,
}

#[derive(serde::Deserialize)]
struct BraveContextSource {
    #[serde(default)]
    title: Option<String>,
}

/// Render the Brave context payload as a plaintext block the model can
/// consume directly. Skips entries with no URL or no usable snippets; falls
/// back to sources[url].title when the grounding entry has no title of its
/// own. Mirrors `context_response_to_web_results` in cloud-api/brave.rs.
///
/// Truncates at `MAX_FORMATTED_OUTPUT_BYTES` with a marker, since we don't
/// want to depend on Brave honoring its input caps. The truncated output
/// is what we both emit to the client AND feed back to the model on the
/// next iteration, so this also bounds prompt growth across iterations.
fn format_context_response(resp: &BraveContextResponse) -> String {
    let mut out = String::new();
    let mut n: u32 = 0;
    let mut truncated = false;
    'outer: for entry in &resp.grounding.generic {
        let url = entry.url.trim();
        if url.is_empty() {
            continue;
        }
        let snippets: Vec<&str> = entry
            .snippets
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if snippets.is_empty() {
            continue;
        }
        let title = entry
            .title
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .or_else(|| {
                resp.sources
                    .get(url)
                    .and_then(|s| s.title.as_deref())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| url.to_string());
        n += 1;
        let separator = if n > 1 { "\n\n" } else { "" };
        let header = format!("{separator}[{n}] {title}\n{url}\n");
        let joined = snippets.join("\n\n");
        for piece in [header.as_str(), joined.as_str()] {
            if out.len() + piece.len() > MAX_FORMATTED_OUTPUT_BYTES {
                let remaining = MAX_FORMATTED_OUTPUT_BYTES.saturating_sub(out.len());
                // Find a UTF-8 char boundary at or before `remaining`.
                let mut cut = remaining;
                while cut > 0 && !piece.is_char_boundary(cut) {
                    cut -= 1;
                }
                out.push_str(&piece[..cut]);
                truncated = true;
                break 'outer;
            }
            out.push_str(piece);
        }
    }
    if truncated {
        out.push_str("\n[truncated]");
    }
    if out.is_empty() {
        "No results.".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_in_detects_exact_single_tool() {
        let req = json!({"tools": [{"type": "web_context_search"}]});
        assert!(is_web_context_search_request(&req));
    }

    #[test]
    fn opt_in_rejects_mixed_tools() {
        let req = json!({"tools": [
            {"type": "web_context_search"},
            {"type": "function", "function": {"name": "x"}}
        ]});
        assert!(!is_web_context_search_request(&req));
    }

    #[test]
    fn opt_in_rejects_function_only() {
        let req = json!({"tools": [{"type": "function", "function": {"name": "x"}}]});
        assert!(!is_web_context_search_request(&req));
    }

    #[test]
    fn opt_in_rejects_no_tools() {
        let req = json!({"messages": []});
        assert!(!is_web_context_search_request(&req));
    }

    #[test]
    fn opt_in_rejects_two_web_context_search_entries() {
        let req = json!({"tools": [
            {"type": "web_context_search"},
            {"type": "web_context_search"}
        ]});
        assert!(!is_web_context_search_request(&req));
    }

    #[test]
    fn rewrite_tool_replaces_with_function_def() {
        let mut req = json!({"tools": [{"type": "web_context_search"}]});
        rewrite_tool_for_upstream(&mut req);
        let tools = req["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], WEB_CONTEXT_SEARCH_TOOL_NAME);
        assert!(tools[0]["function"]["parameters"]["required"]
            .as_array()
            .unwrap()
            .contains(&Value::String("query".to_string())));
    }

    #[test]
    fn parse_query_handles_well_formed() {
        assert_eq!(
            parse_query(r#"{"query":"  hello world  "}"#),
            Some("hello world".to_string())
        );
    }

    #[test]
    fn parse_query_handles_malformed() {
        assert_eq!(parse_query("not json"), None);
        assert_eq!(parse_query("{}"), None);
        assert_eq!(parse_query(r#"{"other":"x"}"#), None);
    }

    #[test]
    fn merge_tool_call_deltas_assembles_split_arguments() {
        let mut acc: Vec<Value> = Vec::new();
        merge_tool_call_deltas(
            &mut acc,
            &[json!({
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_context_search", "arguments": ""}
            })],
        );
        merge_tool_call_deltas(
            &mut acc,
            &[json!({
                "index": 0,
                "function": {"arguments": r#"{"query":"#}
            })],
        );
        merge_tool_call_deltas(
            &mut acc,
            &[json!({
                "index": 0,
                "function": {"arguments": r#""rust"}"#}
            })],
        );
        assert_eq!(acc.len(), 1);
        assert_eq!(acc[0]["id"], "call_1");
        assert_eq!(acc[0]["function"]["name"], "web_context_search");
        assert_eq!(acc[0]["function"]["arguments"], r#"{"query":"rust"}"#);
    }

    #[test]
    fn all_calls_predicate_requires_correct_name() {
        let calls = vec![json!({"function": {"name": "web_context_search"}})];
        assert!(all_calls_are_web_context_search(&calls));
        let bad = vec![json!({"function": {"name": "something_else"}})];
        assert!(!all_calls_are_web_context_search(&bad));
        let mixed = vec![
            json!({"function": {"name": "web_context_search"}}),
            json!({"function": {"name": "other"}}),
        ];
        assert!(!all_calls_are_web_context_search(&mixed));
    }

    #[test]
    fn format_context_response_renders_block() {
        let resp = BraveContextResponse {
            grounding: BraveContextGrounding {
                generic: vec![
                    BraveContextResult {
                        url: "https://example.com/a".to_string(),
                        title: Some("Title A".to_string()),
                        snippets: vec!["snippet 1".to_string(), "snippet 2".to_string()],
                    },
                    BraveContextResult {
                        url: "".to_string(), // empty url — skip
                        title: Some("skipped".to_string()),
                        snippets: vec!["x".to_string()],
                    },
                    BraveContextResult {
                        url: "https://example.com/b".to_string(),
                        title: None,
                        snippets: vec!["only".to_string()],
                    },
                ],
            },
            sources: std::collections::HashMap::from([(
                "https://example.com/b".to_string(),
                BraveContextSource {
                    title: Some("Source B Title".to_string()),
                },
            )]),
        };
        let out = format_context_response(&resp);
        assert!(out.contains("[1] Title A"));
        assert!(out.contains("https://example.com/a"));
        assert!(out.contains("snippet 1\n\nsnippet 2"));
        assert!(out.contains("[2] Source B Title"));
        assert!(out.contains("https://example.com/b"));
        assert!(!out.contains("skipped"));
    }

    #[test]
    fn format_context_response_empty_falls_back_to_no_results() {
        let resp = BraveContextResponse {
            grounding: BraveContextGrounding::default(),
            sources: std::collections::HashMap::new(),
        };
        assert_eq!(format_context_response(&resp), "No results.");
    }
}
