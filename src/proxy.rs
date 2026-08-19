use std::sync::Arc;

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};

use crate::cache::ChatCache;
use crate::error::AppError;
use crate::signing::SigningPair;
use crate::{AppState, TracingIds};

#[cfg(test)]
#[path = "proxy_upstream_error_tests.rs"]
mod proxy_upstream_error_tests;
#[cfg(test)]
#[path = "proxy_validation_error_tests.rs"]
mod proxy_validation_error_tests;

/// Parsed upstream error info for logging and re-wrapping.
pub struct UpstreamErrorInfo {
    pub message: String,
    pub error_type: String,
}

/// Parse an upstream error body to extract message and type.
/// Handles both vLLM flat format (`{"object":"error","message":"...","type":"..."}`)
/// and nested format (`{"error":{"message":"...","type":"..."}}`).
/// Returns None if the body is not parseable JSON with the expected fields.
///
/// The extracted message is sanitized to strip user data from validation error
/// details (the `'input'` and `'ctx'` fields in Python-formatted validation dicts).
pub fn parse_upstream_error(body: &[u8]) -> Option<UpstreamErrorInfo> {
    let json: serde_json::Value = serde_json::from_slice(body).ok()?;

    // Try nested format first: {"error": {"message": "...", "type": "..."}}
    if let Some(error_obj) = json.get("error").filter(|v| v.is_object()) {
        let message = error_obj.get("message")?.as_str()?.to_string();
        let error_type = error_obj
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        return Some(UpstreamErrorInfo {
            message: sanitize_validation_errors(&message),
            error_type,
        });
    }

    // Try vLLM flat format: {"message": "...", "type": "...", "object": "error"}
    let message = json.get("message")?.as_str()?.to_string();
    let error_type = json
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    Some(UpstreamErrorInfo {
        message: sanitize_validation_errors(&message),
        error_type,
    })
}

/// Regex to extract 'type' values from Python-formatted validation error dicts.
/// Handles both single-quoted and double-quoted strings.
static VALIDATION_TYPE_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r#"'type':\s*(?:'([^']+)'|"([^"]+)")"#).unwrap());

/// Regex to extract 'msg' values from Python-formatted validation error dicts.
/// Handles both single-quoted and double-quoted strings.
static VALIDATION_MSG_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r#"'msg':\s*(?:'([^']*)'|"([^"]*)")"#).unwrap());

/// Regex to extract error type from pydantic v2 bracket sections: [type=missing, ...]
static PYDANTIC_V2_TYPE_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"type=(\w+)").unwrap());

/// Sanitize validation error messages to prevent leaking user conversation content.
///
/// Backend validation errors (from SGLang/vLLM) include `'input'` and `'ctx'` fields
/// containing the original request data, which may include user messages, AI responses,
/// and other sensitive conversation content. This function strips those fields while
/// preserving useful error type and message information.
pub fn sanitize_validation_errors(message: &str) -> String {
    // Check for sensitive fields in both Python dict format ('input':, 'ctx':)
    // and pydantic v2 format (input_value=, input_type=)
    let has_python_dict_fields = message.contains("'input':") || message.contains("'ctx':");
    let has_pydantic_v2_fields =
        message.contains("input_value=") || message.contains("input_type=");

    if !has_python_dict_fields && !has_pydantic_v2_fields {
        return message.to_string();
    }

    message
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();

            // Skip stack traces, HTTP method lines, and pydantic v2 "For further information" URLs
            if trimmed.starts_with("File \"")
                || trimmed.starts_with("POST ")
                || trimmed.starts_with("GET ")
                || trimmed.starts_with("For further information visit")
            {
                return None;
            }

            // Python dict format: lines with 'input' or 'ctx' fields
            if trimmed.contains("'input':") || trimmed.contains("'ctx':") {
                let error_type = VALIDATION_TYPE_RE
                    .captures(trimmed)
                    .and_then(|c| c.get(1).or_else(|| c.get(2)))
                    .map(|m| m.as_str());
                let error_msg = VALIDATION_MSG_RE
                    .captures(trimmed)
                    .and_then(|c| c.get(1).or_else(|| c.get(2)))
                    .map(|m| m.as_str());

                let sanitized = match (error_type, error_msg) {
                    (Some(t), Some(m)) => format!("  {}: {}", t, m),
                    (Some(t), None) => format!("  {}", t),
                    _ => "  (validation error)".to_string(),
                };
                Some(sanitized)
            } else if trimmed.contains("input_value=") || trimmed.contains("input_type=") {
                // Pydantic v2 format: "Field required [type=missing, input_value={...}, input_type=dict]"
                // Extract only the description and error type, discard input_value which contains user data
                let desc = trimmed.split('[').next().unwrap_or("").trim();
                // Guard: if desc itself contains sensitive data (unexpected format), use placeholder
                let desc = if desc.contains("input_value=") || desc.contains("input_type=") {
                    "(validation error)"
                } else {
                    desc
                };
                let error_type = PYDANTIC_V2_TYPE_RE
                    .captures(trimmed)
                    .and_then(|c| c.get(1))
                    .map(|m| m.as_str());
                match error_type {
                    Some(t) => Some(format!("  {} [type={}]", desc, t)),
                    None => Some(format!("  {}", desc)),
                }
            } else {
                Some(line.to_string())
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render an upstream URL for logs without caller-controlled credential/query data.
pub(crate) fn sanitized_upstream_url_for_logs(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(mut parsed) => {
            let _ = parsed.set_username("");
            let _ = parsed.set_password(None);
            parsed.set_query(None);
            parsed.set_fragment(None);
            parsed.to_string()
        }
        Err(_) => url
            .split(['?', '#'])
            .next()
            .unwrap_or("<invalid upstream url>")
            .to_string(),
    }
}

/// Parse an upstream error body, log it, and return the parsed info.
pub(crate) fn log_upstream_error(
    status: reqwest::StatusCode,
    url: &str,
    body: &[u8],
    tracing_ids: Option<&TracingIds>,
) -> Option<UpstreamErrorInfo> {
    let info = parse_upstream_error(body);
    let upstream_url = sanitized_upstream_url_for_logs(url);
    let (request_id, org_id, workspace_id) = match tracing_ids {
        Some(ids) => (
            ids.request_id.as_str(),
            ids.org_id_or_empty(),
            ids.workspace_id_or_empty(),
        ),
        None => ("", "", ""),
    };
    warn!(
        request_id = %request_id,
        org_id = %org_id,
        workspace_id = %workspace_id,
        upstream_status = %status,
        upstream_url = %upstream_url,
        upstream_error_parseable = info.is_some(),
        upstream_error_body_bytes = body.len(),
        "Backend returned non-success status"
    );
    info
}

/// Map an upstream error to the status we return downstream, downgrading a
/// backend 5xx to **400** when the failure is really a *client media-fetch 4xx*.
///
/// vLLM/SGLang fetch caller-supplied image/video URLs server-side and wrap a
/// failed fetch as a generic HTTP 500 whose JSON `message` is the aiohttp
/// `ClientResponseError.__str__` — `NNN, message='...', url='...'` (or the
/// requests form `NNN Client Error: ... for url: ...`). When that embedded
/// fetch status is a 4xx the fault is the caller's URL (e.g. Wikimedia 403s a
/// default User-Agent), not our backend: forwarding the 500 makes cloud-api
/// retry the identical request across providers and ultimately surface a
/// misleading 502 "model unavailable" (nearai/cloud-api#606).
///
/// We normalize to **400**, never the raw 4xx: cloud-api maps upstream
/// 401/403/407 to 5xx (it assumes those signal *our* backend credentials), so
/// returning the literal 403 would just round-trip back into a 502.
/// Takes the already-parsed [`UpstreamErrorInfo`] (from [`log_upstream_error`])
/// to avoid re-parsing/sanitizing the body on the error path.
pub(crate) fn effective_error_status(
    upstream_status: u16,
    info: Option<&UpstreamErrorInfo>,
) -> StatusCode {
    let passthrough = StatusCode::from_u16(upstream_status).unwrap_or(StatusCode::BAD_GATEWAY);
    // Only reinterpret server errors; a genuine 4xx is already correct.
    if !(500..600).contains(&upstream_status) {
        return passthrough;
    }
    match info {
        Some(info)
            if message_is_client_fetch_4xx(&info.message)
                || message_is_allowed_media_domain_error(&info.message) =>
        {
            StatusCode::BAD_REQUEST
        }
        _ => passthrough,
    }
}

/// True when an upstream error `message` describes a client media-URL fetch that
/// the remote host answered with an explicit **4xx**. Anchored on `url=` /
/// `for url:` so only genuine outbound-URL fetches qualify (the only per-request
/// outbound HTTP the engine performs), never an incidental 4-something elsewhere.
fn message_is_client_fetch_4xx(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    if !lower.contains("url=") && !lower.contains("for url:") {
        return false;
    }
    // The pattern only matches 4xx codes, so a match is sufficient — no need to
    // capture/parse the number back out.
    static FETCH_4XX: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        // aiohttp ClientResponseError str:  `NNN, message=`
        // aiohttp exception status field:   `status=NNN`
        // requests/urllib:                  `NNN client error:`
        regex::Regex::new(r"\b4\d\d, message=|status=4\d\d\b|\b4\d\d client error:")
            .expect("static regex compiles")
    });
    FETCH_4XX.is_match(&lower)
}

/// True when an upstream error `message` describes a media URL rejected because
/// its domain is not on the allowlist. vLLM wraps this client-side validation
/// failure as a generic 500, so without this check cloud-api would retry the
/// identical request and surface a misleading 502 "model unavailable".
fn message_is_allowed_media_domain_error(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("url must be from one of the allowed domains")
}

/// Reports usage to the cloud API for billing.
///
/// Posts to `POST {cloud_api_url}/v1/internal/usage` with `Authorization:
/// Bearer {cloud_api_usage_token}` (a shared infrastructure secret) and carries
/// `organization_id` + `workspace_id` + `api_key_id` in the body so cloud-api
/// can attribute the usage. The legacy `sk-`-authenticated `POST /v1/usage`
/// path has been removed from cloud-api, so reporting is skipped (with an error
/// log) when the usage token or any identity field is missing — rather than
/// posting to a deleted endpoint.
#[derive(Clone)]
pub struct UsageReporter {
    pub http_client: reqwest::Client,
    pub cloud_api_url: String,
    pub model_name: String,
    /// Shared infrastructure token for the service-token reporting path.
    /// Required: usage is reported only via `/v1/internal/usage`. cloud-api
    /// removed the legacy `sk-`-authenticated `POST /v1/usage` endpoint, so
    /// reporting is skipped (with an error log) when this or any identity
    /// field below is missing.
    pub cloud_api_usage_token: Option<String>,
    pub org_id: Option<String>,
    pub workspace_id: Option<String>,
    pub api_key_id: Option<String>,
    /// Safe correlation context for logs and the Cloud API request. None of
    /// these values are emitted as metric labels.
    pub request_id: Option<String>,
    pub request_source: crate::auth::RequestSource,
}

impl UsageReporter {
    /// Whether the service-token reporting path can be used. False when the
    /// usage token isn't configured or the auth response was missing identity
    /// fields; in that case usage reporting is skipped (the legacy `sk-`-bearer
    /// `/v1/usage` endpoint no longer exists on cloud-api).
    #[cfg(test)]
    fn can_use_service_token_path(&self) -> bool {
        self.service_path_unavailable_reason().is_none()
    }

    fn service_path_unavailable_reason(&self) -> Option<UsageReportOutcome> {
        if self.cloud_api_usage_token.is_none() {
            Some(UsageReportOutcome::MissingUsageToken)
        } else if self.org_id.is_none() || self.workspace_id.is_none() || self.api_key_id.is_none()
        {
            Some(UsageReportOutcome::MissingAuthIdentity)
        } else {
            None
        }
    }
}

/// Terminal state of one direct-key usage-reporting flow. Values are bounded
/// and intentionally exclude tenant or request identifiers, so they are safe
/// as metric labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UsageReportOutcome {
    Accepted,
    Http4xx,
    Http5xx,
    HttpOther,
    Timeout,
    ConnectError,
    TransportError,
    MissingUsageToken,
    MissingAuthIdentity,
    InvalidBody,
    MissingBillableUsage,
    MissingResponseId,
}

impl UsageReportOutcome {
    fn as_label(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Http4xx => "http_4xx",
            Self::Http5xx => "http_5xx",
            Self::HttpOther => "http_other",
            Self::Timeout => "timeout",
            Self::ConnectError => "connect_error",
            Self::TransportError => "transport_error",
            Self::MissingUsageToken => "missing_usage_token",
            Self::MissingAuthIdentity => "missing_auth_identity",
            Self::InvalidBody => "invalid_body",
            Self::MissingBillableUsage => "missing_billable_usage",
            Self::MissingResponseId => "missing_response_id",
        }
    }
}

fn classify_usage_http_status(status: reqwest::StatusCode) -> UsageReportOutcome {
    if status.is_success() {
        UsageReportOutcome::Accepted
    } else if status.is_client_error() {
        UsageReportOutcome::Http4xx
    } else if status.is_server_error() {
        UsageReportOutcome::Http5xx
    } else {
        UsageReportOutcome::HttpOther
    }
}

fn classify_usage_request_error(error: &reqwest::Error) -> UsageReportOutcome {
    if error.is_timeout() {
        UsageReportOutcome::Timeout
    } else if error.is_connect() {
        UsageReportOutcome::ConnectError
    } else {
        UsageReportOutcome::TransportError
    }
}

fn record_usage_report_outcome(
    reporter: &UsageReporter,
    outcome: UsageReportOutcome,
    duration: Option<std::time::Duration>,
) {
    let labels = [
        ("outcome", outcome.as_label()),
        ("auth_path", reporter.request_source.auth_path.as_label()),
        (
            "ingress_route",
            reporter.request_source.ingress_route.as_label(),
        ),
    ];
    metrics::counter!("inference_proxy_usage_reports_total", &labels).increment(1);
    if let Some(duration) = duration {
        metrics::histogram!("inference_proxy_usage_report_duration_seconds", &labels)
            .record(duration.as_secs_f64());
    }
}

/// What kind of usage to extract from the response, and the `inference_type`
/// label to report it under.
#[derive(Clone, Default)]
pub enum UsageType {
    /// Extract prompt_tokens / completion_tokens from the `usage` object.
    #[default]
    ChatCompletion,
    /// Count items in the `data` array (for image generation).
    ImageGeneration,
    /// Input-token-billed, output-less kinds. They share the embeddings
    /// response shape (top-level `usage.prompt_tokens`, no completion
    /// tokens) and differ only in the reported label, so cloud-api records
    /// the correct `inference_type` instead of mislabeling them as
    /// `chat_completion`. See nearai/infra#169.
    Embedding,
    Rerank,
    Score,
}

/// Shape of the reassembled non-streaming response when `proxy_json_request`
/// converts an upstream SSE stream back into a single JSON body.
///
/// `/v1/chat/completions` and `/v1/completions` both stream when forwarded
/// internally, but the SSE chunk shape and the final response shape differ:
/// chat completions carry `choices[].delta.{role,content,...}` and reassemble
/// to `{object: "chat.completion", choices: [{message: {...}}]}`, whereas
/// text completions carry `choices[].text` and reassemble to
/// `{object: "text_completion", choices: [{text: "..."}]}`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResponseShape {
    #[default]
    ChatCompletion,
    TextCompletion,
}

/// Build a `UsageReporter` if the request was authenticated with a cloud API
/// key. The reporter captures the subject identity (`org_id` / `workspace_id` /
/// `api_key_id`) from the auth response plus the configured shared
/// `cloud_api_usage_token`, so usage can be reported via the service-token
/// `/v1/internal/usage` path. Returns `None` for non-`sk-` (proxy-token) requests.
pub fn make_usage_reporter(
    auth: &crate::auth::RequireAuth,
    state: &AppState,
) -> Option<UsageReporter> {
    // Gate: a reporter exists only for direct sk- requests. The key value
    // itself is no longer stored — usage is attributed via the identity fields
    // below, not the sk-.
    auth.cloud_api_key.as_ref()?;
    let url = state.config.cloud_api_url.as_ref()?;
    Some(UsageReporter {
        http_client: state.http_client.clone(),
        cloud_api_url: url.clone(),
        model_name: state.config.model_name.clone(),
        cloud_api_usage_token: state.config.cloud_api_usage_token.clone(),
        org_id: auth.org_id.clone(),
        workspace_id: auth.workspace_id.clone(),
        api_key_id: auth.api_key_id.clone(),
        request_id: auth.request_id.clone(),
        request_source: auth.request_source,
    })
}

/// Build the `/v1/internal/usage` request body for a parsed response, or
/// `None` when there is nothing billable to report. Pure (no I/O) so it can
/// be unit-tested directly; `try_report_usage` wraps it with the
/// fire-and-forget send.
fn build_usage_body(
    usage_type: &UsageType,
    response_data: &serde_json::Value,
    model_name: &str,
    id: &str,
) -> Option<serde_json::Value> {
    match usage_type {
        UsageType::ChatCompletion => {
            let usage = response_data.get("usage")?;
            let input = usage
                .get("prompt_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let output = usage
                .get("completion_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            if input == 0 && output == 0 {
                return None;
            }
            Some(serde_json::json!({
                "type": "chat_completion",
                "model": model_name,
                "input_tokens": input,
                "output_tokens": output,
                "id": id,
            }))
        }
        UsageType::ImageGeneration => {
            let count = response_data
                .get("data")
                .and_then(|d| d.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            if count == 0 {
                return None;
            }
            Some(serde_json::json!({
                "type": "image_generation",
                "model": model_name,
                "image_count": count,
                "id": id,
            }))
        }
        UsageType::Embedding => input_only_usage_body("embedding", response_data, model_name, id),
        UsageType::Rerank => input_only_usage_body("rerank", response_data, model_name, id),
        UsageType::Score => input_only_usage_body("score", response_data, model_name, id),
    }
}

/// Build the body for an input-token-billed, output-less kind
/// (`embedding`/`rerank`/`score`): the input token count comes from the
/// top-level `usage.prompt_tokens`, and only that count is reported under
/// the given `inference_type` label.
fn input_only_usage_body(
    type_label: &str,
    response_data: &serde_json::Value,
    model_name: &str,
    id: &str,
) -> Option<serde_json::Value> {
    let input = response_data
        .get("usage")?
        .get("prompt_tokens")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    if input == 0 {
        return None;
    }
    Some(serde_json::json!({
        "type": type_label,
        "model": model_name,
        "input_tokens": input,
        "id": id,
    }))
}

/// Extract usage from a parsed JSON response and fire-and-forget a report to the cloud API.
pub(crate) fn try_report_usage(response_data: &serde_json::Value, id: &str, opts: &ProxyOpts) {
    let Some(reporter) = &opts.usage_reporter else {
        return;
    };
    match build_usage_body(&opts.usage_type, response_data, &reporter.model_name, id) {
        Some(body) => spawn_usage_report(reporter, body),
        None => {
            record_usage_report_outcome(reporter, UsageReportOutcome::MissingBillableUsage, None);
            warn!(
                request_id = %reporter.request_id.as_deref().unwrap_or(""),
                org_id = %reporter.org_id.as_deref().unwrap_or(""),
                workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                model = %reporter.model_name,
                auth_path = reporter.request_source.auth_path.as_label(),
                ingress_route = reporter.request_source.ingress_route.as_label(),
                "Skipping direct-key usage report: response contained no billable usage"
            );
        }
    }
}

/// Report chat-completion usage when both cumulative token counts and a
/// provider response ID are available. Missing pieces become explicit terminal
/// outcomes instead of silent gaps in the direct-key reporting funnel.
pub(crate) fn report_chat_usage_if_present(
    reporter: &UsageReporter,
    usage: Option<(i64, i64)>,
    response_id: Option<&str>,
) -> bool {
    let Some((input, output)) = usage else {
        record_usage_report_outcome(reporter, UsageReportOutcome::MissingBillableUsage, None);
        warn!(
            request_id = %reporter.request_id.as_deref().unwrap_or(""),
            org_id = %reporter.org_id.as_deref().unwrap_or(""),
            workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
            api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
            model = %reporter.model_name,
            auth_path = reporter.request_source.auth_path.as_label(),
            ingress_route = reporter.request_source.ingress_route.as_label(),
            "Skipping direct-key usage report: no cumulative token usage was observed"
        );
        return false;
    };
    let Some(id) = response_id else {
        record_usage_report_outcome(reporter, UsageReportOutcome::MissingResponseId, None);
        warn!(
            request_id = %reporter.request_id.as_deref().unwrap_or(""),
            org_id = %reporter.org_id.as_deref().unwrap_or(""),
            workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
            api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
            model = %reporter.model_name,
            input_tokens = input,
            output_tokens = output,
            auth_path = reporter.request_source.auth_path.as_label(),
            ingress_route = reporter.request_source.ingress_route.as_label(),
            "Skipping direct-key usage report: provider response ID was not observed"
        );
        return false;
    };

    let body = serde_json::json!({
        "type": "chat_completion",
        "model": reporter.model_name,
        "input_tokens": input,
        "output_tokens": output,
        "id": id,
    });
    spawn_usage_report(reporter, body);
    true
}

/// Report token usage to cloud-api when a streaming response finalizes — even if
/// it did NOT complete cleanly (client disconnect, upstream error, or no `[DONE]`).
/// Billing must not depend on a clean `[DONE]` (nearai/infra#98): the tokens were
/// already produced. The caller keeps signing/caching gated on clean completion;
/// a partial response cannot be verified, but it was still billed.
///
/// Shared by `proxy_streaming_request` and `proxy_streaming_response` so both
/// streaming paths bill identically. The reporter only exists for direct `sk-`
/// requests (`RequireAuth.cloud_api_key`); cloud-api's own `InterceptStream` is
/// not in that path, so this is the sole biller and there is no double-billing.
fn report_stream_usage_on_finalize(
    usage_reporter: &Option<UsageReporter>,
    usage: Option<(i64, i64)>,
    chat_id: Option<&str>,
    completed_cleanly: bool,
    log_request_id: &str,
    log_org_id: &str,
    log_workspace_id: &str,
) {
    let Some(reporter) = usage_reporter else {
        return;
    };
    let reported = report_chat_usage_if_present(reporter, usage, chat_id);

    if reported && !completed_cleanly {
        let (input, output) = usage.expect("reported usage requires token counts");
        let id = chat_id.expect("reported usage requires a response id");
        info!(
            request_id = %log_request_id,
            org_id = %log_org_id,
            workspace_id = %log_workspace_id,
            chat_id = %id,
            input_tokens = input,
            output_tokens = output,
            "Reported usage for interrupted stream"
        );
    }
}

/// Fire-and-forget POST of a usage event to cloud-api's `/v1/internal/usage`
/// (service-token authenticated). The legacy `sk-`-authenticated `/v1/usage`
/// endpoint has been removed from cloud-api, so when the service-token path is
/// unavailable (missing usage token or identity fields) we log an error and
/// skip — never post to the deleted endpoint.
pub(crate) fn spawn_usage_report(reporter: &UsageReporter, mut body: serde_json::Value) {
    if let Some(outcome) = reporter.service_path_unavailable_reason() {
        record_usage_report_outcome(reporter, outcome, None);
        // No fallback exists anymore. This is a misconfiguration (usage token
        // not set) or an auth response missing identity fields — surface it
        // loudly rather than silently dropping billing.
        error!(
            has_usage_token = reporter.cloud_api_usage_token.is_some(),
            has_org = reporter.org_id.is_some(),
            has_workspace = reporter.workspace_id.is_some(),
            has_api_key_id = reporter.api_key_id.is_some(),
            request_id = %reporter.request_id.as_deref().unwrap_or(""),
            org_id = %reporter.org_id.as_deref().unwrap_or(""),
            workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
            api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
            model = %reporter.model_name,
            auth_path = reporter.request_source.auth_path.as_label(),
            ingress_route = reporter.request_source.ingress_route.as_label(),
            outcome = outcome.as_label(),
            "Skipping usage report: service-token path unavailable and the legacy \
             /v1/usage endpoint is removed — usage NOT billed"
        );
        return;
    }

    // Inject subject identity into the body. Cloud-api's `/v1/internal/usage`
    // handler reads these to attribute the usage.
    match &mut body {
        serde_json::Value::Object(map) => {
            // `unwrap` is fine — all three `Option`s are checked by
            // `can_use_service_token_path` above.
            map.insert(
                "organization_id".to_string(),
                serde_json::Value::String(reporter.org_id.clone().unwrap()),
            );
            map.insert(
                "workspace_id".to_string(),
                serde_json::Value::String(reporter.workspace_id.clone().unwrap()),
            );
            map.insert(
                "api_key_id".to_string(),
                serde_json::Value::String(reporter.api_key_id.clone().unwrap()),
            );
        }
        other => {
            record_usage_report_outcome(reporter, UsageReportOutcome::InvalidBody, None);
            // Today every call site builds the body via `serde_json::json!({…})`
            // so it's always an object. Guard against a future caller passing
            // something else: drop the report rather than send un-attributable
            // bytes to cloud-api, which would silently fail to write a usage row.
            warn!(
                body_kind = %match other {
                    serde_json::Value::Null => "null",
                    serde_json::Value::Bool(_) => "bool",
                    serde_json::Value::Number(_) => "number",
                    serde_json::Value::String(_) => "string",
                    serde_json::Value::Array(_) => "array",
                    serde_json::Value::Object(_) => unreachable!(),
                },
                request_id = %reporter.request_id.as_deref().unwrap_or(""),
                org_id = %reporter.org_id.as_deref().unwrap_or(""),
                workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                model = %reporter.model_name,
                auth_path = reporter.request_source.auth_path.as_label(),
                ingress_route = reporter.request_source.ingress_route.as_label(),
                "Skipping usage report: body is not a JSON object — identity fields \
                 can't be injected, refusing to send unattributable report"
            );
            return;
        }
    }

    let client = reporter.http_client.clone();
    let url = format!("{}/v1/internal/usage", reporter.cloud_api_url);
    let auth = format!("Bearer {}", reporter.cloud_api_usage_token.clone().unwrap());
    let reporter = reporter.clone();
    tokio::spawn(async move {
        let started_at = std::time::Instant::now();
        let mut request = client
            .post(&url)
            .header("authorization", &auth)
            .json(&body)
            .timeout(std::time::Duration::from_secs(5));
        if let Some(request_id) = reporter.request_id.as_deref() {
            request = request.header("x-request-id", request_id);
        }
        match request.send().await {
            Ok(resp) => {
                let status = resp.status();
                let outcome = classify_usage_http_status(status);
                record_usage_report_outcome(&reporter, outcome, Some(started_at.elapsed()));
                if outcome == UsageReportOutcome::Accepted {
                    info!(
                        request_id = %reporter.request_id.as_deref().unwrap_or(""),
                        org_id = %reporter.org_id.as_deref().unwrap_or(""),
                        workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                        api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                        model = %reporter.model_name,
                        status = %status,
                        duration_ms = started_at.elapsed().as_millis() as u64,
                        auth_path = reporter.request_source.auth_path.as_label(),
                        ingress_route = reporter.request_source.ingress_route.as_label(),
                        "Direct-key usage report accepted by Cloud API"
                    );
                } else {
                    warn!(
                        request_id = %reporter.request_id.as_deref().unwrap_or(""),
                        org_id = %reporter.org_id.as_deref().unwrap_or(""),
                        workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                        api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                        model = %reporter.model_name,
                        status = %status,
                        duration_ms = started_at.elapsed().as_millis() as u64,
                        auth_path = reporter.request_source.auth_path.as_label(),
                        ingress_route = reporter.request_source.ingress_route.as_label(),
                        outcome = outcome.as_label(),
                        "Usage reporting returned non-success"
                    );
                }
            }
            Err(error) => {
                let outcome = classify_usage_request_error(&error);
                record_usage_report_outcome(&reporter, outcome, Some(started_at.elapsed()));
                warn!(
                    request_id = %reporter.request_id.as_deref().unwrap_or(""),
                    org_id = %reporter.org_id.as_deref().unwrap_or(""),
                    workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                    api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                    model = %reporter.model_name,
                    error = %error,
                    duration_ms = started_at.elapsed().as_millis() as u64,
                    auth_path = reporter.request_source.auth_path.as_label(),
                    ingress_route = reporter.request_source.ingress_route.as_label(),
                    outcome = outcome.as_label(),
                    "Usage reporting failed"
                );
            }
        }
    });
}

/// Options for proxy requests that need signing.
pub struct ProxyOpts {
    pub signing: Arc<SigningPair>,
    pub cache: Arc<ChatCache>,
    /// Prefix for auto-generated IDs (e.g., "chatcmpl", "img", "emb").
    pub id_prefix: String,
    /// Model name included in the signed text.
    pub model_name: String,
    /// If set, report usage to the cloud API after a successful response.
    pub usage_reporter: Option<UsageReporter>,
    /// What kind of usage to extract from the response.
    pub usage_type: UsageType,
    /// Pre-computed SHA-256 hex hash of the original request body.
    /// When set, this hash is used in the signature instead of hashing the
    /// (possibly decrypted/modified) body that is forwarded to the backend.
    /// This matches the Python proxy behavior where signatures cover the
    /// original client-sent body, not the decrypted version.
    pub request_hash: Option<String>,
    /// Applied to response JSON after signing, before sending to client.
    pub response_transform: Option<crate::encryption::ResponseTransform>,
    /// Applied to each SSE chunk JSON before forwarding to client.
    pub chunk_transform: Option<crate::encryption::ChunkTransform>,
    /// RAII guard for backend connection tracking. For streaming requests,
    /// this is moved into the spawned task so active_conns stays incremented
    /// for the full duration of the stream (not just until the handler returns).
    pub backend_guard: Option<crate::backend_pool::BackendGuard>,
    /// Maximum idle time between upstream SSE chunks. Zero disables the
    /// watchdog. Kept per-request so routes can apply the configured value to
    /// both native streaming and internally reassembled JSON requests.
    pub stream_idle_timeout_secs: u64,
    /// Shape of the reassembled response when forwarding an SSE stream as
    /// a non-streaming JSON body. Defaults to `ChatCompletion`; the
    /// `/v1/completions` route sets this to `TextCompletion`.
    pub response_shape: ResponseShape,
    /// Tracing correlation IDs (request_id / org_id / workspace_id) parsed by
    /// `request_id_middleware`. When `Some`, the corresponding headers are
    /// forwarded to the upstream engine and emitted on the per-request log
    /// line. Direct completion and authenticated passthrough routes pass the selected
    /// request ID; tenant IDs appear only on trusted config-token paths.
    pub tracing_ids: Option<TracingIds>,
}

/// Apply upstream tracing headers to a `reqwest::RequestBuilder`. No-op when
/// `tracing_ids` is `None`.
pub(crate) fn apply_tracing_headers(
    mut req: reqwest::RequestBuilder,
    tracing_ids: Option<&TracingIds>,
) -> reqwest::RequestBuilder {
    let Some(ids) = tracing_ids else {
        return req;
    };
    for (k, v) in ids.upstream_headers() {
        req = req.header(k, v);
    }
    req
}

/// Owned `(request_id, org_id, workspace_id)` strings for emission on the
/// per-request log line. Each value defaults to `""` when absent — matches the
/// pre-refactor behavior so existing Datadog queries keep working.
fn log_ids_or_empty(tracing_ids: &Option<TracingIds>) -> (String, String, String) {
    match tracing_ids {
        Some(ids) => (
            ids.request_id.clone(),
            ids.org_id_or_empty().to_string(),
            ids.workspace_id_or_empty().to_string(),
        ),
        None => (String::new(), String::new(), String::new()),
    }
}

#[derive(Clone, Copy)]
struct RequestMetricLabels {
    auth_path: &'static str,
    ingress_route: &'static str,
    tenant_context: &'static str,
    request_id_origin: &'static str,
}

fn request_metric_labels(tracing_ids: &Option<TracingIds>) -> RequestMetricLabels {
    match tracing_ids {
        Some(ids) => {
            let tenant_context = match (
                ids.request_source.map(|source| source.auth_path),
                ids.org_id.is_some(),
                ids.workspace_id.is_some(),
            ) {
                (Some(crate::auth::AuthPath::CloudApiKey), true, true) => "verified",
                (Some(crate::auth::AuthPath::CloudApiKey), true, false)
                | (Some(crate::auth::AuthPath::CloudApiKey), false, true) => "verified_partial",
                (Some(crate::auth::AuthPath::TrustedConfigToken), true, _)
                | (Some(crate::auth::AuthPath::TrustedConfigToken), _, true) => "trusted_headers",
                (_, false, false) => "absent",
                _ => "present_unknown",
            };
            RequestMetricLabels {
                auth_path: ids
                    .request_source
                    .map(|source| source.auth_path.as_label())
                    .unwrap_or("unknown"),
                ingress_route: ids
                    .request_source
                    .map(|source| source.ingress_route.as_label())
                    .unwrap_or("unknown"),
                tenant_context,
                request_id_origin: if ids.request_id_inbound {
                    "inbound"
                } else {
                    "generated"
                },
            }
        }
        None => RequestMetricLabels {
            auth_path: "unknown",
            ingress_route: "unknown",
            tenant_context: "unknown",
            request_id_origin: "unknown",
        },
    }
}

fn record_completed_request_metrics(
    labels: RequestMetricLabels,
    input_tokens: i64,
    total_duration_ms: u128,
    mode: &'static str,
) {
    let common_labels = [
        ("auth_path", labels.auth_path),
        ("ingress_route", labels.ingress_route),
        ("tenant_context", labels.tenant_context),
        ("request_id_origin", labels.request_id_origin),
        ("mode", mode),
    ];
    metrics::counter!("inference_proxy_completed_requests_total", &common_labels).increment(1);
    metrics::counter!("inference_proxy_input_tokens_total", &common_labels)
        .increment(input_tokens.max(0) as u64);
    metrics::histogram!("inference_proxy_input_tokens", &common_labels)
        .record(input_tokens.max(0) as f64);
    metrics::histogram!("inference_proxy_request_duration_seconds", &common_labels)
        .record(total_duration_ms as f64 / 1_000.0);
}

/// Proxy a non-streaming JSON request to the backend using internal streaming.
///
/// Sends the request to the backend with `stream: true` injected, consumes
/// the SSE stream to reassemble a complete non-streaming response, then signs
/// and returns it. This approach has two advantages over a plain blocking POST:
///
/// 1. **Cancellation**: When the downstream connection drops (e.g. cloud-api
///    timeout), the byte stream is dropped, closing the TCP connection to the
///    backend. The backend (SGLang/vLLM) detects the closed connection and
///    aborts generation, preventing zombie requests from consuming GPU.
///
/// 2. **Idle watchdog**: When configured, an upstream stream that stops
///    producing chunks fails as a typed 504 before any downstream response is
///    sent. The ordinary reqwest timeout remains a total-request bound.
pub async fn proxy_json_request(
    client: &reqwest::Client,
    url: &str,
    request_body: Vec<u8>,
    mut opts: ProxyOpts,
) -> Result<Response, AppError> {
    // Hash the ORIGINAL request body for signing (before we inject stream: true).
    let request_sha256 = opts
        .request_hash
        .take()
        .unwrap_or_else(|| hex::encode(Sha256::digest(&request_body)));

    // Inject stream: true and stream_options.include_usage into the body.
    let streaming_body = inject_streaming(&request_body)?;

    let upstream_start = std::time::Instant::now();
    let req = apply_tracing_headers(
        client
            .post(url)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream"),
        opts.tracing_ids.as_ref(),
    );
    let response = req
        .body(streaming_body)
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    metrics::histogram!("upstream_request_duration_seconds", "endpoint" => "json_via_stream")
        .record(upstream_start.elapsed().as_secs_f64());

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_else(|_| Bytes::from("{}"));
        let info = log_upstream_error(status, url, &body, opts.tracing_ids.as_ref());
        return Err(AppError::Upstream {
            status: effective_error_status(status.as_u16(), info.as_ref()),
            body,
        });
    }

    // Check if the backend actually returned SSE. Some backends may ignore
    // stream: true and return a plain JSON response (e.g. non-chat endpoints
    // routed here, or backends that don't support streaming). In that case,
    // fall back to the original non-streaming flow.
    let is_sse = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.contains("text/event-stream"))
        .unwrap_or(false);

    let mut response_data = if is_sse {
        // Consume the SSE stream and reassemble into a non-streaming response.
        let mut assembler = StreamingResponseAssembler::new(opts.response_shape);
        let mut stream_parser = SseParser::new();
        {
            use futures_util::StreamExt;
            let mut byte_stream = std::pin::pin!(response.bytes_stream());
            loop {
                let next_chunk = if opts.stream_idle_timeout_secs == 0 {
                    byte_stream.next().await
                } else {
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(opts.stream_idle_timeout_secs),
                        byte_stream.next(),
                    )
                    .await
                    {
                        Ok(chunk) => chunk,
                        Err(_) => {
                            metrics::counter!(
                                "upstream_stream_incomplete_total",
                                "reason" => "idle_timeout",
                                "mode" => "json_via_stream"
                            )
                            .increment(1);
                            let (request_id, org_id, workspace_id) =
                                log_ids_or_empty(&opts.tracing_ids);
                            warn!(
                                request_id = %request_id,
                                org_id = %org_id,
                                workspace_id = %workspace_id,
                                model = %opts.model_name.to_lowercase(),
                                timeout_secs = opts.stream_idle_timeout_secs,
                                "Upstream SSE stream exceeded the idle timeout"
                            );
                            return Err(AppError::UpstreamParsed {
                                status: StatusCode::GATEWAY_TIMEOUT,
                                message: "Upstream response stream timed out".to_string(),
                                error_type: "upstream_stream_idle_timeout".to_string(),
                            });
                        }
                    }
                };
                let Some(chunk) = next_chunk else {
                    break;
                };
                let chunk = chunk.map_err(|e| AppError::Internal(e.into()))?;
                stream_parser.process_chunk(&chunk);
                assembler.process_chunk(&chunk);
            }
        }
        stream_parser.finish();
        if opts.stream_idle_timeout_secs > 0 && !stream_parser.seen_done {
            metrics::counter!(
                "upstream_stream_incomplete_total",
                "reason" => "missing_done",
                "mode" => "json_via_stream"
            )
            .increment(1);
            let (request_id, org_id, workspace_id) = log_ids_or_empty(&opts.tracing_ids);
            warn!(
                request_id = %request_id,
                org_id = %org_id,
                workspace_id = %workspace_id,
                model = %opts.model_name.to_lowercase(),
                "Upstream SSE stream ended without [DONE]"
            );
            return Err(AppError::UpstreamParsed {
                status: StatusCode::BAD_GATEWAY,
                message: "Upstream response stream ended before completion".to_string(),
                error_type: "upstream_stream_incomplete".to_string(),
            });
        }
        // If the stream surfaced an upstream error chunk (e.g. SGLang queue-full
        // abort), propagate it as a real upstream error. Otherwise the empty
        // `choices: []` final chunk would be signed and returned as HTTP 200,
        // hiding the failure from cloud-api's retry logic.
        if let Some(err) = assembler.take_error() {
            let status_code = err
                .get("code")
                .and_then(|v| v.as_u64())
                .and_then(|c| u16::try_from(c).ok())
                .and_then(|c| StatusCode::from_u16(c).ok())
                .unwrap_or(StatusCode::BAD_GATEWAY);
            let body_bytes = Bytes::from(
                serde_json::to_vec(&serde_json::json!({ "error": err }))
                    .map_err(|e| AppError::Internal(e.into()))?,
            );
            let reqwest_status = reqwest::StatusCode::from_u16(status_code.as_u16())
                .unwrap_or(reqwest::StatusCode::BAD_GATEWAY);
            let info =
                log_upstream_error(reqwest_status, url, &body_bytes, opts.tracing_ids.as_ref());
            return Err(AppError::Upstream {
                // A client media-fetch failure can arrive as an SSE error chunk
                // with code:500 + `403, message='…', url='…'` — downgrade to 400
                // here too so it isn't retried/masked as a 502 (cloud-api#606).
                status: effective_error_status(status_code.as_u16(), info.as_ref()),
                body: body_bytes,
            });
        }
        assembler.into_response(&opts.id_prefix)
    } else {
        // Backend returned plain JSON — process as before.
        let response_bytes = response
            .bytes()
            .await
            .map_err(|e| AppError::Internal(e.into()))?;
        let mut data: serde_json::Value =
            serde_json::from_slice(&response_bytes).map_err(|e| AppError::Internal(e.into()))?;
        // Generate an ID if not present.
        if data.get("id").and_then(|v| v.as_str()).is_none() {
            let id = format!(
                "{}-{}",
                opts.id_prefix,
                &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]
            );
            if let Some(obj) = data.as_object_mut() {
                obj.insert("id".to_string(), serde_json::Value::String(id));
            }
        }
        data
    };

    // Report usage for cloud API key requests (before encryption, needs plaintext fields)
    let chat_id = response_data["id"].as_str().unwrap_or("").to_string();
    try_report_usage(&response_data, &chat_id, &opts);

    // Structured completion log — one line per successful non-streaming request.
    // Mirrors the log emitted by proxy_streaming_request so both paths are
    // queryable in Datadog by request_id / org_id.
    {
        let input_tokens = response_data
            .pointer("/usage/prompt_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let output_tokens = response_data
            .pointer("/usage/completion_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let total_ms = upstream_start.elapsed().as_millis();
        let (log_request_id, log_org_id, log_workspace_id) = log_ids_or_empty(&opts.tracing_ids);
        let source_labels = request_metric_labels(&opts.tracing_ids);
        record_completed_request_metrics(source_labels, input_tokens, total_ms, "json_via_stream");
        info!(
            request_id = %log_request_id,
            org_id = %log_org_id,
            workspace_id = %log_workspace_id,
            model = %opts.model_name.to_lowercase(),
            chat_id = %chat_id,
            input_tokens,
            output_tokens,
            total_duration_ms = total_ms,
            auth_path = source_labels.auth_path,
            ingress_route = source_labels.ingress_route,
            tenant_context = source_labels.tenant_context,
            request_id_origin = source_labels.request_id_origin,
            "request completed"
        );
    }

    // Apply response transform (e.g., encryption) before hashing/signing.
    if let Some(transform) = opts.response_transform.take() {
        transform(&mut response_data)?;
    }

    // Serialize with compact separators (matching Python's separators=(",",":"))
    let final_body =
        serde_json::to_string(&response_data).map_err(|e| AppError::Internal(e.into()))?;
    let response_sha256 = hex::encode(Sha256::digest(final_body.as_bytes()));

    // Sign and cache
    let text = format!("{}:{request_sha256}:{response_sha256}", opts.model_name);
    let signed = opts.signing.sign_chat(&text).map_err(|e| {
        error!(error = %e, "Signing failed");
        AppError::Internal(e)
    })?;
    let signed_json = serde_json::to_string(&signed).map_err(|e| AppError::Internal(e.into()))?;
    opts.cache.set_chat(&chat_id, &signed_json);

    Ok((
        StatusCode::OK,
        [("content-type", "application/json")],
        final_body,
    )
        .into_response())
}

/// Inject `"stream": true` and `"stream_options": {"include_usage": true}`
/// into a JSON request body for internal streaming.
fn inject_streaming(body: &[u8]) -> Result<Vec<u8>, AppError> {
    let mut json: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;
    json["stream"] = true.into();
    json["stream_options"] = serde_json::json!({"include_usage": true});
    serde_json::to_vec(&json).map_err(|e| AppError::Internal(e.into()))
}

/// Reassembles streaming SSE chunks into a single non-streaming response.
///
/// Processes `data:` lines from the SSE stream. The capture rules and final
/// envelope shape depend on `shape`:
/// - `ChatCompletion`: concatenates `delta.content` / `delta.reasoning_content`,
///   merges `delta.tool_calls` by index, and emits a `chat.completion` object
///   with `choices[].message`.
/// - `TextCompletion`: concatenates `choices[].text` and emits a
///   `text_completion` object with `choices[].text`.
struct StreamingResponseAssembler {
    line_buffer: String,
    id: Option<String>,
    model: Option<String>,
    created: Option<i64>,
    /// Per-choice state, keyed by choice index.
    choices: Vec<ChoiceAssembler>,
    usage: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
    shape: ResponseShape,
    /// First `event["error"]` object seen in the stream. SGLang aborts (e.g.
    /// `--max-queued-requests` overflow) emit `data: {"error": {...}}` then
    /// continue with `[DONE]` plus an empty-choices/zero-usage chunk, so the
    /// upstream returns HTTP 200 with a valid SSE shape. Capturing this lets
    /// the caller surface a real upstream error instead of returning a
    /// phantom HTTP 200 + empty choices.
    error: Option<serde_json::Value>,
    /// Unknown top-level fields seen in the stream (e.g. sglang `sglext`),
    /// preserved verbatim so they survive reassembly. Last-writer-wins.
    top_extra: serde_json::Map<String, serde_json::Value>,
}

/// Accumulates delta fields for a single choice.
struct ChoiceAssembler {
    role: Option<String>,
    content: String,
    reasoning_content: String,
    tool_calls: Vec<serde_json::Value>,
    finish_reason: Option<String>,
    logprobs: Option<serde_json::Value>,
    /// Unknown per-choice and per-delta fields, preserved verbatim and emitted
    /// as choice-level siblings of `message`. This is how `delta.hidden_states`
    /// (sglang `return_hidden_states`) survives reassembly — it maps onto the
    /// native non-streaming `choices[].hidden_states`. Last-writer-wins, which
    /// for sglang is the final hidden-states chunk (last-token states), since
    /// the proxy forces `stream:true` upstream.
    extra: serde_json::Map<String, serde_json::Value>,
}

/// Merge preserved unknown fields into an assembled object, without
/// overwriting fields we already populated.
fn merge_extra(target: &mut serde_json::Value, extra: serde_json::Map<String, serde_json::Value>) {
    if extra.is_empty() {
        return;
    }
    if let Some(obj) = target.as_object_mut() {
        for (k, v) in extra {
            obj.entry(k).or_insert(v);
        }
    }
}

impl StreamingResponseAssembler {
    fn new(shape: ResponseShape) -> Self {
        Self {
            line_buffer: String::new(),
            id: None,
            model: None,
            created: None,
            choices: Vec::new(),
            usage: None,
            metadata: None,
            shape,
            error: None,
            top_extra: serde_json::Map::new(),
        }
    }

    /// Returns the upstream error chunk captured during stream processing, if any.
    fn take_error(&mut self) -> Option<serde_json::Value> {
        self.error.take()
    }

    fn process_chunk(&mut self, chunk: &[u8]) {
        match std::str::from_utf8(chunk) {
            Ok(s) => self.line_buffer.push_str(s),
            Err(_) => self.line_buffer.push_str(&String::from_utf8_lossy(chunk)),
        }

        while let Some(newline_pos) = self.line_buffer.find('\n') {
            let line_end =
                if newline_pos > 0 && self.line_buffer.as_bytes()[newline_pos - 1] == b'\r' {
                    newline_pos - 1
                } else {
                    newline_pos
                };

            let line = &self.line_buffer[..line_end];
            let data = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
                .unwrap_or("")
                .trim();

            if !data.is_empty() && data != "[DONE]" {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                    self.ingest_event(&parsed);
                }
            }

            self.line_buffer.drain(..newline_pos + 1);
        }
    }

    fn ingest_event(&mut self, event: &serde_json::Value) {
        // Capture the first upstream error chunk. SGLang surfaces aborts
        // (queue-full, priority-disabled, waiting timeout) by emitting
        // `data: {"error": {"object":"error","message":"...","type":"...","code":<http_status>}}`
        // mid-stream while keeping the SSE response otherwise well-formed.
        if self.error.is_none() {
            if let Some(err) = event.get("error").filter(|v| v.is_object()) {
                self.error = Some(err.clone());
            }
        }

        // Capture top-level fields from the first event.
        if self.id.is_none() {
            self.id = event.get("id").and_then(|v| v.as_str()).map(String::from);
        }
        if self.model.is_none() {
            self.model = event
                .get("model")
                .and_then(|v| v.as_str())
                .map(String::from);
        }
        if self.created.is_none() {
            self.created = event.get("created").and_then(|v| v.as_i64());
        }
        if self.metadata.is_none() {
            if let Some(m) = event.get("metadata").filter(|v| v.is_object()) {
                self.metadata = Some(m.clone());
            }
        }

        // Capture usage (typically in the final chunk with empty choices).
        if let Some(u) = event.get("usage").filter(|v| v.is_object()) {
            self.usage = Some(u.clone());
        }

        // Preserve unknown top-level fields (e.g. sglang `sglext`) verbatim,
        // so they survive reassembly instead of being silently dropped.
        if let Some(obj) = event.as_object() {
            for (k, v) in obj {
                if v.is_null()
                    || matches!(
                        k.as_str(),
                        "id" | "object"
                            | "model"
                            | "created"
                            | "choices"
                            | "usage"
                            | "metadata"
                            | "error"
                    )
                {
                    continue;
                }
                self.top_extra.insert(k.clone(), v.clone());
            }
        }

        // Process choices/deltas.
        if let Some(choices) = event.get("choices").and_then(|v| v.as_array()) {
            for choice in choices {
                let index = choice.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

                // Grow the choices vec if needed.
                while self.choices.len() <= index {
                    self.choices.push(ChoiceAssembler::new());
                }
                let ca = &mut self.choices[index];

                match self.shape {
                    ResponseShape::ChatCompletion => {
                        if let Some(delta) = choice.get("delta").filter(|v| v.is_object()) {
                            if let Some(role) = delta.get("role").and_then(|v| v.as_str()) {
                                if ca.role.is_none() {
                                    ca.role = Some(role.to_string());
                                }
                            }
                            if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
                                ca.content.push_str(c);
                            }
                            // Some upstream parsers (vLLM `qwen3` reasoning parser) emit
                            // `delta.reasoning` instead of the standard `delta.reasoning_content`;
                            // accept either so non-streaming clients of Qwen reasoning models
                            // see reasoning text in the assembled response.
                            if let Some(r) = delta
                                .get("reasoning_content")
                                .and_then(|v| v.as_str())
                                .or_else(|| delta.get("reasoning").and_then(|v| v.as_str()))
                            {
                                ca.reasoning_content.push_str(r);
                            }
                            if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                                ca.merge_tool_calls(tcs);
                            }
                            // Preserve unknown delta fields (e.g. sglang
                            // `delta.hidden_states` from return_hidden_states),
                            // hoisted to choice level to match the native
                            // non-streaming `choices[].hidden_states` shape.
                            if let Some(delta_obj) = delta.as_object() {
                                for (k, v) in delta_obj {
                                    if v.is_null()
                                        || matches!(
                                            k.as_str(),
                                            "role"
                                                | "content"
                                                | "reasoning"
                                                | "reasoning_content"
                                                | "tool_calls"
                                        )
                                    {
                                        continue;
                                    }
                                    ca.extra.insert(k.clone(), v.clone());
                                }
                            }
                        }
                    }
                    ResponseShape::TextCompletion => {
                        // vLLM/SGLang text-completion SSE chunks emit incremental
                        // tokens at `choices[].text` (no `delta` wrapper).
                        if let Some(t) = choice.get("text").and_then(|v| v.as_str()) {
                            ca.content.push_str(t);
                        }
                    }
                }

                if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    ca.finish_reason = Some(fr.to_string());
                }
                if let Some(lp) = choice.get("logprobs").filter(|v| !v.is_null()) {
                    ca.logprobs = Some(lp.clone());
                }

                // Preserve unknown choice-level fields (e.g. `matched_stop`, or
                // a per-choice `hidden_states`) verbatim.
                if let Some(obj) = choice.as_object() {
                    for (k, v) in obj {
                        if v.is_null()
                            || matches!(
                                k.as_str(),
                                "index"
                                    | "delta"
                                    | "text"
                                    | "finish_reason"
                                    | "logprobs"
                                    | "message"
                            )
                        {
                            continue;
                        }
                        ca.extra.insert(k.clone(), v.clone());
                    }
                }
            }
        }
    }

    /// Build the final non-streaming response JSON. The `object` field and
    /// per-choice shape depend on `self.shape`.
    fn into_response(self, id_prefix: &str) -> serde_json::Value {
        let id = self.id.unwrap_or_else(|| {
            format!(
                "{}-{}",
                id_prefix,
                &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]
            )
        });

        let shape = self.shape;
        let choices: Vec<serde_json::Value> = self
            .choices
            .into_iter()
            .enumerate()
            .map(|(i, ca)| ca.into_choice_json(i, shape))
            .collect();

        let object = match shape {
            ResponseShape::ChatCompletion => "chat.completion",
            ResponseShape::TextCompletion => "text_completion",
        };

        let mut resp = serde_json::json!({
            "id": id,
            "object": object,
            "choices": choices,
        });

        if let Some(model) = self.model {
            resp["model"] = model.into();
        }
        if let Some(created) = self.created {
            resp["created"] = created.into();
        }
        if let Some(usage) = self.usage {
            resp["usage"] = usage;
        }
        if let Some(metadata) = self.metadata {
            resp["metadata"] = metadata;
        }

        // Re-attach any unknown top-level provider fields (e.g. sglang `sglext`).
        merge_extra(&mut resp, self.top_extra);

        resp
    }
}

impl ChoiceAssembler {
    fn new() -> Self {
        Self {
            role: None,
            content: String::new(),
            reasoning_content: String::new(),
            tool_calls: Vec::new(),
            finish_reason: None,
            logprobs: None,
            extra: serde_json::Map::new(),
        }
    }

    /// Merge streaming tool_call deltas by index.
    ///
    /// First delta for an index carries `id`, `type`, `function.name`.
    /// Subsequent deltas for the same index append to `function.arguments`.
    fn merge_tool_calls(&mut self, deltas: &[serde_json::Value]) {
        for tc_delta in deltas {
            let idx = tc_delta.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

            while self.tool_calls.len() <= idx {
                self.tool_calls.push(serde_json::json!({
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""}
                }));
            }

            let existing = &mut self.tool_calls[idx];

            if let Some(id) = tc_delta.get("id").and_then(|v| v.as_str()) {
                existing["id"] = id.into();
            }
            if let Some(t) = tc_delta.get("type").and_then(|v| v.as_str()) {
                existing["type"] = t.into();
            }
            if let Some(func) = tc_delta.get("function").filter(|v| v.is_object()) {
                if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                    if !name.is_empty() {
                        existing["function"]["name"] = name.into();
                    }
                }
                if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                    let prev = existing["function"]["arguments"].as_str().unwrap_or("");
                    let mut combined = prev.to_string();
                    combined.push_str(args);
                    existing["function"]["arguments"] = combined.into();
                }
            }
        }
    }

    fn into_choice_json(self, index: usize, shape: ResponseShape) -> serde_json::Value {
        match shape {
            ResponseShape::ChatCompletion => {
                let mut message = serde_json::json!({
                    "role": self.role.unwrap_or_else(|| "assistant".to_string()),
                });

                // Include content/reasoning_content: use null when empty (matches SGLang behavior).
                if self.content.is_empty() {
                    message["content"] = serde_json::Value::Null;
                } else {
                    message["content"] = self.content.into();
                }

                if !self.reasoning_content.is_empty() {
                    message["reasoning_content"] = self.reasoning_content.into();
                }

                if !self.tool_calls.is_empty() {
                    message["tool_calls"] = self.tool_calls.into();
                }

                let mut choice = serde_json::json!({
                    "index": index,
                    "message": message,
                    "finish_reason": self.finish_reason,
                    "logprobs": self.logprobs,
                });
                // Re-attach preserved per-choice fields (e.g. hidden_states)
                // as siblings of `message`, matching native sglang.
                merge_extra(&mut choice, self.extra);
                choice
            }
            ResponseShape::TextCompletion => {
                let mut choice = serde_json::json!({
                    "index": index,
                    "text": self.content,
                    "finish_reason": self.finish_reason,
                    "logprobs": self.logprobs,
                });
                merge_extra(&mut choice, self.extra);
                choice
            }
        }
    }
}

/// Proxy a streaming SSE request. Hashes all chunks, signs at end, caches signature.
pub async fn proxy_streaming_request(
    client: &reqwest::Client,
    url: &str,
    request_body: Vec<u8>,
    mut opts: ProxyOpts,
) -> Result<Response, AppError> {
    let request_sha256 = opts
        .request_hash
        .take()
        .unwrap_or_else(|| hex::encode(Sha256::digest(&request_body)));

    let upstream_start = std::time::Instant::now();
    let req = apply_tracing_headers(
        client
            .post(url)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream"),
        opts.tracing_ids.as_ref(),
    );
    let response = req
        .body(request_body)
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    metrics::histogram!("upstream_request_duration_seconds", "endpoint" => "streaming")
        .record(upstream_start.elapsed().as_secs_f64());

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_else(|_| Bytes::from("{}"));
        let info = log_upstream_error(status, url, &body, opts.tracing_ids.as_ref());
        return Err(AppError::Upstream {
            status: effective_error_status(status.as_u16(), info.as_ref()),
            body,
        });
    }

    // Capture log fields before any partial moves from opts.
    let (log_request_id, log_org_id, log_workspace_id) = log_ids_or_empty(&opts.tracing_ids);
    let source_labels = request_metric_labels(&opts.tracing_ids);

    let signing = opts.signing.clone();
    let cache = opts.cache.clone();
    let usage_reporter = opts.usage_reporter.clone();
    let model_name = opts.model_name.clone();
    let chunk_transform = opts.chunk_transform;
    let backend_guard = opts.backend_guard;
    let stream_idle_timeout_secs = opts.stream_idle_timeout_secs;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    // Spawn a task to consume upstream and forward chunks.
    // Uses select! on tx.closed() to detect client disconnect while waiting
    // for upstream data, preventing resource leaks from abandoned connections.
    let byte_stream = response.bytes_stream();
    tokio::spawn(async move {
        use futures_util::StreamExt;

        let _guard = StreamingGuard::new();
        // Keep backend_guard alive for the full duration of the stream
        // so active_conns tracking is accurate for least-connections selection.
        let _backend_guard = backend_guard;

        let mut byte_stream = std::pin::pin!(byte_stream);
        let mut hasher = Sha256::new();
        let mut parser = SseParser::new();
        let mut upstream_error = false;
        let mut downstream_closed = false;
        let mut incomplete_reason = None;
        let mut transformer = SseTransformer::new(chunk_transform);

        loop {
            tokio::select! {
                chunk = byte_stream.next() => {
                    match chunk {
                        Some(Ok(chunk)) => {
                            parser.process_chunk(&chunk);

                            // Normalize (and encrypt, if active) the chunk, then hash
                            // what the client actually receives for signatures.
                            let to_send = match transformer.process_chunk(&chunk) {
                                Ok(transformed) => transformed,
                                Err(e) => {
                                    error!(error = %e, "Stream transform failed");
                                    let _ = tx.send(Err(std::io::Error::other(
                                        "Stream transform failed",
                                    ))).await;
                                    upstream_error = true;
                                    incomplete_reason = Some("transform_error");
                                    break;
                                }
                            };
                            hasher.update(&to_send);

                            if tx.send(Ok(to_send)).await.is_err() {
                                downstream_closed = true;
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            error!(error = %e, "Error reading upstream stream");
                            upstream_error = true;
                            incomplete_reason = Some("upstream_read_error");
                            let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                            break;
                        }
                        None => break, // stream ended
                    }
                }
                _ = tx.closed() => {
                    info!("Client disconnected, aborting upstream stream processing");
                    downstream_closed = true;
                    break;
                }
                _ = tokio::time::sleep(std::time::Duration::from_secs(
                    stream_idle_timeout_secs,
                )), if stream_idle_timeout_secs > 0 => {
                    warn!(
                        request_id = %log_request_id,
                        org_id = %log_org_id,
                        workspace_id = %log_workspace_id,
                        model = %model_name.to_lowercase(),
                        timeout_secs = stream_idle_timeout_secs,
                        "Upstream SSE stream exceeded the idle timeout"
                    );
                    upstream_error = true;
                    incomplete_reason = Some("idle_timeout");
                    let _ = tx.send(Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "Upstream response stream timed out",
                    ))).await;
                    break;
                }
            }
        }

        parser.finish();

        // Flush any remaining buffered content in the transformer
        if !upstream_error && !downstream_closed {
            match transformer.flush() {
                Ok(flushed) if !flushed.is_empty() => {
                    hasher.update(&flushed);
                    if tx.send(Ok(flushed)).await.is_err() {
                        downstream_closed = true;
                    }
                }
                Err(e) => {
                    error!(error = %e, "Stream transform flush failed");
                    let _ = tx
                        .send(Err(std::io::Error::other("Stream transform failed")))
                        .await;
                    upstream_error = true;
                    incomplete_reason = Some("transform_error");
                }
                _ => {}
            }
        }

        // Once the watchdog is enabled, an upstream EOF without the protocol
        // terminator is an explicit downstream body error rather than a
        // successful-looking truncated HTTP 200.
        if stream_idle_timeout_secs > 0
            && !upstream_error
            && !downstream_closed
            && !parser.seen_done
        {
            incomplete_reason = Some("missing_done");
            let _ = tx
                .send(Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Upstream response stream ended before [DONE]",
                )))
                .await;
        }

        let completed_cleanly = !upstream_error && !downstream_closed && parser.seen_done;
        if !completed_cleanly && !downstream_closed {
            let reason = incomplete_reason.unwrap_or("missing_done");
            metrics::counter!(
                "upstream_stream_incomplete_total",
                "reason" => reason,
                "mode" => "streaming_request"
            )
            .increment(1);
        }

        // Bill for the tokens the backend already produced, even when the stream
        // did NOT finish cleanly (nearai/infra#98). With continuous_usage_stats
        // forced on cloud-key streaming (routes/chat.rs, routes/completions.rs),
        // `parser.usage` holds the running cumulative counts from the last chunk
        // we saw — the best figure at the point of interruption. Signing/caching
        // stays gated on a clean [DONE] below.
        report_stream_usage_on_finalize(
            &usage_reporter,
            parser.usage,
            parser.chat_id.as_deref(),
            completed_cleanly,
            &log_request_id,
            &log_org_id,
            &log_workspace_id,
        );

        // Only sign and cache for a fully completed stream
        if completed_cleanly {
            let response_sha256 = hex::encode(hasher.finalize());
            if let Some(ref id) = parser.chat_id {
                let text = format!("{model_name}:{request_sha256}:{response_sha256}");
                match signing.sign_chat(&text) {
                    Ok(signed) => {
                        if let Ok(signed_json) = serde_json::to_string(&signed) {
                            cache.set_chat(id, &signed_json);
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Signing failed for streaming response");
                    }
                }

                // Structured completion log — one line per successful streaming request.
                // Carries org_id/request_id propagated from cloud-api so this line is
                // joinable with cloud-api and nginx logs in Datadog.
                let (input_tokens, output_tokens) = parser.usage.unwrap_or((0, 0));
                let total_ms = upstream_start.elapsed().as_millis();
                record_completed_request_metrics(
                    source_labels,
                    input_tokens,
                    total_ms,
                    "streaming_request",
                );
                info!(
                    request_id = %log_request_id,
                    org_id = %log_org_id,
                    workspace_id = %log_workspace_id,
                    model = %model_name.to_lowercase(),
                    chat_id = %id,
                    input_tokens,
                    output_tokens,
                    total_duration_ms = total_ms,
                    auth_path = source_labels.auth_path,
                    ingress_route = source_labels.ingress_route,
                    tenant_context = source_labels.tenant_context,
                    request_id_origin = source_labels.request_id_origin,
                    "request completed"
                );
            } else {
                error!("Chat id could not be extracted from the completed streaming response");
            }
        } else {
            info!(
                upstream_error,
                downstream_closed,
                seen_done = parser.seen_done,
                "Skipping streaming signature cache: stream did not complete cleanly"
            );
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(body)
        .unwrap())
}

/// Normalize OpenAI-compat chat completion streaming chunks emitted by upstream
/// reasoning parsers that use the non-standard `delta.reasoning` field
/// (vLLM `qwen3` parser as of v0.10) so downstream clients see the standard
/// `delta.reasoning_content` field consistently across reasoning models.
/// If both fields are present the existing `reasoning_content` is kept.
pub(crate) fn normalize_chat_chunk(val: &mut serde_json::Value) {
    let Some(choices) = val.get_mut("choices").and_then(|c| c.as_array_mut()) else {
        return;
    };
    for choice in choices {
        let Some(delta) = choice.get_mut("delta").and_then(|d| d.as_object_mut()) else {
            continue;
        };
        if delta.contains_key("reasoning") && !delta.contains_key("reasoning_content") {
            if let Some(v) = delta.remove("reasoning") {
                delta.insert("reasoning_content".to_string(), v);
            }
        }
    }
}

/// Line-buffered SSE transformer that handles data split across chunk boundaries.
/// Always normalizes chat completion chunks (`delta.reasoning` → `delta.reasoning_content`)
/// and optionally applies an additional transform (e.g. encryption).
/// Fail-closed: if a data line contains JSON that cannot be transformed, the stream errors.
struct SseTransformer {
    line_buffer: String,
    extra_transform: Option<crate::encryption::ChunkTransform>,
}

impl SseTransformer {
    fn new(extra_transform: Option<crate::encryption::ChunkTransform>) -> Self {
        Self {
            line_buffer: String::new(),
            extra_transform,
        }
    }

    /// Feed raw bytes into the buffer and return all complete transformed lines.
    /// Incomplete lines are buffered for the next call.
    fn process_chunk(&mut self, chunk: &[u8]) -> Result<Bytes, AppError> {
        let s = std::str::from_utf8(chunk).map_err(|e| {
            AppError::Internal(anyhow::anyhow!("Received invalid UTF-8 in SSE stream: {e}"))
        })?;
        self.line_buffer.push_str(s);

        let mut output = String::new();

        while let Some(newline_pos) = self.line_buffer.find('\n') {
            // Extract the complete line including the newline
            let full_line = self.line_buffer[..=newline_pos].to_string();
            self.line_buffer.drain(..=newline_pos);

            let trimmed = full_line.trim_end_matches(['\n', '\r']);
            let data = trimmed
                .strip_prefix("data: ")
                .or_else(|| trimmed.strip_prefix("data:"));

            if let Some(data) = data {
                let data = data.trim();
                if !data.is_empty() && data != "[DONE]" {
                    // This is a JSON data line — must transform or fail
                    let mut parsed: serde_json::Value =
                        serde_json::from_str(data).map_err(|e| {
                            AppError::Internal(anyhow::anyhow!(
                                "Failed to parse SSE data line: {e}"
                            ))
                        })?;
                    normalize_chat_chunk(&mut parsed);
                    if let Some(ref extra) = self.extra_transform {
                        (extra)(&mut parsed)?;
                    }
                    let re_serialized =
                        serde_json::to_string(&parsed).map_err(|e| AppError::Internal(e.into()))?;
                    output.push_str("data: ");
                    output.push_str(&re_serialized);
                    // Preserve the original line ending
                    let ending = &full_line[trimmed.len()..];
                    output.push_str(ending);
                    continue;
                }
            }
            // Pass through non-data lines, empty lines, and [DONE]
            output.push_str(&full_line);
        }

        Ok(Bytes::from(output))
    }

    /// Flush any remaining buffered content at stream end.
    /// A well-formed SSE stream always ends lines with `\n`, but if the backend
    /// sends a final line without one, this ensures it is still transformed and
    /// forwarded so the signature hash (which covers all raw bytes) matches
    /// what the client receives.
    fn flush(&mut self) -> Result<Bytes, AppError> {
        if self.line_buffer.is_empty() {
            return Ok(Bytes::new());
        }
        let remaining = std::mem::take(&mut self.line_buffer);
        let trimmed = remaining.trim_end_matches(['\n', '\r']);
        let data = trimmed
            .strip_prefix("data: ")
            .or_else(|| trimmed.strip_prefix("data:"));

        if let Some(data) = data {
            let data = data.trim();
            if !data.is_empty() && data != "[DONE]" {
                let mut parsed: serde_json::Value = serde_json::from_str(data).map_err(|e| {
                    AppError::Internal(anyhow::anyhow!("Failed to parse SSE data line: {e}"))
                })?;
                normalize_chat_chunk(&mut parsed);
                if let Some(ref extra) = self.extra_transform {
                    (extra)(&mut parsed)?;
                }
                let re_serialized =
                    serde_json::to_string(&parsed).map_err(|e| AppError::Internal(e.into()))?;
                let mut output = String::from("data: ");
                output.push_str(&re_serialized);
                output.push_str(&remaining[trimmed.len()..]);
                return Ok(Bytes::from(output));
            }
        }
        Ok(Bytes::from(remaining))
    }
}

/// Proxy a multipart request. Caller provides pre-computed request hash covering all field bytes.
pub async fn proxy_multipart_request(
    client: &reqwest::Client,
    url: &str,
    form: reqwest::multipart::Form,
    request_sha256: &str,
    mut opts: ProxyOpts,
) -> Result<Response, AppError> {
    let upstream_start = std::time::Instant::now();
    let req = apply_tracing_headers(client.post(url).multipart(form), opts.tracing_ids.as_ref());
    let response = req.send().await.map_err(|e| AppError::Internal(e.into()))?;
    metrics::histogram!("upstream_request_duration_seconds", "endpoint" => "multipart")
        .record(upstream_start.elapsed().as_secs_f64());

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_else(|_| Bytes::from("{}"));
        let info = log_upstream_error(status, url, &body, opts.tracing_ids.as_ref());
        return Err(AppError::Upstream {
            status: effective_error_status(status.as_u16(), info.as_ref()),
            body,
        });
    }

    let response_bytes = response
        .bytes()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    let mut response_data: serde_json::Value =
        serde_json::from_slice(&response_bytes).map_err(|e| AppError::Internal(e.into()))?;

    let response_id = match response_data.get("id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => {
            let id = format!(
                "{}-{}",
                opts.id_prefix,
                &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]
            );
            if let Some(obj) = response_data.as_object_mut() {
                obj.insert("id".to_string(), serde_json::Value::String(id.clone()));
            }
            id
        }
    };

    // Report usage for cloud API key requests (before encryption, needs plaintext fields)
    try_report_usage(&response_data, &response_id, &opts);

    // Apply response transform (e.g., encryption) before hashing/signing.
    // The signature covers the response bytes the client actually receives.
    if let Some(transform) = opts.response_transform.take() {
        transform(&mut response_data)?;
    }

    // Serialize with compact separators (matching Python's separators=(",",":"))
    let final_body =
        serde_json::to_string(&response_data).map_err(|e| AppError::Internal(e.into()))?;
    let response_sha256 = hex::encode(Sha256::digest(final_body.as_bytes()));

    // Sign and cache
    let text = format!("{}:{request_sha256}:{response_sha256}", opts.model_name);
    let signed = opts.signing.sign_chat(&text).map_err(|e| {
        error!(error = %e, "Signing failed");
        AppError::Internal(e)
    })?;
    let signed_json = serde_json::to_string(&signed).map_err(|e| AppError::Internal(e.into()))?;
    opts.cache.set_chat(&response_id, &signed_json);

    Ok((
        StatusCode::OK,
        [("content-type", "application/json")],
        final_body,
    )
        .into_response())
}

/// Simple proxy without signing (for tokenize, metrics, models).
pub async fn proxy_simple(
    client: &reqwest::Client,
    url: &str,
    method: reqwest::Method,
    body: Option<&[u8]>,
    content_type: &str,
    timeout: Option<std::time::Duration>,
    tracing_ids: Option<&TracingIds>,
) -> Result<Response, AppError> {
    let mut builder = client.request(method, url);

    if let Some(body) = body {
        builder = builder
            .header("content-type", "application/json")
            .body(body.to_vec());
    }

    if let Some(timeout) = timeout {
        builder = builder.timeout(timeout);
    }
    builder = apply_tracing_headers(builder, tracing_ids);

    let upstream_start = std::time::Instant::now();
    let response = builder
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    metrics::histogram!("upstream_request_duration_seconds", "endpoint" => "simple")
        .record(upstream_start.elapsed().as_secs_f64());

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_else(|_| Bytes::from("{}"));
        let info = log_upstream_error(status, url, &body, tracing_ids);
        return Err(AppError::Upstream {
            status: effective_error_status(status.as_u16(), info.as_ref()),
            body,
        });
    }

    let response_bytes = response
        .bytes()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", content_type)
        .body(Body::from(response_bytes))
        .unwrap())
}

/// Sign already-fetched JSON response bytes, cache the signature, and return a JSON response.
/// Used by catch-all when content-type is already known to be JSON.
pub async fn sign_and_cache_json_response(
    response_bytes: &[u8],
    request_sha256: &str,
    mut opts: ProxyOpts,
    status: StatusCode,
) -> Result<Response, AppError> {
    // Parse JSON; if the backend sent content-type: application/json but the body
    // is empty or not valid JSON, wrap it in an empty object so we can still
    // generate an ID, sign, and cache.
    let mut response_data: serde_json::Value = match serde_json::from_slice(response_bytes) {
        Ok(data) => data,
        Err(e) => {
            debug!(error = %e, "Response body not valid JSON, wrapping in empty object");
            serde_json::json!({})
        }
    };

    let chat_id = match response_data.get("id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => {
            let id = format!(
                "{}-{}",
                opts.id_prefix,
                &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]
            );
            if let Some(obj) = response_data.as_object_mut() {
                obj.insert("id".to_string(), serde_json::Value::String(id.clone()));
            }
            debug!(id = %id, "Generated response ID");
            id
        }
    };

    // Report usage for cloud API key requests (before encryption, needs plaintext fields)
    try_report_usage(&response_data, &chat_id, &opts);

    // Apply response transform (e.g., encryption) before hashing/signing.
    // The signature covers the response bytes the client actually receives.
    if let Some(transform) = opts.response_transform.take() {
        transform(&mut response_data)?;
    }

    // Serialize with compact separators (matching Python's separators=(",",":"))
    let final_body =
        serde_json::to_string(&response_data).map_err(|e| AppError::Internal(e.into()))?;
    let response_sha256 = hex::encode(Sha256::digest(final_body.as_bytes()));

    // Sign and cache
    let text = format!("{}:{request_sha256}:{response_sha256}", opts.model_name);
    let signed = opts.signing.sign_chat(&text).map_err(|e| {
        error!(error = %e, "Signing failed");
        AppError::Internal(e)
    })?;
    let signed_json = serde_json::to_string(&signed).map_err(|e| AppError::Internal(e.into()))?;
    opts.cache.set_chat(&chat_id, &signed_json);

    Ok(Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(final_body))
        .unwrap())
}

/// Proxy an already-received streaming SSE response. Hashes all chunks, signs at end, caches.
/// Used by catch-all when content-type is already known to be SSE.
pub async fn proxy_streaming_response(
    response: reqwest::Response,
    request_sha256: &str,
    opts: ProxyOpts,
    status: StatusCode,
) -> Result<Response, AppError> {
    // Capture log fields before any partial moves from opts.
    let (log_request_id, log_org_id, log_workspace_id) = log_ids_or_empty(&opts.tracing_ids);
    let source_labels = request_metric_labels(&opts.tracing_ids);
    let stream_start = std::time::Instant::now();

    let signing = opts.signing.clone();
    let cache = opts.cache.clone();
    let usage_reporter = opts.usage_reporter.clone();
    let model_name = opts.model_name.clone();
    let chunk_transform = opts.chunk_transform;
    let backend_guard = opts.backend_guard;
    let stream_idle_timeout_secs = opts.stream_idle_timeout_secs;
    let request_sha256 = request_sha256.to_string();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    let byte_stream = response.bytes_stream();
    tokio::spawn(async move {
        use futures_util::StreamExt;

        let _guard = StreamingGuard::new();
        let _backend_guard = backend_guard;

        let mut byte_stream = std::pin::pin!(byte_stream);
        let mut hasher = Sha256::new();
        let mut parser = SseParser::new();
        let mut upstream_error = false;
        let mut downstream_closed = false;
        let mut incomplete_reason = None;
        let mut transformer = SseTransformer::new(chunk_transform);

        loop {
            tokio::select! {
                chunk = byte_stream.next() => {
                    match chunk {
                        Some(Ok(chunk)) => {
                            parser.process_chunk(&chunk);

                            // Normalize (and encrypt, if active) the chunk, then hash
                            // what the client actually receives for signatures.
                            let to_send = match transformer.process_chunk(&chunk) {
                                Ok(transformed) => transformed,
                                Err(e) => {
                                    error!(error = %e, "Stream transform failed");
                                    let _ = tx.send(Err(std::io::Error::other(
                                        "Stream transform failed",
                                    ))).await;
                                    upstream_error = true;
                                    incomplete_reason = Some("transform_error");
                                    break;
                                }
                            };

                            hasher.update(&to_send);

                            if tx.send(Ok(to_send)).await.is_err() {
                                downstream_closed = true;
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            error!(error = %e, "Error reading upstream stream");
                            upstream_error = true;
                            incomplete_reason = Some("upstream_read_error");
                            let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                            break;
                        }
                        None => break, // stream ended
                    }
                }
                _ = tx.closed() => {
                    info!("Client disconnected, aborting upstream stream processing");
                    downstream_closed = true;
                    break;
                }
                _ = tokio::time::sleep(std::time::Duration::from_secs(
                    stream_idle_timeout_secs,
                )), if stream_idle_timeout_secs > 0 => {
                    warn!(
                        request_id = %log_request_id,
                        org_id = %log_org_id,
                        workspace_id = %log_workspace_id,
                        model = %model_name.to_lowercase(),
                        timeout_secs = stream_idle_timeout_secs,
                        "Upstream SSE stream exceeded the idle timeout"
                    );
                    upstream_error = true;
                    incomplete_reason = Some("idle_timeout");
                    let _ = tx.send(Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "Upstream response stream timed out",
                    ))).await;
                    break;
                }
            }
        }

        parser.finish();

        // Flush any remaining buffered content in the transformer
        if !upstream_error && !downstream_closed {
            match transformer.flush() {
                Ok(flushed) if !flushed.is_empty() => {
                    hasher.update(&flushed);
                    if tx.send(Ok(flushed)).await.is_err() {
                        downstream_closed = true;
                    }
                }
                Err(e) => {
                    error!(error = %e, "Stream transform flush failed");
                    let _ = tx
                        .send(Err(std::io::Error::other("Stream transform failed")))
                        .await;
                    upstream_error = true;
                    incomplete_reason = Some("transform_error");
                }
                _ => {}
            }
        }

        if stream_idle_timeout_secs > 0
            && !upstream_error
            && !downstream_closed
            && !parser.seen_done
        {
            incomplete_reason = Some("missing_done");
            let _ = tx
                .send(Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Upstream response stream ended before [DONE]",
                )))
                .await;
        }

        let completed_cleanly = !upstream_error && !downstream_closed && parser.seen_done;
        if !completed_cleanly && !downstream_closed {
            let reason = incomplete_reason.unwrap_or("missing_done");
            metrics::counter!(
                "upstream_stream_incomplete_total",
                "reason" => reason,
                "mode" => "streaming_response"
            )
            .increment(1);
        }

        // Bill for tokens already produced even on an interrupted stream
        // (nearai/infra#98). Shared with proxy_streaming_request so both
        // streaming proxy paths have identical billing semantics. This path is
        // reachable from the authenticated catch-all SSE proxy. Signing/caching
        // stays gated on a clean [DONE] below.
        report_stream_usage_on_finalize(
            &usage_reporter,
            parser.usage,
            parser.chat_id.as_deref(),
            completed_cleanly,
            &log_request_id,
            &log_org_id,
            &log_workspace_id,
        );

        if completed_cleanly {
            let response_sha256 = hex::encode(hasher.finalize());
            if let Some(ref id) = parser.chat_id {
                let text = format!("{model_name}:{request_sha256}:{response_sha256}");
                match signing.sign_chat(&text) {
                    Ok(signed) => {
                        if let Ok(signed_json) = serde_json::to_string(&signed) {
                            cache.set_chat(id, &signed_json);
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Signing failed for streaming response");
                    }
                }

                // Structured completion log — one line per successful streaming response.
                let (input_tokens, output_tokens) = parser.usage.unwrap_or((0, 0));
                let total_ms = stream_start.elapsed().as_millis();
                record_completed_request_metrics(
                    source_labels,
                    input_tokens,
                    total_ms,
                    "streaming_response",
                );
                info!(
                    request_id = %log_request_id,
                    org_id = %log_org_id,
                    workspace_id = %log_workspace_id,
                    model = %model_name.to_lowercase(),
                    chat_id = %id,
                    input_tokens,
                    output_tokens,
                    total_duration_ms = total_ms,
                    auth_path = source_labels.auth_path,
                    ingress_route = source_labels.ingress_route,
                    tenant_context = source_labels.tenant_context,
                    request_id_origin = source_labels.request_id_origin,
                    "request completed"
                );
            } else {
                error!("Chat id could not be extracted from the completed streaming response");
            }
        } else {
            info!(
                upstream_error,
                downstream_closed,
                seen_done = parser.seen_done,
                "Skipping streaming signature cache: stream did not complete cleanly"
            );
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    Ok(Response::builder()
        .status(status)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(body)
        .unwrap())
}

/// Drop guard that tracks the streaming_connections gauge.
/// Increments on creation, decrements on drop — guarantees they stay paired.
pub(crate) struct StreamingGuard;

impl StreamingGuard {
    pub(crate) fn new() -> Self {
        metrics::gauge!("streaming_connections").increment(1);
        Self
    }
}

impl Drop for StreamingGuard {
    fn drop(&mut self) {
        metrics::gauge!("streaming_connections").decrement(1);
    }
}

/// Line-buffered SSE parser that handles data split across chunk boundaries.
/// Extracts `chat_id` from the first JSON chunk and detects the `[DONE]` marker.
pub struct SseParser {
    line_buffer: String,
    pub chat_id: Option<String>,
    pub seen_done: bool,
    /// Token usage extracted from the final SSE chunk (prompt_tokens, completion_tokens).
    pub usage: Option<(i64, i64)>,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            line_buffer: String::new(),
            chat_id: None,
            seen_done: false,
            usage: None,
        }
    }

    pub fn process_chunk(&mut self, chunk: &[u8]) {
        match std::str::from_utf8(chunk) {
            Ok(s) => self.line_buffer.push_str(s),
            Err(_) => self.line_buffer.push_str(&String::from_utf8_lossy(chunk)),
        }

        // Process all complete lines in the buffer.
        // We extract state changes from borrowed data first, then mutate,
        // to avoid allocating a String copy of each line.
        while let Some(newline_pos) = self.line_buffer.find('\n') {
            let line_end =
                if newline_pos > 0 && self.line_buffer.as_bytes()[newline_pos - 1] == b'\r' {
                    newline_pos - 1
                } else {
                    newline_pos
                };

            // Borrow the line from the buffer, extract what we need, then release the borrow
            let (is_done, extracted_id, extracted_usage) = {
                let line = &self.line_buffer[..line_end];
                let data = line
                    .strip_prefix("data: ")
                    .or_else(|| line.strip_prefix("data:"))
                    .unwrap_or(line)
                    .trim();

                if data.is_empty() {
                    (false, None, None)
                } else if data == "[DONE]" {
                    (true, None, None)
                } else if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                    let id = if self.chat_id.is_none() {
                        parsed
                            .get("id")
                            .and_then(|id| id.as_str().map(String::from))
                    } else {
                        None
                    };
                    // Capture usage from any chunk that has it (typically the final one)
                    let usage = parsed
                        .get("usage")
                        .filter(|u| u.is_object())
                        .and_then(|usage| {
                            let input = usage
                                .get("prompt_tokens")
                                .and_then(|v| v.as_i64())
                                .unwrap_or(0);
                            let output = usage
                                .get("completion_tokens")
                                .and_then(|v| v.as_i64())
                                .unwrap_or(0);
                            if input > 0 || output > 0 {
                                Some((input, output))
                            } else {
                                None
                            }
                        });
                    (false, id, usage)
                } else {
                    (false, None, None)
                }
            };

            if is_done {
                self.seen_done = true;
            }
            if let Some(id) = extracted_id {
                self.chat_id = Some(id);
            }
            if let Some(usage) = extracted_usage {
                self.usage = Some(usage);
            }

            // Remove the processed line in-place (no allocation, just memmove)
            self.line_buffer.drain(..newline_pos + 1);
        }
    }

    /// Dispatch a final unterminated SSE line at end-of-stream. SSE permits
    /// the last event line to omit its newline, including `data: [DONE]`.
    pub fn finish(&mut self) {
        if !self.line_buffer.is_empty() {
            self.process_chunk(b"\n");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{AuthPath, IngressRouteKind, RequestSource};
    use crate::cache::ChatCache;
    use crate::encryption::ChunkTransform;
    use crate::signing::{EcdsaContext, Ed25519Context, SigningPair};

    #[test]
    fn request_metric_labels_preserve_only_bounded_source_dimensions() {
        let tracing_ids = Some(TracingIds {
            request_id: uuid::Uuid::new_v4().to_string(),
            request_id_inbound: true,
            org_id: Some("not-emitted-as-a-label".to_string()),
            workspace_id: None,
            request_source: Some(RequestSource {
                auth_path: AuthPath::TrustedConfigToken,
                ingress_route: IngressRouteKind::LongIndexed,
            }),
            forward_tenant_headers: true,
        });

        let labels = request_metric_labels(&tracing_ids);
        assert_eq!(labels.auth_path, "trusted_config_token");
        assert_eq!(labels.ingress_route, "long_indexed");
        assert_eq!(labels.tenant_context, "trusted_headers");
        assert_eq!(labels.request_id_origin, "inbound");

        let direct_ids = Some(TracingIds {
            request_id: uuid::Uuid::new_v4().to_string(),
            request_id_inbound: false,
            org_id: Some("org-verified-but-not-a-metric-label".to_string()),
            workspace_id: Some("workspace-verified-but-not-a-metric-label".to_string()),
            request_source: Some(RequestSource {
                auth_path: AuthPath::CloudApiKey,
                ingress_route: IngressRouteKind::Canonical,
            }),
            forward_tenant_headers: false,
        });
        let direct_labels = request_metric_labels(&direct_ids);
        assert_eq!(direct_labels.auth_path, "cloud_api_key");
        assert_eq!(direct_labels.ingress_route, "canonical");
        assert_eq!(direct_labels.tenant_context, "verified");
        assert_eq!(direct_labels.request_id_origin, "generated");
    }

    fn reporter_with(
        usage_token: Option<&str>,
        org_id: Option<&str>,
        workspace_id: Option<&str>,
        api_key_id: Option<&str>,
    ) -> UsageReporter {
        UsageReporter {
            http_client: reqwest::Client::new(),
            cloud_api_url: "http://cloud-api.invalid".to_string(),
            model_name: "test-model".to_string(),
            cloud_api_usage_token: usage_token.map(String::from),
            org_id: org_id.map(String::from),
            workspace_id: workspace_id.map(String::from),
            api_key_id: api_key_id.map(String::from),
            request_id: Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
            request_source: RequestSource {
                auth_path: AuthPath::CloudApiKey,
                ingress_route: IngressRouteKind::Canonical,
            },
        }
    }

    #[test]
    fn test_can_use_service_token_path_requires_all_fields() {
        // Token + all three identity fields: green light.
        assert!(
            reporter_with(Some("tok"), Some("org"), Some("ws"), Some("key"),)
                .can_use_service_token_path()
        );

        // Token missing: reporting is skipped (no legacy fallback) even if
        // identity is fully populated.
        assert!(
            !reporter_with(None, Some("org"), Some("ws"), Some("key")).can_use_service_token_path()
        );

        // Any identity field missing: reporting is skipped. Mirrors the
        // "older cloud-api hasn't shipped /v1/check_api_key changes yet"
        // case where the auth response returned `None`s.
        for (org, ws, key) in [
            (None, Some("ws"), Some("key")),
            (Some("org"), None, Some("key")),
            (Some("org"), Some("ws"), None),
        ] {
            assert!(
                !reporter_with(Some("tok"), org, ws, key).can_use_service_token_path(),
                "missing identity field should disable service-token path: \
                 org={org:?} ws={ws:?} key={key:?}"
            );
        }

        // Nothing configured: reporting is skipped.
        assert!(!reporter_with(None, None, None, None).can_use_service_token_path());
    }

    #[test]
    fn usage_report_outcomes_are_bounded_and_status_classification_is_stable() {
        assert_eq!(
            classify_usage_http_status(reqwest::StatusCode::OK),
            UsageReportOutcome::Accepted
        );
        assert_eq!(
            classify_usage_http_status(reqwest::StatusCode::UNAUTHORIZED),
            UsageReportOutcome::Http4xx
        );
        assert_eq!(
            classify_usage_http_status(reqwest::StatusCode::SERVICE_UNAVAILABLE),
            UsageReportOutcome::Http5xx
        );
        assert_eq!(
            classify_usage_http_status(reqwest::StatusCode::TEMPORARY_REDIRECT),
            UsageReportOutcome::HttpOther
        );

        let labels = [
            UsageReportOutcome::Accepted,
            UsageReportOutcome::Http4xx,
            UsageReportOutcome::Http5xx,
            UsageReportOutcome::HttpOther,
            UsageReportOutcome::Timeout,
            UsageReportOutcome::ConnectError,
            UsageReportOutcome::TransportError,
            UsageReportOutcome::MissingUsageToken,
            UsageReportOutcome::MissingAuthIdentity,
            UsageReportOutcome::InvalidBody,
            UsageReportOutcome::MissingBillableUsage,
            UsageReportOutcome::MissingResponseId,
        ]
        .map(UsageReportOutcome::as_label);
        assert_eq!(labels.len(), 12);
        assert!(labels.iter().all(|label| !label.is_empty()));
    }

    #[test]
    fn test_build_usage_body_chat_completion() {
        let resp = serde_json::json!({
            "usage": {"prompt_tokens": 12, "completion_tokens": 7}
        });
        let body = build_usage_body(&UsageType::ChatCompletion, &resp, "m", "id-1").unwrap();
        assert_eq!(body["type"], "chat_completion");
        assert_eq!(body["model"], "m");
        assert_eq!(body["input_tokens"], 12);
        assert_eq!(body["output_tokens"], 7);
        assert_eq!(body["id"], "id-1");
    }

    #[test]
    fn test_build_usage_body_image_generation() {
        let resp = serde_json::json!({"data": [{}, {}, {}]});
        let body = build_usage_body(&UsageType::ImageGeneration, &resp, "m", "id-2").unwrap();
        assert_eq!(body["type"], "image_generation");
        assert_eq!(body["image_count"], 3);
        assert!(body.get("input_tokens").is_none());
    }

    #[test]
    fn test_build_usage_body_input_only_kinds_label_and_shape() {
        // embedding / rerank / score share the embeddings response shape
        // (top-level usage.prompt_tokens, no completion tokens) and report
        // input-only usage under their own label (nearai/infra#169).
        let resp = serde_json::json!({
            "usage": {"prompt_tokens": 42, "total_tokens": 42}
        });
        for (ty, label) in [
            (UsageType::Embedding, "embedding"),
            (UsageType::Rerank, "rerank"),
            (UsageType::Score, "score"),
        ] {
            let body = build_usage_body(&ty, &resp, "m", "id-3").unwrap();
            assert_eq!(body["type"], label, "wrong label for {label}");
            assert_eq!(body["input_tokens"], 42);
            // No output_tokens for input-only kinds — cloud-api's wire
            // variant carries input_tokens only.
            assert!(
                body.get("output_tokens").is_none(),
                "{label} must not report output_tokens"
            );
            assert_eq!(body["id"], "id-3");
        }
    }

    #[test]
    fn test_build_usage_body_returns_none_when_nothing_billable() {
        // No usage object at all.
        assert!(
            build_usage_body(&UsageType::Embedding, &serde_json::json!({}), "m", "x").is_none()
        );
        // Zero tokens.
        let zero = serde_json::json!({"usage": {"prompt_tokens": 0}});
        assert!(build_usage_body(&UsageType::Rerank, &zero, "m", "x").is_none());
        // Chat with both token counts zero.
        let zero_chat = serde_json::json!({"usage": {"prompt_tokens": 0, "completion_tokens": 0}});
        assert!(build_usage_body(&UsageType::ChatCompletion, &zero_chat, "m", "x").is_none());
        // Empty image data array.
        let no_images = serde_json::json!({"data": []});
        assert!(build_usage_body(&UsageType::ImageGeneration, &no_images, "m", "x").is_none());
    }

    /// Build a ProxyOpts with fixed signing keys for deterministic tests.
    fn test_proxy_opts() -> ProxyOpts {
        let ecdsa_key: [u8; 32] = [
            0xac, 0x09, 0x74, 0xbe, 0xc3, 0x9a, 0x17, 0xe3, 0x6b, 0xa4, 0xa6, 0xb4, 0xd2, 0x38,
            0xff, 0x94, 0x4b, 0xac, 0xb3, 0x5e, 0x5d, 0xc4, 0xaf, 0x0f, 0x33, 0x47, 0xe5, 0x87,
            0x31, 0x79, 0x67, 0x0f,
        ];
        let ed25519_key: [u8; 32] = [
            0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec,
            0x2c, 0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03,
            0x1c, 0xae, 0x7f, 0x60,
        ];
        let ecdsa = EcdsaContext::from_key_bytes(&ecdsa_key).unwrap();
        let ed25519 = Ed25519Context::from_key_bytes(&ed25519_key).unwrap();
        let signing = Arc::new(SigningPair { ecdsa, ed25519 });
        let cache = Arc::new(ChatCache::new("test-model", 1200));
        ProxyOpts {
            signing,
            cache,
            id_prefix: "test".to_string(),
            model_name: "test-model".to_string(),
            usage_reporter: None,
            usage_type: UsageType::default(),
            request_hash: None,
            response_transform: None,
            chunk_transform: None,
            backend_guard: None,
            stream_idle_timeout_secs: 0,
            response_shape: ResponseShape::default(),
            tracing_ids: None,
        }
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_empty_body() {
        let opts = test_proxy_opts();
        let request_sha256 = hex::encode(Sha256::digest(b"test-request"));

        let result = sign_and_cache_json_response(b"", &request_sha256, opts, StatusCode::OK).await;

        let resp = result.expect("empty body should not return error");
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Should have a generated ID starting with the prefix
        let id = parsed["id"].as_str().unwrap();
        assert!(id.starts_with("test-"), "id should start with prefix: {id}");
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_invalid_json_body() {
        let opts = test_proxy_opts();
        let request_sha256 = hex::encode(Sha256::digest(b"test-request"));

        let result = sign_and_cache_json_response(
            b"this is not json",
            &request_sha256,
            opts,
            StatusCode::OK,
        )
        .await;

        let resp = result.expect("invalid JSON should not return error");
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let id = parsed["id"].as_str().unwrap();
        assert!(id.starts_with("test-"), "id should start with prefix: {id}");
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_valid_body_with_id() {
        let opts = test_proxy_opts();
        let cache = opts.cache.clone();
        let request_sha256 = hex::encode(Sha256::digest(b"test-request"));
        let body = br#"{"id":"existing-id","text":"hello"}"#;

        let result =
            sign_and_cache_json_response(body, &request_sha256, opts, StatusCode::OK).await;

        let resp = result.expect("valid JSON should succeed");
        assert_eq!(resp.status(), StatusCode::OK);

        let resp_body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
        assert_eq!(parsed["id"], "existing-id");

        // Signature should be cached under "existing-id"
        assert!(cache.get_chat("existing-id").is_some());
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_valid_body_without_id() {
        let opts = test_proxy_opts();
        let cache = opts.cache.clone();
        let request_sha256 = hex::encode(Sha256::digest(b"test-request"));
        let body = br#"{"text":"hello"}"#;

        let result =
            sign_and_cache_json_response(body, &request_sha256, opts, StatusCode::OK).await;

        let resp = result.expect("valid JSON without id should succeed");
        let resp_body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
        let id = parsed["id"].as_str().unwrap();
        assert!(id.starts_with("test-"), "should generate id with prefix");

        // Signature should be cached under the generated id
        assert!(cache.get_chat(id).is_some());
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_preserves_status_code() {
        let opts = test_proxy_opts();
        let request_sha256 = hex::encode(Sha256::digest(b"test"));
        let body = br#"{"id":"s1"}"#;

        let resp = sign_and_cache_json_response(body, &request_sha256, opts, StatusCode::CREATED)
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    /// Verifies that the streaming task exits promptly when the client disconnects
    /// (i.e. the response body receiver is dropped) even if the upstream is still
    /// producing data. Without the `tx.closed()` branch in `select!`, the task
    /// would block on `byte_stream.next().await` indefinitely.
    #[tokio::test]
    async fn test_streaming_task_cancels_on_client_disconnect() {
        use std::time::Duration;

        // Simulate a slow upstream that hasn't sent anything yet.
        // We keep _upstream_tx alive so the upstream channel stays open — this means
        // byte_stream.next() will block forever waiting for data, which is exactly
        // the scenario that tx.closed() needs to rescue us from.
        let (_upstream_tx, upstream_rx) =
            tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(1);

        // This is the downstream channel (proxy -> client)
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);

        let handle = tokio::spawn(async move {
            use futures_util::StreamExt;
            let byte_stream = tokio_stream::wrappers::ReceiverStream::new(upstream_rx);
            let mut byte_stream = std::pin::pin!(byte_stream);

            loop {
                tokio::select! {
                    chunk = byte_stream.next() => {
                        match chunk {
                            Some(Ok(data)) => {
                                if tx.send(Ok(data)).await.is_err() {
                                    break;
                                }
                            }
                            _ => break,
                        }
                    }
                    _ = tx.closed() => {
                        // Client disconnected — exit immediately
                        break;
                    }
                }
            }
        });

        // Drop the receiver to simulate client disconnect
        drop(rx);

        // The task should exit promptly thanks to tx.closed().
        // Without the tx.closed() branch in select!, the task would block forever
        // on byte_stream.next() since _upstream_tx is alive and never sends data.
        let result = tokio::time::timeout(Duration::from_millis(100), handle).await;
        assert!(
            result.is_ok(),
            "Streaming task should exit promptly on client disconnect"
        );
    }

    async fn raw_sse_response(body: Option<&'static str>, hold_open: bool) -> reqwest::Response {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 2048];
            let _ = socket.read(&mut request).await;
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n",
                )
                .await
                .unwrap();
            if let Some(body) = body {
                let encoded = format!("{:X}\r\n{}\r\n0\r\n\r\n", body.len(), body);
                socket.write_all(encoded.as_bytes()).await.unwrap();
            }
            if hold_open {
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
        });

        reqwest::Client::new()
            .get(format!("http://{address}"))
            .send()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_stream_idle_watchdog_fails_downstream_body() {
        let upstream = raw_sse_response(None, true).await;
        let mut opts = test_proxy_opts();
        opts.stream_idle_timeout_secs = 1;
        let response = proxy_streaming_response(upstream, "request-sha256", opts, StatusCode::OK)
            .await
            .unwrap();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            axum::body::to_bytes(response.into_body(), 1024 * 1024),
        )
        .await
        .expect("idle watchdog should fire before the test timeout");
        assert!(
            result.is_err(),
            "stalled stream must fail the response body"
        );
    }

    #[tokio::test]
    async fn test_missing_done_fails_downstream_body_when_watchdog_enabled() {
        let upstream =
            raw_sse_response(Some("data: {\"id\":\"chat-1\",\"choices\":[]}\n\n"), false).await;
        let mut opts = test_proxy_opts();
        opts.stream_idle_timeout_secs = 1;
        let response = proxy_streaming_response(upstream, "request-sha256", opts, StatusCode::OK)
            .await
            .unwrap();

        let result = axum::body::to_bytes(response.into_body(), 1024 * 1024).await;
        assert!(
            result.is_err(),
            "stream without [DONE] must fail the response body"
        );
    }

    #[tokio::test]
    async fn test_done_without_trailing_newline_completes_cleanly() {
        let upstream = raw_sse_response(
            Some("data: {\"id\":\"chat-1\",\"choices\":[]}\n\ndata: [DONE]"),
            false,
        )
        .await;
        let mut opts = test_proxy_opts();
        opts.stream_idle_timeout_secs = 1;
        let response = proxy_streaming_response(upstream, "request-sha256", opts, StatusCode::OK)
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .expect("valid final [DONE] line must not fail the response body");
        assert!(body.ends_with(b"data: [DONE]"));
    }

    #[test]
    fn test_sse_parser_normal_sse() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"id\":\"chat-1\",\"content\":\"hi\"}\n\ndata: [DONE]\n\n");
        assert_eq!(parser.chat_id.as_deref(), Some("chat-1"));
        assert!(parser.seen_done);
    }

    #[test]
    fn test_sse_parser_done_split_across_chunks() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"id\":\"chat-2\"}\n\ndata: [DO");
        assert_eq!(parser.chat_id.as_deref(), Some("chat-2"));
        assert!(!parser.seen_done);

        parser.process_chunk(b"NE]\n\n");
        assert!(parser.seen_done);
    }

    #[test]
    fn test_sse_parser_id_split_across_chunks() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"id\":\"cha");
        assert!(parser.chat_id.is_none());

        parser.process_chunk(b"t-3\",\"choices\":[]}\n\n");
        assert_eq!(parser.chat_id.as_deref(), Some("chat-3"));
    }

    #[test]
    fn test_sse_parser_no_space_after_data_colon() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data:{\"id\":\"chat-4\"}\n\ndata:[DONE]\n\n");
        assert_eq!(parser.chat_id.as_deref(), Some("chat-4"));
        assert!(parser.seen_done);
    }

    #[test]
    fn test_sse_parser_crlf_line_endings() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"id\":\"chat-5\"}\r\n\r\ndata: [DONE]\r\n\r\n");
        assert_eq!(parser.chat_id.as_deref(), Some("chat-5"));
        assert!(parser.seen_done);
    }

    #[test]
    fn test_sse_parser_no_done_marker() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"id\":\"chat-6\"}\n\n");
        assert_eq!(parser.chat_id.as_deref(), Some("chat-6"));
        assert!(!parser.seen_done);
    }

    #[test]
    fn test_sse_parser_done_without_trailing_newline() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: [DONE]");
        assert!(!parser.seen_done);

        parser.finish();
        assert!(parser.seen_done);
    }

    #[test]
    fn test_sse_parser_multiple_json_chunks() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"id\":\"chat-7\",\"delta\":\"a\"}\n\n");
        parser.process_chunk(b"data: {\"id\":\"chat-7\",\"delta\":\"b\"}\n\n");
        parser.process_chunk(b"data: [DONE]\n\n");
        // Should use the first id
        assert_eq!(parser.chat_id.as_deref(), Some("chat-7"));
        assert!(parser.seen_done);
    }

    // ── Fix 1: SseTransformer line buffering and fail-closed tests ──

    #[test]
    fn test_sse_transformer_data_split_across_chunks() {
        // Simulate a "data: {...}\n" line split across two TCP chunks
        let transform: ChunkTransform = Arc::new(|v| {
            if let Some(s) = v
                .get_mut("text")
                .and_then(|t| t.as_str().map(|s| s.to_string()))
            {
                v["text"] = serde_json::Value::String(format!("ENC:{s}"));
            }
            Ok(())
        });

        let mut transformer = SseTransformer::new(Some(transform));

        // First chunk: incomplete line
        let out1 = transformer.process_chunk(b"data: {\"text\":\"hel").unwrap();
        assert_eq!(out1.as_ref(), b""); // No complete line yet

        // Second chunk: completes the line
        let out2 = transformer.process_chunk(b"lo\"}\n").unwrap();
        let out_str = std::str::from_utf8(&out2).unwrap();
        assert!(out_str.contains("\"ENC:hello\""), "Got: {out_str}");
    }

    #[test]
    fn test_sse_transformer_multiple_lines_in_one_chunk() {
        let transform: ChunkTransform = Arc::new(|v| {
            if let Some(s) = v
                .get_mut("x")
                .and_then(|t| t.as_str().map(|s| s.to_string()))
            {
                v["x"] = serde_json::Value::String(format!("T:{s}"));
            }
            Ok(())
        });

        let mut transformer = SseTransformer::new(Some(transform));
        let chunk = b"data: {\"x\":\"a\"}\ndata: {\"x\":\"b\"}\n\n";
        let out = transformer.process_chunk(chunk).unwrap();
        let out_str = std::str::from_utf8(&out).unwrap();
        assert!(out_str.contains("\"T:a\""), "Got: {out_str}");
        assert!(out_str.contains("\"T:b\""), "Got: {out_str}");
    }

    #[test]
    fn test_sse_transformer_fail_closed_on_bad_json() {
        let transform: ChunkTransform = Arc::new(|_| Ok(()));

        let mut transformer = SseTransformer::new(Some(transform));
        // Invalid JSON in data line
        let result = transformer.process_chunk(b"data: {not json}\n");
        assert!(result.is_err(), "Should fail-closed on bad JSON");
    }

    #[test]
    fn test_sse_transformer_fail_closed_on_transform_error() {
        let transform: ChunkTransform =
            Arc::new(|_| Err(AppError::Internal(anyhow::anyhow!("transform failed"))));

        let mut transformer = SseTransformer::new(Some(transform));
        let result = transformer.process_chunk(b"data: {\"x\":1}\n");
        assert!(result.is_err(), "Should fail-closed on transform error");
    }

    #[test]
    fn test_sse_transformer_passes_through_done_and_empty_lines() {
        let transform: ChunkTransform = Arc::new(|_| Ok(()));

        let mut transformer = SseTransformer::new(Some(transform));
        let chunk = b"data: [DONE]\n\n";
        let out = transformer.process_chunk(chunk).unwrap();
        let out_str = std::str::from_utf8(&out).unwrap();
        assert!(out_str.contains("[DONE]"));
    }

    #[test]
    fn test_sse_transformer_flush_incomplete_line() {
        // Simulate a data line that arrives without a trailing newline (stream ends mid-line).
        let transform: ChunkTransform = Arc::new(|val| {
            // Uppercase the "text" field to prove the transform ran
            if let Some(t) = val
                .get_mut("text")
                .and_then(|v| v.as_str().map(|s| s.to_uppercase()))
            {
                val["text"] = serde_json::Value::String(t);
            }
            Ok(())
        });

        let mut transformer = SseTransformer::new(Some(transform));

        // Send a partial chunk with no trailing newline
        let chunk = b"data: {\"text\":\"hello\"}";
        let out = transformer.process_chunk(chunk).unwrap();
        // Should be buffered, nothing emitted yet
        assert!(
            out.is_empty(),
            "Expected buffered, got: {:?}",
            std::str::from_utf8(&out)
        );

        // Flush should emit the transformed line
        let flushed = transformer.flush().unwrap();
        let flushed_str = std::str::from_utf8(&flushed).unwrap();
        assert!(
            flushed_str.contains("HELLO"),
            "Expected transformed text, got: {flushed_str}"
        );

        // Second flush should be empty
        let flushed2 = transformer.flush().unwrap();
        assert!(flushed2.is_empty());
    }

    #[test]
    fn test_sse_transformer_flush_empty_buffer() {
        let transform: ChunkTransform = Arc::new(|_| Ok(()));

        let mut transformer = SseTransformer::new(Some(transform));
        let flushed = transformer.flush().unwrap();
        assert!(flushed.is_empty());
    }

    #[test]
    fn test_sse_transformer_flush_done_marker() {
        // A [DONE] marker buffered without trailing newline should pass through unchanged.
        let transform: ChunkTransform = Arc::new(|_| Ok(()));

        let mut transformer = SseTransformer::new(Some(transform));
        let chunk = b"data: [DONE]";
        let out = transformer.process_chunk(chunk).unwrap();
        assert!(out.is_empty());

        let flushed = transformer.flush().unwrap();
        let flushed_str = std::str::from_utf8(&flushed).unwrap();
        assert!(flushed_str.contains("[DONE]"));
    }

    // ── normalize_chat_chunk + SseTransformer normalization tests ──

    #[test]
    fn test_normalize_renames_delta_reasoning() {
        let mut v = serde_json::json!({
            "choices": [{"index": 0, "delta": {"reasoning": "Thinking..."}}]
        });
        normalize_chat_chunk(&mut v);
        assert_eq!(
            v["choices"][0]["delta"]["reasoning_content"],
            serde_json::Value::String("Thinking...".into())
        );
        assert!(v["choices"][0]["delta"].get("reasoning").is_none());
    }

    #[test]
    fn test_normalize_keeps_reasoning_content_when_both_present() {
        let mut v = serde_json::json!({
            "choices": [{"index": 0, "delta": {
                "reasoning": "raw",
                "reasoning_content": "canonical"
            }}]
        });
        normalize_chat_chunk(&mut v);
        // reasoning_content stays canonical; reasoning is left as-is (we only
        // rename when reasoning_content is absent, to avoid clobbering).
        assert_eq!(
            v["choices"][0]["delta"]["reasoning_content"],
            serde_json::Value::String("canonical".into())
        );
    }

    #[test]
    fn test_normalize_passthrough_when_no_reasoning() {
        let mut v = serde_json::json!({
            "choices": [{"index": 0, "delta": {"content": "hi"}}]
        });
        let before = v.clone();
        normalize_chat_chunk(&mut v);
        assert_eq!(v, before);
    }

    #[test]
    fn test_normalize_handles_missing_choices_or_delta() {
        // No choices array (e.g. usage-only final chunk).
        let mut v = serde_json::json!({"usage": {"prompt_tokens": 1}});
        let before = v.clone();
        normalize_chat_chunk(&mut v);
        assert_eq!(v, before);

        // Choices with no delta (text completion shape).
        let mut v = serde_json::json!({"choices": [{"text": "hello"}]});
        let before = v.clone();
        normalize_chat_chunk(&mut v);
        assert_eq!(v, before);
    }

    #[test]
    fn test_sse_transformer_normalizes_without_extra_transform() {
        let mut transformer = SseTransformer::new(None);
        let chunk = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"think\"}}]}\n";
        let out = transformer.process_chunk(chunk).unwrap();
        let out_str = std::str::from_utf8(&out).unwrap();
        assert!(
            out_str.contains("\"reasoning_content\":\"think\""),
            "Got: {out_str}"
        );
        assert!(
            !out_str.contains("\"reasoning\":\"think\""),
            "Got: {out_str}"
        );
    }

    #[test]
    fn test_sse_transformer_normalize_then_extra_transform() {
        // Extra transform runs after normalization — it should see reasoning_content.
        let extra: ChunkTransform = Arc::new(|v| {
            let saw_reasoning_content = v["choices"][0]["delta"].get("reasoning_content").is_some();
            v["saw_reasoning_content"] = serde_json::Value::Bool(saw_reasoning_content);
            Ok(())
        });
        let mut transformer = SseTransformer::new(Some(extra));
        let chunk = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"x\"}}]}\n";
        let out = transformer.process_chunk(chunk).unwrap();
        let out_str = std::str::from_utf8(&out).unwrap();
        assert!(
            out_str.contains("\"saw_reasoning_content\":true"),
            "Got: {out_str}"
        );
        assert!(
            out_str.contains("\"reasoning_content\":\"x\""),
            "Got: {out_str}"
        );
    }

    #[test]
    fn test_assembler_accumulates_delta_reasoning_fallback() {
        // Upstream emits `delta.reasoning` (vLLM qwen3 parser); assembler should
        // still surface it as `reasoning_content` in the assembled message.
        let mut asm = StreamingResponseAssembler::new(ResponseShape::ChatCompletion);
        asm.process_chunk(
            b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"Step 1\"}}]}\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\" Step 2\"}}]}\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n",
        );
        asm.process_chunk(b"data: [DONE]\n");

        let resp = asm.into_response("chatcmpl");
        let msg = &resp["choices"][0]["message"];
        assert_eq!(msg["content"], serde_json::Value::String("hi".into()));
        assert_eq!(
            msg["reasoning_content"],
            serde_json::Value::String("Step 1 Step 2".into())
        );
    }

    #[test]
    fn test_sanitize_pydantic_v2_unexpected_format_no_brackets() {
        // Edge case: input_value= appears without bracket structure
        let message = "  input_value={'role': 'user', 'content': 'secret'}, input_type=dict";
        let result = sanitize_validation_errors(message);
        assert!(!result.contains("secret"), "leaked user content: {result}");
        assert!(result.contains("(validation error)"));
    }

    #[test]
    fn test_sanitize_mixed_python_dict_and_pydantic_v2() {
        // Mixed format (both SGLang and vLLM style in one message)
        let message = concat!(
            "2 errors:\n",
            "  {'type': 'value_error', 'msg': 'bad role', 'input': 'secret data'}\n",
            "  Input should be 'text' [type=literal_error, input_value='secret', input_type=str]"
        );
        let result = sanitize_validation_errors(message);
        assert!(!result.contains("secret"), "leaked data: {result}");
        assert!(result.contains("value_error: bad role"));
        assert!(result.contains("Input should be 'text' [type=literal_error]"));
    }

    // ── StreamingResponseAssembler tests ──

    #[test]
    fn test_assembler_basic_content() {
        let mut asm = StreamingResponseAssembler::new(ResponseShape::ChatCompletion);
        asm.process_chunk(
            b"data: {\"id\":\"c1\",\"model\":\"m\",\"created\":100,\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello \"},\"finish_reason\":null}]}\n\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}]}\n\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"c1\",\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\ndata: [DONE]\n\n",
        );

        let resp = asm.into_response("chatcmpl");
        assert_eq!(resp["id"], "c1");
        assert_eq!(resp["object"], "chat.completion");
        assert_eq!(resp["model"], "m");
        assert_eq!(resp["created"], 100);
        assert_eq!(resp["choices"][0]["message"]["content"], "hello world");
        assert_eq!(resp["choices"][0]["message"]["role"], "assistant");
        assert_eq!(resp["choices"][0]["finish_reason"], "stop");
        assert_eq!(resp["usage"]["prompt_tokens"], 5);
        assert_eq!(resp["usage"]["completion_tokens"], 2);
    }

    #[test]
    fn test_assembler_preserves_hidden_states_and_unknown_fields() {
        // sglang `return_hidden_states` streams activations in a dedicated chunk
        // as `choices[].delta.hidden_states`. Because the proxy forces upstream
        // streaming and reassembles, the assembler must keep them (hoisted to
        // choice level, matching native non-streaming `choices[].hidden_states`),
        // plus any other unknown provider fields (top-level + per-choice).
        let mut asm = StreamingResponseAssembler::new(ResponseShape::ChatCompletion);
        asm.process_chunk(
            b"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"}}],\"sglext\":{\"spec_verify_ct\":3}}\n\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"hidden_states\":[[0.1,0.2]]},\"finish_reason\":\"stop\",\"matched_stop\":2}]}\n\n",
        );
        asm.process_chunk(b"data: [DONE]\n\n");

        let resp = asm.into_response("chatcmpl");

        // Normal content still assembles, known fields are not clobbered.
        assert_eq!(resp["choices"][0]["message"]["content"], "hi");
        assert_eq!(resp["choices"][0]["finish_reason"], "stop");
        // hidden_states preserved at CHOICE level (sibling of message), not inside it.
        assert_eq!(resp["choices"][0]["hidden_states"][0][1], 0.2);
        assert!(resp["choices"][0]["message"].get("hidden_states").is_none());
        // Other unknown fields preserved: per-choice `matched_stop` + top-level `sglext`.
        assert_eq!(resp["choices"][0]["matched_stop"], 2);
        assert_eq!(resp["sglext"]["spec_verify_ct"], 3);
    }

    #[test]
    fn test_assembler_reasoning_content() {
        let mut asm = StreamingResponseAssembler::new(ResponseShape::ChatCompletion);
        asm.process_chunk(
            b"data: {\"id\":\"r1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":null}}]}\n\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"r1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"r1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"ing\"},\"finish_reason\":\"stop\"}]}\n\n",
        );
        asm.process_chunk(b"data: [DONE]\n\n");

        let resp = asm.into_response("chatcmpl");
        assert_eq!(
            resp["choices"][0]["message"]["reasoning_content"],
            "thinking"
        );
        // content should be null since it was empty
        assert!(resp["choices"][0]["message"]["content"].is_null());
    }

    #[test]
    fn test_assembler_tool_calls() {
        let mut asm = StreamingResponseAssembler::new(ResponseShape::ChatCompletion);
        // First tool call chunk: id + name
        asm.process_chunk(
            b"data: {\"id\":\"t1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n",
        );
        // Arguments chunks
        asm.process_chunk(
            b"data: {\"id\":\"t1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\"\"}}]}}]}\n\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"t1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\": \\\"NYC\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        );
        asm.process_chunk(b"data: [DONE]\n\n");

        let resp = asm.into_response("chatcmpl");
        let tc = &resp["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["id"], "call_1");
        assert_eq!(tc["function"]["name"], "get_weather");
        assert_eq!(tc["function"]["arguments"], "{\"city\": \"NYC\"}");
    }

    #[test]
    fn test_assembler_generates_id_when_missing() {
        let mut asm = StreamingResponseAssembler::new(ResponseShape::ChatCompletion);
        asm.process_chunk(
            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        );

        let resp = asm.into_response("chatcmpl");
        let id = resp["id"].as_str().unwrap();
        assert!(id.starts_with("chatcmpl-"), "should generate id: {id}");
    }

    #[test]
    fn test_assembler_split_across_chunks() {
        let mut asm = StreamingResponseAssembler::new(ResponseShape::ChatCompletion);
        // Split a single SSE line across two TCP chunks
        asm.process_chunk(b"data: {\"id\":\"s1\",\"choices\":[{\"inde");
        asm.process_chunk(
            b"x\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        );

        let resp = asm.into_response("chatcmpl");
        assert_eq!(resp["id"], "s1");
        assert_eq!(resp["choices"][0]["message"]["content"], "ok");
    }

    #[test]
    fn test_assembler_text_completion_shape() {
        // vLLM/SGLang `/v1/completions` SSE chunks carry incremental tokens at
        // `choices[].text` (no `delta` wrapper). The assembler must concatenate
        // them and emit `object: text_completion` with `choices[].text`.
        let mut asm = StreamingResponseAssembler::new(ResponseShape::TextCompletion);
        asm.process_chunk(
            b"data: {\"id\":\"cmpl-1\",\"object\":\"text_completion\",\"model\":\"m\",\"created\":100,\"choices\":[{\"index\":0,\"text\":\"The \",\"finish_reason\":null,\"logprobs\":null}]}\n\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"text\":\"capital \",\"finish_reason\":null,\"logprobs\":null}]}\n\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"text\":\"of France\",\"finish_reason\":\"length\",\"logprobs\":null}]}\n\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"cmpl-1\",\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\ndata: [DONE]\n\n",
        );

        let resp = asm.into_response("cmpl");
        assert_eq!(resp["id"], "cmpl-1");
        assert_eq!(resp["object"], "text_completion");
        assert_eq!(resp["model"], "m");
        assert_eq!(resp["created"], 100);
        assert_eq!(resp["choices"][0]["text"], "The capital of France");
        assert_eq!(resp["choices"][0]["finish_reason"], "length");
        assert_eq!(resp["choices"][0]["index"], 0);
        // No chat-shape fields.
        assert!(resp["choices"][0].get("message").is_none());
        assert_eq!(resp["usage"]["completion_tokens"], 3);
    }

    #[test]
    fn test_assembler_text_completion_empty_emits_empty_text() {
        // Edge case: backend emits only a finish_reason chunk with no text
        // (e.g. when max_tokens forces termination at the prompt boundary).
        // Assembler must still emit a valid `text` field rather than null.
        let mut asm = StreamingResponseAssembler::new(ResponseShape::TextCompletion);
        asm.process_chunk(
            b"data: {\"id\":\"cmpl-2\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n",
        );

        let resp = asm.into_response("cmpl");
        assert_eq!(resp["object"], "text_completion");
        assert_eq!(resp["choices"][0]["text"], "");
        assert!(!resp["choices"][0]["text"].is_null());
    }

    #[test]
    fn test_assembler_text_completion_generates_id_with_cmpl_prefix() {
        let mut asm = StreamingResponseAssembler::new(ResponseShape::TextCompletion);
        asm.process_chunk(
            b"data: {\"choices\":[{\"index\":0,\"text\":\"hi\",\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        );

        let resp = asm.into_response("cmpl");
        let id = resp["id"].as_str().unwrap();
        assert!(id.starts_with("cmpl-"), "should generate id: {id}");
    }

    #[test]
    fn test_assembler_captures_sglang_queue_full_abort() {
        // SGLang's `--max-queued-requests` abort emits an error data chunk
        // mid-stream, then continues with an empty-choices/zero-usage chunk
        // and `[DONE]`. The assembler must capture the error so the caller
        // can surface it as a real upstream error instead of returning HTTP
        // 200 with empty choices.
        let mut asm = StreamingResponseAssembler::new(ResponseShape::ChatCompletion);
        asm.process_chunk(
            b"data: {\"error\":{\"object\":\"error\",\"message\":\"The request queue is full.\",\"type\":\"SERVICE_UNAVAILABLE\",\"code\":503}}\n\n",
        );
        asm.process_chunk(
            b"data: {\"id\":\"chatcmpl-abc\",\"object\":\"chat.completion.chunk\",\"model\":\"glm-5.1\",\"created\":1779313724,\"choices\":[],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}\n\n",
        );
        asm.process_chunk(b"data: [DONE]\n\n");

        let err = asm.take_error().expect("error chunk must be captured");
        assert_eq!(err["code"], 503);
        assert_eq!(err["type"], "SERVICE_UNAVAILABLE");
        assert_eq!(err["message"], "The request queue is full.");
        // Subsequent take returns None.
        assert!(asm.take_error().is_none());
    }

    #[test]
    fn test_assembler_ignores_non_object_error_field() {
        // A `null` or string `error` field on a chunk must not be mistaken
        // for an upstream abort.
        let mut asm = StreamingResponseAssembler::new(ResponseShape::ChatCompletion);
        asm.process_chunk(
            b"data: {\"id\":\"c1\",\"error\":null,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        );
        assert!(asm.take_error().is_none());
        let resp = asm.into_response("chatcmpl");
        assert_eq!(resp["choices"][0]["message"]["content"], "hi");
    }

    #[test]
    fn test_assembler_keeps_first_error_chunk() {
        // If the stream emits multiple error chunks (defensive — not observed
        // in practice), keep the first so we surface the original failure.
        let mut asm = StreamingResponseAssembler::new(ResponseShape::ChatCompletion);
        asm.process_chunk(
            b"data: {\"error\":{\"object\":\"error\",\"message\":\"first\",\"type\":\"SERVICE_UNAVAILABLE\",\"code\":503}}\n\n",
        );
        asm.process_chunk(
            b"data: {\"error\":{\"object\":\"error\",\"message\":\"second\",\"type\":\"INTERNAL\",\"code\":500}}\n\n",
        );
        let err = asm.take_error().unwrap();
        assert_eq!(err["message"], "first");
    }

    #[test]
    fn test_inject_streaming() {
        let body = br#"{"messages":[{"role":"user","content":"hi"}]}"#;
        let result = inject_streaming(body).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&result).unwrap();
        assert_eq!(json["stream"], true);
        assert_eq!(json["stream_options"]["include_usage"], true);
        assert_eq!(json["messages"][0]["content"], "hi");
    }

    #[test]
    fn test_inject_streaming_preserves_existing_fields() {
        let body = br#"{"messages":[],"max_tokens":100,"temperature":0.7}"#;
        let result = inject_streaming(body).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&result).unwrap();
        assert_eq!(json["stream"], true);
        assert_eq!(json["max_tokens"], 100);
        assert_eq!(json["temperature"], 0.7);
    }
}
