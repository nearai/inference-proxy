use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use tokio::net::UnixStream;
use tracing::warn;

use crate::backend_pool::BackendGuard;
use crate::AppState;

/// GET / → {}
pub async fn root() -> impl IntoResponse {
    Json(serde_json::json!({}))
}

/// GET /version → {"version": "...", "type": "proxy"}
pub async fn version(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "version": state.config.git_rev,
        "type": "proxy",
    }))
}

// ── /healthz ────────────────────────────────────────────────────────

/// Per-check budget. Both checks run concurrently, so the total handler
/// latency is `max(DSTACK, BACKEND)` plus parsing overhead.
///
/// `BACKEND_PROBE_TIMEOUT` targets `/health`, a lightweight FastAPI route
/// that bypasses the inference event loop on both vLLM and SGLang; typical
/// responses are sub-50ms. However, under sustained KV-cache pressure SGLang
/// can starve the Python GIL, causing even this lightweight route to miss the
/// previous 1200ms cap (observed on GLM-5 hosts, May 2026). 3000ms gives the
/// backend enough headroom to recover from transient GIL saturation without
/// prematurely marking a live host as unhealthy.
const DSTACK_PROBE_TIMEOUT: Duration = Duration::from_millis(300);
const BACKEND_PROBE_TIMEOUT: Duration = Duration::from_millis(3000);

/// Path used to probe the inference backend. `/health` is a lightweight
/// FastAPI/uvicorn route on both vLLM and SGLang that does not go through
/// the engine's request scheduler — unlike `/v1/models`, which serializes
/// against the OpenAI-compatible request loop and can stall for >1s while
/// the engine is mid-prefill on a large request. The original `/v1/models`
/// probe path produced spurious 503s under heavy load (PR #106 follow-up,
/// reported via Datadog `service:nginx "GET /healthz HTTP/1.1" 503` on the
/// GLM-5.1 hosts when sglang was busy).
///
/// Also used by `backend_pool::spawn_health_check` for the per-backend
/// liveness probe in multi-backend deployments (e.g. Qwen3.5-122B,
/// gpt-oss). Keeping both probes on the same path means a wedged backend
/// is dropped from both the inference-proxy's internal pool and the
/// upstream model-proxy's pool consistently.
pub const BACKEND_HEALTH_PATH: &str = "/health";

/// Stable diagnostic codes returned to unauthenticated callers. Detailed
/// errors (paths, URLs, OS errno text) are logged server-side via tracing
/// so operators can still debug; the response body avoids leaking internal
/// topology to anyone who can hit `/healthz`.
const STATUS_OK: &str = "ok";
const DSTACK_UNREACHABLE: &str = "unreachable";
const DSTACK_TIMEOUT: &str = "timeout";
const BACKEND_TIMEOUT: &str = "timeout";
const BACKEND_UNREACHABLE: &str = "unreachable";

/// GET /healthz — readiness probe for upstream load balancers.
///
/// Probes critical subsystems whose failure would otherwise be invisible to a
/// plain `/v1/models` health check, leading to traffic being routed at a host
/// that silently fails specific endpoints. Currently checks:
///
/// - **dstack guest-agent socket**: `connect()` to the configured unix socket.
///   When unreachable, `/v1/attestation/report` returns 500 ("Failed to connect
///   to unix stream"), which breaks cloud-api's E2EE provider discovery and
///   blocks the model in cloud-api's pubkey routing — but the backend
///   (sglang/vLLM) keeps serving `/v1/models`, so a `/v1/models` probe sees a
///   healthy backend and keeps routing traffic to it.
/// - **inference backend**: HTTP `GET /health` against a backend selected
///   from the pool. Catches sglang/vLLM crashes and unreachable backends.
///   We deliberately do not use `/v1/models` here — it's served by the same
///   handler that processes inference and can stall for over a second when
///   the engine is mid-prefill, producing 503s on otherwise-healthy hosts.
///
/// Returns 200 with `{"status":"ok","checks":{...}}` when both checks pass,
/// 503 with `{"status":"unhealthy","checks":{...}}` otherwise. Each entry in
/// `checks` is a stable token (`"ok"`, `"unreachable"`, `"timeout"`, or
/// `"http_<code>"`) — detailed errors (paths, URLs, OS messages) are logged
/// server-side rather than returned to the unauthenticated caller.
pub async fn healthz(State(state): State<AppState>) -> impl IntoResponse {
    let dstack_path = state.config.dstack_socket_path.clone();
    // Hold the BackendGuard for the full probe so the backend's
    // `active_connections` count reflects the in-flight check; otherwise the
    // probe slot is "free" the moment we read the URL and least-connections
    // accounting under-counts the load.
    let (backend_url, _guard) = state.backend_pool.select_url(BACKEND_HEALTH_PATH);
    let client = state.http_client.clone();

    let (dstack_result, backend_result) = tokio::join!(
        check_dstack(&dstack_path),
        check_backend(&client, &backend_url, &_guard),
    );

    let healthy = dstack_result.is_ok() && backend_result.is_ok();
    let body = serde_json::json!({
        "status": if healthy { STATUS_OK } else { "unhealthy" },
        "checks": {
            "dstack": status_token(&dstack_result),
            "backend": status_token(&backend_result),
        },
    });

    let status = if healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(body))
}

fn status_token(result: &Result<(), &'static str>) -> &'static str {
    match result {
        Ok(()) => STATUS_OK,
        Err(token) => token,
    }
}

async fn check_dstack(path: &str) -> Result<(), &'static str> {
    match tokio::time::timeout(DSTACK_PROBE_TIMEOUT, UnixStream::connect(path)).await {
        Ok(Ok(_stream)) => Ok(()),
        Ok(Err(e)) => {
            warn!(
                check = "dstack",
                socket_path = %path,
                error = %e,
                "dstack socket unreachable"
            );
            Err(DSTACK_UNREACHABLE)
        }
        Err(_) => {
            warn!(
                check = "dstack",
                socket_path = %path,
                timeout_ms = DSTACK_PROBE_TIMEOUT.as_millis() as u64,
                "dstack socket connect timed out"
            );
            Err(DSTACK_TIMEOUT)
        }
    }
}

async fn check_backend(
    client: &reqwest::Client,
    url: &str,
    _guard: &BackendGuard,
) -> Result<(), &'static str> {
    let send = client.get(url).timeout(BACKEND_PROBE_TIMEOUT).send();
    match send.await {
        Ok(resp) if resp.status().is_success() => {
            // Drain the body so reqwest can return the connection to the
            // keep-alive pool. Without this, every probe opens a new TCP +
            // TLS connection — wasteful at the 5s interval model-proxy uses.
            if let Err(e) = resp.bytes().await {
                warn!(check = "backend", url = %url, error = %e, "drain failed");
                return Err(BACKEND_UNREACHABLE);
            }
            Ok(())
        }
        Ok(resp) => {
            let status = resp.status();
            let _ = resp.bytes().await;
            warn!(check = "backend", url = %url, status = %status, "backend returned non-success");
            Err(http_status_token(status))
        }
        Err(e) if e.is_timeout() => {
            warn!(
                check = "backend",
                url = %url,
                timeout_ms = BACKEND_PROBE_TIMEOUT.as_millis() as u64,
                "backend probe timed out"
            );
            Err(BACKEND_TIMEOUT)
        }
        Err(e) => {
            warn!(check = "backend", url = %url, error = %e, "backend probe failed");
            Err(BACKEND_UNREACHABLE)
        }
    }
}

/// Map an HTTP status to a small set of stable tokens. We bucket so the
/// response body never reveals more than the broad failure mode (bad request,
/// rate limited, server error, etc.); the exact code is in the warn log.
fn http_status_token(status: StatusCode) -> &'static str {
    match status.as_u16() {
        408 => "http_408",
        429 => "http_429",
        500..=599 => "http_5xx",
        400..=499 => "http_4xx",
        _ => BACKEND_UNREACHABLE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[tokio::test]
    async fn dstack_check_passes_when_socket_listens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dstack.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        // Accept loop so the connect completes.
        tokio::spawn(async move { while let Ok((_s, _)) = listener.accept().await {} });

        let result = check_dstack(path.to_str().unwrap()).await;
        assert!(result.is_ok(), "expected ok, got {result:?}");
    }

    #[tokio::test]
    async fn dstack_check_returns_unreachable_when_path_missing() {
        let result = check_dstack("/nonexistent/dstack.sock").await;
        assert_eq!(result, Err(DSTACK_UNREACHABLE));
    }

    #[tokio::test]
    async fn dstack_check_returns_unreachable_when_path_is_regular_file() {
        // A regular file at the socket path: connect() will fail with ECONNREFUSED.
        // Mirrors the on-host failure mode where the socket doesn't exist OR exists
        // but no daemon is listening. Either case must surface as "unreachable",
        // not as "ok".
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-socket");
        std::fs::write(&path, b"").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let result = check_dstack(path.to_str().unwrap()).await;
        assert_eq!(result, Err(DSTACK_UNREACHABLE));
    }

    #[test]
    fn http_status_token_buckets_known_classes() {
        assert_eq!(http_status_token(StatusCode::REQUEST_TIMEOUT), "http_408");
        assert_eq!(http_status_token(StatusCode::TOO_MANY_REQUESTS), "http_429");
        assert_eq!(
            http_status_token(StatusCode::INTERNAL_SERVER_ERROR),
            "http_5xx"
        );
        assert_eq!(http_status_token(StatusCode::BAD_GATEWAY), "http_5xx");
        assert_eq!(http_status_token(StatusCode::NOT_FOUND), "http_4xx");
    }
}
