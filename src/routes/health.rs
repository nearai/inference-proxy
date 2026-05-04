use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use tokio::net::UnixStream;

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
/// latency is `max(DSTACK, BACKEND)` plus parsing overhead. Kept tight to
/// stay under model-proxy's `health_check.slow_threshold_ms` (1500ms by
/// default) — slower-than-threshold probes are treated as failures there.
const DSTACK_PROBE_TIMEOUT: Duration = Duration::from_millis(300);
const BACKEND_PROBE_TIMEOUT: Duration = Duration::from_millis(1200);

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
/// - **inference backend**: HTTP `GET /v1/models` against a backend selected
///   from the pool. Catches sglang/vLLM crashes and unreachable backends.
///
/// Returns 200 with `{"status":"ok","checks":{...}}` when both checks pass,
/// 503 with `{"status":"unhealthy","checks":{...}}` otherwise. Each entry in
/// `checks` is `"ok"` or a short error string for diagnostics.
pub async fn healthz(State(state): State<AppState>) -> impl IntoResponse {
    let dstack_path = state.config.dstack_socket_path.clone();
    let backend_url = {
        let (url, _guard) = state.backend_pool.select_url("/v1/models");
        url
    };
    let client = state.http_client.clone();

    let (dstack_result, backend_result) = tokio::join!(
        check_dstack(&dstack_path),
        check_backend(&client, &backend_url),
    );

    let healthy = dstack_result.is_ok() && backend_result.is_ok();
    let body = serde_json::json!({
        "status": if healthy { "ok" } else { "unhealthy" },
        "checks": {
            "dstack": check_value(&dstack_result),
            "backend": check_value(&backend_result),
        },
    });

    let status = if healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(body))
}

fn check_value(result: &Result<(), String>) -> String {
    match result {
        Ok(()) => "ok".to_string(),
        Err(e) => e.clone(),
    }
}

async fn check_dstack(path: &str) -> Result<(), String> {
    match tokio::time::timeout(DSTACK_PROBE_TIMEOUT, UnixStream::connect(path)).await {
        Ok(Ok(_stream)) => Ok(()),
        Ok(Err(e)) => Err(format!("connect {path}: {e}")),
        Err(_) => Err(format!(
            "timeout after {}ms connecting to {path}",
            DSTACK_PROBE_TIMEOUT.as_millis()
        )),
    }
}

async fn check_backend(client: &reqwest::Client, url: &str) -> Result<(), String> {
    let send = client.get(url).timeout(BACKEND_PROBE_TIMEOUT).send();
    match send.await {
        Ok(resp) if resp.status().is_success() => Ok(()),
        Ok(resp) => Err(format!("GET {url} -> {}", resp.status())),
        Err(e) if e.is_timeout() => Err(format!(
            "timeout after {}ms GET {url}",
            BACKEND_PROBE_TIMEOUT.as_millis()
        )),
        Err(e) => Err(format!("GET {url}: {e}")),
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
    async fn dstack_check_fails_when_path_missing() {
        let result = check_dstack("/nonexistent/dstack.sock").await;
        let err = result.unwrap_err();
        assert!(
            err.contains("/nonexistent/dstack.sock"),
            "error should mention the path, got: {err}"
        );
    }

    #[tokio::test]
    async fn dstack_check_fails_when_path_is_regular_file() {
        // A regular file at the socket path: connect() will fail with ECONNREFUSED.
        // Mirrors the on-host failure mode where the socket doesn't exist OR exists
        // but no daemon is listening.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-socket");
        std::fs::write(&path, b"").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let result = check_dstack(path.to_str().unwrap()).await;
        assert!(result.is_err(), "regular file should not pass as a socket");
    }
}
