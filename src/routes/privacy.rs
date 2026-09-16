use axum::body::Body;
use axum::extract::{OriginalUri, State};
use axum::http::HeaderMap;
use axum::response::Response;
use axum::Extension;
use sha2::{Digest, Sha256};

use crate::auth::RequireAuth;
use crate::error::AppError;
use crate::proxy::{self, make_usage_reporter, ProxyOpts, ResponseShape, UsageType};
use crate::routes::chat::read_body_with_limit;
use crate::{AppState, TracingIds};

const EXCLUDED_REQUEST_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "te",
    "trailer",
    "upgrade",
    "proxy-authenticate",
    "proxy-authorization",
    "authorization",
    "x-request-id",
    "x-org-id",
    "x-workspace-id",
    crate::vllm_dp_affinity::DATA_PARALLEL_RANK_HEADER,
];

const EXCLUDED_RESPONSE_HEADERS: &[&str] = &[
    "transfer-encoding",
    "connection",
    "keep-alive",
    "te",
    "trailer",
    "upgrade",
];

/// Direct privacy-filter classification.
///
/// This is deliberately an exact route rather than a generic backend
/// passthrough. Privacy-filter deployments expose this product endpoint, while
/// backend maintenance and discovery paths remain unreachable.
pub async fn classify(
    State(state): State<AppState>,
    auth: RequireAuth,
    Extension(tracing_ids): Extension<TracingIds>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, AppError> {
    let request_body = read_body_with_limit(body, state.config.max_request_size).await?;
    let request_sha256 = hex::encode(Sha256::digest(&request_body));
    let path_with_query = match uri.query() {
        Some(query) => format!("{}?{query}", super::ROUTE_PRIVACY_CLASSIFY),
        None => super::ROUTE_PRIVACY_CLASSIFY.to_string(),
    };
    let (backend_url, backend_guard) = state.backend_pool.select_url(&path_with_query);
    let tracing_ids = tracing_ids.with_authenticated_context(&headers, &auth);

    let mut request = state.backend_client.post(&backend_url);
    for (name, value) in &headers {
        if !EXCLUDED_REQUEST_HEADERS.contains(&name.as_str()) {
            request = request.header(name, value);
        }
    }
    request = proxy::apply_tracing_headers(request, Some(&tracing_ids)).body(request_body);

    let upstream_start = std::time::Instant::now();
    let response = request
        .send()
        .await
        .map_err(|error| AppError::Internal(error.into()))?;
    metrics::histogram!(
        "upstream_request_duration_seconds",
        "endpoint" => "privacy_classify"
    )
    .record(upstream_start.elapsed().as_secs_f64());

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_default();
        let info = proxy::log_upstream_error(status, &backend_url, &body, Some(&tracing_ids));
        return Err(AppError::UpstreamParsed {
            status: proxy::effective_error_status(status.as_u16(), info.as_ref(), false),
            message: info
                .as_ref()
                .map(|error| error.message.clone())
                .unwrap_or_else(|| format!("Upstream request failed with status {status}")),
            error_type: info
                .as_ref()
                .map(|error| error.error_type.clone())
                .unwrap_or_else(|| "upstream_error".to_string()),
        });
    }

    let axum_status =
        axum::http::StatusCode::from_u16(status.as_u16()).unwrap_or(axum::http::StatusCode::OK);
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();

    if content_type.contains("application/json") {
        let response_bytes = response
            .bytes()
            .await
            .map_err(|error| AppError::Internal(error.into()))?;
        if response_bytes.len() > state.config.max_request_size {
            return Err(AppError::PayloadTooLarge {
                max_size: state.config.max_request_size,
            });
        }
        let opts = ProxyOpts {
            signing: state.signing.clone(),
            cache: state.cache.clone(),
            id_prefix: "privacy".to_string(),
            model_name: state.config.model_name.clone(),
            usage_reporter: make_usage_reporter(&auth, &state),
            usage_type: UsageType::PrivacyClassify,
            request_hash: None,
            response_transform: None,
            chunk_transform: None,
            backend_guard: Some(backend_guard),
            stream_idle_timeout_secs: state.config.stream_idle_timeout_secs,
            sse_keepalive_secs: 0,
            map_queue_full_to_429: false,
            stream_error_peek_ms: 0,
            response_shape: ResponseShape::ChatCompletion,
            tracing_ids: Some(tracing_ids),
            upstream_data_parallel_rank: None,
            admission: None,
            connect_failover: None,
        };
        proxy::sign_and_cache_json_response(
            &response_bytes,
            &request_sha256,
            opts,
            axum_status,
            Some(proxy::CompletionContext {
                started_at: upstream_start,
                mode: "privacy_classify_json",
            }),
        )
        .await
    } else {
        let mut builder = Response::builder().status(axum_status);
        for (name, value) in response.headers() {
            if !EXCLUDED_RESPONSE_HEADERS.contains(&name.as_str()) {
                builder = builder.header(name, value);
            }
        }
        Ok(builder
            .body(Body::from_stream(response.bytes_stream()))
            .unwrap())
    }
}
