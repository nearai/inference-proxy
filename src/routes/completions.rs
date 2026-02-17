use axum::body::Body;
use axum::extract::State;
use axum::response::Response;

use crate::auth::RequireAuth;
use crate::error::AppError;
use crate::proxy::{self, make_usage_reporter, ProxyOpts, UsageType};
use crate::routes::chat::read_body_with_limit;
use crate::AppState;

/// POST /v1/completions
pub async fn completions(
    State(state): State<AppState>,
    auth: RequireAuth,
    body: Body,
) -> Result<Response, AppError> {
    let request_body = read_body_with_limit(body, state.config.max_request_size).await?;

    let mut request_json: serde_json::Value = serde_json::from_slice(&request_body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;

    let is_stream = request_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // For cloud API key requests with streaming, force include_usage
    // so the backend always sends token counts for billing.
    if auth.cloud_api_key.is_some() && is_stream {
        let stream_opts = request_json
            .get("stream_options")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let mut stream_opts = stream_opts;
        stream_opts.insert("include_usage".into(), true.into());
        request_json["stream_options"] = serde_json::Value::Object(stream_opts);
    }

    let modified_body =
        serde_json::to_vec(&request_json).map_err(|e| AppError::Internal(e.into()))?;

    let opts = ProxyOpts {
        signing: state.signing.clone(),
        cache: state.cache.clone(),
        id_prefix: "cmpl".to_string(),
        usage_reporter: make_usage_reporter(auth.cloud_api_key.as_ref(), &state),
        usage_type: UsageType::ChatCompletion,
    };

    if is_stream {
        proxy::proxy_streaming_request(
            &state.http_client,
            &state.config.completions_url,
            modified_body,
            &opts,
        )
        .await
    } else {
        proxy::proxy_json_request(
            &state.http_client,
            &state.config.completions_url,
            modified_body,
            &opts,
        )
        .await
    }
}
