use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::Extension;

use crate::admission::RejectReason;
use crate::auth::RequireAuth;
use crate::backend_pool;
use crate::encryption::{self, Endpoint};
use crate::error::AppError;
use crate::proxy::{
    self, make_usage_reporter, ConnectFailover, ProxyOpts, ResponseShape, UsageType,
};
use crate::routes::chat::{read_body_with_limit, resolve_request_hash_for_signing};
use crate::{AppState, TracingIds};

/// POST /v1/completions
pub async fn completions(
    State(state): State<AppState>,
    auth: RequireAuth,
    Extension(tracing_ids): Extension<TracingIds>,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, AppError> {
    let request_body = read_body_with_limit(body, state.config.max_request_size).await?;

    let mut request_json: serde_json::Value = serde_json::from_slice(&request_body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;

    // Engine `priority`: the proxy decides it (trusted callers may set it via
    // header; any client value is discarded).
    crate::priority::apply_priority(&mut request_json, &headers, auth.cloud_api_key.is_none());

    // Extract encryption context from headers
    let enc_ctx = encryption::extract_encryption_context(&headers)?;
    let tracing_ids = tracing_ids.with_authenticated_context(&headers, &auth);

    let original_request_hash = Some(resolve_request_hash_for_signing(
        &headers,
        &request_body,
        auth.cloud_api_key.is_none(),
    ));

    // Decrypt request fields if encryption is active
    if let Some(ref ctx) = enc_ctx {
        encryption::decrypt_request_fields(
            &mut request_json,
            Endpoint::Completions,
            ctx,
            &state.signing,
        )?;
    }

    let is_stream = request_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // For cloud API key requests with streaming, force include_usage AND
    // continuous_usage_stats so the backend sends running cumulative token counts
    // on every chunk, making an interrupted stream billable (nearai/infra#98).
    // include_usage alone only emits usage in the final chunk, which a stream
    // interrupted before [DONE] never reaches.
    if auth.cloud_api_key.is_some() && is_stream {
        let stream_opts = request_json
            .get("stream_options")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let mut stream_opts = stream_opts;
        stream_opts.insert("include_usage".into(), true.into());
        stream_opts.insert("continuous_usage_stats".into(), true.into());
        request_json["stream_options"] = serde_json::Value::Object(stream_opts);
    }

    let modified_body =
        serde_json::to_vec(&request_json).map_err(|e| AppError::Internal(e.into()))?;

    // Build encryption transforms if active
    let (response_transform, chunk_transform) = if let Some(ctx) = enc_ctx {
        let signing = state.signing.clone();
        (
            Some(encryption::make_response_transform(
                Endpoint::Completions,
                ctx.clone(),
                signing.clone(),
            )),
            Some(encryption::make_chunk_transform(
                Endpoint::Completions,
                ctx,
                signing,
            )),
        )
    } else {
        (None, None)
    };

    // Long-context tier and lane admission (gateway mode), see the chat
    // route. Token ids in `prompt` are counted exactly; text is estimated.
    let tier = crate::context_tier::decide(
        &state.backend_pool,
        state.config.long_context_above_tokens,
        || crate::context_tier::completion_estimate(&request_json),
    );
    let permit = state.admission.try_admit(&state.backend_pool, tier)?;
    let host_share = state
        .admission
        .host_share(state.backend_pool.healthy_count());
    let mut restrict = tier.and_then(|tier| tier.restrict);
    let place = |tier| {
        let policy = backend_pool::Policy {
            max_conns: host_share,
            avoid: &|index| state.admission.backend_saturated(index),
            engine: &|index| state.admission.engine(index),
            tier,
        };
        state
            .backend_affinity
            .place(&state.backend_pool, None, "/v1/completions", &policy)
    };
    // See the chat route: a tier that just emptied falls back, and `restrict`
    // follows the placement.
    let mut placement = place(restrict);
    if placement.is_none()
        && restrict.is_some_and(|tier| {
            crate::context_tier::recheck_restriction(&state.backend_pool, tier).is_none()
        })
    {
        restrict = None;
        placement = place(None);
    }
    let placement =
        placement.ok_or_else(|| AppError::from(state.admission.reject(RejectReason::HostShare)))?;
    if let Some(permit) = permit.as_ref() {
        permit.attach_backend(placement.index);
    }
    let connect_failover = state
        .config
        .backend_connect_failover
        .then(|| ConnectFailover {
            pool: state.backend_pool.clone(),
            path: "/v1/completions",
            index: placement.index,
            tier: restrict,
            affinity: None,
        });
    let url = placement.url;

    let opts = ProxyOpts {
        signing: state.signing.clone(),
        cache: state.cache.clone(),
        id_prefix: "cmpl".to_string(),
        model_name: state.config.model_name.clone(),
        usage_reporter: make_usage_reporter(&auth, &state),
        usage_type: UsageType::ChatCompletion,
        request_hash: original_request_hash,
        response_transform,
        chunk_transform,
        backend_guard: Some(placement.guard),
        stream_idle_timeout_secs: state.config.stream_idle_timeout_secs,
        sse_keepalive_secs: state.config.sse_keepalive_secs,
        map_queue_full_to_429: state.config.map_queue_full_to_429,
        stream_error_peek_ms: state.config.stream_error_peek_ms,
        stream_commit_ms: state.config.stream_commit_ms,
        response_shape: ResponseShape::TextCompletion,
        tracing_ids: Some(tracing_ids),
        upstream_data_parallel_rank: None,
        admission: permit,
        connect_failover,
    };

    if is_stream {
        proxy::proxy_streaming_request(&state.backend_client, &url, modified_body, opts).await
    } else {
        proxy::proxy_json_request(&state.backend_client, &url, modified_body, opts).await
    }
}
