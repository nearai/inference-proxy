//! Model routes (gateway mode, `VLLM_PROXY_MODEL_ROUTES`): one public gateway
//! fronting other models, each served by its own gateway.
//!
//! A gateway is model-scoped — one `MODEL_NAME`, one admission budget, one
//! billing identity, one backend pool. An aggregator that reaches us at a
//! single base URL picks the model in the request body instead, so the public
//! gateway reads the body's `model` and, for a routed model, hands the request
//! as it arrived to that model's gateway (typically on loopback) and streams
//! the answer back as it left. Everything that makes a lane a lane — auth,
//! admission, priority, reasoning switch, usage billing — happens there, once.
//! Here a routed request takes no admission slot and is never billed.
//!
//! With no routes configured, the inference handlers are called exactly as
//! before: the body is not even read here.

use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::handler::Handler;
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{post, MethodRouter};
use futures_util::TryStreamExt;
use serde_json::Value;

use crate::config::{Config, ModelRoute};
use crate::error::AppError;
use crate::routes::chat::read_body_with_limit;
use crate::{AppState, TracingIds};

/// How long `/v1/models` waits for a routed gateway's listing.
const ROUTED_MODELS_TIMEOUT: Duration = Duration::from_secs(3);

/// What the front does with one inference request.
#[derive(Debug, PartialEq, Eq)]
enum Target<'a> {
    /// This gateway's own model (or no usable `model`): the existing handler.
    Local,
    /// Another model's gateway.
    Routed(&'a ModelRoute),
    /// A model neither served here nor routed.
    Unknown(String),
}

/// Decide from the buffered body. Only a non-empty string `model` that is not
/// `MODEL_NAME` leaves the local path; a body that is not JSON, has no
/// `model`, or names it with another type is the local handler's to judge.
fn target<'a>(model_name: &str, routes: &'a [ModelRoute], body: &[u8]) -> Target<'a> {
    #[derive(serde::Deserialize)]
    struct ModelOnly {
        model: Option<Value>,
    }
    let Ok(ModelOnly {
        model: Some(Value::String(model)),
    }) = serde_json::from_slice::<ModelOnly>(body)
    else {
        return Target::Local;
    };
    if model.is_empty() || model == model_name {
        return Target::Local;
    }
    match routes.iter().find(|r| r.model == model) {
        Some(route) => Target::Routed(route),
        None => Target::Unknown(model),
    }
}

/// A POST inference route whose body may name a routed model. `limit` is the
/// handler's own body-size limit, so a request is refused for size exactly as
/// it would be without routes.
pub fn post_by_model<H, T>(handler: H, limit: fn(&Config) -> usize) -> MethodRouter<AppState>
where
    H: Handler<T, AppState>,
    T: 'static,
{
    post(move |State(state): State<AppState>, request: Request| {
        let handler = handler.clone();
        async move { dispatch(handler, state, request, limit).await }
    })
}

async fn dispatch<H, T>(
    handler: H,
    state: AppState,
    request: Request,
    limit: fn(&Config) -> usize,
) -> Response
where
    H: Handler<T, AppState>,
    T: 'static,
{
    if state.config.model_routes.is_empty() {
        return handler.call(request, state).await;
    }
    let (parts, body) = request.into_parts();
    let body = match read_body_with_limit(body, limit(&state.config)).await {
        Ok(body) => body,
        Err(error) => return error.into_response(),
    };
    match target(&state.config.model_name, &state.config.model_routes, &body) {
        Target::Local => {
            handler
                .call(Request::from_parts(parts, Body::from(body)), state)
                .await
        }
        Target::Routed(route) => {
            let route = route.clone();
            forward(&state, &route, parts, body).await
        }
        Target::Unknown(model) => {
            metrics::counter!("model_route_requests_total", "model" => "unknown", "status" => "404")
                .increment(1);
            model_not_found(&model)
        }
    }
}

/// OpenAI's answer for a model the caller cannot use.
fn model_not_found(model: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!("The model `{model}` does not exist or you do not have access to it."),
            "type": "invalid_request_error",
            "param": null,
            "code": "model_not_found",
        }
    });
    (StatusCode::NOT_FOUND, axum::Json(body)).into_response()
}

/// Hop-by-hop headers (RFC 9110 §7.6.1) plus the ones the client library
/// recomputes for the new connection.
fn is_hop_by_hop(name: &HeaderName, connection_tokens: &[String]) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-connection"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    ) || connection_tokens.iter().any(|t| t == name.as_str())
}

fn connection_tokens(headers: &HeaderMap) -> Vec<String> {
    headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(|t| t.trim().to_ascii_lowercase())
        .filter(|t| !t.is_empty())
        .collect()
}

fn end_to_end(headers: &HeaderMap, also_drop: &[HeaderName]) -> HeaderMap {
    let tokens = connection_tokens(headers);
    let mut out = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        if !is_hop_by_hop(name, &tokens) && !also_drop.contains(name) {
            out.append(name.clone(), value.clone());
        }
    }
    out
}

/// Send the request as it arrived to the routed gateway and stream its answer
/// back as it left. Uses the general client (`http_client`), never the
/// backend client: that one carries this lane's backend bearer by default,
/// which must not reach another gateway as a stand-in for a missing key.
async fn forward(
    state: &AppState,
    route: &ModelRoute,
    parts: axum::http::request::Parts,
    body: Vec<u8>,
) -> Response {
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or_else(|| parts.uri.path());
    let url = format!("{}{}", route.base_url, path_and_query);
    let mut headers = end_to_end(&parts.headers, &[header::HOST, header::CONTENT_LENGTH]);
    // The routed gateway logs under the same request id as this one.
    if let Some(ids) = parts.extensions.get::<TracingIds>() {
        if let Ok(value) = HeaderValue::from_str(&ids.request_id) {
            headers.insert("x-request-id", value);
        }
    }
    tracing::info!(model = %route.model, "Forwarding request to the model's gateway");
    let label = |status: &str| {
        metrics::counter!(
            "model_route_requests_total",
            "model" => route.model.clone(),
            "status" => status.to_string()
        )
        .increment(1);
    };

    let upstream = state
        .http_client
        .request(parts.method.clone(), &url)
        .headers(headers)
        .body(body)
        .send()
        .await;
    let upstream = match upstream {
        Ok(upstream) => upstream,
        Err(error) => {
            let (status, message, error_type, label_value) = if error.is_timeout() {
                (
                    StatusCode::GATEWAY_TIMEOUT,
                    "Upstream request timed out",
                    "upstream_request_timeout",
                    "timeout",
                )
            } else {
                (
                    StatusCode::BAD_GATEWAY,
                    "The model's inference service is not reachable",
                    "upstream_unreachable",
                    "unreachable",
                )
            };
            label(label_value);
            tracing::warn!(model = %route.model, error = %error, "Model route upstream failed");
            return AppError::UpstreamParsed {
                status,
                message: message.to_string(),
                error_type: error_type.to_string(),
            }
            .into_response();
        }
    };

    let status = upstream.status();
    label(status.as_str());
    let mut response = Response::builder().status(status);
    if let Some(out) = response.headers_mut() {
        *out = end_to_end(upstream.headers(), &[]);
    }
    let stream = upstream.bytes_stream().map_err(std::io::Error::other);
    response
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

/// `/v1/models` with model routes: the front's own listing plus each routed
/// gateway's entry for its model, verbatim (capacity, `discount_to_user`,
/// OpenRouter slug are that gateway's). Listings are fetched in parallel with
/// a short timeout; a gateway that does not answer, or does not list its
/// model, is left out with a warning rather than failing the whole document.
pub async fn merge_routed_models(state: &AppState, own: Response) -> Result<Response, AppError> {
    let routes = &state.config.model_routes;
    if routes.is_empty() || own.status() != StatusCode::OK {
        return Ok(own);
    }
    let (parts, body) = own.into_parts();
    let bytes = axum::body::to_bytes(body, usize::MAX)
        .await
        .map_err(|error| AppError::Internal(anyhow::anyhow!("models document: {error}")))?;
    let mut document: Value = match serde_json::from_slice(&bytes) {
        Ok(document @ Value::Object(_)) if document.get("data").is_some_and(Value::is_array) => {
            document
        }
        _ => {
            tracing::warn!("Own model list is not a model list; routed models not merged");
            return Ok(Response::from_parts(parts, Body::from(bytes)));
        }
    };
    let fetched =
        futures_util::future::join_all(routes.iter().map(|route| routed_entries(state, route)))
            .await;
    if let Some(data) = document.get_mut("data").and_then(Value::as_array_mut) {
        data.extend(fetched.into_iter().flatten());
    }
    Ok((StatusCode::OK, axum::Json(document)).into_response())
}

async fn routed_entries(state: &AppState, route: &ModelRoute) -> Vec<Value> {
    let url = format!("{}{}", route.base_url, crate::routes::ROUTE_V1_MODELS);
    let result = async {
        let response = state
            .http_client
            .get(&url)
            .timeout(ROUTED_MODELS_TIMEOUT)
            .send()
            .await?;
        if !response.status().is_success() {
            anyhow::bail!("answered {}", response.status());
        }
        let list: Value = response.json().await?;
        let entries: Vec<Value> = list
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("no `data` array"))?
            .iter()
            .filter(|entry| entry.get("id").and_then(Value::as_str) == Some(route.model.as_str()))
            .cloned()
            .collect();
        if entries.is_empty() {
            anyhow::bail!("its list has no entry for the model");
        }
        Ok(entries)
    }
    .await;
    result.unwrap_or_else(|error: anyhow::Error| {
        metrics::counter!("model_route_listing_failures_total", "model" => route.model.clone())
            .increment(1);
        tracing::warn!(
            model = %route.model,
            error = %error,
            "Routed gateway's model list unavailable, leaving the model out of /v1/models"
        );
        Vec::new()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_selection() {
        let routes = crate::config::parse_model_routes("other=http://127.0.0.1:1", "glm").unwrap();
        let target = |body: &[u8]| target("glm", &routes, body);
        assert_eq!(target(br#"{"model":"glm"}"#), Target::Local);
        assert_eq!(target(br#"{"messages":[]}"#), Target::Local);
        assert_eq!(target(br#"{"model":""}"#), Target::Local);
        assert_eq!(target(br#"{"model":7}"#), Target::Local);
        assert_eq!(target(b"not json"), Target::Local);
        assert_eq!(
            target(br#"{"model":"other","messages":[]}"#),
            Target::Routed(&routes[0])
        );
        assert_eq!(
            target(br#"{"model":"nope"}"#),
            Target::Unknown("nope".to_string())
        );
    }

    #[test]
    fn hop_by_hop_headers_are_dropped() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer sk-x"));
        headers.insert("x-nearai-priority", HeaderValue::from_static("5"));
        headers.insert("connection", HeaderValue::from_static("keep-alive, x-hop"));
        headers.insert("x-hop", HeaderValue::from_static("1"));
        headers.insert("transfer-encoding", HeaderValue::from_static("chunked"));
        headers.insert("host", HeaderValue::from_static("front"));
        let out = end_to_end(&headers, &[header::HOST]);
        assert_eq!(out.get("authorization").unwrap(), "Bearer sk-x");
        assert_eq!(out.get("x-nearai-priority").unwrap(), "5");
        for gone in ["connection", "x-hop", "transfer-encoding", "host"] {
            assert!(out.get(gone).is_none(), "{gone}");
        }
    }
}
