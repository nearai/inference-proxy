use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::MockServer;

mod common;
use common::*;

const BLOCKED_PATHS: &[&str] = &[
    "/openapi.json",
    "/get_model_info",
    "/generate",
    "/flush_cache",
    "/abort_request",
    "/pause_generation",
    "/release_memory_occupation",
    "/resume_memory_occupation",
    "/configure_logging",
    "/set_internal_state",
    "/start_profile",
    "/stop_profile",
    "/update_weights_from_disk",
    "/update_weights_from_distributed",
    "/update_weights_from_tensor",
    "/load_lora_adapter",
    "/unload_lora_adapter",
    "/v1/custom/endpoint",
    "/v1/chat/completions/",
    "/v1/chat/%63ompletions",
    "/v1/responsesx",
    "/v1/privacy/redact",
];

#[tokio::test]
async fn unknown_and_backend_control_plane_routes_never_reach_backend() {
    let backend = MockServer::start().await;
    let app = build_test_app(&backend.uri(), TestAppOptions::default());

    for path in BLOCKED_PATHS {
        for method in [Method::GET, Method::POST, Method::PUT] {
            for authorization in [
                None,
                Some("Bearer test-token"),
                Some("Bearer sk-test-customer"),
            ] {
                let mut request = Request::builder()
                    .method(method.clone())
                    .uri(*path)
                    .header("content-type", "application/json");
                if let Some(value) = authorization {
                    request = request.header("authorization", value);
                }

                let response = app
                    .clone()
                    .oneshot(request.body(Body::from(r#"{"probe":true}"#)).unwrap())
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    StatusCode::NOT_FOUND,
                    "{method} {path} with authorization={authorization:?}"
                );
                let body: serde_json::Value = serde_json::from_slice(
                    &response.into_body().collect().await.unwrap().to_bytes(),
                )
                .unwrap();
                assert_eq!(body["error"]["type"], "not_found");
            }
        }
    }

    assert!(
        backend
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty(),
        "deny-by-default routes must be rejected locally"
    );
}
