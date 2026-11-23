//! Deprecation headers on direct endpoint responses (`direct_deprecation`).

use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[path = "common/body_to_bytes.rs"]
mod body_to_bytes;
mod common;

use body_to_bytes::*;
use common::*;
use vllm_proxy_rs::direct_deprecation::{DEPRECATION, DEPRECATION_LINK, SUNSET};

const API_KEY: &str = "Bearer sk-test-public-key-1234567890123";
const CONFIG_TOKEN: &str = "Bearer test-token";
const CHAT: &str = r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#;
const CHAT_STREAM: &str =
    r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
const CHAT_RESPONSE: &str = r#"{"id":"chatcmpl-deprecation","object":"chat.completion","model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
const CHAT_SSE: &str = concat!(
    "data: {\"id\":\"chatcmpl-deprecation\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-deprecation\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
    "data: [DONE]\n\n",
);

struct Servers {
    backend: MockServer,
    cloud_api: MockServer,
}

async fn start_servers(key_status: u16) -> Servers {
    let backend = MockServer::start().await;
    let cloud_api = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .respond_with(
            ResponseTemplate::new(key_status).set_body_json(serde_json::json!({
                "organization_id": "org-test",
                "workspace_id": "ws-test",
                "api_key_id": "key-test"
            })),
        )
        .mount(&cloud_api)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/internal/usage"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&cloud_api)
        .await;
    Servers { backend, cloud_api }
}

async fn mount_chat(backend: &MockServer, status: u16, body: &str, content_type: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(status).set_body_raw(body.as_bytes(), content_type))
        .mount(backend)
        .await;
}

async fn send(
    servers: &Servers,
    host: &str,
    authorization: &str,
    body: &'static str,
) -> Response<Body> {
    build_test_app(
        &servers.backend.uri(),
        TestAppOptions {
            cloud_api_url: Some(servers.cloud_api.uri()),
            ..Default::default()
        },
    )
    .oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("host", host)
            .header("authorization", authorization)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap(),
    )
    .await
    .unwrap()
}

fn assert_notice(response: &Response<Body>, case: &str) {
    let headers = response.headers();
    assert_eq!(headers.get("deprecation").unwrap(), DEPRECATION, "{case}");
    assert_eq!(headers.get("sunset").unwrap(), SUNSET, "{case}");
    assert_eq!(headers.get("link").unwrap(), DEPRECATION_LINK, "{case}");
}

fn assert_no_notice(response: &Response<Body>, case: &str) {
    let headers = response.headers();
    for name in ["deprecation", "sunset", "link"] {
        assert!(headers.get(name).is_none(), "{case}: unexpected {name}");
    }
}

#[tokio::test]
async fn api_key_on_a_model_domain_gets_the_notice() {
    let servers = start_servers(200).await;
    mount_chat(&servers.backend, 200, CHAT_RESPONSE, "application/json").await;
    for host in [
        "test-model.completions.near.ai",
        "test-model.completions.near.ai:443",
        "test-model-i0.completions.near.ai",
        "test-model-long.completions.near.ai",
        "test-model-long-i1.completions.near.ai",
        "test-model.completions-stg.near.ai",
    ] {
        let response = send(&servers, host, API_KEY, CHAT).await;
        assert_eq!(response.status(), StatusCode::OK, "{host}");
        assert_notice(&response, host);
        // The body is what the engine returned; only headers were added.
        let body: serde_json::Value =
            serde_json::from_slice(&body_to_bytes(response).await).expect("chat response is JSON");
        assert_eq!(body["choices"][0]["message"]["content"], "ok", "{host}");
    }
}

#[tokio::test]
async fn streaming_and_upstream_errors_get_the_notice() {
    let servers = start_servers(200).await;
    mount_chat(&servers.backend, 200, CHAT_SSE, "text/event-stream").await;
    let response = send(
        &servers,
        "test-model.completions.near.ai",
        API_KEY,
        CHAT_STREAM,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_notice(&response, "streaming");
    assert!(!body_to_bytes(response).await.is_empty());

    let servers = start_servers(200).await;
    mount_chat(
        &servers.backend,
        400,
        r#"{"error":{"message":"bad request","type":"BadRequestError"}}"#,
        "application/json",
    )
    .await;
    let response = send(&servers, "test-model.completions.near.ai", API_KEY, CHAT).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_notice(&response, "upstream 400");
}

#[tokio::test]
async fn other_callers_get_no_notice() {
    let servers = start_servers(200).await;
    mount_chat(&servers.backend, 200, CHAT_RESPONSE, "application/json").await;

    // The deployment token on a model domain.
    let response = send(
        &servers,
        "test-model.completions.near.ai",
        CONFIG_TOKEN,
        CHAT,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_no_notice(&response, "config token");

    // An API key on any other host: a gateway domain or an OHTTP loopback.
    for host in [
        "127.0.0.1:8000",
        "gateway.example.com",
        "completions.near.ai",
    ] {
        let response = send(&servers, host, API_KEY, CHAT).await;
        assert_eq!(response.status(), StatusCode::OK, "{host}");
        assert_no_notice(&response, host);
    }

    // A key the Cloud API refused never authenticated.
    let refused = start_servers(401).await;
    let response = send(&refused, "test-model.completions.near.ai", API_KEY, CHAT).await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_no_notice(&response, "refused key");
}
