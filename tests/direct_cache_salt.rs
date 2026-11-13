//! `cache_salt` on chat/completions: a Cloud API key caller always gets a
//! proxy-set value bound to its organization and workspace; a config-token
//! caller's body keeps whatever it sent.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;
#[path = "common/minimal_sse_response.rs"]
mod minimal_sse_response;

use common::*;
use minimal_sse_response::*;

const KEY_ORG_A: &str = "sk-test-cache-salt-org-a-0000000001";
const KEY_ORG_A_OTHER_WS: &str = "sk-test-cache-salt-org-a-0000000002";
const KEY_ORG_B: &str = "sk-test-cache-salt-org-b-0000000003";
const KEY_NO_OWNER: &str = "sk-test-cache-salt-no-owner-0000004";
const TRUSTED: &str = "test-token";

const COMPLETION_SSE: &str = "data: {\"id\":\"cmpl-1\",\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"ok\",\"finish_reason\":null}]}\n\n\
     data: {\"id\":\"cmpl-1\",\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n\
     data: [DONE]\n\n";

struct Harness {
    app: axum::Router,
    backend: MockServer,
    _cloud_api: MockServer,
}

async fn mount_key(cloud_api: &MockServer, key: &str, identity: Value) {
    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .and(header("authorization", format!("Bearer {key}").as_str()))
        .respond_with(ResponseTemplate::new(200).set_body_json(identity))
        .mount(cloud_api)
        .await;
}

async fn harness() -> Harness {
    let backend = MockServer::start().await;
    let cloud_api = MockServer::start().await;

    mount_key(
        &cloud_api,
        KEY_ORG_A,
        json!({"organization_id": "org-a", "workspace_id": "ws-1", "api_key_id": "key-a1"}),
    )
    .await;
    mount_key(
        &cloud_api,
        KEY_ORG_A_OTHER_WS,
        json!({"organization_id": "org-a", "workspace_id": "ws-2", "api_key_id": "key-a2"}),
    )
    .await;
    mount_key(
        &cloud_api,
        KEY_ORG_B,
        json!({"organization_id": "org-b", "workspace_id": "ws-1", "api_key_id": "key-b1"}),
    )
    .await;
    mount_key(&cloud_api, KEY_NO_OWNER, json!({"api_key_id": "key-x"})).await;
    Mock::given(method("POST"))
        .and(path("/v1/internal/usage"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&cloud_api)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(minimal_sse_response("chatcmpl-salt"), "text/event-stream"),
        )
        .mount(&backend)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(COMPLETION_SSE.as_bytes(), "text/event-stream"),
        )
        .mount(&backend)
        .await;

    let app = build_test_app(
        &backend.uri(),
        TestAppOptions {
            cloud_api_url: Some(cloud_api.uri()),
            ..Default::default()
        },
    );
    Harness {
        app,
        backend,
        _cloud_api: cloud_api,
    }
}

fn body_for(uri: &str, stream: bool, client_salt: Option<Value>) -> Value {
    let mut body = if uri == "/v1/completions" {
        json!({"model": "test-model", "prompt": "hi"})
    } else {
        json!({"model": "test-model", "messages": [{"role": "user", "content": "hi"}]})
    };
    if stream {
        body["stream"] = true.into();
    }
    if let Some(salt) = client_salt {
        body["cache_salt"] = salt;
    }
    body
}

/// Send one request and return the status plus the `cache_salt` the backend
/// received for it (`None` when the field was absent upstream).
async fn send(
    h: &Harness,
    uri: &str,
    bearer: &str,
    body: &Value,
) -> (StatusCode, Option<Option<Value>>) {
    let before = h
        .backend
        .received_requests()
        .await
        .unwrap_or_default()
        .len();
    let response = h
        .app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {bearer}"))
                .body(Body::from(serde_json::to_vec(body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    // Drain the body so a streaming request has finished upstream.
    let _ = http_body_util::BodyExt::collect(response.into_body()).await;
    let received = h.backend.received_requests().await.unwrap_or_default();
    let forwarded = received[before..]
        .iter()
        .rfind(|request| request.url.path() == uri)
        .map(|request| {
            let json: Value = serde_json::from_slice(&request.body).unwrap();
            json.get("cache_salt").cloned()
        });
    (status, forwarded)
}

/// The `cache_salt` a direct request was forwarded with; it must be present.
async fn direct_salt(h: &Harness, uri: &str, key: &str, body: &Value) -> String {
    let (status, forwarded) = send(h, uri, key, body).await;
    assert_eq!(status, StatusCode::OK, "{uri} {body}");
    let salt = forwarded
        .expect("request reached the backend")
        .expect("cache_salt is always set for a direct caller");
    let salt = salt.as_str().expect("cache_salt is a string").to_string();
    assert_eq!(salt.len(), 64, "{uri}");
    salt
}

const ROUTES: [&str; 2] = ["/v1/chat/completions", "/v1/completions"];

#[tokio::test]
async fn direct_tenants_never_share_a_namespace() {
    let h = harness().await;
    for uri in ROUTES {
        for stream in [false, true] {
            for client in [None, Some(json!("shared-value"))] {
                let body = body_for(uri, stream, client.clone());
                let a = direct_salt(&h, uri, KEY_ORG_A, &body).await;
                let a_other_ws = direct_salt(&h, uri, KEY_ORG_A_OTHER_WS, &body).await;
                let b = direct_salt(&h, uri, KEY_ORG_B, &body).await;
                assert_ne!(a, b, "{uri} stream={stream} client={client:?}");
                assert_ne!(a, a_other_ws, "{uri} stream={stream} client={client:?}");
                assert_ne!(b, a_other_ws, "{uri} stream={stream} client={client:?}");
                // Stable for the same tenant and the same client value.
                assert_eq!(a, direct_salt(&h, uri, KEY_ORG_A, &body).await);
            }
        }
    }
}

#[tokio::test]
async fn direct_client_value_is_isolated_and_never_forwarded() {
    let h = harness().await;
    // The value another tenant's request was forwarded with.
    let other_tenant = direct_salt(
        &h,
        "/v1/chat/completions",
        KEY_ORG_B,
        &body_for("/v1/chat/completions", false, Some(json!("theirs"))),
    )
    .await;
    for uri in ROUTES {
        for stream in [false, true] {
            let omitted = direct_salt(&h, uri, KEY_ORG_A, &body_for(uri, stream, None)).await;
            let first = direct_salt(
                &h,
                uri,
                KEY_ORG_A,
                &body_for(uri, stream, Some(json!("s1"))),
            )
            .await;
            let second = direct_salt(
                &h,
                uri,
                KEY_ORG_A,
                &body_for(uri, stream, Some(json!("s2"))),
            )
            .await;
            assert_ne!(first, "s1");
            assert_ne!(first, second, "a client can still split its own traffic");
            assert_ne!(first, omitted);
            // Sending another tenant's forwarded value verbatim does not reach it.
            let replay = direct_salt(
                &h,
                uri,
                KEY_ORG_A,
                &body_for(uri, stream, Some(json!(other_tenant.clone()))),
            )
            .await;
            assert_ne!(replay, other_tenant, "{uri} stream={stream}");
            // Non-string values are replaced, never forwarded.
            for odd in [json!(null), json!(42), json!({"k": "v"})] {
                let salt = direct_salt(&h, uri, KEY_ORG_A, &body_for(uri, stream, Some(odd))).await;
                assert_ne!(salt, other_tenant);
            }
        }
    }
}

#[tokio::test]
async fn trusted_caller_cache_salt_passes_through_unchanged() {
    let h = harness().await;
    for uri in ROUTES {
        for stream in [false, true] {
            let body = body_for(uri, stream, Some(json!("gateway-derived-value")));
            let (status, forwarded) = send(&h, uri, TRUSTED, &body).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(
                forwarded.expect("request reached the backend"),
                Some(json!("gateway-derived-value")),
                "{uri} stream={stream}"
            );
            // A trusted caller without a value gets none added.
            let (status, forwarded) = send(&h, uri, TRUSTED, &body_for(uri, stream, None)).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(
                forwarded.expect("request reached the backend"),
                None,
                "{uri} stream={stream}"
            );
        }
    }
}

#[tokio::test]
async fn direct_key_without_an_owner_never_reaches_the_backend() {
    let h = harness().await;
    for uri in ROUTES {
        let body = body_for(uri, false, Some(json!("mine")));
        let (status, forwarded) = send(&h, uri, KEY_NO_OWNER, &body).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{uri}");
        assert!(forwarded.is_none(), "{uri}");
    }
}
