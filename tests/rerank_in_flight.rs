//! `/v1/rerank` in-flight limit: refusal before dispatch, pools keyed by
//! server-side identity, and permits held until the upstream completes.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use vllm_proxy_rs::rerank_limit::{RerankLimiter, RerankLimits, Tenant};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;
use common::*;

const RERANK_BODY: &str = r#"{"model":"test-model","query":"q","documents":["a","b"]}"#;
const KEY_A: &str = "sk-test-workspace-a-1111111111111111";
const KEY_A2: &str = "sk-test-workspace-a-2222222222222222";
const KEY_B: &str = "sk-test-workspace-b-3333333333333333";

fn limiter(per_workspace: usize, direct: usize, trusted: usize) -> Arc<RerankLimiter> {
    Arc::new(RerankLimiter::new(RerankLimits {
        per_workspace,
        direct,
        trusted,
    }))
}

fn rerank_request(bearer: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/rerank")
        .header("authorization", format!("Bearer {bearer}"))
        .header("content-type", "application/json")
        // Client-supplied tenant hints must not select the pool.
        .header("x-org-id", "org-a")
        .header("x-workspace-id", "ws-a")
        .header("x-forwarded-for", "203.0.113.7")
        .body(Body::from(RERANK_BODY))
        .unwrap()
}

async fn rerank_backend(delay: Duration) -> MockServer {
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({
                    "id": "rerank-limit",
                    "results": [{"index": 0, "relevance_score": 0.9}],
                    "usage": {"prompt_tokens": 4, "total_tokens": 4}
                }))
                .set_delay(delay),
        )
        .mount(&backend)
        .await;
    backend
}

/// Cloud-api answering KEY_A and KEY_A2 with one workspace, KEY_B with another.
async fn cloud_api() -> MockServer {
    let cloud_api = MockServer::start().await;
    for (key, org, workspace, key_id) in [
        (KEY_A, "org-a", "ws-a", "key-a"),
        (KEY_A2, "org-a", "ws-a", "key-a2"),
        (KEY_B, "org-b", "ws-b", "key-b"),
    ] {
        Mock::given(method("POST"))
            .and(path("/v1/check_api_key"))
            .and(header("authorization", format!("Bearer {key}").as_str()))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "organization_id": org,
                "workspace_id": workspace,
                "api_key_id": key_id
            })))
            .mount(&cloud_api)
            .await;
    }
    Mock::given(method("POST"))
        .and(path("/v1/internal/usage"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&cloud_api)
        .await;
    cloud_api
}

async fn backend_requests(backend: &MockServer) -> usize {
    backend
        .received_requests()
        .await
        .expect("backend records requests")
        .len()
}

async fn assert_refused(response: axum::response::Response) {
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "overloaded");
}

#[tokio::test]
async fn trusted_pool_full_is_refused_before_dispatch() {
    let backend = rerank_backend(Duration::ZERO).await;
    let limiter = limiter(1, 1, 1);
    let app = build_test_app(
        &backend.uri(),
        TestAppOptions {
            rerank_limiter: Some(limiter.clone()),
            ..Default::default()
        },
    );

    let held = limiter.try_acquire(Tenant::Trusted).unwrap();
    assert_refused(
        app.clone()
            .oneshot(rerank_request("test-token"))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(backend_requests(&backend).await, 0);

    drop(held);
    let response = app.oneshot(rerank_request("test-token")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(backend_requests(&backend).await, 1);
    assert_eq!(limiter.in_flight(), (0, 0, 0));
}

#[tokio::test]
async fn direct_keys_are_limited_per_cloud_api_workspace() {
    let backend = rerank_backend(Duration::ZERO).await;
    let cloud_api = cloud_api().await;
    let limiter = limiter(1, 8, 1);
    let app = build_test_app(
        &backend.uri(),
        TestAppOptions {
            cloud_api_url: Some(cloud_api.uri()),
            rerank_limiter: Some(limiter.clone()),
            ..Default::default()
        },
    );

    let held = limiter
        .try_acquire(Tenant::Workspace {
            org_id: "org-a".into(),
            workspace_id: "ws-a".into(),
        })
        .unwrap();
    // Both keys of workspace A share its pool.
    for key in [KEY_A, KEY_A2] {
        assert_refused(app.clone().oneshot(rerank_request(key)).await.unwrap()).await;
    }
    assert_eq!(backend_requests(&backend).await, 0);

    // Workspace B is unaffected, although its request claims org-a/ws-a in
    // headers.
    let response = app.clone().oneshot(rerank_request(KEY_B)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // The trusted pool is separate from the direct-key pools.
    let response = app
        .clone()
        .oneshot(rerank_request("test-token"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(backend_requests(&backend).await, 2);

    drop(held);
    let response = app.oneshot(rerank_request(KEY_A2)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(backend_requests(&backend).await, 3);
    assert_eq!(limiter.in_flight(), (0, 0, 0));
}

#[tokio::test]
async fn direct_key_ceiling_spans_workspaces() {
    let backend = rerank_backend(Duration::ZERO).await;
    let cloud_api = cloud_api().await;
    let limiter = limiter(4, 1, 1);
    let app = build_test_app(
        &backend.uri(),
        TestAppOptions {
            cloud_api_url: Some(cloud_api.uri()),
            rerank_limiter: Some(limiter.clone()),
            ..Default::default()
        },
    );

    let _held = limiter
        .try_acquire(Tenant::Workspace {
            org_id: "org-other".into(),
            workspace_id: "ws-other".into(),
        })
        .unwrap();
    assert_refused(app.oneshot(rerank_request(KEY_B)).await.unwrap()).await;
    assert_eq!(backend_requests(&backend).await, 0);
}

#[tokio::test]
async fn permit_is_held_until_the_upstream_response_completes() {
    let backend = rerank_backend(Duration::from_secs(3)).await;
    let limiter = limiter(1, 1, 1);
    let app = build_test_app(
        &backend.uri(),
        TestAppOptions {
            rerank_limiter: Some(limiter.clone()),
            ..Default::default()
        },
    );

    let first = tokio::spawn(app.clone().oneshot(rerank_request("test-token")));
    // Wait until the first request holds its permit (no fixed sleep).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while limiter.in_flight().0 == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "first request never took a permit"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(limiter.in_flight(), (1, 0, 0));

    assert_refused(
        app.clone()
            .oneshot(rerank_request("test-token"))
            .await
            .unwrap(),
    )
    .await;

    let response = first.await.unwrap().unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // Only the admitted request reached the backend.
    assert_eq!(backend_requests(&backend).await, 1);
    assert_eq!(limiter.in_flight(), (0, 0, 0));
}

#[tokio::test]
async fn default_limits_admit_a_large_legitimate_request() {
    let backend = rerank_backend(Duration::ZERO).await;
    let app = build_test_app(&backend.uri(), TestAppOptions::default());
    let documents = vec!["a"; 2_000];
    let body = serde_json::to_vec(&serde_json::json!({
        "model": "test-model",
        "query": "q",
        "documents": documents,
    }))
    .unwrap();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/rerank")
                .header("authorization", "Bearer test-token")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(backend_requests(&backend).await, 1);
}
