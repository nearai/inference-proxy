//! Standalone media keeps its original payload and does not depend on text admission.
mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn cloud_key_audio_preserves_upload_and_format_without_budget_service() {
    let backend = MockServer::start().await;
    let cloud = MockServer::start().await;
    Mock::given(path("/v1/check_api_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "organization_id": "org-test", "workspace_id": "ws-test", "api_key_id": "key-test"
        })))
        .mount(&cloud)
        .await;
    Mock::given(path("/v1/internal/budget"))
        .respond_with(ResponseTemplate::new(503))
        .expect(0)
        .mount(&cloud)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "audio-original", "text": "transcript"
        })))
        .expect(1)
        .mount(&backend)
        .await;
    let app = build_test_app(
        &backend.uri(),
        TestAppOptions {
            cloud_api_url: Some(cloud.uri()),
            ..Default::default()
        },
    );
    let body = concat!(
        "--media\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ntest-model\r\n",
        "--media\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n",
        "--media\r\nContent-Disposition: form-data; name=\"file\"; filename=\"original.webm\"\r\n",
        "Content-Type: audio/webm\r\n\r\nopaque original audio bytes\0\r\n--media--\r\n"
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header("authorization", "Bearer sk-test-customer")
                .header("content-type", "multipart/form-data; boundary=media")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let requests = backend.received_requests().await.unwrap();
    let upload = &requests[0].body;
    for original in [
        b"opaque original audio bytes\0".as_slice(),
        b"filename=\"original.webm\"",
        b"audio/webm",
        b"name=\"response_format\"\r\n\r\njson",
    ] {
        assert!(upload.windows(original.len()).any(|part| part == original));
    }
    assert!(!upload
        .windows(b"audio.flac".len())
        .any(|part| part == b"audio.flac"));
}

#[tokio::test]
async fn cloud_key_non_text_json_keeps_payload_without_budget_service() {
    use serde_json::json;
    let backend = MockServer::start().await;
    let cloud = MockServer::start().await;
    Mock::given(path("/v1/check_api_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "organization_id": "org-test", "workspace_id": "ws-test", "api_key_id": "key-test"
        })))
        .mount(&cloud)
        .await;
    Mock::given(path("/v1/internal/budget"))
        .respond_with(ResponseTemplate::new(503))
        .expect(0)
        .mount(&cloud)
        .await;
    let app = build_test_app(
        &backend.uri(),
        TestAppOptions {
            cloud_api_url: Some(cloud.uri()),
            ..Default::default()
        },
    );
    for (route, fields) in [
        ("/v1/embeddings", json!({"input": ["one", "two"]})),
        ("/v1/rerank", json!({"query": "one", "documents": ["two"]})),
        ("/v1/score", json!({"text_1": "one", "text_2": "two"})),
        ("/v1/images/generations", json!({"prompt": "a cat", "n": 2})),
    ] {
        let mut payload = fields;
        payload["model"] = json!("test-model");
        Mock::given(method("POST"))
            .and(path(route))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "original-result", "data": [], "results": [],
                "usage": {"prompt_tokens": 2, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&backend)
            .await;
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(route)
                    .header("authorization", "Bearer sk-test-customer")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{route}");
        let requests = backend.received_requests().await.unwrap();
        let forwarded = requests.iter().find(|r| r.url.path() == route).unwrap();
        let forwarded: serde_json::Value = serde_json::from_slice(&forwarded.body).unwrap();
        for (field, value) in payload.as_object().unwrap() {
            assert_eq!(&forwarded[field], value, "{route}: {field}");
        }
        assert!(forwarded.get("max_tokens").is_none());
        assert!(forwarded.get("max_completion_tokens").is_none());
    }
}
