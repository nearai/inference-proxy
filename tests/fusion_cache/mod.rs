use super::*;
use serde_json::{json, Value};
use wiremock::matchers::{body_partial_json, body_string_contains};

#[tokio::test]
async fn fusion_cache_json_reports_aggregate_once() {
    assert_fusion_cache_report(false).await;
}

#[tokio::test]
async fn fusion_cache_stream_reports_aggregate_once() {
    assert_fusion_cache_report(true).await;
}

async fn assert_fusion_cache_report(stream: bool) {
    let backend = MockServer::start().await;
    let cloud_api = MockServer::start().await;
    mount_fusion_components(&backend).await;
    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "organization_id": "org-test",
            "workspace_id": "ws-test",
            "api_key_id": "key-test"
        })))
        .expect(1)
        .mount(&cloud_api)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/internal/usage"))
        .and(header("authorization", "Bearer test-usage-token"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&cloud_api)
        .await;

    let app = build_test_app_inner_with_fusion(
        &backend.uri(),
        TestAppOptions {
            fusion_enabled: true,
            fusion_endpoints_url: Some(format!("{}/endpoints", backend.uri())),
            cloud_api_url: Some(cloud_api.uri()),
            ..Default::default()
        },
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer sk-test-valid-key-12345678901")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Pick a color"}],
                        "tools": [{
                            "type": "openrouter:fusion",
                            "parameters": {
                                "analysis_models": ["panel-a", "panel-b"],
                                "model": "panel-a"
                            }
                        }],
                        "tool_choice": "required",
                        "stream": stream
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_usage = if stream {
        let body = String::from_utf8(body_to_bytes(response).await).unwrap();
        assert!(body.ends_with("data: [DONE]\n\n"));
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .find_map(|chunk| chunk.get("usage").cloned())
            .expect("fusion stream should contain its aggregate usage")
    } else {
        let body = body_to_json(response).await;
        assert_eq!(body["id"], "chatcmpl-fusion-cache");
        body["usage"].clone()
    };

    wait_for_usage_request(&cloud_api, 1).await;
    let reports = get_usage_requests(&cloud_api).await;
    assert_eq!(reports.len(), 1);
    assert_eq!(
        reports[0],
        json!({
            "type": "chat_completion",
            "model": "test-model",
            "input_tokens": 71,
            "output_tokens": 15,
            "cache_read_tokens": 52,
            "id": "chatcmpl-fusion-cache",
            "organization_id": "org-test",
            "workspace_id": "ws-test",
            "api_key_id": "key-test"
        })
    );
    assert_eq!(response_usage["prompt_tokens"], 71);
    assert_eq!(response_usage["completion_tokens"], 15);
    assert_eq!(response_usage["total_tokens"], 86);
    assert_eq!(response_usage["prompt_tokens_details"]["cached_tokens"], 52);
    cloud_api.verify().await;
    backend.verify().await;
}

async fn mount_fusion_components(backend: &MockServer) {
    Mock::given(method("GET"))
        .and(path("/endpoints"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "endpoints": [{"domain": backend.uri(), "models": ["panel-a", "panel-b"]}]
        })))
        .expect(1)
        .mount(backend)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/attestation/report"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .mount(backend)
        .await;
    for (model, input, output, cached) in [("panel-a", 10, 2, 6), ("panel-b", 11, 3, 7)] {
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer fusion-token"))
            .and(body_partial_json(json!({"model": model})))
            .and(body_string_contains(
                "one member of a private multi-model panel",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(component_response(
                "Blue.",
                json!({
                    "prompt_tokens": input,
                    "completion_tokens": output,
                    "prompt_tokens_details": {"cached_tokens": cached}
                }),
            )))
            .expect(1)
            .mount(backend)
            .await;
    }
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer fusion-token"))
        .and(body_string_contains("strict JSON only"))
        .respond_with(ResponseTemplate::new(200).set_body_json(component_response(
            r#"{"consensus":"blue","disagreements":[],"strengths":[],"risks":[],"synthesis_guidance":"answer blue"}"#,
            json!({
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 15}
            }),
        )))
        .expect(1)
        .mount(backend)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({"model": "test-model"})))
        .and(body_string_contains("final synthesis model"))
        .respond_with(ResponseTemplate::new(200).set_body_json(component_response(
            "Blue.",
            json!({
                "prompt_tokens": 30,
                "completion_tokens": 6,
                "prompt_tokens_details": {"cached_tokens": 24}
            }),
        )))
        .expect(1)
        .mount(backend)
        .await;
}

fn component_response(content: &str, usage: Value) -> Value {
    json!({
        "id": "chatcmpl-fusion-cache",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": usage
    })
}
