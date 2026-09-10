use super::*;

#[tokio::test]
async fn direct_cache_json_reports_glm_and_normalizes_invalid_details() {
    for (detail, expected) in [
        (json!({"cached_tokens": 1856}), 1856),
        (Value::Null, 0),
        (json!({}), 0),
        (json!({"cached_tokens": null}), 0),
        (json!({"cached_tokens": 0}), 0),
        (json!({"cached_tokens": -1}), 0),
        (json!({"cached_tokens": 2000}), 1858),
        (json!({"cached_tokens": 1.5}), 0),
        (json!({"cached_tokens": "1856"}), 0),
        (json!({"cached_tokens": 2147483648_i64}), 0),
        (json!({"cached_tokens": -2147483649_i64}), 0),
        (json!({"cached_tokens": u64::MAX}), 0),
    ] {
        let backend = MockServer::start().await;
        let cloud = cloud_fixture().await;
        Mock::given(method("POST")).and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": ID, "object": "chat.completion", "model": "test-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1858, "completion_tokens": 8, "prompt_tokens_details": detail}
            }))).expect(1).mount(&backend).await;
        let app = build_test_app_with_cloud_api(&backend.uri(), &cloud.uri());
        let response = app
            .oneshot(request("/v1/chat/completions", false))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_to_json(response).await;
        assert_eq!(body["usage"]["prompt_tokens"], 1858);
        assert_report(&cloud, expected, 8).await;
        backend.verify().await;
    }
}

#[tokio::test]
async fn direct_cache_reassembled_chat_and_text_retain_or_reset_cache() {
    for route in ["/v1/chat/completions", "/v1/completions"] {
        for (detail, expected) in [
            (None, 1856),
            (Some(Value::Null), 1856),
            (
                Some(json!({"cached_tokens": null, "provider_detail": 42})),
                1856,
            ),
            (Some(json!({"cached_tokens": 0})), 0),
        ] {
            let backend = MockServer::start().await;
            let cloud = cloud_fixture().await;
            let mut usage =
                json!({"prompt_tokens": 1858, "completion_tokens": 8, "provider_usage": 42});
            if let Some(detail) = detail {
                usage["prompt_tokens_details"] = detail;
            }
            let sse = first_frame() + &frame(usage.clone()) + "data: [DONE]\n\n";
            Mock::given(method("POST"))
                .and(path(route))
                .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
                .expect(1)
                .mount(&backend)
                .await;
            let app = build_test_app_with_cloud_api(&backend.uri(), &cloud.uri());
            let response = app.oneshot(request(route, false)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let response = body_to_json(response).await;
            assert_eq!(response["usage"]["completion_tokens"], 8);
            assert_eq!(response["usage"]["provider_usage"], 42);
            assert_eq!(
                response["usage"]["prompt_tokens_details"]["cached_tokens"],
                expected
            );
            if usage
                .pointer("/prompt_tokens_details/provider_detail")
                .is_some()
            {
                assert_eq!(
                    response["usage"]["prompt_tokens_details"]["provider_detail"],
                    42
                );
            }
            assert_report(&cloud, expected, 8).await;
            backend.verify().await;
        }
    }
}
