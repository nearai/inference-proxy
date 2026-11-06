//! Request-sized admission, including encrypted input at the trusted counting boundary.
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};
mod common;

fn request(path: &str) -> axum::http::request::Builder {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
}
async fn json_body(response: axum::response::Response) -> Value {
    serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
}
async fn auth(server: &MockServer) {
    Mock::given(path("/v1/check_api_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "organization_id":"org", "workspace_id":"workspace", "api_key_id":"key"
        })))
        .mount(server)
        .await;
}

#[tokio::test]
async fn private_counter_decrypts_v3_and_returns_only_metadata() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/e2ee-v3.json")).unwrap();
    for vector in fixture["vectors"].as_array().unwrap() {
        let backend = MockServer::start().await;
        let plaintext = vector["plaintext"].clone();
        let expected = plaintext.clone();
        Mock::given(path("/tokenize"))
            .respond_with(move |r: &wiremock::Request| {
                let body: Value = serde_json::from_slice(&r.body).unwrap();
                assert_eq!(body["messages"][0]["content"], expected);
                assert_eq!(body["add_generation_prompt"], true);
                ResponseTemplate::new(200).set_body_json(
                    json!({"count":17, "tokens":[100,200], "prompt":"must not escape"}),
                )
            })
            .expect(1)
            .mount(&backend)
            .await;
        let app = common::build_test_app(&backend.uri(), Default::default());
        let input = json!({"kind":"chat", "request":{"model":"test-model","messages":[{"role":"user","content":vector["ciphertext"]}]}});
        let response = app
            .clone()
            .oneshot(
                request("/v1/internal/input-tokens")
                    .header("authorization", "Bearer test-token")
                    .header("x-signing-algo", vector["signing_algo"].as_str().unwrap())
                    .header("x-client-pub-key", vector["reply_public"].as_str().unwrap())
                    .header("x-encryption-version", "3")
                    .body(Body::from(input.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(json_body(response).await, json!({"input_tokens":17}));
        let wrong_key = app
            .oneshot(
                request("/v1/internal/input-tokens")
                    .header("authorization", "Bearer test-token")
                    .header("x-signing-algo", vector["signing_algo"].as_str().unwrap())
                    .header("x-client-pub-key", vector["model_public"].as_str().unwrap())
                    .header("x-encryption-version", "3")
                    .body(Body::from(input.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(wrong_key.status(), StatusCode::BAD_REQUEST);
    }
}

#[tokio::test]
async fn customer_key_cannot_call_the_private_counter_even_with_delegation_headers() {
    let backend = MockServer::start().await;
    let app = common::build_test_app(&backend.uri(), Default::default());
    let response = app
        .oneshot(
            request("/v1/internal/input-tokens")
                .header("authorization", "Bearer sk-customer")
                .header("x-nearai-billing", "per_attempt_v1")
                .body(Body::from("invalid JSON"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(backend.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn private_counter_enforces_content_and_image_policy_before_engine_access() {
    let backend = MockServer::start().await;
    let app = common::build_test_app(
        &backend.uri(),
        common::TestAppOptions {
            validate_images: true,
            rejected_content_part_types: vec!["video_url".into()],
            ..Default::default()
        },
    );
    for part in [
        json!({"type":"image_url","image_url":{"url":"http://127.0.0.1/private"}}),
        json!({"type":"image_url","image_url":{"url":"file:///private-image"}}),
        json!({"type":"video_url","video_url":{"url":"https://example.com/video"}}),
    ] {
        let response = app
            .clone()
            .oneshot(
                request("/v1/internal/input-tokens")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(
                        json!({"kind":"chat","request":{
                            "model":"test-model","messages":[{"role":"user","content":[part]}]
                        }})
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{part}");
    }
    assert!(backend.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn counts_batches_and_repeated_pair_inputs_before_admission() {
    for (kind, input, calls) in [
        ("completion", json!({"prompt":["one","two"]}), 2),
        ("embedding", json!({"input":["one","two","three"]}), 3),
        ("privacy", json!({"input":["one","two"]}), 2),
        (
            "rerank",
            json!({"query":"query", "documents":["one","two"]}),
            4,
        ),
        (
            "score",
            json!({"text_1":"query", "text_2":["one","two"]}),
            4,
        ),
    ] {
        let backend = MockServer::start().await;
        Mock::given(path("/tokenize"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"count":11})))
            .expect(calls)
            .mount(&backend)
            .await;
        let app = common::build_test_app(&backend.uri(), Default::default());
        let response = app
            .oneshot(
                request("/v1/internal/input-tokens")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(json!({"kind":kind,"request":input}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{kind}");
        assert_eq!(
            json_body(response).await["input_tokens"],
            calls * 11,
            "{kind}"
        );
    }
}

#[tokio::test]
async fn malformed_counter_never_reserves_or_executes_and_does_not_reflect_content() {
    for response in [
        json!({"count":-1}),
        json!({"count":2147483648_i64}),
        json!({"count":"17"}),
        json!({"prompt":"sensitive marker"}),
    ] {
        let backend = MockServer::start().await;
        let cloud = MockServer::start().await;
        auth(&cloud).await;
        Mock::given(path("/tokenize"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response))
            .expect(1)
            .mount(&backend)
            .await;
        let app = common::build_test_app(
            &backend.uri(),
            common::TestAppOptions {
                cloud_api_url: Some(cloud.uri()),
                ..Default::default()
            },
        );
        let response = app.oneshot(request("/v1/chat/completions")
            .header("authorization", "Bearer sk-customer")
            .body(Body::from(json!({"model":"test-model","messages":[{"role":"user","content":"secret"}],"max_tokens":9}).to_string())).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(!json_body(response)
            .await
            .to_string()
            .contains("sensitive marker"));
        assert!(cloud
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|r| r.url.path() == "/v1/check_api_key"));
        assert!(backend
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|r| r.url.path() == "/tokenize"));
    }
}

#[tokio::test]
async fn admission_uses_backend_count_and_enforces_the_output_ceiling() {
    for delegated in [false, true] {
        let backend = MockServer::start().await;
        let cloud = MockServer::start().await;
        auth(&cloud).await;
        Mock::given(path("/tokenize"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"count":7})))
            .expect(1)
            .mount(&backend)
            .await;
        Mock::given(path("/v1/internal/budget"))
            .respond_with(move |r: &wiremock::Request| {
                let b: Value = serde_json::from_slice(&r.body).unwrap();
                assert_eq!(b["input_units"], 7);
                assert_eq!(b["max_output_tokens"], 9);
                assert_eq!(b["choices"], 3);
                assert_eq!(b["executions"], 1);
                assert!(b.get("messages").is_none());
                if !delegated {
                    assert_eq!(b["api_key_id"], "key");
                }
                ResponseTemplate::new(200)
                    .set_body_json(json!({"id":b["id"],"model":"test-model","max_output_tokens":9}))
            })
            .expect(1)
            .mount(&cloud)
            .await;
        Mock::given(method("POST")).and(path("/v1/chat/completions")).respond_with(|r:&wiremock::Request| {
            let b:Value=serde_json::from_slice(&r.body).unwrap(); assert_eq!(b["max_tokens"],9);
            assert!(!r.headers.contains_key("x-nearai-billing-key"));
            ResponseTemplate::new(200).set_body_json(json!({"id":"complete","choices":[{"index":0,"message":{"role":"assistant","content":"yes"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":1}}))
        }).expect(1).mount(&backend).await;
        let app = common::build_test_app(
            &backend.uri(),
            common::TestAppOptions {
                cloud_api_url: Some(cloud.uri()),
                ..Default::default()
            },
        );
        let identity = "00000000-0000-0000-0000-000000000001";
        let response=app.oneshot(request("/v1/chat/completions")
            .header("authorization",if delegated {"Bearer test-token"} else {"Bearer sk-customer"})
            .header("x-nearai-billing","per_attempt_v1")
            .header("x-nearai-billing-org",identity).header("x-nearai-billing-workspace",identity).header("x-nearai-billing-key",identity)
            .body(Body::from(json!({"model":"test-model","messages":[{"role":"user","content":"short"}],"max_tokens":9,"best_of":3,"input_units":1}).to_string())).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        json_body(response).await;
    }
}

#[tokio::test]
async fn encrypted_counting_covers_every_existing_token_input_shape() {
    use vllm_proxy_rs::{
        encryption::{encrypt_string, EncryptionAlgo, EncryptionContext},
        signing,
    };
    let server = common::test_signing_pair();
    let sender = signing::SigningPair {
        ecdsa: signing::EcdsaContext::from_key_bytes(&[7; 32]).unwrap(),
        ed25519: signing::Ed25519Context::from_key_bytes(&[7; 32]).unwrap(),
    };
    for version in [1, 2] {
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: hex::decode(&server.ed25519.signing_public_key).unwrap(),
            version,
            encrypt_all_fields: true,
        };
        let cipher = encrypt_string("private input", &ctx, &sender).unwrap();
        for (kind, body, calls) in [
            (
                "chat",
                json!({"messages":[{"role":"user","content":cipher}]}),
                1,
            ),
            ("completion", json!({"prompt":cipher}), 1),
            ("embedding", json!({"input":[cipher,cipher]}), 2),
            ("privacy", json!({"input":cipher}), 1),
            ("rerank", json!({"query":cipher,"documents":[cipher]}), 2),
            ("score", json!({"text_1":cipher,"text_2":cipher}), 2),
        ] {
            let backend = MockServer::start().await;
            Mock::given(path("/tokenize"))
                .respond_with(|r: &wiremock::Request| {
                    let b: Value = serde_json::from_slice(&r.body).unwrap();
                    if b.get("messages").is_some() {
                        assert_eq!(b["messages"][0]["content"], "private input");
                    } else {
                        assert_eq!(b["prompt"], "private input");
                    }
                    ResponseTemplate::new(200).set_body_json(json!({"count":7}))
                })
                .expect(calls)
                .mount(&backend)
                .await;
            let app = common::build_test_app(&backend.uri(), Default::default());
            let response = app
                .oneshot(
                    request("/v1/internal/input-tokens")
                        .header("authorization", "Bearer test-token")
                        .header("x-signing-algo", "ed25519")
                        .header("x-client-pub-key", &sender.ed25519.signing_public_key)
                        .header("x-encryption-version", version.to_string())
                        .header("x-encrypt-all-fields", "true")
                        .body(Body::from(json!({"kind":kind,"request":body}).to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{kind} v{version}");
            assert_eq!(json_body(response).await, json!({"input_tokens":calls*7}));
        }
    }
}
