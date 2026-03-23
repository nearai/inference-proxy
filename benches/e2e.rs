//! End-to-end benchmarks for attestation and completion flows.
//!
//! These benchmarks measure the proxy overhead — the time spent in our code
//! between receiving a request and forwarding/returning a response. Backend
//! latency is simulated with wiremock returning instantly.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sha2::{Digest, Sha256};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use vllm_proxy_rs::*;

// ── Test keys (same as integration tests) ──

const ECDSA_KEY: [u8; 32] = [
    0xac, 0x09, 0x74, 0xbe, 0xc3, 0x9a, 0x17, 0xe3, 0x6b, 0xa4, 0xa6, 0xb4, 0xd2, 0x38, 0xff,
    0x94, 0x4b, 0xac, 0xb3, 0x5e, 0x5d, 0xc4, 0xaf, 0x0f, 0x33, 0x47, 0xe5, 0x87, 0x31, 0x79,
    0x67, 0x0f,
];

const ED25519_KEY: [u8; 32] = [
    0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c,
    0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae,
    0x7f, 0x60,
];

fn build_test_app(mock_url: &str) -> axum::Router {
    let base = mock_url.trim_end_matches('/');

    let config = config::Config {
        model_name: "bench-model".to_string(),
        token: "bench-token".to_string(),
        vllm_base_url: mock_url.to_string(),
        chat_completions_url: format!("{base}/v1/chat/completions"),
        completions_url: format!("{base}/v1/completions"),
        tokenize_url: format!("{base}/tokenize"),
        metrics_url: format!("{base}/metrics"),
        models_url: format!("{base}/v1/models"),
        images_url: format!("{base}/v1/images/generations"),
        images_edits_url: format!("{base}/v1/images/edits"),
        transcriptions_url: format!("{base}/v1/audio/transcriptions"),
        embeddings_url: format!("{base}/v1/embeddings"),
        rerank_url: format!("{base}/v1/rerank"),
        score_url: format!("{base}/v1/score"),
        max_keepalive: 100,
        max_request_size: 1024 * 1024,
        max_image_request_size: 5 * 1024 * 1024,
        max_audio_request_size: 10 * 1024 * 1024,
        chat_cache_expiration_secs: 1200,
        attestation_cache_ttl_secs: 300,
        dev_mode: true,
        gpu_no_hw_mode: true,
        git_rev: "bench".to_string(),
        rate_limit_per_second: 10000,
        rate_limit_burst_size: 20000,
        rate_limit_trust_proxy_headers: false,
        cloud_api_url: None,
        tls_cert_path: None,
        timeout_secs: 30,
        timeout_tokenize_secs: 5,
        openai_chat_compatibility_check_enabled: false,
        startup_check_retries: 0,
        startup_check_retry_delay_secs: 0,
        startup_check_timeout_secs: 1,
    };

    let ecdsa = signing::EcdsaContext::from_key_bytes(&ECDSA_KEY).unwrap();
    let ed25519 = signing::Ed25519Context::from_key_bytes(&ED25519_KEY).unwrap();
    let signing_pair = signing::SigningPair { ecdsa, ed25519 };

    let chat_cache = cache::ChatCache::new("bench-model", 1200);
    let http_client = reqwest::Client::new();

    let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();

    let state = AppState {
        config: Arc::new(config),
        signing: Arc::new(signing_pair),
        cache: Arc::new(chat_cache),
        attestation_cache: Arc::new(attestation::AttestationCache::new(300)),
        http_client,
        metrics_handle,
        tls_cert_fingerprint: None,
    };

    let rate_limiter = rate_limit::build_rate_limiter(10000, 20000);
    let rate_limit_state = rate_limit::RateLimitState {
        limiter: rate_limiter,
        trust_proxy_headers: false,
    };

    routes::build_router()
        .layer(middleware::from_fn(rate_limit::rate_limit_middleware))
        .layer(axum::Extension(rate_limit_state))
        .layer(middleware::from_fn(request_id_middleware))
        .with_state(state)
}

// ── Helpers ──

fn make_chat_request(messages: usize) -> String {
    let msgs: Vec<serde_json::Value> = (0..messages)
        .map(|i| {
            serde_json::json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("Message number {i} with some typical content that a user might send in a conversation.")
            })
        })
        .collect();
    serde_json::json!({
        "model": "bench-model",
        "messages": msgs,
        "stream": false
    })
    .to_string()
}

fn make_chat_response(id: &str) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": 1234567890,
        "model": "bench-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a typical assistant response with some content that would be returned from the model."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80
        }
    })
    .to_string()
}

fn make_streaming_response(id: &str, num_chunks: usize) -> String {
    let mut body = String::new();
    for i in 0..num_chunks {
        let chunk = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "bench-model",
            "choices": [{
                "index": 0,
                "delta": { "content": format!("word{i} ") },
                "finish_reason": null
            }]
        });
        body.push_str(&format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()));
    }
    // Final chunk with usage
    let final_chunk = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "bench-model",
        "choices": [],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": num_chunks as i64,
            "total_tokens": 50 + num_chunks as i64
        }
    });
    body.push_str(&format!(
        "data: {}\n\n",
        serde_json::to_string(&final_chunk).unwrap()
    ));
    body.push_str("data: [DONE]\n\n");
    body
}

// ── Attestation benchmarks ──

fn bench_attestation_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("attestation_cache");
    let rt = tokio::runtime::Runtime::new().unwrap();

    let cache = attestation::AttestationCache::new(300);

    // Pre-populate cache
    let report = types::AttestationReport {
        model_name: "bench-model".to_string(),
        signing_address: "0xabcdef1234567890".to_string(),
        signing_algo: "ecdsa".to_string(),
        signing_public_key: "04abcdef".to_string(),
        request_nonce: hex::encode([0xAA; 32]),
        intel_quote: "base64quote".repeat(100),
        nvidia_payload: serde_json::json!({
            "nonce": hex::encode([0xAA; 32]),
            "evidence_list": [{"gpu": "H100"}],
            "arch": "HOPPER"
        })
        .to_string(),
        event_log: serde_json::json!({"entries": []}),
        info: serde_json::json!({"version": "1.0"}),
        tls_cert_fingerprint: None,
    };
    rt.block_on(cache.set("ecdsa", false, report.clone()));

    group.bench_function("cache_hit", |b| {
        b.to_async(&rt).iter(|| async {
            black_box(cache.get("ecdsa", false).await)
        })
    });

    group.bench_function("cache_miss", |b| {
        b.to_async(&rt).iter(|| async {
            black_box(cache.get("ed25519", true).await)
        })
    });

    group.bench_function("cache_set", |b| {
        b.to_async(&rt).iter(|| async {
            cache.set("ecdsa", false, report.clone()).await
        })
    });

    group.bench_function("semaphore_acquire_uncontended", |b| {
        b.to_async(&rt).iter(|| async {
            let permit = cache.acquire_gpu_permit().await;
            drop(black_box(permit));
        })
    });

    group.finish();
}

fn bench_attestation_report_serialization(c: &mut Criterion) {
    let report = types::AttestationReport {
        model_name: "bench-model".to_string(),
        signing_address: "0xabcdef1234567890abcdef1234567890abcdef12".to_string(),
        signing_algo: "ecdsa".to_string(),
        signing_public_key: hex::encode([0xAB; 65]),
        request_nonce: hex::encode([0xCD; 32]),
        intel_quote: "base64encodedquotedata".repeat(50),
        nvidia_payload: serde_json::json!({
            "nonce": hex::encode([0xCD; 32]),
            "evidence_list": [
                {"gpu": "H100", "evidence": "base64data".repeat(20)},
                {"gpu": "H100", "evidence": "base64data".repeat(20)},
            ],
            "arch": "HOPPER"
        })
        .to_string(),
        event_log: serde_json::json!({"entries": [
            {"type": "init", "data": "some event"},
            {"type": "measure", "data": "another event"},
        ]}),
        info: serde_json::json!({"version": "1.0", "tcb": "123"}),
        tls_cert_fingerprint: None,
    };

    let response = types::AttestationResponse {
        report: report.clone(),
        all_attestations: vec![report],
    };

    c.bench_function("attestation_response_serialize", |b| {
        b.iter(|| serde_json::to_value(black_box(&response)).unwrap())
    });
}

// ── Completion flow benchmarks (end-to-end through the proxy) ──

fn bench_json_completion_e2e(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_completion_e2e");
    let rt = tokio::runtime::Runtime::new().unwrap();

    for msg_count in [1, 5, 20] {
        let request_body = make_chat_request(msg_count);
        let response_body = make_chat_response(&format!("chatcmpl-bench-{msg_count}"));

        // Set up mock server once per parameter
        let mock_server = rt.block_on(async {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(&response_body)
                        .insert_header("content-type", "application/json"),
                )
                .mount(&server)
                .await;
            server
        });

        let mock_uri = mock_server.uri();

        let app = build_test_app(&mock_uri);

        group.bench_with_input(
            BenchmarkId::new("messages", msg_count),
            &request_body,
            |b, req_body| {
                b.to_async(&rt).iter(|| {
                    let req_body = req_body.clone();
                    let app = app.clone();
                    async move {
                        let request = Request::builder()
                            .method("POST")
                            .uri("/v1/chat/completions")
                            .header("content-type", "application/json")
                            .header("authorization", "Bearer bench-token")
                            .body(Body::from(req_body))
                            .unwrap();

                        let response = app.oneshot(request).await.unwrap();
                        let status = response.status();
                        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                            .await
                            .unwrap();
                        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
                        black_box(body)
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_streaming_completion_e2e(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_completion_e2e");
    // Give streaming benchmarks more time since they involve spawned tasks
    group.sample_size(50);
    let rt = tokio::runtime::Runtime::new().unwrap();

    for chunk_count in [5, 20, 50] {
        let request_body = {
            let mut v: serde_json::Value = serde_json::from_str(&make_chat_request(3)).unwrap();
            v["stream"] = serde_json::Value::Bool(true);
            serde_json::to_string(&v).unwrap()
        };
        let response_body = make_streaming_response("chatcmpl-stream-bench", chunk_count);

        let mock_server = rt.block_on(async {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_string(&response_body)
                        .insert_header("content-type", "text/event-stream"),
                )
                .mount(&server)
                .await;
            server
        });

        let mock_uri = mock_server.uri();
        let app = build_test_app(&mock_uri);

        group.bench_with_input(
            BenchmarkId::new("chunks", chunk_count),
            &request_body,
            |b, req_body| {
                b.to_async(&rt).iter(|| {
                    let req_body = req_body.clone();
                    let app = app.clone();
                    async move {
                        let request = Request::builder()
                            .method("POST")
                            .uri("/v1/chat/completions")
                            .header("content-type", "application/json")
                            .header("authorization", "Bearer bench-token")
                            .body(Body::from(req_body))
                            .unwrap();

                        let response = app.oneshot(request).await.unwrap();
                        assert_eq!(response.status(), StatusCode::OK);
                        let body = axum::body::to_bytes(response.into_body(), 10 * 1024 * 1024)
                            .await
                            .unwrap();
                        black_box(body)
                    }
                })
            },
        );
    }

    group.finish();
}

// ── Proxy overhead component benchmarks ──

fn bench_request_body_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("request_body_processing");

    for msg_count in [1, 5, 20, 50] {
        let body = make_chat_request(msg_count);
        let body_bytes = body.as_bytes();

        group.bench_with_input(
            BenchmarkId::new("sha256_hash", msg_count),
            body_bytes,
            |b, data| {
                b.iter(|| hex::encode(Sha256::digest(black_box(data))))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("json_parse", msg_count),
            &body,
            |b, data| {
                b.iter(|| {
                    let v: serde_json::Value = serde_json::from_str(black_box(data)).unwrap();
                    black_box(v)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("json_parse_and_reserialize", msg_count),
            &body,
            |b, data| {
                b.iter(|| {
                    let v: serde_json::Value = serde_json::from_str(black_box(data)).unwrap();
                    let out = serde_json::to_vec(&v).unwrap();
                    black_box(out)
                })
            },
        );
    }

    group.finish();
}

fn bench_response_signing_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("response_signing_full");

    let ecdsa = signing::EcdsaContext::from_key_bytes(&ECDSA_KEY).unwrap();
    let ed25519 = signing::Ed25519Context::from_key_bytes(&ED25519_KEY).unwrap();
    let pair = Arc::new(signing::SigningPair { ecdsa, ed25519 });
    let cache = Arc::new(cache::ChatCache::new("bench-model", 1200));

    // Simulate the full response processing pipeline:
    // parse JSON -> extract ID -> hash -> sign -> serialize signature -> cache
    for response_size in ["small", "medium", "large"] {
        let response_body = match response_size {
            "small" => serde_json::json!({
                "id": "chatcmpl-bench",
                "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
            }),
            "medium" => serde_json::json!({
                "id": "chatcmpl-bench",
                "choices": [{"message": {"content": "x".repeat(1000)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250}
            }),
            "large" => serde_json::json!({
                "id": "chatcmpl-bench",
                "choices": [{"message": {"content": "x".repeat(10000)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 2000, "total_tokens": 2500}
            }),
            _ => unreachable!(),
        };
        let response_bytes = serde_json::to_vec(&response_body).unwrap();
        let request_sha256 = hex::encode(Sha256::digest(b"test request"));

        group.bench_with_input(
            BenchmarkId::new("pipeline", response_size),
            &(response_bytes, request_sha256),
            |b, (resp_bytes, req_hash)| {
                let pair = pair.clone();
                let cache = cache.clone();
                b.iter(|| {
                    // 1. Parse response JSON
                    let response_data: serde_json::Value =
                        serde_json::from_slice(black_box(resp_bytes)).unwrap();

                    // 2. Extract ID
                    let chat_id = response_data["id"].as_str().unwrap();

                    // 3. Serialize to final form
                    let final_body = serde_json::to_string(&response_data).unwrap();

                    // 4. Hash response
                    let response_sha256 = hex::encode(Sha256::digest(final_body.as_bytes()));

                    // 5. Sign
                    let text = format!("bench-model:{req_hash}:{response_sha256}");
                    let signed = pair.sign_chat(&text).unwrap();

                    // 6. Serialize signature and cache
                    let signed_json = serde_json::to_string(&signed).unwrap();
                    cache.set_chat(chat_id, &signed_json);

                    black_box(final_body)
                })
            },
        );
    }

    group.finish();
}

fn bench_streaming_sse_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_sse_processing");

    // Benchmark the SSE parser + hasher pipeline (what runs per-chunk in streaming)
    for chunk_count in [10, 50, 200] {
        let chunks: Vec<Vec<u8>> = (0..chunk_count)
            .map(|i| {
                let chunk = serde_json::json!({
                    "id": "chatcmpl-bench",
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": format!("word{i} ")}}]
                });
                format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()).into_bytes()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("parse_and_hash", chunk_count),
            &chunks,
            |b, chunks| {
                b.iter(|| {
                    let mut parser = proxy::SseParser::new();
                    let mut hasher = Sha256::new();
                    for chunk in chunks {
                        parser.process_chunk(black_box(chunk));
                        hasher.update(chunk);
                    }
                    // Final sign text construction
                    let response_sha256 = hex::encode(hasher.finalize());
                    black_box(response_sha256);
                    black_box(&parser.chat_id);
                })
            },
        );
    }

    group.finish();
}

fn bench_auth_token_comparison(c: &mut Criterion) {
    use subtle::ConstantTimeEq;

    let token = "rr9w3S91rog35JM6Sgr2YqwbMvKrbnLA95hQoiwip+4=";
    let matching = "rr9w3S91rog35JM6Sgr2YqwbMvKrbnLA95hQoiwip+4=";
    let non_matching = "xx9w3S91rog35JM6Sgr2YqwbMvKrbnLA95hQoiwip+4=";

    let mut group = c.benchmark_group("auth_token");

    group.bench_function("constant_time_match", |b| {
        b.iter(|| {
            let result: bool = token
                .as_bytes()
                .ct_eq(black_box(matching.as_bytes()))
                .into();
            black_box(result)
        })
    });

    group.bench_function("constant_time_mismatch", |b| {
        b.iter(|| {
            let result: bool = token
                .as_bytes()
                .ct_eq(black_box(non_matching.as_bytes()))
                .into();
            black_box(result)
        })
    });

    group.finish();
}

fn bench_json_body_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_round_trip");

    // This measures the overhead of parsing request body, modifying it
    // (strip_empty_tool_calls, force stream_options), and re-serializing.
    let body_with_tools = serde_json::json!({
        "model": "bench-model",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "tool_calls": []},
            {"role": "user", "content": "How are you?"}
        ],
        "stream": true,
        "stream_options": {"include_usage": false}
    });
    let body_str = serde_json::to_string(&body_with_tools).unwrap();

    group.bench_function("parse_modify_reserialize", |b| {
        b.iter(|| {
            let mut v: serde_json::Value =
                serde_json::from_str(black_box(&body_str)).unwrap();

            // strip_empty_tool_calls equivalent
            if let Some(messages) = v.get_mut("messages").and_then(|m| m.as_array_mut()) {
                for message in messages.iter_mut() {
                    if let Some(obj) = message.as_object_mut() {
                        if let Some(tool_calls) = obj.get("tool_calls") {
                            if tool_calls.as_array().map(|a| a.is_empty()).unwrap_or(false) {
                                obj.remove("tool_calls");
                            }
                        }
                    }
                }
            }

            // Force stream_options
            if let Some(stream_opts) = v.get_mut("stream_options").and_then(|v| v.as_object_mut()) {
                stream_opts.insert("include_usage".into(), true.into());
            }

            let out = serde_json::to_vec(&v).unwrap();
            black_box(out)
        })
    });

    group.finish();
}

// ── Criterion configuration ──

criterion_group!(
    attestation,
    bench_attestation_cache_operations,
    bench_attestation_report_serialization,
);

criterion_group!(
    completion,
    bench_json_completion_e2e,
    bench_streaming_completion_e2e,
);

criterion_group!(
    proxy_overhead,
    bench_request_body_processing,
    bench_response_signing_full,
    bench_streaming_sse_processing,
    bench_auth_token_comparison,
    bench_json_body_round_trip,
);

criterion_main!(attestation, completion, proxy_overhead);
