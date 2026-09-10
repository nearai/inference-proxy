use super::*;
use serde_json::{json, Value};

#[test]
fn direct_cache_json_normalizes_cached_subset() {
    for (detail, expected) in [
        (json!({"cached_tokens": 1856}), 1856),
        (json!({}), 0),
        (Value::Null, 0),
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
        let response = json!({"usage": {"prompt_tokens": 1858,
            "completion_tokens": 8, "prompt_tokens_details": detail}});
        let body = build_usage_body(&UsageType::ChatCompletion, &response, "m", "id").unwrap();
        assert_eq!(body["input_tokens"], 1858);
        assert_eq!(body["output_tokens"], 8);
        assert_eq!(body["cache_read_tokens"], expected, "detail={detail}");
    }
}

#[test]
fn direct_cache_assembler_retains_observation_for_chat_and_text() {
    for shape in [ResponseShape::ChatCompletion, ResponseShape::TextCompletion] {
        for (detail, expected) in [
            (None, 1856),
            (Some(Value::Null), 1856),
            (
                Some(json!({"cached_tokens": null, "provider_detail": 42})),
                1856,
            ),
            (Some(json!({"cached_tokens": 1.5})), 1856),
            (Some(json!({"cached_tokens": 0})), 0),
        ] {
            let mut assembler = StreamingResponseAssembler::new(shape);
            assembler.ingest_event(&json!({"id": "cache-chat", "choices": [], "usage": {
                "prompt_tokens": 1858, "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 1856}
            }}));
            let mut last = json!({"prompt_tokens": 1858, "completion_tokens": 8,
                "provider_usage": {"kept": true}});
            if let Some(detail) = detail {
                last["prompt_tokens_details"] = detail;
            }
            assembler.ingest_event(&json!({"usage": last}));
            let response = assembler.into_response("chatcmpl");
            assert_eq!(response["usage"]["completion_tokens"], 8);
            assert_eq!(response["usage"]["provider_usage"], json!({"kept": true}));
            assert_eq!(
                response["usage"]["prompt_tokens_details"]["cached_tokens"],
                expected
            );
            if last
                .pointer("/prompt_tokens_details/provider_detail")
                .is_some()
            {
                assert_eq!(
                    response["usage"]["prompt_tokens_details"]["provider_detail"],
                    42
                );
            }
            let report =
                build_usage_body(&UsageType::ChatCompletion, &response, "m", "id").unwrap();
            assert_eq!(report["cache_read_tokens"], expected);
        }
    }
}

#[test]
fn direct_cache_characterizes_primary_validation_and_non_chat_shapes() {
    let response = json!({"usage": {"prompt_tokens": -2, "completion_tokens": 8,
        "prompt_tokens_details": {"cached_tokens": 3}}});
    let body = build_usage_body(&UsageType::ChatCompletion, &response, "m", "id").unwrap();
    assert_eq!(body["input_tokens"], -2);
    assert_eq!(body["output_tokens"], 8);
    for kind in [UsageType::Embedding, UsageType::Rerank, UsageType::Score] {
        let response = json!({"usage": {"prompt_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 3}}});
        let body = build_usage_body(&kind, &response, "m", "id").unwrap();
        assert_eq!(body["input_tokens"], 8);
        assert!(body.get("cache_read_tokens").is_none());
        assert!(body.get("output_tokens").is_none());
    }
}

#[test]
fn direct_cache_sse_empty_usage_retains_accepted_snapshot() {
    let mut parser = SseParser::new();
    parser.process_chunk(b"data: {\"usage\":{\"prompt_tokens\":1858,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":1856}}}\n\n");

    parser.process_chunk(b"data: {\"usage\":{}}\n\ndata: [DONE]\n\n");

    assert_eq!(
        parser.usage,
        Some(ChatUsage {
            input_tokens: 1858,
            output_tokens: 3,
            cache_read_tokens: 1856,
        })
    );
    assert!(parser.seen_done);
}

#[test]
fn direct_cache_sse_ignored_primary_preserves_unobserved_cache() {
    for primary in [json!(0), json!(-1), Value::Null, json!(1.5), json!("0")] {
        for detail in [
            None,
            Some(Value::Null),
            Some(json!({})),
            Some(json!({"cached_tokens": null})),
            Some(json!({"cached_tokens": 1.5})),
            Some(json!({"cached_tokens": "1856"})),
        ] {
            let mut parser = SseParser::new();
            parser.process_chunk(b"data: {\"usage\":{\"prompt_tokens\":1858,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":1856}}}\n\n");
            let accepted = parser.usage;
            let mut usage = json!({"prompt_tokens": primary, "completion_tokens": primary});
            if let Some(detail) = detail {
                usage["prompt_tokens_details"] = detail;
            }

            parser.process_chunk(format!("data: {}\n\n", json!({"usage": usage})).as_bytes());

            assert_eq!(parser.usage, accepted, "ignored usage={usage}");
        }
    }
}

#[test]
fn direct_cache_sse_ignored_primary_applies_integer_cache_observations() {
    for (cached, expected) in [
        (json!(0), 0),
        (json!(-1), 0),
        (json!(2147483648_i64), 0),
        (json!(-2147483649_i64), 0),
        (json!(u64::MAX), 0),
        (json!(1000), 1000),
        (json!(2000), 1858),
    ] {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"usage\":{\"prompt_tokens\":1858,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":1856}}}\n\n");
        let usage = json!({"prompt_tokens": 0, "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": cached}});

        parser.process_chunk(format!("data: {}\n\n", json!({"usage": usage})).as_bytes());

        assert_eq!(
            parser.usage,
            Some(ChatUsage {
                input_tokens: 1858,
                output_tokens: 3,
                cache_read_tokens: expected,
            }),
            "cache observation={cached}"
        );
    }
}

#[test]
fn direct_cache_sse_accepted_snapshot_clamps_to_new_input() {
    for input in [100, 0, -1] {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"usage\":{\"prompt_tokens\":1858,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":1856}}}\n\n");
        let usage = json!({"prompt_tokens": input, "completion_tokens": 8});

        parser.process_chunk(format!("data: {}\n\n", json!({"usage": usage})).as_bytes());

        assert_eq!(
            parser.usage,
            Some(ChatUsage {
                input_tokens: input,
                output_tokens: 8,
                cache_read_tokens: input.max(0),
            })
        );
    }
}

#[test]
fn direct_cache_sse_cache_only_does_not_create_billable_snapshot() {
    let mut parser = SseParser::new();

    parser.process_chunk(b"data: {\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":1856}}}\n\ndata: [DONE]\n\n");

    assert_eq!(parser.usage, None);
    assert!(parser.seen_done);
}
