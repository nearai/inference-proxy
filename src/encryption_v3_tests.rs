use super::*;
use serde_json::{json, Value};

fn vectors() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("../tests/fixtures/e2ee-v3.json")).unwrap()
        ["vectors"]
        .as_array()
        .unwrap()
        .clone()
}

fn pair(seed: &[u8]) -> SigningPair {
    let seed: [u8; 32] = seed.try_into().unwrap();
    SigningPair {
        ecdsa: crate::signing::EcdsaContext::from_key_bytes(&seed).unwrap(),
        ed25519: crate::signing::Ed25519Context::from_key_bytes(&seed).unwrap(),
    }
}

fn context(vector: &Value, flag: Option<&str>) -> EncryptionContext {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        "x-signing-algo",
        vector["signing_algo"].as_str().unwrap().parse().unwrap(),
    );
    headers.insert(
        "x-client-pub-key",
        vector["reply_public"].as_str().unwrap().parse().unwrap(),
    );
    headers.insert("x-encryption-version", "3".parse().unwrap());
    if let Some(flag) = flag {
        headers.insert("x-encrypt-all-fields", flag.parse().unwrap());
    }
    extract_encryption_context(&headers).unwrap().unwrap()
}

fn encrypt_request(
    text: &str,
    field: &str,
    ctx: &EncryptionContext,
    server: &SigningPair,
) -> String {
    let aad = ctx.request_aad(field).unwrap();
    let bytes = match ctx.algo {
        EncryptionAlgo::Ecdsa => ecies::encrypt_with_aad(
            text.as_bytes(),
            &ecies::parse_client_pubkey(&hex::decode(&server.ecdsa.signing_public_key).unwrap())
                .unwrap(),
            &aad,
        )
        .unwrap(),
        EncryptionAlgo::Ed25519 => {
            let public = hex::decode(&server.ed25519.signing_public_key)
                .unwrap()
                .try_into()
                .unwrap();
            nacl_v2::encrypt_with_aad(
                text.as_bytes(),
                &nacl::ed25519_public_to_x25519(&public).unwrap(),
                &aad,
            )
            .unwrap()
        }
    };
    hex::encode(bytes)
}

#[test]
fn shared_vectors_bind_canonical_reply_key_and_reject_downgrade() {
    for vector in vectors() {
        let server = pair(&hex::decode(vector["model_secret"].as_str().unwrap()).unwrap());
        let ctx = context(&vector, None);
        let field = vector["field"].as_str().unwrap();
        assert_eq!(hex::encode(ctx.request_aad(field).unwrap()), vector["aad"]);
        let ciphertext = vector["ciphertext"].as_str().unwrap();
        assert_eq!(
            decrypt_string_for_field(ciphertext, &ctx, &server, field).unwrap(),
            vector["plaintext"]
        );
        if ctx.algo == EncryptionAlgo::Ecdsa {
            let mut raw = ctx.clone();
            raw.client_pub_key.remove(0);
            assert_eq!(
                ctx.request_aad(field).unwrap(),
                raw.request_aad(field).unwrap()
            );
            assert!(decrypt_string_for_field(ciphertext, &raw, &server, field).is_ok());
        }
        for version in [1, 2] {
            let mut downgraded = ctx.clone();
            downgraded.version = version;
            assert!(decrypt_string_for_field(ciphertext, &downgraded, &server, field).is_err());
        }
        let mut replaced = ctx.clone();
        replaced.client_pub_key = hex::decode(vector["model_public"].as_str().unwrap()).unwrap();
        assert!(decrypt_string_for_field(ciphertext, &replaced, &server, field).is_err());
    }
}

#[test]
fn v3_authenticates_every_supported_chat_field_strictly() {
    for vector in vectors() {
        let server = pair(&hex::decode(vector["model_secret"].as_str().unwrap()).unwrap());
        let ctx = context(&vector, Some("false"));
        let fields = [
            ("/messages/0/content", "messages[].content"),
            ("/messages/0/reasoning", "messages[].reasoning"),
            (
                "/messages/0/reasoning_content",
                "messages[].reasoning_content",
            ),
            ("/messages/0/name", "messages[].name"),
            ("/messages/0/refusal", "messages[].refusal"),
            ("/messages/0/audio/data", "messages[].audio.data"),
            (
                "/messages/0/tool_calls/0/function/name",
                "messages[].tool_calls[].function.name",
            ),
            (
                "/messages/0/tool_calls/0/function/arguments",
                "messages[].tool_calls[].function.arguments",
            ),
            (
                "/messages/0/function_call/name",
                "messages[].function_call.name",
            ),
            (
                "/messages/0/function_call/arguments",
                "messages[].function_call.arguments",
            ),
            ("/tools/0/function/name", "tools[].function.name"),
            (
                "/tools/0/function/description",
                "tools[].function.description",
            ),
            (
                "/tools/0/function/parameters",
                "tools[].function.parameters",
            ),
            ("/tool_choice/function/name", "tool_choice.function.name"),
            ("/function_call/name", "function_call.name"),
        ];
        let mut request = json!({
            "messages": [{
                "content": "[{\"type\":\"text\",\"text\":\"secret\"}]",
                "reasoning": "reason", "reasoning_content": "thought", "name": "name", "refusal": "refusal",
                "audio": {"data": "audio"},
                "tool_calls": [{"function": {"name": "tool", "arguments": "{}"}}],
                "function_call": {"name": "legacy", "arguments": "{}"}
            }],
            "tools": [{"function": {"name": "tool", "description": "description", "parameters": "{\"type\":\"object\"}"}}],
            "tool_choice": {"function": {"name": "tool"}},
            "function_call": {"name": "legacy"}
        });
        for (path, field) in fields {
            let value = request.pointer_mut(path).unwrap();
            *value = json!(encrypt_request(
                value.as_str().unwrap(),
                field,
                &ctx,
                &server
            ));
        }
        let original = request.clone();
        decrypt_request_fields(&mut request, Endpoint::ChatCompletions, &ctx, &server).unwrap();
        assert_eq!(request["messages"][0]["content"][0]["text"], "secret");
        assert_eq!(
            request["tools"][0]["function"]["parameters"],
            json!({"type":"object"})
        );
        for (path, _) in fields {
            for invalid in [
                json!("cleartext"),
                json!({"text":"cleartext"}),
                json!(["cleartext"]),
            ] {
                let mut modified = original.clone();
                *modified.pointer_mut(path).unwrap() = invalid;
                assert!(
                    decrypt_request_fields(&mut modified, Endpoint::ChatCompletions, &ctx, &server)
                        .is_err(),
                    "{path}"
                );
            }
        }
        for (path, field) in fields.iter().skip(1) {
            let mut relocated = original.clone();
            *relocated.pointer_mut(path).unwrap() = original["messages"][0]["content"].clone();
            assert!(
                decrypt_request_fields(&mut relocated, Endpoint::ChatCompletions, &ctx, &server)
                    .is_err(),
                "{field}"
            );
        }
        let mut malformed_schema = original;
        malformed_schema["tools"][0]["function"]["parameters"] = json!(encrypt_request(
            "invalid JSON",
            "tools[].function.parameters",
            &ctx,
            &server
        ));
        assert!(decrypt_request_fields(
            &mut malformed_schema,
            Endpoint::ChatCompletions,
            &ctx,
            &server
        )
        .is_err());
        let mut optional = json!({"messages":[{"content":null,"refusal":"","name":null}]});
        decrypt_request_fields(&mut optional, Endpoint::ChatCompletions, &ctx, &server).unwrap();
    }
}

#[test]
fn v3_encrypts_json_and_stream_fields_regardless_of_flag() {
    for vector in vectors() {
        let server = pair(&hex::decode(vector["model_secret"].as_str().unwrap()).unwrap());
        let client = pair(&hex::decode(vector["reply_secret"].as_str().unwrap()).unwrap());
        for flag in [None, Some("false"), Some("true")] {
            let ctx = context(&vector, flag);
            assert!(ctx.encrypt_all_fields);
            // Response wire format remains v2 (empty AAD).
            let response_ctx = EncryptionContext {
                version: 2,
                ..ctx.clone()
            };
            for stream in [false, true] {
                let key = if stream { "delta" } else { "message" };
                let mut response = json!({"choices":[{key:{"content":"answer","refusal":"no","tool_calls":[{"function":{"name":"tool","arguments":"{}"}}],"function_call":{"name":"old","arguments":"{}"}},"logprobs":{"content":[{"token":"secret","bytes":[115],"top_logprobs":[{"token":"other","bytes":[111]}]}]}}, {"logprobs":{"refusal":[{"token":"hidden","bytes":[104]}]}}]});
                if stream {
                    encrypt_streaming_chunk(
                        &mut response,
                        Endpoint::ChatCompletions,
                        &ctx,
                        &server,
                    )
                    .unwrap();
                } else {
                    encrypt_response_fields(
                        &mut response,
                        Endpoint::ChatCompletions,
                        &ctx,
                        &server,
                    )
                    .unwrap();
                }
                for (path, expected) in [
                    (format!("/choices/0/{key}/content"), "answer"),
                    (
                        format!("/choices/0/{key}/tool_calls/0/function/arguments"),
                        "{}",
                    ),
                    ("/choices/0/logprobs/content/0/token".into(), "secret"),
                    ("/choices/0/logprobs/content/0/bytes".into(), "[115]"),
                    (
                        "/choices/0/logprobs/content/0/top_logprobs/0/token".into(),
                        "other",
                    ),
                    ("/choices/1/logprobs/refusal/0/token".into(), "hidden"),
                ] {
                    let cipher = response.pointer(&path).unwrap().as_str().unwrap();
                    assert_ne!(cipher, expected);
                    assert_eq!(
                        decrypt_string(cipher, &response_ctx, &client).unwrap(),
                        expected
                    );
                }
            }
        }
    }
}

#[test]
fn malformed_version_does_not_select_legacy() {
    let vector = vectors().remove(0);
    for version in ["invalid", "", "4"] {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-signing-algo", "ecdsa".parse().unwrap());
        headers.insert(
            "x-client-pub-key",
            vector["reply_public"].as_str().unwrap().parse().unwrap(),
        );
        headers.insert("x-encryption-version", version.parse().unwrap());
        assert!(extract_encryption_context(&headers).is_err());
    }
}

#[test]
fn v3_protects_other_endpoint_fields_and_completion_logprobs() {
    for vector in vectors() {
        let server = pair(&hex::decode(vector["model_secret"].as_str().unwrap()).unwrap());
        let client = pair(&hex::decode(vector["reply_secret"].as_str().unwrap()).unwrap());
        let ctx = context(&vector, None);
        let e =
            |field| encrypt_request(vector["plaintext"].as_str().unwrap(), field, &ctx, &server);
        for (endpoint, mut body, path) in [
            (
                Endpoint::Completions,
                json!({"prompt":[e("prompt")]}),
                "/prompt/0",
            ),
            (
                Endpoint::Embeddings,
                json!({"input":[e("input")]}),
                "/input/0",
            ),
            (
                Endpoint::ImagesGenerations,
                json!({"prompt":e("prompt")}),
                "/prompt",
            ),
            (
                Endpoint::ImagesEdits,
                json!({"prompt":e("prompt")}),
                "/prompt",
            ),
            (
                Endpoint::AudioTranscriptions,
                json!({"prompt":e("prompt")}),
                "/prompt",
            ),
            (
                Endpoint::Rerank,
                json!({"query":e("query"),"documents":[e("documents[]"),{"text":e("documents[].text")}]}),
                "/documents/1/text",
            ),
            (
                Endpoint::Score,
                json!({"text_1":e("text_1"),"text_2":e("text_2")}),
                "/text_2",
            ),
        ] {
            decrypt_request_fields(&mut body, endpoint, &ctx, &server).unwrap();
            assert_eq!(body.pointer(path).unwrap(), &vector["plaintext"]);
        }
        let response_ctx = EncryptionContext {
            version: 2,
            ..ctx.clone()
        };
        let logprobs =
            json!({"tokens":["private token"],"top_logprobs":[{"private alternative":-1.0}]});
        for stream in [false, true] {
            let mut body = json!({"choices":[{"text":"answer","logprobs":logprobs}]});
            if stream {
                encrypt_streaming_chunk(&mut body, Endpoint::Completions, &ctx, &server).unwrap();
            } else {
                encrypt_response_fields(&mut body, Endpoint::Completions, &ctx, &server).unwrap();
            }
            let clear = decrypt_string(
                body["choices"][0]["logprobs"].as_str().unwrap(),
                &response_ctx,
                &client,
            )
            .unwrap();
            assert_eq!(serde_json::from_str::<Value>(&clear).unwrap(), logprobs);
        }
    }
}

#[test]
fn v3_completion_metadata_options_keep_legacy_compatibility() {
    let vector = vectors().remove(0);
    let server = pair(&hex::decode(vector["model_secret"].as_str().unwrap()).unwrap());
    for version in [1, 2, 3] {
        let ctx = EncryptionContext {
            version,
            ..context(&vector, None)
        };
        for endpoint in [Endpoint::ChatCompletions, Endpoint::Completions] {
            for (field, value) in [
                ("return_token_ids", json!(true)),
                ("return_token_ids", json!("true")),
                ("return_prompt_text", json!(true)),
                ("prompt_logprobs", json!(0)),
                ("prompt_logprobs", json!(1)),
            ] {
                let mut request = json!({field: value});
                let result = decrypt_request_fields(&mut request, endpoint, &ctx, &server);
                if version == 3 {
                    assert!(
                        matches!(result, Err(AppError::BadRequest(message)) if message.contains(field))
                    );
                } else {
                    result.unwrap();
                    assert_eq!(request[field], value);
                }
            }
            let mut request = json!({
                "return_token_ids": false,
                "return_prompt_text": null,
                "prompt_logprobs": null,
                "logprobs": true,
            });
            decrypt_request_fields(&mut request, endpoint, &ctx, &server).unwrap();
        }
    }
}

#[test]
fn v3_removes_completion_metadata_from_json_and_sse() {
    let vector = vectors().remove(0);
    let server = pair(&hex::decode(vector["model_secret"].as_str().unwrap()).unwrap());
    let metadata = json!({
        "prompt_token_ids": [101, 202],
        "token_ids": [303, 404],
        "prompt_text": "private prompt",
        "matched_stop": "private stop",
        "stop_reason": 42,
        "hidden_states": [[0.5]],
        "meta_info": {"private": "value"},
        "sglext": {"private": "value"},
        "future_engine_field": "private value",
        "prompt_logprobs": [{"101": {"decoded_token": "private", "logprob": -1.0}}],
    });
    for version in [1, 2, 3] {
        let ctx = EncryptionContext {
            version,
            ..context(&vector, None)
        };
        for endpoint in [Endpoint::ChatCompletions, Endpoint::Completions] {
            for stream in [false, true] {
                let key = if stream { "delta" } else { "message" };
                let mut choice = metadata.clone();
                choice[key] = metadata.clone();
                choice["index"] = json!(0);
                let mut response = metadata.clone();
                response["choices"] = json!([choice]);
                response["id"] = json!("chat-test");
                response["usage"] = json!({"prompt_tokens": 2});
                if stream {
                    encrypt_streaming_chunk(&mut response, endpoint, &ctx, &server).unwrap();
                } else {
                    encrypt_response_fields(&mut response, endpoint, &ctx, &server).unwrap();
                }
                for object in [
                    &response,
                    &response["choices"][0],
                    &response["choices"][0][key],
                ] {
                    for (field, expected) in metadata.as_object().unwrap() {
                        if version == 3 {
                            assert!(object.get(field).is_none(), "{endpoint:?}: {field}");
                        } else {
                            assert_eq!(&object[field], expected);
                        }
                    }
                }
                assert_eq!(response["id"], "chat-test");
                assert_eq!(response["usage"]["prompt_tokens"], 2);
                assert_eq!(response["choices"][0]["index"], 0);
            }
        }
    }
}

#[test]
fn v3_filters_nested_response_extensions() {
    let vector = vectors().remove(0);
    let server = pair(&hex::decode(vector["model_secret"].as_str().unwrap()).unwrap());
    let ctx = context(&vector, None);
    for streaming in [false, true] {
        let field = if streaming { "delta" } else { "message" };
        let mut response = json!({
            "choices": [{
                "index": 0,
                field: {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer", "extension": "private"}],
                    "audio": {"id": "audio-1", "data": "audio", "transcript": "private"},
                    "tool_calls": [{"id": "call-1", "extension": "private", "function": {
                        "name": "weather", "arguments": "{}", "extension": "private"
                    }}],
                    "function_call": {"name": "weather", "arguments": "{}", "extension": "private"},
                    "nearai_tool_result": {"name": "private", "status": "private", "output": "result"}
                },
                "logprobs": {"content": [{"token": "answer", "bytes": [97], "logprob": -0.1,
                    "extension": "private", "top_logprobs": [{"token": "other", "bytes": [98], "logprob": -1.0, "extension": "private"}]
                }], "extension": "private"}
            }],
            "usage": {"prompt_tokens": 2, "extension": "private", "prompt_tokens_details": {"cached_tokens": 1, "extension": "private"}}
        });
        if streaming {
            encrypt_streaming_chunk(&mut response, Endpoint::ChatCompletions, &ctx, &server)
                .unwrap();
        } else {
            encrypt_response_fields(&mut response, Endpoint::ChatCompletions, &ctx, &server)
                .unwrap();
        }
        assert!(!response.to_string().contains("private"));
        assert_eq!(response["choices"][0][field]["role"], "assistant");
        assert_eq!(
            response["choices"][0][field]["tool_calls"][0]["id"],
            "call-1"
        );
        assert_eq!(
            response["choices"][0]["logprobs"]["content"][0]["logprob"],
            -0.1
        );
        assert_eq!(
            response["usage"]["prompt_tokens_details"]["cached_tokens"],
            1
        );
    }
}

#[test]
fn v3_encrypts_rerank_scores() {
    for vector in vectors() {
        let server = pair(&hex::decode(vector["model_secret"].as_str().unwrap()).unwrap());
        let client = pair(&hex::decode(vector["reply_secret"].as_str().unwrap()).unwrap());
        for version in [2, 3] {
            let ctx = EncryptionContext {
                version,
                ..context(&vector, None)
            };
            let mut response = json!({"results":[{"index":0,"relevance_score":0.25,"document":{"text":"document"}},{"index":1,"relevance_score":0.75}]});
            encrypt_response_fields(&mut response, Endpoint::Rerank, &ctx, &server).unwrap();
            if version == 3 {
                let response_ctx = EncryptionContext { version: 2, ..ctx };
                assert_eq!(
                    decrypt_string(
                        response["results"][0]["relevance_score"].as_str().unwrap(),
                        &response_ctx,
                        &client
                    )
                    .unwrap(),
                    "0.75"
                );
            } else {
                assert_eq!(response["results"][0]["relevance_score"], 0.25);
            }
            assert_eq!(
                response["results"][0]["index"],
                if version == 3 { 1 } else { 0 }
            );
        }
    }
}

#[test]
fn encrypted_error_events_use_fixed_messages_in_every_version() {
    let vector = vectors().remove(0);
    let server = pair(&hex::decode(vector["model_secret"].as_str().unwrap()).unwrap());
    for version in [1, 2, 3] {
        let ctx = EncryptionContext {
            version,
            ..context(&vector, None)
        };
        for error in [
            json!({"error":{"message":"private value","type":"private type","code":400},"extra":"private value"}),
            json!({"object":"error","message":"private value","type":"BadRequestError"}),
        ] {
            for streaming in [false, true] {
                let mut value = error.clone();
                if streaming {
                    encrypt_streaming_chunk(&mut value, Endpoint::ChatCompletions, &ctx, &server)
                        .unwrap();
                } else {
                    encrypt_response_fields(&mut value, Endpoint::ChatCompletions, &ctx, &server)
                        .unwrap();
                }
                assert_eq!(
                    value["error"]["message"],
                    crate::error::E2EE_UPSTREAM_MESSAGE
                );
                assert!(!value.to_string().contains("private"));
            }
        }
    }
}
