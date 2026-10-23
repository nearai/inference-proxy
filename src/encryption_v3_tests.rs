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

fn encrypt_request(text: &str, ctx: &EncryptionContext, server: &SigningPair) -> String {
    let aad = ctx.request_aad().unwrap();
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
        assert_eq!(hex::encode(ctx.request_aad().unwrap()), vector["aad"]);
        let ciphertext = vector["ciphertext"].as_str().unwrap();
        assert_eq!(
            decrypt_string(ciphertext, &ctx, &server).unwrap(),
            vector["plaintext"]
        );
        if ctx.algo == EncryptionAlgo::Ecdsa {
            let mut raw = ctx.clone();
            raw.client_pub_key.remove(0);
            assert_eq!(ctx.request_aad().unwrap(), raw.request_aad().unwrap());
            assert!(decrypt_string(ciphertext, &raw, &server).is_ok());
        }
        for version in [1, 2] {
            let mut downgraded = ctx.clone();
            downgraded.version = version;
            assert!(decrypt_string(ciphertext, &downgraded, &server).is_err());
        }
        let mut replaced = ctx.clone();
        replaced.client_pub_key = hex::decode(vector["model_public"].as_str().unwrap()).unwrap();
        assert!(decrypt_string(ciphertext, &replaced, &server).is_err());
    }
}

#[test]
fn v3_authenticates_every_supported_chat_field_strictly() {
    for vector in vectors() {
        let server = pair(&hex::decode(vector["model_secret"].as_str().unwrap()).unwrap());
        let ctx = context(&vector, Some("false"));
        let e = |s| encrypt_request(s, &ctx, &server);
        let mut request = json!({"messages":[{"content":e("[{\"type\":\"text\",\"text\":\"secret\"}]"),"reasoning":e("reason"),"reasoning_content":e("thought"),"name":e("name"),"refusal":e("refusal"),"audio":{"data":e("audio")},"tool_calls":[{"function":{"name":e("tool"),"arguments":e("{}")}}],"function_call":{"name":e("legacy"),"arguments":e("{}")}}],"tools":[{"function":{"name":e("tool"),"description":e("description"),"parameters":e("{\"type\":\"object\"}")}}],"tool_choice":{"function":{"name":e("tool")}},"function_call":{"name":e("legacy")}});
        let original = request.clone();
        decrypt_request_fields(&mut request, Endpoint::ChatCompletions, &ctx, &server).unwrap();
        assert_eq!(request["messages"][0]["content"][0]["text"], "secret");
        assert_eq!(
            request["tools"][0]["function"]["parameters"],
            json!({"type":"object"})
        );
        for path in [
            "/messages/0/content",
            "/messages/0/reasoning",
            "/messages/0/reasoning_content",
            "/messages/0/name",
            "/messages/0/refusal",
            "/messages/0/audio/data",
            "/messages/0/tool_calls/0/function/name",
            "/messages/0/tool_calls/0/function/arguments",
            "/messages/0/function_call/name",
            "/messages/0/function_call/arguments",
            "/tools/0/function/name",
            "/tools/0/function/description",
            "/tools/0/function/parameters",
            "/tool_choice/function/name",
            "/function_call/name",
        ] {
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
        let mut malformed_schema = original;
        malformed_schema["tools"][0]["function"]["parameters"] = json!(e("invalid JSON"));
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
        let cipher = &vector["ciphertext"];
        for (endpoint, mut body, path) in [
            (
                Endpoint::Completions,
                json!({"prompt":[cipher]}),
                "/prompt/0",
            ),
            (Endpoint::Embeddings, json!({"input":[cipher]}), "/input/0"),
            (
                Endpoint::ImagesGenerations,
                json!({"prompt":cipher}),
                "/prompt",
            ),
            (Endpoint::ImagesEdits, json!({"prompt":cipher}), "/prompt"),
            (
                Endpoint::AudioTranscriptions,
                json!({"prompt":cipher}),
                "/prompt",
            ),
            (
                Endpoint::Rerank,
                json!({"query":cipher,"documents":[cipher,{"text":cipher}]}),
                "/documents/1/text",
            ),
            (
                Endpoint::Score,
                json!({"text_1":cipher,"text_2":cipher}),
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
