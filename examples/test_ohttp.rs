//! Test OHTTP gateway against a live deployment.
//!
//! Usage:
//!     cargo run --example test_ohttp -- <base_url> [--token <token>]
//!
//! Examples:
//!     cargo run --example test_ohttp -- http://127.0.0.1:8000 --token <token>
//!     cargo run --example test_ohttp -- https://<model>.completions.near.ai
//!
//! Tests:
//!     1. Fetch OHTTP key config from /.well-known/ohttp-gateway
//!     2. OHTTP-encrypted chat completion (non-streaming)
//!     3. OHTTP-encrypted /v1/models GET
//!     4. OHTTP with missing auth (expect encrypted 401)

use std::io::Cursor;

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(run());
}

async fn run() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <base_url> [--token <token>] [--model <model>]",
            args[0]
        );
        std::process::exit(1);
    }

    let base_url = args[1].trim_end_matches('/');
    let token = get_arg(&args, "--token").unwrap_or_else(|| "test-token".to_string());
    let model = get_arg(&args, "--model").unwrap_or_else(|| "zai-org/GLM-5-FP8".to_string());

    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true) // for local testing
        .build()
        .unwrap();

    println!("=== OHTTP Gateway Test ===");
    println!("Base URL: {base_url}");
    println!("Model:    {model}");
    println!();

    // Test 1: Fetch key config
    print!("1. Fetching OHTTP key config... ");
    let config_url = format!("{base_url}/.well-known/ohttp-gateway");
    let resp = client.get(&config_url).send().await.unwrap();
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        println!("FAIL (status {status}): {body}");
        std::process::exit(1);
    }
    let config_bytes = resp.bytes().await.unwrap();
    let mut key_config = ohttp::KeyConfig::decode(&config_bytes).unwrap();
    println!(
        "OK ({} bytes, key_id={})",
        config_bytes.len(),
        config_bytes[0]
    );

    // Also test the alias
    print!("   Checking /v1/ohttp/config alias... ");
    let alias_resp = client
        .get(format!("{base_url}/v1/ohttp/config"))
        .send()
        .await
        .unwrap();
    let alias_bytes = alias_resp.bytes().await.unwrap();
    if alias_bytes == config_bytes {
        println!("OK (matches)");
    } else {
        println!("FAIL (bytes differ!)");
    }
    println!();

    // Test 2: OHTTP-encrypted chat completion
    print!("2. OHTTP chat completion (non-streaming)... ");
    let chat_body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Say 'OHTTP works!' and nothing else."}],
        "max_tokens": 20,
        "stream": false
    });

    let mut inner_req = bhttp::Message::request(
        b"POST".to_vec(),
        b"https".to_vec(),
        b"localhost".to_vec(),
        b"/v1/chat/completions".to_vec(),
    );
    inner_req.put_header("content-type", "application/json");
    inner_req.put_header("authorization", format!("Bearer {token}"));
    inner_req.write_content(serde_json::to_vec(&chat_body).unwrap());

    let (enc_req, client_response) = encrypt_request(&mut key_config, &inner_req);
    let start = std::time::Instant::now();
    let resp = client
        .post(format!("{base_url}/ohttp"))
        .header("content-type", "message/ohttp-req")
        .body(enc_req)
        .send()
        .await
        .unwrap();

    let ohttp_status = resp.status();
    let enc_resp = resp.bytes().await.unwrap();
    let elapsed = start.elapsed();

    if !ohttp_status.is_success() {
        println!(
            "FAIL (OHTTP layer returned {ohttp_status}): {}",
            String::from_utf8_lossy(&enc_resp)
        );
        std::process::exit(1);
    }

    let inner_resp = decrypt_response(client_response, &enc_resp);
    let inner_status = inner_resp.control().status().unwrap().code();
    let body: serde_json::Value = serde_json::from_slice(inner_resp.content()).unwrap_or_default();

    if inner_status == 200 {
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("<no content>");
        println!("OK ({elapsed:?}, inner 200)");
        println!("   Response: {content}");
    } else {
        println!("FAIL (inner status {inner_status})");
        println!(
            "   Body: {}",
            serde_json::to_string_pretty(&body).unwrap_or_default()
        );
        std::process::exit(1);
    }
    println!();

    // Test 3: OHTTP-encrypted GET /v1/models
    print!("3. OHTTP GET /v1/models... ");
    let models_req = bhttp::Message::request(
        b"GET".to_vec(),
        b"https".to_vec(),
        b"localhost".to_vec(),
        b"/v1/models".to_vec(),
    );

    // Need a fresh KeyConfig for each encapsulation (consumed by encapsulate)
    let mut key_config2 = ohttp::KeyConfig::decode(&config_bytes).unwrap();
    let (enc_req, client_response) = encrypt_request(&mut key_config2, &models_req);
    let resp = client
        .post(format!("{base_url}/ohttp"))
        .header("content-type", "message/ohttp-req")
        .body(enc_req)
        .send()
        .await
        .unwrap();

    let enc_resp = resp.bytes().await.unwrap();
    let inner_resp = decrypt_response(client_response, &enc_resp);
    let inner_status = inner_resp.control().status().unwrap().code();

    if inner_status == 200 {
        let body: serde_json::Value =
            serde_json::from_slice(inner_resp.content()).unwrap_or_default();
        let model_count = body["data"].as_array().map(|a| a.len()).unwrap_or(0);
        println!("OK (inner 200, {model_count} models)");
    } else {
        println!("inner status {inner_status}");
        println!("   Body: {}", String::from_utf8_lossy(inner_resp.content()));
    }
    println!();

    // Test 4: OHTTP with no auth (expect encrypted 401)
    print!("4. OHTTP without auth (expect 401 inside envelope)... ");
    let mut noauth_req = bhttp::Message::request(
        b"POST".to_vec(),
        b"https".to_vec(),
        b"localhost".to_vec(),
        b"/v1/chat/completions".to_vec(),
    );
    noauth_req.put_header("content-type", "application/json");
    // No authorization header
    noauth_req.write_content(serde_json::to_vec(&chat_body).unwrap());

    let mut key_config3 = ohttp::KeyConfig::decode(&config_bytes).unwrap();
    let (enc_req, client_response) = encrypt_request(&mut key_config3, &noauth_req);
    let resp = client
        .post(format!("{base_url}/ohttp"))
        .header("content-type", "message/ohttp-req")
        .body(enc_req)
        .send()
        .await
        .unwrap();

    let ohttp_status = resp.status();
    let enc_resp = resp.bytes().await.unwrap();

    if ohttp_status.is_success() {
        let inner_resp = decrypt_response(client_response, &enc_resp);
        let inner_status = inner_resp.control().status().unwrap().code();
        if inner_status == 401 {
            println!("OK (OHTTP 200, inner 401 as expected)");
        } else {
            println!("UNEXPECTED (OHTTP 200, inner {inner_status} — expected 401)");
        }
    } else {
        println!("FAIL (OHTTP layer returned {ohttp_status})");
    }
    println!();

    println!("=== All tests done ===");
}

fn encrypt_request(
    key_config: &mut ohttp::KeyConfig,
    inner_msg: &bhttp::Message,
) -> (Vec<u8>, ohttp::ClientResponse) {
    let client_request = ohttp::ClientRequest::from_config(key_config).unwrap();

    let mut bhttp_bytes = Vec::new();
    inner_msg
        .write_bhttp(bhttp::Mode::KnownLength, &mut bhttp_bytes)
        .unwrap();

    client_request.encapsulate(&bhttp_bytes).unwrap()
}

fn decrypt_response(client_response: ohttp::ClientResponse, enc_response: &[u8]) -> bhttp::Message {
    let bhttp_bytes = client_response.decapsulate(enc_response).unwrap();
    bhttp::Message::read_bhttp(&mut Cursor::new(&bhttp_bytes[..])).unwrap()
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
