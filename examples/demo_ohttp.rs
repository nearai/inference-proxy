//! OHTTP (Oblivious HTTP) Demo — shows the full privacy flow
//!
//! Usage:
//!     cargo run --example demo_ohttp -- <base_url> --token <token>
//!
//! Example:
//!     cargo run --example demo_ohttp -- https://<model>.completions.near.ai
//!     cargo run --example demo_ohttp -- http://127.0.0.1:8000 --token <token>

use std::io::Cursor;

const BLUE: &str = "\x1b[34m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";
const CYAN: &str = "\x1b[36m";

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(run());
}

async fn run() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <base_url> [--token <token>] [--model <model>] [--prompt <prompt>]",
            args[0]
        );
        std::process::exit(1);
    }

    let base_url = args[1].trim_end_matches('/');
    let token = get_arg(&args, "--token").unwrap_or_else(|| "test-token".to_string());
    let model = get_arg(&args, "--model").unwrap_or_else(|| "zai-org/GLM-5.1-FP8".to_string());
    let prompt = get_arg(&args, "--prompt")
        .unwrap_or_else(|| "What is the capital of France? Answer in one sentence.".to_string());

    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .unwrap();

    println!();
    println!("{BOLD}╔══════════════════════════════════════════════════════════════════╗{RESET}");
    println!("{BOLD}║           Oblivious HTTP (OHTTP) — RFC 9458 Demo               ║{RESET}");
    println!("{BOLD}╚══════════════════════════════════════════════════════════════════╝{RESET}");
    println!();
    println!("{DIM}OHTTP ensures no single party can link a user's identity to their");
    println!("query. A relay sees the client IP but only encrypted blobs; the");
    println!("gateway sees the query but only the relay's IP.{RESET}");
    println!();
    println!("  Gateway: {CYAN}{base_url}{RESET}");
    println!("  Model:   {CYAN}{model}{RESET}");
    println!();

    // ── Step 1: Fetch key config ──────────────────────────────────────
    println!("{BOLD}┌─ Step 1: Fetch Gateway's OHTTP Key Config{RESET}");
    println!("{DIM}│  The client fetches the gateway's HPKE public key.{RESET}");
    println!("{DIM}│  This key is deterministically derived from the TEE's{RESET}");
    println!("{DIM}│  Ed25519 signing key and can be verified via attestation.{RESET}");
    println!("│");

    let config_url = format!("{base_url}/.well-known/ohttp-gateway");
    let resp = client.get(&config_url).send().await.unwrap();
    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        println!("│  {BOLD}\x1b[31mERROR:{RESET} {body}");
        println!("│  Is OHTTP_ENABLED=true on this deployment?");
        std::process::exit(1);
    }
    let config_bytes = resp.bytes().await.unwrap();
    println!("│  {GREEN}Key config:{RESET} {} bytes", config_bytes.len());
    println!("│  {DIM}{}...{RESET}", hex_preview(&config_bytes, 32));
    println!("└──");
    println!();

    // ── Step 2: Build inner HTTP request ──────────────────────────────
    println!("{BOLD}┌─ Step 2: Build Inner HTTP Request{RESET}");
    println!("{DIM}│  This is the real API request — it will be encrypted{RESET}");
    println!("{DIM}│  before leaving the client.{RESET}");
    println!("│");

    let chat_body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
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

    let mut bhttp_bytes = Vec::new();
    inner_req
        .write_bhttp(bhttp::Mode::KnownLength, &mut bhttp_bytes)
        .unwrap();

    println!("│  {BLUE}POST /v1/chat/completions{RESET}");
    println!(
        "│  {BLUE}Authorization: Bearer {}...{RESET}",
        &token[..std::cmp::min(8, token.len())]
    );
    println!("│  {BLUE}Prompt: \"{prompt}\"{RESET}");
    println!(
        "│  {DIM}Binary HTTP encoded: {} bytes{RESET}",
        bhttp_bytes.len()
    );
    println!("└──");
    println!();

    // ── Step 3: HPKE encrypt ──────────────────────────────────────────
    println!("{BOLD}┌─ Step 3: HPKE Encrypt (Client-Side){RESET}");
    println!("{DIM}│  The client encrypts the Binary HTTP request using the{RESET}");
    println!("{DIM}│  gateway's HPKE public key. Only the gateway's private{RESET}");
    println!("{DIM}│  key (inside the TEE) can decrypt this.{RESET}");
    println!("│");

    let mut key_config = ohttp::KeyConfig::decode(&config_bytes).unwrap();
    let client_request = ohttp::ClientRequest::from_config(&mut key_config).unwrap();
    let (enc_request, client_response) = client_request.encapsulate(&bhttp_bytes).unwrap();
    let enc_request_len = enc_request.len();

    println!("│  {YELLOW}Encrypted request:{RESET} {enc_request_len} bytes");
    println!("│  {DIM}{}...{RESET}", hex_preview(&enc_request, 48));
    println!("│");
    println!("│  {DIM}The relay would forward this opaque blob.{RESET}");
    println!("│  {DIM}It cannot read the prompt, auth token, or any content.{RESET}");
    println!("└──");
    println!();

    // ── Step 4: Send to gateway ───────────────────────────────────────
    println!("{BOLD}┌─ Step 4: Send Encrypted Request to Gateway{RESET}");
    println!("{DIM}│  In production, this goes: Client → Relay → Gateway.{RESET}");
    println!("{DIM}│  The relay only sees the encrypted blob above.{RESET}");
    println!("│");

    let start = std::time::Instant::now();
    let resp = client
        .post(format!("{base_url}/ohttp"))
        .header("content-type", "message/ohttp-req")
        .body(enc_request)
        .send()
        .await
        .unwrap();

    let ohttp_status = resp.status();
    let enc_response = resp.bytes().await.unwrap();
    let elapsed = start.elapsed();

    println!(
        "│  {GREEN}OHTTP response:{RESET} status={ohttp_status}, {} bytes, {elapsed:.1?}",
        enc_response.len()
    );
    println!("│  {DIM}{}...{RESET}", hex_preview(&enc_response, 48));
    println!("│");
    println!("│  {DIM}The relay forwards this encrypted response back.{RESET}");
    println!("│  {DIM}It cannot read the model's answer.{RESET}");
    println!("└──");
    println!();

    // ── Step 5: Decrypt response ──────────────────────────────────────
    println!("{BOLD}┌─ Step 5: HPKE Decrypt (Client-Side){RESET}");
    println!("{DIM}│  The client decrypts using the HPKE context from Step 3.{RESET}");
    println!("│");

    if !ohttp_status.is_success() {
        println!("│  {BOLD}\x1b[31mGateway returned {ohttp_status} — cannot decrypt.{RESET}");
        println!("│  Body: {}", String::from_utf8_lossy(&enc_response));
        std::process::exit(1);
    }

    let bhttp_resp_bytes = client_response.decapsulate(&enc_response).unwrap();
    let inner_resp = bhttp::Message::read_bhttp(&mut Cursor::new(&bhttp_resp_bytes[..])).unwrap();

    let inner_status = inner_resp.control().status().unwrap().code();
    let body: serde_json::Value = serde_json::from_slice(inner_resp.content()).unwrap_or_default();

    println!("│  {GREEN}Inner HTTP status:{RESET} {inner_status}");

    if inner_status == 200 {
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .filter(|s| !s.is_empty());
        let reasoning = body["choices"][0]["message"]["reasoning_content"]
            .as_str()
            .filter(|s| !s.is_empty());

        if let Some(r) = reasoning {
            let preview = if r.len() > 200 { &r[..200] } else { r };
            println!("│  {DIM}Reasoning: {preview}...{RESET}");
        }
        if let Some(c) = content {
            println!("│  {GREEN}Response: {c}{RESET}");
        } else if reasoning.is_some() {
            println!("│  {DIM}(Model used all tokens on reasoning, no final answer){RESET}");
        } else {
            println!("│  {DIM}(Empty response){RESET}");
        }

        let usage = &body["usage"];
        println!(
            "│  {DIM}Tokens: {} prompt, {} completion{RESET}",
            usage["prompt_tokens"], usage["completion_tokens"]
        );
    } else {
        let err_msg = body["error"]["message"].as_str().unwrap_or("unknown error");
        println!("│  \x1b[31mError: {err_msg}{RESET}");
    }

    println!("└──");
    println!();

    // ── Summary ───────────────────────────────────────────────────────
    println!("{BOLD}╔══════════════════════════════════════════════════════════════════╗{RESET}");
    println!("{BOLD}║  Privacy Summary                                               ║{RESET}");
    println!("{BOLD}╠══════════════════════════════════════════════════════════════════╣{RESET}");
    println!("{BOLD}║{RESET}                                                                {BOLD}║{RESET}");
    println!("{BOLD}║{RESET}  Relay sees:    Client IP + {YELLOW}{enc_request_len} bytes of encrypted noise{RESET}    {BOLD}║{RESET}");
    println!("{BOLD}║{RESET}  Relay learns:  {GREEN}Nothing about the query or response{RESET}        {BOLD}║{RESET}");
    println!("{BOLD}║{RESET}                                                                {BOLD}║{RESET}");
    println!("{BOLD}║{RESET}  Gateway sees:  Relay IP + decrypted query                     {BOLD}║{RESET}");
    println!("{BOLD}║{RESET}  Gateway learns: {GREEN}Nothing about who sent the query{RESET}          {BOLD}║{RESET}");
    println!("{BOLD}║{RESET}                                                                {BOLD}║{RESET}");
    println!("{BOLD}║{RESET}  {DIM}Neither the relay nor the gateway alone can link{RESET}             {BOLD}║{RESET}");
    println!("{BOLD}║{RESET}  {DIM}a user's identity to their query content.{RESET}                   {BOLD}║{RESET}");
    println!("{BOLD}║{RESET}                                                                {BOLD}║{RESET}");
    println!("{BOLD}╚══════════════════════════════════════════════════════════════════╝{RESET}");
    println!();
}

fn hex_preview(data: &[u8], max_bytes: usize) -> String {
    let n = std::cmp::min(data.len(), max_bytes);
    hex::encode(&data[..n])
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
