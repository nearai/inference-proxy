use std::sync::Arc;

use crate::error::AppError;
use crate::signing::SigningPair;

/// One-shot transform applied to a full JSON response body.
pub type ResponseTransform = Box<dyn FnOnce(&mut serde_json::Value) -> Result<(), AppError> + Send>;

/// Reusable transform applied to each SSE chunk.
pub type ChunkTransform = Arc<dyn Fn(&mut serde_json::Value) -> Result<(), AppError> + Send + Sync>;

// ── Algorithm enum ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionAlgo {
    Ed25519,
    Ecdsa,
}

// ── Encryption context (parsed from request headers) ────────────────

#[derive(Debug, Clone)]
pub struct EncryptionContext {
    pub algo: EncryptionAlgo,
    /// Raw client public key bytes.
    pub client_pub_key: Vec<u8>,
    /// Encryption protocol version (1 = legacy NaCl box, 2 = HKDF + XChaCha20-Poly1305).
    pub version: u8,
    /// When true, tool call arguments (`tools[].function.description/parameters` in requests
    /// and `tool_calls[].function.arguments` in responses) are also encrypted/decrypted.
    /// Enabled by the `X-Encrypt-Tool-Calls: true` request header.
    pub encrypt_tool_calls: bool,
}

/// Extract encryption context from request headers.
/// Both `X-Signing-Algo` and `X-Client-Pub-Key` must be present together, or neither.
pub fn extract_encryption_context(
    headers: &axum::http::HeaderMap,
) -> Result<Option<EncryptionContext>, AppError> {
    let algo_hdr = headers
        .get("x-signing-algo")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let key_hdr = headers
        .get("x-client-pub-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    match (algo_hdr, key_hdr) {
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => Err(AppError::BadRequest(
            "Both X-Signing-Algo and X-Client-Pub-Key headers must be present together".to_string(),
        )),
        (Some(algo_str), Some(key_hex)) => {
            let algo = match algo_str.to_lowercase().as_str() {
                "ed25519" => EncryptionAlgo::Ed25519,
                "ecdsa" => EncryptionAlgo::Ecdsa,
                _ => {
                    return Err(AppError::BadRequest(format!(
                        "Invalid X-Signing-Algo: {algo_str}. Must be 'ecdsa' or 'ed25519'"
                    )));
                }
            };

            let key_bytes = hex::decode(&key_hex).map_err(|_| {
                AppError::BadRequest("X-Client-Pub-Key must be valid hex".to_string())
            })?;

            // Validate key length
            match algo {
                EncryptionAlgo::Ed25519 => {
                    if key_bytes.len() != 32 {
                        return Err(AppError::BadRequest(format!(
                            "Ed25519 client public key must be 32 bytes, got {}",
                            key_bytes.len()
                        )));
                    }
                }
                EncryptionAlgo::Ecdsa => {
                    // Accept 64 bytes (raw) or 65 bytes (with 0x04 prefix)
                    if key_bytes.len() != 64 && key_bytes.len() != 65 {
                        return Err(AppError::BadRequest(format!(
                            "ECDSA client public key must be 64 or 65 bytes, got {}",
                            key_bytes.len()
                        )));
                    }
                }
            }

            let version = headers
                .get("x-encryption-version")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(1);

            if version != 1 && version != 2 {
                return Err(AppError::BadRequest(format!(
                    "Invalid X-Encryption-Version: {version}. Must be 1 or 2"
                )));
            }

            let encrypt_tool_calls = headers
                .get("x-encrypt-tool-calls")
                .and_then(|v| v.to_str().ok())
                .map(|s| { let s = s.trim(); s == "1" || s.eq_ignore_ascii_case("true") })
                .unwrap_or(false);

            Ok(Some(EncryptionContext {
                algo,
                client_pub_key: key_bytes,
                version,
                encrypt_tool_calls,
            }))
        }
    }
}

// ── Ed25519 / NaCl Box encryption ───────────────────────────────────

mod nacl {
    use crypto_box::{
        aead::{Aead, AeadCore, OsRng},
        ChaChaBox, PublicKey, SalsaBox, SecretKey,
    };

    /// Encrypt bytes using NaCl box (X25519 + XChaCha20-Poly1305).
    /// Wire format: [ephemeral_pubkey(32)][nonce(24)][ciphertext+tag]
    pub fn encrypt(plaintext: &[u8], recipient_pub: &[u8; 32]) -> Result<Vec<u8>, String> {
        let recipient = PublicKey::from(*recipient_pub);
        let ephemeral_secret = SecretKey::generate(&mut OsRng);
        let ephemeral_public = ephemeral_secret.public_key();

        let chacha_box = ChaChaBox::new(&recipient, &ephemeral_secret);
        let nonce = ChaChaBox::generate_nonce(&mut OsRng);
        let ciphertext = chacha_box
            .encrypt(&nonce, plaintext)
            .map_err(|e| format!("NaCl encrypt failed: {e}"))?;

        let mut out = Vec::with_capacity(32 + 24 + ciphertext.len());
        out.extend_from_slice(ephemeral_public.as_bytes());
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt bytes using NaCl box.
    /// Tries XChaCha20-Poly1305 first, falls back to XSalsa20-Poly1305 for
    /// backward compatibility with older clients.
    /// Expects wire format: [ephemeral_pubkey(32)][nonce(24)][ciphertext+tag]
    pub fn decrypt(data: &[u8], secret_key_bytes: &[u8; 32]) -> Result<Vec<u8>, String> {
        if data.len() < 32 + 24 + 16 {
            return Err("NaCl ciphertext too short".to_string());
        }

        let ephemeral_pub = PublicKey::from(<[u8; 32]>::try_from(&data[..32]).unwrap());
        #[allow(deprecated)] // generic-array 0.x from_slice deprecation; fixed when deps upgrade
        let nonce = crypto_box::Nonce::from_slice(&data[32..56]);
        let ciphertext = &data[56..];

        let secret = SecretKey::from(*secret_key_bytes);

        // Try ChaCha20 first (documented algorithm)
        let chacha_box = ChaChaBox::new(&ephemeral_pub, &secret);
        if let Ok(plaintext) = chacha_box.decrypt(nonce, ciphertext) {
            return Ok(plaintext);
        }

        // Fall back to Salsa20 for backward compatibility
        let salsa_box = SalsaBox::new(&ephemeral_pub, &secret);
        salsa_box
            .decrypt(nonce, ciphertext)
            .map_err(|e| format!("NaCl decrypt failed: {e}"))
    }

    /// Convert an Ed25519 secret key (32 bytes) to an X25519 secret key.
    /// Ed25519 signing keys are clamped SHA-512 hashes; for NaCl box we need
    /// the X25519 private scalar, which is the first 32 bytes of SHA-512(seed)
    /// with clamping applied.
    pub fn ed25519_secret_to_x25519(ed25519_secret: &[u8; 32]) -> [u8; 32] {
        use sha2::Digest;
        let hash = sha2::Sha512::digest(ed25519_secret);
        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&hash[..32]);
        // Clamp
        scalar[0] &= 248;
        scalar[31] &= 127;
        scalar[31] |= 64;
        scalar
    }

    /// Convert an Ed25519 public key (32 bytes) to an X25519 public key.
    /// Uses the birational map from Edwards to Montgomery form.
    pub fn ed25519_public_to_x25519(ed25519_pub: &[u8; 32]) -> Result<[u8; 32], String> {
        use ed25519_dalek::VerifyingKey;
        let vk = VerifyingKey::from_bytes(ed25519_pub)
            .map_err(|e| format!("Invalid Ed25519 public key: {e}"))?;
        Ok(vk.to_montgomery().to_bytes())
    }
}

// ── Ed25519 / v2 encryption (X25519 + HKDF-SHA256 + XChaCha20-Poly1305) ──

mod nacl_v2 {
    use chacha20poly1305::{
        aead::{Aead, AeadCore, KeyInit, OsRng},
        XChaCha20Poly1305,
    };
    use hkdf::Hkdf;
    use sha2::Sha256;
    use x25519_dalek::{EphemeralSecret, PublicKey, StaticSecret};

    /// Encrypt bytes using X25519 ECDH + HKDF-SHA256 + XChaCha20-Poly1305.
    /// Wire format: [ephemeral_pubkey(32)][nonce(24)][ciphertext+tag]
    pub fn encrypt(plaintext: &[u8], recipient_pub: &[u8; 32]) -> Result<Vec<u8>, String> {
        let recipient = PublicKey::from(*recipient_pub);
        let ephemeral_secret = EphemeralSecret::random_from_rng(OsRng);
        let ephemeral_public = PublicKey::from(&ephemeral_secret);

        let shared_secret = ephemeral_secret.diffie_hellman(&recipient);

        let hkdf = Hkdf::<Sha256>::new(None, shared_secret.as_bytes());
        let mut symmetric_key = [0u8; 32];
        hkdf.expand(b"ed25519_encryption", &mut symmetric_key)
            .map_err(|e| format!("HKDF expand failed: {e}"))?;

        let cipher = XChaCha20Poly1305::new_from_slice(&symmetric_key)
            .map_err(|e| format!("XChaCha20Poly1305 key init failed: {e}"))?;
        let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(&nonce, plaintext)
            .map_err(|e| format!("XChaCha20Poly1305 encrypt failed: {e}"))?;

        let mut out = Vec::with_capacity(32 + 24 + ciphertext.len());
        out.extend_from_slice(ephemeral_public.as_bytes());
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt bytes using X25519 ECDH + HKDF-SHA256 + XChaCha20-Poly1305.
    /// Expects wire format: [ephemeral_pubkey(32)][nonce(24)][ciphertext+tag]
    pub fn decrypt(data: &[u8], secret_key_bytes: &[u8; 32]) -> Result<Vec<u8>, String> {
        if data.len() < 32 + 24 + 16 {
            return Err("Ciphertext too short".to_string());
        }

        let ephemeral_pub = PublicKey::from(<[u8; 32]>::try_from(&data[..32]).unwrap());
        #[allow(deprecated)] // generic-array 0.x from_slice deprecation; fixed when deps upgrade
        let nonce = chacha20poly1305::XNonce::from_slice(&data[32..56]);
        let ciphertext = &data[56..];

        let secret = StaticSecret::from(*secret_key_bytes);
        let shared_secret = secret.diffie_hellman(&ephemeral_pub);

        let hkdf = Hkdf::<Sha256>::new(None, shared_secret.as_bytes());
        let mut symmetric_key = [0u8; 32];
        hkdf.expand(b"ed25519_encryption", &mut symmetric_key)
            .map_err(|e| format!("HKDF expand failed: {e}"))?;

        let cipher = XChaCha20Poly1305::new_from_slice(&symmetric_key)
            .map_err(|e| format!("XChaCha20Poly1305 key init failed: {e}"))?;
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| format!("Decrypt failed: {e}"))
    }
}

// ── ECDSA / ECIES encryption ────────────────────────────────────────

mod ecies {
    use aes_gcm::{
        aead::{Aead, AeadCore, KeyInit, OsRng},
        Aes256Gcm,
    };
    use hkdf::Hkdf;
    use k256::{ecdh::EphemeralSecret, elliptic_curve::sec1::ToEncodedPoint, PublicKey};
    use sha2::Sha256;

    /// Encrypt bytes using ECIES (secp256k1 ECDH + HKDF-SHA256 + AES-256-GCM).
    /// Wire format: [ephemeral_pubkey_uncompressed(65)][nonce(12)][ciphertext+tag]
    pub fn encrypt(plaintext: &[u8], recipient_pub: &PublicKey) -> Result<Vec<u8>, String> {
        let ephemeral_secret = EphemeralSecret::random(&mut OsRng);
        let ephemeral_public = ephemeral_secret.public_key().to_encoded_point(false);

        let shared_secret = ephemeral_secret.diffie_hellman(recipient_pub);

        let hkdf = Hkdf::<Sha256>::new(None, shared_secret.raw_secret_bytes());
        let mut aes_key = [0u8; 32];
        hkdf.expand(b"ecdsa_encryption", &mut aes_key)
            .map_err(|e| format!("HKDF expand failed: {e}"))?;

        let cipher =
            Aes256Gcm::new_from_slice(&aes_key).map_err(|e| format!("AES key init failed: {e}"))?;
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(&nonce, plaintext)
            .map_err(|e| format!("AES-GCM encrypt failed: {e}"))?;

        let pub_bytes = ephemeral_public.as_bytes();
        let mut out = Vec::with_capacity(pub_bytes.len() + 12 + ciphertext.len());
        out.extend_from_slice(pub_bytes);
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt bytes using ECIES.
    /// Expects wire format: [ephemeral_pubkey_uncompressed(65)][nonce(12)][ciphertext+tag]
    pub fn decrypt(data: &[u8], secret_key: &k256::ecdsa::SigningKey) -> Result<Vec<u8>, String> {
        if data.len() < 65 + 12 + 16 {
            return Err("ECIES ciphertext too short".to_string());
        }

        let ephemeral_pub = PublicKey::from_sec1_bytes(&data[..65])
            .map_err(|e| format!("Invalid ephemeral public key: {e}"))?;
        #[allow(deprecated)] // generic-array 0.x from_slice deprecation; fixed when deps upgrade
        let nonce = aes_gcm::Nonce::from_slice(&data[65..77]);
        let ciphertext = &data[77..];

        // Perform ECDH using the server's secret key
        let shared_secret =
            k256::ecdh::diffie_hellman(secret_key.as_nonzero_scalar(), ephemeral_pub.as_affine());

        let hkdf = Hkdf::<Sha256>::new(None, shared_secret.raw_secret_bytes());
        let mut aes_key = [0u8; 32];
        hkdf.expand(b"ecdsa_encryption", &mut aes_key)
            .map_err(|e| format!("HKDF expand failed: {e}"))?;

        let cipher =
            Aes256Gcm::new_from_slice(&aes_key).map_err(|e| format!("AES key init failed: {e}"))?;
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| format!("AES-GCM decrypt failed: {e}"))
    }

    /// Parse a client public key (64 bytes raw or 65 bytes with 0x04 prefix) into a k256 PublicKey.
    pub fn parse_client_pubkey(bytes: &[u8]) -> Result<PublicKey, String> {
        match bytes.len() {
            65 => PublicKey::from_sec1_bytes(bytes)
                .map_err(|e| format!("Invalid ECDSA public key: {e}")),
            64 => {
                let mut prefixed = vec![0x04];
                prefixed.extend_from_slice(bytes);
                PublicKey::from_sec1_bytes(&prefixed)
                    .map_err(|e| format!("Invalid ECDSA public key: {e}"))
            }
            n => Err(format!("ECDSA public key must be 64 or 65 bytes, got {n}")),
        }
    }
}

// ── High-level encrypt/decrypt string wrappers ──────────────────────

/// Encrypt a plaintext string, returning a hex-encoded ciphertext string.
///
/// Only uses `ctx.client_pub_key` and `ctx.algo`. The `signing` parameter is
/// accepted for API symmetry with `decrypt_string` but is not used for
/// encryption — the recipient's public key (from the client) is sufficient.
pub fn encrypt_string(
    plaintext: &str,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<String, AppError> {
    if plaintext.is_empty() {
        return Ok(String::new());
    }
    let _ = signing;
    let ciphertext = match ctx.algo {
        EncryptionAlgo::Ed25519 => {
            let client_ed25519: [u8; 32] =
                ctx.client_pub_key.as_slice().try_into().map_err(|_| {
                    AppError::BadRequest(format!(
                        "invalid Ed25519 public key length: expected 32 bytes, got {}",
                        ctx.client_pub_key.len()
                    ))
                })?;
            let client_x25519 =
                nacl::ed25519_public_to_x25519(&client_ed25519).map_err(AppError::BadRequest)?;
            let encrypt_fn = if ctx.version >= 2 {
                nacl_v2::encrypt
            } else {
                nacl::encrypt
            };
            encrypt_fn(plaintext.as_bytes(), &client_x25519)
                .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        }
        EncryptionAlgo::Ecdsa => {
            let client_pk =
                ecies::parse_client_pubkey(&ctx.client_pub_key).map_err(AppError::BadRequest)?;
            ecies::encrypt(plaintext.as_bytes(), &client_pk)
                .map_err(|e| AppError::Internal(anyhow::anyhow!(e)))?
        }
    };
    Ok(hex::encode(ciphertext))
}

/// Decrypt a hex-encoded ciphertext string, returning plaintext.
pub fn decrypt_string(
    hex_ciphertext: &str,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<String, AppError> {
    if hex_ciphertext.is_empty() {
        return Ok(String::new());
    }
    let data = hex::decode(hex_ciphertext)
        .map_err(|_| AppError::BadRequest("Encrypted field is not valid hex".to_string()))?;

    // Use a generic error message for all decryption failures to avoid
    // leaking oracle information (e.g. distinguishing "too short" from
    // "authentication tag mismatch").
    let plaintext_bytes = match ctx.algo {
        EncryptionAlgo::Ed25519 => {
            let x25519_secret = nacl::ed25519_secret_to_x25519(signing.ed25519.secret_bytes());
            let decrypt_fn = if ctx.version >= 2 {
                nacl_v2::decrypt
            } else {
                nacl::decrypt
            };
            decrypt_fn(&data, &x25519_secret)
                .map_err(|_| AppError::BadRequest("Decryption failed".to_string()))?
        }
        EncryptionAlgo::Ecdsa => ecies::decrypt(&data, signing.ecdsa.secret_key())
            .map_err(|_| AppError::BadRequest("Decryption failed".to_string()))?,
    };

    String::from_utf8(plaintext_bytes)
        .map_err(|_| AppError::BadRequest("Decrypted content is not valid UTF-8".to_string()))
}

// ── Field-level JSON transformers ───────────────────────────────────

/// Endpoint type for determining which fields to encrypt/decrypt.
#[derive(Debug, Clone, Copy)]
pub enum Endpoint {
    ChatCompletions,
    Completions,
    ImagesGenerations,
    ImagesEdits,
    Embeddings,
    AudioTranscriptions,
    Rerank,
    Score,
}

/// Decrypt request fields in-place based on the endpoint type.
pub fn decrypt_request_fields(
    value: &mut serde_json::Value,
    endpoint: Endpoint,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<(), AppError> {
    match endpoint {
        Endpoint::ChatCompletions => {
            if let Some(messages) = value.get_mut("messages").and_then(|m| m.as_array_mut()) {
                for msg in messages.iter_mut() {
                    decrypt_chat_message_fields(msg, ctx, signing)?;
                }
            }
            if ctx.encrypt_tool_calls {
                decrypt_tools_array(value, ctx, signing)?;
            }
        }
        Endpoint::Completions => {
            // prompt can be a string or an array of strings
            decrypt_field_or_array(value, "prompt", ctx, signing)?;
        }
        Endpoint::ImagesGenerations | Endpoint::ImagesEdits => {
            decrypt_field(value, "prompt", ctx, signing)?;
        }
        Endpoint::Embeddings => {
            // input can be a string or array of strings; non-string elements
            // (token ID arrays) pass through unchanged — matches Python proxy.
            decrypt_field_or_array(value, "input", ctx, signing)?;
        }
        Endpoint::AudioTranscriptions => {
            decrypt_field(value, "prompt", ctx, signing)?;
        }
        Endpoint::Rerank => {
            decrypt_field(value, "query", ctx, signing)?;
            decrypt_rerank_documents(value, ctx, signing)?;
        }
        Endpoint::Score => {
            decrypt_field(value, "text_1", ctx, signing)?;
            decrypt_field(value, "text_2", ctx, signing)?;
        }
    }
    Ok(())
}

/// Encrypt response fields in-place based on the endpoint type.
pub fn encrypt_response_fields(
    value: &mut serde_json::Value,
    endpoint: Endpoint,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<(), AppError> {
    match endpoint {
        Endpoint::ChatCompletions => {
            encrypt_chat_response_choices(value, ctx, signing, false)?;
        }
        Endpoint::Completions => {
            if let Some(choices) = value.get_mut("choices").and_then(|c| c.as_array_mut()) {
                for choice in choices.iter_mut() {
                    encrypt_field(choice, "text", ctx, signing)?;
                }
            }
        }
        Endpoint::ImagesGenerations | Endpoint::ImagesEdits => {
            if let Some(data) = value.get_mut("data").and_then(|d| d.as_array_mut()) {
                for item in data.iter_mut() {
                    encrypt_field(item, "b64_json", ctx, signing)?;
                    encrypt_field(item, "revised_prompt", ctx, signing)?;
                }
            }
        }
        Endpoint::Embeddings => {
            // Encrypt each embedding: serialize float array to JSON, then encrypt
            if let Some(data) = value.get_mut("data").and_then(|d| d.as_array_mut()) {
                for item in data.iter_mut() {
                    if let Some(embedding) = item.get_mut("embedding") {
                        if embedding.is_array() {
                            let json_str = serde_json::to_string(embedding)
                                .map_err(|e| AppError::Internal(e.into()))?;
                            let encrypted = encrypt_string(&json_str, ctx, signing)?;
                            *embedding = serde_json::Value::String(encrypted);
                        }
                    }
                }
            }
        }
        Endpoint::AudioTranscriptions => {
            encrypt_field(value, "text", ctx, signing)?;
        }
        Endpoint::Rerank => {
            if let Some(results) = value.get_mut("results").and_then(|r| r.as_array_mut()) {
                for result in results.iter_mut() {
                    if let Some(doc) = result.get_mut("document") {
                        encrypt_field(doc, "text", ctx, signing)?;
                    }
                }
            }
        }
        Endpoint::Score => {
            // Serialize score value to JSON string, then encrypt
            if let Some(score) = value.get_mut("score") {
                let json_str =
                    serde_json::to_string(score).map_err(|e| AppError::Internal(e.into()))?;
                let encrypted = encrypt_string(&json_str, ctx, signing)?;
                *score = serde_json::Value::String(encrypted);
            }
        }
    }
    Ok(())
}

/// Encrypt fields in a streaming SSE chunk, dispatching by endpoint type.
pub fn encrypt_streaming_chunk(
    value: &mut serde_json::Value,
    endpoint: Endpoint,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<(), AppError> {
    match endpoint {
        Endpoint::ChatCompletions => encrypt_chat_response_choices(value, ctx, signing, true),
        Endpoint::Completions => {
            // Completions streams emit choices[*].text (not delta)
            if let Some(choices) = value.get_mut("choices").and_then(|c| c.as_array_mut()) {
                for choice in choices.iter_mut() {
                    encrypt_field(choice, "text", ctx, signing)?;
                }
            }
            Ok(())
        }
        // These endpoints don't use streaming — no fields to encrypt
        Endpoint::ImagesGenerations
        | Endpoint::ImagesEdits
        | Endpoint::Embeddings
        | Endpoint::AudioTranscriptions
        | Endpoint::Rerank
        | Endpoint::Score => Ok(()),
    }
}

// ── Internal helpers ────────────────────────────────────────────────

fn decrypt_chat_message_fields(
    msg: &mut serde_json::Value,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<(), AppError> {
    // content can be a string (encrypted) or array of content parts.
    // When encrypted, the client sends a single encrypted string; after decryption
    // we try json.loads to restore multimodal content arrays (matching Python proxy).
    if let Some(content) = msg.get_mut("content") {
        if let Some(s) = content.as_str() {
            if !s.is_empty() {
                let decrypted = decrypt_string(s, ctx, signing)?;
                // Try to parse as JSON array (multimodal content parts)
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&decrypted) {
                    if parsed.is_array() {
                        *content = parsed;
                    } else {
                        *content = serde_json::Value::String(decrypted);
                    }
                } else {
                    *content = serde_json::Value::String(decrypted);
                }
            }
        }
        // If content is already an array, it was not encrypted — pass through
    }

    decrypt_field(msg, "reasoning_content", ctx, signing)?;
    decrypt_field(msg, "reasoning", ctx, signing)?;

    // audio.data
    if let Some(audio) = msg.get_mut("audio") {
        decrypt_field(audio, "data", ctx, signing)?;
    }

    if ctx.encrypt_tool_calls {
        decrypt_message_tool_calls(msg, ctx, signing)?;
    }

    Ok(())
}

fn encrypt_chat_response_choices(
    value: &mut serde_json::Value,
    ctx: &EncryptionContext,
    signing: &SigningPair,
    is_streaming: bool,
) -> Result<(), AppError> {
    if let Some(choices) = value.get_mut("choices").and_then(|c| c.as_array_mut()) {
        for choice in choices.iter_mut() {
            let msg_key = if is_streaming { "delta" } else { "message" };
            if let Some(msg) = choice.get_mut(msg_key) {
                encrypt_content_field(msg, ctx, signing)?;
                encrypt_field(msg, "reasoning_content", ctx, signing)?;
                encrypt_field(msg, "reasoning", ctx, signing)?;
                if let Some(audio) = msg.get_mut("audio") {
                    encrypt_field(audio, "data", ctx, signing)?;
                }
                if ctx.encrypt_tool_calls {
                    encrypt_message_tool_calls(msg, ctx, signing)?;
                }
            }
        }
    }
    Ok(())
}

fn encrypt_content_field(
    msg: &mut serde_json::Value,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<(), AppError> {
    if let Some(content) = msg.get_mut("content") {
        match content {
            serde_json::Value::String(s) => {
                if !s.is_empty() {
                    let encrypted = encrypt_string(s, ctx, signing)?;
                    *s = encrypted;
                }
            }
            serde_json::Value::Array(parts) => {
                for part in parts.iter_mut() {
                    encrypt_field(part, "text", ctx, signing)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Decrypt a single string field on an object, in-place.
fn decrypt_field(
    obj: &mut serde_json::Value,
    field: &str,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<(), AppError> {
    if let Some(val) = obj.get_mut(field) {
        if let Some(s) = val.as_str() {
            if !s.is_empty() {
                let s_owned = s.to_string();
                let decrypted = decrypt_string(&s_owned, ctx, signing)?;
                *val = serde_json::Value::String(decrypted);
            }
        }
    }
    Ok(())
}

/// Decrypt a field that may be a string or an array of encrypted strings.
/// Handles the completions `prompt` field which can be either form.
fn decrypt_field_or_array(
    obj: &mut serde_json::Value,
    field: &str,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<(), AppError> {
    if let Some(val) = obj.get_mut(field) {
        match val {
            serde_json::Value::String(s) => {
                if !s.is_empty() {
                    let decrypted = decrypt_string(s, ctx, signing)?;
                    *s = decrypted;
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr.iter_mut() {
                    if let Some(s) = item.as_str() {
                        if !s.is_empty() {
                            let s_owned = s.to_string();
                            let decrypted = decrypt_string(&s_owned, ctx, signing)?;
                            *item = serde_json::Value::String(decrypted);
                        }
                    }
                    // Non-string array elements (e.g. token ID arrays) are not encrypted
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Decrypt rerank `documents[]` in-place.
/// Each element can be a string (decrypt directly) or an object with a `text` field.
fn decrypt_rerank_documents(
    value: &mut serde_json::Value,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<(), AppError> {
    if let Some(docs) = value.get_mut("documents").and_then(|d| d.as_array_mut()) {
        for doc in docs.iter_mut() {
            match doc {
                serde_json::Value::String(s) => {
                    if !s.is_empty() {
                        let decrypted = decrypt_string(s, ctx, signing)?;
                        *s = decrypted;
                    }
                }
                serde_json::Value::Object(_) => {
                    decrypt_field(doc, "text", ctx, signing)?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Decrypt `tool_calls[].function.arguments` in a single message object.
fn decrypt_message_tool_calls(
    msg: &mut serde_json::Value,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<(), AppError> {
    if let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut()) {
        for tc in tool_calls.iter_mut() {
            if let Some(func) = tc.get_mut("function") {
                decrypt_field(func, "arguments", ctx, signing)?;
            }
        }
    }
    Ok(())
}

/// Decrypt `tools[].function.description` and `tools[].function.parameters` in-place.
/// The client serializes `parameters` (a JSON object) to a JSON string before encrypting it;
/// after decryption we parse it back to a JSON value.
fn decrypt_tools_array(
    value: &mut serde_json::Value,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<(), AppError> {
    if let Some(tools) = value.get_mut("tools").and_then(|t| t.as_array_mut()) {
        for tool in tools.iter_mut() {
            if let Some(func) = tool.get_mut("function") {
                decrypt_field(func, "description", ctx, signing)?;
                // parameters is a JSON object encrypted as a string — decrypt then re-parse.
                if let Some(params) = func.get_mut("parameters") {
                    if let serde_json::Value::String(s) = params {
                        if !s.is_empty() {
                            let decrypted = decrypt_string(s, ctx, signing)?;
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&decrypted)
                            {
                                *params = parsed;
                            } else {
                                *params = serde_json::Value::String(decrypted);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Encrypt `tool_calls[].function.arguments` in a message/delta object.
fn encrypt_message_tool_calls(
    msg: &mut serde_json::Value,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<(), AppError> {
    if let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut()) {
        for tc in tool_calls.iter_mut() {
            if let Some(func) = tc.get_mut("function") {
                encrypt_field(func, "arguments", ctx, signing)?;
            }
        }
    }
    Ok(())
}

/// Encrypt a single string field on an object, in-place.
fn encrypt_field(
    obj: &mut serde_json::Value,
    field: &str,
    ctx: &EncryptionContext,
    signing: &SigningPair,
) -> Result<(), AppError> {
    if let Some(val) = obj.get_mut(field) {
        if let Some(s) = val.as_str() {
            if !s.is_empty() {
                let s_owned = s.to_string();
                let encrypted = encrypt_string(&s_owned, ctx, signing)?;
                *val = serde_json::Value::String(encrypted);
            }
        }
    }
    Ok(())
}

/// Build a response transform closure for `ProxyOpts`.
pub fn make_response_transform(
    endpoint: Endpoint,
    ctx: EncryptionContext,
    signing: Arc<SigningPair>,
) -> ResponseTransform {
    Box::new(move |value: &mut serde_json::Value| {
        encrypt_response_fields(value, endpoint, &ctx, &signing)
    })
}

/// Build a streaming chunk transform closure for `ProxyOpts`.
pub fn make_chunk_transform(
    endpoint: Endpoint,
    ctx: EncryptionContext,
    signing: Arc<SigningPair>,
) -> ChunkTransform {
    Arc::new(move |value: &mut serde_json::Value| {
        encrypt_streaming_chunk(value, endpoint, &ctx, &signing)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signing::{EcdsaContext, Ed25519Context, SigningPair};

    const TEST_ECDSA_KEY: [u8; 32] = [
        0xac, 0x09, 0x74, 0xbe, 0xc3, 0x9a, 0x17, 0xe3, 0x6b, 0xa4, 0xa6, 0xb4, 0xd2, 0x38, 0xff,
        0x94, 0x4b, 0xac, 0xb3, 0x5e, 0x5d, 0xc4, 0xaf, 0x0f, 0x33, 0x47, 0xe5, 0x87, 0x31, 0x79,
        0x67, 0x0f,
    ];

    const TEST_ED25519_KEY: [u8; 32] = [
        0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c,
        0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae,
        0x7f, 0x60,
    ];

    fn test_signing_pair() -> SigningPair {
        let ecdsa = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
        let ed25519 = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();
        SigningPair { ecdsa, ed25519 }
    }

    // ── NaCl round-trip tests ───────────────────────────────────────

    #[test]
    fn test_nacl_round_trip() {
        let server_pair = test_signing_pair();
        let server_x25519_secret =
            nacl::ed25519_secret_to_x25519(server_pair.ed25519.secret_bytes());

        // Derive the server's X25519 public key from the secret
        let server_x25519_pub = {
            let sk = crypto_box::SecretKey::from(server_x25519_secret);
            *sk.public_key().as_bytes()
        };

        let plaintext = b"Hello, encrypted world!";
        let encrypted = nacl::encrypt(plaintext, &server_x25519_pub).unwrap();
        let decrypted = nacl::decrypt(&encrypted, &server_x25519_secret).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_nacl_empty_plaintext() {
        let server_pair = test_signing_pair();
        let server_x25519_secret =
            nacl::ed25519_secret_to_x25519(server_pair.ed25519.secret_bytes());
        let server_x25519_pub = {
            let sk = crypto_box::SecretKey::from(server_x25519_secret);
            *sk.public_key().as_bytes()
        };

        let encrypted = nacl::encrypt(b"", &server_x25519_pub).unwrap();
        let decrypted = nacl::decrypt(&encrypted, &server_x25519_secret).unwrap();
        assert_eq!(decrypted, b"");
    }

    #[test]
    fn test_nacl_large_payload() {
        let server_pair = test_signing_pair();
        let server_x25519_secret =
            nacl::ed25519_secret_to_x25519(server_pair.ed25519.secret_bytes());
        let server_x25519_pub = {
            let sk = crypto_box::SecretKey::from(server_x25519_secret);
            *sk.public_key().as_bytes()
        };

        let plaintext = vec![0x42u8; 100_000];
        let encrypted = nacl::encrypt(&plaintext, &server_x25519_pub).unwrap();
        let decrypted = nacl::decrypt(&encrypted, &server_x25519_secret).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_nacl_tampered_ciphertext() {
        let server_pair = test_signing_pair();
        let server_x25519_secret =
            nacl::ed25519_secret_to_x25519(server_pair.ed25519.secret_bytes());
        let server_x25519_pub = {
            let sk = crypto_box::SecretKey::from(server_x25519_secret);
            *sk.public_key().as_bytes()
        };

        let mut encrypted = nacl::encrypt(b"secret", &server_x25519_pub).unwrap();
        // Tamper with ciphertext byte
        let last = encrypted.len() - 1;
        encrypted[last] ^= 0xff;
        assert!(nacl::decrypt(&encrypted, &server_x25519_secret).is_err());
    }

    #[test]
    fn test_nacl_too_short() {
        let server_pair = test_signing_pair();
        let server_x25519_secret =
            nacl::ed25519_secret_to_x25519(server_pair.ed25519.secret_bytes());
        assert!(nacl::decrypt(&[0u8; 10], &server_x25519_secret).is_err());
    }

    /// Verify key conversion and decryption are compatible with libsodium/PyNaCl.
    /// Test vector generated by PyNaCl 1.6.2 (libsodium binding).
    #[test]
    fn test_nacl_libsodium_cross_compat() {
        let server_pair = test_signing_pair();
        let server_x25519_secret =
            nacl::ed25519_secret_to_x25519(server_pair.ed25519.secret_bytes());

        // Expected X25519 keys from libsodium's crypto_sign_ed25519_{sk,pk}_to_curve25519
        assert_eq!(
            hex::encode(server_x25519_secret),
            "307c83864f2833cb427a2ef1c00a013cfdff2768d980c0a3a520f006904de94f",
            "X25519 secret key must match libsodium"
        );

        let ed25519_pub_bytes: [u8; 32] = hex::decode(&server_pair.ed25519.signing_public_key)
            .unwrap()
            .try_into()
            .unwrap();
        let server_x25519_pub = nacl::ed25519_public_to_x25519(&ed25519_pub_bytes).unwrap();
        assert_eq!(
            hex::encode(server_x25519_pub),
            "d85e07ec22b0ad881537c2f44d662d1a143cf830c57aca4305d85c7a90f6b62e",
            "X25519 public key must match libsodium"
        );

        // Ciphertext produced by PyNaCl: Box(ephemeral_sk, server_x25519_pk).encrypt(b"cross-compat test")
        // Wire format: [ephemeral_pubkey(32)][nonce(24)][ciphertext+mac]
        let wire = hex::decode(
            "4f3d96155afaa5a7aa4a988fcbd318f973dd3ae41f2a31e91ab768e0774baa75\
             05b5b46e1a991ea89e91c900ae6ea5e7b5a915e30fd4307aa710fd21515eaf95\
             555ccfa2a1a3efcad963fd7bc5bf2ece02e0b17a9fe0f55fba",
        )
        .unwrap();

        let decrypted = nacl::decrypt(&wire, &server_x25519_secret).unwrap();
        assert_eq!(decrypted, b"cross-compat test");
    }

    /// Verify that the primary ChaCha20 path works with known Ed25519 keys.
    #[test]
    fn test_nacl_chacha_round_trip_known_keys() {
        let server_pair = test_signing_pair();
        let server_x25519_secret =
            nacl::ed25519_secret_to_x25519(server_pair.ed25519.secret_bytes());

        let ed25519_pub_bytes: [u8; 32] = hex::decode(&server_pair.ed25519.signing_public_key)
            .unwrap()
            .try_into()
            .unwrap();
        let server_x25519_pub = nacl::ed25519_public_to_x25519(&ed25519_pub_bytes).unwrap();

        let plaintext = b"chacha20 round-trip test";
        let encrypted = nacl::encrypt(plaintext, &server_x25519_pub).unwrap();
        let decrypted = nacl::decrypt(&encrypted, &server_x25519_secret).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    // ── NaCl v2 (HKDF + XChaCha20-Poly1305) tests ────────────────────

    #[test]
    fn test_nacl_v2_round_trip() {
        let server_pair = test_signing_pair();
        let server_x25519_secret =
            nacl::ed25519_secret_to_x25519(server_pair.ed25519.secret_bytes());
        let server_x25519_pub = {
            let sk = x25519_dalek::StaticSecret::from(server_x25519_secret);
            *x25519_dalek::PublicKey::from(&sk).as_bytes()
        };

        let plaintext = b"Hello from v2!";
        let encrypted = nacl_v2::encrypt(plaintext, &server_x25519_pub).unwrap();
        let decrypted = nacl_v2::decrypt(&encrypted, &server_x25519_secret).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_nacl_v2_empty_plaintext() {
        let server_pair = test_signing_pair();
        let server_x25519_secret =
            nacl::ed25519_secret_to_x25519(server_pair.ed25519.secret_bytes());
        let server_x25519_pub = {
            let sk = x25519_dalek::StaticSecret::from(server_x25519_secret);
            *x25519_dalek::PublicKey::from(&sk).as_bytes()
        };

        let encrypted = nacl_v2::encrypt(b"", &server_x25519_pub).unwrap();
        let decrypted = nacl_v2::decrypt(&encrypted, &server_x25519_secret).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn test_nacl_v2_tampered_ciphertext() {
        let server_pair = test_signing_pair();
        let server_x25519_secret =
            nacl::ed25519_secret_to_x25519(server_pair.ed25519.secret_bytes());
        let server_x25519_pub = {
            let sk = x25519_dalek::StaticSecret::from(server_x25519_secret);
            *x25519_dalek::PublicKey::from(&sk).as_bytes()
        };

        let mut encrypted = nacl_v2::encrypt(b"secret", &server_x25519_pub).unwrap();
        let last = encrypted.len() - 1;
        encrypted[last] ^= 0xff;
        assert!(nacl_v2::decrypt(&encrypted, &server_x25519_secret).is_err());
    }

    #[test]
    fn test_nacl_v2_too_short() {
        let server_pair = test_signing_pair();
        let server_x25519_secret =
            nacl::ed25519_secret_to_x25519(server_pair.ed25519.secret_bytes());
        assert!(nacl_v2::decrypt(&[0u8; 10], &server_x25519_secret).is_err());
    }

    /// Verify that v1 (NaCl box) and v2 (HKDF) ciphertexts are NOT cross-compatible.
    #[test]
    fn test_nacl_v1_v2_not_cross_compatible() {
        let server_pair = test_signing_pair();
        let server_x25519_secret =
            nacl::ed25519_secret_to_x25519(server_pair.ed25519.secret_bytes());
        let server_x25519_pub = {
            let sk = x25519_dalek::StaticSecret::from(server_x25519_secret);
            *x25519_dalek::PublicKey::from(&sk).as_bytes()
        };

        // v1 ciphertext cannot be decrypted by v2
        let v1_encrypted = nacl::encrypt(b"v1 message", &server_x25519_pub).unwrap();
        assert!(nacl_v2::decrypt(&v1_encrypted, &server_x25519_secret).is_err());

        // v2 ciphertext cannot be decrypted by v1
        let v2_encrypted = nacl_v2::encrypt(b"v2 message", &server_x25519_pub).unwrap();
        assert!(nacl::decrypt(&v2_encrypted, &server_x25519_secret).is_err());
    }

    // ── ECIES round-trip tests ──────────────────────────────────────

    #[test]
    fn test_ecies_round_trip() {
        let server_pair = test_signing_pair();
        let server_pk = server_pair.ecdsa.secret_key().verifying_key();
        let k256_pub = k256::PublicKey::from(server_pk);

        let plaintext = b"Hello, ECIES!";
        let encrypted = ecies::encrypt(plaintext, &k256_pub).unwrap();
        let decrypted = ecies::decrypt(&encrypted, server_pair.ecdsa.secret_key()).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_ecies_empty_plaintext() {
        let server_pair = test_signing_pair();
        let server_pk = server_pair.ecdsa.secret_key().verifying_key();
        let k256_pub = k256::PublicKey::from(server_pk);

        let encrypted = ecies::encrypt(b"", &k256_pub).unwrap();
        let decrypted = ecies::decrypt(&encrypted, server_pair.ecdsa.secret_key()).unwrap();
        assert_eq!(decrypted, b"");
    }

    #[test]
    fn test_ecies_large_payload() {
        let server_pair = test_signing_pair();
        let server_pk = server_pair.ecdsa.secret_key().verifying_key();
        let k256_pub = k256::PublicKey::from(server_pk);

        let plaintext = vec![0x42u8; 100_000];
        let encrypted = ecies::encrypt(&plaintext, &k256_pub).unwrap();
        let decrypted = ecies::decrypt(&encrypted, server_pair.ecdsa.secret_key()).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_ecies_tampered_ciphertext() {
        let server_pair = test_signing_pair();
        let server_pk = server_pair.ecdsa.secret_key().verifying_key();
        let k256_pub = k256::PublicKey::from(server_pk);

        let mut encrypted = ecies::encrypt(b"secret", &k256_pub).unwrap();
        let last = encrypted.len() - 1;
        encrypted[last] ^= 0xff;
        assert!(ecies::decrypt(&encrypted, server_pair.ecdsa.secret_key()).is_err());
    }

    #[test]
    fn test_ecies_too_short() {
        let server_pair = test_signing_pair();
        assert!(ecies::decrypt(&[0u8; 10], server_pair.ecdsa.secret_key()).is_err());
    }

    #[test]
    fn test_ecies_parse_client_pubkey_65_bytes() {
        let server_pair = test_signing_pair();
        let vk = server_pair.ecdsa.secret_key().verifying_key();
        let encoded = vk.to_encoded_point(false);
        let bytes = encoded.as_bytes();
        assert_eq!(bytes.len(), 65);
        assert!(ecies::parse_client_pubkey(bytes).is_ok());
    }

    #[test]
    fn test_ecies_parse_client_pubkey_64_bytes() {
        let server_pair = test_signing_pair();
        let vk = server_pair.ecdsa.secret_key().verifying_key();
        let encoded = vk.to_encoded_point(false);
        let bytes = &encoded.as_bytes()[1..]; // strip 0x04 prefix
        assert_eq!(bytes.len(), 64);
        assert!(ecies::parse_client_pubkey(bytes).is_ok());
    }

    // ── High-level encrypt/decrypt string tests ─────────────────────

    #[test]
    fn test_encrypt_decrypt_string_ed25519() {
        let server_pair = test_signing_pair();

        // Generate a "client" keypair (reuse server key for test simplicity)
        let client_ed25519_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();

        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_ed25519_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let plaintext = "Hello from client!";
        let encrypted = encrypt_string(plaintext, &ctx, &server_pair).unwrap();
        assert!(!encrypted.is_empty());
        assert_ne!(encrypted, plaintext);

        // Decrypt with server key (which is same as client key in this test)
        let decrypted = decrypt_string(&encrypted, &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_decrypt_string_ecdsa() {
        let server_pair = test_signing_pair();

        // Use server's ECDSA public key as "client" public key
        let vk = server_pair.ecdsa.secret_key().verifying_key();
        let encoded = vk.to_encoded_point(false);
        let client_pub = encoded.as_bytes()[1..].to_vec(); // 64 bytes raw

        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ecdsa,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let plaintext = "Hello from client!";
        let encrypted = encrypt_string(plaintext, &ctx, &server_pair).unwrap();
        assert!(!encrypted.is_empty());

        let decrypted = decrypt_string(&encrypted, &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_decrypt_empty_string() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let encrypted = encrypt_string("", &ctx, &server_pair).unwrap();
        assert_eq!(encrypted, "");

        let decrypted = decrypt_string("", &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, "");
    }

    #[test]
    fn test_encrypt_decrypt_unicode() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let plaintext = "こんにちは世界 🌍 Привет мир";
        let encrypted = encrypt_string(plaintext, &ctx, &server_pair).unwrap();
        let decrypted = decrypt_string(&encrypted, &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    // ── Field-level transformer tests ───────────────────────────────

    #[test]
    fn test_decrypt_chat_message_string_content() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        // Encrypt "Hello" to simulate client-encrypted message
        let encrypted_hello = encrypt_string("Hello", &ctx, &server_pair).unwrap();

        let mut request = serde_json::json!({
            "messages": [
                {"role": "user", "content": encrypted_hello}
            ]
        });

        decrypt_request_fields(&mut request, Endpoint::ChatCompletions, &ctx, &server_pair)
            .unwrap();

        assert_eq!(request["messages"][0]["content"], "Hello");
    }

    #[test]
    fn test_decrypt_chat_message_array_content() {
        // When content is an encrypted string that decrypts to a JSON array,
        // it should be restored to the array form (matching Python proxy's json.loads behavior).
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        // Client encrypts a JSON-serialized content array
        let content_array = serde_json::json!([
            {"type": "text", "text": "Hello world"},
            {"type": "image_url", "image_url": {"url": "data:..."}}
        ]);
        let encrypted_content = encrypt_string(
            &serde_json::to_string(&content_array).unwrap(),
            &ctx,
            &server_pair,
        )
        .unwrap();

        let mut request = serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": encrypted_content
                }
            ]
        });

        decrypt_request_fields(&mut request, Endpoint::ChatCompletions, &ctx, &server_pair)
            .unwrap();

        // Content should be restored to the original array
        assert_eq!(request["messages"][0]["content"][0]["text"], "Hello world");
        assert_eq!(request["messages"][0]["content"][1]["type"], "image_url");
    }

    #[test]
    fn test_encrypt_chat_response() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let mut response = serde_json::json!({
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!"
                    }
                }
            ]
        });

        encrypt_response_fields(&mut response, Endpoint::ChatCompletions, &ctx, &server_pair)
            .unwrap();

        let encrypted_content = response["choices"][0]["message"]["content"]
            .as_str()
            .unwrap();
        assert_ne!(encrypted_content, "Hello!");
        assert!(!encrypted_content.is_empty());

        // Verify we can decrypt it
        let decrypted = decrypt_string(encrypted_content, &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, "Hello!");
    }

    #[test]
    fn test_encrypt_streaming_chunk() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let mut chunk = serde_json::json!({
            "id": "chatcmpl-123",
            "choices": [
                {
                    "delta": {
                        "content": "Hi"
                    }
                }
            ]
        });

        super::encrypt_streaming_chunk(&mut chunk, Endpoint::ChatCompletions, &ctx, &server_pair)
            .unwrap();

        let encrypted = chunk["choices"][0]["delta"]["content"].as_str().unwrap();
        assert_ne!(encrypted, "Hi");

        let decrypted = decrypt_string(encrypted, &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, "Hi");
    }

    #[test]
    fn test_encrypt_completions_response() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let mut response = serde_json::json!({
            "choices": [
                {"text": "completed text"}
            ]
        });

        encrypt_response_fields(&mut response, Endpoint::Completions, &ctx, &server_pair).unwrap();

        let encrypted = response["choices"][0]["text"].as_str().unwrap();
        let decrypted = decrypt_string(encrypted, &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, "completed text");
    }

    #[test]
    fn test_encrypt_images_response() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let mut response = serde_json::json!({
            "data": [
                {"b64_json": "base64imagedata"}
            ]
        });

        encrypt_response_fields(
            &mut response,
            Endpoint::ImagesGenerations,
            &ctx,
            &server_pair,
        )
        .unwrap();

        let encrypted = response["data"][0]["b64_json"].as_str().unwrap();
        let decrypted = decrypt_string(encrypted, &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, "base64imagedata");
    }

    #[test]
    fn test_encrypt_audio_transcription_response() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let mut response = serde_json::json!({
            "text": "transcribed text"
        });

        encrypt_response_fields(
            &mut response,
            Endpoint::AudioTranscriptions,
            &ctx,
            &server_pair,
        )
        .unwrap();

        let encrypted = response["text"].as_str().unwrap();
        let decrypted = decrypt_string(encrypted, &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, "transcribed text");
    }

    #[test]
    fn test_embeddings_response_encrypted() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let mut response = serde_json::json!({
            "data": [
                {"embedding": [0.1, 0.2, 0.3]}
            ]
        });

        encrypt_response_fields(&mut response, Endpoint::Embeddings, &ctx, &server_pair).unwrap();
        // Embedding should now be an encrypted hex string, not an array
        let encrypted = response["data"][0]["embedding"].as_str().unwrap();
        let decrypted = decrypt_string(encrypted, &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, "[0.1,0.2,0.3]");
    }

    #[test]
    fn test_null_content_not_encrypted() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let mut response = serde_json::json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": null
                    }
                }
            ]
        });

        encrypt_response_fields(&mut response, Endpoint::ChatCompletions, &ctx, &server_pair)
            .unwrap();
        assert!(response["choices"][0]["message"]["content"].is_null());
    }

    // ── Header extraction tests ─────────────────────────────────────

    #[test]
    fn test_extract_encryption_context_none() {
        let headers = axum::http::HeaderMap::new();
        assert!(extract_encryption_context(&headers).unwrap().is_none());
    }

    #[test]
    fn test_extract_encryption_context_missing_one_header() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-signing-algo", "ed25519".parse().unwrap());
        assert!(extract_encryption_context(&headers).is_err());

        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-client-pub-key", "abcd".parse().unwrap());
        assert!(extract_encryption_context(&headers).is_err());
    }

    #[test]
    fn test_extract_encryption_context_invalid_algo() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-signing-algo", "rsa".parse().unwrap());
        headers.insert("x-client-pub-key", "aa".repeat(32).parse().unwrap());
        assert!(extract_encryption_context(&headers).is_err());
    }

    #[test]
    fn test_extract_encryption_context_invalid_hex() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-signing-algo", "ed25519".parse().unwrap());
        headers.insert("x-client-pub-key", "not-hex".parse().unwrap());
        assert!(extract_encryption_context(&headers).is_err());
    }

    #[test]
    fn test_extract_encryption_context_wrong_key_length() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-signing-algo", "ed25519".parse().unwrap());
        headers.insert("x-client-pub-key", "aabb".parse().unwrap()); // 2 bytes, need 32
        assert!(extract_encryption_context(&headers).is_err());
    }

    #[test]
    fn test_extract_encryption_context_valid_ed25519() {
        let server_pair = test_signing_pair();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-signing-algo", "ed25519".parse().unwrap());
        headers.insert(
            "x-client-pub-key",
            server_pair.ed25519.signing_public_key.parse().unwrap(),
        );
        let ctx = extract_encryption_context(&headers).unwrap().unwrap();
        assert_eq!(ctx.algo, EncryptionAlgo::Ed25519);
        assert_eq!(ctx.client_pub_key.len(), 32);
    }

    // ── Fix 2: Completions streaming encryption ───────────────────

    #[test]
    fn test_encrypt_streaming_chunk_completions() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let mut chunk = serde_json::json!({
            "id": "cmpl-123",
            "choices": [
                {
                    "text": "hello world"
                }
            ]
        });

        super::encrypt_streaming_chunk(&mut chunk, Endpoint::Completions, &ctx, &server_pair)
            .unwrap();

        let encrypted = chunk["choices"][0]["text"].as_str().unwrap();
        assert_ne!(encrypted, "hello world");

        let decrypted = decrypt_string(encrypted, &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, "hello world");
    }

    // ── Embeddings input decryption ──────────────────────────────────

    #[test]
    fn test_decrypt_embeddings_input_string() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        // Client encrypts the input string directly (matching Python proxy's decrypt_prompt)
        let encrypted = encrypt_string("hello world", &ctx, &server_pair).unwrap();

        let mut request = serde_json::json!({
            "input": encrypted,
            "model": "text-embedding"
        });

        decrypt_request_fields(&mut request, Endpoint::Embeddings, &ctx, &server_pair).unwrap();

        assert_eq!(request["input"], serde_json::json!("hello world"));
    }

    #[test]
    fn test_decrypt_embeddings_input_array() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        // Client encrypts each string element individually (matching Python proxy)
        let enc_hello = encrypt_string("hello", &ctx, &server_pair).unwrap();
        let enc_world = encrypt_string("world", &ctx, &server_pair).unwrap();

        let mut request = serde_json::json!({
            "input": [enc_hello, enc_world],
            "model": "text-embedding"
        });

        decrypt_request_fields(&mut request, Endpoint::Embeddings, &ctx, &server_pair).unwrap();

        assert_eq!(request["input"], serde_json::json!(["hello", "world"]));
    }

    #[test]
    fn test_decrypt_embeddings_input_token_array() {
        // Token ID arrays are not encrypted — they pass through unchanged
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let mut request = serde_json::json!({
            "input": [[1, 2, 3], [4, 5, 6]],
            "model": "text-embedding"
        });

        decrypt_request_fields(&mut request, Endpoint::Embeddings, &ctx, &server_pair).unwrap();

        // Token arrays pass through unchanged
        assert_eq!(request["input"], serde_json::json!([[1, 2, 3], [4, 5, 6]]));
    }

    // ── Decryption error message is generic (#3) ─────────────────────

    #[test]
    fn test_decrypt_error_is_generic() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        // Tampered ciphertext — should fail with generic message
        let bad_hex = hex::encode(vec![0u8; 100]);
        let err = decrypt_string(&bad_hex, &ctx, &server_pair).unwrap_err();
        let err_msg = format!("{err}");
        // Must not contain crypto implementation details like "tag mismatch" or "too short"
        assert!(
            err_msg.contains("Decryption failed"),
            "Error should be generic, got: {err_msg}"
        );
        assert!(
            !err_msg.contains("NaCl") && !err_msg.contains("AES") && !err_msg.contains("tag"),
            "Error should not leak crypto details, got: {err_msg}"
        );
    }

    // Embeddings input type validation removed — now using decrypt_field_or_array
    // which matches Python proxy behavior (decrypt strings, skip non-strings).

    // ── Completions prompt array decryption (#9) ───────────────────

    #[test]
    fn test_decrypt_completions_prompt_string() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let encrypted = encrypt_string("hello", &ctx, &server_pair).unwrap();

        let mut request = serde_json::json!({
            "prompt": encrypted,
            "model": "test"
        });

        decrypt_request_fields(&mut request, Endpoint::Completions, &ctx, &server_pair).unwrap();
        assert_eq!(request["prompt"], "hello");
    }

    #[test]
    fn test_decrypt_completions_prompt_array() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        let enc1 = encrypt_string("hello", &ctx, &server_pair).unwrap();
        let enc2 = encrypt_string("world", &ctx, &server_pair).unwrap();

        let mut request = serde_json::json!({
            "prompt": [enc1, enc2],
            "model": "test"
        });

        decrypt_request_fields(&mut request, Endpoint::Completions, &ctx, &server_pair).unwrap();
        assert_eq!(request["prompt"][0], "hello");
        assert_eq!(request["prompt"][1], "world");
    }

    #[test]
    fn test_decrypt_completions_prompt_array_with_tokens() {
        let server_pair = test_signing_pair();
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        let ctx = EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls: false,
        };

        // Array with mixed types: encrypted strings and token ID arrays (not encrypted)
        let enc1 = encrypt_string("hello", &ctx, &server_pair).unwrap();

        let mut request = serde_json::json!({
            "prompt": [enc1, [1, 2, 3]],
            "model": "test"
        });

        decrypt_request_fields(&mut request, Endpoint::Completions, &ctx, &server_pair).unwrap();
        assert_eq!(request["prompt"][0], "hello");
        // Token arrays are left unchanged
        assert_eq!(request["prompt"][1], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_extract_encryption_context_valid_ecdsa() {
        let server_pair = test_signing_pair();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-signing-algo", "ecdsa".parse().unwrap());
        headers.insert(
            "x-client-pub-key",
            server_pair.ecdsa.signing_public_key.parse().unwrap(), // 128 hex = 64 bytes
        );
        let ctx = extract_encryption_context(&headers).unwrap().unwrap();
        assert_eq!(ctx.algo, EncryptionAlgo::Ecdsa);
        assert_eq!(ctx.client_pub_key.len(), 64);
    }

    // ── Tool call encryption tests ──────────────────────────────────

    fn make_tool_call_ctx(server_pair: &SigningPair, encrypt_tool_calls: bool) -> EncryptionContext {
        let client_pub = hex::decode(&server_pair.ed25519.signing_public_key).unwrap();
        EncryptionContext {
            algo: EncryptionAlgo::Ed25519,
            client_pub_key: client_pub,
            version: 1,
            encrypt_tool_calls,
        }
    }

    #[test]
    fn test_encrypt_tool_calls_in_response() {
        let server_pair = test_signing_pair();
        let ctx = make_tool_call_ctx(&server_pair, true);

        let mut response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"San Francisco\"}"
                        }
                    }]
                }
            }]
        });

        encrypt_response_fields(&mut response, Endpoint::ChatCompletions, &ctx, &server_pair)
            .unwrap();

        let encrypted_args = response["choices"][0]["message"]["tool_calls"][0]["function"]
            ["arguments"]
            .as_str()
            .unwrap();
        assert_ne!(encrypted_args, "{\"location\":\"San Francisco\"}");

        let decrypted = decrypt_string(encrypted_args, &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, "{\"location\":\"San Francisco\"}");
    }

    #[test]
    fn test_tool_calls_not_encrypted_without_flag() {
        let server_pair = test_signing_pair();
        let ctx = make_tool_call_ctx(&server_pair, false);

        let args = "{\"location\":\"San Francisco\"}";
        let mut response = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": args}
                    }]
                }
            }]
        });

        encrypt_response_fields(&mut response, Endpoint::ChatCompletions, &ctx, &server_pair)
            .unwrap();

        // arguments must be unchanged when flag is off
        assert_eq!(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
            args
        );
    }

    #[test]
    fn test_encrypt_tool_calls_in_streaming_chunk() {
        let server_pair = test_signing_pair();
        let ctx = make_tool_call_ctx(&server_pair, true);

        let mut chunk = serde_json::json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "{\"city\":\"Tokyo\"}"}
                    }]
                }
            }]
        });

        encrypt_streaming_chunk(&mut chunk, Endpoint::ChatCompletions, &ctx, &server_pair)
            .unwrap();

        let encrypted = chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert_ne!(encrypted, "{\"city\":\"Tokyo\"}");
        let decrypted = decrypt_string(encrypted, &ctx, &server_pair).unwrap();
        assert_eq!(decrypted, "{\"city\":\"Tokyo\"}");
    }

    #[test]
    fn test_decrypt_message_tool_calls_in_request() {
        let server_pair = test_signing_pair();
        let ctx = make_tool_call_ctx(&server_pair, true);

        let encrypted_args =
            encrypt_string("{\"location\":\"Paris\"}", &ctx, &server_pair).unwrap();

        let mut request = serde_json::json!({
            "messages": [{
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": encrypted_args
                    }
                }]
            }]
        });

        decrypt_request_fields(&mut request, Endpoint::ChatCompletions, &ctx, &server_pair)
            .unwrap();

        assert_eq!(
            request["messages"][0]["tool_calls"][0]["function"]["arguments"],
            "{\"location\":\"Paris\"}"
        );
    }

    #[test]
    fn test_decrypt_tools_array_in_request() {
        let server_pair = test_signing_pair();
        let ctx = make_tool_call_ctx(&server_pair, true);

        let encrypted_desc = encrypt_string("Get the weather", &ctx, &server_pair).unwrap();
        let params_obj = serde_json::json!({"type": "object", "properties": {}});
        let encrypted_params = encrypt_string(
            &serde_json::to_string(&params_obj).unwrap(),
            &ctx,
            &server_pair,
        )
        .unwrap();

        let mut request = serde_json::json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": encrypted_desc,
                    "parameters": encrypted_params
                }
            }]
        });

        decrypt_request_fields(&mut request, Endpoint::ChatCompletions, &ctx, &server_pair)
            .unwrap();

        assert_eq!(
            request["tools"][0]["function"]["description"],
            "Get the weather"
        );
        assert_eq!(request["tools"][0]["function"]["parameters"], params_obj);
    }

    #[test]
    fn test_tools_array_not_decrypted_without_flag() {
        let server_pair = test_signing_pair();
        let ctx = make_tool_call_ctx(&server_pair, false);

        // Even if the description looks like ciphertext, it should pass through unchanged
        let raw_desc = "plaintext description";
        let mut request = serde_json::json!({
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "description": raw_desc}
            }]
        });

        decrypt_request_fields(&mut request, Endpoint::ChatCompletions, &ctx, &server_pair)
            .unwrap();

        assert_eq!(
            request["tools"][0]["function"]["description"],
            raw_desc
        );
    }

    #[test]
    fn test_extract_encryption_context_encrypt_tool_calls_header() {
        let server_pair = test_signing_pair();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-signing-algo", "ed25519".parse().unwrap());
        headers.insert(
            "x-client-pub-key",
            server_pair.ed25519.signing_public_key.parse().unwrap(),
        );
        headers.insert("x-encrypt-tool-calls", "true".parse().unwrap());
        let ctx = extract_encryption_context(&headers).unwrap().unwrap();
        assert!(ctx.encrypt_tool_calls);

        // Also test the "1" form
        let mut headers2 = axum::http::HeaderMap::new();
        headers2.insert("x-signing-algo", "ed25519".parse().unwrap());
        headers2.insert(
            "x-client-pub-key",
            server_pair.ed25519.signing_public_key.parse().unwrap(),
        );
        headers2.insert("x-encrypt-tool-calls", "1".parse().unwrap());
        let ctx2 = extract_encryption_context(&headers2).unwrap().unwrap();
        assert!(ctx2.encrypt_tool_calls);

        // Case-insensitive: "True" and "TRUE" should also work
        for val in &["True", "TRUE"] {
            let mut h = axum::http::HeaderMap::new();
            h.insert("x-signing-algo", "ed25519".parse().unwrap());
            h.insert(
                "x-client-pub-key",
                server_pair.ed25519.signing_public_key.parse().unwrap(),
            );
            h.insert("x-encrypt-tool-calls", val.parse().unwrap());
            let c = extract_encryption_context(&h).unwrap().unwrap();
            assert!(c.encrypt_tool_calls, "failed for value {val}");
        }
    }

    #[test]
    fn test_extract_encryption_context_encrypt_tool_calls_default_false() {
        let server_pair = test_signing_pair();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("x-signing-algo", "ed25519".parse().unwrap());
        headers.insert(
            "x-client-pub-key",
            server_pair.ed25519.signing_public_key.parse().unwrap(),
        );
        let ctx = extract_encryption_context(&headers).unwrap().unwrap();
        assert!(!ctx.encrypt_tool_calls);
    }
}
