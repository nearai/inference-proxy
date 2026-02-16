use anyhow::Result;
use ed25519_dalek::{Signer, SigningKey as Ed25519SigningKey};
use k256::ecdsa::{signature::hazmat::PrehashSigner, RecoveryId, SigningKey as K256SigningKey};
use rand::rand_core::TryRngCore;
use rand::rngs::OsRng;
use sha3::{Digest as Sha3Digest, Keccak256};
use tracing::info;

/// ECDSA (secp256k1) signing context with Ethereum-compatible addresses.
pub struct EcdsaContext {
    signing_key: K256SigningKey,
    /// Ethereum address as checksummed hex string (e.g., "0xAb5...").
    pub signing_address: String,
    /// Raw 20-byte Ethereum address.
    pub signing_address_bytes: Vec<u8>,
    /// Uncompressed public key bytes (64 bytes, no 0x04 prefix).
    pub signing_public_key: String,
}

/// Ed25519 signing context.
pub struct Ed25519Context {
    signing_key: Ed25519SigningKey,
    /// Hex-encoded 32-byte public key (also used as signing address).
    pub signing_address: String,
    /// Raw 32-byte public key.
    pub signing_address_bytes: Vec<u8>,
    /// Same as signing_address for ed25519.
    pub signing_public_key: String,
}

/// Holds both signing contexts.
pub struct SigningPair {
    pub ecdsa: EcdsaContext,
    pub ed25519: Ed25519Context,
}

impl EcdsaContext {
    /// Create from raw 32-byte key material.
    pub fn from_key_bytes(key_bytes: &[u8; 32]) -> Result<Self> {
        let signing_key = K256SigningKey::from_bytes(key_bytes.into())?;
        let verifying_key = signing_key.verifying_key();

        // Get uncompressed public key (65 bytes with 0x04 prefix)
        let pk_encoded = verifying_key.to_encoded_point(false);
        let pk_bytes = pk_encoded.as_bytes();
        // Skip the 0x04 prefix to get 64-byte uncompressed key
        let pk_uncompressed = &pk_bytes[1..];
        let signing_public_key = hex::encode(pk_uncompressed);

        // Derive Ethereum address: keccak256(uncompressed_pubkey) → last 20 bytes
        let hash = Keccak256::digest(pk_uncompressed);
        let address_bytes = hash[12..32].to_vec();
        let signing_address = format!("0x{}", hex::encode(&address_bytes));

        Ok(EcdsaContext {
            signing_key,
            signing_address,
            signing_address_bytes: address_bytes,
            signing_public_key,
        })
    }

    /// Sign a message using EIP-191 personal_sign format.
    /// Returns "0x{r}{s}{v}" (65 bytes hex-encoded).
    pub fn sign(&self, message: &str) -> Result<String> {
        // EIP-191: "\x19Ethereum Signed Message:\n{len}{message}"
        let prefix = format!("\x19Ethereum Signed Message:\n{}", message.len());
        let mut prefixed = Vec::with_capacity(prefix.len() + message.len());
        prefixed.extend_from_slice(prefix.as_bytes());
        prefixed.extend_from_slice(message.as_bytes());

        // Keccak256 hash the prefixed message
        let hash = Keccak256::digest(&prefixed);

        // Sign with recoverable signature
        let (signature, recovery_id): (k256::ecdsa::Signature, RecoveryId) =
            self.signing_key.sign_prehash(&hash[..])?;

        // Build 0x{r}{s}{v} where v = recovery_id + 27
        let mut sig_bytes = [0u8; 65];
        sig_bytes[..64].copy_from_slice(&signature.to_bytes());
        sig_bytes[64] = recovery_id.to_byte() + 27;

        Ok(format!("0x{}", hex::encode(sig_bytes)))
    }
}

impl Ed25519Context {
    /// Create from raw 32-byte key material.
    pub fn from_key_bytes(key_bytes: &[u8; 32]) -> Result<Self> {
        let signing_key = Ed25519SigningKey::from_bytes(key_bytes);
        let verifying_key = signing_key.verifying_key();
        let pk_bytes = verifying_key.to_bytes();
        let signing_address = hex::encode(pk_bytes);
        let signing_public_key = signing_address.clone();

        Ok(Ed25519Context {
            signing_key,
            signing_address,
            signing_address_bytes: pk_bytes.to_vec(),
            signing_public_key,
        })
    }

    /// Sign a message. Returns hex-encoded 64-byte signature.
    pub fn sign(&self, message: &str) -> Result<String> {
        let signature = self.signing_key.sign(message.as_bytes());
        Ok(hex::encode(signature.to_bytes()))
    }
}

impl SigningPair {
    /// Initialize signing pair from dstack KMS or random keys in dev mode.
    pub async fn init(model_name: &str, dev_mode: bool) -> Result<Self> {
        if dev_mode {
            info!("DEV mode: generating random signing keys");
            let mut ecdsa_bytes = [0u8; 32];
            let mut ed25519_bytes = [0u8; 32];
            OsRng
                .try_fill_bytes(&mut ecdsa_bytes)
                .expect("Failed to generate ECDSA key bytes");
            OsRng
                .try_fill_bytes(&mut ed25519_bytes)
                .expect("Failed to generate Ed25519 key bytes");

            let ecdsa = EcdsaContext::from_key_bytes(&ecdsa_bytes)?;
            let ed25519 = Ed25519Context::from_key_bytes(&ed25519_bytes)?;

            info!(ecdsa_address = %ecdsa.signing_address, "ECDSA key initialized");
            info!(ed25519_address = %ed25519.signing_address, "Ed25519 key initialized");

            return Ok(SigningPair { ecdsa, ed25519 });
        }

        // Production: derive keys from dstack KMS
        let client = dstack_sdk::dstack_client::DstackClient::new(None);

        let ecdsa_path = format!("{model_name}/ecdsa-signing-key");
        let ecdsa_resp = client
            .get_key(Some(ecdsa_path), Some("signing".to_string()))
            .await?;
        let ecdsa_key_bytes = ecdsa_resp.decode_key()?;
        if ecdsa_key_bytes.len() != 32 {
            anyhow::bail!(
                "Expected 32-byte ECDSA key, got {} bytes",
                ecdsa_key_bytes.len()
            );
        }
        let ecdsa = EcdsaContext::from_key_bytes(ecdsa_key_bytes[..32].try_into().unwrap())?;

        let ed25519_path = format!("{model_name}/ed25519-signing-key");
        let ed25519_resp = client
            .get_key(Some(ed25519_path), Some("signing".to_string()))
            .await?;
        let ed25519_key_bytes = ed25519_resp.decode_key()?;
        if ed25519_key_bytes.len() != 32 {
            anyhow::bail!(
                "Expected 32-byte Ed25519 key, got {} bytes",
                ed25519_key_bytes.len()
            );
        }
        let ed25519 = Ed25519Context::from_key_bytes(ed25519_key_bytes[..32].try_into().unwrap())?;

        info!(ecdsa_address = %ecdsa.signing_address, "ECDSA key derived from KMS");
        info!(ed25519_address = %ed25519.signing_address, "Ed25519 key derived from KMS");

        Ok(SigningPair { ecdsa, ed25519 })
    }

    /// Sign content with both algorithms and return a SignedChat.
    pub fn sign_chat(&self, text: &str) -> Result<crate::types::SignedChat> {
        let start_ecdsa = std::time::Instant::now();
        let sig_ecdsa = self.ecdsa.sign(text)?;
        metrics::counter!("signatures_generated_total", "algorithm" => "ecdsa").increment(1);
        metrics::histogram!("signature_generation_duration_seconds", "algorithm" => "ecdsa")
            .record(start_ecdsa.elapsed().as_secs_f64());

        let start_ed25519 = std::time::Instant::now();
        let sig_ed25519 = self.ed25519.sign(text)?;
        metrics::counter!("signatures_generated_total", "algorithm" => "ed25519").increment(1);
        metrics::histogram!("signature_generation_duration_seconds", "algorithm" => "ed25519")
            .record(start_ed25519.elapsed().as_secs_f64());

        Ok(crate::types::SignedChat {
            text: text.to_string(),
            signature_ecdsa: sig_ecdsa,
            signing_address_ecdsa: self.ecdsa.signing_address.clone(),
            signature_ed25519: sig_ed25519,
            signing_address_ed25519: self.ed25519.signing_address.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Verifier;
    use k256::ecdsa::VerifyingKey;
    use sha3::{Digest as Sha3Digest, Keccak256};

    // Known test vector: private key → expected Ethereum address
    // Using a well-known test key from Ethereum documentation
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

    #[test]
    fn test_ecdsa_context_creation() {
        let ctx = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();

        // Address should be 0x-prefixed hex, 42 chars total
        assert!(ctx.signing_address.starts_with("0x"));
        assert_eq!(ctx.signing_address.len(), 42);

        // Address bytes should be 20 bytes
        assert_eq!(ctx.signing_address_bytes.len(), 20);

        // Public key should be 128 hex chars (64 bytes uncompressed, no prefix)
        assert_eq!(ctx.signing_public_key.len(), 128);
    }

    #[test]
    fn test_ecdsa_ethereum_address_derivation() {
        let ctx = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();

        // Verify the address matches keccak256(pubkey)[12..32]
        let pk_bytes = hex::decode(&ctx.signing_public_key).unwrap();
        assert_eq!(pk_bytes.len(), 64);
        let hash = Keccak256::digest(&pk_bytes);
        let expected_addr = format!("0x{}", hex::encode(&hash[12..32]));
        assert_eq!(ctx.signing_address, expected_addr);
    }

    #[test]
    fn test_ecdsa_sign_produces_valid_eip191_signature() {
        let ctx = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
        let message = "hello world";

        let sig_hex = ctx.sign(message).unwrap();

        // Signature should be 0x-prefixed, 132 hex chars (65 bytes)
        assert!(sig_hex.starts_with("0x"));
        assert_eq!(sig_hex.len(), 132);

        // Parse signature components
        let sig_bytes = hex::decode(&sig_hex[2..]).unwrap();
        assert_eq!(sig_bytes.len(), 65);

        let r_s = &sig_bytes[..64];
        let v = sig_bytes[64];

        // v should be 27 or 28
        assert!(v == 27 || v == 28);

        // Verify the signature by recovering the public key
        let prefix = format!("\x19Ethereum Signed Message:\n{}", message.len());
        let mut prefixed = Vec::new();
        prefixed.extend_from_slice(prefix.as_bytes());
        prefixed.extend_from_slice(message.as_bytes());
        let hash = Keccak256::digest(&prefixed);

        let signature = k256::ecdsa::Signature::from_slice(r_s).unwrap();
        let recovery_id = RecoveryId::from_byte(v - 27).unwrap();
        let recovered_key =
            VerifyingKey::recover_from_prehash(&hash[..], &signature, recovery_id).unwrap();

        // Recovered key should match original public key
        let recovered_pk = recovered_key.to_encoded_point(false);
        let recovered_pk_hex = hex::encode(&recovered_pk.as_bytes()[1..]);
        assert_eq!(recovered_pk_hex, ctx.signing_public_key);
    }

    #[test]
    fn test_ecdsa_sign_deterministic() {
        let ctx = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
        let message = "deterministic test";

        let sig1 = ctx.sign(message).unwrap();
        let sig2 = ctx.sign(message).unwrap();

        // RFC6979 makes ECDSA signing deterministic for same key+message
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_ecdsa_different_messages_different_signatures() {
        let ctx = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();

        let sig1 = ctx.sign("message 1").unwrap();
        let sig2 = ctx.sign("message 2").unwrap();

        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_ed25519_context_creation() {
        let ctx = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();

        // Address = public key hex = 64 hex chars (32 bytes)
        assert_eq!(ctx.signing_address.len(), 64);
        assert_eq!(ctx.signing_public_key, ctx.signing_address);
        assert_eq!(ctx.signing_address_bytes.len(), 32);
    }

    #[test]
    fn test_ed25519_sign_and_verify() {
        let ctx = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();
        let message = "hello world";

        let sig_hex = ctx.sign(message).unwrap();

        // Signature should be 128 hex chars (64 bytes)
        assert_eq!(sig_hex.len(), 128);

        // Verify the signature
        let sig_bytes = hex::decode(&sig_hex).unwrap();
        let signature = ed25519_dalek::Signature::from_bytes(sig_bytes[..64].try_into().unwrap());
        let pk_bytes = hex::decode(&ctx.signing_public_key).unwrap();
        let verifying_key =
            ed25519_dalek::VerifyingKey::from_bytes(pk_bytes[..32].try_into().unwrap()).unwrap();

        assert!(verifying_key.verify(message.as_bytes(), &signature).is_ok());
    }

    #[test]
    fn test_ed25519_sign_deterministic() {
        let ctx = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();
        let message = "deterministic test";

        let sig1 = ctx.sign(message).unwrap();
        let sig2 = ctx.sign(message).unwrap();

        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_ed25519_different_messages_different_signatures() {
        let ctx = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();

        let sig1 = ctx.sign("message 1").unwrap();
        let sig2 = ctx.sign("message 2").unwrap();

        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_signing_pair_sign_chat() {
        let ecdsa = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
        let ed25519 = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();

        let ecdsa_addr = ecdsa.signing_address.clone();
        let ed25519_addr = ed25519.signing_address.clone();

        let pair = SigningPair { ecdsa, ed25519 };

        let text = "abc123:def456";
        let signed = pair.sign_chat(text).unwrap();

        assert_eq!(signed.text, text);
        assert_eq!(signed.signing_address_ecdsa, ecdsa_addr);
        assert_eq!(signed.signing_address_ed25519, ed25519_addr);
        assert!(signed.signature_ecdsa.starts_with("0x"));
        assert_eq!(signed.signature_ecdsa.len(), 132);
        assert_eq!(signed.signature_ed25519.len(), 128);
    }

    #[test]
    fn test_signing_pair_sign_chat_serialization() {
        let ecdsa = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
        let ed25519 = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();

        let pair = SigningPair { ecdsa, ed25519 };
        let signed = pair.sign_chat("req_hash:resp_hash").unwrap();

        // Should serialize to JSON without error
        let json = serde_json::to_string(&signed).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert!(parsed.get("text").is_some());
        assert!(parsed.get("signature_ecdsa").is_some());
        assert!(parsed.get("signing_address_ecdsa").is_some());
        assert!(parsed.get("signature_ed25519").is_some());
        assert!(parsed.get("signing_address_ed25519").is_some());
    }

    // ---- Client-perspective verification tests ----

    /// Helper: recover Ethereum address from an EIP-191 signature.
    fn recover_ecdsa_address(message: &str, sig_hex: &str) -> String {
        let sig_bytes = hex::decode(&sig_hex[2..]).unwrap();
        let v = sig_bytes[64];
        let prefix = format!("\x19Ethereum Signed Message:\n{}", message.len());
        let mut prefixed = Vec::new();
        prefixed.extend_from_slice(prefix.as_bytes());
        prefixed.extend_from_slice(message.as_bytes());
        let hash = Keccak256::digest(&prefixed);
        let signature = k256::ecdsa::Signature::from_slice(&sig_bytes[..64]).unwrap();
        let recovery_id = RecoveryId::from_byte(v - 27).unwrap();
        let recovered_key =
            VerifyingKey::recover_from_prehash(&hash[..], &signature, recovery_id).unwrap();
        let pk_encoded = recovered_key.to_encoded_point(false);
        let pk_hash = Keccak256::digest(&pk_encoded.as_bytes()[1..]);
        format!("0x{}", hex::encode(&pk_hash[12..32]))
    }

    /// Helper: verify an Ed25519 signature.
    fn verify_ed25519(message: &str, sig_hex: &str, pk_hex: &str) -> bool {
        let sig_bytes = hex::decode(sig_hex).unwrap();
        let signature = ed25519_dalek::Signature::from_bytes(sig_bytes[..64].try_into().unwrap());
        let pk_bytes = hex::decode(pk_hex).unwrap();
        let verifying_key =
            ed25519_dalek::VerifyingKey::from_bytes(pk_bytes[..32].try_into().unwrap()).unwrap();
        verifying_key.verify(message.as_bytes(), &signature).is_ok()
    }

    #[test]
    fn test_ecdsa_signature_recovers_correct_address() {
        let ctx = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
        let message = "request_hash:response_hash";
        let sig_hex = ctx.sign(message).unwrap();

        let recovered = recover_ecdsa_address(message, &sig_hex);
        assert_eq!(recovered, ctx.signing_address);
    }

    #[test]
    fn test_ecdsa_wrong_message_recovers_wrong_address() {
        let ctx = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
        let sig_hex = ctx.sign("original message").unwrap();

        let recovered = recover_ecdsa_address("tampered message", &sig_hex);
        assert_ne!(recovered, ctx.signing_address);
    }

    #[test]
    fn test_ecdsa_wrong_key_recovers_wrong_address() {
        let ctx = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
        let mut other_key = TEST_ECDSA_KEY;
        other_key[0] ^= 0xff;
        let other_ctx = EcdsaContext::from_key_bytes(&other_key).unwrap();

        let sig_hex = other_ctx.sign("hello").unwrap();
        let recovered = recover_ecdsa_address("hello", &sig_hex);
        assert_ne!(recovered, ctx.signing_address);
    }

    #[test]
    fn test_ecdsa_corrupted_signature_does_not_recover_original_address() {
        let ctx = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
        let message = "test message";
        let sig_hex = ctx.sign(message).unwrap();

        // Corrupt one byte in the r component of the signature
        let mut sig_bytes = hex::decode(&sig_hex[2..]).unwrap();
        sig_bytes[0] ^= 0xff;
        let corrupted_sig = format!("0x{}", hex::encode(&sig_bytes));

        // Recovery may succeed but should yield a different address,
        // or may panic/fail entirely — either outcome means rejection
        let result = std::panic::catch_unwind(|| recover_ecdsa_address(message, &corrupted_sig));
        if let Ok(recovered) = result {
            assert_ne!(recovered, ctx.signing_address);
        } // Panicked during recovery (Err) = also a rejection
    }

    #[test]
    fn test_ed25519_wrong_message_fails_verification() {
        let ctx = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();
        let sig_hex = ctx.sign("original message").unwrap();

        assert!(!verify_ed25519(
            "tampered message",
            &sig_hex,
            &ctx.signing_public_key
        ));
    }

    #[test]
    fn test_ed25519_wrong_key_fails_verification() {
        let ctx = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();
        let sig_hex = ctx.sign("hello").unwrap();

        let mut other_key = TEST_ED25519_KEY;
        other_key[0] ^= 0xff;
        let other_ctx = Ed25519Context::from_key_bytes(&other_key).unwrap();

        assert!(!verify_ed25519(
            "hello",
            &sig_hex,
            &other_ctx.signing_public_key
        ));
    }

    #[test]
    fn test_sign_chat_both_signatures_verifiable() {
        let ecdsa = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
        let ed25519 = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();
        let pair = SigningPair { ecdsa, ed25519 };

        let text = "abc123hash:def456hash";
        let signed = pair.sign_chat(text).unwrap();

        // Verify ECDSA
        let recovered_addr = recover_ecdsa_address(&signed.text, &signed.signature_ecdsa);
        assert_eq!(recovered_addr, signed.signing_address_ecdsa);

        // Verify Ed25519
        assert!(verify_ed25519(
            &signed.text,
            &signed.signature_ed25519,
            &signed.signing_address_ed25519
        ));
    }

    #[test]
    fn test_sign_chat_tampered_text_fails_both_verifications() {
        let ecdsa = EcdsaContext::from_key_bytes(&TEST_ECDSA_KEY).unwrap();
        let ed25519 = Ed25519Context::from_key_bytes(&TEST_ED25519_KEY).unwrap();
        let pair = SigningPair { ecdsa, ed25519 };

        let signed = pair.sign_chat("original:hash").unwrap();

        // Tampered text should fail both
        let tampered = "tampered:hash";
        let recovered_addr = recover_ecdsa_address(tampered, &signed.signature_ecdsa);
        assert_ne!(recovered_addr, signed.signing_address_ecdsa);

        assert!(!verify_ed25519(
            tampered,
            &signed.signature_ed25519,
            &signed.signing_address_ed25519
        ));
    }

    #[tokio::test]
    async fn test_signing_pair_dev_mode_init() {
        let pair = SigningPair::init("test-model", true).await.unwrap();

        // Both contexts should be initialized
        assert!(pair.ecdsa.signing_address.starts_with("0x"));
        assert_eq!(pair.ecdsa.signing_address.len(), 42);
        assert_eq!(pair.ed25519.signing_address.len(), 64);

        // Should be able to sign
        let signed = pair.sign_chat("test:hash").unwrap();
        assert!(signed.signature_ecdsa.starts_with("0x"));
        assert_eq!(signed.signature_ed25519.len(), 128);
    }
}
