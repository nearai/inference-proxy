use ohttp::hpke::{Aead, Kdf, Kem};
use ohttp::{KeyConfig, Server, ServerResponse, SymmetricSuite};

/// Arbitrary key ID for this deployment's OHTTP key configuration.
const OHTTP_KEY_ID: u8 = 1;

/// OHTTP Gateway (RFC 9458) — handles HPKE decapsulation/encapsulation.
///
/// The HPKE keypair is deterministically derived from the Ed25519 signing key
/// seed (loaded from dstack KMS). All instances of the same model share the
/// same KMS key, so they produce identical OHTTP key configurations.
pub struct OhttpGateway {
    server: Server,
    /// Pre-encoded key configuration bytes for the well-known endpoint.
    config_bytes: Vec<u8>,
}

impl OhttpGateway {
    /// Create from Ed25519 secret key material (32 bytes from dstack KMS).
    ///
    /// Uses HPKE `DeriveKeyPair` internally — the resulting X25519 keypair is
    /// domain-separated from the E2EE X25519 key (which uses a different
    /// derivation path via `SHA512(seed)[..32]` with RFC 7748 clamping).
    pub fn new(ed25519_secret: &[u8; 32]) -> anyhow::Result<Self> {
        let config = KeyConfig::derive(
            OHTTP_KEY_ID,
            Kem::X25519Sha256,
            vec![
                SymmetricSuite::new(Kdf::HkdfSha256, Aead::Aes128Gcm),
                SymmetricSuite::new(Kdf::HkdfSha256, Aead::ChaCha20Poly1305),
            ],
            ed25519_secret,
        )?;
        let config_bytes = config.encode()?;
        let server = Server::new(config)?;
        Ok(Self {
            server,
            config_bytes,
        })
    }

    /// Encoded key configuration bytes (served at `/.well-known/ohttp-gateway`).
    pub fn config_bytes(&self) -> &[u8] {
        &self.config_bytes
    }

    /// Decapsulate a standard OHTTP request.
    ///
    /// Returns the plaintext Binary HTTP request bytes and a `ServerResponse`
    /// that must be used to encapsulate the response.
    pub fn decapsulate(
        &self,
        enc_request: &[u8],
    ) -> Result<(Vec<u8>, ServerResponse), ohttp::Error> {
        self.server.decapsulate(enc_request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed test key (same as ed25519_key in integration tests).
    const TEST_KEY: [u8; 32] = [
        0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c,
        0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae,
        0x7f, 0x60,
    ];

    #[test]
    fn test_init_from_known_key() {
        let gw = OhttpGateway::new(&TEST_KEY).unwrap();
        assert!(!gw.config_bytes().is_empty());
    }

    #[test]
    fn test_deterministic_config() {
        let gw1 = OhttpGateway::new(&TEST_KEY).unwrap();
        let gw2 = OhttpGateway::new(&TEST_KEY).unwrap();
        assert_eq!(gw1.config_bytes(), gw2.config_bytes());
    }

    #[test]
    fn test_roundtrip() {
        let gw = OhttpGateway::new(&TEST_KEY).unwrap();

        // Client side: encrypt a request
        let mut config = KeyConfig::decode(gw.config_bytes()).unwrap();
        let client_request = ohttp::ClientRequest::from_config(&mut config).unwrap();

        let inner_request = b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let (enc_request, client_response) = client_request.encapsulate(inner_request).unwrap();

        // Server side: decapsulate
        let (plaintext, server_response) = gw.decapsulate(&enc_request).unwrap();
        assert_eq!(plaintext, inner_request);

        // Server side: encapsulate response
        let inner_response = b"HTTP/1.1 200 OK\r\n\r\n{\"models\":[]}";
        let enc_response = server_response.encapsulate(inner_response).unwrap();

        // Client side: decrypt response
        let decrypted = client_response.decapsulate(&enc_response).unwrap();
        assert_eq!(decrypted, inner_response);
    }

    #[test]
    fn test_different_keys_produce_different_configs() {
        let key_a = [0x01u8; 32];
        let key_b = [0x02u8; 32];
        let gw_a = OhttpGateway::new(&key_a).unwrap();
        let gw_b = OhttpGateway::new(&key_b).unwrap();
        assert_ne!(gw_a.config_bytes(), gw_b.config_bytes());
    }

    #[test]
    fn test_ohttp_key_differs_from_e2ee_x25519_key() {
        // The OHTTP key is derived via HPKE DeriveKeyPair, which uses a
        // different derivation path than the E2EE X25519 key (SHA512 + clamp).
        // Verify they're not the same.
        let gw = OhttpGateway::new(&TEST_KEY).unwrap();

        // Reproduce the E2EE X25519 derivation: SHA512(seed)[..32] with clamping.
        use sha2::Digest;
        let hash = sha2::Sha512::digest(TEST_KEY);
        let mut e2ee_scalar = [0u8; 32];
        e2ee_scalar.copy_from_slice(&hash[..32]);
        e2ee_scalar[0] &= 248;
        e2ee_scalar[31] &= 127;
        e2ee_scalar[31] |= 64;
        let e2ee_x25519_public =
            x25519_dalek::PublicKey::from(&x25519_dalek::StaticSecret::from(e2ee_scalar));

        // The OHTTP config bytes should NOT contain the E2EE public key bytes
        // as a substring (different derivation → different key).
        let e2ee_pub_bytes = e2ee_x25519_public.as_bytes();
        let config = gw.config_bytes();
        let found = config
            .windows(32)
            .any(|window| window == e2ee_pub_bytes.as_slice());
        assert!(
            !found,
            "OHTTP key config contains the E2EE X25519 public key — domain separation broken"
        );
    }
}
