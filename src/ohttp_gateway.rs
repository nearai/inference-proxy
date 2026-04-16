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

    /// Decapsulate a standard OHTTP request (RFC 9458).
    ///
    /// Returns the plaintext Binary HTTP request bytes and a `ServerResponse`
    /// that must be used to encapsulate the response.
    pub fn decapsulate(
        &self,
        enc_request: &[u8],
    ) -> Result<(Vec<u8>, ServerResponse), ohttp::Error> {
        self.server.decapsulate(enc_request)
    }

    /// Clone the inner `Server` for use with streaming APIs.
    ///
    /// Needed because `Server::decapsulate_stream` consumes the server.
    /// Use: `gateway.clone_server().decapsulate_stream(src)`.
    pub fn clone_server(&self) -> Server {
        self.server.clone()
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

    #[tokio::test]
    async fn test_stream_roundtrip() {
        use futures_util::{AsyncReadExt, AsyncWriteExt};
        use tokio_util::compat::TokioAsyncWriteCompatExt;

        let gw = OhttpGateway::new(&TEST_KEY).unwrap();

        // Client side: encrypt a request using the stream API
        let mut config = KeyConfig::decode(gw.config_bytes()).unwrap();
        let client = ohttp::ClientRequest::from_config(&mut config).unwrap();

        // Write the request into a pipe, encrypted
        let (mut request_pipe_read, request_pipe_write) = tokio::io::duplex(8192);
        let mut client_request = client
            .encapsulate_stream(request_pipe_write.compat_write())
            .unwrap();
        let inner_request = b"hello from the client";
        client_request.write_all(inner_request).await.unwrap();
        client_request.close().await.unwrap();

        // Read the encrypted bytes from the pipe
        let mut enc_request = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut request_pipe_read, &mut enc_request)
            .await
            .unwrap();

        // Server side: decapsulate stream
        let server = gw.clone_server();
        let mut server_request = server.decapsulate_stream(&enc_request[..]);
        let mut decrypted_request = Vec::new();
        server_request
            .read_to_end(&mut decrypted_request)
            .await
            .unwrap();
        assert_eq!(decrypted_request, inner_request);

        // Server side: write response through the stream API
        let (mut response_pipe_read, response_pipe_write) = tokio::io::duplex(8192);
        let mut server_response = server_request
            .response(response_pipe_write.compat_write())
            .unwrap();
        let inner_response = b"hello from the server";
        server_response.write_all(inner_response).await.unwrap();
        server_response.close().await.unwrap();

        // Read the encrypted response from the tokio duplex pipe
        let mut enc_response = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut response_pipe_read, &mut enc_response)
            .await
            .unwrap();

        // Client side: decrypt the response stream
        let mut client_response = client_request.response(&enc_response[..]).unwrap();
        let mut decrypted_response = Vec::new();
        client_response
            .read_to_end(&mut decrypted_response)
            .await
            .unwrap();
        assert_eq!(decrypted_response, inner_response);
    }

    /// Test that a large payload (>16KB) works through the streaming API.
    /// The ohttp crate encrypts in 16KB chunks — this verifies multi-chunk roundtrip.
    #[tokio::test]
    async fn test_stream_roundtrip_large_payload() {
        use futures_util::{AsyncReadExt, AsyncWriteExt};
        use tokio_util::compat::TokioAsyncWriteCompatExt;

        let gw = OhttpGateway::new(&TEST_KEY).unwrap();
        let mut config = KeyConfig::decode(gw.config_bytes()).unwrap();
        let client = ohttp::ClientRequest::from_config(&mut config).unwrap();

        // Small request, large response (32KB — spans 2+ AEAD chunks)
        let inner_request = b"short request";
        let (mut req_read, req_write) = tokio::io::duplex(8192);
        let mut client_request = client.encapsulate_stream(req_write.compat_write()).unwrap();
        client_request.write_all(inner_request).await.unwrap();
        client_request.close().await.unwrap();
        let mut enc_request = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut req_read, &mut enc_request)
            .await
            .unwrap();

        let server = gw.clone_server();
        let mut server_request = server.decapsulate_stream(&enc_request[..]);
        let mut dec_req = Vec::new();
        server_request.read_to_end(&mut dec_req).await.unwrap();
        assert_eq!(dec_req, inner_request);

        // Write a 32KB response
        let large_response = vec![0xABu8; 32 * 1024];
        let (mut resp_read, resp_write) = tokio::io::duplex(64 * 1024);
        let mut server_response = server_request.response(resp_write.compat_write()).unwrap();
        server_response.write_all(&large_response).await.unwrap();
        server_response.close().await.unwrap();

        let mut enc_response = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut resp_read, &mut enc_response)
            .await
            .unwrap();

        // Encrypted must be larger (AEAD overhead per chunk)
        assert!(enc_response.len() > large_response.len());

        let mut client_response = client_request.response(&enc_response[..]).unwrap();
        let mut dec_resp = Vec::new();
        client_response.read_to_end(&mut dec_resp).await.unwrap();
        assert_eq!(dec_resp, large_response);
    }

    /// Test that an empty response body works through the streaming API.
    #[tokio::test]
    async fn test_stream_roundtrip_empty_response() {
        use futures_util::{AsyncReadExt, AsyncWriteExt};
        use tokio_util::compat::TokioAsyncWriteCompatExt;

        let gw = OhttpGateway::new(&TEST_KEY).unwrap();
        let mut config = KeyConfig::decode(gw.config_bytes()).unwrap();
        let client = ohttp::ClientRequest::from_config(&mut config).unwrap();

        let (mut req_read, req_write) = tokio::io::duplex(8192);
        let mut client_request = client.encapsulate_stream(req_write.compat_write()).unwrap();
        client_request.write_all(b"request").await.unwrap();
        client_request.close().await.unwrap();
        let mut enc_request = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut req_read, &mut enc_request)
            .await
            .unwrap();

        let server = gw.clone_server();
        let mut server_request = server.decapsulate_stream(&enc_request[..]);
        let mut dec_req = Vec::new();
        server_request.read_to_end(&mut dec_req).await.unwrap();

        // Empty response
        let (mut resp_read, resp_write) = tokio::io::duplex(8192);
        let mut server_response = server_request.response(resp_write.compat_write()).unwrap();
        // Write nothing, just close
        server_response.close().await.unwrap();

        let mut enc_response = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut resp_read, &mut enc_response)
            .await
            .unwrap();

        // Should still produce some encrypted bytes (nonce + final chunk tag)
        assert!(!enc_response.is_empty());

        let mut client_response = client_request.response(&enc_response[..]).unwrap();
        let mut dec_resp = Vec::new();
        client_response.read_to_end(&mut dec_resp).await.unwrap();
        assert!(dec_resp.is_empty());
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
