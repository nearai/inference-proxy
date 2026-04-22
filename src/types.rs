use serde::{Deserialize, Serialize};

/// Cached signature data for a chat completion, stored in the cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedChat {
    pub text: String,
    pub signature_ecdsa: String,
    pub signing_address_ecdsa: String,
    pub signature_ed25519: String,
    pub signing_address_ed25519: String,
}

/// Attestation report returned by GET /v1/attestation/report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationReport {
    pub model_name: String,
    pub signing_address: String,
    pub signing_algo: String,
    pub signing_public_key: String,
    pub request_nonce: String,
    pub intel_quote: String,
    pub nvidia_payload: String,
    pub event_log: serde_json::Value,
    pub info: serde_json::Value,
    /// SHA-256 hash of the TLS certificate's SPKI, if configured.
    /// When present, report_data[..32] = SHA256(signing_address_bytes || cert_fingerprint_bytes).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tls_cert_fingerprint: Option<String>,
}

/// Response for GET /v1/attestation/report (wraps with all_attestations).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationResponse {
    #[serde(flatten)]
    pub report: AttestationReport,
    pub all_attestations: Vec<AttestationReport>,
    /// Compose-manager deployment attestation, if available.
    /// Contains the compose-manager's TDX-attested action log (deployment events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compose_manager_attestation: Option<serde_json::Value>,
    /// Legacy flat OHTTP key config field kept for backward compatibility.
    /// Mirrors `ohttp_attestation.key_config` when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ohttp_key_config: Option<String>,
    /// OHTTP key attestation payload, if OHTTP is enabled.
    /// Includes key config bytes and an Ed25519 signature over decoded `key_config` bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ohttp_attestation: Option<OhttpAttestation>,
}

impl AttestationResponse {
    pub fn new(
        report: AttestationReport,
        all_attestations: Vec<AttestationReport>,
        compose_manager_attestation: Option<serde_json::Value>,
        ohttp_attestation: Option<OhttpAttestation>,
    ) -> Self {
        let ohttp_key_config = ohttp_attestation.as_ref().map(|att| att.key_config.clone());
        Self {
            report,
            all_attestations,
            compose_manager_attestation,
            ohttp_key_config,
            ohttp_attestation,
        }
    }
}

/// Attestation payload for OHTTP key configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhttpAttestation {
    /// Signing algorithm used for `signature`.
    pub signing_algo: String,
    /// Hex-encoded public key corresponding to the signing key.
    pub signing_key: String,
    /// Hex-encoded OHTTP key configuration bytes (RFC 9458).
    pub key_config: String,
    /// Signature over the decoded `key_config` bytes.
    /// Clients verify this signature against the attested Ed25519 public key.
    pub signature: String,
}

/// Response for GET /v1/signature/{chat_id}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureResponse {
    pub text: String,
    pub signature: String,
    pub signing_address: String,
    pub signing_algo: String,
}
