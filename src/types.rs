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
}

/// Response for GET /v1/signature/{chat_id}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureResponse {
    pub text: String,
    pub signature: String,
    pub signing_address: String,
    pub signing_algo: String,
}
