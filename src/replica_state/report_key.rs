//! Per-boot ed25519 key that signs replica reports, bound to the CVM's
//! attestation by recording its public half in the dstack event log.

use ed25519_dalek::SigningKey;
use sha2::Digest;

/// dstack event name under which the report public key is recorded.
pub const REPORT_KEY_EVENT: &str = "nearai-replica-report-key-v1";

pub struct ReportKey {
    signing: SigningKey,
    pub key_id: String,
    pub boot_id: String,
}

impl ReportKey {
    /// Fresh random key for this boot. `key_id` is the first 16 hex chars of
    /// sha256(public key); `boot_id` is a random UUID v4.
    pub fn generate() -> Self {
        let seed: [u8; 32] = rand::random();
        let signing = SigningKey::from_bytes(&seed);
        let key_id =
            hex::encode(sha2::Sha256::digest(signing.verifying_key().to_bytes()))[..16].to_string();
        Self {
            signing,
            key_id,
            boot_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    pub fn signing_key(&self) -> &SigningKey {
        &self.signing
    }

    pub fn public_hex(&self) -> String {
        hex::encode(self.signing.verifying_key().to_bytes())
    }

    /// Event-log payload:
    /// `{"key_id","public_key_hex","boot_id","host_id","model","replica_ids"}`.
    /// Binds the key to the host and replicas it may report for. Public data only.
    pub fn event_payload(&self, host_id: &str, model: &str, replica_ids: &[String]) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "key_id": self.key_id,
            "public_key_hex": self.public_hex(),
            "boot_id": self.boot_id,
            "host_id": host_id,
            "model": model,
            "replica_ids": replica_ids,
        }))
        .expect("static JSON object always serializes")
    }
}

impl std::fmt::Debug for ReportKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReportKey")
            .field("key_id", &self.key_id)
            .field("boot_id", &self.boot_id)
            .finish_non_exhaustive()
    }
}

/// Ok(true) if recorded in the dstack event log; Ok(false) if skipped (dev mode or non-TEE).
pub async fn bind_to_attestation(
    key: &ReportKey,
    host_id: &str,
    model: &str,
    replica_ids: &[String],
    skip: bool,
) -> anyhow::Result<bool> {
    if skip {
        tracing::info!(key_id = %key.key_id, "Replica report key not bound: not in a TEE");
        return Ok(false);
    }
    dstack_sdk::dstack_client::DstackClient::new(None)
        .emit_event(
            REPORT_KEY_EVENT.to_string(),
            key.event_payload(host_id, model, replica_ids),
        )
        .await?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_and_ids_differ_per_boot_and_derive_correctly() {
        let a = ReportKey::generate();
        let b = ReportKey::generate();
        assert_ne!(a.public_hex(), b.public_hex());
        assert_ne!(a.boot_id, b.boot_id);
        use sha2::Digest;
        assert_eq!(
            a.key_id,
            hex::encode(sha2::Sha256::digest(
                a.signing_key().verifying_key().to_bytes()
            ))[..16]
                .to_string()
        );
    }

    #[test]
    fn event_payload_binds_host_and_exposes_no_secret() {
        use sha2::Digest;
        let k = ReportKey::generate();
        let ids = vec!["r1".to_string(), "r2".to_string()];
        let v: serde_json::Value =
            serde_json::from_slice(&k.event_payload("host-a", "m", &ids)).unwrap();
        assert_eq!(v.as_object().unwrap().len(), 6);
        assert_eq!(v["key_id"], k.key_id);
        assert_eq!(v["public_key_hex"], k.public_hex());
        assert_eq!(v["boot_id"], k.boot_id);
        assert_eq!(v["host_id"], "host-a");
        assert_eq!(v["model"], "m");
        assert_eq!(v["replica_ids"], serde_json::json!(["r1", "r2"]));
        let pk = hex::decode(v["public_key_hex"].as_str().unwrap()).unwrap();
        assert_eq!(
            v["key_id"].as_str().unwrap(),
            &hex::encode(sha2::Sha256::digest(&pk))[..16]
        );
        assert!(!v
            .to_string()
            .contains(&hex::encode(k.signing_key().to_bytes())));
        assert!(!format!("{k:?}").contains(&hex::encode(k.signing_key().to_bytes())));
    }

    #[tokio::test]
    async fn binding_is_skipped_when_asked() {
        let ids = vec!["r1".to_string()];
        assert!(
            !bind_to_attestation(&ReportKey::generate(), "h", "m", &ids, true)
                .await
                .unwrap()
        );
    }
}
