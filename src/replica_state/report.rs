//! ReplicaReport v1 and the signed envelope that carries it over Redis.
//!
//! The envelope carries the report as the exact signed JSON string (`frame`)
//! so that readers in any language can verify the received bytes before
//! parsing them. [`open`] never re-serializes `frame`; it verifies the bytes
//! as received, then parses them. The envelope's `key_id` is an unsigned
//! lookup hint for picking the verifying key; the signed `report_key_id`
//! inside `frame` must equal it.

use base64::Engine as _;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

/// Domain-separation prefix prepended to `frame` bytes before signing.
pub const SIGNING_DOMAIN: &[u8] = b"nearai-replica-report-v1\n";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Lifecycle {
    Warming,
    Ready,
    Degraded,
    Unhealthy,
    Draining,
    Drained,
} // v1 writer emits Warming/Ready/Unhealthy

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Sglang,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Limits {
    pub max_running: Option<u32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Load {
    pub running: Option<u32>,
    pub queued: Option<u32>,
    pub prefill_backlog_tokens: Option<u64>,
    pub kv_usage: Option<f64>,
    pub gen_tps: Option<f64>,
    pub cached_token_ratio: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReplicaReport {
    pub schema: u8,
    pub host_id: String,
    pub replica_id: String,
    pub boot_id: String,
    pub seq: u64,
    /// Engine's own sample time; `None` until the replica has been read once.
    pub engine_sampled_at_ms: Option<u64>,
    /// Wall-clock time the frame was sealed, after this tick's reads.
    pub reported_at_ms: u64,
    pub lifecycle_state: Lifecycle,
    pub model: String,
    pub engine: Engine,
    pub engine_version: Option<String>,
    pub limits: Limits,
    pub load: Load,
    pub proxy_inflight: u32,
    pub report_key_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Envelope {
    pub frame: String,
    pub sig: String,
    /// Unsigned hint naming the key that signed `frame`; not trusted on its own.
    pub key_id: String,
}

fn message(frame: &str) -> Vec<u8> {
    let mut m = SIGNING_DOMAIN.to_vec();
    m.extend_from_slice(frame.as_bytes());
    m
}

pub fn seal(report: &ReplicaReport, key: &SigningKey) -> Envelope {
    let frame = serde_json::to_string(report).expect("ReplicaReport always serializes");
    let sig =
        base64::engine::general_purpose::STANDARD.encode(key.sign(&message(&frame)).to_bytes());
    Envelope {
        frame,
        sig,
        key_id: report.report_key_id.clone(),
    }
}

/// Verify, then parse. None on a bad signature, bad JSON, or when the signed
/// `report_key_id` disagrees with the envelope's `key_id` hint.
pub fn open(env: &Envelope, key: &VerifyingKey) -> Option<ReplicaReport> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(&env.sig)
        .ok()?;
    let sig = Signature::from_bytes(&<[u8; 64]>::try_from(raw.as_slice()).ok()?);
    key.verify(&message(&env.frame), &sig).ok()?;
    let report: ReplicaReport = serde_json::from_str(&env.frame).ok()?;
    (report.report_key_id == env.key_id).then_some(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn report() -> ReplicaReport {
        ReplicaReport {
            schema: 1,
            host_id: "glm53-gpu03".into(),
            replica_id: "r1".into(),
            boot_id: "00000000-0000-4000-8000-000000000001".into(),
            seq: 7,
            engine_sampled_at_ms: Some(1_790_000_000_011),
            reported_at_ms: 1_790_000_000_123,
            lifecycle_state: Lifecycle::Ready,
            model: "z-ai/glm-5.3-flash".into(),
            engine: Engine::Sglang,
            engine_version: None,
            limits: Limits {
                max_running: Some(32),
            },
            load: Load {
                running: Some(14),
                queued: Some(0),
                prefill_backlog_tokens: Some(51200),
                ..Default::default()
            },
            proxy_inflight: 16,
            report_key_id: "0123456789abcdef".into(),
        }
    }

    #[test]
    fn seal_open_roundtrip_through_json() {
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let wire = serde_json::to_string(&seal(&report(), &sk)).unwrap();
        let env: Envelope = serde_json::from_str(&wire).unwrap();
        assert_eq!(open(&env, &sk.verifying_key()), Some(report()));
    }

    #[test]
    fn frame_keeps_nulls_and_is_compact() {
        let env = seal(&report(), &SigningKey::from_bytes(&[7u8; 32]));
        assert!(env
            .frame
            .starts_with(r#"{"schema":1,"host_id":"glm53-gpu03""#));
        assert!(env.frame.contains(r#""kv_usage":null"#));
        assert!(!env.frame.contains(": "));
    }

    #[test]
    fn tampered_frame_or_wrong_key_fails() {
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let mut env = seal(&report(), &sk);
        env.frame = env.frame.replace(r#""running":14"#, r#""running":0"#);
        assert!(open(&env, &sk.verifying_key()).is_none());
        let env = seal(&report(), &sk);
        assert!(open(&env, &SigningKey::from_bytes(&[8u8; 32]).verifying_key()).is_none());
    }

    #[test]
    fn envelope_carries_unsigned_key_hint_that_must_match_signed_key_id() {
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let env = seal(&report(), &sk);
        assert_eq!(env.key_id, "0123456789abcdef");
        let wire: serde_json::Value = serde_json::to_value(&env).unwrap();
        assert_eq!(wire.as_object().unwrap().len(), 3);

        let swapped = Envelope {
            key_id: "fedcba9876543210".into(),
            ..env
        };
        assert!(open(&swapped, &sk.verifying_key()).is_none());
    }

    #[test]
    fn unknown_sample_time_serializes_as_null() {
        let r = ReplicaReport {
            engine_sampled_at_ms: None,
            ..report()
        };
        let env = seal(&r, &SigningKey::from_bytes(&[7u8; 32]));
        assert!(env.frame.contains(r#""engine_sampled_at_ms":null"#));
    }

    #[test]
    fn signature_without_domain_prefix_is_rejected() {
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let frame = serde_json::to_string(&report()).unwrap();
        let sig =
            base64::engine::general_purpose::STANDARD.encode(sk.sign(frame.as_bytes()).to_bytes());
        let key_id = report().report_key_id;
        assert!(open(&Envelope { frame, sig, key_id }, &sk.verifying_key()).is_none());
    }

    #[test]
    fn malformed_signatures_are_rejected_without_panicking() {
        let key = SigningKey::from_bytes(&[7u8; 32]);
        let mut env = seal(&report(), &key);
        env.sig = "not-base64!!".to_string();
        assert!(open(&env, &key.verifying_key()).is_none());
        env.sig = base64::engine::general_purpose::STANDARD.encode([1u8; 32]);
        assert!(open(&env, &key.verifying_key()).is_none());
    }
}
