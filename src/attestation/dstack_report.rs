use anyhow::{bail, ensure, Context, Result};
use dstack_sdk::dstack_client::{AttestResponse, DstackClient};
use http_client_unix_domain_socket::ErrorAndResponseJson;
use parity_scale_codec::Decode;
use serde::{de::IgnoredAny, Deserialize, Serialize};
use std::io::Cursor;

const MAX_ATTESTATION_BYTES: usize = 10 * 1024 * 1024;
const RUNTIME_EVENT: u32 = 0x0800_0001;

#[derive(Clone)]
pub(super) struct GuestInfo {
    pub info: serde_json::Value,
    pub modern_event_log: bool,
}

impl GuestInfo {
    pub async fn fetch() -> Result<Self> {
        let client = DstackClient::new(None);
        let (info, version) = tokio::join!(client.info(), client.version());
        // Attest predates Version. Guests before 0.6 strip runtime digests.
        let modern_event_log = match version {
            Ok(response) => {
                let version = semver::Version::parse(&response.version)?;
                (version.major, version.minor) >= (0, 6)
            }
            Err(error) => {
                let status = match error.downcast_ref::<ErrorAndResponseJson<serde_json::Value>>() {
                    Some(ErrorAndResponseJson::ResponseUnsuccessful(status, _)) => Some(*status),
                    _ => error
                        .downcast_ref::<reqwest::Error>()
                        .and_then(reqwest::Error::status),
                };
                // The 0.5.6 guest returns 400 for an unknown RPC. Other transports
                // may return 404. Retry other failures on the next request.
                if !matches!(status.map(|s| s.as_u16()), Some(400 | 404)) {
                    return Err(error.context("Failed to query dstack guest version"));
                }
                false
            }
        };
        Ok(Self {
            info: serde_json::to_value(info?)?,
            modern_event_log,
        })
    }
}

#[derive(Decode, Deserialize)]
pub(super) struct TdxEvidence {
    pub quote: Vec<u8>,
    event_log: Vec<TdxEvent>,
}

#[derive(Decode, Deserialize, Serialize)]
struct TdxEvent {
    imr: u32,
    event_type: u32,
    #[serde(with = "serde_human_bytes", default)]
    digest: Vec<u8>,
    event: String,
    #[serde(with = "serde_human_bytes")]
    event_payload: Vec<u8>,
    #[codec(skip)]
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<u32>,
    #[codec(skip)]
    #[serde(skip_serializing_if = "Option::is_none")]
    preimage: Option<String>,
}

#[derive(Deserialize)]
struct AttestationV1 {
    version: u64,
    platform: PlatformEvidence,
    #[serde(rename = "stack")]
    _stack: IgnoredAny,
}

#[derive(Deserialize)]
struct PlatformEvidence {
    kind: String,
    data: TdxEvidence,
}

impl TdxEvidence {
    pub fn decode(response: &AttestResponse) -> Result<Self> {
        ensure!(
            response.attestation.len() <= 2 * MAX_ATTESTATION_BYTES,
            "dstack attestation exceeds size limit"
        );
        let bytes = response
            .decode_attestation()
            .context("Invalid dstack attestation hex")?;
        match bytes.as_slice() {
            [0, platform, rest @ ..] => {
                let mut tail = rest;
                ensure!(*platform <= 1, "Expected TDX evidence in SCALE attestation");
                let evidence = <Self as Decode>::decode(&mut tail)
                    .context("Invalid dstack SCALE TDX evidence")?;
                if *platform == 1 {
                    // GCP appends a TPM quote after the TDX evidence.
                    type Pcr = (u32, String, Vec<u8>);
                    type TpmQuote = (Vec<u8>, Vec<u8>, Vec<Pcr>, Vec<u8>, u8, Vec<(u32, Vec<u8>)>);
                    TpmQuote::skip(&mut tail)?;
                }
                // Shared SCALE trailer: runtime events, report data, and VM config.
                <(Vec<(String, Vec<u8>)>, [u8; 64], String)>::skip(&mut tail)?;
                ensure!(tail.is_empty(), "Trailing data in SCALE attestation");
                Ok(evidence)
            }
            [0x80..=0x8f | 0xde | 0xdf, ..] => {
                let mut cursor = Cursor::new(&bytes);
                let att: AttestationV1 = rmp_serde::from_read(&mut cursor)
                    .context("Invalid dstack MessagePack attestation")?;
                ensure!(
                    cursor.position() == bytes.len() as u64,
                    "Trailing data in MessagePack attestation"
                );
                ensure!(
                    att.version == 1,
                    "Unsupported dstack attestation version {}",
                    att.version
                );
                ensure!(
                    matches!(att.platform.kind.as_str(), "tdx" | "gcp-tdx"),
                    "Expected TDX platform evidence"
                );
                Ok(att.platform.data)
            }
            _ => bail!("Unknown dstack attestation format"),
        }
    }

    pub fn report_event_log(&mut self, modern_event_log: bool) -> Result<serde_json::Value> {
        for entry in &mut self.event_log {
            if entry.event_type == RUNTIME_EVENT {
                if !modern_event_log && entry.version.unwrap_or(1) == 1 {
                    entry.digest.clear();
                }
            } else if !(modern_event_log
                && entry.imr == 0
                && entry.event_type == 10
                && entry.event_payload == b"ACPI DATA")
            {
                entry.event_payload.clear();
            }
        }
        Ok(serde_json::to_value(&self.event_log)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parity_scale_codec::Encode;
    use serde_json::json;

    fn attest(bytes: &[u8]) -> AttestResponse {
        AttestResponse {
            attestation: hex::encode(bytes),
        }
    }

    #[test]
    fn scale_projects_the_same_quote_and_legacy_getquote_log() {
        let events = vec![
            (0u32, 10u32, vec![0xab_u8], "", b"ACPI DATA".to_vec()),
            (3, RUNTIME_EVENT, vec![0xcd], "test", vec![0x23]),
        ];
        let runtime_events = vec![("test", vec![0x23u8])];
        let bytes = (
            0u8,
            0u8,
            vec![0xb0u8, 0xa1],
            events,
            runtime_events,
            [0x42u8; 64],
            "{}",
        )
            .encode();
        let mut evidence = TdxEvidence::decode(&attest(&bytes)).unwrap();
        assert_eq!(hex::encode(&evidence.quote), "b0a1");
        assert_eq!(
            evidence.report_event_log(false).unwrap(),
            json!([
                {"imr":0, "event_type":10, "digest":"ab", "event":"", "event_payload":""},
                {"imr":3, "event_type":RUNTIME_EVENT, "digest":"", "event":"test", "event_payload":"23"}
            ])
        );
        let mut evidence = TdxEvidence::decode(&attest(&bytes)).unwrap();
        let log = evidence.report_event_log(true).unwrap();
        assert_eq!(log[0]["event_payload"], hex::encode("ACPI DATA"));
        assert_eq!(log[1]["digest"], "cd");
        assert!(TdxEvidence::decode(&attest(&bytes[..bytes.len() - 1])).is_err());
    }

    #[test]
    fn msgpack_preserves_the_v2_runtime_digest_and_preimage() {
        let envelope = |version, kind| {
            json!({
                "version": version,
                "platform": {"kind": kind, "data": {"quote":[171,205], "event_log":[
                    {"imr":1,"event_type":1,"event":"boot","event_payload":[52],"digest":[43]},
                    {"imr":3,"event_type":RUNTIME_EVENT,"event":"test","event_payload":[35],
                     "digest":[205],"version":2,"preimage":"abcd"}
                ]}},
                "stack": {"kind":"dstack","data":{"report_data":vec![66;64],"runtime_events":[],"config":"{}"}}
            })
        };
        let encode = |value| rmp_serde::to_vec_named(&value).unwrap();
        let mut bytes = encode(envelope(1, "tdx"));
        let mut evidence = TdxEvidence::decode(&attest(&bytes)).unwrap();
        assert_eq!(hex::encode(&evidence.quote), "abcd");
        assert_eq!(
            evidence.report_event_log(true).unwrap(),
            json!([
                {"imr":1,"event_type":1,"event":"boot","event_payload":"","digest":"2b"},
                {"imr":3,"event_type":RUNTIME_EVENT,"event":"test","event_payload":"23",
                 "digest":"cd","version":2,"preimage":"abcd"}
            ])
        );
        bytes.push(0);
        assert!(TdxEvidence::decode(&attest(&bytes)).is_err());
        assert!(TdxEvidence::decode(&attest(&encode(envelope(2, "tdx")))).is_err());
        assert!(TdxEvidence::decode(&attest(&encode(envelope(1, "sev-snp")))).is_err());
    }
}
