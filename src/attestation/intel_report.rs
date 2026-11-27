use anyhow::{bail, ensure, Context, Result};
use dstack_sdk::dstack_client::{AttestResponse, DstackClient};
use http_client_unix_domain_socket::ErrorAndResponseJson;
use parity_scale_codec::Decode;
use serde::{Deserialize, Serialize};
use std::io::Cursor;

const MAX_ATTESTATION_BYTES: usize = 10 * 1024 * 1024;
const RUNTIME_EVENT: u32 = 0x0800_0001;

/// Intel `GetQuote` event log format for the dstack guest agent version.
#[derive(Clone, Copy)]
pub(super) enum IntelEventLogFormat {
    /// dstack 0.5.x returns an empty digest for runtime events.
    Before06,
    /// The format introduced in dstack 0.6.x keeps runtime digests and the `ACPI DATA` marker.
    Since06,
}

impl IntelEventLogFormat {
    pub(super) async fn from_guest(client: &DstackClient) -> Result<Self> {
        match client.version().await {
            Ok(response) => {
                let version = semver::Version::parse(&response.version)?;
                Ok(if (version.major, version.minor) >= (0, 6) {
                    Self::Since06
                } else {
                    Self::Before06
                })
            }
            Err(error) => {
                let status = match error.downcast_ref::<ErrorAndResponseJson<serde_json::Value>>() {
                    Some(ErrorAndResponseJson::ResponseUnsuccessful(status, _)) => Some(*status),
                    _ => error
                        .downcast_ref::<reqwest::Error>()
                        .and_then(reqwest::Error::status),
                };
                // Attest predates Version. The 0.5.6 guest returns 400 for an unknown RPC.
                if matches!(status.map(|s| s.as_u16()), Some(400 | 404)) {
                    Ok(Self::Before06)
                } else {
                    Err(error.context("Failed to query dstack guest version"))
                }
            }
        }
    }
}

#[derive(Decode, Deserialize)]
struct TdxEvidence {
    quote: Vec<u8>,
    event_log: Vec<TdxEvent>,
}

#[derive(Decode, Deserialize, Serialize)]
struct TdxEvent {
    imr: u32,
    event_type: u32,
    #[serde(
        deserialize_with = "serde_bytes::deserialize",
        serialize_with = "hex::serde::serialize",
        default
    )]
    digest: Vec<u8>,
    event: String,
    #[serde(
        deserialize_with = "serde_bytes::deserialize",
        serialize_with = "hex::serde::serialize"
    )]
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
    platform: TdxPlatform,
}

#[derive(Deserialize)]
#[serde(tag = "kind", content = "data")]
enum TdxPlatform {
    #[serde(rename = "tdx", alias = "gcp-tdx")]
    Tdx(TdxEvidence),
}

/// Read the Intel quote and recreate the event log returned by the guest's GetQuote RPC.
pub(super) fn legacy_fields(
    response: &AttestResponse,
    format: IntelEventLogFormat,
) -> Result<(String, serde_json::Value)> {
    ensure!(
        response.attestation.len() <= 2 * MAX_ATTESTATION_BYTES,
        "dstack attestation exceeds size limit"
    );
    let bytes = response
        .decode_attestation()
        .context("Invalid dstack attestation hex")?;
    let mut evidence = match bytes.as_slice() {
        [0, kind @ (0 | 1), rest @ ..] => {
            let mut tail = rest;
            let tdx =
                TdxEvidence::decode(&mut tail).context("Invalid dstack SCALE TDX evidence")?;
            if *kind == 1 {
                // GCP encodes a TPM quote between the TDX evidence and the common trailer.
                type Pcr = (u32, String, Vec<u8>);
                type TpmQuote = (Vec<u8>, Vec<u8>, Vec<Pcr>, Vec<u8>, u8, Vec<(u32, Vec<u8>)>);
                TpmQuote::skip(&mut tail)?;
            }
            <(Vec<(String, Vec<u8>)>, [u8; 64], String)>::skip(&mut tail)?;
            ensure!(tail.is_empty(), "Trailing data in SCALE attestation");
            tdx
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
            let TdxPlatform::Tdx(tdx) = att.platform;
            tdx
        }
        _ => bail!("Expected a dstack TDX attestation"),
    };
    ensure!(!evidence.quote.is_empty(), "Empty dstack TDX quote");
    let modern = matches!(format, IntelEventLogFormat::Since06);
    for entry in &mut evidence.event_log {
        if entry.event_type == RUNTIME_EVENT {
            if !modern {
                entry.digest.clear();
            }
        } else if !(modern
            && entry.imr == 0
            && entry.event_type == 10
            && entry.event_payload == b"ACPI DATA")
        {
            entry.event_payload.clear();
        }
        if entry.version == Some(1) {
            entry.version = None;
        }
    }
    Ok((
        hex::encode(evidence.quote),
        serde_json::to_value(evidence.event_log)?,
    ))
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

    #[tokio::test]
    async fn intel_uses_one_attest_request_and_preserves_the_getquote_fields() {
        use wiremock::{
            matchers::{body_json, method, path},
            Mock, MockServer, ResponseTemplate,
        };
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
        let server = MockServer::start().await;
        let client = DstackClient::new(Some(&server.uri()));
        let report_data = vec![0x42; 64];
        let encoded = hex::encode(&bytes);
        Mock::given(method("POST"))
            .and(path("/Version"))
            .respond_with(
                ResponseTemplate::new(400).set_body_json(json!({"error":"method not found"})),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/Attest"))
            .and(body_json(json!({"report_data":hex::encode(&report_data)})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"attestation":encoded})))
            .expect(1)
            .mount(&server)
            .await;
        let format = IntelEventLogFormat::from_guest(&client).await.unwrap();
        let fields = super::super::collect_cvm_evidence(&client, report_data, Some(format))
            .await
            .unwrap();
        assert_eq!(fields.dstack_attestation, encoded);
        assert_eq!(fields.intel_quote, "b0a1");
        let log = fields.event_log;
        assert_eq!(
            log,
            json!([
                {"imr":0, "event_type":10, "digest":"ab", "event":"", "event_payload":""},
                {"imr":3, "event_type":RUNTIME_EVENT, "digest":"", "event":"test", "event_payload":"23"}
            ])
        );
        let (_, log) = legacy_fields(&attest(&bytes), IntelEventLogFormat::Since06).unwrap();
        assert_eq!(log[0]["event_payload"], hex::encode("ACPI DATA"));
        assert_eq!(log[1]["digest"], "cd");
        assert!(legacy_fields(
            &attest(&bytes[..bytes.len() - 1]),
            IntelEventLogFormat::Since06
        )
        .is_err());
    }

    #[test]
    fn msgpack_preserves_the_v2_runtime_digest_and_preimage() {
        let envelope = |version, kind| {
            json!({
                "version": version,
                "platform": {"data": {"quote":[171,205], "event_log":[
                    {"imr":1,"event_type":1,"event":"boot","event_payload":[52],"digest":[43],"version":1},
                    {"imr":3,"event_type":RUNTIME_EVENT,"event":"test","event_payload":[35],
                     "digest":[205],"version":2,"preimage":"abcd"}
                ]}, "kind":kind},
                "stack": {"kind":"dstack","data":{"report_data":vec![66;64],"runtime_events":[],"config":"{}"}}
            })
        };
        let encode = |value| rmp_serde::to_vec_named(&value).unwrap();
        let mut bytes = encode(envelope(1, "tdx"));
        let (quote, log) = legacy_fields(&attest(&bytes), IntelEventLogFormat::Since06).unwrap();
        assert_eq!(quote, "abcd");
        assert_eq!(
            log,
            json!([
                {"imr":1,"event_type":1,"event":"boot","event_payload":"","digest":"2b"},
                {"imr":3,"event_type":RUNTIME_EVENT,"event":"test","event_payload":"23",
                 "digest":"cd","version":2,"preimage":"abcd"}
            ])
        );
        bytes.push(0);
        assert!(legacy_fields(&attest(&bytes), IntelEventLogFormat::Since06).is_err());
        assert!(legacy_fields(
            &attest(&encode(envelope(2, "tdx"))),
            IntelEventLogFormat::Since06
        )
        .is_err());
        assert!(legacy_fields(
            &attest(&encode(envelope(1, "sev-snp"))),
            IntelEventLogFormat::Since06
        )
        .is_err());
    }
}
