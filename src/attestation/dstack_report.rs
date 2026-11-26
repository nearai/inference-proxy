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
    pub async fn fetch(client: &DstackClient, platform: &PlatformEvidence) -> Result<Self> {
        if let PlatformEvidence::SevSnp(_) = platform {
            return Ok(Self {
                info: client.info_raw().await?,
                modern_event_log: false,
            });
        }
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
    platform: PlatformEvidence,
    #[serde(rename = "stack")]
    _stack: IgnoredAny,
}

#[derive(Deserialize)]
#[serde(tag = "kind", content = "data")]
pub(super) enum PlatformEvidence {
    #[serde(rename = "tdx", alias = "gcp-tdx")]
    Tdx(TdxEvidence),
    #[serde(rename = "sev-snp")]
    SevSnp(SnpEvidence),
}

#[derive(Decode, Deserialize)]
pub(super) struct SnpEvidence {
    report: Vec<u8>,
    #[serde(rename = "cert_chain")]
    _cert_chain: Vec<Vec<u8>>,
    #[serde(rename = "mr_config")]
    _mr_config: String,
}

impl PlatformEvidence {
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
                let evidence = match platform {
                    0 | 1 => {
                        let tdx = TdxEvidence::decode(&mut tail)
                            .context("Invalid dstack SCALE TDX evidence")?;
                        if *platform == 1 {
                            // GCP appends a TPM quote after the TDX evidence.
                            type Pcr = (u32, String, Vec<u8>);
                            type TpmQuote =
                                (Vec<u8>, Vec<u8>, Vec<Pcr>, Vec<u8>, u8, Vec<(u32, Vec<u8>)>);
                            TpmQuote::skip(&mut tail)?;
                        }
                        Self::Tdx(tdx)
                    }
                    3 => Self::SevSnp(
                        SnpEvidence::decode(&mut tail)
                            .context("Invalid dstack SCALE SEV-SNP evidence")?,
                    ),
                    _ => bail!("Unsupported dstack SCALE platform {platform}"),
                };
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
                Ok(att.platform)
            }
            _ => bail!("Unknown dstack attestation format"),
        }
    }

    pub fn into_intel_fields(self, modern_event_log: bool) -> Result<(String, serde_json::Value)> {
        match self {
            Self::Tdx(mut tdx) => {
                let event_log = tdx.report_event_log(modern_event_log)?;
                Ok((hex::encode(tdx.quote), event_log))
            }
            Self::SevSnp(snp) => {
                ensure!(!snp.report.is_empty(), "Empty dstack SEV-SNP report");
                Ok((String::new(), serde_json::json!([])))
            }
        }
    }
}

impl TdxEvidence {
    fn report_event_log(&mut self, modern_event_log: bool) -> Result<serde_json::Value> {
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
        let evidence = PlatformEvidence::decode(&attest(&bytes)).unwrap();
        let (quote, log) = evidence.into_intel_fields(false).unwrap();
        assert_eq!(quote, "b0a1");
        assert_eq!(
            log,
            json!([
                {"imr":0, "event_type":10, "digest":"ab", "event":"", "event_payload":""},
                {"imr":3, "event_type":RUNTIME_EVENT, "digest":"", "event":"test", "event_payload":"23"}
            ])
        );
        let evidence = PlatformEvidence::decode(&attest(&bytes)).unwrap();
        let (_, log) = evidence.into_intel_fields(true).unwrap();
        assert_eq!(log[0]["event_payload"], hex::encode("ACPI DATA"));
        assert_eq!(log[1]["digest"], "cd");
        assert!(PlatformEvidence::decode(&attest(&bytes[..bytes.len() - 1])).is_err());
    }

    #[tokio::test]
    async fn snp_accepts_nullable_info_and_has_no_legacy_intel_fields() {
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };

        let server = MockServer::start().await;
        let tcb = json!({"mrtd":null, "rtmr3":null, "event_log":[]});
        Mock::given(method("POST"))
            .and(path("/Info"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"tcb_info":tcb.to_string()})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let client = DstackClient::new(Some(&server.uri()));

        let report = vec![0x42u8; 1184];
        let cert_chain = vec![vec![1u8, 2]];
        let scale = (
            0u8,
            3u8,
            &report,
            &cert_chain,
            "{}",
            Vec::<(String, Vec<u8>)>::new(),
            [0u8; 64],
            "{}",
        )
            .encode();
        let msgpack = rmp_serde::to_vec_named(&json!({
            "version": 1,
            "platform": {"kind":"sev-snp", "data":{"report":report, "cert_chain":cert_chain, "mr_config":"{}"}},
            "stack": {"kind":"dstack", "data":{"report_data":vec![0;64], "runtime_events":[], "config":"{}"}}
        })).unwrap();
        let evidence = PlatformEvidence::decode(&attest(&scale)).unwrap();
        let guest = GuestInfo::fetch(&client, &evidence).await.unwrap();
        assert_eq!(guest.info["tcb_info"], tcb);
        let expected = (String::new(), json!([]));
        assert_eq!(
            evidence.into_intel_fields(guest.modern_event_log).unwrap(),
            expected
        );
        assert_eq!(
            PlatformEvidence::decode(&attest(&msgpack))
                .unwrap()
                .into_intel_fields(false)
                .unwrap(),
            expected
        );
        assert!(PlatformEvidence::decode(&attest(&scale[..scale.len() - 1])).is_err());
    }

    #[test]
    fn msgpack_preserves_the_v2_runtime_digest_and_preimage() {
        let envelope = |version, kind| {
            json!({
                "version": version,
                "platform": {"data": {"quote":[171,205], "event_log":[
                    {"imr":1,"event_type":1,"event":"boot","event_payload":[52],"digest":[43]},
                    {"imr":3,"event_type":RUNTIME_EVENT,"event":"test","event_payload":[35],
                     "digest":[205],"version":2,"preimage":"abcd"}
                ]}, "kind":kind},
                "stack": {"kind":"dstack","data":{"report_data":vec![66;64],"runtime_events":[],"config":"{}"}}
            })
        };
        let encode = |value| rmp_serde::to_vec_named(&value).unwrap();
        let mut bytes = encode(envelope(1, "tdx"));
        let evidence = PlatformEvidence::decode(&attest(&bytes)).unwrap();
        let (quote, log) = evidence.into_intel_fields(true).unwrap();
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
        assert!(PlatformEvidence::decode(&attest(&bytes)).is_err());
        assert!(PlatformEvidence::decode(&attest(&encode(envelope(2, "tdx")))).is_err());
        assert!(PlatformEvidence::decode(&attest(&encode(envelope(1, "sev-snp")))).is_err());
    }
}
