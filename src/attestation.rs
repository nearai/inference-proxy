use tracing::{error, info};

use crate::types::AttestationReport;

/// Errors from attestation generation.
#[derive(Debug, thiserror::Error)]
pub enum AttestationError {
    /// User-provided nonce is invalid (bad hex or wrong length).
    #[error("{0}")]
    InvalidNonce(String),
    /// Internal error (dstack, GPU subprocess, etc.).
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

/// Build TDX report data: [signing_address (padded to 32 bytes) || nonce (32 bytes)].
fn build_report_data(signing_address_bytes: &[u8], nonce: &[u8; 32]) -> Vec<u8> {
    let mut data = vec![0u8; 64];
    let len = signing_address_bytes.len().min(32);
    data[..len].copy_from_slice(&signing_address_bytes[..len]);
    data[32..64].copy_from_slice(nonce);
    data
}

/// Parse nonce from hex string or generate random 32 bytes.
fn parse_nonce(nonce: Option<&str>) -> Result<[u8; 32], AttestationError> {
    match nonce {
        Some(hex_str) => {
            let bytes = hex::decode(hex_str).map_err(|_| {
                AttestationError::InvalidNonce("Nonce must be hex-encoded".to_string())
            })?;
            if bytes.len() != 32 {
                return Err(AttestationError::InvalidNonce(
                    "Nonce must be 32 bytes".to_string(),
                ));
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            Ok(arr)
        }
        None => {
            let mut arr = [0u8; 32];
            use rand::RngCore;
            rand::rngs::OsRng.fill_bytes(&mut arr);
            Ok(arr)
        }
    }
}

/// Collect GPU evidence via Python subprocess.
/// In no_gpu_mode, produces canned evidence.
async fn collect_gpu_evidence(
    nonce_hex: &str,
    no_gpu_mode: bool,
) -> anyhow::Result<serde_json::Value> {
    if no_gpu_mode {
        info!("GPU evidence no-GPU mode enabled; using canned evidence");
    }

    // Build a small Python script that collects GPU evidence.
    // ppcie_mode=False is required on PPCIE systems (the default True triggers a
    // "standalone mode not supported" error). Safe on non-PPCIE systems too.
    let script = if no_gpu_mode {
        format!(
            r#"
import json
from verifier import cc_admin
evidence = cc_admin.collect_gpu_evidence_remote("{nonce_hex}", no_gpu_mode=True)
print(json.dumps(evidence))
"#,
        )
    } else {
        format!(
            r#"
import json
from verifier import cc_admin
evidence = cc_admin.collect_gpu_evidence_remote("{nonce_hex}", ppcie_mode=False)
print(json.dumps(evidence))
"#,
        )
    };

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        tokio::process::Command::new("python3")
            .arg("-c")
            .arg(&script)
            .output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("GPU evidence subprocess timed out after 60s"))?
    .map_err(|e| anyhow::anyhow!("Failed to run GPU evidence subprocess: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        error!(stderr = %stderr, "GPU evidence subprocess failed");
        anyhow::bail!("GPU evidence collection failed: {stderr}");
    }

    // The Python verifier library prints info messages to stdout (e.g. "Number of GPUs
    // available : 8"). Extract only the last line, which contains the JSON evidence array.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_line = stdout
        .lines()
        .rev()
        .find(|line| line.starts_with('['))
        .ok_or_else(|| anyhow::anyhow!("No JSON array found in GPU evidence output"))?;
    let evidence: serde_json::Value = serde_json::from_str(json_line)
        .map_err(|e| anyhow::anyhow!("Failed to parse GPU evidence JSON: {e}"))?;

    Ok(evidence)
}

/// Build NVIDIA payload JSON.
fn build_nvidia_payload(nonce_hex: &str, evidences: &serde_json::Value) -> String {
    serde_json::json!({
        "nonce": nonce_hex,
        "evidence_list": evidences,
        "arch": "HOPPER",
    })
    .to_string()
}

/// Generate a complete attestation report.
pub async fn generate_attestation(
    signing_address: &str,
    signing_algo: &str,
    signing_public_key: &str,
    signing_address_bytes: &[u8],
    nonce: Option<&str>,
    gpu_no_hw_mode: bool,
) -> Result<AttestationReport, AttestationError> {
    let nonce_bytes = parse_nonce(nonce)?;
    let nonce_hex = hex::encode(nonce_bytes);

    // Build TDX report data
    let report_data = build_report_data(signing_address_bytes, &nonce_bytes);

    // Get TDX quote from dstack
    let client = dstack_sdk::dstack_client::DstackClient::new(None);
    let quote_result = client.get_quote(report_data).await?;
    let event_log: serde_json::Value =
        serde_json::from_str(&quote_result.event_log).map_err(anyhow::Error::from)?;

    // Collect GPU evidence
    let gpu_evidence = collect_gpu_evidence(&nonce_hex, gpu_no_hw_mode).await?;
    let nvidia_payload = build_nvidia_payload(&nonce_hex, &gpu_evidence);

    // Get system info
    let info = client.info().await?;
    let info_value = serde_json::to_value(&info).map_err(anyhow::Error::from)?;

    Ok(AttestationReport {
        signing_address: signing_address.to_string(),
        signing_algo: signing_algo.to_string(),
        signing_public_key: signing_public_key.to_string(),
        request_nonce: nonce_hex,
        intel_quote: quote_result.quote,
        nvidia_payload,
        event_log,
        info: info_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_report_data_structure() {
        // 20-byte Ethereum address
        let address = vec![0xABu8; 20];
        let nonce = [0xCDu8; 32];

        let data = build_report_data(&address, &nonce);

        assert_eq!(data.len(), 64);
        // First 20 bytes = address
        assert_eq!(&data[..20], &[0xAB; 20]);
        // Bytes 20..32 = zero padding
        assert_eq!(&data[20..32], &[0x00; 12]);
        // Last 32 bytes = nonce
        assert_eq!(&data[32..64], &[0xCD; 32]);
    }

    #[test]
    fn test_build_report_data_32_byte_address() {
        // Ed25519 public key (32 bytes) fills entire first half
        let address = vec![0xFFu8; 32];
        let nonce = [0x11u8; 32];

        let data = build_report_data(&address, &nonce);

        assert_eq!(data.len(), 64);
        assert_eq!(&data[..32], &[0xFF; 32]);
        assert_eq!(&data[32..64], &[0x11; 32]);
    }

    #[test]
    fn test_build_report_data_oversized_address_truncated() {
        // Address larger than 32 bytes gets truncated
        let address = vec![0xAA; 40];
        let nonce = [0x00; 32];

        let data = build_report_data(&address, &nonce);

        assert_eq!(data.len(), 64);
        // Only first 32 bytes of address used
        assert_eq!(&data[..32], &[0xAA; 32]);
    }

    #[test]
    fn test_parse_nonce_valid_hex() {
        let hex_str = "a".repeat(64); // 32 bytes as hex
        let result = parse_nonce(Some(&hex_str)).unwrap();
        assert_eq!(result, [0xAA; 32]);
    }

    #[test]
    fn test_parse_nonce_invalid_hex() {
        let result = parse_nonce(Some("not_valid_hex!"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("hex-encoded"));
    }

    #[test]
    fn test_parse_nonce_wrong_length() {
        // 16 bytes (32 hex chars) is too short
        let short_hex = "ab".repeat(16);
        let result = parse_nonce(Some(&short_hex));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("32 bytes"));
    }

    #[test]
    fn test_parse_nonce_none_generates_random() {
        let nonce1 = parse_nonce(None).unwrap();
        let nonce2 = parse_nonce(None).unwrap();
        // Two random nonces should (almost certainly) differ
        assert_ne!(nonce1, nonce2);
        assert_eq!(nonce1.len(), 32);
    }

    #[test]
    fn test_build_nvidia_payload_structure() {
        let nonce = "abc123";
        let evidence = serde_json::json!([{"gpu": "H100"}]);

        let payload_str = build_nvidia_payload(nonce, &evidence);
        let payload: serde_json::Value = serde_json::from_str(&payload_str).unwrap();

        assert_eq!(payload["nonce"], "abc123");
        assert_eq!(payload["arch"], "HOPPER");
        assert_eq!(payload["evidence_list"][0]["gpu"], "H100");
    }
}
