use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use sha2::{Digest, Sha256};
use tokio::sync::{RwLock, Semaphore};
use tracing::{error, info, warn};

use crate::types::AttestationReport;

/// Cache key for nonce-less attestation reports.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct AttestationCacheKey {
    signing_algo: String,
    include_tls_fingerprint: bool,
}

struct CachedReport {
    report: AttestationReport,
    created_at: Instant,
}

/// Caches nonce-less attestation reports and serializes GPU evidence collection.
///
/// GPU evidence collection spawns a Python subprocess that calls `nvmlInit()`.
/// Under heavy GPU load, `nvmlInit` can intermittently time out (5s timeout in
/// the NVIDIA verifier library). This cache:
/// 1. Serves pre-generated reports for requests without a nonce (the common case).
/// 2. Serializes subprocess calls so only one `nvmlInit` runs at a time.
/// 3. Retries once on GPU evidence failure.
pub struct AttestationCache {
    /// Cached reports keyed by (signing_algo, include_tls_fingerprint).
    reports: RwLock<HashMap<AttestationCacheKey, CachedReport>>,
    /// Serializes GPU evidence subprocess calls (only 1 at a time).
    gpu_semaphore: Semaphore,
    /// Cache TTL in seconds.
    ttl_secs: u64,
}

impl AttestationCache {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            reports: RwLock::new(HashMap::new()),
            gpu_semaphore: Semaphore::new(1),
            ttl_secs,
        }
    }

    /// Get a cached report if it exists and is fresh.
    pub async fn get(
        &self,
        signing_algo: &str,
        include_tls_fingerprint: bool,
    ) -> Option<AttestationReport> {
        let key = AttestationCacheKey {
            signing_algo: signing_algo.to_string(),
            include_tls_fingerprint,
        };
        let reports = self.reports.read().await;
        if let Some(cached) = reports.get(&key) {
            if cached.created_at.elapsed().as_secs() < self.ttl_secs {
                metrics::counter!("attestation_cache_hits_total").increment(1);
                return Some(cached.report.clone());
            }
        }
        metrics::counter!("attestation_cache_misses_total").increment(1);
        None
    }

    /// Store a report in the cache.
    pub async fn set(
        &self,
        signing_algo: &str,
        include_tls_fingerprint: bool,
        report: AttestationReport,
    ) {
        let key = AttestationCacheKey {
            signing_algo: signing_algo.to_string(),
            include_tls_fingerprint,
        };
        let mut reports = self.reports.write().await;
        reports.insert(
            key,
            CachedReport {
                report,
                created_at: Instant::now(),
            },
        );
    }

    /// Acquire the GPU evidence semaphore (serializes subprocess calls).
    pub async fn acquire_gpu_permit(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.gpu_semaphore
            .acquire()
            .await
            .expect("semaphore closed")
    }
}

/// Spawn a background task that periodically refreshes cached attestation reports.
pub fn spawn_cache_refresh_task(
    cache: Arc<AttestationCache>,
    model_name: String,
    signing: Arc<crate::signing::SigningPair>,
    gpu_no_hw_mode: bool,
    tls_cert_fingerprint: Option<String>,
    refresh_interval_secs: u64,
) {
    tokio::spawn(async move {
        // Initial delay to let the server start up.
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        loop {
            for algo in &["ecdsa", "ed25519"] {
                let (signing_address, signing_address_bytes, signing_public_key) = match *algo {
                    "ecdsa" => (
                        signing.ecdsa.signing_address.clone(),
                        signing.ecdsa.signing_address_bytes.clone(),
                        signing.ecdsa.signing_public_key.clone(),
                    ),
                    "ed25519" => (
                        signing.ed25519.signing_address.clone(),
                        signing.ed25519.signing_address_bytes.clone(),
                        signing.ed25519.signing_public_key.clone(),
                    ),
                    _ => unreachable!(),
                };

                // Refresh without TLS fingerprint (most common).
                let _permit = cache.acquire_gpu_permit().await;
                match generate_attestation_inner(AttestationParams {
                    model_name: &model_name,
                    signing_address: &signing_address,
                    signing_algo: algo,
                    signing_public_key: &signing_public_key,
                    signing_address_bytes: &signing_address_bytes,
                    nonce: None,
                    gpu_no_hw_mode,
                    tls_cert_fingerprint: None,
                })
                .await
                {
                    Ok(report) => {
                        cache.set(algo, false, report).await;
                        info!(algo, "Background attestation cache refresh succeeded");
                    }
                    Err(e) => {
                        warn!(algo, error = %e, "Background attestation cache refresh failed");
                    }
                }
                drop(_permit);

                // Also refresh with TLS fingerprint if configured.
                if let Some(ref fp) = tls_cert_fingerprint {
                    let _permit = cache.acquire_gpu_permit().await;
                    match generate_attestation_inner(AttestationParams {
                        model_name: &model_name,
                        signing_address: &signing_address,
                        signing_algo: algo,
                        signing_public_key: &signing_public_key,
                        signing_address_bytes: &signing_address_bytes,
                        nonce: None,
                        gpu_no_hw_mode,
                        tls_cert_fingerprint: Some(fp.as_str()),
                    })
                    .await
                    {
                        Ok(report) => {
                            cache.set(algo, true, report).await;
                        }
                        Err(e) => {
                            warn!(algo, error = %e, "Background attestation cache refresh (with TLS) failed");
                        }
                    }
                    drop(_permit);
                }
            }

            let sleep_secs = if refresh_interval_secs == 0 {
                warn!("refresh_interval_secs was 0; clamping to 1s to avoid busy loop");
                1
            } else {
                refresh_interval_secs
            };
            tokio::time::sleep(std::time::Duration::from_secs(sleep_secs)).await;
        }
    });
}

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

/// Build TDX report data (64 bytes).
///
/// Without cert fingerprint: `[signing_address (padded to 32) || nonce (32)]`
/// With cert fingerprint:    `[SHA256(signing_address || cert_fingerprint) || nonce (32)]`
fn build_report_data(
    signing_address_bytes: &[u8],
    nonce: &[u8; 32],
    cert_fingerprint: Option<&[u8]>,
) -> Vec<u8> {
    let mut data = vec![0u8; 64];
    match cert_fingerprint {
        Some(fp) => {
            let mut hasher = Sha256::new();
            hasher.update(signing_address_bytes);
            hasher.update(fp);
            let hash = hasher.finalize();
            data[..32].copy_from_slice(&hash);
        }
        None => {
            let len = signing_address_bytes.len().min(32);
            data[..len].copy_from_slice(&signing_address_bytes[..len]);
        }
    }
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
            use rand::rand_core::TryRngCore;
            rand::rngs::OsRng
                .try_fill_bytes(&mut arr)
                .expect("Failed to generate random nonce bytes");
            Ok(arr)
        }
    }
}

/// Collect GPU evidence via Python subprocess (single attempt).
async fn collect_gpu_evidence_once(
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

/// Collect GPU evidence with one retry on failure.
///
/// nvmlInit can intermittently time out under heavy GPU load. A single retry
/// after a short delay often succeeds once the driver lock is released.
async fn collect_gpu_evidence(
    nonce_hex: &str,
    no_gpu_mode: bool,
) -> anyhow::Result<serde_json::Value> {
    match collect_gpu_evidence_once(nonce_hex, no_gpu_mode).await {
        Ok(evidence) => Ok(evidence),
        Err(first_err) => {
            warn!(
                error = %first_err,
                "GPU evidence collection failed, retrying after 2s"
            );
            metrics::counter!("gpu_evidence_retries_total").increment(1);
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            collect_gpu_evidence_once(nonce_hex, no_gpu_mode).await
        }
    }
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

/// Compute SHA-256 hash of a PEM certificate's Subject Public Key Info (SPKI).
pub fn compute_spki_hash(cert_path: &str) -> anyhow::Result<String> {
    let pem_data = std::fs::read(cert_path)
        .map_err(|e| anyhow::anyhow!("failed to read cert {cert_path}: {e}"))?;
    let (_, pem) = x509_parser::pem::parse_x509_pem(&pem_data)
        .map_err(|e| anyhow::anyhow!("failed to parse PEM: {e}"))?;
    let (_, cert) = x509_parser::parse_x509_certificate(&pem.contents)
        .map_err(|e| anyhow::anyhow!("failed to parse X.509: {e}"))?;
    let spki_der = cert.tbs_certificate.subject_pki.raw;
    let hash = Sha256::digest(spki_der);
    Ok(hex::encode(hash))
}

#[cfg(test)]
mod tests_spki {
    use super::compute_spki_hash;
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::fs::File;
    use std::io::Write;

    // A small, valid self-signed test certificate in PEM format.
    // This is only used for unit testing of SPKI hashing.
    const TEST_CERT_PEM: &str = r#"-----BEGIN CERTIFICATE-----
MIIDEzCCAfugAwIBAgIUc8i7HuXjfzh0UgxHI50TZ5VvEMswDQYJKoZIhvcNAQEL
BQAwGTEXMBUGA1UEAwwOdGVzdC1sb2NhbGhvc3QwHhcNMjYwMjEzMTMwODAzWhcN
MzYwMjExMTMwODAzWjAZMRcwFQYDVQQDDA50ZXN0LWxvY2FsaG9zdDCCASIwDQYJ
KoZIhvcNAQEBBQADggEPADCCAQoCggEBAJ3j+xeMEJ9c4nfYNXLOFwkdBU1lxI/u
qWHCnHoNwbmVFBZDvksf9jv8KQwfqaOj8VwBVHat1rbpkgCkcwVHnmZBB6DjDhhs
2wp8MDnjHR58J3tqvgZmrf6Dp4TkziwAlGWHM//wI9km8KWr0cX2p/z3YfHOWj3F
yaRbJ6b/QFJ3fyuk8UY9d9WlKG91wPX8Oeg3d2rSiAXx3daO/MbkRroT2XpKaYux
qTDsxAWRqxkCcQsdHxXG+rbA3HPTpirNWDxLRmxm0Q8PCEFG9EF+Mu1XVmOgkUTp
7p98vdwtP3c6HnfoMkpobfEUmTbtcXkJHMTPr2IrqxMC/8I+8+F5lrMCAwEAAaNT
MFEwHQYDVR0OBBYEFJsscWLVB2QcCxb9PxMMG9vxZZ/8MB8GA1UdIwQYMBaAFJss
cWLVB2QcCxb9PxMMG9vxZZ/8MA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQEL
BQADggEBAIPwnN16vmNi26XppI4E6TzOY4EXyqhPhtGNeos7Hxsw6DXKA28iaaOW
xnH5LeNFP1//9hojTCo/w6CS4BWJNlGoFPfAHIAHFAIVkqOcmO+YLGYotcR67ftd
loGVCS8p4a88M7X2JeziizPlssmbzQkcAGQ3latUu5O6wxUATFFWmdPELhm8xRdW
qB2wGiBhxD46CKcMKZrtW+P8SjhhxXEJ2x+UYdSxXSTTnrBAZi23yo4TNFVXw5jA
Tw4GxEVK193pwe3l749yk1dkJkxAfRCavr3BVP5Br53GWHVFBDOR2tPw83frzTBJ
nU+jXBG7tgClr/DntUBJx+xfNWpxLKE=
-----END CERTIFICATE-----
"#;

    fn write_temp_file(prefix: &str, contents: &str) -> String {
        let mut path = std::env::temp_dir();
        // Use the process ID and a monotonic counter to reduce collision risk.
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let filename = format!("{}_{}_{}.pem", prefix, std::process::id(), id);
        path.push(filename);

        let mut file = File::create(&path).expect("failed to create temp file");
        file.write_all(contents.as_bytes())
            .expect("failed to write temp file");

        path.to_string_lossy().to_string()
    }

    #[test]
    fn test_compute_spki_hash_valid_cert() {
        let path = write_temp_file("valid_cert", TEST_CERT_PEM);

        // Independently compute expected SPKI hash from the in-memory PEM.
        let pem_bytes = TEST_CERT_PEM.as_bytes();
        let (_, pem) =
            x509_parser::pem::parse_x509_pem(pem_bytes).expect("failed to parse test PEM");
        let (_, cert) =
            x509_parser::parse_x509_certificate(&pem.contents).expect("failed to parse test X.509");
        let spki_der = cert.tbs_certificate.subject_pki.raw;
        let expected_hash = {
            let hash = Sha256::digest(spki_der);
            hex::encode(hash)
        };

        let actual_hash =
            compute_spki_hash(&path).expect("compute_spki_hash should succeed for valid cert");

        // Clean up the temp file; ignore errors.
        let _ = fs::remove_file(&path);

        assert_eq!(actual_hash, expected_hash);
    }

    #[test]
    fn test_compute_spki_hash_invalid_pem() {
        let path = write_temp_file("invalid_pem", "this is not a valid PEM certificate");

        let result = compute_spki_hash(&path);

        // Clean up the temp file; ignore errors.
        let _ = fs::remove_file(&path);

        assert!(result.is_err(), "expected error for invalid PEM input");
    }

    #[test]
    fn test_compute_spki_hash_missing_file() {
        // Use a path that is very unlikely to exist.
        let path = "/nonexistent/path/to/cert_for_spki_hash_test.pem";

        let result = compute_spki_hash(path);

        assert!(
            result.is_err(),
            "expected error for missing certificate file"
        );
    }
}
/// Parameters for generating an attestation report.
pub struct AttestationParams<'a> {
    pub model_name: &'a str,
    pub signing_address: &'a str,
    pub signing_algo: &'a str,
    pub signing_public_key: &'a str,
    pub signing_address_bytes: &'a [u8],
    pub nonce: Option<&'a str>,
    pub gpu_no_hw_mode: bool,
    pub tls_cert_fingerprint: Option<&'a str>,
}

/// Generate a complete attestation report (core logic, no caching).
async fn generate_attestation_inner(
    params: AttestationParams<'_>,
) -> Result<AttestationReport, AttestationError> {
    let nonce_bytes = parse_nonce(params.nonce)?;
    let nonce_hex = hex::encode(nonce_bytes);

    // Build TDX report data (binds cert fingerprint when present)
    let fp_bytes = params
        .tls_cert_fingerprint
        .map(hex::decode)
        .transpose()
        .map_err(|e| {
            AttestationError::Internal(anyhow::anyhow!("bad cert fingerprint hex: {e}"))
        })?;
    let report_data = build_report_data(
        params.signing_address_bytes,
        &nonce_bytes,
        fp_bytes.as_deref(),
    );

    // Get TDX quote from dstack
    let client = dstack_sdk::dstack_client::DstackClient::new(None);
    let quote_result = client.get_quote(report_data).await?;
    let event_log: serde_json::Value =
        serde_json::from_str(&quote_result.event_log).map_err(anyhow::Error::from)?;

    // Collect GPU evidence
    let gpu_evidence = collect_gpu_evidence(&nonce_hex, params.gpu_no_hw_mode).await?;
    let nvidia_payload = build_nvidia_payload(&nonce_hex, &gpu_evidence);

    // Get system info
    let info = client.info().await?;
    let info_value = serde_json::to_value(&info).map_err(anyhow::Error::from)?;

    Ok(AttestationReport {
        model_name: params.model_name.to_string(),
        signing_address: params.signing_address.to_string(),
        signing_algo: params.signing_algo.to_string(),
        signing_public_key: params.signing_public_key.to_string(),
        request_nonce: nonce_hex,
        intel_quote: quote_result.quote,
        nvidia_payload,
        event_log,
        info: info_value,
        tls_cert_fingerprint: params.tls_cert_fingerprint.map(|s| s.to_string()),
    })
}

/// Generate an attestation report, using the cache for nonce-less requests.
///
/// When a caller provides a nonce, the GPU evidence and TDX quote are
/// cryptographically bound to that nonce, so we must generate fresh.
/// When no nonce is provided, we serve a cached report (which contains its
/// own randomly-generated nonce) — the caller accepts whatever nonce we return.
pub async fn generate_attestation(
    params: AttestationParams<'_>,
    cache: Option<&AttestationCache>,
) -> Result<AttestationReport, AttestationError> {
    let is_nonceless = params.nonce.is_none();
    let include_tls = params.tls_cert_fingerprint.is_some();
    let signing_algo = params.signing_algo.to_string();

    // For nonce-less requests, try the cache first.
    if is_nonceless {
        if let Some(cache) = cache {
            if let Some(report) = cache.get(&signing_algo, include_tls).await {
                return Ok(report);
            }
        }
    }

    // Generate fresh report. Acquire semaphore to serialize GPU evidence calls.
    let report = if let Some(cache) = cache {
        let _permit = cache.acquire_gpu_permit().await;
        // Double-check cache after acquiring permit (another request may have filled it).
        if is_nonceless {
            if let Some(report) = cache.get(&signing_algo, include_tls).await {
                return Ok(report);
            }
        }
        let report = generate_attestation_inner(params).await?;
        if is_nonceless {
            cache.set(&signing_algo, include_tls, report.clone()).await;
        }
        report
    } else {
        generate_attestation_inner(params).await?
    };

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_report_data_structure() {
        // 20-byte Ethereum address, no cert fingerprint
        let address = vec![0xABu8; 20];
        let nonce = [0xCDu8; 32];

        let data = build_report_data(&address, &nonce, None);

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

        let data = build_report_data(&address, &nonce, None);

        assert_eq!(data.len(), 64);
        assert_eq!(&data[..32], &[0xFF; 32]);
        assert_eq!(&data[32..64], &[0x11; 32]);
    }

    #[test]
    fn test_build_report_data_oversized_address_truncated() {
        // Address larger than 32 bytes gets truncated
        let address = vec![0xAA; 40];
        let nonce = [0x00; 32];

        let data = build_report_data(&address, &nonce, None);

        assert_eq!(data.len(), 64);
        // Only first 32 bytes of address used
        assert_eq!(&data[..32], &[0xAA; 32]);
    }

    #[test]
    fn test_build_report_data_with_cert_fingerprint() {
        let address = vec![0xABu8; 20];
        let nonce = [0xCDu8; 32];
        let cert_fp = vec![0xEEu8; 32];

        let data = build_report_data(&address, &nonce, Some(&cert_fp));

        assert_eq!(data.len(), 64);
        // First 32 bytes = SHA256(address || cert_fingerprint)
        let mut hasher = Sha256::new();
        hasher.update(&address);
        hasher.update(&cert_fp);
        let expected_hash = hasher.finalize();
        assert_eq!(&data[..32], &expected_hash[..]);
        // Last 32 bytes = nonce
        assert_eq!(&data[32..64], &[0xCD; 32]);
    }

    #[test]
    fn test_build_report_data_cert_fingerprint_changes_output() {
        let address = vec![0xABu8; 20];
        let nonce = [0xCDu8; 32];

        let data_without = build_report_data(&address, &nonce, None);
        let data_with = build_report_data(&address, &nonce, Some(&[0xEE; 32]));

        // First 32 bytes must differ
        assert_ne!(&data_without[..32], &data_with[..32]);
        // Nonce (last 32) stays the same
        assert_eq!(&data_without[32..], &data_with[32..]);
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

    fn make_test_report(algo: &str, nonce: &str) -> AttestationReport {
        AttestationReport {
            model_name: "test-model".to_string(),
            signing_address: "0xtest".to_string(),
            signing_algo: algo.to_string(),
            signing_public_key: "pk".to_string(),
            request_nonce: nonce.to_string(),
            intel_quote: "quote".to_string(),
            nvidia_payload: "payload".to_string(),
            event_log: serde_json::json!({}),
            info: serde_json::json!({}),
            tls_cert_fingerprint: None,
        }
    }

    #[tokio::test]
    async fn test_attestation_cache_hit() {
        let cache = AttestationCache::new(300);
        let report = make_test_report("ecdsa", "aabb");
        cache.set("ecdsa", false, report.clone()).await;

        let result = cache.get("ecdsa", false).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().request_nonce, "aabb");
    }

    #[tokio::test]
    async fn test_attestation_cache_miss_different_algo() {
        let cache = AttestationCache::new(300);
        cache
            .set("ecdsa", false, make_test_report("ecdsa", "aa"))
            .await;

        assert!(cache.get("ed25519", false).await.is_none());
    }

    #[tokio::test]
    async fn test_attestation_cache_miss_different_tls() {
        let cache = AttestationCache::new(300);
        cache
            .set("ecdsa", false, make_test_report("ecdsa", "aa"))
            .await;

        assert!(cache.get("ecdsa", true).await.is_none());
    }

    #[tokio::test]
    async fn test_attestation_cache_ttl_expiry() {
        let cache = AttestationCache::new(1);
        cache
            .set("ecdsa", false, make_test_report("ecdsa", "aa"))
            .await;

        assert!(cache.get("ecdsa", false).await.is_some());
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        assert!(cache.get("ecdsa", false).await.is_none());
    }

    #[tokio::test]
    async fn test_gpu_semaphore_serializes() {
        let cache = Arc::new(AttestationCache::new(300));

        // Hold the first permit.
        let permit1 = cache.acquire_gpu_permit().await;

        // Spawn a task that tries to acquire the semaphore while we hold it.
        let cache2 = cache.clone();
        let mut handle = tokio::spawn(async move {
            let _permit = cache2.acquire_gpu_permit().await;
        });

        // The second acquire should block (not complete within 50ms).
        let result = tokio::time::timeout(std::time::Duration::from_millis(50), &mut handle).await;
        assert!(
            result.is_err(),
            "second acquire should block while first permit is held"
        );

        // Release the first permit — the second task should now complete.
        drop(permit1);
        tokio::time::timeout(std::time::Duration::from_millis(50), handle)
            .await
            .expect("second acquire should complete after first permit is dropped")
            .expect("task should not panic");
    }
}
