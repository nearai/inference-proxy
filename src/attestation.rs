use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, OnceCell, RwLock};
use tracing::{error, info, warn};

use crate::types::AttestationReport;

/// Cache key for nonce-less attestation reports.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct AttestationCacheKey {
    signing_algo: String,
    include_tls_fingerprint: bool,
}

struct CachedReport {
    /// Pre-serialized JSON bytes of the full AttestationResponse.
    /// Avoids re-serializing 297KB on every cache hit.
    response_bytes: bytes::Bytes,
    /// The report struct, needed for background refresh to build new responses.
    report: AttestationReport,
    created_at: Instant,
}

/// Persistent Python worker process for GPU evidence collection.
///
/// Keeps the Python interpreter, verifier module imports, and NVML driver
/// initialized across requests, avoiding ~0.5-2s startup overhead per call.
/// Communication is via JSON lines over stdin/stdout pipes.
///
/// The worker is automatically restarted if it dies. All access is serialized
/// by the gpu_semaphore in AttestationCache (only one evidence collection at a time).
struct GpuEvidenceWorker {
    stdin: tokio::process::ChildStdin,
    stdout: BufReader<tokio::process::ChildStdout>,
    child: tokio::process::Child,
}

/// Path to the worker script, resolved relative to the binary.
fn worker_script_path() -> String {
    // In Docker: /app/gpu_evidence_worker.py (next to /app/vllm-proxy-rs)
    // In dev: ./gpu_evidence_worker.py
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    if let Some(dir) = exe_dir {
        let candidate = dir.join("gpu_evidence_worker.py");
        if candidate.exists() {
            return candidate.to_string_lossy().to_string();
        }
    }
    // Fallback: current directory or CARGO_MANIFEST_DIR for dev
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidate = std::path::Path::new(&manifest).join("gpu_evidence_worker.py");
        if candidate.exists() {
            return candidate.to_string_lossy().to_string();
        }
    }
    "gpu_evidence_worker.py".to_string()
}

impl GpuEvidenceWorker {
    /// Spawn a new persistent Python worker process.
    async fn spawn() -> anyhow::Result<Self> {
        let script_path = worker_script_path();
        info!(script = %script_path, "Spawning GPU evidence worker");

        let mut child = tokio::process::Command::new("python3")
            .arg(&script_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn GPU evidence worker: {e}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture worker stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture worker stdout"))?;
        let mut stdout = BufReader::new(stdout);

        // Wait for the ready signal (first line of output).
        let mut ready_line = String::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(30),
            stdout.read_line(&mut ready_line),
        )
        .await
        .map_err(|_| anyhow::anyhow!("GPU evidence worker did not send ready signal within 30s"))?
        .map_err(|e| anyhow::anyhow!("Failed to read worker ready signal: {e}"))?;

        let ready: serde_json::Value = serde_json::from_str(ready_line.trim())
            .map_err(|e| anyhow::anyhow!("Worker ready signal is not valid JSON: {e}"))?;

        if ready.get("ready") != Some(&serde_json::Value::Bool(true)) {
            anyhow::bail!("Worker sent unexpected ready signal: {}", ready_line.trim());
        }

        let import_ok = ready
            .get("import_ok")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !import_ok {
            let err = ready
                .get("import_error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            warn!(error = %err, "GPU evidence worker started but verifier import failed");
        } else {
            info!("GPU evidence worker ready");
        }

        Ok(Self {
            stdin,
            stdout,
            child,
        })
    }

    /// Send a nonce to the worker and read back GPU evidence.
    async fn collect(
        &mut self,
        nonce_hex: &str,
        no_gpu_mode: bool,
    ) -> anyhow::Result<serde_json::Value> {
        let request = serde_json::json!({
            "nonce": nonce_hex,
            "no_gpu_mode": no_gpu_mode,
        });
        let mut request_line = serde_json::to_string(&request)?;
        request_line.push('\n');

        // Write request
        self.stdin
            .write_all(request_line.as_bytes())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to write to GPU evidence worker: {e}"))?;
        self.stdin
            .flush()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to flush GPU evidence worker stdin: {e}"))?;

        // Read response (with timeout)
        let mut response_line = String::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(60),
            self.stdout.read_line(&mut response_line),
        )
        .await
        .map_err(|_| anyhow::anyhow!("GPU evidence worker timed out after 60s"))?
        .map_err(|e| anyhow::anyhow!("Failed to read from GPU evidence worker: {e}"))?;

        if response_line.is_empty() {
            anyhow::bail!("GPU evidence worker closed stdout (process may have died)");
        }

        let response: serde_json::Value = serde_json::from_str(response_line.trim())
            .map_err(|e| anyhow::anyhow!("Worker response is not valid JSON: {e}"))?;

        if response.get("ok") == Some(&serde_json::Value::Bool(true)) {
            response
                .get("evidence")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Worker response missing 'evidence' field"))
        } else {
            let err = response
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("GPU evidence worker error: {err}")
        }
    }

    /// Check if the worker process is still alive.
    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

/// Caches nonce-less attestation reports and serializes GPU evidence collection.
///
/// GPU evidence collection uses a persistent Python worker process that keeps
/// the verifier module and NVML driver initialized. This cache:
/// 1. Serves pre-generated reports for requests without a nonce (the common case).
/// 2. Serializes evidence calls so only one `nvmlInit`-using request runs at a time.
/// 3. Retries once on GPU evidence failure (restarting the worker if needed).
pub struct AttestationCache {
    /// Cached reports keyed by (signing_algo, include_tls_fingerprint).
    reports: RwLock<HashMap<AttestationCacheKey, CachedReport>>,
    /// Cache TTL in seconds.
    ttl_secs: u64,
    /// Cached dstack info (static for the lifetime of the process).
    dstack_info: OnceCell<serde_json::Value>,
    /// Persistent GPU evidence worker process. Protected by Mutex which also
    /// serializes GPU evidence calls (only one NVML call at a time).
    /// The outer Option is None until first use; the worker is lazily spawned.
    gpu_worker: Mutex<Option<GpuEvidenceWorker>>,
}

impl AttestationCache {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            reports: RwLock::new(HashMap::new()),
            ttl_secs,
            dstack_info: OnceCell::new(),
            gpu_worker: Mutex::new(None),
        }
    }

    /// Get cached dstack info, fetching it once on first call.
    async fn get_dstack_info(&self) -> anyhow::Result<serde_json::Value> {
        self.dstack_info
            .get_or_try_init(|| async {
                let client = dstack_sdk::dstack_client::DstackClient::new(None);
                let info = client.info().await?;
                serde_json::to_value(&info).map_err(anyhow::Error::from)
            })
            .await
            .cloned()
    }

    /// Collect GPU evidence using the persistent worker, with auto-restart.
    ///
    /// Caller must hold the gpu_semaphore permit.
    async fn collect_gpu_evidence(
        &self,
        nonce_hex: &str,
        no_gpu_mode: bool,
    ) -> anyhow::Result<serde_json::Value> {
        let mut worker_guard = self.gpu_worker.lock().await;

        // Ensure we have a live worker
        let needs_spawn = match worker_guard.as_mut() {
            Some(w) => !w.is_alive(),
            None => true,
        };
        if needs_spawn {
            match GpuEvidenceWorker::spawn().await {
                Ok(w) => {
                    *worker_guard = Some(w);
                }
                Err(e) => {
                    warn!(error = %e, "Failed to spawn GPU evidence worker, falling back to subprocess");
                    *worker_guard = None;
                    // Fall back to one-shot subprocess
                    return collect_gpu_evidence_subprocess(nonce_hex, no_gpu_mode).await;
                }
            }
        }

        let worker = worker_guard.as_mut().unwrap();
        match worker.collect(nonce_hex, no_gpu_mode).await {
            Ok(evidence) => Ok(evidence),
            Err(first_err) => {
                warn!(error = %first_err, "GPU evidence worker failed, restarting and retrying");
                metrics::counter!("gpu_evidence_retries_total").increment(1);

                // Kill old worker, spawn fresh one and retry
                *worker_guard = None;
                match GpuEvidenceWorker::spawn().await {
                    Ok(mut new_worker) => match new_worker.collect(nonce_hex, no_gpu_mode).await {
                        Ok(evidence) => {
                            *worker_guard = Some(new_worker);
                            Ok(evidence)
                        }
                        Err(retry_err) => {
                            warn!(error = %retry_err, "Worker retry also failed, falling back to subprocess");
                            *worker_guard = None;
                            collect_gpu_evidence_subprocess(nonce_hex, no_gpu_mode).await
                        }
                    },
                    Err(spawn_err) => {
                        warn!(error = %spawn_err, "Worker restart failed, falling back to subprocess");
                        collect_gpu_evidence_subprocess(nonce_hex, no_gpu_mode).await
                    }
                }
            }
        }
    }

    /// Get pre-serialized JSON bytes for a cached report, if fresh.
    pub async fn get_bytes(
        &self,
        signing_algo: &str,
        include_tls_fingerprint: bool,
    ) -> Option<bytes::Bytes> {
        let key = AttestationCacheKey {
            signing_algo: signing_algo.to_string(),
            include_tls_fingerprint,
        };
        let reports = self.reports.read().await;
        if let Some(cached) = reports.get(&key) {
            if cached.created_at.elapsed().as_secs() < self.ttl_secs {
                metrics::counter!("attestation_cache_hits_total").increment(1);
                return Some(cached.response_bytes.clone());
            }
        }
        metrics::counter!("attestation_cache_misses_total").increment(1);
        None
    }

    /// Get a cached report struct if it exists and is fresh.
    /// Used by background refresh to check if a refresh is needed.
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
                return Some(cached.report.clone());
            }
        }
        None
    }

    /// Store a report in the cache, pre-serializing to JSON bytes.
    pub async fn set(
        &self,
        signing_algo: &str,
        include_tls_fingerprint: bool,
        report: AttestationReport,
        compose_manager_attestation: Option<serde_json::Value>,
    ) {
        let response = crate::types::AttestationResponse {
            report: report.clone(),
            all_attestations: vec![report.clone()],
            compose_manager_attestation,
        };
        let response_bytes = match serde_json::to_vec(&response) {
            Ok(bytes) => bytes::Bytes::from(bytes),
            Err(e) => {
                error!(error = %e, "Failed to serialize attestation response for cache");
                return;
            }
        };
        let key = AttestationCacheKey {
            signing_algo: signing_algo.to_string(),
            include_tls_fingerprint,
        };
        let mut reports = self.reports.write().await;
        reports.insert(
            key,
            CachedReport {
                response_bytes,
                report,
                created_at: Instant::now(),
            },
        );
    }
}

/// Fetch compose-manager attestation report from the given URL.
///
/// Returns `None` on any error (timeout, connection refused, bad JSON, etc.)
/// so that compose-manager unavailability never blocks inference attestation.
pub async fn fetch_compose_manager_attestation(
    http_client: &reqwest::Client,
    compose_manager_url: &str,
    nonce: Option<&str>,
) -> Option<serde_json::Value> {
    let mut url = match reqwest::Url::parse(compose_manager_url) {
        Ok(u) => u,
        Err(e) => {
            warn!(error = %e, "Invalid compose-manager base URL");
            return None;
        }
    };
    url.set_path("/v1/attestation/report");
    if let Some(nonce) = nonce {
        url.query_pairs_mut().append_pair("nonce", nonce);
    }
    match http_client
        .get(url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => match resp.json::<serde_json::Value>().await {
            Ok(val) => Some(val),
            Err(e) => {
                warn!(error = %e, "Failed to parse compose-manager attestation response");
                None
            }
        },
        Ok(resp) => {
            warn!(status = %resp.status(), "Compose-manager attestation returned non-success status");
            None
        }
        Err(e) => {
            warn!(error = %e, "Failed to fetch compose-manager attestation");
            None
        }
    }
}

/// Compose-manager connection info for fetching deployment attestation.
pub struct ComposeManagerConfig {
    pub http_client: reqwest::Client,
    pub url: String,
}

/// Spawn a background task that periodically refreshes cached attestation reports.
pub fn spawn_cache_refresh_task(
    cache: Arc<AttestationCache>,
    model_name: String,
    signing: Arc<crate::signing::SigningPair>,
    gpu_no_hw_mode: bool,
    tls_cert_fingerprint: Option<String>,
    refresh_interval_secs: u64,
    compose_manager: Option<ComposeManagerConfig>,
) {
    tokio::spawn(async move {
        // Initial delay to let the server start up.
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        loop {
            // Fetch compose-manager attestation once per refresh cycle (shared across algos).
            let cm_attestation = if let Some(ref cm) = compose_manager {
                fetch_compose_manager_attestation(&cm.http_client, &cm.url, None).await
            } else {
                None
            };

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
                // GPU evidence serialization is handled by the worker Mutex.
                match generate_attestation_inner(
                    AttestationParams {
                        model_name: &model_name,
                        signing_address: &signing_address,
                        signing_algo: algo,
                        signing_public_key: &signing_public_key,
                        signing_address_bytes: &signing_address_bytes,
                        nonce: None,
                        gpu_no_hw_mode,
                        tls_cert_fingerprint: None,
                    },
                    Some(&cache),
                )
                .await
                {
                    Ok(report) => {
                        cache.set(algo, false, report, cm_attestation.clone()).await;
                        info!(algo, "Background attestation cache refresh succeeded");
                    }
                    Err(e) => {
                        warn!(algo, error = %e, "Background attestation cache refresh failed");
                    }
                }

                // Also refresh with TLS fingerprint if configured.
                if let Some(ref fp) = tls_cert_fingerprint {
                    match generate_attestation_inner(
                        AttestationParams {
                            model_name: &model_name,
                            signing_address: &signing_address,
                            signing_algo: algo,
                            signing_public_key: &signing_public_key,
                            signing_address_bytes: &signing_address_bytes,
                            nonce: None,
                            gpu_no_hw_mode,
                            tls_cert_fingerprint: Some(fp.as_str()),
                        },
                        Some(&cache),
                    )
                    .await
                    {
                        Ok(report) => {
                            cache.set(algo, true, report, cm_attestation.clone()).await;
                        }
                        Err(e) => {
                            warn!(algo, error = %e, "Background attestation cache refresh (with TLS) failed");
                        }
                    }
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

/// Fallback: collect GPU evidence via one-shot Python subprocess.
///
/// Used when the persistent worker cannot be spawned (e.g., script not found,
/// Python not installed). Slower due to Python startup + module import overhead.
async fn collect_gpu_evidence_subprocess(
    nonce_hex: &str,
    no_gpu_mode: bool,
) -> anyhow::Result<serde_json::Value> {
    if no_gpu_mode {
        info!("GPU evidence no-GPU mode enabled; using canned evidence");
    }

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
///
/// Parallelizes the two slow operations:
/// - TDX quote generation (dstack Unix socket RPC)
/// - GPU evidence collection (Python subprocess with NVML)
///
/// dstack info is cached for the process lifetime (it never changes).
async fn generate_attestation_inner(
    params: AttestationParams<'_>,
    cache: Option<&AttestationCache>,
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

    // Run TDX quote and GPU evidence collection in parallel.
    // These are independent: TDX quote talks to dstack via Unix socket,
    // GPU evidence uses the persistent Python worker (or subprocess fallback).
    let gpu_no_hw_mode = params.gpu_no_hw_mode;
    let nonce_hex_clone = nonce_hex.clone();
    let (quote_result, gpu_evidence) = tokio::try_join!(
        async {
            let client = dstack_sdk::dstack_client::DstackClient::new(None);
            client
                .get_quote(report_data)
                .await
                .map_err(AttestationError::Internal)
        },
        async {
            if let Some(cache) = cache {
                cache
                    .collect_gpu_evidence(&nonce_hex_clone, gpu_no_hw_mode)
                    .await
                    .map_err(AttestationError::Internal)
            } else {
                collect_gpu_evidence_subprocess(&nonce_hex_clone, gpu_no_hw_mode)
                    .await
                    .map_err(AttestationError::Internal)
            }
        },
    )?;

    let event_log: serde_json::Value = serde_json::from_str(&quote_result.event_log)
        .map_err(|e| AttestationError::Internal(anyhow::Error::from(e)))?;
    let nvidia_payload = build_nvidia_payload(&nonce_hex, &gpu_evidence);

    // dstack info is static — use cached value if available.
    let info_value = if let Some(cache) = cache {
        cache
            .get_dstack_info()
            .await
            .map_err(AttestationError::Internal)?
    } else {
        let client = dstack_sdk::dstack_client::DstackClient::new(None);
        let info = client.info().await.map_err(AttestationError::Internal)?;
        serde_json::to_value(&info)
            .map_err(|e| AttestationError::Internal(anyhow::Error::from(e)))?
    };

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

/// Result of attestation generation — either pre-serialized cached bytes or a fresh report.
pub enum AttestationResult {
    /// Cache hit: pre-serialized JSON bytes ready to send.
    CachedBytes(bytes::Bytes),
    /// Fresh report that needs serialization.
    Fresh(Box<AttestationReport>),
}

/// Generate an attestation report, using the cache for nonce-less requests.
///
/// When a caller provides a nonce, the GPU evidence and TDX quote are
/// cryptographically bound to that nonce, so we must generate fresh.
/// When no nonce is provided, we serve a cached report (which contains its
/// own randomly-generated nonce) — the caller accepts whatever nonce we return.
///
/// GPU evidence collection is serialized by the worker Mutex (NVML constraint),
/// but TDX quotes and dstack info calls run concurrently with other requests.
pub async fn generate_attestation(
    params: AttestationParams<'_>,
    cache: Option<&AttestationCache>,
) -> Result<AttestationResult, AttestationError> {
    let is_nonceless = params.nonce.is_none();
    let include_tls = params.tls_cert_fingerprint.is_some();
    let signing_algo = params.signing_algo.to_string();

    // For nonce-less requests, try the cache first (returns pre-serialized bytes).
    if is_nonceless {
        if let Some(cache) = cache {
            if let Some(bytes) = cache.get_bytes(&signing_algo, include_tls).await {
                return Ok(AttestationResult::CachedBytes(bytes));
            }
        }
    }

    // Generate fresh report. GPU evidence is serialized by the worker Mutex,
    // but TDX quote runs concurrently.
    let report = generate_attestation_inner(params, cache).await?;
    // Don't cache here — the caller (route handler) caches after fetching
    // compose-manager attestation so cached responses include the full chain.

    Ok(AttestationResult::Fresh(Box::new(report)))
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
        cache.set("ecdsa", false, report.clone(), None).await;

        let result = cache.get("ecdsa", false).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().request_nonce, "aabb");
    }

    #[tokio::test]
    async fn test_attestation_cache_miss_different_algo() {
        let cache = AttestationCache::new(300);
        cache
            .set("ecdsa", false, make_test_report("ecdsa", "aa"), None)
            .await;

        assert!(cache.get("ed25519", false).await.is_none());
    }

    #[tokio::test]
    async fn test_attestation_cache_miss_different_tls() {
        let cache = AttestationCache::new(300);
        cache
            .set("ecdsa", false, make_test_report("ecdsa", "aa"), None)
            .await;

        assert!(cache.get("ecdsa", true).await.is_none());
    }

    #[tokio::test]
    async fn test_attestation_cache_ttl_expiry() {
        let cache = AttestationCache::new(1);
        cache
            .set("ecdsa", false, make_test_report("ecdsa", "aa"), None)
            .await;

        assert!(cache.get("ecdsa", false).await.is_some());
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        assert!(cache.get("ecdsa", false).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_get_bytes_returns_preserialized() {
        let cache = AttestationCache::new(300);
        let report = make_test_report("ecdsa", "aabb");
        cache.set("ecdsa", false, report, None).await;

        let bytes = cache.get_bytes("ecdsa", false).await;
        assert!(bytes.is_some());
        let parsed: serde_json::Value =
            serde_json::from_slice(&bytes.unwrap()).expect("cached bytes should be valid JSON");
        assert_eq!(parsed["request_nonce"], "aabb");
        assert!(parsed["all_attestations"].is_array());
    }

    #[tokio::test]
    async fn test_cache_includes_compose_manager_attestation() {
        let cache = AttestationCache::new(300);
        let report = make_test_report("ecdsa", "aabb");
        let cm_attestation = serde_json::json!({
            "actions": [{"action": "compose_up", "tag": "v1.0"}],
            "actions_hash": "deadbeef",
            "quote": "some_tdx_quote"
        });
        cache
            .set("ecdsa", false, report, Some(cm_attestation.clone()))
            .await;

        // Verify the struct getter doesn't include compose-manager attestation
        // (it only returns AttestationReport, not the full response)
        let result = cache.get("ecdsa", false).await;
        assert!(result.is_some());

        // Verify pre-serialized bytes include compose_manager_attestation
        let bytes = cache.get_bytes("ecdsa", false).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed["compose_manager_attestation"], cm_attestation);
    }

    #[tokio::test]
    async fn test_cache_omits_compose_manager_attestation_when_none() {
        let cache = AttestationCache::new(300);
        let report = make_test_report("ecdsa", "aabb");
        cache.set("ecdsa", false, report, None).await;

        let bytes = cache.get_bytes("ecdsa", false).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(parsed.get("compose_manager_attestation").is_none());
    }
}

#[cfg(test)]
mod tests_fetch_compose_manager {
    use super::*;

    #[tokio::test]
    async fn test_fetch_success() {
        let mock = wiremock::MockServer::start().await;
        let cm_response = serde_json::json!({
            "actions": [],
            "actions_hash": "abc123",
            "nonce": "def456",
            "quote": "tdx_quote_data"
        });
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/attestation/report"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(&cm_response))
            .mount(&mock)
            .await;

        let client = reqwest::Client::new();
        let result = fetch_compose_manager_attestation(&client, &mock.uri(), None).await;
        assert_eq!(result, Some(cm_response));
    }

    #[tokio::test]
    async fn test_fetch_passes_nonce() {
        let mock = wiremock::MockServer::start().await;
        let nonce = "aa".repeat(32);
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/attestation/report"))
            .and(wiremock::matchers::query_param("nonce", &nonce))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"nonce": &nonce})),
            )
            .mount(&mock)
            .await;

        let client = reqwest::Client::new();
        let result = fetch_compose_manager_attestation(&client, &mock.uri(), Some(&nonce)).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap()["nonce"], nonce);
    }

    #[tokio::test]
    async fn test_fetch_returns_none_on_server_error() {
        let mock = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/attestation/report"))
            .respond_with(wiremock::ResponseTemplate::new(503))
            .mount(&mock)
            .await;

        let client = reqwest::Client::new();
        let result = fetch_compose_manager_attestation(&client, &mock.uri(), None).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_fetch_returns_none_on_connection_refused() {
        // Bind to an ephemeral port, then drop the listener to guarantee nothing is listening.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");
        drop(listener);

        let client = reqwest::Client::new();
        let result = fetch_compose_manager_attestation(&client, &base_url, None).await;
        assert!(result.is_none());
    }
}
