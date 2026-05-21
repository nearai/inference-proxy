use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, OnceCell, RwLock};
use tracing::{error, info, warn};

use crate::types::AttestationReport;

/// Per-call timeout for the Python evidence collector (persistent worker
/// `collect()` and one-shot subprocess fallback).
///
/// Sized to fail before cloud-api's 30s HTTP timeout: when NVML hangs on
/// a contended shared-GPU host (the production race that produces
/// `NONCE_NOT_MATCHING` and outright hangs alike), we want to give up
/// quickly so the existing PR #51 cache-layer retry can spawn a fresh
/// worker and answer the same request. A 60s timeout was strictly worse
/// — cloud-api had already disconnected by the time we recovered, and
/// cloud-api saw a transport-level "error sending request for url"
/// instead of the structured 5xx we'd otherwise return.
const GPU_EVIDENCE_COLLECTION_TIMEOUT_SECS: u64 = 20;

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

        // Read response (with timeout). See GPU_EVIDENCE_COLLECTION_TIMEOUT_SECS
        // for why this is sized to fail before cloud-api's HTTP timeout.
        let mut response_line = String::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(GPU_EVIDENCE_COLLECTION_TIMEOUT_SECS),
            self.stdout.read_line(&mut response_line),
        )
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "GPU evidence worker timed out after {GPU_EVIDENCE_COLLECTION_TIMEOUT_SECS}s"
            )
        })?
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

    /// Drop every cached report. Called after the TLS cert rotates so the
    /// next request regenerates a TDX quote bound to the new fingerprint
    /// rather than serving a stale, soon-to-fail report from cache.
    pub async fn clear_all(&self) {
        let mut reports = self.reports.write().await;
        let n = reports.len();
        reports.clear();
        drop(reports);
        info!(invalidated_entries = n, "Attestation cache cleared");
    }

    /// Store a report in the cache, pre-serializing to JSON bytes.
    pub async fn set(
        &self,
        signing_algo: &str,
        include_tls_fingerprint: bool,
        report: AttestationReport,
        compose_manager_attestation: Option<Box<serde_json::value::RawValue>>,
        ohttp_attestation: Option<crate::types::OhttpAttestation>,
    ) {
        let response = crate::types::AttestationResponse::new(
            report.clone(),
            vec![report.clone()],
            compose_manager_attestation,
            ohttp_attestation,
        );
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
/// Returns the raw response body as a `RawValue` so the bytes compose-manager
/// signed are forwarded verbatim. This preserves the `sha256(actions) ==
/// actions_hash` binding for downstream verifiers — re-parsing through
/// `serde_json::Value` would reorder object keys alphabetically and break it.
///
/// Returns `None` on any error (timeout, connection refused, bad JSON, etc.)
/// so that compose-manager unavailability never blocks inference attestation.
pub async fn fetch_compose_manager_attestation(
    http_client: &reqwest::Client,
    compose_manager_url: &str,
    nonce: Option<&str>,
) -> Option<Box<serde_json::value::RawValue>> {
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
        Ok(resp) if resp.status().is_success() => match resp.text().await {
            Ok(body) => match serde_json::value::RawValue::from_string(body) {
                Ok(raw) => Some(raw),
                Err(e) => {
                    warn!(error = %e, "Compose-manager attestation response was not valid JSON");
                    None
                }
            },
            Err(e) => {
                warn!(error = %e, "Failed to read compose-manager attestation response body");
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

/// Owned-lifetime version of `DelegateContext` used by the background
/// cache refresh task (which doesn't have access to the request-scoped
/// `&Config`/`&Client`). Holds a clone of the `reqwest::Client` and an
/// `Arc<Config>` so the spawned task is `'static`.
pub struct DelegateRefreshConfig {
    pub config: Arc<crate::config::Config>,
    pub http_client: reqwest::Client,
}

/// Build OHTTP attestation payload for the process-wide OHTTP gateway config.
pub fn build_ohttp_attestation(
    signing: &crate::signing::SigningPair,
    gateway: &crate::ohttp_gateway::OhttpGateway,
) -> anyhow::Result<crate::types::OhttpAttestation> {
    let key_config = hex::encode(gateway.config_bytes());
    let signature = signing
        .ed25519
        .sign_bytes(gateway.config_bytes())
        .map_err(|e| anyhow::anyhow!("failed to sign OHTTP attestation: {e}"))?;
    Ok(crate::types::OhttpAttestation {
        signing_algo: "ed25519".to_string(),
        signing_key: signing.ed25519.signing_public_key.clone(),
        key_config,
        signature,
    })
}

/// Spawn a background task that periodically refreshes cached attestation reports.
#[allow(clippy::too_many_arguments)]
pub fn spawn_cache_refresh_task(
    cache: Arc<AttestationCache>,
    model_name: String,
    signing: Arc<crate::signing::SigningPair>,
    gpu_no_hw_mode: bool,
    tls_cert_tracker: Arc<TlsCertTracker>,
    refresh_interval_secs: u64,
    compose_manager: Option<ComposeManagerConfig>,
    ohttp_attestation_ed25519: Option<crate::types::OhttpAttestation>,
    delegate_refresh: Option<DelegateRefreshConfig>,
) {
    tokio::spawn(async move {
        // Initial delay to let the server start up.
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        loop {
            // Reconcile our cached TLS-cert SPKI with whatever the ingress
            // sidecar wrote to disk. If certbot just renewed the cert, the
            // tracker swaps to the new hash and the cache is cleared so the
            // next request regenerates a TDX quote bound to it.
            if tls_cert_tracker.refresh_if_changed().is_some() {
                cache.clear_all().await;
            }

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
                let delegate_ctx = delegate_refresh.as_ref().map(|d| DelegateContext {
                    config: &d.config,
                    http_client: &d.http_client,
                });
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
                    delegate_ctx.as_ref(),
                )
                .await
                {
                    Ok(report) => {
                        cache
                            .set(
                                algo,
                                false,
                                report,
                                cm_attestation.clone(),
                                ohttp_attestation_ed25519.clone(),
                            )
                            .await;
                        info!(algo, "Background attestation cache refresh succeeded");
                    }
                    Err(e) => {
                        warn!(algo, error = %e, "Background attestation cache refresh failed");
                    }
                }

                // Also refresh with TLS fingerprint if configured. Read the
                // current value through the tracker — it may have just been
                // updated by `refresh_if_changed` above.
                if let Some(ref fp) = tls_cert_tracker.current() {
                    let delegate_ctx = delegate_refresh.as_ref().map(|d| DelegateContext {
                        config: &d.config,
                        http_client: &d.http_client,
                    });
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
                        delegate_ctx.as_ref(),
                    )
                    .await
                    {
                        Ok(report) => {
                            cache
                                .set(
                                    algo,
                                    true,
                                    report,
                                    cm_attestation.clone(),
                                    ohttp_attestation_ed25519.clone(),
                                )
                                .await;
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
        None => Ok(rand::random()),
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
        std::time::Duration::from_secs(GPU_EVIDENCE_COLLECTION_TIMEOUT_SECS),
        tokio::process::Command::new("python3")
            .arg("-c")
            .arg(&script)
            .output(),
    )
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "GPU evidence subprocess timed out after {GPU_EVIDENCE_COLLECTION_TIMEOUT_SECS}s"
        )
    })?
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

/// Tracks the SHA-256 SPKI hash of the local TLS certificate and refreshes it
/// when the file on disk changes.
///
/// We previously snapshotted the fingerprint at startup and reused it for the
/// lifetime of the process. That broke whenever certbot (running in the
/// co-located `cvm-ingress` sidecar) renewed the cert — nginx picked up the
/// new cert immediately via `nginx -s reload`, but the inference-proxy kept
/// reporting the pre-renewal SPKI in its attestation. Cloud-api pinned the
/// stale fingerprint, then rejected the actual (post-renewal) cert at TLS
/// handshake time, producing the 503-on-attestation-report bursts we hunted
/// down on staging.
///
/// The tracker is consulted on every request via [`Self::current`] (cheap
/// `RwLock` read) and reconciled with the on-disk cert via
/// [`Self::refresh_if_changed`] inside the background attestation refresh
/// task. We use `mtime` as a coarse change signal because the renewal daemon
/// rewrites the file atomically, so any actual rotation advances `mtime`. A
/// touch that doesn't change the content is rehashed but produces no
/// fingerprint change and triggers no cache invalidation.
pub struct TlsCertTracker {
    /// Configured cert path. `None` when `TLS_CERT_PATH` is unset, in which
    /// case the tracker is permanently inert.
    cert_path: Option<String>,
    inner: std::sync::RwLock<TlsCertTrackerState>,
}

#[derive(Debug)]
struct TlsCertTrackerState {
    /// Latest hash that the rest of the process should report. `None` when
    /// no cert path is configured.
    fingerprint: Option<String>,
    /// Last observed mtime; used as the cheap "did anything change?" gate.
    last_mtime: Option<std::time::SystemTime>,
}

impl TlsCertTracker {
    /// Build a tracker, seeding the fingerprint and `mtime` from disk.
    ///
    /// Returns `Err` only when `cert_path` is configured but the initial read
    /// fails — preserving the existing startup contract that an unreadable
    /// `TLS_CERT_PATH` is fatal.
    pub fn new(cert_path: Option<String>) -> anyhow::Result<Self> {
        let inner = match &cert_path {
            Some(path) => {
                let last_mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
                let fingerprint = Some(compute_spki_hash(path)?);
                info!(
                    tls_cert_path = %path,
                    fingerprint = %fingerprint.as_deref().unwrap_or(""),
                    "TLS certificate SPKI hash computed"
                );
                TlsCertTrackerState {
                    fingerprint,
                    last_mtime,
                }
            }
            None => TlsCertTrackerState {
                fingerprint: None,
                last_mtime: None,
            },
        };
        Ok(Self {
            cert_path,
            inner: std::sync::RwLock::new(inner),
        })
    }

    /// Current fingerprint (cloned). Hot path; called on every attestation.
    pub fn current(&self) -> Option<String> {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .fingerprint
            .clone()
    }

    /// Re-stat the cert file. If its `mtime` advanced, re-compute the SPKI
    /// hash; if the hash actually changed, swap it into the tracker.
    ///
    /// Returns `Some(new_fingerprint)` when the fingerprint changed, `None`
    /// otherwise (no path configured, file unchanged, file unreadable, or the
    /// content was touched but produced an identical hash). The boolean
    /// distinction matters to the caller: a non-`None` return is the signal
    /// to invalidate the attestation cache so the next request regenerates
    /// a TDX quote bound to the new fingerprint.
    pub fn refresh_if_changed(&self) -> Option<String> {
        let path = self.cert_path.as_deref()?;

        let new_mtime = match std::fs::metadata(path).and_then(|m| m.modified()) {
            Ok(t) => Some(t),
            Err(e) => {
                warn!(
                    tls_cert_path = %path,
                    error = %e,
                    "failed to stat TLS cert file, keeping cached fingerprint"
                );
                return None;
            }
        };

        let old_mtime = self
            .inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .last_mtime;
        if new_mtime == old_mtime {
            return None;
        }

        let new_fp = match compute_spki_hash(path) {
            Ok(fp) => fp,
            Err(e) => {
                warn!(
                    tls_cert_path = %path,
                    error = %e,
                    "failed to re-hash TLS cert after rotation, keeping cached fingerprint"
                );
                return None;
            }
        };

        let mut state = self.inner.write().unwrap_or_else(|e| e.into_inner());
        let changed = state.fingerprint.as_deref() != Some(new_fp.as_str());
        state.fingerprint = Some(new_fp.clone());
        state.last_mtime = new_mtime;
        drop(state);

        if changed {
            info!(
                tls_cert_path = %path,
                fingerprint = %new_fp,
                "TLS cert SPKI fingerprint updated after on-disk rotation"
            );
            Some(new_fp)
        } else {
            // mtime advanced but content was the same (touch-only). Don't
            // bother invalidating the cache.
            None
        }
    }
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

/// Context the delegate-dispatch path needs at the call site.
///
/// Carries the resolved `Config` (for the delegate URL/timeout/auth
/// token) and the shared `reqwest::Client` we use across the proxy.
/// Lifetime-bound to the caller's `AppState` so we don't clone the
/// client per request.
pub struct DelegateContext<'a> {
    pub config: &'a crate::config::Config,
    pub http_client: &'a reqwest::Client,
}

/// Maximum attempts for `collect_gpu_evidence_with_nonce_check`.
///
/// 4 attempts (1 initial + 3 retries) with exponential backoff between
/// them. The race appears to be at the NVML/GPU-firmware level —
/// transient, but with enough variance that a single retry isn't
/// always enough. Worst-case wait time across all retries is bounded
/// by `GPU_EVIDENCE_NONCE_BACKOFF_BASE_MS * (2^0 + 2^1 + 2^2)` plus
/// the four collection latencies.
const GPU_EVIDENCE_NONCE_MAX_ATTEMPTS: usize = 4;

/// Initial backoff before the second attempt. Doubles before each
/// subsequent retry: 100ms, 200ms, 400ms — total worst-case wait 700ms.
/// Short enough to stay under cloud-api's per-request timeout while
/// giving the firmware time to settle if the cause is contention.
const GPU_EVIDENCE_NONCE_BACKOFF_BASE_MS: u64 = 100;

/// SPDM-style request opcode header at the start of every per-GPU
/// attestation report binary, observed across PASS and FAIL responses
/// captured from production.
const GPU_EVIDENCE_HEADER: [u8; 4] = [0x11, 0xE0, 0x01, 0xFF];

/// Byte range where the per-GPU attestation report binary embeds the
/// caller-provided nonce. Verified against captured production responses
/// (working glm-5 capture and a tampered-evidence FAIL capture both have
/// the request nonce at offset 4..36 of the base64-decoded `evidence`
/// field).
const GPU_EVIDENCE_NONCE_OFFSET: usize = 4;
const GPU_EVIDENCE_NONCE_LEN: usize = 32;

/// Decode a single per-GPU `evidence` (base64) and check that the
/// 32-byte slice at offset 4..36 equals the caller's nonce.
///
/// Returns `false` if the field is missing, not base64, too short, or
/// the nonce doesn't match (NRAS rejects all of these as
/// `NONCE_NOT_MATCHING`/`INVALID_EVIDENCE_*`).
///
/// Returns `true` (fail-open) when the evidence has an SPDM header we
/// don't recognise. Verification is best-effort — if NVIDIA bumps the
/// request opcode in a future driver, we'd rather let NRAS render
/// judgment than reject every PASS evidence in our fleet.
fn evidence_has_correct_nonce(
    evidence_entry: &serde_json::Value,
    expected_nonce: &[u8; 32],
) -> bool {
    let Some(b64) = evidence_entry.get("evidence").and_then(|v| v.as_str()) else {
        return false;
    };
    use base64::Engine;
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) else {
        return false;
    };
    if bytes.len() < GPU_EVIDENCE_NONCE_OFFSET + GPU_EVIDENCE_NONCE_LEN {
        return false;
    }
    if bytes[..GPU_EVIDENCE_HEADER.len()] != GPU_EVIDENCE_HEADER {
        // Unknown SPDM revision — fail open (see doc comment above).
        return true;
    }
    bytes[GPU_EVIDENCE_NONCE_OFFSET..GPU_EVIDENCE_NONCE_OFFSET + GPU_EVIDENCE_NONCE_LEN]
        == expected_nonce[..]
}

/// Why a per-GPU nonce binding check failed.
///
/// `NoEvidenceList` covers both "not a JSON array" and "empty array" —
/// either way there's nothing to verify, and bubbling that up as a
/// failure (rather than silently treating it as "all GPUs verified")
/// is what closes the bypass Copilot flagged.
#[derive(Debug, Clone, PartialEq, Eq)]
enum NonceMismatch {
    /// `evidence_list` was missing, not an array, or empty — no per-GPU
    /// evidence to verify. NRAS would reject the resulting payload, and
    /// we don't want a malformed shape to slip past as "verified".
    NoEvidenceList,
    /// A specific GPU's bound nonce doesn't match the request nonce.
    GpuIndex(usize),
}

impl std::fmt::Display for NonceMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NonceMismatch::NoEvidenceList => write!(f, "evidence_list missing/empty/non-array"),
            NonceMismatch::GpuIndex(idx) => write!(f, "GPU index {idx}"),
        }
    }
}

/// Walk every entry in an `evidence_list` and verify that each GPU's
/// embedded nonce matches the request nonce. Returns `Ok(())` only when
/// the list is a non-empty array and every entry verifies; otherwise
/// returns the specific reason.
fn check_evidence_nonce_binding(
    evidences: &serde_json::Value,
    expected_nonce: &[u8; 32],
) -> Result<(), NonceMismatch> {
    let arr = evidences.as_array().ok_or(NonceMismatch::NoEvidenceList)?;
    if arr.is_empty() {
        return Err(NonceMismatch::NoEvidenceList);
    }
    for (idx, entry) in arr.iter().enumerate() {
        if !evidence_has_correct_nonce(entry, expected_nonce) {
            return Err(NonceMismatch::GpuIndex(idx));
        }
    }
    Ok(())
}

/// Collect GPU evidence and verify that the firmware bound the caller's
/// nonce into the signed report. Retries with exponential backoff if any
/// GPU's evidence has the wrong nonce — matching the production failure
/// mode where NRAS rejects with `NONCE_NOT_MATCHING` (error 4010), which
/// we believe is a race in NVML/GPU-firmware when multiple inference-proxy
/// processes on a shared GPU host call the verifier concurrently.
///
/// We deliberately do **not** kill the persistent Python worker on
/// mismatch: the worker is a thin pass-through to `cc_admin` and
/// produces evidence successfully — just with the wrong nonce baked in
/// some of the time. Restarting it would add ~1–2s spawn overhead per
/// retry and briefly drop our in-process serialization, plausibly making
/// the cross-process race worse for whichever request slips in. The
/// backoff (100ms → 200ms → 400ms) is what gives the firmware time to
/// settle.
///
/// Failures (transport errors, repeated nonce mismatches) bubble up so
/// cloud-api can rotate to a different backend instead of submitting
/// known-bad evidence to NRAS.
pub(crate) async fn collect_gpu_evidence_with_nonce_check(
    nonce_hex: &str,
    nonce_bytes: &[u8; 32],
    gpu_no_hw_mode: bool,
    cache: Option<&AttestationCache>,
    delegate_ctx: Option<&DelegateContext<'_>>,
) -> anyhow::Result<serde_json::Value> {
    let mut last_failure: Option<NonceMismatch> = None;

    for attempt in 1..=GPU_EVIDENCE_NONCE_MAX_ATTEMPTS {
        // Backoff before retries (not before the first attempt).
        if attempt > 1 {
            let delay_ms = GPU_EVIDENCE_NONCE_BACKOFF_BASE_MS << (attempt - 2);
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }

        // Four backends, in priority order:
        //   1. delegate proxy (HTTP, opt-in via GPU_EVIDENCE_DELEGATE_URL)
        //      — used to serialize NVML across multiple proxies sharing a
        //      host. Only the delegate touches local NVML.
        //   2. nv-attestation-sdk (Rust → C FFI, opt-in via env var)
        //   3. cache's persistent Python worker (existing default)
        //   4. one-shot Python subprocess (fallback when no cache)
        // The self-check + retry below applies regardless of which one
        // produced the evidence — including evidence returned by the
        // delegate (defense in depth, plus catches the rare "delegate
        // returned 200 but with bytes from a different request").
        let evidence = if let Some(dctx) = delegate_ctx.filter(|_| !gpu_no_hw_mode) {
            // Delegate path. `gpu_no_hw_mode` doesn't make sense across
            // an HTTP hop; fall through to local paths if it's set.
            crate::gpu_evidence_delegate::collect_via_delegate(
                dctx.config,
                dctx.http_client,
                nonce_hex,
                gpu_no_hw_mode,
            )
            .await?
        } else if crate::attestation_sdk::is_active() && !gpu_no_hw_mode {
            // SDK path doesn't support no_gpu_mode (it requires real
            // hardware via NVML); fall back to the Python paths for
            // dev/test environments without GPUs.
            crate::attestation_sdk::collect_gpu_evidence_via_sdk(nonce_hex).await?
        } else if let Some(cache) = cache {
            cache
                .collect_gpu_evidence(nonce_hex, gpu_no_hw_mode)
                .await?
        } else {
            collect_gpu_evidence_subprocess(nonce_hex, gpu_no_hw_mode).await?
        };

        // `no_gpu_mode` returns canned evidence (nv-attestation-sdk fixture
        // for hosts without GPUs); skip the nonce-binding check there since
        // the canned bytes don't carry our nonce.
        if gpu_no_hw_mode {
            return Ok(evidence);
        }

        match check_evidence_nonce_binding(&evidence, nonce_bytes) {
            Ok(()) => return Ok(evidence),
            Err(reason) => {
                metrics::counter!("gpu_evidence_nonce_mismatch_total").increment(1);
                warn!(
                    attempt,
                    max_attempts = GPU_EVIDENCE_NONCE_MAX_ATTEMPTS,
                    failure = %reason,
                    "GPU evidence nonce binding check failed — collector returned evidence whose embedded nonce differs from the request nonce, or the evidence_list shape was unusable"
                );
                last_failure = Some(reason);
            }
        }
    }

    metrics::counter!("gpu_evidence_nonce_mismatch_exhausted_total").increment(1);
    let failure = last_failure
        .map(|r| r.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    anyhow::bail!(
        "GPU evidence nonce binding check failed after {} attempts ({failure})",
        GPU_EVIDENCE_NONCE_MAX_ATTEMPTS
    )
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
    delegate_ctx: Option<&DelegateContext<'_>>,
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
    let nonce_bytes_for_verify = nonce_bytes;
    let (quote_result, gpu_evidence) = tokio::try_join!(
        async {
            let client = dstack_sdk::dstack_client::DstackClient::new(None);
            client
                .get_quote(report_data)
                .await
                .map_err(AttestationError::Internal)
        },
        async {
            collect_gpu_evidence_with_nonce_check(
                &nonce_hex_clone,
                &nonce_bytes_for_verify,
                gpu_no_hw_mode,
                cache,
                delegate_ctx,
            )
            .await
            .map_err(AttestationError::Internal)
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
    delegate_ctx: Option<&DelegateContext<'_>>,
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
    let report = generate_attestation_inner(params, cache, delegate_ctx).await?;
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

    /// Build a synthetic per-GPU evidence binary in the format observed in
    /// captured production responses: 4-byte SPDM header + 32-byte nonce +
    /// padding. Encoded as base64 so it can drop into a fake evidence_list
    /// entry.
    fn fake_evidence_b64(nonce: &[u8; 32]) -> String {
        use base64::Engine;
        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(&GPU_EVIDENCE_HEADER);
        bytes.extend_from_slice(nonce);
        bytes.extend_from_slice(&[0u8; 28]); // tail padding (real evidence has more)
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    }

    #[test]
    fn evidence_has_correct_nonce_accepts_matching_binding() {
        let nonce = [0xABu8; 32];
        let entry = serde_json::json!({"evidence": fake_evidence_b64(&nonce)});
        assert!(evidence_has_correct_nonce(&entry, &nonce));
    }

    #[test]
    fn evidence_has_correct_nonce_rejects_wrong_nonce() {
        // This is the production failure mode: header is correct but
        // bytes 4..36 are some other 32-byte value.
        let request_nonce = [0xABu8; 32];
        let evidence_nonce = [0xCDu8; 32];
        let entry = serde_json::json!({"evidence": fake_evidence_b64(&evidence_nonce)});
        assert!(!evidence_has_correct_nonce(&entry, &request_nonce));
    }

    #[test]
    fn evidence_has_correct_nonce_rejects_missing_or_malformed_evidence() {
        let nonce = [0xABu8; 32];
        // Missing field
        assert!(!evidence_has_correct_nonce(&serde_json::json!({}), &nonce));
        // Not a string
        assert!(!evidence_has_correct_nonce(
            &serde_json::json!({"evidence": 42}),
            &nonce
        ));
        // Not base64
        assert!(!evidence_has_correct_nonce(
            &serde_json::json!({"evidence": "not base64!!"}),
            &nonce
        ));
        // Too short
        use base64::Engine;
        let short = base64::engine::general_purpose::STANDARD.encode([0u8; 10]);
        assert!(!evidence_has_correct_nonce(
            &serde_json::json!({"evidence": short}),
            &nonce
        ));
    }

    #[test]
    fn evidence_has_correct_nonce_passes_unknown_header_through() {
        // If NVIDIA changes the SPDM opcode header in a future driver,
        // we don't want to fail-closed and reject every PASS evidence.
        // The check returns true ("can't verify, assume ok") and lets
        // NRAS make the call.
        let nonce = [0xABu8; 32];
        use base64::Engine;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0x99, 0x88, 0x77, 0x66]); // unknown header
        bytes.extend_from_slice(&[0u8; 32]); // not the expected nonce
        bytes.extend_from_slice(&[0u8; 28]);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        assert!(evidence_has_correct_nonce(
            &serde_json::json!({"evidence": b64}),
            &nonce
        ));
    }

    #[test]
    fn check_evidence_nonce_binding_finds_offending_gpu() {
        let good_nonce = [0xABu8; 32];
        let bad_nonce = [0xCDu8; 32];
        // 4 GPUs: 0,1 ok, 2 mismatched, 3 ok
        let evidences = serde_json::json!([
            {"evidence": fake_evidence_b64(&good_nonce)},
            {"evidence": fake_evidence_b64(&good_nonce)},
            {"evidence": fake_evidence_b64(&bad_nonce)},
            {"evidence": fake_evidence_b64(&good_nonce)},
        ]);
        assert_eq!(
            check_evidence_nonce_binding(&evidences, &good_nonce),
            Err(NonceMismatch::GpuIndex(2))
        );
    }

    #[test]
    fn check_evidence_nonce_binding_returns_ok_when_all_match() {
        let nonce = [0xABu8; 32];
        let evidences = serde_json::json!([
            {"evidence": fake_evidence_b64(&nonce)},
            {"evidence": fake_evidence_b64(&nonce)},
            {"evidence": fake_evidence_b64(&nonce)},
        ]);
        assert_eq!(check_evidence_nonce_binding(&evidences, &nonce), Ok(()));
    }

    #[test]
    fn check_evidence_nonce_binding_rejects_non_array() {
        // Closes the bypass Copilot flagged: a non-array `evidence_list`
        // (or any malformed shape) used to silently propagate `None` and
        // be interpreted as "all GPUs verified". Now it surfaces as a
        // verification failure that drives the retry path.
        let nonce = [0xABu8; 32];
        assert_eq!(
            check_evidence_nonce_binding(&serde_json::json!({}), &nonce),
            Err(NonceMismatch::NoEvidenceList)
        );
        assert_eq!(
            check_evidence_nonce_binding(&serde_json::json!("string"), &nonce),
            Err(NonceMismatch::NoEvidenceList)
        );
        assert_eq!(
            check_evidence_nonce_binding(&serde_json::json!(null), &nonce),
            Err(NonceMismatch::NoEvidenceList)
        );
    }

    #[test]
    fn check_evidence_nonce_binding_rejects_empty_array() {
        let nonce = [0xABu8; 32];
        assert_eq!(
            check_evidence_nonce_binding(&serde_json::json!([]), &nonce),
            Err(NonceMismatch::NoEvidenceList)
        );
    }

    #[test]
    fn nonce_mismatch_display_formats_clearly() {
        // Pinned because the Display impl is what shows up in operator
        // log lines via `failure = %reason` and in the error message
        // bubbled up to cloud-api.
        assert_eq!(
            NonceMismatch::NoEvidenceList.to_string(),
            "evidence_list missing/empty/non-array"
        );
        assert_eq!(NonceMismatch::GpuIndex(3).to_string(), "GPU index 3");
    }

    #[test]
    fn nonce_check_retry_constants_are_consistent() {
        // Pin the retry policy so tweaks are intentional. 4 attempts with
        // 100ms exponential-doubling backoff = 100+200+400 = 700ms wait
        // worst-case, plus four collection latencies.
        assert_eq!(GPU_EVIDENCE_NONCE_MAX_ATTEMPTS, 4);
        assert_eq!(GPU_EVIDENCE_NONCE_BACKOFF_BASE_MS, 100);
        let waits: Vec<u64> = (2..=GPU_EVIDENCE_NONCE_MAX_ATTEMPTS)
            .map(|attempt| GPU_EVIDENCE_NONCE_BACKOFF_BASE_MS << (attempt - 2))
            .collect();
        assert_eq!(waits, vec![100, 200, 400]);
        assert_eq!(waits.iter().sum::<u64>(), 700);
    }

    /// Local-only sanity check against captured production NRAS payloads.
    /// Set `NRAS_PAYLOAD` to a JSON file containing the `nvidia_payload`
    /// (string) value pulled from a real `/v1/attestation/report` response.
    /// Reproduces the offset assumption — bytes 4..36 of each evidence's
    /// base64-decoded `evidence` field carry the request nonce.
    #[test]
    #[ignore]
    fn evidence_nonce_offset_matches_captured_response() {
        let path = match std::env::var("NRAS_PAYLOAD") {
            Ok(v) => v,
            Err(_) => {
                eprintln!("skipped: NRAS_PAYLOAD env var not set");
                return;
            }
        };
        let raw = std::fs::read_to_string(&path).unwrap();
        let payload: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let nonce_hex = payload["nonce"].as_str().unwrap();
        let nonce_bytes: [u8; 32] = hex::decode(nonce_hex).unwrap().try_into().unwrap();
        let mismatch = check_evidence_nonce_binding(&payload["evidence_list"], &nonce_bytes);
        eprintln!("captured: nonce={nonce_hex}");
        eprintln!("captured: mismatch_index={:?}", mismatch);
        // For a captured PASS payload we expect None; for a captured
        // FAIL payload we expect Some(idx). Either way, the function
        // must run cleanly — that's what this test guards.
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
        cache.set("ecdsa", false, report.clone(), None, None).await;

        let result = cache.get("ecdsa", false).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().request_nonce, "aabb");
    }

    #[tokio::test]
    async fn test_attestation_cache_miss_different_algo() {
        let cache = AttestationCache::new(300);
        cache
            .set("ecdsa", false, make_test_report("ecdsa", "aa"), None, None)
            .await;

        assert!(cache.get("ed25519", false).await.is_none());
    }

    #[tokio::test]
    async fn test_attestation_cache_miss_different_tls() {
        let cache = AttestationCache::new(300);
        cache
            .set("ecdsa", false, make_test_report("ecdsa", "aa"), None, None)
            .await;

        assert!(cache.get("ecdsa", true).await.is_none());
    }

    #[tokio::test]
    async fn test_attestation_cache_ttl_expiry() {
        let cache = AttestationCache::new(1);
        cache
            .set("ecdsa", false, make_test_report("ecdsa", "aa"), None, None)
            .await;

        assert!(cache.get("ecdsa", false).await.is_some());
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        assert!(cache.get("ecdsa", false).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_get_bytes_returns_preserialized() {
        let cache = AttestationCache::new(300);
        let report = make_test_report("ecdsa", "aabb");
        cache.set("ecdsa", false, report, None, None).await;

        let bytes = cache.get_bytes("ecdsa", false).await;
        assert!(bytes.is_some());
        let parsed: serde_json::Value =
            serde_json::from_slice(&bytes.unwrap()).expect("cached bytes should be valid JSON");
        assert_eq!(parsed["request_nonce"], "aabb");
        assert!(parsed["all_attestations"].is_array());
    }

    #[tokio::test]
    async fn test_cache_keeps_legacy_ohttp_key_config_alias() {
        let cache = AttestationCache::new(300);
        let report = make_test_report("ecdsa", "aabb");
        let ohttp_attestation = crate::types::OhttpAttestation {
            signing_algo: "ed25519".to_string(),
            signing_key: "11".repeat(32),
            key_config: "aa55".to_string(),
            signature: "cc77".to_string(),
        };
        cache
            .set("ecdsa", false, report, None, Some(ohttp_attestation))
            .await;

        let bytes = cache.get_bytes("ecdsa", false).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed["ohttp_key_config"], "aa55");
        assert_eq!(parsed["ohttp_attestation"]["key_config"], "aa55");
        assert!(parsed["ohttp_attestation"].get("text").is_none());
    }

    #[tokio::test]
    async fn test_cache_includes_compose_manager_attestation() {
        let cache = AttestationCache::new(300);
        let report = make_test_report("ecdsa", "aabb");
        // Use a hand-rolled string with non-alphabetical key order so the
        // assertion below also catches any reordering regression.
        let cm_raw = r#"{"quote":"some_tdx_quote","actions_hash":"deadbeef","actions":[{"action":"compose_up","tag":"v1.0"}]}"#;
        let cm_attestation = serde_json::value::RawValue::from_string(cm_raw.to_string()).unwrap();
        cache
            .set("ecdsa", false, report, Some(cm_attestation), None)
            .await;

        // Verify the struct getter doesn't include compose-manager attestation
        // (it only returns AttestationReport, not the full response)
        let result = cache.get("ecdsa", false).await;
        assert!(result.is_some());

        // Verify pre-serialized bytes preserve the exact bytes (key order included).
        let bytes = cache.get_bytes("ecdsa", false).await.unwrap();
        let body = std::str::from_utf8(&bytes).unwrap();
        assert!(
            body.contains(cm_raw),
            "compose_manager_attestation should round-trip byte-exact; got body: {body}"
        );
    }

    #[tokio::test]
    async fn test_cache_omits_compose_manager_attestation_when_none() {
        let cache = AttestationCache::new(300);
        let report = make_test_report("ecdsa", "aabb");
        cache.set("ecdsa", false, report, None, None).await;

        let bytes = cache.get_bytes("ecdsa", false).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(parsed.get("compose_manager_attestation").is_none());
    }

    /// Regression: a verifier must be able to recover `actions_hash` by
    /// re-hashing the `actions` field as it appears in inference-proxy's
    /// response. Previously the value was held as `serde_json::Value` which
    /// alphabetized object keys on re-serialization, breaking this binding.
    #[tokio::test]
    async fn test_actions_hash_round_trips_via_cache() {
        use sha2::{Digest, Sha256};

        let cache = AttestationCache::new(300);
        let report = make_test_report("ecdsa", "aabb");

        // Mimic compose-manager: serialize an actions list (struct field order),
        // hash it, then embed both in the attestation body.
        let actions_json = r#"[{"timestamp":"2026-05-21T10:01:40Z","action":"compose_up","tag":"v0.0.169","commit":"8e71b71b","file":"small-models.yaml","file_sha256":"7f60fb50"}]"#;
        let actions_hash = hex::encode(Sha256::digest(actions_json.as_bytes()));
        let cm_raw =
            format!(r#"{{"actions":{actions_json},"actions_hash":"{actions_hash}","quote":"q"}}"#);
        let cm_attestation = serde_json::value::RawValue::from_string(cm_raw.clone()).unwrap();

        cache
            .set("ecdsa", false, report, Some(cm_attestation), None)
            .await;

        // Verifier path: parse the response, extract the actions field, hash it.
        let bytes = cache.get_bytes("ecdsa", false).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let cm = &parsed["compose_manager_attestation"];

        // `serde_json::Value` reorders keys, so we cannot recover actions by
        // re-serializing `cm["actions"]`. Instead, the response embeds the raw
        // compose-manager body — locate the substring and hash directly.
        let body = std::str::from_utf8(&bytes).unwrap();
        let needle = r#""actions":"#;
        let start = body.find(needle).unwrap() + needle.len();
        // Walk until the matching close-bracket of the JSON array.
        let mut depth = 0i32;
        let mut end = start;
        for (i, c) in body[start..].char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        let extracted = &body[start..end];
        let recomputed = hex::encode(Sha256::digest(extracted.as_bytes()));
        assert_eq!(
            recomputed,
            cm["actions_hash"].as_str().unwrap(),
            "verifier must be able to recompute actions_hash from raw response bytes"
        );
    }
}

#[cfg(test)]
mod tests_fetch_compose_manager {
    use super::*;

    #[tokio::test]
    async fn test_fetch_success() {
        let mock = wiremock::MockServer::start().await;
        // Use a hand-rolled body so we control the exact byte sequence and
        // verify the fetcher forwards it verbatim (no key reordering, no
        // re-formatting).
        let cm_body =
            r#"{"quote":"tdx_quote_data","actions":[],"actions_hash":"abc123","nonce":"def456"}"#;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/attestation/report"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_string(cm_body)
                    .insert_header("content-type", "application/json"),
            )
            .mount(&mock)
            .await;

        let client = reqwest::Client::new();
        let result = fetch_compose_manager_attestation(&client, &mock.uri(), None).await;
        let raw = result.expect("expected successful fetch");
        assert_eq!(
            raw.get(),
            cm_body,
            "RawValue bytes must match upstream response byte-for-byte"
        );
    }

    #[tokio::test]
    async fn test_fetch_passes_nonce() {
        let mock = wiremock::MockServer::start().await;
        let nonce = "aa".repeat(32);
        let body = format!(r#"{{"nonce":"{nonce}"}}"#);
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/attestation/report"))
            .and(wiremock::matchers::query_param("nonce", &nonce))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_string(body.clone())
                    .insert_header("content-type", "application/json"),
            )
            .mount(&mock)
            .await;

        let client = reqwest::Client::new();
        let result = fetch_compose_manager_attestation(&client, &mock.uri(), Some(&nonce)).await;
        let raw = result.expect("expected successful fetch");
        assert_eq!(raw.get(), body);
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

#[cfg(test)]
mod tests_tls_tracker {
    use super::TlsCertTracker;
    use std::fs;
    use std::io::Write;

    // Same valid PEM as `tests_spki`. Duplicated here so the two test
    // modules stay independent.
    const TEST_CERT_PEM_A: &str = r#"-----BEGIN CERTIFICATE-----
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

    // A distinct self-signed cert. Public key differs from PEM_A so any
    // honest re-hash produces a different SPKI fingerprint.
    const TEST_CERT_PEM_B: &str = r#"-----BEGIN CERTIFICATE-----
MIIDEzCCAfugAwIBAgIUCF/LqQXgEWr2ZmQrYTNMaM1GsUowDQYJKoZIhvcNAQEL
BQAwGTEXMBUGA1UEAwwOdGVzdC10cmFja2VyLWIwHhcNMjYwNTE5MTEzNTE4WhcN
MzYwNTE2MTEzNTE4WjAZMRcwFQYDVQQDDA50ZXN0LXRyYWNrZXItYjCCASIwDQYJ
KoZIhvcNAQEBBQADggEPADCCAQoCggEBALZYU2Q9XtlxoWFS5LcNEb0y5vMM/Z38
2Fh10omoAiBlkTGtu5eHdM5DFqg4HTnjxh8ch6+OSmZ3sPIMKkUZyp9hgd3y4L53
GLmSkug5cNcM5qoGvmUL01NivzlEQAp8A2SrsqfCb29+pqO5tauhciHynKFSitW7
+nr9fVg2BML4kIOUDrAjejv+tL9X8AwKpuWmClZoYgyoJRZ63Wxql/b1eeCedXy4
+P0Ri1iF3M1pCD7DKSGomlNhA69zfBmjW3Fhofejewps9jv8EnPrO54fA6mB4msm
LJRRi0UH/+1Nst8mUCbfd9wj8nUwDTwqTNRD2Yn1530ZIMjwtyqBZb8CAwEAAaNT
MFEwHQYDVR0OBBYEFE6+caJsXpeQu5CgIxOqDZSRg3tDMB8GA1UdIwQYMBaAFE6+
caJsXpeQu5CgIxOqDZSRg3tDMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQEL
BQADggEBAKN5RyjfQ5sxA8E1effeL8UVXQsY3T/8bFL6FSHMCT+4AQQqhZN5Edpo
oDzzP/OEyb9Xu+ZKyL4PhLQPDiTcRalY67Cgc8F3AsC1uo+mtbwthU8vew+xA89J
jQV2HpRTfPhYNVycb0wEoTZPB8Xo9Lb4mSmUwiP9WrrttiQXBHvJhnFC6l5ymS69
4TiavnAdLTuNnWvNj4Rm9dXiwayWJGxU7veNzwPzLL41WFgrvjcbYqDw+tqlQl4o
vPKm1JDghjp3Oa6sER4/Ei9k/gqh0h2eCAIRHW36HC/pvKVYM3dAJUhkMv7HKpQc
7qBO0wxl/8MPbMS2gs7vcfJVV9jHGJ4=
-----END CERTIFICATE-----
"#;

    fn write_cert(path: &std::path::Path, pem: &str) {
        let mut f = fs::File::create(path).expect("create cert");
        f.write_all(pem.as_bytes()).expect("write cert");
    }

    /// Returns a unique temp path under /tmp for a test PEM.
    fn tmp_path(suffix: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        p.push(format!(
            "tracker_{}_{}_{}.pem",
            suffix,
            std::process::id(),
            id
        ));
        p
    }

    #[test]
    fn none_path_yields_inert_tracker() {
        let t = TlsCertTracker::new(None).expect("None must construct");
        assert!(t.current().is_none());
        assert!(t.refresh_if_changed().is_none());
    }

    #[test]
    fn fails_when_cert_unreadable_at_init() {
        // Path that surely doesn't exist.
        let path = "/nonexistent/should-not-exist/tracker.pem";
        let res = TlsCertTracker::new(Some(path.to_string()));
        assert!(res.is_err(), "expected init failure for missing cert");
    }

    #[test]
    fn current_returns_initial_hash() {
        let path = tmp_path("initial");
        write_cert(&path, TEST_CERT_PEM_A);
        let t = TlsCertTracker::new(Some(path.to_string_lossy().into_owned()))
            .expect("init must succeed");
        let fp = t.current().expect("fingerprint expected");
        // Cleanup before assertion so a panic doesn't leak the file.
        let _ = fs::remove_file(&path);
        assert_eq!(fp.len(), 64, "SHA-256 hex = 64 chars");
    }

    #[test]
    fn refresh_detects_rotation_to_distinct_cert() {
        let path = tmp_path("rotate");
        write_cert(&path, TEST_CERT_PEM_A);
        let t = TlsCertTracker::new(Some(path.to_string_lossy().into_owned()))
            .expect("init must succeed");
        let fp_a = t.current().expect("fingerprint A");

        // Bump mtime forward so the `==` check trips even when the OS has a
        // 1-second mtime granularity. `set_file_mtime` from the stdlib isn't
        // available, but writing a different cert atomically and sleeping
        // briefly is reliable enough for a unit test.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        write_cert(&path, TEST_CERT_PEM_B);

        let new_fp = t.refresh_if_changed();
        let final_fp = t.current();
        let _ = fs::remove_file(&path);

        let new_fp = new_fp.expect("rotation must surface a new fingerprint");
        assert_ne!(new_fp, fp_a, "fingerprint must differ after rotation");
        assert_eq!(final_fp.as_deref(), Some(new_fp.as_str()));
    }

    #[test]
    fn refresh_is_no_op_when_unchanged() {
        let path = tmp_path("noop");
        write_cert(&path, TEST_CERT_PEM_A);
        let t = TlsCertTracker::new(Some(path.to_string_lossy().into_owned()))
            .expect("init must succeed");
        let before = t.current();
        let res = t.refresh_if_changed();
        let after = t.current();
        let _ = fs::remove_file(&path);

        assert!(res.is_none(), "no rotation, no return value");
        assert_eq!(before, after, "fingerprint must not move");
    }

    #[test]
    fn refresh_is_no_op_when_content_identical_after_touch() {
        let path = tmp_path("touch");
        write_cert(&path, TEST_CERT_PEM_A);
        let t = TlsCertTracker::new(Some(path.to_string_lossy().into_owned()))
            .expect("init must succeed");
        let before = t.current();

        // Re-write the same content. mtime advances but the hash doesn't.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        write_cert(&path, TEST_CERT_PEM_A);
        let res = t.refresh_if_changed();
        let after = t.current();
        let _ = fs::remove_file(&path);

        assert!(
            res.is_none(),
            "identical content rewrite must not signal rotation"
        );
        assert_eq!(before, after);
    }

    #[test]
    fn refresh_handles_missing_file_gracefully() {
        let path = tmp_path("vanish");
        write_cert(&path, TEST_CERT_PEM_A);
        let t = TlsCertTracker::new(Some(path.to_string_lossy().into_owned()))
            .expect("init must succeed");
        let before = t.current();

        // File disappears. tracker must keep the cached value and not panic.
        let _ = fs::remove_file(&path);
        let res = t.refresh_if_changed();
        let after = t.current();

        assert!(res.is_none(), "missing file: no rotation signal");
        assert_eq!(
            before, after,
            "missing file must not clear the cached fingerprint"
        );
    }
}
