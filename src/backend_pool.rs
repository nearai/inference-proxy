use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tracing::{debug, info, warn};

/// A single backend instance (e.g., one vLLM process).
pub struct Backend {
    pub base_url: String,
    pub healthy: AtomicBool,
    pub active_conns: AtomicU32,
    pub consecutive_failures: AtomicU32,
}

impl Backend {
    fn new(base_url: String) -> Self {
        Self {
            base_url,
            healthy: AtomicBool::new(true),
            active_conns: AtomicU32::new(0),
            consecutive_failures: AtomicU32::new(0),
        }
    }

    /// Build a full URL by appending a path to this backend's base URL.
    pub fn url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        let path = path.trim_start_matches('/');
        if path.is_empty() {
            base.to_string()
        } else {
            format!("{base}/{path}")
        }
    }
}

impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Backend")
            .field("base_url", &self.base_url)
            .field("healthy", &self.healthy.load(Ordering::Relaxed))
            .field("active_conns", &self.active_conns.load(Ordering::Relaxed))
            .finish()
    }
}

/// RAII guard that decrements active_conns on drop.
pub struct BackendGuard {
    backend: Arc<Backend>,
}

impl BackendGuard {
    pub fn new(backend: Arc<Backend>) -> Self {
        backend.active_conns.fetch_add(1, Ordering::Relaxed);
        Self { backend }
    }
}

impl Drop for BackendGuard {
    fn drop(&mut self) {
        self.backend.active_conns.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Pool of backends for the same model, with least-connections selection.
pub struct BackendPool {
    backends: Vec<Arc<Backend>>,
}

impl BackendPool {
    pub fn new(base_urls: Vec<String>) -> Self {
        assert!(!base_urls.is_empty(), "at least one backend URL required");
        let backends = base_urls
            .into_iter()
            .map(|u| Arc::new(Backend::new(u)))
            .collect();
        Self { backends }
    }

    /// Number of backends in the pool.
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Select a backend using least-connections among healthy backends.
    /// If all are unhealthy, picks the least-loaded one anyway.
    pub fn select(&self) -> Arc<Backend> {
        if self.backends.len() == 1 {
            return self.backends[0].clone();
        }

        // Try healthy backends first
        let healthy = self
            .backends
            .iter()
            .filter(|b| b.healthy.load(Ordering::Relaxed))
            .min_by_key(|b| b.active_conns.load(Ordering::Relaxed));

        if let Some(b) = healthy {
            return b.clone();
        }

        // All unhealthy — pick least-loaded anyway
        self.backends
            .iter()
            .min_by_key(|b| b.active_conns.load(Ordering::Relaxed))
            .expect("backends is non-empty")
            .clone()
    }

    /// Select a backend and return (full_url, guard).
    /// The guard tracks active connections and decrements on drop.
    pub fn select_url(&self, path: &str) -> (String, BackendGuard) {
        let backend = self.select();
        let url = backend.url(path);
        let guard = BackendGuard::new(backend);
        (url, guard)
    }

    /// Get a reference to the backends (for health checking).
    pub fn backends(&self) -> &[Arc<Backend>] {
        &self.backends
    }
}

/// Spawn a background health check task that pings each backend periodically.
pub fn spawn_health_check(
    pool: Arc<BackendPool>,
    client: reqwest::Client,
    interval: Duration,
    timeout: Duration,
    max_failures: u32,
    health_path: &str,
) {
    let health_path = health_path.to_string();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        tick.tick().await; // skip immediate first tick
        loop {
            tick.tick().await;
            for backend in pool.backends() {
                let url = backend.url(&health_path);
                let result = client.get(&url).timeout(timeout).send().await;

                match result {
                    Ok(resp) if resp.status().is_success() => {
                        let was_unhealthy = !backend.healthy.load(Ordering::Relaxed);
                        backend.consecutive_failures.store(0, Ordering::Relaxed);
                        backend.healthy.store(true, Ordering::Relaxed);
                        if was_unhealthy {
                            info!(backend = %backend.base_url, "Backend recovered");
                        }
                    }
                    Ok(resp) => {
                        let failures =
                            backend.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
                        if failures >= max_failures {
                            let was_healthy = backend.healthy.swap(false, Ordering::Relaxed);
                            if was_healthy {
                                warn!(
                                    backend = %backend.base_url,
                                    status = %resp.status(),
                                    failures,
                                    "Backend marked unhealthy"
                                );
                            }
                        } else {
                            debug!(
                                backend = %backend.base_url,
                                status = %resp.status(),
                                failures,
                                max_failures,
                                "Backend health check failed"
                            );
                        }
                    }
                    Err(e) => {
                        let failures =
                            backend.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
                        if failures >= max_failures {
                            let was_healthy = backend.healthy.swap(false, Ordering::Relaxed);
                            if was_healthy {
                                warn!(
                                    backend = %backend.base_url,
                                    error = %e,
                                    failures,
                                    "Backend marked unhealthy"
                                );
                            }
                        } else {
                            debug!(
                                backend = %backend.base_url,
                                error = %e,
                                failures,
                                max_failures,
                                "Backend health check failed"
                            );
                        }
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_backend_always_selected() {
        let pool = BackendPool::new(vec!["http://localhost:8000".to_string()]);
        let b = pool.select();
        assert_eq!(b.base_url, "http://localhost:8000");
    }

    #[test]
    fn test_least_connections_selection() {
        let pool = BackendPool::new(vec![
            "http://b1:8000".to_string(),
            "http://b2:8000".to_string(),
        ]);
        // Simulate b1 having more connections
        pool.backends[0].active_conns.store(5, Ordering::Relaxed);
        pool.backends[1].active_conns.store(1, Ordering::Relaxed);

        let selected = pool.select();
        assert_eq!(selected.base_url, "http://b2:8000");
    }

    #[test]
    fn test_unhealthy_backend_skipped() {
        let pool = BackendPool::new(vec![
            "http://b1:8000".to_string(),
            "http://b2:8000".to_string(),
        ]);
        pool.backends[0].healthy.store(false, Ordering::Relaxed);

        let selected = pool.select();
        assert_eq!(selected.base_url, "http://b2:8000");
    }

    #[test]
    fn test_all_unhealthy_still_selects() {
        let pool = BackendPool::new(vec![
            "http://b1:8000".to_string(),
            "http://b2:8000".to_string(),
        ]);
        pool.backends[0].healthy.store(false, Ordering::Relaxed);
        pool.backends[1].healthy.store(false, Ordering::Relaxed);
        pool.backends[0].active_conns.store(3, Ordering::Relaxed);
        pool.backends[1].active_conns.store(1, Ordering::Relaxed);

        let selected = pool.select();
        assert_eq!(selected.base_url, "http://b2:8000");
    }

    #[test]
    fn test_backend_guard_tracks_connections() {
        let pool = BackendPool::new(vec!["http://b1:8000".to_string()]);
        assert_eq!(pool.backends[0].active_conns.load(Ordering::Relaxed), 0);

        let guard = BackendGuard::new(pool.backends[0].clone());
        assert_eq!(pool.backends[0].active_conns.load(Ordering::Relaxed), 1);

        drop(guard);
        assert_eq!(pool.backends[0].active_conns.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_select_url_builds_correct_url() {
        let pool = BackendPool::new(vec!["http://b1:8000".to_string()]);
        let (url, _guard) = pool.select_url("/v1/chat/completions");
        assert_eq!(url, "http://b1:8000/v1/chat/completions");
    }

    #[test]
    fn test_backend_url_handles_trailing_slash() {
        let b = Backend::new("http://b1:8000/".to_string());
        assert_eq!(b.url("/v1/models"), "http://b1:8000/v1/models");
        assert_eq!(b.url(""), "http://b1:8000");
    }
}
