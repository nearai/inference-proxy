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

/// Why `BackendPool::select_with_preference` chose the backend it did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionOutcome {
    /// Only one backend in the pool.
    Single,
    /// No preference given: plain least-connections.
    New,
    /// The preferred backend was healthy and within the imbalance bound.
    Pinned,
    /// The preferred backend was healthy but too far above the least-loaded one.
    Rebalanced,
    /// The preferred backend was unhealthy.
    Unhealthy,
}

impl SelectionOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            SelectionOutcome::Single => "single",
            SelectionOutcome::New => "new",
            SelectionOutcome::Pinned => "pinned",
            SelectionOutcome::Rebalanced => "rebalanced",
            SelectionOutcome::Unhealthy => "unhealthy",
        }
    }
}

/// Result of `BackendPool::select_with_preference`.
pub struct Selection {
    pub backend: Arc<Backend>,
    /// Position of `backend` in the pool (stable for the process lifetime).
    pub index: usize,
    pub outcome: SelectionOutcome,
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

    /// Whether the pool is empty (always false — constructor requires ≥1 backend).
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }

    /// Select a backend using least-connections among healthy backends.
    /// If all are unhealthy, picks the least-loaded one anyway.
    pub fn select(&self) -> Arc<Backend> {
        if self.backends.len() == 1 {
            return self.backends[0].clone();
        }
        self.backends[self.least_connections_index()].clone()
    }

    /// Least-connections selection with an optional preferred backend.
    ///
    /// The preferred backend wins when it is healthy and carries at most
    /// `max_imbalance` more in-flight requests than the least-loaded healthy
    /// backend; otherwise selection falls back to plain least-connections. An
    /// index outside the pool is treated as no preference.
    pub fn select_with_preference(
        &self,
        preferred: Option<usize>,
        max_imbalance: u32,
    ) -> Selection {
        if self.backends.len() == 1 {
            return Selection {
                backend: self.backends[0].clone(),
                index: 0,
                outcome: SelectionOutcome::Single,
            };
        }

        let least_index = self.least_connections_index();
        let least = &self.backends[least_index];
        let Some(preferred_index) = preferred.filter(|index| *index < self.backends.len()) else {
            return Selection {
                backend: least.clone(),
                index: least_index,
                outcome: SelectionOutcome::New,
            };
        };

        let candidate = &self.backends[preferred_index];
        if !candidate.healthy.load(Ordering::Relaxed) {
            return Selection {
                backend: least.clone(),
                index: least_index,
                outcome: SelectionOutcome::Unhealthy,
            };
        }

        let candidate_conns = candidate.active_conns.load(Ordering::Relaxed);
        let least_conns = least.active_conns.load(Ordering::Relaxed);
        if candidate_conns <= least_conns.saturating_add(max_imbalance) {
            Selection {
                backend: candidate.clone(),
                index: preferred_index,
                outcome: SelectionOutcome::Pinned,
            }
        } else {
            Selection {
                backend: least.clone(),
                index: least_index,
                outcome: SelectionOutcome::Rebalanced,
            }
        }
    }

    /// Index of the least-loaded healthy backend, or of the least-loaded
    /// backend overall when none is healthy.
    fn least_connections_index(&self) -> usize {
        let healthy = self
            .backends
            .iter()
            .enumerate()
            .filter(|(_, b)| b.healthy.load(Ordering::Relaxed))
            .min_by_key(|(_, b)| b.active_conns.load(Ordering::Relaxed))
            .map(|(index, _)| index);

        healthy.unwrap_or_else(|| {
            self.backends
                .iter()
                .enumerate()
                .min_by_key(|(_, b)| b.active_conns.load(Ordering::Relaxed))
                .map(|(index, _)| index)
                .expect("backends is non-empty")
        })
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

    fn two_backends() -> BackendPool {
        BackendPool::new(vec![
            "http://b1:8000".to_string(),
            "http://b2:8000".to_string(),
        ])
    }

    #[test]
    fn test_preference_pins_within_imbalance_bound() {
        let pool = two_backends();
        pool.backends[0].active_conns.store(4, Ordering::Relaxed);
        pool.backends[1].active_conns.store(1, Ordering::Relaxed);

        // 4 <= 1 + 8: the preferred backend keeps the request.
        let selection = pool.select_with_preference(Some(0), 8);
        assert_eq!(selection.index, 0);
        assert_eq!(selection.outcome, SelectionOutcome::Pinned);
        assert_eq!(selection.backend.base_url, "http://b1:8000");
    }

    #[test]
    fn test_preference_rebalances_beyond_imbalance_bound() {
        let pool = two_backends();
        pool.backends[0].active_conns.store(4, Ordering::Relaxed);
        pool.backends[1].active_conns.store(1, Ordering::Relaxed);

        // 4 > 1 + 2: fall back to least-connections.
        let selection = pool.select_with_preference(Some(0), 2);
        assert_eq!(selection.index, 1);
        assert_eq!(selection.outcome, SelectionOutcome::Rebalanced);
    }

    #[test]
    fn test_preference_skips_unhealthy_backend() {
        let pool = two_backends();
        pool.backends[0].healthy.store(false, Ordering::Relaxed);

        let selection = pool.select_with_preference(Some(0), 8);
        assert_eq!(selection.index, 1);
        assert_eq!(selection.outcome, SelectionOutcome::Unhealthy);
    }

    #[test]
    fn test_no_or_invalid_preference_is_least_connections() {
        let pool = two_backends();
        pool.backends[0].active_conns.store(2, Ordering::Relaxed);

        let selection = pool.select_with_preference(None, 8);
        assert_eq!(selection.index, 1);
        assert_eq!(selection.outcome, SelectionOutcome::New);

        let selection = pool.select_with_preference(Some(7), 8);
        assert_eq!(selection.index, 1);
        assert_eq!(selection.outcome, SelectionOutcome::New);
    }

    #[test]
    fn test_single_backend_ignores_preference() {
        let pool = BackendPool::new(vec!["http://b1:8000".to_string()]);
        let selection = pool.select_with_preference(Some(3), 0);
        assert_eq!(selection.index, 0);
        assert_eq!(selection.outcome, SelectionOutcome::Single);
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
