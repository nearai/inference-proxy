use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tracing::{debug, info, warn};

use crate::error::AppError;

/// A single backend instance (e.g., one vLLM process).
pub struct Backend {
    pub base_url: String,
    pub healthy: AtomicBool,
    pub active_conns: AtomicU32,
    pub consecutive_failures: AtomicU32,
}

impl Backend {
    fn new(base_url: String, healthy: bool) -> Self {
        Self {
            base_url,
            healthy: AtomicBool::new(healthy),
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
///
/// Holds an `Arc<Backend>`, so an in-flight request keeps its backend alive
/// (and correctly accounted) even after discovery removes that backend from
/// the pool's membership.
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
    /// The preferred backend is no longer a member of the pool (removed by
    /// discovery); selection fell back to least-connections.
    Departed,
}

impl SelectionOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            SelectionOutcome::Single => "single",
            SelectionOutcome::New => "new",
            SelectionOutcome::Pinned => "pinned",
            SelectionOutcome::Rebalanced => "rebalanced",
            SelectionOutcome::Unhealthy => "unhealthy",
            SelectionOutcome::Departed => "departed",
        }
    }
}

/// Result of `BackendPool::select_with_preference`.
pub struct Selection {
    pub backend: Arc<Backend>,
    /// Position of `backend` in the membership snapshot the selection was made
    /// from. Only meaningful as a bounded metric label; membership can change
    /// between calls, so never use it to identify a backend across requests.
    pub index: usize,
    pub outcome: SelectionOutcome,
}

/// Membership delta returned by `BackendPool::set_backends`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MembershipChange {
    pub added: Vec<String>,
    pub removed: Vec<String>,
}

impl MembershipChange {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }
}

/// Pool of backends for the same model, with least-connections selection.
///
/// Membership is a copy-on-write snapshot: readers take an `Arc` to the
/// current `Vec` (no lock held while selecting), and `set_backends` swaps in a
/// new `Vec` while preserving the `Backend` instances — and therefore their
/// health and in-flight counters — of every URL that stays a member. The pool
/// may be empty (before the first successful discovery poll); selection then
/// fails with `AppError::NoBackendsAvailable` (HTTP 503).
pub struct BackendPool {
    backends: RwLock<Arc<Vec<Arc<Backend>>>>,
}

impl BackendPool {
    pub fn new(base_urls: Vec<String>) -> Self {
        let backends = base_urls
            .into_iter()
            .map(|u| Arc::new(Backend::new(u, true)))
            .collect();
        Self {
            backends: RwLock::new(Arc::new(backends)),
        }
    }

    /// Current membership snapshot. Cheap (one Arc clone); the returned Vec is
    /// immutable, so iterate it freely without holding any lock.
    pub fn backends(&self) -> Arc<Vec<Arc<Backend>>> {
        self.backends
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Number of backends in the pool.
    pub fn len(&self) -> usize {
        self.backends().len()
    }

    /// Whether the pool currently has no members.
    pub fn is_empty(&self) -> bool {
        self.backends().is_empty()
    }

    /// Replace the membership with `members` (base URL, initial health for a
    /// backend that is new to the pool). URLs already in the pool keep their
    /// existing `Backend` — health state owned by the health checker and the
    /// live `active_conns` count survive the swap. Duplicates are ignored.
    /// Returns which URLs were added and removed.
    pub fn set_backends(&self, members: Vec<(String, bool)>) -> MembershipChange {
        let mut guard = self
            .backends
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let existing: HashMap<&str, &Arc<Backend>> =
            guard.iter().map(|b| (b.base_url.as_str(), b)).collect();

        let mut change = MembershipChange::default();
        let mut next: Vec<Arc<Backend>> = Vec::with_capacity(members.len());
        for (url, healthy) in members {
            let url = url.trim_end_matches('/').to_string();
            if next.iter().any(|b| b.base_url == url) {
                continue;
            }
            match existing.get(url.as_str()) {
                Some(backend) => next.push(Arc::clone(backend)),
                None => {
                    change.added.push(url.clone());
                    next.push(Arc::new(Backend::new(url, healthy)));
                }
            }
        }
        for backend in guard.iter() {
            if !next.iter().any(|b| b.base_url == backend.base_url) {
                change.removed.push(backend.base_url.clone());
            }
        }
        if !change.is_empty() {
            *guard = Arc::new(next);
        }
        change
    }

    /// Select a backend using least-connections among healthy backends.
    /// If all are unhealthy, picks the least-loaded one anyway. `None` only
    /// when the pool is empty.
    pub fn select(&self) -> Option<Arc<Backend>> {
        let backends = self.backends();
        if backends.len() == 1 {
            return Some(backends[0].clone());
        }
        least_connections_index(&backends).map(|i| backends[i].clone())
    }

    /// Least-connections selection with an optional preferred backend (by base
    /// URL, the stable identity of a backend across membership changes).
    ///
    /// The preferred backend wins when it is still a member, healthy, and
    /// carries at most `max_imbalance` more in-flight requests than the
    /// least-loaded healthy backend; otherwise selection falls back to plain
    /// least-connections. `None` only when the pool is empty.
    pub fn select_with_preference(
        &self,
        preferred: Option<&str>,
        max_imbalance: u32,
    ) -> Option<Selection> {
        let backends = self.backends();
        if backends.len() == 1 {
            return Some(Selection {
                backend: backends[0].clone(),
                index: 0,
                outcome: SelectionOutcome::Single,
            });
        }

        let least_index = least_connections_index(&backends)?;
        let least = &backends[least_index];
        let Some(preferred) = preferred else {
            return Some(Selection {
                backend: least.clone(),
                index: least_index,
                outcome: SelectionOutcome::New,
            });
        };

        let Some(preferred_index) = backends.iter().position(|b| b.base_url == preferred) else {
            return Some(Selection {
                backend: least.clone(),
                index: least_index,
                outcome: SelectionOutcome::Departed,
            });
        };

        let candidate = &backends[preferred_index];
        if !candidate.healthy.load(Ordering::Relaxed) {
            return Some(Selection {
                backend: least.clone(),
                index: least_index,
                outcome: SelectionOutcome::Unhealthy,
            });
        }

        let candidate_conns = candidate.active_conns.load(Ordering::Relaxed);
        let least_conns = least.active_conns.load(Ordering::Relaxed);
        if candidate_conns <= least_conns.saturating_add(max_imbalance) {
            Some(Selection {
                backend: candidate.clone(),
                index: preferred_index,
                outcome: SelectionOutcome::Pinned,
            })
        } else {
            Some(Selection {
                backend: least.clone(),
                index: least_index,
                outcome: SelectionOutcome::Rebalanced,
            })
        }
    }

    /// Select a backend and return (full_url, guard).
    /// The guard tracks active connections and decrements on drop.
    pub fn select_url(&self, path: &str) -> Result<(String, BackendGuard), AppError> {
        let backend = self.select().ok_or(AppError::NoBackendsAvailable)?;
        let url = backend.url(path);
        let guard = BackendGuard::new(backend);
        Ok((url, guard))
    }
}

/// Index of the least-loaded healthy backend, or of the least-loaded backend
/// overall when none is healthy. `None` for an empty slice.
fn least_connections_index(backends: &[Arc<Backend>]) -> Option<usize> {
    let healthy = backends
        .iter()
        .enumerate()
        .filter(|(_, b)| b.healthy.load(Ordering::Relaxed))
        .min_by_key(|(_, b)| b.active_conns.load(Ordering::Relaxed))
        .map(|(index, _)| index);

    healthy.or_else(|| {
        backends
            .iter()
            .enumerate()
            .min_by_key(|(_, b)| b.active_conns.load(Ordering::Relaxed))
            .map(|(index, _)| index)
    })
}

/// Spawn a background health check task that pings each backend periodically.
/// Reads the membership snapshot on every tick, so backends added or removed
/// by discovery are picked up without a restart.
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
            for backend in pool.backends().iter() {
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
            let backends = pool.backends();
            let healthy = backends
                .iter()
                .filter(|b| b.healthy.load(Ordering::Relaxed))
                .count();
            metrics::gauge!("backend_pool_size").set(backends.len() as f64);
            metrics::gauge!("backend_pool_healthy").set(healthy as f64);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_backend_always_selected() {
        let pool = BackendPool::new(vec!["http://localhost:8000".to_string()]);
        let b = pool.select().unwrap();
        assert_eq!(b.base_url, "http://localhost:8000");
    }

    #[test]
    fn test_least_connections_selection() {
        let pool = BackendPool::new(vec![
            "http://b1:8000".to_string(),
            "http://b2:8000".to_string(),
        ]);
        // Simulate b1 having more connections
        pool.backends()[0].active_conns.store(5, Ordering::Relaxed);
        pool.backends()[1].active_conns.store(1, Ordering::Relaxed);

        let selected = pool.select().unwrap();
        assert_eq!(selected.base_url, "http://b2:8000");
    }

    #[test]
    fn test_unhealthy_backend_skipped() {
        let pool = BackendPool::new(vec![
            "http://b1:8000".to_string(),
            "http://b2:8000".to_string(),
        ]);
        pool.backends()[0].healthy.store(false, Ordering::Relaxed);

        let selected = pool.select().unwrap();
        assert_eq!(selected.base_url, "http://b2:8000");
    }

    #[test]
    fn test_all_unhealthy_still_selects() {
        let pool = BackendPool::new(vec![
            "http://b1:8000".to_string(),
            "http://b2:8000".to_string(),
        ]);
        pool.backends()[0].healthy.store(false, Ordering::Relaxed);
        pool.backends()[1].healthy.store(false, Ordering::Relaxed);
        pool.backends()[0].active_conns.store(3, Ordering::Relaxed);
        pool.backends()[1].active_conns.store(1, Ordering::Relaxed);

        let selected = pool.select().unwrap();
        assert_eq!(selected.base_url, "http://b2:8000");
    }

    #[test]
    fn test_empty_pool_selects_nothing() {
        let pool = BackendPool::new(Vec::new());
        assert!(pool.is_empty());
        assert!(pool.select().is_none());
        assert!(pool
            .select_with_preference(Some("http://b1:8000"), 8)
            .is_none());
        assert!(matches!(
            pool.select_url("/v1/models"),
            Err(AppError::NoBackendsAvailable)
        ));
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
        pool.backends()[0].active_conns.store(4, Ordering::Relaxed);
        pool.backends()[1].active_conns.store(1, Ordering::Relaxed);

        // 4 <= 1 + 8: the preferred backend keeps the request.
        let selection = pool
            .select_with_preference(Some("http://b1:8000"), 8)
            .unwrap();
        assert_eq!(selection.index, 0);
        assert_eq!(selection.outcome, SelectionOutcome::Pinned);
        assert_eq!(selection.backend.base_url, "http://b1:8000");
    }

    #[test]
    fn test_preference_rebalances_beyond_imbalance_bound() {
        let pool = two_backends();
        pool.backends()[0].active_conns.store(4, Ordering::Relaxed);
        pool.backends()[1].active_conns.store(1, Ordering::Relaxed);

        // 4 > 1 + 2: fall back to least-connections.
        let selection = pool
            .select_with_preference(Some("http://b1:8000"), 2)
            .unwrap();
        assert_eq!(selection.index, 1);
        assert_eq!(selection.outcome, SelectionOutcome::Rebalanced);
    }

    #[test]
    fn test_preference_skips_unhealthy_backend() {
        let pool = two_backends();
        pool.backends()[0].healthy.store(false, Ordering::Relaxed);

        let selection = pool
            .select_with_preference(Some("http://b1:8000"), 8)
            .unwrap();
        assert_eq!(selection.index, 1);
        assert_eq!(selection.outcome, SelectionOutcome::Unhealthy);
    }

    #[test]
    fn test_no_or_departed_preference_is_least_connections() {
        let pool = two_backends();
        pool.backends()[0].active_conns.store(2, Ordering::Relaxed);

        let selection = pool.select_with_preference(None, 8).unwrap();
        assert_eq!(selection.index, 1);
        assert_eq!(selection.outcome, SelectionOutcome::New);

        let selection = pool
            .select_with_preference(Some("http://gone:8000"), 8)
            .unwrap();
        assert_eq!(selection.index, 1);
        assert_eq!(selection.outcome, SelectionOutcome::Departed);
    }

    #[test]
    fn test_single_backend_ignores_preference() {
        let pool = BackendPool::new(vec!["http://b1:8000".to_string()]);
        let selection = pool
            .select_with_preference(Some("http://other:8000"), 0)
            .unwrap();
        assert_eq!(selection.index, 0);
        assert_eq!(selection.outcome, SelectionOutcome::Single);
    }

    #[test]
    fn test_backend_guard_tracks_connections() {
        let pool = BackendPool::new(vec!["http://b1:8000".to_string()]);
        assert_eq!(pool.backends()[0].active_conns.load(Ordering::Relaxed), 0);

        let guard = BackendGuard::new(pool.backends()[0].clone());
        assert_eq!(pool.backends()[0].active_conns.load(Ordering::Relaxed), 1);

        drop(guard);
        assert_eq!(pool.backends()[0].active_conns.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_select_url_builds_correct_url() {
        let pool = BackendPool::new(vec!["http://b1:8000".to_string()]);
        let (url, _guard) = pool.select_url("/v1/chat/completions").unwrap();
        assert_eq!(url, "http://b1:8000/v1/chat/completions");
    }

    #[test]
    fn test_backend_url_handles_trailing_slash() {
        let b = Backend::new("http://b1:8000/".to_string(), true);
        assert_eq!(b.url("/v1/models"), "http://b1:8000/v1/models");
        assert_eq!(b.url(""), "http://b1:8000");
    }

    #[test]
    fn test_set_backends_preserves_existing_state_and_reports_delta() {
        let pool = two_backends();
        pool.backends()[0].active_conns.store(7, Ordering::Relaxed);
        pool.backends()[1].healthy.store(false, Ordering::Relaxed);
        let b1 = pool.backends()[0].clone();

        // b2 leaves, b3 arrives (initially unhealthy per the registry), b1 stays.
        let change = pool.set_backends(vec![
            ("http://b1:8000/".to_string(), true),
            ("http://b3:8000".to_string(), false),
            ("http://b3:8000".to_string(), true), // duplicate ignored
        ]);
        assert_eq!(change.added, vec!["http://b3:8000".to_string()]);
        assert_eq!(change.removed, vec!["http://b2:8000".to_string()]);

        let backends = pool.backends();
        assert_eq!(backends.len(), 2);
        assert!(Arc::ptr_eq(&backends[0], &b1), "b1 instance must survive");
        assert_eq!(backends[0].active_conns.load(Ordering::Relaxed), 7);
        assert_eq!(backends[1].base_url, "http://b3:8000");
        assert!(!backends[1].healthy.load(Ordering::Relaxed));

        // No change → identical snapshot, empty delta.
        let before = pool.backends();
        let change = pool.set_backends(vec![
            ("http://b1:8000".to_string(), true),
            ("http://b3:8000".to_string(), true),
        ]);
        assert!(change.is_empty());
        assert!(Arc::ptr_eq(&before, &pool.backends()));
    }

    #[test]
    fn test_removed_backend_stays_alive_for_in_flight_guard() {
        let pool = two_backends();
        let (url, guard) = pool.select_url("/v1/chat/completions").unwrap();
        assert_eq!(url, "http://b1:8000/v1/chat/completions");

        let change = pool.set_backends(vec![("http://b2:8000".to_string(), true)]);
        assert_eq!(change.removed, vec!["http://b1:8000".to_string()]);
        assert_eq!(pool.len(), 1);

        // New selections only see b2; the in-flight request keeps its guard.
        let (url, _g2) = pool.select_url("/v1/chat/completions").unwrap();
        assert_eq!(url, "http://b2:8000/v1/chat/completions");
        assert_eq!(guard.backend.active_conns.load(Ordering::Relaxed), 1);
        drop(guard);
    }
}
