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

    /// Wrap a slot that `BackendPool::reserve` already took on
    /// `backend.active_conns` (no second increment).
    fn reserved(backend: Arc<Backend>) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &Arc<Backend> {
        &self.backend
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

/// Result of a reserving selection: the backend, its stable index in the
/// pool, why it was chosen, and the in-flight slot already taken on it.
pub struct Selection {
    pub backend: Arc<Backend>,
    /// Position of `backend` in the pool (stable for the process lifetime).
    pub index: usize,
    pub outcome: SelectionOutcome,
    pub guard: BackendGuard,
}

/// Attempts to turn a pick into a reservation before giving up; a lost race
/// (another request took the last slot under the bound) picks again.
const RESERVE_ATTEMPTS: usize = 16;

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
    /// If all are unhealthy, picks the least-loaded one anyway. No slot is
    /// reserved; callers create their own `BackendGuard`.
    pub fn select(&self) -> Arc<Backend> {
        let index = self
            .least_index(|_, backend| backend.healthy.load(Ordering::Relaxed))
            .unwrap_or_else(|| {
                self.least_index(|_, _| true)
                    .expect("backends is non-empty")
            });
        self.backends[index].clone()
    }

    /// Least-connections selection with an optional preferred backend; the
    /// returned `Selection` holds the in-flight slot.
    ///
    /// The preferred backend wins when it is healthy and carries at most
    /// `max_imbalance` more in-flight requests than the least-loaded healthy
    /// backend; otherwise selection falls back to plain least-connections. An
    /// index outside the pool is treated as no preference. With every backend
    /// unhealthy the least-loaded one is returned anyway (a probe will re-mark
    /// it; refusing everything would be worse).
    pub fn select_with_preference(
        &self,
        preferred: Option<usize>,
        max_imbalance: u32,
    ) -> Selection {
        self.reserve(preferred, max_imbalance, None, &|_| false, true)
            .expect("unbounded selection always yields a backend")
    }

    /// `select_with_preference` restricted to backends that are not `avoid`ed
    /// and have fewer than `max_conns` requests in flight (the admission
    /// per-host share, see `admission.rs`). The slot is taken atomically, so
    /// concurrent selections cannot overshoot the bound. A preferred backend
    /// that is avoided or at the bound is treated like an overloaded one and
    /// the turn is rebalanced; `None` when no backend is eligible.
    pub fn select_with_preference_bounded(
        &self,
        preferred: Option<usize>,
        max_imbalance: u32,
        max_conns: Option<u32>,
        avoid: &dyn Fn(usize) -> bool,
    ) -> Option<Selection> {
        // Without a bound (admission off) keep the legacy degradation when
        // every backend is unhealthy; with one, refusing is the point.
        self.reserve(
            preferred,
            max_imbalance,
            max_conns,
            avoid,
            max_conns.is_none(),
        )
    }

    /// Least-loaded eligible backend other than `excluded` (connection
    /// fail-over), under the same bound and avoidance as
    /// `select_with_preference_bounded`; `None` when there is none.
    pub fn select_excluding(
        &self,
        excluded: usize,
        max_conns: Option<u32>,
        avoid: &dyn Fn(usize) -> bool,
    ) -> Option<Selection> {
        if self.backends.len() == 1 {
            return None;
        }
        self.reserve(
            None,
            0,
            max_conns,
            &|index| index == excluded || avoid(index),
            false,
        )
    }

    /// Number of backends currently marked healthy.
    pub fn healthy_count(&self) -> usize {
        self.backends
            .iter()
            .filter(|b| b.healthy.load(Ordering::Relaxed))
            .count()
    }

    /// Pick, then take the slot with a compare-and-swap so the bound holds
    /// under concurrency; a lost race picks again.
    fn reserve(
        &self,
        preferred: Option<usize>,
        max_imbalance: u32,
        max_conns: Option<u32>,
        avoid: &dyn Fn(usize) -> bool,
        degrade_when_all_unhealthy: bool,
    ) -> Option<Selection> {
        for _ in 0..RESERVE_ATTEMPTS {
            let (index, outcome) = self.pick(
                preferred,
                max_imbalance,
                max_conns,
                avoid,
                degrade_when_all_unhealthy,
            )?;
            let backend = &self.backends[index];
            let taken =
                backend
                    .active_conns
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |conns| {
                        max_conns
                            .is_none_or(|max| conns < max)
                            .then(|| conns.saturating_add(1))
                    });
            if taken.is_ok() {
                return Some(Selection {
                    backend: backend.clone(),
                    index,
                    outcome,
                    guard: BackendGuard::reserved(backend.clone()),
                });
            }
        }
        None
    }

    /// The selection policy on a snapshot of the counters (no reservation).
    fn pick(
        &self,
        preferred: Option<usize>,
        max_imbalance: u32,
        max_conns: Option<u32>,
        avoid: &dyn Fn(usize) -> bool,
        degrade_when_all_unhealthy: bool,
    ) -> Option<(usize, SelectionOutcome)> {
        let under_bound = |backend: &Backend| {
            max_conns.is_none_or(|max| backend.active_conns.load(Ordering::Relaxed) < max)
        };
        if self.backends.len() == 1 {
            let only = &self.backends[0];
            return (!avoid(0) && under_bound(only)).then_some((0, SelectionOutcome::Single));
        }

        let eligible = |index: usize, backend: &Backend| {
            !avoid(index) && backend.healthy.load(Ordering::Relaxed) && under_bound(backend)
        };
        let least_index = match self.least_index(eligible) {
            Some(index) => index,
            // Everything unhealthy: degrade to the least-loaded backend that is
            // not avoided rather than refusing every request (a probe will
            // re-mark them). Never for a bounded or fail-over selection.
            None if degrade_when_all_unhealthy => {
                self.least_index(|index, backend| !avoid(index) && under_bound(backend))?
            }
            None => return None,
        };
        let Some(preferred_index) = preferred.filter(|index| *index < self.backends.len()) else {
            return Some((least_index, SelectionOutcome::New));
        };

        let candidate = &self.backends[preferred_index];
        if !candidate.healthy.load(Ordering::Relaxed) {
            return Some((least_index, SelectionOutcome::Unhealthy));
        }
        let candidate_conns = candidate.active_conns.load(Ordering::Relaxed);
        let least_conns = self.backends[least_index]
            .active_conns
            .load(Ordering::Relaxed);
        if eligible(preferred_index, candidate)
            && candidate_conns <= least_conns.saturating_add(max_imbalance)
        {
            Some((preferred_index, SelectionOutcome::Pinned))
        } else {
            Some((least_index, SelectionOutcome::Rebalanced))
        }
    }

    /// Index of the least-loaded backend among those `eligible` accepts.
    fn least_index(&self, eligible: impl Fn(usize, &Backend) -> bool) -> Option<usize> {
        self.backends
            .iter()
            .enumerate()
            .filter(|(index, backend)| eligible(*index, backend))
            .min_by_key(|(_, backend)| backend.active_conns.load(Ordering::Relaxed))
            .map(|(index, _)| index)
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
            let healthy = pool
                .backends()
                .iter()
                .filter(|b| b.healthy.load(Ordering::Relaxed))
                .count();
            metrics::gauge!("backend_pool_size").set(pool.len() as f64);
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

    fn no_avoid(_: usize) -> bool {
        false
    }

    #[test]
    fn test_bounded_selection_skips_backends_at_the_share() {
        let pool = two_backends();
        pool.backends()[0].active_conns.store(2, Ordering::Relaxed);
        pool.backends()[1].active_conns.store(1, Ordering::Relaxed);
        // Share 2: only b2 has room, and the reservation is taken on it.
        let sel = pool
            .select_with_preference_bounded(None, 8, Some(2), &no_avoid)
            .unwrap();
        assert_eq!(sel.index, 1);
        assert_eq!(pool.backends()[1].active_conns.load(Ordering::Relaxed), 2);
        drop(sel);
        assert_eq!(pool.backends()[1].active_conns.load(Ordering::Relaxed), 1);
        // A pin on the full backend is rebalanced even within the imbalance bound.
        let sel = pool
            .select_with_preference_bounded(Some(0), 8, Some(2), &no_avoid)
            .unwrap();
        assert_eq!(sel.index, 1);
        assert_eq!(sel.outcome, SelectionOutcome::Rebalanced);
        drop(sel);
        // Everyone at the share: nothing to pick.
        pool.backends()[1].active_conns.store(2, Ordering::Relaxed);
        assert!(pool
            .select_with_preference_bounded(Some(1), 8, Some(2), &no_avoid)
            .is_none());
        assert!(pool
            .select_with_preference_bounded(None, 8, Some(2), &no_avoid)
            .is_none());
        // Without a bound the same state still selects.
        assert!(pool
            .select_with_preference_bounded(None, 8, None, &no_avoid)
            .is_some());
    }

    #[test]
    fn test_avoided_backends_are_skipped_and_pins_move_off_them() {
        let pool = two_backends();
        let avoid_b1 = |index: usize| index == 0;
        let sel = pool
            .select_with_preference_bounded(Some(0), 8, None, &avoid_b1)
            .unwrap();
        assert_eq!(sel.index, 1);
        assert_eq!(sel.outcome, SelectionOutcome::Rebalanced);
        drop(sel);
        assert!(pool
            .select_with_preference_bounded(None, 8, None, &|_| true)
            .is_none());
        let single = BackendPool::new(vec!["http://only:8000".to_string()]);
        assert!(single
            .select_with_preference_bounded(None, 0, None, &|_| true)
            .is_none());
    }

    #[test]
    fn test_bounded_selection_single_backend() {
        let pool = BackendPool::new(vec!["http://only:8000".to_string()]);
        pool.backends()[0].active_conns.store(3, Ordering::Relaxed);
        assert!(pool
            .select_with_preference_bounded(None, 0, Some(3), &no_avoid)
            .is_none());
        let sel = pool
            .select_with_preference_bounded(None, 0, Some(4), &no_avoid)
            .unwrap();
        assert_eq!(sel.outcome, SelectionOutcome::Single);
        assert_eq!(pool.backends()[0].active_conns.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_select_excluding_picks_another_healthy_backend() {
        let pool = two_backends();
        assert_eq!(pool.select_excluding(0, None, &no_avoid).unwrap().index, 1);
        assert_eq!(pool.select_excluding(1, None, &no_avoid).unwrap().index, 0);
        pool.backends()[1].healthy.store(false, Ordering::Relaxed);
        assert!(pool.select_excluding(0, None, &no_avoid).is_none());
        assert_eq!(pool.healthy_count(), 1);
        pool.backends()[1].healthy.store(true, Ordering::Relaxed);
        pool.backends()[1].active_conns.store(2, Ordering::Relaxed);
        assert!(pool.select_excluding(0, Some(2), &no_avoid).is_none());
        assert!(pool
            .select_excluding(0, None, &|index| index == 1)
            .is_none());
        let single = BackendPool::new(vec!["http://only:8000".to_string()]);
        assert!(single.select_excluding(0, None, &no_avoid).is_none());
    }

    #[test]
    fn test_bounded_reservation_never_overshoots_under_concurrency() {
        let pool = Arc::new(two_backends());
        let bound = 3u32;
        let overshoot = Arc::new(AtomicBool::new(false));
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let pool = pool.clone();
                let overshoot = overshoot.clone();
                std::thread::spawn(move || {
                    for _ in 0..500 {
                        if let Some(sel) =
                            pool.select_with_preference_bounded(None, 8, Some(bound), &no_avoid)
                        {
                            if sel.backend.active_conns.load(Ordering::Relaxed) > bound {
                                overshoot.store(true, Ordering::Relaxed);
                            }
                            std::hint::spin_loop();
                            drop(sel);
                        }
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        assert!(
            !overshoot.load(Ordering::Relaxed),
            "a backend exceeded its share"
        );
        assert_eq!(pool.backends()[0].active_conns.load(Ordering::Relaxed), 0);
        assert_eq!(pool.backends()[1].active_conns.load(Ordering::Relaxed), 0);
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
