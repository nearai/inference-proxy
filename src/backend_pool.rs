use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tracing::{debug, info, warn};

use crate::context_tier::ContextTier;

/// A single backend instance (e.g., one vLLM process).
pub struct Backend {
    pub base_url: String,
    /// Which context tier this backend serves. Everything is `Base` unless
    /// `VLLM_BACKEND_LONG_CONTEXT_URLS` is configured (see `context_tier.rs`).
    pub tier: ContextTier,
    pub healthy: AtomicBool,
    /// Every request placed on this backend (least-connections signal).
    pub active_conns: AtomicU32,
    /// Only budgeted lane requests (the admission per-host share is applied
    /// to this, so embeddings, images, probes and the like cannot fill it).
    pub lane_conns: AtomicU32,
    pub consecutive_failures: AtomicU32,
}

impl Backend {
    fn new(base_url: String, tier: ContextTier) -> Self {
        Self {
            base_url,
            tier,
            healthy: AtomicBool::new(true),
            active_conns: AtomicU32::new(0),
            lane_conns: AtomicU32::new(0),
            consecutive_failures: AtomicU32::new(0),
        }
    }

    /// Whether this backend may serve a request restricted to `tier`
    /// (`None` = no restriction).
    fn in_tier(&self, tier: Option<ContextTier>) -> bool {
        tier.is_none_or(|tier| tier == self.tier)
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
            .field("tier", &self.tier)
            .field("healthy", &self.healthy.load(Ordering::Relaxed))
            .field("active_conns", &self.active_conns.load(Ordering::Relaxed))
            .field("lane_conns", &self.lane_conns.load(Ordering::Relaxed))
            .finish()
    }
}

/// RAII guard that decrements the connection counters on drop.
pub struct BackendGuard {
    backend: Arc<Backend>,
    /// The slot was reserved on `lane_conns` too (a budgeted lane request).
    lane: bool,
}

impl BackendGuard {
    pub fn new(backend: Arc<Backend>) -> Self {
        backend.active_conns.fetch_add(1, Ordering::Relaxed);
        Self {
            backend,
            lane: false,
        }
    }

    /// Wrap a slot that `BackendPool::reserve` already took on both
    /// `lane_conns` and `active_conns` (no second increment).
    fn reserved(backend: Arc<Backend>) -> Self {
        Self {
            backend,
            lane: true,
        }
    }

    pub fn backend(&self) -> &Arc<Backend> {
        &self.backend
    }
}

impl Drop for BackendGuard {
    fn drop(&mut self) {
        self.backend.active_conns.fetch_sub(1, Ordering::Relaxed);
        if self.lane {
            self.backend.lane_conns.fetch_sub(1, Ordering::Relaxed);
        }
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

/// What a placement may consider besides health and least-connections.
#[derive(Clone, Copy)]
pub struct Policy<'a> {
    /// Per-backend bound on lane requests in flight (`None` = unbounded).
    pub max_conns: Option<u32>,
    /// Destination-specific bounds override the uniform limit.
    pub max_conns_by_backend: Option<&'a [u32]>,
    /// Estimated request tier, retained when destination restrictions fall back.
    pub requested_tier: Option<ContextTier>,
    /// Backends to steer around (recent engine rejection, non-empty engine
    /// queue).
    pub avoid: &'a (dyn Fn(usize) -> bool + Sync),
    /// Fresh engine view `(running, queued)` when the engines are polled
    /// (`engine_load.rs`). It ranks backends for new conversations and keeps a
    /// pinned conversation where it is regardless of running counts: a host
    /// with a free batch slot serves the cached prefix at once, and only a
    /// queueing host (via `avoid`) moves it.
    pub engine: &'a (dyn Fn(usize) -> Option<(u32, u32)> + Sync),
    /// Only backends of this context tier are candidates (`None` = the whole
    /// pool; see `context_tier.rs`).
    pub tier: Option<ContextTier>,
}

fn never(_: usize) -> bool {
    false
}

fn unknown(_: usize) -> Option<(u32, u32)> {
    None
}

impl Policy<'_> {
    fn limit(&self, index: usize) -> Option<u32> {
        self.max_conns_by_backend
            .map(|limits| limits.get(index).copied().unwrap_or(0))
            .or(self.max_conns)
    }

    fn bounded(&self) -> bool {
        self.max_conns.is_some() || self.max_conns_by_backend.is_some()
    }
}

impl Policy<'static> {
    /// Health and least-connections only (admission off, no engine polling).
    pub const NONE: Policy<'static> = Policy {
        max_conns: None,
        max_conns_by_backend: None,
        requested_tier: None,
        avoid: &never,
        engine: &unknown,
        tier: None,
    };
}

/// Minimum attempts to turn a pick into a reservation before giving up; a
/// lost race (another request took the last slot under the bound) picks
/// again, and a burst can lose once per backend, so the pool scales this.
const RESERVE_ATTEMPTS: usize = 16;

/// Pool of backends for the same model, with least-connections selection.
pub struct BackendPool {
    backends: Vec<Arc<Backend>>,
    /// `Some(Base)` once a long-context tier exists: traffic that carries no
    /// tier of its own (tokenize, media, models, health probes) stays off the
    /// long hosts, which are idle by design and would otherwise attract all
    /// of it. `None` for an untiered pool.
    untiered: Option<ContextTier>,
}

impl BackendPool {
    pub fn new(base_urls: Vec<String>) -> Self {
        Self::with_long_context(base_urls, Vec::new())
    }

    /// Pool with a long-context tier appended after the base backends
    /// (`VLLM_BACKEND_LONG_CONTEXT_URLS`): the base backends keep their
    /// indexes, so everything keyed on them — back-pressure slots, engine
    /// probes, affinity assignments — is unaffected by the tier's existence.
    pub fn with_long_context(base_urls: Vec<String>, long_urls: Vec<String>) -> Self {
        assert!(!base_urls.is_empty(), "at least one backend URL required");
        let untiered = (!long_urls.is_empty()).then_some(ContextTier::Base);
        let backends = base_urls
            .into_iter()
            .map(|u| Arc::new(Backend::new(u, ContextTier::Base)))
            .chain(
                long_urls
                    .into_iter()
                    .map(|u| Arc::new(Backend::new(u, ContextTier::Long))),
            )
            .collect();
        Self { backends, untiered }
    }

    /// Number of backends in the pool.
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Whether the pool is empty (always false — constructor requires ≥1 backend).
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }

    /// Select a backend using least-connections among healthy backends of the
    /// untiered half of the pool (see `untiered`). If none is healthy, picks
    /// the least-loaded backend of the whole pool anyway — a healthy long host
    /// beats a dead base one when the fleet is down, and `/healthz` must not
    /// report 503 while the gateway still answers through the fallback. No
    /// slot is reserved; callers create their own `BackendGuard`.
    pub fn select(&self) -> Arc<Backend> {
        let conns = |_: usize, backend: &Backend| backend.active_conns.load(Ordering::Relaxed);
        let index = self
            .least_index(
                |_, backend| {
                    backend.in_tier(self.untiered) && backend.healthy.load(Ordering::Relaxed)
                },
                conns,
            )
            .unwrap_or_else(|| {
                self.least_index(|_, _| true, conns)
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
        self.reserve(preferred, max_imbalance, &Policy::NONE, true)
            .expect("unbounded selection always yields a backend")
    }

    /// `select_with_preference` under a `Policy`: only backends that are not
    /// avoided and have fewer than `max_conns` lane requests in flight (the
    /// admission per-host share, see `admission.rs`), ranked by the engine
    /// view when polled. The slot is taken atomically, so concurrent
    /// selections cannot overshoot the bound. A preferred backend that is
    /// avoided or at the bound is treated like an overloaded one and the turn
    /// is rebalanced; `None` when no backend is eligible.
    pub fn select_with_preference_bounded(
        &self,
        preferred: Option<usize>,
        max_imbalance: u32,
        policy: &Policy<'_>,
    ) -> Option<Selection> {
        // Without a bound (admission off) keep the legacy degradation when
        // every backend is unhealthy; with one, refusing is the point.
        self.reserve(preferred, max_imbalance, policy, !policy.bounded())
    }

    /// Least-loaded eligible backend other than `excluded` (connection
    /// fail-over), under the same policy as `select_with_preference_bounded`;
    /// `None` when there is none.
    pub fn select_excluding(&self, excluded: usize, policy: &Policy<'_>) -> Option<Selection> {
        if self.backends.len() == 1 {
            return None;
        }
        let avoid = |index: usize| index == excluded || (policy.avoid)(index);
        let policy = Policy {
            avoid: &avoid,
            ..*policy
        };
        self.reserve(None, 0, &policy, false)
    }

    /// Whether some backend other than `index` is healthy (fail-over could
    /// go somewhere, share and avoidance aside).
    pub fn has_healthy_other_than(&self, index: usize) -> bool {
        self.backends
            .iter()
            .enumerate()
            .any(|(i, b)| i != index && b.healthy.load(Ordering::Relaxed))
    }

    /// Number of backends currently marked healthy.
    pub fn healthy_count(&self) -> usize {
        self.healthy_count_in(None)
    }

    /// Number of healthy backends in `tier` (`None` = the whole pool). By
    /// default a tier with none is why a request falls back to the other one
    /// instead of waiting for a host that is not there; with
    /// `VLLM_BACKEND_TIER_STRICT` it is refused instead (`context_tier.rs`).
    pub fn healthy_count_in(&self, tier: Option<ContextTier>) -> usize {
        self.backends
            .iter()
            .filter(|b| b.in_tier(tier) && b.healthy.load(Ordering::Relaxed))
            .count()
    }

    /// Pick, then take the slot with a compare-and-swap so the bound holds
    /// under concurrency; a lost race picks again.
    fn reserve(
        &self,
        preferred: Option<usize>,
        max_imbalance: u32,
        policy: &Policy<'_>,
        degrade_when_all_unhealthy: bool,
    ) -> Option<Selection> {
        let attempts = (self.backends.len() * 4).max(RESERVE_ATTEMPTS);
        for _ in 0..attempts {
            let Some((index, outcome)) =
                self.pick(preferred, max_imbalance, policy, degrade_when_all_unhealthy)
            else {
                self.record_selection_refusal(policy, None);
                return None;
            };
            let backend = &self.backends[index];
            let taken =
                backend
                    .lane_conns
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |conns| {
                        policy
                            .limit(index)
                            .is_none_or(|max| conns < max)
                            .then(|| conns.saturating_add(1))
                    });
            if taken.is_ok() {
                backend.active_conns.fetch_add(1, Ordering::Relaxed);
                return Some(Selection {
                    backend: backend.clone(),
                    index,
                    outcome,
                    guard: BackendGuard::reserved(backend.clone()),
                });
            }
        }
        self.record_selection_refusal(policy, Some("reservation_contention"));
        None
    }

    fn record_selection_refusal(&self, policy: &Policy<'_>, reason: Option<&'static str>) {
        if !policy.bounded() {
            return;
        }
        let mut healthy = false;
        let mut limited = false;
        let mut avoided = false;
        for (index, backend) in self.backends.iter().enumerate() {
            if !backend.in_tier(policy.tier) || !backend.healthy.load(Ordering::Relaxed) {
                continue;
            }
            healthy = true;
            limited |= policy
                .limit(index)
                .is_some_and(|max| backend.lane_conns.load(Ordering::Relaxed) >= max);
            avoided |= (policy.avoid)(index);
        }
        let reason = reason.unwrap_or(match (healthy, limited, avoided) {
            (false, _, _) => "no_healthy",
            (_, true, true) => "mixed",
            (_, true, false) => "host_limit",
            (_, false, true) => "backpressure",
            _ => "reservation_contention",
        });
        metrics::counter!("admission_selection_failures_total", "requested_tier" => policy.requested_tier.map_or("base", ContextTier::as_str), "tier" => policy.tier.map_or("fallback", ContextTier::as_str), "reason" => reason).increment(1);
    }

    /// The selection policy on a snapshot of the counters (no reservation).
    fn pick(
        &self,
        preferred: Option<usize>,
        max_imbalance: u32,
        policy: &Policy<'_>,
        degrade_when_all_unhealthy: bool,
    ) -> Option<(usize, SelectionOutcome)> {
        let avoid = policy.avoid;
        let under_bound = |index: usize, backend: &Backend| {
            policy
                .limit(index)
                .is_none_or(|max| backend.lane_conns.load(Ordering::Relaxed) < max)
        };
        // Engine view when polled (running + queued), the gateway's own
        // connection count otherwise.
        let load = |index: usize, backend: &Backend| {
            (policy.engine)(index)
                .map(|(running, queued)| running.saturating_add(queued))
                .unwrap_or_else(|| backend.active_conns.load(Ordering::Relaxed))
        };
        if self.backends.len() == 1 {
            let only = &self.backends[0];
            let usable = only.healthy.load(Ordering::Relaxed) || degrade_when_all_unhealthy;
            return (usable && !avoid(0) && under_bound(0, only) && only.in_tier(policy.tier))
                .then_some((0, SelectionOutcome::Single));
        }

        let eligible = |index: usize, backend: &Backend| {
            !avoid(index)
                && backend.healthy.load(Ordering::Relaxed)
                && under_bound(index, backend)
                && backend.in_tier(policy.tier)
        };
        let least_index = match self.least_index(eligible, load) {
            Some(index) => index,
            // Everything unhealthy: degrade to the least-loaded backend that is
            // not avoided rather than refusing every request (a probe will
            // re-mark them). Never for a bounded or fail-over selection, and
            // never inside a tier restriction — a host known to be down is
            // worse than the other tier, which the caller reaches by placing
            // again without the restriction.
            None if degrade_when_all_unhealthy && policy.tier.is_none() => self.least_index(
                |index, backend| !avoid(index) && under_bound(index, backend),
                load,
            )?,
            None => return None,
        };
        let Some(preferred_index) = preferred.filter(|index| *index < self.backends.len()) else {
            return Some((least_index, SelectionOutcome::New));
        };

        let candidate = &self.backends[preferred_index];
        if !candidate.healthy.load(Ordering::Relaxed) {
            return Some((least_index, SelectionOutcome::Unhealthy));
        }
        if !eligible(preferred_index, candidate) {
            return Some((least_index, SelectionOutcome::Rebalanced));
        }
        // With the engine view the pin holds: a host that is not queueing
        // serves the cached prefix at once, whatever its running count. Only
        // the gateway-count fallback applies the imbalance bound.
        let within_bound = (policy.engine)(preferred_index).is_some()
            || candidate.active_conns.load(Ordering::Relaxed)
                <= self.backends[least_index]
                    .active_conns
                    .load(Ordering::Relaxed)
                    .saturating_add(max_imbalance);
        if within_bound {
            Some((preferred_index, SelectionOutcome::Pinned))
        } else {
            Some((least_index, SelectionOutcome::Rebalanced))
        }
    }

    /// Index of the least-loaded backend among those `eligible` accepts.
    fn least_index(
        &self,
        eligible: impl Fn(usize, &Backend) -> bool,
        load: impl Fn(usize, &Backend) -> u32,
    ) -> Option<usize> {
        self.backends
            .iter()
            .enumerate()
            .filter(|(index, backend)| eligible(*index, backend))
            .min_by_key(|(index, backend)| load(*index, backend))
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
        set_conns(&pool, 0, 5);
        set_conns(&pool, 1, 1);

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
        set_conns(&pool, 0, 3);
        set_conns(&pool, 1, 1);

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
        set_conns(&pool, 0, 4);
        set_conns(&pool, 1, 1);

        // 4 <= 1 + 8: the preferred backend keeps the request.
        let selection = pool.select_with_preference(Some(0), 8);
        assert_eq!(selection.index, 0);
        assert_eq!(selection.outcome, SelectionOutcome::Pinned);
        assert_eq!(selection.backend.base_url, "http://b1:8000");
    }

    #[test]
    fn test_preference_rebalances_beyond_imbalance_bound() {
        let pool = two_backends();
        set_conns(&pool, 0, 4);
        set_conns(&pool, 1, 1);

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
        set_conns(&pool, 0, 2);

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

    fn bounded(max_conns: u32) -> Policy<'static> {
        Policy {
            max_conns: Some(max_conns),
            ..Policy::NONE
        }
    }

    fn avoiding(avoid: &(dyn Fn(usize) -> bool + Sync)) -> Policy<'_> {
        Policy {
            avoid,
            ..Policy::NONE
        }
    }

    /// Simulate `n` requests on backend `index` (both counters).
    fn set_conns(pool: &BackendPool, index: usize, n: u32) {
        pool.backends()[index]
            .active_conns
            .store(n, Ordering::Relaxed);
        pool.backends()[index]
            .lane_conns
            .store(n, Ordering::Relaxed);
    }

    #[test]
    fn test_bounded_selection_skips_backends_at_the_share() {
        let pool = two_backends();
        set_conns(&pool, 0, 2);
        set_conns(&pool, 1, 1);
        // Share 2: only b2 has room, and the reservation is taken on it.
        let sel = pool
            .select_with_preference_bounded(None, 8, &bounded(2))
            .unwrap();
        assert_eq!(sel.index, 1);
        assert_eq!(pool.backends()[1].active_conns.load(Ordering::Relaxed), 2);
        drop(sel);
        assert_eq!(pool.backends()[1].active_conns.load(Ordering::Relaxed), 1);
        // A pin on the full backend is rebalanced even within the imbalance bound.
        let sel = pool
            .select_with_preference_bounded(Some(0), 8, &bounded(2))
            .unwrap();
        assert_eq!(sel.index, 1);
        assert_eq!(sel.outcome, SelectionOutcome::Rebalanced);
        drop(sel);
        // Everyone at the share: nothing to pick.
        set_conns(&pool, 1, 2);
        assert!(pool
            .select_with_preference_bounded(Some(1), 8, &bounded(2))
            .is_none());
        assert!(pool
            .select_with_preference_bounded(None, 8, &bounded(2))
            .is_none());
        // Without a bound the same state still selects.
        assert!(pool
            .select_with_preference_bounded(None, 8, &Policy::NONE)
            .is_some());
    }

    #[test]
    fn test_avoided_backends_are_skipped_and_pins_move_off_them() {
        let pool = two_backends();
        let avoid_b1 = |index: usize| index == 0;
        let sel = pool
            .select_with_preference_bounded(Some(0), 8, &avoiding(&avoid_b1))
            .unwrap();
        assert_eq!(sel.index, 1);
        assert_eq!(sel.outcome, SelectionOutcome::Rebalanced);
        drop(sel);
        assert!(pool
            .select_with_preference_bounded(None, 8, &avoiding(&|_| true))
            .is_none());
        let single = BackendPool::new(vec!["http://only:8000".to_string()]);
        assert!(single
            .select_with_preference_bounded(None, 0, &avoiding(&|_| true))
            .is_none());
    }

    #[test]
    fn test_bounded_selection_single_backend() {
        let pool = BackendPool::new(vec!["http://only:8000".to_string()]);
        set_conns(&pool, 0, 3);
        assert!(pool
            .select_with_preference_bounded(None, 0, &bounded(3))
            .is_none());
        let sel = pool
            .select_with_preference_bounded(None, 0, &bounded(4))
            .unwrap();
        assert_eq!(sel.outcome, SelectionOutcome::Single);
        assert_eq!(pool.backends()[0].active_conns.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_select_excluding_picks_another_healthy_backend() {
        let pool = two_backends();
        assert_eq!(pool.select_excluding(0, &Policy::NONE).unwrap().index, 1);
        assert_eq!(pool.select_excluding(1, &Policy::NONE).unwrap().index, 0);
        pool.backends()[1].healthy.store(false, Ordering::Relaxed);
        assert!(pool.select_excluding(0, &Policy::NONE).is_none());
        assert_eq!(pool.healthy_count(), 1);
        pool.backends()[1].healthy.store(true, Ordering::Relaxed);
        set_conns(&pool, 1, 2);
        assert!(pool.select_excluding(0, &bounded(2)).is_none());
        assert!(pool
            .select_excluding(0, &avoiding(&|index| index == 1))
            .is_none());
        let single = BackendPool::new(vec!["http://only:8000".to_string()]);
        assert!(single.select_excluding(0, &Policy::NONE).is_none());
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
                            pool.select_with_preference_bounded(None, 8, &bounded(bound))
                        {
                            if sel.backend.lane_conns.load(Ordering::Relaxed) > bound {
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
    fn test_reservation_counts_lane_and_total_separately() {
        let pool = two_backends();
        // Non-lane traffic (a plain guard) does not consume the share.
        let plain = BackendGuard::new(pool.backends()[0].clone());
        let plain2 = BackendGuard::new(pool.backends()[0].clone());
        let sel = pool
            .select_with_preference_bounded(Some(0), 8, &bounded(1))
            .unwrap();
        assert_eq!(sel.index, 0, "share applies to lane requests only");
        assert_eq!(pool.backends()[0].lane_conns.load(Ordering::Relaxed), 1);
        assert_eq!(pool.backends()[0].active_conns.load(Ordering::Relaxed), 3);
        drop(sel);
        assert_eq!(pool.backends()[0].lane_conns.load(Ordering::Relaxed), 0);
        assert_eq!(pool.backends()[0].active_conns.load(Ordering::Relaxed), 2);
        drop((plain, plain2));
        assert_eq!(pool.backends()[0].active_conns.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_bounded_single_backend_honors_health() {
        let pool = BackendPool::new(vec!["http://only:8000".to_string()]);
        pool.backends()[0].healthy.store(false, Ordering::Relaxed);
        assert!(pool
            .select_with_preference_bounded(None, 0, &bounded(4))
            .is_none());
        // Admission off: the legacy degradation still serves it.
        assert_eq!(pool.select_with_preference(None, 0).index, 0);
        assert!(!pool.has_healthy_other_than(0));
    }

    #[test]
    fn test_a_burst_across_a_large_pool_finds_every_free_slot() {
        let pool = Arc::new(BackendPool::new(
            (0..40).map(|i| format!("http://b{i}:8000")).collect(),
        ));
        let barrier = Arc::new(std::sync::Barrier::new(40));
        let handles: Vec<_> = (0..40)
            .map(|_| {
                let pool = pool.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    pool.select_with_preference_bounded(None, 8, &bounded(1))
                })
            })
            .collect();
        let selections: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert!(
            selections.iter().all(|s| s.is_some()),
            "40 requests, 40 backends with share 1: nobody should be refused"
        );
        let mut indexes: Vec<_> = selections
            .iter()
            .map(|s| s.as_ref().unwrap().index)
            .collect();
        indexes.sort_unstable();
        indexes.dedup();
        assert_eq!(indexes.len(), 40);
    }

    #[test]
    fn test_engine_view_ranks_new_conversations_and_keeps_pins() {
        let pool = two_backends();
        // Gateway counts favor b1 (0 vs 3), the engines say b1 is the busy one.
        set_conns(&pool, 1, 3);
        let engine = |index: usize| Some(if index == 0 { (40, 0) } else { (5, 0) });
        let policy = Policy {
            engine: &engine,
            ..Policy::NONE
        };
        let sel = pool
            .select_with_preference_bounded(None, 8, &policy)
            .unwrap();
        assert_eq!(
            sel.index, 1,
            "a new conversation goes to the least-loaded engine"
        );
        drop(sel);
        // A conversation pinned to the busy host stays: it is not queueing, and
        // running counts never move a pin (the gateway-count bound would have).
        let sel = pool
            .select_with_preference_bounded(Some(0), 0, &policy)
            .unwrap();
        assert_eq!((sel.index, sel.outcome), (0, SelectionOutcome::Pinned));
        drop(sel);
        // Without the engine view the gateway-count bound still applies.
        set_conns(&pool, 0, 9);
        set_conns(&pool, 1, 0);
        let sel = pool
            .select_with_preference_bounded(Some(0), 8, &Policy::NONE)
            .unwrap();
        assert_eq!(sel.outcome, SelectionOutcome::Rebalanced);
    }

    fn tiered() -> BackendPool {
        BackendPool::with_long_context(
            vec!["http://b1:8000".to_string(), "http://b2:8000".to_string()],
            vec!["http://long:8000".to_string()],
        )
    }

    fn in_tier(tier: ContextTier) -> Policy<'static> {
        Policy {
            tier: Some(tier),
            ..Policy::NONE
        }
    }

    #[test]
    fn test_untiered_traffic_stays_off_the_long_hosts() {
        let pool = tiered();
        // Least-connections alone would send tokenize, media and health
        // probes to the idle long host; they stay on the base fleet.
        set_conns(&pool, 0, 4);
        set_conns(&pool, 1, 4);
        assert_eq!(pool.select().base_url, "http://b1:8000");
        pool.backends()[0].healthy.store(false, Ordering::Relaxed);
        pool.backends()[1].healthy.store(false, Ordering::Relaxed);
        assert_eq!(
            pool.select().base_url,
            "http://long:8000",
            "with the base fleet down, a healthy long host beats a dead base one"
        );
        // An untiered pool is unaffected.
        let plain = two_backends();
        set_conns(&plain, 0, 4);
        assert_eq!(plain.select().base_url, "http://b2:8000");
    }

    #[test]
    fn test_selection_stays_inside_the_requested_tier() {
        let pool = tiered();
        assert_eq!(
            pool.len(),
            3,
            "the long tier is appended, base indexes hold"
        );
        assert_eq!(pool.healthy_count(), 3);
        assert_eq!(pool.healthy_count_in(Some(ContextTier::Long)), 1);

        let sel = pool
            .select_with_preference_bounded(None, 8, &in_tier(ContextTier::Long))
            .unwrap();
        assert_eq!(sel.index, 2);
        drop(sel);
        // A conversation pinned on a base host is no pin for a long request:
        // it is placed fresh in its tier (and `place` re-pins it there).
        let sel = pool
            .select_with_preference_bounded(Some(0), 8, &in_tier(ContextTier::Long))
            .unwrap();
        assert_eq!((sel.index, sel.outcome), (2, SelectionOutcome::Rebalanced));
        drop(sel);
        // A base request never lands on the long host, however idle it is.
        let held: Vec<_> = (0..4)
            .map(|_| {
                pool.select_with_preference_bounded(None, 8, &in_tier(ContextTier::Base))
                    .unwrap()
            })
            .collect();
        assert!(held.iter().all(|sel| sel.index < 2));
        drop(held);
        // The long tier at its share is a refusal, not a spill onto the base
        // fleet: keeping the big prefills off it is the whole point.
        set_conns(&pool, 2, 2);
        let full = Policy {
            max_conns: Some(2),
            ..in_tier(ContextTier::Long)
        };
        assert!(pool
            .select_with_preference_bounded(None, 8, &full)
            .is_none());
        // Fail-over picks another host of the same tier, or none.
        assert!(pool
            .select_excluding(2, &in_tier(ContextTier::Long))
            .is_none());
        assert_eq!(
            pool.select_excluding(0, &in_tier(ContextTier::Base))
                .unwrap()
                .index,
            1
        );
    }

    #[test]
    fn test_a_restricted_tier_never_degrades_onto_an_unhealthy_host() {
        let pool = tiered();
        pool.backends()[2].healthy.store(false, Ordering::Relaxed);
        // Admission off (no share bound), so selection may degrade onto an
        // unhealthy host — but not inside a tier: the caller places again
        // without the restriction and reaches the healthy base fleet instead
        // of a host it knows is down.
        assert!(pool
            .select_with_preference_bounded(None, 8, &in_tier(ContextTier::Long))
            .is_none());
        assert!(pool
            .select_with_preference_bounded(Some(2), 8, &in_tier(ContextTier::Long))
            .is_none());
        assert_eq!(
            pool.select_with_preference_bounded(None, 8, &Policy::NONE)
                .unwrap()
                .index,
            0
        );
        // With the whole pool down and nothing restricted, degrading is still
        // better than refusing everything.
        for backend in pool.backends() {
            backend.healthy.store(false, Ordering::Relaxed);
        }
        assert!(pool
            .select_with_preference_bounded(None, 8, &Policy::NONE)
            .is_some());
    }

    #[test]
    fn test_select_url_builds_correct_url() {
        let pool = BackendPool::new(vec!["http://b1:8000".to_string()]);
        let (url, _guard) = pool.select_url("/v1/chat/completions");
        assert_eq!(url, "http://b1:8000/v1/chat/completions");
    }

    #[test]
    fn test_backend_url_handles_trailing_slash() {
        let b = Backend::new("http://b1:8000/".to_string(), ContextTier::Base);
        assert_eq!(b.url("/v1/models"), "http://b1:8000/v1/models");
        assert_eq!(b.url(""), "http://b1:8000");
    }
}
