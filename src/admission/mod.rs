//! Lane admission for gateway mode (`VLLM_PROXY_ADMISSION_*`).
//!
//! The gateway fronts a fixed set of inference hosts that also serve direct
//! traffic. Under overload its job is to refuse *early* — a 429 with
//! `Retry-After` before anything is sent upstream — instead of queueing and
//! eventually failing with a 5xx or a minutes-long time to first token. Three
//! checks run on every chat/completions request, in this order:
//!
//! 1. **Observed overload.** Over the last minute, at least 20 lane requests
//!    reached the engine and 5 % of them (at least two) waited longer than the
//!    configured bound for their first generation event; or every healthy
//!    backend answered an engine admission rejection ("The request queue is
//!    full.", "aborted by a higher priority request") within the last few
//!    seconds. Either means the engines are saturated for lane traffic, so new
//!    work is refused until the signal ages out. A backend that rejected
//!    recently is also steered around at selection time while others have
//!    room.
//! 2. **Global in-flight budget.** At most `budget` lane requests are in
//!    flight across the fleet. The budget starts at `start_inflight` and grows
//!    by `ramp_step` every `ramp_interval` up to `max_inflight`, but only after
//!    an interval without any overload signal.
//! 3. **Per-host share.** By default `ceil(budget / healthy backends)` in flight per
//!    backend, so a conversation-affinity pin cannot pile the whole budget onto
//!    one host. Selection (`BackendPool::select_with_preference_bounded`)
//!    reserves the slot atomically and only on backends under their share: a
//!    pinned conversation moves when its host is full, and only when no host
//!    has room is the request refused.
//!
//! The controller is a facade over `TtftBreaker` (1), `BackendSaturation`
//! (1, and placement steering) and `Budget` (2, 3); an overload signal from
//! the first two is counted against the budget's ramp here, so the
//! components never reference each other. `precheck` runs once per request,
//! early; `try_admit` re-runs only the re-runnable part
//! (`check_signals_and_budget_at`) before reserving a slot.
//!
//! Opt-in tier borrowing uses configured counts instead: base hosts may each
//! use ceil(budget / base hosts); long hosts keep a separate ceiling. See
//! `backend_limits`. By default the global budget is shared, with no reserved
//! tier slots: a flood of short requests can fill it and long-context requests
//! are then refused with the rest. `long_reserved_inflight` keeps that many
//! slots of the budget for requests bound for the long tier: everything else
//! is admitted only while fewer than `budget - reserve` such requests are in
//! flight, and the base hosts share that smaller number.
//!
//! Everything is derived from what the gateway observes itself; no admin
//! endpoint or extra token is involved. Disabled (`max_inflight = 0`) the
//! module is inert and the in-CVM behavior is unchanged.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::debug;

use crate::backend_pool::BackendPool;
use crate::context_tier::{ContextTier, TierDecision};
use crate::engine_load::EngineLoad;

mod budget;
mod permit;
mod saturation;
mod ttft;
use budget::Budget;
pub use permit::Permit;
use saturation::BackendSaturation;
use ttft::TtftBreaker;
pub use ttft::{TTFT_BREACH_FRACTION, TTFT_MIN_BREACHES, TTFT_MIN_SAMPLES, TTFT_WINDOW};

/// Operator settings, parsed and validated by `Config::from_env`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionConfig {
    /// Hard ceiling on lane requests in flight across the fleet.
    pub max_inflight: u32,
    pub tier_borrowing: bool,
    pub long_max_inflight_per_host: u32,
    /// Budget slots kept for requests bound for the long tier
    /// (`VLLM_PROXY_ADMISSION_LONG_RESERVED_INFLIGHT`, default 0: none).
    /// Requests that are not are admitted only while fewer than
    /// `budget - long_reserved_inflight` of them are in flight.
    pub long_reserved_inflight: u32,
    /// Budget at start-up; ramps toward `max_inflight`.
    pub start_inflight: u32,
    /// Budget increase per clean ramp interval.
    pub ramp_step: u32,
    /// Length of a ramp interval.
    pub ramp_interval: Duration,
    /// Refuse new work while enough of the window waited longer than this for
    /// the first generation event (`None` = no TTFT check).
    pub ttft_p95_max: Option<Duration>,
    /// How long an engine admission rejection counts against its backend.
    pub backpressure_ttl: Duration,
    /// Engine-reported queue depth at or above which a backend counts as
    /// saturated (`VLLM_PROXY_ADMISSION_QUEUE_SATURATED_AT`, default 1: a
    /// queue of at least one request).
    pub queue_saturated_at: u32,
    /// `Retry-After` value on every refusal.
    pub retry_after: Duration,
}

/// Why a request was refused. The label of `admission_rejections_total`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// The global in-flight budget is used up.
    Budget,
    /// The request is not bound for the long tier and what is left of the
    /// budget is reserved for it (`long_reserved_inflight`).
    LongReserve,
    /// Every backend is at its per-host share (or steered around).
    HostShare,
    /// Every healthy backend recently rejected at engine admission.
    BackendQueue,
    /// The lane's own time-to-first-generation is above the bound.
    Ttft,
    /// Strict context tiers (`VLLM_BACKEND_TIER_STRICT`): the request's tier
    /// has no healthy backend at all, and strict mode refuses rather than
    /// placing it on the other tier. See `context_tier.rs`.
    TierUnavailable,
}

impl RejectReason {
    pub fn as_str(self) -> &'static str {
        match self {
            RejectReason::Budget => "budget",
            RejectReason::LongReserve => "long_reserve",
            RejectReason::HostShare => "host_share",
            RejectReason::BackendQueue => "backend_queue",
            RejectReason::Ttft => "ttft",
            RejectReason::TierUnavailable => "tier_unavailable",
        }
    }
}

/// A refusal, ready to become a 429.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rejected {
    pub reason: RejectReason,
    pub retry_after: Duration,
}

/// Fleet-wide admission state shared by every request (`AppState.admission`).
pub struct AdmissionController {
    config: Option<AdmissionConfig>,
    budget: Budget,
    ttft: TtftBreaker,
    saturation: BackendSaturation,
}

impl AdmissionController {
    /// `backend_count` sizes the per-backend back-pressure slots; it must be
    /// the pool size (backend indexes are stable for the process lifetime).
    pub fn new(
        config: Option<AdmissionConfig>,
        backend_count: usize,
        engine: Arc<EngineLoad>,
    ) -> Self {
        let now = Instant::now();
        let budget = config.as_ref().map_or(0, |c| c.start_inflight);
        if let Some(config) = &config {
            metrics::gauge!("admission_budget").set(f64::from(budget));
            metrics::gauge!("admission_inflight").set(0.0);
            metrics::gauge!("admission_inflight_base").set(0.0);
            metrics::gauge!("admission_long_reserve").set(f64::from(config.long_reserved_inflight));
            debug_assert!(config.start_inflight <= config.max_inflight);
        }
        let reserve = config.as_ref().map_or(0, |c| c.long_reserved_inflight);
        Self {
            config,
            budget: Budget::new(budget, reserve, now),
            ttft: TtftBreaker::new(now),
            saturation: BackendSaturation::new(backend_count, engine, now),
        }
    }

    /// An inert controller: every request is admitted, nothing is counted.
    pub fn disabled() -> Self {
        Self::new(None, 0, Arc::new(EngineLoad::disabled()))
    }

    /// Fresh engine view `(running, queued)` for backend `index`, when polled.
    pub fn engine(&self, index: usize) -> Option<(u32, u32)> {
        self.saturation.engine(index)
    }

    pub fn is_enabled(&self) -> bool {
        self.config.is_some()
    }

    pub fn config(&self) -> Option<&AdmissionConfig> {
        self.config.as_ref()
    }

    /// Current effective budget (0 when disabled).
    pub fn budget(&self) -> u32 {
        self.budget.limit()
    }

    /// Lane requests currently in flight.
    pub fn inflight(&self) -> u32 {
        self.budget.inflight()
    }

    /// Lane requests in flight that are not bound for the long tier.
    pub fn inflight_base(&self) -> u32 {
        self.budget.inflight_base()
    }

    /// Ceiling on in-flight requests that are not bound for the long tier:
    /// the current budget less the long-tier reserve (the whole budget when
    /// no reserve is configured, 0 when disabled).
    pub fn base_budget(&self) -> u32 {
        self.budget.base_limit()
    }

    fn take_base_slot(&self) -> Result<(), Rejected> {
        if self.budget.take_base_slot() {
            Ok(())
        } else {
            Err(self.reject(RejectReason::LongReserve))
        }
    }

    /// Per-backend in-flight bound for the current budget, `None` when
    /// admission is disabled. With no healthy backend the share is computed
    /// as for one, so the (degraded) selection still has a bound to apply.
    pub fn host_share(&self, healthy_backends: usize) -> Option<u32> {
        self.config.as_ref()?;
        let budget = self.budget.limit().max(1);
        let hosts = u32::try_from(healthy_backends.max(1)).unwrap_or(u32::MAX);
        Some(budget.div_ceil(hosts))
    }

    /// Snapshot limits by destination backend, shared by placement and failover.
    /// Configured counts deliberately prevent load concentration on host loss.
    pub fn backend_limits(&self, pool: &BackendPool) -> Option<Vec<u32>> {
        let config = self.config.as_ref()?;
        let budget = self.budget().max(1);
        if !config.tier_borrowing {
            return Some(vec![self.host_share(pool.healthy_count())?; pool.len()]);
        }
        let base_count = pool
            .backends()
            .iter()
            .filter(|b| b.tier == ContextTier::Base)
            .count()
            .max(1) as u32;
        let long_limit = budget
            .div_ceil(pool.len().max(1) as u32)
            .min(config.long_max_inflight_per_host);
        // Base hosts share what the long-tier reserve leaves, so their bounds
        // add up to what the base tier may hold at all.
        let base_budget = self.base_budget().max(1);
        Some(
            pool.backends()
                .iter()
                .map(|backend| match backend.tier {
                    ContextTier::Base => base_budget.div_ceil(base_count),
                    ContextTier::Long => long_limit,
                })
                .collect(),
        )
    }

    /// Scrape-time snapshots avoid racing gauge set operations on reservation/drop.
    pub fn record_backend_metrics(&self, pool: &BackendPool) {
        if let Some(limits) = self.backend_limits(pool) {
            for (index, (backend, limit)) in pool.backends().iter().zip(limits).enumerate() {
                metrics::gauge!("admission_backend_limit", "backend" => index.to_string(), "tier" => backend.tier.as_str()).set(f64::from(limit));
                metrics::gauge!("admission_backend_inflight", "backend" => index.to_string(), "tier" => backend.tier.as_str()).set(f64::from(backend.lane_conns.load(Ordering::Acquire)));
            }
        }
    }

    /// Whether backend `index` is saturated right now — its engine reports a
    /// queue of at least `queue_saturated_at` requests, or it rejected a lane
    /// request at engine admission within the TTL — i.e. selection should
    /// steer around it while other hosts have room.
    pub fn backend_saturated(&self, index: usize) -> bool {
        self.backend_saturated_at(index, Instant::now())
    }

    pub(crate) fn backend_saturated_at(&self, index: usize, now: Instant) -> bool {
        self.saturation
            .saturated_at(self.config.as_ref(), index, now)
    }

    /// Build (and count) a refusal for `reason`.
    pub fn reject(&self, reason: RejectReason) -> Rejected {
        metrics::counter!("admission_rejections_total", "reason" => reason.as_str()).increment(1);
        debug!(
            reason = reason.as_str(),
            "Lane request refused at admission"
        );
        Rejected {
            reason,
            retry_after: self
                .config
                .as_ref()
                .map_or(Duration::from_secs(1), |c| c.retry_after),
        }
    }

    /// The overload and budget checks without taking a slot: a cheap early
    /// refusal for routes that still have expensive work (image validation)
    /// ahead of `try_admit`. Nothing is reserved; the later `try_admit` can
    /// still refuse. The tier decision restricts the fleet-wide queue check
    /// to the backends this request may use (`None` = the whole pool).
    pub fn precheck(&self, pool: &BackendPool, tier: Option<TierDecision>) -> Result<(), Rejected> {
        self.precheck_at(pool, tier, Instant::now())
    }

    pub(crate) fn precheck_at(
        &self,
        pool: &BackendPool,
        tier: Option<TierDecision>,
        now: Instant,
    ) -> Result<(), Rejected> {
        self.check_signals_and_budget_at(pool, tier, now)
    }

    /// The overload signals and the budget peek. Read-only apart from the
    /// ramp tick, and safe to run more than once per request: `precheck`
    /// runs it early and `try_admit` again right before reserving the slot,
    /// since the lane may have changed in between (image validation). A rule
    /// that consumes something per request belongs in `precheck_at`, never
    /// here.
    fn check_signals_and_budget_at(
        &self,
        pool: &BackendPool,
        tier: Option<TierDecision>,
        now: Instant,
    ) -> Result<(), Rejected> {
        let Some(config) = &self.config else {
            return Ok(());
        };
        // Overload first, so a signal that just arrived cannot be preceded by
        // a ramp step that treats the interval as clean. The window holds base
        // requests only (a long prefill is no lane observation), so its
        // verdict says nothing about a request that is actually going to the
        // long tier — but it does apply to one that fell back onto the base
        // fleet, which is the fleet the breaker just declared overloaded.
        let on_long_tier = tier.is_some_and(|tier| tier.restrict == Some(ContextTier::Long));
        if !on_long_tier && self.ttft_over_bound(config, now) {
            return Err(self.reject(RejectReason::Ttft));
        }
        if self.every_backend_queued(config, pool, tier.and_then(|tier| tier.restrict), now) {
            return Err(self.reject(RejectReason::BackendQueue));
        }
        self.budget.tick_ramp(config, now);
        if self.budget.is_full() {
            return Err(self.reject(RejectReason::Budget));
        }
        if config.long_reserved_inflight > 0 && !on_long_tier && self.budget.base_full() {
            return Err(self.reject(RejectReason::LongReserve));
        }
        Ok(())
    }

    /// Run the overload and budget checks for a new request. `Ok(None)` when
    /// admission is disabled; `Ok(Some(permit))` holds one budget slot until
    /// the permit is dropped. The per-host share is applied by the caller at
    /// selection time (`host_share`, `backend_saturated`), since it needs the
    /// chosen backend.
    pub fn try_admit(
        self: &Arc<Self>,
        pool: &BackendPool,
        tier: Option<TierDecision>,
    ) -> Result<Option<Permit>, Rejected> {
        self.try_admit_at(pool, tier, Instant::now())
    }

    pub(crate) fn try_admit_at(
        self: &Arc<Self>,
        pool: &BackendPool,
        tier: Option<TierDecision>,
        now: Instant,
    ) -> Result<Option<Permit>, Rejected> {
        let Some(config) = &self.config else {
            return Ok(None);
        };
        self.check_signals_and_budget_at(pool, tier, now)?;
        // A long-bound request starts outside the base count. If placement or
        // connection fail-over sends it to a base host, it moves there then.
        let base = !tier.is_some_and(|tier| tier.restrict == Some(ContextTier::Long));
        // With a reserve, take the base slot first: a base request that then
        // loses the race for the global slot gives it back, and never holds a
        // global slot a long-tier request could have had.
        let capped = base && config.long_reserved_inflight > 0;
        if capped {
            self.take_base_slot()?;
        }
        if !self.budget.try_reserve() {
            if capped {
                self.budget.unreserve_base();
            }
            return Err(self.reject(RejectReason::Budget));
        }
        if base && !capped {
            self.budget.count_base();
        }
        if base {
            self.budget.gauge_base_admitted();
        }
        Ok(Some(Permit::new(
            Arc::clone(self),
            tier.is_some_and(|tier| tier.estimated == ContextTier::Long),
            base,
        )))
    }

    fn mark_dirty(&self) {
        self.budget.mark_dirty();
    }

    /// One time-to-first-generation observation (or a censored one: the
    /// request ended without a generation event after waiting `ttft`).
    fn record_ttft(&self, now: Instant, ttft: Duration) {
        let max = self.config.as_ref().and_then(|c| c.ttft_p95_max);
        if self.ttft.record(now, ttft, max) {
            self.mark_dirty();
        }
    }

    /// `(samples, breaches)` over the window, whatever the count.
    #[cfg(test)]
    fn ttft_totals(&self, now: Instant) -> (usize, usize) {
        self.ttft.totals(now)
    }

    /// `(samples, breaches)` in the window, or `None` below the minimum.
    pub fn ttft_breaches(&self, now: Instant) -> Option<(usize, usize)> {
        self.ttft.breaches(now)
    }

    fn ttft_over_bound(&self, config: &AdmissionConfig, now: Instant) -> bool {
        let over = self.ttft.over_bound(config.ttft_p95_max, now);
        if over {
            self.mark_dirty();
        }
        over
    }

    fn record_backpressure(&self, backend: usize, now: Instant) {
        self.saturation.record_backpressure(backend, now);
        self.mark_dirty();
    }

    fn every_backend_queued(
        &self,
        config: &AdmissionConfig,
        pool: &BackendPool,
        tier: Option<ContextTier>,
        now: Instant,
    ) -> bool {
        self.saturation
            .every_backend_queued(config, pool, tier, now)
    }
}

#[cfg(test)]
mod tests;
