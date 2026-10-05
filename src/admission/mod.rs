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

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

use crate::backend_pool::BackendPool;
use crate::context_tier::{ContextTier, TierDecision};
use crate::engine_load::EngineLoad;

mod ttft;
use ttft::TtftBreaker;
pub use ttft::{TTFT_BREACH_FRACTION, TTFT_MIN_BREACHES, TTFT_MIN_SAMPLES, TTFT_WINDOW};

/// Marker for a permit that has not been attached to a backend yet.
const NO_BACKEND: usize = usize::MAX;

// Permit lifecycle. Only `DISPATCHED` yields a TTFT observation.
const PENDING: u8 = 0;
const DISPATCHED: u8 = 1;
const GENERATED: u8 = 2;
const ABANDONED: u8 = 3;

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

struct Ramp {
    interval_started: Instant,
    /// An overload signal was seen in the current interval.
    dirty: bool,
}

/// Fleet-wide admission state shared by every request (`AppState.admission`).
pub struct AdmissionController {
    config: Option<AdmissionConfig>,
    inflight: AtomicU32,
    /// The part of `inflight` that is not bound for the long tier. Capped at
    /// `budget - long_reserved_inflight` when a reserve is configured.
    inflight_base: AtomicU32,
    budget: AtomicU32,
    ramp: Mutex<Ramp>,
    ttft: TtftBreaker,
    /// Per backend index: last engine admission rejection as milliseconds
    /// since `epoch`, plus one so that zero means "never".
    backpressure: Vec<AtomicU64>,
    /// Live engine view per backend when `VLLM_BACKEND_PROBE_URLS` is set.
    engine: Arc<EngineLoad>,
    epoch: Instant,
    /// Latched "every backend queues" state, one per tier: the verdict is
    /// computed over the request's own tier, so a single flag would flap
    /// between a queueing base fleet and an idle long host.
    queue_tripped: [AtomicBool; 2],
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
        Self {
            config,
            inflight: AtomicU32::new(0),
            inflight_base: AtomicU32::new(0),
            budget: AtomicU32::new(budget),
            ramp: Mutex::new(Ramp {
                interval_started: now,
                dirty: false,
            }),
            ttft: TtftBreaker::new(now),
            backpressure: (0..backend_count).map(|_| AtomicU64::new(0)).collect(),
            engine,
            epoch: now,
            queue_tripped: [AtomicBool::new(false), AtomicBool::new(false)],
        }
    }

    /// An inert controller: every request is admitted, nothing is counted.
    pub fn disabled() -> Self {
        Self::new(None, 0, Arc::new(EngineLoad::disabled()))
    }

    /// Fresh engine view `(running, queued)` for backend `index`, when polled.
    pub fn engine(&self, index: usize) -> Option<(u32, u32)> {
        self.engine.get(index).map(|s| (s.running, s.queued))
    }

    pub fn is_enabled(&self) -> bool {
        self.config.is_some()
    }

    pub fn config(&self) -> Option<&AdmissionConfig> {
        self.config.as_ref()
    }

    /// Current effective budget (0 when disabled).
    pub fn budget(&self) -> u32 {
        self.budget.load(Ordering::Relaxed)
    }

    /// Lane requests currently in flight.
    pub fn inflight(&self) -> u32 {
        self.inflight.load(Ordering::Relaxed)
    }

    /// Lane requests in flight that are not bound for the long tier.
    pub fn inflight_base(&self) -> u32 {
        self.inflight_base.load(Ordering::Relaxed)
    }

    /// Ceiling on in-flight requests that are not bound for the long tier:
    /// the current budget less the long-tier reserve (the whole budget when
    /// no reserve is configured, 0 when disabled).
    pub fn base_budget(&self) -> u32 {
        let reserve = self
            .config
            .as_ref()
            .map_or(0, |config| config.long_reserved_inflight);
        self.budget().saturating_sub(reserve)
    }

    fn take_base_slot(&self) -> Result<(), Rejected> {
        if self
            .config
            .as_ref()
            .is_none_or(|config| config.long_reserved_inflight == 0)
        {
            self.inflight_base.fetch_add(1, Ordering::AcqRel);
            return Ok(());
        }
        let mut current = self.inflight_base.load(Ordering::Acquire);
        loop {
            if current >= self.base_budget() {
                return Err(self.reject(RejectReason::LongReserve));
            }
            match self.inflight_base.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(actual) => current = actual,
            }
        }
    }

    /// Per-backend in-flight bound for the current budget, `None` when
    /// admission is disabled. With no healthy backend the share is computed
    /// as for one, so the (degraded) selection still has a bound to apply.
    pub fn host_share(&self, healthy_backends: usize) -> Option<u32> {
        self.config.as_ref()?;
        let budget = self.budget.load(Ordering::Relaxed).max(1);
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
        // The engine sample is checked before the config `None` return: a
        // gateway that sets `VLLM_BACKEND_PROBE_URLS` without admission still
        // steers placement around a queueing backend (threshold 1, same as
        // admission's own default).
        let threshold = self.config.as_ref().map_or(1, |c| c.queue_saturated_at);
        if self
            .engine
            .get_at(index, now)
            .is_some_and(|s| s.queued >= threshold)
        {
            return true;
        }
        let Some(config) = &self.config else {
            return false;
        };
        let stamp = self
            .backpressure
            .get(index)
            .map_or(0, |s| s.load(Ordering::Relaxed));
        if stamp == 0 {
            return false;
        }
        let at = self.epoch + Duration::from_millis(stamp - 1);
        now.saturating_duration_since(at) <= config.backpressure_ttl
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
        self.tick_ramp(config, now);
        if self.inflight.load(Ordering::Acquire) >= self.budget.load(Ordering::Relaxed) {
            return Err(self.reject(RejectReason::Budget));
        }
        if config.long_reserved_inflight > 0
            && !on_long_tier
            && self.inflight_base.load(Ordering::Acquire) >= self.base_budget()
        {
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
        self.precheck_at(pool, tier, now)?;
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
        let mut current = self.inflight.load(Ordering::Acquire);
        loop {
            if current >= self.budget.load(Ordering::Relaxed) {
                if capped {
                    self.inflight_base.fetch_sub(1, Ordering::AcqRel);
                }
                return Err(self.reject(RejectReason::Budget));
            }
            match self.inflight.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
        if base && !capped {
            self.inflight_base.fetch_add(1, Ordering::AcqRel);
        }
        metrics::gauge!("admission_inflight").increment(1.0);
        if base {
            metrics::gauge!("admission_inflight_base").increment(1.0);
        }
        Ok(Some(Permit {
            controller: Arc::clone(self),
            backend: AtomicUsize::new(NO_BACKEND),
            long_request: tier.is_some_and(|tier| tier.estimated == ContextTier::Long),
            base: AtomicBool::new(base),
            state: AtomicU8::new(PENDING),
            dispatched_at: OnceLock::new(),
        }))
    }

    fn ramp(&self) -> MutexGuard<'_, Ramp> {
        self.ramp.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Grow the budget by one step when a whole interval passed without an
    /// overload signal; a dirty interval just restarts the clock.
    fn tick_ramp(&self, config: &AdmissionConfig, now: Instant) {
        let mut ramp = self.ramp();
        if now.saturating_duration_since(ramp.interval_started) < config.ramp_interval {
            return;
        }
        let clean = !ramp.dirty;
        ramp.dirty = false;
        ramp.interval_started = now;
        drop(ramp);

        let budget = self.budget.load(Ordering::Relaxed);
        if budget >= config.max_inflight {
            return;
        }
        if clean {
            let next = budget
                .saturating_add(config.ramp_step)
                .min(config.max_inflight);
            self.budget.store(next, Ordering::Relaxed);
            metrics::gauge!("admission_budget").set(f64::from(next));
            info!(
                from = budget,
                to = next,
                max = config.max_inflight,
                "Admission budget ramped up"
            );
        } else {
            info!(
                budget,
                max = config.max_inflight,
                "Admission budget held: overload signals during the last interval"
            );
        }
    }

    fn mark_dirty(&self) {
        self.ramp().dirty = true;
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
        if let Some(slot) = self.backpressure.get(backend) {
            let millis = u64::try_from(now.saturating_duration_since(self.epoch).as_millis())
                .unwrap_or(u64::MAX - 1);
            slot.store(millis + 1, Ordering::Relaxed);
        }
        metrics::counter!("admission_backpressure_total", "backend" => backend.to_string())
            .increment(1);
        self.mark_dirty();
    }

    /// Every healthy backend of the request's tier is saturated: its engine
    /// queue is at or above `queue_saturated_at`, or it rejected a lane
    /// request at engine admission within the TTL.
    fn every_backend_queued(
        &self,
        config: &AdmissionConfig,
        pool: &BackendPool,
        tier: Option<ContextTier>,
        now: Instant,
    ) -> bool {
        let mut healthy = 0usize;
        let mut all_queued = true;
        for (index, backend) in pool.backends().iter().enumerate() {
            if !backend.healthy.load(Ordering::Relaxed)
                || tier.is_some_and(|tier| tier != backend.tier)
            {
                continue;
            }
            healthy += 1;
            if !self.backend_saturated_at(index, now) {
                all_queued = false;
                break;
            }
        }
        let queued = healthy > 0 && all_queued;
        let latch = &self.queue_tripped[usize::from(tier == Some(ContextTier::Long))];
        if queued != latch.swap(queued, Ordering::Relaxed) {
            let tier = tier.map_or("any", ContextTier::as_str);
            if queued {
                warn!(
                    tier,
                    healthy_backends = healthy,
                    queue_saturated_at = config.queue_saturated_at,
                    ttl_secs = config.backpressure_ttl.as_secs(),
                    "Every backend's engine queue is at or above the saturation threshold, or it rejected a lane request at engine admission recently; refusing new work"
                );
            } else {
                info!(tier, "A backend accepts lane work again");
            }
        }
        queued
    }
}

/// One admitted request's budget slot. Dropping it releases the slot; the
/// streaming path moves it into the pump task next to the backend guard so
/// the slot is held for the whole stream.
///
/// Lifecycle: `PENDING` (admitted) → `DISPATCHED` (the engine accepted the
/// request; the TTFT clock starts here, after any pre-dispatch work such as
/// image validation) → `GENERATED` (first generation event: one sample) or
/// `ABANDONED` (the accepted request turned out to be an engine rejection:
/// no sample). A permit released while still `DISPATCHED` records the time
/// it waited as a censored sample.
pub struct Permit {
    controller: Arc<AdmissionController>,
    backend: AtomicUsize,
    /// The request's estimated input is above the long-context threshold.
    /// Decided once, at admission: a prefill of that size takes tens of
    /// seconds on either tier, so the wait is not a lane observation wherever
    /// the request ends up running.
    long_request: bool,
    /// Counted in `inflight_base`, including after a fallback to a base host.
    base: AtomicBool,
    state: AtomicU8,
    dispatched_at: OnceLock<Instant>,
}

impl Permit {
    /// Count a long-bound request against the base allowance before placing
    /// it on a base host. Once counted, it stays there even if it moves back
    /// to a long host.
    pub fn count_as_base(&self) -> Result<(), Rejected> {
        if self.base.load(Ordering::Acquire) {
            return Ok(());
        }
        self.controller.take_base_slot()?;
        if self.base.swap(true, Ordering::AcqRel) {
            self.controller.inflight_base.fetch_sub(1, Ordering::AcqRel);
        } else {
            metrics::gauge!("admission_inflight_base").increment(1.0);
        }
        Ok(())
    }

    pub fn requested_tier(&self) -> ContextTier {
        if self.long_request {
            ContextTier::Long
        } else {
            ContextTier::Base
        }
    }

    /// Record which backend the request was placed on (needed to attribute
    /// engine back-pressure). Called again after a connection fail-over.
    pub fn attach_backend(&self, index: usize) {
        self.backend.store(index, Ordering::Relaxed);
    }

    pub fn backend(&self) -> Option<usize> {
        match self.backend.load(Ordering::Relaxed) {
            NO_BACKEND => None,
            index => Some(index),
        }
    }

    /// Steer selection around backends that are queueing or rejected recently.
    pub fn backend_saturated(&self, index: usize) -> bool {
        self.controller.backend_saturated(index)
    }

    /// Fresh engine view for backend `index` (see `AdmissionController::engine`).
    pub fn engine(&self, index: usize) -> Option<(u32, u32)> {
        self.controller.engine(index)
    }

    pub fn backend_limits(&self, pool: &BackendPool) -> Option<Vec<u32>> {
        self.controller.backend_limits(pool)
    }

    /// A refusal because no backend has room under its share.
    pub fn reject_host_share(&self) -> Rejected {
        self.controller.reject(RejectReason::HostShare)
    }

    /// A refusal because strict mode's tier has no healthy backend at all
    /// (`context_tier.rs`), distinct from `reject_host_share`'s "backends
    /// exist but are full or steered around".
    pub fn reject_tier_unavailable(&self) -> Rejected {
        self.controller.reject(RejectReason::TierUnavailable)
    }

    /// The engine accepted the request; the clock now runs against the
    /// first generation event.
    pub fn mark_dispatched(&self) {
        self.mark_dispatched_at(Instant::now());
    }

    /// When the request went out, i.e. when its time-to-first-token clock
    /// started. `None` until it is dispatched.
    pub fn dispatched_at(&self) -> Option<Instant> {
        self.dispatched_at.get().copied()
    }

    pub(crate) fn mark_dispatched_at(&self, now: Instant) {
        if self
            .state
            .compare_exchange(PENDING, DISPATCHED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.dispatched_at.set(now);
        }
    }

    /// The accepted request turned out to be an engine rejection (an error
    /// event in the stream): no TTFT observation for it. A no-op once a
    /// generation event was seen (a later error is a mid-stream failure).
    pub fn abandon(&self) {
        let _ =
            self.state
                .compare_exchange(DISPATCHED, ABANDONED, Ordering::AcqRel, Ordering::Acquire);
    }

    /// The first generation event arrived: one TTFT sample. Idempotent, and
    /// a no-op unless the request was dispatched and not abandoned.
    pub fn observe_generation_started(&self) {
        self.observe_generation_started_at(Instant::now());
    }

    pub(crate) fn observe_generation_started_at(&self, now: Instant) {
        if self
            .state
            .compare_exchange(DISPATCHED, GENERATED, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        self.record_ttft(now);
    }

    /// One time-to-first-generation observation for the lane — unless this is
    /// a long-context request, whose >100k-token prefill takes tens of seconds
    /// by nature: that wait says nothing about the lane's health and would
    /// trip its breaker for everyone.
    fn record_ttft(&self, now: Instant) {
        if self.long_request {
            return;
        }
        if let Some(dispatched_at) = self.dispatched_at.get() {
            self.controller
                .record_ttft(now, now.saturating_duration_since(*dispatched_at));
        }
    }

    /// The engine refused this request at admission (queue full or displaced
    /// by a higher-priority request).
    pub fn observe_backpressure(&self) {
        self.observe_backpressure_at(Instant::now());
    }

    pub(crate) fn observe_backpressure_at(&self, now: Instant) {
        if let Some(backend) = self.backend() {
            self.controller.record_backpressure(backend, now);
        } else {
            self.controller.mark_dirty();
        }
    }

    pub(crate) fn release_at(&self, now: Instant) {
        // A dispatched request that ended (client gone, idle timeout, stream
        // cut) before any generation event waited at least this long: record
        // it, or slow requests that clients give up on would never count.
        if self
            .state
            .compare_exchange(DISPATCHED, GENERATED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.record_ttft(now);
        }
        self.controller.inflight.fetch_sub(1, Ordering::AcqRel);
        metrics::gauge!("admission_inflight").decrement(1.0);
        if self.base.load(Ordering::Acquire) {
            self.controller.inflight_base.fetch_sub(1, Ordering::AcqRel);
            metrics::gauge!("admission_inflight_base").decrement(1.0);
        }
    }
}

impl std::fmt::Debug for Permit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Permit")
            .field("backend", &self.backend())
            .field("long_request", &self.long_request)
            .field("base", &self.base.load(Ordering::Relaxed))
            .field("state", &self.state.load(Ordering::Relaxed))
            .finish()
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.release_at(Instant::now());
    }
}

#[cfg(test)]
mod tests;
