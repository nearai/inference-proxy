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
//! `backend_limits`. The global budget is shared, with no reserved tier slots.
//!
//! Everything is derived from what the gateway observes itself; no admin
//! endpoint or extra token is involved. Disabled (`max_inflight = 0`) the
//! module is inert and the in-CVM behavior is unchanged.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

use crate::backend_pool::BackendPool;
use crate::context_tier::{ContextTier, TierDecision};
use crate::engine_load::EngineLoad;

/// Window over which time-to-first-generation observations are counted, as
/// one-second buckets of `(samples, breaches)`: bounded memory and work at
/// any request rate, and exactly this long.
pub const TTFT_WINDOW: Duration = Duration::from_secs(60);
/// Samples needed in the window before the bound is enforced at all.
pub const TTFT_MIN_SAMPLES: usize = 20;
/// Fraction of the window that must breach the bound to trip, with a floor
/// of `TTFT_MIN_BREACHES`, so one slow request cannot close the fleet.
pub const TTFT_BREACH_FRACTION: f64 = 0.05;
pub const TTFT_MIN_BREACHES: usize = 2;
/// The breaker is re-evaluated at most this often; refusals in between reuse
/// the cached verdict.
const BREAKER_REEVALUATE_AFTER: Duration = Duration::from_secs(1);
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
    /// `Retry-After` value on every refusal.
    pub retry_after: Duration,
    /// A bounded exception for recently completed large-prefix continuations.
    pub continuation: Option<ContinuationConfig>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinuationConfig {
    pub min_input_tokens: u64,
    pub max_age: Duration,
    pub max_wait: Duration,
    pub max_waiters: usize,
    pub max_buffered_bytes: u64,
    pub max_estimated_tokens: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionClass {
    Cold,
    RecentPrefix {
        preferred_backend: usize,
        body_bytes: u64,
        estimated_tokens: u64,
    },
}

/// Why a request was refused. The label of `admission_rejections_total`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// The global in-flight budget is used up.
    Budget,
    /// Every backend is at its per-host share (or steered around).
    HostShare,
    /// Every healthy backend recently rejected at engine admission.
    BackendQueue,
    /// The lane's own time-to-first-generation is above the bound.
    Ttft,
    /// A released slot is being handed to an older warm continuation.
    ContinuationHandoff,
}

impl RejectReason {
    pub fn as_str(self) -> &'static str {
        match self {
            RejectReason::Budget => "budget",
            RejectReason::HostShare => "host_share",
            RejectReason::BackendQueue => "backend_queue",
            RejectReason::Ttft => "ttft",
            RejectReason::ContinuationHandoff => "continuation_handoff",
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

struct Breaker {
    evaluated_at: Option<Instant>,
    tripped: bool,
}

struct ContinuationWaiter {
    ticket: u64,
    preferred_backend: usize,
    body_bytes: u64,
    estimated_tokens: u64,
    tier: Option<ContextTier>,
}

#[derive(Default)]
struct ContinuationState {
    next_ticket: u64,
    waiters: VecDeque<ContinuationWaiter>,
    buffered_bytes: u64,
    estimated_tokens: u64,
    /// Backend slots handed to admitted warm requests which have not yet
    /// completed atomic backend placement.
    handoffs: HashMap<usize, u32>,
}

/// One second of time-to-first-generation observations.
#[derive(Clone, Copy, Default)]
struct Bucket {
    /// Seconds since `epoch` this bucket currently holds (+1; 0 = unused).
    second: u64,
    samples: u32,
    breaches: u32,
}

const TTFT_BUCKETS: usize = TTFT_WINDOW.as_secs() as usize;

/// Fleet-wide admission state shared by every request (`AppState.admission`).
pub struct AdmissionController {
    config: Option<AdmissionConfig>,
    inflight: AtomicU32,
    budget: AtomicU32,
    ramp: Mutex<Ramp>,
    /// Ring of one-second buckets covering the last `TTFT_WINDOW`.
    ttft: Mutex<[Bucket; TTFT_BUCKETS]>,
    breaker: Mutex<Breaker>,
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
    continuation: Mutex<ContinuationState>,
    capacity_changed: tokio::sync::Notify,
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
            debug_assert!(config.start_inflight <= config.max_inflight);
        }
        Self {
            config,
            inflight: AtomicU32::new(0),
            budget: AtomicU32::new(budget),
            ramp: Mutex::new(Ramp {
                interval_started: now,
                dirty: false,
            }),
            ttft: Mutex::new([Bucket::default(); TTFT_BUCKETS]),
            breaker: Mutex::new(Breaker {
                evaluated_at: None,
                tripped: false,
            }),
            backpressure: (0..backend_count).map(|_| AtomicU64::new(0)).collect(),
            engine,
            epoch: now,
            queue_tripped: [AtomicBool::new(false), AtomicBool::new(false)],
            continuation: Mutex::new(ContinuationState::default()),
            capacity_changed: tokio::sync::Notify::new(),
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

    pub fn continuation_config(&self) -> Option<&ContinuationConfig> {
        self.config.as_ref()?.continuation.as_ref()
    }

    fn continuation_state(&self) -> MutexGuard<'_, ContinuationState> {
        self.continuation.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn add_handoff(&self, backend: usize) {
        let mut state = self.continuation_state();
        *state.handoffs.entry(backend).or_default() += 1;
    }

    fn remove_handoff(&self, backend: usize) {
        let mut state = self.continuation_state();
        if let Some(count) = state.handoffs.get_mut(&backend) {
            *count -= 1;
            if *count == 0 {
                state.handoffs.remove(&backend);
            }
        }
    }

    fn handoff_active(&self, backend: usize) -> bool {
        self.continuation_state().handoffs.contains_key(&backend)
    }

    fn oldest_ready_waiter(&self, state: &ContinuationState, pool: &BackendPool) -> Option<u64> {
        state.waiters.iter().find_map(|waiter| {
            let backend = pool.backends().get(waiter.preferred_backend)?;
            (backend.healthy.load(Ordering::Relaxed)
                && waiter.tier.is_none_or(|tier| tier == backend.tier)
                && !self.backend_saturated(waiter.preferred_backend))
            .then_some(waiter.ticket)
        })
    }

    pub fn waiter_count(&self) -> usize {
        self.continuation_state().waiters.len()
    }

    pub fn waiter_usage(&self) -> (u64, u64) {
        let state = self.continuation_state();
        (state.buffered_bytes, state.estimated_tokens)
    }

    /// Current effective budget (0 when disabled).
    pub fn budget(&self) -> u32 {
        self.budget.load(Ordering::Relaxed)
    }

    /// Lane requests currently in flight.
    pub fn inflight(&self) -> u32 {
        self.inflight.load(Ordering::Relaxed)
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
        Some(
            pool.backends()
                .iter()
                .map(|backend| match backend.tier {
                    ContextTier::Base => budget.div_ceil(base_count),
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
    /// non-empty queue, or it rejected a lane request at engine admission
    /// within the TTL — i.e. selection should steer around it while other
    /// hosts have room.
    pub fn backend_saturated(&self, index: usize) -> bool {
        self.backend_saturated_at(index, Instant::now())
    }

    pub(crate) fn backend_saturated_at(&self, index: usize, now: Instant) -> bool {
        if self.engine.get_at(index, now).is_some_and(|s| s.queued > 0) {
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

    /// Early validation for routes with pre-dispatch work. Recent-prefix work
    /// may proceed to the consuming admission call when only budget or engine
    /// queue pressure is blocking it; TTFT overload still fails immediately.
    pub fn precheck_for(
        &self,
        pool: &BackendPool,
        tier: Option<TierDecision>,
        class: AdmissionClass,
    ) -> Result<(), Rejected> {
        if class == AdmissionClass::Cold || self.continuation_config().is_none() {
            return self.precheck(pool, tier);
        }
        let Some(config) = &self.config else {
            return Ok(());
        };
        let now = Instant::now();
        let on_long_tier = tier.is_some_and(|tier| tier.restrict == Some(ContextTier::Long));
        if !on_long_tier && self.ttft_over_bound(config, now) {
            return Err(self.reject(RejectReason::Ttft));
        }
        self.tick_ramp(config, now);
        Ok(())
    }

    pub(crate) fn precheck_at(
        &self,
        pool: &BackendPool,
        tier: Option<TierDecision>,
        now: Instant,
    ) -> Result<(), Rejected> {
        self.check_at(pool, tier, now)
            .map_err(|reason| self.reject(reason))
    }

    fn check_at(
        &self,
        pool: &BackendPool,
        tier: Option<TierDecision>,
        now: Instant,
    ) -> Result<(), RejectReason> {
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
            return Err(RejectReason::Ttft);
        }
        if self.every_backend_queued(config, pool, tier.and_then(|tier| tier.restrict), now) {
            return Err(RejectReason::BackendQueue);
        }
        self.tick_ramp(config, now);
        if self.inflight.load(Ordering::Acquire) >= self.budget.load(Ordering::Relaxed) {
            return Err(RejectReason::Budget);
        }
        Ok(())
    }

    /// Admit one request. Cold work keeps the existing fail-fast behavior.
    /// An eligible recent-prefix continuation may wait briefly for budget or
    /// engine-queue pressure to clear; its permit is pinned to the backend
    /// which still holds the prefix.
    pub async fn admit(
        self: &Arc<Self>,
        pool: &BackendPool,
        tier: Option<TierDecision>,
        class: AdmissionClass,
    ) -> Result<Option<Permit>, Rejected> {
        let Some(continuation) = self.continuation_config() else {
            return self.try_admit(pool, tier);
        };

        if class == AdmissionClass::Cold {
            let ready_waiter = {
                let state = self.continuation_state();
                self.oldest_ready_waiter(&state, pool).is_some()
            };
            if ready_waiter && self.inflight() < self.budget() {
                return Err(self.reject(RejectReason::ContinuationHandoff));
            }
            return self.try_admit(pool, tier);
        }

        let AdmissionClass::RecentPrefix {
            preferred_backend,
            body_bytes,
            estimated_tokens,
        } = class
        else {
            unreachable!()
        };
        let tier_compatible = pool
            .backends()
            .get(preferred_backend)
            .is_some_and(|backend| {
                backend.healthy.load(Ordering::Relaxed)
                    && tier
                        .and_then(|decision| decision.restrict)
                        .is_none_or(|required| required == backend.tier)
            });
        if !tier_compatible
            || estimated_tokens < continuation.min_input_tokens
            || body_bytes > continuation.max_buffered_bytes
            || estimated_tokens > continuation.max_estimated_tokens
        {
            return self.try_admit(pool, tier);
        }

        let immediate = if self.backend_saturated(preferred_backend) {
            Err(RejectReason::BackendQueue)
        } else {
            self.try_admit_reason_at(pool, tier, Instant::now())
        };
        match immediate {
            Ok(permit) => {
                metrics::counter!("continuation_admission_total", "outcome" => "immediate", "reason" => "none")
                    .increment(1);
                Ok(permit.map(|permit| permit.pin(preferred_backend)))
            }
            Err(reason) if !matches!(reason, RejectReason::Budget | RejectReason::BackendQueue) => {
                Err(self.reject(reason))
            }
            Err(initial_reason) => {
                let ticket = {
                    let mut state = self.continuation_state();
                    if state.waiters.len() >= continuation.max_waiters
                        || state.buffered_bytes.saturating_add(body_bytes)
                            > continuation.max_buffered_bytes
                        || state.estimated_tokens.saturating_add(estimated_tokens)
                            > continuation.max_estimated_tokens
                    {
                        metrics::counter!("continuation_admission_total", "outcome" => "rejected", "reason" => initial_reason.as_str())
                            .increment(1);
                        return Err(self.reject(initial_reason));
                    }
                    let ticket = state.next_ticket;
                    state.next_ticket = state.next_ticket.wrapping_add(1);
                    state.waiters.push_back(ContinuationWaiter {
                        ticket,
                        preferred_backend,
                        body_bytes,
                        estimated_tokens,
                        tier: tier.and_then(|decision| decision.restrict),
                    });
                    state.buffered_bytes += body_bytes;
                    state.estimated_tokens += estimated_tokens;
                    metrics::gauge!("continuation_admission_waiters")
                        .set(state.waiters.len() as f64);
                    ticket
                };
                let mut registration = WaiterRegistration::new(Arc::clone(self), ticket);
                let deadline = tokio::time::Instant::now() + continuation.max_wait;
                loop {
                    let notified = self.capacity_changed.notified();
                    let engine_revision = self.engine.revision();
                    let is_oldest_ready = {
                        let state = self.continuation_state();
                        self.oldest_ready_waiter(&state, pool) == Some(ticket)
                    };
                    if is_oldest_ready {
                        if let Ok(permit) = self.try_admit_reason_at(pool, tier, Instant::now()) {
                            registration.remove();
                            metrics::counter!("continuation_admission_total", "outcome" => "admitted", "reason" => initial_reason.as_str())
                                .increment(1);
                            return Ok(permit.map(|permit| permit.pin(preferred_backend)));
                        }
                    }
                    if tokio::time::Instant::now() >= deadline {
                        metrics::counter!("continuation_admission_total", "outcome" => "timeout", "reason" => initial_reason.as_str())
                            .increment(1);
                        return Err(self.reject(initial_reason));
                    }
                    tokio::select! {
                        _ = notified => {}
                        _ = self.engine.changed_since(engine_revision) => {}
                        _ = tokio::time::sleep_until(deadline) => {}
                    }
                }
            }
        }
    }

    /// Synchronous primitive retained for the unbuffered path and invariant
    /// tests. Production routes use `admit` so there is one admission path.
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
        self.try_admit_reason_at(pool, tier, now)
            .map_err(|reason| self.reject(reason))
    }

    fn try_admit_reason_at(
        self: &Arc<Self>,
        pool: &BackendPool,
        tier: Option<TierDecision>,
        now: Instant,
    ) -> Result<Option<Permit>, RejectReason> {
        if self.config.is_none() {
            return Ok(None);
        }
        self.check_at(pool, tier, now)?;
        let mut current = self.inflight.load(Ordering::Acquire);
        loop {
            if current >= self.budget.load(Ordering::Relaxed) {
                return Err(RejectReason::Budget);
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
        metrics::gauge!("admission_inflight").increment(1.0);
        Ok(Some(Permit {
            controller: Arc::clone(self),
            backend: AtomicUsize::new(NO_BACKEND),
            long_request: tier.is_some_and(|tier| tier.estimated == ContextTier::Long),
            state: AtomicU8::new(PENDING),
            dispatched_at: OnceLock::new(),
            required_backend: None,
            completion_affinity: OnceLock::new(),
            handoff_active: AtomicBool::new(false),
        }))
    }

    fn ramp(&self) -> MutexGuard<'_, Ramp> {
        self.ramp.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn ttft_buckets(&self) -> MutexGuard<'_, [Bucket; TTFT_BUCKETS]> {
        self.ttft.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Seconds since `epoch`, plus one (bucket stamps use 0 for "unused").
    fn stamp(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.epoch).as_secs() + 1
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
        metrics::histogram!("admission_ttft_seconds").record(ttft.as_secs_f64());
        let Some(max) = self.config.as_ref().and_then(|c| c.ttft_p95_max) else {
            return;
        };
        let breach = ttft > max;
        if breach {
            // A breach counts against the current ramp interval even if the
            // breaker is not (yet) tripped.
            self.mark_dirty();
        }
        let stamp = self.stamp(now);
        let mut buckets = self.ttft_buckets();
        let bucket = &mut buckets[(stamp % TTFT_BUCKETS as u64) as usize];
        if bucket.second != stamp {
            *bucket = Bucket {
                second: stamp,
                samples: 0,
                breaches: 0,
            };
        }
        bucket.samples = bucket.samples.saturating_add(1);
        if breach {
            bucket.breaches = bucket.breaches.saturating_add(1);
        }
    }

    /// `(samples, breaches)` over the window, whatever the count.
    fn ttft_totals(&self, now: Instant) -> (usize, usize) {
        let newest = self.stamp(now);
        let oldest = newest.saturating_sub(TTFT_BUCKETS as u64 - 1);
        let buckets = self.ttft_buckets();
        buckets
            .iter()
            .filter(|b| b.second != 0 && b.second >= oldest && b.second <= newest)
            .fold((0, 0), |(s, b), bucket| {
                (s + bucket.samples as usize, b + bucket.breaches as usize)
            })
    }

    /// `(samples, breaches)` in the window, or `None` below the minimum.
    pub fn ttft_breaches(&self, now: Instant) -> Option<(usize, usize)> {
        let (samples, breaches) = self.ttft_totals(now);
        (samples >= TTFT_MIN_SAMPLES).then_some((samples, breaches))
    }

    fn ttft_over_bound(&self, config: &AdmissionConfig, now: Instant) -> bool {
        let Some(max) = config.ttft_p95_max else {
            return false;
        };
        let mut breaker = self.breaker.lock().unwrap_or_else(|e| e.into_inner());
        let fresh = breaker
            .evaluated_at
            .is_some_and(|at| now.saturating_duration_since(at) < BREAKER_REEVALUATE_AFTER);
        if !fresh {
            let over = self.ttft_breaches(now).is_some_and(|(samples, breaches)| {
                let needed = ((samples as f64 * TTFT_BREACH_FRACTION).ceil() as usize)
                    .max(TTFT_MIN_BREACHES);
                breaches >= needed
            });
            if over != breaker.tripped {
                if over {
                    warn!(
                        max_ms = max.as_millis(),
                        window_secs = TTFT_WINDOW.as_secs(),
                        "Lane time-to-first-generation above bound, refusing new work"
                    );
                } else {
                    info!("Lane time-to-first-generation back under bound");
                }
            }
            breaker.tripped = over;
            breaker.evaluated_at = Some(now);
        }
        let over = breaker.tripped;
        drop(breaker);
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

    /// Every healthy backend of the request's tier rejected at engine
    /// admission within the TTL.
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
                    ttl_secs = config.backpressure_ttl.as_secs(),
                    "Every backend rejected at engine admission recently, refusing new work"
                );
            } else {
                info!(tier, "A backend accepts lane work again");
            }
        }
        queued
    }
}

/// Removes queue accounting even when an admission future is cancelled.
struct WaiterRegistration {
    controller: Arc<AdmissionController>,
    ticket: Option<u64>,
}

impl WaiterRegistration {
    fn new(controller: Arc<AdmissionController>, ticket: u64) -> Self {
        Self {
            controller,
            ticket: Some(ticket),
        }
    }

    fn remove(&mut self) {
        let Some(ticket) = self.ticket.take() else {
            return;
        };
        let mut state = self.controller.continuation_state();
        if let Some(index) = state
            .waiters
            .iter()
            .position(|waiter| waiter.ticket == ticket)
        {
            let waiter = state.waiters.remove(index).expect("waiter index exists");
            state.buffered_bytes = state.buffered_bytes.saturating_sub(waiter.body_bytes);
            state.estimated_tokens = state
                .estimated_tokens
                .saturating_sub(waiter.estimated_tokens);
            metrics::gauge!("continuation_admission_waiters").set(state.waiters.len() as f64);
        }
        drop(state);
        self.controller.capacity_changed.notify_waiters();
    }
}

impl Drop for WaiterRegistration {
    fn drop(&mut self) {
        self.remove();
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
    state: AtomicU8,
    dispatched_at: OnceLock<Instant>,
    required_backend: Option<usize>,
    completion_affinity: OnceLock<(
        Arc<crate::backend_affinity::BackendConversationAffinity>,
        crate::backend_affinity::ConversationKey,
    )>,
    handoff_active: AtomicBool,
}

impl Permit {
    fn pin(mut self, backend: usize) -> Self {
        self.required_backend = Some(backend);
        self.controller.add_handoff(backend);
        self.handoff_active.store(true, Ordering::Release);
        self
    }

    /// Whether placement may use this backend. A warm continuation must use
    /// the backend whose cache made it eligible.
    pub fn backend_allowed(&self, index: usize) -> bool {
        self.required_backend.map_or_else(
            || !self.controller.handoff_active(index),
            |required| required == index,
        )
    }

    pub fn track_affinity_completion(
        &self,
        affinity: Arc<crate::backend_affinity::BackendConversationAffinity>,
        key: crate::backend_affinity::ConversationKey,
    ) {
        let _ = self.completion_affinity.set((affinity, key));
    }

    /// Mark the currently attached backend warm only after a valid terminal
    /// completion. Failover updates `backend`, so stale destinations cannot be
    /// recorded as warm.
    pub fn observe_completion(&self) {
        let (Some((affinity, key)), Some(backend)) =
            (self.completion_affinity.get(), self.backend())
        else {
            return;
        };
        affinity.mark_completed(*key, backend);
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
        self.release_handoff();
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
        self.release_handoff();
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
        self.controller.capacity_changed.notify_waiters();
    }

    fn release_handoff(&self) {
        if self.handoff_active.swap(false, Ordering::AcqRel) {
            if let Some(backend) = self.required_backend {
                self.controller.remove_handoff(backend);
            }
        }
    }
}

impl std::fmt::Debug for Permit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Permit")
            .field("backend", &self.backend())
            .field("long_request", &self.long_request)
            .field("required_backend", &self.required_backend)
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
mod tests {
    use super::*;

    fn config() -> AdmissionConfig {
        AdmissionConfig {
            max_inflight: 8,
            tier_borrowing: false,
            long_max_inflight_per_host: 0,
            start_inflight: 2,
            ramp_step: 2,
            ramp_interval: Duration::from_secs(60),
            ttft_p95_max: Some(Duration::from_secs(10)),
            backpressure_ttl: Duration::from_secs(10),
            retry_after: Duration::from_secs(3),
            continuation: None,
        }
    }

    fn continuation_config() -> ContinuationConfig {
        ContinuationConfig {
            min_input_tokens: 100_000,
            max_age: Duration::from_secs(180),
            max_wait: Duration::from_millis(100),
            max_waiters: 8,
            max_buffered_bytes: 8 * 1024 * 1024,
            max_estimated_tokens: 2_000_000,
        }
    }

    fn recent(backend: usize) -> AdmissionClass {
        AdmissionClass::RecentPrefix {
            preferred_backend: backend,
            body_bytes: 1024,
            estimated_tokens: 150_000,
        }
    }

    fn pool(n: usize) -> BackendPool {
        BackendPool::new((0..n).map(|i| format!("http://b{i}:8000")).collect())
    }

    fn controller(config: AdmissionConfig, backends: usize) -> Arc<AdmissionController> {
        Arc::new(AdmissionController::new(
            Some(config),
            backends,
            Arc::new(EngineLoad::disabled()),
        ))
    }

    /// A tier decision for a request estimated onto `estimated` that may use
    /// the backends of `restrict`.
    fn tier(estimated: ContextTier, restrict: Option<ContextTier>) -> Option<TierDecision> {
        Some(TierDecision {
            estimated,
            restrict,
        })
    }

    /// Admit at `t0` and record a first-generation sample `ttft` later.
    fn sample(c: &Arc<AdmissionController>, p: &BackendPool, t0: Instant, ttft: Duration) {
        let permit = c.try_admit_at(p, None, t0).unwrap().unwrap();
        permit.mark_dispatched_at(t0);
        permit.observe_generation_started_at(t0 + ttft);
    }

    #[test]
    fn borrowing_limits_follow_budget_but_not_health() {
        let p = BackendPool::with_long_context(
            vec!["b0".into(), "b1".into(), "b2".into()],
            vec!["long".into()],
        );
        let c = controller(
            AdmissionConfig {
                max_inflight: 64,
                start_inflight: 32,
                tier_borrowing: true,
                long_max_inflight_per_host: 12,
                ..config()
            },
            4,
        );
        for (budget, expected) in [
            (32, vec![11, 11, 11, 8]),
            (40, vec![14, 14, 14, 10]),
            (48, vec![16, 16, 16, 12]),
            (56, vec![19, 19, 19, 12]),
            (64, vec![22, 22, 22, 12]),
        ] {
            c.budget.store(budget, Ordering::Relaxed);
            assert_eq!(c.backend_limits(&p).unwrap(), expected);
            p.backends()[0].healthy.store(false, Ordering::Relaxed);
            assert_eq!(c.backend_limits(&p).unwrap(), expected);
            p.backends()[0].healthy.store(true, Ordering::Relaxed);
        }
        let legacy = controller(
            AdmissionConfig {
                max_inflight: 48,
                start_inflight: 48,
                ..config()
            },
            4,
        );
        assert_eq!(legacy.backend_limits(&p).unwrap(), vec![12; 4]);
        p.backends()[0].healthy.store(false, Ordering::Relaxed);
        assert_eq!(legacy.backend_limits(&p).unwrap(), vec![16; 4]);
        assert_eq!(AdmissionController::disabled().backend_limits(&p), None);
    }

    #[test]
    fn borrowing_reservations_enforce_shared_and_long_limits_under_concurrency() {
        use crate::backend_pool::Policy;
        use std::sync::Barrier;
        let p = Arc::new(BackendPool::with_long_context(
            vec!["b0".into(), "b1".into(), "b2".into()],
            vec!["long".into()],
        ));
        let c = controller(
            AdmissionConfig {
                max_inflight: 48,
                start_inflight: 48,
                tier_borrowing: true,
                long_max_inflight_per_host: 12,
                ..config()
            },
            4,
        );
        let limits = c.backend_limits(&p).unwrap();
        let long_policy = Policy {
            max_conns_by_backend: Some(&limits),
            tier: Some(ContextTier::Long),
            ..Policy::NONE
        };
        let mut long = Vec::new();
        for _ in 0..12 {
            let permit = c
                .try_admit(&p, tier(ContextTier::Long, Some(ContextTier::Long)))
                .unwrap();
            let selection = p
                .select_with_preference_bounded(None, 8, &long_policy)
                .unwrap();
            long.push((permit, selection));
        }
        assert!(p
            .select_with_preference_bounded(None, 8, &long_policy)
            .is_none());
        let barrier = Arc::new(Barrier::new(65));
        let workers: Vec<_> = (0..64)
            .map(|_| {
                let (p, c, barrier) = (p.clone(), c.clone(), barrier.clone());
                std::thread::spawn(move || {
                    let limits = c.backend_limits(&p).unwrap();
                    let policy = Policy {
                        max_conns_by_backend: Some(&limits),
                        tier: Some(ContextTier::Base),
                        ..Policy::NONE
                    };
                    barrier.wait();
                    let held = c
                        .try_admit(&p, tier(ContextTier::Base, Some(ContextTier::Base)))
                        .ok()
                        .map(|permit| {
                            let selection = p
                                .select_with_preference_bounded(Some(0), 8, &policy)
                                .unwrap();
                            (permit, selection)
                        });
                    barrier.wait();
                    barrier.wait();
                    held.is_some()
                })
            })
            .collect();
        barrier.wait();
        barrier.wait();
        assert_eq!(c.inflight(), 48);
        assert!(p.backends()[..3]
            .iter()
            .all(|b| b.lane_conns.load(Ordering::Acquire) <= 16));
        assert_eq!(p.backends()[3].lane_conns.load(Ordering::Acquire), 12);
        barrier.wait();
        assert_eq!(
            workers
                .into_iter()
                .map(|w| usize::from(w.join().unwrap()))
                .sum::<usize>(),
            36
        );
        drop(long);
        assert_eq!(c.inflight(), 0);
        assert!(p
            .backends()
            .iter()
            .all(|b| b.lane_conns.load(Ordering::Acquire) == 0));
        // No reservation for long traffic: base can consume the entire budget.
        let policy = Policy {
            max_conns_by_backend: Some(&limits),
            tier: Some(ContextTier::Base),
            ..Policy::NONE
        };
        let held: Vec<_> = (0..48)
            .map(|_| {
                (
                    c.try_admit(&p, None).unwrap(),
                    p.select_with_preference_bounded(None, 8, &policy).unwrap(),
                )
            })
            .collect();
        assert!(c
            .try_admit(&p, tier(ContextTier::Long, Some(ContextTier::Long)))
            .is_err());
        drop(held);
        assert_eq!(c.inflight(), 0);
    }

    #[test]
    fn disabled_controller_admits_everything_without_counting() {
        let c = Arc::new(AdmissionController::disabled());
        let p = pool(1);
        for _ in 0..100 {
            assert!(c.try_admit(&p, None).unwrap().is_none());
        }
        assert_eq!(c.inflight(), 0);
        assert_eq!(c.host_share(1), None);
        assert!(!c.backend_saturated(0));
    }

    #[test]
    fn budget_bounds_inflight_and_permits_release_on_drop() {
        let c = controller(config(), 2);
        let p = pool(2);
        let a = c.try_admit(&p, None).unwrap().unwrap();
        let b = c.try_admit(&p, None).unwrap().unwrap();
        assert_eq!(c.inflight(), 2);
        let rejected = c.try_admit(&p, None).unwrap_err();
        assert_eq!(rejected.reason, RejectReason::Budget);
        assert_eq!(rejected.retry_after, Duration::from_secs(3));
        drop(a);
        assert_eq!(c.inflight(), 1);
        let _c2 = c.try_admit(&p, None).unwrap().unwrap();
        drop(b);
        assert_eq!(c.inflight(), 1);
    }

    #[test]
    fn host_share_is_the_budget_split_over_healthy_backends() {
        let c = controller(config(), 3);
        // budget 2 over 3 hosts: 1 each; over 1 host: 2.
        assert_eq!(c.host_share(3), Some(1));
        assert_eq!(c.host_share(1), Some(2));
        assert_eq!(c.host_share(0), Some(2));
    }

    #[test]
    fn budget_ramps_only_after_a_clean_interval() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        assert_eq!(c.budget(), 2);
        // Half an interval: nothing.
        drop(
            c.try_admit_at(&p, None, t0 + Duration::from_secs(30))
                .unwrap(),
        );
        assert_eq!(c.budget(), 2);
        // A full clean interval: one step.
        drop(
            c.try_admit_at(&p, None, t0 + Duration::from_secs(61))
                .unwrap(),
        );
        assert_eq!(c.budget(), 4);
        // Back-pressure during the next interval holds the budget...
        let permit = c
            .try_admit_at(&p, None, t0 + Duration::from_secs(70))
            .unwrap()
            .unwrap();
        permit.attach_backend(0);
        permit.observe_backpressure_at(t0 + Duration::from_secs(70));
        drop(permit);
        drop(
            c.try_admit_at(&p, None, t0 + Duration::from_secs(130))
                .unwrap(),
        );
        assert_eq!(c.budget(), 4);
        // ...and the interval after that is clean again.
        drop(
            c.try_admit_at(&p, None, t0 + Duration::from_secs(200))
                .unwrap(),
        );
        assert_eq!(c.budget(), 6);
        drop(
            c.try_admit_at(&p, None, t0 + Duration::from_secs(270))
                .unwrap(),
        );
        assert_eq!(c.budget(), 8);
        // Capped at max.
        drop(
            c.try_admit_at(&p, None, t0 + Duration::from_secs(340))
                .unwrap(),
        );
        assert_eq!(c.budget(), 8);
    }

    #[test]
    fn a_slow_sample_late_in_the_interval_holds_the_ramp() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        // One breach (below the trip threshold) just before the boundary.
        sample(
            &c,
            &p,
            t0 + Duration::from_secs(50),
            Duration::from_secs(11),
        );
        drop(
            c.try_admit_at(&p, None, t0 + Duration::from_secs(62))
                .unwrap(),
        );
        assert_eq!(c.budget(), 2, "the interval with a breach is not clean");
        drop(
            c.try_admit_at(&p, None, t0 + Duration::from_secs(125))
                .unwrap(),
        );
        assert_eq!(c.budget(), 4);
    }

    #[test]
    fn every_backend_queued_refuses_until_the_signal_ages_out() {
        let c = controller(config(), 2);
        let p = pool(2);
        let t0 = Instant::now();
        // Only backend 0 rejected: it is steered around, the other may have room.
        let permit = c.try_admit_at(&p, None, t0).unwrap().unwrap();
        permit.attach_backend(0);
        permit.observe_backpressure_at(t0);
        drop(permit);
        assert!(c.backend_saturated_at(0, t0 + Duration::from_secs(1)));
        assert!(!c.backend_saturated_at(1, t0 + Duration::from_secs(1)));
        assert!(c
            .try_admit_at(&p, None, t0 + Duration::from_secs(1))
            .is_ok());
        // Both rejected: refuse.
        let permit = c.try_admit_at(&p, None, t0).unwrap().unwrap();
        permit.attach_backend(1);
        permit.observe_backpressure_at(t0 + Duration::from_secs(2));
        drop(permit);
        let rejected = c
            .try_admit_at(&p, None, t0 + Duration::from_secs(3))
            .unwrap_err();
        assert_eq!(rejected.reason, RejectReason::BackendQueue);
        // An unhealthy backend does not count; the healthy one is still queued.
        p.backends()[1].healthy.store(false, Ordering::Relaxed);
        let rejected = c
            .try_admit_at(&p, None, t0 + Duration::from_secs(5))
            .unwrap_err();
        assert_eq!(rejected.reason, RejectReason::BackendQueue);
        // Past the TTL the marks expire.
        assert!(!c.backend_saturated_at(0, t0 + Duration::from_secs(20)));
        assert!(c
            .try_admit_at(&p, None, t0 + Duration::from_secs(20))
            .is_ok());
    }

    #[test]
    fn ttft_breaker_needs_enough_samples_and_more_than_one_breach() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        // 19 fast samples + one very slow one: below the minimum sample count.
        for i in 0..19 {
            sample(
                &c,
                &p,
                t0 + Duration::from_millis(i),
                Duration::from_secs(1),
            );
        }
        sample(&c, &p, t0, Duration::from_secs(40));
        assert_eq!(c.ttft_breaches(t0 + Duration::from_secs(41)), Some((20, 1)));
        // 20 samples with a single breach: one slow request is not overload.
        assert!(c
            .try_admit_at(&p, None, t0 + Duration::from_secs(41))
            .is_ok());
        // A second breach (2 of 21 ≥ max(2, ceil(5 %))) trips the breaker.
        sample(&c, &p, t0 + Duration::from_secs(1), Duration::from_secs(41));
        let rejected = c
            .try_admit_at(&p, None, t0 + Duration::from_secs(43))
            .unwrap_err();
        assert_eq!(rejected.reason, RejectReason::Ttft);
        // Samples (stamped when their generation started, ≤ t0+42) fall out
        // of the window and admission resumes.
        let later = t0 + Duration::from_secs(42) + TTFT_WINDOW + Duration::from_secs(2);
        assert!(c.try_admit_at(&p, None, later).is_ok());
        assert_eq!(c.ttft_breaches(later), None);
    }

    #[test]
    fn ttft_breaker_verdict_is_cached_for_a_second() {
        let c = controller(config(), 1);
        let p = pool(1);
        // Everything happens well after the controller's epoch so dispatch
        // times can precede their observations.
        let t0 = Instant::now() + Duration::from_secs(100);
        for i in 0..20 {
            sample(
                &c,
                &p,
                t0 + Duration::from_millis(i),
                Duration::from_secs(1),
            );
        }
        // Evaluated (clean) at t1; two breaches observed right after are not
        // seen until the cache expires.
        let t1 = t0 + Duration::from_secs(2);
        assert!(c.try_admit_at(&p, None, t1).is_ok());
        for _ in 0..2 {
            let permit = c.try_admit_at(&p, None, t1).unwrap().unwrap();
            permit.mark_dispatched_at(t1 - Duration::from_secs(20));
            permit.observe_generation_started_at(t1);
        }
        assert!(c
            .try_admit_at(&p, None, t1 + Duration::from_millis(500))
            .is_ok());
        let rejected = c
            .try_admit_at(&p, None, t1 + Duration::from_millis(1500))
            .unwrap_err();
        assert_eq!(rejected.reason, RejectReason::Ttft);
    }

    #[test]
    fn ttft_check_is_off_without_a_bound() {
        let c = controller(
            AdmissionConfig {
                ttft_p95_max: None,
                ..config()
            },
            1,
        );
        let p = pool(1);
        let t0 = Instant::now();
        for _ in 0..25 {
            sample(&c, &p, t0, Duration::from_secs(100));
        }
        assert!(c
            .try_admit_at(&p, None, t0 + Duration::from_secs(1))
            .is_ok());
    }

    #[test]
    fn generation_start_is_recorded_once_and_only_after_dispatch() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        // Not dispatched yet: nothing to observe.
        let permit = c.try_admit_at(&p, None, t0).unwrap().unwrap();
        permit.observe_generation_started_at(t0 + Duration::from_secs(50));
        assert_eq!(c.ttft_totals(t0 + Duration::from_secs(50)), (0, 0));
        // Dispatched 5 s after admission (image validation): the clock starts
        // at dispatch, and a second observation (50 s, a breach) is ignored.
        permit.mark_dispatched_at(t0 + Duration::from_secs(5));
        permit.observe_generation_started_at(t0 + Duration::from_millis(5100));
        permit.observe_generation_started_at(t0 + Duration::from_secs(55));
        permit.release_at(t0 + Duration::from_secs(60));
        std::mem::forget(permit);
        assert_eq!(c.ttft_totals(t0 + Duration::from_secs(60)), (1, 0));
    }

    #[test]
    fn a_dispatched_request_without_generation_is_a_censored_sample() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        // Never reached the engine: nothing recorded.
        let permit = c.try_admit_at(&p, None, t0).unwrap().unwrap();
        permit.release_at(t0 + Duration::from_secs(30));
        std::mem::forget(permit);
        assert_eq!(c.ttft_totals(t0 + Duration::from_secs(30)), (0, 0));
        // Accepted, then the client gave up after 30 s: one breaching sample.
        let permit = c.try_admit_at(&p, None, t0).unwrap().unwrap();
        permit.mark_dispatched_at(t0);
        permit.release_at(t0 + Duration::from_secs(30));
        std::mem::forget(permit);
        assert_eq!(c.ttft_totals(t0 + Duration::from_secs(30)), (1, 1));
        // Accepted but the stream's first event was an engine rejection: no
        // sample, and a generation event after abandoning is ignored too.
        let permit = c.try_admit_at(&p, None, t0).unwrap().unwrap();
        permit.mark_dispatched_at(t0);
        permit.abandon();
        permit.observe_generation_started_at(t0 + Duration::from_secs(1));
        permit.release_at(t0 + Duration::from_secs(1));
        std::mem::forget(permit);
        assert_eq!(c.ttft_totals(t0 + Duration::from_secs(30)), (1, 1));
        // An error after generation started is a mid-stream failure, not a
        // rejection: the sample stays.
        let permit = c.try_admit_at(&p, None, t0).unwrap().unwrap();
        permit.mark_dispatched_at(t0);
        permit.observe_generation_started_at(t0 + Duration::from_secs(1));
        permit.abandon();
        permit.release_at(t0 + Duration::from_secs(2));
        std::mem::forget(permit);
        assert_eq!(c.ttft_totals(t0 + Duration::from_secs(30)), (2, 1));
        assert_eq!(c.inflight(), 0);
    }

    #[test]
    fn window_covers_exactly_the_last_minute_at_any_rate() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        // A burst of slow samples, then far more fast ones than any cap.
        for _ in 0..50 {
            sample(&c, &p, t0, Duration::from_secs(20));
        }
        for i in 0..10_000u64 {
            sample(
                &c,
                &p,
                t0 + Duration::from_millis(i % 30_000),
                Duration::from_millis(1),
            );
        }
        let (samples, breaches) = c.ttft_totals(t0 + Duration::from_secs(30));
        assert_eq!(
            (samples, breaches),
            (10_050, 50),
            "nothing evicted inside the window"
        );
        // The slow burst (observed at t0+20) ages out a minute later, the
        // fast samples (observed up to t0+30) ten seconds after that.
        assert_eq!(c.ttft_totals(t0 + Duration::from_secs(79)).1, 50);
        assert_eq!(c.ttft_totals(t0 + Duration::from_secs(81)).1, 0);
        assert_eq!(c.ttft_totals(t0 + Duration::from_secs(120)), (0, 0));
    }

    #[test]
    fn a_queueing_engine_counts_as_saturated_until_the_sample_ages() {
        let engine = Arc::new(EngineLoad::new(2, Duration::from_secs(6)));
        let c = Arc::new(AdmissionController::new(Some(config()), 2, engine.clone()));
        let p = pool(2);
        let t0 = Instant::now();
        let busy = crate::engine_load::Sample {
            running: 30,
            queued: 2,
        };
        let idle = crate::engine_load::Sample {
            running: 3,
            queued: 0,
        };
        engine.record_at(0, busy, t0);
        engine.record_at(1, idle, t0);
        assert!(c.backend_saturated_at(0, t0 + Duration::from_secs(1)));
        assert!(!c.backend_saturated_at(1, t0 + Duration::from_secs(1)));
        assert_eq!(c.engine(0), Some((30, 2)));
        assert!(c
            .try_admit_at(&p, None, t0 + Duration::from_secs(1))
            .is_ok());
        // Every host queueing: refuse before dispatch.
        engine.record_at(1, busy, t0 + Duration::from_secs(2));
        let rejected = c
            .try_admit_at(&p, None, t0 + Duration::from_secs(3))
            .unwrap_err();
        assert_eq!(rejected.reason, RejectReason::BackendQueue);
        // Stale samples are unknown, not saturation.
        assert!(!c.backend_saturated_at(0, t0 + Duration::from_secs(10)));
        assert!(c
            .try_admit_at(&p, None, t0 + Duration::from_secs(10))
            .is_ok());
    }

    #[test]
    fn a_refusal_after_dispatch_records_nothing() {
        // Fail-over that ends in a refusal (every other host at its share):
        // the request was marked dispatched but never waited on an engine.
        let c = controller(config(), 2);
        let p = pool(2);
        let t0 = Instant::now();
        let permit = c.try_admit_at(&p, None, t0).unwrap().unwrap();
        permit.mark_dispatched_at(t0);
        permit.abandon();
        permit.release_at(t0 + Duration::from_millis(30));
        std::mem::forget(permit);
        assert_eq!(c.ttft_totals(t0 + Duration::from_secs(1)), (0, 0));
        assert_eq!(c.inflight(), 0);
    }

    #[test]
    fn a_long_context_request_does_not_add_a_ttft_sample() {
        let c = controller(config(), 2);
        let p = pool(2);
        let t0 = Instant::now();
        // A base request that waited 40 s is one breaching sample...
        let base = tier(ContextTier::Base, Some(ContextTier::Base));
        let permit = c.try_admit_at(&p, base, t0).unwrap().unwrap();
        permit.attach_backend(0);
        permit.mark_dispatched_at(t0);
        permit.observe_generation_started_at(t0 + Duration::from_secs(40));
        drop(permit);
        assert_eq!(c.ttft_totals(t0 + Duration::from_secs(40)), (1, 1));
        // ...the same wait on a long-context request is not a sample at all:
        // a 100k-token prefill takes that long by nature, on either tier —
        // this one fell back onto the base fleet.
        let long = tier(ContextTier::Long, None);
        let permit = c.try_admit_at(&p, long, t0).unwrap().unwrap();
        permit.attach_backend(1);
        permit.mark_dispatched_at(t0);
        permit.observe_generation_started_at(t0 + Duration::from_secs(40));
        drop(permit);
        // Nor is the censored one a client gave up on.
        let permit = c.try_admit_at(&p, long, t0).unwrap().unwrap();
        permit.attach_backend(1);
        permit.mark_dispatched_at(t0);
        permit.release_at(t0 + Duration::from_secs(60));
        std::mem::forget(permit);
        assert_eq!(c.ttft_totals(t0 + Duration::from_secs(60)), (1, 1));
    }

    #[test]
    fn the_ttft_breaker_does_not_refuse_long_context_requests() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        // Trip the breaker on base traffic: 20 samples, two of them slow.
        for i in 0..18 {
            sample(
                &c,
                &p,
                t0 + Duration::from_millis(i),
                Duration::from_secs(1),
            );
        }
        for _ in 0..2 {
            sample(&c, &p, t0, Duration::from_secs(40));
        }
        let t1 = t0 + Duration::from_secs(41);
        assert_eq!(
            c.try_admit_at(&p, None, t1).unwrap_err().reason,
            RejectReason::Ttft
        );
        // An oversized request bound for the long tier is not what the window
        // measured, and that host may well have room: it is still admitted.
        assert!(c
            .try_admit_at(&p, tier(ContextTier::Long, Some(ContextTier::Long)), t1)
            .is_ok());
        // One that fell back onto the base fleet, though, is going exactly
        // where the breaker is tripped.
        assert_eq!(
            c.try_admit_at(&p, tier(ContextTier::Long, None), t1)
                .unwrap_err()
                .reason,
            RejectReason::Ttft
        );
    }

    #[test]
    fn the_queue_refusal_counts_only_the_requests_own_tier() {
        let c = controller(config(), 2);
        let p = BackendPool::with_long_context(
            vec!["http://b0:8000".to_string()],
            vec!["http://long:8000".to_string()],
        );
        let t0 = Instant::now();
        // The base host rejected at engine admission, the long one has room.
        let permit = c.try_admit_at(&p, None, t0).unwrap().unwrap();
        permit.attach_backend(0);
        permit.observe_backpressure_at(t0);
        drop(permit);
        let t1 = t0 + Duration::from_secs(1);
        let base = tier(ContextTier::Base, Some(ContextTier::Base));
        let rejected = c.try_admit_at(&p, base, t1).unwrap_err();
        assert_eq!(rejected.reason, RejectReason::BackendQueue);
        let long = tier(ContextTier::Long, Some(ContextTier::Long));
        assert!(c.try_admit_at(&p, long, t1).is_ok());
        assert!(c.try_admit_at(&p, None, t1).is_ok());
    }

    #[test]
    fn precheck_refuses_without_taking_a_slot() {
        let c = controller(config(), 1);
        let p = pool(1);
        assert!(c.precheck(&p, None).is_ok());
        let _a = c.try_admit(&p, None).unwrap().unwrap();
        let _b = c.try_admit(&p, None).unwrap().unwrap();
        assert_eq!(
            c.precheck(&p, None).unwrap_err().reason,
            RejectReason::Budget
        );
        assert_eq!(c.inflight(), 2);
    }

    #[tokio::test]
    async fn empty_recency_state_keeps_cold_admission_behavior() {
        let c = controller(
            AdmissionConfig {
                max_inflight: 1,
                start_inflight: 1,
                continuation: Some(continuation_config()),
                ..config()
            },
            1,
        );
        let p = pool(1);
        let held = c
            .admit(&p, None, AdmissionClass::Cold)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(c.waiter_count(), 0);
        assert_eq!(
            c.admit(&p, None, AdmissionClass::Cold)
                .await
                .unwrap_err()
                .reason,
            RejectReason::Budget
        );
        assert_eq!(c.waiter_count(), 0, "cold requests are never buffered");
        drop(held);
    }

    #[tokio::test]
    async fn recent_prefix_gets_the_released_slot_before_new_cold_work() {
        let c = controller(
            AdmissionConfig {
                max_inflight: 1,
                start_inflight: 1,
                continuation: Some(continuation_config()),
                ..config()
            },
            1,
        );
        let p = Arc::new(pool(1));
        let held = c
            .admit(&p, None, AdmissionClass::Cold)
            .await
            .unwrap()
            .unwrap();

        let waiting = {
            let c = c.clone();
            let p = p.clone();
            tokio::spawn(async move { c.admit(&p, None, recent(0)).await })
        };
        tokio::time::timeout(Duration::from_secs(1), async {
            while c.waiter_count() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        drop(held);
        assert_eq!(
            c.admit(&p, None, AdmissionClass::Cold)
                .await
                .unwrap_err()
                .reason,
            RejectReason::ContinuationHandoff
        );
        let hot = waiting.await.unwrap().unwrap().unwrap();
        assert!(hot.backend_allowed(0));
        assert!(!hot.backend_allowed(1));
        hot.attach_backend(0);
        drop(hot);
        assert_eq!(c.waiter_count(), 0);
    }

    #[tokio::test]
    async fn timed_out_waiter_releases_every_bound() {
        let mut continuation = continuation_config();
        continuation.max_wait = Duration::from_millis(10);
        let c = controller(
            AdmissionConfig {
                max_inflight: 1,
                start_inflight: 1,
                continuation: Some(continuation),
                ..config()
            },
            1,
        );
        let p = pool(1);
        let held = c
            .admit(&p, None, AdmissionClass::Cold)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            c.admit(&p, None, recent(0)).await.unwrap_err().reason,
            RejectReason::Budget
        );
        assert_eq!(c.waiter_count(), 0);
        assert_eq!(c.waiter_usage(), (0, 0));
        drop(held);
    }

    #[tokio::test]
    async fn backend_probe_wakes_a_waiter_without_blocking_ready_cold_capacity() {
        let engine = Arc::new(EngineLoad::new(2, Duration::from_secs(10)));
        let c = Arc::new(AdmissionController::new(
            Some(AdmissionConfig {
                max_inflight: 2,
                start_inflight: 2,
                continuation: Some(continuation_config()),
                ..config()
            }),
            2,
            engine.clone(),
        ));
        let p = Arc::new(pool(2));
        engine.record_at(
            0,
            crate::engine_load::Sample {
                running: 1,
                queued: 1,
            },
            Instant::now(),
        );
        let waiting = {
            let c = c.clone();
            let p = p.clone();
            tokio::spawn(async move { c.admit(&p, None, recent(0)).await })
        };
        tokio::time::timeout(Duration::from_secs(1), async {
            while c.waiter_count() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let cold = c
            .admit(&p, None, AdmissionClass::Cold)
            .await
            .expect("a blocked warm backend must not reserve unrelated capacity")
            .unwrap();
        engine.record_at(
            0,
            crate::engine_load::Sample {
                running: 1,
                queued: 0,
            },
            Instant::now(),
        );
        let hot = tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(hot.backend_allowed(0));
        drop(cold);
        drop(hot);
    }

    #[tokio::test]
    async fn admitted_warm_handoff_keeps_cold_placement_off_its_backend() {
        let c = controller(
            AdmissionConfig {
                max_inflight: 2,
                start_inflight: 2,
                continuation: Some(continuation_config()),
                ..config()
            },
            2,
        );
        let p = pool(2);
        let hot = c.admit(&p, None, recent(0)).await.unwrap().unwrap();
        let cold = c
            .admit(&p, None, AdmissionClass::Cold)
            .await
            .unwrap()
            .unwrap();
        assert!(!cold.backend_allowed(0));
        assert!(cold.backend_allowed(1));
        hot.attach_backend(0);
        assert!(cold.backend_allowed(0));
    }
}
