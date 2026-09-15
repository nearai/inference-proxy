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
//! 3. **Per-host share.** `ceil(budget / healthy backends)` in flight per
//!    backend, so a conversation-affinity pin cannot pile the whole budget onto
//!    one host. Selection (`BackendPool::select_with_preference_bounded`)
//!    reserves the slot atomically and only on backends under their share: a
//!    pinned conversation moves when its host is full, and only when no host
//!    has room is the request refused.
//!
//! Everything is derived from what the gateway observes itself; no admin
//! endpoint or extra token is involved. Disabled (`max_inflight = 0`) the
//! module is inert and the in-CVM behavior is unchanged.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

use crate::backend_pool::BackendPool;

/// Window over which time-to-first-generation samples are kept.
pub const TTFT_WINDOW: Duration = Duration::from_secs(60);
/// Samples needed in the window before the bound is enforced at all.
pub const TTFT_MIN_SAMPLES: usize = 20;
/// Cap on retained samples (oldest dropped first) so evaluation stays cheap.
pub const TTFT_MAX_SAMPLES: usize = 4096;
/// Fraction of the window that must breach the bound to trip, with a floor
/// of `TTFT_MIN_BREACHES`, so one slow request cannot close the fleet.
pub const TTFT_BREACH_FRACTION: f64 = 0.05;
pub const TTFT_MIN_BREACHES: usize = 2;
/// The breaker is re-evaluated at most this often; refusals in between reuse
/// the cached verdict.
const BREAKER_REEVALUATE_AFTER: Duration = Duration::from_secs(1);
/// Marker for a permit that has not been attached to a backend yet.
const NO_BACKEND: usize = usize::MAX;

/// Operator settings, parsed and validated by `Config::from_env`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionConfig {
    /// Hard ceiling on lane requests in flight across the fleet.
    pub max_inflight: u32,
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
}

impl RejectReason {
    pub fn as_str(self) -> &'static str {
        match self {
            RejectReason::Budget => "budget",
            RejectReason::HostShare => "host_share",
            RejectReason::BackendQueue => "backend_queue",
            RejectReason::Ttft => "ttft",
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

/// Fleet-wide admission state shared by every request (`AppState.admission`).
pub struct AdmissionController {
    config: Option<AdmissionConfig>,
    inflight: AtomicU32,
    budget: AtomicU32,
    ramp: Mutex<Ramp>,
    /// `(observed_at, ttft)` samples, oldest first, pruned to `TTFT_WINDOW`
    /// and capped at `TTFT_MAX_SAMPLES`.
    ttft: Mutex<VecDeque<(Instant, Duration)>>,
    breaker: Mutex<Breaker>,
    /// Per backend index: last engine admission rejection as milliseconds
    /// since `epoch`, plus one so that zero means "never".
    backpressure: Vec<AtomicU64>,
    epoch: Instant,
    queue_tripped: AtomicBool,
}

impl AdmissionController {
    /// `backend_count` sizes the per-backend back-pressure slots; it must be
    /// the pool size (backend indexes are stable for the process lifetime).
    pub fn new(config: Option<AdmissionConfig>, backend_count: usize) -> Self {
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
            ttft: Mutex::new(VecDeque::new()),
            breaker: Mutex::new(Breaker {
                evaluated_at: None,
                tripped: false,
            }),
            backpressure: (0..backend_count).map(|_| AtomicU64::new(0)).collect(),
            epoch: now,
            queue_tripped: AtomicBool::new(false),
        }
    }

    /// An inert controller: every request is admitted, nothing is counted.
    pub fn disabled() -> Self {
        Self::new(None, 0)
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

    /// Per-backend in-flight bound for the current budget, `None` when
    /// admission is disabled. With no healthy backend the share is computed
    /// as for one, so the (degraded) selection still has a bound to apply.
    pub fn host_share(&self, healthy_backends: usize) -> Option<u32> {
        self.config.as_ref()?;
        let budget = self.budget.load(Ordering::Relaxed).max(1);
        let hosts = u32::try_from(healthy_backends.max(1)).unwrap_or(u32::MAX);
        Some(budget.div_ceil(hosts))
    }

    /// Whether backend `index` rejected at engine admission within the TTL,
    /// i.e. selection should steer around it while other hosts have room.
    pub fn backend_saturated(&self, index: usize) -> bool {
        self.backend_saturated_at(index, Instant::now())
    }

    pub(crate) fn backend_saturated_at(&self, index: usize, now: Instant) -> bool {
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

    /// Run the overload and budget checks for a new request. `Ok(None)` when
    /// admission is disabled; `Ok(Some(permit))` holds one budget slot until
    /// the permit is dropped. The per-host share is applied by the caller at
    /// selection time (`host_share`, `backend_saturated`), since it needs the
    /// chosen backend.
    pub fn try_admit(self: &Arc<Self>, pool: &BackendPool) -> Result<Option<Permit>, Rejected> {
        self.try_admit_at(pool, Instant::now())
    }

    pub(crate) fn try_admit_at(
        self: &Arc<Self>,
        pool: &BackendPool,
        now: Instant,
    ) -> Result<Option<Permit>, Rejected> {
        let Some(config) = &self.config else {
            return Ok(None);
        };
        // Overload first, so a signal that just arrived cannot be preceded by
        // a ramp step that treats the interval as clean.
        if self.ttft_over_bound(config, now) {
            return Err(self.reject(RejectReason::Ttft));
        }
        if self.every_backend_queued(config, pool, now) {
            return Err(self.reject(RejectReason::BackendQueue));
        }
        self.tick_ramp(config, now);
        let mut current = self.inflight.load(Ordering::Acquire);
        loop {
            if current >= self.budget.load(Ordering::Relaxed) {
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
        metrics::gauge!("admission_inflight").increment(1.0);
        Ok(Some(Permit {
            controller: Arc::clone(self),
            started: now,
            backend: AtomicUsize::new(NO_BACKEND),
            dispatched: AtomicBool::new(false),
            generation_seen: AtomicBool::new(false),
        }))
    }

    fn ramp(&self) -> MutexGuard<'_, Ramp> {
        self.ramp.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn ttft_window(&self) -> MutexGuard<'_, VecDeque<(Instant, Duration)>> {
        self.ttft.lock().unwrap_or_else(|e| e.into_inner())
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
        if self
            .config
            .as_ref()
            .and_then(|c| c.ttft_p95_max)
            .is_some_and(|max| ttft > max)
        {
            // A breach counts against the current ramp interval even if the
            // breaker is not (yet) tripped.
            self.mark_dirty();
        }
        let mut window = self.ttft_window();
        window.push_back((now, ttft));
        while window.len() > TTFT_MAX_SAMPLES {
            window.pop_front();
        }
        prune(&mut window, now);
    }

    /// `(samples, breaches)` in the window, or `None` below the minimum.
    pub fn ttft_breaches(&self, max: Duration, now: Instant) -> Option<(usize, usize)> {
        let mut window = self.ttft_window();
        prune(&mut window, now);
        if window.len() < TTFT_MIN_SAMPLES {
            return None;
        }
        let breaches = window.iter().filter(|(_, d)| *d > max).count();
        Some((window.len(), breaches))
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
            let over = self
                .ttft_breaches(max, now)
                .is_some_and(|(samples, breaches)| {
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

    /// Every healthy backend rejected at engine admission within the TTL.
    fn every_backend_queued(
        &self,
        config: &AdmissionConfig,
        pool: &BackendPool,
        now: Instant,
    ) -> bool {
        let mut healthy = 0usize;
        let mut all_queued = true;
        for (index, backend) in pool.backends().iter().enumerate() {
            if !backend.healthy.load(Ordering::Relaxed) {
                continue;
            }
            healthy += 1;
            if !self.backend_saturated_at(index, now) {
                all_queued = false;
                break;
            }
        }
        let queued = healthy > 0 && all_queued;
        if queued != self.queue_tripped.swap(queued, Ordering::Relaxed) {
            if queued {
                warn!(
                    healthy_backends = healthy,
                    ttl_secs = config.backpressure_ttl.as_secs(),
                    "Every backend rejected at engine admission recently, refusing new work"
                );
            } else {
                info!("A backend accepts lane work again");
            }
        }
        queued
    }
}

fn prune(window: &mut VecDeque<(Instant, Duration)>, now: Instant) {
    while let Some((at, _)) = window.front() {
        if now.saturating_duration_since(*at) > TTFT_WINDOW {
            window.pop_front();
        } else {
            break;
        }
    }
}

/// One admitted request's budget slot. Dropping it releases the slot; the
/// streaming path moves it into the pump task next to the backend guard so
/// the slot is held for the whole stream.
pub struct Permit {
    controller: Arc<AdmissionController>,
    started: Instant,
    backend: AtomicUsize,
    /// The engine accepted the request (2xx); a first-generation sample is
    /// expected, and its absence at drop is a censored slow observation.
    dispatched: AtomicBool,
    generation_seen: AtomicBool,
}

impl Permit {
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

    /// Steer selection around backends that rejected recently.
    pub fn backend_saturated(&self, index: usize) -> bool {
        self.controller.backend_saturated(index)
    }

    /// The engine accepted the request; the clock now runs against the
    /// first generation event.
    pub fn mark_dispatched(&self) {
        self.dispatched.store(true, Ordering::Relaxed);
    }

    /// The accepted request turned out to be an engine rejection (an error
    /// event in the stream): no TTFT observation for it.
    pub fn abandon(&self) {
        self.dispatched.store(false, Ordering::Relaxed);
    }

    /// The first generation event arrived: one TTFT sample. Idempotent.
    pub fn observe_generation_started(&self) {
        self.observe_generation_started_at(Instant::now());
    }

    pub(crate) fn observe_generation_started_at(&self, now: Instant) {
        if self.generation_seen.swap(true, Ordering::Relaxed) {
            return;
        }
        self.controller
            .record_ttft(now, now.saturating_duration_since(self.started));
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
        if self.dispatched.load(Ordering::Relaxed)
            && !self.generation_seen.swap(true, Ordering::Relaxed)
        {
            self.controller
                .record_ttft(now, now.saturating_duration_since(self.started));
        }
        self.controller.inflight.fetch_sub(1, Ordering::AcqRel);
        metrics::gauge!("admission_inflight").decrement(1.0);
    }
}

impl std::fmt::Debug for Permit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Permit")
            .field("backend", &self.backend())
            .field("dispatched", &self.dispatched.load(Ordering::Relaxed))
            .field(
                "generation_seen",
                &self.generation_seen.load(Ordering::Relaxed),
            )
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
            start_inflight: 2,
            ramp_step: 2,
            ramp_interval: Duration::from_secs(60),
            ttft_p95_max: Some(Duration::from_secs(10)),
            backpressure_ttl: Duration::from_secs(10),
            retry_after: Duration::from_secs(3),
        }
    }

    fn pool(n: usize) -> BackendPool {
        BackendPool::new((0..n).map(|i| format!("http://b{i}:8000")).collect())
    }

    fn controller(config: AdmissionConfig, backends: usize) -> Arc<AdmissionController> {
        Arc::new(AdmissionController::new(Some(config), backends))
    }

    /// Admit at `t0` and record a first-generation sample `ttft` later.
    fn sample(c: &Arc<AdmissionController>, p: &BackendPool, t0: Instant, ttft: Duration) {
        let permit = c.try_admit_at(p, t0).unwrap().unwrap();
        permit.mark_dispatched();
        permit.observe_generation_started_at(t0 + ttft);
    }

    #[test]
    fn disabled_controller_admits_everything_without_counting() {
        let c = Arc::new(AdmissionController::disabled());
        let p = pool(1);
        for _ in 0..100 {
            assert!(c.try_admit(&p).unwrap().is_none());
        }
        assert_eq!(c.inflight(), 0);
        assert_eq!(c.host_share(1), None);
        assert!(!c.backend_saturated(0));
    }

    #[test]
    fn budget_bounds_inflight_and_permits_release_on_drop() {
        let c = controller(config(), 2);
        let p = pool(2);
        let a = c.try_admit(&p).unwrap().unwrap();
        let b = c.try_admit(&p).unwrap().unwrap();
        assert_eq!(c.inflight(), 2);
        let rejected = c.try_admit(&p).unwrap_err();
        assert_eq!(rejected.reason, RejectReason::Budget);
        assert_eq!(rejected.retry_after, Duration::from_secs(3));
        drop(a);
        assert_eq!(c.inflight(), 1);
        let _c2 = c.try_admit(&p).unwrap().unwrap();
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
        drop(c.try_admit_at(&p, t0 + Duration::from_secs(30)).unwrap());
        assert_eq!(c.budget(), 2);
        // A full clean interval: one step.
        drop(c.try_admit_at(&p, t0 + Duration::from_secs(61)).unwrap());
        assert_eq!(c.budget(), 4);
        // Back-pressure during the next interval holds the budget...
        let permit = c
            .try_admit_at(&p, t0 + Duration::from_secs(70))
            .unwrap()
            .unwrap();
        permit.attach_backend(0);
        permit.observe_backpressure_at(t0 + Duration::from_secs(70));
        drop(permit);
        drop(c.try_admit_at(&p, t0 + Duration::from_secs(130)).unwrap());
        assert_eq!(c.budget(), 4);
        // ...and the interval after that is clean again.
        drop(c.try_admit_at(&p, t0 + Duration::from_secs(200)).unwrap());
        assert_eq!(c.budget(), 6);
        drop(c.try_admit_at(&p, t0 + Duration::from_secs(270)).unwrap());
        assert_eq!(c.budget(), 8);
        // Capped at max.
        drop(c.try_admit_at(&p, t0 + Duration::from_secs(340)).unwrap());
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
        drop(c.try_admit_at(&p, t0 + Duration::from_secs(62)).unwrap());
        assert_eq!(c.budget(), 2, "the interval with a breach is not clean");
        drop(c.try_admit_at(&p, t0 + Duration::from_secs(125)).unwrap());
        assert_eq!(c.budget(), 4);
    }

    #[test]
    fn every_backend_queued_refuses_until_the_signal_ages_out() {
        let c = controller(config(), 2);
        let p = pool(2);
        let t0 = Instant::now();
        // Only backend 0 rejected: it is steered around, the other may have room.
        let permit = c.try_admit_at(&p, t0).unwrap().unwrap();
        permit.attach_backend(0);
        permit.observe_backpressure_at(t0);
        drop(permit);
        assert!(c.backend_saturated_at(0, t0 + Duration::from_secs(1)));
        assert!(!c.backend_saturated_at(1, t0 + Duration::from_secs(1)));
        assert!(c.try_admit_at(&p, t0 + Duration::from_secs(1)).is_ok());
        // Both rejected: refuse.
        let permit = c.try_admit_at(&p, t0).unwrap().unwrap();
        permit.attach_backend(1);
        permit.observe_backpressure_at(t0 + Duration::from_secs(2));
        drop(permit);
        let rejected = c.try_admit_at(&p, t0 + Duration::from_secs(3)).unwrap_err();
        assert_eq!(rejected.reason, RejectReason::BackendQueue);
        // An unhealthy backend does not count; the healthy one is still queued.
        p.backends()[1].healthy.store(false, Ordering::Relaxed);
        let rejected = c.try_admit_at(&p, t0 + Duration::from_secs(5)).unwrap_err();
        assert_eq!(rejected.reason, RejectReason::BackendQueue);
        // Past the TTL the marks expire.
        assert!(!c.backend_saturated_at(0, t0 + Duration::from_secs(20)));
        assert!(c.try_admit_at(&p, t0 + Duration::from_secs(20)).is_ok());
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
        assert_eq!(
            c.ttft_breaches(Duration::from_secs(10), t0 + Duration::from_secs(41)),
            Some((20, 1))
        );
        // 20 samples with a single breach: one slow request is not overload.
        assert!(c.try_admit_at(&p, t0 + Duration::from_secs(41)).is_ok());
        // A second breach (2 of 21 ≥ max(2, ceil(5 %))) trips the breaker.
        sample(&c, &p, t0 + Duration::from_secs(1), Duration::from_secs(41));
        let rejected = c
            .try_admit_at(&p, t0 + Duration::from_secs(43))
            .unwrap_err();
        assert_eq!(rejected.reason, RejectReason::Ttft);
        // Samples (stamped when their generation started, ≤ t0+42) fall out
        // of the window and admission resumes.
        let later = t0 + Duration::from_secs(42) + TTFT_WINDOW + Duration::from_secs(2);
        assert!(c.try_admit_at(&p, later).is_ok());
        assert_eq!(c.ttft_breaches(Duration::from_secs(10), later), None);
    }

    #[test]
    fn ttft_breaker_verdict_is_cached_for_a_second() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        for i in 0..20 {
            sample(
                &c,
                &p,
                t0 + Duration::from_millis(i),
                Duration::from_secs(1),
            );
        }
        // Evaluated (clean) at t1; two breaches recorded right after are not
        // seen until the cache expires.
        let t1 = t0 + Duration::from_secs(2);
        assert!(c.try_admit_at(&p, t1).is_ok());
        sample(&c, &p, t1, Duration::from_secs(20));
        sample(&c, &p, t1, Duration::from_secs(20));
        assert!(c.try_admit_at(&p, t1 + Duration::from_millis(500)).is_ok());
        let rejected = c
            .try_admit_at(&p, t1 + Duration::from_millis(1500))
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
        assert!(c.try_admit_at(&p, t0 + Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn generation_start_is_recorded_once_per_permit() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        let permit = c.try_admit_at(&p, t0).unwrap().unwrap();
        permit.mark_dispatched();
        permit.observe_generation_started_at(t0 + Duration::from_millis(100));
        permit.observe_generation_started_at(t0 + Duration::from_secs(50));
        permit.release_at(t0 + Duration::from_secs(60));
        std::mem::forget(permit);
        let window = c.ttft_window();
        assert_eq!(window.len(), 1);
        assert_eq!(window[0].1, Duration::from_millis(100));
    }

    #[test]
    fn a_dispatched_request_without_generation_is_a_censored_sample() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        // Never reached the engine: nothing recorded.
        let permit = c.try_admit_at(&p, t0).unwrap().unwrap();
        permit.release_at(t0 + Duration::from_secs(30));
        std::mem::forget(permit);
        assert!(c.ttft_window().is_empty());
        // Accepted, then the client gave up after 30 s: a 30 s observation.
        let permit = c.try_admit_at(&p, t0).unwrap().unwrap();
        permit.mark_dispatched();
        permit.release_at(t0 + Duration::from_secs(30));
        std::mem::forget(permit);
        assert_eq!(c.ttft_window()[0].1, Duration::from_secs(30));
        // Accepted but the stream's first event was an engine rejection.
        let permit = c.try_admit_at(&p, t0).unwrap().unwrap();
        permit.mark_dispatched();
        permit.abandon();
        permit.release_at(t0 + Duration::from_secs(1));
        std::mem::forget(permit);
        assert_eq!(c.ttft_window().len(), 1);
        assert_eq!(c.inflight(), 0);
    }

    #[test]
    fn window_is_capped() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        for _ in 0..(TTFT_MAX_SAMPLES + 10) {
            sample(&c, &p, t0, Duration::from_millis(5));
        }
        assert_eq!(c.ttft_window().len(), TTFT_MAX_SAMPLES);
    }
}
