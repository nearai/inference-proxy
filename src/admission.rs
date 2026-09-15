//! Lane admission for gateway mode (`VLLM_PROXY_ADMISSION_*`).
//!
//! The gateway fronts a fixed set of inference hosts that also serve direct
//! traffic. Under overload its job is to refuse *early* — a 429 with
//! `Retry-After` before anything is sent upstream — instead of queueing and
//! eventually failing with a 5xx or a minutes-long time to first token. Three
//! checks run on every chat/completions request, in this order:
//!
//! 1. **Observed overload.** The lane's own time-to-first-chunk over the last
//!    minute has a p95 above the configured bound, or every healthy backend
//!    answered an engine admission rejection ("The request queue is full.",
//!    "aborted by a higher priority request") within the last few seconds.
//!    Either means the engines are saturated for lane traffic, so new work is
//!    refused until the signal ages out.
//! 2. **Global in-flight budget.** At most `budget` lane requests are in
//!    flight across the fleet. The budget starts at `start_inflight` and grows
//!    by `ramp_step` every `ramp_interval` up to `max_inflight`, but only after
//!    an interval without any overload signal.
//! 3. **Per-host share.** `ceil(budget / healthy backends)` in flight per
//!    backend, so a conversation-affinity pin cannot pile the whole budget onto
//!    one host. Selection (`BackendPool::select_with_preference_bounded`) only
//!    considers backends under their share: a pinned conversation moves when
//!    its host is full, and only when no host has room is the request refused.
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

/// Window over which lane time-to-first-chunk samples are kept.
pub const TTFT_WINDOW: Duration = Duration::from_secs(60);
/// Minimum samples in the window before the p95 bound is enforced.
pub const TTFT_MIN_SAMPLES: usize = 5;
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
    /// Refuse new work while the lane's TTFT p95 over the window is above
    /// this (`None` = no TTFT check).
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
    /// Every backend is at its per-host share.
    HostShare,
    /// Every healthy backend recently rejected at engine admission.
    BackendQueue,
    /// The lane's own time-to-first-chunk p95 is above the bound.
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

/// Fleet-wide admission state shared by every request (`AppState.admission`).
pub struct AdmissionController {
    config: Option<AdmissionConfig>,
    inflight: AtomicU32,
    budget: AtomicU32,
    ramp: Mutex<Ramp>,
    /// `(observed_at, ttft)` samples, oldest first, pruned to `TTFT_WINDOW`.
    ttft: Mutex<VecDeque<(Instant, Duration)>>,
    /// Per backend index: last engine admission rejection as milliseconds
    /// since `epoch`, plus one so that zero means "never".
    backpressure: Vec<AtomicU64>,
    epoch: Instant,
    ttft_tripped: AtomicBool,
    queue_tripped: AtomicBool,
}

impl AdmissionController {
    /// `backend_count` sizes the per-backend back-pressure slots; it must be
    /// the pool size (backend indexes are stable for the process lifetime).
    pub fn new(config: Option<AdmissionConfig>, backend_count: usize) -> Self {
        let now = Instant::now();
        let budget = config.as_ref().map_or(0, |c| c.start_inflight);
        if let Some(config) = &config {
            metrics::gauge!("admission_budget").set(budget as f64);
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
            backpressure: (0..backend_count).map(|_| AtomicU64::new(0)).collect(),
            epoch: now,
            ttft_tripped: AtomicBool::new(false),
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
    /// selection time (`host_share`), since it needs the chosen backend.
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
        self.tick_ramp(config, now);
        if self.ttft_over_bound(config, now) {
            return Err(self.reject(RejectReason::Ttft));
        }
        if self.every_backend_queued(config, pool, now) {
            return Err(self.reject(RejectReason::BackendQueue));
        }
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
        metrics::gauge!("admission_inflight").set(f64::from(current + 1));
        Ok(Some(Permit {
            controller: Arc::clone(self),
            started: now,
            backend: AtomicUsize::new(NO_BACKEND),
            first_chunk_seen: AtomicBool::new(false),
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

    fn record_ttft(&self, now: Instant, ttft: Duration) {
        metrics::histogram!("admission_ttft_seconds").record(ttft.as_secs_f64());
        let mut window = self.ttft_window();
        window.push_back((now, ttft));
        prune(&mut window, now);
    }

    /// TTFT p95 over the window, once there are enough samples.
    pub fn ttft_p95(&self, now: Instant) -> Option<Duration> {
        let mut window = self.ttft_window();
        prune(&mut window, now);
        if window.len() < TTFT_MIN_SAMPLES {
            return None;
        }
        let mut samples: Vec<Duration> = window.iter().map(|(_, d)| *d).collect();
        samples.sort_unstable();
        // Nearest-rank p95.
        let rank = (samples.len() as f64 * 0.95).ceil() as usize;
        Some(samples[rank.clamp(1, samples.len()) - 1])
    }

    fn ttft_over_bound(&self, config: &AdmissionConfig, now: Instant) -> bool {
        let Some(max) = config.ttft_p95_max else {
            return false;
        };
        let p95 = self.ttft_p95(now);
        let over = p95.is_some_and(|p95| p95 > max);
        if over != self.ttft_tripped.swap(over, Ordering::Relaxed) {
            if over {
                warn!(
                    p95_ms = p95.map_or(0, |d| d.as_millis()),
                    max_ms = max.as_millis(),
                    "Lane time-to-first-chunk p95 above bound, refusing new work"
                );
            } else {
                info!("Lane time-to-first-chunk p95 back under bound");
            }
        }
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
            let stamp = self
                .backpressure
                .get(index)
                .map_or(0, |s| s.load(Ordering::Relaxed));
            if stamp == 0 {
                all_queued = false;
                break;
            }
            let at = self.epoch + Duration::from_millis(stamp - 1);
            if now.saturating_duration_since(at) > config.backpressure_ttl {
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
    first_chunk_seen: AtomicBool,
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

    /// The first upstream response chunk arrived: one TTFT sample. Idempotent.
    pub fn observe_first_chunk(&self) {
        self.observe_first_chunk_at(Instant::now());
    }

    pub(crate) fn observe_first_chunk_at(&self, now: Instant) {
        if self.first_chunk_seen.swap(true, Ordering::Relaxed) {
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
}

impl std::fmt::Debug for Permit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Permit")
            .field("backend", &self.backend())
            .field(
                "first_chunk_seen",
                &self.first_chunk_seen.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        let remaining = self.controller.inflight.fetch_sub(1, Ordering::AcqRel) - 1;
        metrics::gauge!("admission_inflight").set(f64::from(remaining));
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

    #[test]
    fn disabled_controller_admits_everything_without_counting() {
        let c = Arc::new(AdmissionController::disabled());
        let p = pool(1);
        for _ in 0..100 {
            assert!(c.try_admit(&p).unwrap().is_none());
        }
        assert_eq!(c.inflight(), 0);
        assert_eq!(c.host_share(1), None);
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
    fn every_backend_queued_refuses_until_the_signal_ages_out() {
        let c = controller(config(), 2);
        let p = pool(2);
        let t0 = Instant::now();
        // Only backend 0 rejected: the other one may still have room.
        let permit = c.try_admit_at(&p, t0).unwrap().unwrap();
        permit.attach_backend(0);
        permit.observe_backpressure_at(t0);
        drop(permit);
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
        assert!(c.try_admit_at(&p, t0 + Duration::from_secs(20)).is_ok());
    }

    #[test]
    fn ttft_p95_over_bound_refuses_and_recovers_when_samples_expire() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        // Four slow samples are not enough to judge.
        for i in 0..4 {
            let permit = c.try_admit_at(&p, t0).unwrap().unwrap();
            permit.observe_first_chunk_at(t0 + Duration::from_secs(20 + i));
        }
        assert!(c.try_admit_at(&p, t0 + Duration::from_secs(1)).is_ok());
        assert_eq!(c.ttft_p95(t0 + Duration::from_secs(1)), None);
        // The fifth one trips the bound.
        let permit = c.try_admit_at(&p, t0).unwrap().unwrap();
        permit.observe_first_chunk_at(t0 + Duration::from_secs(25));
        drop(permit);
        assert_eq!(
            c.ttft_p95(t0 + Duration::from_secs(1)),
            Some(Duration::from_secs(25))
        );
        let rejected = c.try_admit_at(&p, t0 + Duration::from_secs(1)).unwrap_err();
        assert_eq!(rejected.reason, RejectReason::Ttft);
        // Samples (stamped when their first chunk arrived, t0+20..t0+25) fall
        // out of the window and admission resumes.
        let later = t0 + Duration::from_secs(25) + TTFT_WINDOW + Duration::from_secs(1);
        assert!(c.try_admit_at(&p, later).is_ok());
        assert_eq!(c.ttft_p95(later), None);
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
        for _ in 0..6 {
            let permit = c.try_admit_at(&p, t0).unwrap().unwrap();
            permit.observe_first_chunk_at(t0 + Duration::from_secs(100));
        }
        assert!(c.try_admit_at(&p, t0 + Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn first_chunk_is_recorded_once_per_permit() {
        let c = controller(config(), 1);
        let p = pool(1);
        let t0 = Instant::now();
        let permit = c.try_admit_at(&p, t0).unwrap().unwrap();
        permit.observe_first_chunk_at(t0 + Duration::from_millis(100));
        permit.observe_first_chunk_at(t0 + Duration::from_secs(50));
        drop(permit);
        assert_eq!(c.ttft_window().len(), 1);
        assert_eq!(c.ttft_window()[0].1, Duration::from_millis(100));
    }
}
