//! The lane's global in-flight budget and its ramp: start at
//! `start_inflight`, grow by `ramp_step` every `ramp_interval` up to
//! `max_inflight` — only after an interval without an overload signal.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use tracing::info;

use super::AdmissionConfig;

struct Ramp {
    interval_started: Instant,
    /// An overload signal was seen in the current interval.
    dirty: bool,
}

pub(super) struct Budget {
    pub(super) inflight: AtomicU32,
    /// The part of `inflight` that is not bound for the long tier. Capped at
    /// `limit - reserve` when a reserve is configured.
    inflight_base: AtomicU32,
    /// Budget slots kept for the long tier (`long_reserved_inflight`).
    reserve: u32,
    pub(super) limit: AtomicU32,
    ramp: Mutex<Ramp>,
}

impl Budget {
    pub(super) fn new(start: u32, reserve: u32, now: Instant) -> Self {
        Self {
            inflight: AtomicU32::new(0),
            inflight_base: AtomicU32::new(0),
            reserve,
            limit: AtomicU32::new(start),
            ramp: Mutex::new(Ramp {
                interval_started: now,
                dirty: false,
            }),
        }
    }

    pub(super) fn limit(&self) -> u32 {
        self.limit.load(Ordering::Relaxed)
    }

    pub(super) fn inflight(&self) -> u32 {
        self.inflight.load(Ordering::Relaxed)
    }

    /// No slot left at the current budget (the `precheck` peek).
    pub(super) fn is_full(&self) -> bool {
        self.inflight.load(Ordering::Acquire) >= self.limit.load(Ordering::Relaxed)
    }

    /// Take one slot if the budget has room.
    pub(super) fn try_reserve(&self) -> bool {
        let mut current = self.inflight.load(Ordering::Acquire);
        loop {
            if current >= self.limit.load(Ordering::Relaxed) {
                return false;
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
        true
    }

    pub(super) fn release(&self) {
        self.inflight.fetch_sub(1, Ordering::AcqRel);
        metrics::gauge!("admission_inflight").decrement(1.0);
    }

    pub(super) fn inflight_base(&self) -> u32 {
        self.inflight_base.load(Ordering::Relaxed)
    }

    /// Ceiling on in-flight requests that are not bound for the long tier:
    /// the current budget less the long-tier reserve.
    pub(super) fn base_limit(&self) -> u32 {
        self.limit().saturating_sub(self.reserve)
    }

    /// The base allowance has no slot left (the `precheck` peek).
    pub(super) fn base_full(&self) -> bool {
        self.inflight_base.load(Ordering::Acquire) >= self.base_limit()
    }

    /// Take one base slot: always, without a reserve; otherwise only while
    /// fewer than `base_limit` are held.
    pub(super) fn take_base_slot(&self) -> bool {
        if self.reserve == 0 {
            self.inflight_base.fetch_add(1, Ordering::AcqRel);
            return true;
        }
        let mut current = self.inflight_base.load(Ordering::Acquire);
        loop {
            if current >= self.base_limit() {
                return false;
            }
            match self.inflight_base.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    /// Count a base request that holds a global slot and no base slot yet.
    pub(super) fn count_base(&self) {
        self.inflight_base.fetch_add(1, Ordering::AcqRel);
    }

    /// Give back a base slot (no gauge: it was never published).
    pub(super) fn unreserve_base(&self) {
        self.inflight_base.fetch_sub(1, Ordering::AcqRel);
    }

    /// A base slot now backs an admitted request.
    pub(super) fn gauge_base_admitted(&self) {
        metrics::gauge!("admission_inflight_base").increment(1.0);
    }

    /// Release a base slot that `gauge_base_admitted` published.
    pub(super) fn release_base(&self) {
        self.inflight_base.fetch_sub(1, Ordering::AcqRel);
        metrics::gauge!("admission_inflight_base").decrement(1.0);
    }

    /// Grow the budget by one step when a whole interval passed without an
    /// overload signal; a dirty interval just restarts the clock.
    pub(super) fn tick_ramp(&self, config: &AdmissionConfig, now: Instant) {
        let mut ramp = self.ramp();
        if now.saturating_duration_since(ramp.interval_started) < config.ramp_interval {
            return;
        }
        let clean = !ramp.dirty;
        ramp.dirty = false;
        ramp.interval_started = now;
        drop(ramp);

        let budget = self.limit.load(Ordering::Relaxed);
        if budget >= config.max_inflight {
            return;
        }
        if clean {
            let next = budget
                .saturating_add(config.ramp_step)
                .min(config.max_inflight);
            self.limit.store(next, Ordering::Relaxed);
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

    pub(super) fn mark_dirty(&self) {
        self.ramp().dirty = true;
    }

    fn ramp(&self) -> MutexGuard<'_, Ramp> {
        self.ramp.lock().unwrap_or_else(|e| e.into_inner())
    }
}
