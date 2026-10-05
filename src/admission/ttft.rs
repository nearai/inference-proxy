//! Lane time-to-first-generation: a one-minute window of observations and the
//! breaker that refuses new work while too many of them breached the bound.

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tracing::{info, warn};

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

const TTFT_BUCKETS: usize = TTFT_WINDOW.as_secs() as usize;

/// One second of time-to-first-generation observations.
#[derive(Clone, Copy, Default)]
struct Bucket {
    /// Seconds since `epoch` this bucket currently holds (+1; 0 = unused).
    second: u64,
    samples: u32,
    breaches: u32,
}

struct Breaker {
    evaluated_at: Option<Instant>,
    tripped: bool,
}

pub(super) struct TtftBreaker {
    epoch: Instant,
    /// Ring of one-second buckets covering the last `TTFT_WINDOW`.
    ring: Mutex<[Bucket; TTFT_BUCKETS]>,
    breaker: Mutex<Breaker>,
}

impl TtftBreaker {
    pub(super) fn new(epoch: Instant) -> Self {
        Self {
            epoch,
            ring: Mutex::new([Bucket::default(); TTFT_BUCKETS]),
            breaker: Mutex::new(Breaker {
                evaluated_at: None,
                tripped: false,
            }),
        }
    }

    /// One observation (or a censored one). Returns whether it breached
    /// `max`; the caller counts a breach against the ramp interval even if the
    /// breaker is not (yet) tripped. `None` = no TTFT check: the histogram is still
    /// recorded, but nothing enters the window and nothing counts as a breach.
    pub(super) fn record(&self, now: Instant, ttft: Duration, max: Option<Duration>) -> bool {
        metrics::histogram!("admission_ttft_seconds").record(ttft.as_secs_f64());
        let Some(max) = max else {
            return false;
        };
        let breach = ttft > max;
        let stamp = self.stamp(now);
        let mut ring = self.ring();
        let bucket = &mut ring[(stamp % TTFT_BUCKETS as u64) as usize];
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
        breach
    }

    /// `(samples, breaches)` over the window, whatever the count.
    pub(super) fn totals(&self, now: Instant) -> (usize, usize) {
        let newest = self.stamp(now);
        let oldest = newest.saturating_sub(TTFT_BUCKETS as u64 - 1);
        let buckets = self.ring();
        buckets
            .iter()
            .filter(|b| b.second != 0 && b.second >= oldest && b.second <= newest)
            .fold((0, 0), |(s, b), bucket| {
                (s + bucket.samples as usize, b + bucket.breaches as usize)
            })
    }

    /// `(samples, breaches)` in the window, or `None` below the minimum.
    pub(super) fn breaches(&self, now: Instant) -> Option<(usize, usize)> {
        let (samples, breaches) = self.totals(now);
        (samples >= TTFT_MIN_SAMPLES).then_some((samples, breaches))
    }

    /// The (cached for `BREAKER_REEVALUATE_AFTER`) breaker verdict; `false`
    /// when there is no bound. The caller marks the ramp dirty when `true`.
    pub(super) fn over_bound(&self, max: Option<Duration>, now: Instant) -> bool {
        let Some(max) = max else {
            return false;
        };
        let mut breaker = self.breaker.lock().unwrap_or_else(|e| e.into_inner());
        let fresh = breaker
            .evaluated_at
            .is_some_and(|at| now.saturating_duration_since(at) < BREAKER_REEVALUATE_AFTER);
        if !fresh {
            let over = self.breaches(now).is_some_and(|(samples, breaches)| {
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
        over
    }

    /// Seconds since `epoch`, plus one (bucket stamps use 0 for "unused").
    fn stamp(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.epoch).as_secs() + 1
    }

    fn ring(&self) -> MutexGuard<'_, [Bucket; TTFT_BUCKETS]> {
        self.ring.lock().unwrap_or_else(|e| e.into_inner())
    }
}
