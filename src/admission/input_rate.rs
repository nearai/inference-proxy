//! Input-token rate (`VLLM_PROXY_ADMISSION_INPUT_RATE`): requests per minute
//! capped by estimated input size, one token bucket per row of a table. A
//! token is taken once per request, in `AdmissionController::precheck_at`.

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// One row: requests whose estimated input (`context_tier::Estimate::tokens`)
/// is below `below_tokens` and at or above the previous row's bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputRateBucket {
    pub below_tokens: u64,
    /// Sustained requests per minute per gateway instance; 0 = unlimited.
    pub per_minute: u32,
    /// Requests a full bucket lets through at once.
    pub burst: u32,
}

/// Rows with strictly increasing bounds. An estimate at or above the last
/// bound matches no row and is never limited.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputRateTable {
    buckets: Vec<InputRateBucket>,
}

impl InputRateTable {
    pub fn new(buckets: Vec<InputRateBucket>) -> anyhow::Result<Self> {
        anyhow::ensure!(!buckets.is_empty(), "input rate table has no rows");
        let mut previous = 0;
        for bucket in &buckets {
            anyhow::ensure!(
                bucket.below_tokens > previous,
                "input rate table bounds must be positive and strictly increasing"
            );
            anyhow::ensure!(
                bucket.per_minute == 0 || bucket.burst > 0,
                "a limited input rate row needs a burst of at least one"
            );
            previous = bucket.below_tokens;
        }
        Ok(Self { buckets })
    }

    /// The hardcoded V0 table. Sized on the 2026-10-05 flood of
    /// 700–1,500-token requests: 100/min per instance across two gateway
    /// instances is about the pre-flood p99 for prompts under 2,000 tokens.
    /// V1 replaces it at runtime from AppConfig.
    pub fn v0() -> Self {
        Self {
            buckets: vec![InputRateBucket {
                below_tokens: 2_000,
                per_minute: 100,
                burst: 20,
            }],
        }
    }

    pub fn buckets(&self) -> &[InputRateBucket] {
        &self.buckets
    }

    fn row(&self, estimated_tokens: u64) -> Option<usize> {
        self.buckets
            .iter()
            .position(|bucket| estimated_tokens < bucket.below_tokens)
    }
}

/// A refusal: the row that was empty and the wait until it holds a token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct RateLimited {
    pub(super) bucket: InputRateBucket,
    pub(super) retry_after: Duration,
}

/// Classic token bucket: `burst` capacity, refilled continuously at
/// `per_minute / 60` tokens per second.
struct TokenBucket {
    tokens: f64,
    refilled_at: Instant,
    per_second: f64,
    capacity: f64,
}

impl TokenBucket {
    fn full(bucket: InputRateBucket, now: Instant) -> Self {
        Self {
            tokens: f64::from(bucket.burst),
            refilled_at: now,
            per_second: f64::from(bucket.per_minute) / 60.0,
            capacity: f64::from(bucket.burst),
        }
    }

    /// Take one token, or return the wait until one is available. A `now`
    /// earlier than the last refill (a caller that read the clock before a
    /// concurrent one) refills nothing.
    fn try_take_at(&mut self, now: Instant) -> Result<(), Duration> {
        let elapsed = now
            .saturating_duration_since(self.refilled_at)
            .as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.per_second).min(self.capacity);
        self.refilled_at = self.refilled_at.max(now);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            return Ok(());
        }
        Err(Duration::from_secs_f64(
            (1.0 - self.tokens) / self.per_second,
        ))
    }
}

struct Rows {
    table: InputRateTable,
    /// One per row; `None` for an unlimited row.
    buckets: Vec<Option<TokenBucket>>,
}

impl Rows {
    fn new(table: InputRateTable, now: Instant) -> Self {
        let buckets = table
            .buckets
            .iter()
            .map(|&bucket| (bucket.per_minute > 0).then(|| TokenBucket::full(bucket, now)))
            .collect();
        Self { table, buckets }
    }
}

/// The table and its buckets, swappable at runtime (V1: AppConfig).
pub(super) struct InputRateLimiter {
    rows: Mutex<Option<Rows>>,
}

impl InputRateLimiter {
    pub(super) fn new(table: Option<InputRateTable>, now: Instant) -> Self {
        Self {
            rows: Mutex::new(table.map(|table| Rows::new(table, now))),
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.lock_rows().is_some()
    }

    /// Swap the table; every row of a changed table starts with a full
    /// bucket. An identical table is a no-op: buckets keep their level, so a
    /// periodic re-push cannot hand out a fresh burst.
    pub(super) fn replace(&self, table: Option<InputRateTable>, now: Instant) {
        let mut guard = self.lock_rows();
        if guard.as_ref().map(|rows| &rows.table) == table.as_ref() {
            return;
        }
        *guard = table.map(|table| Rows::new(table, now));
    }

    pub(super) fn take_at(&self, estimated_tokens: u64, now: Instant) -> Result<(), RateLimited> {
        let mut guard = self.lock_rows();
        let Some(rows) = guard.as_mut() else {
            return Ok(());
        };
        let Some(row) = rows.table.row(estimated_tokens) else {
            return Ok(());
        };
        let Some(token_bucket) = rows.buckets[row].as_mut() else {
            return Ok(());
        };
        token_bucket.try_take_at(now).map_err(|wait| RateLimited {
            bucket: rows.table.buckets[row],
            retry_after: Duration::from_secs(wait.as_secs_f64().ceil().max(1.0) as u64),
        })
    }

    fn lock_rows(&self) -> MutexGuard<'_, Option<Rows>> {
        self.rows.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(below_tokens: u64, per_minute: u32, burst: u32) -> InputRateBucket {
        InputRateBucket {
            below_tokens,
            per_minute,
            burst,
        }
    }

    fn limiter(rows: &[InputRateBucket], now: Instant) -> InputRateLimiter {
        InputRateLimiter::new(Some(InputRateTable::new(rows.to_vec()).unwrap()), now)
    }

    #[test]
    fn token_bucket_allows_the_burst_then_waits_for_the_next_token() {
        let t0 = Instant::now();
        let mut bucket = TokenBucket::full(row(2_000, 60, 3), t0);
        for _ in 0..3 {
            assert_eq!(bucket.try_take_at(t0), Ok(()));
        }
        assert_eq!(bucket.try_take_at(t0), Err(Duration::from_secs(1)));
    }

    #[test]
    fn token_bucket_refills_at_the_rate_and_caps_at_the_burst() {
        let t0 = Instant::now();
        let mut bucket = TokenBucket::full(row(2_000, 60, 1), t0);
        assert_eq!(bucket.try_take_at(t0), Ok(()));
        assert!(bucket.try_take_at(t0 + Duration::from_millis(500)).is_err());
        assert_eq!(bucket.try_take_at(t0 + Duration::from_secs(1)), Ok(()));
        let later = t0 + Duration::from_secs(3_600);
        assert_eq!(bucket.try_take_at(later), Ok(()));
        assert!(
            bucket.try_take_at(later).is_err(),
            "idling never exceeds the burst"
        );
    }

    #[test]
    fn token_bucket_tolerates_an_earlier_now() {
        let t0 = Instant::now() + Duration::from_secs(10);
        let mut bucket = TokenBucket::full(row(2_000, 60, 1), t0);
        assert_eq!(bucket.try_take_at(t0), Ok(()));
        assert!(bucket.try_take_at(t0 - Duration::from_secs(5)).is_err());
        assert_eq!(bucket.try_take_at(t0 + Duration::from_secs(1)), Ok(()));
    }

    #[test]
    fn retry_after_is_the_wait_rounded_up_to_whole_seconds() {
        // 6/min = one token per 10 s.
        let t0 = Instant::now();
        let limiter = limiter(&[row(2_000, 6, 1)], t0);
        assert_eq!(limiter.take_at(100, t0), Ok(()));
        let refused = limiter
            .take_at(100, t0 + Duration::from_millis(2_500))
            .unwrap_err();
        assert_eq!(refused.retry_after, Duration::from_secs(8)); // ceil(7.5)
        assert_eq!(refused.bucket, row(2_000, 6, 1));
    }

    #[test]
    fn rows_are_independent_and_estimates_past_the_last_bound_are_unlimited() {
        let t0 = Instant::now();
        let limiter = limiter(&[row(500, 60, 1), row(2_000, 60, 1)], t0);
        assert_eq!(limiter.take_at(499, t0), Ok(()));
        assert!(limiter.take_at(0, t0).is_err());
        assert_eq!(limiter.take_at(500, t0), Ok(()));
        assert!(limiter.take_at(1_999, t0).is_err());
        for _ in 0..100 {
            assert_eq!(limiter.take_at(2_000, t0), Ok(()));
        }
    }

    #[test]
    fn a_zero_per_minute_row_is_unlimited() {
        let t0 = Instant::now();
        let limiter = limiter(&[row(500, 0, 0), row(2_000, 60, 1)], t0);
        for _ in 0..100 {
            assert_eq!(limiter.take_at(10, t0), Ok(()));
        }
    }

    #[test]
    fn replacing_the_table_starts_full_and_none_disables() {
        let t0 = Instant::now();
        let limiter = limiter(&[row(2_000, 60, 1)], t0);
        assert!(limiter.enabled());
        assert_eq!(limiter.take_at(10, t0), Ok(()));
        assert!(limiter.take_at(10, t0).is_err());
        limiter.replace(
            Some(InputRateTable::new(vec![row(2_000, 60, 2)]).unwrap()),
            t0,
        );
        assert_eq!(limiter.take_at(10, t0), Ok(()));
        assert_eq!(limiter.take_at(10, t0), Ok(()));
        assert!(limiter.take_at(10, t0).is_err());
        limiter.replace(None, t0);
        assert!(!limiter.enabled());
        assert_eq!(limiter.take_at(10, t0), Ok(()));
    }

    #[test]
    fn replacing_with_an_identical_table_keeps_the_buckets() {
        let t0 = Instant::now();
        let limiter = limiter(&[row(2_000, 60, 1)], t0);
        assert_eq!(limiter.take_at(10, t0), Ok(()));
        assert!(limiter.take_at(10, t0).is_err());
        limiter.replace(
            Some(InputRateTable::new(vec![row(2_000, 60, 1)]).unwrap()),
            t0,
        );
        assert!(limiter.take_at(10, t0).is_err());
        limiter.replace(
            Some(InputRateTable::new(vec![row(2_000, 60, 2)]).unwrap()),
            t0,
        );
        assert_eq!(limiter.take_at(10, t0), Ok(()));
    }

    #[test]
    fn no_table_never_limits() {
        let t0 = Instant::now();
        let limiter = InputRateLimiter::new(None, t0);
        assert!(!limiter.enabled());
        for _ in 0..1_000 {
            assert_eq!(limiter.take_at(10, t0), Ok(()));
        }
    }

    #[test]
    fn table_validation() {
        assert!(InputRateTable::new(vec![]).is_err(), "empty");
        assert!(
            InputRateTable::new(vec![row(2_000, 60, 1), row(500, 60, 1)]).is_err(),
            "decreasing"
        );
        assert!(
            InputRateTable::new(vec![row(500, 60, 1), row(500, 60, 1)]).is_err(),
            "duplicate"
        );
        assert!(
            InputRateTable::new(vec![row(0, 60, 1)]).is_err(),
            "zero bound"
        );
        assert!(
            InputRateTable::new(vec![row(2_000, 60, 0)]).is_err(),
            "limited row without burst"
        );
        assert!(
            InputRateTable::new(vec![row(2_000, 0, 0)]).is_ok(),
            "unlimited row"
        );
    }

    #[test]
    fn v0_table_is_the_reviewed_one() {
        assert_eq!(InputRateTable::v0().buckets(), &[row(2_000, 100, 20)]);
    }

    #[test]
    fn v0_table_satisfies_the_validator() {
        assert!(InputRateTable::new(InputRateTable::v0().buckets().to_vec()).is_ok());
    }
}
