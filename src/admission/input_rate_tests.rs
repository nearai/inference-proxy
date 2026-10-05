//! Facade tests for the input-token rate limiter wired into `AdmissionController`.
use super::tests::{config, pool};
use super::*;

fn one_per_minute_burst_one() -> InputRateTable {
    InputRateTable::new(vec![InputRateBucket {
        below_tokens: 2_000,
        per_minute: 1,
        burst: 1,
    }])
    .unwrap()
}

#[test]
fn precheck_takes_an_input_rate_token_even_without_a_budget() {
    let c = AdmissionController::disabled().with_input_rate(Some(one_per_minute_burst_one()));
    let p = pool(1);
    let t0 = Instant::now();
    assert!(c.input_rate_enabled());
    assert_eq!(c.precheck_at(&p, None, Some(100), t0), Ok(()));
    let refused = c.precheck_at(&p, None, Some(100), t0).unwrap_err();
    assert_eq!(refused.reason, RejectReason::InputTokens);
    assert_eq!(refused.reason.as_str(), "input_tokens");
    assert_eq!(refused.retry_after, Duration::from_secs(60));
    // No estimate, or one past the table: nothing to limit.
    assert_eq!(c.precheck_at(&p, None, None, t0), Ok(()));
    assert_eq!(c.precheck_at(&p, None, Some(2_000), t0), Ok(()));
}

#[test]
fn try_admit_never_takes_an_input_rate_token() {
    let c = Arc::new(
        AdmissionController::new(Some(config()), 1, Arc::new(EngineLoad::disabled()))
            .with_input_rate(Some(one_per_minute_burst_one())),
    );
    let p = pool(1);
    let t0 = Instant::now();
    assert_eq!(c.precheck_at(&p, None, Some(100), t0), Ok(()));
    // try_admit re-runs the signal checks only; the bucket is untouched.
    drop(c.try_admit_at(&p, None, t0).unwrap());
    drop(c.try_admit_at(&p, None, t0).unwrap());
    assert_eq!(
        c.precheck_at(&p, None, Some(100), t0).unwrap_err().reason,
        RejectReason::InputTokens
    );
}

#[test]
fn an_overload_refusal_does_not_take_an_input_rate_token() {
    let c = Arc::new(
        AdmissionController::new(Some(config()), 1, Arc::new(EngineLoad::disabled()))
            .with_input_rate(Some(one_per_minute_burst_one())),
    );
    let p = pool(1);
    let t0 = Instant::now();
    // Fill the budget (config(): start_inflight 2).
    let first = c.try_admit_at(&p, None, t0).unwrap();
    let _second = c.try_admit_at(&p, None, t0).unwrap();
    assert_eq!(
        c.precheck_at(&p, None, Some(100), t0).unwrap_err().reason,
        RejectReason::Budget
    );
    drop(first);
    // The budget refusal left the token in the bucket.
    assert_eq!(c.precheck_at(&p, None, Some(100), t0), Ok(()));
}
