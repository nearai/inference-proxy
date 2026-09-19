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
