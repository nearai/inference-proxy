//! The `sk-` keys cloud-api accepted, kept for 5 minutes so `RequireAuth`
//! (`auth.rs`) does not ask `/v1/check_api_key` about the same key on every
//! request. A gateway sees one or two keys for all of its traffic.

use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use sha2::{Digest, Sha256};
use tokio::time::Instant;

use crate::auth::AuthSubject;
use crate::error::AppError;

/// How long a key cloud-api accepted stays accepted without asking again.
/// The cost: a key revoked or out of credits keeps working here for up to
/// this long (usage is still reported per request), so it is kept short; one
/// check per key every 5 minutes is still nothing next to one per request.
const ACCEPTED_FOR: Duration = Duration::from_secs(300);

/// Keys tracked at once; a flood of random keys can only evict accepted
/// ones, which then cost one more check.
const CAPACITY: u64 = 10_000;

/// Remembers the keys cloud-api accepted (200), for [`ACCEPTED_FOR`]. Nothing
/// else is kept: a refusal, a 402, a 429 or an unreachable cloud-api drops the
/// key's entry, and every request for the key goes to cloud-api until one gets
/// a 200. Nor is a 200 whose identity is incomplete (a body that could not be
/// read or parsed, an older cloud-api): that request gets in, but its usage
/// cannot be reported, so the next one asks again. Keyed by the SHA-256 of
/// the key, never the key itself.
///
/// Requests for a key that needs checking wait for the check in flight
/// rather than each calling cloud-api: a 200 then serves them all. Once a
/// key's last check was not a 200, nobody waits: each request makes its own
/// call, side by side, as if there were no cache.
pub struct ApiKeyCache {
    slots: moka::sync::Cache<[u8; 32], Arc<KeySlot>>,
    accepted_for: Duration,
}

#[derive(Default)]
struct KeySlot {
    /// Numbers the checks in the order they start, so the outcome of the
    /// newest one is what counts, whichever finishes last.
    started: AtomicU64,
    /// The newest check was not a 200 that could be kept, or one was dropped
    /// before it finished: requests stop waiting for each other until it is.
    failing: AtomicBool,
    /// Held by the check the other requests wait for.
    in_flight: tokio::sync::Mutex<()>,
    state: Mutex<KeyState>,
}

#[derive(Default)]
struct KeyState {
    /// The subject cloud-api returned with the newest 200, and when.
    accepted: Option<(AuthSubject, Instant)>,
    /// The number of the newest check whose outcome is recorded.
    recorded: u64,
}

/// Marks the key failing if a check is dropped (its request went away)
/// before it finished, so the requests waiting for it do not each wait in
/// turn.
struct Unfinished<'a> {
    failing: &'a AtomicBool,
    finished: bool,
}

impl Drop for Unfinished<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.failing.store(true, Ordering::Release);
        }
    }
}

impl KeySlot {
    fn accepted(&self, accepted_for: Duration) -> Option<AuthSubject> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state
            .accepted
            .as_ref()
            .filter(|(_, at)| at.elapsed() < accepted_for)
            .map(|(subject, _)| subject.clone())
    }

    async fn check<F, Fut>(&self, check: F) -> Result<AuthSubject, AppError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<AuthSubject, AppError>>,
    {
        let number = self.started.fetch_add(1, Ordering::Relaxed) + 1;
        let mut unfinished = Unfinished {
            failing: &self.failing,
            finished: false,
        };
        let outcome = check().await;
        unfinished.finished = true;

        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if number > state.recorded {
            state.recorded = number;
            state.accepted = outcome
                .as_ref()
                .ok()
                .filter(|subject| has_identity(subject))
                .map(|subject| (subject.clone(), Instant::now()));
            self.failing
                .store(state.accepted.is_none(), Ordering::Release);
        }
        outcome
    }
}

/// Whether a 200's subject carries everything a usage report needs
/// (`proxy::spawn_usage_report` skips the report otherwise).
fn has_identity(subject: &AuthSubject) -> bool {
    subject.org_id.is_some() && subject.workspace_id.is_some() && subject.api_key_id.is_some()
}

impl Default for ApiKeyCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiKeyCache {
    pub fn new() -> Self {
        Self::accepting_for(ACCEPTED_FOR)
    }

    fn accepting_for(accepted_for: Duration) -> Self {
        Self {
            slots: moka::sync::Cache::builder()
                .max_capacity(CAPACITY)
                .time_to_idle(accepted_for)
                .build(),
            accepted_for,
        }
    }

    /// The subject for `token`: from the cache while cloud-api's 200 is
    /// fresh, otherwise from `check` (which asks cloud-api).
    pub(crate) async fn subject<F, Fut>(
        &self,
        token: &str,
        check: F,
    ) -> Result<AuthSubject, AppError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<AuthSubject, AppError>>,
    {
        let key: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        let slot = self.slots.get_with(key, Default::default);

        if let Some(subject) = slot.accepted(self.accepted_for) {
            metrics::counter!("cloud_api_auth_cache_total", "outcome" => "hit").increment(1);
            return Ok(subject);
        }
        if !slot.failing.load(Ordering::Acquire) {
            let _turn = slot.in_flight.lock().await;
            if let Some(subject) = slot.accepted(self.accepted_for) {
                metrics::counter!("cloud_api_auth_cache_total", "outcome" => "shared").increment(1);
                return Ok(subject);
            }
            if !slot.failing.load(Ordering::Acquire) {
                metrics::counter!("cloud_api_auth_cache_total", "outcome" => "miss").increment(1);
                return slot.check(check).await;
            }
        }
        // The key's last check was not a 200 (perhaps the one this request
        // just waited for): ask cloud-api, alongside everyone else.
        metrics::counter!("cloud_api_auth_cache_total", "outcome" => "direct").increment(1);
        slot.check(check).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn subject(org: &str) -> AuthSubject {
        AuthSubject {
            org_id: Some(org.to_string()),
            workspace_id: Some(format!("{org}-workspace")),
            api_key_id: Some(format!("{org}-key")),
        }
    }

    /// What a fake cloud-api answers: a 200 for an org, or a refusal.
    #[derive(Clone, Copy, Debug)]
    enum Answer {
        Accepted(&'static str),
        /// A 200 whose body could not be read: no identity.
        AcceptedWithoutIdentity,
        Unauthorized,
        InsufficientCredits,
        RateLimited,
    }

    /// A check that counts its calls and gives `answer` after `delay`.
    async fn counted(
        calls: &AtomicUsize,
        delay: Duration,
        answer: Answer,
    ) -> Result<AuthSubject, AppError> {
        calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(delay).await;
        match answer {
            Answer::Accepted(org) => Ok(subject(org)),
            Answer::AcceptedWithoutIdentity => Ok(AuthSubject::default()),
            Answer::Unauthorized => Err(AppError::Unauthorized),
            Answer::InsufficientCredits => Err(AppError::InsufficientCredits),
            Answer::RateLimited => Err(AppError::RateLimited),
        }
    }

    fn org(outcome: Result<AuthSubject, AppError>) -> Option<String> {
        outcome
            .unwrap_or_else(|error| panic!("expected an accepted key, got {error:?}"))
            .org_id
    }

    async fn ask(
        cache: &ApiKeyCache,
        calls: &AtomicUsize,
        delay: Duration,
        answer: Answer,
    ) -> Result<AuthSubject, AppError> {
        cache
            .subject("sk-a", || counted(calls, delay, answer))
            .await
    }

    /// `n` concurrent requests for one key, each answered with `answer`
    /// after `delay`; returns their outcomes.
    async fn concurrently(
        cache: &Arc<ApiKeyCache>,
        calls: &Arc<AtomicUsize>,
        n: usize,
        delay: Duration,
        answer: Answer,
    ) -> Vec<Result<AuthSubject, AppError>> {
        let tasks: Vec<_> = (0..n)
            .map(|_| {
                let (cache, calls) = (cache.clone(), calls.clone());
                tokio::spawn(async move { ask(&cache, &calls, delay, answer).await })
            })
            .collect();
        let mut outcomes = Vec::new();
        for task in tasks {
            outcomes.push(task.await.unwrap());
        }
        outcomes
    }

    #[test]
    fn the_cache_accepts_for_five_minutes() {
        assert_eq!(ACCEPTED_FOR, Duration::from_secs(300));
        assert_eq!(ApiKeyCache::new().accepted_for, ACCEPTED_FOR);
    }

    #[tokio::test]
    async fn keeps_an_accepted_key_until_it_expires() {
        let cache = ApiKeyCache::accepting_for(Duration::from_millis(200));
        let calls = AtomicUsize::new(0);
        for _ in 0..3 {
            let got = ask(&cache, &calls, Duration::ZERO, Answer::Accepted("org-a")).await;
            assert_eq!(org(got).as_deref(), Some("org-a"));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        tokio::time::sleep(Duration::from_millis(250)).await;
        ask(&cache, &calls, Duration::ZERO, Answer::Accepted("org-a"))
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// Every request for a key cloud-api did not accept asks cloud-api again,
    /// until one gets a 200, which is then kept.
    #[tokio::test]
    async fn asks_again_until_a_200() {
        let cache = ApiKeyCache::new();
        let calls = AtomicUsize::new(0);
        for answer in [
            Answer::Unauthorized,
            Answer::InsufficientCredits,
            Answer::RateLimited,
        ] {
            for _ in 0..2 {
                assert!(ask(&cache, &calls, Duration::ZERO, answer).await.is_err());
            }
        }
        assert_eq!(calls.load(Ordering::SeqCst), 6);
        for _ in 0..2 {
            let got = ask(&cache, &calls, Duration::ZERO, Answer::Accepted("org-a")).await;
            assert_eq!(org(got).as_deref(), Some("org-a"));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 7);
    }

    #[tokio::test]
    async fn drops_an_accepted_key_cloud_api_later_refuses() {
        let cache = ApiKeyCache::accepting_for(Duration::from_millis(100));
        let calls = AtomicUsize::new(0);
        ask(&cache, &calls, Duration::ZERO, Answer::Accepted("org-a"))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;
        let got = ask(&cache, &calls, Duration::ZERO, Answer::InsufficientCredits).await;
        assert!(matches!(got, Err(AppError::InsufficientCredits)));
        // The refusal replaced the acceptance: the next request asks again.
        let got = ask(&cache, &calls, Duration::ZERO, Answer::Unauthorized).await;
        assert!(matches!(got, Err(AppError::Unauthorized)));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn keeps_keys_apart() {
        let cache = ApiKeyCache::new();
        let calls = AtomicUsize::new(0);
        ask(&cache, &calls, Duration::ZERO, Answer::Accepted("org-a"))
            .await
            .unwrap();
        let b = cache
            .subject("sk-b", || {
                counted(&calls, Duration::ZERO, Answer::Unauthorized)
            })
            .await;
        assert!(matches!(b, Err(AppError::Unauthorized)));
        let a = ask(&cache, &calls, Duration::ZERO, Answer::Unauthorized).await;
        assert_eq!(org(a).as_deref(), Some("org-a"));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// Concurrent requests for one key wait for the check in flight: a 200
    /// serves them all with one call.
    #[tokio::test]
    async fn runs_one_check_for_concurrent_requests_when_it_is_accepted() {
        let cache = Arc::new(ApiKeyCache::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let outcomes = concurrently(
            &cache,
            &calls,
            8,
            Duration::from_millis(50),
            Answer::Accepted("org-a"),
        )
        .await;
        for outcome in outcomes {
            assert_eq!(org(outcome).as_deref(), Some("org-a"));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// When the check they waited for is not a 200, each waiting request
    /// makes its own call, and they make them side by side, not in turn.
    #[tokio::test]
    async fn lets_every_request_through_when_the_check_fails() {
        let cache = Arc::new(ApiKeyCache::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let started = std::time::Instant::now();
        let outcomes = concurrently(
            &cache,
            &calls,
            8,
            Duration::from_millis(100),
            Answer::Unauthorized,
        )
        .await;
        for outcome in outcomes {
            assert!(matches!(outcome, Err(AppError::Unauthorized)));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 8);
        // One check, then the other seven at once: ~200 ms, not ~800 ms.
        assert!(
            started.elapsed() < Duration::from_millis(600),
            "{:?}",
            started.elapsed()
        );

        // Still failing: nobody waits for anybody.
        let started = std::time::Instant::now();
        concurrently(
            &cache,
            &calls,
            8,
            Duration::from_millis(100),
            Answer::Unauthorized,
        )
        .await;
        assert_eq!(calls.load(Ordering::SeqCst), 16);
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "{:?}",
            started.elapsed()
        );
    }

    /// After a refusal the key is failing, so the next request does not wait
    /// for anyone: it calls cloud-api directly. A 200 it gets that way is
    /// kept for everyone, like one from a request that waited.
    #[tokio::test]
    async fn keeps_a_200_from_a_request_that_did_not_wait() {
        let cache = ApiKeyCache::new();
        let calls = AtomicUsize::new(0);
        assert!(ask(&cache, &calls, Duration::ZERO, Answer::Unauthorized)
            .await
            .is_err());
        let got = ask(&cache, &calls, Duration::ZERO, Answer::Accepted("org-a")).await;
        assert_eq!(org(got).as_deref(), Some("org-a"));
        let got = ask(&cache, &calls, Duration::ZERO, Answer::Unauthorized).await;
        assert_eq!(org(got).as_deref(), Some("org-a"));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// A 200 without the identity a usage report needs (its body could not
    /// be read) lets that request in but is not kept: the next request asks
    /// again, and a complete 200 is then kept.
    #[tokio::test]
    async fn does_not_keep_a_200_without_identity() {
        let cache = ApiKeyCache::new();
        let calls = AtomicUsize::new(0);
        for _ in 0..2 {
            let got = ask(
                &cache,
                &calls,
                Duration::ZERO,
                Answer::AcceptedWithoutIdentity,
            )
            .await;
            assert_eq!(org(got), None);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        ask(&cache, &calls, Duration::ZERO, Answer::Accepted("org-a"))
            .await
            .unwrap();
        let got = ask(&cache, &calls, Duration::ZERO, Answer::Unauthorized).await;
        assert_eq!(org(got).as_deref(), Some("org-a"));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    /// A key in constant use (the gateway's case) still expires on time: its
    /// slot never goes idle, so this is the acceptance's own deadline.
    #[tokio::test]
    async fn an_active_key_is_rechecked_at_its_deadline() {
        let cache = ApiKeyCache::accepting_for(Duration::from_millis(200));
        let calls = AtomicUsize::new(0);
        ask(&cache, &calls, Duration::ZERO, Answer::Accepted("org-a"))
            .await
            .unwrap();
        let started = std::time::Instant::now();
        while started.elapsed() < Duration::from_millis(150) {
            ask(&cache, &calls, Duration::ZERO, Answer::Unauthorized)
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        tokio::time::sleep(Duration::from_millis(80)).await;
        let got = ask(&cache, &calls, Duration::ZERO, Answer::InsufficientCredits).await;
        assert!(matches!(got, Err(AppError::InsufficientCredits)));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// A refusal from a newer check clears a 200 an older check stored a
    /// moment earlier, while that 200 is still fresh.
    #[tokio::test]
    async fn a_newer_refusal_clears_a_fresh_200() {
        let cache = Arc::new(ApiKeyCache::new());
        let calls = Arc::new(AtomicUsize::new(0));
        assert!(ask(&cache, &calls, Duration::ZERO, Answer::Unauthorized)
            .await
            .is_err());

        let spawn = |delay, answer| {
            let (cache, calls) = (cache.clone(), calls.clone());
            tokio::spawn(async move { ask(&cache, &calls, delay, answer).await })
        };
        // The older check finishes first, with a 200; the newer one after it,
        // with a 402.
        let older = spawn(Duration::from_millis(50), Answer::Accepted("org-a"));
        tokio::time::sleep(Duration::from_millis(10)).await;
        let newer = spawn(Duration::from_millis(100), Answer::InsufficientCredits);
        assert!(older.await.unwrap().is_ok());
        assert!(matches!(
            newer.await.unwrap(),
            Err(AppError::InsufficientCredits)
        ));

        let got = ask(&cache, &calls, Duration::ZERO, Answer::RateLimited).await;
        assert!(matches!(got, Err(AppError::RateLimited)));
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }

    /// The newest check decides, whichever finishes last: a slow 200 that
    /// started before a refusal does not bring the key back.
    #[tokio::test]
    async fn the_newest_check_decides() {
        let cache = Arc::new(ApiKeyCache::new());
        let calls = Arc::new(AtomicUsize::new(0));
        assert!(ask(&cache, &calls, Duration::ZERO, Answer::Unauthorized)
            .await
            .is_err());

        let slow_200 = {
            let (cache, calls) = (cache.clone(), calls.clone());
            tokio::spawn(async move {
                ask(
                    &cache,
                    &calls,
                    Duration::from_millis(100),
                    Answer::Accepted("org-a"),
                )
                .await
            })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        let refusal = ask(&cache, &calls, Duration::ZERO, Answer::Unauthorized).await;
        assert!(refusal.is_err());
        assert!(slow_200.await.unwrap().is_ok());

        // The refusal started last, so it is what counts: ask again.
        let got = ask(&cache, &calls, Duration::ZERO, Answer::RateLimited).await;
        assert!(matches!(got, Err(AppError::RateLimited)));
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }

    /// A check dropped before it finished (its request went away) does not
    /// leave the requests that waited for it checking one at a time.
    #[tokio::test]
    async fn a_dropped_check_lets_the_waiters_through_together() {
        let cache = Arc::new(ApiKeyCache::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let stuck = {
            let (cache, calls) = (cache.clone(), calls.clone());
            tokio::spawn(async move {
                ask(
                    &cache,
                    &calls,
                    Duration::from_secs(60),
                    Answer::Accepted("org-a"),
                )
                .await
            })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        let waiters = {
            let (cache, calls) = (cache.clone(), calls.clone());
            tokio::spawn(async move {
                concurrently(
                    &cache,
                    &calls,
                    4,
                    Duration::from_millis(100),
                    Answer::Unauthorized,
                )
                .await
            })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        let started = std::time::Instant::now();
        stuck.abort();
        for outcome in waiters.await.unwrap() {
            assert!(outcome.is_err());
        }
        assert_eq!(calls.load(Ordering::SeqCst), 5);
        assert!(
            started.elapsed() < Duration::from_millis(350),
            "{:?}",
            started.elapsed()
        );
    }
}
