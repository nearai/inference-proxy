//! In-flight limit for `/v1/rerank`.
//!
//! Requests are counted while they run, from admission until the upstream
//! response has been read in full, and refused with 429 + `Retry-After`
//! before anything is forwarded once a pool is full.
//!
//! Pools:
//! - every config `TOKEN` shares one trusted pool (`trusted`);
//! - a Cloud API key counts against its organization + workspace, as returned
//!   by cloud-api (`per_workspace`), and against the aggregate of all direct
//!   keys (`direct`). The two partitions are separate, so direct keys cannot
//!   starve trusted traffic or the reverse; the process-wide ceiling is
//!   `trusted + direct`.
//!
//! Per-workspace state exists only while a request holds a permit: an entry is
//! inserted on admission and removed when its last permit is dropped. The map
//! therefore never holds more than `direct` entries, however many distinct
//! workspaces call, and needs no expiry.
//!
//! The limits come from `VLLM_PROXY_RERANK_MAX_INFLIGHT_PER_WORKSPACE`,
//! `_DIRECT` and `_TRUSTED` (`Config::rerank_limits`, defaults below; each
//! at least 1). Sizing: `trusted + direct` is what the rerank engine may be
//! asked to run at once, so keep it within the concurrency the engine serves
//! at an acceptable latency; `per_workspace` bounds what one tenant can take
//! of `direct`.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::auth::{AuthPath, RequireAuth};
use crate::error::AppError;

/// Concurrent rerank requests one organization + workspace may run.
pub const DEFAULT_MAX_IN_FLIGHT_PER_WORKSPACE: usize = 2;
/// Concurrent rerank requests all Cloud API keys together may run.
pub const DEFAULT_MAX_IN_FLIGHT_DIRECT: usize = 16;
/// Concurrent rerank requests all config tokens together may run.
pub const DEFAULT_MAX_IN_FLIGHT_TRUSTED: usize = 16;
/// `Retry-After` on a refusal: rerank requests finish in seconds.
const RETRY_AFTER_SECS: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RerankLimits {
    pub per_workspace: usize,
    pub direct: usize,
    pub trusted: usize,
}

impl Default for RerankLimits {
    fn default() -> Self {
        Self {
            per_workspace: DEFAULT_MAX_IN_FLIGHT_PER_WORKSPACE,
            direct: DEFAULT_MAX_IN_FLIGHT_DIRECT,
            trusted: DEFAULT_MAX_IN_FLIGHT_TRUSTED,
        }
    }
}

/// Whose pool a request counts against.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Tenant {
    /// Any config `TOKEN`.
    Trusted,
    /// A Cloud API key, by the identity cloud-api returned for it.
    Workspace {
        org_id: String,
        workspace_id: String,
    },
}

impl Tenant {
    /// The pool for an authenticated request. Only server-side identity is
    /// used: the auth path, and for a Cloud API key the organization and
    /// workspace from cloud-api's answer. A Cloud API key without both is
    /// refused (authentication already refuses it; this keeps the limiter
    /// from ever sharing one bucket among unidentified keys).
    pub fn of(auth: &RequireAuth) -> Result<Self, AppError> {
        match auth.request_source.auth_path {
            AuthPath::TrustedConfigToken => Ok(Self::Trusted),
            AuthPath::CloudApiKey => {
                let org_id = auth.org_id.as_deref().filter(|id| !id.is_empty());
                let workspace_id = auth.workspace_id.as_deref().filter(|id| !id.is_empty());
                match (org_id, workspace_id) {
                    (Some(org_id), Some(workspace_id)) => Ok(Self::Workspace {
                        org_id: org_id.to_owned(),
                        workspace_id: workspace_id.to_owned(),
                    }),
                    _ => Err(AppError::BillingUnavailable),
                }
            }
        }
    }
}

/// Which limit refused a request (bounded metric label).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    Trusted,
    Direct,
    Workspace,
}

impl Refused {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Trusted => "trusted",
            Self::Direct => "direct",
            Self::Workspace => "workspace",
        }
    }
}

impl From<Refused> for AppError {
    fn from(_: Refused) -> Self {
        AppError::Overloaded {
            reason: "rerank_in_flight",
            retry_after_secs: RETRY_AFTER_SECS,
        }
    }
}

type WorkspaceKey = (String, String);

#[derive(Default)]
struct InFlight {
    trusted: usize,
    direct: usize,
    per_workspace: HashMap<WorkspaceKey, usize>,
}

pub struct RerankLimiter {
    limits: RerankLimits,
    in_flight: Mutex<InFlight>,
}

impl Default for RerankLimiter {
    fn default() -> Self {
        Self::new(RerankLimits::default())
    }
}

/// Held while a rerank request runs; dropping it frees the slot.
pub struct RerankPermit {
    limiter: Arc<RerankLimiter>,
    workspace: Option<WorkspaceKey>,
}

impl RerankLimiter {
    /// Limits below 1 are raised to 1.
    pub fn new(limits: RerankLimits) -> Self {
        Self {
            limits: RerankLimits {
                per_workspace: limits.per_workspace.max(1),
                direct: limits.direct.max(1),
                trusted: limits.trusted.max(1),
            },
            in_flight: Mutex::new(InFlight::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, InFlight> {
        self.in_flight
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Admit an authenticated request or refuse it with 429.
    pub fn admit(self: &Arc<Self>, auth: &RequireAuth) -> Result<RerankPermit, AppError> {
        let tenant = Tenant::of(auth)?;
        self.try_acquire(tenant).map_err(|refused| {
            metrics::counter!(
                "inference_proxy_rerank_refused_total",
                "limit" => refused.as_str()
            )
            .increment(1);
            tracing::info!(
                limit = refused.as_str(),
                "Rerank request refused: in-flight limit reached"
            );
            refused.into()
        })
    }

    /// Take a slot for `tenant`, or say which limit is full. Every limit is
    /// checked before anything is counted, so a refusal changes nothing.
    pub fn try_acquire(self: &Arc<Self>, tenant: Tenant) -> Result<RerankPermit, Refused> {
        let mut in_flight = self.lock();
        let workspace = match tenant {
            Tenant::Trusted => {
                if in_flight.trusted >= self.limits.trusted {
                    return Err(Refused::Trusted);
                }
                in_flight.trusted += 1;
                None
            }
            Tenant::Workspace {
                org_id,
                workspace_id,
            } => {
                if in_flight.direct >= self.limits.direct {
                    return Err(Refused::Direct);
                }
                let key = (org_id, workspace_id);
                let running = in_flight.per_workspace.get(&key).copied().unwrap_or(0);
                if running >= self.limits.per_workspace {
                    return Err(Refused::Workspace);
                }
                in_flight.direct += 1;
                in_flight.per_workspace.insert(key.clone(), running + 1);
                Some(key)
            }
        };
        Ok(RerankPermit {
            limiter: Arc::clone(self),
            workspace,
        })
    }

    /// Requests running now: (trusted, direct, workspaces with one running).
    pub fn in_flight(&self) -> (usize, usize, usize) {
        let in_flight = self.lock();
        (
            in_flight.trusted,
            in_flight.direct,
            in_flight.per_workspace.len(),
        )
    }
}

impl Drop for RerankPermit {
    fn drop(&mut self) {
        let mut in_flight = self.limiter.lock();
        match self.workspace.take() {
            None => in_flight.trusted = in_flight.trusted.saturating_sub(1),
            Some(key) => {
                in_flight.direct = in_flight.direct.saturating_sub(1);
                if let Entry::Occupied(mut running) = in_flight.per_workspace.entry(key) {
                    *running.get_mut() = running.get().saturating_sub(1);
                    if *running.get() == 0 {
                        running.remove();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{IngressRouteKind, RequestSource, CALLER_API_KEY, CALLER_UNLABELED};

    fn limiter(per_workspace: usize, direct: usize, trusted: usize) -> Arc<RerankLimiter> {
        Arc::new(RerankLimiter::new(RerankLimits {
            per_workspace,
            direct,
            trusted,
        }))
    }

    fn workspace(org: &str, ws: &str) -> Tenant {
        Tenant::Workspace {
            org_id: org.to_string(),
            workspace_id: ws.to_string(),
        }
    }

    fn auth(auth_path: AuthPath, org: Option<&str>, workspace: Option<&str>) -> RequireAuth {
        RequireAuth {
            cloud_api_key: matches!(auth_path, AuthPath::CloudApiKey)
                .then(|| "sk-test-synthetic".to_string()),
            org_id: org.map(str::to_string),
            workspace_id: workspace.map(str::to_string),
            api_key_id: Some("key".to_string()),
            request_id: None,
            request_source: RequestSource {
                auth_path,
                ingress_route: IngressRouteKind::Missing,
                caller: match auth_path {
                    AuthPath::CloudApiKey => CALLER_API_KEY,
                    AuthPath::TrustedConfigToken => CALLER_UNLABELED,
                },
            },
        }
    }

    #[test]
    fn defaults_are_nonzero_and_trusted_is_a_lane() {
        let limits = RerankLimits::default();
        assert!(limits.per_workspace >= 1);
        assert!(limits.per_workspace <= limits.direct);
        assert!(limits.trusted >= limits.direct);
    }

    #[test]
    fn tenant_comes_from_server_side_identity_only() {
        assert_eq!(
            Tenant::of(&auth(AuthPath::TrustedConfigToken, None, None)).unwrap(),
            Tenant::Trusted
        );
        // A config token stays in the trusted pool even if ids are present.
        assert_eq!(
            Tenant::of(&auth(AuthPath::TrustedConfigToken, Some("o"), Some("w"))).unwrap(),
            Tenant::Trusted
        );
        assert_eq!(
            Tenant::of(&auth(AuthPath::CloudApiKey, Some("o"), Some("w"))).unwrap(),
            workspace("o", "w")
        );
    }

    #[test]
    fn a_cloud_api_key_without_identity_fails_closed() {
        for (org, ws) in [
            (None, Some("w")),
            (Some("o"), None),
            (None, None),
            (Some(""), Some("w")),
            (Some("o"), Some("")),
        ] {
            assert!(matches!(
                Tenant::of(&auth(AuthPath::CloudApiKey, org, ws)),
                Err(AppError::BillingUnavailable)
            ));
        }
        let limiter = limiter(4, 4, 4);
        assert!(limiter
            .admit(&auth(AuthPath::CloudApiKey, None, None))
            .is_err());
        assert_eq!(limiter.in_flight(), (0, 0, 0));
    }

    #[test]
    fn caps_one_workspace_and_releases_on_drop() {
        let limiter = limiter(2, 10, 10);
        let a = workspace("org-a", "ws-a");
        let first = limiter.try_acquire(a.clone()).unwrap();
        let second = limiter.try_acquire(a.clone()).unwrap();
        assert_eq!(
            limiter.try_acquire(a.clone()).err(),
            Some(Refused::Workspace)
        );
        assert_eq!(limiter.in_flight(), (0, 2, 1));
        drop(first);
        let third = limiter.try_acquire(a.clone()).unwrap();
        drop(second);
        drop(third);
        // The workspace entry is gone once nothing runs for it.
        assert_eq!(limiter.in_flight(), (0, 0, 0));
    }

    #[test]
    fn workspaces_are_isolated_including_within_one_org() {
        let limiter = limiter(1, 10, 10);
        let _a = limiter.try_acquire(workspace("org-a", "ws-1")).unwrap();
        assert_eq!(
            limiter.try_acquire(workspace("org-a", "ws-1")).err(),
            Some(Refused::Workspace)
        );
        let _b = limiter.try_acquire(workspace("org-a", "ws-2")).unwrap();
        let _c = limiter.try_acquire(workspace("org-b", "ws-1")).unwrap();
        // Tuple keys: ("a:b", "c") and ("a", "b:c") are different tenants.
        let _d = limiter.try_acquire(workspace("x:y", "z")).unwrap();
        let _e = limiter.try_acquire(workspace("x", "y:z")).unwrap();
        assert_eq!(limiter.in_flight(), (0, 5, 5));
    }

    #[test]
    fn direct_ceiling_spans_workspaces_and_a_refusal_counts_nothing() {
        let limiter = limiter(4, 3, 10);
        let held: Vec<_> = (0..3)
            .map(|i| {
                limiter
                    .try_acquire(workspace("org", &format!("ws-{i}")))
                    .unwrap()
            })
            .collect();
        assert_eq!(
            limiter.try_acquire(workspace("org", "ws-new")).err(),
            Some(Refused::Direct)
        );
        assert_eq!(limiter.in_flight(), (0, 3, 3));
        drop(held);
        assert_eq!(limiter.in_flight(), (0, 0, 0));
    }

    #[test]
    fn trusted_and_direct_pools_are_separate() {
        let limiter = limiter(1, 1, 2);
        let _direct = limiter.try_acquire(workspace("o", "w")).unwrap();
        let t1 = limiter.try_acquire(Tenant::Trusted).unwrap();
        let _t2 = limiter.try_acquire(Tenant::Trusted).unwrap();
        assert_eq!(
            limiter.try_acquire(Tenant::Trusted).err(),
            Some(Refused::Trusted)
        );
        assert_eq!(
            limiter.try_acquire(workspace("o2", "w2")).err(),
            Some(Refused::Direct)
        );
        drop(t1);
        let _t3 = limiter.try_acquire(Tenant::Trusted).unwrap();
        assert_eq!(limiter.in_flight(), (2, 1, 1));
    }

    #[test]
    fn many_distinct_workspaces_cannot_grow_state_past_the_ceiling() {
        let limiter = limiter(1, 64, 1);
        let mut held = Vec::new();
        for i in 0..10_000 {
            if let Ok(permit) = limiter.try_acquire(workspace(&format!("org-{i}"), "ws")) {
                held.push(permit);
            }
        }
        assert_eq!(held.len(), 64);
        assert_eq!(limiter.in_flight(), (0, 64, 64));
        drop(held);
        assert_eq!(limiter.in_flight(), (0, 0, 0));
    }

    #[test]
    fn zero_limits_are_raised_to_one() {
        let limiter = limiter(0, 0, 0);
        let _t = limiter.try_acquire(Tenant::Trusted).unwrap();
        let _w = limiter.try_acquire(workspace("o", "w")).unwrap();
    }

    #[tokio::test]
    async fn refusal_is_a_429_with_retry_after() {
        let limiter = limiter(1, 1, 1);
        let _held = limiter
            .admit(&auth(AuthPath::TrustedConfigToken, None, None))
            .unwrap();
        let error = match limiter.admit(&auth(AuthPath::TrustedConfigToken, None, None)) {
            Err(error) => error,
            Ok(_) => panic!("second trusted request should be refused"),
        };
        let response = axum::response::IntoResponse::into_response(error);
        assert_eq!(response.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some("1")
        );
    }
}
