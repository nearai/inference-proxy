use crate::admission::{Permit, RejectReason};
use crate::backend_affinity::ConversationKey;
use crate::backend_pool::{BackendGuard, Policy};
use crate::context_tier::TierDecision;
use crate::error::AppError;
use crate::proxy::ConnectFailover;
use crate::AppState;

/// Placement resources shared by chat and text completions for the lifetime
/// of their upstream request.
pub(super) struct PlacedCompletion {
    pub url: String,
    pub backend_guard: BackendGuard,
    pub admission: Option<Permit>,
    pub connect_failover: Option<ConnectFailover>,
}

/// Admit and place one chat or text-completion request.
///
/// The routes retain request adaptation and transport choices. This function
/// owns the common gateway policy that must stay aligned across both routes.
pub(super) fn place_completion(
    state: &AppState,
    path: &'static str,
    tier: Option<TierDecision>,
    affinity_key: Option<ConversationKey>,
) -> Result<PlacedCompletion, AppError> {
    let strict = state.config.backend_tier_strict;
    let admission = state.admission.try_admit(&state.backend_pool, tier)?;
    let limits = state.admission.backend_limits(&state.backend_pool);
    let mut restrict = tier.and_then(|decision| decision.restrict);
    let requested_tier = tier.map(|decision| decision.estimated);
    let place = |tier| {
        let policy = Policy {
            max_conns: None,
            max_conns_by_backend: limits.as_deref(),
            requested_tier,
            avoid: &|index| state.admission.backend_saturated(index),
            engine: &|index| state.admission.engine(index),
            tier,
        };
        state
            .backend_affinity
            .place(&state.backend_pool, affinity_key, path, &policy)
    };

    let mut placement = place(restrict);
    // A tier can empty after the initial decision when its final backend is
    // marked unhealthy. Fall back to the remaining fleet rather than refuse
    // (non-strict only: strict's `recheck_restriction` never returns `None`,
    // so this never widens the search there).
    if placement.is_none()
        && restrict.is_some_and(|tier| {
            crate::context_tier::recheck_restriction(&state.backend_pool, tier, strict).is_none()
        })
    {
        restrict = None;
        placement = place(None);
    }
    let placement = match placement {
        Some(placement) => placement,
        None => {
            // `restrict` can only still be `Some(tier)` here with `tier`
            // truly empty in strict mode (non-strict already cleared it
            // above whenever that was the reason placement failed): refuse
            // deterministically instead of the generic host-share rejection,
            // which is for backends that exist but are full or steered
            // around, not for a tier with none at all.
            //
            // Gated on `strict` explicitly rather than leaning on `restrict`
            // alone: a concurrent request can empty the tier between the
            // widening check above and here even in non-strict mode, and
            // this block must not fire for that race — non-strict behavior
            // must stay exactly what it was before strict mode existed.
            if strict {
                if let Some(tier) = restrict {
                    if state.backend_pool.healthy_count_in(Some(tier)) == 0 {
                        return Err(match admission.as_ref() {
                            Some(permit) => AppError::from(permit.reject_tier_unavailable()),
                            None => {
                                AppError::tier_unavailable(state.config.admission_retry_after_secs)
                            }
                        });
                    }
                }
            }
            return Err(AppError::from(
                state.admission.reject(RejectReason::HostShare),
            ));
        }
    };
    if let Some(permit) = admission.as_ref() {
        permit.attach_backend(placement.index);
    }
    let connect_failover = state
        .config
        .backend_connect_failover
        .then(|| ConnectFailover {
            pool: state.backend_pool.clone(),
            path,
            index: placement.index,
            tier: restrict,
            strict,
            retry_after_secs: state.config.admission_retry_after_secs,
            affinity: affinity_key.map(|key| (state.backend_affinity.clone(), key)),
        });

    Ok(PlacedCompletion {
        url: placement.url,
        backend_guard: placement.guard,
        admission,
        connect_failover,
    })
}
