//! Per-backend saturation: the engine's own queue depth (when probed) and
//! recent engine admission rejections, plus the fleet-wide "every backend
//! queues" verdict latched per tier.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::{info, warn};

use super::AdmissionConfig;
use crate::backend_pool::BackendPool;
use crate::context_tier::ContextTier;
use crate::engine_load::EngineLoad;

pub(super) struct BackendSaturation {
    epoch: Instant,
    /// Per backend index: last engine admission rejection as milliseconds
    /// since `epoch`, plus one so that zero means "never".
    backpressure: Vec<AtomicU64>,
    /// Live engine view per backend when `VLLM_BACKEND_PROBE_URLS` is set.
    engine: Arc<EngineLoad>,
    /// Latched "every backend queues" state, one per tier: the verdict is
    /// computed over the request's own tier, so a single flag would flap
    /// between a queueing base fleet and an idle long host.
    queue_tripped: [AtomicBool; 2],
}

impl BackendSaturation {
    pub(super) fn new(backend_count: usize, engine: Arc<EngineLoad>, epoch: Instant) -> Self {
        Self {
            epoch,
            backpressure: (0..backend_count).map(|_| AtomicU64::new(0)).collect(),
            engine,
            queue_tripped: [AtomicBool::new(false), AtomicBool::new(false)],
        }
    }

    pub(super) fn engine(&self, index: usize) -> Option<(u32, u32)> {
        self.engine.get(index).map(|s| (s.running, s.queued))
    }

    pub(super) fn saturated_at(
        &self,
        config: Option<&AdmissionConfig>,
        index: usize,
        now: Instant,
    ) -> bool {
        // The engine sample is checked before the config `None` return: a
        // gateway that sets `VLLM_BACKEND_PROBE_URLS` without admission still
        // steers placement around a queueing backend (threshold 1, same as
        // admission's own default).
        let threshold = config.map_or(1, |c| c.queue_saturated_at);
        if self
            .engine
            .get_at(index, now)
            .is_some_and(|s| s.queued >= threshold)
        {
            return true;
        }
        let Some(config) = config else {
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

    /// Stamp an engine admission rejection on `backend`. The caller counts it
    /// against the ramp interval.
    pub(super) fn record_backpressure(&self, backend: usize, now: Instant) {
        if let Some(slot) = self.backpressure.get(backend) {
            let millis = u64::try_from(now.saturating_duration_since(self.epoch).as_millis())
                .unwrap_or(u64::MAX - 1);
            slot.store(millis + 1, Ordering::Relaxed);
        }
        metrics::counter!("admission_backpressure_total", "backend" => backend.to_string())
            .increment(1);
    }

    /// Every healthy backend of the request's tier is saturated: its engine
    /// queue is at or above `queue_saturated_at`, or it rejected a lane
    /// request at engine admission within the TTL.
    pub(super) fn every_backend_queued(
        &self,
        config: &AdmissionConfig,
        pool: &BackendPool,
        tier: Option<ContextTier>,
        now: Instant,
    ) -> bool {
        let mut healthy = 0usize;
        let mut all_queued = true;
        for (index, backend) in pool.backends().iter().enumerate() {
            if !backend.healthy.load(Ordering::Relaxed)
                || tier.is_some_and(|tier| tier != backend.tier)
            {
                continue;
            }
            healthy += 1;
            if !self.saturated_at(Some(config), index, now) {
                all_queued = false;
                break;
            }
        }
        let queued = healthy > 0 && all_queued;
        let latch = &self.queue_tripped[usize::from(tier == Some(ContextTier::Long))];
        if queued != latch.swap(queued, Ordering::Relaxed) {
            let tier = tier.map_or("any", ContextTier::as_str);
            if queued {
                warn!(
                    tier,
                    healthy_backends = healthy,
                    queue_saturated_at = config.queue_saturated_at,
                    ttl_secs = config.backpressure_ttl.as_secs(),
                    "Every backend's engine queue is at or above the saturation threshold, or it rejected a lane request at engine admission recently; refusing new work"
                );
            } else {
                info!(tier, "A backend accepts lane work again");
            }
        }
        queued
    }
}
