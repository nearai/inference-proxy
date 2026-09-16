//! Live engine load per backend (`VLLM_BACKEND_PROBE_URLS`, gateway mode).
//!
//! The gateway's own connection counts say nothing about the direct traffic
//! the hosts also serve. Each backend's engine metrics (`/v1/metrics`, the
//! same route model-proxy samples) are polled on a short interval for the
//! running and queued request counts, which then drive placement (least
//! loaded host for a new conversation, a queueing host is steered around)
//! and admission (every host queueing = refuse). A sample older than three
//! intervals counts as unknown, so a probe outage degrades to the gateway's
//! own view instead of blocking the lane.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::{debug, warn};

/// One reading of a backend's engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    pub running: u32,
    pub queued: u32,
}

pub struct EngineLoad {
    samples: Vec<Mutex<Option<(Instant, Sample)>>>,
    stale_after: Duration,
}

impl EngineLoad {
    pub fn new(backends: usize, stale_after: Duration) -> Self {
        Self {
            samples: (0..backends).map(|_| Mutex::new(None)).collect(),
            stale_after,
        }
    }

    /// No probes configured: every backend reads as unknown.
    pub fn disabled() -> Self {
        Self::new(0, Duration::ZERO)
    }

    /// The latest fresh sample for backend `index`, if any.
    pub fn get(&self, index: usize) -> Option<Sample> {
        self.get_at(index, Instant::now())
    }

    pub(crate) fn get_at(&self, index: usize, now: Instant) -> Option<Sample> {
        let slot = self
            .samples
            .get(index)?
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        slot.filter(|(at, _)| now.saturating_duration_since(*at) <= self.stale_after)
            .map(|(_, sample)| sample)
    }

    pub(crate) fn record_at(&self, index: usize, sample: Sample, now: Instant) {
        if let Some(slot) = self.samples.get(index) {
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some((now, sample));
        }
        metrics::gauge!("backend_engine_running", "backend" => index.to_string())
            .set(f64::from(sample.running));
        metrics::gauge!("backend_engine_queued", "backend" => index.to_string())
            .set(f64::from(sample.queued));
    }
}

/// Sum of a Prometheus gauge over all its label sets, under any of `names`
/// (SGLang and vLLM spell them differently).
pub fn metric_sum(body: &str, names: &[&str]) -> Option<u32> {
    let mut total = 0.0f64;
    let mut found = false;
    for line in body.lines() {
        let Some(name) = names.iter().copied().find(|name| {
            line.starts_with(name)
                && line[name.len()..]
                    .chars()
                    .next()
                    .is_some_and(|c| c == '{' || c == ' ')
        }) else {
            continue;
        };
        let value = line[name.len()..]
            .rsplit(' ')
            .next()
            .and_then(|v| v.parse::<f64>().ok());
        if let Some(value) = value {
            total += value;
            found = true;
        }
    }
    found.then(|| total.max(0.0).round() as u32)
}

pub fn parse_sample(body: &str) -> Option<Sample> {
    Some(Sample {
        running: metric_sum(
            body,
            &["sglang:num_running_reqs", "vllm:num_requests_running"],
        )?,
        queued: metric_sum(
            body,
            &["sglang:num_queue_reqs", "vllm:num_requests_waiting"],
        )
        .unwrap_or(0),
    })
}

/// Poll every probe URL's `/v1/metrics` on `interval`; `probe_urls[i]` is
/// backend `i`. Failures leave the previous sample to age out.
pub fn spawn_engine_load_poller(
    load: Arc<EngineLoad>,
    client: reqwest::Client,
    probe_urls: Vec<String>,
    interval: Duration,
) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        loop {
            tick.tick().await;
            for (index, probe) in probe_urls.iter().enumerate() {
                let url = format!("{}/v1/metrics", probe.trim_end_matches('/'));
                let body = match client.get(&url).timeout(interval).send().await {
                    Ok(response) if response.status().is_success() => response.text().await.ok(),
                    Ok(response) => {
                        debug!(backend = index, status = %response.status(), "Engine metrics probe failed");
                        None
                    }
                    Err(error) => {
                        debug!(backend = index, error = %error, "Engine metrics probe failed");
                        None
                    }
                };
                match body.as_deref().and_then(parse_sample) {
                    Some(sample) => load.record_at(index, sample, Instant::now()),
                    None => {
                        metrics::counter!("backend_engine_probe_failures_total", "backend" => index.to_string())
                            .increment(1);
                        if body.is_some() {
                            warn!(
                                backend = index,
                                "Engine metrics did not expose a running-requests gauge"
                            );
                        }
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const SGLANG: &str = "# HELP sglang:num_running_reqs The number of running requests.\n\
# TYPE sglang:num_running_reqs gauge\n\
sglang:num_running_reqs{engine_type=\"unified\",model_name=\"m\",tp_rank=\"0\"} 4.0\n\
sglang:num_running_reqs{engine_type=\"unified\",model_name=\"m\",tp_rank=\"1\"} 4.0\n\
sglang:num_queue_reqs{engine_type=\"unified\",model_name=\"m\"} 3.0\n\
sglang:num_running_reqs_total 100\n";

    #[test]
    fn sums_gauges_across_label_sets_and_ignores_prefixed_names() {
        assert_eq!(
            parse_sample(SGLANG),
            Some(Sample {
                running: 8,
                queued: 3
            })
        );
        assert_eq!(
            parse_sample("vllm:num_requests_running{model=\"m\"} 2.0\n"),
            Some(Sample {
                running: 2,
                queued: 0
            })
        );
        assert_eq!(parse_sample("something_else 1\n"), None);
    }

    #[test]
    fn samples_age_out() {
        let load = EngineLoad::new(2, Duration::from_secs(6));
        let t0 = Instant::now();
        let sample = Sample {
            running: 1,
            queued: 2,
        };
        load.record_at(1, sample, t0);
        assert_eq!(load.get_at(1, t0 + Duration::from_secs(5)), Some(sample));
        assert_eq!(load.get_at(1, t0 + Duration::from_secs(7)), None);
        assert_eq!(load.get_at(0, t0), None);
        assert_eq!(load.get_at(9, t0), None);
        assert_eq!(EngineLoad::disabled().get_at(0, t0), None);
    }
}
