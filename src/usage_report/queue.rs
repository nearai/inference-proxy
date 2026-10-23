//! The queue behind a delivery: the reports waiting for a place, the places
//! taken, and the drain. A place is held by a task that sends reports through
//! the sender, each until it is accepted, dropped or out of attempts: one at a
//! time, or, for a sender that takes a batch, the ones already waiting.

use std::collections::VecDeque;
use std::sync::{Arc, MutexGuard};
use std::time::{Duration, Instant};

use super::log::{GiveUp, Step};
use super::{Drained, Job, UsageReportDelivery, MAX_DROPS_BEING_LOGGED};
use crate::model_metrics::ModelLabel;

#[derive(Default)]
pub(super) struct State {
    /// Reports waiting for a place, oldest first. (A waiting report refers
    /// back to its delivery through its reporter, until it leaves the queue.)
    pub(super) waiting: VecDeque<Job>,
    /// Places taken: reports being sent or backing off before a retry.
    pub(super) in_flight: usize,
    /// Reports dropped from a full queue whose line is still to be written.
    pub(super) drops_being_logged: usize,
}

impl State {
    pub(super) fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.in_flight == 0 && self.drops_being_logged == 0
    }
}

/// A place among the `max_in_flight`, held by one worker task.
pub(super) struct Place {
    pub(super) delivery: Arc<UsageReportDelivery>,
    /// The model of the report in the place, for the gauge.
    pub(super) model: ModelLabel,
    pub(super) held: bool,
}

impl Place {
    /// Deliver `first`, then whatever waits, and give the place back when
    /// nothing does. A sender with a `max_batch` above 1 takes the waiting
    /// reports up to that many at once, never waiting for more to come.
    async fn work(mut self, first: Job) {
        let max_batch = self.delivery.sender.max_batch().max(1);
        let mut jobs = vec![first];
        loop {
            if max_batch == 1 {
                self.delivery.deliver(&jobs[0]).await;
            } else {
                self.delivery.deliver_batch(jobs).await;
            }
            jobs = self.next(max_batch);
            if jobs.is_empty() {
                return;
            }
            // A backlog of reports past their deadline is dropped without a
            // single await; other tasks get their turn between two of them.
            tokio::task::yield_now().await;
        }
    }

    /// The next reports to deliver in this place, the oldest first and at most
    /// `max`, or none after giving the place back because nothing waits. One
    /// lock for both, so a report queued in between cannot be left waiting
    /// with a place free. The place counts as the model of the first report.
    pub(super) fn next(&mut self, max: usize) -> Vec<Job> {
        let (next, idle) = {
            let mut state = self.delivery.state();
            let count = state.waiting.len().min(max);
            let next: Vec<Job> = state.waiting.drain(..count).collect();
            if next.is_empty() {
                state.in_flight -= 1;
                self.held = false;
            }
            (next, state.is_idle())
        };
        self.delivery.in_flight_gauge(self.model, Step::Down);
        if let Some(first) = next.first() {
            // The in-flight gauge counts places, not reports (it is stepped
            // down above for the place's old model): one step up for the
            // batch, under the model of its first report. The queue depth
            // counts reports, so it steps down for each one claimed.
            self.model = first.reporter.model_label;
            for job in &next {
                self.delivery
                    .waiting_gauge(job.reporter.model_label, Step::Down);
            }
            self.delivery.in_flight_gauge(self.model, Step::Up);
        }
        if idle {
            self.delivery.idle.notify_waiters();
        }
        next
    }
}

impl Drop for Place {
    /// A worker that ends without giving its place back (it panicked, or the
    /// runtime is going away) must not take the place with it: under a cap
    /// that would stop delivery for good once every place was lost. Reports
    /// still waiting are picked up by the worker of the next report.
    fn drop(&mut self) {
        if self.held {
            let idle = {
                let mut state = self.delivery.state();
                state.in_flight -= 1;
                state.is_idle()
            };
            self.delivery.in_flight_gauge(self.model, Step::Down);
            if idle {
                self.delivery.idle.notify_waiters();
            }
        }
    }
}

impl UsageReportDelivery {
    pub(super) fn is_idle(&self) -> bool {
        self.state().is_idle()
    }

    pub fn pending(&self) -> (usize, usize) {
        let state = self.state();
        (state.waiting.len(), state.in_flight)
    }

    pub(super) fn state(&self) -> MutexGuard<'_, State> {
        // Nothing panics while holding the lock; if something ever does, the
        // counts are still the best there is.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn submit(self: &Arc<Self>, mut job: Job) {
        if let Err(refused) = self.sender.prepare(&mut job) {
            return self.give_up(&job, GiveUp::Refused(refused));
        }
        let model = job.reporter.model_label;
        let evicted = {
            let mut state = self.state();
            if self.policy.max_in_flight == 0 || state.in_flight < self.policy.max_in_flight {
                state.in_flight += 1;
                drop(state);
                // Held from here on, whatever becomes of the task.
                let place = Place {
                    delivery: Arc::clone(self),
                    model,
                    held: true,
                };
                self.in_flight_gauge(model, Step::Up);
                // Nothing waits; said once so the series is there to read
                // before the first report ever has to.
                self.waiting_gauge(model, Step::None);
                tokio::spawn(place.work(job));
                return;
            }
            // The queue is full: the report that has waited longest goes. It
            // has the least time left before its deadline.
            let evicted = if state.waiting.len() >= self.policy.max_queued {
                state.waiting.pop_front().map(|evicted| {
                    let in_own_task = state.drops_being_logged < MAX_DROPS_BEING_LOGGED;
                    if in_own_task {
                        state.drops_being_logged += 1;
                    }
                    (evicted, in_own_task)
                })
            } else {
                None
            };
            state.waiting.push_back(job);
            evicted
        };
        self.waiting_gauge(model, Step::Up);
        if let Some((evicted, in_own_task)) = evicted {
            self.waiting_gauge(evicted.reporter.model_label, Step::Down);
            if in_own_task {
                // The line is about another request's report: it belongs
                // neither on this caller's path nor in its request span.
                let delivery = Arc::clone(self);
                tokio::spawn(async move {
                    delivery.give_up(&evicted, GiveUp::QueueFull);
                    let idle = {
                        let mut state = delivery.state();
                        state.drops_being_logged -= 1;
                        state.is_idle()
                    };
                    if idle {
                        delivery.idle.notify_waiters();
                    }
                });
            } else {
                self.give_up(&evicted, GiveUp::QueueFull);
            }
        }
    }

    pub async fn drain(&self, limit: Duration) -> Drained {
        let started_at = Instant::now();
        let give_up_at = tokio::time::Instant::now() + limit.min(Duration::from_secs(86_400));
        let (waiting, in_flight) = self.pending();
        let pending = waiting + in_flight;
        loop {
            // Registered before the state is read, so the notification of a
            // report that ends in between is not missed.
            let idle = self.idle.notified();
            tokio::pin!(idle);
            idle.as_mut().enable();
            if self.is_idle() || tokio::time::timeout_at(give_up_at, idle).await.is_err() {
                let (left_waiting, left_in_flight) = self.pending();
                return Drained {
                    pending,
                    left_waiting,
                    left_in_flight,
                    waited: started_at.elapsed(),
                };
            }
        }
    }
}
