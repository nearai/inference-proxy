//! The SQS sender: reports to the usage queue, carried as `cloud-platform-events`
//! envelopes, in `SendMessageBatch` calls signed with SigV4 (service `sqs`). It is
//! selected by `VLLM_PROXY_USAGE_REPORT_TRANSPORT=sqs` and is the only code that
//! builds the envelope: the HTTP sender posts the report's bytes as they are.
//!
//! Each attempt is still one report's, decided by the shared delivery as before.
//! The attempts under way at once are gathered into batches: a batching task
//! takes the first attempt waiting, keeps its batch open for the batch window
//! (`VLLM_PROXY_USAGE_REPORT_SQS_BATCH_WINDOW_MS`) or until it holds ten reports
//! or as many bytes as one envelope may have, and sends it in one call. SQS bills
//! a batch as one request, and one call carries up to ten reports, so a busy
//! process makes up to ten times fewer calls. Every report gets its own answer
//! from the batch's: the answer to the call when the call failed as a whole,
//! else the entry's.
//!
//! The sender classifies each answer for the shared delivery. Only explicit,
//! non-retryable answers end a report: a denied access, an invalid parameter, a
//! missing queue, and a report too large for the queue. Everything else is
//! retried, credentials included: a refused or unreadable credential is fetched
//! again before the next attempt. A report retried after its batch was sent
//! after all is deduplicated by the billing API on its completion id.
//!
//! The report is validated and serialized once, when it is submitted
//! (`UsageSender::prepare`); each batch only wraps, signs and sends those bytes.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use aws_credential_types::Credentials;
use aws_sigv4::http_request::{sign, SignableBody, SignableRequest, SigningSettings};
use aws_sigv4::sign::v4;
use bytes::Bytes;
use cloud_platform_events::{
    EncodeError, InternalUsageReport, Producer, UsageEnvelope, CURRENT_USAGE_VERSION,
    MAX_ENVELOPE_BYTES,
};
use futures_util::future::BoxFuture;
use reqwest::StatusCode;
use serde::Deserialize;
use tokio::sync::{mpsc, oneshot};

use super::{AttemptLog, Job, Refused, SendOutcome, SqsSettings, TransportKind, UsageSender};
use crate::aws_credentials::{
    no_redirect_client, read_bounded, CredentialFailure, CredentialProvider,
};
use crate::proxy::{classify_usage_request_error, UsageReportOutcome};

const SERVICE: &str = "sqs";
const TARGET: &str = "AmazonSQS.SendMessageBatch";
const CHECK_TARGET: &str = "AmazonSQS.GetQueueAttributes";
const CONTENT_TYPE: &str = "application/x-amz-json-1.0";
/// The most of an SQS error body that is read for its `__type`.
const MAX_ERROR_BODY_BYTES: usize = 16 * 1024;
/// The most of a successful answer that is read: a batch's answer names each
/// of its entries in a few hundred bytes.
const MAX_ANSWER_BYTES: usize = 64 * 1024;
/// The most entries one `SendMessageBatch` may carry.
const MAX_BATCH_ENTRIES: usize = 10;
/// The most message bytes one batch carries: those of the largest envelope, so
/// that any report the queue takes fits a batch of its own, and a batch stays
/// within the queue's limit on a whole batch, which is at least this.
const MAX_BATCH_BYTES: usize = MAX_ENVELOPE_BYTES;

/// What one report's attempt came to: its outcome, and the status the queue
/// answered or the error that stood in the way.
type Answer = (UsageReportOutcome, Result<StatusCode, String>);

pub(crate) struct SqsSender {
    queue: Arc<Queue>,
    /// How long a batch stays open for more reports after its first.
    batch_window: Duration,
    /// Where an attempt hands its report to the batching task. The task is
    /// started by the first attempt (it needs the runtime), and again by an
    /// attempt that finds it ended.
    batcher: Mutex<Option<mpsc::UnboundedSender<Pending>>>,
}

/// The usage queue and how to reach it: what every call needs, shared by the
/// sender and its batching task.
pub(crate) struct Queue {
    pub(crate) queue_url: String,
    pub(crate) region: String,
    /// Where the request goes: the regional SQS endpoint, or the override
    /// `VLLM_PROXY_USAGE_REPORT_SQS_ENDPOINT` (a local emulator).
    pub(crate) endpoint: String,
    pub(crate) credentials: CredentialProvider,
    /// The client of the credentials fetch and of the send. Not the shared
    /// proxy client: it never follows a redirect, so the metadata session
    /// token and the signed request go to the configured hosts only.
    pub(crate) client: reqwest::Client,
}

/// A report waiting for its batch.
struct Pending {
    /// The `MessageBody`: the report's envelope.
    message: String,
    /// When its attempt gives up: a batch's call gets the time its most
    /// pressed report has left.
    deadline: tokio::time::Instant,
    /// Where its answer goes.
    answer: oneshot::Sender<Answer>,
}

/// What one signed call came to.
struct CallAnswer {
    outcome: UsageReportOutcome,
    /// The status the queue answered, or the error that stood in the way.
    status: Result<StatusCode, String>,
    /// The error type an error answer names (`QueueDoesNotExist`, `AccessDenied`, ...).
    error_type: Option<String>,
    /// The body of a successful answer; empty otherwise.
    body: Vec<u8>,
}

impl CallAnswer {
    fn failed(outcome: UsageReportOutcome, error: String) -> Self {
        Self {
            outcome,
            status: Err(error),
            error_type: None,
            body: Vec::new(),
        }
    }
}

impl SqsSender {
    pub(crate) fn new(settings: &SqsSettings) -> Self {
        let endpoint = settings
            .endpoint
            .clone()
            .unwrap_or_else(|| format!("https://sqs.{}.amazonaws.com/", settings.region));
        Self::with_queue(
            Queue {
                queue_url: settings.queue_url.clone(),
                region: settings.region.clone(),
                endpoint,
                credentials: CredentialProvider::new(settings.credentials_url.clone()),
                client: no_redirect_client(),
            },
            settings.batch_window,
        )
    }

    pub(crate) fn with_queue(queue: Queue, batch_window: Duration) -> Self {
        Self {
            queue: Arc::new(queue),
            batch_window,
            batcher: Mutex::new(None),
        }
    }

    /// One attempt for `job`, which gives up after `timeout`: its report joins
    /// a batch, and the attempt waits for the batch's answer for it.
    async fn attempt(&self, job: &Job, timeout: Duration) -> Answer {
        let Ok(message) = std::str::from_utf8(&job.body) else {
            return (
                UsageReportOutcome::SqsInvalid,
                Err("message is not UTF-8".to_string()),
            );
        };
        let (answer, answered) = oneshot::channel();
        self.enqueue(Pending {
            message: message.to_owned(),
            deadline: tokio::time::Instant::now() + timeout,
            answer,
        });
        match tokio::time::timeout(timeout, answered).await {
            Ok(Ok(answer)) => answer,
            // The batching task ended without answering: the runtime is going away.
            Ok(Err(_)) => (
                UsageReportOutcome::SqsUnknown,
                Err("the batch ended without an answer".to_string()),
            ),
            Err(_) => (
                UsageReportOutcome::Timeout,
                Err("no answer within the attempt's timeout".to_string()),
            ),
        }
    }

    /// Hands `pending` to the batching task, starting it if there is none.
    fn enqueue(&self, pending: Pending) {
        let mut batcher = self.batcher.lock().unwrap_or_else(|e| e.into_inner());
        let pending = match batcher.as_ref() {
            Some(running) => match running.send(pending) {
                Ok(()) => return,
                // The task ended (its runtime went away): another is started.
                Err(mpsc::error::SendError(pending)) => pending,
            },
            None => pending,
        };
        let (sender, receiver) = mpsc::unbounded_channel();
        tokio::spawn(gather(Arc::clone(&self.queue), self.batch_window, receiver));
        // The receiver is alive: its task was just spawned.
        let _ = sender.send(pending);
        *batcher = Some(sender);
    }
}

/// The batching task: gathers the reports waiting into batches and sends each
/// in a task of its own, so a slow call does not hold up the next batch. It
/// ends with its sender.
async fn gather(
    queue: Arc<Queue>,
    window: Duration,
    mut waiting: mpsc::UnboundedReceiver<Pending>,
) {
    // The report that did not fit the last batch: the first of the next.
    let mut left_over = None;
    loop {
        let first = match left_over.take() {
            Some(first) => first,
            None => match waiting.recv().await {
                Some(first) => first,
                None => return,
            },
        };
        let closes_at = tokio::time::Instant::now() + window;
        let mut bytes = first.message.len();
        let mut batch = vec![first];
        while batch.len() < MAX_BATCH_ENTRIES {
            let next = match waiting.try_recv() {
                Ok(next) => next,
                Err(mpsc::error::TryRecvError::Disconnected) => break,
                Err(mpsc::error::TryRecvError::Empty) => {
                    match tokio::time::timeout_at(closes_at, waiting.recv()).await {
                        Ok(Some(next)) => next,
                        Ok(None) | Err(_) => break,
                    }
                }
            };
            if bytes + next.message.len() > MAX_BATCH_BYTES {
                left_over = Some(next);
                break;
            }
            bytes += next.message.len();
            batch.push(next);
        }
        tokio::spawn(Arc::clone(&queue).send_batch(batch));
    }
}

impl Queue {
    /// One `SendMessageBatch` of `batch`, within the time its most pressed
    /// report has left; every report gets its answer.
    async fn send_batch(self: Arc<Self>, batch: Vec<Pending>) {
        let now = tokio::time::Instant::now();
        let timeout = batch
            .iter()
            .map(|pending| pending.deadline.saturating_duration_since(now))
            .min()
            .unwrap_or_default();
        let entries: Vec<_> = batch
            .iter()
            .enumerate()
            .map(|(id, pending)| {
                serde_json::json!({"Id": id.to_string(), "MessageBody": pending.message})
            })
            .collect();
        let body = Bytes::from(
            serde_json::json!({"QueueUrl": self.queue_url, "Entries": entries}).to_string(),
        );
        let answer = self.call(TARGET, &body, timeout).await;
        let answers = entry_answers(answer, batch.len());
        for (pending, answer) in batch.into_iter().zip(answers) {
            // An attempt that gave up no longer listens.
            let _ = pending.answer.send(answer);
        }
    }

    /// The credentials for one call, within `timeout`. A failure is the
    /// call's outcome and its log text.
    async fn credentials_within(
        &self,
        timeout: Duration,
    ) -> Result<Credentials, (UsageReportOutcome, String)> {
        match tokio::time::timeout(timeout, self.credentials.get(&self.client)).await {
            Ok(Ok(credentials)) => Ok(credentials),
            Ok(Err(CredentialFailure(error))) => Err((UsageReportOutcome::SqsCredentials, error)),
            Err(_) => Err((
                UsageReportOutcome::Timeout,
                "credentials were not received within the attempt's timeout".to_string(),
            )),
        }
    }

    /// The error type an SQS error answer names: its header, else its body.
    /// The body is read for its error type only, and only up to a bound; what
    /// it says is not logged.
    async fn error_type(response: reqwest::Response) -> Option<String> {
        let from_header = response
            .headers()
            .get("x-amzn-errortype")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let body = read_bounded(response, MAX_ERROR_BODY_BYTES)
            .await
            .unwrap_or_default();
        from_header.or_else(|| error_type_in_body(&String::from_utf8_lossy(&body)))
    }

    /// One signed call of the SQS action `target` with `body`, which gives up
    /// after `timeout` in all: the credentials and the send share that budget.
    async fn call(&self, target: &str, body: &Bytes, timeout: Duration) -> CallAnswer {
        let started_at = Instant::now();
        let credentials = match self.credentials_within(timeout).await {
            Ok(credentials) => credentials,
            Err((outcome, error)) => return CallAnswer::failed(outcome, error),
        };
        let Some(remaining) = timeout
            .checked_sub(started_at.elapsed())
            .filter(|left| !left.is_zero())
        else {
            return CallAnswer::failed(
                UsageReportOutcome::Timeout,
                "the attempt's time was used up fetching credentials".to_string(),
            );
        };
        let headers = match sign_request(
            &self.endpoint,
            &self.region,
            &credentials,
            target,
            body,
            SystemTime::now(),
        ) {
            Ok(headers) => headers,
            Err(error) => return CallAnswer::failed(UsageReportOutcome::SqsInvalid, error),
        };

        let mut request = self
            .client
            .post(&self.endpoint)
            .timeout(remaining)
            .header(reqwest::header::CONTENT_TYPE, CONTENT_TYPE)
            .header("x-amz-target", target)
            .body(body.clone());
        // The headers SigV4 added: the date and the authorization.
        for (name, value) in headers {
            request = request.header(name, value);
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                let outcome = match classify_usage_request_error(&error) {
                    UsageReportOutcome::TransportError if error.is_builder() => {
                        UsageReportOutcome::SqsInvalid
                    }
                    other => other,
                };
                return CallAnswer::failed(outcome, error.to_string());
            }
        };
        let status = response.status();
        if status.is_success() {
            // A batch's answer says which of its entries were taken: an answer
            // that cannot be read names none, and every entry is sent again.
            return match read_bounded(response, MAX_ANSWER_BYTES).await {
                Ok(body) => CallAnswer {
                    outcome: UsageReportOutcome::Accepted,
                    status: Ok(status),
                    error_type: None,
                    body,
                },
                Err(error) => {
                    CallAnswer::failed(UsageReportOutcome::SqsUnknown, format!("answer {error}"))
                }
            };
        }
        // A redirect is final whatever its body says: nothing in it is read.
        let error_type = if status.is_redirection() {
            None
        } else {
            Self::error_type(response).await
        };
        let outcome = classify_sqs_answer(status, error_type.as_deref());
        if matches!(
            outcome,
            UsageReportOutcome::SqsCredentials | UsageReportOutcome::SqsDenied
        ) {
            // Credentials the service refused are not sent again: the next
            // attempt fetches new ones.
            self.credentials.forget(&credentials);
        }
        CallAnswer {
            outcome,
            status: Ok(status),
            error_type,
            body: Vec::new(),
        }
    }
}

/// A `SendMessageBatch` answer: the entries taken, and those refused.
#[derive(Default, Deserialize)]
#[serde(rename_all = "PascalCase", default)]
struct BatchAnswer {
    successful: Vec<BatchEntryTaken>,
    failed: Vec<BatchEntryRefused>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct BatchEntryTaken {
    id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct BatchEntryRefused {
    id: String,
    /// Whether the entry itself is at fault (a 4xx of its own) rather than the
    /// service (a 5xx).
    #[serde(default)]
    sender_fault: bool,
    #[serde(default)]
    code: String,
}

/// The answer of each of a batch's `entries` (`Id` `"0"`, `"1"`, ... in order).
/// A call that failed as a whole fails every entry the same way. A call taken
/// answers each entry by its own: taken, refused for the reason it names
/// (classified as a call answering that error would be), or, if the answer
/// names it nowhere, unknown and sent again.
fn entry_answers(call: CallAnswer, entries: usize) -> Vec<Answer> {
    let status = match (call.outcome, &call.status) {
        (UsageReportOutcome::Accepted, Ok(status)) => *status,
        _ => return vec![(call.outcome, call.status); entries],
    };
    let answer: BatchAnswer = match serde_json::from_slice(&call.body) {
        Ok(answer) => answer,
        Err(_) => {
            // Its length only: an answer is not logged.
            tracing::warn!(
                answer_bytes = call.body.len(),
                entries,
                "SendMessageBatch answer could not be read; its entries are sent again"
            );
            BatchAnswer::default()
        }
    };
    (0..entries)
        .map(|entry| {
            let id = entry.to_string();
            if answer.successful.iter().any(|taken| taken.id == id) {
                return (UsageReportOutcome::Accepted, Ok(status));
            }
            match answer.failed.iter().find(|refused| refused.id == id) {
                Some(refused) => {
                    let as_status = if refused.sender_fault {
                        StatusCode::BAD_REQUEST
                    } else {
                        StatusCode::INTERNAL_SERVER_ERROR
                    };
                    (
                        classify_sqs_answer(as_status, Some(&refused.code)),
                        Err(format!("entry refused: {}", error_name(&refused.code))),
                    )
                }
                None => (
                    UsageReportOutcome::SqsUnknown,
                    Err("the answer does not name the entry".to_string()),
                ),
            }
        })
        .collect()
}

impl UsageSender for SqsSender {
    fn kind(&self) -> TransportKind {
        TransportKind::Sqs
    }

    /// The report's `MessageBody`, built once: its envelope. A report the
    /// queue cannot carry is refused here.
    fn prepare(&self, job: &mut Job) -> Result<(), Refused> {
        job.body = Bytes::from(message_body(job)?);
        Ok(())
    }

    /// `GetQueueAttributes` (`QueueArn`) on the usage queue: one attempt that
    /// proves the queue exists and these credentials may reach it, signed like
    /// a `SendMessageBatch`. Needs `sqs:GetQueueAttributes` besides `sqs:SendMessage`.
    fn startup_check<'a>(
        &'a self,
        timeout: Duration,
    ) -> BoxFuture<'a, Result<(), (UsageReportOutcome, String)>> {
        Box::pin(async move {
            let body = Bytes::from(
                serde_json::json!({"QueueUrl": self.queue.queue_url, "AttributeNames": ["QueueArn"]})
                    .to_string(),
            );
            let answer = self.queue.call(CHECK_TARGET, &body, timeout).await;
            match answer {
                CallAnswer {
                    outcome: UsageReportOutcome::Accepted,
                    ..
                } => Ok(()),
                CallAnswer {
                    outcome,
                    status: Ok(status),
                    error_type,
                    ..
                } => Err((
                    outcome,
                    match error_type.as_deref().map(error_name) {
                        Some(name) => format!("status {status}, {name}"),
                        None => format!("status {status}"),
                    },
                )),
                CallAnswer {
                    outcome,
                    status: Err(error),
                    ..
                } => Err((outcome, error)),
            }
        })
    }

    fn send<'a>(
        &'a self,
        job: &'a Job,
        timeout: Duration,
    ) -> BoxFuture<'a, (SendOutcome, AttemptLog)> {
        Box::pin(async move {
            let started_at = Instant::now();
            let (outcome, answer) = self.attempt(job, timeout).await;
            let decision = match outcome {
                UsageReportOutcome::Accepted => SendOutcome::Accepted,
                _ if outcome.is_retryable() => SendOutcome::Retryable(outcome.as_label()),
                _ => SendOutcome::Final(outcome.as_label()),
            };
            (
                decision,
                AttemptLog {
                    answer,
                    outcome,
                    duration: started_at.elapsed(),
                },
            )
        })
    }
}

/// An error type as it may be logged: AWS's name for what is wrong, which is
/// no secret but not trusted text either, so it is cut to a name's characters.
/// A name with none of them left (an entry refused without a code) is `unknown`.
fn error_name(error_type: &str) -> String {
    let name: String = short_error_name(error_type)
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        .take(64)
        .collect();
    if name.is_empty() {
        "unknown".to_string()
    } else {
        name
    }
}

/// The `MessageBody`: the report as a usage envelope, emitted when the
/// request completed. The usage is the report's own body read into the events
/// crate's payload type. That read must lose nothing: a field the schema does
/// not carry would reach the HTTP billing and not this queue, so the usage
/// written back has to equal the body, or the report is refused here as final.
/// An envelope the queue cannot carry is refused too, as `sqs_oversize`; the
/// log names its length only.
fn message_body(job: &Job) -> Result<String, Refused> {
    let refuse = |error: String| Refused {
        reason: "sqs_invalid",
        outcome: UsageReportOutcome::SqsInvalid,
        error,
    };
    let original: serde_json::Value = serde_json::from_slice(&job.body)
        .map_err(|_| refuse("usage body is not JSON".to_string()))?;
    let usage: InternalUsageReport = serde_json::from_value(original.clone())
        .map_err(|_| refuse("usage body does not match the usage schema".to_string()))?;
    let written_back = serde_json::to_value(&usage)
        .map_err(|_| refuse("usage body does not match the usage schema".to_string()))?;
    if written_back != original {
        return Err(refuse(
            "usage body has fields the usage schema does not carry".to_string(),
        ));
    }
    UsageEnvelope {
        v: CURRENT_USAGE_VERSION,
        producer: Producer::OpenrouterGateway,
        emitted_at: job.emitted_at,
        usage,
    }
    .try_to_json()
    .map_err(|error| match error {
        EncodeError::EnvelopeTooLarge(length) => Refused {
            reason: "sqs_oversize",
            outcome: UsageReportOutcome::SqsInvalid,
            error: format!("usage envelope of {length} bytes is over the queue's limit"),
        },
        EncodeError::Serialize => refuse("usage envelope could not be serialized".to_string()),
    })
}

/// The headers SigV4 adds to an SQS call of `target` with `payload` to `endpoint` at
/// `time`, with the two the request carries itself (content type and target)
/// signed.
fn sign_request(
    endpoint: &str,
    region: &str,
    credentials: &aws_credential_types::Credentials,
    target: &str,
    payload: &[u8],
    time: SystemTime,
) -> Result<Vec<(String, String)>, String> {
    let identity = credentials.clone().into();
    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name(SERVICE)
        .time(time)
        .settings(SigningSettings::default())
        .build()
        .map_err(|error| error.to_string())?;
    let headers = [("content-type", CONTENT_TYPE), ("x-amz-target", target)];
    let request = SignableRequest::new(
        "POST",
        endpoint,
        headers.into_iter(),
        SignableBody::Bytes(payload),
    )
    .map_err(|error| error.to_string())?;
    let output = sign(request, &params.into()).map_err(|error| error.to_string())?;
    let (instructions, _signature) = output.into_parts();
    Ok(instructions
        .headers()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect())
}

/// The error type of an SQS error body (`{"__type": "com.amazonaws.sqs#OverLimit"}`).
fn error_type_in_body(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    value.get("__type")?.as_str().map(str::to_owned)
}

/// The short name of an error type: the part after the namespace (`#`), and
/// before any `:` a header value appends.
fn short_error_name(error_type: &str) -> &str {
    let name = error_type.rsplit('#').next().unwrap_or(error_type);
    name.split(':').next().unwrap_or(name)
}

/// What an SQS answer means. A 2xx is accepted. Only an explicit answer that no
/// other attempt can change ends the report: access denied, an invalid
/// parameter or a missing queue. Throttling, 5xx and credential failures
/// (refused, expired, skewed) are retried; so is any answer that names no known
/// error, since dropping a report for an answer nobody classified is the loss
/// this sender exists to avoid.
pub(crate) fn classify_sqs_answer(
    status: StatusCode,
    error_type: Option<&str>,
) -> UsageReportOutcome {
    if status.is_success() {
        return UsageReportOutcome::Accepted;
    }
    // Redirects are never followed: a queue endpoint that redirects is
    // misconfigured, and asking again does not change that. This comes before
    // the error type, so a redirect whose body names a throttle is not retried.
    if status.is_redirection() {
        return UsageReportOutcome::SqsInvalid;
    }
    match error_type.map(short_error_name) {
        Some(
            "ThrottlingException"
            | "Throttling"
            | "RequestThrottled"
            | "RequestThrottledException"
            | "OverLimit",
        ) => UsageReportOutcome::SqsThrottled,
        // A credential the service does not accept now: fetched again.
        // Clock skew (RequestExpired, RequestTimeTooSkewed, and a signature
        // that is wrong because the clock is) is one of these.
        Some(
            "ExpiredToken"
            | "ExpiredTokenException"
            | "RequestExpired"
            | "RequestTimeTooSkewed"
            | "InvalidClientTokenId"
            | "SignatureDoesNotMatch"
            | "UnrecognizedClientException"
            | "MissingAuthenticationToken"
            | "IncompleteSignature",
        ) => UsageReportOutcome::SqsCredentials,
        Some("AccessDenied" | "AccessDeniedException") => UsageReportOutcome::SqsDenied,
        Some(
            "InvalidParameterValue"
            | "InvalidParameterCombination"
            | "MissingParameter"
            | "InvalidMessageContents"
            | "InvalidAttributeName"
            | "InvalidAttributeValue"
            | "QueueDoesNotExist"
            | "AWS.SimpleQueueService.NonExistentQueue",
        ) => UsageReportOutcome::SqsInvalid,
        _ => match status {
            StatusCode::TOO_MANY_REQUESTS => UsageReportOutcome::SqsThrottled,
            _ if status.is_server_error() => UsageReportOutcome::Sqs5xx,
            StatusCode::FORBIDDEN | StatusCode::UNAUTHORIZED => UsageReportOutcome::SqsCredentials,
            _ => UsageReportOutcome::SqsUnknown,
        },
    }
}

#[cfg(test)]
#[path = "sqs_sender_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "sqs_sender_redirect_tests.rs"]
mod redirect_tests;

#[cfg(test)]
#[path = "sqs_sender_startup_tests.rs"]
mod startup_tests;

#[cfg(test)]
#[path = "sqs_sender_batch_tests.rs"]
mod batch_tests;
