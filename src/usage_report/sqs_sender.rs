//! The SQS sender: the reports to the usage queue, each carried as a
//! `cloud-platform-events` envelope, in a `SendMessageBatch` of the reports
//! already waiting (a retry is a single `SendMessage`), signed with SigV4
//! (service `sqs`). It is selected by `VLLM_PROXY_USAGE_REPORT_TRANSPORT=sqs`
//! and is the only code that builds the envelope: the HTTP sender posts the
//! report's bytes as they are.
//!
//! The sender makes one attempt and classifies its answer for the shared
//! delivery. A batch answers per report: a call that fails as a whole gives
//! every report the outcome a single send would get, and an answer that
//! names failed entries classifies each by its error code. Only explicit, non-retryable answers end a report: a denied
//! access, an invalid parameter, a missing queue, and a report too large for
//! the queue. Everything else is retried, credentials included: a refused or
//! unreadable credential is fetched again before the next attempt.
//!
//! The report is validated and serialized once, when it is submitted
//! (`UsageSender::prepare`); each attempt only signs and sends those bytes.

use std::time::{Duration, Instant, SystemTime};

use aws_credential_types::Credentials;
use aws_sigv4::http_request::{sign, SignableBody, SignableRequest, SigningSettings};
use aws_sigv4::sign::v4;
use bytes::Bytes;
use cloud_platform_events::{
    EncodeError, InternalUsageReport, Producer, UsageEnvelope, CURRENT_USAGE_VERSION,
};
use futures_util::future::BoxFuture;
use reqwest::StatusCode;

use super::{AttemptLog, Job, Refused, SendOutcome, SqsSettings, TransportKind, UsageSender};
use crate::aws_credentials::{
    no_redirect_client, read_bounded, CredentialFailure, CredentialProvider,
};
use crate::proxy::{classify_usage_request_error, UsageReportOutcome};

const SERVICE: &str = "sqs";
const TARGET: &str = "AmazonSQS.SendMessage";
const BATCH_TARGET: &str = "AmazonSQS.SendMessageBatch";
const CHECK_TARGET: &str = "AmazonSQS.GetQueueAttributes";
const CONTENT_TYPE: &str = "application/x-amz-json-1.0";
/// The most of an SQS error body that is read for its `__type`.
const MAX_ERROR_BODY_BYTES: usize = 16 * 1024;
/// SQS takes at most 10 entries in a `SendMessageBatch`.
const MAX_BATCH_ENTRIES: usize = 10;
/// The most message bytes in one batch. SQS allows 1 MiB for a batch of a queue
/// whose messages may be that large, and 256 KiB for older queues; the smaller
/// one is kept, which is also the most one envelope may be
/// (`cloud_platform_events::MAX_ENVELOPE_BYTES`). Reports are about 1 KB, so
/// this only splits a batch of unusually large ones.
const MAX_BATCH_BYTES: usize = 262_144;
/// The most of a `SendMessageBatch` answer that is read: ten entries and their
/// error messages are far less.
const MAX_BATCH_RESPONSE_BYTES: usize = 64 * 1024;

pub(crate) struct SqsSender {
    queue_url: String,
    region: String,
    /// Where the request goes: the regional SQS endpoint, or the override
    /// `VLLM_PROXY_USAGE_REPORT_SQS_ENDPOINT` (a local emulator).
    endpoint: String,
    credentials: CredentialProvider,
    /// The client of the credentials fetch and of the send. Not the shared
    /// proxy client: it never follows a redirect, so the metadata session
    /// token and the signed request go to the configured hosts only.
    client: reqwest::Client,
}

impl SqsSender {
    pub(crate) fn new(settings: &SqsSettings) -> Self {
        let endpoint = settings
            .endpoint
            .clone()
            .unwrap_or_else(|| format!("https://sqs.{}.amazonaws.com/", settings.region));
        Self {
            queue_url: settings.queue_url.clone(),
            region: settings.region.clone(),
            endpoint,
            credentials: CredentialProvider::new(settings.credentials_url.clone()),
            client: no_redirect_client(),
        }
    }

    /// The credentials for one attempt, within `timeout`. A failure is the
    /// attempt's outcome and its log text.
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

    /// One attempt, which gives up after `timeout` in all: the credentials and
    /// the send share that budget. Returns the attempt's outcome, which decides
    /// its retry, and the answer for the log line.
    async fn attempt(
        &self,
        job: &Job,
        timeout: Duration,
    ) -> (UsageReportOutcome, Result<StatusCode, String>) {
        let called = self.call(TARGET, &job.body, timeout, false).await;
        (called.outcome, called.answer)
    }

    /// One signed call of the SQS action `target` with `body`, within `timeout`.
    /// Besides the outcome and the answer, the error type the answer names, if
    /// it does (`QueueDoesNotExist`, `AccessDenied`, ...), and, with
    /// `read_answer`, the body of a successful answer.
    async fn call(
        &self,
        target: &str,
        body: &Bytes,
        timeout: Duration,
        read_answer: bool,
    ) -> Called {
        let started_at = Instant::now();
        let credentials = match self.credentials_within(timeout).await {
            Ok(credentials) => credentials,
            Err((outcome, error)) => return Called::failed(outcome, error),
        };
        let Some(remaining) = timeout
            .checked_sub(started_at.elapsed())
            .filter(|left| !left.is_zero())
        else {
            return Called::failed(
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
            Err(error) => return Called::failed(UsageReportOutcome::SqsInvalid, error),
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
                return Called::failed(outcome, error.to_string());
            }
        };
        let status = response.status();
        // A redirect is final whatever its body says: nothing in it is read.
        let mut answer = None;
        let error_type = if status.is_success() {
            if read_answer {
                answer = read_bounded(response, MAX_BATCH_RESPONSE_BYTES).await.ok();
            }
            None
        } else if status.is_redirection() {
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
        Called {
            outcome,
            answer: Ok(status),
            error_type,
            body: answer,
        }
    }

    /// The `SendMessageBatch` of `messages` (at most 10, within the byte limit):
    /// one result per message, in order. Entry `Id`s are the positions.
    async fn send_chunk(
        &self,
        messages: &[&str],
        timeout: Duration,
    ) -> Vec<(SendOutcome, AttemptLog)> {
        let started_at = Instant::now();
        let entries: Vec<_> = messages
            .iter()
            .enumerate()
            .map(|(id, message)| serde_json::json!({"Id": id.to_string(), "MessageBody": message}))
            .collect();
        let body = Bytes::from(
            serde_json::json!({"QueueUrl": self.queue_url, "Entries": entries}).to_string(),
        );
        let called = self.call(BATCH_TARGET, &body, timeout, true).await;
        let duration = started_at.elapsed();
        let result = |outcome: UsageReportOutcome, answer: Result<StatusCode, String>| {
            (
                decide(outcome),
                AttemptLog {
                    answer,
                    outcome,
                    duration,
                },
            )
        };
        // The call failed as a whole: each report has what a single send
        // would have got.
        if called.outcome != UsageReportOutcome::Accepted {
            return messages
                .iter()
                .map(|_| result(called.outcome, called.answer.clone()))
                .collect();
        }
        let status = called.answer.clone().unwrap_or(StatusCode::OK);
        let Some(answered) = called
            .body
            .as_deref()
            .and_then(|body| serde_json::from_slice::<BatchAnswer>(body).ok())
        else {
            // Accepted, but which messages were taken is not known. They are
            // sent again; the billing side deduplicates.
            return messages
                .iter()
                .map(|_| {
                    result(
                        UsageReportOutcome::SqsUnknown,
                        Err("the batch answer could not be read".to_string()),
                    )
                })
                .collect();
        };
        // A message the answer does not name is sent again too.
        let mut results: Vec<_> = messages
            .iter()
            .map(|_| {
                result(
                    UsageReportOutcome::SqsUnknown,
                    Err("the batch answer does not name the message".to_string()),
                )
            })
            .collect();
        for entry in answered.successful {
            if let Some(slot) = entry_slot(&entry.id, &mut results) {
                *slot = result(UsageReportOutcome::Accepted, Ok(status));
            }
        }
        for entry in answered.failed {
            let Some(slot) = entry_slot(&entry.id, &mut results) else {
                continue;
            };
            let code = entry.code.as_deref();
            // The status a single send of this entry would have been answered
            // with: 400 for the sender's fault, else 500.
            let mut outcome = classify_sqs_answer(
                if entry.sender_fault {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                },
                code,
            );
            // Not the sender's fault: another attempt may pass, whatever the
            // code says.
            if !entry.sender_fault && !outcome.is_retryable() {
                outcome = UsageReportOutcome::Sqs5xx;
            }
            let error = match code.map(sanitized_name) {
                Some(name) if !name.is_empty() => format!("batch entry failed, {name}"),
                _ => "batch entry failed".to_string(),
            };
            *slot = result(outcome, Err(error));
        }
        results
    }
}

/// The answer of one `call`.
struct Called {
    outcome: UsageReportOutcome,
    answer: Result<StatusCode, String>,
    /// The error type the answer names, if it does.
    error_type: Option<String>,
    /// The body of a successful answer, when it was asked for and could be read.
    body: Option<Vec<u8>>,
}

impl Called {
    fn failed(outcome: UsageReportOutcome, error: String) -> Self {
        Self {
            outcome,
            answer: Err(error),
            error_type: None,
            body: None,
        }
    }
}

/// What the delivery does with an attempt that ended in `outcome`.
fn decide(outcome: UsageReportOutcome) -> SendOutcome {
    match outcome {
        UsageReportOutcome::Accepted => SendOutcome::Accepted,
        _ if outcome.is_retryable() => SendOutcome::Retryable(outcome.as_label()),
        _ => SendOutcome::Final(outcome.as_label()),
    }
}

/// The result of the entry with this `Id` (its position in the call).
fn entry_slot<'a>(
    id: &str,
    results: &'a mut [(SendOutcome, AttemptLog)],
) -> Option<&'a mut (SendOutcome, AttemptLog)> {
    results.get_mut(id.parse::<usize>().ok()?)
}

/// A `SendMessageBatch` answer: the entries taken and the entries refused.
#[derive(serde::Deserialize)]
struct BatchAnswer {
    #[serde(default, rename = "Successful")]
    successful: Vec<BatchEntryAnswer>,
    #[serde(default, rename = "Failed")]
    failed: Vec<BatchEntryAnswer>,
}

#[derive(serde::Deserialize)]
struct BatchEntryAnswer {
    #[serde(rename = "Id")]
    id: String,
    #[serde(default, rename = "SenderFault")]
    sender_fault: bool,
    #[serde(default, rename = "Code")]
    code: Option<String>,
}

/// The `MessageBody` of a prepared report: the `SendMessage` request it carries.
fn message_of(job: &Job) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct Prepared {
        #[serde(rename = "MessageBody")]
        message: String,
    }
    serde_json::from_slice::<Prepared>(&job.body)
        .ok()
        .map(|prepared| prepared.message)
}

/// Splits `messages` (with their positions) into calls of at most 10 messages
/// and `MAX_BATCH_BYTES` of message text, in order.
fn chunks(messages: &[(usize, String)]) -> Vec<&[(usize, String)]> {
    let mut chunks = Vec::new();
    let (mut start, mut bytes) = (0, 0);
    for (index, (_, message)) in messages.iter().enumerate() {
        if index > start
            && (index - start == MAX_BATCH_ENTRIES || bytes + message.len() > MAX_BATCH_BYTES)
        {
            chunks.push(&messages[start..index]);
            (start, bytes) = (index, 0);
        }
        bytes += message.len();
    }
    if start < messages.len() {
        chunks.push(&messages[start..]);
    }
    chunks
}

impl UsageSender for SqsSender {
    fn kind(&self) -> TransportKind {
        TransportKind::Sqs
    }

    /// The `SendMessage` request for the report, built once: the envelope, the
    /// queue and the body. A report the queue cannot carry is refused here.
    fn prepare(&self, job: &mut Job) -> Result<(), Refused> {
        let message = message_body(job)?;
        let payload = serde_json::to_vec(&serde_json::json!({
            "QueueUrl": self.queue_url,
            "MessageBody": message,
        }))
        .map_err(|_| Refused {
            reason: "sqs_invalid",
            outcome: UsageReportOutcome::SqsInvalid,
            error: "request body does not serialize".to_string(),
        })?;
        job.body = Bytes::from(payload);
        Ok(())
    }

    /// `GetQueueAttributes` (`QueueArn`) on the usage queue: one attempt that
    /// proves the queue exists and these credentials may reach it, signed like
    /// a `SendMessage`. Needs `sqs:GetQueueAttributes` besides `sqs:SendMessage`.
    fn startup_check<'a>(
        &'a self,
        timeout: Duration,
    ) -> BoxFuture<'a, Result<(), (UsageReportOutcome, String)>> {
        Box::pin(async move {
            let body = Bytes::from(
                serde_json::json!({"QueueUrl": self.queue_url, "AttributeNames": ["QueueArn"]})
                    .to_string(),
            );
            let called = self.call(CHECK_TARGET, &body, timeout, false).await;
            match (called.outcome, called.answer, called.error_type) {
                (UsageReportOutcome::Accepted, ..) => Ok(()),
                (outcome, Ok(status), error_type) => Err((
                    outcome,
                    // The error type is AWS's name for what is wrong with the
                    // queue or the permission: no secret, but not trusted
                    // text, so it is cut to a name's characters.
                    match error_type.as_deref().map(short_error_name) {
                        Some(name) => format!("status {status}, {}", sanitized(name)),
                        None => format!("status {status}"),
                    },
                )),
                (outcome, Err(error), _) => Err((outcome, error)),
            }
        })
    }

    fn max_batch(&self) -> usize {
        MAX_BATCH_ENTRIES
    }

    /// `SendMessageBatch` of the reports, split into several calls (sent
    /// together) only when they are more than a call may carry in bytes.
    fn send_batch<'a>(
        &'a self,
        jobs: &'a [&'a Job],
        timeout: Duration,
    ) -> BoxFuture<'a, Vec<(SendOutcome, AttemptLog)>> {
        Box::pin(async move {
            let mut results: Vec<Option<(SendOutcome, AttemptLog)>> =
                jobs.iter().map(|_| None).collect();
            let mut messages = Vec::with_capacity(jobs.len());
            for (index, job) in jobs.iter().enumerate() {
                match message_of(job) {
                    Some(message) => messages.push((index, message)),
                    // Prepared at submit, so not reached in practice. Final,
                    // as a report that cannot be built.
                    None => {
                        results[index] = Some((
                            SendOutcome::Final("sqs_invalid"),
                            AttemptLog {
                                answer: Err("prepared report has no message".to_string()),
                                outcome: UsageReportOutcome::SqsInvalid,
                                duration: Duration::ZERO,
                            },
                        ))
                    }
                }
            }
            let calls = chunks(&messages).into_iter().map(|chunk| async move {
                let texts: Vec<&str> = chunk.iter().map(|(_, text)| text.as_str()).collect();
                (chunk, self.send_chunk(&texts, timeout).await)
            });
            for (chunk, answers) in futures_util::future::join_all(calls).await {
                for ((index, _), answer) in chunk.iter().zip(answers) {
                    results[*index] = Some(answer);
                }
            }
            results.into_iter().flatten().collect()
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
            (
                decide(outcome),
                AttemptLog {
                    answer,
                    outcome,
                    duration: started_at.elapsed(),
                },
            )
        })
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

/// An error name cut to a name's characters: AWS's text, not trusted.
fn sanitized_name(name: &str) -> String {
    sanitized(short_error_name(name))
}

fn sanitized(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        .take(64)
        .collect()
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
#[path = "sqs_sender_batch_tests.rs"]
mod batch_tests;

#[cfg(test)]
#[path = "sqs_sender_redirect_tests.rs"]
mod redirect_tests;

#[cfg(test)]
#[path = "sqs_sender_startup_tests.rs"]
mod startup_tests;
