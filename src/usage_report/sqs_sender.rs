//! The SQS sender: one `SendMessage` of a report to the usage queue, carrying
//! the report as a `cloud-platform-events` envelope and signed with SigV4
//! (service `sqs`). It is selected by `VLLM_PROXY_USAGE_REPORT_TRANSPORT=sqs`
//! and is the only code that builds the envelope: the HTTP sender posts the
//! report's bytes as they are.
//!
//! The sender makes one attempt and classifies its answer for the shared
//! delivery. Only explicit, non-retryable answers end a report: a denied
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
use crate::aws_credentials::{read_bounded, CredentialFailure, CredentialProvider};
use crate::proxy::{classify_usage_request_error, UsageReportOutcome};

const SERVICE: &str = "sqs";
const TARGET: &str = "AmazonSQS.SendMessage";
const CONTENT_TYPE: &str = "application/x-amz-json-1.0";
/// The most of an SQS error body that is read for its `__type`.
const MAX_ERROR_BODY_BYTES: usize = 16 * 1024;

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

/// A client that treats a 3xx as the answer it is. Timeouts stay per call
/// (the attempt's budget); bodies stay bounded by the callers.
pub(crate) fn no_redirect_client() -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .pool_idle_timeout(Duration::from_secs(30))
        .build()
        .expect("a client with default TLS settings builds")
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
                "credentials were not had within the attempt's timeout".to_string(),
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
        let started_at = Instant::now();
        let credentials = match self.credentials_within(timeout).await {
            Ok(credentials) => credentials,
            Err((outcome, error)) => return (outcome, Err(error)),
        };
        let Some(remaining) = timeout
            .checked_sub(started_at.elapsed())
            .filter(|left| !left.is_zero())
        else {
            return (
                UsageReportOutcome::Timeout,
                Err("the attempt's time went on the credentials".to_string()),
            );
        };
        let headers = match sign_request(
            &self.endpoint,
            &self.region,
            &credentials,
            &job.body,
            SystemTime::now(),
        ) {
            Ok(headers) => headers,
            Err(error) => return (UsageReportOutcome::SqsInvalid, Err(error)),
        };

        let mut request = self
            .client
            .post(&self.endpoint)
            .timeout(remaining)
            .header(reqwest::header::CONTENT_TYPE, CONTENT_TYPE)
            .header("x-amz-target", TARGET)
            .body(job.body.clone());
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
                return (outcome, Err(error.to_string()));
            }
        };
        let status = response.status();
        let error_type = if status.is_success() {
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
        (outcome, Ok(status))
    }
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

/// The headers SigV4 adds to a `SendMessage` of `payload` to `endpoint` at
/// `time`, with the two the request carries itself (content type and target)
/// signed.
fn sign_request(
    endpoint: &str,
    region: &str,
    credentials: &aws_credential_types::Credentials,
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
    let headers = [("content-type", CONTENT_TYPE), ("x-amz-target", TARGET)];
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
            // Redirects are never followed: a queue endpoint that redirects is
            // misconfigured, and asking again does not change that.
            _ if status.is_redirection() => UsageReportOutcome::SqsInvalid,
            StatusCode::FORBIDDEN | StatusCode::UNAUTHORIZED => UsageReportOutcome::SqsCredentials,
            _ => UsageReportOutcome::SqsUnknown,
        },
    }
}

#[cfg(test)]
#[path = "sqs_sender_tests.rs"]
mod tests;
