//! The SQS sender against a stand-in SQS endpoint (wiremock), and its
//! classification of SQS answers. The sender is `pub(crate)`, so these tests
//! live beside it rather than under `tests/`.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use aws_credential_types::Credentials;
use bytes::Bytes;
use cloud_platform_events::{parse_message, ParsedMessage, Producer};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusRecorder};
use reqwest::StatusCode;
use serde_json::{json, Value};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
use crate::auth::{AuthPath, IngressRouteKind, RequestSource};
use crate::aws_credentials::test_support::{mount_imds_session, ROLE_CREDENTIALS_PATH};
use crate::proxy::{build_usage_body, complete_usage_body, UsageReporter, UsageType};
use crate::usage_report::{UsageReportDelivery, UsageReportPolicy};
use aws_sigv4::http_request::{sign, SignableBody, SignableRequest, SigningSettings};
use aws_sigv4::sign::v4;

const QUEUE: &str = "https://sqs.us-east-1.amazonaws.com/123456789012/cloud-platform-usage";
const ACCESS_KEY: &str = "AKIDEXAMPLE";
const SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";

/// The sum of the series of `family` whose line carries `label`.
fn counter(recorder: &PrometheusRecorder, family: &str, label: &str) -> f64 {
    recorder
        .handle()
        .render()
        .lines()
        .filter(|line| line.starts_with(family) && line.contains(label))
        .map(|line| line.rsplit(' ').next().unwrap().parse::<f64>().unwrap())
        .sum()
}

/// `2026-10-20T12:00:00Z`, the time every signature here is made at.
fn fixed_time() -> SystemTime {
    SystemTime::from(
        chrono::DateTime::parse_from_rfc3339("2026-10-20T12:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc),
    )
}

/// A sender against `endpoint` (a stand-in for SQS), holding credentials
/// already cached, so no credentials endpoint is involved.
fn sender(endpoint: &str) -> SqsSender {
    SqsSender {
        queue_url: QUEUE.to_string(),
        region: "us-east-1".to_string(),
        endpoint: endpoint.to_string(),
        client: no_redirect_client(),
        credentials: CredentialProvider::seeded(Credentials::new(
            ACCESS_KEY, SECRET_KEY, None, None, "test",
        )),
    }
}

/// How many times the credentials themselves were fetched from `server`.
async fn credential_fetches(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|request| request.url.path() == ROLE_CREDENTIALS_PATH)
        .count()
}

/// A sender whose credentials come from the endpoint `credentials_url`.
fn sender_with_endpoint_credentials(endpoint: &str, credentials_url: &str) -> SqsSender {
    SqsSender {
        queue_url: QUEUE.to_string(),
        region: "us-east-1".to_string(),
        endpoint: endpoint.to_string(),
        client: no_redirect_client(),
        credentials: CredentialProvider::new(credentials_url.to_string()),
    }
}

/// The usage body of a chat completion, as `spawn_usage_report` completes it
/// for a deployment that bills at a discount.
fn chat_body(id: &str) -> Value {
    json!({
        "type": "chat_completion",
        "model": "test-model",
        "input_tokens": 1200,
        "output_tokens": 340,
        "cache_read_tokens": 800,
        "id": id,
        "organization_id": "00000000-0000-0000-0000-000000000001",
        "workspace_id": "00000000-0000-0000-0000-000000000002",
        "api_key_id": "00000000-0000-0000-0000-000000000003",
        "discount_to_user": 0.2,
    })
}

fn job_with(body: &Value, delivery: &Arc<UsageReportDelivery>) -> Job {
    Job {
        reporter: UsageReporter {
            http_client: reqwest::Client::new(),
            cloud_api_url: None,
            model_name: "test-model".to_string(),
            cloud_api_usage_token: Some("usage-secret".to_string()),
            org_id: Some("org-1".to_string()),
            workspace_id: Some("ws-1".to_string()),
            api_key_id: Some("key-1".to_string()),
            discount_to_user: Some(0.2),
            model_label: None,
            request_id: Some("request-1".to_string()),
            request_source: RequestSource {
                auth_path: AuthPath::CloudApiKey,
                ingress_route: IngressRouteKind::Canonical,
            },
            delivery: delivery.clone(),
        },
        body: Bytes::from(serde_json::to_vec(body).unwrap()),
        completed_at: Instant::now(),
        emitted_at: chrono::DateTime::parse_from_rfc3339("2026-10-20T12:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc),
    }
}

/// One attempt as the delivery makes it: the report is prepared at submit,
/// and a refused report is not sent at all.
async fn attempt(sender: &SqsSender, job: &Job) -> (SendOutcome, AttemptLog) {
    let mut job = job.clone();
    match sender.prepare(&mut job) {
        Ok(()) => sender.send(&job, Duration::from_secs(5)).await,
        Err(refused) => (
            SendOutcome::Final(refused.reason),
            AttemptLog {
                answer: Err(refused.error),
                outcome: refused.outcome,
                duration: Duration::ZERO,
            },
        ),
    }
}

async fn sqs_answering(status: u16, body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn sqs_send_is_signed_and_carries_the_http_usage_body() {
    let server = sqs_answering(200, json!({"MessageId": "m-1"})).await;
    for body in [
        chat_body("chatcmpl-1"),
        json!({"type": "image_generation", "model": "test-model", "image_count": 2, "id": "img-1",
               "organization_id": "00000000-0000-0000-0000-000000000001",
               "workspace_id": "00000000-0000-0000-0000-000000000002",
               "api_key_id": "00000000-0000-0000-0000-000000000003"}),
        json!({"type": "privacy_classify", "model": "test-model", "input_tokens": 77, "id": "priv-1",
               "organization_id": "00000000-0000-0000-0000-000000000001",
               "workspace_id": "00000000-0000-0000-0000-000000000002",
               "api_key_id": "00000000-0000-0000-0000-000000000003"}),
    ] {
        let job = job_with(&body, &Arc::new(UsageReportDelivery::default()));
        let (outcome, log) = attempt(&sender(&server.uri()), &job).await;
        assert_eq!(outcome, SendOutcome::Accepted);
        assert_eq!(log.outcome, UsageReportOutcome::Accepted);
        assert_eq!(log.answer.unwrap(), StatusCode::OK);
    }

    // The same bodies as the HTTP sender would post them, one by one.
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 3);
    for (request, body) in requests.iter().zip([
        chat_body("chatcmpl-1"),
        json!({"type": "image_generation", "model": "test-model", "image_count": 2, "id": "img-1",
               "organization_id": "00000000-0000-0000-0000-000000000001",
               "workspace_id": "00000000-0000-0000-0000-000000000002",
               "api_key_id": "00000000-0000-0000-0000-000000000003"}),
        json!({"type": "privacy_classify", "model": "test-model", "input_tokens": 77, "id": "priv-1",
               "organization_id": "00000000-0000-0000-0000-000000000001",
               "workspace_id": "00000000-0000-0000-0000-000000000002",
               "api_key_id": "00000000-0000-0000-0000-000000000003"}),
    ]) {
        let outer: Value = serde_json::from_slice(&request.body).unwrap();
        let ParsedMessage::Usage(envelope) =
            parse_message(outer["MessageBody"].as_str().unwrap().as_bytes()).unwrap()
        else {
            panic!("a usage message");
        };
        assert_eq!(envelope.v, 1);
        assert_eq!(envelope.producer, Producer::OpenrouterGateway);
        assert_eq!(
            serde_json::to_value(&envelope.usage).unwrap(),
            body,
            "the usage body is the one the HTTP sender posts"
        );
    }
}

/// The usage body the proxy builds for each usage type, through its own
/// `build_usage_body` and `complete_usage_body`, is the body the queue carries
/// field for field.
#[test]
fn the_queue_carries_the_body_the_proxy_builds_for_each_usage_type() {
    let delivery = Arc::new(UsageReportDelivery::default());
    let probe = job_with(&json!({}), &delivery);
    let cases = [
        (
            UsageType::ChatCompletion,
            json!({"usage": {"prompt_tokens": 1200, "completion_tokens": 340,
                   "prompt_tokens_details": {"cached_tokens": 800}}}),
        ),
        (UsageType::ImageGeneration, json!({"data": [{}, {}]})),
        (
            UsageType::PrivacyClassify,
            json!({"data": [{"usage": {"input_tokens": 77}}]}),
        ),
        (
            UsageType::Embedding,
            json!({"usage": {"prompt_tokens": 512}}),
        ),
        (UsageType::Rerank, json!({"usage": {"prompt_tokens": 2048}})),
        (UsageType::Score, json!({"usage": {"prompt_tokens": 96}})),
    ];
    for (usage_type, response) in cases {
        let mut body = build_usage_body(&usage_type, &response, "test-model", "id-1")
            .expect("a billable response");
        complete_usage_body(&probe.reporter, &mut body).expect("a JSON object");
        let job = job_with(&body, &delivery);
        let message =
            message_body(&job).unwrap_or_else(|refused| panic!("{}: {body}", refused.error));
        let ParsedMessage::Usage(envelope) = parse_message(message.as_bytes()).unwrap() else {
            panic!("a usage message");
        };
        assert_eq!(serde_json::to_value(&envelope.usage).unwrap(), body);
    }
}

/// A field the usage schema does not carry is refused as a final
/// `sqs_invalid`, and nothing is sent: it is not silently dropped.
#[tokio::test]
async fn a_field_the_usage_schema_does_not_carry_is_refused_not_dropped() {
    let server = sqs_answering(200, json!({"MessageId": "m"})).await;
    let body = chat_body("chatcmpl-9");
    // A top-level unknown key, then a nested one inside an unknown object.
    for (key, value) in [
        ("future_field", json!("value")),
        ("future_detail", json!({"nested": {"deeper": 1}})),
    ] {
        let mut body = body.clone();
        body[key] = value;
        let job = job_with(&body, &Arc::new(UsageReportDelivery::default()));
        assert!(message_body(&job).is_err(), "{key}");
        let (outcome, log) = attempt(&sender(&server.uri()), &job).await;
        assert_eq!(outcome, SendOutcome::Final("sqs_invalid"), "{key}");
        assert_eq!(log.outcome, UsageReportOutcome::SqsInvalid, "{key}");
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn sqs_request_has_the_sqs_target_content_type_and_a_sigv4_authorization() {
    let server = sqs_answering(200, json!({"MessageId": "m-1"})).await;
    let job = job_with(
        &chat_body("chatcmpl-2"),
        &Arc::new(UsageReportDelivery::default()),
    );
    attempt(&sender(&server.uri()), &job).await;

    let requests = server.received_requests().await.unwrap();
    let request = &requests[0];
    assert_eq!(request.headers["x-amz-target"], "AmazonSQS.SendMessage");
    assert_eq!(
        request.headers["content-type"],
        "application/x-amz-json-1.0"
    );
    let authorization = request.headers["authorization"].to_str().unwrap();
    assert!(authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"));
    assert!(authorization.contains("/us-east-1/sqs/aws4_request"));
    assert!(authorization.contains("SignedHeaders="));
    assert!(authorization.contains("x-amz-target"));
    assert!(request.headers.contains_key("x-amz-date"));
}

/// The AWS SigV4 test suite's `post-vanilla` case
/// (`aws-signing-test-suite/v4/post-vanilla` in the aws-sigv4 package, from
/// the CRT suite): fixed keys, time, region and service, an empty body. The
/// signature is the suite's published value. The signing calls are the ones
/// `sign_request` makes.
#[test]
fn sigv4_matches_aws_test_vector() {
    let credentials = Credentials::new(
        "AKIDEXAMPLE",
        "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        None,
        None,
        "test",
    );
    let identity = credentials.into();
    let time = SystemTime::from(
        chrono::DateTime::parse_from_rfc3339("2015-08-30T12:36:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc),
    );
    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region("us-east-1")
        .name("service")
        .time(time)
        .settings(SigningSettings::default())
        .build()
        .unwrap();
    let request = SignableRequest::new(
        "POST",
        "https://example.amazonaws.com/",
        std::iter::empty(),
        SignableBody::Bytes(b""),
    )
    .unwrap();
    let (instructions, _signature) = sign(request, &params.into()).unwrap().into_parts();
    let authorization = instructions
        .headers()
        .find(|(name, _)| *name == "authorization")
        .map(|(_, value)| value)
        .unwrap();
    assert_eq!(
        authorization,
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date, Signature=5da7c1a2acd57cee7505fc6676e4e544621c30862966e37dddb68e92efbe5d6b"
    );
}

/// The signature of a fixed SQS request at a fixed time, with the example keys
/// of the AWS documentation. The expected value was computed independently:
/// Python's `hmac` and `hashlib` over a canonical request written out by hand
/// (POST `/`, host `sqs.us-east-1.amazonaws.com`, signed headers
/// `content-type;host;x-amz-date;x-amz-target`, payload hash of the body
/// below), with the SigV4 key chain for `20261020/us-east-1/sqs`.
#[test]
fn sigv4_signature_matches_the_independent_computation() {
    let credentials = Credentials::new(ACCESS_KEY, SECRET_KEY, None, None, "test");
    let payload = br#"{"QueueUrl":"x"}"#;
    let headers = sign_request(
        "https://sqs.us-east-1.amazonaws.com/",
        "us-east-1",
        &credentials,
        TARGET,
        payload,
        fixed_time(),
    )
    .unwrap();
    let find = |name: &str| {
        headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("missing header {name}: {headers:?}"))
    };
    assert_eq!(find("x-amz-date"), "20261020T120000Z");
    assert_eq!(
        find("authorization"),
        SIGNING_VECTOR_AUTHORIZATION,
        "signed headers: {headers:?}"
    );
}

const SIGNING_VECTOR_AUTHORIZATION: &str = "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261020/us-east-1/sqs/aws4_request, SignedHeaders=content-type;host;x-amz-date;x-amz-target, Signature=79062887be578b0e59a90b2783701217339d4d9dc397f9ef867adaf600bce2f1";

#[tokio::test]
async fn sqs_throttling_is_retried_and_named_so() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(400).set_body_json(
            json!({"__type": "com.amazonaws.sqs#RequestThrottled", "message": "slow down"}),
        ))
        .mount(&server)
        .await;
    let job = job_with(
        &chat_body("chatcmpl-3"),
        &Arc::new(UsageReportDelivery::default()),
    );
    let (outcome, log) = attempt(&sender(&server.uri()), &job).await;
    assert_eq!(outcome, SendOutcome::Retryable("sqs_throttled"));
    assert_eq!(log.outcome, UsageReportOutcome::SqsThrottled);
    assert_eq!(log.answer.unwrap(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn sqs_access_denied_is_retried_and_invalid_parameters_are_final() {
    for (status, error_type, decision, outcome) in [
        (
            403,
            "AccessDeniedException",
            SendOutcome::Retryable("sqs_denied"),
            UsageReportOutcome::SqsDenied,
        ),
        (
            400,
            "InvalidParameterValue",
            SendOutcome::Final("sqs_invalid"),
            UsageReportOutcome::SqsInvalid,
        ),
        (
            400,
            "AWS.SimpleQueueService.NonExistentQueue",
            SendOutcome::Final("sqs_invalid"),
            UsageReportOutcome::SqsInvalid,
        ),
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/"))
            .respond_with(
                ResponseTemplate::new(status)
                    .set_body_json(json!({"__type": format!("com.amazonaws.sqs#{error_type}")})),
            )
            .mount(&server)
            .await;
        let job = job_with(
            &chat_body("chatcmpl-4"),
            &Arc::new(UsageReportDelivery::default()),
        );
        let (sent, log) = attempt(&sender(&server.uri()), &job).await;
        assert_eq!(sent, decision, "{error_type}");
        assert_eq!(log.outcome, outcome, "{error_type}");
    }
}

#[test]
fn sqs_answers_are_classified_by_name_then_status() {
    use StatusCode as S;
    use UsageReportOutcome as O;
    let cases = [
        (S::OK, None, O::Accepted),
        (S::ACCEPTED, None, O::Accepted),
        (S::BAD_REQUEST, Some("ThrottlingException"), O::SqsThrottled),
        (
            S::BAD_REQUEST,
            Some("com.amazonaws.sqs#OverLimit"),
            O::SqsThrottled,
        ),
        (
            S::BAD_REQUEST,
            Some("RequestThrottled:http://internal.amazon.com/coral/x"),
            O::SqsThrottled,
        ),
        (S::TOO_MANY_REQUESTS, None, O::SqsThrottled),
        (S::INTERNAL_SERVER_ERROR, None, O::Sqs5xx),
        (
            S::SERVICE_UNAVAILABLE,
            Some("ServiceUnavailable"),
            O::Sqs5xx,
        ),
        (S::BAD_REQUEST, Some("ExpiredToken"), O::SqsCredentials),
        (S::BAD_REQUEST, Some("RequestExpired"), O::SqsCredentials),
        (
            S::BAD_REQUEST,
            Some("RequestTimeTooSkewed"),
            O::SqsCredentials,
        ),
        (
            S::BAD_REQUEST,
            Some("InvalidClientTokenId"),
            O::SqsCredentials,
        ),
        (
            S::BAD_REQUEST,
            Some("SignatureDoesNotMatch"),
            O::SqsCredentials,
        ),
        (S::FORBIDDEN, None, O::SqsCredentials),
        (S::BAD_REQUEST, Some("AccessDenied"), O::SqsDenied),
        (S::BAD_REQUEST, Some("QueueDoesNotExist"), O::SqsInvalid),
        (S::BAD_REQUEST, Some("MissingParameter"), O::SqsInvalid),
        (S::BAD_REQUEST, Some("SomethingNew"), O::SqsUnknown),
        (S::BAD_REQUEST, None, O::SqsUnknown),
    ];
    for (status, name, expected) in cases {
        assert_eq!(
            classify_sqs_answer(status, name),
            expected,
            "{status} {name:?}"
        );
    }
}

/// Requirement: no attempt exceeds its deadline when the credentials endpoint
/// hangs, and each attempt fetches at most once. Exactly one fetch is not
/// required.
#[tokio::test]
async fn a_hung_credentials_endpoint_bounds_every_attempt() {
    let creds = MockServer::start().await;
    mount_imds_session(&creds).await;
    Mock::given(method("GET"))
        .and(path(ROLE_CREDENTIALS_PATH))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(10)))
        .mount(&creds)
        .await;
    let sqs = sqs_answering(200, json!({"MessageId": "m"})).await;
    let sender = Arc::new(sender_with_endpoint_credentials(&sqs.uri(), &creds.uri()));
    let job = job_with(
        &chat_body("chatcmpl-14"),
        &Arc::new(UsageReportDelivery::default()),
    );
    let started = Instant::now();
    let outcomes = futures_util::future::join_all((0..10).map(|_| {
        let (sender, job) = (sender.clone(), job.clone());
        async move { attempt_with_timeout(&sender, &job, Duration::from_millis(300)).await }
    }))
    .await;
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "{:?}",
        started.elapsed()
    );
    for (outcome, log) in outcomes {
        assert_eq!(outcome, SendOutcome::Retryable("timeout"));
        assert_eq!(log.outcome, UsageReportOutcome::Timeout);
    }
    let fetches = credential_fetches(&creds).await;
    assert!(
        (1..=10).contains(&fetches),
        "{fetches} fetches for 10 attempts"
    );
    assert!(sqs.received_requests().await.unwrap().is_empty());
}

async fn attempt_with_timeout(
    sender: &SqsSender,
    job: &Job,
    timeout: Duration,
) -> (SendOutcome, AttemptLog) {
    let mut job = job.clone();
    sender.prepare(&mut job).expect("a report that prepares");
    sender.send(&job, timeout).await
}

#[tokio::test]
async fn sqs_5xx_is_retried_until_the_attempts_run_out() {
    let server = sqs_answering(500, json!({"__type": "InternalFailure"})).await;
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let delivery = UsageReportDelivery::with_sender(
        UsageReportPolicy {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(1),
            ..Default::default()
        },
        Arc::new(sender(&server.uri())),
    );
    let job = job_with(&chat_body("chatcmpl-5"), &delivery);
    delivery.submit(job);
    let drained = delivery.drain(Duration::from_secs(30)).await;
    assert_eq!(drained.left(), 0, "{drained:?}");
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
    assert_eq!(
        counter(
            &recorder,
            "inference_proxy_usage_reports_dropped_total",
            r#"reason="attempts_exhausted""#
        ),
        1.0
    );
}

#[tokio::test]
async fn an_unreachable_queue_is_a_retryable_connect_error() {
    // Port 1 is never listening (see usage_report_tests).
    let (outcome, log) = attempt(
        &sender("http://127.0.0.1:1"),
        &job_with(
            &chat_body("chatcmpl-6"),
            &Arc::new(UsageReportDelivery::default()),
        ),
    )
    .await;
    assert_eq!(outcome, SendOutcome::Retryable("connect_error"));
    assert_eq!(log.outcome, UsageReportOutcome::ConnectError);
    assert!(log.answer.is_err());
}

#[tokio::test]
async fn endpoint_credentials_are_cached_until_near_expiry_and_sent_as_a_session() {
    let server = MockServer::start().await;
    mount_imds_session(&server).await;
    let in_an_hour = (chrono::Utc::now() + chrono::TimeDelta::hours(1)).to_rfc3339();
    Mock::given(method("GET"))
        .and(path(ROLE_CREDENTIALS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "Code": "Success",
            "AccessKeyId": "ASIAEXAMPLE",
            "SecretAccessKey": SECRET_KEY,
            "Token": "session-token",
            "Expiration": in_an_hour,
        })))
        .expect(1)
        .mount(&server)
        .await;
    let sqs = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .and(header("x-amz-security-token", "session-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"MessageId": "m"})))
        .expect(2)
        .mount(&sqs)
        .await;

    let sender = SqsSender {
        queue_url: QUEUE.to_string(),
        region: "us-east-1".to_string(),
        endpoint: sqs.uri(),
        client: no_redirect_client(),
        credentials: CredentialProvider::new(server.uri()),
    };
    let job = job_with(
        &chat_body("chatcmpl-7"),
        &Arc::new(UsageReportDelivery::default()),
    );
    assert_eq!(attempt(&sender, &job).await.0, SendOutcome::Accepted);
    assert_eq!(attempt(&sender, &job).await.0, SendOutcome::Accepted);
    // One fetch for both sends: the cached credentials were reused.
    assert_eq!(credential_fetches(&server).await, 1);
    let authorization = sqs.received_requests().await.unwrap()[1].headers["authorization"]
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        authorization.contains("Credential=ASIAEXAMPLE/"),
        "{authorization}"
    );
    // Fetched once: `expect(1)` is verified when the server drops.
}

#[tokio::test]
async fn endpoint_credentials_due_for_refresh_are_fetched_again() {
    let server = MockServer::start().await;
    mount_imds_session(&server).await;
    let in_a_minute = (chrono::Utc::now() + chrono::TimeDelta::minutes(1)).to_rfc3339();
    Mock::given(method("GET"))
        .and(path(ROLE_CREDENTIALS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "AccessKeyId": "ASIAEXAMPLE",
            "SecretAccessKey": SECRET_KEY,
            "Token": "session-token",
            "Expiration": in_a_minute,
        })))
        .expect(2)
        .mount(&server)
        .await;
    let sqs = sqs_answering(200, json!({"MessageId": "m"})).await;
    let sender = SqsSender {
        queue_url: QUEUE.to_string(),
        region: "us-east-1".to_string(),
        endpoint: sqs.uri(),
        client: no_redirect_client(),
        credentials: CredentialProvider::new(server.uri()),
    };
    let job = job_with(
        &chat_body("chatcmpl-8"),
        &Arc::new(UsageReportDelivery::default()),
    );
    attempt(&sender, &job).await;
    attempt(&sender, &job).await;
    assert_eq!(credential_fetches(&server).await, 2);
}

#[tokio::test]
async fn shutdown_drain_waits_for_reports_sent_over_sqs() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(100))
                .set_body_json(json!({"MessageId": "m"})),
        )
        .mount(&server)
        .await;
    let delivery = UsageReportDelivery::with_sender(
        UsageReportPolicy {
            max_in_flight: 4,
            ..Default::default()
        },
        Arc::new(sender(&server.uri())),
    );
    for id in ["a", "b"] {
        delivery.submit(job_with(&chat_body(id), &delivery));
    }
    let drained = delivery.drain(Duration::from_secs(10)).await;
    assert_eq!(drained.left(), 0, "{drained:?}");
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

/// The SQS transport is named by a gauge of 1; the default HTTP transport
/// adds no series at all, so its `/metrics` stay as they were.
#[test]
fn the_transport_gauge_names_a_non_default_transport_only() {
    let recorder: PrometheusRecorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let _http = UsageReportDelivery::default();
    assert!(!recorder.handle().render().contains("transport_info"));
    let _sqs = UsageReportDelivery::with_sender(
        UsageReportPolicy::default(),
        Arc::new(sender("http://127.0.0.1:1")),
    );
    let rendered = recorder.handle().render();
    assert!(
        rendered.contains(r#"inference_proxy_usage_report_transport_info{transport="sqs"} 1"#),
        "{rendered}"
    );
}

/// Credentials are fetched again, and a refused credential is not sent again:
/// an `InvalidClientTokenId` is retried with a fresh fetch and then accepted.
#[tokio::test]
async fn refused_credentials_are_forgotten_and_fetched_again() {
    let server = MockServer::start().await;
    mount_imds_session(&server).await;
    let in_an_hour = (chrono::Utc::now() + chrono::TimeDelta::hours(1)).to_rfc3339();
    Mock::given(method("GET"))
        .and(path(ROLE_CREDENTIALS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "AccessKeyId": "ASIAEXAMPLE",
            "SecretAccessKey": SECRET_KEY,
            "Token": "session-token",
            "Expiration": in_an_hour,
        })))
        .expect(2)
        .mount(&server)
        .await;
    let sqs = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_json(json!({"__type": "com.amazonaws.sqs#InvalidClientTokenId"})),
        )
        .up_to_n_times(1)
        .mount(&sqs)
        .await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"MessageId": "m"})))
        .mount(&sqs)
        .await;
    let sender = sender_with_endpoint_credentials(&sqs.uri(), &server.uri());
    let job = job_with(
        &chat_body("chatcmpl-10"),
        &Arc::new(UsageReportDelivery::default()),
    );
    let (first, log) = attempt(&sender, &job).await;
    assert_eq!(first, SendOutcome::Retryable("sqs_credentials"));
    assert_eq!(log.outcome, UsageReportOutcome::SqsCredentials);
    assert_eq!(attempt(&sender, &job).await.0, SendOutcome::Accepted);
    // Two fetches: `expect(2)` is verified when the server drops.
}

/// An endpoint that cannot give credentials ends no report: the attempt is
/// retryable and nothing is sent to the queue without credentials.
#[tokio::test]
async fn credentials_endpoint_failures_are_retried_and_nothing_is_sent() {
    let creds = MockServer::start().await;
    mount_imds_session(&creds).await;
    Mock::given(method("GET"))
        .and(path(ROLE_CREDENTIALS_PATH))
        .respond_with(ResponseTemplate::new(404))
        .mount(&creds)
        .await;
    let sqs = sqs_answering(200, json!({"MessageId": "m"})).await;
    let sender = sender_with_endpoint_credentials(&sqs.uri(), &creds.uri());
    let job = job_with(
        &chat_body("chatcmpl-11"),
        &Arc::new(UsageReportDelivery::default()),
    );
    let (outcome, log) = attempt(&sender, &job).await;
    assert_eq!(outcome, SendOutcome::Retryable("sqs_credentials"));
    assert_eq!(log.outcome, UsageReportOutcome::SqsCredentials);
    assert!(log.answer.is_err());
    assert!(sqs.received_requests().await.unwrap().is_empty());
}

/// One attempt has one budget: the credentials and the send together do not
/// take twice the timeout. The send gets what the credentials left.
#[tokio::test]
async fn one_attempt_has_one_timeout_across_credentials_and_send() {
    let creds = MockServer::start().await;
    mount_imds_session(&creds).await;
    let in_an_hour = (chrono::Utc::now() + chrono::TimeDelta::hours(1)).to_rfc3339();
    Mock::given(method("GET"))
        .and(path(ROLE_CREDENTIALS_PATH))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(250))
                .set_body_json(json!({
                    "AccessKeyId": "ASIAEXAMPLE",
                    "SecretAccessKey": SECRET_KEY,
                    "Token": "session-token",
                    "Expiration": in_an_hour,
                })),
        )
        .mount(&creds)
        .await;
    let sqs = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(250))
                .set_body_json(json!({"MessageId": "m"})),
        )
        .mount(&sqs)
        .await;
    let sender = sender_with_endpoint_credentials(&sqs.uri(), &creds.uri());
    let job = job_with(
        &chat_body("chatcmpl-12"),
        &Arc::new(UsageReportDelivery::default()),
    );
    let mut prepared = job.clone();
    sender.prepare(&mut prepared).unwrap();
    let started = Instant::now();
    let (outcome, log) = sender.send(&prepared, Duration::from_millis(400)).await;
    assert!(
        started.elapsed() < Duration::from_millis(600),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(outcome, SendOutcome::Retryable("timeout"));
    assert_eq!(log.outcome, UsageReportOutcome::Timeout);
}

/// A report the queue cannot carry is refused at submit: it is never sent,
/// it is counted as `sqs_oversize`, and the log names its length, not its content.
#[tokio::test]
async fn an_oversize_report_is_refused_at_submit_and_counted() {
    let sqs = sqs_answering(200, json!({"MessageId": "m"})).await;
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let delivery = UsageReportDelivery::with_sender(
        UsageReportPolicy::default(),
        Arc::new(sender(&sqs.uri())),
    );
    let mut body = chat_body("chatcmpl-13");
    body["model"] = json!("oversize-model-name-".repeat(20_000));
    let job = job_with(&body, &delivery);
    let mut prepared = job.clone();
    let refused = sender(&sqs.uri()).prepare(&mut prepared).unwrap_err();
    assert_eq!(refused.reason, "sqs_oversize");
    assert_eq!(refused.outcome, UsageReportOutcome::SqsInvalid);
    assert!(refused.error.contains("bytes"), "{}", refused.error);
    assert!(
        !refused.error.contains("oversize-model"),
        "{}",
        refused.error
    );

    delivery.submit(job);
    let drained = delivery.drain(Duration::from_secs(10)).await;
    assert_eq!(drained.left(), 0, "{drained:?}");
    assert!(sqs.received_requests().await.unwrap().is_empty());
    assert_eq!(
        counter(
            &recorder,
            "inference_proxy_usage_reports_dropped_total",
            r#"reason="sqs_oversize""#
        ),
        1.0
    );
}

/// AccessDenied can follow a fresh session that the service has not yet
/// accepted: the next attempt fetches new credentials, and a report accepted
/// then is billed.
#[tokio::test]
async fn a_denied_fresh_session_is_retried_with_new_credentials() {
    let creds = MockServer::start().await;
    mount_imds_session(&creds).await;
    let in_an_hour = (chrono::Utc::now() + chrono::TimeDelta::hours(1)).to_rfc3339();
    Mock::given(method("GET"))
        .and(path(ROLE_CREDENTIALS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "AccessKeyId": "ASIAEXAMPLE",
            "SecretAccessKey": SECRET_KEY,
            "Token": "session-token",
            "Expiration": in_an_hour,
        })))
        .expect(2)
        .mount(&creds)
        .await;
    let sqs = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(
            ResponseTemplate::new(403)
                .set_body_json(json!({"__type": "com.amazonaws.sqs#AccessDeniedException"})),
        )
        .up_to_n_times(1)
        .mount(&sqs)
        .await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"MessageId": "m"})))
        .mount(&sqs)
        .await;
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let delivery = UsageReportDelivery::with_sender(
        UsageReportPolicy {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(1),
            ..Default::default()
        },
        Arc::new(sender_with_endpoint_credentials(&sqs.uri(), &creds.uri())),
    );
    delivery.submit(job_with(&chat_body("chatcmpl-15"), &delivery));
    let drained = delivery.drain(Duration::from_secs(10)).await;
    assert_eq!(drained.left(), 0, "{drained:?}");
    assert_eq!(sqs.received_requests().await.unwrap().len(), 2);
    assert_eq!(credential_fetches(&creds).await, 2);
    assert_eq!(
        counter(&recorder, "inference_proxy_usage_reports_dropped_total", ""),
        0.0
    );
}

/// Still denied once the attempts run out: the report is dropped, counted as
/// `attempts_exhausted`, and the final denial is what the attempt metric shows.
#[tokio::test]
async fn a_denial_that_persists_is_dropped_once_the_attempts_run_out() {
    let creds = MockServer::start().await;
    mount_imds_session(&creds).await;
    Mock::given(method("GET"))
        .and(path(ROLE_CREDENTIALS_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "AccessKeyId": "ASIAEXAMPLE",
            "SecretAccessKey": SECRET_KEY,
            "Token": "session-token",
        })))
        .mount(&creds)
        .await;
    let sqs = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(
            ResponseTemplate::new(403)
                .set_body_json(json!({"__type": "com.amazonaws.sqs#AccessDeniedException"})),
        )
        .expect(2)
        .mount(&sqs)
        .await;
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let delivery = UsageReportDelivery::with_sender(
        UsageReportPolicy {
            max_attempts: 2,
            initial_backoff: Duration::from_millis(1),
            ..Default::default()
        },
        Arc::new(sender_with_endpoint_credentials(&sqs.uri(), &creds.uri())),
    );
    delivery.submit(job_with(&chat_body("chatcmpl-16"), &delivery));
    let drained = delivery.drain(Duration::from_secs(10)).await;
    assert_eq!(drained.left(), 0, "{drained:?}");
    assert_eq!(sqs.received_requests().await.unwrap().len(), 2);
    assert_eq!(
        counter(
            &recorder,
            "inference_proxy_usage_reports_dropped_total",
            r#"reason="attempts_exhausted""#
        ),
        1.0
    );
}

/// A delivery whose startup check tries quickly: the policy's attempt timeout
/// and backoff are what bound it.
fn checking(sender: SqsSender) -> Arc<UsageReportDelivery> {
    UsageReportDelivery::with_sender(
        UsageReportPolicy {
            attempt_timeout: Duration::from_secs(1),
            initial_backoff: Duration::from_millis(1),
            ..Default::default()
        },
        Arc::new(sender),
    )
}

/// The startup check is a signed `GetQueueAttributes` for `QueueArn` on the
/// usage queue, made once when it succeeds.
#[tokio::test]
async fn startup_check_gets_the_queue_arn_once() {
    let sqs = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .and(header("x-amz-target", "AmazonSQS.GetQueueAttributes"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"Attributes": {}})))
        .expect(1)
        .mount(&sqs)
        .await;
    let delivery = checking(sender(&sqs.uri()));
    delivery
        .check_startup(&reqwest::Client::new())
        .await
        .unwrap();
    let requests = sqs.received_requests().await.unwrap();
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(
        body,
        json!({"QueueUrl": QUEUE, "AttributeNames": ["QueueArn"]})
    );
    assert!(requests[0].headers.contains_key("authorization"));
}

/// A queue that does not exist refuses startup at once, naming the category.
#[tokio::test]
async fn startup_check_refuses_a_missing_queue() {
    let sqs = sqs_answering(
        400,
        json!({"__type": "com.amazonaws.sqs#QueueDoesNotExist"}),
    )
    .await;
    let delivery = checking(sender(&sqs.uri()));
    let error = delivery
        .check_startup(&reqwest::Client::new())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("sqs_invalid"), "{error}");
    assert_eq!(sqs.received_requests().await.unwrap().len(), 1);
}

/// Credentials that cannot be had are tried a few times, then startup is refused.
#[tokio::test]
async fn startup_check_refuses_when_credentials_fail() {
    let creds = MockServer::start().await;
    mount_imds_session(&creds).await;
    Mock::given(method("GET"))
        .and(path(ROLE_CREDENTIALS_PATH))
        .respond_with(ResponseTemplate::new(500))
        .expect(4)
        .mount(&creds)
        .await;
    let sqs = sqs_answering(200, json!({})).await;
    let delivery = checking(sender_with_endpoint_credentials(&sqs.uri(), &creds.uri()));
    let error = delivery
        .check_startup(&reqwest::Client::new())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("sqs_credentials"), "{error}");
    assert!(sqs.received_requests().await.unwrap().is_empty());
}

/// A failure that passes is retried: the second try succeeds.
#[tokio::test]
async fn startup_check_retries_a_transient_failure() {
    let sqs = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&sqs)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&sqs)
        .await;
    checking(sender(&sqs.uri()))
        .check_startup(&reqwest::Client::new())
        .await
        .unwrap();
    assert_eq!(sqs.received_requests().await.unwrap().len(), 2);
}

/// Over HTTP the check does nothing: no request is made.
#[tokio::test]
async fn startup_check_makes_no_call_over_http() {
    let server = MockServer::start().await;
    let client = reqwest::Client::new();
    UsageReportDelivery::default()
        .check_startup(&client)
        .await
        .unwrap();
    assert!(server.received_requests().await.unwrap().is_empty());
}

/// A redirect from the queue endpoint is not followed: the signed request goes
/// to the configured host only, and the answer is final (a redirecting
/// endpoint is misconfigured; asking again changes nothing).
#[tokio::test]
async fn a_redirect_from_the_queue_endpoint_is_not_followed() {
    let elsewhere = MockServer::start().await;
    Mock::given(wiremock::matchers::any())
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"MessageId": "m"})))
        .mount(&elsewhere)
        .await;
    let sqs = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(302).insert_header("location", elsewhere.uri()))
        .mount(&sqs)
        .await;
    let sender = sender(&sqs.uri());
    let job = job_with(
        &chat_body("chatcmpl-redirect"),
        &Arc::new(UsageReportDelivery::default()),
    );
    let (outcome, log) = attempt(&sender, &job).await;
    assert_eq!(outcome, SendOutcome::Final("sqs_invalid"));
    assert_eq!(log.outcome, UsageReportOutcome::SqsInvalid);
    assert_eq!(sqs.received_requests().await.unwrap().len(), 1);
    assert!(elsewhere.received_requests().await.unwrap().is_empty());
}

/// A redirect from the credentials endpoint is not followed either, at the
/// token step and at the credentials step (which carries the session token):
/// nothing reaches the other host, and the attempt fails as a credentials
/// failure, which is retried like any other.
#[tokio::test]
async fn a_redirect_from_the_credentials_endpoint_is_not_followed() {
    use wiremock::matchers::any;
    let elsewhere = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_string("stolen"))
        .mount(&elsewhere)
        .await;
    let sqs = sqs_answering(200, json!({"MessageId": "m"})).await;

    // Redirected at the token step.
    let creds = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(302).insert_header("location", elsewhere.uri()))
        .mount(&creds)
        .await;
    let job = job_with(
        &chat_body("chatcmpl-redirect-creds"),
        &Arc::new(UsageReportDelivery::default()),
    );
    let sender_a = sender_with_endpoint_credentials(&sqs.uri(), &creds.uri());
    let (outcome, log) = attempt(&sender_a, &job).await;
    assert_eq!(outcome, SendOutcome::Retryable("sqs_credentials"));
    assert_eq!(log.outcome, UsageReportOutcome::SqsCredentials);

    // Redirected at the credentials step, which carries the token header.
    let creds = MockServer::start().await;
    mount_imds_session(&creds).await;
    Mock::given(method("GET"))
        .and(path(ROLE_CREDENTIALS_PATH))
        .respond_with(ResponseTemplate::new(302).insert_header("location", elsewhere.uri()))
        .mount(&creds)
        .await;
    let sender_b = sender_with_endpoint_credentials(&sqs.uri(), &creds.uri());
    let (outcome, _) = attempt(&sender_b, &job).await;
    assert_eq!(outcome, SendOutcome::Retryable("sqs_credentials"));

    assert!(elsewhere.received_requests().await.unwrap().is_empty());
    assert!(sqs.received_requests().await.unwrap().is_empty());
}

fn sqs_transport() -> crate::usage_report::UsageReportTransport {
    crate::usage_report::UsageReportTransport::Sqs(crate::usage_report::SqsSettings {
        queue_url: QUEUE.to_string(),
        region: "us-east-1".to_string(),
        endpoint: None,
        credentials_url: "http://127.0.0.1:9911".to_string(),
    })
}

const TRANSPORT_INFO: &str = "inference_proxy_usage_report_transport_info";

/// A queue that passes its check keeps the process on SQS.
#[tokio::test]
async fn a_passing_queue_check_keeps_the_sqs_transport() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let sqs = sqs_answering(200, json!({})).await;
    let resolved = checking(sender(&sqs.uri()))
        .resolve_after_check(sqs_transport(), true)
        .await
        .unwrap();
    assert_eq!(resolved, sqs_transport());
    assert_eq!(
        counter(&recorder, TRANSPORT_INFO, r#"transport="http""#),
        0.0
    );
}

/// A failing queue check with HTTP fully configured falls back to HTTP, and
/// the active transport is exposed for alerting.
#[tokio::test]
async fn a_failing_queue_check_falls_back_to_http_when_it_is_configured() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let sqs = sqs_answering(
        400,
        json!({"__type": "com.amazonaws.sqs#QueueDoesNotExist"}),
    )
    .await;
    let resolved = checking(sender(&sqs.uri()))
        .resolve_after_check(sqs_transport(), true)
        .await
        .unwrap();
    assert_eq!(resolved, crate::usage_report::UsageReportTransport::Http);
    assert_eq!(
        counter(&recorder, TRANSPORT_INFO, r#"transport="http""#),
        1.0
    );
    assert_eq!(
        counter(&recorder, TRANSPORT_INFO, r#"transport="sqs""#),
        0.0
    );
}

/// A failing queue check without a configured HTTP transport refuses startup:
/// there is never a process with no billing path.
#[tokio::test]
async fn a_failing_queue_check_without_http_configured_refuses_startup() {
    let sqs = sqs_answering(
        400,
        json!({"__type": "com.amazonaws.sqs#QueueDoesNotExist"}),
    )
    .await;
    let error = checking(sender(&sqs.uri()))
        .resolve_after_check(sqs_transport(), false)
        .await
        .unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("sqs_invalid"), "{message}");
    assert!(message.contains("billing path"), "{message}");
}
