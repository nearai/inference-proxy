//! The startup check of the usage queue and the choice between SQS and the
//! HTTP fallback it feeds. Beside `sqs_sender_tests.rs`, which uses its helpers.

use std::sync::Arc;
use std::time::Duration;

use metrics_exporter_prometheus::PrometheusBuilder;
use serde_json::{json, Value};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::tests::{counter, sender, sender_with_endpoint_credentials, sqs_answering, QUEUE};
use super::*;
use crate::aws_credentials::test_support::{mount_imds_session, ROLE_CREDENTIALS_PATH};
use crate::usage_report::{UsageReportDelivery, UsageReportPolicy};

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
    delivery.check_startup().await.unwrap();
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
    let error = delivery.check_startup().await.unwrap_err().to_string();
    assert!(error.contains("sqs_invalid"), "{error}");
    // The error type AWS names is in the text, for the operator reading the log.
    assert!(error.contains("QueueDoesNotExist"), "{error}");
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
    let error = delivery.check_startup().await.unwrap_err().to_string();
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
    checking(sender(&sqs.uri())).check_startup().await.unwrap();
    assert_eq!(sqs.received_requests().await.unwrap().len(), 2);
}

/// Over HTTP the check does nothing: no request is made.
#[tokio::test]
async fn startup_check_makes_no_call_over_http() {
    let server = MockServer::start().await;
    UsageReportDelivery::default()
        .check_startup()
        .await
        .unwrap();
    assert!(server.received_requests().await.unwrap().is_empty());
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
    // The gauges are emitted once the process's recorder is installed.
    crate::usage_report::record_fallback_to_http();
    let exposition = recorder.handle().render();
    assert!(
        exposition.contains(r#"inference_proxy_usage_report_transport_info{transport="http"} 1"#),
        "{exposition}"
    );
    assert!(
        exposition.contains(r#"inference_proxy_usage_report_transport_info{transport="sqs"} 0"#),
        "{exposition}"
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
