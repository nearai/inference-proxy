//! Redirects from the queue endpoint and from the credentials endpoint: never
//! followed, and final or a credentials failure, whatever their body says.
//! Beside `sqs_sender_tests.rs`, which uses its helpers.

use std::sync::Arc;

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::tests::{
    attempt, chat_body, job_with, sender, sender_with_endpoint_credentials, sqs_taking, TakesEvery,
};
use super::*;
use crate::aws_credentials::test_support::{mount_imds_session, ROLE_CREDENTIALS_PATH};
use crate::usage_report::UsageReportDelivery;

/// A redirect whose body (or header) names an AWS error type the sender
/// otherwise acts on is still the final `sqs_invalid`: the status decides
/// first, so a throttle, a credentials error or a denial named in a redirect
/// is neither retried nor treated as a credentials failure.
#[tokio::test]
async fn a_redirect_is_final_whatever_error_type_it_names() {
    for status in [301, 302, 307, 308] {
        for error_type in ["ThrottlingException", "ExpiredToken", "AccessDenied"] {
            for in_header in [false, true] {
                let sqs = MockServer::start().await;
                let answer = ResponseTemplate::new(status).insert_header("location", sqs.uri());
                let answer = if in_header {
                    answer.insert_header("x-amzn-errortype", error_type)
                } else {
                    answer.set_body_json(json!({"__type": error_type}))
                };
                Mock::given(method("POST"))
                    .respond_with(answer)
                    .mount(&sqs)
                    .await;
                let sender = sender(&sqs.uri());
                let job = job_with(
                    &chat_body("chatcmpl-redirect-type"),
                    &Arc::new(UsageReportDelivery::default()),
                );
                let (outcome, log) = attempt(&sender, &job).await;
                let case = format!("{status} {error_type} header={in_header}");
                assert_eq!(outcome, SendOutcome::Final("sqs_invalid"), "{case}");
                assert_eq!(log.outcome, UsageReportOutcome::SqsInvalid, "{case}");
                // Not followed (the redirect points back at the same server),
                // and sent once.
                assert_eq!(sqs.received_requests().await.unwrap().len(), 1, "{case}");
            }
        }
    }
}

/// A redirect from the queue endpoint is not followed: the signed request goes
/// to the configured host only, and the answer is final (a redirecting
/// endpoint is misconfigured; asking again changes nothing).
#[tokio::test]
async fn a_redirect_from_the_queue_endpoint_is_not_followed() {
    let elsewhere = MockServer::start().await;
    Mock::given(wiremock::matchers::any())
        .respond_with(TakesEvery::now())
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
    let sqs = sqs_taking().await;

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

/// The classification alone: a redirect status wins over any error type, and
/// over none.
#[test]
fn a_redirect_status_is_classified_before_its_error_type() {
    for status in [301, 302, 303, 307, 308] {
        let status = StatusCode::from_u16(status).unwrap();
        for name in [None, Some("ThrottlingException"), Some("ExpiredToken")] {
            assert_eq!(
                classify_sqs_answer(status, name),
                UsageReportOutcome::SqsInvalid,
                "{status} {name:?}"
            );
        }
    }
}
