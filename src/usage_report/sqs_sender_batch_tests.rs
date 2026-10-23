//! `SendMessageBatch`: what the sender puts in a call, how it maps the answer
//! onto the reports of the call, and how a place of the queue fills one.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use cloud_platform_events::{parse_message, ParsedMessage};
use metrics_exporter_prometheus::PrometheusBuilder;
use reqwest::StatusCode;
use serde_json::{json, Value};
use wiremock::matchers::{header, method};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use super::tests::{chat_body, counter, job_with, sender, sqs_answering, QUEUE};
use super::*;
use crate::usage_report::{UsageReportDelivery, UsageReportPolicy};

const TIMEOUT: Duration = Duration::from_secs(5);

fn delivery() -> Arc<UsageReportDelivery> {
    Arc::new(UsageReportDelivery::default())
}

/// `count` prepared reports, ids `r-0`, `r-1`, ...
fn prepared(sender: &SqsSender, count: usize) -> Vec<Job> {
    (0..count)
        .map(|i| {
            let mut job = job_with(&chat_body(&format!("r-{i}")), &delivery());
            sender.prepare(&mut job).unwrap();
            job
        })
        .collect()
}

async fn send_batch(sender: &SqsSender, jobs: &[Job]) -> Vec<(SendOutcome, AttemptLog)> {
    let refs: Vec<&Job> = jobs.iter().collect();
    sender.send_batch(&refs, TIMEOUT).await
}

/// The completion id of the report in a message.
fn report_id(message: &str) -> String {
    let envelope: Value = serde_json::from_str(message).unwrap();
    envelope["usage"]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("no id in {envelope}"))
        .to_string()
}

fn json_body(request: &Request) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

fn target(request: &Request) -> &str {
    request
        .headers
        .get("x-amz-target")
        .unwrap()
        .to_str()
        .unwrap()
}

/// The `MessageBody` a single `SendMessage` of the job carries.
fn single_message(job: &Job) -> String {
    json_body_of(&job.body)["MessageBody"]
        .as_str()
        .unwrap()
        .to_string()
}

fn json_body_of(bytes: &Bytes) -> Value {
    serde_json::from_slice(bytes).unwrap()
}

fn batch_answer(successful: &[usize], failed: Value) -> Value {
    json!({
        "Successful": successful
            .iter()
            .map(|id| json!({"Id": id.to_string(), "MessageId": format!("m-{id}"), "MD5OfMessageBody": "x"}))
            .collect::<Vec<_>>(),
        "Failed": failed,
    })
}

#[tokio::test]
async fn a_batch_of_one_is_a_send_message_batch_with_the_message_of_a_single_send() {
    let server = sqs_answering(200, batch_answer(&[0], json!([]))).await;
    let sender = sender(&server.uri());
    let jobs = prepared(&sender, 1);
    let results = send_batch(&sender, &jobs).await;

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, SendOutcome::Accepted);
    assert_eq!(results[0].1.outcome, UsageReportOutcome::Accepted);
    assert_eq!(results[0].1.answer.as_ref().unwrap(), &StatusCode::OK);

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(target(&requests[0]), "AmazonSQS.SendMessageBatch");
    assert_eq!(
        requests[0].headers.get("content-type").unwrap(),
        "application/x-amz-json-1.0"
    );
    assert!(requests[0]
        .headers
        .get("authorization")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("/us-east-1/sqs/aws4_request"));
    let body = json_body(&requests[0]);
    assert_eq!(body["QueueUrl"], QUEUE);
    assert_eq!(body["Entries"].as_array().unwrap().len(), 1);
    assert_eq!(body["Entries"][0]["Id"], "0");
    assert_eq!(body["Entries"][0]["MessageBody"], single_message(&jobs[0]));
}

#[tokio::test]
async fn entries_have_distinct_ids_and_the_exact_message_of_a_single_send() {
    let server = sqs_answering(200, batch_answer(&(0..10).collect::<Vec<_>>(), json!([]))).await;
    let sender = sender(&server.uri());
    let jobs = prepared(&sender, 10);
    let results = send_batch(&sender, &jobs).await;
    assert!(results
        .iter()
        .all(|(sent, _)| *sent == SendOutcome::Accepted));

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body = json_body(&requests[0]);
    let entries = body["Entries"].as_array().unwrap();
    assert_eq!(entries.len(), 10);
    let ids: std::collections::HashSet<_> =
        entries.iter().map(|e| e["Id"].as_str().unwrap()).collect();
    assert_eq!(ids.len(), 10);
    for (entry, job) in entries.iter().zip(&jobs) {
        assert_eq!(entry["MessageBody"], single_message(job));
        // Still a usage envelope the consumer parses.
        let message = entry["MessageBody"].as_str().unwrap();
        assert!(matches!(
            parse_message(message.as_bytes()),
            Ok(ParsedMessage::Usage(_))
        ));
    }
}

#[tokio::test]
async fn each_entry_of_a_partial_failure_gets_its_own_outcome() {
    let failed = json!([
        {"Id": "1", "SenderFault": true, "Code": "InvalidParameterValue", "Message": "bad"},
        {"Id": "2", "SenderFault": false, "Code": "InternalError", "Message": "oops"},
        {"Id": "4", "SenderFault": true, "Code": "RequestThrottled"},
        {"Id": "5", "SenderFault": true, "Code": "SomethingNew"},
        {"Id": "6", "SenderFault": false, "Code": "InvalidParameterValue"},
    ]);
    let server = sqs_answering(200, batch_answer(&[0, 3, 7], failed)).await;
    let sender = sender(&server.uri());
    let jobs = prepared(&sender, 8);
    let results = send_batch(&sender, &jobs).await;
    assert_eq!(results.len(), 8);

    let decisions: Vec<_> = results.iter().map(|(sent, _)| *sent).collect();
    assert_eq!(
        decisions,
        vec![
            SendOutcome::Accepted,
            // A sender fault of a code the single send ends the report for.
            SendOutcome::Final("sqs_invalid"),
            // Not the sender's fault.
            SendOutcome::Retryable("sqs_5xx"),
            SendOutcome::Accepted,
            // Classified by code, as a single send's answer would be.
            SendOutcome::Retryable("sqs_throttled"),
            // A sender fault no one classified: as a single send's 400.
            SendOutcome::Retryable("sqs_unknown"),
            // Retryable whatever the code says.
            SendOutcome::Retryable("sqs_5xx"),
            SendOutcome::Accepted,
        ]
    );
    let error = results[1].1.answer.as_ref().unwrap_err();
    assert!(error.contains("InvalidParameterValue"), "{error}");
    assert_eq!(results[1].1.outcome, UsageReportOutcome::SqsInvalid);
    assert_eq!(results[0].1.outcome, UsageReportOutcome::Accepted);
}

#[tokio::test]
async fn a_call_that_fails_as_a_whole_gives_every_report_the_single_send_outcome() {
    for (status, body) in [
        (500, json!({"__type": "com.amazonaws.sqs#InternalError"})),
        (503, json!({})),
        (400, json!({"__type": "com.amazonaws.sqs#RequestThrottled"})),
        (429, json!({})),
        (
            403,
            json!({"__type": "com.amazonaws.sqs#AccessDeniedException"}),
        ),
        (403, json!({"__type": "com.amazonaws.sqs#ExpiredToken"})),
        (
            400,
            json!({"__type": "com.amazonaws.sqs#InvalidParameterValue"}),
        ),
        (
            400,
            json!({"__type": "com.amazonaws.sqs#QueueDoesNotExist"}),
        ),
    ] {
        let server = sqs_answering(status, body.clone()).await;
        // A sender of its own for each call: a refused credential is dropped.
        let single_sender = sender(&server.uri());
        let jobs = prepared(&single_sender, 3);
        let single = single_sender.send(&jobs[0], TIMEOUT).await;
        let results = send_batch(&sender(&server.uri()), &jobs).await;
        assert_eq!(results.len(), 3);
        for (sent, log) in &results {
            assert_eq!(*sent, single.0, "{status} {body}");
            assert_eq!(log.outcome, single.1.outcome, "{status} {body}");
            assert_eq!(
                log.answer.as_ref().unwrap(),
                single.1.answer.as_ref().unwrap()
            );
        }
    }
}

#[tokio::test]
async fn an_unreachable_queue_is_retryable_for_every_report() {
    let sender = sender("http://127.0.0.1:1/");
    let jobs = prepared(&sender, 2);
    for (sent, log) in send_batch(&sender, &jobs).await {
        assert_eq!(sent, SendOutcome::Retryable("connect_error"));
        assert_eq!(log.outcome, UsageReportOutcome::ConnectError);
    }
}

#[tokio::test]
async fn an_answer_that_does_not_say_what_became_of_a_report_gets_it_sent_again() {
    // Unreadable.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;
    let sender_ = sender(&server.uri());
    let jobs = prepared(&sender_, 2);
    for (sent, log) in send_batch(&sender_, &jobs).await {
        assert_eq!(sent, SendOutcome::Retryable("sqs_unknown"));
        assert_eq!(log.outcome, UsageReportOutcome::SqsUnknown);
    }

    // Names only one of the two.
    let server = sqs_answering(200, batch_answer(&[1], json!([]))).await;
    let sender_ = sender(&server.uri());
    let results = send_batch(&sender_, &jobs).await;
    assert_eq!(results[0].0, SendOutcome::Retryable("sqs_unknown"));
    assert_eq!(results[1].0, SendOutcome::Accepted);
}

#[tokio::test]
async fn reports_over_the_byte_limit_of_a_call_are_split_into_several() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(|request: &Request| {
            let entries = json_body(request)["Entries"].as_array().unwrap().len();
            ResponseTemplate::new(200)
                .set_body_json(batch_answer(&(0..entries).collect::<Vec<_>>(), json!([])))
        })
        .mount(&server)
        .await;
    let sender = sender(&server.uri());
    let mut jobs = prepared(&sender, 3);
    // Three messages of 100 KB: two fit in 256 KiB, the third does not.
    for job in &mut jobs {
        let filler = "x".repeat(100_000);
        job.body = Bytes::from(
            serde_json::to_vec(&json!({"QueueUrl": QUEUE, "MessageBody": filler})).unwrap(),
        );
    }
    let results = send_batch(&sender, &jobs).await;
    assert!(results
        .iter()
        .all(|(sent, _)| *sent == SendOutcome::Accepted));
    let mut sizes: Vec<_> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|request| json_body(request)["Entries"].as_array().unwrap().len())
        .collect();
    sizes.sort();
    assert_eq!(sizes, vec![1, 2]);
}

#[test]
fn chunks_hold_at_most_ten_messages_and_the_byte_limit_in_order() {
    let messages: Vec<(usize, String)> = (0..25).map(|i| (i, "m".to_string())).collect();
    let sizes: Vec<_> = chunks(&messages).iter().map(|c| c.len()).collect();
    assert_eq!(sizes, vec![10, 10, 5]);
    assert!(chunks(&[]).is_empty());

    let big = "x".repeat(MAX_BATCH_BYTES);
    let messages = vec![(0, "a".to_string()), (1, "a".to_string()), (2, big)];
    let sizes: Vec<_> = chunks(&messages).iter().map(|c| c.len()).collect();
    assert_eq!(sizes, vec![2, 1]);
}

#[test]
fn only_the_sqs_sender_batches() {
    assert_eq!(sender("http://localhost/").max_batch(), 10);
    assert_eq!(crate::usage_report::http_sender::HttpSender.max_batch(), 1);
}

fn delivery_for(sender: SqsSender, max_in_flight: usize) -> Arc<UsageReportDelivery> {
    UsageReportDelivery::with_sender(
        UsageReportPolicy {
            attempt_timeout: TIMEOUT,
            max_attempts: 3,
            initial_backoff: Duration::from_millis(1),
            deadline: None,
            max_in_flight,
            max_queued: 1000,
            shutdown_drain: Duration::from_secs(10),
        },
        Arc::new(sender),
    )
}

/// A stand-in that takes every message of every call.
async fn queue_taking_everything() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("x-amz-target", "AmazonSQS.SendMessageBatch"))
        .respond_with(|request: &Request| {
            let entries = json_body(request)["Entries"].as_array().unwrap().len();
            ResponseTemplate::new(200)
                .set_body_json(batch_answer(&(0..entries).collect::<Vec<_>>(), json!([])))
        })
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(header("x-amz-target", "AmazonSQS.SendMessage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"MessageId": "m"})))
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn a_backlog_is_sent_ten_to_a_call_and_never_more() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let server = queue_taking_everything().await;
    let delivery = delivery_for(sender(&server.uri()), 1);
    // Nothing runs until the test awaits: one report takes the place and the
    // other 24 wait behind it.
    for i in 0..25 {
        delivery.submit(job_with(&chat_body(&format!("b-{i}")), &delivery));
    }
    assert_eq!(delivery.pending(), (24, 1));
    let drained = delivery.drain(Duration::from_secs(10)).await;
    assert_eq!(drained.left(), 0);

    let requests = server.received_requests().await.unwrap();
    assert!(requests
        .iter()
        .all(|request| target(request) == "AmazonSQS.SendMessageBatch"));
    let sizes: Vec<_> = requests
        .iter()
        .map(|request| json_body(request)["Entries"].as_array().unwrap().len())
        .collect();
    assert_eq!(sizes, vec![1, 10, 10, 4]);
    // Every report was sent once, in order.
    let mut ids = Vec::new();
    for request in &requests {
        for entry in json_body(request)["Entries"].as_array().unwrap() {
            let message = entry["MessageBody"].as_str().unwrap();
            ids.push(report_id(message));
        }
    }
    let expected: Vec<_> = (0..25).map(|i| format!("b-{i}")).collect();
    assert_eq!(ids, expected);

    assert_eq!(
        counter(
            &recorder,
            "inference_proxy_usage_report_attempts_total",
            "accepted"
        ),
        25.0
    );
    let metrics = recorder.handle().render();
    assert!(
        metrics.contains("inference_proxy_usage_report_batch_size_count 4"),
        "{metrics}"
    );
    assert_eq!(delivery.pending(), (0, 0));
}

#[tokio::test]
async fn a_lone_report_is_sent_at_once_without_waiting_for_company() {
    let server = queue_taking_everything().await;
    let delivery = delivery_for(sender(&server.uri()), 16);
    let started = Instant::now();
    delivery.submit(job_with(&chat_body("lone"), &delivery));
    let drained = delivery.drain(Duration::from_secs(10)).await;
    assert_eq!(drained.left(), 0);
    assert!(started.elapsed() < Duration::from_secs(1));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        json_body(&requests[0])["Entries"].as_array().unwrap().len(),
        1
    );
}

#[tokio::test]
async fn a_report_a_batch_did_not_take_is_retried_alone_and_the_rest_are_not_sent_again() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let server = MockServer::start().await;
    // The first call of the first batch refuses its second entry; the rest
    // is taken.
    Mock::given(method("POST"))
        .and(header("x-amz-target", "AmazonSQS.SendMessageBatch"))
        .respond_with(|request: &Request| {
            let entries = json_body(request)["Entries"].as_array().unwrap().len();
            let taken: Vec<usize> = (0..entries).filter(|id| *id != 1).collect();
            ResponseTemplate::new(200).set_body_json(batch_answer(
                &taken,
                json!([{"Id": "1", "SenderFault": false, "Code": "InternalError"}]),
            ))
        })
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(header("x-amz-target", "AmazonSQS.SendMessage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"MessageId": "m"})))
        .mount(&server)
        .await;
    let delivery = delivery_for(sender(&server.uri()), 1);
    for i in 0..3 {
        delivery.submit(job_with(&chat_body(&format!("p-{i}")), &delivery));
    }
    let drained = delivery.drain(Duration::from_secs(10)).await;
    assert_eq!(drained.left(), 0);

    // Batch [p-0], batch [p-1, p-2] whose entry 1 (p-2) is refused, then p-2
    // alone as a SendMessage.
    let requests = server.received_requests().await.unwrap();
    let targets: Vec<_> = requests.iter().map(target).collect();
    assert_eq!(targets.iter().filter(|t| t.ends_with("Batch")).count(), 2);
    let singles: Vec<_> = requests
        .iter()
        .filter(|request| target(request) == "AmazonSQS.SendMessage")
        .collect();
    assert_eq!(singles.len(), 1);
    let message = json_body(singles[0])["MessageBody"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(report_id(&message), "p-2");
    assert_eq!(
        counter(
            &recorder,
            "inference_proxy_usage_report_retries_total",
            "sqs_5xx"
        ),
        1.0
    );
    assert_eq!(
        counter(&recorder, "inference_proxy_usage_reports_dropped_total", ""),
        0.0
    );
}

#[tokio::test]
async fn a_refused_report_ends_without_a_second_attempt() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(|request: &Request| {
            let entries = json_body(request)["Entries"].as_array().unwrap().len();
            let taken: Vec<usize> = (0..entries).filter(|id| *id != 0).collect();
            ResponseTemplate::new(200).set_body_json(batch_answer(
                &taken,
                json!([{"Id": "0", "SenderFault": true, "Code": "InvalidParameterValue"}]),
            ))
        })
        .mount(&server)
        .await;
    let delivery = delivery_for(sender(&server.uri()), 1);
    delivery.submit(job_with(&chat_body("only"), &delivery));
    let drained = delivery.drain(Duration::from_secs(10)).await;
    assert_eq!(drained.left(), 0);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
    assert_eq!(
        counter(
            &recorder,
            "inference_proxy_usage_reports_dropped_total",
            "sqs_invalid"
        ),
        1.0
    );
}
