//! Batching: the reports sent at once share one `SendMessageBatch`, within
//! its limits, and each is answered by its own part of the batch's answer.
//! Beside `sqs_sender_tests.rs`; the stand-in queue's batch answers live here.

use std::sync::Arc;
use std::time::Duration;

use cloud_platform_events::{parse_message, ParsedMessage, MAX_ENVELOPE_BYTES};
use metrics_exporter_prometheus::PrometheusBuilder;
use reqwest::StatusCode;
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::tests::{batching_sender, chat_body, counter, job_with};
use super::*;
use crate::usage_report::{UsageReportDelivery, UsageReportPolicy};

/// The `SendMessageBatch` answer that takes every entry `request` carries.
pub(super) struct TakesEvery(Duration);

impl TakesEvery {
    pub(super) fn now() -> Self {
        Self(Duration::ZERO)
    }

    pub(super) fn after(delay: Duration) -> Self {
        Self(delay)
    }
}

impl wiremock::Respond for TakesEvery {
    fn respond(&self, request: &wiremock::Request) -> ResponseTemplate {
        let taken: Vec<Value> = entries(request)
            .iter()
            .map(|entry| json!({"Id": entry["Id"], "MessageId": format!("m-{}", entry["Id"])}))
            .collect();
        ResponseTemplate::new(200)
            .set_delay(self.0)
            .set_body_json(json!({"Successful": taken, "Failed": []}))
    }
}

/// The entries of the `SendMessageBatch` `request`.
pub(super) fn entries(request: &wiremock::Request) -> Vec<Value> {
    let body: Value = serde_json::from_slice(&request.body).unwrap();
    body["Entries"].as_array().cloned().unwrap_or_default()
}

/// A stand-in queue that takes every report it is sent.
pub(super) async fn sqs_taking() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(TakesEvery::now())
        .mount(&server)
        .await;
    server
}

/// A delivery over `sender` with room for `in_flight` reports at once, each
/// retried after a millisecond.
fn batching_delivery(sender: SqsSender, in_flight: usize) -> Arc<UsageReportDelivery> {
    UsageReportDelivery::with_sender(
        UsageReportPolicy {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(1),
            max_in_flight: in_flight,
            ..Default::default()
        },
        Arc::new(sender),
    )
}

/// The entries of each call `server` received, in the order received.
async fn batches(server: &MockServer) -> Vec<Vec<Value>> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(entries)
        .collect()
}

/// Reports sent at once share one `SendMessageBatch`, each its own entry, and
/// each is accepted by the batch's answer for it.
#[tokio::test]
async fn reports_sent_at_once_share_one_batch() {
    let sqs = sqs_taking().await;
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let delivery = batching_delivery(batching_sender(&sqs.uri(), Duration::from_millis(200)), 20);
    for id in ["chatcmpl-b1", "chatcmpl-b2", "chatcmpl-b3"] {
        delivery.submit(job_with(&chat_body(id), &delivery));
    }
    let drained = delivery.drain(Duration::from_secs(10)).await;
    assert_eq!(drained.left(), 0, "{drained:?}");

    let batches = batches(&sqs).await;
    assert_eq!(batches.len(), 1, "{batches:?}");
    let mut ids: Vec<String> = batches[0]
        .iter()
        .map(|entry| {
            let ParsedMessage::Usage(envelope) =
                parse_message(entry["MessageBody"].as_str().unwrap().as_bytes()).unwrap()
            else {
                panic!("a usage message");
            };
            serde_json::to_value(&envelope.usage).unwrap()["id"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    ids.sort();
    assert_eq!(ids, ["chatcmpl-b1", "chatcmpl-b2", "chatcmpl-b3"]);
    let entry_ids: Vec<_> = batches[0].iter().map(|entry| entry["Id"].clone()).collect();
    assert_eq!(entry_ids, [json!("0"), json!("1"), json!("2")]);
    assert_eq!(
        counter(&recorder, "inference_proxy_usage_reports_dropped_total", ""),
        0.0
    );
}

/// A batch carries at most ten reports: the rest go in the next.
#[tokio::test]
async fn a_batch_carries_at_most_ten_reports() {
    let sqs = sqs_taking().await;
    let delivery = batching_delivery(batching_sender(&sqs.uri(), Duration::from_millis(200)), 50);
    for n in 0..23 {
        delivery.submit(job_with(&chat_body(&format!("chatcmpl-{n}")), &delivery));
    }
    let drained = delivery.drain(Duration::from_secs(10)).await;
    assert_eq!(drained.left(), 0, "{drained:?}");
    let mut sizes: Vec<_> = batches(&sqs).await.iter().map(Vec::len).collect();
    sizes.sort();
    assert_eq!(sizes, [3, 10, 10]);
}

/// A batch carries at most the bytes of one envelope, so that a report the
/// queue takes always fits: a report that would take a batch past them opens
/// the next one.
#[tokio::test]
async fn a_batch_carries_at_most_the_bytes_of_one_envelope() {
    let sqs = sqs_taking().await;
    let delivery = batching_delivery(batching_sender(&sqs.uri(), Duration::from_millis(200)), 20);
    for n in 0..3 {
        let mut body = chat_body(&format!("chatcmpl-large-{n}"));
        // About 40% of an envelope each: two fit a batch, three do not.
        body["model"] = json!("m".repeat(MAX_ENVELOPE_BYTES * 2 / 5));
        delivery.submit(job_with(&body, &delivery));
    }
    let drained = delivery.drain(Duration::from_secs(10)).await;
    assert_eq!(drained.left(), 0, "{drained:?}");
    let batches = batches(&sqs).await;
    let mut sizes: Vec<_> = batches.iter().map(Vec::len).collect();
    sizes.sort();
    assert_eq!(sizes, [1, 2]);
    for batch in &batches {
        let bytes: usize = batch
            .iter()
            .map(|entry| entry["MessageBody"].as_str().unwrap().len())
            .sum();
        assert!(bytes <= MAX_BATCH_BYTES, "{bytes}");
    }
}

/// Only the report the batch's answer refused is sent again, alone; the one it
/// took is done.
#[tokio::test]
async fn only_a_refused_entry_is_sent_again() {
    let sqs = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "Successful": [{"Id": "0", "MessageId": "m-0"}],
            "Failed": [{"Id": "1", "SenderFault": false, "Code": "InternalError", "Message": "x"}],
        })))
        .up_to_n_times(1)
        .mount(&sqs)
        .await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(TakesEvery::now())
        .mount(&sqs)
        .await;
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let delivery = batching_delivery(batching_sender(&sqs.uri(), Duration::from_millis(200)), 20);
    for id in ["chatcmpl-r1", "chatcmpl-r2"] {
        delivery.submit(job_with(&chat_body(id), &delivery));
    }
    let drained = delivery.drain(Duration::from_secs(10)).await;
    assert_eq!(drained.left(), 0, "{drained:?}");
    let batches = batches(&sqs).await;
    let sizes: Vec<_> = batches.iter().map(Vec::len).collect();
    assert_eq!(sizes, [2, 1]);
    // The report sent again is the one refused (`Id` "1"), not the one taken.
    let body = |entry: &Value| entry["MessageBody"].as_str().unwrap().to_string();
    let refused = batches[0].iter().find(|entry| entry["Id"] == "1").unwrap();
    let taken = batches[0].iter().find(|entry| entry["Id"] == "0").unwrap();
    assert_eq!(body(&batches[1][0]), body(refused));
    assert_ne!(body(&batches[1][0]), body(taken));
    assert_eq!(
        counter(
            &recorder,
            "inference_proxy_usage_report_retries_total",
            r#"reason="sqs_5xx""#
        ),
        1.0
    );
    assert_eq!(
        counter(&recorder, "inference_proxy_usage_reports_dropped_total", ""),
        0.0
    );
}

/// Each entry is answered by its own part of a batch's answer: taken, refused
/// for a reason classified as a call answering it would be, or, named nowhere,
/// unknown and sent again. A call that failed as a whole fails every entry.
#[test]
fn each_entry_is_answered_by_its_own_part_of_the_answer() {
    use UsageReportOutcome as O;
    let taken = |body: Value| CallAnswer {
        outcome: O::Accepted,
        status: Ok(StatusCode::OK),
        error_type: None,
        body: serde_json::to_vec(&body).unwrap(),
    };
    let outcomes = |answers: Vec<Answer>| answers.into_iter().map(|(o, _)| o).collect::<Vec<_>>();

    let answers = entry_answers(
        taken(json!({
            "Successful": [{"Id": "0", "MessageId": "m"}],
            "Failed": [
                {"Id": "1", "SenderFault": true, "Code": "InvalidMessageContents", "Message": "x"},
                {"Id": "2", "SenderFault": false, "Code": "InternalError", "Message": "x"},
                {"Id": "3", "SenderFault": true, "Code": "SomethingNew"},
            ],
        })),
        5,
    );
    assert_eq!(answers[0].1, Ok(StatusCode::OK));
    assert_eq!(
        answers[1].1,
        Err("entry refused: InvalidMessageContents".to_string())
    );
    assert_eq!(
        outcomes(answers),
        [
            O::Accepted,
            O::SqsInvalid,
            O::Sqs5xx,
            O::SqsUnknown,
            O::SqsUnknown
        ]
    );
    // An answer that cannot be read names no entry.
    assert_eq!(
        outcomes(entry_answers(
            CallAnswer {
                body: b"not json".to_vec(),
                ..taken(json!({}))
            },
            2
        )),
        [O::SqsUnknown, O::SqsUnknown]
    );
    // A call that failed as a whole.
    let failed = CallAnswer {
        outcome: O::SqsThrottled,
        status: Ok(StatusCode::BAD_REQUEST),
        error_type: Some("RequestThrottled".to_string()),
        body: Vec::new(),
    };
    let answers = entry_answers(failed, 2);
    assert!(answers
        .iter()
        .all(|answer| *answer == (O::SqsThrottled, Ok(StatusCode::BAD_REQUEST))));
}
