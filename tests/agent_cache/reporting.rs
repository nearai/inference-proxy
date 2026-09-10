use super::*;
use serde_json::{json, Value};
use tokio::sync::Notify;

pub(super) async fn usage_server() -> (MockServer, Arc<Notify>) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "organization_id": "org-test",
            "workspace_id": "ws-test",
            "api_key_id": "key-test"
        })))
        .mount(&server)
        .await;
    let received = Arc::new(Notify::new());
    let notify = Arc::clone(&received);
    Mock::given(method("POST"))
        .and(path("/v1/internal/usage"))
        .and(header("authorization", "Bearer test-usage-token"))
        .respond_with(move |_: &wiremock::Request| {
            notify.notify_one();
            ResponseTemplate::new(200)
        })
        .expect(1)
        .mount(&server)
        .await;
    (server, received)
}

pub(super) async fn reported_usage(server: &MockServer, received: &Notify) -> Value {
    tokio::time::timeout(std::time::Duration::from_secs(2), received.notified())
        .await
        .expect("agent loop must report usage after terminating");
    let requests = server.received_requests().await.unwrap();
    let reports: Vec<_> = requests
        .iter()
        .filter(|request| request.url.path() == "/v1/internal/usage")
        .collect();
    assert_eq!(
        reports.len(),
        1,
        "one report per complete or interrupted loop"
    );
    serde_json::from_slice(&reports[0].body).unwrap()
}

#[derive(Clone, Copy)]
enum FinalTurn {
    Complete,
    Interrupted,
    ExplicitZero,
}

async fn assert_two_iteration_usage(final_turn: FinalTurn) {
    // Given two iterations, both with cumulative cache observations.
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;
    let (cloud_api, received) = usage_server().await;
    let first = upstream_tool_call_sse("chatcmpl-CACHE", "rust").replace(
        "\"total_tokens\":15",
        "\"total_tokens\":15,\"prompt_tokens_details\":{\"cached_tokens\":7}",
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("\"continuous_usage_stats\":true"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(first, "text/event-stream"))
        .up_to_n_times(1)
        .expect(1)
        .mount(&upstream)
        .await;
    let mut final_sse = upstream_final_answer_sse_no_done("chatcmpl-FINAL");
    match final_turn {
        FinalTurn::Complete => final_sse.push_str("data: [DONE]\n\n"),
        FinalTurn::Interrupted => {}
        FinalTurn::ExplicitZero => {
            final_sse = final_sse.replace("\"cached_tokens\":null", "\"cached_tokens\":0");
            final_sse.push_str("data: [DONE]\n\n");
        }
    }
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("\"role\":\"tool\""))
        .respond_with(ResponseTemplate::new(200).set_body_raw(final_sse, "text/event-stream"))
        .expect(1)
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path("/res/v1/llm/context"))
        .respond_with(ResponseTemplate::new(200).set_body_json(brave_context_json()))
        .expect(1)
        .mount(&brave)
        .await;
    let app = build_agent_loop_app_with_cloud(
        &upstream.uri(),
        Some(&format!("{}/res/v1/llm/context", brave.uri())),
        Some(&cloud_api.uri()),
    );

    // When the final iteration completes or ends before [DONE].
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-test-cache-loop-key")
                .header("content-type", "application/json")
                .body(Body::from(agent_loop_request_body(true).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_string(response).await;
    let usage = reported_usage(&cloud_api, &received).await;

    // Then only the final cumulative counts from each iteration are billed.
    assert_eq!(usage["type"], "chat_completion");
    assert_eq!(usage["input_tokens"], 25);
    assert_eq!(usage["output_tokens"], 8);
    let expected_cache = match final_turn {
        FinalTurn::Complete | FinalTurn::Interrupted => 19,
        FinalTurn::ExplicitZero => 7,
    };
    assert_eq!(usage["cache_read_tokens"], expected_cache);
    assert_eq!(usage["organization_id"], "org-test");
    assert_eq!(usage["workspace_id"], "ws-test");
    assert_eq!(usage["api_key_id"], "key-test");
    assert_eq!(usage["id"], "chatcmpl-CACHE");
    match final_turn {
        FinalTurn::Complete | FinalTurn::ExplicitZero => {
            assert!(body.ends_with("data: [DONE]\n\n"));
        }
        FinalTurn::Interrupted => {
            assert!(!body.contains("data: [DONE]"));
            let signature = app
                .oneshot(
                    Request::builder()
                        .uri("/v1/signature/chatcmpl-CACHE")
                        .header("authorization", "Bearer test-token")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(signature.status(), StatusCode::NOT_FOUND);
        }
    }
    upstream.verify().await;
    brave.verify().await;
    cloud_api.verify().await;
}

#[tokio::test]
async fn completed_agent_cache_report_sums_iterations_once() {
    assert_two_iteration_usage(FinalTurn::Complete).await;
}

#[tokio::test]
async fn interrupted_agent_cache_report_retains_earlier_cumulative_observation() {
    assert_two_iteration_usage(FinalTurn::Interrupted).await;
}

#[tokio::test]
async fn completed_agent_cache_report_honors_explicit_zero() {
    assert_two_iteration_usage(FinalTurn::ExplicitZero).await;
}
