use wiremock::MockServer;

pub(crate) async fn assert_backend_saw_request_id(
    mock_server: &MockServer,
    expected_request_id: &str,
) {
    let received = mock_server
        .received_requests()
        .await
        .expect("mock server records requests");
    assert_eq!(received.len(), 1, "expected one upstream request");
    let observed_request_id = received[0]
        .headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("<missing>");
    eprintln!(
        "manual-qa: wiremock observed x-request-id={observed_request_id} expected={expected_request_id}"
    );
    assert_eq!(observed_request_id, expected_request_id);
}
