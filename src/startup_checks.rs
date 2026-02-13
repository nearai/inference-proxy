use std::time::Duration;

use tracing::{debug, error, info, warn};

use crate::config::Config;

#[derive(Debug, thiserror::Error)]
pub enum StartupCheckError {
    #[error("Backend not reachable at {url}: {source}")]
    ConnectionFailed { url: String, source: reqwest::Error },

    #[error("Backend returned HTTP {status} for {url}: {body}")]
    UnexpectedStatus {
        url: String,
        status: u16,
        body: String,
    },

    #[error("Model '{expected}' not found in /v1/models response. Available: {available:?}")]
    ModelNotFound {
        expected: String,
        available: Vec<String>,
    },

    #[error("Chat completions check failed: {reason}")]
    ChatCompletionFailed { reason: String },

    #[error("All {retries} retries exhausted for '{check_name}': {last_error}")]
    RetriesExhausted {
        check_name: String,
        retries: usize,
        last_error: String,
    },
}

impl StartupCheckError {
    fn is_retryable(&self) -> bool {
        match self {
            StartupCheckError::ConnectionFailed { .. } => true,
            StartupCheckError::UnexpectedStatus { status, .. } => *status == 503,
            StartupCheckError::ModelNotFound { .. } => false,
            StartupCheckError::ChatCompletionFailed { .. } => false,
            StartupCheckError::RetriesExhausted { .. } => false,
        }
    }
}

/// Run all startup health checks against the backend.
pub async fn run_startup_checks(
    client: &reqwest::Client,
    config: &Config,
) -> Result<(), StartupCheckError> {
    info!(
        backend = %config.vllm_base_url,
        model = %config.model_name,
        retries = config.startup_check_retries,
        retry_delay_secs = config.startup_check_retry_delay_secs,
        timeout_secs = config.startup_check_timeout_secs,
        "Running startup health checks"
    );

    let timeout = Duration::from_secs(config.startup_check_timeout_secs);
    let delay = Duration::from_secs(config.startup_check_retry_delay_secs);
    let retries = config.startup_check_retries;

    // Check 1: /v1/models — backend reachable, model loaded
    run_with_retries("models", retries, delay, || {
        check_models(client, &config.models_url, &config.model_name, timeout)
    })
    .await?;
    info!(
        "Startup check passed: model '{}' found in /v1/models",
        config.model_name
    );

    // Check 2: non-streaming chat completions with tools
    run_with_retries("chat_completions_tools", retries, delay, || {
        check_chat_completions_with_tools(
            client,
            &config.chat_completions_url,
            &config.model_name,
            false,
            timeout,
        )
    })
    .await?;
    info!("Startup check passed: non-streaming chat completions with tools");

    // Check 3: streaming chat completions with tools
    run_with_retries("chat_completions_tools_streaming", retries, delay, || {
        check_chat_completions_with_tools(
            client,
            &config.chat_completions_url,
            &config.model_name,
            true,
            timeout,
        )
    })
    .await?;
    info!("Startup check passed: streaming chat completions with tools");

    info!("All startup checks passed");
    Ok(())
}

async fn run_with_retries<F, Fut>(
    check_name: &str,
    max_retries: usize,
    delay: Duration,
    check_fn: F,
) -> Result<(), StartupCheckError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<(), StartupCheckError>>,
{
    let mut last_error = String::new();

    for attempt in 1..=max_retries {
        match check_fn().await {
            Ok(()) => return Ok(()),
            Err(e) => {
                last_error = e.to_string();

                if !e.is_retryable() {
                    error!(
                        check = check_name,
                        attempt,
                        error = %e,
                        "Startup check failed (not retryable)"
                    );
                    return Err(e);
                }

                if attempt < max_retries {
                    warn!(
                        check = check_name,
                        attempt,
                        max_retries,
                        error = %e,
                        retry_in_secs = delay.as_secs(),
                        "Startup check failed, retrying..."
                    );
                    tokio::time::sleep(delay).await;
                } else {
                    error!(
                        check = check_name,
                        attempt,
                        error = %e,
                        "Startup check failed, no retries remaining"
                    );
                }
            }
        }
    }

    Err(StartupCheckError::RetriesExhausted {
        check_name: check_name.to_string(),
        retries: max_retries,
        last_error,
    })
}

async fn check_models(
    client: &reqwest::Client,
    models_url: &str,
    expected_model: &str,
    timeout: Duration,
) -> Result<(), StartupCheckError> {
    info!(url = %models_url, "Checking /v1/models endpoint...");

    let start = std::time::Instant::now();
    let response = client
        .get(models_url)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| {
            error!(
                url = %models_url,
                elapsed_ms = start.elapsed().as_millis() as u64,
                error = %e,
                "Failed to connect to backend /v1/models"
            );
            StartupCheckError::ConnectionFailed {
                url: models_url.to_string(),
                source: e,
            }
        })?;

    let status = response.status();
    debug!(
        url = %models_url,
        status = status.as_u16(),
        elapsed_ms = start.elapsed().as_millis() as u64,
        "Received /v1/models response"
    );

    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        error!(
            url = %models_url,
            status = status.as_u16(),
            response_body = %body,
            "Backend /v1/models returned non-success status"
        );
        return Err(StartupCheckError::UnexpectedStatus {
            url: models_url.to_string(),
            status: status.as_u16(),
            body,
        });
    }

    let body: serde_json::Value =
        response
            .json()
            .await
            .map_err(|e| StartupCheckError::ConnectionFailed {
                url: models_url.to_string(),
                source: e,
            })?;

    // vLLM /v1/models returns: {"data": [{"id": "model-name", ...}, ...]}
    let model_ids: Vec<String> = body
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|v| v.as_str()))
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();

    info!(
        expected = %expected_model,
        available = ?model_ids,
        "Models endpoint returned model list"
    );

    if model_ids.iter().any(|id| id == expected_model) {
        Ok(())
    } else {
        error!(
            expected = %expected_model,
            available = ?model_ids,
            "Expected model not found in backend model list"
        );
        Err(StartupCheckError::ModelNotFound {
            expected: expected_model.to_string(),
            available: model_ids,
        })
    }
}

fn build_tools_request(model_name: &str, stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": model_name,
        "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }],
        "max_tokens": 50,
        "stream": stream,
    })
}

async fn check_chat_completions_with_tools(
    client: &reqwest::Client,
    chat_completions_url: &str,
    model_name: &str,
    stream: bool,
    timeout: Duration,
) -> Result<(), StartupCheckError> {
    let mode = if stream { "streaming" } else { "non-streaming" };
    let request_body = build_tools_request(model_name, stream);

    info!(
        url = %chat_completions_url,
        mode,
        model = %model_name,
        request = %serde_json::to_string(&request_body).unwrap_or_default(),
        "Sending chat completions check with tools..."
    );

    let start = std::time::Instant::now();
    let response = client
        .post(chat_completions_url)
        .timeout(timeout)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| {
            error!(
                url = %chat_completions_url,
                mode,
                elapsed_ms = start.elapsed().as_millis() as u64,
                error = %e,
                "Failed to connect to backend for chat completions check"
            );
            StartupCheckError::ConnectionFailed {
                url: chat_completions_url.to_string(),
                source: e,
            }
        })?;

    let status = response.status();
    debug!(
        url = %chat_completions_url,
        mode,
        status = status.as_u16(),
        elapsed_ms = start.elapsed().as_millis() as u64,
        "Received chat completions response"
    );

    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        error!(
            url = %chat_completions_url,
            mode,
            status = status.as_u16(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            response_body = %body,
            "Backend chat completions returned non-success status"
        );
        return Err(StartupCheckError::UnexpectedStatus {
            url: chat_completions_url.to_string(),
            status: status.as_u16(),
            body,
        });
    }

    let result = if stream {
        validate_streaming_response(response).await
    } else {
        validate_non_streaming_response(response).await
    };

    let elapsed = start.elapsed();
    match &result {
        Ok(()) => info!(
            mode,
            elapsed_ms = elapsed.as_millis() as u64,
            "Chat completions check with tools validated successfully"
        ),
        Err(e) => error!(
            mode,
            elapsed_ms = elapsed.as_millis() as u64,
            error = %e,
            "Chat completions check with tools validation failed"
        ),
    }

    result
}

async fn validate_non_streaming_response(
    response: reqwest::Response,
) -> Result<(), StartupCheckError> {
    let body: serde_json::Value =
        response
            .json()
            .await
            .map_err(|e| StartupCheckError::ChatCompletionFailed {
                reason: format!("Failed to parse response as JSON: {e}"),
            })?;

    debug!(
        response = %serde_json::to_string_pretty(&body).unwrap_or_default(),
        "Non-streaming response body"
    );

    let choices = body
        .get("choices")
        .and_then(|c| c.as_array())
        .ok_or_else(|| StartupCheckError::ChatCompletionFailed {
            reason: format!(
                "Response missing choices array. Response: {}",
                serde_json::to_string_pretty(&body).unwrap_or_default()
            ),
        })?;

    if choices.is_empty() {
        return Err(StartupCheckError::ChatCompletionFailed {
            reason: "Response has empty choices array".to_string(),
        });
    }

    // Log what the model returned (text vs tool call)
    for (i, choice) in choices.iter().enumerate() {
        if let Some(msg) = choice.get("message") {
            let finish_reason = choice
                .get("finish_reason")
                .and_then(|f| f.as_str())
                .unwrap_or("unknown");
            if msg.get("tool_calls").is_some() {
                let tool_names: Vec<&str> = msg
                    .get("tool_calls")
                    .and_then(|tc| tc.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|tc| {
                                tc.get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(|n| n.as_str())
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                info!(
                    choice_index = i,
                    finish_reason,
                    tool_names = ?tool_names,
                    "Model returned tool call(s)"
                );
            } else {
                let content = msg
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("<no content>");
                let preview: String = content.chars().take(100).collect();
                info!(
                    choice_index = i,
                    finish_reason,
                    content_preview = %preview,
                    "Model returned text response"
                );
            }
        }
    }

    // Validate tool call arguments JSON if present
    validate_tool_calls_json(choices)?;

    Ok(())
}

async fn validate_streaming_response(response: reqwest::Response) -> Result<(), StartupCheckError> {
    use futures_util::StreamExt;

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut got_any_data = false;
    let mut seen_done = false;
    let mut chunk_count: usize = 0;
    let mut accumulated_tool_args: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| StartupCheckError::ChatCompletionFailed {
            reason: format!("Error reading stream: {e}"),
        })?;

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete lines
        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
            buffer = buffer[newline_pos + 1..].to_string();

            let data = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
                .unwrap_or(&line)
                .trim();

            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                seen_done = true;
                continue;
            }

            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                got_any_data = true;
                chunk_count += 1;
                debug!(chunk = chunk_count, data = %data, "Received SSE chunk");

                // Accumulate tool call argument deltas
                if let Some(choices) = parsed.get("choices").and_then(|c| c.as_array()) {
                    for choice in choices {
                        if let Some(tool_calls) = choice
                            .get("delta")
                            .and_then(|d| d.get("tool_calls"))
                            .and_then(|tc| tc.as_array())
                        {
                            for tool_call in tool_calls {
                                let index =
                                    tool_call.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                                let key = format!(
                                    "{}-{}",
                                    choice.get("index").and_then(|i| i.as_u64()).unwrap_or(0),
                                    index
                                );
                                if let Some(args) = tool_call
                                    .get("function")
                                    .and_then(|f| f.get("arguments"))
                                    .and_then(|a| a.as_str())
                                {
                                    accumulated_tool_args.entry(key).or_default().push_str(args);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if !got_any_data {
        error!("Streaming response contained no SSE data chunks");
        return Err(StartupCheckError::ChatCompletionFailed {
            reason: "Streaming response contained no data chunks".to_string(),
        });
    }

    if !seen_done {
        error!("Streaming response ended without [DONE] marker — stream may have been truncated");
        return Err(StartupCheckError::ChatCompletionFailed {
            reason: "Streaming response missing [DONE] marker".to_string(),
        });
    }

    info!(
        chunk_count = chunk_count,
        tool_call_count = accumulated_tool_args.len(),
        "Streaming response complete, validating tool call arguments"
    );

    // Validate accumulated tool call arguments are valid JSON
    for (key, args) in &accumulated_tool_args {
        if !args.is_empty() {
            debug!(
                tool_call_index = %key,
                accumulated_arguments = %args,
                "Validating tool call arguments JSON"
            );
            if serde_json::from_str::<serde_json::Value>(args).is_err() {
                error!(
                    tool_call_index = %key,
                    arguments = %args,
                    "Streaming tool call produced malformed JSON arguments"
                );
                return Err(StartupCheckError::ChatCompletionFailed {
                    reason: format!(
                        "Streaming tool call (index {key}) has malformed JSON arguments: {args}"
                    ),
                });
            }
            info!(
                tool_call_index = %key,
                arguments = %args,
                "Tool call arguments validated as valid JSON"
            );
        }
    }

    Ok(())
}

fn validate_tool_calls_json(choices: &[serde_json::Value]) -> Result<(), StartupCheckError> {
    for choice in choices {
        let tool_calls = choice
            .get("message")
            .and_then(|m| m.get("tool_calls"))
            .and_then(|tc| tc.as_array());

        if let Some(tool_calls) = tool_calls {
            for tool_call in tool_calls {
                if let Some(args_str) = tool_call
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                {
                    if serde_json::from_str::<serde_json::Value>(args_str).is_err() {
                        return Err(StartupCheckError::ChatCompletionFailed {
                            reason: format!("Tool call has malformed JSON arguments: {args_str}"),
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // --- Models check tests ---

    #[tokio::test]
    async fn test_check_models_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"id": "test-model", "object": "model"}]
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/models", mock_server.uri());
        let result = check_models(&client, &url, "test-model", Duration::from_secs(5)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_check_models_not_found() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"id": "other-model", "object": "model"}]
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/models", mock_server.uri());
        let result = check_models(&client, &url, "test-model", Duration::from_secs(5)).await;
        assert!(matches!(
            result,
            Err(StartupCheckError::ModelNotFound { .. })
        ));
        assert!(!result.unwrap_err().is_retryable());
    }

    #[tokio::test]
    async fn test_check_models_connection_refused() {
        let client = reqwest::Client::new();
        let result = check_models(
            &client,
            "http://127.0.0.1:1/v1/models",
            "m",
            Duration::from_secs(1),
        )
        .await;
        assert!(matches!(
            result,
            Err(StartupCheckError::ConnectionFailed { .. })
        ));
        assert!(result.unwrap_err().is_retryable());
    }

    #[tokio::test]
    async fn test_check_models_503_is_retryable() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(503).set_body_string("loading"))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/models", mock_server.uri());
        let result = check_models(&client, &url, "m", Duration::from_secs(5)).await;
        assert!(matches!(
            result,
            Err(StartupCheckError::UnexpectedStatus { .. })
        ));
        assert!(result.unwrap_err().is_retryable());
    }

    #[tokio::test]
    async fn test_check_models_500_not_retryable() {
        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/models", mock_server.uri());
        let result = check_models(&client, &url, "m", Duration::from_secs(5)).await;
        assert!(matches!(
            result,
            Err(StartupCheckError::UnexpectedStatus { .. })
        ));
        assert!(!result.unwrap_err().is_retryable());
    }

    // --- Non-streaming with tools tests ---

    #[tokio::test]
    async fn test_non_streaming_tools_success_with_tool_call() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-tools",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\":\"San Francisco\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", mock_server.uri());
        let result = check_chat_completions_with_tools(
            &client,
            &url,
            "test-model",
            false,
            Duration::from_secs(5),
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_non_streaming_tools_success_text_response() {
        // Model may respond with text instead of calling a tool — still valid
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-text",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "I can't check weather."},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16}
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", mock_server.uri());
        let result = check_chat_completions_with_tools(
            &client,
            &url,
            "test-model",
            false,
            Duration::from_secs(5),
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_non_streaming_tools_malformed_json_args() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-bad",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{malformed json"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", mock_server.uri());
        let result = check_chat_completions_with_tools(
            &client,
            &url,
            "test-model",
            false,
            Duration::from_secs(5),
        )
        .await;
        assert!(matches!(
            result,
            Err(StartupCheckError::ChatCompletionFailed { .. })
        ));
    }

    #[tokio::test]
    async fn test_non_streaming_tools_empty_choices() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-empty",
                "choices": []
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", mock_server.uri());
        let result = check_chat_completions_with_tools(
            &client,
            &url,
            "test-model",
            false,
            Duration::from_secs(5),
        )
        .await;
        assert!(matches!(
            result,
            Err(StartupCheckError::ChatCompletionFailed { .. })
        ));
    }

    // --- Streaming with tools tests ---

    #[tokio::test]
    async fn test_streaming_tools_success() {
        let mock_server = MockServer::start().await;

        let sse_body = "\
data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n\
data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"location\\\"\"}}]}}]}\n\n\
data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\": \\\"San Francisco\\\"}\"}}]}}]}\n\n\
data: [DONE]\n\n";

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(sse_body)
                    .insert_header("content-type", "text/event-stream"),
            )
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", mock_server.uri());
        let result = check_chat_completions_with_tools(
            &client,
            &url,
            "test-model",
            true,
            Duration::from_secs(5),
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_streaming_tools_malformed_json_args() {
        let mock_server = MockServer::start().await;

        let sse_body = "\
data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n\
data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{malformed\"}}]}}]}\n\n\
data: [DONE]\n\n";

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(sse_body)
                    .insert_header("content-type", "text/event-stream"),
            )
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", mock_server.uri());
        let result = check_chat_completions_with_tools(
            &client,
            &url,
            "test-model",
            true,
            Duration::from_secs(5),
        )
        .await;
        assert!(matches!(
            result,
            Err(StartupCheckError::ChatCompletionFailed { .. })
        ));
    }

    #[tokio::test]
    async fn test_streaming_missing_done_marker() {
        let mock_server = MockServer::start().await;

        let sse_body = "\
data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n";

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(sse_body)
                    .insert_header("content-type", "text/event-stream"),
            )
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", mock_server.uri());
        let result = check_chat_completions_with_tools(
            &client,
            &url,
            "test-model",
            true,
            Duration::from_secs(5),
        )
        .await;
        assert!(matches!(
            result,
            Err(StartupCheckError::ChatCompletionFailed { .. })
        ));
    }

    #[tokio::test]
    async fn test_streaming_text_response_success() {
        // Model responds with text instead of tool call in streaming — still valid
        let mock_server = MockServer::start().await;

        let sse_body = "\
data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"I\"}}]}\n\n\
data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" can't\"}}]}\n\n\
data: [DONE]\n\n";

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(sse_body)
                    .insert_header("content-type", "text/event-stream"),
            )
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", mock_server.uri());
        let result = check_chat_completions_with_tools(
            &client,
            &url,
            "test-model",
            true,
            Duration::from_secs(5),
        )
        .await;
        assert!(result.is_ok());
    }

    // --- Retry logic tests ---

    #[tokio::test]
    async fn test_retry_succeeds_after_transient_failure() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let mock_server = MockServer::start().await;
        let attempt = Arc::new(AtomicUsize::new(0));
        let attempt_clone = attempt.clone();
        let uri = mock_server.uri();

        // First 2 requests return 503, third succeeds
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(2)
            .mount(&mock_server)
            .await;

        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"id": "test-model"}]
            })))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let result = run_with_retries("test", 3, Duration::from_millis(10), || {
            let client = client.clone();
            let uri = uri.clone();
            let attempt = attempt_clone.clone();
            async move {
                attempt.fetch_add(1, Ordering::SeqCst);
                check_models(
                    &client,
                    &format!("{uri}/v1/models"),
                    "test-model",
                    Duration::from_secs(5),
                )
                .await
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(attempt.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_stops_on_non_retryable_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let attempt = Arc::new(AtomicUsize::new(0));
        let attempt_clone = attempt.clone();

        let result = run_with_retries("test", 5, Duration::from_millis(10), || {
            let attempt = attempt_clone.clone();
            async move {
                attempt.fetch_add(1, Ordering::SeqCst);
                Err(StartupCheckError::ModelNotFound {
                    expected: "test-model".to_string(),
                    available: vec!["other".to_string()],
                })
            }
        })
        .await;

        assert!(matches!(
            result,
            Err(StartupCheckError::ModelNotFound { .. })
        ));
        assert_eq!(attempt.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_exhaustion() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let attempt = Arc::new(AtomicUsize::new(0));
        let attempt_clone = attempt.clone();

        let mock_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&mock_server)
            .await;

        let client = reqwest::Client::new();
        let uri = mock_server.uri();

        let result = run_with_retries("test", 2, Duration::from_millis(10), || {
            let client = client.clone();
            let uri = uri.clone();
            let attempt = attempt_clone.clone();
            async move {
                attempt.fetch_add(1, Ordering::SeqCst);
                check_models(
                    &client,
                    &format!("{uri}/v1/models"),
                    "m",
                    Duration::from_secs(5),
                )
                .await
            }
        })
        .await;

        assert!(matches!(
            result,
            Err(StartupCheckError::RetriesExhausted { retries: 2, .. })
        ));
        assert_eq!(attempt.load(Ordering::SeqCst), 2);
    }
}
