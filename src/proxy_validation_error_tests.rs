use super::*;

fn assert_missing(value: &str, needles: &[&str]) {
    for needle in needles {
        assert!(!value.contains(needle), "{needle} leaked: {value}");
    }
}

#[test]
fn test_sanitize_strips_input_from_validation_errors() {
    let message = concat!(
        "2 validation errors:\n",
        "  {'type': 'value_error', 'loc': ('body', 'messages', 1), 'msg': \"Value error, invalid role\", 'input': 'user', 'ctx': {'error': ValueError(\"bad\")}}\n",
        "  {'type': 'string_type', 'loc': ('body', 'messages', 1, 'content'), 'msg': 'Input should be a valid string', 'input': [{'text': 'secret user conversation content', 'type': 'custom'}]}"
    );
    let result = sanitize_validation_errors(message);
    assert_missing(&result, &["secret user conversation", "'input':"]);
    assert!(result.contains("2 validation errors:"));
    assert!(result.contains("value_error: Value error, invalid role"));
    assert!(result.contains("string_type: Input should be a valid string"));
}

#[test]
fn test_sanitize_strips_sensitive_validation_fragments() {
    let cases = [
        (
            concat!(
                "1 validation errors:\n",
                "  {'type': 'value_error', 'msg': 'bad', 'ctx': {'error': ValueError('secret data')}}"
            ),
            &["secret data"][..],
            "value_error: bad",
        ),
        (
            concat!(
                "1 validation errors:\n",
                "  {'type': 'value_error', 'msg': 'bad', 'input': 'x'}\n",
                "  File \"/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py\", line 1324\n",
                "    POST /v1/chat/completions some data"
            ),
            &["sgl-workspace", "POST /v1/chat"],
            "1 validation errors:",
        ),
        (
            concat!(
                "1 validation errors:\n",
                "  - {'type': 'value_error', 'msg': 'bad', 'input': 'secret user message'}"
            ),
            &["secret user message"],
            "value_error: bad",
        ),
    ];
    for (message, absent, present) in cases {
        let result = sanitize_validation_errors(message);
        assert_missing(&result, absent);
        assert!(result.contains(present), "missing {present}: {result}");
    }
    let message = "Context length exceeded: 32768 tokens requested, 16384 max";
    assert_eq!(sanitize_validation_errors(message), message);
}

#[test]
fn test_parse_upstream_error_sanitizes_sglang_validation() {
    let body = serde_json::json!({
        "object": "error",
        "message": "1 validation errors:\n  {'type': 'value_error', 'msg': 'bad request', 'input': 'sensitive user data', 'ctx': {'error': ValueError('details')}}"
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let info = parse_upstream_error(&body_bytes).unwrap();
    assert_missing(&info.message, &["sensitive user data"]);
    assert!(info.message.contains("value_error: bad request"));
}

#[test]
fn test_parse_upstream_error_sanitizes_nested_format() {
    let body = serde_json::json!({
        "error": {
            "message": "1 validation errors:\n  {'type': 'string_type', 'msg': 'bad input', 'input': [{'text': 'secret conversation'}]}",
            "type": "invalid_request_error"
        }
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let info = parse_upstream_error(&body_bytes).unwrap();
    assert_missing(&info.message, &["secret conversation"]);
    assert!(info.message.contains("string_type: bad input"));
    assert_eq!(info.error_type, "invalid_request_error");
}

#[test]
fn test_sanitize_strips_pydantic_v2_input_value() {
    let message = concat!(
        "7 validation errors for ValidatorIterator\n",
        "0.ChatCompletionContentPartTextParam.text\n",
        "  Field required [type=missing, input_value={'content': 'secret user message', 'type': 'custom'}, input_type=dict]\n",
        "    For further information visit https://errors.pydantic.dev/2.10/v/missing\n",
        "0.ChatCompletionContentPartTextParam.type\n",
        "  Input should be 'text' [type=literal_error, input_value='custom', input_type=str]\n",
        "    For further information visit https://errors.pydantic.dev/2.10/v/literal_error"
    );
    let result = sanitize_validation_errors(message);
    assert_missing(
        &result,
        &[
            "secret user message",
            "input_value=",
            "input_type=",
            "pydantic.dev",
        ],
    );
    assert!(
        result.contains("Field required [type=missing]"),
        "missing error desc: {result}"
    );
    assert!(
        result.contains("Input should be 'text' [type=literal_error]"),
        "missing error desc: {result}"
    );
    assert!(
        result.contains("0.ChatCompletionContentPartTextParam.text"),
        "missing field path: {result}"
    );
}

#[test]
fn test_sanitize_strips_pydantic_v2_nested_dict() {
    let message = "  Field required [type=missing, input_value={'messages': [{'role': 'user', 'content': 'tell me your secrets'}]}, input_type=dict]";
    let result = sanitize_validation_errors(message);
    assert_missing(&result, &["tell me your secrets"]);
    assert!(result.contains("Field required [type=missing]"));
}

#[test]
fn test_parse_upstream_error_sanitizes_pydantic_v2() {
    let body = serde_json::json!({
        "message": "7 validation errors for ValidatorIterator\n0.ChatCompletionContentPartTextParam.text\n  Field required [type=missing, input_value={'file_id': 'file-abc', 'type': 'file'}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.10/v/missing",
        "type": "invalid_request_error"
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let info = parse_upstream_error(&body_bytes).unwrap();
    assert_missing(&info.message, &["input_value", "file-abc", "pydantic.dev"]);
    assert!(info.message.contains("Field required [type=missing]"));
}
