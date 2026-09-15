//! Pre-dispatch policy for chat content parts.
//!
//! Engines answer an unsupported modality inconsistently: SGLang fetches a
//! `video_url` (an outbound request from inside the CVM) and then fails with a
//! decoder error; an intermediate gateway may turn that into a 502. A model
//! that intentionally does not serve a modality should refuse it up front with
//! a deterministic 400 and never dispatch the request. Opt-in via
//! `VLLM_PROXY_REJECTED_CONTENT_PART_TYPES` (comma-separated `type` values such
//! as `video_url,input_audio,file`); the default is empty, so existing
//! deployments are unchanged.

use serde_json::Value;

use crate::error::AppError;

/// Reject the request if any `messages[*].content[*].type` is in `rejected`.
/// Matching is exact and case-sensitive, like the OpenAI schema itself.
pub fn reject_unsupported_content_parts(
    request_json: &Value,
    rejected: &[String],
) -> Result<(), AppError> {
    if rejected.is_empty() {
        return Ok(());
    }
    let Some(messages) = request_json.get("messages").and_then(|m| m.as_array()) else {
        return Ok(());
    };
    for msg in messages {
        let Some(parts) = msg.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for part in parts {
            let Some(part_type) = part.get("type").and_then(|t| t.as_str()) else {
                continue;
            };
            if rejected.iter().any(|r| r == part_type) {
                metrics::counter!(
                    "rejected_content_parts_total",
                    "part_type" => part_type.to_string()
                )
                .increment(1);
                return Err(AppError::BadRequest(format!(
                    "Content part type '{part_type}' is not supported by this model."
                )));
            }
        }
    }
    Ok(())
}

/// Parse the env value: comma-separated, trimmed, empty entries dropped.
pub fn parse_rejected_types(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn rejected() -> Vec<String> {
        parse_rejected_types("video_url, input_audio,,file ")
    }

    #[test]
    fn parses_env_list() {
        assert_eq!(rejected(), vec!["video_url", "input_audio", "file"]);
        assert!(parse_rejected_types("").is_empty());
    }

    #[test]
    fn rejects_listed_part_types_anywhere_in_history() {
        let req = json!({"messages": [
            {"role": "user", "content": "plain text turn"},
            {"role": "assistant", "content": null},
            {"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}
            ]}
        ]});
        let err = reject_unsupported_content_parts(&req, &rejected()).unwrap_err();
        match err {
            AppError::BadRequest(msg) => assert!(msg.contains("'video_url'"), "{msg}"),
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn allows_text_and_images_and_mentions_of_video() {
        let req = json!({"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "this text mentions a video_url but is text"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]}
        ]});
        assert!(reject_unsupported_content_parts(&req, &rejected()).is_ok());
    }

    #[test]
    fn empty_policy_or_no_messages_is_a_noop() {
        let req = json!({"messages": [{"role": "user", "content": [{"type": "video_url"}]}]});
        assert!(reject_unsupported_content_parts(&req, &[]).is_ok());
        assert!(reject_unsupported_content_parts(&json!({"prompt": "x"}), &rejected()).is_ok());
        assert!(reject_unsupported_content_parts(&json!({"messages": "bad"}), &rejected()).is_ok());
    }
}
