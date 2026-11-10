//! Request feature extraction only; monetary policy lives in Cloud API.
use crate::error::AppError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
fn overflow() -> AppError {
    AppError::BadRequest("Request estimate overflow".into())
}
/// Counts computed locally where the request is already visible. No content crosses admission.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UsageFeatures {
    pub inference_type: String,
    pub text_bytes: u64,
    pub token_ids: u64,
    pub images: u64,
    pub audio: u64,
    pub video: u64,
    pub unknown_media: u64,
    pub batch_items: u64,
    pub choices: u64,
    pub executions: u64,
    pub requested_output_tokens: Option<u64>,
    pub model_output_tokens: Option<u64>,
    pub media_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_duration_seconds: Option<u64>,
    pub requested_images: u64,
    pub questions: u64,
}

impl UsageFeatures {
    pub fn from_json(kind: String, body: &Value) -> Result<Self, AppError> {
        let positive = |name: &str| -> Result<Option<u64>, AppError> {
            body.get(name)
                .filter(|v| !v.is_null())
                .map(|v| {
                    v.as_u64().filter(|n| *n > 0).ok_or_else(|| {
                        AppError::BadRequest("invalid generation count or output limit".into())
                    })
                })
                .transpose()
        };
        let mut features = Self {
            inference_type: kind,
            text_bytes: 0,
            token_ids: 0,
            images: 0,
            audio: 0,
            video: 0,
            unknown_media: 0,
            batch_items: 1,
            choices: positive("n")?
                .unwrap_or(1)
                .max(positive("best_of")?.unwrap_or(1)),
            executions: 1,
            requested_output_tokens: positive("max_completion_tokens")?
                .or(positive("max_output_tokens")?)
                .or(positive("max_tokens")?),
            model_output_tokens: None,
            media_bytes: 0,
            audio_duration_seconds: None,
            requested_images: positive("n")?.unwrap_or(1),
            questions: 0,
        };
        for name in [
            "messages",
            "prompt",
            "input",
            "instructions",
            "system",
            "state",
            "questions",
            "assets",
        ] {
            if let Some(value) = body.get(name) {
                features.count(value)?;
            }
        }
        // Schemas are part of the model input. Count their serialized keys and values;
        // the media walker intentionally skips fields such as `data`, `id`, and URLs.
        for value in [
            body.get("tools"),
            body.get("functions"),
            body.get("tool_choice"),
            body.get("response_format"),
            body.pointer("/text/format"),
        ]
        .into_iter()
        .flatten()
        {
            features.text_bytes = features
                .text_bytes
                .checked_add(value.to_string().len() as u64)
                .ok_or_else(overflow)?;
        }
        if let Some(items) = body.get("prompt").and_then(Value::as_array) {
            if !items.iter().all(Value::is_number) {
                features.batch_items = items.len().max(1) as u64;
            }
        }
        features.questions = match body.get("questions") {
            Some(Value::Array(items)) => items.len().max(1) as u64,
            Some(Value::Object(items)) => items.len().max(1) as u64,
            _ => 1,
        };
        Ok(features)
    }

    fn count(&mut self, value: &Value) -> Result<(), AppError> {
        let add = |target: &mut u64, n: u64| -> Result<(), AppError> {
            *target = target.checked_add(n).ok_or_else(overflow)?;
            Ok(())
        };
        match value {
            Value::String(text) => add(&mut self.text_bytes, text.len() as u64)?,
            Value::Array(items) if !items.is_empty() && items.iter().all(Value::is_number) => {
                add(&mut self.token_ids, items.len() as u64)?
            }
            Value::Array(items) => {
                for item in items {
                    self.count(item)?;
                }
            }
            Value::Object(fields) => {
                let kind = fields.get("type").and_then(Value::as_str).unwrap_or("");
                let media = match kind {
                    "image" | "image_url" | "input_image" => Some(&mut self.images),
                    "audio" | "input_audio" | "audio_url" => Some(&mut self.audio),
                    "video" | "video_url" | "input_video" => Some(&mut self.video),
                    "file" | "input_file" | "document" => Some(&mut self.unknown_media),
                    _ => None,
                };
                if let Some(count) = media {
                    add(count, 1)?;
                } else {
                    for (name, item) in fields {
                        if !matches!(
                            name.as_str(),
                            "id" | "url"
                                | "file_id"
                                | "image_url"
                                | "audio_url"
                                | "data"
                                | "b64_json"
                        ) {
                            self.count(item)?;
                        } else if matches!(name.as_str(), "image_url" | "audio_url" | "file_id") {
                            add(&mut self.unknown_media, 1)?;
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schema_keys_and_media_named_properties_count_as_model_input() {
        use serde_json::json;
        // Large keys and "data"/"id" properties used to be omitted by the media walker.
        let schema = json!({"type": "object", "properties": {
            "data": {"const": "x".repeat(120_000)},
            "id": {"enum": ["y".repeat(60_000)]},
            "z".repeat(10_000): {"type": "number"}
        }});
        for body in [
            json!({"response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}),
            json!({"text": {"format": {"type": "json_schema", "schema": schema}}}),
            json!({"tools": [{"type": "function", "function": {"parameters": schema}}]}),
            json!({"tools": [{"name": "test", "input_schema": schema}]}),
            json!({"functions": [{"parameters": schema}]}),
        ] {
            let original = body.clone();
            let features = UsageFeatures::from_json("chat_completion".into(), &body).unwrap();
            assert!(features.text_bytes >= 190_000);
            assert_eq!((features.images, features.unknown_media), (0, 0));
            assert_eq!(body, original, "estimation must not alter inference");
        }
    }
}
