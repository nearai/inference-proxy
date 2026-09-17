//! Pre-dispatch normalisation of tool-call history.
//!
//! The OpenAI schema types `messages[*].tool_calls[*].function.arguments` as a
//! string holding a JSON object. Clients replaying a conversation send it in
//! every other shape too: an empty string for a call without parameters, the
//! object itself instead of its serialisation, a JSON-encoded string of the
//! JSON (double encoding), an array or a scalar, a string cut mid-way by a
//! broken stream, or no `arguments` key at all. SGLang renders the arguments
//! through the chat template and refuses the whole request with `400
//! "Assistant tool call function.arguments must be valid JSON"` / `"... must
//! be a JSON object"`, so one odd historical turn blocks every later turn of
//! that conversation. That was most of what OpenRouter users saw on the first
//! day of the GLM-5.3 Flash listing (nearai/inference-proxy#239).
//!
//! The proxy repairs the field before dispatch instead of failing the request:
//! nothing, `null`, blank and JSON `null` become `{}`; a double-encoded object
//! is unwrapped; an object sent as JSON is serialised; anything else that is
//! not an object is wrapped as `{"value": ...}` so the model still sees what
//! the call carried. Well-formed arguments are left byte-for-byte unchanged.

use serde_json::{Map, Value};

/// Counter label for each repair, so a client that keeps sending odd history
/// is visible in `tool_call_arguments_normalized_total`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Repair {
    /// No `arguments` key, or `arguments: null`.
    Missing,
    /// An empty or whitespace-only string.
    Empty,
    /// A string holding JSON `null`.
    NullJson,
    /// A string holding a JSON string that itself holds a JSON object.
    DoubleEncoded,
    /// A JSON object sent as an object rather than its serialisation.
    ObjectToString,
    /// Valid JSON that is not an object (array, number, bool, string).
    NonObject,
    /// A string that is not valid JSON.
    InvalidJson,
}

impl Repair {
    pub fn as_str(self) -> &'static str {
        match self {
            Repair::Missing => "missing",
            Repair::Empty => "empty",
            Repair::NullJson => "null_json",
            Repair::DoubleEncoded => "double_encoded",
            Repair::ObjectToString => "object_to_string",
            Repair::NonObject => "non_object",
            Repair::InvalidJson => "invalid_json",
        }
    }
}

/// Normalise every `messages[*].tool_calls[*].function.arguments` in place.
/// Returns the repairs made, in document order (empty when nothing changed).
pub fn normalize_tool_call_arguments(request_json: &mut Value) -> Vec<Repair> {
    let mut repairs = Vec::new();
    let Some(messages) = request_json
        .get_mut("messages")
        .and_then(|m| m.as_array_mut())
    else {
        return repairs;
    };
    for message in messages.iter_mut() {
        let Some(tool_calls) = message.get_mut("tool_calls").and_then(|t| t.as_array_mut()) else {
            continue;
        };
        for call in tool_calls.iter_mut() {
            let Some(function) = call.get_mut("function").and_then(|f| f.as_object_mut()) else {
                continue;
            };
            let (replacement, repair) = match function.get("arguments") {
                None | Some(Value::Null) => {
                    (Some(Value::String("{}".to_string())), Repair::Missing)
                }
                Some(Value::String(raw)) => match normalize_string(raw) {
                    Some((fixed, repair)) => (Some(Value::String(fixed)), repair),
                    None => (None, Repair::Missing),
                },
                Some(Value::Object(obj)) => (
                    Some(Value::String(serialize(&Value::Object(obj.clone())))),
                    Repair::ObjectToString,
                ),
                Some(other) => (Some(Value::String(wrap(other.clone()))), Repair::NonObject),
            };
            if let Some(value) = replacement {
                function.insert("arguments".to_string(), value);
                metrics::counter!(
                    "tool_call_arguments_normalized_total",
                    "kind" => repair.as_str()
                )
                .increment(1);
                repairs.push(repair);
            }
        }
    }
    repairs
}

/// `None` when the string already holds a JSON object (leave it untouched).
fn normalize_string(raw: &str) -> Option<(String, Repair)> {
    if raw.trim().is_empty() {
        return Some(("{}".to_string(), Repair::Empty));
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(_)) => None,
        Ok(Value::Null) => Some(("{}".to_string(), Repair::NullJson)),
        Ok(Value::String(inner)) => {
            if matches!(serde_json::from_str::<Value>(&inner), Ok(Value::Object(_))) {
                Some((inner, Repair::DoubleEncoded))
            } else {
                Some((wrap(Value::String(inner)), Repair::NonObject))
            }
        }
        Ok(other) => Some((wrap(other), Repair::NonObject)),
        Err(_) => Some((wrap(Value::String(raw.to_string())), Repair::InvalidJson)),
    }
}

fn wrap(value: Value) -> String {
    let mut obj = Map::new();
    obj.insert("value".to_string(), value);
    serialize(&Value::Object(obj))
}

fn serialize(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(arguments: Value) -> Value {
        let mut function = json!({"name": "get_time"});
        function["arguments"] = arguments;
        json!({"messages": [
            {"role": "user", "content": "time?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": function}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "12:00"}
        ]})
    }

    fn arguments_of(request_json: &Value) -> &Value {
        &request_json["messages"][1]["tool_calls"][0]["function"]["arguments"]
    }

    #[test]
    fn well_formed_arguments_are_untouched() {
        for raw in ["{}", "{\"a\": 1}", " {\"a\":{\"b\":[1,2]}} "] {
            let mut req = request(json!(raw));
            assert!(normalize_tool_call_arguments(&mut req).is_empty(), "{raw}");
            assert_eq!(arguments_of(&req), &json!(raw));
        }
    }

    #[test]
    fn empty_null_and_missing_become_an_empty_object() {
        let mut req = request(json!(""));
        assert_eq!(normalize_tool_call_arguments(&mut req), vec![Repair::Empty]);
        assert_eq!(arguments_of(&req), &json!("{}"));

        let mut req = request(json!("   "));
        assert_eq!(normalize_tool_call_arguments(&mut req), vec![Repair::Empty]);
        assert_eq!(arguments_of(&req), &json!("{}"));

        let mut req = request(json!("null"));
        assert_eq!(
            normalize_tool_call_arguments(&mut req),
            vec![Repair::NullJson]
        );
        assert_eq!(arguments_of(&req), &json!("{}"));

        let mut req = request(Value::Null);
        assert_eq!(
            normalize_tool_call_arguments(&mut req),
            vec![Repair::Missing]
        );
        assert_eq!(arguments_of(&req), &json!("{}"));

        let mut req = request(json!("{}"));
        req["messages"][1]["tool_calls"][0]["function"]
            .as_object_mut()
            .unwrap()
            .remove("arguments");
        assert_eq!(
            normalize_tool_call_arguments(&mut req),
            vec![Repair::Missing]
        );
        assert_eq!(arguments_of(&req), &json!("{}"));
    }

    #[test]
    fn double_encoded_object_is_unwrapped() {
        let mut req = request(json!("\"{\\\"city\\\": \\\"Paris\\\"}\""));
        assert_eq!(
            normalize_tool_call_arguments(&mut req),
            vec![Repair::DoubleEncoded]
        );
        assert_eq!(arguments_of(&req), &json!("{\"city\": \"Paris\"}"));
    }

    #[test]
    fn object_sent_as_json_is_serialised() {
        let mut req = request(json!({"city": "Paris"}));
        assert_eq!(
            normalize_tool_call_arguments(&mut req),
            vec![Repair::ObjectToString]
        );
        assert_eq!(arguments_of(&req), &json!("{\"city\":\"Paris\"}"));
    }

    #[test]
    fn non_objects_and_broken_json_are_wrapped() {
        let mut req = request(json!("[1, 2]"));
        assert_eq!(
            normalize_tool_call_arguments(&mut req),
            vec![Repair::NonObject]
        );
        assert_eq!(arguments_of(&req), &json!("{\"value\":[1,2]}"));

        let mut req = request(json!("\"paris\""));
        assert_eq!(
            normalize_tool_call_arguments(&mut req),
            vec![Repair::NonObject]
        );
        assert_eq!(arguments_of(&req), &json!("{\"value\":\"paris\"}"));

        let mut req = request(json!(42));
        assert_eq!(
            normalize_tool_call_arguments(&mut req),
            vec![Repair::NonObject]
        );
        assert_eq!(arguments_of(&req), &json!("{\"value\":42}"));

        let mut req = request(json!("{\"city\": \"Par"));
        assert_eq!(
            normalize_tool_call_arguments(&mut req),
            vec![Repair::InvalidJson]
        );
        assert_eq!(
            arguments_of(&req),
            &json!("{\"value\":\"{\\\"city\\\": \\\"Par\"}")
        );
    }

    #[test]
    fn every_call_in_every_message_is_visited() {
        let mut req = json!({"messages": [
            {"role": "assistant", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "f", "arguments": ""}},
                {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{\"ok\":true}"}}
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "x"},
            {"role": "assistant", "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "h", "arguments": "[]"}}
            ]}
        ]});
        assert_eq!(
            normalize_tool_call_arguments(&mut req),
            vec![Repair::Empty, Repair::NonObject]
        );
        assert_eq!(
            req["messages"][0]["tool_calls"][1]["function"]["arguments"],
            json!("{\"ok\":true}")
        );
        assert_eq!(
            req["messages"][2]["tool_calls"][0]["function"]["arguments"],
            json!("{\"value\":[]}")
        );
    }

    #[test]
    fn odd_shapes_around_the_field_are_ignored() {
        for req in [
            json!({"prompt": "no messages"}),
            json!({"messages": "not a list"}),
            json!({"messages": [{"role": "assistant", "tool_calls": "not a list"}]}),
            json!({"messages": [{"role": "assistant", "tool_calls": [{"id": "x", "function": "not an object"}]}]}),
            json!({"messages": [{"role": "assistant", "tool_calls": [{"id": "x"}]}]}),
        ] {
            let mut req = req;
            let before = req.clone();
            assert!(normalize_tool_call_arguments(&mut req).is_empty());
            assert_eq!(req, before);
        }
    }
}
