//! Pre-dispatch repair of `response_format.json_schema`.
//!
//! OpenAI's structured-output shape is
//! `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}`
//! and `name` is required by the OpenAI and OpenRouter specs. Some producers
//! leave it out (a raw request body pasted into a generic OpenAI-compatible
//! client, extensions that never set one) and a few send the JSON Schema
//! itself as `json_schema` with no `schema` wrapper at all. Several
//! OpenRouter providers accept both, so the producer never notices; SGLang
//! types `JsonSchemaResponseFormat.name` as a required `str` and refuses the
//! request with a pydantic validation `400` whose sanitised form carries no
//! field path, which is what the OpenRouter lane saw at up to ~9k/h from
//! 2026-09-25 (nearai/inference-proxy#279). On OpenRouter every such 400 is a
//! request served by another provider instead.
//!
//! The proxy repairs the object in place before dispatch, on every lane, the
//! way [`crate::tool_calls::normalize_tool_call_arguments`] repairs tool-call
//! history:
//!
//! * `json_schema` is an object without `name` but with a wrapper key
//!   (`schema` or `strict`), or the request is not a `json_schema` request at
//!   all: insert `name: "response_schema"`. The schema, strictness,
//!   description and any unknown keys are untouched.
//! * `response_format.type` is `json_schema` and `json_schema` is an object
//!   with neither `name`, `schema` nor `strict`: it holds the schema itself,
//!   so it is wrapped as `{"name": "response_schema", "schema": <object>}`.
//!
//! An explicit `name` of any value (including `""`, `null` or a number) is
//! preserved so the engine's native validation still applies, and a
//! `json_schema` that is not an object is left for the engine to reject.
//! `schema: true` is not special-cased: SGLang only accepts a dict there, so
//! the request still fails after the name is defaulted, just with a clearer
//! error. Each repair is counted in
//! `json_schema_response_format_repaired_total{repair}`; no payload is logged.

use serde_json::Value;

/// Name inserted when the client sent none. Matches OpenAI's
/// `^[a-zA-Z0-9_-]{1,64}$` constraint; SGLang does not use it for decoding.
pub const DEFAULT_JSON_SCHEMA_NAME: &str = "response_schema";

/// Counter label for each repair, so the shape a producer keeps sending is
/// visible in `json_schema_response_format_repaired_total`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Repair {
    /// `json_schema` was a wrapper object with no `name`.
    NameDefaulted,
    /// `json_schema` held the JSON Schema itself; it was wrapped and named.
    BareSchemaWrapped,
}

impl Repair {
    pub fn as_str(self) -> &'static str {
        match self {
            Repair::NameDefaulted => "name_defaulted",
            Repair::BareSchemaWrapped => "bare_schema_wrapped",
        }
    }
}

/// Repair `response_format.json_schema` in place. Returns the repair made,
/// `None` when the request was left byte-for-byte unchanged.
pub fn repair_json_schema_response_format(request: &mut Value) -> Option<Repair> {
    let response_format = request
        .get_mut("response_format")
        .and_then(Value::as_object_mut)?;
    let is_json_schema_request =
        response_format.get("type").and_then(Value::as_str) == Some("json_schema");
    let json_schema = response_format
        .get_mut("json_schema")
        .and_then(Value::as_object_mut)?;
    if json_schema.contains_key("name") {
        return None;
    }

    let has_wrapper_key = json_schema.contains_key("schema") || json_schema.contains_key("strict");
    let repair = if has_wrapper_key || !is_json_schema_request {
        Repair::NameDefaulted
    } else {
        let bare_schema = std::mem::take(json_schema);
        json_schema.insert("schema".to_string(), Value::Object(bare_schema));
        Repair::BareSchemaWrapped
    };
    json_schema.insert(
        "name".to_string(),
        Value::String(DEFAULT_JSON_SCHEMA_NAME.to_string()),
    );
    metrics::counter!(
        "json_schema_response_format_repaired_total",
        "repair" => repair.as_str()
    )
    .increment(1);
    Some(repair)
}

#[cfg(test)]
mod tests {
    use super::{repair_json_schema_response_format, Repair, DEFAULT_JSON_SCHEMA_NAME};
    use serde_json::json;

    fn request(response_format: serde_json::Value) -> serde_json::Value {
        json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": response_format
        })
    }

    #[test]
    fn defaults_missing_name_and_is_idempotent() {
        let mut req = request(json!({
            "type": "json_schema",
            "json_schema": {
                "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
                "strict": true,
                "description": "d",
                "x-unknown": 1
            }
        }));
        let before = req.clone();

        assert_eq!(
            repair_json_schema_response_format(&mut req),
            Some(Repair::NameDefaulted)
        );
        let mut expected = before;
        expected["response_format"]["json_schema"]["name"] = json!(DEFAULT_JSON_SCHEMA_NAME);
        assert_eq!(req, expected);

        assert_eq!(repair_json_schema_response_format(&mut req), None);
        assert_eq!(req, expected);
    }

    #[test]
    fn strict_only_wrapper_gets_a_name_but_no_schema() {
        // Nothing to wrap: the engine's "schema_ is required" error is the
        // right answer, and it is far clearer than the nameless 400.
        let mut req = request(json!({"type": "json_schema", "json_schema": {"strict": true}}));
        assert_eq!(
            repair_json_schema_response_format(&mut req),
            Some(Repair::NameDefaulted)
        );
        assert_eq!(
            req["response_format"]["json_schema"],
            json!({"strict": true, "name": DEFAULT_JSON_SCHEMA_NAME})
        );
    }

    #[test]
    fn wraps_a_bare_schema() {
        let schema = json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        });
        let mut req = request(json!({"type": "json_schema", "json_schema": schema}));

        assert_eq!(
            repair_json_schema_response_format(&mut req),
            Some(Repair::BareSchemaWrapped)
        );
        assert_eq!(
            req["response_format"],
            json!({
                "type": "json_schema",
                "json_schema": {"name": DEFAULT_JSON_SCHEMA_NAME, "schema": schema}
            })
        );
        assert_eq!(repair_json_schema_response_format(&mut req), None);
    }

    #[test]
    fn wraps_an_empty_json_schema_object() {
        // `{}` cannot be told apart from an empty bare schema; wrapping it
        // turns the nameless 400 into the engine's explicit schema error.
        let mut req = request(json!({"type": "json_schema", "json_schema": {}}));
        assert_eq!(
            repair_json_schema_response_format(&mut req),
            Some(Repair::BareSchemaWrapped)
        );
        assert_eq!(
            req["response_format"]["json_schema"],
            json!({"name": DEFAULT_JSON_SCHEMA_NAME, "schema": {}})
        );
    }

    #[test]
    fn only_names_when_type_is_not_json_schema() {
        // The engine validates `json_schema` whatever `type` says, so a
        // nameless wrapper next to `json_object` is rejected too. Naming it is
        // enough; wrapping would invent a schema the request never asked for.
        for ty in ["json_object", "text"] {
            let mut req = request(json!({"type": ty, "json_schema": {"a": 1}}));
            assert_eq!(
                repair_json_schema_response_format(&mut req),
                Some(Repair::NameDefaulted)
            );
            assert_eq!(
                req["response_format"],
                json!({"type": ty, "json_schema": {"a": 1, "name": DEFAULT_JSON_SCHEMA_NAME}})
            );
        }
    }

    #[test]
    fn boolean_schema_only_gets_a_name() {
        // SGLang rejects a boolean `schema` regardless; the repair does not
        // pretend otherwise, it just removes the missing-name error.
        let mut req = request(json!({"type": "json_schema", "json_schema": {"schema": true}}));
        assert_eq!(
            repair_json_schema_response_format(&mut req),
            Some(Repair::NameDefaulted)
        );
        assert_eq!(
            req["response_format"]["json_schema"],
            json!({"schema": true, "name": DEFAULT_JSON_SCHEMA_NAME})
        );
    }

    #[test]
    fn preserves_every_explicit_name_value() {
        for name in [
            json!("custom-name"),
            json!(""),
            json!(null),
            json!(42),
            json!(true),
        ] {
            let mut req = request(json!({
                "type": "json_schema",
                "json_schema": {
                    "schema": {},
                    "name": name,
                    "strict": true,
                    "description": "A named response"
                }
            }));
            let original = req.clone();
            assert_eq!(repair_json_schema_response_format(&mut req), None);
            assert_eq!(req, original);
        }
    }

    #[test]
    fn ignores_shapes_the_engine_should_judge() {
        for response_format in [
            json!({"type": "json_schema"}),
            json!({"type": "json_schema", "json_schema": null}),
            json!({"type": "json_schema", "json_schema": "{}"}),
            json!({"type": "json_schema", "json_schema": []}),
            json!({"type": "json_object"}),
            json!({"type": "text"}),
            json!("json_schema"),
            json!(null),
        ] {
            let mut req = request(response_format.clone());
            assert_eq!(repair_json_schema_response_format(&mut req), None);
            assert_eq!(req["response_format"], response_format);
        }
        let mut req = json!({"model": "m", "messages": []});
        assert_eq!(repair_json_schema_response_format(&mut req), None);
    }

    #[test]
    fn counts_each_repair_under_its_label() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let mut named = request(json!({"type": "json_schema", "json_schema": {"schema": {}}}));
        let mut bare = request(json!({"type": "json_schema", "json_schema": {"type": "object"}}));

        metrics::with_local_recorder(&recorder, || {
            assert!(repair_json_schema_response_format(&mut named).is_some());
            assert!(repair_json_schema_response_format(&mut named).is_none());
            assert!(repair_json_schema_response_format(&mut bare).is_some());
        });

        let rendered = recorder.handle().render();
        assert!(rendered
            .contains("json_schema_response_format_repaired_total{repair=\"name_defaulted\"} 1"));
        assert!(rendered.contains(
            "json_schema_response_format_repaired_total{repair=\"bare_schema_wrapped\"} 1"
        ));
    }
}
