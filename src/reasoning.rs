//! An aggregator's reasoning controls, mapped to the engine's switch.
//!
//! OpenRouter sends `{"reasoning": {"enabled": false}}` or
//! `{"reasoning": {"effort": "none" | "minimal" | "low" | ...}}`, while the
//! engines only read the OpenAI-style top-level `reasoning_effort`, which the
//! chat template turns into the model's thinking budget. Left alone, the
//! object is ignored: the model keeps thinking, the caller pays for reasoning
//! tokens it asked not to have, and a small `max_tokens` comes back with
//! empty `content`.
//!
//! "Off" is model-specific. GLM-5.3 Flash's template only knows `low` and
//! `high` (anything else means max), and with thinking switched off outright
//! the model writes its reasoning as visible content. So the value that
//! stands for "off" is configuration (`VLLM_PROXY_REASONING_OFF_EFFORT`,
//! default `none`, `low` for that model), applied to `enabled: false`, to an
//! `effort` of `none` or `minimal`, and to the same values sent as
//! `reasoning_effort` directly. Other efforts are copied as they are; a
//! caller's explicit `reasoning_effort` outside the off values is respected.
//! The mapped value is written to `reasoning_effort` and back into
//! `reasoning.effort`: the engine reads the object's field first when both
//! are present, so leaving `none` there would undo the mapping.
//! `exclude` and `max_tokens` have no engine equivalent and are left to the
//! aggregator. Gateway mode only.

use serde_json::Value;

/// Effort values that mean "as little reasoning as possible".
const OFF_EFFORTS: [&str; 2] = ["none", "minimal"];

/// Map the caller's reasoning intent onto `reasoning_effort`, using
/// `off_effort` as the model's cleanest minimum. Returns what was applied,
/// for the caller's metrics.
pub fn apply_reasoning_switch(request_json: &mut Value, off_effort: &str) -> Option<&'static str> {
    let body = request_json.as_object_mut()?;
    let explicit = body
        .get("reasoning_effort")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let reasoning = body.get("reasoning").and_then(|r| r.as_object());
    let (effort, kind) = match (explicit.as_deref(), reasoning) {
        (Some(value), _) if OFF_EFFORTS.contains(&value) && value != off_effort => {
            (off_effort.to_string(), "off_effort")
        }
        (Some(_), _) => return None,
        (None, Some(reasoning)) if reasoning.get("enabled") == Some(&Value::Bool(false)) => {
            (off_effort.to_string(), "disabled")
        }
        (None, Some(reasoning)) => match reasoning.get("effort").and_then(|e| e.as_str()) {
            Some(value) if OFF_EFFORTS.contains(&value) => (off_effort.to_string(), "disabled"),
            Some(value) => (value.to_string(), "effort"),
            None => return None,
        },
        (None, None) => return None,
    };
    // The engine reads the object's own `effort` first when both are present,
    // so the mapped value goes into both places.
    if let Some(reasoning) = body.get_mut("reasoning").and_then(|r| r.as_object_mut()) {
        reasoning.insert("effort".to_string(), Value::String(effort.clone()));
    }
    body.insert("reasoning_effort".to_string(), Value::String(effort));
    metrics::counter!("reasoning_switch_applied_total", "kind" => kind).increment(1);
    Some(kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn enabled_false_becomes_the_off_effort() {
        let mut req = json!({"model": "m", "reasoning": {"enabled": false}});
        assert_eq!(apply_reasoning_switch(&mut req, "low"), Some("disabled"));
        assert_eq!(req["reasoning_effort"], "low");
        assert_eq!(req["reasoning"]["enabled"], false, "the object stays");
        assert_eq!(
            req["reasoning"]["effort"], "low",
            "the object carries the mapped effort too"
        );

        let mut req = json!({"reasoning": {"enabled": false, "effort": "high"}});
        assert_eq!(apply_reasoning_switch(&mut req, "none"), Some("disabled"));
        assert_eq!(req["reasoning_effort"], "none");
        assert_eq!(req["reasoning"]["effort"], "none");
    }

    #[test]
    fn off_efforts_become_the_off_effort_and_others_are_copied() {
        for effort in ["none", "minimal"] {
            let mut req = json!({"reasoning": {"effort": effort, "exclude": true}});
            assert_eq!(apply_reasoning_switch(&mut req, "low"), Some("disabled"));
            assert_eq!(req["reasoning_effort"], "low");
            assert_eq!(req["reasoning"]["effort"], "low");
            assert_eq!(req["reasoning"]["exclude"], true);
        }
        for effort in ["low", "medium", "high", "xhigh", "max"] {
            let mut req = json!({"reasoning": {"effort": effort}});
            assert_eq!(apply_reasoning_switch(&mut req, "low"), Some("effort"));
            assert_eq!(req["reasoning_effort"], effort);
        }
    }

    #[test]
    fn an_explicit_reasoning_effort_is_respected_unless_it_means_off() {
        let mut req = json!({"reasoning_effort": "high", "reasoning": {"enabled": false}});
        assert_eq!(apply_reasoning_switch(&mut req, "low"), None);
        assert_eq!(req["reasoning_effort"], "high");

        let mut req = json!({"reasoning_effort": "none"});
        assert_eq!(apply_reasoning_switch(&mut req, "low"), Some("off_effort"));
        assert_eq!(req["reasoning_effort"], "low");
        assert!(req.get("reasoning").is_none(), "no object is invented");

        let mut req = json!({"reasoning_effort": "minimal", "reasoning": {"effort": "minimal"}});
        assert_eq!(apply_reasoning_switch(&mut req, "low"), Some("off_effort"));
        assert_eq!(req["reasoning_effort"], "low");
        assert_eq!(req["reasoning"]["effort"], "low");

        // With the default off value nothing needs rewriting.
        let mut req = json!({"reasoning_effort": "none"});
        assert_eq!(apply_reasoning_switch(&mut req, "none"), None);
        assert_eq!(req["reasoning_effort"], "none");
    }

    #[test]
    fn nothing_to_map_is_a_noop() {
        for req in [
            json!({"model": "m"}),
            json!({"reasoning": {"enabled": true}}),
            json!({"reasoning": {"exclude": true}}),
            json!({"reasoning": {"max_tokens": 0}}),
            json!({"reasoning": "none"}),
            json!({"reasoning": {"effort": 3}}),
            json!("not an object"),
        ] {
            let mut req = req;
            let before = req.clone();
            assert_eq!(apply_reasoning_switch(&mut req, "low"), None, "{before}");
            assert_eq!(req, before);
        }
    }
}
