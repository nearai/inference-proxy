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
//!
//! The efforts a chat template accepts are model-specific too: one that
//! knows `low`, `medium` and `xhigh` answers a 400 to `high`, which callers
//! of an aggregator send routinely. A model of a gateway list can therefore
//! name the efforts its engine refuses and what to send in their place
//! (`reasoning_effort_map`, `EffortMap`). The map runs on what the switch
//! above left in the body, and on both places the engine reads an effort
//! from, so no value it names reaches the engine. Whatever means "off" stays
//! with the switch: the map can neither take an off value nor change the
//! model's off effort (`EffortMap::new`).

use std::collections::BTreeMap;

use serde_json::Value;

use crate::model_metrics::{model_counter, ModelLabel};

/// Effort values that mean "as little reasoning as possible".
const OFF_EFFORTS: [&str; 2] = ["none", "minimal"];

/// Entries a `reasoning_effort_map` may have. Its targets are a metric label
/// (`apply_effort_map`), so this bounds that label's values per model.
const EFFORT_MAP_MAX_ENTRIES: usize = 16;

/// Longest effort a `reasoning_effort_map` may name, in bytes.
const EFFORT_MAX_LEN: usize = 32;

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

/// A model's `reasoning_effort_map` (gateway list mode): efforts its engine
/// refuses, each with the effort sent in its place. Empty for a model without
/// the key, and for the one model of a process without a list.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct EffortMap(BTreeMap<String, String>);

impl EffortMap {
    /// The map of a model whose "off" is `off_effort`, from the pairs its
    /// list entry wrote. Refused, so that a list never starts with one:
    ///
    /// - a key or a target that is not 1 to 32 characters of `a-z`, `0-9`,
    ///   `_` and `-`, a key written twice, more than 16 entries;
    /// - a chain, i.e. a target that is also a key (`{"a": "b", "b": "c"}`,
    ///   or `{"a": "a"}`): every effort is looked up once, so what a request
    ///   is sent with never depends on an order;
    /// - `none` or `minimal` as a key: those mean "off", which is
    ///   `off_effort` (`apply_reasoning_switch`);
    /// - `off_effort` itself as a key: it is what the switch writes for
    ///   "off", and mapping it would send something else than the configured
    ///   off effort;
    /// - `none` or `minimal` as a target, unless it is `off_effort`: the
    ///   switch replaces those with `off_effort`, and the map must not write
    ///   one back after it.
    ///
    /// Together these keep the two mechanisms apart: the map never changes
    /// what the switch wrote for "off", and never writes an off value the
    /// switch would have replaced.
    pub fn new(pairs: Vec<(String, String)>, off_effort: &str) -> anyhow::Result<Self> {
        const KEY: &str = "`reasoning_effort_map`";
        if pairs.len() > EFFORT_MAP_MAX_ENTRIES {
            anyhow::bail!("{KEY} has more than {EFFORT_MAP_MAX_ENTRIES} entries");
        }
        let mut map = BTreeMap::new();
        for (from, to) in pairs {
            // Not echoed: a string of the wrong shape can be anything.
            if !is_effort(&from) || !is_effort(&to) {
                anyhow::bail!(
                    "{KEY}: every key and value must be 1 to {EFFORT_MAX_LEN} characters of a-z, 0-9, `_` or `-`"
                );
            }
            if OFF_EFFORTS.contains(&from.as_str()) {
                anyhow::bail!(
                    "{KEY}: {from:?} cannot be a key: `none` and `minimal` become `reasoning_off_effort`"
                );
            }
            if from == off_effort {
                anyhow::bail!(
                    "{KEY}: {from:?} cannot be a key: it is this model's `reasoning_off_effort`, which is sent as it is"
                );
            }
            if OFF_EFFORTS.contains(&to.as_str()) && to != off_effort {
                anyhow::bail!(
                    "{KEY}: {from:?} cannot become {to:?}: `none` and `minimal` are sent as this model's `reasoning_off_effort` ({off_effort:?})"
                );
            }
            if map.contains_key(&from) {
                anyhow::bail!("{KEY}: {from:?} is a key more than once");
            }
            map.insert(from, to);
        }
        if let Some(chained) = map.values().find(|to| map.contains_key(*to)) {
            anyhow::bail!(
                "{KEY}: {chained:?} is both a key and a value: an effort is mapped once, never through a chain"
            );
        }
        Ok(Self(map))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// What `effort` is replaced with, if the map names it. An exact match,
    /// like the off values.
    pub fn get(&self, effort: &str) -> Option<&str> {
        self.0.get(effort).map(String::as_str)
    }

    /// Replace `effort` in place when it is a string the map names.
    fn rewrite(&self, effort: &mut Value) -> Option<&str> {
        let target = self.get(effort.as_str()?)?;
        *effort = Value::String(target.to_string());
        Some(target)
    }
}

/// As the map itself, `{"high": "xhigh"}`: this is what the startup line of
/// a model prints.
impl std::fmt::Debug for EffortMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// An effort a map may name: short, and of the characters effort names are
/// made of. A target is sent to the engine and is a metric label value.
fn is_effort(value: &str) -> bool {
    (1..=EFFORT_MAX_LEN).contains(&value.len())
        && value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// Replace the efforts `map` names in the two places the engine reads one
/// from: the top-level `reasoning_effort` and `reasoning.effort`. Runs after
/// `apply_reasoning_switch`, on what it left, so the switch's copy of
/// `reasoning.effort` into `reasoning_effort` is covered and the off values
/// are already the model's off effort, which no map names. Each field is
/// looked up on its own: a caller that sent two different efforts gets each
/// one mapped or passed through, and the body never holds a value the map
/// names, whichever field the engine prefers. Anything else (an effort the
/// map does not name, a value that is not a string, no effort at all) is
/// left as it is, and no field is added.
///
/// A request the map changed is counted once in
/// `inference_proxy_model_reasoning_effort_mapped_total{effort, model}`:
/// `model` is the configured id (`ModelView::label`) and `effort` the value
/// that was sent instead, which is returned. Both are configuration; the
/// caller's own effort is never a label. When the two fields were mapped to
/// different values it is the object's, the one the engine reads first.
pub fn apply_effort_map<'a>(
    request_json: &mut Value,
    map: &'a EffortMap,
    model: ModelLabel,
) -> Option<&'a str> {
    let body = request_json.as_object_mut()?;
    let in_object = body
        .get_mut("reasoning")
        .and_then(Value::as_object_mut)
        .and_then(|reasoning| reasoning.get_mut("effort"))
        .and_then(|effort| map.rewrite(effort));
    let top_level = body
        .get_mut("reasoning_effort")
        .and_then(|effort| map.rewrite(effort));
    let target = in_object.or(top_level)?;
    model_counter!(
        model,
        "inference_proxy_model_reasoning_effort_mapped_total",
        "effort" => target.to_string(),
    )
    .increment(1);
    Some(target)
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

    fn pairs(entries: &[(&str, &str)]) -> Vec<(String, String)> {
        entries
            .iter()
            .map(|(from, to)| (from.to_string(), to.to_string()))
            .collect()
    }

    /// The map of a model whose engine knows `low`, `medium` and `xhigh`.
    fn high_to_xhigh() -> EffortMap {
        EffortMap::new(pairs(&[("high", "xhigh")]), "low").unwrap()
    }

    /// What the gateway does to a body: the switch, then the map.
    fn switch_then_map(req: &mut Value, off_effort: &str, map: &EffortMap) -> Option<String> {
        apply_reasoning_switch(req, off_effort);
        apply_effort_map(req, map, Some("example/alpha")).map(str::to_string)
    }

    /// No effort the map names is left where the engine reads one.
    fn assert_no_effort(req: &Value, unmapped: &str) {
        assert_ne!(req["reasoning_effort"], unmapped, "{req}");
        assert_ne!(req["reasoning"]["effort"], unmapped, "{req}");
    }

    #[test]
    fn a_mapped_effort_is_replaced_in_both_request_shapes() {
        let map = high_to_xhigh();

        // The OpenAI shape: nothing but the top-level value, and no object
        // is invented for it.
        let mut req = json!({"model": "m", "reasoning_effort": "high"});
        assert_eq!(
            switch_then_map(&mut req, "low", &map).as_deref(),
            Some("xhigh")
        );
        assert_eq!(req, json!({"model": "m", "reasoning_effort": "xhigh"}));

        // The aggregator's object: the switch copies its effort to the top
        // level, and both places then carry the mapped value.
        let mut req = json!({"reasoning": {"effort": "high", "exclude": true}});
        assert_eq!(
            switch_then_map(&mut req, "low", &map).as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            req,
            json!({
                "reasoning": {"effort": "xhigh", "exclude": true},
                "reasoning_effort": "xhigh"
            })
        );

        // Both at once, as an aggregator that sets the two fields sends it.
        let mut req = json!({"reasoning_effort": "high", "reasoning": {"effort": "high"}});
        assert_eq!(
            switch_then_map(&mut req, "low", &map).as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            req,
            json!({"reasoning_effort": "xhigh", "reasoning": {"effort": "xhigh"}})
        );
    }

    #[test]
    fn each_field_is_mapped_on_its_own() {
        let map = EffortMap::new(pairs(&[("high", "xhigh"), ("max", "medium")]), "low").unwrap();

        // An explicit `reasoning_effort` keeps the switch out, so the object
        // still holds what the caller wrote: mapped where the map names it...
        let mut req = json!({"reasoning_effort": "high", "reasoning": {"effort": "max"}});
        assert_eq!(
            switch_then_map(&mut req, "low", &map).as_deref(),
            Some("medium"),
            "the object's target is the one reported"
        );
        assert_eq!(
            req,
            json!({"reasoning_effort": "xhigh", "reasoning": {"effort": "medium"}})
        );

        // ... and left alone where it does not.
        let mut req = json!({"reasoning_effort": "high", "reasoning": {"effort": "low"}});
        assert_eq!(
            switch_then_map(&mut req, "low", &map).as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            req,
            json!({"reasoning_effort": "xhigh", "reasoning": {"effort": "low"}})
        );
        let mut req = json!({"reasoning_effort": "low", "reasoning": {"effort": "high"}});
        assert_eq!(
            switch_then_map(&mut req, "low", &map).as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            req,
            json!({"reasoning_effort": "low", "reasoning": {"effort": "xhigh"}})
        );
        for mapped in ["high", "max"] {
            assert_no_effort(&req, mapped);
        }
    }

    #[test]
    fn an_effort_the_map_does_not_name_passes_through() {
        let map = high_to_xhigh();
        for effort in ["low", "medium", "xhigh", "max", "High", " high", "highest"] {
            let mut req = json!({"reasoning_effort": effort});
            assert_eq!(switch_then_map(&mut req, "low", &map), None, "{effort}");
            assert_eq!(req, json!({"reasoning_effort": effort}));

            let mut req = json!({"reasoning": {"effort": effort}});
            assert_eq!(switch_then_map(&mut req, "low", &map), None, "{effort}");
            assert_eq!(
                req,
                json!({"reasoning": {"effort": effort}, "reasoning_effort": effort})
            );
        }
        // Nothing to look up: no effort, or one that is not a string.
        for req in [
            json!({"model": "m"}),
            json!({"reasoning": {"enabled": true}}),
            json!({"reasoning": "high"}),
            json!({"reasoning": {"effort": 3}, "reasoning_effort": ["high"]}),
            json!({"reasoning_effort": null}),
            json!("high"),
        ] {
            let mut req = req;
            let before = req.clone();
            assert_eq!(apply_effort_map(&mut req, &map, None), None, "{before}");
            assert_eq!(req, before);
        }
        // An empty map changes nothing either.
        let mut req = json!({"reasoning_effort": "high"});
        assert_eq!(
            apply_effort_map(&mut req, &EffortMap::default(), None),
            None
        );
        assert_eq!(req, json!({"reasoning_effort": "high"}));
    }

    #[test]
    fn off_values_stay_with_the_switch() {
        let map = high_to_xhigh();
        // Every way of asking for no reasoning still ends as the model's off
        // effort, which the map cannot name.
        for req in [
            json!({"reasoning": {"enabled": false}}),
            json!({"reasoning": {"enabled": false, "effort": "high"}}),
            json!({"reasoning": {"effort": "none"}}),
            json!({"reasoning": {"effort": "minimal"}}),
            json!({"reasoning_effort": "none"}),
            json!({"reasoning_effort": "minimal", "reasoning": {"effort": "high"}}),
        ] {
            let mut req = req;
            let asked = req.clone();
            assert_eq!(switch_then_map(&mut req, "low", &map), None, "{asked}");
            assert_eq!(req["reasoning_effort"], "low", "{asked}");
            if asked.get("reasoning").is_some() {
                assert_eq!(req["reasoning"]["effort"], "low", "{asked}");
            }
            assert_no_effort(&req, "high");
        }

        // A map may send an effort to the model's own off effort, and that
        // is then all the engine sees of it.
        let to_off = EffortMap::new(pairs(&[("low", "none")]), "none").unwrap();
        let mut req = json!({"reasoning": {"effort": "low"}});
        assert_eq!(
            switch_then_map(&mut req, "none", &to_off).as_deref(),
            Some("none")
        );
        assert_eq!(
            req,
            json!({"reasoning": {"effort": "none"}, "reasoning_effort": "none"})
        );
    }

    #[test]
    fn a_mapped_request_is_counted_once_under_its_model_and_target() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let map = high_to_xhigh();
            for req in [
                json!({"reasoning_effort": "high"}),
                // Two fields rewritten, one request.
                json!({"reasoning_effort": "high", "reasoning": {"effort": "high"}}),
                json!({"reasoning_effort": "medium"}),
                json!({"reasoning_effort": "caller-supplied-effort"}),
            ] {
                let mut req = req;
                apply_effort_map(&mut req, &map, Some("example/alpha"));
            }
        });
        let rendered = handle.render();
        assert!(
            rendered.contains(
                "inference_proxy_model_reasoning_effort_mapped_total{effort=\"xhigh\",model=\"example/alpha\"} 2"
            ),
            "{rendered}"
        );
        // One series: what the caller sent is not a label, mapped or not.
        let series = rendered
            .lines()
            .filter(|line| line.starts_with("inference_proxy_model_reasoning_effort_mapped_total"))
            .count();
        assert_eq!(series, 1, "{rendered}");
        assert!(!rendered.contains("caller-supplied-effort"), "{rendered}");
        assert!(!rendered.contains("\"high\""), "{rendered}");
    }

    #[test]
    fn a_map_is_refused_when_it_is_malformed_or_chained() {
        let long = "x".repeat(EFFORT_MAX_LEN + 1);
        for (entries, expected) in [
            (vec![("", "xhigh")], "every key and value must be 1 to 32"),
            (vec![("high", "")], "every key and value must be 1 to 32"),
            (vec![("high", " ")], "every key and value must be 1 to 32"),
            (
                vec![(" high", "xhigh")],
                "every key and value must be 1 to 32",
            ),
            (
                vec![("High", "xhigh")],
                "every key and value must be 1 to 32",
            ),
            (
                vec![("high", "x high")],
                "every key and value must be 1 to 32",
            ),
            (
                vec![("high", "x\"high")],
                "every key and value must be 1 to 32",
            ),
            (
                vec![("high", "xhigh\n")],
                "every key and value must be 1 to 32",
            ),
            (
                vec![(long.as_str(), "xhigh")],
                "every key and value must be 1 to 32",
            ),
            (
                vec![("high", long.as_str())],
                "every key and value must be 1 to 32",
            ),
            // A chain, in either order, and the shortest one.
            (
                vec![("high", "xhigh"), ("xhigh", "max")],
                "\"xhigh\" is both a key and a value",
            ),
            (
                vec![("xhigh", "max"), ("high", "xhigh")],
                "\"xhigh\" is both a key and a value",
            ),
            (
                vec![("high", "medium"), ("medium", "high")],
                "is both a key and a value",
            ),
            (vec![("high", "high")], "\"high\" is both a key and a value"),
            (
                vec![("high", "xhigh"), ("high", "max")],
                "\"high\" is a key more than once",
            ),
        ] {
            let error = EffortMap::new(pairs(&entries), "low")
                .expect_err("the map must be refused")
                .to_string();
            assert!(
                error.starts_with("`reasoning_effort_map`") && error.contains(expected),
                "{entries:?}: {error}"
            );
            // A string of the wrong shape is not repeated.
            assert!(!error.contains(&long), "{error}");
        }

        let too_many: Vec<(String, String)> = (0..=EFFORT_MAP_MAX_ENTRIES)
            .map(|n| (format!("effort-{n}"), "xhigh".to_string()))
            .collect();
        let error = EffortMap::new(too_many.clone(), "low")
            .unwrap_err()
            .to_string();
        assert!(error.contains("more than 16 entries"), "{error}");
        // The limit itself is fine, and so is every character an effort is
        // written with, at the longest length.
        EffortMap::new(too_many[1..].to_vec(), "low").unwrap();
        let longest = "a-z_09".repeat(6)[..EFFORT_MAX_LEN].to_string();
        let map = EffortMap::new(pairs(&[(&longest, "x_high-2")]), "low").unwrap();
        assert_eq!(map.get(&longest), Some("x_high-2"));
        assert!(EffortMap::new(Vec::new(), "low").unwrap().is_empty());
    }

    #[test]
    fn a_map_that_contradicts_the_off_handling_is_refused() {
        for (entries, off_effort, expected) in [
            // The off values belong to the switch.
            (
                vec![("none", "low")],
                "low",
                "\"none\" cannot be a key: `none` and `minimal` become `reasoning_off_effort`",
            ),
            (
                vec![("minimal", "low")],
                "none",
                "\"minimal\" cannot be a key",
            ),
            // The off effort is sent as configured, not mapped again.
            (
                vec![("low", "medium")],
                "low",
                "\"low\" cannot be a key: it is this model's `reasoning_off_effort`",
            ),
            // No off value the switch would have replaced is put back.
            (
                vec![("high", "none")],
                "low",
                "\"high\" cannot become \"none\"",
            ),
            (
                vec![("high", "minimal")],
                "none",
                "\"high\" cannot become \"minimal\"",
            ),
            (
                vec![("high", "none")],
                "minimal",
                "\"high\" cannot become \"none\"",
            ),
        ] {
            let error = EffortMap::new(pairs(&entries), off_effort)
                .expect_err("the map must be refused")
                .to_string();
            assert!(
                error.starts_with("`reasoning_effort_map`") && error.contains(expected),
                "{entries:?} with off {off_effort:?}: {error}"
            );
        }
        // The model's own off effort is a fine target, whatever it is called.
        for off_effort in ["none", "minimal", "low"] {
            let map = EffortMap::new(pairs(&[("high", off_effort)]), off_effort).unwrap();
            assert_eq!(map.get("high"), Some(off_effort));
        }
    }

    #[test]
    fn a_map_prints_as_its_entries() {
        let map = EffortMap::new(pairs(&[("max", "xhigh"), ("high", "xhigh")]), "low").unwrap();
        assert_eq!(format!("{map:?}"), r#"{"high": "xhigh", "max": "xhigh"}"#);
        assert_eq!(format!("{:?}", EffortMap::default()), "{}");
    }
}
