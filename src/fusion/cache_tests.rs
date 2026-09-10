use super::Usage;
use serde_json::{json, Value};

#[test]
fn fusion_cache_usage_aggregates_components_once() {
    let mut aggregate = Usage::default();
    for (input, output, cached) in [(10, 2, 6), (11, 3, 7), (20, 4, 15), (30, 6, 24)] {
        aggregate.add(&Usage::from_response(&json!({
            "usage": {
                "prompt_tokens": input,
                "completion_tokens": output,
                "prompt_tokens_details": {"cached_tokens": cached}
            }
        })));
    }

    assert_eq!(
        aggregate.to_json(),
        json!({
            "prompt_tokens": 71,
            "completion_tokens": 15,
            "total_tokens": 86,
            "prompt_tokens_details": {"cached_tokens": 52}
        })
    );
}

#[test]
fn fusion_cache_usage_normalizes_each_component_before_aggregation() {
    let cases = [
        (Value::Null, 0),
        (json!(0), 0),
        (json!(-3), 0),
        (json!(7), 7),
        (json!(30), 10),
        (json!("7"), 0),
        (json!(7.5), 0),
        (json!(i64::from(i32::MAX) + 1), 0),
        (json!(u64::MAX), 0),
    ];
    let mut aggregate = Usage::default();
    for (cached, expected) in cases {
        let usage = Usage::from_response(&json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": cached}
            }
        }));
        assert_eq!(
            usage.to_json()["prompt_tokens_details"]["cached_tokens"],
            expected,
            "cached observation: {cached}"
        );
        aggregate.add(&usage);
    }
    assert_eq!(aggregate.to_json()["prompt_tokens"], 90);
    assert_eq!(
        aggregate.to_json()["prompt_tokens_details"]["cached_tokens"],
        17
    );
}

#[test]
fn fusion_cache_usage_preserves_primary_counter_semantics() {
    for input in [-3, 0, 10] {
        let usage = Usage::from_response(&json!({
            "usage": {"prompt_tokens": input, "completion_tokens": -2}
        }));
        assert_eq!(usage.to_json()["prompt_tokens"], input);
        assert_eq!(usage.to_json()["completion_tokens"], -2);
        assert_eq!(usage.to_json()["prompt_tokens_details"]["cached_tokens"], 0);
    }
    let usage = Usage::from_response(&json!({
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 4}
        }
    }));
    assert_eq!(usage.to_json()["prompt_tokens"], -1);
    assert_eq!(usage.to_json()["prompt_tokens_details"]["cached_tokens"], 0);
}
