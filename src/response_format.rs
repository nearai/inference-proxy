use serde_json::Value;

const DEFAULT_JSON_SCHEMA_NAME: &str = "response_schema";

pub fn default_json_schema_name(request: &mut Value) -> bool {
    let Some(response_format) = request
        .get_mut("response_format")
        .and_then(Value::as_object_mut)
    else {
        return false;
    };
    if response_format.get("type").and_then(Value::as_str) != Some("json_schema") {
        return false;
    }

    let Some(json_schema) = response_format
        .get_mut("json_schema")
        .and_then(Value::as_object_mut)
    else {
        return false;
    };
    let Some(schema) = json_schema.get("schema") else {
        return false;
    };
    if !(schema.is_object() || schema.is_boolean()) || json_schema.contains_key("name") {
        return false;
    }

    json_schema.insert(
        "name".to_string(),
        Value::String(DEFAULT_JSON_SCHEMA_NAME.to_string()),
    );
    metrics::counter!("json_schema_names_defaulted_total").increment(1);
    true
}

#[cfg(test)]
mod tests {
    use super::default_json_schema_name;
    use serde_json::json;

    #[test]
    fn defaults_object_schema_and_is_idempotent() {
        let mut request = json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}, "strict": true}
            }
        });

        assert!(default_json_schema_name(&mut request));
        assert_eq!(
            request["response_format"]["json_schema"]["name"],
            "response_schema"
        );
        assert_eq!(
            request["response_format"]["json_schema"]["schema"],
            json!({"type": "object"})
        );
        assert_eq!(request["response_format"]["json_schema"]["strict"], true);
        assert!(!default_json_schema_name(&mut request));
    }

    #[test]
    fn defaults_boolean_schema() {
        let mut request = json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": true}
            }
        });

        assert!(default_json_schema_name(&mut request));
        assert_eq!(
            request["response_format"]["json_schema"]["name"],
            "response_schema"
        );
        assert_eq!(request["response_format"]["json_schema"]["schema"], true);
    }

    #[test]
    fn counts_insertions_without_counting_idempotent_calls() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let mut request = json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {}}
            }
        });

        metrics::with_local_recorder(&recorder, || {
            assert!(default_json_schema_name(&mut request));
            assert!(!default_json_schema_name(&mut request));
        });

        assert!(recorder
            .handle()
            .render()
            .contains("json_schema_names_defaulted_total 1"));
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
            let mut request = json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": {},
                        "name": name,
                        "strict": true,
                        "description": "A named response"
                    }
                }
            });
            let original = request.clone();
            assert!(!default_json_schema_name(&mut request));
            assert_eq!(request, original);
        }
    }

    #[test]
    fn ignores_malformed_schema_wrappers_and_other_formats() {
        for response_format in [
            json!({"type": "json_schema", "json_schema": {}}),
            json!({"type": "json_schema", "json_schema": {"schema": null}}),
            json!({"type": "json_schema", "json_schema": {"schema": []}}),
            json!({"type": "json_schema", "json_schema": {"schema": "object"}}),
            json!({"type": "json_schema", "json_schema": []}),
            json!({"type": "json_object"}),
            json!({"type": "text"}),
        ] {
            let mut request = json!({"response_format": response_format});
            assert!(!default_json_schema_name(&mut request));
            assert_eq!(request["response_format"], response_format);
        }
    }
}
