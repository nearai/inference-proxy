use super::*;

/// The built-in defaults: no per-model variable is set.
fn defaults() -> ModelDefaults {
    ModelDefaults {
        admission_queue_saturated_at: 1,
        reasoning_off_effort: "none".to_string(),
        ..Default::default()
    }
}

fn process() -> ProcessSettings {
    ProcessSettings {
        admission_ramp_step: 8,
        admission_ramp_interval_secs: 1800,
        admission_backpressure_secs: 10,
        admission_retry_after_secs: 2,
        bills_usage: true,
    }
}

fn lookup(name: &str) -> Option<String> {
    match name {
        "VLLM_BACKEND_TOKEN_ALPHA" => Some("alpha-secret".to_string()),
        "VLLM_BACKEND_TOKEN_BETA" => Some(" beta-secret ".to_string()),
        "VLLM_BACKEND_TOKEN_BLANK" => Some("  ".to_string()),
        _ => None,
    }
}

fn parse_with(
    list: serde_json::Value,
    defaults: &ModelDefaults,
    process: &ProcessSettings,
) -> anyhow::Result<Vec<ModelConfig>> {
    parse(&list.to_string(), defaults, process, &lookup)
}

fn parse_list(list: serde_json::Value) -> anyhow::Result<Vec<ModelConfig>> {
    parse_with(list, &defaults(), &process())
}

fn error_of(list: serde_json::Value) -> String {
    parse_list(list)
        .expect_err("the list must be refused")
        .to_string()
}

/// The three-model list of `docs/gateway-mode.md`, read from the document
/// itself so the example operators copy is one this code accepts.
fn documented_list() -> serde_json::Value {
    let doc = include_str!("../docs/gateway-mode.md");
    let (_, after) = doc
        .split_once("Three models, one with a long-context tier and two plain ones:")
        .expect("the example's lead-in");
    let (_, after) = after.split_once("```json\n").expect("the example");
    let (example, _) = after.split_once("```").expect("the example's end");
    serde_json::from_str(example).expect("the example is JSON")
}

#[test]
fn the_documented_list_resolves_to_three_models() {
    let models = parse_list(documented_list()).unwrap();
    assert_eq!(models.len(), 3);

    let alpha = &models[0];
    assert_eq!(alpha.id, "example/alpha");
    assert_eq!(alpha.backend_urls.len(), 3);
    assert_eq!(alpha.backend_probe_urls.len(), 3);
    assert_eq!(
        alpha.backend_long_context_urls,
        ["https://alpha-long-b4.inference.example"]
    );
    assert_eq!(
        alpha.pool_probe_urls().last().unwrap(),
        "http://alpha-4.internal.example:8000"
    );
    assert_eq!(alpha.long_context_above_tokens, 100_000);
    assert!(alpha.backend_tier_strict);
    assert!(alpha.admission_tier_borrowing);
    assert_eq!(alpha.admission_long_max_inflight_per_host, 12);
    assert_eq!(alpha.admission_long_reserved_inflight, 12);
    assert_eq!(alpha.admission_max_inflight, 48);
    assert_eq!(alpha.admission_start_inflight, 48);
    assert_eq!(alpha.admission_queue_saturated_at, 4);
    assert_eq!(alpha.capacity_requests_per_minute, 150);
    assert_eq!(alpha.discount_to_user, Some(0.3));
    assert_eq!(alpha.reasoning_off_effort, "low");
    assert_eq!(alpha.backend_token.as_deref(), Some("alpha-secret"));
    assert_eq!(alpha.backend_priority, Some(-1));

    // A plain model: its own budget and token, everything else the built-in
    // defaults, and no tier at all.
    let beta = &models[1];
    assert_eq!(beta.admission_max_inflight, 16);
    assert_eq!(beta.admission_start_inflight, 16);
    assert_eq!(beta.admission_queue_saturated_at, 1);
    assert_eq!(beta.backend_token.as_deref(), Some("beta-secret"));
    assert!(beta.backend_long_context_urls.is_empty());
    assert_eq!(beta.long_context_above_tokens, 0);
    assert!(!beta.backend_tier_strict);
    assert_eq!(beta.discount_to_user, None);
    assert_eq!(beta.reasoning_off_effort, "none");

    // `id` and `backend_urls` alone are a model.
    let gamma = &models[2];
    assert_eq!(gamma.backend_urls, ["https://gamma-b1.inference.example"]);
    assert_eq!(gamma.admission_max_inflight, 0);
    assert_eq!(gamma.backend_token, None);
    assert_eq!(gamma.backend_priority, None);
}

#[test]
fn omitted_keys_fall_back_to_the_process_level_values() {
    let defaults = ModelDefaults {
        long_context_above_tokens: 100_000,
        backend_tier_strict: true,
        admission_max_inflight: 48,
        admission_start_inflight: Some(24),
        admission_queue_saturated_at: 4,
        admission_tier_borrowing: true,
        admission_long_max_inflight_per_host: 12,
        admission_long_reserved_inflight: 8,
        capacity_requests_per_minute: 150,
        discount_to_user: Some(0.2),
        reasoning_off_effort: "low".to_string(),
        backend_token: Some("shared-secret".to_string()),
        backend_priority: Some(-1),
    };
    let models = parse_with(
        serde_json::json!({"models": [
            {
                "id": "tiered",
                "backend_urls": ["https://a.example"],
                "long_context": {"backend_urls": ["https://a-long.example"]}
            },
            {"id": "plain", "backend_urls": ["https://b.example"]},
            {
                "id": "own",
                "backend_urls": ["https://c.example"],
                "admission_max_inflight": 0,
                "admission_queue_saturated_at": 2,
                "capacity_requests_per_minute": 0,
                "discount_to_user": 0,
                "reasoning_off_effort": "none",
                "backend_token_env": "VLLM_BACKEND_TOKEN_ALPHA",
                "backend_priority": 5
            }
        ]}),
        &defaults,
        &process(),
    )
    .unwrap();

    let tiered = &models[0];
    assert_eq!(tiered.long_context_above_tokens, 100_000);
    assert!(tiered.backend_tier_strict);
    assert!(tiered.admission_tier_borrowing);
    assert_eq!(tiered.admission_long_max_inflight_per_host, 12);
    assert_eq!(tiered.admission_long_reserved_inflight, 8);
    assert_eq!(tiered.admission_max_inflight, 48);
    assert_eq!(tiered.admission_start_inflight, 24);
    assert_eq!(tiered.admission_queue_saturated_at, 4);
    assert_eq!(tiered.capacity_requests_per_minute, 150);
    assert_eq!(tiered.discount_to_user, Some(0.2));
    assert_eq!(tiered.reasoning_off_effort, "low");
    assert_eq!(tiered.backend_token.as_deref(), Some("shared-secret"));
    assert_eq!(tiered.backend_priority, Some(-1));

    // The tier variables are defaults for a tier, not a tier: a model
    // without `long_context` has none, whatever they say.
    let plain = &models[1];
    assert_eq!(plain.long_context_above_tokens, 0);
    assert!(!plain.backend_tier_strict);
    assert!(!plain.admission_tier_borrowing);
    assert_eq!(plain.admission_long_max_inflight_per_host, 0);
    assert_eq!(plain.admission_long_reserved_inflight, 0);
    assert_eq!(plain.admission_max_inflight, 48);
    assert_eq!(plain.discount_to_user, Some(0.2));

    // An entry's own value wins, including the ones that switch a default off.
    let own = &models[2];
    assert_eq!(own.admission_max_inflight, 0);
    assert_eq!(own.admission_queue_saturated_at, 2);
    assert_eq!(own.capacity_requests_per_minute, 0);
    assert_eq!(own.discount_to_user, None);
    assert_eq!(own.reasoning_off_effort, "none");
    assert_eq!(own.backend_token.as_deref(), Some("alpha-secret"));
    assert_eq!(own.backend_priority, Some(5));
}

#[test]
fn start_inflight_is_the_entrys_then_the_variable_then_the_models_own_maximum() {
    let start_of = |entry: serde_json::Value, variable: Option<u32>| {
        let defaults = ModelDefaults {
            admission_max_inflight: 48,
            admission_start_inflight: variable,
            ..defaults()
        };
        parse_with(
            serde_json::json!({"models": [entry]}),
            &defaults,
            &process(),
        )
        .map(|models| models[0].admission_start_inflight)
    };
    let entry = |extra: serde_json::Value| {
        let mut entry = serde_json::json!({"id": "m", "backend_urls": ["https://a.example"]});
        entry
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        entry
    };
    assert_eq!(start_of(entry(serde_json::json!({})), None).unwrap(), 48);
    assert_eq!(
        start_of(entry(serde_json::json!({})), Some(24)).unwrap(),
        24
    );
    assert_eq!(
        start_of(
            entry(serde_json::json!({"admission_max_inflight": 16})),
            None
        )
        .unwrap(),
        16
    );
    assert_eq!(
        start_of(
            entry(serde_json::json!({"admission_start_inflight": 8})),
            Some(24)
        )
        .unwrap(),
        8
    );
    // A smaller budget under a larger process-level start is not clamped
    // silently: the entry has to say where it starts.
    let error = start_of(
        entry(serde_json::json!({"admission_max_inflight": 16})),
        Some(24),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("model \"m\"") && error.contains("VLLM_PROXY_ADMISSION_START_INFLIGHT"),
        "{error}"
    );
}

#[test]
fn a_backend_token_is_read_from_the_named_variable_and_never_printed() {
    let models = parse_list(documented_list()).unwrap();
    let printed = format!("{models:?}");
    assert!(!printed.contains("alpha-secret"), "{printed}");
    assert!(!printed.contains("beta-secret"), "{printed}");
    assert!(
        printed.contains("backend_token: Some(\"<set>\")"),
        "{printed}"
    );
    assert!(printed.contains("backend_token: None"), "{printed}");
}

fn one(extra: serde_json::Value) -> serde_json::Value {
    let mut entry = serde_json::json!({"id": "m", "backend_urls": ["https://a.example"]});
    entry
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    serde_json::json!({"models": [entry]})
}

fn tiered(tier: serde_json::Value, extra: serde_json::Value) -> serde_json::Value {
    let mut long_context = serde_json::json!({
        "backend_urls": ["https://a-long.example"],
        "above_tokens": 100000
    });
    long_context
        .as_object_mut()
        .unwrap()
        .extend(tier.as_object().unwrap().clone());
    let mut extra = extra;
    extra["long_context"] = long_context;
    one(extra)
}

#[test]
fn a_list_that_is_not_a_model_list_is_refused() {
    for (text, expected) in [
        ("", "not a valid model list"),
        ("[]", "not a valid model list"),
        ("{}", "missing field `models`"),
        ("{\"models\": {}}", "not a valid model list"),
        (
            "{\"models\": [{\"backend_urls\": [\"https://a.example\"]}]}",
            "missing field `id`",
        ),
        (
            "{\"models\": [{\"id\": \"m\"}]}",
            "missing field `backend_urls`",
        ),
        (
            "{\"models\": [], \"defaults\": {}}",
            "unknown field `defaults`",
        ),
    ] {
        let error = parse(text, &defaults(), &process(), &lookup)
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{text}: {error}");
    }
    assert!(error_of(serde_json::json!({"models": []})).contains("`models` is empty"));
}

#[test]
fn a_mistyped_key_is_refused_rather_than_defaulted() {
    let error = error_of(one(serde_json::json!({"admission_max_inflght": 48})));
    assert!(
        error.contains("unknown field `admission_max_inflght`"),
        "{error}"
    );
    let error = error_of(tiered(
        serde_json::json!({"above_token": 1}),
        serde_json::json!({}),
    ));
    assert!(error.contains("unknown field `above_token`"), "{error}");
    // The file holds the name of the token's variable, never a token.
    let error = error_of(one(serde_json::json!({"backend_token": "secret"})));
    assert!(error.contains("unknown field `backend_token`"), "{error}");
}

#[test]
fn ids_must_be_exact_and_unique() {
    for id in ["", " m", "m ", "m\n"] {
        let error = error_of(serde_json::json!({"models": [
            {"id": id, "backend_urls": ["https://a.example"]}
        ]}));
        assert!(
            error.contains("exact cloud-api model name"),
            "{id:?}: {error}"
        );
    }
    let error = error_of(serde_json::json!({"models": [
        {"id": "m", "backend_urls": ["https://a.example"]},
        {"id": "m", "backend_urls": ["https://b.example"]}
    ]}));
    assert!(
        error.contains("model \"m\" is listed more than once"),
        "{error}"
    );
    // Ids are compared byte for byte, so these are two models.
    parse_list(serde_json::json!({"models": [
        {"id": "org/m", "backend_urls": ["https://a.example"]},
        {"id": "org/M", "backend_urls": ["https://b.example"]}
    ]}))
    .unwrap();
}

#[test]
fn a_model_needs_backends_and_one_probe_for_each() {
    let error = error_of(serde_json::json!({"models": [{"id": "m", "backend_urls": []}]}));
    assert!(
        error.contains("model \"m\"") && error.contains("`backend_urls` is empty"),
        "{error}"
    );
    for url in ["", "a.example", "ftp://a.example"] {
        let error = error_of(serde_json::json!({"models": [{"id": "m", "backend_urls": [url]}]}));
        assert!(error.contains("is not an http(s) URL"), "{url:?}: {error}");
    }
    let error = error_of(one(serde_json::json!({
        "backend_probe_urls": ["http://p1.example:8000", "http://p2.example:8000"]
    })));
    assert!(
        error.contains(
            "VLLM_BACKEND_PROBE_URLS must list one probe URL per VLLM_BACKEND_URLS entry"
        ),
        "{error}"
    );
}

#[test]
fn the_tier_rules_of_the_variables_hold_for_an_entry() {
    let cases = [
        // A tier without a threshold, from the entry or the variable.
        (
            one(serde_json::json!({"long_context": {"backend_urls": ["https://a-long.example"]}})),
            "VLLM_BACKEND_LONG_CONTEXT_URLS requires VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS",
        ),
        (
            one(serde_json::json!({"long_context": {"backend_urls": [], "above_tokens": 1}})),
            "`long_context.backend_urls` is empty",
        ),
        (
            tiered(serde_json::json!({"borrowing": true}), serde_json::json!({})),
            "VLLM_PROXY_ADMISSION_TIER_BORROWING requires admission",
        ),
        (
            tiered(
                serde_json::json!({"borrowing": true}),
                serde_json::json!({"admission_max_inflight": 8}),
            ),
            "VLLM_PROXY_ADMISSION_TIER_BORROWING requires admission",
        ),
        (
            tiered(
                serde_json::json!({"reserved_inflight": 2}),
                serde_json::json!({"admission_max_inflight": 8}),
            ),
            "VLLM_PROXY_ADMISSION_LONG_RESERVED_INFLIGHT requires VLLM_PROXY_ADMISSION_TIER_BORROWING",
        ),
        (
            tiered(
                serde_json::json!({"borrowing": true, "max_inflight_per_host": 4, "reserved_inflight": 8}),
                serde_json::json!({"admission_max_inflight": 8}),
            ),
            "VLLM_PROXY_ADMISSION_LONG_RESERVED_INFLIGHT must be below VLLM_PROXY_ADMISSION_START_INFLIGHT",
        ),
        (
            tiered(
                serde_json::json!({"backend_urls": ["https://a.example"]}),
                serde_json::json!({}),
            ),
            "is listed in both VLLM_BACKEND_URLS and VLLM_BACKEND_LONG_CONTEXT_URLS",
        ),
        // Probes for one tier only.
        (
            tiered(
                serde_json::json!({}),
                serde_json::json!({"backend_probe_urls": ["http://p1.example:8000"]}),
            ),
            "VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS must list one probe URL per",
        ),
        (
            tiered(
                serde_json::json!({"backend_probe_urls": ["http://p2.example:8000"]}),
                serde_json::json!({}),
            ),
            "VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS must list one probe URL per",
        ),
        (
            tiered(
                serde_json::json!({"backend_probe_urls": ["http://p1.example:8000"]}),
                serde_json::json!({"backend_probe_urls": ["http://p1.example:8000"]}),
            ),
            "is listed in both VLLM_BACKEND_PROBE_URLS and VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS",
        ),
    ];
    for (list, expected) in cases {
        let error = error_of(list.clone());
        assert!(
            error.contains("model \"m\"") && error.contains(expected),
            "{list}: {error}"
        );
    }
    // And a complete tier passes them.
    parse_list(tiered(
        serde_json::json!({
            "backend_probe_urls": ["http://p2.example:8000"],
            "strict": true,
            "borrowing": true,
            "max_inflight_per_host": 4,
            "reserved_inflight": 4
        }),
        serde_json::json!({
            "backend_probe_urls": ["http://p1.example:8000"],
            "admission_max_inflight": 8
        }),
    ))
    .unwrap();
}

#[test]
fn the_admission_rules_of_the_variables_hold_for_an_entry() {
    for (extra, expected) in [
        (
            serde_json::json!({"admission_max_inflight": 8, "admission_start_inflight": 9}),
            "VLLM_PROXY_ADMISSION_START_INFLIGHT must be between 1 and VLLM_PROXY_ADMISSION_MAX_INFLIGHT",
        ),
        (
            serde_json::json!({"admission_max_inflight": 8, "admission_start_inflight": 0}),
            "VLLM_PROXY_ADMISSION_START_INFLIGHT must be between 1 and VLLM_PROXY_ADMISSION_MAX_INFLIGHT",
        ),
        (
            serde_json::json!({"admission_max_inflight": 8, "admission_queue_saturated_at": 0}),
            "VLLM_PROXY_ADMISSION_QUEUE_SATURATED_AT must be at least 1",
        ),
    ] {
        let error = error_of(one(extra.clone()));
        assert!(
            error.contains("model \"m\"") && error.contains(expected),
            "{extra}: {error}"
        );
    }
    // The process-level half is checked for every entry that has a budget,
    // even when the process-level budget itself is off.
    let error = parse_with(
        one(serde_json::json!({"admission_max_inflight": 8})),
        &defaults(),
        &ProcessSettings {
            admission_retry_after_secs: 0,
            ..process()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("VLLM_PROXY_ADMISSION_RETRY_AFTER_SECS must be at least 1"),
        "{error}"
    );
    // Without a budget there is nothing to check.
    parse_list(one(serde_json::json!({"admission_queue_saturated_at": 0}))).unwrap();
}

#[test]
fn discount_priority_and_reasoning_effort_are_validated() {
    for (extra, expected) in [
        (
            serde_json::json!({"discount_to_user": 1}),
            "VLLM_PROXY_DISCOUNT_TO_USER must be a number in [0, 1)",
        ),
        (
            serde_json::json!({"discount_to_user": -0.1}),
            "VLLM_PROXY_DISCOUNT_TO_USER must be a number in [0, 1)",
        ),
        (
            serde_json::json!({"discount_to_user": 0.12345}),
            "VLLM_PROXY_DISCOUNT_TO_USER must have at most four decimal places",
        ),
        (
            serde_json::json!({"discount_to_user": "0.3"}),
            "not a valid model list",
        ),
        (
            serde_json::json!({"backend_priority": 1001}),
            "VLLM_BACKEND_PRIORITY: priority must be within",
        ),
        (
            serde_json::json!({"reasoning_off_effort": " "}),
            "`reasoning_off_effort` is empty",
        ),
    ] {
        let error = error_of(one(extra.clone()));
        assert!(error.contains(expected), "{extra}: {error}");
    }
}

/// `one(extra)` with a backend token, which a model with a map needs.
fn one_lane(extra: serde_json::Value) -> serde_json::Value {
    let mut extra = extra;
    extra["backend_token_env"] = serde_json::json!("VLLM_BACKEND_TOKEN_ALPHA");
    one(extra)
}

#[test]
fn a_reasoning_effort_map_is_read_from_the_entry_that_writes_one() {
    // The documented example: alpha maps one effort, the others none.
    let models = parse_list(documented_list()).unwrap();
    assert_eq!(models[0].reasoning_effort_map.get("high"), Some("xhigh"));
    assert_eq!(models[0].reasoning_effort_map.get("xhigh"), None);
    assert_eq!(models[0].reasoning_effort_map.get("low"), None);
    assert!(models[1].reasoning_effort_map.is_empty());
    assert!(models[2].reasoning_effort_map.is_empty());
    let printed = format!("{:?}", models[0]);
    assert!(
        printed.contains(r#"reasoning_effort_map: {"high": "xhigh"}"#),
        "{printed}"
    );

    // Several efforts may share a target; an empty object, `null` and no key
    // at all are a model without a map, token or not.
    let models = parse_list(one_lane(serde_json::json!({
        "reasoning_effort_map": {"high": "xhigh", "max": "xhigh", "medium": "low"}
    })))
    .unwrap();
    let map = &models[0].reasoning_effort_map;
    assert_eq!(
        [map.get("high"), map.get("max"), map.get("medium")],
        [Some("xhigh"), Some("xhigh"), Some("low")]
    );
    for extra in [
        serde_json::json!({"reasoning_effort_map": {}}),
        serde_json::json!({"reasoning_effort_map": null}),
        serde_json::json!({}),
    ] {
        let models = parse_list(one(extra.clone())).unwrap();
        assert!(models[0].reasoning_effort_map.is_empty(), "{extra}");
    }
}

#[test]
fn a_reasoning_effort_map_of_another_shape_is_refused() {
    for map in [
        serde_json::json!("high=xhigh"),
        serde_json::json!(["high", "xhigh"]),
        serde_json::json!([{"high": "xhigh"}]),
        serde_json::json!({"high": 1}),
        serde_json::json!({"high": null}),
        serde_json::json!({"high": ["xhigh"]}),
        serde_json::json!({"high": {"effort": "xhigh"}}),
        serde_json::json!(true),
    ] {
        let error = error_of(one_lane(serde_json::json!({"reasoning_effort_map": map})));
        assert!(error.contains("not a valid model list"), "{map}: {error}");
    }
    // A key written twice is not settled by which one came last. (Written
    // out: a `json!` object cannot hold it.)
    let error = parse(
        r#"{"models": [{
            "id": "m",
            "backend_urls": ["https://a.example"],
            "backend_token_env": "VLLM_BACKEND_TOKEN_ALPHA",
            "reasoning_effort_map": {"high": "xhigh", "high": "max"}
        }]}"#,
        &defaults(),
        &process(),
        &lookup,
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("not a valid model list")
            && error.contains("`reasoning_effort_map` has the same key more than once"),
        "{error}"
    );
    // It is a key of a model, not of its tier or of the file.
    for list in [
        tiered(
            serde_json::json!({"reasoning_effort_map": {"high": "xhigh"}}),
            serde_json::json!({}),
        ),
        serde_json::json!({
            "models": [{"id": "m", "backend_urls": ["https://a.example"]}],
            "reasoning_effort_map": {"high": "xhigh"}
        }),
    ] {
        let error = error_of(list);
        assert!(
            error.contains("unknown field `reasoning_effort_map`"),
            "{error}"
        );
    }
}

#[test]
fn a_reasoning_effort_map_is_validated() {
    for (map, expected) in [
        // A chain: the result would depend on the order of two lookups.
        (
            serde_json::json!({"high": "xhigh", "xhigh": "max"}),
            "\"xhigh\" is both a key and a value",
        ),
        (
            serde_json::json!({"high": "high"}),
            "\"high\" is both a key and a value",
        ),
        (
            serde_json::json!({"": "xhigh"}),
            "every key and value must be 1 to 32 characters",
        ),
        (
            serde_json::json!({"high": ""}),
            "every key and value must be 1 to 32 characters",
        ),
        (
            serde_json::json!({"high": "x high"}),
            "every key and value must be 1 to 32 characters",
        ),
        (
            serde_json::json!({"high": "x".repeat(33)}),
            "every key and value must be 1 to 32 characters",
        ),
        // The off values and the off effort belong to `reasoning_off_effort`
        // (the built-in `none` here).
        (
            serde_json::json!({"none": "low"}),
            "\"none\" cannot be a key",
        ),
        (
            serde_json::json!({"minimal": "low"}),
            "\"minimal\" cannot be a key",
        ),
        (
            serde_json::json!({"high": "minimal"}),
            "\"high\" cannot become \"minimal\"",
        ),
    ] {
        let error = error_of(one_lane(
            serde_json::json!({"reasoning_effort_map": map.clone()}),
        ));
        assert!(
            error.contains("model \"m\": `reasoning_effort_map`") && error.contains(expected),
            "{map}: {error}"
        );
    }
    let too_many: serde_json::Map<String, serde_json::Value> = (0..17)
        .map(|n| (format!("effort-{n}"), serde_json::json!("xhigh")))
        .collect();
    let error = error_of(one_lane(
        serde_json::json!({"reasoning_effort_map": too_many}),
    ));
    assert!(error.contains("more than 16 entries"), "{error}");
}

#[test]
fn a_reasoning_effort_map_is_checked_against_the_off_effort_the_model_resolves_to() {
    let low_by_default = ModelDefaults {
        reasoning_off_effort: "low".to_string(),
        ..defaults()
    };
    let parsed = |extra: serde_json::Value, defaults: &ModelDefaults| {
        parse_with(one_lane(extra), defaults, &process())
            .map(|models| models[0].clone())
            .map_err(|error| error.to_string())
    };

    // The entry's own off effort.
    let error = parsed(
        serde_json::json!({
            "reasoning_off_effort": "low",
            "reasoning_effort_map": {"low": "medium"}
        }),
        &defaults(),
    )
    .unwrap_err();
    assert!(
        error.contains("\"low\" cannot be a key: it is this model's `reasoning_off_effort`"),
        "{error}"
    );
    let error = parsed(
        serde_json::json!({
            "reasoning_off_effort": "low",
            "reasoning_effort_map": {"high": "none"}
        }),
        &defaults(),
    )
    .unwrap_err();
    assert!(
        error.contains("\"high\" cannot become \"none\"") && error.contains("(\"low\")"),
        "{error}"
    );

    // The process-level one, for an entry that leaves it out ...
    let error = parsed(
        serde_json::json!({"reasoning_effort_map": {"low": "medium"}}),
        &low_by_default,
    )
    .unwrap_err();
    assert!(error.contains("\"low\" cannot be a key"), "{error}");
    // ... and not for one that has its own: `low` is an effort like any
    // other for this model, and `none` is its off effort.
    let model = parsed(
        serde_json::json!({
            "reasoning_off_effort": "none",
            "reasoning_effort_map": {"low": "medium", "max": "none"}
        }),
        &low_by_default,
    )
    .unwrap();
    assert_eq!(model.reasoning_off_effort, "none");
    assert_eq!(model.reasoning_effort_map.get("low"), Some("medium"));
    assert_eq!(model.reasoning_effort_map.get("max"), Some("none"));
    // The model's off effort as a target, when it is not an off value.
    let model = parsed(
        serde_json::json!({"reasoning_effort_map": {"medium": "low"}}),
        &low_by_default,
    )
    .unwrap();
    assert_eq!(model.reasoning_effort_map.get("medium"), Some("low"));
}

#[test]
fn a_reasoning_effort_map_requires_a_backend_token() {
    // The reasoning handling only runs for a model with a backend token, so
    // a map without one would be configured and never applied.
    let error = error_of(one(
        serde_json::json!({"reasoning_effort_map": {"high": "xhigh"}}),
    ));
    assert!(
        error.contains("model \"m\": `reasoning_effort_map` requires a backend token"),
        "{error}"
    );
    // The model's own token or the process-level one.
    parse_list(one_lane(
        serde_json::json!({"reasoning_effort_map": {"high": "xhigh"}}),
    ))
    .unwrap();
    let with_default_token = ModelDefaults {
        backend_token: Some("shared-secret".to_string()),
        ..defaults()
    };
    parse_with(
        one(serde_json::json!({"reasoning_effort_map": {"high": "xhigh"}})),
        &with_default_token,
        &process(),
    )
    .unwrap();
}

#[test]
fn a_token_variable_must_be_of_the_backend_token_family_and_set() {
    let error = error_of(one(
        serde_json::json!({"backend_token_env": "VLLM_BACKEND_TOKEN_UNSET"}),
    ));
    assert!(
        error.contains("model \"m\"")
            && error
                .contains("`backend_token_env` names VLLM_BACKEND_TOKEN_UNSET, which is not set"),
        "{error}"
    );
    let error = error_of(one(
        serde_json::json!({"backend_token_env": "VLLM_BACKEND_TOKEN_BLANK"}),
    ));
    assert!(error.contains("which is not set"), "{error}");
    // Another secret of the process, or the token pasted where its
    // variable's name belongs: refused, and not repeated in the message.
    for name in [
        "CLOUD_API_USAGE_TOKEN",
        "TOKEN",
        "sk-live-pasted-secret",
        "VLLM_BACKEND_TOKEN pasted-secret",
        "",
    ] {
        let error = error_of(one(serde_json::json!({"backend_token_env": name})));
        assert!(
            error.contains(
                "must be the name of an environment variable starting with VLLM_BACKEND_TOKEN"
            ),
            "{name:?}: {error}"
        );
        // (`TOKEN` alone is part of the family's own name.)
        assert!(
            name.is_empty() || name == "TOKEN" || !error.contains(name),
            "{error}"
        );
    }
}

#[test]
fn a_backend_token_requires_a_process_that_can_bill() {
    let cannot_bill = ProcessSettings {
        bills_usage: false,
        ..process()
    };
    let with_default_token = ModelDefaults {
        backend_token: Some("shared-secret".to_string()),
        ..defaults()
    };
    for (list, defaults) in [
        (
            one(serde_json::json!({"backend_token_env": "VLLM_BACKEND_TOKEN_ALPHA"})),
            defaults(),
        ),
        (one(serde_json::json!({})), with_default_token),
    ] {
        let error = parse_with(list, &defaults, &cannot_bill)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("VLLM_BACKEND_TOKEN requires CLOUD_API_URL and CLOUD_API_USAGE_TOKEN"),
            "{error}"
        );
    }
    // No token, nothing to bill for.
    parse_with(one(serde_json::json!({})), &defaults(), &cannot_bill).unwrap();
}

#[test]
fn a_backend_or_a_probe_belongs_to_one_model() {
    let error = error_of(serde_json::json!({"models": [
        {"id": "a", "backend_urls": ["https://shared.example", "https://a.example"]},
        {"id": "b", "backend_urls": ["https://shared.example/"]}
    ]}));
    assert!(
        error.contains("backend URL https://shared.example is listed under both \"a\" and \"b\""),
        "{error}"
    );
    // One model's long-context host is not another model's base host either.
    let error = error_of(serde_json::json!({"models": [
        {
            "id": "a",
            "backend_urls": ["https://a.example"],
            "long_context": {"backend_urls": ["https://shared.example"], "above_tokens": 1}
        },
        {"id": "b", "backend_urls": ["https://shared.example"]}
    ]}));
    assert!(
        error.contains("is listed under both \"a\" and \"b\""),
        "{error}"
    );
    let error = error_of(serde_json::json!({"models": [
        {
            "id": "a",
            "backend_urls": ["https://a.example"],
            "backend_probe_urls": ["http://shared.example:8000"]
        },
        {
            "id": "b",
            "backend_urls": ["https://b.example"],
            "backend_probe_urls": ["http://shared.example:8000"]
        }
    ]}));
    assert!(
        error.contains("probe URL http://shared.example:8000 is listed under both \"a\" and \"b\""),
        "{error}"
    );
}

#[test]
fn one_backend_under_two_spellings_is_still_one_backend() {
    let two = |a: &str, b: &str, key: &str| {
        let mut model_a = serde_json::json!({"id": "a", "backend_urls": ["https://a.example"]});
        let mut model_b = serde_json::json!({"id": "b", "backend_urls": ["https://b.example"]});
        model_a[key] = serde_json::json!([a]);
        model_b[key] = serde_json::json!([b]);
        serde_json::json!({"models": [model_a, model_b]})
    };
    // The host in another case, the scheme's own port written out, a path
    // that resolves to the same place, a trailing dot on the host, and an
    // address written another way: one endpoint each.
    let same = [
        ("http://shared.example:8000", "http://SHARED.example:8000"),
        ("https://shared.example", "https://shared.example:443"),
        ("http://shared.example", "http://shared.example:80/"),
        ("https://shared.example", "https://shared.example/v1/.."),
        ("https://shared.example/v1", "https://shared.example/v1/./"),
        ("https://shared.example", "https://shared.example."),
        ("https://shared.example", "HTTPS://Shared.Example"),
        ("http://127.0.0.1:8000", "http://127.1:8000"),
    ];
    for key in ["backend_urls", "backend_probe_urls"] {
        let what = if key == "backend_urls" {
            "backend URL"
        } else {
            "probe URL"
        };
        for (first, second) in same {
            let error = error_of(two(first, second, key));
            assert!(
                error.contains(what)
                    && error.contains(&format!("is the same endpoint as {first}"))
                    && error.contains("is listed under both \"a\" and \"b\""),
                "{first} / {second}: {error}"
            );
        }
    }
    // The long-context tier of one model against the base tier of another.
    let error = error_of(serde_json::json!({"models": [
        {
            "id": "a",
            "backend_urls": ["https://a.example"],
            "long_context": {"backend_urls": ["https://Shared.example:443"], "above_tokens": 1}
        },
        {"id": "b", "backend_urls": ["https://shared.example"]}
    ]}));
    assert!(
        error.contains("is the same endpoint as https://Shared.example:443"),
        "{error}"
    );

    // Another port, scheme, path or host is another backend.
    let different = [
        ("https://shared.example", "https://shared.example:8443"),
        ("http://shared.example", "https://shared.example"),
        ("https://shared.example/a", "https://shared.example/b"),
        ("https://shared.example", "https://shared.example.org"),
        ("http://127.0.0.1:8000", "http://127.0.0.2:8000"),
    ];
    for (first, second) in different {
        let models = parse_list(two(first, second, "backend_urls"))
            .unwrap_or_else(|error| panic!("{first} / {second}: {error}"));
        // The URLs are kept as they were written; only the comparison is
        // on where they point.
        assert_eq!(models[0].backend_urls, [first]);
        assert_eq!(models[1].backend_urls, [second]);
    }
}

#[test]
fn one_backend_serves_one_tier_however_it_is_spelled() {
    // The tier rules of the variables, with the comparison a list makes.
    let error = error_of(tiered(
        serde_json::json!({"backend_urls": ["https://A.example:443/"]}),
        serde_json::json!({}),
    ));
    assert!(
        error.contains("model \"m\"")
            && error
                .contains("is listed in both VLLM_BACKEND_URLS and VLLM_BACKEND_LONG_CONTEXT_URLS"),
        "{error}"
    );
    let error = error_of(tiered(
        serde_json::json!({"backend_probe_urls": ["http://P1.example:8000"]}),
        serde_json::json!({"backend_probe_urls": ["http://p1.example:8000/"]}),
    ));
    assert!(
        error.contains(
            "is listed in both VLLM_BACKEND_PROBE_URLS and VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS"
        ),
        "{error}"
    );
}

#[test]
fn a_backend_url_is_a_base_url_without_credentials() {
    for url in [
        "https://a.example?x=1",
        "https://a.example/v1?x=1",
        "https://a.example#top",
    ] {
        let error = error_of(serde_json::json!({"models": [{"id": "m", "backend_urls": [url]}]}));
        assert!(
            error.contains("must be a base URL, without a query or fragment"),
            "{url}: {error}"
        );
    }
    // Credentials do not belong in the file, and are not repeated.
    for url in [
        "https://operator:pasted-secret@a.example",
        "https://pasted-secret@a.example",
    ] {
        for key in ["backend_urls", "backend_probe_urls"] {
            let mut model = serde_json::json!({"id": "m", "backend_urls": ["https://a.example"]});
            model[key] = serde_json::json!([url]);
            let error = error_of(serde_json::json!({"models": [model]}));
            assert!(
                error.contains(&format!("`{key}`: a URL must not carry credentials")),
                "{error}"
            );
            assert!(!error.contains("pasted-secret"), "{error}");
        }
    }
}

#[test]
fn a_requested_model_is_exact_a_case_variant_another_string_or_missing() {
    let ids = ["org/Model-A", "org/model-b"];
    let class = |requested: Option<&str>| classify(requested, ids.iter().copied());
    assert_eq!(class(Some("org/Model-A")), ModelMatch::Exact);
    assert_eq!(class(Some("org/model-b")), ModelMatch::Exact);
    assert_eq!(class(Some("org/model-a")), ModelMatch::CaseDiffers);
    assert_eq!(class(Some("ORG/MODEL-B")), ModelMatch::CaseDiffers);
    assert_eq!(class(Some("org/model-c")), ModelMatch::Other);
    assert_eq!(class(Some("")), ModelMatch::Other);
    assert_eq!(class(Some(" org/model-b")), ModelMatch::Other);
    assert_eq!(class(None), ModelMatch::Missing);

    // Only a string is a model name.
    let requested = |body: serde_json::Value| requested_model(&body).map(str::to_string);
    assert_eq!(
        requested(serde_json::json!({"model": "m"})).as_deref(),
        Some("m")
    );
    for body in [
        serde_json::json!({}),
        serde_json::json!({"model": null}),
        serde_json::json!({"model": 7}),
        serde_json::json!({"model": ["m"]}),
        serde_json::json!({"model": {"id": "m"}}),
        serde_json::json!([]),
    ] {
        assert_eq!(requested(body.clone()), None, "{body}");
    }
    assert_eq!(
        [
            ModelMatch::Exact,
            ModelMatch::CaseDiffers,
            ModelMatch::Other,
            ModelMatch::Missing
        ]
        .map(ModelMatch::as_str),
        ["exact", "case_differs", "other", "missing"]
    );
}

#[derive(Clone, Default)]
struct Logs(Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for Logs {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn a_single_models_log_lines_carry_no_model_field() {
    let admission = || {
        Some(crate::admission::AdmissionConfig {
            max_inflight: 1,
            tier_borrowing: false,
            long_max_inflight_per_host: 0,
            long_reserved_inflight: 0,
            start_inflight: 1,
            ramp_step: 8,
            ramp_interval: Duration::from_secs(1800),
            ttft_p95_max: None,
            backpressure_ttl: Duration::from_secs(10),
            queue_saturated_at: 1,
            retry_after: Duration::from_secs(2),
        })
    };
    let engine = || Arc::new(EngineLoad::disabled());
    // The lines a lane writes are shared code: the same call, once for the
    // single model of a process and once for an entry of a list.
    let refuse_once_each = || {
        AdmissionController::new(admission(), 1, engine())
            .reject(crate::admission::RejectReason::Budget);
        AdmissionController::for_model(Some("example/alpha"), admission(), 1, engine())
            .reject(crate::admission::RejectReason::Budget);
    };

    // tracing caches a callsite's interest for the whole process, and while
    // exactly one subscriber is registered it derives that from the calling
    // thread's default: another test's thread, which has none, would switch
    // these shared callsites off for this capture too. A second registered
    // dispatcher makes it ask every registered subscriber instead.
    let _registered = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());

    let logs = Logs::default();
    let writer = logs.clone();
    let guard = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(false)
            .without_time()
            .with_writer(move || writer.clone())
            .finish(),
    );
    refuse_once_each();
    drop(guard);
    let captured = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    assert_eq!(
        captured.lines().collect::<Vec<_>>(),
        [
            "DEBUG vllm_proxy_rs::admission: Lane request refused at admission reason=\"budget\"",
            "DEBUG vllm_proxy_rs::admission: Lane request refused at admission model=\"example/alpha\" reason=\"budget\"",
        ]
    );

    // And as JSON (`LOG_FORMAT=json`).
    let logs = Logs::default();
    let writer = logs.clone();
    let guard = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .json()
            .with_max_level(tracing::Level::DEBUG)
            .without_time()
            .with_writer(move || writer.clone())
            .finish(),
    );
    refuse_once_each();
    drop(guard);
    let captured = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    let fields: Vec<serde_json::Value> = captured
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap()["fields"].take())
        .collect();
    assert_eq!(
        fields,
        [
            serde_json::json!({"message": "Lane request refused at admission", "reason": "budget"}),
            serde_json::json!({
                "message": "Lane request refused at admission",
                "model": "example/alpha",
                "reason": "budget"
            }),
        ]
    );
}
