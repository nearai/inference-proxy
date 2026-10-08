//! Gateway list mode: one process serving several models
//! (`VLLM_PROXY_MODEL_LIST_FILE`), and what must not change for a process
//! that serves one.
//!
//! A list gateway here is the one `main` runs: the environment and the list
//! file go through `Config::from_env`, the state comes from
//! `model_list::app_state` and the routes and middleware from
//! `routes::build_app`, the two functions `main` calls. Only the signing keys
//! and the HTTP clients are the test's own. One test starts the built binary
//! itself.

use std::collections::BTreeSet;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{any, body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use vllm_proxy_rs::*;

const ALPHA: &str = "example/alpha";
const BETA: &str = "example/beta";
const GAMMA: &str = "example/gamma";
const ROUTES: [&str; 2] = [routes::ROUTE_CHAT_COMPLETIONS, routes::ROUTE_COMPLETIONS];

/// `Config::from_env` reads the process environment, which every test of
/// this binary shares. Everything that reads or writes it — setting the
/// variables, parsing, and building HTTP clients (they look for proxy
/// variables) — happens under this lock, inside `start`.
static ENV: Mutex<()> = Mutex::new(());

struct Gateway {
    app: axum::Router,
    state: AppState,
}

impl Gateway {
    fn model(&self, id: &str) -> &model_list::ServedModel {
        self.state
            .models
            .as_ref()
            .expect("list mode")
            .get(id)
            .expect("configured model")
    }
}

/// A gateway serving `list` (list mode), with `env` on top of what list mode
/// requires.
fn start_list(list: &Value, env: &[(&str, &str)], metrics: Option<PrometheusHandle>) -> Gateway {
    start(Some(list), env, metrics)
}

/// A process serving one model from its environment: `env` names it.
fn start_single(env: &[(&str, &str)], metrics: Option<PrometheusHandle>) -> Gateway {
    start(None, env, metrics)
}

fn start(list: Option<&Value>, env: &[(&str, &str)], metrics: Option<PrometheusHandle>) -> Gateway {
    let _env = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let file = tempfile::NamedTempFile::new().unwrap();
    let mut vars: Vec<(String, String)> = [
        ("TOKEN", "test-token"),
        ("NON_TEE_DEPLOYMENT", "1"),
        ("DEV", "1"),
        ("GPU_NO_HW_MODE", "1"),
        ("CLOUD_API_AUTH_MAX_ATTEMPTS", "1"),
        ("VLLM_PROXY_IMAGE_VALIDATION_DISABLED", "1"),
        // The pool health checkers stay out of the way unless a test is
        // about them: a probe must not land on a mock that expects none.
        ("HEALTH_CHECK_INTERVAL_SECS", "3600"),
    ]
    .iter()
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .collect();
    if let Some(list) = list {
        file.as_file()
            .write_all(list.to_string().as_bytes())
            .unwrap();
        vars.push((
            model_list::MODEL_LIST_FILE_ENV.to_string(),
            file.path().to_str().unwrap().to_string(),
        ));
        // Required in list mode; a test that reads `/v1/models` names a real one.
        vars.push((
            "VLLM_PROXY_MODELS_DOCUMENT_URL".to_string(),
            "http://catalog.invalid/v1/models".to_string(),
        ));
    }
    vars.extend(
        env.iter()
            .map(|(key, value)| (key.to_string(), value.to_string())),
    );
    for (key, value) in &vars {
        std::env::set_var(key, value);
    }
    let config = config::Config::from_env();
    for (key, _) in &vars {
        std::env::remove_var(key);
    }
    let config = Arc::new(config.expect("valid configuration"));
    assert_eq!(config.model_list.is_some(), list.is_some());

    let ecdsa_key: [u8; 32] = [
        0xac, 0x09, 0x74, 0xbe, 0xc3, 0x9a, 0x17, 0xe3, 0x6b, 0xa4, 0xa6, 0xb4, 0xd2, 0x38, 0xff,
        0x94, 0x4b, 0xac, 0xb3, 0x5e, 0x5d, 0xc4, 0xaf, 0x0f, 0x33, 0x47, 0xe5, 0x87, 0x31, 0x79,
        0x67, 0x0f,
    ];
    let ed25519_key: [u8; 32] = [
        0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c,
        0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae,
        0x7f, 0x60,
    ];
    let process = model_list::Process {
        config,
        signing: signing::SigningPair {
            ecdsa: signing::EcdsaContext::from_key_bytes(&ecdsa_key).unwrap(),
            ed25519: signing::Ed25519Context::from_key_bytes(&ed25519_key).unwrap(),
        },
        http_client: reqwest::Client::new(),
        metrics_handle: metrics
            .unwrap_or_else(|| PrometheusBuilder::new().build_recorder().handle()),
        backend_client: &|headers| {
            Ok(reqwest::Client::builder()
                .default_headers(headers)
                .build()?)
        },
    };
    let state = if process.config.model_list.is_some() {
        model_list::app_state(process).expect("the list starts")
    } else {
        single_model_state(process)
    };
    Gateway {
        app: routes::build_app(state.clone()),
        state,
    }
}

/// The state of a process that serves one model, wired as `main` wires it.
/// (That wiring lives in the binary, so this is a copy of it, as in the
/// other test files; a list goes through `model_list::app_state` instead.)
fn single_model_state(process: model_list::Process<'_>) -> AppState {
    let model_list::Process {
        config,
        signing,
        http_client,
        metrics_handle,
        backend_client,
    } = process;
    let mut backend_headers = reqwest::header::HeaderMap::new();
    if let Some(token) = &config.backend_token {
        backend_headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
    }
    if let Some(priority) = config.backend_priority {
        backend_headers.insert(
            priority::PRIORITY_HEADER,
            reqwest::header::HeaderValue::from(priority),
        );
    }
    let backend_pool = Arc::new(backend_pool::BackendPool::with_long_context(
        config.backend_urls.clone(),
        config.backend_long_context_urls.clone(),
    ));
    let engine_load = Arc::new(engine_load::EngineLoad::new(
        backend_pool.len(),
        Duration::from_secs(config.backend_probe_interval_secs) * 3,
    ));
    AppState {
        signing: Arc::new(signing),
        cache: Arc::new(cache::ChatCache::new(
            &config.model_name,
            config.chat_cache_expiration_secs,
        )),
        attestation_cache: Arc::new(attestation::AttestationCache::new(300)),
        backend_client: if backend_headers.is_empty() {
            http_client.clone()
        } else {
            backend_client(backend_headers).unwrap()
        },
        http_client,
        metrics_handle,
        tls_cert_fingerprint: Arc::new(attestation::TlsCertTracker::new(None).unwrap()),
        admission: Arc::new(admission::AdmissionController::new(
            config.admission(),
            backend_pool.len(),
            engine_load,
        )),
        backend_affinity: Arc::new(backend_affinity::BackendConversationAffinity::new(
            config.backend_conversation_affinity,
            backend_pool.len(),
            config.backend_affinity_max_imbalance,
            config.chat_cache_expiration_secs,
        )),
        backend_pool,
        ohttp_gateway: None,
        ohttp_attestation_ed25519: None,
        fusion_caches: Arc::new(fusion::FusionCaches::default()),
        vllm_dp_affinity: Arc::new(vllm_dp_affinity::VllmDpAffinity::new(None, 1_200)),
        models: None,
        usage_report_delivery: usage_report::UsageReportDelivery::new(config.usage_report.clone()),
        config,
    }
}

fn completion_json() -> Value {
    json!({
        "id": "chatcmpl-list-1",
        "object": "chat.completion",
        "model": "served",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    })
}

/// Answer `expected` chat/completions requests on `route`.
async fn mount_completions(backend: &MockServer, route: &str, expected: u64) {
    Mock::given(method("POST"))
        .and(path(route))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_json()))
        .expect(expected)
        .mount(backend)
        .await;
}

/// A backend that must not be asked anything at all.
async fn expect_untouched(backend: &MockServer) {
    Mock::given(any())
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(backend)
        .await;
}

fn body_for(model: Option<Value>) -> Value {
    let mut body = json!({
        "messages": [{"role": "user", "content": "hello"}],
        "prompt": "hello"
    });
    if let Some(model) = model {
        body["model"] = model;
    }
    body
}

fn post(route: &str, bearer: Option<&str>, body: &Value) -> Request<Body> {
    let mut request = Request::builder()
        .method("POST")
        .uri(route)
        .header("content-type", "application/json");
    if let Some(bearer) = bearer {
        request = request.header("authorization", format!("Bearer {bearer}"));
    }
    request
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap()
}

/// A chat request for `model` from the operator's config token.
fn chat(model: &str) -> Request<Body> {
    post(
        routes::ROUTE_CHAT_COMPLETIONS,
        Some("test-token"),
        &body_for(Some(json!(model))),
    )
}

/// A request for `model` on `route` whose prompt is `bytes` long: a chat
/// message or a text prompt, estimated at a quarter of that in tokens.
fn sized(route: &str, bearer: &str, model: &str, bytes: usize) -> Request<Body> {
    let body = if route == routes::ROUTE_CHAT_COMPLETIONS {
        json!({"model": model, "messages": [{"role": "user", "content": "x".repeat(bytes)}]})
    } else {
        json!({"model": model, "prompt": "x".repeat(bytes)})
    };
    post(route, Some(bearer), &body)
}

/// `get` from the operator: the gateway's own config token as bearer.
fn get_as(route: &str, bearer: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(route)
        .header("authorization", format!("Bearer {bearer}"))
        .body(Body::empty())
        .unwrap()
}

fn get(route: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(route)
        .body(Body::empty())
        .unwrap()
}

async fn text_body(response: axum::response::Response) -> String {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8_lossy(&bytes).into_owned()
}

async fn json_body(response: axum::response::Response) -> Value {
    let text = text_body(response).await;
    serde_json::from_str(&text).unwrap_or_else(|_| panic!("non-JSON body: {text}"))
}

async fn assert_model_not_found(response: axum::response::Response) -> Value {
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = json_body(response).await;
    assert_eq!(body["error"]["code"], "model_not_found", "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert!(body["error"]["param"].is_null(), "{body}");
    body
}

/// A mock cloud-api: `key` is a valid customer key of organization `org`,
/// and usage reports are accepted under the shared usage token.
async fn cloud_api(key: &str) -> MockServer {
    let cloud = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .and(header("authorization", format!("Bearer {key}").as_str()))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "valid": true,
            "organization_id": "org",
            "workspace_id": "ws",
            "api_key_id": "k"
        })))
        .mount(&cloud)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/internal/usage"))
        .and(header("authorization", "Bearer usage-secret"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&cloud)
        .await;
    cloud
}

async fn requests_to(server: &MockServer, route: &str) -> Vec<wiremock::Request> {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|request| request.url.path() == route)
        .collect()
}

/// The usage reports posted so far, once there are `expected` of them.
async fn usage_reports(cloud: &MockServer, expected: usize) -> Vec<Value> {
    for _ in 0..150 {
        let reports = requests_to(cloud, "/v1/internal/usage").await;
        if reports.len() >= expected {
            return reports
                .iter()
                .map(|report| serde_json::from_slice(&report.body).unwrap())
                .collect();
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("expected {expected} usage reports");
}

fn unreachable_backend_url() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    format!("http://127.0.0.1:{port}")
}

/// Poll `ready` until it holds, for at most `secs` seconds.
async fn eventually(secs: u64, what: &str, mut ready: impl FnMut() -> bool) {
    for _ in 0..secs * 50 {
        if ready() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("timed out waiting for {what}");
}

fn bearer_of(request: &wiremock::Request) -> Option<String> {
    request
        .headers
        .get("authorization")
        .map(|value| value.to_str().unwrap().to_string())
}

// ---------------------------------------------------------------------------
// Model selection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_request_is_served_by_the_model_its_body_names() {
    let alpha = MockServer::start().await;
    let beta = MockServer::start().await;
    for (backend, id) in [(&alpha, ALPHA), (&beta, BETA)] {
        for route in ROUTES {
            // The body is forwarded with the id it arrived with.
            Mock::given(method("POST"))
                .and(path(route))
                .and(body_partial_json(json!({"model": id})))
                .respond_with(ResponseTemplate::new(200).set_body_json(completion_json()))
                .expect(1)
                .mount(backend)
                .await;
        }
    }
    let gateway = start_list(
        &json!({"models": [
            {"id": ALPHA, "backend_urls": [alpha.uri()]},
            {"id": BETA, "backend_urls": [beta.uri()]}
        ]}),
        &[],
        None,
    );
    for id in [ALPHA, BETA] {
        for route in ROUTES {
            let request = post(route, Some("test-token"), &body_for(Some(json!(id))));
            let response = gateway.app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{id} {route}");
        }
    }
    // Each backend saw exactly its own model's two requests.
    alpha.verify().await;
    beta.verify().await;
}

#[tokio::test]
async fn an_unknown_or_missing_model_is_a_404_and_nothing_is_dispatched_or_billed() {
    let alpha = MockServer::start().await;
    let beta = MockServer::start().await;
    expect_untouched(&alpha).await;
    expect_untouched(&beta).await;
    let cloud = cloud_api("sk-live-customer").await;
    let gateway = start_list(
        &json!({"models": [
            {"id": ALPHA, "backend_urls": [alpha.uri()], "admission_max_inflight": 1},
            {"id": BETA, "backend_urls": [beta.uri()], "admission_max_inflight": 1}
        ]}),
        &[
            ("CLOUD_API_URL", &cloud.uri()),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
        ],
        None,
    );
    // Every model's budget is in use for the whole test: a request that
    // needed a slot, of any model, would be a 429 here, never a 404.
    let held = [ALPHA, BETA].map(|id| {
        let model = gateway.model(id);
        model
            .admission
            .try_admit(&model.backend_pool, None)
            .unwrap()
            .expect("admission is on")
    });
    let response = gateway.app.clone().oneshot(chat(ALPHA)).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

    // Not a configured id — another model, or a configured one in another
    // case — and no usable `model` at all.
    let unknown = [
        json!(GAMMA),
        json!("Example/Alpha"),
        json!(""),
        json!(" example/alpha"),
    ];
    let missing = [
        None,
        Some(Value::Null),
        Some(json!(7)),
        Some(json!([ALPHA])),
    ];
    for route in ROUTES {
        for bearer in ["sk-live-customer", "test-token"] {
            for model in &unknown {
                let request = post(route, Some(bearer), &body_for(Some(model.clone())));
                let response = gateway.app.clone().oneshot(request).await.unwrap();
                let body = assert_model_not_found(response).await;
                let name = model.as_str().unwrap();
                assert_eq!(
                    body["error"]["message"],
                    format!("The model `{name}` does not exist or you do not have access to it."),
                );
            }
            for model in &missing {
                let request = post(route, Some(bearer), &body_for(model.clone()));
                let response = gateway.app.clone().oneshot(request).await.unwrap();
                assert_model_not_found(response).await;
            }
        }
    }

    // The 404 comes after authentication: without a valid key the answer is
    // the same 401 whether or not the model exists, so it tells nothing.
    for route in ROUTES {
        for model in [GAMMA, ALPHA] {
            for bearer in [None, Some("sk-live-unknown"), Some("not-a-key")] {
                let request = post(route, bearer, &body_for(Some(json!(model))));
                let response = gateway.app.clone().oneshot(request).await.unwrap();
                assert_eq!(
                    response.status(),
                    StatusCode::UNAUTHORIZED,
                    "{model} {bearer:?}"
                );
            }
        }
    }
    // A body that is not JSON is still a 400, as it is for a single model.
    let request = Request::builder()
        .method("POST")
        .uri(routes::ROUTE_CHAT_COMPLETIONS)
        .header("authorization", "Bearer test-token")
        .body(Body::from("not json"))
        .unwrap();
    let response = gateway.app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Nothing reached a backend, and nothing was billed. No slot was asked
    // for either: the budgets were full throughout, and every answer above
    // was the 404 or the 401, with only the held slots in flight.
    alpha.verify().await;
    beta.verify().await;
    for id in [ALPHA, BETA] {
        assert_eq!(gateway.model(id).admission.inflight(), 1);
    }
    drop(held);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(requests_to(&cloud, "/v1/internal/usage").await.is_empty());
}

// ---------------------------------------------------------------------------
// Billing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn usage_is_billed_under_the_selected_model_at_its_own_discount() {
    let backends = [
        MockServer::start().await,
        MockServer::start().await,
        MockServer::start().await,
    ];
    for backend in &backends {
        for route in ROUTES {
            mount_completions(backend, route, 1).await;
        }
    }
    let cloud = cloud_api("sk-live-customer").await;
    let gateway = start_list(
        &json!({"models": [
            // Its own discount, none at all, and the process-level default.
            {"id": ALPHA, "backend_urls": [backends[0].uri()], "discount_to_user": 0.3},
            {"id": BETA, "backend_urls": [backends[1].uri()], "discount_to_user": 0},
            {"id": GAMMA, "backend_urls": [backends[2].uri()]}
        ]}),
        &[
            ("CLOUD_API_URL", &cloud.uri()),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
            ("VLLM_PROXY_DISCOUNT_TO_USER", "0.15"),
        ],
        None,
    );
    // One route at a time, so each route's reports are read on their own.
    for (round, route) in ROUTES.into_iter().enumerate() {
        for id in [ALPHA, BETA, GAMMA] {
            let request = post(route, Some("sk-live-customer"), &body_for(Some(json!(id))));
            let response = gateway.app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{route} {id}");
            json_body(response).await;
        }

        let reports = usage_reports(&cloud, 3 * (round + 1)).await;
        let reports = &reports[3 * round..];
        assert_eq!(reports.len(), 3, "{route}");
        let report_of = |id: &str| {
            reports
                .iter()
                .find(|report| report["model"] == id)
                .unwrap_or_else(|| panic!("{route}: no usage report for {id}: {reports:?}"))
        };
        for id in [ALPHA, BETA, GAMMA] {
            let report = report_of(id);
            assert_eq!(report["organization_id"], "org");
            assert_eq!(report["input_tokens"], 3);
            assert_eq!(report["output_tokens"], 1);
        }
        assert_eq!(report_of(ALPHA)["discount_to_user"].as_f64(), Some(0.3));
        assert!(report_of(BETA).get("discount_to_user").is_none());
        assert_eq!(report_of(GAMMA)["discount_to_user"].as_f64(), Some(0.15));
    }
    for backend in &backends {
        backend.verify().await;
    }
}

#[tokio::test]
async fn text_completions_are_placed_billed_and_counted_as_their_model() {
    let alpha = MockServer::start().await;
    let alpha_long = MockServer::start().await;
    let beta = MockServer::start().await;
    for backend in [&alpha, &alpha_long, &beta] {
        mount_completions(backend, routes::ROUTE_COMPLETIONS, 1).await;
    }
    let cloud = cloud_api("sk-live-customer").await;
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let gateway = start_list(
        &json!({"models": [
            {
                "id": ALPHA,
                "backend_urls": [alpha.uri()],
                "long_context": {"backend_urls": [alpha_long.uri()], "above_tokens": 1000},
                "discount_to_user": 0.3
            },
            {"id": BETA, "backend_urls": [beta.uri()]}
        ]}),
        &[
            ("CLOUD_API_URL", &cloud.uri()),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
        ],
        Some(handle.clone()),
    );
    // ~100 and ~1,000 estimated tokens: under and over alpha's threshold,
    // which is alpha's alone. beta has no tier.
    for (id, bytes) in [(ALPHA, 400), (ALPHA, 4_000), (BETA, 4_000)] {
        let request = sized(routes::ROUTE_COMPLETIONS, "sk-live-customer", id, bytes);
        let response = gateway.app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{id} {bytes}");
        json_body(response).await;
    }
    // Placed by alpha's own tier: one prompt on each of its tiers.
    alpha.verify().await;
    alpha_long.verify().await;
    beta.verify().await;

    // Billed under the model's id, at its discount.
    let reports = usage_reports(&cloud, 3).await;
    let of = |id: &'static str| reports.iter().filter(move |report| report["model"] == id);
    assert_eq!(of(ALPHA).count(), 2, "{reports:?}");
    assert_eq!(of(BETA).count(), 1, "{reports:?}");
    assert!(of(ALPHA).all(|report| report["discount_to_user"].as_f64() == Some(0.3)));
    assert!(of(BETA).all(|report| report.get("discount_to_user").is_none()));

    // Counted under the model's label.
    let rendered = handle.render();
    let completed = |id: &str, count: u32| {
        format!(
            "inference_proxy_completed_requests_total{{auth_path=\"cloud_api_key\",\
             ingress_route=\"missing\",tenant_context=\"verified\",\
             request_id_origin=\"generated\",mode=\"json_via_stream\",model=\"{id}\"}} {count}"
        )
    };
    assert!(rendered.contains(&completed(ALPHA, 2)), "{rendered}");
    assert!(rendered.contains(&completed(BETA, 1)), "{rendered}");
    for expected in [
        "backend_tier_requests_total{tier=\"base\",outcome=\"routed\",model=\"example/alpha\"} 1",
        "backend_tier_requests_total{tier=\"long\",outcome=\"routed\",model=\"example/alpha\"} 1",
    ] {
        assert!(rendered.contains(expected), "{expected}: {rendered}");
    }
    for series in series(&rendered) {
        if series.starts_with("inference_proxy_completed_requests_total")
            || series.starts_with("backend_tier_requests_total")
        {
            assert!(series.contains("model=\""), "{series}");
        }
    }
}

const ONE_TOKEN_STREAM: &str = concat!(
    "data: {\"id\":\"chatcmpl-s\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-s\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn a_stream_is_served_billed_and_refused_as_its_own_model() {
    let stream = || {
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(ONE_TOKEN_STREAM)
    };
    // alpha's engine is slow to its first token; beta's answers at once.
    let alpha = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(routes::ROUTE_CHAT_COMPLETIONS))
        .respond_with(stream().set_delay(Duration::from_secs(30)))
        .mount(&alpha)
        .await;
    let beta = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(routes::ROUTE_CHAT_COMPLETIONS))
        .respond_with(stream())
        .expect(1)
        .mount(&beta)
        .await;
    let cloud = cloud_api("sk-live-customer").await;
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let gateway = start_list(
        &json!({"models": [
            {"id": ALPHA, "backend_urls": [alpha.uri()], "admission_max_inflight": 2},
            {"id": BETA, "backend_urls": [beta.uri()], "discount_to_user": 0.25}
        ]}),
        &[
            ("CLOUD_API_URL", &cloud.uri()),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
            // A stream timing: one value for the process, applied per request.
            ("VLLM_PROXY_FIRST_TOKEN_DEADLINE_MS", "1000"),
        ],
        Some(handle.clone()),
    );
    let stream_for = |model: &str| {
        post(
            routes::ROUTE_CHAT_COMPLETIONS,
            Some("sk-live-customer"),
            &json!({
                "model": model,
                "stream": true,
                "messages": [{"role": "user", "content": "hello"}]
            }),
        )
    };

    let response = gateway.app.clone().oneshot(stream_for(BETA)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let body = text_body(response).await;
    assert!(
        body.contains("chatcmpl-s") && body.contains("[DONE]"),
        "{body}"
    );
    let reports = usage_reports(&cloud, 1).await;
    assert_eq!(reports[0]["model"], BETA);
    assert_eq!(reports[0]["input_tokens"], 2);
    assert_eq!(reports[0]["output_tokens"], 1);
    assert_eq!(reports[0]["discount_to_user"].as_f64(), Some(0.25));

    // alpha misses its first-token deadline: refused, its slot released, and
    // counted against alpha alone.
    let response = gateway
        .app
        .clone()
        .oneshot(stream_for(ALPHA))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers().get("retry-after").unwrap(), "2");
    // (The slot goes back when the streaming task sees the refusal.)
    for _ in 0..100 {
        if gateway.model(ALPHA).admission.inflight() == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(gateway.model(ALPHA).admission.inflight(), 0);

    let mut rendered = handle.render();
    for _ in 0..100 {
        if rendered.contains("mode=\"streaming_request\",model=\"example/beta\"") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        rendered = handle.render();
    }
    assert!(
        rendered.contains("first_token_deadline_refusals_total{model=\"example/alpha\"} 1"),
        "{rendered}"
    );
    assert!(
        !rendered.contains("first_token_deadline_refusals_total{model=\"example/beta\"}"),
        "{rendered}"
    );
    assert!(
        rendered.contains("mode=\"streaming_request\",model=\"example/beta\"} 1"),
        "{rendered}"
    );
    assert_eq!(requests_to(&cloud, "/v1/internal/usage").await.len(), 1);
    beta.verify().await;
}

// ---------------------------------------------------------------------------
// Isolation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_model_at_its_budget_or_without_backends_refuses_nothing_for_another() {
    let alpha = MockServer::start().await;
    let beta = MockServer::start().await;
    mount_completions(&alpha, routes::ROUTE_CHAT_COMPLETIONS, 0).await;
    mount_completions(&beta, routes::ROUTE_CHAT_COMPLETIONS, 3).await;
    let gateway = start_list(
        &json!({"models": [
            {"id": ALPHA, "backend_urls": [alpha.uri()], "admission_max_inflight": 1},
            {"id": BETA, "backend_urls": [beta.uri()], "admission_max_inflight": 1},
            {"id": GAMMA, "backend_urls": [unreachable_backend_url()]}
        ]}),
        &[("VLLM_BACKEND_CONNECT_FAILOVER", "1")],
        None,
    );

    // alpha's whole budget is in flight.
    let alpha_model = gateway.model(ALPHA);
    let held = alpha_model
        .admission
        .try_admit(&alpha_model.backend_pool, None)
        .unwrap()
        .expect("admission is on");
    assert_eq!(alpha_model.admission.inflight(), 1);
    let response = gateway.app.clone().oneshot(chat(ALPHA)).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(json_body(response).await["error"]["type"], "overloaded");
    // beta has its own budget, untouched by alpha's.
    assert_eq!(gateway.model(BETA).admission.inflight(), 0);
    let response = gateway.app.clone().oneshot(chat(BETA)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // gamma's only backend is down: its request fails, its pool empties.
    let response = gateway.app.clone().oneshot(chat(GAMMA)).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(gateway.model(GAMMA).backend_pool.healthy_count(), 0);
    // beta's pool and budget are its own.
    assert_eq!(gateway.model(BETA).backend_pool.healthy_count(), 1);
    let response = gateway.app.clone().oneshot(chat(BETA)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // An engine rejection marks alpha's backend, not beta's.
    alpha_model
        .admission
        .try_admit(&alpha_model.backend_pool, None)
        .unwrap_err();
    held.attach_backend(0);
    held.observe_backpressure();
    assert!(alpha_model.admission.backend_saturated(0));
    assert!(!gateway.model(BETA).admission.backend_saturated(0));
    drop(held);
    let response = gateway.app.clone().oneshot(chat(BETA)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    alpha.verify().await;
    beta.verify().await;
}

#[tokio::test]
async fn a_models_backend_token_and_priority_reach_only_its_own_backends() {
    let alpha = MockServer::start().await;
    let beta = MockServer::start().await;
    let gamma = MockServer::start().await;
    for backend in [&alpha, &beta, &gamma] {
        for route in ROUTES {
            mount_completions(backend, route, 2).await;
        }
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200))
            .mount(backend)
            .await;
    }
    let cloud = cloud_api("sk-live-customer").await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"id": ALPHA}, {"id": BETA}, {"id": GAMMA}]
        })))
        .mount(&cloud)
        .await;
    let gateway = start_list(
        &json!({"models": [
            {
                "id": ALPHA,
                "backend_urls": [alpha.uri()],
                "backend_token_env": "VLLM_BACKEND_TOKEN_ALPHA",
                "backend_priority": -1
            },
            {
                "id": BETA,
                "backend_urls": [beta.uri()],
                "backend_token_env": "VLLM_BACKEND_TOKEN_BETA",
                "backend_priority": 5
            },
            // No token of its own and no process-level one to fall back to.
            {"id": GAMMA, "backend_urls": [gamma.uri()]}
        ]}),
        &[
            ("CLOUD_API_URL", &cloud.uri()),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
            (
                "VLLM_PROXY_MODELS_DOCUMENT_URL",
                &format!("{}/v1/models", cloud.uri()),
            ),
            ("VLLM_BACKEND_TOKEN_ALPHA", "alpha-secret"),
            ("VLLM_BACKEND_TOKEN_BETA", "beta-secret"),
        ],
        None,
    );

    // Inference from a customer key and from the operator token, the models
    // document and the health probes: every outbound path of the process.
    for id in [ALPHA, BETA, GAMMA] {
        for route in ROUTES {
            for bearer in ["sk-live-customer", "test-token"] {
                let request = post(route, Some(bearer), &body_for(Some(json!(id))));
                let response = gateway.app.clone().oneshot(request).await.unwrap();
                assert_eq!(response.status(), StatusCode::OK, "{id} {route} {bearer}");
                json_body(response).await;
            }
        }
    }
    for route in [routes::ROUTE_V1_MODELS, routes::ROUTE_HEALTHZ] {
        let response = gateway.app.clone().oneshot(get(route)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{route}");
    }
    usage_reports(&cloud, 6).await;

    let expected = [
        (&alpha, Some("Bearer alpha-secret"), Some("-1")),
        (&beta, Some("Bearer beta-secret"), Some("5")),
        (&gamma, None, None),
    ];
    for (backend, bearer, priority) in expected {
        let inference: Vec<_> = backend
            .received_requests()
            .await
            .unwrap()
            .into_iter()
            .filter(|request| request.method.as_str() == "POST")
            .collect();
        assert_eq!(inference.len(), 4);
        for request in &inference {
            // The model's own bearer — never the caller's, never another model's.
            assert_eq!(bearer_of(request).as_deref(), bearer);
            assert_eq!(
                request
                    .headers
                    .get(priority::PRIORITY_HEADER)
                    .map(|value| value.to_str().unwrap()),
                priority
            );
        }
        // The health probe goes out on the general client: no bearer at all.
        let probes = requests_to(backend, "/health").await;
        assert!(!probes.is_empty());
        for probe in &probes {
            assert_eq!(bearer_of(probe), None);
        }
    }
    // cloud-api sees the customer key on the key check and the usage token on
    // the reports; no backend token on anything.
    let to_cloud = cloud.received_requests().await.unwrap();
    assert!(to_cloud.len() >= 13, "{}", to_cloud.len());
    for request in &to_cloud {
        let expected = match request.url.path() {
            "/v1/check_api_key" => Some("Bearer sk-live-customer"),
            "/v1/internal/usage" => Some("Bearer usage-secret"),
            "/v1/models" => None,
            other => panic!("unexpected request to cloud-api: {other}"),
        };
        assert_eq!(bearer_of(request).as_deref(), expected, "{}", request.url);
        let raw = format!(
            "{:?}{}",
            request.headers,
            String::from_utf8_lossy(&request.body)
        );
        assert!(!raw.contains("alpha-secret") && !raw.contains("beta-secret"));
    }
}

#[tokio::test]
async fn each_model_keeps_its_own_tier_and_reasoning_switch() {
    let alpha = MockServer::start().await;
    let alpha_long = MockServer::start().await;
    let beta = MockServer::start().await;
    // alpha: a long-context tier above 1,000 estimated tokens, and `low` as
    // its "no reasoning". beta: one flat pool, and the default `none`.
    Mock::given(method("POST"))
        .and(path(routes::ROUTE_CHAT_COMPLETIONS))
        .and(body_partial_json(json!({"reasoning_effort": "low"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_json()))
        .expect(1)
        .mount(&alpha)
        .await;
    Mock::given(method("POST"))
        .and(path(routes::ROUTE_CHAT_COMPLETIONS))
        .and(body_partial_json(json!({"reasoning_effort": "low"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_json()))
        .expect(1)
        .mount(&alpha_long)
        .await;
    Mock::given(method("POST"))
        .and(path(routes::ROUTE_CHAT_COMPLETIONS))
        .and(body_partial_json(json!({"reasoning_effort": "none"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_json()))
        .expect(2)
        .mount(&beta)
        .await;
    let cloud = cloud_api("sk-live-customer").await;
    let gateway = start_list(
        &json!({"models": [
            {
                "id": ALPHA,
                "backend_urls": [alpha.uri()],
                "long_context": {
                    "backend_urls": [alpha_long.uri()],
                    "above_tokens": 1000,
                    "strict": true
                },
                "reasoning_off_effort": "low",
                "backend_token_env": "VLLM_BACKEND_TOKEN_ALPHA"
            },
            {
                "id": BETA,
                "backend_urls": [beta.uri()],
                "backend_token_env": "VLLM_BACKEND_TOKEN_BETA"
            }
        ]}),
        &[
            ("CLOUD_API_URL", &cloud.uri()),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
            ("VLLM_BACKEND_TOKEN_ALPHA", "alpha-secret"),
            ("VLLM_BACKEND_TOKEN_BETA", "beta-secret"),
        ],
        None,
    );
    for id in [ALPHA, BETA] {
        // ~100 and ~1,300 estimated tokens: under and over alpha's threshold.
        for bytes in [400, 4_000] {
            let request = post(
                routes::ROUTE_CHAT_COMPLETIONS,
                Some("test-token"),
                &json!({
                    "model": id,
                    "reasoning": {"enabled": false},
                    "messages": [{"role": "user", "content": "x".repeat(bytes)}]
                }),
            );
            let response = gateway.app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{id} {bytes}");
        }
    }
    alpha.verify().await;
    alpha_long.verify().await;
    beta.verify().await;
}

#[tokio::test]
async fn a_tier_is_strict_for_the_model_that_says_so_on_both_routes() {
    // alpha isolates its tiers, beta does not, gamma leaves it to the
    // process — once with the variable unset and once with it set, so each
    // model's own word is on the other side of the default at least once.
    for process_default in [false, true] {
        let bases = [
            MockServer::start().await,
            MockServer::start().await,
            MockServer::start().await,
        ];
        let strict = [true, false, process_default];
        for (base, strict) in bases.iter().zip(strict) {
            for route in ROUTES {
                mount_completions(base, route, u64::from(!strict)).await;
            }
        }
        let tier = |extra: Value| {
            let mut tier = json!({
                "backend_urls": [unreachable_backend_url()],
                "above_tokens": 1000
            });
            tier.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            tier
        };
        let gateway = start_list(
            &json!({"models": [
                {"id": ALPHA, "backend_urls": [bases[0].uri()],
                 "long_context": tier(json!({"strict": true}))},
                {"id": BETA, "backend_urls": [bases[1].uri()],
                 "long_context": tier(json!({"strict": false}))},
                {"id": GAMMA, "backend_urls": [bases[2].uri()], "long_context": tier(json!({}))}
            ]}),
            &[(
                "VLLM_BACKEND_TIER_STRICT",
                if process_default { "1" } else { "" },
            )],
            None,
        );
        // Every model's long-context tier is down, and its pool knows.
        for id in [ALPHA, BETA, GAMMA] {
            gateway.model(id).backend_pool.backends()[1]
                .healthy
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        for route in ROUTES {
            for (id, strict) in [ALPHA, BETA, GAMMA].into_iter().zip(strict) {
                // An oversized prompt belongs on the tier that is down.
                let request = sized(route, "test-token", id, 4_000);
                let response = gateway.app.clone().oneshot(request).await.unwrap();
                let context = format!("{route} {id} default={process_default}");
                if strict {
                    // Refused rather than put in front of the short requests.
                    assert_eq!(
                        response.status(),
                        StatusCode::SERVICE_UNAVAILABLE,
                        "{context}"
                    );
                    let body = json_body(response).await;
                    assert_eq!(body["error"]["type"], "tier_unavailable", "{context}");
                } else {
                    // Served by the base hosts instead.
                    assert_eq!(response.status(), StatusCode::OK, "{context}");
                }
            }
        }
        for base in &bases {
            base.verify().await;
        }
    }
}

/// Turn `turn` of one append-only conversation with `model`.
fn conversation(model: &str, turn: usize) -> Request<Body> {
    let mut messages = vec![
        json!({"role": "system", "content": "be brief"}),
        json!({"role": "user", "content": "start"}),
    ];
    for i in 0..turn {
        messages.push(json!({"role": "assistant", "content": "ok"}));
        messages.push(json!({"role": "user", "content": format!("more {i}")}));
    }
    post(
        routes::ROUTE_CHAT_COMPLETIONS,
        Some("test-token"),
        &json!({"model": model, "messages": messages}),
    )
}

#[tokio::test]
async fn a_conversation_stays_on_its_backend_within_its_own_model() {
    let alpha = [MockServer::start().await, MockServer::start().await];
    let beta = [MockServer::start().await, MockServer::start().await];
    mount_completions(&alpha[0], routes::ROUTE_CHAT_COMPLETIONS, 0).await;
    mount_completions(&alpha[1], routes::ROUTE_CHAT_COMPLETIONS, 2).await;
    mount_completions(&beta[0], routes::ROUTE_CHAT_COMPLETIONS, 1).await;
    mount_completions(&beta[1], routes::ROUTE_CHAT_COMPLETIONS, 0).await;
    let gateway = start_list(
        &json!({"models": [
            {"id": ALPHA, "backend_urls": [alpha[0].uri(), alpha[1].uri()]},
            {"id": BETA, "backend_urls": [beta[0].uri(), beta[1].uri()]}
        ]}),
        &[("VLLM_BACKEND_CONVERSATION_AFFINITY", "1")],
        None,
    );

    // The first turn finds alpha's first backend busy and starts on the second.
    let busy =
        backend_pool::BackendGuard::new(gateway.model(ALPHA).backend_pool.backends()[0].clone());
    let response = gateway
        .app
        .clone()
        .oneshot(conversation(ALPHA, 0))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    drop(busy);
    // Both are idle now, and least-connections alone would take the first.
    // The conversation is pinned where its prefix is cached: the second.
    let response = gateway
        .app
        .clone()
        .oneshot(conversation(ALPHA, 1))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // The same conversation with beta is beta's own: nothing pins it, and it
    // goes to beta's first backend, not to "backend 1" as alpha's does.
    let response = gateway
        .app
        .clone()
        .oneshot(conversation(BETA, 1))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    for backend in alpha.iter().chain(&beta) {
        backend.verify().await;
    }
}

fn engine_metrics(running: u32, queued: u32) -> String {
    format!(
        "sglang:num_running_reqs{{model_name=\"m\"}} {running}.0\nsglang:num_queue_reqs{{model_name=\"m\"}} {queued}.0\n"
    )
}

#[tokio::test]
async fn every_model_gets_a_health_checker_and_an_engine_poller_that_carry_no_bearer() {
    // alpha's one backend fails its first health probe and passes the next.
    let alpha = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(500))
        .up_to_n_times(1)
        .mount(&alpha)
        .await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&alpha)
        .await;
    let alpha_engine = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_string(engine_metrics(3, 2)))
        .mount(&alpha_engine)
        .await;
    let beta = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&beta)
        .await;
    let cloud = cloud_api("sk-live-customer").await;
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let gateway = start_list(
        &json!({"models": [
            {
                "id": ALPHA,
                "backend_urls": [alpha.uri()],
                "backend_probe_urls": [alpha_engine.uri()],
                "backend_token_env": "VLLM_BACKEND_TOKEN_ALPHA",
                "backend_priority": -1
            },
            {"id": BETA, "backend_urls": [beta.uri()]}
        ]}),
        &[
            ("CLOUD_API_URL", &cloud.uri()),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
            ("VLLM_BACKEND_TOKEN_ALPHA", "alpha-secret"),
            ("HEALTH_CHECK_INTERVAL_SECS", "1"),
            ("HEALTH_CHECK_MAX_FAILURES", "1"),
            ("VLLM_BACKEND_PROBE_INTERVAL_SECS", "1"),
        ],
        Some(handle.clone()),
    );
    let (alpha_model, beta_model) = (gateway.model(ALPHA), gateway.model(BETA));

    // The poller: alpha's engine view arrives; beta, without probes, has none.
    eventually(5, "alpha's engine sample", || {
        alpha_model.admission.engine(0) == Some((3, 2))
    })
    .await;
    assert_eq!(beta_model.admission.engine(0), None);

    // The checker, for a pool of one backend too: alpha's host is taken out
    // on its failed probe and comes back on the next one.
    assert_eq!(alpha_model.backend_pool.healthy_count(), 1);
    eventually(5, "alpha's backend to be marked down", || {
        alpha_model.backend_pool.healthy_count() == 0
    })
    .await;
    assert_eq!(beta_model.backend_pool.healthy_count(), 1);
    eventually(5, "alpha's backend to come back", || {
        alpha_model.backend_pool.healthy_count() == 1
    })
    .await;

    // Both ran on the general client: neither probe carries alpha's bearer
    // or its priority.
    let health_probes = requests_to(&alpha, "/health").await;
    let engine_probes = requests_to(&alpha_engine, "/v1/metrics").await;
    assert!(health_probes.len() >= 2, "{}", health_probes.len());
    assert!(!engine_probes.is_empty());
    for probe in health_probes.iter().chain(&engine_probes) {
        assert_eq!(bearer_of(probe), None);
        assert!(probe.headers.get(priority::PRIORITY_HEADER).is_none());
    }
    assert!(!requests_to(&beta, "/health").await.is_empty());

    let rendered = handle.render();
    for expected in [
        "backend_engine_running{backend=\"0\",model=\"example/alpha\"} 3",
        "backend_engine_queued{backend=\"0\",model=\"example/alpha\"} 2",
        "backend_pool_size{model=\"example/alpha\"} 1",
        "backend_pool_healthy{model=\"example/alpha\"} 1",
        "backend_pool_size{model=\"example/beta\"} 1",
        "backend_pool_healthy{model=\"example/beta\"} 1",
    ] {
        assert!(rendered.contains(expected), "{expected}: {rendered}");
    }
    assert!(!rendered.contains("model=\"example/beta\",backend_engine"));
    assert!(!rendered.contains("backend_engine_running{backend=\"0\",model=\"example/beta\"}"));
}

// ---------------------------------------------------------------------------
// /v1/models
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_models_document_lists_every_configured_model_with_its_own_capacity_and_discount() {
    let backend = MockServer::start().await;
    expect_untouched(&backend).await;
    let source = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                {"id": "example/unrelated", "name": "Unrelated"},
                {"id": BETA, "name": "Beta", "discount_to_user": 0.9},
                {"id": ALPHA, "name": "Alpha", "pricing": {"prompt": "0.00000015"},
                 "openrouter": {"slug": "example/alpha"}}
            ]
        })))
        .expect(1)
        .mount(&source)
        .await;

    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let gateway = start_list(
        &json!({"models": [
            {
                "id": ALPHA,
                "backend_urls": [format!("{}/a", backend.uri())],
                "admission_max_inflight": 48,
                "capacity_requests_per_minute": 150,
                "discount_to_user": 0.3
            },
            {"id": BETA, "backend_urls": [format!("{}/b", backend.uri())], "admission_max_inflight": 16},
            // Configured here, but not in the catalog.
            {"id": GAMMA, "backend_urls": [format!("{}/c", backend.uri())]}
        ]}),
        &[(
            "VLLM_PROXY_MODELS_DOCUMENT_URL",
            &format!("{}/v1/models", source.uri()),
        )],
        Some(handle.clone()),
    );
    let response = gateway
        .app
        .clone()
        .oneshot(get(routes::ROUTE_V1_MODELS))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let document = json_body(response).await;
    assert_eq!(document["object"], "list");
    // The source's entries for the configured models, in the source's order.
    let data = document["data"].as_array().unwrap();
    assert_eq!(data.len(), 2, "{document}");
    assert_eq!(data[0]["id"], BETA);
    assert_eq!(data[0]["name"], "Beta");
    assert_eq!(
        data[0]["capacity"],
        json!([{"type": "concurrency", "unit": "request", "value": 16}])
    );
    // No discount configured for beta: the one in the source is not this
    // lane's, and is dropped.
    assert!(data[0].get("discount_to_user").is_none(), "{document}");
    assert_eq!(data[1]["id"], ALPHA);
    assert_eq!(data[1]["pricing"]["prompt"], "0.00000015");
    assert_eq!(data[1]["openrouter"]["slug"], "example/alpha");
    assert_eq!(
        data[1]["capacity"],
        json!([
            {"type": "concurrency", "unit": "request", "value": 48},
            {"type": "request", "unit": "request", "per": "minute", "value": 150}
        ])
    );
    assert_eq!(data[1]["discount_to_user"].as_f64(), Some(0.3));

    // The model the source does not list is left out, and counted.
    let rendered = handle.render();
    assert!(
        rendered.contains("models_document_missing_models_total{model=\"example/gamma\"} 1"),
        "{rendered}"
    );
    assert!(!rendered.contains("models_document_missing_models_total{model=\"example/alpha\"}"));
    source.verify().await;
    backend.verify().await;
}

#[tokio::test]
async fn an_unreadable_models_document_is_a_502_and_never_an_engine_list() {
    let backend = MockServer::start().await;
    expect_untouched(&backend).await;
    let failing = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&failing)
        .await;
    let not_a_list = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models": []})))
        .mount(&not_a_list)
        .await;
    let list = json!({"models": [
        {"id": ALPHA, "backend_urls": [format!("{}/a", backend.uri())], "discount_to_user": 0.3},
        {"id": BETA, "backend_urls": [format!("{}/b", backend.uri())]}
    ]});
    for source in [failing.uri(), not_a_list.uri(), unreachable_backend_url()] {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _metrics = metrics::set_default_local_recorder(&recorder);
        let gateway = start_list(
            &list,
            &[(
                "VLLM_PROXY_MODELS_DOCUMENT_URL",
                &format!("{source}/v1/models"),
            )],
            Some(handle.clone()),
        );
        let response = gateway
            .app
            .clone()
            .oneshot(get(routes::ROUTE_V1_MODELS))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY, "{source}");
        let body = json_body(response).await;
        assert_eq!(body["error"]["type"], "models_document_unavailable");
        assert!(body.get("data").is_none(), "{body}");
        assert!(
            handle
                .render()
                .contains("models_document_source_failures_total 1"),
            "{}",
            handle.render()
        );
    }
    // No engine was asked for its list in the source's place.
    backend.verify().await;
}

#[tokio::test]
async fn a_model_missing_from_the_document_is_logged_when_that_changes_and_counted_every_time() {
    let backend = MockServer::start().await;
    let source = MockServer::start().await;
    let document = |ids: &[&str]| {
        let data: Vec<Value> = ids.iter().map(|id| json!({"id": id})).collect();
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"object": "list", "data": data})),
            )
    };
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _logs = logs.capture();
    let gateway = start_list(
        &json!({"models": [
            {"id": ALPHA, "backend_urls": [format!("{}/a", backend.uri())]},
            {"id": GAMMA, "backend_urls": [format!("{}/c", backend.uri())]}
        ]}),
        &[(
            "VLLM_PROXY_MODELS_DOCUMENT_URL",
            &format!("{}/v1/models", source.uri()),
        )],
        Some(handle.clone()),
    );
    let listed = || async {
        let response = gateway
            .app
            .clone()
            .oneshot(get(routes::ROUTE_V1_MODELS))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        json_body(response).await["data"].as_array().unwrap().len()
    };
    let lines = |message: &str| {
        logs.contents()
            .lines()
            .filter(|line| line.contains(message) && line.contains("model=example/gamma"))
            .count()
    };
    let missing = "Configured model is not in the models document";
    let back = "Configured model is in the models document again";
    let counted = |count: u32| {
        format!("models_document_missing_models_total{{model=\"example/gamma\"}} {count}")
    };

    // The route is polled: gamma is counted on every read it is missing
    // from, and logged once.
    document(&[ALPHA]).mount(&source).await;
    for _ in 0..3 {
        assert_eq!(listed().await, 1);
    }
    assert_eq!((lines(missing), lines(back)), (1, 0), "{}", logs.contents());
    assert!(handle.render().contains(&counted(3)), "{}", handle.render());

    // Back in the document: said once, and no longer counted.
    source.reset().await;
    document(&[ALPHA, GAMMA]).mount(&source).await;
    for _ in 0..2 {
        assert_eq!(listed().await, 2);
    }
    assert_eq!((lines(missing), lines(back)), (1, 1), "{}", logs.contents());
    assert!(handle.render().contains(&counted(3)), "{}", handle.render());

    // Gone again: a new change, a new line.
    source.reset().await;
    document(&[ALPHA]).mount(&source).await;
    for _ in 0..2 {
        assert_eq!(listed().await, 1);
    }
    assert_eq!((lines(missing), lines(back)), (2, 1), "{}", logs.contents());
    assert!(handle.render().contains(&counted(5)), "{}", handle.render());
    // alpha was there throughout: never a line about it.
    assert!(!logs
        .contents()
        .lines()
        .any(|line| line.contains("models document") && line.contains("model=example/alpha")));
    assert!(!handle
        .render()
        .contains("models_document_missing_models_total{model=\"example/alpha\"}"));
}

// ---------------------------------------------------------------------------
// /healthz
// ---------------------------------------------------------------------------

#[tokio::test]
async fn healthz_is_ok_while_one_model_is_and_names_the_models_to_the_operator_only() {
    let alpha = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/healthz"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&alpha)
        .await;
    let gamma = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/healthz"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&gamma)
        .await;
    let cloud = cloud_api("sk-live-customer").await;
    let gateway = start_list(
        &json!({"models": [
            {"id": ALPHA, "backend_urls": [alpha.uri()]},
            {"id": BETA, "backend_urls": [unreachable_backend_url()]},
            {"id": GAMMA, "backend_urls": [gamma.uri()]}
        ]}),
        &[
            ("VLLM_BACKEND_HEALTH_PATH", "/healthz"),
            ("CLOUD_API_URL", &cloud.uri()),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
        ],
        None,
    );
    let healthz = |bearer: Option<&'static str>| {
        let request = match bearer {
            Some(bearer) => get_as(routes::ROUTE_HEALTHZ, bearer),
            None => get(routes::ROUTE_HEALTHZ),
        };
        let app = gateway.app.clone();
        async move {
            let response = app.oneshot(request).await.unwrap();
            (response.status(), text_body(response).await)
        }
    };
    let json = |body: &str| serde_json::from_str::<Value>(body).unwrap();

    // Two of three models are down; the process still serves the third.
    // The operator, who holds the gateway's own token, sees each model.
    let (status, body) = healthz(Some("test-token")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json(&body),
        json!({
            "status": "ok",
            "checks": {"dstack": "skipped", "backend": "ok"},
            "models": [
                {"id": ALPHA, "backend": "ok"},
                {"id": BETA, "backend": "unreachable"},
                {"id": GAMMA, "backend": "http_5xx"}
            ]
        })
    );
    // Anyone else gets the same verdict and no model: the ids are not for
    // unauthenticated callers, and a customer key is not the operator.
    for bearer in [None, Some("sk-live-customer"), Some("not-the-token")] {
        let (status, body) = healthz(bearer).await;
        assert_eq!(status, StatusCode::OK, "{bearer:?}");
        assert_eq!(
            json(&body),
            json!({"status": "ok", "checks": {"dstack": "skipped", "backend": "ok"}}),
            "{bearer:?}"
        );
        for id in [ALPHA, BETA, GAMMA, "models"] {
            assert!(!body.contains(id), "{bearer:?}: {body}");
        }
    }

    // Once a model's pool knows it has no healthy backend, it is reported
    // without being probed again.
    let before = requests_to(&gamma, "/healthz").await.len();
    for backend in gateway.model(GAMMA).backend_pool.backends() {
        backend
            .healthy
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }
    let (status, body) = healthz(Some("test-token")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json(&body)["models"][2],
        json!({"id": GAMMA, "backend": "unhealthy"})
    );
    assert_eq!(requests_to(&gamma, "/healthz").await.len(), before);

    // No model left: the process is down, for everyone.
    alpha.reset().await;
    Mock::given(method("GET"))
        .and(path("/healthz"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&alpha)
        .await;
    let (status, body) = healthz(None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        json(&body),
        json!({"status": "unhealthy", "checks": {"dstack": "skipped", "backend": "unhealthy"}})
    );
    let (status, body) = healthz(Some("test-token")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        json(&body),
        json!({
            "status": "unhealthy",
            "checks": {"dstack": "skipped", "backend": "unhealthy"},
            "models": [
                {"id": ALPHA, "backend": "http_5xx"},
                {"id": BETA, "backend": "unreachable"},
                {"id": GAMMA, "backend": "unhealthy"}
            ]
        })
    );
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_mode_serves_only_the_routes_defined_for_several_models() {
    let backend = MockServer::start().await;
    expect_untouched(&backend).await;
    let gateway = start_list(
        &json!({"models": [{"id": ALPHA, "backend_urls": [backend.uri()]}]}),
        &[],
        None,
    );
    for route in [routes::ROUTE_VERSION, routes::ROUTE_METRICS] {
        let response = gateway.app.clone().oneshot(get(route)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{route}");
    }
    // Everything that assumes the one model of a single-model process is the
    // same 404 as an undeclared route, before authentication.
    for route in [
        routes::ROUTE_ROOT,
        routes::ROUTE_V1_METRICS,
        routes::ROUTE_ATTESTATION_REPORT,
        "/v1/signature/chatcmpl-1",
        routes::ROUTE_OHTTP_CONFIG,
        routes::ROUTE_OHTTP_WELL_KNOWN,
    ] {
        let response = gateway.app.clone().oneshot(get(route)).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{route}");
        let body = json_body(response).await;
        assert_eq!(body["error"]["message"], "Endpoint not found", "{route}");
    }
    for route in [
        routes::ROUTE_TOKENIZE,
        routes::ROUTE_EMBEDDINGS,
        routes::ROUTE_RERANK,
        routes::ROUTE_SCORE,
        routes::ROUTE_PRIVACY_CLASSIFY,
        routes::ROUTE_IMAGES_GENERATIONS,
        routes::ROUTE_IMAGES_EDITS,
        routes::ROUTE_AUDIO_TRANSCRIPTIONS,
        routes::ROUTE_INTERNAL_GPU_EVIDENCE,
        routes::ROUTE_OHTTP_RELAY,
    ] {
        let request = post(route, Some("test-token"), &body_for(Some(json!(ALPHA))));
        let response = gateway.app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{route}");
        let body = json_body(response).await;
        assert_eq!(body["error"]["message"], "Endpoint not found", "{route}");
    }
    backend.verify().await;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// The identity of every series in a Prometheus rendering: name and labels.
fn series(rendered: &str) -> BTreeSet<String> {
    rendered
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(|line| line.rsplit_once(' ').unwrap().0.to_string())
        .collect()
}

/// `series` without the `model` label, and whether it had one.
fn without_model(series: &str) -> (String, bool) {
    let Some((name, labels)) = series.split_once('{') else {
        return (series.to_string(), false);
    };
    let labels: Vec<&str> = labels.trim_end_matches('}').split(',').collect();
    let kept: Vec<&str> = labels
        .iter()
        .copied()
        .filter(|label| !label.starts_with("model=\""))
        .collect();
    let had_model = kept.len() != labels.len();
    if kept.is_empty() {
        (name.to_string(), had_model)
    } else {
        (format!("{name}{{{}}}", kept.join(",")), had_model)
    }
}

/// The same traffic against a gateway: a customer request (billed), an
/// oversized request (long tier), a refusal at the budget, a scrape.
async fn exercise(
    gateway: &Gateway,
    cloud: &MockServer,
    take_budget: impl FnOnce() -> Option<admission::Permit>,
) -> String {
    let customer = post(
        routes::ROUTE_CHAT_COMPLETIONS,
        Some("sk-live-customer"),
        &body_for(Some(json!(ALPHA))),
    );
    let response = gateway.app.clone().oneshot(customer).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    json_body(response).await;
    let oversized = post(
        routes::ROUTE_CHAT_COMPLETIONS,
        Some("test-token"),
        &json!({"model": ALPHA, "messages": [{"role": "user", "content": "x".repeat(4_000)}]}),
    );
    let response = gateway.app.clone().oneshot(oversized).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    json_body(response).await;
    usage_reports(cloud, 1).await;
    // The report's own outcome is recorded once cloud-api has answered it.
    for _ in 0..150 {
        let response = gateway
            .app
            .clone()
            .oneshot(get(routes::ROUTE_METRICS))
            .await
            .unwrap();
        if text_body(response)
            .await
            .contains("inference_proxy_usage_report_duration_seconds")
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let held = take_budget().expect("admission is on");
    let response = gateway.app.clone().oneshot(chat(ALPHA)).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    drop(held);

    let response = gateway
        .app
        .clone()
        .oneshot(get(routes::ROUTE_METRICS))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    text_body(response).await
}

#[tokio::test]
async fn per_model_series_carry_a_model_label_in_list_mode_and_none_for_a_single_model() {
    // One lane, configured twice with the same values: as the single model of
    // a process, and as the only entry of a list.
    let lane = [
        ("VLLM_PROXY_ADMISSION_MAX_INFLIGHT", "1"),
        ("VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS", "1000"),
        ("VLLM_PROXY_DISCOUNT_TO_USER", "0.3"),
        ("VLLM_BACKEND_TOKEN", "backend-secret"),
        ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
    ];

    let single = {
        let (base, long) = (MockServer::start().await, MockServer::start().await);
        mount_completions(&base, routes::ROUTE_CHAT_COMPLETIONS, 1).await;
        mount_completions(&long, routes::ROUTE_CHAT_COMPLETIONS, 1).await;
        let cloud = cloud_api("sk-live-customer").await;
        let recorder = PrometheusBuilder::new().build_recorder();
        let _metrics = metrics::set_default_local_recorder(&recorder);
        let mut env = lane.to_vec();
        let (base_url, long_url, cloud_url) = (base.uri(), long.uri(), cloud.uri());
        env.extend([
            ("MODEL_NAME", ALPHA),
            ("VLLM_BACKEND_URLS", base_url.as_str()),
            ("VLLM_BACKEND_LONG_CONTEXT_URLS", long_url.as_str()),
            ("CLOUD_API_URL", cloud_url.as_str()),
            (
                "VLLM_PROXY_MODELS_DOCUMENT_URL",
                "http://catalog.invalid/v1/models",
            ),
        ]);
        let gateway = start_single(&env, Some(recorder.handle()));
        let rendered = exercise(&gateway, &cloud, || {
            gateway
                .state
                .admission
                .try_admit(&gateway.state.backend_pool, None)
                .unwrap()
        })
        .await;
        base.verify().await;
        long.verify().await;
        rendered
    };

    let listed = {
        let (base, long) = (MockServer::start().await, MockServer::start().await);
        mount_completions(&base, routes::ROUTE_CHAT_COMPLETIONS, 1).await;
        mount_completions(&long, routes::ROUTE_CHAT_COMPLETIONS, 1).await;
        let cloud = cloud_api("sk-live-customer").await;
        let recorder = PrometheusBuilder::new().build_recorder();
        let _metrics = metrics::set_default_local_recorder(&recorder);
        let mut env = lane.to_vec();
        let cloud_url = cloud.uri();
        env.push(("CLOUD_API_URL", cloud_url.as_str()));
        let gateway = start_list(
            &json!({"models": [{
                "id": ALPHA,
                "backend_urls": [base.uri()],
                "long_context": {"backend_urls": [long.uri()]}
            }]}),
            &env,
            Some(recorder.handle()),
        );
        let rendered = exercise(&gateway, &cloud, || {
            let model = gateway.model(ALPHA);
            model
                .admission
                .try_admit(&model.backend_pool, None)
                .unwrap()
        })
        .await;
        base.verify().await;
        long.verify().await;
        rendered
    };

    // A single-model process: the series dashboards and alerts match on,
    // with no `model` label anywhere.
    let single = series(&single);
    for expected in [
        "admission_budget",
        "admission_inflight",
        "admission_inflight_base",
        "admission_long_reserve",
        "admission_rejections_total{reason=\"budget\"}",
        "admission_backend_limit{backend=\"0\",tier=\"base\"}",
        "admission_backend_inflight{backend=\"1\",tier=\"long\"}",
        "backend_tier_requests_total{tier=\"base\",outcome=\"routed\"}",
        "backend_tier_requests_total{tier=\"long\",outcome=\"routed\"}",
        "request_model_match_total{result=\"exact\"}",
    ] {
        assert!(single.contains(expected), "{expected} missing: {single:#?}");
    }
    for series in &single {
        assert!(!series.contains("model=\""), "{series}");
    }

    // The list: the very same series, each per-model one told apart by
    // `model` as its last label.
    let listed = series(&listed);
    for expected in [
        "admission_budget{model=\"example/alpha\"}",
        "admission_inflight{model=\"example/alpha\"}",
        "admission_rejections_total{reason=\"budget\",model=\"example/alpha\"}",
        "admission_backend_limit{backend=\"0\",tier=\"base\",model=\"example/alpha\"}",
        "admission_backend_inflight{backend=\"1\",tier=\"long\",model=\"example/alpha\"}",
        "backend_tier_requests_total{tier=\"long\",outcome=\"routed\",model=\"example/alpha\"}",
        "request_model_match_total{result=\"exact\"}",
    ] {
        assert!(listed.contains(expected), "{expected} missing: {listed:#?}");
    }
    let per_model = [
        "admission_",
        "backend_tier_",
        "backend_failover_",
        "backend_affinity_",
        "backend_engine_",
        "backend_pool_",
        "request_estimated_prompt_tokens",
        "first_token_deadline_refusals_total",
        "inference_proxy_usage_report",
        "inference_proxy_completed_requests_total",
        "inference_proxy_input_tokens",
        "inference_proxy_request_duration_seconds",
    ];
    let mut labelled = 0;
    let mut stripped = BTreeSet::new();
    for series in &listed {
        let (plain, had_model) = without_model(series);
        let is_per_model = per_model.iter().any(|family| series.starts_with(family));
        assert_eq!(had_model, is_per_model, "{series}");
        labelled += usize::from(had_model);
        stripped.insert(plain);
    }
    assert!(labelled >= 15, "{labelled}: {listed:#?}");
    // (The pool gauges are written by the health checker on its own clock,
    // and only a list starts one for a one-backend tier.)
    let on_a_timer = |series: &String| series.starts_with("backend_pool_");
    stripped.retain(|series| !on_a_timer(series));
    let single: BTreeSet<String> = single.into_iter().filter(|s| !on_a_timer(s)).collect();
    assert_eq!(stripped, single);
}

// ---------------------------------------------------------------------------
// Observe before enforce (single model)
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Logs {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Keeps a `Logs` capture going until dropped.
struct LogCapture {
    _subscriber: tracing::subscriber::DefaultGuard,
    _registered: tracing::Dispatch,
}

impl Logs {
    /// Every log line of the proxy itself, at any level, on this thread.
    fn capture(&self) -> LogCapture {
        let logs = self.clone();
        let subscriber = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::new("vllm_proxy_rs=trace"))
                .with_ansi(false)
                .with_writer(move || logs.clone())
                .finish(),
        );
        // tracing caches every callsite's interest for the whole process.
        // While exactly one subscriber is registered it derives that from the
        // calling thread's default, so a test on another thread, which has
        // none, would switch a shared callsite off for this capture as well.
        // With a second one registered it asks every registered subscriber.
        let registered = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        LogCapture {
            _subscriber: subscriber,
            _registered: registered,
        }
    }

    fn contents(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
    }
}

/// What a caller may put in `model`: never a label, never a log line.
const REQUESTED: [&str; 2] = ["Example/ALPHA", "someone-elses/model-sentinel"];

#[tokio::test]
async fn a_single_model_lane_counts_how_the_requested_model_compares_and_serves_as_before() {
    let backend = MockServer::start().await;
    for route in ROUTES {
        mount_completions(&backend, route, 6).await;
    }
    let cloud = cloud_api("sk-live-customer").await;
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _logs = logs.capture();
    let gateway = start_single(
        &[
            ("MODEL_NAME", ALPHA),
            ("VLLM_BACKEND_URLS", &backend.uri()),
            ("VLLM_BACKEND_TOKEN", "backend-secret"),
            ("CLOUD_API_URL", &cloud.uri()),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
        ],
        Some(handle.clone()),
    );

    let bodies = [
        Some(json!(ALPHA)),
        Some(json!(REQUESTED[0])),
        Some(json!(REQUESTED[1])),
        None,
        Some(Value::Null),
        Some(json!(42)),
    ];
    for route in ROUTES {
        for model in &bodies {
            // Whatever the body names, the one model answers — as it always has.
            let request = post(route, Some("sk-live-customer"), &body_for(model.clone()));
            let response = gateway.app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{route} {model:?}");
            json_body(response).await;
        }
    }
    backend.verify().await;
    // ... and is billed, under MODEL_NAME.
    let reports = usage_reports(&cloud, 12).await;
    assert!(reports.iter().all(|report| report["model"] == ALPHA));

    let rendered = handle.render();
    for (result, count) in [
        ("exact", 2),
        ("case_differs", 2),
        ("other", 2),
        ("missing", 6),
    ] {
        let line = format!("request_model_match_total{{result=\"{result}\"}} {count}");
        assert!(rendered.contains(&line), "{line}: {rendered}");
    }
    let captured = logs.contents();
    assert!(captured.contains("request completed"), "{captured}");
    for requested in REQUESTED {
        assert!(!rendered.contains(requested), "{rendered}");
        assert!(!captured.contains(requested), "{captured}");
    }
}

#[tokio::test]
async fn a_process_that_is_not_a_lane_does_not_look_at_the_requested_model() {
    let cloud = cloud_api("sk-live-customer").await;
    let cloud_url = cloud.uri();
    // What a proxy inside a CVM looks like: no backend token. And one that
    // was given a backend token there all the same: still inside a TEE, so
    // still not a lane.
    let inside_a_cvm: [Vec<(&str, &str)>; 2] = [
        vec![],
        vec![
            ("NON_TEE_DEPLOYMENT", ""),
            ("VLLM_BACKEND_TOKEN", "backend-secret"),
            ("CLOUD_API_URL", &cloud_url),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
        ],
    ];
    for extra in inside_a_cvm {
        let backend = MockServer::start().await;
        mount_completions(&backend, routes::ROUTE_CHAT_COMPLETIONS, 2).await;
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _metrics = metrics::set_default_local_recorder(&recorder);
        let backend_url = backend.uri();
        let mut env = vec![
            ("MODEL_NAME", ALPHA),
            ("VLLM_BASE_URL", backend_url.as_str()),
        ];
        env.extend(extra.iter().copied());
        let gateway = start_single(&env, Some(handle.clone()));
        assert_eq!(gateway.state.config.non_tee_deployment, extra.is_empty());
        for model in [ALPHA, REQUESTED[1]] {
            let response = gateway.app.clone().oneshot(chat(model)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            json_body(response).await;
        }
        backend.verify().await;
        let rendered = handle.render();
        assert!(
            !rendered.contains("request_model_match_total"),
            "{extra:?}: {rendered}"
        );
        assert!(!rendered.contains("model=\""), "{extra:?}: {rendered}");
    }
}

#[tokio::test]
async fn list_mode_counts_refused_models_without_naming_them() {
    let backend = MockServer::start().await;
    mount_completions(&backend, routes::ROUTE_CHAT_COMPLETIONS, 1).await;
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _logs = logs.capture();
    let gateway = start_list(
        &json!({"models": [{"id": ALPHA, "backend_urls": [backend.uri()]}]}),
        &[],
        Some(handle.clone()),
    );
    let response = gateway.app.clone().oneshot(chat(ALPHA)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    json_body(response).await;
    for requested in REQUESTED {
        let response = gateway.app.clone().oneshot(chat(requested)).await.unwrap();
        assert_model_not_found(response).await;
    }
    let request = post(
        routes::ROUTE_CHAT_COMPLETIONS,
        Some("test-token"),
        &body_for(None),
    );
    assert_model_not_found(gateway.app.clone().oneshot(request).await.unwrap()).await;

    let rendered = handle.render();
    for result in ["exact", "case_differs", "other", "missing"] {
        let line = format!("request_model_match_total{{result=\"{result}\"}} 1");
        assert!(rendered.contains(&line), "{line}: {rendered}");
    }
    assert!(
        rendered.contains("http_errors_total{error_type=\"model_not_found\"} 3"),
        "{rendered}"
    );
    let captured = logs.contents();
    assert!(captured.contains("Serving model"), "{captured}");
    for requested in REQUESTED {
        assert!(!rendered.contains(requested), "{rendered}");
        assert!(!captured.contains(requested), "{captured}");
    }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

#[tokio::test]
async fn in_list_mode_the_single_model_state_serves_nothing() {
    let seen = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200))
        .mount(&seen)
        .await;
    let backend = MockServer::start().await;
    let cloud = cloud_api("sk-live-customer").await;
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    // The environment of a single-model lane, minus its name and backends:
    // in list mode all of it is defaults for the entries.
    let gateway = start_list(
        &json!({"models": [{"id": ALPHA, "backend_urls": [backend.uri()]}]}),
        &[
            ("CLOUD_API_URL", &cloud.uri()),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
            ("VLLM_BACKEND_TOKEN", "shared-secret"),
            ("VLLM_BACKEND_PRIORITY", "-1"),
            ("VLLM_PROXY_ADMISSION_MAX_INFLIGHT", "8"),
            ("VLLM_BACKEND_CONVERSATION_AFFINITY", "1"),
            ("VLLM_PROXY_DISCOUNT_TO_USER", "0.2"),
        ],
        Some(handle.clone()),
    );
    let state = &gateway.state;

    // The model got them ...
    let model = gateway.model(ALPHA);
    assert!(model.admission.is_enabled());
    assert_eq!(model.config.discount_to_user, Some(0.2));
    model.backend_client.get(seen.uri()).send().await.unwrap();
    // ... and the process itself is no model: one backend that cannot
    // resolve, no budget, nothing to pin, nothing to bill as, and a client
    // without the bearer or the priority.
    let single_model_backends: Vec<_> = state
        .backend_pool
        .backends()
        .iter()
        .map(|backend| backend.base_url.as_str())
        .collect();
    assert_eq!(single_model_backends, [model_list::UNROUTED_BACKEND_URL]);
    assert!(model_list::UNROUTED_BACKEND_URL.ends_with(".invalid"));
    assert!(!state.admission.is_enabled());
    assert!(!state.backend_affinity.is_active());
    assert_eq!(state.config.model_name, "");
    assert!(state.config.backend_token.is_none());
    assert!(state.config.discount_to_user.is_none());
    state.backend_client.get(seen.uri()).send().await.unwrap();

    let requests = seen.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        bearer_of(&requests[0]).as_deref(),
        Some("Bearer shared-secret")
    );
    assert_eq!(
        requests[0]
            .headers
            .get(priority::PRIORITY_HEADER)
            .map(|value| value.to_str().unwrap()),
        Some("-1")
    );
    assert_eq!(bearer_of(&requests[1]), None);
    assert!(requests[1].headers.get(priority::PRIORITY_HEADER).is_none());

    // No lane without a model: every admission series is alpha's.
    let response = gateway
        .app
        .clone()
        .oneshot(get(routes::ROUTE_METRICS))
        .await
        .unwrap();
    let rendered = text_body(response).await;
    let admission: Vec<_> = series(&rendered)
        .into_iter()
        .filter(|series| series.starts_with("admission_"))
        .collect();
    assert!(admission.contains(&"admission_budget{model=\"example/alpha\"}".to_string()));
    for series in &admission {
        assert!(series.contains("model=\"example/alpha\""), "{series}");
    }
}

/// The built binary serving a list, until dropped.
struct Binary {
    child: std::process::Child,
    base: String,
    log: tempfile::NamedTempFile,
    _list: tempfile::NamedTempFile,
}

impl Drop for Binary {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Binary {
    /// Start `vllm-proxy-rs` on a free loopback port with `list` and `env`,
    /// and wait until it answers.
    async fn start(list: &Value, env: &[(&str, &str)]) -> Binary {
        let mut last_log = String::new();
        // The port is picked by binding and releasing it; if something else
        // takes it in between, the binary exits and another one is tried.
        for _ in 0..5 {
            let port = {
                let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                listener.local_addr().unwrap().port()
            };
            let mut list_file = tempfile::NamedTempFile::new().unwrap();
            list_file.write_all(list.to_string().as_bytes()).unwrap();
            let log = tempfile::NamedTempFile::new().unwrap();
            let child = std::process::Command::new(env!("CARGO_BIN_EXE_vllm-proxy-rs"))
                .env_clear()
                .envs([
                    ("TOKEN", "test-token"),
                    ("NON_TEE_DEPLOYMENT", "1"),
                    ("DEV", "1"),
                    ("GPU_NO_HW_MODE", "1"),
                    ("LOG_FORMAT", "json"),
                    ("LISTEN_ADDR", "127.0.0.1"),
                    ("CLOUD_API_AUTH_MAX_ATTEMPTS", "1"),
                    ("VLLM_PROXY_IMAGE_VALIDATION_DISABLED", "1"),
                ])
                .env("LISTEN_PORT", port.to_string())
                .env(model_list::MODEL_LIST_FILE_ENV, list_file.path())
                .envs(env.iter().copied())
                .stdin(std::process::Stdio::null())
                .stdout(log.reopen().unwrap())
                .stderr(log.reopen().unwrap())
                .spawn()
                .expect("the binary starts");
            let mut binary = Binary {
                child,
                base: format!("http://127.0.0.1:{port}"),
                log,
                _list: list_file,
            };
            let client = reqwest::Client::new();
            for _ in 0..400 {
                if binary.child.try_wait().unwrap().is_some() {
                    break;
                }
                let version = client.get(format!("{}/version", binary.base)).send().await;
                if version.is_ok_and(|response| response.status().is_success()) {
                    return binary;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            last_log = binary.log();
        }
        panic!("the binary did not come up: {last_log}");
    }

    fn log(&self) -> String {
        std::fs::read_to_string(self.log.path()).unwrap()
    }

    /// The `fields` of every JSON log line with this message.
    fn logged(&self, message: &str) -> Vec<Value> {
        self.log()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|line| line["fields"]["message"] == message)
            .map(|mut line| line["fields"].take())
            .collect()
    }
}

/// `main` itself, in list mode: the process the deploy starts, with the
/// environment and the file it is given, answering over a real socket.
#[tokio::test]
async fn the_binary_serves_a_model_list() {
    let alpha = MockServer::start().await;
    let beta = MockServer::start().await;
    for backend in [&alpha, &beta] {
        mount_completions(backend, routes::ROUTE_CHAT_COMPLETIONS, 1).await;
        Mock::given(method("GET"))
            .and(path("/healthz"))
            .respond_with(ResponseTemplate::new(200))
            .mount(backend)
            .await;
    }
    let alpha_engine = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_string(engine_metrics(3, 0)))
        .mount(&alpha_engine)
        .await;
    let cloud = cloud_api("sk-live-customer").await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"id": ALPHA, "name": "Alpha"}, {"id": BETA, "name": "Beta"}]
        })))
        .mount(&cloud)
        .await;

    let binary = Binary::start(
        &json!({"models": [
            {
                "id": ALPHA,
                "backend_urls": [alpha.uri()],
                "backend_probe_urls": [alpha_engine.uri()],
                "admission_max_inflight": 4,
                "discount_to_user": 0.3,
                "backend_token_env": "VLLM_BACKEND_TOKEN_ALPHA",
                "backend_priority": -1
            },
            {"id": BETA, "backend_urls": [beta.uri()]}
        ]}),
        &[
            ("CLOUD_API_URL", &cloud.uri()),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
            (
                "VLLM_PROXY_MODELS_DOCUMENT_URL",
                &format!("{}/v1/models", cloud.uri()),
            ),
            ("VLLM_BACKEND_TOKEN_ALPHA", "alpha-secret"),
            ("VLLM_BACKEND_HEALTH_PATH", "/healthz"),
            ("HEALTH_CHECK_INTERVAL_SECS", "1"),
            ("VLLM_BACKEND_PROBE_INTERVAL_SECS", "1"),
        ],
    )
    .await;
    let client = reqwest::Client::new();
    let url = |route: &str| format!("{}{route}", binary.base);
    let chat = |bearer: &'static str, model: &'static str| {
        client
            .post(url(routes::ROUTE_CHAT_COMPLETIONS))
            .bearer_auth(bearer)
            .json(&body_for(Some(json!(model))))
            .send()
    };

    // Each model answers from its own backend; another model is a 404, and
    // so is a route that needs the one model of a single-model process.
    let response = chat("sk-live-customer", ALPHA).await.unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert_eq!(
        response.json::<Value>().await.unwrap()["choices"][0]["message"]["content"],
        "hi"
    );
    let response = chat("test-token", BETA).await.unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let response = chat("test-token", GAMMA).await.unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
    assert_eq!(
        response.json::<Value>().await.unwrap()["error"]["code"],
        "model_not_found"
    );
    let response = client
        .post(url(routes::ROUTE_EMBEDDINGS))
        .bearer_auth("test-token")
        .json(&json!({"model": ALPHA, "input": "x"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);

    let models: Value = client
        .get(url(routes::ROUTE_V1_MODELS))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        models["data"],
        json!([
            {
                "id": ALPHA,
                "name": "Alpha",
                "capacity": [{"type": "concurrency", "unit": "request", "value": 4}],
                "discount_to_user": 0.3
            },
            {"id": BETA, "name": "Beta"}
        ])
    );

    // `/healthz`: the verdict for everyone, the models for the operator.
    let health = client.get(url(routes::ROUTE_HEALTHZ)).send().await.unwrap();
    assert_eq!(health.status(), reqwest::StatusCode::OK);
    assert_eq!(
        health.json::<Value>().await.unwrap(),
        json!({"status": "ok", "checks": {"dstack": "skipped", "backend": "ok"}})
    );
    let health: Value = client
        .get(url(routes::ROUTE_HEALTHZ))
        .bearer_auth("test-token")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        health["models"],
        json!([{"id": ALPHA, "backend": "ok"}, {"id": BETA, "backend": "ok"}])
    );

    // The customer's request was billed as alpha, at alpha's discount.
    let reports = usage_reports(&cloud, 1).await;
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0]["model"], ALPHA);
    assert_eq!(reports[0]["discount_to_user"].as_f64(), Some(0.3));

    // Every model's health checker and alpha's engine poller run in the
    // real process, and the scrape tells the models apart.
    let mut rendered = String::new();
    let waited_for = [
        "backend_pool_healthy{model=\"example/alpha\"} 1",
        "backend_pool_healthy{model=\"example/beta\"} 1",
        "backend_engine_running{backend=\"0\",model=\"example/alpha\"} 3",
        "inference_proxy_usage_reports_total{outcome=\"accepted\"",
    ];
    for _ in 0..300 {
        rendered = client
            .get(url(routes::ROUTE_METRICS))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        if waited_for.iter().all(|series| rendered.contains(series)) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    for expected in waited_for.into_iter().chain([
        "admission_budget{model=\"example/alpha\"} 4",
        "request_model_match_total{result=\"exact\"} 2",
        "request_model_match_total{result=\"other\"} 1",
        "http_errors_total{error_type=\"model_not_found\"} 1",
        // The HTTP metrics middleware is part of what is served.
        "http_requests_total{method=\"POST\",endpoint=\"/v1/chat/completions\",status=\"200\"} 2",
        "http_requests_total{method=\"POST\",endpoint=\"/v1/chat/completions\",status=\"404\"} 1",
    ]) {
        assert!(rendered.contains(expected), "{expected}: {rendered}");
    }
    // The process itself is no lane: no admission series without a model.
    for series in series(&rendered) {
        if series.starts_with("admission_") || series.starts_with("backend_pool_") {
            assert!(series.contains("model=\""), "{series}");
        }
    }

    // alpha's bearer and priority went to alpha's inference requests and
    // nowhere else: not to the probes, not to beta, not to cloud-api.
    let inference = requests_to(&alpha, routes::ROUTE_CHAT_COMPLETIONS).await;
    assert_eq!(inference.len(), 1);
    assert_eq!(
        bearer_of(&inference[0]).as_deref(),
        Some("Bearer alpha-secret")
    );
    assert_eq!(
        inference[0].headers.get(priority::PRIORITY_HEADER).unwrap(),
        "-1"
    );
    let probes: Vec<_> = requests_to(&alpha, "/healthz")
        .await
        .into_iter()
        .chain(requests_to(&alpha_engine, "/v1/metrics").await)
        .collect();
    assert!(probes.len() >= 3, "{}", probes.len());
    for request in probes
        .iter()
        .chain(&beta.received_requests().await.unwrap())
        .chain(&cloud.received_requests().await.unwrap())
    {
        let raw = format!(
            "{:?}{}",
            request.headers,
            String::from_utf8_lossy(&request.body)
        );
        assert!(!raw.contains("alpha-secret"), "{}", request.url);
        assert!(request.headers.get(priority::PRIORITY_HEADER).is_none());
    }

    // What the process says at startup, and what it must not say.
    let starting = binary.logged("Starting vllm-proxy-rs");
    assert_eq!(starting.len(), 1);
    assert!(starting[0].get("model").is_none(), "{}", starting[0]);
    let serving = binary.logged("Serving model");
    let served: Vec<_> = serving.iter().map(|line| line["model"].clone()).collect();
    assert_eq!(served, [ALPHA, BETA]);
    assert_eq!(serving[0]["backend_token"], true);
    assert_eq!(serving[0]["engine_probes"], true);
    assert_eq!(serving[0]["admission_max_inflight"], 4);
    assert_eq!(serving[1]["backend_token"], false);
    let enabled = binary.logged("Model list enabled");
    assert_eq!(enabled.len(), 1);
    assert_eq!(enabled[0]["models"], 2);
    assert_eq!(binary.logged("Rate limiter configured").len(), 1);
    let log = binary.log();
    for secret in [
        "alpha-secret",
        "usage-secret",
        "test-token",
        "sk-live-customer",
    ] {
        assert!(!log.contains(secret), "{secret} in the log");
    }
    alpha.verify().await;
    beta.verify().await;
}

// ---------------------------------------------------------------------------
// Usage report delivery (`VLLM_PROXY_USAGE_REPORT_*`)
// ---------------------------------------------------------------------------

/// The delivery series follow the rule of the usage-report series that were
/// there before them: `model` in list mode, no such label for one model.
#[tokio::test]
async fn usage_report_delivery_series_carry_the_model_label_in_list_mode_only() {
    let lane = [
        ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
        ("VLLM_PROXY_USAGE_REPORT_TIMEOUT_SECS", "30"),
        ("VLLM_PROXY_USAGE_REPORT_MAX_ATTEMPTS", "5"),
        ("VLLM_PROXY_USAGE_REPORT_DEADLINE_SECS", "300"),
        ("VLLM_PROXY_USAGE_REPORT_MAX_IN_FLIGHT", "8"),
    ];
    for listed in [false, true] {
        let backend = MockServer::start().await;
        mount_completions(&backend, routes::ROUTE_CHAT_COMPLETIONS, 1).await;
        let cloud = cloud_api("sk-live-customer").await;
        let recorder = PrometheusBuilder::new().build_recorder();
        let _metrics = metrics::set_default_local_recorder(&recorder);
        let (backend_url, cloud_url) = (backend.uri(), cloud.uri());
        let mut env = lane.to_vec();
        env.push(("CLOUD_API_URL", cloud_url.as_str()));
        let gateway = if listed {
            start_list(
                &json!({"models": [{"id": ALPHA, "backend_urls": [backend_url]}]}),
                &env,
                Some(recorder.handle()),
            )
        } else {
            env.extend([
                ("MODEL_NAME", ALPHA),
                ("VLLM_BACKEND_URLS", backend_url.as_str()),
            ]);
            start_single(&env, Some(recorder.handle()))
        };
        // One delivery for the process, with the settings of its environment.
        let policy = gateway.state.usage_report_delivery.policy();
        assert_eq!(policy, &gateway.state.config.usage_report);
        assert_eq!((policy.max_attempts, policy.max_in_flight), (5, 8));

        let customer = post(
            routes::ROUTE_CHAT_COMPLETIONS,
            Some("sk-live-customer"),
            &body_for(Some(json!(ALPHA))),
        );
        let response = gateway.app.clone().oneshot(customer).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        json_body(response).await;
        let drained = gateway
            .state
            .usage_report_delivery
            .drain(Duration::from_secs(30))
            .await;
        assert_eq!(drained.left(), 0);
        assert_eq!(usage_reports(&cloud, 1).await.len(), 1);

        let response = gateway
            .app
            .clone()
            .oneshot(get(routes::ROUTE_METRICS))
            .await
            .unwrap();
        let rendered = text_body(response).await;
        let usage: Vec<String> = series(&rendered)
            .into_iter()
            .filter(|series| series.starts_with("inference_proxy_usage_report"))
            .collect();
        for family in [
            "inference_proxy_usage_reports_total{outcome=\"accepted\"",
            "inference_proxy_usage_report_duration_seconds_count{outcome=\"accepted\"",
            "inference_proxy_usage_report_attempts_total{outcome=\"accepted\"",
            "inference_proxy_usage_report_time_to_accepted_seconds_count",
            "inference_proxy_usage_reports_in_flight",
        ] {
            assert!(
                usage.iter().any(|series| series.starts_with(family)),
                "{family} missing: {usage:#?}"
            );
        }
        for series in &usage {
            let (_, had_model) = without_model(series);
            assert_eq!(had_model, listed, "{series}");
            assert_eq!(
                series.contains("model=\"example/alpha\""),
                listed,
                "{series}"
            );
        }
        backend.verify().await;
    }
}

/// `main` at shutdown. With a drain time, a usage report the billing API has
/// not answered yet is waited for after SIGTERM, and the process says so.
/// With no usage-report setting it exits at once, as it always has.
#[cfg(unix)]
#[tokio::test]
async fn the_binary_waits_for_pending_usage_reports_at_shutdown_only_when_told_to() {
    const ACCEPTED: &str = "Direct-key usage report accepted by Cloud API";
    const RETRYING: &str = "Usage report attempt failed, retrying";
    const DRAINED: &str = "Usage reports drained before shutdown";
    for configured in [true, false] {
        let backend = MockServer::start().await;
        mount_completions(&backend, routes::ROUTE_CHAT_COMPLETIONS, 1).await;
        // A cloud-api whose usage intake takes its time, after one failure
        // for the process that retries.
        let cloud = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/check_api_key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "valid": true,
                "organization_id": "org",
                "workspace_id": "ws",
                "api_key_id": "k"
            })))
            .mount(&cloud)
            .await;
        if configured {
            Mock::given(method("POST"))
                .and(path("/v1/internal/usage"))
                .respond_with(ResponseTemplate::new(503))
                .up_to_n_times(1)
                .mount(&cloud)
                .await;
        }
        Mock::given(method("POST"))
            .and(path("/v1/internal/usage"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(2_500)))
            .mount(&cloud)
            .await;
        let (cloud_url, catalog_url) = (cloud.uri(), format!("{}/v1/models", cloud.uri()));
        let mut env = vec![
            ("CLOUD_API_URL", cloud_url.as_str()),
            ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
            ("VLLM_PROXY_MODELS_DOCUMENT_URL", catalog_url.as_str()),
        ];
        if configured {
            env.extend([
                ("VLLM_PROXY_USAGE_REPORT_MAX_ATTEMPTS", "3"),
                ("VLLM_PROXY_USAGE_REPORT_INITIAL_BACKOFF_MS", "20"),
                ("VLLM_PROXY_USAGE_REPORT_MAX_IN_FLIGHT", "4"),
                ("VLLM_PROXY_USAGE_REPORT_SHUTDOWN_DRAIN_SECS", "30"),
            ]);
        }
        let mut binary = Binary::start(
            &json!({"models": [{"id": ALPHA, "backend_urls": [backend.uri()]}]}),
            &env,
        )
        .await;

        // A customer request is answered while its report is on its way.
        {
            let response = reqwest::Client::new()
                .post(format!("{}{}", binary.base, routes::ROUTE_CHAT_COMPLETIONS))
                .bearer_auth("sk-live-customer")
                .json(&body_for(Some(json!(ALPHA))))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            response.bytes().await.unwrap();
        }
        // The attempt the intake is slow to answer is in flight: the first
        // one, or the retry after the 503.
        let attempts = if configured { 2 } else { 1 };
        let reports = requests_to_eventually(&cloud, "/v1/internal/usage", attempts).await;
        assert!(binary.logged(ACCEPTED).is_empty());

        let terminated = std::process::Command::new("kill")
            .args(["-TERM", &binary.child.id().to_string()])
            .status()
            .unwrap();
        assert!(terminated.success());
        let mut exit = None;
        for _ in 0..600 {
            exit = binary.child.try_wait().unwrap();
            if exit.is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(exit.expect("the process exits").success());

        let log = binary.log();
        let at = |message: &str| log.find(message);
        assert_eq!(binary.logged("Server shut down").len(), 1, "{log}");
        if configured {
            // The retry carried the bytes of the first attempt.
            assert_eq!(reports[0].body, reports[1].body);
            // The report was accepted after the signal and before the exit.
            let accepted = binary.logged(ACCEPTED);
            assert_eq!(accepted.len(), 1, "{log}");
            assert_eq!(accepted[0]["attempts"], 2);
            // The line a report has always ended with keeps its log target.
            let targets: Vec<Value> = log
                .lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .filter(|line| line["fields"]["message"] == ACCEPTED)
                .map(|mut line| line["target"].take())
                .collect();
            assert_eq!(targets, ["vllm_proxy_rs::proxy"]);
            let drained = binary.logged(DRAINED);
            assert_eq!(drained.len(), 1, "{log}");
            assert_eq!(drained[0]["pending"], 1);
            assert!(at("Shutdown signal received") < at(ACCEPTED), "{log}");
            assert!(at(ACCEPTED) < at(DRAINED), "{log}");
            assert!(at(DRAINED) < at("Server shut down"), "{log}");
            assert_eq!(binary.logged("Usage report delivery configured").len(), 1);
            // A failed attempt is logged with ids, labels and numbers only.
            let retrying = binary.logged(RETRYING);
            assert_eq!(retrying.len(), 1, "{log}");
            let fields: BTreeSet<&str> = retrying[0]
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            assert_eq!(
                fields,
                BTreeSet::from([
                    "message",
                    "request_id",
                    "org_id",
                    "workspace_id",
                    "api_key_id",
                    "model",
                    "status",
                    "duration_ms",
                    "auth_path",
                    "ingress_route",
                    "outcome",
                    "attempt",
                    "max_attempts",
                    "retry_in_ms",
                ])
            );
            assert_eq!(retrying[0]["status"], "503 Service Unavailable");
            assert_eq!(retrying[0]["outcome"], "http_5xx");
        } else {
            // Nothing was waited for and nothing new was said.
            let accepted = binary.logged(ACCEPTED);
            assert!(accepted.is_empty(), "{log}");
            assert!(!log.contains("Usage reports"), "{log}");
            assert!(!log.contains("Usage report delivery"), "{log}");
        }
        // No secret and nothing of the request's content in the log.
        for unwanted in ["usage-secret", "sk-live-customer", "hello"] {
            assert!(!log.contains(unwanted), "{unwanted} in the log");
        }
        backend.verify().await;
    }
}

/// The requests `server` got on `route`, once there are `expected` of them.
async fn requests_to_eventually(
    server: &MockServer,
    route: &str,
    expected: usize,
) -> Vec<wiremock::Request> {
    for _ in 0..250 {
        let requests = requests_to(server, route).await;
        if requests.len() >= expected {
            return requests;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("expected {expected} requests on {route}");
}
