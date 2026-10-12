//! Per-organization model access (`VLLM_PROXY_MODEL_ACCESS_REFRESH_SECS`,
//! gateway mode).
//!
//! cloud-api can restrict a model to a set of organizations, and applies that
//! on its own inference path. A gateway lane is not on that path: it checks
//! the key and bills the usage itself, so without this a valid key of any
//! organization reaches a restricted model through the lane. cloud-api
//! publishes the rule as one document, and the gateway applies it:
//!
//! ```text
//! GET {CLOUD_API_URL}/v1/internal/model-access
//! Authorization: Bearer <CLOUD_API_USAGE_TOKEN>
//!
//! {"object": "model_access",
//!  "models": [{"id": "<model name>", "organization_ids": ["<uuid>", …]}]}
//! ```
//!
//! The document lists the restricted models, each with the organizations that
//! may use it, possibly none. An `id` is the model's canonical catalog name,
//! the string a gateway is configured with (`MODEL_NAME`, a list entry's
//! `id`). A model the document does not list is open to every valid key.
//!
//! It is read once before the process accepts its first request (`start`),
//! and then on the configured interval, off the request path. The latest one
//! is kept in memory: a request only ever reads that snapshot. A revocation
//! in cloud-api therefore applies here one interval and one read later at
//! most, as long as reads succeed.
//!
//! A request made with a cloud-api key for a restricted model, from an
//! organization that is not on the model's list, gets the answer of a model
//! this process does not serve (`AppError::ModelNotFound`, with the model
//! string the caller sent). A key whose check returned no organization id is
//! refused the same way. A caller that presents the proxy's own config token
//! is not checked: it is trusted to have applied the rule itself.
//!
//! A read that fails leaves the last snapshot in force, however old it is: an
//! outage of cloud-api must not open a restricted model, nor close one to the
//! organizations that have it. Until a first document has been read nothing
//! is restricted, and the read is tried again every few seconds rather than
//! once an interval, so that state does not last. `model_access_enforcing`
//! says which of the two a process is in. One failure is not waited out: an
//! answer of 401 or 403 to the read at startup means the token is not the
//! one cloud-api expects, so the process would never enforce anything, and it
//! does not start. The static `VLLM_PROXY_ALLOWED_ORG_IDS` list applies
//! throughout.
//!
//! The read follows no redirect and takes at most 1 MiB, and the service
//! token is sent to that one URL only.
//!
//! Unset or `0`, the default, none of this runs: no request to cloud-api, no
//! check and no series.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::header::{HeaderValue, AUTHORIZATION};
use reqwest::StatusCode;
use serde::Deserialize;
use tracing::{info, warn};

use crate::auth::RequireAuth;
use crate::config::Config;
use crate::error::AppError;
use crate::model_metrics::{model_counter, ModelLabel};
use crate::AppState;

/// cloud-api's route for the document, under `CLOUD_API_URL`.
pub const MODEL_ACCESS_PATH: &str = "/v1/internal/model-access";

/// Timeout of one read of the document.
const READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Time between two reads while no document has been read yet. Nothing is
/// restricted in that state, so it is not left to last a whole interval.
const RETRY_WITHOUT_DOCUMENT: Duration = Duration::from_secs(5);

/// Largest document that is read. A cloud-api with thousands of restricted
/// models and grants stays far below it; anything larger is not the document.
const MAX_DOCUMENT_BYTES: usize = 1024 * 1024;

/// The restricted models of one document: for each, the organizations that
/// may use it.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    allowed: HashMap<String, HashSet<String>>,
}

#[derive(Deserialize)]
struct Document {
    models: Vec<DocumentModel>,
}

#[derive(Deserialize)]
struct DocumentModel {
    id: String,
    organization_ids: Vec<String>,
}

impl Snapshot {
    /// Parse cloud-api's document. Fields this version does not know are
    /// ignored. A document without `models`, with an entry that lacks its
    /// `id` or its `organization_ids`, or with a model listed twice is
    /// refused as a whole: half a document, or a guess at which of two lists
    /// applies, would restrict or open something cloud-api did not say.
    pub fn parse(body: &[u8]) -> anyhow::Result<Self> {
        let document: Document = serde_json::from_slice(body)?;
        let mut allowed = HashMap::with_capacity(document.models.len());
        for model in document.models {
            match allowed.entry(model.id) {
                Entry::Occupied(listed) => {
                    anyhow::bail!("model {:?} is listed twice", listed.key())
                }
                Entry::Vacant(new) => {
                    new.insert(model.organization_ids.into_iter().collect());
                }
            }
        }
        Ok(Self { allowed })
    }

    /// Whether a cloud-api key of organization `org_id` may use `model`: the
    /// model is not restricted, or the organization is on its list. Both are
    /// compared byte for byte, as list mode matches a request's `model` and
    /// the static allow-list matches an organization. A key without an
    /// organization id cannot be on a list.
    pub fn allows(&self, model: &str, org_id: Option<&str>) -> bool {
        match self.allowed.get(model) {
            None => true,
            Some(organizations) => org_id.is_some_and(|org| organizations.contains(org)),
        }
    }

    fn restricted_models(&self) -> usize {
        self.allowed.len()
    }

    /// The models among `served` that the document restricts, each with the
    /// number of organizations on its list, as one line for the log. Model
    /// ids and counts only: never an organization id. `None` when it
    /// restricts none of them.
    fn restricted_among(&self, served: &[String]) -> Option<String> {
        let restricted: Vec<String> = served
            .iter()
            .filter_map(|id| {
                let organizations = self.allowed.get(id)?.len();
                Some(format!("{id} (organizations: {organizations})"))
            })
            .collect();
        (!restricted.is_empty()).then(|| restricted.join(", "))
    }
}

/// The decision for one request: whether it may be served as `model`.
///
/// `snapshot` is `None` until a first document has been read, and always when
/// the setting is off: nothing is restricted then. `customer_key` is false
/// for a caller that authenticated with the proxy's own config token, which
/// is never checked.
fn permits(
    snapshot: Option<&Snapshot>,
    customer_key: bool,
    org_id: Option<&str>,
    model: &str,
) -> bool {
    !customer_key || snapshot.is_none_or(|snapshot| snapshot.allows(model, org_id))
}

/// Why a read of the document did not produce a snapshot. `reason` is the
/// label of `model_access_refresh_failures_total`.
enum Failure {
    /// 404: this cloud-api does not serve the document.
    RouteMissing,
    /// 401 or 403: cloud-api does not accept the service token.
    Unauthorized(StatusCode),
    /// Any 3xx. The read follows no redirect: the token and the trust in
    /// the answer are for the URL it was sent to.
    Redirect(StatusCode),
    Status(StatusCode),
    Transport(reqwest::Error),
    /// Not the document: not valid, a model listed twice, or too large.
    Malformed(String),
}

impl Failure {
    fn reason(&self) -> &'static str {
        match self {
            Failure::RouteMissing => "route_missing",
            Failure::Unauthorized(_) => "unauthorized",
            Failure::Redirect(_) => "redirect",
            Failure::Status(_) => "status",
            Failure::Transport(_) => "transport",
            Failure::Malformed(_) => "malformed",
        }
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::RouteMissing => f.write_str("cloud-api answered 404"),
            Failure::Unauthorized(status) => {
                write!(f, "cloud-api answered {status}: the token is not accepted")
            }
            Failure::Redirect(status) => {
                write!(f, "cloud-api answered {status}: a redirect is not followed")
            }
            Failure::Status(status) => write!(f, "cloud-api answered {status}"),
            Failure::Transport(error) => write!(f, "{error}"),
            Failure::Malformed(what) => write!(f, "not a model access document: {what}"),
        }
    }
}

/// When the document is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timing {
    /// Time between two reads once a document is in force
    /// (`VLLM_PROXY_MODEL_ACCESS_REFRESH_SECS`).
    pub refresh: Duration,
    /// Time between two reads while none is, when that is shorter: 5 s.
    pub retry_without_document: Duration,
    /// Timeout of one read: 5 s.
    pub read_timeout: Duration,
}

impl Timing {
    /// The timing of a process whose setting is `refresh`.
    pub fn new(refresh: Duration) -> Self {
        Self {
            refresh,
            retry_without_document: RETRY_WITHOUT_DOCUMENT,
            read_timeout: READ_TIMEOUT,
        }
    }
}

/// Where the document is read from and how. Not `Debug`: it holds the token.
struct Source {
    url: String,
    /// `Bearer <CLOUD_API_USAGE_TOKEN>`, for this one URL.
    authorization: HeaderValue,
    /// A client that follows no redirect (`ModelAccess::new`).
    client: reqwest::Client,
    timing: Timing,
    /// The ids of the models this process serves: `MODEL_NAME`, or the ids
    /// of its list. What a new document says about them goes to the log.
    served: Vec<String>,
}

/// `Bearer <token>` as the header of the read, with the white space around
/// the token removed (a token read from a file often ends in a newline).
/// Fails for a token that cannot be a header value, without echoing it.
pub(crate) fn bearer(token: &str) -> anyhow::Result<HeaderValue> {
    let mut value = HeaderValue::from_str(&format!("Bearer {}", token.trim())).map_err(|_| {
        anyhow::anyhow!(
            "CLOUD_API_USAGE_TOKEN is not a valid HTTP header value, and VLLM_PROXY_MODEL_ACCESS_REFRESH_SECS sends it as the bearer of the model access read"
        )
    })?;
    value.set_sensitive(true);
    Ok(value)
}

/// `ModelAccess::last_refresh` after a read that produced a snapshot. After
/// one that did not, it holds the `Failure::reason`.
const READ: &str = "read";

/// The process's view of cloud-api's model access document
/// (`AppState::model_access`). One per process, shared by every model of a
/// list: the document covers all of cloud-api's models.
#[derive(Default)]
pub struct ModelAccess {
    /// `None` = the setting is off: nothing is read and nothing is checked.
    source: Option<Source>,
    /// The last document read. `None` until there is one.
    snapshot: RwLock<Option<Arc<Snapshot>>>,
    /// How the last read ended (`READ` or a `Failure::reason`), `None` before
    /// the first one. A read that ends like the one before it is not logged
    /// again: the log has the changes, not every poll.
    last_refresh: Mutex<Option<&'static str>>,
}

impl ModelAccess {
    /// Off unless `VLLM_PROXY_MODEL_ACCESS_REFRESH_SECS` is set. Nothing is
    /// read here: `start` does that before the process serves, and then on
    /// the configured interval.
    ///
    /// `no_redirect_client` builds the client of the read, with the
    /// process's own connection settings and no redirect followed. It is
    /// only called when the setting is on. Fails, as startup does, when the
    /// setting is on without `CLOUD_API_URL` or with a token that cannot be
    /// sent as a header.
    pub fn new(
        config: &Config,
        no_redirect_client: &dyn Fn() -> anyhow::Result<reqwest::Client>,
    ) -> anyhow::Result<Self> {
        if config.model_access_refresh_secs == 0 {
            return Ok(Self::default());
        }
        let (Some(base), Some(token)) = (&config.cloud_api_url, &config.cloud_api_usage_token)
        else {
            anyhow::bail!(
                "VLLM_PROXY_MODEL_ACCESS_REFRESH_SECS requires CLOUD_API_URL and CLOUD_API_USAGE_TOKEN"
            );
        };
        let served = match &config.model_list {
            Some(list) => list.models.iter().map(|model| model.id.clone()).collect(),
            None => vec![config.model_name.clone()],
        };
        let source = Source {
            url: format!("{}{MODEL_ACCESS_PATH}", base.trim_end_matches('/')),
            authorization: bearer(token)?,
            client: no_redirect_client()?,
            timing: Timing::new(Duration::from_secs(config.model_access_refresh_secs)),
            served,
        };
        Ok(Self {
            source: Some(source),
            ..Self::default()
        })
    }

    /// Other intervals and another read timeout than the process's, for a
    /// test that cannot wait for them. Nothing changes when the setting is
    /// off.
    pub fn with_timing(mut self, timing: Timing) -> Self {
        if let Some(source) = &mut self.source {
            source.timing = timing;
        }
        self
    }

    /// The setting is on.
    pub fn is_enabled(&self) -> bool {
        self.source.is_some()
    }

    /// A document has been read: restricted models are enforced.
    pub fn has_snapshot(&self) -> bool {
        self.current().is_some()
    }

    fn current(&self) -> Option<Arc<Snapshot>> {
        self.snapshot
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Whether the caller may have a request served as `model`: the model
    /// the request would be served as, so the configured one, or in list
    /// mode the entry the body's `model` selected. A refusal is counted in
    /// `model_access_rejections_total` (`model` label in list mode), and
    /// nothing is logged per request.
    ///
    /// For a route with an answer of its own for what it does not have; the
    /// routes that take a `model` use `check`.
    pub fn allows(&self, auth: &RequireAuth, model: &str, label: ModelLabel) -> bool {
        if self.source.is_none() {
            return true;
        }
        let snapshot = self.current();
        let allowed = permits(
            snapshot.as_deref(),
            auth.cloud_api_key.is_some(),
            auth.org_id.as_deref(),
            model,
        );
        if !allowed {
            model_counter!(label, "model_access_rejections_total").increment(1);
        }
        allowed
    }

    /// Refuse a request whose organization may not use `model`, with the
    /// answer of a model this process does not serve. `requested` gives the
    /// `model` string the caller sent, which that answer echoes as the one
    /// for an unknown model does: never a configured id the caller did not
    /// name. It is only called for a refusal.
    ///
    /// Called after authentication and model selection, before anything is
    /// counted for the model, admitted, dispatched or billed.
    pub fn check(
        &self,
        auth: &RequireAuth,
        model: &str,
        label: ModelLabel,
        requested: impl FnOnce() -> Option<String>,
    ) -> Result<(), AppError> {
        if self.allows(auth, model, label) {
            return Ok(());
        }
        Err(AppError::ModelNotFound { model: requested() })
    }

    /// Read the document once and keep it. A failure keeps what was read
    /// before. Does nothing when the setting is off.
    pub async fn refresh(&self) {
        let _ = self.read(false).await;
    }

    /// `refresh`, saying how the read ended. `at_startup` is the read
    /// `start` waits for, whose refused token ends the process: the line
    /// about trying again is not written for it.
    async fn read(&self, at_startup: bool) -> Result<(), Failure> {
        let Some(source) = &self.source else {
            return Ok(());
        };
        match read_document(source).await {
            Ok(snapshot) => {
                self.keep(source, snapshot);
                Ok(())
            }
            Err(failure) => {
                let fatal = at_startup && matches!(failure, Failure::Unauthorized(_));
                self.note_failure(source, &failure, fatal);
                Err(failure)
            }
        }
    }

    fn keep(&self, source: &Source, snapshot: Snapshot) {
        let current = Arc::new(snapshot);
        // Requests read the snapshot under this lock, so it is held for the
        // swap alone. The document it replaces is compared, and freed, once
        // the lock is released.
        let previous = self
            .snapshot
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .replace(current.clone());
        let changed = previous.as_deref() != Some(&*current);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |since_epoch| since_epoch.as_secs_f64());
        metrics::gauge!("model_access_last_refresh_timestamp_seconds").set(now);
        metrics::gauge!("model_access_enforcing").set(1.0);
        let recovered = self.note(READ);
        // One line per document that differs from the one before it, and it
        // names what the document means for the models served here: that is
        // where an id that does not match a configured one shows.
        if changed {
            let document_models = current.restricted_models();
            match current.restricted_among(&source.served) {
                Some(restricted) => info!(
                    document_models,
                    restricted = %restricted,
                    "Model access document applied: it restricts these models served here"
                ),
                None => info!(
                    document_models,
                    served = ?source.served,
                    "Model access document applied: none of the models served here is in it, so none is restricted"
                ),
            }
        } else if recovered {
            info!("Model access document read again, unchanged");
        }
    }

    fn note_failure(&self, source: &Source, failure: &Failure, fatal: bool) {
        let reason = failure.reason();
        metrics::counter!("model_access_refresh_failures_total", "reason" => reason).increment(1);
        let enforcing = self.has_snapshot();
        if !enforcing {
            metrics::gauge!("model_access_enforcing").set(0.0);
        }
        if !self.note(reason) || fatal {
            return;
        }
        if enforcing {
            warn!(
                reason,
                error = %failure,
                "Model access document could not be read, keeping the last one"
            );
        } else {
            warn!(
                reason,
                error = %failure,
                retry_secs = source.timing.without_document().as_secs_f64(),
                "Model access document could not be read: model access is NOT enforced until it can"
            );
        }
    }

    /// Record how a read ended, and say whether that differs from the read
    /// before it.
    fn note(&self, outcome: &'static str) -> bool {
        self.last_refresh
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .replace(outcome)
            != Some(outcome)
    }
}

impl Timing {
    /// Time between two reads while no document is in force: the retry
    /// interval, unless the setting asks for reads more often than that.
    fn without_document(&self) -> Duration {
        self.refresh.min(self.retry_without_document)
    }
}

/// One read of the document. The service token is attached to this request
/// only and the client follows no redirect, so the token goes to
/// `{CLOUD_API_URL}/v1/internal/model-access` and nowhere else.
async fn read_document(source: &Source) -> Result<Snapshot, Failure> {
    let response = source
        .client
        .get(&source.url)
        .header(AUTHORIZATION, source.authorization.clone())
        .timeout(source.timing.read_timeout)
        .send()
        .await
        .map_err(Failure::Transport)?;
    let status = response.status();
    if !status.is_success() {
        return Err(match status {
            StatusCode::NOT_FOUND => Failure::RouteMissing,
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Failure::Unauthorized(status),
            status if status.is_redirection() => Failure::Redirect(status),
            status => Failure::Status(status),
        });
    }
    let body = read_capped(response).await?;
    Snapshot::parse(&body).map_err(|error| Failure::Malformed(error.to_string()))
}

/// The body of the answer, up to `MAX_DOCUMENT_BYTES`. A longer one is not
/// read to its end.
async fn read_capped(mut response: reqwest::Response) -> Result<Vec<u8>, Failure> {
    let too_large = || Failure::Malformed(format!("larger than {MAX_DOCUMENT_BYTES} bytes"));
    if response
        .content_length()
        .is_some_and(|length| length > MAX_DOCUMENT_BYTES as u64)
    {
        return Err(too_large());
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(Failure::Transport)? {
        if body.len() + chunk.len() > MAX_DOCUMENT_BYTES {
            return Err(too_large());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Keep the document fresh for the life of the process: a read every
/// interval, counted from the end of the read before it, so a slow cloud-api
/// is never asked twice at once nor again the moment it has answered. While
/// no document has been read yet the wait is the shorter retry interval.
/// The first read is `start`'s: this task begins with a wait. Does nothing
/// when the setting is off.
pub fn spawn_refresh(access: Arc<ModelAccess>) {
    let Some(timing) = access.source.as_ref().map(|source| source.timing) else {
        return;
    };
    tokio::spawn(async move {
        loop {
            let wait = if access.has_snapshot() {
                timing.refresh
            } else {
                timing.without_document()
            };
            tokio::time::sleep(wait).await;
            access.refresh().await;
        }
    });
}

/// What a process does before it accepts requests, when
/// `VLLM_PROXY_MODEL_ACCESS_REFRESH_SECS` is set: read the document once and
/// wait for the answer, so that the first request is already checked against
/// it, then start the task that keeps it fresh. The wait is one read, so at
/// most the read's timeout.
///
/// A first read that fails or times out does not hold the process back or
/// stop it: it starts with nothing restricted, as before any document has
/// been read, and the task tries again every few seconds. The exception is
/// an answer of 401 or 403: the token is not the one cloud-api expects, no
/// later read would do better, and a process that would never enforce model
/// access does not start.
///
/// Does nothing when the setting is off. Must run inside a Tokio runtime.
pub async fn start(state: &AppState) -> anyhow::Result<()> {
    let access = &state.model_access;
    let Some(source) = &access.source else {
        return Ok(());
    };
    metrics::gauge!("model_access_enforcing").set(0.0);
    info!(
        refresh_secs = source.timing.refresh.as_secs(),
        "Model access: reading cloud-api's document before serving"
    );
    if let Err(Failure::Unauthorized(status)) = access.read(true).await {
        anyhow::bail!(
            "VLLM_PROXY_MODEL_ACCESS_REFRESH_SECS: cloud-api answered {status} to the read of {MODEL_ACCESS_PATH}. CLOUD_API_USAGE_TOKEN is not the service token it expects, so this process would never enforce model access"
        );
    }
    spawn_refresh(access.clone());
    Ok(())
}

/// `ModelAccess::check` for a route only a single-model process serves: the
/// model is the configured one, whatever the request names.
pub fn check_single_model(
    state: &AppState,
    auth: &RequireAuth,
    requested: impl FnOnce() -> Option<String>,
) -> Result<(), AppError> {
    state
        .model_access
        .check(auth, &state.config.model_name, None, requested)
}

/// The `model` a JSON request body names, for the message of a refusal on a
/// route that forwards its body without reading it. `None` when the body is
/// not a JSON object or has no `model` string.
pub(crate) fn model_in_json(body: &[u8]) -> Option<String> {
    let body: serde_json::Value = serde_json::from_slice(body).ok()?;
    body.get("model")?.as_str().map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(document: serde_json::Value) -> Snapshot {
        Snapshot::parse(document.to_string().as_bytes()).expect("a valid document")
    }

    #[test]
    fn parses_the_document_and_ignores_unknown_fields() {
        let parsed = snapshot(serde_json::json!({
            "object": "model_access",
            "generated_at": "2026-01-01T00:00:00Z",
            "models": [
                {"id": "vendor/alpha", "organization_ids": ["org-a", "org-b"], "note": "x"},
                {"id": "vendor/beta", "organization_ids": []}
            ]
        }));
        assert_eq!(parsed.restricted_models(), 2);
        assert!(parsed.allows("vendor/alpha", Some("org-a")));
        assert!(parsed.allows("vendor/alpha", Some("org-b")));
        assert!(!parsed.allows("vendor/alpha", Some("org-c")));
        // Listed with nobody on its list: restricted to no one.
        assert!(!parsed.allows("vendor/beta", Some("org-a")));
        // Not listed: not restricted.
        assert!(parsed.allows("vendor/gamma", Some("org-c")));
        assert!(parsed.allows("vendor/gamma", None));
    }

    #[test]
    fn an_empty_document_restricts_nothing() {
        let parsed = snapshot(serde_json::json!({"object": "model_access", "models": []}));
        assert_eq!(parsed, Snapshot::default());
        assert!(parsed.allows("vendor/alpha", None));
    }

    #[test]
    fn a_model_listed_twice_is_not_a_document() {
        // Which of the two lists applies is not for the gateway to guess,
        // whether or not they say the same.
        for second in [serde_json::json!(["org-b"]), serde_json::json!(["org-a"])] {
            let body = serde_json::json!({"models": [
                {"id": "vendor/alpha", "organization_ids": ["org-a"]},
                {"id": "vendor/beta", "organization_ids": []},
                {"id": "vendor/alpha", "organization_ids": second}
            ]})
            .to_string();
            let error = Snapshot::parse(body.as_bytes()).unwrap_err().to_string();
            assert_eq!(error, "model \"vendor/alpha\" is listed twice");
        }
    }

    #[test]
    fn a_new_document_is_described_by_the_served_models_it_restricts() {
        let parsed = snapshot(serde_json::json!({"models": [
            {"id": "vendor/alpha", "organization_ids": ["org-a", "org-b"]},
            {"id": "vendor/beta", "organization_ids": []},
            {"id": "vendor/elsewhere", "organization_ids": ["org-c"]}
        ]}));
        let served = |ids: &[&str]| ids.iter().map(|id| id.to_string()).collect::<Vec<_>>();
        // In the order the process serves them, with counts and no
        // organization id.
        assert_eq!(
            parsed
                .restricted_among(&served(&["vendor/beta", "vendor/gamma", "vendor/alpha"]))
                .as_deref(),
            Some("vendor/beta (organizations: 0), vendor/alpha (organizations: 2)")
        );
        // None of the served models is in the document: a name that does
        // not match shows here.
        assert_eq!(
            parsed.restricted_among(&served(&["vendor/Alpha", "alpha"])),
            None
        );
        assert_eq!(parsed.restricted_among(&[]), None);
    }

    #[test]
    fn a_malformed_document_is_refused_as_a_whole() {
        for body in [
            "",
            "not json",
            "[]",
            "{}",
            r#"{"object": "model_access"}"#,
            r#"{"models": null}"#,
            r#"{"models": {"vendor/alpha": ["org-a"]}}"#,
            r#"{"models": ["vendor/alpha"]}"#,
            r#"{"models": [{"organization_ids": ["org-a"]}]}"#,
            r#"{"models": [{"id": "vendor/alpha"}]}"#,
            r#"{"models": [{"id": "vendor/alpha", "organization_ids": null}]}"#,
            r#"{"models": [{"id": "vendor/alpha", "organization_ids": "org-a"}]}"#,
            r#"{"models": [{"id": "vendor/alpha", "organization_ids": [1]}]}"#,
            r#"{"models": [{"id": 1, "organization_ids": []}]}"#,
            // One bad entry spoils the document: the good one is not applied.
            r#"{"models": [{"id": "vendor/alpha", "organization_ids": []}, {"id": "vendor/beta"}]}"#,
            r#"{"error": {"message": "unauthorized"}}"#,
        ] {
            assert!(Snapshot::parse(body.as_bytes()).is_err(), "{body}");
        }
    }

    #[test]
    fn matching_is_exact() {
        let parsed = snapshot(serde_json::json!({"models": [
            {"id": "vendor/alpha", "organization_ids": ["org-a"]}
        ]}));
        // Another case or a longer name is another model, which the document
        // does not restrict.
        assert!(parsed.allows("vendor/Alpha", Some("org-z")));
        assert!(parsed.allows("vendor/alpha-2", Some("org-z")));
        // An organization is on the list as written or not at all.
        assert!(!parsed.allows("vendor/alpha", Some("ORG-A")));
        assert!(!parsed.allows("vendor/alpha", Some("org-a ")));
        assert!(!parsed.allows("vendor/alpha", Some("")));
    }

    #[test]
    fn the_decision() {
        let held = snapshot(serde_json::json!({"models": [
            {"id": "vendor/alpha", "organization_ids": ["org-a"]}
        ]}));
        let held = Some(&held);
        // A granted organization, and anyone on a model that is not restricted.
        assert!(permits(held, true, Some("org-a"), "vendor/alpha"));
        assert!(permits(held, true, Some("org-b"), "vendor/beta"));
        assert!(permits(held, true, None, "vendor/beta"));
        // Another organization, and a key without one.
        assert!(!permits(held, true, Some("org-b"), "vendor/alpha"));
        assert!(!permits(held, true, None, "vendor/alpha"));
        // The config token is not checked, with or without an organization.
        assert!(permits(held, false, None, "vendor/alpha"));
        assert!(permits(held, false, Some("org-b"), "vendor/alpha"));
        // No document read yet: nothing is restricted.
        assert!(permits(None, true, Some("org-b"), "vendor/alpha"));
        assert!(permits(None, true, None, "vendor/alpha"));
    }

    #[test]
    fn a_read_is_logged_when_its_outcome_changes() {
        let access = ModelAccess::default();
        assert!(access.note(READ));
        assert!(!access.note(READ));
        assert!(access.note("status"));
        assert!(!access.note("status"));
        assert!(access.note("transport"));
        assert!(access.note(READ));
    }

    #[test]
    fn off_by_default() {
        let access = ModelAccess::default();
        assert!(!access.is_enabled());
        assert!(!access.has_snapshot());
    }

    fn access_with(timing: Timing) -> ModelAccess {
        ModelAccess {
            source: Some(Source {
                url: format!("http://cloud-api.invalid{MODEL_ACCESS_PATH}"),
                authorization: bearer("usage-secret").unwrap(),
                client: reqwest::Client::new(),
                timing,
                served: vec!["vendor/alpha".to_string()],
            }),
            ..ModelAccess::default()
        }
    }

    #[test]
    fn without_a_document_the_next_read_does_not_wait_an_interval() {
        let timing = Timing::new(Duration::from_secs(300));
        assert_eq!(timing.refresh, Duration::from_secs(300));
        assert_eq!(timing.retry_without_document, Duration::from_secs(5));
        assert_eq!(timing.read_timeout, Duration::from_secs(5));
        assert_eq!(timing.without_document(), Duration::from_secs(5));
        // An interval shorter than the retry is not made longer by it.
        let timing = Timing::new(Duration::from_secs(2));
        assert_eq!(timing.without_document(), Duration::from_secs(2));
    }

    #[test]
    fn a_refusal_is_given_the_name_the_caller_sent_and_asks_for_it_only_then() {
        let access = access_with(Timing::new(Duration::from_secs(60)));
        *access.snapshot.write().unwrap() = Some(Arc::new(snapshot(serde_json::json!({
            "models": [{"id": "vendor/alpha", "organization_ids": ["org-a"]}]
        }))));
        let key_of = |org: &str| RequireAuth {
            cloud_api_key: Some("sk-key".to_string()),
            org_id: Some(org.to_string()),
            workspace_id: None,
            api_key_id: None,
            request_id: None,
            request_source: crate::auth::RequestSource {
                auth_path: crate::auth::AuthPath::CloudApiKey,
                ingress_route: crate::auth::IngressRouteKind::Other,
            },
        };
        // Served: the request's own `model` is not even looked at.
        let granted = access.check(&key_of("org-a"), "vendor/alpha", None, || {
            panic!("the name is only needed for a refusal")
        });
        assert!(granted.is_ok());
        // Refused: with what the caller sent, never the configured id.
        for sent in [Some("anything/else".to_string()), None] {
            let wanted = sent.clone();
            match access.check(&key_of("org-b"), "vendor/alpha", None, || sent) {
                Err(AppError::ModelNotFound { model }) => assert_eq!(model, wanted),
                _ => panic!("served to an organization without access"),
            }
        }
    }

    #[test]
    fn the_token_is_trimmed_and_has_to_be_a_header_value() {
        for token in ["usage-secret", " usage-secret\n", "\tusage-secret \r\n"] {
            let value = bearer(token).unwrap();
            assert_eq!(value.as_bytes(), b"Bearer usage-secret", "{token:?}");
            assert!(value.is_sensitive());
        }
        for token in ["usage\nsecret", "usage\u{0}secret", "usage\u{7f}"] {
            let error = bearer(token).unwrap_err().to_string();
            assert!(error.contains("CLOUD_API_USAGE_TOKEN"), "{error}");
            // The token itself is not in the message.
            assert!(
                !error.contains("usage\n") && !error.contains("secret"),
                "{error}"
            );
        }
    }

    #[test]
    fn the_model_a_json_body_names() {
        assert_eq!(
            model_in_json(br#"{"model": "vendor/alpha", "prompt": "x"}"#).as_deref(),
            Some("vendor/alpha")
        );
        for body in [
            &b""[..],
            b"not json",
            b"[]",
            b"{}",
            br#"{"model": 7}"#,
            br#"{"model": null}"#,
        ] {
            assert_eq!(model_in_json(body), None);
        }
    }

    /// An answer whose body arrives in `chunks`, without a declared length.
    fn streamed(chunks: Vec<Vec<u8>>) -> reqwest::Response {
        let chunks = chunks
            .into_iter()
            .map(|chunk| Ok::<_, std::io::Error>(bytes::Bytes::from(chunk)));
        let body = reqwest::Body::wrap_stream(futures_util::stream::iter(chunks));
        reqwest::Response::from(axum::http::Response::new(body))
    }

    #[tokio::test]
    async fn a_document_larger_than_the_cap_is_not_read() {
        let at_most = |bytes: usize| vec![b'x'; bytes];
        // Up to the cap, whole or in pieces.
        let read = read_capped(streamed(vec![at_most(MAX_DOCUMENT_BYTES)])).await;
        assert_eq!(read.ok().map(|body| body.len()), Some(MAX_DOCUMENT_BYTES));
        let half = MAX_DOCUMENT_BYTES / 2;
        let read = read_capped(streamed(vec![at_most(half), at_most(half)])).await;
        assert_eq!(read.ok().map(|body| body.len()), Some(MAX_DOCUMENT_BYTES));

        // One byte more, with and without a declared length: not the
        // document, whatever it holds.
        let declared = axum::http::Response::new(at_most(MAX_DOCUMENT_BYTES + 1));
        for larger in [
            reqwest::Response::from(declared),
            streamed(vec![at_most(half), at_most(half), at_most(1)]),
        ] {
            match read_capped(larger).await {
                Err(failure @ Failure::Malformed(_)) => {
                    assert_eq!(failure.reason(), "malformed");
                    assert!(failure.to_string().contains("larger than 1048576 bytes"));
                }
                _ => panic!("a body past the cap was read"),
            }
        }
    }
}
