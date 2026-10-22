//! AWS credentials for the SQS usage-report sender.
//!
//! Credentials come only from an instance-metadata-service (IMDSv2) endpoint:
//! on a gateway host that is `aws_signing_helper serve` (IAM Roles Anywhere),
//! reached on localhost, which answers only requests that carry a session
//! token. The configured URL is the endpoint's base (`http://127.0.0.1:9911`);
//! a fetch is
//! 1. `PUT /latest/api/token` with `X-aws-ec2-metadata-token-ttl-seconds`, the token;
//! 2. `GET /latest/meta-data/iam/security-credentials/`, the role name (first line);
//! 3. `GET /latest/meta-data/iam/security-credentials/<role>`, the credentials
//!    (`AccessKeyId`, `SecretAccessKey`, `Token`, `Expiration`),
//!
//! steps 2 and 3 sending the token in `X-aws-ec2-metadata-token`. The token is
//! kept until shortly before its time to live ends; a 401 drops it and the
//! fetch is made once more with a new one. There are no static keys: the
//! environment is never read for credentials.
//!
//! Credentials are cached until five minutes before they expire, and fetched
//! again after that. The cache is one lock, held across a fetch: concurrent
//! attempts that need credentials wait for it and take what it fetched. The
//! caller bounds the wait with its own timeout; a caller that gives up drops
//! the fetch, and the next one fetches again.
//!
//! Every failure to get credentials is worth another attempt: the endpoint may
//! pass, and the next attempt fetches again.
//!
//! Secrets never reach a log line or a `Debug` output: the types below either
//! do not derive `Debug` or hold only a description of the failure.

use std::time::{Duration, SystemTime};

use aws_credential_types::Credentials;
use chrono::{DateTime, Utc};
use serde::Deserialize;

/// A client that treats a 3xx as the answer it is. The metadata requests carry
/// a session token in a custom header, which reqwest's redirect handling does
/// not strip when it follows a redirect to another host; every client passed
/// to [`CredentialProvider::get`] must be this one or one that is as strict.
/// Timeouts stay per call (the caller's budget); bodies stay bounded.
pub(crate) fn no_redirect_client() -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .pool_idle_timeout(Duration::from_secs(30))
        .build()
}

/// Credentials from an endpoint are fetched again this long before they expire.
const REFRESH_BEFORE_EXPIRY: Duration = Duration::from_secs(300);
/// Endpoint credentials that carry no `Expiration` are fetched again after this long.
const WITHOUT_EXPIRATION_TTL: Duration = Duration::from_secs(15 * 60);
/// The most of an endpoint's answer that is read: credentials are a few hundred bytes.
const MAX_ANSWER_BYTES: usize = 64 * 1024;
/// The session token's time to live that is asked for (the helper's maximum).
const TOKEN_TTL_SECS: u64 = 21_600;
/// A token is replaced this long before its time to live ends.
const TOKEN_REFRESH_BEFORE_EXPIRY: Duration = Duration::from_secs(300);
const TOKEN_PATH: &str = "/latest/api/token";
const ROLES_PATH: &str = "/latest/meta-data/iam/security-credentials/";
const TOKEN_HEADER: &str = "x-aws-ec2-metadata-token";
const TOKEN_TTL_HEADER: &str = "x-aws-ec2-metadata-token-ttl-seconds";

/// Why credentials could not be had for one attempt. Retryable by nature.
#[derive(Debug, Clone)]
pub(crate) struct CredentialFailure(pub(crate) String);

/// Hands out credentials for signing, caching the endpoint's answer.
pub(crate) struct CredentialProvider {
    /// The metadata endpoint's base URL (`VLLM_PROXY_AWS_CREDENTIALS_URL`),
    /// without a trailing slash.
    base: String,
    /// The cached credentials and session token. Held across a fetch, so one
    /// fetch serves the attempts that wait for it.
    cached: tokio::sync::Mutex<State>,
}

#[derive(Default)]
struct State {
    credentials: Option<Cached>,
    token: Option<SessionToken>,
}

/// A metadata session token. Not `Debug`: it is a secret.
struct SessionToken {
    value: String,
    valid_until: std::time::Instant,
}

struct Cached {
    credentials: Credentials,
    fetched_at: std::time::Instant,
}

/// The endpoint's answer, as the instance metadata service words it.
#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct EndpointCredentials {
    access_key_id: String,
    secret_access_key: String,
    #[serde(default)]
    token: Option<String>,
    #[serde(default)]
    expiration: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

impl CredentialProvider {
    /// The provider for the metadata endpoint at `url` (its base URL).
    pub(crate) fn new(url: String) -> Self {
        Self {
            base: url.trim_end_matches('/').to_string(),
            cached: tokio::sync::Mutex::new(State::default()),
        }
    }

    /// Credentials to sign with, fetched from the endpoint when none is cached
    /// or the cached one is due. The lock is held across the fetch on purpose:
    /// concurrent callers wait for the one fetch and share its result, so the
    /// endpoint sees one flow, not one per caller. Callers bound this with
    /// their own timeout.
    pub(crate) async fn get(
        &self,
        client: &reqwest::Client,
    ) -> Result<Credentials, CredentialFailure> {
        let mut state = self.cached.lock().await;
        if let Some(fresh) = state.credentials.as_ref().filter(|cached| is_fresh(cached)) {
            return Ok(fresh.credentials.clone());
        }
        let credentials = fetch(client, &self.base, &mut state.token).await?;
        state.credentials = Some(Cached {
            credentials: credentials.clone(),
            fetched_at: std::time::Instant::now(),
        });
        Ok(credentials)
    }

    /// Drop the cached credentials the service refused, if they are still the
    /// ones it saw. Never waits: when a fetch holds the cache, that fetch is
    /// replacing the entry anyway, so skipping is safe.
    pub(crate) fn forget(&self, refused: &Credentials) {
        if let Ok(mut state) = self.cached.try_lock() {
            if state
                .credentials
                .as_ref()
                .is_some_and(|cached| cached.credentials.access_key_id() == refused.access_key_id())
            {
                state.credentials = None;
            }
        }
    }

    /// Seeds the cache with `credentials`, for tests of the signing path that
    /// must not reach an endpoint.
    #[cfg(test)]
    pub(crate) fn seeded(credentials: Credentials) -> Self {
        let provider = Self::new("http://127.0.0.1:1".to_string());
        provider.cached.try_lock().unwrap().credentials = Some(Cached {
            credentials,
            fetched_at: std::time::Instant::now(),
        });
        provider
    }
}

fn is_fresh(cached: &Cached) -> bool {
    match cached.credentials.expiry() {
        // The endpoint gives an absolute time, so the wall clock is the only
        // clock that can be compared with it; the margin absorbs small jumps.
        Some(expiry) => SystemTime::now() + REFRESH_BEFORE_EXPIRY < expiry,
        // No expiration given: a short fixed life, not forever.
        None => cached.fetched_at.elapsed() < WITHOUT_EXPIRATION_TTL,
    }
}

/// One fetch of credentials: the IMDSv2 flow against `base`. A 401 from the
/// endpoint drops the session token, and the flow is run once more with a new
/// one (the helper restarted, or the token expired early).
async fn fetch(
    client: &reqwest::Client,
    base: &str,
    token: &mut Option<SessionToken>,
) -> Result<Credentials, CredentialFailure> {
    match fetch_with_token(client, base, token).await {
        Err(Failure::Unauthorized) => {
            *token = None;
            fetch_with_token(client, base, token)
                .await
                .map_err(Failure::into_credential_failure)
        }
        other => other.map_err(Failure::into_credential_failure),
    }
}

enum Failure {
    /// The endpoint answered 401: the session token is not (or no longer) good.
    Unauthorized,
    Other(CredentialFailure),
}

impl Failure {
    fn into_credential_failure(self) -> CredentialFailure {
        match self {
            Failure::Unauthorized => {
                CredentialFailure("credentials endpoint refused the session token (401)".into())
            }
            Failure::Other(failure) => failure,
        }
    }
}

fn other(message: impl Into<String>) -> Failure {
    Failure::Other(CredentialFailure(message.into()))
}

async fn fetch_with_token(
    client: &reqwest::Client,
    base: &str,
    token: &mut Option<SessionToken>,
) -> Result<Credentials, Failure> {
    let value = match token
        .as_ref()
        .filter(|t| std::time::Instant::now() < t.valid_until)
    {
        Some(token) => token.value.clone(),
        None => {
            let fresh = fetch_token(client, base).await?;
            let value = fresh.value.clone();
            *token = Some(fresh);
            value
        }
    };
    let roles = get_text(client, &format!("{base}{ROLES_PATH}"), &value).await?;
    let role = roles.lines().next().map(str::trim).unwrap_or_default();
    if role.is_empty() || role.contains(['/', '?', '#', '%']) {
        return Err(other("credentials endpoint listed no usable role name"));
    }
    let bytes = get_bytes(client, &format!("{base}{ROLES_PATH}{role}"), &value).await?;
    // The body holds secrets: it is parsed, never echoed into an error.
    let body: EndpointCredentials = serde_json::from_slice(&bytes)
        .map_err(|_| other("credentials endpoint answered something other than credentials"))?;
    // `Code` is a status word, not a secret: it says why there are no
    // credentials (`AssumeRoleUnauthorizedAccess`, `ExpiredToken`, ...).
    if let Some(code) = body.code.as_deref().filter(|code| *code != "Success") {
        return Err(other(format!(
            "credentials endpoint answered Code {:?}",
            code.chars().take(64).collect::<String>()
        )));
    }
    let expiration = body
        .expiration
        .map(|text| parse_expiration(&text).map_err(other))
        .transpose()?;
    Ok(Credentials::new(
        body.access_key_id,
        body.secret_access_key,
        body.token,
        expiration,
        "credentials-endpoint",
    ))
}

/// Step 1: a session token. The token is a secret and is never echoed.
async fn fetch_token(client: &reqwest::Client, base: &str) -> Result<SessionToken, Failure> {
    let response = client
        .put(format!("{base}{TOKEN_PATH}"))
        .header(TOKEN_TTL_HEADER, TOKEN_TTL_SECS)
        .send()
        .await
        .map_err(|error| other(error.to_string()))?;
    check_status(&response)?;
    let bytes = read_bounded(response, MAX_ANSWER_BYTES)
        .await
        .map_err(read_failure)?;
    let value = String::from_utf8(bytes)
        .map(|text| text.trim().to_string())
        .ok()
        .filter(|text| !text.is_empty() && text.bytes().all(|b| b.is_ascii_graphic()))
        .ok_or_else(|| other("credentials endpoint answered no usable session token"))?;
    Ok(SessionToken {
        value,
        valid_until: std::time::Instant::now()
            + Duration::from_secs(TOKEN_TTL_SECS).saturating_sub(TOKEN_REFRESH_BEFORE_EXPIRY),
    })
}

async fn get_bytes(client: &reqwest::Client, url: &str, token: &str) -> Result<Vec<u8>, Failure> {
    let response = client
        .get(url)
        .header(TOKEN_HEADER, token)
        .send()
        .await
        .map_err(|error| other(error.to_string()))?;
    check_status(&response)?;
    read_bounded(response, MAX_ANSWER_BYTES)
        .await
        .map_err(read_failure)
}

async fn get_text(client: &reqwest::Client, url: &str, token: &str) -> Result<String, Failure> {
    let bytes = get_bytes(client, url, token).await?;
    String::from_utf8(bytes)
        .map_err(|_| other("credentials endpoint answered text that is not UTF-8"))
}

fn check_status(response: &reqwest::Response) -> Result<(), Failure> {
    let status = response.status();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        Err(Failure::Unauthorized)
    } else if !status.is_success() {
        Err(other(format!("credentials endpoint answered {status}")))
    } else {
        Ok(())
    }
}

fn read_failure(reason: String) -> Failure {
    other(format!("credentials endpoint answer {reason}"))
}

/// The body of `response`, read for at most `limit` bytes. The error says
/// whether the body is longer than that or could not be read.
pub(crate) async fn read_bounded(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| format!("could not be read: {error}"))?
    {
        if body.len() + chunk.len() > limit {
            return Err("is larger than it may be".to_string());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn parse_expiration(text: &str) -> Result<SystemTime, String> {
    DateTime::parse_from_rfc3339(text)
        .map(|time| SystemTime::from(time.with_timezone(&Utc)))
        .map_err(|_| "credentials endpoint gave an Expiration that is not RFC 3339".to_string())
}

/// Mocks of an IMDSv2 endpoint, shared with the sender's tests.
#[cfg(test)]
pub(crate) mod test_support {
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    pub(crate) const TOKEN: &str = "test-session-token";
    pub(crate) const ROLE: &str = "test-role";
    /// Where the credentials of `ROLE` are served.
    pub(crate) const ROLE_CREDENTIALS_PATH: &str =
        "/latest/meta-data/iam/security-credentials/test-role";

    /// Mounts the token and role-listing steps; the caller mounts the
    /// credentials at `ROLE_CREDENTIALS_PATH`. Every other request is a 401,
    /// as the helper answers a request without a valid token.
    pub(crate) async fn mount_imds_session(server: &MockServer) {
        Mock::given(method("PUT"))
            .and(path("/latest/api/token"))
            .and(header("x-aws-ec2-metadata-token-ttl-seconds", "21600"))
            .respond_with(ResponseTemplate::new(200).set_body_string(TOKEN))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/latest/meta-data/iam/security-credentials/"))
            .and(header("x-aws-ec2-metadata-token", TOKEN))
            .respond_with(ResponseTemplate::new(200).set_body_string(format!("{ROLE}\n")))
            .mount(server)
            .await;
        reject_tokenless(server).await;
    }

    /// Anything not matched by a mock with the token is a 401 ("no token provided").
    pub(crate) async fn reject_tokenless(server: &MockServer) {
        Mock::given(wiremock::matchers::any())
            .respond_with(ResponseTemplate::new(401).set_body_string("no token provided"))
            .with_priority(200)
            .mount(server)
            .await;
    }

    /// The credentials mock: `ROLE_CREDENTIALS_PATH` with the token.
    pub(crate) fn credentials_mock() -> wiremock::MockBuilder {
        Mock::given(method("GET"))
            .and(path(ROLE_CREDENTIALS_PATH))
            .and(header("x-aws-ec2-metadata-token", TOKEN))
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use std::time::Instant;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    fn good_credentials() -> serde_json::Value {
        serde_json::json!({
            "AccessKeyId": "ASIAEXAMPLE",
            "SecretAccessKey": "secret",
            "Token": "session-token",
            "Code": "Success",
            "Expiration": (Utc::now() + chrono::TimeDelta::hours(1)).to_rfc3339(),
        })
    }

    async fn requests(server: &MockServer, verb: &str) -> usize {
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .filter(|r: &&Request| r.method.as_str() == verb)
            .count()
    }

    #[test]
    fn expiration_must_be_rfc3339() {
        assert!(parse_expiration("2026-10-20T12:00:00Z").is_ok());
        assert!(parse_expiration("tomorrow").is_err());
    }

    /// The full IMDSv2 flow against a server that, like `aws_signing_helper
    /// serve`, answers 401 to anything without the token (a plain GET of the
    /// configured URL, the old behaviour, fails here).
    #[tokio::test]
    async fn the_imdsv2_flow_fetches_credentials_from_a_server_that_rejects_tokenless_requests() {
        let server = MockServer::start().await;
        mount_imds_session(&server).await;
        credentials_mock()
            .respond_with(ResponseTemplate::new(200).set_body_json(good_credentials()))
            .mount(&server)
            .await;
        // A trailing slash on the configured URL makes no difference.
        let provider = CredentialProvider::new(format!("{}/", server.uri()));
        let credentials = provider.get(&reqwest::Client::new()).await.unwrap();
        assert_eq!(credentials.access_key_id(), "ASIAEXAMPLE");
        assert_eq!(credentials.secret_access_key(), "secret");
        assert_eq!(credentials.session_token(), Some("session-token"));
        assert!(credentials.expiry().is_some());
        assert_eq!(requests(&server, "PUT").await, 1);
        assert_eq!(requests(&server, "GET").await, 2);
    }

    #[tokio::test]
    async fn a_plain_get_of_the_url_is_the_bug_and_is_refused_by_such_a_server() {
        let server = MockServer::start().await;
        mount_imds_session(&server).await;
        let plain = reqwest::Client::new()
            .get(server.uri())
            .send()
            .await
            .unwrap();
        assert_eq!(plain.status(), 401);
    }

    #[tokio::test]
    async fn the_first_line_of_the_role_listing_is_the_role() {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/latest/api/token"))
            .respond_with(ResponseTemplate::new(200).set_body_string(TOKEN))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/latest/meta-data/iam/security-credentials/"))
            .respond_with(ResponseTemplate::new(200).set_body_string("test-role\nother-role\n"))
            .mount(&server)
            .await;
        credentials_mock()
            .respond_with(ResponseTemplate::new(200).set_body_json(good_credentials()))
            .mount(&server)
            .await;
        let provider = CredentialProvider::new(server.uri());
        provider.get(&reqwest::Client::new()).await.unwrap();
    }

    #[tokio::test]
    async fn an_empty_or_unusable_role_listing_fails() {
        for listing in ["", "\n", "a/b", "a%2Fb"] {
            let server = MockServer::start().await;
            Mock::given(method("PUT"))
                .and(path("/latest/api/token"))
                .respond_with(ResponseTemplate::new(200).set_body_string(TOKEN))
                .mount(&server)
                .await;
            Mock::given(method("GET"))
                .and(path("/latest/meta-data/iam/security-credentials/"))
                .respond_with(ResponseTemplate::new(200).set_body_string(listing))
                .mount(&server)
                .await;
            let failure = CredentialProvider::new(server.uri())
                .get(&reqwest::Client::new())
                .await
                .unwrap_err();
            assert!(failure.0.contains("role"), "{listing:?}: {}", failure.0);
        }
    }

    #[tokio::test]
    async fn the_session_token_is_reused_until_near_its_time_to_live() {
        let server = MockServer::start().await;
        mount_imds_session(&server).await;
        credentials_mock()
            .respond_with(ResponseTemplate::new(200).set_body_json(good_credentials()))
            .mount(&server)
            .await;
        let provider = CredentialProvider::new(server.uri());
        let client = reqwest::Client::new();

        provider.get(&client).await.unwrap();
        // Credentials due again: the token is reused, not fetched.
        provider.cached.lock().await.credentials = None;
        provider.get(&client).await.unwrap();
        assert_eq!(requests(&server, "PUT").await, 1);

        // The token past its time to live is fetched again.
        provider.cached.lock().await.credentials = None;
        provider
            .cached
            .lock()
            .await
            .token
            .as_mut()
            .unwrap()
            .valid_until = Instant::now();
        provider.get(&client).await.unwrap();
        assert_eq!(requests(&server, "PUT").await, 2);
    }

    /// A token the server no longer knows (401): the token is fetched once
    /// more and the fetch then succeeds.
    #[tokio::test]
    async fn a_401_refetches_the_token_once_and_succeeds() {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/latest/api/token"))
            .respond_with(ResponseTemplate::new(200).set_body_string("stale-token"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        mount_imds_session(&server).await;
        credentials_mock()
            .respond_with(ResponseTemplate::new(200).set_body_json(good_credentials()))
            .mount(&server)
            .await;
        let provider = CredentialProvider::new(server.uri());
        provider.get(&reqwest::Client::new()).await.unwrap();
        assert_eq!(requests(&server, "PUT").await, 2);
        assert_eq!(
            provider.cached.lock().await.token.as_ref().unwrap().value,
            TOKEN
        );
    }

    /// A server that keeps answering 401: one refetch of the token, then a failure.
    #[tokio::test]
    async fn a_persistent_401_is_one_refetch_then_a_retryable_failure() {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/latest/api/token"))
            .respond_with(ResponseTemplate::new(200).set_body_string("useless-token"))
            .mount(&server)
            .await;
        reject_tokenless(&server).await;
        let provider = CredentialProvider::new(server.uri());
        let failure = provider.get(&reqwest::Client::new()).await.unwrap_err();
        assert!(failure.0.contains("401"), "{}", failure.0);
        assert_eq!(requests(&server, "PUT").await, 2);
    }

    #[tokio::test]
    async fn credentials_without_an_expiration_are_fetched_again_after_fifteen_minutes() {
        let server = MockServer::start().await;
        mount_imds_session(&server).await;
        credentials_mock()
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "AccessKeyId": "ASIAEXAMPLE",
                "SecretAccessKey": "secret",
            })))
            .mount(&server)
            .await;
        let provider = CredentialProvider::new(server.uri());
        let client = reqwest::Client::new();

        provider.get(&client).await.unwrap();
        provider.get(&client).await.unwrap();
        assert_eq!(requests(&server, "GET").await, 2);

        // Age the cached entry past its time to live: it is fetched again.
        provider
            .cached
            .lock()
            .await
            .credentials
            .as_mut()
            .unwrap()
            .fetched_at = Instant::now() - Duration::from_secs(16 * 60);
        provider.get(&client).await.unwrap();
        assert_eq!(requests(&server, "GET").await, 4);
    }

    #[tokio::test]
    async fn concurrent_refreshes_share_one_fetch() {
        let server = MockServer::start().await;
        mount_imds_session(&server).await;
        credentials_mock()
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_millis(200))
                    .set_body_json(good_credentials()),
            )
            .expect(1)
            .mount(&server)
            .await;
        let provider = CredentialProvider::new(server.uri());
        let client = reqwest::Client::new();

        let answers = futures_util::future::join_all((0..10).map(|_| {
            let (provider, client) = (&provider, &client);
            async move {
                provider
                    .get(client)
                    .await
                    .map(|c| c.access_key_id().to_string())
            }
        }))
        .await;
        for answer in answers {
            assert_eq!(answer.unwrap(), "ASIAEXAMPLE");
        }
        // One flow for ten concurrent attempts: `expect(1)` is verified on drop.
        assert_eq!(requests(&server, "PUT").await, 1);
    }

    /// Requirement: no attempt exceeds its deadline, however the endpoint
    /// behaves, and each attempt fetches at most once. Exactly one fetch is not
    /// required: a cancelled fetch lets the next attempt fetch.
    #[tokio::test]
    async fn a_hung_endpoint_is_bounded_by_the_attempts_and_fetches_once() {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/latest/api/token"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(10)))
            .mount(&server)
            .await;
        let provider = CredentialProvider::new(server.uri());
        let client = reqwest::Client::new();
        let deadline = Duration::from_millis(300);
        let started = Instant::now();
        // Ten attempts at once, each bounded the way `SqsSender::attempt` bounds it.
        let outcomes = futures_util::future::join_all((0..10).map(|_| {
            let (provider, client) = (&provider, &client);
            async move { tokio::time::timeout(deadline, provider.get(client)).await }
        }))
        .await;
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
        assert!(outcomes.iter().all(|outcome| outcome.is_err()));
        let fetches = requests(&server, "PUT").await;
        assert!(
            (1..=10).contains(&fetches),
            "{fetches} fetches for 10 attempts"
        );
    }

    #[tokio::test]
    async fn a_failing_endpoint_is_a_retryable_failure_and_is_asked_again() {
        let server = MockServer::start().await;
        mount_imds_session(&server).await;
        // First answer: a 404. Then a body that is not credentials. Then good.
        credentials_mock()
            .respond_with(ResponseTemplate::new(404))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        credentials_mock()
            .respond_with(ResponseTemplate::new(200).set_body_string("<html>not json</html>"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        credentials_mock()
            .respond_with(ResponseTemplate::new(200).set_body_json(good_credentials()))
            .mount(&server)
            .await;
        let provider = CredentialProvider::new(server.uri());
        let client = reqwest::Client::new();

        let first = provider.get(&client).await.unwrap_err();
        assert!(first.0.contains("404"), "{}", first.0);
        let second = provider.get(&client).await.unwrap_err();
        assert!(!second.0.contains("<html>"), "a body is never echoed");
        assert!(second.0.contains("other than credentials"), "{}", second.0);
        provider.get(&client).await.unwrap();
        assert_eq!(requests(&server, "PUT").await, 1, "the token is kept");
    }

    #[tokio::test]
    async fn malformed_or_unsuccessful_credentials_are_refused_without_echoing_them() {
        for body in [
            r#"{"AccessKeyId": "AKIDLEAK", "SecretAccessKey"#.to_string(),
            r#"{"AccessKeyId": "AKIDLEAK"}"#.to_string(),
            r#"{"Code":"AssumeRoleUnauthorizedAccess","AccessKeyId":"AKIDLEAK","SecretAccessKey":"s"}"#
                .to_string(),
            r#"{"AccessKeyId":"AKIDLEAK","SecretAccessKey":"s","Expiration":"tomorrow"}"#
                .to_string(),
        ] {
            let server = MockServer::start().await;
            mount_imds_session(&server).await;
            credentials_mock()
                .respond_with(ResponseTemplate::new(200).set_body_string(body.clone()))
                .mount(&server)
                .await;
            let failure = CredentialProvider::new(server.uri())
                .get(&reqwest::Client::new())
                .await
                .unwrap_err();
            assert!(!failure.0.contains("AKIDLEAK"), "{body}: {}", failure.0);
        }
    }

    /// The reason the endpoint gives (`Code`) is in the failure; the secrets
    /// in the same body are not.
    #[tokio::test]
    async fn an_unsuccessful_code_is_named_in_the_failure() {
        let server = MockServer::start().await;
        mount_imds_session(&server).await;
        credentials_mock()
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{"Code":"AssumeRoleUnauthorizedAccess","AccessKeyId":"AKIDLEAK","SecretAccessKey":"s"}"#,
            ))
            .mount(&server)
            .await;
        let failure = CredentialProvider::new(server.uri())
            .get(&reqwest::Client::new())
            .await
            .unwrap_err();
        assert!(
            failure.0.contains("AssumeRoleUnauthorizedAccess"),
            "{}",
            failure.0
        );
        assert!(!failure.0.contains("AKIDLEAK"), "{}", failure.0);
    }

    /// A redirect from the endpoint to another host must not carry the session
    /// token there: with `no_redirect_client` it is not followed at all. The
    /// plain client is the control: it does follow, and the token goes along.
    #[tokio::test]
    async fn a_redirect_from_the_endpoint_does_not_carry_the_token_to_another_host() {
        let elsewhere = MockServer::start().await;
        Mock::given(wiremock::matchers::any())
            .respond_with(ResponseTemplate::new(200).set_body_string("test-role"))
            .mount(&elsewhere)
            .await;
        let endpoint = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/latest/api/token"))
            .respond_with(ResponseTemplate::new(200).set_body_string(TOKEN))
            .mount(&endpoint)
            .await;
        Mock::given(method("GET"))
            .and(path("/latest/meta-data/iam/security-credentials/"))
            .respond_with(
                ResponseTemplate::new(307)
                    .insert_header("location", format!("{}/roles", elsewhere.uri())),
            )
            .mount(&endpoint)
            .await;

        let failure = CredentialProvider::new(endpoint.uri())
            .get(&no_redirect_client().unwrap())
            .await
            .unwrap_err();
        assert!(failure.0.contains("307"), "{}", failure.0);
        assert!(elsewhere.received_requests().await.unwrap().is_empty());

        let _ = CredentialProvider::new(endpoint.uri())
            .get(&reqwest::Client::new())
            .await;
        let followed = elsewhere.received_requests().await.unwrap();
        assert_eq!(followed.len(), 1, "the control client follows the redirect");
        assert!(followed[0].headers.contains_key("x-aws-ec2-metadata-token"));
    }

    #[tokio::test]
    async fn an_oversized_endpoint_answer_is_refused_not_read_whole() {
        // The credentials, the role listing and the token are each bounded.
        for step in ["credentials", "roles", "token"] {
            let server = MockServer::start().await;
            let big = ResponseTemplate::new(200).set_body_bytes(vec![b'a'; MAX_ANSWER_BYTES + 1]);
            mount_imds_session(&server).await;
            let provider = CredentialProvider::new(server.uri());
            match step {
                "credentials" => {
                    credentials_mock().respond_with(big).mount(&server).await;
                }
                "roles" => {
                    Mock::given(method("GET"))
                        .and(path("/latest/meta-data/iam/security-credentials/"))
                        .and(header("x-aws-ec2-metadata-token", TOKEN))
                        .respond_with(big)
                        .with_priority(1)
                        .mount(&server)
                        .await;
                }
                _ => {
                    Mock::given(method("PUT"))
                        .and(path("/latest/api/token"))
                        .respond_with(big)
                        .with_priority(1)
                        .mount(&server)
                        .await;
                }
            }
            let failure = provider.get(&reqwest::Client::new()).await.unwrap_err();
            assert!(failure.0.contains("larger"), "{step}: {}", failure.0);
        }
    }

    #[tokio::test]
    async fn forget_clears_only_the_credentials_that_were_refused() {
        let provider =
            CredentialProvider::seeded(Credentials::new("AKID-OLD", "s", None, None, "t"));
        provider.forget(&Credentials::new("AKID-OTHER", "s", None, None, "t"));
        assert!(
            provider.cached.lock().await.credentials.is_some(),
            "a different key is not cleared"
        );
        provider.forget(&Credentials::new("AKID-OLD", "s", None, None, "t"));
        assert!(provider.cached.lock().await.credentials.is_none());
    }

    #[tokio::test]
    async fn forget_never_waits_for_a_fetch_in_progress() {
        let provider =
            CredentialProvider::seeded(Credentials::new("AKID-OLD", "s", None, None, "t"));
        let _fetching = provider.cached.lock().await;
        // The lock is held: forget returns at once and leaves the entry to the fetch.
        provider.forget(&Credentials::new("AKID-OLD", "s", None, None, "t"));
        drop(_fetching);
        assert!(provider.cached.lock().await.credentials.is_some());
    }
}
