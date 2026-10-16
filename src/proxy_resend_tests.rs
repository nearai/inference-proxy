//! A request the backend provably never read is sent once more
//! (`send_resending_undelivered`); anything else is not.
//!
//! The backend here is a raw HTTP/1.1 server driven by a script, because the
//! failures under test happen below HTTP: a keep-alive connection the server
//! closes while a request sits unread in its socket (the kernel answers with a
//! reset), and a connection closed after the whole request was read.

use super::*;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// What the backend does with the next request, on whatever connection it
/// arrives.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// Read it, answer 200 and keep the connection open.
    Answer,
    /// Wait until it arrives, then close the connection without reading it:
    /// the unread bytes make the kernel reset the connection. This is the
    /// engine's keep-alive timer closing a pooled connection as a request
    /// goes out on it.
    ResetUnread,
    /// Read all of it, then close the connection without answering.
    CloseAfterRead,
}

struct Backend {
    addr: SocketAddr,
    /// Connections accepted.
    connections: Arc<AtomicUsize>,
    /// Requests read in full, i.e. that the backend could have acted on.
    requests_read: Arc<AtomicUsize>,
    /// Steps not played yet.
    script: Arc<Mutex<VecDeque<Step>>>,
}

impl Backend {
    async fn start(script: &[Step]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = Backend {
            addr: listener.local_addr().unwrap(),
            connections: Arc::new(AtomicUsize::new(0)),
            requests_read: Arc::new(AtomicUsize::new(0)),
            script: Arc::new(Mutex::new(script.iter().copied().collect())),
        };
        let (connections, requests_read, script) = (
            backend.connections.clone(),
            backend.requests_read.clone(),
            backend.script.clone(),
        );
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                connections.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(serve(stream, requests_read.clone(), script.clone()));
            }
        });
        backend
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }

    fn steps_left(&self) -> usize {
        self.script.lock().unwrap().len()
    }
}

async fn serve(
    mut stream: TcpStream,
    requests_read: Arc<AtomicUsize>,
    script: Arc<Mutex<VecDeque<Step>>>,
) {
    loop {
        // Wait for the next request before taking a step: an idle pooled
        // connection must not use one up.
        if stream.readable().await.is_err() {
            return;
        }
        let mut probe = [0u8; 1];
        match stream.peek(&mut probe).await {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let step = script.lock().unwrap().pop_front().unwrap_or(Step::Answer);
        match step {
            Step::ResetUnread => {
                // Let the request land in the socket buffer, then drop it.
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                return;
            }
            Step::CloseAfterRead => {
                if read_request(&mut stream).await {
                    requests_read.fetch_add(1, Ordering::SeqCst);
                }
                return;
            }
            Step::Answer => {
                if !read_request(&mut stream).await {
                    return;
                }
                requests_read.fetch_add(1, Ordering::SeqCst);
                let body = br#"{"id":"chatcmpl-resend","object":"chat.completion","created":0,"model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
                    body.len()
                );
                if stream.write_all(head.as_bytes()).await.is_err()
                    || stream.write_all(body).await.is_err()
                {
                    return;
                }
            }
        }
    }
}

/// Read one request (headers, then a `content-length` body). False when the
/// connection ends first.
async fn read_request(stream: &mut TcpStream) -> bool {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let header_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => return false,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_ascii_lowercase();
    let length: usize = head
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .map_or(0, |value| value.trim().parse().unwrap());
    let mut have = buf.len() - header_end;
    while have < length {
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => return false,
            Ok(n) => have += n,
        }
    }
    true
}

async fn send(client: &reqwest::Client, url: &str) -> Result<reqwest::Response, AppError> {
    let mut url = url.to_string();
    let mut opts = super::tests::test_proxy_opts();
    send_upstream(
        client,
        &mut url,
        Bytes::from_static(br#"{"messages":[]}"#),
        &mut opts,
        "streaming",
    )
    .await
}

fn internal_message(error: AppError) -> String {
    match error {
        AppError::Internal(error) => error.to_string(),
        other => panic!("expected the generic 500, got {other:?}"),
    }
}

/// The production case: a pooled connection the engine closes just as the
/// next request goes out on it. The engine never read that request, so it is
/// sent again, on a new connection, and the caller gets the answer.
#[tokio::test]
async fn request_on_a_pooled_connection_the_backend_reset_is_sent_again() {
    let backend = Backend::start(&[Step::Answer, Step::ResetUnread, Step::Answer]).await;
    let client = reqwest::Client::new();
    let url = backend.url("/v1/chat/completions");

    let first = send(&client, &url).await.unwrap();
    assert!(first.status().is_success());
    first.bytes().await.unwrap(); // back to the pool
    assert_eq!(backend.connections.load(Ordering::SeqCst), 1);

    let second = send(&client, &url).await.unwrap();
    assert!(second.status().is_success());
    assert_eq!(
        backend.connections.load(Ordering::SeqCst),
        2,
        "the resend goes out on a new connection"
    );
    assert_eq!(
        backend.requests_read.load(Ordering::SeqCst),
        2,
        "the backend read each request once: the reset one never"
    );
    assert_eq!(backend.steps_left(), 0);
}

/// Same on a new connection: the reset proves the backend did not read it.
#[tokio::test]
async fn request_reset_on_a_new_connection_is_sent_again() {
    let backend = Backend::start(&[Step::ResetUnread, Step::Answer]).await;
    let response = send(
        &reqwest::Client::new(),
        &backend.url("/v1/chat/completions"),
    )
    .await
    .unwrap();
    assert!(response.status().is_success());
    assert_eq!(backend.connections.load(Ordering::SeqCst), 2);
    assert_eq!(backend.requests_read.load(Ordering::SeqCst), 1);
}

/// One resend, not more: a second reset is the caller's 500, and its log
/// line says what happened instead of reqwest's bare "error sending request".
#[tokio::test]
async fn second_reset_is_returned_with_its_cause() {
    let backend = Backend::start(&[Step::ResetUnread, Step::ResetUnread, Step::Answer]).await;
    let error = send(
        &reqwest::Client::new(),
        &backend.url("/v1/chat/completions"),
    )
    .await
    .unwrap_err();
    let message = internal_message(error);
    assert!(
        message.starts_with("error sending request for url"),
        "{message}"
    );
    assert!(
        message.contains("reset by peer") || message.contains("Broken pipe"),
        "the chain names the cause: {message}"
    );
    assert_eq!(backend.connections.load(Ordering::SeqCst), 2);
    assert_eq!(backend.requests_read.load(Ordering::SeqCst), 0);
    assert_eq!(backend.steps_left(), 1, "no third attempt");
}

/// A connection closed after the backend read the whole request is not
/// proof that it did nothing with it: never sent twice.
#[tokio::test]
async fn request_the_backend_read_is_not_sent_again() {
    let backend = Backend::start(&[Step::CloseAfterRead, Step::Answer]).await;
    let error = send(
        &reqwest::Client::new(),
        &backend.url("/v1/chat/completions"),
    )
    .await
    .unwrap_err();
    let message = internal_message(error);
    assert!(
        message.contains("connection closed before message completed"),
        "{message}"
    );
    assert_eq!(backend.connections.load(Ordering::SeqCst), 1);
    assert_eq!(backend.requests_read.load(Ordering::SeqCst), 1);
    assert_eq!(backend.steps_left(), 1, "no second attempt");
}

/// Which failures count as "never read".
#[tokio::test]
async fn only_a_reset_counts_as_undelivered() {
    let client = reqwest::Client::new();

    let backend = Backend::start(&[Step::ResetUnread]).await;
    let reset = client
        .post(backend.url("/"))
        .body("{}")
        .send()
        .await
        .unwrap_err();
    assert!(undelivered_request(&reset).is_some(), "{reset:?}");

    let backend = Backend::start(&[Step::CloseAfterRead]).await;
    let closed = client
        .post(backend.url("/"))
        .body("{}")
        .send()
        .await
        .unwrap_err();
    assert_eq!(undelivered_request(&closed), None, "{closed:?}");

    // Nothing listens: a connect error, handled (and failed over) elsewhere.
    let refused = {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        client
            .post(format!("http://{addr}/"))
            .send()
            .await
            .unwrap_err()
    };
    assert!(refused.is_connect());
    assert_eq!(undelivered_request(&refused), None);

    // No answer in time: the backend may well be working on it.
    let silent = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = silent.local_addr().unwrap();
    let _held = tokio::spawn(async move {
        let (stream, _) = silent.accept().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        drop(stream);
    });
    let timeout = client
        .post(format!("http://{addr}/"))
        .timeout(std::time::Duration::from_millis(200))
        .send()
        .await
        .unwrap_err();
    assert!(timeout.is_timeout());
    assert_eq!(undelivered_request(&timeout), None);
}

/// A first-token deadline that passed while the first attempt failed rules
/// the resend out, as it rules out a fail-over.
#[tokio::test]
async fn no_resend_once_the_first_token_deadline_has_passed() {
    let backend = Backend::start(&[Step::ResetUnread, Step::Answer]).await;
    let mut url = backend.url("/v1/chat/completions");
    let mut opts = super::tests::test_proxy_opts();
    opts.first_token_deadline = Some(FirstTokenDeadline {
        started_at: std::time::Instant::now(),
        budget: std::time::Duration::ZERO,
        estimated_tokens: 1,
        retry_after_secs: 2,
        model: None,
    });
    let error = send_upstream(
        &reqwest::Client::new(),
        &mut url,
        Bytes::from_static(b"{}"),
        &mut opts,
        "streaming",
    )
    .await
    .unwrap_err();
    assert!(matches!(error, AppError::Internal(_)), "{error:?}");
    assert_eq!(backend.connections.load(Ordering::SeqCst), 1);
    assert_eq!(backend.steps_left(), 1);
}

/// Passthrough routes (`/v1/metrics`, `/v1/models`, ...) share the rule.
#[tokio::test]
async fn passthrough_request_reset_on_a_pooled_connection_is_sent_again() {
    let backend = Backend::start(&[Step::Answer, Step::ResetUnread, Step::Answer]).await;
    let client = reqwest::Client::new();
    let url = backend.url("/metrics");
    for _ in 0..2 {
        let response = proxy_simple(
            &client,
            &url,
            reqwest::Method::GET,
            None,
            "text/plain",
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    assert_eq!(backend.connections.load(Ordering::SeqCst), 2);
    assert_eq!(backend.requests_read.load(Ordering::SeqCst), 2);
}
