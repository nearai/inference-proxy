use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[path = "common/auth_header.rs"]
mod auth_header;
#[path = "common/body_to_bytes.rs"]
mod body_to_bytes;
mod common;
#[path = "common/valid_request_id.rs"]
mod valid_request_id;

use auth_header::*;
use body_to_bytes::*;
use common::*;
use valid_request_id::*;

const STREAMING_CHAT_SSE: &str = concat!(
    "data: {\"id\":\"chatcmpl-route-matrix\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-route-matrix\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
    "data: [DONE]\n\n",
);
const STREAMING_COMPLETIONS_SSE: &str = concat!(
    "data: {\"id\":\"cmpl-route-matrix\",\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"ok\",\"finish_reason\":null,\"logprobs\":null}]}\n\n",
    "data: {\"id\":\"cmpl-route-matrix\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"stop\",\"logprobs\":null}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
    "data: [DONE]\n\n",
);

#[derive(Clone, Copy)]
enum RouteBody {
    Json(&'static str),
    Multipart {
        boundary: &'static str,
        body: &'static str,
    },
}

#[derive(Clone, Copy)]
struct DataPlaneRouteCase {
    name: &'static str,
    public_route: &'static str,
    upstream_route: &'static str,
    body: RouteBody,
    upstream_content_type: &'static str,
    upstream_body: &'static str,
    streaming: bool,
}

impl DataPlaneRouteCase {
    fn content_type(&self) -> String {
        match self.body {
            RouteBody::Json(_) => "application/json".to_string(),
            RouteBody::Multipart { boundary, .. } => {
                format!("multipart/form-data; boundary={boundary}")
            }
        }
    }

    fn request_body(&self) -> Body {
        match self.body {
            RouteBody::Json(body) => Body::from(body),
            RouteBody::Multipart { body, .. } => Body::from(body),
        }
    }
}

fn direct_data_plane_cases() -> [DataPlaneRouteCase; 9] {
    [
        DataPlaneRouteCase {
            name: "chat completions streaming",
            public_route: "/v1/chat/completions",
            upstream_route: "/v1/chat/completions",
            body: RouteBody::Json(
                r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
            ),
            upstream_content_type: "text/event-stream",
            upstream_body: STREAMING_CHAT_SSE,
            streaming: true,
        },
        DataPlaneRouteCase {
            name: "text completions streaming",
            public_route: "/v1/completions",
            upstream_route: "/v1/completions",
            body: RouteBody::Json(r#"{"model":"test-model","prompt":"hi","stream":true}"#),
            upstream_content_type: "text/event-stream",
            upstream_body: STREAMING_COMPLETIONS_SSE,
            streaming: true,
        },
        DataPlaneRouteCase {
            name: "tokenize",
            public_route: "/v1/tokenize",
            upstream_route: "/tokenize",
            body: RouteBody::Json(r#"{"text":"hello"}"#),
            upstream_content_type: "application/json",
            upstream_body: r#"{"tokens":[1],"count":1}"#,
            streaming: false,
        },
        DataPlaneRouteCase {
            name: "embeddings",
            public_route: "/v1/embeddings",
            upstream_route: "/v1/embeddings",
            body: RouteBody::Json(r#"{"model":"test-model","input":"hello"}"#),
            upstream_content_type: "application/json",
            upstream_body: r#"{"id":"emb-route-matrix","data":[{"embedding":[0.1],"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}}"#,
            streaming: false,
        },
        DataPlaneRouteCase {
            name: "rerank",
            public_route: "/v1/rerank",
            upstream_route: "/v1/rerank",
            body: RouteBody::Json(r#"{"model":"test-model","query":"q","documents":["a"]}"#),
            upstream_content_type: "application/json",
            upstream_body: r#"{"id":"rerank-route-matrix","results":[{"index":0,"relevance_score":0.9}]}"#,
            streaming: false,
        },
        DataPlaneRouteCase {
            name: "score",
            public_route: "/v1/score",
            upstream_route: "/v1/score",
            body: RouteBody::Json(r#"{"model":"test-model","text_1":"a","text_2":"b"}"#),
            upstream_content_type: "application/json",
            upstream_body: r#"{"id":"score-route-matrix","results":[{"score":0.8}]}"#,
            streaming: false,
        },
        DataPlaneRouteCase {
            name: "image generations",
            public_route: "/v1/images/generations",
            upstream_route: "/v1/images/generations",
            body: RouteBody::Json(r#"{"model":"test-model","prompt":"a cat"}"#),
            upstream_content_type: "application/json",
            upstream_body: r#"{"id":"img-route-matrix","data":[{"url":"https://example.com/image.png"}]}"#,
            streaming: false,
        },
        DataPlaneRouteCase {
            name: "image edits",
            public_route: "/v1/images/edits",
            upstream_route: "/v1/images/edits",
            body: RouteBody::Multipart {
                boundary: "----RouteMatrixImageEdit",
                body: "------RouteMatrixImageEdit\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\na cat\r\n------RouteMatrixImageEdit\r\nContent-Disposition: form-data; name=\"image\"; filename=\"test.png\"\r\nContent-Type: image/png\r\n\r\nfakepngdata\r\n------RouteMatrixImageEdit--\r\n",
            },
            upstream_content_type: "application/json",
            upstream_body: r#"{"id":"img-edit-route-matrix","data":[{"url":"https://example.com/edited.png"}]}"#,
            streaming: false,
        },
        DataPlaneRouteCase {
            name: "audio transcriptions",
            public_route: "/v1/audio/transcriptions",
            upstream_route: "/v1/audio/transcriptions",
            body: RouteBody::Multipart {
                boundary: "----RouteMatrixAudio",
                body: "------RouteMatrixAudio\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n------RouteMatrixAudio\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\nfakeaudiodata\r\n------RouteMatrixAudio--\r\n",
            },
            upstream_content_type: "application/json",
            upstream_body: r#"{"id":"trans-route-matrix","text":"hello"}"#,
            streaming: false,
        },
    ]
}

#[tokio::test]
async fn direct_data_plane_routes_forward_selected_request_id_to_upstream() {
    for case in direct_data_plane_cases() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(case.upstream_route))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(case.upstream_body.as_bytes(), case.upstream_content_type),
            )
            .mount(&mock_server)
            .await;

        let app = build_test_app(&mock_server.uri(), TestAppOptions::default());
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(case.public_route)
                    .header(auth_header().0, auth_header().1)
                    .header("content-type", case.content_type())
                    .header("x-request-id", VALID_REQUEST_ID)
                    .body(case.request_body())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK, "{}", case.name);
        assert_eq!(
            response.headers().get("x-request-id").unwrap(),
            VALID_REQUEST_ID
        );
        if case.streaming {
            eprintln!(
                "manual-qa: {} response headers observed before reading body; x-request-id={}",
                case.name, VALID_REQUEST_ID
            );
        }
        let body = body_to_bytes(response).await;
        assert!(
            !body.is_empty(),
            "{} response body should be readable",
            case.name
        );

        let received = mock_server
            .received_requests()
            .await
            .expect("wiremock records upstream requests");
        assert_eq!(received.len(), 1, "{} upstream request count", case.name);
        assert_eq!(
            received[0].url.path(),
            case.upstream_route,
            "{} upstream path",
            case.name
        );
        assert_eq!(
            received[0]
                .headers
                .get("x-request-id")
                .and_then(|value| value.to_str().ok()),
            Some(VALID_REQUEST_ID),
            "{} upstream x-request-id",
            case.name
        );
        eprintln!(
            "manual-qa: {} {} -> {} forwarded x-request-id={}",
            case.name, case.public_route, case.upstream_route, VALID_REQUEST_ID
        );
    }
}
