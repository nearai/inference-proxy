use axum::body::Body;

pub(crate) fn streaming_chat_body() -> Body {
    Body::from(
        r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
    )
}
