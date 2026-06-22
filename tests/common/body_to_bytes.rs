use axum::body::Body;

pub(crate) async fn body_to_bytes(response: axum::http::Response<Body>) -> Vec<u8> {
    use http_body_util::BodyExt;
    response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec()
}
