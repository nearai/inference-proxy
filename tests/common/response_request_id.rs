pub(crate) fn request_id_from_response(response: &axum::response::Response) -> String {
    response
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .expect("response must include x-request-id")
        .to_string()
}

pub(crate) fn assert_uuid(value: &str) {
    uuid::Uuid::parse_str(value).expect("x-request-id must be a UUID");
}
