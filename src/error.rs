use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use tracing::error;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("upstream error")]
    Upstream { status: StatusCode, body: Bytes },

    #[error("upstream request failed with status {status}")]
    UpstreamParsed {
        status: StatusCode,
        message: String,
        error_type: String,
    },

    #[error("{0}")]
    BadRequest(String),

    #[error("unauthorized")]
    Unauthorized,

    #[error("insufficient credits")]
    InsufficientCredits,

    #[error("{0}")]
    NotFound(String),

    #[error("payload too large (max {max_size} bytes)")]
    PayloadTooLarge { max_size: usize },

    #[error("rate limit exceeded")]
    RateLimited,

    #[error("{0}")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message, error_type) = match &self {
            AppError::Upstream { status, body } => {
                // Forward upstream errors directly
                metrics::counter!("http_errors_total", "error_type" => "upstream").increment(1);
                return Response::builder()
                    .status(*status)
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body.clone()))
                    .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
            }
            AppError::UpstreamParsed {
                status,
                message,
                error_type,
            } => {
                metrics::counter!("http_errors_total", "error_type" => "upstream").increment(1);
                let body = serde_json::json!({
                    "error": {
                        "message": message,
                        "type": error_type,
                        "param": null,
                        "code": null,
                    }
                });
                return (*status, axum::Json(body)).into_response();
            }
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone(), "bad_request"),
            AppError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "Invalid or missing Authorization header".to_string(),
                "unauthorized",
            ),
            AppError::InsufficientCredits => (
                StatusCode::PAYMENT_REQUIRED,
                "Insufficient credits".to_string(),
                "insufficient_credits",
            ),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone(), "not_found"),
            AppError::PayloadTooLarge { max_size } => (
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("Request body too large. Maximum: {max_size} bytes"),
                "payload_too_large",
            ),
            AppError::RateLimited => (
                StatusCode::TOO_MANY_REQUESTS,
                "Rate limit exceeded. Please try again later.".to_string(),
                "rate_limited",
            ),
            AppError::Internal(ref e) => {
                error!(error = %e, "Internal server error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal server error".to_string(),
                    "server_error",
                )
            }
        };

        metrics::counter!("http_errors_total", "error_type" => error_type).increment(1);

        let body = serde_json::json!({
            "error": {
                "message": message,
                "type": error_type,
                "param": null,
                "code": null,
            }
        });

        (status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;

    async fn response_to_json(response: Response) -> (StatusCode, serde_json::Value) {
        let status = response.status();
        let body = response.into_body();
        let bytes = body.collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        (status, json)
    }

    #[tokio::test]
    async fn test_bad_request_error() {
        let err = AppError::BadRequest("invalid field".to_string());
        let response = err.into_response();
        let (status, json) = response_to_json(response).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["message"], "invalid field");
        assert_eq!(json["error"]["type"], "bad_request");
        assert!(json["error"]["param"].is_null());
        assert!(json["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn test_unauthorized_error() {
        let err = AppError::Unauthorized;
        let response = err.into_response();
        let (status, json) = response_to_json(response).await;

        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(
            json["error"]["message"],
            "Invalid or missing Authorization header"
        );
        assert_eq!(json["error"]["type"], "unauthorized");
    }

    #[tokio::test]
    async fn test_not_found_error() {
        let err = AppError::NotFound("Chat id not found".to_string());
        let response = err.into_response();
        let (status, json) = response_to_json(response).await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(json["error"]["message"], "Chat id not found");
        assert_eq!(json["error"]["type"], "not_found");
    }

    #[tokio::test]
    async fn test_payload_too_large_error() {
        let err = AppError::PayloadTooLarge {
            max_size: 10 * 1024 * 1024,
        };
        let response = err.into_response();
        let (status, json) = response_to_json(response).await;

        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            json["error"]["message"],
            "Request body too large. Maximum: 10485760 bytes"
        );
        assert_eq!(json["error"]["type"], "payload_too_large");
    }

    #[tokio::test]
    async fn test_rate_limited_error() {
        let err = AppError::RateLimited;
        let response = err.into_response();
        let (status, json) = response_to_json(response).await;

        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            json["error"]["message"],
            "Rate limit exceeded. Please try again later."
        );
        assert_eq!(json["error"]["type"], "rate_limited");
    }

    #[tokio::test]
    async fn test_internal_error_hides_details() {
        let err = AppError::Internal(anyhow::anyhow!("secret database password exposed"));
        let response = err.into_response();
        let (status, json) = response_to_json(response).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        // Should NOT leak the internal error message
        assert_eq!(json["error"]["message"], "Internal server error");
        assert_eq!(json["error"]["type"], "server_error");
    }

    #[tokio::test]
    async fn test_upstream_parsed_error() {
        let err = AppError::UpstreamParsed {
            status: StatusCode::BAD_REQUEST,
            message: "This model's maximum context length is 2048 tokens".to_string(),
            error_type: "BadRequestError".to_string(),
        };
        let response = err.into_response();
        let (status, json) = response_to_json(response).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["type"], "BadRequestError");
        assert_eq!(
            json["error"]["message"],
            "This model's maximum context length is 2048 tokens"
        );
        assert!(json["error"]["param"].is_null());
        assert!(json["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn test_upstream_parsed_preserves_status_codes() {
        for (code, expected_status) in [
            (400, StatusCode::BAD_REQUEST),
            (404, StatusCode::NOT_FOUND),
            (422, StatusCode::UNPROCESSABLE_ENTITY),
            (500, StatusCode::INTERNAL_SERVER_ERROR),
            (501, StatusCode::NOT_IMPLEMENTED),
        ] {
            let err = AppError::UpstreamParsed {
                status: StatusCode::from_u16(code).unwrap(),
                message: "test".to_string(),
                error_type: "TestError".to_string(),
            };
            let response = err.into_response();
            assert_eq!(response.status(), expected_status, "status code {code}");
        }
    }

    #[tokio::test]
    async fn test_upstream_error_passthrough() {
        let upstream_body = serde_json::json!({
            "error": {"message": "model not found", "type": "not_found"}
        });
        let body_bytes = serde_json::to_vec(&upstream_body).unwrap();

        let err = AppError::Upstream {
            status: StatusCode::NOT_FOUND,
            body: Bytes::from(body_bytes),
        };
        let response = err.into_response();
        let (status, json) = response_to_json(response).await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(json["error"]["message"], "model not found");
    }

    #[tokio::test]
    async fn test_error_response_is_openai_compatible() {
        // All non-upstream errors should produce the standard OpenAI error shape
        let err = AppError::BadRequest("test".to_string());
        let response = err.into_response();
        let (_, json) = response_to_json(response).await;

        // Must have the "error" top-level key with sub-fields
        assert!(json.get("error").is_some());
        let error_obj = &json["error"];
        assert!(error_obj.get("message").is_some());
        assert!(error_obj.get("type").is_some());
        assert!(error_obj.get("param").is_some());
        assert!(error_obj.get("code").is_some());
    }
}
