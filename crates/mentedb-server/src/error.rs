//! API error types with automatic HTTP status code mapping.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Unified API error type.
#[derive(Debug)]
#[allow(dead_code)]
pub enum ApiError {
    BadRequest(String),
    NotFound(String),
    Conflict(String),
    Unauthorized(String),
    Forbidden(String),
    RateLimited,
    Internal(String),
    ServiceUnavailable(String),
    /// The request's deadline passed before the work finished.
    GatewayTimeout(String),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadRequest(msg) => write!(f, "bad request: {msg}"),
            Self::NotFound(msg) => write!(f, "not found: {msg}"),
            Self::Conflict(msg) => write!(f, "conflict: {msg}"),
            Self::Unauthorized(msg) => write!(f, "unauthorized: {msg}"),
            Self::Forbidden(msg) => write!(f, "forbidden: {msg}"),
            Self::RateLimited => write!(f, "rate limited"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
            Self::ServiceUnavailable(msg) => write!(f, "service unavailable: {msg}"),
            Self::GatewayTimeout(msg) => write!(f, "deadline exceeded: {msg}"),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            Self::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            Self::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            Self::Conflict(msg) => (StatusCode::CONFLICT, msg.clone()),
            Self::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg.clone()),
            Self::Forbidden(msg) => (StatusCode::FORBIDDEN, msg.clone()),
            Self::GatewayTimeout(msg) => (StatusCode::GATEWAY_TIMEOUT, msg.clone()),
            Self::RateLimited => (
                StatusCode::TOO_MANY_REQUESTS,
                "too many requests".to_string(),
            ),
            Self::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            Self::ServiceUnavailable(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg.clone()),
        };

        let body = json!({ "error": message });
        (status, axum::Json(body)).into_response()
    }
}
