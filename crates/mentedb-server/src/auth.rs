//! JWT authentication middleware and token management.

use std::sync::Arc;

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::ApiError;
use crate::state::AppState;

/// JWT claims embedded in every token.
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub agent_id: String,
    pub exp: usize,
}

/// Request body for the token generation endpoint.
#[derive(Deserialize)]
pub struct TokenRequest {
    pub agent_id: String,
    #[serde(default = "default_expiry_hours")]
    pub expiry_hours: u64,
}

fn default_expiry_hours() -> u64 {
    24
}

/// Create a signed JWT for the given agent.
pub fn create_token(secret: &str, agent_id: &str, expiry_hours: u64) -> String {
    let exp = jsonwebtoken::get_current_timestamp() as usize + (expiry_hours as usize * 3600);
    let claims = Claims {
        agent_id: agent_id.to_string(),
        exp,
    };
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .expect("JWT encoding should not fail with valid inputs")
}

/// Validate a JWT and return the embedded claims.
pub fn validate_token(secret: &str, token: &str) -> Result<Claims, ApiError> {
    decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )
    .map(|data| data.claims)
    .map_err(|e| ApiError::Unauthorized(format!("invalid token: {e}")))
}

/// Handler: POST /v1/auth/token — generate a new JWT.
pub async fn generate_token(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let secret = state.jwt_secret.as_deref().ok_or_else(|| {
        ApiError::BadRequest("auth is disabled (no jwt-secret configured)".into())
    })?;

    let token = create_token(secret, &req.agent_id, req.expiry_hours);
    Ok(Json(json!({ "token": token, "agent_id": req.agent_id })))
}

/// Middleware that enforces JWT auth when a secret is configured.
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let secret = match &state.jwt_secret {
        Some(s) => s.clone(),
        // No secret configured means development mode; skip auth.
        None => return next.run(request).await,
    };

    // Allow health and token endpoints without auth.
    let path = request.uri().path();
    if path == "/v1/health" || path == "/v1/auth/token" {
        return next.run(request).await;
    }

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    let token = match auth_header {
        Some(h) if h.starts_with("Bearer ") => &h[7..],
        _ => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "error": "missing or invalid Authorization header" })),
            )
                .into_response();
        }
    };

    match validate_token(&secret, token) {
        Ok(_claims) => next.run(request).await,
        Err(e) => e.into_response(),
    }
}
