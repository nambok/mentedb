//! JWT authentication middleware and token management.

use std::sync::Arc;

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http_body_util::BodyExt;
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::ApiError;
use crate::state::AppState;

#[derive(Debug, Clone)]
pub struct AuthenticatedAgent {
    /// Agent the token was issued for; empty when the token is scoped to
    /// databases rather than to one agent.
    pub agent_id: String,
    pub admin: bool,
    /// Databases this token may use. `None` means every database (tokens
    /// issued before database scopes existed, and admin tokens).
    pub dbs: Option<Vec<String>>,
    /// Read-only tokens may search and read but not write.
    pub read_only: bool,
}

impl AuthenticatedAgent {
    /// Allow or refuse access to database `name`.
    pub fn authorize_db(&self, name: &str, write: bool) -> Result<(), ApiError> {
        if write && self.read_only {
            return Err(ApiError::Forbidden("token is read-only".into()));
        }
        if self.admin {
            return Ok(());
        }
        match &self.dbs {
            None => Ok(()),
            Some(list) if list.iter().any(|d| d == "*" || d == name) => Ok(()),
            Some(_) => Err(ApiError::Forbidden(format!(
                "token is not scoped to database '{name}'"
            ))),
        }
    }

    /// The agent the token is bound to, if any.
    pub fn agent(&self) -> Result<Option<mentedb_core::types::AgentId>, ApiError> {
        if self.agent_id.is_empty() {
            return Ok(None);
        }
        self.agent_id
            .parse()
            .map(Some)
            .map_err(|_| ApiError::Unauthorized("token contains invalid agent_id UUID".into()))
    }
}

/// JWT claims embedded in every token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// The agent ID this token was issued for. Optional for tokens scoped to
    /// databases instead.
    #[serde(default)]
    pub agent_id: String,
    /// Whether this token grants admin privileges.
    #[serde(default)]
    pub admin: bool,
    /// Token expiration as a Unix timestamp.
    pub exp: usize,
    /// Databases the token may use (`"*"` for all). Absent means all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dbs: Option<Vec<String>>,
    /// Read-only access.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub read_only: bool,
}

impl Claims {
    fn into_agent(self) -> AuthenticatedAgent {
        AuthenticatedAgent {
            agent_id: self.agent_id,
            admin: self.admin,
            dbs: self.dbs,
            read_only: self.read_only,
        }
    }
}

/// How bearer tokens are checked.
///
/// A shared secret (HS256) lets the server issue tokens itself. A public key
/// (RS256, ES256 or EdDSA) lets another service sign tokens, for example one
/// per database, while the server holds no secret at all.
pub struct TokenVerifier {
    key: DecodingKey,
    validation: Validation,
}

impl TokenVerifier {
    /// Verify HS256 tokens signed with `secret`.
    pub fn shared_secret(secret: &str) -> Self {
        Self {
            key: DecodingKey::from_secret(secret.as_bytes()),
            validation: Validation::default(),
        }
    }

    /// Verify tokens signed by the private half of a PEM public key. The
    /// algorithm follows the key: RSA keys use RS256, EC P-256 keys ES256,
    /// Ed25519 keys EdDSA.
    pub fn public_key_pem(pem: &[u8]) -> Result<Self, String> {
        let text = String::from_utf8_lossy(pem);
        let (key, alg) = if let Ok(k) = DecodingKey::from_rsa_pem(pem) {
            (k, Algorithm::RS256)
        } else if let Ok(k) = DecodingKey::from_ec_pem(pem) {
            (k, Algorithm::ES256)
        } else if let Ok(k) = DecodingKey::from_ed_pem(pem) {
            (k, Algorithm::EdDSA)
        } else {
            return Err(format!(
                "not an RSA, EC or Ed25519 public key in PEM form ({} bytes)",
                text.len()
            ));
        };
        Ok(Self {
            key,
            validation: Validation::new(alg),
        })
    }

    /// Also require these `iss` / `aud` values when set.
    pub fn with_issuer_audience(
        mut self,
        issuer: Option<String>,
        audience: Option<String>,
    ) -> Self {
        if let Some(iss) = issuer {
            self.validation.set_issuer(&[iss]);
        }
        match audience {
            Some(aud) => self.validation.set_audience(&[aud]),
            None => self.validation.validate_aud = false,
        }
        self
    }

    /// Validate a token and return its claims.
    pub fn verify(&self, token: &str) -> Result<Claims, ApiError> {
        decode::<Claims>(token, &self.key, &self.validation)
            .map(|data| data.claims)
            .map_err(|e| ApiError::Unauthorized(format!("invalid token: {e}")))
    }

    /// Validate a token into the caller it represents.
    pub fn authenticate(&self, token: &str) -> Result<AuthenticatedAgent, ApiError> {
        self.verify(token).map(Claims::into_agent)
    }
}

/// Request body for the token generation endpoint.
#[derive(Deserialize)]
pub struct TokenRequest {
    /// The agent ID to issue a token for (may be empty for a token scoped to
    /// databases).
    #[serde(default)]
    pub agent_id: String,
    /// How many hours until the token expires (default: 24).
    #[serde(default = "default_expiry_hours")]
    pub expiry_hours: u64,
    /// Databases the token may use. Absent means all.
    #[serde(default)]
    pub dbs: Option<Vec<String>>,
    /// Issue a read-only token.
    #[serde(default)]
    pub read_only: bool,
}

fn default_expiry_hours() -> u64 {
    24
}

/// Create a signed JWT for the given agent.
pub fn create_token(secret: &str, agent_id: &str, admin: bool, expiry_hours: u64) -> String {
    create_scoped_token(secret, agent_id, admin, expiry_hours, None, false)
}

/// Create a signed JWT limited to `dbs` (all when `None`).
pub fn create_scoped_token(
    secret: &str,
    agent_id: &str,
    admin: bool,
    expiry_hours: u64,
    dbs: Option<Vec<String>>,
    read_only: bool,
) -> String {
    let exp = jsonwebtoken::get_current_timestamp() as usize + (expiry_hours as usize * 3600);
    let claims = Claims {
        agent_id: agent_id.to_string(),
        admin,
        exp,
        dbs,
        read_only,
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

/// Handler: POST /v1/auth/token: generate a new JWT.
fn extract_admin_key(request: &Request<Body>) -> Option<String> {
    if let Some(v) = request
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
    {
        return Some(v.to_string());
    }
    request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(String::from)
}

/// Authorize an admin request from request headers against the configured admin
/// key. Errors if no admin key is configured (admin endpoints are disabled) or the
/// provided `x-api-key` / bearer does not match. Used by the /v1/admin endpoints.
pub fn admin_authorized(
    headers: &axum::http::HeaderMap,
    admin_key: &Option<String>,
) -> Result<(), ApiError> {
    let expected = admin_key
        .as_deref()
        .ok_or_else(|| ApiError::Unauthorized("admin endpoints disabled: no admin key".into()))?;
    let provided = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
        })
        .ok_or_else(|| ApiError::Unauthorized("admin key required".into()))?;
    if provided != expected {
        return Err(ApiError::Forbidden("invalid admin key".into()));
    }
    Ok(())
}

pub async fn generate_token(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
) -> Result<impl IntoResponse, ApiError> {
    let secret = state.jwt_secret.as_deref().ok_or_else(|| {
        ApiError::BadRequest("auth is disabled (no jwt-secret configured)".into())
    })?;
    match &state.admin_key {
        None => {
            return Err(ApiError::Forbidden(
                "token endpoint disabled: no admin key configured".into(),
            ));
        }
        Some(expected) => {
            let provided = extract_admin_key(&request)
                .ok_or_else(|| ApiError::Unauthorized("admin key required".into()))?;
            if provided != *expected {
                return Err(ApiError::Forbidden("invalid admin key".into()));
            }
        }
    }
    let body_bytes = request
        .into_body()
        .collect()
        .await
        .map_err(|e| ApiError::BadRequest(format!("failed to read body: {e}")))?
        .to_bytes();
    let req: TokenRequest = serde_json::from_slice(&body_bytes)
        .map_err(|e| ApiError::BadRequest(format!("invalid JSON: {e}")))?;
    if let Some(dbs) = &req.dbs {
        for name in dbs.iter().filter(|d| d.as_str() != "*") {
            crate::databases::validate_name(name)?;
        }
    }
    let token = create_scoped_token(
        secret,
        &req.agent_id,
        false,
        req.expiry_hours,
        req.dbs.clone(),
        req.read_only,
    );
    Ok(Json(json!({
        "token": token,
        "agent_id": req.agent_id,
        "dbs": req.dbs,
        "read_only": req.read_only,
    })))
}

/// Middleware that enforces JWT auth when a secret is configured.
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    let verifier = match &state.verifier {
        Some(v) => v.clone(),
        // No secret or key configured means development mode; skip auth.
        None => return next.run(request).await,
    };

    // Allow health, token, metrics, and peer-to-peer gossip endpoints without auth.
    let path = request.uri().path();
    // Admin endpoints are gated by the admin key inside the handler, not the JWT,
    // so the bundled console (which sends x-api-key) can reach them.
    if path == "/v1/health"
        || path == "/v1/ready"
        || path == "/v1/auth/token"
        || path == "/metrics"
        || path == "/console"
        || path.starts_with("/v1/admin")
        || path == crate::cluster::GOSSIP_PATH
    {
        return next.run(request).await;
    }

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(String::from);

    let token = match auth_header.as_deref() {
        Some(h) if h.starts_with("Bearer ") => &h[7..],
        _ => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "error": "missing or invalid Authorization header" })),
            )
                .into_response();
        }
    };

    match verifier.authenticate(token) {
        Ok(agent) => {
            request.extensions_mut().insert(agent);
            next.run(request).await
        }
        Err(e) => e.into_response(),
    }
}
