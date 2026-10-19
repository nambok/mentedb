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

/// What a token may do. Levels are ordered: each includes the ones before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Scope {
    /// Read and search only.
    Read,
    /// Read and write memories.
    Write,
    /// Write, plus dropping and backing up the databases the token names.
    Maintenance,
}

impl Scope {
    /// Parse the `scope` claim: `read`, `write` or `maintenance`.
    pub fn parse(s: &str) -> Result<Self, ApiError> {
        match s.trim().to_lowercase().as_str() {
            "read" => Ok(Self::Read),
            "write" => Ok(Self::Write),
            "maintenance" => Ok(Self::Maintenance),
            other => Err(ApiError::Unauthorized(format!(
                "unknown scope '{other}': use read, write or maintenance"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Maintenance => "maintenance",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AuthenticatedAgent {
    /// Agent the token was issued for; empty when the token is scoped to
    /// databases rather than to one agent.
    pub agent_id: String,
    pub admin: bool,
    /// Databases this token may use: exact names or globs where `*` matches
    /// any run of characters. `None` means every database (allowed only when
    /// the server does not require the claim).
    pub dbs: Option<Vec<String>>,
    /// Read-only tokens may search and read but not write. Same as
    /// `scope == Scope::Read`.
    pub read_only: bool,
    /// What the token may do.
    pub scope: Scope,
    /// The token's `sub` claim.
    pub subject: Option<String>,
    /// The token's `jti` claim.
    pub token_id: Option<String>,
}

/// Whether `text` matches `pattern`, where `*` in the pattern matches any run
/// of characters (including none) and everything else matches itself.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark) = (None::<usize>, 0usize);
    while ti < t.len() {
        if pi < p.len() && p[pi] != '*' && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

impl AuthenticatedAgent {
    /// Whether the token's `dbs` claim admits database `name`.
    fn names_db(&self, name: &str) -> bool {
        match &self.dbs {
            None => true,
            Some(list) => list.iter().any(|d| d == "*" || glob_match(d, name)),
        }
    }

    /// Allow or refuse access to database `name`.
    pub fn authorize_db(&self, name: &str, write: bool) -> Result<(), ApiError> {
        if write && self.scope < Scope::Write {
            return Err(ApiError::Forbidden("token is read-only".into()));
        }
        if self.admin || self.names_db(name) {
            Ok(())
        } else {
            Err(ApiError::Forbidden(format!(
                "token is not scoped to database '{name}'"
            )))
        }
    }

    /// Allow or refuse dropping or backing up database `name`: admin tokens,
    /// or a token with the maintenance scope whose `dbs` names the database.
    pub fn authorize_maintenance(&self, name: &str) -> Result<(), ApiError> {
        if self.admin {
            return Ok(());
        }
        if self.scope < Scope::Maintenance {
            return Err(ApiError::Forbidden(
                "this operation needs a token with the maintenance scope".into(),
            ));
        }
        self.authorize_db(name, true)
    }

    /// The key requests from this caller are rate limited under: the token's
    /// `sub`, else its `jti`, else its agent.
    pub fn rate_key(&self) -> Option<String> {
        self.subject
            .clone()
            .or_else(|| self.token_id.clone())
            .or_else(|| (!self.agent_id.is_empty()).then(|| self.agent_id.clone()))
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
    /// `read`, `write` or `maintenance`. Absent means `write`, or `read` when
    /// `read_only` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Subject the token was issued to; binds `dbs` when the server sets a
    /// subject pattern, and keys the rate limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sub: Option<String>,
    /// Token id; keys the rate limit when there is no `sub`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jti: Option<String>,
}

impl Claims {
    fn into_agent(self) -> Result<AuthenticatedAgent, ApiError> {
        let scope = match self.scope.as_deref() {
            Some(s) => Scope::parse(s)?,
            None if self.read_only => Scope::Read,
            None => Scope::Write,
        };
        Ok(AuthenticatedAgent {
            agent_id: self.agent_id,
            admin: self.admin,
            dbs: self.dbs,
            read_only: scope == Scope::Read,
            scope,
            subject: self.sub,
            token_id: self.jti,
        })
    }
}

/// Rules a token must meet beyond a valid signature.
#[derive(Clone, Debug, Default)]
pub struct TokenPolicy {
    /// Refuse non admin tokens without a `dbs` claim (by default an absent
    /// claim means every database).
    pub require_dbs: bool,
    /// A glob every `dbs` entry must match after `{sub}` is replaced by the
    /// token's `sub`, e.g. `tenant.{sub}.*`. Tokens without a `sub`, or whose
    /// `sub` holds anything but letters, digits, `-` and `_`, are refused.
    pub dbs_sub_pattern: Option<String>,
}

impl TokenPolicy {
    /// Check a decoded caller against the policy.
    pub fn check(&self, agent: &AuthenticatedAgent) -> Result<(), ApiError> {
        if agent.admin {
            return Ok(());
        }
        if self.require_dbs && agent.dbs.is_none() {
            return Err(ApiError::Unauthorized(
                "token has no 'dbs' claim; this server requires one".into(),
            ));
        }
        if let Some(pattern) = &self.dbs_sub_pattern {
            let sub = agent.subject.as_deref().unwrap_or("");
            let safe = !sub.is_empty()
                && sub
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
            if !safe {
                return Err(ApiError::Unauthorized(
                    "token needs a 'sub' of letters, digits, '-' or '_' to use databases".into(),
                ));
            }
            let bound = pattern.replace("{sub}", sub);
            let dbs = agent.dbs.as_deref().ok_or_else(|| {
                ApiError::Unauthorized(
                    "token has no 'dbs' claim; this server binds databases to 'sub'".into(),
                )
            })?;
            if let Some(bad) = dbs.iter().find(|d| !glob_match(&bound, d)) {
                return Err(ApiError::Unauthorized(format!(
                    "dbs entry '{bad}' is outside '{bound}', the databases of this subject"
                )));
            }
        }
        Ok(())
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
    policy: TokenPolicy,
}

impl TokenVerifier {
    /// Verify HS256 tokens signed with `secret`.
    pub fn shared_secret(secret: &str) -> Self {
        Self {
            key: DecodingKey::from_secret(secret.as_bytes()),
            validation: Validation::default(),
            policy: TokenPolicy::default(),
        }
    }

    /// Enforce `policy` on every token.
    pub fn with_policy(mut self, policy: TokenPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Verify tokens with a public key given as PEM, or as base64 of the DER
    /// encoded SubjectPublicKeyInfo (what `openssl pkey -pubout -outform DER`
    /// writes), for keys passed inline through an environment variable.
    pub fn public_key_text(text: &str) -> Result<Self, String> {
        let trimmed = text.trim();
        if trimmed.starts_with("-----BEGIN") {
            return Self::public_key_pem(trimmed.as_bytes());
        }
        let b64: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
        if b64.is_empty()
            || !b64
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '='))
        {
            return Err("public key is neither PEM nor base64 DER".to_string());
        }
        let mut pem = String::from("-----BEGIN PUBLIC KEY-----\n");
        for chunk in b64.as_bytes().chunks(64) {
            pem.push_str(&String::from_utf8_lossy(chunk));
            pem.push('\n');
        }
        pem.push_str("-----END PUBLIC KEY-----\n");
        Self::public_key_pem(pem.as_bytes())
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
            policy: TokenPolicy::default(),
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

    /// Validate a token into the caller it represents, enforcing the policy.
    pub fn authenticate(&self, token: &str) -> Result<AuthenticatedAgent, ApiError> {
        let agent = self.verify(token)?.into_agent()?;
        self.policy.check(&agent)?;
        Ok(agent)
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
    /// `read`, `write` or `maintenance`.
    #[serde(default)]
    pub scope: Option<String>,
    /// Subject to put in the `sub` claim.
    #[serde(default)]
    pub sub: Option<String>,
}

/// Extra claims for [`create_token_with`].
#[derive(Clone, Debug, Default)]
pub struct TokenOptions {
    pub dbs: Option<Vec<String>>,
    pub read_only: bool,
    pub scope: Option<Scope>,
    pub sub: Option<String>,
    pub jti: Option<String>,
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
    create_token_with(
        secret,
        agent_id,
        admin,
        expiry_hours,
        TokenOptions {
            dbs,
            read_only,
            ..TokenOptions::default()
        },
    )
}

/// Create a signed HS256 JWT with every optional claim.
pub fn create_token_with(
    secret: &str,
    agent_id: &str,
    admin: bool,
    expiry_hours: u64,
    opts: TokenOptions,
) -> String {
    let exp = jsonwebtoken::get_current_timestamp() as usize + (expiry_hours as usize * 3600);
    let claims = Claims {
        agent_id: agent_id.to_string(),
        admin,
        exp,
        dbs: opts.dbs,
        read_only: opts.read_only,
        scope: opts.scope.map(|s| s.as_str().to_string()),
        sub: opts.sub,
        jti: opts.jti,
    };
    // HS256 encoding of plain claims has no failure mode short of a broken
    // serializer; an empty token is refused by every verifier.
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap_or_default()
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
            // A glob entry is a valid name with `*` standing in for characters.
            crate::databases::validate_name(&name.replace('*', "x"))?;
        }
    }
    let scope = req
        .scope
        .as_deref()
        .map(Scope::parse)
        .transpose()
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let token = create_token_with(
        secret,
        &req.agent_id,
        false,
        req.expiry_hours,
        TokenOptions {
            dbs: req.dbs.clone(),
            read_only: req.read_only,
            scope,
            sub: req.sub.clone(),
            jti: Some(uuid::Uuid::new_v4().to_string()),
        },
    );
    Ok(Json(json!({
        "token": token,
        "agent_id": req.agent_id,
        "dbs": req.dbs,
        "read_only": req.read_only,
        "scope": scope.map(|s| s.as_str()),
        "sub": req.sub,
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
    if path == "/v1/health"
        || path == "/v1/ready"
        || path == "/v1/live"
        || path == "/v1/auth/token"
        || path == "/metrics"
        || path == "/console"
        || path == crate::cluster::GOSSIP_PATH
    {
        return next.run(request).await;
    }
    // Admin endpoints are gated by the admin key or an admin token inside the
    // handler, so the bundled console (which sends x-api-key) can reach them.
    // A valid bearer token is still decoded so admin tokens work too; an
    // invalid or absent one is left for the handler to refuse.
    if is_admin_path(path) {
        if let Some(agent) = request
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|h| h.strip_prefix("Bearer "))
            .and_then(|t| verifier.authenticate(t).ok())
        {
            request.extensions_mut().insert(agent);
        }
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

/// Paths whose handlers check the admin key themselves: `/v1/admin/...`,
/// `/v1/databases...`, `/v1/db/{db}` itself (drop), and per database admin,
/// backup and backups routes. A valid token is still decoded on these, so
/// database scoped maintenance tokens reach the handler too.
fn is_admin_path(path: &str) -> bool {
    let rest = match path.strip_prefix("/v1/db/") {
        Some(r) => match r.split_once('/') {
            Some((_, rest)) => rest,
            None => return !r.is_empty(),
        },
        None => match path.strip_prefix("/v1/") {
            Some(r) => r,
            None => return false,
        },
    };
    rest == "databases"
        || rest.starts_with("databases/")
        || rest == "admin"
        || rest.starts_with("admin/")
        || rest == "backup"
        || rest == "backups"
}

#[cfg(test)]
mod admin_path_tests {
    use super::is_admin_path;

    #[test]
    fn admin_paths_are_recognized_on_every_mount() {
        for p in [
            "/v1/admin/mql",
            "/v1/db/alice/admin/memories",
            "/v1/databases",
            "/v1/databases/alice",
            "/v1/backup",
            "/v1/db/alice/backups",
            // The database itself (DELETE drops it).
            "/v1/db/admin",
            "/v1/db/tenant.alice.notes",
        ] {
            assert!(is_admin_path(p), "{p}");
        }
        for p in [
            "/v1/memories",
            "/v1/db/alice/memories",
            "/v1/db/admin/memories",
            "/v1/db/",
            "/v1/search",
        ] {
            assert!(!is_admin_path(p), "{p}");
        }
    }
}

#[cfg(test)]
mod scope_tests {
    use super::*;

    const SECRET: &str = "test-secret";

    fn token(opts: TokenOptions) -> String {
        create_token_with(SECRET, "", false, 1, opts)
    }

    fn verifier(policy: TokenPolicy) -> TokenVerifier {
        TokenVerifier::shared_secret(SECRET).with_policy(policy)
    }

    #[test]
    fn glob_matching() {
        assert!(glob_match("tenant.alice.*", "tenant.alice.notes"));
        assert!(glob_match("tenant.alice.*", "tenant.alice."));
        assert!(!glob_match("tenant.alice.*", "tenant.alicex.notes"));
        assert!(!glob_match("tenant.alice.*", "tenant.bob.notes"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("a*c*e", "abcde"));
        assert!(!glob_match("a*c*e", "abcd"));
        assert!(glob_match("exact", "exact"));
        assert!(!glob_match("exact", "exactly"));
    }

    #[test]
    fn read_scope_denies_writes_and_write_scope_allows_them() {
        let v = verifier(TokenPolicy::default());
        let read = v
            .authenticate(&token(TokenOptions {
                scope: Some(Scope::Read),
                ..Default::default()
            }))
            .unwrap();
        assert!(read.authorize_db("x", false).is_ok());
        assert!(read.authorize_db("x", true).is_err());
        let legacy = v
            .authenticate(&token(TokenOptions {
                read_only: true,
                ..Default::default()
            }))
            .unwrap();
        assert_eq!(legacy.scope, Scope::Read);
        assert!(legacy.authorize_db("x", true).is_err());
        let write = v.authenticate(&token(TokenOptions::default())).unwrap();
        assert_eq!(write.scope, Scope::Write);
        assert!(write.authorize_db("x", true).is_ok());
    }

    #[test]
    fn maintenance_needs_the_scope_and_the_database() {
        let v = verifier(TokenPolicy::default());
        let maint = v
            .authenticate(&token(TokenOptions {
                scope: Some(Scope::Maintenance),
                dbs: Some(vec!["tenant.alice.*".into()]),
                ..Default::default()
            }))
            .unwrap();
        assert!(maint.authorize_maintenance("tenant.alice.notes").is_ok());
        assert!(maint.authorize_maintenance("tenant.bob.notes").is_err());
        assert!(maint.authorize_db("tenant.alice.notes", true).is_ok());
        let write = v
            .authenticate(&token(TokenOptions {
                dbs: Some(vec!["tenant.alice.notes".into()]),
                ..Default::default()
            }))
            .unwrap();
        assert!(write.authorize_maintenance("tenant.alice.notes").is_err());
    }

    #[test]
    fn unknown_scope_is_refused() {
        let v = verifier(TokenPolicy::default());
        let claims = Claims {
            agent_id: String::new(),
            admin: false,
            exp: jsonwebtoken::get_current_timestamp() as usize + 600,
            dbs: None,
            read_only: false,
            scope: Some("owner".into()),
            sub: None,
            jti: None,
        };
        let t = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(SECRET.as_bytes()),
        )
        .unwrap();
        assert!(v.authenticate(&t).is_err());
    }

    #[test]
    fn required_dbs_refuses_unscoped_tokens() {
        let v = verifier(TokenPolicy {
            require_dbs: true,
            ..Default::default()
        });
        assert!(v.authenticate(&token(TokenOptions::default())).is_err());
        assert!(
            v.authenticate(&token(TokenOptions {
                dbs: Some(vec!["a".into()]),
                ..Default::default()
            }))
            .is_ok()
        );
        let admin = create_token_with(SECRET, "", true, 1, TokenOptions::default());
        assert!(v.authenticate(&admin).is_ok(), "admin tokens are exempt");
        // Without the policy an absent claim still means every database.
        let open = verifier(TokenPolicy::default())
            .authenticate(&token(TokenOptions::default()))
            .unwrap();
        assert!(open.authorize_db("anything", true).is_ok());
    }

    #[test]
    fn sub_pattern_binds_dbs_to_the_subject() {
        let v = verifier(TokenPolicy {
            require_dbs: true,
            dbs_sub_pattern: Some("tenant.{sub}.*".into()),
        });
        let ok = |sub: &str, dbs: &[&str]| {
            v.authenticate(&token(TokenOptions {
                sub: Some(sub.into()),
                dbs: Some(dbs.iter().map(|s| s.to_string()).collect()),
                ..Default::default()
            }))
            .is_ok()
        };
        assert!(ok("alice", &["tenant.alice.notes"]));
        assert!(ok("alice", &["tenant.alice.*"]));
        assert!(!ok("alice", &["tenant.bob.notes"]));
        assert!(!ok("alice", &["tenant.alice.notes", "tenant.bob.notes"]));
        assert!(!ok("alice", &["*"]));
        assert!(!ok("a*", &["tenant.a*.x"]), "glob characters in sub");
        assert!(!ok("a.b", &["tenant.a.b.x"]), "dots in sub");
        assert!(
            v.authenticate(&token(TokenOptions {
                dbs: Some(vec!["tenant.alice.notes".into()]),
                ..Default::default()
            }))
            .is_err(),
            "no sub"
        );
    }

    #[test]
    fn rate_key_prefers_sub_then_jti_then_agent() {
        let v = verifier(TokenPolicy::default());
        let a = v
            .authenticate(&token(TokenOptions {
                sub: Some("s".into()),
                jti: Some("j".into()),
                ..Default::default()
            }))
            .unwrap();
        assert_eq!(a.rate_key().as_deref(), Some("s"));
        let b = v
            .authenticate(&token(TokenOptions {
                jti: Some("j".into()),
                ..Default::default()
            }))
            .unwrap();
        assert_eq!(b.rate_key().as_deref(), Some("j"));
    }

    #[test]
    fn public_key_accepts_pem_or_base64_der() {
        // An Ed25519 public key in PEM and as base64 DER.
        let pem = "-----BEGIN PUBLIC KEY-----\n\
                   MCowBQYDK2VwAyEAGb9ECWmEzf6FQbrBZ9w7lshQhqowtrbLDFw4rXAxZuE=\n\
                   -----END PUBLIC KEY-----\n";
        assert!(TokenVerifier::public_key_text(pem).is_ok());
        assert!(
            TokenVerifier::public_key_text(
                "MCowBQYDK2VwAyEAGb9ECWmEzf6FQbrBZ9w7lshQhqowtrbLDFw4rXAxZuE="
            )
            .is_ok()
        );
        assert!(TokenVerifier::public_key_text("not a key!").is_err());
    }
}
