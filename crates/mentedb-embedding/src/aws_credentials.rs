//! AWS credentials for the hand-rolled SigV4 signers (Bedrock embeddings here,
//! Bedrock extraction in `mentedb-extraction`).
//!
//! Resolution order, checked once at construction:
//! 1. Static keys: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (and
//!    `AWS_SESSION_TOKEN` for temporary credentials).
//! 2. The container credentials endpoint used by ECS task roles and EKS Pod
//!    Identity: `AWS_CONTAINER_CREDENTIALS_FULL_URI` (with
//!    `AWS_CONTAINER_AUTHORIZATION_TOKEN` or `_TOKEN_FILE` when set), else
//!    `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` on `169.254.170.2`.
//!
//! Endpoint credentials are temporary, so they are cached and fetched again
//! shortly before they expire. Profiles and EC2 instance metadata are not
//! resolved; export keys or run in a container for those.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};

use mentedb_core::error::{MenteError, MenteResult};

const ECS_ENDPOINT_HOST: &str = "http://169.254.170.2";

/// A resolved set of AWS credentials.
#[derive(Clone)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

impl AwsCredentials {
    /// Load from `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
    /// `AWS_SESSION_TOKEN` (the last is required for SSO/temporary credentials).
    pub fn from_env() -> MenteResult<Self> {
        let access_key_id = std::env::var("AWS_ACCESS_KEY_ID")
            .map_err(|_| MenteError::Storage("AWS_ACCESS_KEY_ID not set".to_string()))?;
        let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .map_err(|_| MenteError::Storage("AWS_SECRET_ACCESS_KEY not set".to_string()))?;
        Ok(Self {
            access_key_id,
            secret_access_key,
            session_token: std::env::var("AWS_SESSION_TOKEN").ok(),
        })
    }
}

/// Settings for fetching credentials from the container endpoint.
#[derive(Clone, Debug)]
pub struct CredentialRefreshConfig {
    /// Fetch new credentials when the cached ones expire within this window.
    pub refresh_before_expiry: Duration,
    /// Timeout for one call to the credentials endpoint (a link-local address).
    pub endpoint_timeout: Duration,
}

impl Default for CredentialRefreshConfig {
    fn default() -> Self {
        Self {
            refresh_before_expiry: Duration::from_secs(5 * 60),
            endpoint_timeout: Duration::from_secs(2),
        }
    }
}

#[derive(Clone)]
enum ContainerAuth {
    None,
    Token(String),
    TokenFile(PathBuf),
}

enum Source {
    Static(AwsCredentials),
    Container {
        url: String,
        auth: ContainerAuth,
        agent: ureq::Agent,
        refresh: CredentialRefreshConfig,
        cache: Mutex<Option<(AwsCredentials, DateTime<Utc>)>>,
    },
}

/// Where request signing gets its credentials from. Cheap to clone and safe to
/// share across threads; endpoint credentials are cached between calls.
#[derive(Clone)]
pub struct AwsCredentialSource {
    inner: Arc<Source>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
struct EndpointCredentials {
    access_key_id: String,
    secret_access_key: String,
    token: Option<String>,
    expiration: String,
}

impl AwsCredentialSource {
    /// Resolve the source from the environment (see the module docs). Fails
    /// when neither static keys nor a container endpoint are configured. No
    /// network call is made here.
    pub fn from_env() -> MenteResult<Self> {
        Self::from_env_with(CredentialRefreshConfig::default())
    }

    /// [`Self::from_env`] with explicit endpoint refresh settings.
    pub fn from_env_with(refresh: CredentialRefreshConfig) -> MenteResult<Self> {
        if let Ok(creds) = AwsCredentials::from_env() {
            return Ok(Self::fixed(creds));
        }
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let auth = if let Some(file) = var("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE") {
            ContainerAuth::TokenFile(PathBuf::from(file))
        } else if let Some(token) = var("AWS_CONTAINER_AUTHORIZATION_TOKEN") {
            ContainerAuth::Token(token)
        } else {
            ContainerAuth::None
        };
        if let Some(url) = var("AWS_CONTAINER_CREDENTIALS_FULL_URI") {
            return Ok(Self::container(url, auth, refresh));
        }
        if let Some(path) = var("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") {
            return Ok(Self::container(
                format!("{ECS_ENDPOINT_HOST}{path}"),
                auth,
                refresh,
            ));
        }
        Err(MenteError::Storage(
            "no AWS credentials: set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or run with \
             an ECS task role or EKS Pod Identity (AWS_CONTAINER_CREDENTIALS_RELATIVE_URI / \
             AWS_CONTAINER_CREDENTIALS_FULL_URI)"
                .to_string(),
        ))
    }

    /// Always sign with these credentials.
    pub fn fixed(creds: AwsCredentials) -> Self {
        Self {
            inner: Arc::new(Source::Static(creds)),
        }
    }

    fn container(url: String, auth: ContainerAuth, refresh: CredentialRefreshConfig) -> Self {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(refresh.endpoint_timeout))
            .build()
            .into();
        Self {
            inner: Arc::new(Source::Container {
                url,
                auth,
                agent,
                refresh,
                cache: Mutex::new(None),
            }),
        }
    }

    /// Credentials to sign the next request with. Endpoint credentials come
    /// from the cache unless they expire within the refresh window.
    pub fn credentials(&self) -> MenteResult<AwsCredentials> {
        match self.inner.as_ref() {
            Source::Static(creds) => Ok(creds.clone()),
            Source::Container {
                url,
                auth,
                agent,
                refresh,
                cache,
            } => {
                let mut cached = cache.lock().unwrap_or_else(|e| e.into_inner());
                let window = chrono::Duration::from_std(refresh.refresh_before_expiry)
                    .unwrap_or_else(|_| chrono::Duration::minutes(5));
                if let Some((creds, expires)) = cached.as_ref()
                    && *expires - window > Utc::now()
                {
                    return Ok(creds.clone());
                }
                let fresh = fetch_endpoint(agent, url, auth)?;
                *cached = Some(fresh.clone());
                Ok(fresh.0)
            }
        }
    }
}

fn fetch_endpoint(
    agent: &ureq::Agent,
    url: &str,
    auth: &ContainerAuth,
) -> MenteResult<(AwsCredentials, DateTime<Utc>)> {
    let token = match auth {
        ContainerAuth::None => None,
        ContainerAuth::Token(t) => Some(t.clone()),
        ContainerAuth::TokenFile(path) => Some(
            std::fs::read_to_string(path)
                .map_err(|e| {
                    MenteError::Storage(format!(
                        "reading AWS container authorization token {}: {e}",
                        path.display()
                    ))
                })?
                .trim()
                .to_string(),
        ),
    };
    let mut req = agent.get(url).header("accept", "application/json");
    if let Some(token) = &token {
        req = req.header("authorization", token);
    }
    let mut resp = req.call().map_err(|e| {
        MenteError::Storage(format!("AWS container credentials request failed: {e}"))
    })?;
    let parsed: EndpointCredentials = resp
        .body_mut()
        .read_json()
        .map_err(|e| MenteError::Storage(format!("AWS container credentials parse failed: {e}")))?;
    let expires = DateTime::parse_from_rfc3339(&parsed.expiration)
        .map_err(|e| {
            MenteError::Storage(format!(
                "AWS container credentials expiration '{}' is not RFC 3339: {e}",
                parsed.expiration
            ))
        })?
        .with_timezone(&Utc);
    Ok((
        AwsCredentials {
            access_key_id: parsed.access_key_id,
            secret_access_key: parsed.secret_access_key,
            session_token: parsed.token,
        },
        expires,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A one-route credentials endpoint that counts requests and records the
    /// authorization header it saw. Each response expires `ttl` from now.
    fn endpoint(ttl: chrono::Duration) -> (String, Arc<AtomicUsize>, Arc<Mutex<Option<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let seen_auth = Arc::new(Mutex::new(None));
        let (h, a) = (hits.clone(), seen_auth.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                loop {
                    line.clear();
                    if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("authorization: ") {
                        *a.lock().unwrap() = Some(v.trim().to_string());
                    }
                }
                let n = h.fetch_add(1, Ordering::SeqCst) + 1;
                let body = serde_json::json!({
                    "AccessKeyId": format!("AKID{n}"),
                    "SecretAccessKey": "secret",
                    "Token": "session",
                    "Expiration": (Utc::now() + ttl).to_rfc3339(),
                })
                .to_string();
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        (format!("http://{addr}/creds"), hits, seen_auth)
    }

    #[test]
    fn endpoint_credentials_are_cached_until_near_expiry() {
        let (url, hits, _) = endpoint(chrono::Duration::hours(6));
        let src = AwsCredentialSource::container(
            url,
            ContainerAuth::None,
            CredentialRefreshConfig::default(),
        );
        let first = src.credentials().unwrap();
        let second = src.credentials().unwrap();
        assert_eq!(first.access_key_id, "AKID1");
        assert_eq!(second.access_key_id, "AKID1");
        assert_eq!(first.session_token.as_deref(), Some("session"));
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "second call served from cache"
        );
    }

    #[test]
    fn credentials_inside_the_refresh_window_are_fetched_again() {
        // Expires in 2 minutes, inside the default 5 minute window.
        let (url, hits, _) = endpoint(chrono::Duration::minutes(2));
        let src = AwsCredentialSource::container(
            url,
            ContainerAuth::None,
            CredentialRefreshConfig::default(),
        );
        assert_eq!(src.credentials().unwrap().access_key_id, "AKID1");
        assert_eq!(src.credentials().unwrap().access_key_id, "AKID2");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn authorization_token_is_sent() {
        let (url, _, seen) = endpoint(chrono::Duration::hours(1));
        let src = AwsCredentialSource::container(
            url,
            ContainerAuth::Token("pod-token".to_string()),
            CredentialRefreshConfig::default(),
        );
        src.credentials().unwrap();
        assert_eq!(seen.lock().unwrap().as_deref(), Some("pod-token"));
    }

    #[test]
    fn unreachable_endpoint_is_an_error_not_a_hang() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/creds", listener.local_addr().unwrap());
        drop(listener);
        let src = AwsCredentialSource::container(
            url,
            ContainerAuth::None,
            CredentialRefreshConfig::default(),
        );
        assert!(src.credentials().is_err());
    }
}
