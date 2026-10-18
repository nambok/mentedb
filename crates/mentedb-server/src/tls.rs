//! Native TLS for the REST API and gRPC.
//!
//! When `--tls-cert` and `--tls-key` (or `MENTEDB_TLS_CERT_FILE` and
//! `MENTEDB_TLS_KEY_FILE`) are both set, the REST API is served over HTTPS
//! with rustls and gRPC over TLS with the same certificate. Neither set keeps
//! plain HTTP; only one set is a startup error.

use std::future::Future;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum_server::tls_rustls::RustlsConfig;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

/// Certificate chain and private key locations, both PEM.
#[derive(Debug, Clone)]
pub struct TlsPaths {
    pub cert: PathBuf,
    pub key: PathBuf,
}

/// Loaded TLS material, ready for the REST and gRPC listeners.
#[derive(Clone)]
pub struct TlsMaterial {
    /// rustls server config for the REST API (ALPN h2 and http/1.1).
    pub rest: Arc<rustls::ServerConfig>,
    /// Raw PEM bytes for tonic's gRPC identity.
    pub cert_pem: Vec<u8>,
    pub key_pem: Vec<u8>,
}

/// Decide whether TLS is on. Both paths set turns it on, neither keeps plain
/// HTTP, and exactly one is a configuration error.
pub fn resolve(cert: Option<PathBuf>, key: Option<PathBuf>) -> Result<Option<TlsPaths>, String> {
    match (cert, key) {
        (Some(cert), Some(key)) => Ok(Some(TlsPaths { cert, key })),
        (None, None) => Ok(None),
        (Some(_), None) => Err(
            "--tls-cert (MENTEDB_TLS_CERT_FILE) is set but --tls-key (MENTEDB_TLS_KEY_FILE) is \
             not; set both to serve HTTPS, or neither for plain HTTP"
                .to_string(),
        ),
        (None, Some(_)) => Err(
            "--tls-key (MENTEDB_TLS_KEY_FILE) is set but --tls-cert (MENTEDB_TLS_CERT_FILE) is \
             not; set both to serve HTTPS, or neither for plain HTTP"
                .to_string(),
        ),
    }
}

/// Read and validate the certificate chain and key. Fails at startup on a
/// missing file, an empty chain, or a key that does not match.
pub fn load(paths: &TlsPaths) -> Result<TlsMaterial, String> {
    let cert_pem = read(&paths.cert, "certificate")?;
    let key_pem = read(&paths.key, "private key")?;
    let rest = server_config(&cert_pem, &key_pem).map_err(|e| {
        format!(
            "TLS {} / {}: {e}",
            paths.cert.display(),
            paths.key.display()
        )
    })?;
    Ok(TlsMaterial {
        rest: Arc::new(rest),
        cert_pem,
        key_pem,
    })
}

fn read(path: &Path, what: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("reading TLS {what} {}: {e}", path.display()))
}

/// Build the rustls server config from PEM bytes on the ring provider.
pub fn server_config(cert_pem: &[u8], key_pem: &[u8]) -> Result<rustls::ServerConfig, String> {
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(cert_pem)
        .collect::<Result<_, _>>()
        .map_err(|e| format!("invalid certificate PEM: {e}"))?;
    if certs.is_empty() {
        return Err("no certificate found in the certificate PEM".to_string());
    }
    let key = PrivateKeyDer::from_pem_slice(key_pem)
        .map_err(|e| format!("invalid private key PEM: {e}"))?;
    let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|e| e.to_string())?
    .with_no_client_auth()
    .with_single_cert(certs, key)
    .map_err(|e| e.to_string())?;
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(config)
}

/// Serve the REST router on `listener`, over HTTPS when `tls` is set and
/// plain HTTP otherwise, with client addresses available to handlers and a
/// graceful shutdown when `shutdown` resolves.
pub async fn serve_rest<F>(
    listener: tokio::net::TcpListener,
    app: Router,
    tls: Option<Arc<rustls::ServerConfig>>,
    shutdown: F,
    drain_timeout: Duration,
) -> std::io::Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let make_service = app.into_make_service_with_connect_info::<SocketAddr>();
    match tls {
        None => {
            axum::serve(listener, make_service)
                .with_graceful_shutdown(shutdown)
                .await
        }
        Some(config) => {
            let handle = axum_server::Handle::new();
            let shutdown_handle = handle.clone();
            tokio::spawn(async move {
                shutdown.await;
                shutdown_handle.graceful_shutdown(Some(drain_timeout));
            });
            let std_listener = listener.into_std()?;
            axum_server::from_tcp_rustls(std_listener, RustlsConfig::from_config(config))
                .handle(handle)
                .serve(make_service)
                .await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    use axum::middleware;

    use crate::state::AppState;

    fn self_signed() -> (String, String) {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        (cert.cert.pem(), cert.key_pair.serialize_pem())
    }

    #[test]
    fn one_path_without_the_other_is_an_error() {
        assert!(resolve(None, None).unwrap().is_none());
        assert!(
            resolve(Some("c.pem".into()), Some("k.pem".into()))
                .unwrap()
                .is_some()
        );
        assert!(resolve(Some("c.pem".into()), None).is_err());
        assert!(resolve(None, Some("k.pem".into())).is_err());
    }

    #[test]
    fn mismatched_or_garbage_pem_fails_at_startup() {
        let (cert, _) = self_signed();
        let (_, other_key) = self_signed();
        assert!(server_config(cert.as_bytes(), other_key.as_bytes()).is_err());
        assert!(server_config(b"not a pem", b"not a pem").is_err());
    }

    #[tokio::test]
    async fn health_is_served_over_https_with_a_self_signed_cert() {
        let (cert_pem, key_pem) = self_signed();
        let dir = tempfile::TempDir::new().unwrap();
        let cert_path = dir.path().join("cert.pem");
        let key_path = dir.path().join("key.pem");
        std::fs::write(&cert_path, &cert_pem).unwrap();
        std::fs::write(&key_path, &key_pem).unwrap();
        let paths = resolve(Some(cert_path), Some(key_path)).unwrap().unwrap();
        let material = load(&paths).unwrap();

        // The production router and middleware stack on a temp data dir.
        let databases = Arc::new(
            crate::databases::Databases::open(
                &dir.path().join("data"),
                None,
                Default::default(),
                Default::default(),
            )
            .unwrap(),
        );
        let state = Arc::new(AppState {
            db: databases.default_db(),
            databases,
            verifier: None,
            spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
            jwt_secret: None,
            admin_key: None,
            start_time: Instant::now(),
            extraction_config: None,
            auto_extract: false,
            extraction_tx: None,
            cluster: None,
            turn_trackers: Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
        });
        let app = crate::routes::build_router(state.clone())
            .layer(middleware::from_fn_with_state(
                state.clone(),
                crate::auth::auth_middleware,
            ))
            .layer(crate::rate_limit::RateLimiter::default());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(serve_rest(
            listener,
            app,
            Some(material.rest.clone()),
            async {
                let _ = stop_rx.await;
            },
            Duration::from_secs(1),
        ));

        let client = reqwest::Client::builder()
            .add_root_certificate(reqwest::Certificate::from_pem(cert_pem.as_bytes()).unwrap())
            .build()
            .unwrap();
        let resp = client
            .get(format!("https://localhost:{port}/v1/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), reqwest::StatusCode::OK);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "ok", "unexpected health body: {body}");

        // A plain HTTP request to the TLS port must not be served.
        let plain = reqwest::Client::new()
            .get(format!("http://localhost:{port}/v1/health"))
            .send()
            .await;
        assert!(
            plain.map(|r| !r.status().is_success()).unwrap_or(true),
            "plain HTTP must not succeed against the TLS listener"
        );

        let _ = stop_tx.send(());
        tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .expect("server did not shut down")
            .unwrap()
            .unwrap();
    }
}
