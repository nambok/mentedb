//! The verified erase endpoint: `POST /v1/memories/{id}/erase` and its named
//! database form, the receipt it returns, and who may call it.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware;
use http_body_util::BodyExt;
use mentedb_server::auth;
use mentedb_server::routes;
use mentedb_server::state::AppState;
use serde_json::{Value, json};
use tempfile::TempDir;
use tower::ServiceExt;

const AGENT: &str = "00000000-0000-0000-0000-000000000001";
const OTHER_AGENT: &str = "00000000-0000-0000-0000-000000000002";
const JWT_SECRET: &str = "erase-test-secret";
const MARKER: &str = "ERASE-ENDPOINT-MARKER-5e2b: the user's door code is 8812";

fn build_app() -> (axum::Router, TempDir) {
    let tmp = TempDir::new().unwrap();
    let embedder: Arc<dyn mentedb_embedding::EmbeddingProvider> =
        Arc::new(mentedb_embedding::HashEmbeddingProvider::new(384));
    let databases = Arc::new(
        mentedb_server::databases::Databases::open(
            tmp.path(),
            Some(embedder),
            Default::default(),
            Default::default(),
        )
        .unwrap(),
    );
    let state = Arc::new(AppState {
        db: databases.default_db(),
        databases: databases.clone(),
        verifier: Some(Arc::new(auth::TokenVerifier::shared_secret(JWT_SECRET))),
        spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
        jwt_secret: Some(JWT_SECRET.to_string()),
        admin_key: None,
        start_time: Instant::now(),
        extraction_config: None,
        auto_extract: false,
        extraction_tx: None,
        cluster: None,
        turn_trackers: Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
    });
    let app = routes::build_router(state.clone()).layer(middleware::from_fn_with_state(
        state.clone(),
        auth::auth_middleware,
    ));
    (app, tmp)
}

async fn send(
    app: &axum::Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
    token: &str,
) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header("authorization", format!("Bearer {token}"));
    let req = match body {
        Some(b) => req
            .header("content-type", "application/json")
            .body(Body::from(b.to_string())),
        None => req.body(Body::empty()),
    }
    .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

async fn store(app: &axum::Router, prefix: &str, content: &str, token: &str) -> String {
    let (status, body) = send(
        app,
        "POST",
        &format!("{prefix}/memories/batch"),
        Some(json!({ "memories": [{
            "agent_id": AGENT,
            "memory_type": "semantic",
            "content": content,
        }] })),
        token,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    body["ids"][0].as_str().unwrap().to_string()
}

/// Files under `dir` whose bytes contain `needle`.
fn files_with(dir: &std::path::Path, needle: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if std::fs::read(&p)
                .unwrap()
                .windows(needle.len())
                .any(|w| w == needle)
            {
                out.push(p.display().to_string());
            }
        }
    }
    out
}

#[tokio::test]
async fn erase_endpoint_returns_a_receipt_and_removes_the_bytes() {
    let (app, tmp) = build_app();
    let token = auth::create_token(JWT_SECRET, AGENT, false, 1);
    for prefix in ["/v1", "/v1/db/team"] {
        let gone = store(&app, prefix, MARKER, &token).await;
        let kept = store(&app, prefix, "the user enjoys long walks", &token).await;
        assert!(!files_with(tmp.path(), MARKER.as_bytes()).is_empty());

        let (status, receipt) = send(
            &app,
            "POST",
            &format!("{prefix}/memories/{gone}/erase"),
            None,
            &token,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{receipt}");
        assert_eq!(receipt["requested"], gone);
        assert_eq!(receipt["erased"].as_array().unwrap().len(), 1);
        assert_eq!(receipt["erased"][0]["reason"], "requested");
        assert_eq!(
            receipt["erased"][0]["content_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert_eq!(
            receipt["verification"]["residual_found"], false,
            "{receipt}"
        );
        assert_eq!(receipt["steps"]["wal_scrubbed"], true);
        assert!(!receipt.to_string().contains("ERASE-ENDPOINT-MARKER"));
        assert_eq!(
            files_with(tmp.path(), MARKER.as_bytes()),
            Vec::<String>::new()
        );

        let (status, _) = send(
            &app,
            "GET",
            &format!("{prefix}/memories/{gone}"),
            None,
            &token,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _) = send(
            &app,
            "GET",
            &format!("{prefix}/memories/{kept}"),
            None,
            &token,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (status, _) = send(
            &app,
            "POST",
            &format!("{prefix}/memories/{gone}/erase"),
            None,
            &token,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}

#[tokio::test]
async fn erase_endpoint_takes_options_in_the_body() {
    let (app, _tmp) = build_app();
    let token = auth::create_token(JWT_SECRET, AGENT, false, 1);
    let gone = store(&app, "/v1", MARKER, &token).await;
    let (status, receipt) = send(
        &app,
        "POST",
        &format!("/v1/memories/{gone}/erase"),
        Some(json!({ "include_sources": true, "verify": false })),
        &token,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{receipt}");
    assert!(receipt["verification"].is_null(), "{receipt}");
    let (status, body) = send(
        &app,
        "POST",
        &format!("/v1/memories/{gone}/erase"),
        Some(json!({ "verify": "yes" })),
        &token,
    )
    .await;
    assert!(status.is_client_error(), "{status} {body}");
}

#[tokio::test]
async fn erase_needs_a_write_token_for_the_owner() {
    let (app, tmp) = build_app();
    let token = auth::create_token(JWT_SECRET, AGENT, false, 1);
    let read_only = auth::create_scoped_token(JWT_SECRET, AGENT, false, 1, None, true);
    let other = auth::create_token(JWT_SECRET, OTHER_AGENT, false, 1);
    let scoped_elsewhere = auth::create_scoped_token(
        JWT_SECRET,
        AGENT,
        false,
        1,
        Some(vec!["elsewhere".to_string()]),
        false,
    );
    let gone = store(&app, "/v1/db/team", MARKER, &token).await;
    for (who, t) in [
        ("read only", &read_only),
        ("other agent", &other),
        ("other database", &scoped_elsewhere),
    ] {
        let (status, _) = send(
            &app,
            "POST",
            &format!("/v1/db/team/memories/{gone}/erase"),
            None,
            t,
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{who}");
    }
    assert!(!files_with(tmp.path(), MARKER.as_bytes()).is_empty());
    let (status, _) = send(
        &app,
        "POST",
        "/v1/db/team/memories/not-a-uuid/erase",
        None,
        &token,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
