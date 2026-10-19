//! Tests for the operations endpoints: maintenance, enrichment, documents,
//! backups, database management and readiness.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware;
use http_body_util::BodyExt;
use mentedb::storage::backup::BackupManager;
use mentedb_core::error::MenteResult;
use mentedb_embedding::provider::EmbeddingProvider;
use mentedb_server::auth;
use mentedb_server::routes;
use mentedb_server::state::AppState;
use serde_json::{Value, json};
use tempfile::TempDir;
use tower::ServiceExt;

const ADMIN_KEY: &str = "ops-admin-key";
const JWT_SECRET: &str = "ops-test-secret";
const AGENT: &str = "00000000-0000-0000-0000-000000000001";

/// An embedder whose every call fails, for the readiness probe.
struct FailingEmbedder;

impl EmbeddingProvider for FailingEmbedder {
    fn embed(&self, _text: &str) -> MenteResult<Vec<f32>> {
        Err(mentedb_core::MenteError::Storage("embedder is down".into()))
    }
    fn embed_batch(&self, _texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        Err(mentedb_core::MenteError::Storage("embedder is down".into()))
    }
    fn dimensions(&self) -> usize {
        384
    }
    fn model_name(&self) -> &str {
        "failing"
    }
}

fn hash_embedder() -> Option<Arc<dyn EmbeddingProvider>> {
    Some(Arc::new(mentedb_embedding::HashEmbeddingProvider::new(384)))
}

fn app_with(embedder: Option<Arc<dyn EmbeddingProvider>>, jwt: bool) -> (axum::Router, TempDir) {
    let tmp = TempDir::new().unwrap();
    let databases = Arc::new(
        mentedb_server::databases::Databases::open(
            tmp.path(),
            embedder,
            Default::default(),
            Default::default(),
        )
        .unwrap(),
    );
    let state = Arc::new(AppState {
        db: databases.default_db(),
        databases: databases.clone(),
        verifier: jwt.then(|| Arc::new(auth::TokenVerifier::shared_secret(JWT_SECRET))),
        spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
        jwt_secret: jwt.then(|| JWT_SECRET.to_string()),
        admin_key: Some(ADMIN_KEY.to_string()),
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

fn app() -> (axum::Router, TempDir) {
    app_with(hash_embedder(), false)
}

async fn send(
    app: &axum::Router,
    method: &str,
    uri: &str,
    headers: &[(&str, &str)],
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut req = Request::builder().method(method).uri(uri);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let body = match body {
        Some(v) => {
            req = req.header("content-type", "application/json");
            Body::from(serde_json::to_vec(&v).unwrap())
        }
        None => Body::empty(),
    };
    let resp = app.clone().oneshot(req.body(body).unwrap()).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

const ADMIN: &[(&str, &str)] = &[("x-api-key", ADMIN_KEY)];

/// Five paragraphs of 109 characters each, each ending in a distinct word.
fn five_paragraphs() -> Vec<String> {
    (0..5)
        .map(|i| {
            format!(
                "Memory systems keep what matters and let the rest fade away. \
                 Old notes decay. Paragraph {i} closes with marker{i}"
            )
        })
        .collect()
}

async fn load_document(app: &axum::Router, prefix: &str) -> (String, Vec<String>) {
    let content = five_paragraphs().join("\n\n");
    let (status, body) = send(
        app,
        "POST",
        &format!("{prefix}/documents"),
        &[],
        Some(json!({
            "content": content,
            "title": "Handbook",
            "source": "handbook.md",
            "agent_id": AGENT,
            "tags": ["kind:doc"],
            "chunk_chars": 200,
            "overlap_chars": 40,
        })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let ids = body["chunk_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    (body["document_id"].as_str().unwrap().to_string(), ids)
}

// ---------------------------------------------------------------------------
// Maintenance
// ---------------------------------------------------------------------------

#[tokio::test]
async fn maintenance_endpoints_return_counts() {
    let (app, _tmp) = app();
    load_document(&app, "/v1").await;

    let (s, b) = send(&app, "POST", "/v1/maintenance/decay", &[], None).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert!(b["decayed"].is_u64());
    assert_eq!(b["database"], "default");

    let (s, b) = send(&app, "POST", "/v1/maintenance/archive", &[], None).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(b["evaluated"], 5, "every chunk evaluated: {b}");
    for key in [
        "keep",
        "flagged_archive",
        "flagged_consolidate",
        "flagged_delete",
        "forgotten",
    ] {
        assert!(b[key].is_u64(), "{key} in {b}");
    }

    let (s, b) = send(&app, "POST", "/v1/maintenance/consolidate", &[], None).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    for key in [
        "duplicates_merged",
        "clusters_found",
        "clusters_consolidated",
    ] {
        assert!(b[key].is_u64(), "{key} in {b}");
    }

    let (s, b) = send(&app, "POST", "/v1/maintenance/sweep", &[], None).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    for key in ["consolidated", "decayed", "forgotten"] {
        assert!(b[key].is_u64(), "{key} in {b}");
    }

    // The same routes on a named database.
    let (s, b) = send(&app, "POST", "/v1/db/team/maintenance/sweep", &[], None).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(b["database"], "team");
}

// ---------------------------------------------------------------------------
// Enrichment
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enrich_without_an_llm_is_503() {
    let (app, _tmp) = app();
    let (s, b) = send(&app, "POST", "/v1/enrich", &[], None).await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
    assert!(b["error"].as_str().unwrap().contains("LLM"), "{b}");
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

#[tokio::test]
async fn documents_split_into_overlapping_chunks_and_delete_together() {
    let (app, _tmp) = app();
    let (doc_id, ids) = load_document(&app, "/v1").await;
    // Two 109 character paragraphs never fit in 200 characters, so each
    // paragraph is its own chunk.
    assert_eq!(ids.len(), 5);

    let mut chunks = Vec::new();
    for id in &ids {
        let (s, b) = send(&app, "GET", &format!("/v1/memories/{id}"), &[], None).await;
        assert_eq!(s, StatusCode::OK, "{b}");
        chunks.push(b);
    }
    for (i, chunk) in chunks.iter().enumerate() {
        let content = chunk["content"].as_str().unwrap();
        assert!(content.chars().count() <= 200, "{content}");
        assert!(content.contains(&format!("marker{i}")), "{content}");
        if i > 0 {
            // The overlap carries the end of the previous chunk forward.
            assert!(content.contains(&format!("marker{}", i - 1)), "{content}");
        }
        let tags: Vec<&str> = chunk["tags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap())
            .collect();
        assert!(tags.contains(&format!("doc:{doc_id}").as_str()), "{tags:?}");
        assert!(tags.contains(&"kind:doc"), "{tags:?}");
        let attrs = chunk["attributes"].to_string();
        for needle in [
            doc_id.as_str(),
            "Handbook",
            "handbook.md",
            "chunk_index",
            "chunk_count",
        ] {
            assert!(attrs.contains(needle), "{needle} in {attrs}");
        }
        assert_eq!(
            chunk["memory_type"].as_str().unwrap().to_lowercase(),
            "semantic"
        );
    }

    let (s, b) = send(
        &app,
        "DELETE",
        &format!("/v1/documents/{doc_id}"),
        &[],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(b["deleted"], 5);
    for id in &ids {
        let (s, _) = send(&app, "GET", &format!("/v1/memories/{id}"), &[], None).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }
    let (s, _) = send(
        &app,
        "DELETE",
        &format!("/v1/documents/{doc_id}"),
        &[],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn documents_reject_bad_chunking_and_empty_content() {
    let (app, _tmp) = app();
    let cases = [
        json!({ "content": "" }),
        json!({ "content": "text", "chunk_chars": 10 }),
        json!({ "content": "text", "chunk_chars": 1_000_000 }),
        json!({ "content": "text", "chunk_chars": 400, "overlap_chars": 300 }),
        json!({ "content": "text", "memory_type": "nonsense" }),
    ];
    for body in cases {
        let (s, b) = send(&app, "POST", "/v1/documents", &[], Some(body.clone())).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{body} -> {b}");
    }
    let (s, _) = send(&app, "DELETE", "/v1/documents/not-a-uuid", &[], None).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn documents_in_a_named_database_stay_there() {
    let (app, _tmp) = app();
    let (doc_id, ids) = load_document(&app, "/v1/db/acme").await;
    assert_eq!(ids.len(), 5);
    let (s, _) = send(&app, "GET", &format!("/v1/memories/{}", ids[0]), &[], None).await;
    assert_eq!(s, StatusCode::NOT_FOUND, "not in the default database");
    let (s, b) = send(
        &app,
        "DELETE",
        &format!("/v1/db/acme/documents/{doc_id}"),
        &[],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{b}");
}

// ---------------------------------------------------------------------------
// Backups
// ---------------------------------------------------------------------------

#[tokio::test]
async fn backup_writes_a_file_with_a_manifest_and_lists_it() {
    let (app, tmp) = app();
    load_document(&app, "/v1").await;
    // A named database lives under the default one's directory; its files
    // must not end up in the default database's backup.
    load_document(&app, "/v1/db/other").await;

    let (s, b) = send(&app, "POST", "/v1/backup", ADMIN, None).await;
    assert_eq!(s, StatusCode::CREATED, "{b}");
    let file = b["file"].as_str().unwrap().to_string();
    assert!(
        file.starts_with("default-") && file.ends_with(".mentebackup"),
        "{file}"
    );
    assert!(b["manifest"]["memory_count"].as_u64().unwrap() > 0, "{b}");
    assert_eq!(b["manifest"]["version"], "mentedb-backup-v1");

    let dir = tmp.path().join("backups").join("default");
    let path = dir.join(&file);
    assert!(path.is_file());
    let staged: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().collect();
    assert_eq!(staged.len(), 1, "staging directory removed");

    let restored = TempDir::new().unwrap();
    BackupManager::restore_backup(&path, restored.path()).unwrap();
    assert!(
        !restored.path().join("dbs").exists(),
        "named databases left out"
    );
    assert!(
        !restored.path().join("backups").exists(),
        "backups left out"
    );

    let (s, b) = send(&app, "GET", "/v1/backups", ADMIN, None).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    let listed = b["backups"].as_array().unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0]["file"], file.as_str());

    // A named database backs up into its own directory.
    let (s, b) = send(&app, "POST", "/v1/db/other/backup", ADMIN, None).await;
    assert_eq!(s, StatusCode::CREATED, "{b}");
    assert!(b["file"].as_str().unwrap().starts_with("other-"));
    assert!(
        tmp.path()
            .join("backups/other")
            .join(b["file"].as_str().unwrap())
            .is_file()
    );
    let (_, b) = send(&app, "GET", "/v1/db/other/backups", ADMIN, None).await;
    assert_eq!(b["backups"].as_array().unwrap().len(), 1);
    let (_, b) = send(&app, "GET", "/v1/backups", ADMIN, None).await;
    assert_eq!(b["backups"].as_array().unwrap().len(), 1, "per database");
}

// ---------------------------------------------------------------------------
// Admin gating
// ---------------------------------------------------------------------------

#[tokio::test]
async fn admin_routes_refuse_non_admins() {
    let (app, tmp) = app();
    let wrong: &[(&str, &str)] = &[("x-api-key", "wrong")];
    for (method, uri) in [
        ("POST", "/v1/backup"),
        ("GET", "/v1/backups"),
        ("GET", "/v1/databases"),
        ("DELETE", "/v1/databases/x"),
    ] {
        let (s, _) = send(&app, method, uri, &[], None).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{method} {uri} without a key");
        let (s, _) = send(&app, method, uri, wrong, None).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{method} {uri} with a wrong key");
    }
    let (s, _) = send(
        &app,
        "POST",
        "/v1/databases",
        &[],
        Some(json!({"name": "x"})),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);

    // A refused request to a named database must not create it.
    let (s, _) = send(&app, "POST", "/v1/db/ghost/backup", &[], None).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert!(!tmp.path().join("dbs/ghost").exists());
}

#[tokio::test]
async fn admin_routes_with_tokens() {
    let (app, _tmp) = app_with(hash_embedder(), true);
    let user = format!("Bearer {}", auth::create_token(JWT_SECRET, AGENT, false, 1));
    let admin = format!("Bearer {}", auth::create_token(JWT_SECRET, AGENT, true, 1));

    let (s, _) = send(&app, "GET", "/v1/databases", &[], None).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED, "no token");
    let (s, _) = send(
        &app,
        "GET",
        "/v1/databases",
        &[("authorization", &user)],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN, "non admin token");
    let (s, b) = send(
        &app,
        "GET",
        "/v1/databases",
        &[("authorization", &admin)],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "admin token: {b}");
    let (s, b) = send(
        &app,
        "GET",
        "/v1/databases",
        &[("authorization", &user), ("x-api-key", ADMIN_KEY)],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "token plus admin key: {b}");
    let (s, _) = send(
        &app,
        "POST",
        "/v1/backup",
        &[("authorization", &user)],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    let (s, b) = send(
        &app,
        "POST",
        "/v1/backup",
        &[("authorization", &admin)],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{b}");

    // Readiness needs no token.
    let (s, _) = send(&app, "GET", "/v1/ready", &[], None).await;
    assert_eq!(s, StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Databases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn databases_create_list_delete() {
    let (app, tmp) = app();
    let (s, b) = send(
        &app,
        "POST",
        "/v1/databases",
        ADMIN,
        Some(json!({"name": "alpha"})),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{b}");
    assert!(tmp.path().join("dbs/alpha/.mentedb-db").exists());

    let (s, _) = send(
        &app,
        "POST",
        "/v1/databases",
        ADMIN,
        Some(json!({"name": "alpha"})),
    )
    .await;
    assert_eq!(s, StatusCode::CONFLICT);
    let (s, _) = send(
        &app,
        "POST",
        "/v1/databases",
        ADMIN,
        Some(json!({"name": "default"})),
    )
    .await;
    assert_eq!(s, StatusCode::CONFLICT);
    let (s, _) = send(
        &app,
        "POST",
        "/v1/databases",
        ADMIN,
        Some(json!({"name": "bad name"})),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    let (s, _) = send(&app, "POST", "/v1/databases", ADMIN, Some(json!({}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);

    let (s, b) = send(&app, "GET", "/v1/databases", ADMIN, None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(b["default"], "default");
    assert_eq!(b["databases"], json!(["alpha"]));

    let (s, b) = send(&app, "DELETE", "/v1/databases/alpha", ADMIN, None).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert!(!tmp.path().join("dbs/alpha").exists());
    let (_, b) = send(&app, "GET", "/v1/databases", ADMIN, None).await;
    assert_eq!(b["databases"], json!([]));
    let (s, _) = send(&app, "DELETE", "/v1/databases/alpha", ADMIN, None).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    let (s, _) = send(&app, "DELETE", "/v1/databases/default", ADMIN, None).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}

// ---------------------------------------------------------------------------
// Readiness
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ready_with_a_working_embedder() {
    let (app, _tmp) = app();
    let (s, b) = send(&app, "GET", "/v1/ready", &[], None).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(b["status"], "ready");
    assert_eq!(b["checks"]["database"], "ok");
    assert_eq!(b["checks"]["embedder"], "ok");
    // Served from the cached embed check the second time.
    let (s, _) = send(&app, "GET", "/v1/ready", &[], None).await;
    assert_eq!(s, StatusCode::OK);
}

#[tokio::test]
async fn ready_without_an_embedder() {
    let (app, _tmp) = app_with(None, false);
    let (s, b) = send(&app, "GET", "/v1/ready", &[], None).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(b["checks"]["embedder"], "not_configured");
}

#[tokio::test]
async fn not_ready_when_the_embedder_fails() {
    let (app, _tmp) = app_with(Some(Arc::new(FailingEmbedder)), false);
    let (s, b) = send(&app, "GET", "/v1/ready", &[], None).await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{b}");
    assert_eq!(b["status"], "not_ready");
    assert_eq!(b["failed"], json!(["embedder"]));
    assert_eq!(b["checks"]["database"], "ok");
    assert!(
        b["checks"]["embedder"]
            .as_str()
            .unwrap()
            .contains("embedder is down"),
        "{b}"
    );
    // Liveness stays up.
    let (s, _) = send(&app, "GET", "/v1/health", &[], None).await;
    assert_eq!(s, StatusCode::OK);
}
