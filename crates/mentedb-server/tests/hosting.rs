//! HTTP tests for the hosting features: memory updates and supersession,
//! paged lists, token scopes, scoped drops and backups, batch search,
//! deadlines, per caller rate limits, recall_for_action, consolidation,
//! recall only turns, hybrid search options, readiness and liveness.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware;
use http_body_util::BodyExt;
use mentedb_embedding::provider::EmbeddingProvider;
use mentedb_server::auth::{self, Scope, TokenOptions, create_token_with};
use mentedb_server::rate_limit::RateLimiter;
use mentedb_server::routes;
use mentedb_server::state::AppState;
use serde_json::{Value, json};
use tempfile::TempDir;
use tower::ServiceExt;

const ADMIN_KEY: &str = "hosting-admin-key";
const SECRET: &str = "hosting-test-secret";
const AGENT_A: &str = "00000000-0000-0000-0000-00000000000a";
const AGENT_B: &str = "00000000-0000-0000-0000-00000000000b";

struct App {
    router: axum::Router,
    tmp: TempDir,
}

fn build(auth_on: bool, limiter: Option<RateLimiter>, timeout: Option<Duration>) -> App {
    let tmp = TempDir::new().unwrap();
    let embedder: Arc<dyn EmbeddingProvider> =
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
        databases,
        verifier: auth_on.then(|| Arc::new(auth::TokenVerifier::shared_secret(SECRET))),
        spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
        jwt_secret: auth_on.then(|| SECRET.to_string()),
        admin_key: Some(ADMIN_KEY.to_string()),
        start_time: Instant::now(),
        extraction_config: None,
        auto_extract: false,
        extraction_tx: None,
        cluster: None,
        turn_trackers: Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
    });
    let mut router = routes::build_router(state.clone());
    if let Some(l) = limiter {
        router = router.layer(l);
    }
    let router = router
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            timeout,
            mentedb_server::hosting::deadline_middleware,
        ));
    App { router, tmp }
}

fn open_app() -> App {
    build(false, None, None)
}

fn token(opts: TokenOptions) -> String {
    create_token_with(SECRET, "", false, 1, opts)
}

fn agent_token(agent: &str) -> String {
    create_token_with(SECRET, agent, false, 1, TokenOptions::default())
}

async fn send(
    app: &App,
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
    let resp = app
        .router
        .clone()
        .oneshot(req.body(body).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

fn bearer(t: &str) -> String {
    format!("Bearer {t}")
}

async fn store(app: &App, prefix: &str, auth: &[(&str, &str)], body: Value) -> String {
    let (s, b) = send(app, "POST", &format!("{prefix}/memories"), auth, Some(body)).await;
    assert_eq!(s, StatusCode::CREATED, "{b}");
    b["id"].as_str().unwrap().to_string()
}

fn memory(agent: &str, content: &str, tags: &[&str]) -> Value {
    json!({
        "agent_id": agent,
        "memory_type": "semantic",
        "content": content,
        "tags": tags,
    })
}

// ---------------------------------------------------------------------------
// Memory JSON, PATCH, supersede
// ---------------------------------------------------------------------------

#[tokio::test]
async fn memory_json_omits_the_embedding_unless_asked() {
    let app = open_app();
    let id = store(&app, "/v1", &[], memory(AGENT_A, "the sky is blue", &[])).await;
    let (_, b) = send(&app, "GET", &format!("/v1/memories/{id}"), &[], None).await;
    assert!(b.get("embedding").is_none(), "{b}");
    assert!(b.get("valid_from").is_some());
    assert!(b.get("valid_until").is_some());
    assert!(b.get("user_id").is_some());
    let (_, b) = send(
        &app,
        "GET",
        &format!("/v1/memories/{id}?fields=embedding"),
        &[],
        None,
    )
    .await;
    assert_eq!(b["embedding"].as_array().unwrap().len(), 384);
}

#[tokio::test]
async fn patch_updates_in_place_and_keeps_the_id() {
    let app = open_app();
    let mut m = memory(AGENT_A, "deploys run at noon", &["ops"]);
    m["attributes"] = json!({ "env": "prod", "owner": "team-a" });
    let id = store(&app, "/v1", &[], m).await;
    let (s, b) = send(
        &app,
        "PATCH",
        &format!("/v1/memories/{id}"),
        &[],
        Some(json!({
            "attributes": { "env": "staging", "ticket": 42 },
            "tags": ["ops", "schedule"],
            "valid_until": 4_102_444_800_000_000u64,
        })),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(b["id"], id.as_str());
    assert_eq!(b["attributes"]["env"], "staging");
    assert_eq!(b["attributes"]["owner"], "team-a", "merged, not replaced");
    assert_eq!(b["attributes"]["ticket"], 42);
    assert_eq!(b["tags"], json!(["ops", "schedule"]));
    assert_eq!(b["valid_until"], 4_102_444_800_000_000u64);

    // The tag index follows the change.
    let (_, b) = send(&app, "GET", "/v1/memories?tag=schedule", &[], None).await;
    assert_eq!(b["memories"].as_array().unwrap().len(), 1);

    // Clearing valid_until with null.
    let (_, b) = send(
        &app,
        "PATCH",
        &format!("/v1/memories/{id}"),
        &[],
        Some(json!({ "valid_until": null })),
    )
    .await;
    assert!(b["valid_until"].is_null());

    let (s, _) = send(
        &app,
        "PATCH",
        &format!("/v1/memories/{id}"),
        &[],
        Some(json!({})),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "empty patch");
    let missing = uuid::Uuid::new_v4();
    let (s, _) = send(
        &app,
        "PATCH",
        &format!("/v1/memories/{missing}"),
        &[],
        Some(json!({ "tags": [] })),
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn supersede_closes_the_old_memory_and_links_the_new() {
    let app = open_app();
    let old = store(
        &app,
        "/v1",
        &[],
        memory(AGENT_A, "phone number is 1234", &["contact"]),
    )
    .await;
    let (s, b) = send(
        &app,
        "POST",
        &format!("/v1/memories/{old}/supersede"),
        &[],
        Some(json!({ "content": "phone number is 4321", "tags": ["contact"] })),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{b}");
    let new = b["id"].as_str().unwrap().to_string();
    assert_ne!(new, old);
    assert_eq!(b["supersedes"], old.as_str());

    let (_, o) = send(&app, "GET", &format!("/v1/memories/{old}"), &[], None).await;
    assert!(o["valid_until"].is_u64(), "old memory is closed: {o}");
    let (_, n) = send(&app, "GET", &format!("/v1/memories/{new}"), &[], None).await;
    assert_eq!(n["agent_id"], AGENT_A, "owner carried over");
    assert_eq!(n["memory_type"], "semantic", "type defaults to the old one");

    let (_, h) = send(
        &app,
        "GET",
        &format!("/v1/memories/{new}/history"),
        &[],
        None,
    )
    .await;
    let history = h["history"]
        .as_array()
        .or_else(|| h["memories"].as_array())
        .unwrap();
    assert!(history.iter().any(|m| m["id"] == old.as_str()), "{h}");

    // Current lists hide the closed memory; include_inactive shows it.
    let (_, l) = send(&app, "GET", "/v1/memories?tag=contact", &[], None).await;
    let ids: Vec<&str> = l["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec![new.as_str()]);
    let (_, l) = send(
        &app,
        "GET",
        "/v1/memories?tag=contact&include_inactive=1",
        &[],
        None,
    )
    .await;
    assert_eq!(l["memories"].as_array().unwrap().len(), 2);

    let (s, _) = send(
        &app,
        "POST",
        &format!("/v1/memories/{old}/supersede"),
        &[],
        Some(json!({})),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "content is required");
}

// ---------------------------------------------------------------------------
// Lists and token scopes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_is_paged_stable_and_token_scoped() {
    let app = build(true, None, None);
    let ta = agent_token(AGENT_A);
    let tb = agent_token(AGENT_B);
    let auth_a = [("authorization", bearer(&ta))];
    let auth_a: Vec<(&str, &str)> = auth_a.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let auth_b = [("authorization", bearer(&tb))];
    let auth_b: Vec<(&str, &str)> = auth_b.iter().map(|(k, v)| (*k, v.as_str())).collect();
    for i in 0..5 {
        store(
            &app,
            "/v1",
            &auth_a,
            memory(AGENT_A, &format!("a note {i}"), &["n"]),
        )
        .await;
    }
    store(&app, "/v1", &auth_b, memory(AGENT_B, "b note", &["n"])).await;

    let mut seen = Vec::new();
    let mut offset = 0;
    loop {
        let (s, b) = send(
            &app,
            "GET",
            &format!("/v1/memories?limit=2&offset={offset}"),
            &auth_a,
            None,
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{b}");
        for m in b["memories"].as_array().unwrap() {
            assert_eq!(m["agent_id"], AGENT_A, "only the token's agent");
            seen.push(m["id"].as_str().unwrap().to_string());
        }
        match b["next_offset"].as_u64() {
            Some(n) => offset = n as usize,
            None => break,
        }
    }
    assert_eq!(seen.len(), 5);
    let mut dedup = seen.clone();
    dedup.sort();
    dedup.dedup();
    assert_eq!(dedup.len(), 5, "no repeats across pages");

    let (_, b) = send(&app, "GET", "/v1/memories?tag=n", &auth_b, None).await;
    assert_eq!(b["memories"].as_array().unwrap().len(), 1);
    let (s, _) = send(
        &app,
        "GET",
        "/v1/memories?include_inactive=maybe",
        &auth_a,
        None,
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn read_scope_cannot_write_and_dbs_bind_databases() {
    let app = build(true, None, None);
    let read = token(TokenOptions {
        scope: Some(Scope::Read),
        dbs: Some(vec!["tenant.alice.*".into()]),
        ..Default::default()
    });
    let write = token(TokenOptions {
        dbs: Some(vec!["tenant.alice.*".into()]),
        ..Default::default()
    });
    let r = [("authorization", bearer(&read))];
    let r: Vec<(&str, &str)> = r.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let w = [("authorization", bearer(&write))];
    let w: Vec<(&str, &str)> = w.iter().map(|(k, v)| (*k, v.as_str())).collect();

    let (s, _) = send(
        &app,
        "POST",
        "/v1/db/tenant.alice.notes/memories",
        &r,
        Some(memory(AGENT_A, "x", &[])),
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN, "read scope cannot store");
    let id = store(
        &app,
        "/v1/db/tenant.alice.notes",
        &w,
        memory(AGENT_A, "x", &[]),
    )
    .await;
    let (s, _) = send(
        &app,
        "GET",
        &format!("/v1/db/tenant.alice.notes/memories/{id}"),
        &r,
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "read scope can read");
    let (s, _) = send(
        &app,
        "PATCH",
        &format!("/v1/db/tenant.alice.notes/memories/{id}"),
        &r,
        Some(json!({ "tags": ["t"] })),
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN, "read scope cannot patch");
    let (s, _) = send(&app, "GET", "/v1/db/tenant.bob.notes/memories", &w, None).await;
    assert_eq!(s, StatusCode::FORBIDDEN, "another tenant's database");
}

#[tokio::test]
async fn dotted_database_names_work_and_separators_do_not() {
    let app = open_app();
    let id = store(
        &app,
        "/v1/db/tenant.alice.notes",
        &[],
        memory(AGENT_A, "dotted", &[]),
    )
    .await;
    let (s, _) = send(
        &app,
        "GET",
        &format!("/v1/db/tenant.alice.notes/memories/{id}"),
        &[],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert!(app.tmp.path().join("dbs/tenant.alice.notes").is_dir());
    for bad in ["..", "%2E%2E", "a%2Fb", ".hidden"] {
        let (s, _) = send(&app, "GET", &format!("/v1/db/{bad}/memories"), &[], None).await;
        assert!(
            s == StatusCode::BAD_REQUEST || s == StatusCode::NOT_FOUND,
            "{bad}: {s}"
        );
    }
    assert!(!app.tmp.path().join("hidden").exists());
}

// ---------------------------------------------------------------------------
// Scoped drop and backup
// ---------------------------------------------------------------------------

#[tokio::test]
async fn maintenance_tokens_drop_and_back_up_their_own_databases() {
    let app = build(true, None, None);
    let maint = token(TokenOptions {
        scope: Some(Scope::Maintenance),
        dbs: Some(vec!["tenant.alice.*".into()]),
        ..Default::default()
    });
    let write = token(TokenOptions {
        dbs: Some(vec!["tenant.alice.*".into()]),
        ..Default::default()
    });
    let m = [("authorization", bearer(&maint))];
    let m: Vec<(&str, &str)> = m.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let w = [("authorization", bearer(&write))];
    let w: Vec<(&str, &str)> = w.iter().map(|(k, v)| (*k, v.as_str())).collect();

    store(
        &app,
        "/v1/db/tenant.alice.notes",
        &w,
        memory(AGENT_A, "keep me", &[]),
    )
    .await;
    let admin = bearer(&create_token_with(
        SECRET,
        "",
        true,
        1,
        TokenOptions::default(),
    ));
    store(
        &app,
        "/v1/db/tenant.bob.notes",
        &[("authorization", admin.as_str())],
        memory(AGENT_B, "bob", &[]),
    )
    .await;

    // Backup with a chosen relative name.
    let (s, b) = send(
        &app,
        "POST",
        "/v1/db/tenant.alice.notes/backup",
        &m,
        Some(json!({ "name": "nightly/first" })),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{b}");
    assert_eq!(b["file"], "nightly/first.mentebackup");
    assert!(
        app.tmp
            .path()
            .join("backups/tenant.alice.notes/nightly/first.mentebackup")
            .is_file()
    );
    for bad in ["../escape", "/etc/x", "a//b", "a/../b", "sp ace"] {
        let (s, b) = send(
            &app,
            "POST",
            "/v1/db/tenant.alice.notes/backup",
            &m,
            Some(json!({ "name": bad })),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{bad}: {b}");
    }
    let (s, _) = send(&app, "POST", "/v1/db/tenant.alice.notes/backup", &w, None).await;
    assert_eq!(s, StatusCode::FORBIDDEN, "write scope cannot back up");
    let (s, _) = send(&app, "POST", "/v1/db/tenant.bob.notes/backup", &m, None).await;
    assert_eq!(s, StatusCode::FORBIDDEN, "not this token's database");

    // Drop.
    let (s, _) = send(&app, "DELETE", "/v1/db/tenant.alice.notes", &w, None).await;
    assert_eq!(s, StatusCode::FORBIDDEN, "write scope cannot drop");
    let (s, _) = send(&app, "DELETE", "/v1/db/tenant.bob.notes", &m, None).await;
    assert_eq!(s, StatusCode::FORBIDDEN, "not this token's database");
    let (s, b) = send(&app, "DELETE", "/v1/db/tenant.alice.notes", &m, None).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert!(!app.tmp.path().join("dbs/tenant.alice.notes").exists());
    let (s, _) = send(&app, "DELETE", "/v1/db/default", &m, None).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    let (s, _) = send(&app, "DELETE", "/v1/db/tenant.bob.notes", &[], None).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED, "nothing presented");
    let (s, _) = send(
        &app,
        "DELETE",
        "/v1/db/tenant.bob.notes",
        &[("x-api-key", ADMIN_KEY)],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "the admin key drops any database");
}

// ---------------------------------------------------------------------------
// Batch search
// ---------------------------------------------------------------------------

#[tokio::test]
async fn batch_search_spans_databases_within_the_token() {
    let app = build(true, None, None);
    let t = token(TokenOptions {
        dbs: Some(vec!["tenant.alice.*".into()]),
        ..Default::default()
    });
    let h = [("authorization", bearer(&t))];
    let h: Vec<(&str, &str)> = h.iter().map(|(k, v)| (*k, v.as_str())).collect();
    store(
        &app,
        "/v1/db/tenant.alice.a",
        &h,
        memory(AGENT_A, "the garden has tomatoes", &[]),
    )
    .await;
    store(
        &app,
        "/v1/db/tenant.alice.b",
        &h,
        memory(AGENT_A, "the garden has roses", &[]),
    )
    .await;

    let (s, b) = send(
        &app,
        "POST",
        "/v1/batch/search",
        &h,
        Some(json!({
            "databases": ["tenant.alice.a", "tenant.alice.b", "tenant.alice.missing"],
            "query": "garden",
            "k": 5,
            "context": true,
        })),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{b}");
    let results = b["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    for r in results {
        assert!(!r["results"].as_array().unwrap().is_empty(), "{r}");
        assert!(r["context"].is_array());
    }
    assert!(b["errors"]["tenant.alice.missing"].is_string());
    assert!(
        !app.tmp.path().join("dbs/tenant.alice.missing").exists(),
        "a batch read never creates a database"
    );

    let (s, _) = send(
        &app,
        "POST",
        "/v1/batch/search",
        &h,
        Some(json!({ "databases": ["tenant.alice.a", "tenant.bob.x"], "query": "garden" })),
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN, "every database must be in scope");
    let (s, _) = send(
        &app,
        "POST",
        "/v1/batch/search",
        &h,
        Some(json!({ "databases": [] })),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}

// ---------------------------------------------------------------------------
// Deadlines and rate limits
// ---------------------------------------------------------------------------

#[tokio::test]
async fn deadline_header_bounds_the_request() {
    let app = open_app();
    let (s, _) = send(
        &app,
        "GET",
        "/v1/health",
        &[("x-mentedb-deadline-ms", "5000")],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let (s, b) = send(
        &app,
        "GET",
        "/v1/memories",
        &[("x-mentedb-deadline-ms", "0")],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::GATEWAY_TIMEOUT, "{b}");
    let (s, _) = send(
        &app,
        "GET",
        "/v1/health",
        &[("x-mentedb-deadline-ms", "soon")],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn server_timeout_applies_without_a_header() {
    let app = build(false, None, Some(Duration::from_millis(0)));
    let (s, _) = send(&app, "GET", "/v1/memories", &[], None).await;
    assert_eq!(s, StatusCode::GATEWAY_TIMEOUT);
}

#[tokio::test]
async fn rate_limit_is_per_token_subject() {
    let app = build(true, Some(RateLimiter::new(2, 1)), None);
    let alice = token(TokenOptions {
        sub: Some("alice".into()),
        ..Default::default()
    });
    let bob = token(TokenOptions {
        sub: Some("bob".into()),
        ..Default::default()
    });
    let a = bearer(&alice);
    let b = bearer(&bob);
    for _ in 0..2 {
        let (s, _) = send(
            &app,
            "GET",
            "/v1/stats",
            &[("authorization", a.as_str())],
            None,
        )
        .await;
        assert_eq!(s, StatusCode::OK);
    }
    let (s, _) = send(
        &app,
        "GET",
        "/v1/stats",
        &[("authorization", a.as_str())],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    let (s, _) = send(
        &app,
        "GET",
        "/v1/stats",
        &[("authorization", b.as_str())],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "another subject has its own budget");
}

// ---------------------------------------------------------------------------
// recall_for_action, consolidation, recall only turns, hybrid options
// ---------------------------------------------------------------------------

#[tokio::test]
async fn recall_for_action_finds_the_rule_for_an_action() {
    let app = open_app();
    let mut rule = memory(AGENT_A, "always run cargo fmt before committing", &[]);
    rule["memory_type"] = json!("procedural");
    let id = store(&app, "/v1", &[], rule).await;
    store(
        &app,
        "/v1",
        &[],
        memory(AGENT_A, "the office plant needs water", &[]),
    )
    .await;
    let (s, b) = send(
        &app,
        "POST",
        "/v1/recall_for_action",
        &[],
        Some(json!({ "action": "git commit after cargo fmt", "k": 3 })),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{b}");
    let ids: Vec<&str> = b["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&id.as_str()), "{b}");
    let (s, _) = send(&app, "POST", "/v1/recall_for_action", &[], Some(json!({}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn consolidation_candidates_read_and_consolidate_merges_exactly_the_ids() {
    let app = open_app();
    let a = store(&app, "/v1", &[], memory(AGENT_A, "standup is at nine", &[])).await;
    let b = store(
        &app,
        "/v1",
        &[],
        memory(AGENT_A, "the daily meeting starts at nine", &[]),
    )
    .await;
    let c = store(&app, "/v1", &[], memory(AGENT_A, "lunch is at noon", &[])).await;
    let count = |body: &Value| body["memory_count"].as_u64().unwrap();
    let (_, before) = send(&app, "GET", "/v1/stats", &[], None).await;

    let (s, body) = send(
        &app,
        "GET",
        "/v1/consolidation_candidates?min_cluster=2&similarity=0.5",
        &[],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    assert!(body["clusters"].is_array());
    let (_, after) = send(&app, "GET", "/v1/stats", &[], None).await;
    assert_eq!(
        count(&before),
        count(&after),
        "listing candidates merges nothing"
    );
    let (s, _) = send(
        &app,
        "GET",
        "/v1/consolidation_candidates?similarity=2",
        &[],
        None,
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);

    let (s, body) = send(
        &app,
        "POST",
        "/v1/consolidate",
        &[],
        Some(json!({ "ids": [a, b] })),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{body}");
    let merged = body["id"].as_str().unwrap();
    let (_, m) = send(&app, "GET", &format!("/v1/memories/{merged}"), &[], None).await;
    assert_eq!(m["id"], merged);
    let (_, untouched) = send(&app, "GET", &format!("/v1/memories/{c}"), &[], None).await;
    assert!(untouched["valid_until"].is_null(), "not named, not merged");
    let (s, _) = send(
        &app,
        "POST",
        "/v1/consolidate",
        &[],
        Some(json!({ "ids": [c] })),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "one id is not a merge");
}

#[tokio::test]
async fn process_turn_without_store_writes_nothing() {
    let app = open_app();
    store(
        &app,
        "/v1",
        &[],
        memory(AGENT_A, "the user likes green tea", &[]),
    )
    .await;
    let (_, before) = send(&app, "GET", "/v1/stats", &[], None).await;
    let (s, b) = send(
        &app,
        "POST",
        "/v1/process_turn",
        &[],
        Some(json!({ "user_message": "what tea do I like?", "store": false })),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(b["stored"], false);
    assert!(b["stored_ids"].as_array().unwrap().is_empty());
    let (_, after) = send(&app, "GET", "/v1/stats", &[], None).await;
    assert_eq!(before["memory_count"], after["memory_count"]);

    let (_, b) = send(
        &app,
        "POST",
        "/v1/process_turn",
        &[],
        Some(json!({ "user_message": "what tea do I like?" })),
    )
    .await;
    assert_eq!(b["stored"], true);
    let (_, stored) = send(&app, "GET", "/v1/stats", &[], None).await;
    assert!(stored["memory_count"].as_u64() > after["memory_count"].as_u64());
    let (s, _) = send(
        &app,
        "POST",
        "/v1/process_turn",
        &[],
        Some(json!({ "user_message": "x", "store": "no" })),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn hybrid_search_accepts_ranking_options() {
    let app = open_app();
    for c in [
        "red apples",
        "green apples",
        "apple pie recipe",
        "car engines",
    ] {
        store(&app, "/v1", &[], memory(AGENT_A, c, &[])).await;
    }
    let (s, b) = send(
        &app,
        "POST",
        "/v1/search/hybrid",
        &[],
        Some(json!({
            "query": "apples",
            "k": 3,
            "mmr_lambda": 0.5,
            "rerank": true,
            "relevance_first": true,
        })),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert!(!b["results"].as_array().unwrap().is_empty());
    let (s, _) = send(
        &app,
        "POST",
        "/v1/search/hybrid",
        &[],
        Some(json!({ "query": "apples", "mmr_lambda": 3 })),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
}

// ---------------------------------------------------------------------------
// Readiness and liveness
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ready_checks_the_mount_and_a_database_lock_and_live_checks_nothing() {
    let app = build(true, None, None);
    let (s, b) = send(&app, "GET", "/v1/ready", &[], None).await;
    assert_eq!(s, StatusCode::OK, "{b}");
    assert_eq!(b["checks"]["data_dir"], "ok");
    assert_eq!(b["checks"]["database_lock"], "ok");
    let (_, b) = send(&app, "GET", "/v1/ready?embedder=skip", &[], None).await;
    assert_eq!(b["checks"]["embedder"], "skipped");
    let (s, b) = send(&app, "GET", "/v1/live", &[], None).await;
    assert_eq!(s, StatusCode::OK, "no token needed");
    assert_eq!(b["status"], "alive");
}
