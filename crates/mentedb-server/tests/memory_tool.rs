//! Integration tests for `POST /v1/memory_tool`: every command over HTTP on
//! the default and a named database, per agent scoping, token scopes, and
//! path traversal refusal.

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
const JWT_SECRET: &str = "memory-tool-test-secret";

fn build_app(with_auth: bool) -> (axum::Router, TempDir) {
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
        verifier: with_auth.then(|| Arc::new(auth::TokenVerifier::shared_secret(JWT_SECRET))),
        spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
        jwt_secret: with_auth.then(|| JWT_SECRET.to_string()),
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

async fn post(
    app: &axum::Router,
    uri: &str,
    body: Value,
    token: Option<&str>,
) -> (StatusCode, Value) {
    let mut req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(t) = token {
        req = req.header("authorization", format!("Bearer {t}"));
    }
    let resp = app
        .clone()
        .oneshot(req.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// Send one tool call, expect HTTP 200, return `(content, is_error)`.
async fn tool(app: &axum::Router, uri: &str, body: Value, token: Option<&str>) -> (String, bool) {
    let (status, resp) = post(app, uri, body.clone(), token).await;
    assert_eq!(status, StatusCode::OK, "{body} answered {resp}");
    assert_eq!(resp.as_object().unwrap().len(), 2, "exact contract: {resp}");
    (
        resp["content"].as_str().unwrap().to_string(),
        resp["is_error"].as_bool().unwrap(),
    )
}

async fn ok(app: &axum::Router, uri: &str, body: Value, token: Option<&str>) -> String {
    let (content, is_error) = tool(app, uri, body.clone(), token).await;
    assert!(!is_error, "{body} failed: {content}");
    content
}

async fn fails(app: &axum::Router, uri: &str, body: Value, token: Option<&str>) -> String {
    let (content, is_error) = tool(app, uri, body.clone(), token).await;
    assert!(is_error, "{body} should fail: {content}");
    content
}

async fn every_command(app: &axum::Router, uri: &str) {
    let root = ok(
        app,
        uri,
        json!({"command": "view", "path": "/memories"}),
        None,
    )
    .await;
    assert!(root.ends_with("\n0B\t/memories"), "{root}");
    assert_eq!(
        ok(
            app,
            uri,
            json!({"command": "create", "path": "/memories/notes.md", "file_text": "alpha\nbeta\n"}),
            None
        )
        .await,
        "File created successfully at: /memories/notes.md"
    );
    assert_eq!(
        ok(
            app,
            uri,
            json!({"command": "view", "path": "/memories/notes.md"}),
            None
        )
        .await,
        "Here's the content of /memories/notes.md with line numbers:\n     1\talpha\n     2\tbeta"
    );
    assert_eq!(
        ok(
            app,
            uri,
            json!({"command": "view", "path": "/memories/notes.md", "view_range": [2, 2]}),
            None
        )
        .await,
        "Here's the content of /memories/notes.md with line numbers:\n     2\tbeta"
    );
    assert_eq!(
        ok(
            app,
            uri,
            json!({"command": "str_replace", "path": "/memories/notes.md", "old_str": "beta", "new_str": "gamma"}),
            None
        )
        .await,
        "The memory file has been edited."
    );
    assert_eq!(
        ok(
            app,
            uri,
            json!({"command": "insert", "path": "/memories/notes.md", "insert_line": 1, "insert_text": "between"}),
            None
        )
        .await,
        "The file /memories/notes.md has been edited."
    );
    assert_eq!(
        ok(
            app,
            uri,
            json!({"command": "rename", "old_path": "/memories/notes.md", "new_path": "/memories/archive/notes.md"}),
            None
        )
        .await,
        "Successfully renamed /memories/notes.md to /memories/archive/notes.md"
    );
    assert_eq!(
        ok(
            app,
            uri,
            json!({"command": "view", "path": "/memories/archive/notes.md"}),
            None
        )
        .await,
        "Here's the content of /memories/archive/notes.md with line numbers:\n     1\talpha\n     2\tbetween\n     3\tgamma"
    );
    let listing = ok(
        app,
        uri,
        json!({"command": "view", "path": "/memories"}),
        None,
    )
    .await;
    assert!(listing.contains("\t/memories/archive\n"), "{listing}");
    assert!(
        listing.ends_with("\t/memories/archive/notes.md"),
        "{listing}"
    );
    assert!(
        fails(
            app,
            uri,
            json!({"command": "view", "path": "/memories/notes.md"}),
            None
        )
        .await
        .contains("does not exist")
    );
    assert_eq!(
        ok(
            app,
            uri,
            json!({"command": "delete", "path": "/memories/archive"}),
            None
        )
        .await,
        "Successfully deleted /memories/archive"
    );
    let root = ok(
        app,
        uri,
        json!({"command": "view", "path": "/memories"}),
        None,
    )
    .await;
    assert!(root.ends_with("\n0B\t/memories"), "{root}");
    assert!(
        fails(app, uri, json!({"command": "bogus"}), None)
            .await
            .contains("unknown command")
    );
}

#[tokio::test]
async fn every_command_on_the_default_and_a_named_database() {
    let (app, _tmp) = build_app(false);
    every_command(&app, "/v1/memory_tool").await;
    every_command(&app, "/v1/db/team/memory_tool").await;

    // The two databases hold separate trees.
    ok(
        &app,
        "/v1/db/team/memory_tool",
        json!({"command": "create", "path": "/memories/team.md", "file_text": "x"}),
        None,
    )
    .await;
    fails(
        &app,
        "/v1/memory_tool",
        json!({"command": "view", "path": "/memories/team.md"}),
        None,
    )
    .await;
}

#[tokio::test]
async fn traversal_is_refused_and_writes_nothing() {
    let (app, _tmp) = build_app(false);
    for path in [
        "/memories/../secrets.env",
        "/memories/a/../../x",
        "/memories/%2e%2e/x",
        "/memories/%2E%2E%2Fx",
        "/memories/%252e%252e/x",
        "/memories/..%5cx",
        "/memories/..\\x",
        "/memories//x",
        "/etc/passwd",
        "memories/x",
        "/memoriesx",
    ] {
        let msg = fails(
            &app,
            "/v1/memory_tool",
            json!({"command": "create", "path": path, "file_text": "pwned"}),
            None,
        )
        .await;
        assert!(msg.starts_with("Error: Invalid path"), "{path}: {msg}");
        fails(
            &app,
            "/v1/memory_tool",
            json!({"command": "view", "path": path}),
            None,
        )
        .await;
    }
    ok(
        &app,
        "/v1/memory_tool",
        json!({"command": "create", "path": "/memories/a.md", "file_text": "a"}),
        None,
    )
    .await;
    let msg = fails(
        &app,
        "/v1/memory_tool",
        json!({"command": "rename", "old_path": "/memories/a.md", "new_path": "/memories/../a.md"}),
        None,
    )
    .await;
    assert!(msg.starts_with("Error: Invalid path"), "{msg}");
    let root = ok(
        &app,
        "/v1/memory_tool",
        json!({"command": "view", "path": "/memories"}),
        None,
    )
    .await;
    assert!(root.ends_with("\t/memories/a.md"), "{root}");
    assert_eq!(root.matches("\n").count(), 2, "{root}");
}

#[tokio::test]
async fn one_agent_never_sees_another_agents_files() {
    let (app, _tmp) = build_app(true);
    let alice = auth::create_token(JWT_SECRET, AGENT, false, 1);
    let bob = auth::create_token(JWT_SECRET, OTHER_AGENT, false, 1);
    let uri = "/v1/memory_tool";

    ok(
        &app,
        uri,
        json!({"command": "create", "path": "/memories/secret.md", "file_text": "alice only"}),
        Some(&alice),
    )
    .await;

    // Bob sees an empty tree and cannot reach Alice's file by any command.
    let root = ok(
        &app,
        uri,
        json!({"command": "view", "path": "/memories"}),
        Some(&bob),
    )
    .await;
    assert!(root.ends_with("\n0B\t/memories"), "{root}");
    for body in [
        json!({"command": "view", "path": "/memories/secret.md"}),
        json!({"command": "str_replace", "path": "/memories/secret.md", "old_str": "alice", "new_str": "bob"}),
        json!({"command": "insert", "path": "/memories/secret.md", "insert_line": 0, "insert_text": "bob"}),
        json!({"command": "delete", "path": "/memories/secret.md"}),
        json!({"command": "rename", "old_path": "/memories/secret.md", "new_path": "/memories/x.md"}),
    ] {
        fails(&app, uri, body, Some(&bob)).await;
    }

    // Naming Alice in the body with Bob's token is refused outright.
    let (status, _) = post(
        &app,
        uri,
        json!({"command": "view", "path": "/memories/secret.md", "agent_id": AGENT}),
        Some(&bob),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    // Bob writing the same path gets his own file.
    ok(
        &app,
        uri,
        json!({"command": "create", "path": "/memories/secret.md", "file_text": "bob's"}),
        Some(&bob),
    )
    .await;
    assert!(
        ok(
            &app,
            uri,
            json!({"command": "view", "path": "/memories/secret.md", "agent_id": AGENT}),
            Some(&alice),
        )
        .await
        .ends_with("\talice only")
    );

    // A user_id splits one agent's files further.
    let user = "00000000-0000-0000-0000-0000000000aa";
    fails(
        &app,
        uri,
        json!({"command": "view", "path": "/memories/secret.md", "user_id": user}),
        Some(&alice),
    )
    .await;
}

#[tokio::test]
async fn body_agent_id_scopes_files_without_auth() {
    let (app, _tmp) = build_app(false);
    let uri = "/v1/memory_tool";
    ok(
        &app,
        uri,
        json!({"command": "create", "path": "/memories/a.md", "file_text": "a", "agent_id": AGENT}),
        None,
    )
    .await;
    fails(
        &app,
        uri,
        json!({"command": "view", "path": "/memories/a.md", "agent_id": OTHER_AGENT}),
        None,
    )
    .await;
    fails(
        &app,
        uri,
        json!({"command": "view", "path": "/memories/a.md"}),
        None,
    )
    .await;
    ok(
        &app,
        uri,
        json!({"command": "view", "path": "/memories/a.md", "agent_id": AGENT}),
        None,
    )
    .await;
    let (status, _) = post(
        &app,
        uri,
        json!({"command": "view", "path": "/memories", "agent_id": "not-a-uuid"}),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn token_scopes_gate_the_endpoint() {
    let (app, _tmp) = build_app(true);
    let uri = "/v1/db/team/memory_tool";
    let writer = auth::create_token(JWT_SECRET, AGENT, false, 1);
    let read_only = auth::create_scoped_token(JWT_SECRET, AGENT, false, 1, None, true);
    let elsewhere = auth::create_scoped_token(
        JWT_SECRET,
        AGENT,
        false,
        1,
        Some(vec!["elsewhere".to_string()]),
        false,
    );

    let create = json!({"command": "create", "path": "/memories/a.md", "file_text": "a"});
    let view = json!({"command": "view", "path": "/memories/a.md"});

    // No token at all.
    let (status, _) = post(&app, uri, view.clone(), None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    ok(&app, uri, create.clone(), Some(&writer)).await;

    // A read only token can view but not change anything.
    ok(&app, uri, view.clone(), Some(&read_only)).await;
    for body in [
        create.clone(),
        json!({"command": "delete", "path": "/memories/a.md"}),
        json!({"command": "rename", "old_path": "/memories/a.md", "new_path": "/memories/b.md"}),
    ] {
        let (status, _) = post(&app, uri, body, Some(&read_only)).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    // A token scoped to another database reaches nothing here.
    let (status, _) = post(&app, uri, view.clone(), Some(&elsewhere)).await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    // A malformed body is a bad request, not a tool error.
    let (status, _) = post(&app, uri, json!({"path": "/memories"}), Some(&writer)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn concurrent_creates_of_one_path_store_one_file() {
    let (app, _tmp) = build_app(false);
    let mut tasks = Vec::new();
    for i in 0..16 {
        let app = app.clone();
        tasks.push(tokio::spawn(async move {
            ok(
                &app,
                "/v1/memory_tool",
                json!({"command": "create", "path": "/memories/race.md", "file_text": format!("writer {i}")}),
                None,
            )
            .await
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    let root = ok(
        &app,
        "/v1/memory_tool",
        json!({"command": "view", "path": "/memories"}),
        None,
    )
    .await;
    assert_eq!(root.matches("/memories/race.md").count(), 1, "{root}");
    let req = Request::builder()
        .uri("/v1/stats")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let stats: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(stats["memory_count"], 1, "{stats}");
}
