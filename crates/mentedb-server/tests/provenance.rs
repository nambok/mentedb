//! Provenance over REST: written on store, returned on every read, filtered
//! on list and search, kept on re-store and patch, and erase by session.

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
const JWT_SECRET: &str = "provenance-test-secret";

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

async fn store(app: &axum::Router, body: Value, token: &str) -> String {
    let (status, resp) = send(app, "POST", "/v1/memories", Some(body), token).await;
    assert_eq!(status, StatusCode::CREATED, "{resp}");
    resp["id"].as_str().unwrap().to_string()
}

async fn get(app: &axum::Router, id: &str, token: &str) -> Value {
    let (status, body) = send(app, "GET", &format!("/v1/memories/{id}"), None, token).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body
}

fn ids(list: &Value, field: &str) -> std::collections::BTreeSet<String> {
    list[field]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn store_returns_and_filters_provenance() {
    let (app, _tmp) = build_app();
    let token = auth::create_token(JWT_SECRET, AGENT, false, 1);

    let chat_id = store(
        &app,
        json!({
            "agent_id": AGENT,
            "memory_type": "semantic",
            "content": "the user is planning a trip to Kyoto in April",
            "provenance": { "session_id": "s1", "source": "chat", "source_ref": "msg-42" },
        }),
        &token,
    )
    .await;
    // The top level session_id is a shorthand; the source defaults to api.
    let api_id = store(
        &app,
        json!({
            "agent_id": AGENT,
            "memory_type": "semantic",
            "content": "the user keeps a cat called Miso at home",
            "session_id": "s1",
        }),
        &token,
    )
    .await;
    let other_id = store(
        &app,
        json!({
            "agent_id": AGENT,
            "memory_type": "semantic",
            "content": "the user plays the cello every weekend",
            "session_id": "s2",
        }),
        &token,
    )
    .await;

    let m = get(&app, &chat_id, &token).await;
    assert_eq!(m["provenance"]["session_id"], "s1");
    assert_eq!(m["provenance"]["source"], "chat");
    assert_eq!(m["provenance"]["source_ref"], "msg-42");
    assert_eq!(m["provenance"]["written_by"], AGENT);
    assert!(m["provenance"]["written_at"].as_u64().unwrap() > 0);
    assert_eq!(m["provenance"]["derived_from"], json!([]));
    let m = get(&app, &api_id, &token).await;
    assert_eq!(m["provenance"]["source"], "api");
    assert_eq!(m["provenance"]["session_id"], "s1");

    // List by session, by source, by both.
    let (status, list) = send(&app, "GET", "/v1/memories?session_id=s1", None, &token).await;
    assert_eq!(status, StatusCode::OK, "{list}");
    assert_eq!(
        ids(&list, "memories"),
        [chat_id.clone(), api_id.clone()].into()
    );
    let (_, list) = send(&app, "GET", "/v1/memories?source=chat", None, &token).await;
    assert_eq!(ids(&list, "memories"), [chat_id.clone()].into());
    let (_, list) = send(
        &app,
        "GET",
        "/v1/memories?session_id=s1&source=api",
        None,
        &token,
    )
    .await;
    assert_eq!(ids(&list, "memories"), [api_id.clone()].into());

    // Search filtered to a session: the s2 memory never comes back, even
    // for a query that names it.
    let (status, res) = send(
        &app,
        "POST",
        "/v1/search/hybrid",
        Some(json!({ "query": "cello every weekend", "session_id": "s1", "k": 10 })),
        &token,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{res}");
    let found = ids(&res, "results");
    assert!(!found.contains(&other_id), "{res}");
    assert!(found.is_subset(&[chat_id.clone(), api_id.clone()].into()));
    assert!(res["results"][0]["provenance"]["session_id"] == "s1");

    // Bad provenance types are refused.
    let (status, _) = send(
        &app,
        "POST",
        "/v1/memories",
        Some(json!({
            "agent_id": AGENT, "memory_type": "semantic", "content": "x",
            "provenance": { "session_id": 7 },
        })),
        &token,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = send(
        &app,
        "POST",
        "/v1/memories",
        Some(json!({
            "agent_id": AGENT, "memory_type": "semantic", "content": "x",
            "session_id": "a", "provenance": { "session_id": "b" },
        })),
        &token,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn patch_and_batch_keep_provenance() {
    let (app, _tmp) = build_app();
    let token = auth::create_token(JWT_SECRET, AGENT, false, 1);
    let (status, body) = send(
        &app,
        "POST",
        "/v1/memories/batch",
        Some(json!({ "memories": [{
            "agent_id": AGENT,
            "memory_type": "semantic",
            "content": "the user's favourite editor is helix",
            "provenance": { "session_id": "b1", "source": "import", "source_ref": "export.json" },
        }] })),
        &token,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let id = body["ids"][0].as_str().unwrap().to_string();
    let before = get(&app, &id, &token).await["provenance"].clone();
    assert_eq!(before["source"], "import");

    let (status, body) = send(
        &app,
        "PATCH",
        &format!("/v1/memories/{id}"),
        Some(json!({ "tags": ["topic:tools"], "provenance": { "session_id": "forged" } })),
        &token,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(get(&app, &id, &token).await["provenance"], before);
    let (_, list) = send(&app, "GET", "/v1/memories?session_id=b1", None, &token).await;
    assert_eq!(ids(&list, "memories"), [id].into());
}

#[tokio::test]
async fn erase_session_erases_the_session_only() {
    let (app, tmp) = build_app();
    let token = auth::create_token(JWT_SECRET, AGENT, false, 1);
    let marker_a = "SESSION-ERASE-MARKER-a91: the user's locker code is 4471";
    let marker_b = "SESSION-ERASE-MARKER-b17: the user's bike lock is 9020";
    let a = store(
        &app,
        json!({ "agent_id": AGENT, "memory_type": "semantic", "content": marker_a, "session_id": "s1" }),
        &token,
    )
    .await;
    let b = store(
        &app,
        json!({ "agent_id": AGENT, "memory_type": "semantic", "content": marker_b, "session_id": "s1" }),
        &token,
    )
    .await;
    let kept = store(
        &app,
        json!({ "agent_id": AGENT, "memory_type": "semantic", "content": "the user enjoys long walks by the sea", "session_id": "s2" }),
        &token,
    )
    .await;

    let (status, receipt) = send(&app, "POST", "/v1/sessions/s1/erase", None, &token).await;
    assert_eq!(status, StatusCode::OK, "{receipt}");
    assert_eq!(receipt["session_id"], "s1");
    let requested: std::collections::BTreeSet<String> = receipt["requested"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(requested, [a.clone(), b.clone()].into());
    assert_eq!(receipt["erased"].as_array().unwrap().len(), 2);
    assert_eq!(
        receipt["verification"]["residual_found"], false,
        "{receipt}"
    );
    assert!(!receipt.to_string().contains("SESSION-ERASE-MARKER"));
    for m in [marker_a, marker_b] {
        let mut stack = vec![tmp.path().to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    let bytes = std::fs::read(&p).unwrap();
                    assert!(
                        !bytes.windows(m.len()).any(|w| w == m.as_bytes()),
                        "{} still holds erased content",
                        p.display()
                    );
                }
            }
        }
    }
    for id in [&a, &b] {
        let (status, _) = send(&app, "GET", &format!("/v1/memories/{id}"), None, &token).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
    get(&app, &kept, &token).await;

    // An unknown session erases nothing.
    let (status, receipt) = send(&app, "POST", "/v1/sessions/none/erase", None, &token).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(receipt["erased"], json!([]));
}

#[tokio::test]
async fn erase_session_refuses_a_session_holding_another_agents_memories() {
    let (app, _tmp) = build_app();
    let mine = auth::create_token(JWT_SECRET, AGENT, false, 1);
    let theirs = auth::create_token(JWT_SECRET, OTHER_AGENT, false, 1);
    let own = store(
        &app,
        json!({ "agent_id": AGENT, "memory_type": "semantic", "content": "mine in the shared session", "session_id": "shared" }),
        &mine,
    )
    .await;
    let other = store(
        &app,
        json!({ "agent_id": OTHER_AGENT, "memory_type": "semantic", "content": "theirs in the shared session", "session_id": "shared" }),
        &theirs,
    )
    .await;
    let (status, _) = send(&app, "POST", "/v1/sessions/shared/erase", None, &mine).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    get(&app, &own, &mine).await;
    get(&app, &other, &theirs).await;
}
