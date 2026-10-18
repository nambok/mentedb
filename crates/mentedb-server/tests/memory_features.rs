//! Integration tests for the memory feature endpoints (hybrid search, context,
//! history, neighbors, conflicts, pins, profile, entities, batch store,
//! recent) on the default database and on named databases.

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
const JWT_SECRET: &str = "memory-features-test-secret";

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

async fn send(
    app: &axum::Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
    token: Option<&str>,
) -> (StatusCode, Value) {
    let mut req = Request::builder().method(method).uri(uri);
    if let Some(t) = token {
        req = req.header("authorization", format!("Bearer {t}"));
    }
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
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

fn memory(content: &str, memory_type: &str, tags: &[&str]) -> Value {
    json!({
        "agent_id": AGENT,
        "memory_type": memory_type,
        "content": content,
        "tags": tags,
        "confidence": 0.9,
    })
}

/// Store memories in one batch and return their ids.
async fn batch(app: &axum::Router, prefix: &str, items: Vec<Value>) -> Vec<String> {
    let (status, body) = send(
        app,
        "POST",
        &format!("{prefix}/memories/batch"),
        Some(json!({ "memories": items })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    body["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect()
}

fn ids_in(list: &Value) -> Vec<String> {
    list.as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect()
}

/// Every endpoint, driven against one database prefix.
async fn exercise_every_endpoint(app: &axum::Router, prefix: &str) {
    let ids = batch(
        app,
        prefix,
        vec![
            memory(
                "The kafka broker keeps three partitions per topic",
                "semantic",
                &["entity:kafka"],
            ),
            memory("My phone number is 1234", "semantic", &[]),
            memory("My phone number is 4321", "semantic", &[]),
            memory(
                "Always answer in plain words without jargon",
                "procedural",
                &[],
            ),
        ],
    )
    .await;
    assert_eq!(ids.len(), 4);
    let (kafka, old_phone, new_phone, rule) = (&ids[0], &ids[1], &ids[2], &ids[3]);

    // Hybrid search from text, embedded by the server.
    let (status, body) = send(
        app,
        "POST",
        &format!("{prefix}/search/hybrid"),
        Some(json!({ "query": "kafka broker partitions", "k": 5 })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let results = body["results"].as_array().unwrap();
    assert_eq!(results[0]["id"], kafka.as_str(), "{body}");
    assert!(results[0]["score"].is_number());
    assert!(results[0]["content"].as_str().unwrap().contains("kafka"));

    // Edges: the new phone supersedes the old one and contradicts it.
    for edge_type in ["supersedes", "contradicts"] {
        let (status, body) = send(
            app,
            "POST",
            &format!("{prefix}/edges"),
            Some(json!({ "source": new_phone, "target": old_phone, "edge_type": edge_type })),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{body}");
    }

    // History: the chain of facts the new number replaced.
    let (status, body) = send(
        app,
        "GET",
        &format!("{prefix}/memories/{new_phone}/history"),
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["memory"]["id"], new_phone.as_str());
    assert_eq!(ids_in(&body["history"]), vec![old_phone.clone()]);

    // Neighbors: both directions by default, filtered by edge type.
    let (status, body) = send(
        app,
        "GET",
        &format!("{prefix}/memories/{old_phone}/neighbors?depth=1&edge_types=supersedes"),
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(ids_in(&body["neighbors"]), vec![new_phone.clone()]);
    assert_eq!(body["neighbors"][0]["hops"], 1);
    let edges = body["edges"].as_array().unwrap();
    assert!(
        edges
            .iter()
            .all(|e| e["edge_type"] == "supersedes" && e["source"] == new_phone.as_str()),
        "{body}"
    );
    let (_, body) = send(
        app,
        "GET",
        &format!("{prefix}/memories/{old_phone}/neighbors?direction=out&edge_types=supersedes"),
        None,
        None,
    )
    .await;
    assert!(body["neighbors"].as_array().unwrap().is_empty(), "{body}");
    let (_, body) = send(
        app,
        "GET",
        &format!("{prefix}/memories/{old_phone}/neighbors?depth=99"),
        None,
        None,
    )
    .await;
    assert_eq!(body["depth"], 4, "depth is capped by the limits");
    let (status, _) = send(
        app,
        "GET",
        &format!("{prefix}/memories/{old_phone}/neighbors?edge_types=nonsense"),
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Conflicts: the contradiction pair with both memories.
    let (status, body) = send(app, "GET", &format!("{prefix}/conflicts"), None, None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["contradicts"].as_u64().unwrap() >= 1, "{body}");
    let pair = body["conflicts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["edge_type"] == "contradicts")
        .expect("contradiction listed");
    let pair_ids = [
        pair["source"]["id"].as_str().unwrap(),
        pair["target"]["id"].as_str().unwrap(),
    ];
    assert!(pair_ids.contains(&new_phone.as_str()) && pair_ids.contains(&old_phone.as_str()));
    assert!(pair["source"]["content"].is_string());

    // Pins: pin, list, see it in the heads-up context, unpin.
    let (status, body) = send(
        app,
        "POST",
        &format!("{prefix}/memories/{rule}/pin"),
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["pinned"], true);
    assert!(
        body["tags"]
            .as_array()
            .unwrap()
            .contains(&json!("scope:always"))
    );
    let (_, body) = send(app, "GET", &format!("{prefix}/pins"), None, None).await;
    assert_eq!(ids_in(&body["pins"]), vec![rule.clone()]);

    let (status, body) = send(
        app,
        "POST",
        &format!("{prefix}/context"),
        Some(json!({ "query": "what does the kafka broker do", "max_items": 5 })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let selected = body["memories"].as_array().unwrap();
    let pinned = selected
        .iter()
        .find(|m| m["id"] == rule.as_str())
        .expect("pinned memory in context");
    assert_eq!(pinned["reason"], "pinned");
    assert!(
        selected
            .iter()
            .any(|m| m["id"] == kafka.as_str() && m["reason"] == "relevant"),
        "{body}"
    );

    let (status, body) = send(
        app,
        "DELETE",
        &format!("{prefix}/memories/{rule}/pin"),
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["pinned"], false);
    let (_, body) = send(app, "GET", &format!("{prefix}/pins"), None, None).await;
    assert!(body["pins"].as_array().unwrap().is_empty(), "{body}");

    // Profile facts for the owner pair.
    let (status, body) = send(
        app,
        "GET",
        &format!("{prefix}/profile?agent_id={AGENT}"),
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["agent_id"], AGENT);
    assert!(body["profile"].is_null());
    let facts = body["facts"].as_array().unwrap();
    // Entity memories are not profile facts; the phone facts are.
    assert!(
        facts
            .iter()
            .any(|f| f.as_str().unwrap().contains("phone number"))
    );
    assert!(!facts.iter().any(|f| f.as_str().unwrap().contains("kafka")));

    // Entities.
    let (status, body) = send(app, "GET", &format!("{prefix}/entities"), None, None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["entities"][0]["name"], "kafka", "{body}");
    assert_eq!(body["entities"][0]["memory_count"], 1);
    let (status, body) = send(
        app,
        "GET",
        &format!("{prefix}/entities/Kafka/memories"),
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(ids_in(&body["memories"]), vec![kafka.clone()]);

    // Recent, newest first, bounded by limit.
    let (status, body) = send(app, "GET", &format!("{prefix}/recent?limit=2"), None, None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["memories"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn every_endpoint_on_the_default_database() {
    let (app, _tmp) = build_app(false);
    exercise_every_endpoint(&app, "/v1").await;
}

#[tokio::test]
async fn every_endpoint_on_a_named_database() {
    let (app, _tmp) = build_app(false);
    exercise_every_endpoint(&app, "/v1/db/team").await;
}

#[tokio::test]
async fn named_databases_are_isolated() {
    let (app, _tmp) = build_app(false);
    let ids = batch(
        &app,
        "/v1/db/a",
        vec![memory(
            "The kafka broker keeps three partitions per topic",
            "semantic",
            &["entity:kafka"],
        )],
    )
    .await;
    let id = &ids[0];

    let (status, _) = send(&app, "GET", &format!("/v1/db/a/memories/{id}"), None, None).await;
    assert_eq!(status, StatusCode::OK);
    for path in [
        format!("/v1/db/b/memories/{id}"),
        format!("/v1/db/b/memories/{id}/history"),
        format!("/v1/db/b/memories/{id}/neighbors"),
        format!("/v1/memories/{id}/history"),
    ] {
        let (status, _) = send(&app, "GET", &path, None, None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{path}");
    }
    let (status, _) = send(
        &app,
        "POST",
        &format!("/v1/db/b/memories/{id}/pin"),
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    for prefix in ["/v1/db/b", "/v1"] {
        let (_, body) = send(
            &app,
            "POST",
            &format!("{prefix}/search/hybrid"),
            Some(json!({ "query": "kafka broker partitions" })),
            None,
        )
        .await;
        assert!(body["results"].as_array().unwrap().is_empty(), "{body}");
        let (_, body) = send(&app, "GET", &format!("{prefix}/recent"), None, None).await;
        assert!(body["memories"].as_array().unwrap().is_empty(), "{body}");
        let (_, body) = send(&app, "GET", &format!("{prefix}/entities"), None, None).await;
        assert!(body["entities"].as_array().unwrap().is_empty(), "{body}");
    }
}

#[tokio::test]
async fn hybrid_search_takes_a_point_in_time_and_a_raw_embedding() {
    let (app, _tmp) = build_app(false);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    let later = now + 3_600_000_000;
    let mut future = memory("The office moves to the river building", "semantic", &[]);
    future["valid_from"] = json!(later);
    let ids = batch(&app, "/v1", vec![future]).await;

    let search = |as_of: Option<u64>| {
        let mut body = json!({ "query": "office river building", "k": 3 });
        if let Some(t) = as_of {
            body["as_of"] = json!(t);
        }
        body
    };
    let (_, body) = send(&app, "POST", "/v1/search/hybrid", Some(search(None)), None).await;
    assert!(
        !ids_in(&body["results"]).contains(&ids[0]),
        "not valid yet: {body}"
    );
    let (_, body) = send(
        &app,
        "POST",
        "/v1/search/hybrid",
        Some(search(Some(later + 1))),
        None,
    )
    .await;
    assert_eq!(ids_in(&body["results"]), vec![ids[0].clone()], "{body}");
    assert_eq!(body["as_of"], later + 1);

    let (status, body) = send(
        &app,
        "POST",
        "/v1/search/hybrid",
        Some(json!({ "embedding": vec![0.1f32; 384], "k": 1, "as_of": later + 1 })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["results"].as_array().unwrap().len(), 1);

    let (status, _) = send(&app, "POST", "/v1/search/hybrid", Some(json!({})), None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = send(
        &app,
        "POST",
        "/v1/search/hybrid",
        Some(json!({ "query": "x", "tags_mode": "xor" })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn batch_store_validates_every_item_before_storing() {
    let (app, _tmp) = build_app(false);
    let (status, body) = send(
        &app,
        "POST",
        "/v1/memories/batch",
        Some(json!({ "memories": [
            memory("first fact", "semantic", &[]),
            memory("second fact", "nonsense", &[]),
        ]})),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body["error"].as_str().unwrap().starts_with("memories[1]"),
        "{body}"
    );
    let (_, body) = send(&app, "GET", "/v1/recent", None, None).await;
    assert!(
        body["memories"].as_array().unwrap().is_empty(),
        "nothing stored"
    );

    let too_many: Vec<Value> = (0..1001)
        .map(|i| memory(&format!("fact {i}"), "semantic", &[]))
        .collect();
    let (status, _) = send(
        &app,
        "POST",
        "/v1/memories/batch",
        Some(json!({ "memories": too_many })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _) = send(
        &app,
        "POST",
        "/v1/memories/batch",
        Some(json!({ "memories": [] })),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Fields carry through, and items without a vector are embedded.
    let mut item = memory("the deploy runs at noon", "episodic", &["ops"]);
    item["attributes"] = json!({ "env": "prod" });
    item["valid_until"] = json!(u64::MAX);
    let ids = batch(&app, "/v1", vec![item]).await;
    let (_, body) = send(&app, "GET", &format!("/v1/memories/{}", ids[0]), None, None).await;
    assert_eq!(body["memory_type"], "episodic");
    assert_eq!(body["tags"], json!(["ops"]));
    assert_eq!(body["attributes"]["env"], "prod");
    assert_eq!(body["embedding"].as_array().unwrap().len(), 384);
}

#[tokio::test]
async fn token_scopes_apply_to_memory_endpoints() {
    let (app, _tmp) = build_app(true);
    let scoped = auth::create_scoped_token(
        JWT_SECRET,
        AGENT,
        false,
        1,
        Some(vec!["a".to_string()]),
        false,
    );
    let read_only = auth::create_scoped_token(JWT_SECRET, AGENT, false, 1, None, true);

    // A token scoped to database "a" cannot touch "b".
    let (status, body) = send(
        &app,
        "POST",
        "/v1/db/a/memories/batch",
        Some(json!({ "memories": [memory("scoped fact about kafka", "semantic", &[])] })),
        Some(&scoped),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    for (method, path, payload) in [
        ("GET", "/v1/db/b/recent", None),
        ("GET", "/v1/db/b/pins", None),
        ("GET", "/v1/db/b/conflicts", None),
        (
            "POST",
            "/v1/db/b/search/hybrid",
            Some(json!({ "query": "kafka" })),
        ),
        (
            "POST",
            "/v1/db/b/memories/batch",
            Some(json!({ "memories": [memory("x", "semantic", &[])] })),
        ),
    ] {
        let (status, _) = send(&app, method, path, payload, Some(&scoped)).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{method} {path}");
    }
    let (status, _) = send(&app, "GET", "/v1/db/a/recent", None, Some(&scoped)).await;
    assert_eq!(status, StatusCode::OK);

    // A read-only token searches but cannot write.
    let (status, body) = send(
        &app,
        "POST",
        "/v1/db/a/search/hybrid",
        Some(json!({ "query": "kafka" })),
        Some(&read_only),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["results"].as_array().unwrap().len(), 1, "{body}");
    let id = body["results"][0]["id"].as_str().unwrap().to_string();
    let (status, _) = send(
        &app,
        "POST",
        "/v1/search/hybrid",
        Some(json!({ "query": "kafka" })),
        Some(&read_only),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = send(
        &app,
        "POST",
        "/v1/db/a/context",
        Some(json!({ "query": "kafka" })),
        Some(&read_only),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = send(
        &app,
        "POST",
        "/v1/db/a/memories/batch",
        Some(json!({ "memories": [memory("x", "semantic", &[])] })),
        Some(&read_only),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    for method in ["POST", "DELETE"] {
        let (status, _) = send(
            &app,
            method,
            &format!("/v1/db/a/memories/{id}/pin"),
            None,
            Some(&read_only),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{method} pin");
    }
}

#[tokio::test]
async fn agent_bound_tokens_only_reach_their_own_memories() {
    let (app, _tmp) = build_app(true);
    let mine = auth::create_token(JWT_SECRET, AGENT, false, 1);
    let theirs = auth::create_token(JWT_SECRET, OTHER_AGENT, false, 1);

    let (status, body) = send(
        &app,
        "POST",
        "/v1/memories/batch",
        Some(json!({ "memories": [memory("my private kafka note", "semantic", &[])] })),
        Some(&mine),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let id = body["ids"][0].as_str().unwrap().to_string();

    // Storing on behalf of another agent is refused.
    let (status, _) = send(
        &app,
        "POST",
        "/v1/memories/batch",
        Some(json!({ "memories": [memory("x", "semantic", &[])] })),
        Some(&theirs),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    for path in [
        format!("/v1/memories/{id}/history"),
        format!("/v1/memories/{id}/neighbors"),
    ] {
        let (status, _) = send(&app, "GET", &path, None, Some(&theirs)).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{path}");
    }
    let (status, _) = send(
        &app,
        "POST",
        &format!("/v1/memories/{id}/pin"),
        None,
        Some(&theirs),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    let (_, body) = send(&app, "GET", "/v1/recent", None, Some(&theirs)).await;
    assert!(body["memories"].as_array().unwrap().is_empty(), "{body}");
    let (_, body) = send(
        &app,
        "POST",
        "/v1/search/hybrid",
        Some(json!({ "query": "kafka note" })),
        Some(&theirs),
    )
    .await;
    assert!(body["results"].as_array().unwrap().is_empty(), "{body}");
    let (status, _) = send(
        &app,
        "POST",
        "/v1/search/hybrid",
        Some(json!({ "query": "kafka note", "agent_id": AGENT })),
        Some(&theirs),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    let (_, body) = send(&app, "GET", "/v1/recent", None, Some(&mine)).await;
    assert_eq!(ids_in(&body["memories"]), vec![id]);
}

#[tokio::test]
async fn storing_a_duplicate_returns_the_memory_that_holds_it() {
    let (app, _tmp) = build_app(false);
    let body = json!({
        "agent_id": AGENT,
        "memory_type": "semantic",
        "content": "The deploy pipeline runs on GitHub Actions",
    });
    let (status, first) = send(
        &app,
        "POST",
        "/v1/db/dup/memories",
        Some(body.clone()),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(first["status"], "stored");
    let (status, again) = send(&app, "POST", "/v1/db/dup/memories", Some(body), None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(again["status"], "duplicate");
    assert_eq!(
        again["id"], first["id"],
        "the existing memory, not a new id"
    );
    assert_eq!(again["duplicate_of"], first["id"]);
    let id = again["id"].as_str().unwrap();
    let (status, fetched) = send(
        &app,
        "GET",
        &format!("/v1/db/dup/memories/{id}"),
        None,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "the returned id can be fetched");
    assert_eq!(
        fetched["content"],
        "The deploy pipeline runs on GitHub Actions"
    );
}
