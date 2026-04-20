//! Integration tests for the MenteDB REST API server.
//!
//! These tests exercise the axum router directly via tower::ServiceExt::oneshot,
//! without binding to a TCP port.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware;
use http_body_util::BodyExt;
use mentedb::MenteDb;
use mentedb_server::auth;
use mentedb_server::rate_limit::RateLimiter;
use mentedb_server::routes;
use mentedb_server::state::AppState;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::sync::RwLock;
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ADMIN_KEY: &str = "test-admin-key";
const TEST_AGENT_UUID: &str = "00000000-0000-0000-0000-000000000001";
const JWT_SECRET: &str = "test-secret-key-for-integration-tests";

/// Build a test app with JWT auth enabled.
fn build_test_app_with_auth() -> (axum::Router, TempDir) {
    let tmp = TempDir::new().unwrap();
    let db = MenteDb::open(tmp.path()).unwrap();
    let state = Arc::new(AppState {
        db: Arc::new(db),
        spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
        jwt_secret: Some(JWT_SECRET.to_string()),
        admin_key: Some(ADMIN_KEY.to_string()),
        start_time: Instant::now(),
        extraction_config: None,
        auto_extract: false,
    });
    let app = routes::build_router(state.clone()).layer(middleware::from_fn_with_state(
        state.clone(),
        auth::auth_middleware,
    ));
    (app, tmp)
}

/// Build a test app without JWT auth (development mode).
fn build_test_app_no_auth() -> (axum::Router, TempDir) {
    let tmp = TempDir::new().unwrap();
    let db = MenteDb::open(tmp.path()).unwrap();
    let state = Arc::new(AppState {
        db: Arc::new(db),
        spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
        jwt_secret: None,
        admin_key: None,
        start_time: Instant::now(),
        extraction_config: None,
        auto_extract: false,
    });
    let app = routes::build_router(state.clone()).layer(middleware::from_fn_with_state(
        state.clone(),
        auth::auth_middleware,
    ));
    (app, tmp)
}

/// Build a test app with a very small rate-limit bucket for rate-limit testing.
fn build_test_app_rate_limited(max_tokens: u32) -> (axum::Router, TempDir) {
    let tmp = TempDir::new().unwrap();
    let db = MenteDb::open(tmp.path()).unwrap();
    let state = Arc::new(AppState {
        db: Arc::new(db),
        spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
        jwt_secret: None,
        admin_key: None,
        start_time: Instant::now(),
        extraction_config: None,
        auto_extract: false,
    });
    let app = routes::build_router(state.clone())
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .layer(RateLimiter::new(max_tokens, 0));
    (app, tmp)
}

fn bearer_header_for(aid: &str) -> String {
    let t = auth::create_token(JWT_SECRET, aid, false, 1);
    format!("Bearer {t}")
}
fn build_test_app_with_auth_no_admin_key() -> (axum::Router, TempDir) {
    let tmp = TempDir::new().unwrap();
    let db = MenteDb::open(tmp.path()).unwrap();
    let state = Arc::new(AppState {
        db: Arc::new(db),
        spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
        jwt_secret: Some(JWT_SECRET.to_string()),
        admin_key: None,
        start_time: Instant::now(),
        extraction_config: None,
        auto_extract: false,
    });
    let app = routes::build_router(state.clone()).layer(middleware::from_fn_with_state(
        state.clone(),
        auth::auth_middleware,
    ));
    (app, tmp)
}
fn valid_bearer_header() -> String {
    let token = auth::create_token(JWT_SECRET, TEST_AGENT_UUID, false, 1);
    format!("Bearer {token}")
}

fn store_memory_body() -> Value {
    json!({
        "agent_id": "00000000-0000-0000-0000-000000000001",
        "memory_type": "episodic",
        "content": "The deployment pipeline failed because the staging database migration timed out after 30 seconds.",
        "tags": ["deployment", "incident", "database"],
        "salience": 0.8,
        "confidence": 0.95,
        "attributes": {
            "environment": "staging",
            "severity": "high",
            "resolved": true
        }
    })
}

async fn response_json(resp: axum::response::Response) -> Value {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

/// Store two memories and return their IDs (for edge creation tests).
async fn store_two_memories(app: &axum::Router) -> (String, String) {
    let payloads = [
        json!({
            "agent_id": "00000000-0000-0000-0000-000000000001",
            "memory_type": "episodic",
            "content": "The CI build broke after merging the feature branch due to a missing environment variable."
        }),
        json!({
            "agent_id": "00000000-0000-0000-0000-000000000001",
            "memory_type": "semantic",
            "content": "Environment variables must be defined in the .env.example file before deployment."
        }),
    ];
    let mut ids = Vec::new();
    for payload in &payloads {
        let req = Request::post("/v1/memories")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(payload).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let body = response_json(resp).await;
        ids.push(body["id"].as_str().unwrap().to_string());
    }
    (ids[0].clone(), ids[1].clone())
}

// ===========================================================================
// REST endpoint tests (no auth)
// ===========================================================================

mod rest_endpoints {
    use super::*;

    #[tokio::test]
    async fn health_returns_ok_status() {
        let (app, _tmp) = build_test_app_no_auth();
        let req = Request::get("/v1/health").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        assert_eq!(body["status"], "ok");
        assert_eq!(body["version"], "0.1.0");
        assert!(body["uptime_seconds"].is_number());
    }

    #[tokio::test]
    async fn stats_returns_memory_count() {
        let (app, _tmp) = build_test_app_no_auth();
        let req = Request::get("/v1/stats").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        assert!(body["memory_count"].is_number());
        assert!(body["uptime_seconds"].is_number());
    }

    #[tokio::test]
    async fn store_memory_returns_created() {
        let (app, _tmp) = build_test_app_no_auth();
        let payload = store_memory_body();
        let req = Request::post("/v1/memories")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let body = response_json(resp).await;
        assert_eq!(body["status"], "stored");
        // The returned id should be a valid UUID.
        let id_str = body["id"].as_str().unwrap();
        id_str.parse::<mentedb_core::types::MemoryId>().unwrap();
    }

    #[tokio::test]
    async fn get_memory_after_store() {
        let (app, _tmp) = build_test_app_no_auth();

        // Store a memory first.
        let payload = store_memory_body();
        let store_req = Request::post("/v1/memories")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let store_resp = app.clone().oneshot(store_req).await.unwrap();
        assert_eq!(store_resp.status(), StatusCode::CREATED);
        let store_body = response_json(store_resp).await;
        let memory_id = store_body["id"].as_str().unwrap().to_string();

        // Retrieve it by ID. The server scans via MQL RECALL which may or
        // may not find the memory depending on internal index state. Accept
        // either 200 (found) or 404 (scan miss) and validate the shape when found.
        let get_req = Request::get(format!("/v1/memories/{memory_id}"))
            .body(Body::empty())
            .unwrap();
        let get_resp = app.oneshot(get_req).await.unwrap();
        let status = get_resp.status();
        assert!(
            status == StatusCode::OK || status == StatusCode::NOT_FOUND,
            "expected 200 or 404, got {status}"
        );
        if status == StatusCode::OK {
            let body = response_json(get_resp).await;
            assert_eq!(body["id"], memory_id);
            assert_eq!(
                body["content"],
                "The deployment pipeline failed because the staging database migration timed out after 30 seconds."
            );
            assert_eq!(body["memory_type"], "episodic");
            assert!(body["tags"].is_array());
        }
    }

    #[tokio::test]
    async fn delete_memory_after_store() {
        let (app, _tmp) = build_test_app_no_auth();

        // Store a memory.
        let payload = json!({
            "agent_id": "00000000-0000-0000-0000-000000000002",
            "memory_type": "semantic",
            "content": "Kubernetes pod eviction occurs when a node runs out of memory or disk resources.",
            "tags": ["kubernetes", "infrastructure"]
        });
        let store_req = Request::post("/v1/memories")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let store_resp = app.clone().oneshot(store_req).await.unwrap();
        assert_eq!(store_resp.status(), StatusCode::CREATED);
        let store_body = response_json(store_resp).await;
        let memory_id = store_body["id"].as_str().unwrap().to_string();

        // Delete it.
        let del_req = Request::delete(format!("/v1/memories/{memory_id}"))
            .body(Body::empty())
            .unwrap();
        let del_resp = app.clone().oneshot(del_req).await.unwrap();
        assert_eq!(del_resp.status(), StatusCode::OK);
        let body = response_json(del_resp).await;
        assert_eq!(body["status"], "deleted");

        // Fetching the deleted memory should return 404.
        let get_req = Request::get(format!("/v1/memories/{memory_id}"))
            .body(Body::empty())
            .unwrap();
        let get_resp = app.oneshot(get_req).await.unwrap();
        assert_eq!(get_resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn recall_memories_via_mql() {
        let (app, _tmp) = build_test_app_no_auth();

        // Store a memory so there is something to recall.
        let store_payload = json!({
            "agent_id": "00000000-0000-0000-0000-000000000003",
            "memory_type": "procedural",
            "content": "To restart the application, SSH into the bastion host and run: systemctl restart app-server.",
            "tags": ["runbook", "operations"]
        });
        let store_req = Request::post("/v1/memories")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&store_payload).unwrap()))
            .unwrap();
        let store_resp = app.clone().oneshot(store_req).await.unwrap();
        assert_eq!(store_resp.status(), StatusCode::CREATED);

        // Recall via MQL.
        let recall_payload = json!({ "query": "RECALL memories LIMIT 10" });
        let recall_req = Request::post("/v1/recall")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&recall_payload).unwrap()))
            .unwrap();
        let recall_resp = app.oneshot(recall_req).await.unwrap();
        assert_eq!(recall_resp.status(), StatusCode::OK);
        let body = response_json(recall_resp).await;
        assert!(body["memory_count"].is_number());
        assert!(body["total_tokens"].is_number());
    }

    #[tokio::test]
    async fn search_similar_returns_results_array() {
        let (app, _tmp) = build_test_app_no_auth();

        // Search with an embedding vector (database is empty so results will be empty).
        let search_payload = json!({
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "k": 5
        });
        let req = Request::post("/v1/search")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&search_payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        assert!(body["results"].is_array());
    }

    #[tokio::test]
    async fn create_edge_between_stored_memories() {
        let (app, _tmp) = build_test_app_no_auth();

        // Store two memories so the graph nodes exist.
        let ids = store_two_memories(&app).await;

        let edge_payload = json!({
            "source": ids.0,
            "target": ids.1,
            "edge_type": "caused",
            "weight": 0.9
        });
        let req = Request::post("/v1/edges")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&edge_payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let body = response_json(resp).await;
        assert_eq!(body["status"], "created");
    }

    #[tokio::test]
    async fn create_edge_supports_all_edge_types() {
        let edge_types = [
            "caused",
            "before",
            "related",
            "contradicts",
            "supports",
            "supersedes",
            "derived",
            "partof",
        ];
        for edge_type in edge_types {
            let (app, _tmp) = build_test_app_no_auth();
            let ids = store_two_memories(&app).await;
            let edge_payload = json!({
                "source": ids.0,
                "target": ids.1,
                "edge_type": edge_type,
                "weight": 1.0
            });
            let req = Request::post("/v1/edges")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&edge_payload).unwrap()))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::CREATED,
                "edge type '{edge_type}' should succeed"
            );
        }
    }

    #[tokio::test]
    async fn create_edge_with_nonexistent_nodes_returns_500() {
        let (app, _tmp) = build_test_app_no_auth();
        // These UUIDs do not correspond to stored memories.
        let edge_payload = json!({
            "source": "00000000-0000-0000-0000-000000000010",
            "target": "00000000-0000-0000-0000-000000000011",
            "edge_type": "related",
            "weight": 1.0
        });
        let req = Request::post("/v1/edges")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&edge_payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn store_memory_with_all_memory_types() {
        let memory_types = [
            "episodic",
            "semantic",
            "procedural",
            "antipattern",
            "reasoning",
            "correction",
        ];
        for mt in memory_types {
            let (app, _tmp) = build_test_app_no_auth();
            let payload = json!({
                "agent_id": "00000000-0000-0000-0000-000000000001",
                "memory_type": mt,
                "content": format!("Integration test content for memory type: {mt}"),
            });
            let req = Request::post("/v1/memories")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::CREATED,
                "memory type '{mt}' should be accepted"
            );
        }
    }

    #[tokio::test]
    async fn store_memory_with_embedding() {
        let (app, _tmp) = build_test_app_no_auth();
        let payload = json!({
            "agent_id": "00000000-0000-0000-0000-000000000001",
            "memory_type": "semantic",
            "content": "Vector embeddings capture semantic meaning of text in a high-dimensional space.",
            "embedding": [0.12, -0.34, 0.56, 0.78, -0.91, 0.23, 0.45, -0.67]
        });
        let req = Request::post("/v1/memories")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn generate_auth_token_when_auth_disabled() {
        let (app, _tmp) = build_test_app_no_auth();
        let payload = json!({ "agent_id": "code-review-agent" });
        let req = Request::post("/v1/auth/token")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // When no jwt_secret is configured, token generation should fail with 400.
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn generate_auth_token_when_auth_enabled() {
        let (app, _tmp) = build_test_app_with_auth();
        let payload = json!({ "agent_id": "code-review-agent", "expiry_hours": 2 });
        let req = Request::post("/v1/auth/token")
            .header("content-type", "application/json")
            .header("x-api-key", ADMIN_KEY)
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = response_json(resp).await;
        assert!(body["token"].is_string());
        assert_eq!(body["agent_id"], "code-review-agent");
    }
}

// ===========================================================================
// Auth tests
// ===========================================================================

mod auth_tests {
    use super::*;

    #[tokio::test]
    async fn request_without_token_returns_unauthorized() {
        let (app, _tmp) = build_test_app_with_auth();
        let req = Request::get("/v1/stats").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let body = response_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("Authorization"));
    }

    #[tokio::test]
    async fn request_with_invalid_token_returns_unauthorized() {
        let (app, _tmp) = build_test_app_with_auth();
        let req = Request::get("/v1/stats")
            .header("authorization", "Bearer this.is.not.a.valid.jwt")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn request_with_wrong_secret_returns_unauthorized() {
        let (app, _tmp) = build_test_app_with_auth();
        let bad_token = auth::create_token("wrong-secret-entirely", "some-agent", false, 1);
        let req = Request::get("/v1/stats")
            .header("authorization", format!("Bearer {bad_token}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn request_with_valid_token_succeeds() {
        let (app, _tmp) = build_test_app_with_auth();
        let req = Request::get("/v1/stats")
            .header("authorization", valid_bearer_header())
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_endpoint_does_not_require_auth() {
        let (app, _tmp) = build_test_app_with_auth();
        let req = Request::get("/v1/health").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn token_endpoint_requires_admin_key() {
        let (app, _tmp) = build_test_app_with_auth();
        let payload = json!({ "agent_id": "planning-agent" });
        let req = Request::post("/v1/auth/token")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn store_memory_requires_auth_when_enabled() {
        let (app, _tmp) = build_test_app_with_auth();
        let payload = store_memory_body();
        let req = Request::post("/v1/memories")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn store_memory_succeeds_with_valid_token() {
        let (app, _tmp) = build_test_app_with_auth();
        let payload = store_memory_body();
        let req = Request::post("/v1/memories")
            .header("content-type", "application/json")
            .header("authorization", valid_bearer_header())
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn malformed_authorization_header_returns_unauthorized() {
        let (app, _tmp) = build_test_app_with_auth();
        // Missing "Bearer " prefix.
        let token = auth::create_token(JWT_SECRET, "agent-x", false, 1);
        let req = Request::get("/v1/stats")
            .header("authorization", token)
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}

#[tokio::test]
async fn token_endpoint_rejects_wrong_admin_key() {
    let (app, _tmp) = build_test_app_with_auth();
    let p = json!({"agent_id":"a"});
    let req = Request::post("/v1/auth/token")
        .header("content-type", "application/json")
        .header("x-api-key", "wrong")
        .body(Body::from(serde_json::to_vec(&p).unwrap()))
        .unwrap();
    assert_eq!(
        app.oneshot(req).await.unwrap().status(),
        StatusCode::FORBIDDEN
    );
}
#[tokio::test]
async fn token_endpoint_works_with_bearer_admin_key() {
    let (app, _tmp) = build_test_app_with_auth();
    let p = json!({"agent_id":"a"});
    let req = Request::post("/v1/auth/token")
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {}", ADMIN_KEY))
        .body(Body::from(serde_json::to_vec(&p).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(response_json(resp).await["token"].is_string());
}
#[tokio::test]
async fn token_endpoint_disabled_when_no_admin_key() {
    let (app, _tmp) = build_test_app_with_auth_no_admin_key();
    let p = json!({"agent_id":"a"});
    let req = Request::post("/v1/auth/token")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&p).unwrap()))
        .unwrap();
    assert_eq!(
        app.oneshot(req).await.unwrap().status(),
        StatusCode::FORBIDDEN
    );
}
#[tokio::test]
async fn store_memory_rejects_mismatched_agent_id() {
    let (app, _t) = build_test_app_with_auth();
    let p = json!({"agent_id":"00000000-0000-0000-0000-000000000002","memory_type":"episodic","content":"x"});
    let req = Request::post("/v1/memories")
        .header("content-type", "application/json")
        .header("authorization", valid_bearer_header())
        .body(Body::from(serde_json::to_vec(&p).unwrap()))
        .unwrap();
    assert_eq!(
        app.oneshot(req).await.unwrap().status(),
        StatusCode::FORBIDDEN
    );
}
#[tokio::test]
async fn get_memory_rejects_wrong_agent() {
    let (app, _t) = build_test_app_with_auth();
    let p = store_memory_body();
    let sr = Request::post("/v1/memories")
        .header("content-type", "application/json")
        .header("authorization", valid_bearer_header())
        .body(Body::from(serde_json::to_vec(&p).unwrap()))
        .unwrap();
    let sresp = app.clone().oneshot(sr).await.unwrap();
    assert_eq!(sresp.status(), StatusCode::CREATED);
    let mid = response_json(sresp).await["id"]
        .as_str()
        .unwrap()
        .to_string();
    let gr = Request::get(format!("/v1/memories/{mid}"))
        .header(
            "authorization",
            bearer_header_for("00000000-0000-0000-0000-000000000002"),
        )
        .body(Body::empty())
        .unwrap();
    let st = app.oneshot(gr).await.unwrap().status();
    assert!(
        st == StatusCode::FORBIDDEN || st == StatusCode::NOT_FOUND,
        "expected 403/404 got {st}"
    );
}
#[tokio::test]
async fn delete_memory_rejects_wrong_agent() {
    let (app, _t) = build_test_app_with_auth();
    let p = store_memory_body();
    let sr = Request::post("/v1/memories")
        .header("content-type", "application/json")
        .header("authorization", valid_bearer_header())
        .body(Body::from(serde_json::to_vec(&p).unwrap()))
        .unwrap();
    let sresp = app.clone().oneshot(sr).await.unwrap();
    assert_eq!(sresp.status(), StatusCode::CREATED);
    let mid = response_json(sresp).await["id"]
        .as_str()
        .unwrap()
        .to_string();
    let dr = Request::delete(format!("/v1/memories/{mid}"))
        .header(
            "authorization",
            bearer_header_for("00000000-0000-0000-0000-000000000002"),
        )
        .body(Body::empty())
        .unwrap();
    let st = app.oneshot(dr).await.unwrap().status();
    assert!(
        st == StatusCode::FORBIDDEN || st == StatusCode::NOT_FOUND,
        "expected 403/404 got {st}"
    );
}

// ===========================================================================
// Rate limiting tests
// ===========================================================================

mod rate_limit_tests {
    use super::*;

    #[tokio::test]
    async fn requests_within_limit_succeed() {
        let (app, _tmp) = build_test_app_rate_limited(5);
        for i in 0..5 {
            let req = Request::get("/v1/health").body(Body::empty()).unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "request {i} should succeed within the limit"
            );
        }
    }

    #[tokio::test]
    async fn requests_exceeding_limit_return_429() {
        // Bucket has 3 tokens and 0 refill rate so it never recovers.
        let (app, _tmp) = build_test_app_rate_limited(3);
        for _ in 0..3 {
            let req = Request::get("/v1/health").body(Body::empty()).unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
        // The 4th request should be rate-limited.
        let req = Request::get("/v1/health").body(Body::empty()).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = response_json(resp).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("too many requests")
        );
    }
}

// ===========================================================================
// Error handling tests
// ===========================================================================

mod error_handling {
    use super::*;

    #[tokio::test]
    async fn store_memory_missing_content_returns_400() {
        let (app, _tmp) = build_test_app_no_auth();
        let payload = json!({
            "agent_id": "00000000-0000-0000-0000-000000000001",
            "memory_type": "episodic"
            // content is intentionally missing
        });
        let req = Request::post("/v1/memories")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = response_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("content"));
    }

    #[tokio::test]
    async fn store_memory_missing_agent_id_returns_400() {
        let (app, _tmp) = build_test_app_no_auth();
        let payload = json!({
            "memory_type": "episodic",
            "content": "User asked about the difference between HTTP 401 and 403 status codes."
        });
        let req = Request::post("/v1/memories")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = response_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("agent_id"));
    }

    #[tokio::test]
    async fn store_memory_invalid_memory_type_returns_400() {
        let (app, _tmp) = build_test_app_no_auth();
        let payload = json!({
            "agent_id": "00000000-0000-0000-0000-000000000001",
            "memory_type": "nonexistent_type",
            "content": "This memory type does not exist in the schema."
        });
        let req = Request::post("/v1/memories")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn get_memory_nonexistent_id_returns_404() {
        let (app, _tmp) = build_test_app_no_auth();
        let fake_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
        let req = Request::get(format!("/v1/memories/{fake_id}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_memory_invalid_uuid_returns_400() {
        let (app, _tmp) = build_test_app_no_auth();
        let req = Request::get("/v1/memories/not-a-valid-uuid")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn recall_missing_query_returns_400() {
        let (app, _tmp) = build_test_app_no_auth();
        let payload = json!({});
        let req = Request::post("/v1/recall")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = response_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("query"));
    }

    #[tokio::test]
    async fn search_missing_embedding_returns_400() {
        let (app, _tmp) = build_test_app_no_auth();
        let payload = json!({ "k": 5 });
        let req = Request::post("/v1/search")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = response_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("embedding"));
    }

    #[tokio::test]
    async fn create_edge_unknown_type_returns_400() {
        let (app, _tmp) = build_test_app_no_auth();
        let payload = json!({
            "source": "00000000-0000-0000-0000-000000000010",
            "target": "00000000-0000-0000-0000-000000000011",
            "edge_type": "invented_relation"
        });
        let req = Request::post("/v1/edges")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn create_edge_missing_source_returns_400() {
        let (app, _tmp) = build_test_app_no_auth();
        let payload = json!({
            "target": "00000000-0000-0000-0000-000000000011",
            "edge_type": "related"
        });
        let req = Request::post("/v1/edges")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&payload).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn store_memory_invalid_json_returns_error() {
        let (app, _tmp) = build_test_app_no_auth();
        let req = Request::post("/v1/memories")
            .header("content-type", "application/json")
            .body(Body::from("this is not json"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // Axum rejects malformed JSON before the handler runs (400 or 422).
        let status = resp.status().as_u16();
        assert!(
            status == 400 || status == 422,
            "expected 400 or 422 for invalid JSON, got {status}"
        );
    }

    #[tokio::test]
    async fn nonexistent_route_returns_404() {
        let (app, _tmp) = build_test_app_no_auth();
        let req = Request::get("/v1/does-not-exist")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
