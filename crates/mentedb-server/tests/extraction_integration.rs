//! Integration tests for the extraction endpoints and auto-extract mode.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use mentedb::MenteDb;
use mentedb_extraction::{ExtractionConfig, LlmProvider};
use mentedb_server::routes;
use mentedb_server::state::AppState;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::sync::RwLock;
use tower::ServiceExt;

fn test_state(
    extraction_config: Option<ExtractionConfig>,
    auto_extract: bool,
) -> (Arc<AppState>, TempDir) {
    let tmp = TempDir::new().unwrap();
    let db = MenteDb::open(tmp.path()).unwrap();
    let state = Arc::new(AppState {
        db: Arc::new(RwLock::new(db)),
        spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
        jwt_secret: None,
        admin_key: None,
        start_time: Instant::now(),
        extraction_config,
        auto_extract,
    });
    (state, tmp)
}

fn mock_extraction_config() -> ExtractionConfig {
    // Use OpenAI provider type but with a dummy key — MockExtractionProvider
    // tests don't hit the network, so we only need valid config structure.
    ExtractionConfig {
        provider: LlmProvider::OpenAI,
        api_key: Some("test-key".to_string()),
        api_url: "http://localhost:0/fake".to_string(),
        model: "test-model".to_string(),
        max_extractions_per_conversation: 50,
        quality_threshold: 0.7,
        deduplication_threshold: 0.85,
        enable_contradiction_check: true,
        enable_deduplication: true,
    }
}

// -----------------------------------------------------------------------
// POST /v1/ingest returns 503 when no LLM provider is configured
// -----------------------------------------------------------------------

#[tokio::test]
async fn ingest_returns_503_without_provider() {
    let (state, _tmp) = test_state(None, false);
    let app = routes::build_router(state);

    let body = json!({
        "conversation": "User: hello\nAssistant: hi",
        "agent_id": "00000000-0000-0000-0000-000000000001",
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/ingest")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert!(
        json["error"]
            .as_str()
            .unwrap()
            .contains("LLM provider not configured")
    );
}

// -----------------------------------------------------------------------
// POST /v1/ingest returns 400 for missing conversation field
// -----------------------------------------------------------------------

#[tokio::test]
async fn ingest_returns_400_for_missing_conversation() {
    let (state, _tmp) = test_state(Some(mock_extraction_config()), false);
    let app = routes::build_router(state);

    let body = json!({
        "agent_id": "00000000-0000-0000-0000-000000000001",
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/ingest")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// -----------------------------------------------------------------------
// POST /v1/ingest returns 400 for missing agent_id
// -----------------------------------------------------------------------

#[tokio::test]
async fn ingest_returns_400_for_missing_agent_id() {
    let (state, _tmp) = test_state(Some(mock_extraction_config()), false);
    let app = routes::build_router(state);

    let body = json!({
        "conversation": "User: test\nAssistant: response",
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/ingest")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// -----------------------------------------------------------------------
// POST /v1/memories with auto_extract stores and returns extract stats
// -----------------------------------------------------------------------

#[tokio::test]
async fn store_memory_without_auto_extract_returns_normal() {
    let (state, _tmp) = test_state(Some(mock_extraction_config()), false);
    let app = routes::build_router(state);

    let body = json!({
        "agent_id": "00000000-0000-0000-0000-000000000001",
        "memory_type": "episodic",
        "content": "User: I love Rust\nAssistant: That's great!",
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/memories")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(json["status"], "stored");
    // No auto_extract field when auto-extract is disabled
    assert!(json.get("auto_extract").is_none());
}

// -----------------------------------------------------------------------
// Provider config parsing from env vars
// -----------------------------------------------------------------------

#[test]
fn extraction_config_defaults() {
    let config = ExtractionConfig::openai("test-key");
    assert_eq!(config.provider, LlmProvider::OpenAI);
    assert_eq!(config.api_key, Some("test-key".to_string()));
    assert_eq!(config.model, "gpt-4o-mini");
}

#[test]
fn extraction_config_anthropic() {
    let config = ExtractionConfig::anthropic("anthropic-key");
    assert_eq!(config.provider, LlmProvider::Anthropic);
    assert_eq!(config.model, "claude-sonnet-4-20250514");
}

#[test]
fn extraction_config_ollama() {
    let config = ExtractionConfig::ollama();
    assert_eq!(config.provider, LlmProvider::Ollama);
    assert!(config.api_key.is_none());
    assert_eq!(config.model, "llama3");
}

#[test]
fn extraction_config_quality_threshold() {
    let mut config = ExtractionConfig::openai("key");
    config.quality_threshold = 0.9;
    assert!((config.quality_threshold - 0.9).abs() < f32::EPSILON);
}
