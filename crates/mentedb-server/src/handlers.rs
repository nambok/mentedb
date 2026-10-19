//! Axum handler functions for the MenteDB REST API.

use std::collections::HashMap;
use std::sync::Arc;

use crate::auth::AuthenticatedAgent;
use axum::Extension;
use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::{AttributeValue, MemoryType};
use mentedb_core::space::Permission;
use mentedb_core::{MemoryEdge, MemoryNode};
use mentedb_extraction::{
    ExtractionConfig, ExtractionPipeline, HttpExtractionProvider,
    map_extraction_type_to_memory_type,
};
use serde_json::{Map, Value, json};
use tracing::error;
use uuid::Uuid;

use crate::databases::Db;
use crate::error::ApiError;
use crate::state::AppState;
use mentedb_core::types::{AgentId, MemoryId, SpaceId, UserId};

// ---------------------------------------------------------------------------
// GET /v1/health
// ---------------------------------------------------------------------------

/// Health check endpoint. Returns server uptime and status.
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let uptime = state.start_time.elapsed().as_secs();
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": uptime,
    }))
}

// ---------------------------------------------------------------------------
// GET /v1/stats
// ---------------------------------------------------------------------------

/// Returns database statistics (memory count, index size, etc.).
pub async fn stats(
    State(state): State<Arc<AppState>>,
    db: Db,
) -> Result<impl IntoResponse, ApiError> {
    let uptime = state.start_time.elapsed().as_secs();
    let memory_count = db.memory_count();

    Ok(Json(json!({
        "database": db.name,
        "memory_count": memory_count,
        "uptime_seconds": uptime,
    })))
}

// ---------------------------------------------------------------------------
// POST /v1/memories
// ---------------------------------------------------------------------------

/// Stores a new memory node in the database.
pub async fn store_memory(
    State(state): State<Arc<AppState>>,
    agent: Option<Extension<AuthenticatedAgent>>,
    db: Db,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let agent_id = AgentId(parse_uuid(&req, "agent_id")?);
    check_body_agent(&agent, agent_id)?;
    let memory_type = parse_memory_type(&req)?;

    let content = req
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing or invalid 'content'".into()))?
        .to_string();

    let embedding = parse_embedding(&req, false)?;

    let tags: Vec<String> = match req.get("tags").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        None => vec![],
    };

    let attributes = parse_attributes(req.get("attributes"));

    let space_id = match req.get("space_id") {
        Some(v) => v
            .as_str()
            .and_then(|s| s.parse::<SpaceId>().ok())
            .ok_or_else(|| ApiError::BadRequest("invalid 'space_id'".into()))?,
        None => SpaceId::nil(),
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    let id = MemoryId::new();

    let salience = req.get("salience").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
    let confidence = req
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    let node = MemoryNode {
        id,
        agent_id,
        user_id: parse_optional_uuid(&req, "user_id")?
            .map(UserId)
            .unwrap_or_else(UserId::nil),
        memory_type,
        embedding,
        content: content.clone(),
        created_at: now,
        accessed_at: now,
        access_count: 0,
        salience,
        confidence,
        space_id,
        attributes,
        tags,
        valid_from: req.get("valid_from").and_then(|v| v.as_u64()),
        valid_until: req.get("valid_until").and_then(|v| v.as_u64()),
        context: None,
    };

    let handle = db.handle.clone();
    let outcome = tokio::task::spawn_blocking(move || handle.store_with_outcome(node))
        .await
        .map_err(|e| ApiError::Internal(format!("store task failed: {e}")))?
        .map_err(|e| {
            error!("store failed: {e}");
            ApiError::Internal(format!("store failed: {e}"))
        })?;
    // A duplicate was not written: hand back the memory that already holds
    // the fact, never the new id, which does not exist.
    if let mentedb::StoreOutcome::Duplicate { of } = outcome {
        return Ok((
            StatusCode::OK,
            Json(
                json!({ "id": of.to_string(), "status": "duplicate", "duplicate_of": of.to_string() }),
            ),
        ));
    }

    // Auto-extract: if enabled and content looks like a conversation, queue extraction
    if state.auto_extract
        && state.extraction_config.is_some()
        && looks_like_conversation(&content)
        && let Some(tx) = &state.extraction_tx
    {
        let req = crate::extraction_queue::ExtractionRequest {
            config: state.extraction_config.clone().unwrap(),
            content: content.clone(),
            agent_id,
            space_id,
            db: db.handle.clone(),
        };
        if tx.try_send(req).is_err() {
            tracing::warn!("extraction queue full, skipping auto-extract");
        }
    }

    Ok((
        StatusCode::CREATED,
        Json(json!({ "id": id.to_string(), "status": "stored" })),
    ))
}

// ---------------------------------------------------------------------------
// GET /v1/memories/:id
// ---------------------------------------------------------------------------

/// Retrieves a single memory by its UUID.
pub async fn get_memory(
    agent: Option<Extension<AuthenticatedAgent>>,
    db: Db,
    Path(params): Path<HashMap<String, String>>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let id = memory_id_param(&params)?;
    let node = db
        .get_memory(id)
        .map_err(|_| ApiError::NotFound(format!("memory {id} not found")))?;
    check_owner(&agent, &node)?;

    Ok(Json(memory_node_to_json_with(
        &node,
        wants_embedding(&query),
    )))
}

// ---------------------------------------------------------------------------
// DELETE /v1/memories/:id
// ---------------------------------------------------------------------------

/// Deletes a memory from the database.
pub async fn forget_memory(
    agent: Option<Extension<AuthenticatedAgent>>,
    db: Db,
    Path(params): Path<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let id = memory_id_param(&params)?;
    if agent.is_some() {
        let node = db
            .get_memory(id)
            .map_err(|_| ApiError::NotFound(format!("memory {id} not found")))?;
        check_owner(&agent, &node)?;
    }
    db.forget(id).map_err(|e| {
        error!("forget failed: {e}");
        ApiError::Internal(format!("forget failed: {e}"))
    })?;

    Ok(Json(json!({ "status": "deleted" })))
}

// ---------------------------------------------------------------------------
// POST /v1/recall
// ---------------------------------------------------------------------------

/// Recalls memories matching an MQL query string.
pub async fn recall_memories(
    db: Db,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let query = req
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing 'query' field".into()))?;

    let window = db.recall(query).map_err(|e| {
        error!("recall failed: {e}");
        ApiError::Internal(format!("recall failed: {e}"))
    })?;

    let memory_count: usize = window.blocks.iter().map(|b| b.memories.len()).sum();
    Ok(Json(json!({
        "context": window.format,
        "total_tokens": window.total_tokens,
        "memory_count": memory_count,
    })))
}

// ---------------------------------------------------------------------------
// POST /v1/search
// ---------------------------------------------------------------------------

/// Searches for memories similar to a given embedding vector.
pub async fn search_similar(db: Db, Json(req): Json<Value>) -> Result<impl IntoResponse, ApiError> {
    let k = req.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    // Either a raw vector, or text the server embeds with its own embedder.
    let embedding = match req.get("query").and_then(|v| v.as_str()) {
        Some(text) if req.get("embedding").is_none() => embed_query(&db, text).await?,
        _ => parse_embedding(&req, true)?,
    };

    let results = db.recall_similar(&embedding, k).map_err(|e| {
        error!("search failed: {e}");
        ApiError::Internal(format!("search failed: {e}"))
    })?;

    let items: Vec<Value> = results
        .iter()
        .map(|(id, score)| json!({ "id": id.to_string(), "score": score }))
        .collect();

    Ok(Json(json!({ "results": items })))
}

// ---------------------------------------------------------------------------
// POST /v1/edges
// ---------------------------------------------------------------------------

/// Creates a typed, weighted edge between two memories.
pub async fn create_edge(db: Db, Json(req): Json<Value>) -> Result<impl IntoResponse, ApiError> {
    let source = MemoryId(parse_uuid(&req, "source")?);
    let target = MemoryId(parse_uuid(&req, "target")?);

    let edge_type_str = req
        .get("edge_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing 'edge_type'".into()))?;

    let edge_type = parse_edge_type(edge_type_str)
        .ok_or_else(|| ApiError::BadRequest(format!("unknown edge_type: {edge_type_str}")))?;

    let weight = req.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    let valid_from = req.get("valid_from").and_then(|v| v.as_u64());
    let valid_until = req.get("valid_until").and_then(|v| v.as_u64());

    let edge = MemoryEdge {
        source,
        target,
        edge_type,
        weight,
        created_at: now,
        valid_from,
        valid_until,
        label: req
            .get("label")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
    };

    db.relate(edge).map_err(|e| {
        error!("relate failed: {e}");
        ApiError::Internal(format!("relate failed: {e}"))
    })?;

    Ok((StatusCode::CREATED, Json(json!({ "status": "created" }))))
}

// ---------------------------------------------------------------------------
// POST /v1/ingest
// ---------------------------------------------------------------------------

/// Ingest a conversation: extract memories via LLM, filter, and store them.
pub async fn ingest_conversation(
    State(state): State<Arc<AppState>>,
    agent: Option<Extension<AuthenticatedAgent>>,
    db: Db,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let agent_id = AgentId(parse_uuid(&req, "agent_id")?);
    check_body_agent(&agent, agent_id)?;

    let extraction_config = state.extraction_config.as_ref().ok_or_else(|| {
        ApiError::ServiceUnavailable(
            "LLM provider not configured. Set MENTEDB_LLM_PROVIDER env var.".into(),
        )
    })?;

    let conversation = req
        .get("conversation")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing or invalid 'conversation'".into()))?;

    let space_id = match req.get("space_id") {
        Some(v) => v
            .as_str()
            .and_then(|s| s.parse::<SpaceId>().ok())
            .ok_or_else(|| ApiError::BadRequest("invalid 'space_id'".into()))?,
        None => SpaceId::nil(),
    };

    let stats = run_extraction(extraction_config, conversation, agent_id, space_id, &db).await?;

    Ok((StatusCode::OK, Json(stats)))
}

// ---------------------------------------------------------------------------
// POST /v1/process_turn
// ---------------------------------------------------------------------------

/// Run the full cognitive turn pipeline: embed, recall context, store the
/// turn, reconcile, analyze. The same engine call the hosted platform serves;
/// self-hosters get the one-call pipeline instead of assembling it from the
/// store and recall primitives. Delta serving is tracked per agent across
/// requests via `AppState::turn_trackers`.
pub async fn process_turn(
    State(state): State<Arc<AppState>>,
    agent: Option<Extension<AuthenticatedAgent>>,
    db: Db,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let user_message = req
        .get("user_message")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing or invalid 'user_message'".into()))?
        .to_string();
    let assistant_response = req
        .get("assistant_response")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let turn_id = req.get("turn_id").and_then(|v| v.as_u64()).unwrap_or(0);
    let project_context = req
        .get("project_context")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let session_id = req
        .get("session_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let agent_id = req
        .get("agent_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| {
            Uuid::parse_str(s).map_err(|_| ApiError::BadRequest("invalid 'agent_id' UUID".into()))
        })
        .transpose()?;
    let user_id = req
        .get("user_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| {
            Uuid::parse_str(s).map_err(|_| ApiError::BadRequest("invalid 'user_id' UUID".into()))
        })
        .transpose()?;

    // A scoped token must only process turns for its own agent.
    if let Some(aid) = agent_id {
        check_body_agent(&agent, AgentId(aid))?;
    }

    let input = mentedb::process_turn::ProcessTurnInput {
        user_message,
        assistant_response,
        turn_id,
        project_context,
        agent_id,
        user_id,
        session_id,
    };

    // The tracker is keyed by agent (nil for the shared/global agent) so a
    // multi-agent server keeps one delta stream per agent. Taken out of the
    // map for the duration of the blocking call, then put back, so the mutex
    // is never held across the engine turn.
    let tracker_key = format!("{}/{}", db.name, agent_id.unwrap_or_else(Uuid::nil));
    let mut tracker = {
        let mut map = state.turn_trackers.lock().await;
        map.remove(&tracker_key)
            .unwrap_or_else(mentedb::context::DeltaTracker::new)
    };

    // `store: false` runs only the read half of the turn: recall, pain
    // warnings and the delta, with nothing written to the database.
    let store = match req.get("store") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(ApiError::BadRequest("'store' must be a boolean".into())),
    };

    let db = db.handle.clone();
    let joined = tokio::task::spawn_blocking(move || {
        let result = if store {
            db.process_turn(&input, &mut tracker).map(TurnOutcome::Full)
        } else {
            db.recall_turn_context(&input, &mut tracker)
                .map(TurnOutcome::RecallOnly)
        };
        (result, tracker)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("process_turn task failed: {e}")))?;
    let (result, tracker) = joined;
    {
        let mut map = state.turn_trackers.lock().await;
        map.insert(tracker_key, tracker);
    }
    let result = result.map_err(|e| {
        error!("process_turn failed: {e}");
        ApiError::Internal(format!("process_turn failed: {e}"))
    })?;
    let result = match result {
        TurnOutcome::Full(r) => r,
        TurnOutcome::RecallOnly(r) => {
            let context: Vec<Value> = r.context.iter().map(turn_context_json).collect();
            return Ok((
                StatusCode::OK,
                Json(json!({
                    "ok": true,
                    "stored": false,
                    "context": context,
                    "cache_hit": r.cache_hit,
                    "stored_ids": [],
                    "episodic_id": null,
                    "pain_warnings": r.pain_warnings.len(),
                    "delta_added": r.delta_added.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                    "delta_removed": r.delta_removed.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                })),
            ));
        }
    };

    let context: Vec<Value> = result.context.iter().map(turn_context_json).collect();

    Ok((
        StatusCode::OK,
        Json(json!({
            "ok": true,
            "stored": true,
            "context": context,
            "cache_hit": result.cache_hit,
            "stored_ids": result.stored_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
            "episodic_id": result.episodic_id.map(|id| id.to_string()),
            "contradictions": result.contradiction_count,
            "facts_extracted": result.facts_extracted,
            "edges_created": result.edges_created,
            "inference_actions": result.inference_actions,
            "pain_warnings": result.pain_warnings.len(),
            "predicted_topics": result.predicted_topics,
            "delta_added": result.delta_added.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
            "delta_removed": result.delta_removed.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
        })),
    ))
}

/// What a turn ran: the full pipeline, or only its read half.
enum TurnOutcome {
    Full(mentedb::process_turn::ProcessTurnResult),
    RecallOnly(mentedb::process_turn::RecallContextResult),
}

fn turn_context_json(sm: &mentedb::context::ScoredMemory) -> Value {
    json!({
        "id": sm.memory.id.to_string(),
        "content": sm.memory.content,
        "memory_type": format!("{:?}", sm.memory.memory_type).to_lowercase(),
        "relevance_score": sm.score,
        "created_at": sm.memory.created_at,
    })
}

/// Run the extraction pipeline and store accepted memories. Returns JSON stats.
async fn run_extraction(
    config: &ExtractionConfig,
    conversation: &str,
    agent_id: AgentId,
    space_id: SpaceId,
    db: &mentedb::MenteDb,
) -> Result<Value, ApiError> {
    let provider = HttpExtractionProvider::new(config.clone()).map_err(|e| {
        error!("extraction provider init failed: {e}");
        ApiError::Internal(format!("extraction provider init failed: {e}"))
    })?;

    let pipeline = ExtractionPipeline::new(provider, config.clone());

    let all_memories = pipeline
        .extract_from_conversation(conversation)
        .await
        .map_err(|e| {
            error!("extraction failed: {e}");
            ApiError::Internal(format!("extraction failed: {e}"))
        })?;

    let total_extracted = all_memories.len();
    let quality_passed = pipeline.filter_quality(&all_memories);
    let rejected_low_quality = total_extracted - quality_passed.len();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    let mut stored_ids = Vec::new();
    let mut duplicate_ids = Vec::new();
    for memory in &quality_passed {
        let memory_type = map_extraction_type_to_memory_type(&memory.memory_type);
        let id = MemoryId::new();
        let node = MemoryNode {
            id,
            agent_id,
            user_id: UserId::nil(),
            memory_type,
            embedding: vec![],
            content: memory.content.clone(),
            created_at: now,
            accessed_at: now,
            access_count: 0,
            salience: memory.confidence,
            confidence: memory.confidence,
            space_id,
            attributes: std::collections::HashMap::new(),
            tags: memory.tags.clone(),
            valid_from: None,
            valid_until: None,
            context: None,
        };
        match db.store_with_outcome(node) {
            Ok(mentedb::StoreOutcome::Stored) => stored_ids.push(id.to_string()),
            Ok(mentedb::StoreOutcome::Duplicate { of }) => duplicate_ids.push(of.to_string()),
            Err(e) => {
                tracing::warn!(error = %e, "failed to store extracted memory, skipping");
            }
        }
    }

    Ok(json!({
        "memories_stored": stored_ids.len(),
        "rejected_low_quality": rejected_low_quality,
        "rejected_duplicate": duplicate_ids.len(),
        "duplicate_of": duplicate_ids,
        "contradictions": 0,
        "stored_ids": stored_ids,
    }))
}

/// Heuristic: does this content look like a multi-turn conversation?
fn looks_like_conversation(content: &str) -> bool {
    let lower = content.to_lowercase();
    let patterns = ["user:", "assistant:", "human:", "ai:", "\nq:", "\na:"];
    let matches = patterns.iter().filter(|p| lower.contains(*p)).count();
    matches >= 2
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reject a body `agent_id` that differs from the agent the token is bound to.
/// Tokens scoped to databases rather than an agent carry no agent to check.
pub(crate) fn check_body_agent(
    agent: &Option<Extension<AuthenticatedAgent>>,
    agent_id: AgentId,
) -> Result<(), ApiError> {
    if let Some(Extension(authed)) = agent
        && let Some(tid) = authed.agent()?
        && tid != agent_id
    {
        return Err(ApiError::Forbidden(
            "agent_id in request body does not match token".into(),
        ));
    }
    Ok(())
}

/// Reject reading or changing a memory owned by another agent.
pub(crate) fn check_owner(
    agent: &Option<Extension<AuthenticatedAgent>>,
    node: &MemoryNode,
) -> Result<(), ApiError> {
    if let Some(Extension(authed)) = agent
        && !authed.admin
        && let Some(tid) = authed.agent()?
        && node.agent_id != tid
    {
        return Err(ApiError::Forbidden(
            "memory belongs to a different agent".into(),
        ));
    }
    Ok(())
}

/// The `{id}` path parameter as a memory id.
pub(crate) fn memory_id_param(params: &HashMap<String, String>) -> Result<MemoryId, ApiError> {
    params
        .get("id")
        .ok_or_else(|| ApiError::BadRequest("missing memory ID".into()))?
        .parse()
        .map_err(|_| ApiError::BadRequest("invalid memory ID".into()))
}

/// Embed query text with the database's embedder, off the async workers.
pub(crate) async fn embed_query(db: &Db, text: &str) -> Result<Vec<f32>, ApiError> {
    let handle = db.handle.clone();
    let text = text.to_string();
    tokio::task::spawn_blocking(move || handle.embed_text(&text))
        .await
        .map_err(|e| ApiError::Internal(format!("embed task failed: {e}")))?
        .map_err(|e| ApiError::ServiceUnavailable(format!("embedding failed: {e}")))?
        .ok_or_else(|| {
            ApiError::BadRequest(
                "this server has no embedder; send an 'embedding' vector instead".into(),
            )
        })
}

pub(crate) fn parse_optional_uuid(val: &Value, field: &str) -> Result<Option<Uuid>, ApiError> {
    match val
        .get(field)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        Some(s) => Uuid::parse_str(s)
            .map(Some)
            .map_err(|_| ApiError::BadRequest(format!("invalid UUID for '{field}'"))),
        None => Ok(None),
    }
}

pub(crate) fn parse_uuid(val: &Value, field: &str) -> Result<Uuid, ApiError> {
    val.get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest(format!("missing '{field}'")))
        .and_then(|s| {
            Uuid::parse_str(s)
                .map_err(|_| ApiError::BadRequest(format!("invalid UUID for '{field}'")))
        })
}

pub fn parse_memory_type(val: &Value) -> Result<MemoryType, ApiError> {
    let s = val
        .get("memory_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing 'memory_type'".into()))?;

    match s.to_lowercase().as_str() {
        "episodic" => Ok(MemoryType::Episodic),
        "semantic" => Ok(MemoryType::Semantic),
        "procedural" => Ok(MemoryType::Procedural),
        "antipattern" | "anti_pattern" => Ok(MemoryType::AntiPattern),
        "reasoning" => Ok(MemoryType::Reasoning),
        "correction" => Ok(MemoryType::Correction),
        _ => Err(ApiError::BadRequest(format!("unknown memory_type: {s}"))),
    }
}

pub(crate) fn parse_edge_type(s: &str) -> Option<EdgeType> {
    match s.to_lowercase().as_str() {
        "caused" => Some(EdgeType::Caused),
        "before" => Some(EdgeType::Before),
        "related" => Some(EdgeType::Related),
        "contradicts" => Some(EdgeType::Contradicts),
        "supports" => Some(EdgeType::Supports),
        "supersedes" => Some(EdgeType::Supersedes),
        "derived" => Some(EdgeType::Derived),
        "partof" | "part_of" => Some(EdgeType::PartOf),
        _ => None,
    }
}

pub(crate) fn parse_embedding(val: &Value, required: bool) -> Result<Vec<f32>, ApiError> {
    match val.get("embedding").and_then(|v| v.as_array()) {
        Some(arr) => {
            let mut emb = Vec::with_capacity(arr.len());
            for v in arr {
                let f = v.as_f64().ok_or_else(|| {
                    ApiError::BadRequest("embedding values must be numbers".into())
                })?;
                emb.push(f as f32);
            }
            Ok(emb)
        }
        None if required => Err(ApiError::BadRequest("missing 'embedding' array".into())),
        None => Ok(vec![]),
    }
}

pub(crate) fn parse_attributes(val: Option<&Value>) -> HashMap<String, AttributeValue> {
    let mut map = HashMap::new();
    if let Some(Value::Object(obj)) = val {
        for (k, v) in obj {
            let av = match v {
                Value::String(s) => AttributeValue::String(s.clone()),
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        AttributeValue::Integer(i)
                    } else if let Some(f) = n.as_f64() {
                        AttributeValue::Float(f)
                    } else {
                        continue;
                    }
                }
                Value::Bool(b) => AttributeValue::Boolean(*b),
                _ => continue,
            };
            map.insert(k.clone(), av);
        }
    }
    map
}

/// Whether a `fields` query value (comma separated) asks for the embedding.
pub(crate) fn wants_embedding(params: &HashMap<String, String>) -> bool {
    params
        .get("fields")
        .is_some_and(|f| f.split(',').any(|x| x.trim() == "embedding"))
}

/// A memory as JSON, without its embedding vector. Vectors are large and
/// rarely useful to API callers; ask with `?fields=embedding` where supported.
pub(crate) fn memory_node_to_json(node: &MemoryNode) -> Value {
    memory_node_to_json_with(node, false)
}

/// A memory as JSON, with the embedding when `include_embedding` is set.
pub(crate) fn memory_node_to_json_with(node: &MemoryNode, include_embedding: bool) -> Value {
    let mut attrs = Map::new();
    for (k, v) in &node.attributes {
        let jv = match v {
            AttributeValue::String(s) => Value::String(s.clone()),
            AttributeValue::Integer(i) => json!(i),
            AttributeValue::Float(f) => json!(f),
            AttributeValue::Boolean(b) => json!(b),
            AttributeValue::Bytes(b) => json!(b),
        };
        attrs.insert(k.clone(), jv);
    }

    let memory_type_str = format!("{:?}", node.memory_type).to_lowercase();

    let mut out = json!({
        "id": node.id.to_string(),
        "agent_id": node.agent_id.to_string(),
        "user_id": node.user_id.to_string(),
        "memory_type": memory_type_str,
        "content": node.content,
        "created_at": node.created_at,
        "accessed_at": node.accessed_at,
        "access_count": node.access_count,
        "salience": node.salience,
        "confidence": node.confidence,
        "space_id": node.space_id.to_string(),
        "valid_from": node.valid_from,
        "valid_until": node.valid_until,
        "attributes": attrs,
        "tags": node.tags,
    });
    if include_embedding && let Value::Object(m) = &mut out {
        m.insert("embedding".into(), json!(node.embedding));
    }
    out
}

fn parse_permission(s: &str) -> Result<Permission, ApiError> {
    match s.to_lowercase().as_str() {
        "read" => Ok(Permission::Read),
        "write" => Ok(Permission::Write),
        "readwrite" | "read_write" => Ok(Permission::ReadWrite),
        "admin" => Ok(Permission::Admin),
        _ => Err(ApiError::BadRequest(format!("unknown permission: {s}"))),
    }
}
fn resolve_agent_id(
    agent: &Option<Extension<AuthenticatedAgent>>,
    body: Option<&Value>,
) -> Result<AgentId, ApiError> {
    if let Some(Extension(a)) = agent {
        return a
            .agent_id
            .parse::<AgentId>()
            .map_err(|_| ApiError::Unauthorized("invalid agent_id in token".into()));
    }
    if let Some(val) = body {
        return Ok(AgentId(parse_uuid(val, "agent_id")?));
    }
    Err(ApiError::Unauthorized("no agent_id available".into()))
}
fn is_admin_token(agent: &Option<Extension<AuthenticatedAgent>>) -> bool {
    matches!(agent, Some(Extension(a)) if a.admin)
}
pub async fn create_space(
    State(state): State<Arc<AppState>>,
    agent: Option<Extension<AuthenticatedAgent>>,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let agent_id = resolve_agent_id(&agent, Some(&req))?;
    let name = req
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing 'name' field".into()))?;
    let mut spaces = state.spaces.write().await;
    let space = spaces.create_space(name, agent_id);
    Ok((
        StatusCode::CREATED,
        Json(
            json!({"id": space.id.to_string(), "name": space.name, "owner": space.owner.to_string()}),
        ),
    ))
}
pub async fn list_spaces(
    State(state): State<Arc<AppState>>,
    agent: Option<Extension<AuthenticatedAgent>>,
) -> Result<impl IntoResponse, ApiError> {
    let agent_id = resolve_agent_id(&agent, None)?;
    let spaces = state.spaces.read().await;
    let list: Vec<Value> = spaces
        .list_spaces_for_agent(agent_id)
        .iter()
        .map(|s| json!({"id": s.id.to_string(), "name": s.name, "owner": s.owner.to_string()}))
        .collect();
    Ok(Json(json!({"spaces": list})))
}
pub async fn grant_space_access(
    State(state): State<Arc<AppState>>,
    agent: Option<Extension<AuthenticatedAgent>>,
    Path(space_id_str): Path<String>,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let space_id: SpaceId = space_id_str
        .parse()
        .map_err(|_| ApiError::BadRequest("invalid space ID".into()))?;
    let caller_id = resolve_agent_id(&agent, None)?;
    let target_agent = AgentId(parse_uuid(&req, "agent_id")?);
    let perm_str = req
        .get("permission")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing 'permission' field".into()))?;
    let perm = parse_permission(perm_str)?;
    let mut spaces = state.spaces.write().await;
    if !is_admin_token(&agent) && !spaces.check_access(space_id, caller_id, Permission::Admin) {
        return Err(ApiError::Forbidden(
            "only space owner or admin can grant access".into(),
        ));
    }
    spaces.grant_access(space_id, target_agent, perm);
    Ok(Json(json!({"status": "granted"})))
}

// ---------------------------------------------------------------------------
// Admin endpoints (gated by the admin key, for the bundled /console).
// ---------------------------------------------------------------------------

/// Query params for GET /v1/admin/memories.
#[derive(serde::Deserialize)]
pub struct AdminListParams {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    /// Filter by owning agent UUID.
    pub agent: Option<String>,
    /// Filter by memory type name.
    #[serde(rename = "type")]
    pub memory_type: Option<String>,
    /// Case-insensitive content substring.
    pub q: Option<String>,
}

fn memory_type_from_str(s: &str) -> Result<MemoryType, ApiError> {
    match s.to_lowercase().as_str() {
        "episodic" => Ok(MemoryType::Episodic),
        "semantic" => Ok(MemoryType::Semantic),
        "procedural" => Ok(MemoryType::Procedural),
        "antipattern" | "anti_pattern" => Ok(MemoryType::AntiPattern),
        "reasoning" => Ok(MemoryType::Reasoning),
        "correction" => Ok(MemoryType::Correction),
        other => Err(ApiError::BadRequest(format!(
            "unknown memory type: {other}"
        ))),
    }
}

/// GET /v1/admin/memories: a bounded, paginated page of memories for the console.
pub async fn admin_list_memories(
    State(state): State<Arc<AppState>>,
    db: Db,
    headers: HeaderMap,
    Query(p): Query<AdminListParams>,
) -> Result<impl IntoResponse, ApiError> {
    crate::auth::admin_authorized(&headers, &state.admin_key)?;
    let limit = p.limit.unwrap_or(50).clamp(1, 200);
    let offset = p.offset.unwrap_or(0);
    let agent = match p.agent.as_deref().filter(|s| !s.is_empty()) {
        Some(s) => {
            Some(AgentId(s.parse().map_err(|_| {
                ApiError::BadRequest("invalid agent UUID".into())
            })?))
        }
        None => None,
    };
    let mtype = match p.memory_type.as_deref().filter(|s| !s.is_empty()) {
        Some(s) => Some(memory_type_from_str(s)?),
        None => None,
    };
    let q = p.q.as_deref().filter(|s| !s.is_empty());

    let (total, nodes) = db
        .list_memories(limit, offset, agent, mtype, q)
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    let memories: Vec<_> = nodes.iter().map(memory_node_to_json).collect();
    Ok(Json(json!({
        "total": total,
        "limit": limit,
        "offset": offset,
        "memories": memories,
    })))
}

/// DELETE /v1/admin/memories/{id}: forget a memory from the console.
pub async fn admin_delete_memory(
    State(state): State<Arc<AppState>>,
    db: Db,
    headers: HeaderMap,
    Path(params): Path<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    crate::auth::admin_authorized(&headers, &state.admin_key)?;
    let id = memory_id_param(&params)?;
    db.forget(id)
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(Json(json!({ "status": "deleted", "id": id.to_string() })))
}

/// POST /v1/admin/mql: run an MQL query, return the scored matches. Powers the
/// console query box and the `mentedb` CLI's remote mode.
pub async fn admin_run_mql(
    State(state): State<Arc<AppState>>,
    db: Db,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    crate::auth::admin_authorized(&headers, &state.admin_key)?;
    let mql = body
        .get("mql")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing 'mql'".into()))?;
    let scored = db
        .query(mql)
        .map_err(|e| ApiError::BadRequest(format!("query error: {e}")))?;
    let memories: Vec<Value> = scored
        .iter()
        .map(|s| {
            let mut j = memory_node_to_json(&s.memory);
            if let Value::Object(ref mut m) = j {
                m.insert("score".into(), json!(s.score));
            }
            j
        })
        .collect();
    Ok(Json(
        json!({ "count": memories.len(), "memories": memories }),
    ))
}
