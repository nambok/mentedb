//! Axum handler functions for the MenteDB REST API.

use std::collections::HashMap;
use std::sync::Arc;

use crate::auth::AuthenticatedAgent;
use axum::Extension;
use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
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

use crate::error::ApiError;
use crate::state::AppState;
use mentedb_core::types::{AgentId, MemoryId, SpaceId};

// ---------------------------------------------------------------------------
// GET /v1/health
// ---------------------------------------------------------------------------

/// Health check endpoint. Returns server uptime and status.
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let uptime = state.start_time.elapsed().as_secs();
    Json(json!({
        "status": "ok",
        "version": "0.1.0",
        "uptime_seconds": uptime,
    }))
}

// ---------------------------------------------------------------------------
// GET /v1/stats
// ---------------------------------------------------------------------------

/// Returns database statistics (memory count, index size, etc.).
pub async fn stats(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, ApiError> {
    let uptime = state.start_time.elapsed().as_secs();

    // Scan to count memories (no public count method on MenteDb).
    let db = state.db.read().await;
    let memory_count = match db.recall("RECALL memories LIMIT 10000") {
        Ok(window) => window
            .blocks
            .iter()
            .map(|b| b.memories.len())
            .sum::<usize>(),
        Err(_) => 0,
    };

    Ok(Json(json!({
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
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let agent_id = AgentId(parse_uuid(&req, "agent_id")?);
    if let Some(Extension(ref authed)) = agent {
        let tid: AgentId = authed
            .agent_id
            .parse()
            .map_err(|_| ApiError::Internal("token contains invalid agent_id UUID".into()))?;
        if tid != agent_id {
            return Err(ApiError::Forbidden(
                "agent_id in request body does not match token".into(),
            ));
        }
    }
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
    };

    let mut db = state.db.write().await;
    db.store(node).map_err(|e| {
        error!("store failed: {e}");
        ApiError::Internal(format!("store failed: {e}"))
    })?;

    // Auto-extract: if enabled and content looks like a conversation, run extraction
    if state.auto_extract && state.extraction_config.is_some() && looks_like_conversation(&content)
    {
        let extraction_config = state.extraction_config.clone().unwrap();
        match run_extraction(&extraction_config, &content, agent_id, space_id, &mut db).await {
            Ok(extract_stats) => {
                return Ok((
                    StatusCode::CREATED,
                    Json(json!({
                        "id": id.to_string(),
                        "status": "stored",
                        "auto_extract": extract_stats,
                    })),
                ));
            }
            Err(e) => {
                tracing::warn!(error = %e, "auto-extraction failed, memory stored without extraction");
            }
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
    State(state): State<Arc<AppState>>,
    agent: Option<Extension<AuthenticatedAgent>>,
    Path(id_str): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let id: MemoryId = id_str
        .parse()
        .map_err(|_| ApiError::BadRequest("invalid memory ID".into()))?;

    // PointLookup exists in QueryPlan but is not reachable via MQL syntax,
    // so we scan and filter client-side until MenteDb exposes a public get(id).
    let db = state.db.read().await;
    let window = db.recall("RECALL memories LIMIT 1000").map_err(|e| {
        error!("recall failed: {e}");
        ApiError::NotFound(format!("memory {id} not found"))
    })?;

    for block in &window.blocks {
        for scored in &block.memories {
            if scored.memory.id == id {
                if let Some(Extension(ref authed)) = agent {
                    let tid: AgentId = authed.agent_id.parse().map_err(|_| {
                        ApiError::Internal("token contains invalid agent_id UUID".into())
                    })?;
                    if scored.memory.agent_id != tid {
                        return Err(ApiError::Forbidden(
                            "memory belongs to a different agent".into(),
                        ));
                    }
                }
                return Ok(Json(memory_node_to_json(&scored.memory)));
            }
        }
    }

    Err(ApiError::NotFound(format!("memory {id} not found")))
}

// ---------------------------------------------------------------------------
// DELETE /v1/memories/:id
// ---------------------------------------------------------------------------

/// Deletes a memory from the database.
pub async fn forget_memory(
    State(state): State<Arc<AppState>>,
    agent: Option<Extension<AuthenticatedAgent>>,
    Path(id_str): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let id: MemoryId = id_str
        .parse()
        .map_err(|_| ApiError::BadRequest("invalid memory ID".into()))?;

    let mut db = state.db.write().await;
    if let Some(Extension(ref authed)) = agent {
        let tid: AgentId = authed
            .agent_id
            .parse()
            .map_err(|_| ApiError::Internal("token contains invalid agent_id UUID".into()))?;
        let window = db.recall("RECALL memories LIMIT 1000").map_err(|e| {
            error!("recall failed: {e}");
            ApiError::Internal(format!("recall failed: {e}"))
        })?;
        let mut found = false;
        for block in &window.blocks {
            for scored in &block.memories {
                if scored.memory.id == id {
                    if scored.memory.agent_id != tid {
                        return Err(ApiError::Forbidden(
                            "memory belongs to a different agent".into(),
                        ));
                    }
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }
        if !found {
            return Err(ApiError::NotFound(format!("memory {id} not found")));
        }
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
    State(state): State<Arc<AppState>>,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let query = req
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing 'query' field".into()))?;

    let db = state.db.read().await;
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
pub async fn search_similar(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let embedding = parse_embedding(&req, true)?;
    let k = req.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let db = state.db.read().await;
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
pub async fn create_edge(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
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

    let mut db = state.db.write().await;
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
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let agent_id = AgentId(parse_uuid(&req, "agent_id")?);
    if let Some(Extension(ref authed)) = agent {
        let tid: AgentId = authed
            .agent_id
            .parse()
            .map_err(|_| ApiError::Internal("token contains invalid agent_id UUID".into()))?;
        if tid != agent_id {
            return Err(ApiError::Forbidden(
                "agent_id in request body does not match token".into(),
            ));
        }
    }

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

    let mut db = state.db.write().await;
    let stats =
        run_extraction(extraction_config, conversation, agent_id, space_id, &mut db).await?;

    Ok((StatusCode::OK, Json(stats)))
}

/// Run the extraction pipeline and store accepted memories. Returns JSON stats.
async fn run_extraction(
    config: &ExtractionConfig,
    conversation: &str,
    agent_id: AgentId,
    space_id: SpaceId,
    db: &mut mentedb::MenteDb,
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
    for memory in &quality_passed {
        let memory_type = map_extraction_type_to_memory_type(&memory.memory_type);
        let id = MemoryId::new();
        let node = MemoryNode {
            id,
            agent_id,
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
        };
        match db.store(node) {
            Ok(()) => stored_ids.push(id.to_string()),
            Err(e) => {
                tracing::warn!(error = %e, "failed to store extracted memory, skipping");
            }
        }
    }

    Ok(json!({
        "memories_stored": stored_ids.len(),
        "rejected_low_quality": rejected_low_quality,
        "rejected_duplicate": 0,
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

fn parse_uuid(val: &Value, field: &str) -> Result<Uuid, ApiError> {
    val.get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest(format!("missing '{field}'")))
        .and_then(|s| {
            Uuid::parse_str(s)
                .map_err(|_| ApiError::BadRequest(format!("invalid UUID for '{field}'")))
        })
}

fn parse_memory_type(val: &Value) -> Result<MemoryType, ApiError> {
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

fn parse_edge_type(s: &str) -> Option<EdgeType> {
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

fn parse_embedding(val: &Value, required: bool) -> Result<Vec<f32>, ApiError> {
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

fn parse_attributes(val: Option<&Value>) -> HashMap<String, AttributeValue> {
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

fn memory_node_to_json(node: &MemoryNode) -> Value {
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

    json!({
        "id": node.id.to_string(),
        "agent_id": node.agent_id.to_string(),
        "memory_type": memory_type_str,
        "embedding": node.embedding,
        "content": node.content,
        "created_at": node.created_at,
        "accessed_at": node.accessed_at,
        "access_count": node.access_count,
        "salience": node.salience,
        "confidence": node.confidence,
        "space_id": node.space_id.to_string(),
        "attributes": attrs,
        "tags": node.tags,
    })
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
