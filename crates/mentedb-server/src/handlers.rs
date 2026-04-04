//! Axum handler functions for the MenteDB REST API.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::{AttributeValue, MemoryType};
use mentedb_core::{MemoryEdge, MemoryNode};
use serde_json::{Map, Value, json};
use tracing::error;
use uuid::Uuid;

use crate::error::ApiError;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// GET /v1/health
// ---------------------------------------------------------------------------

pub async fn health(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
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

pub async fn stats(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, ApiError> {
    let uptime = state.start_time.elapsed().as_secs();

    // Scan to count memories (no public count method on MenteDb).
    let mut db = state.db.write().await;
    let memory_count = match db.recall("RECALL memories LIMIT 10000") {
        Ok(window) => window.blocks.iter().map(|b| b.memories.len()).sum::<usize>(),
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

pub async fn store_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let agent_id = parse_uuid(&req, "agent_id")?;
    let memory_type = parse_memory_type(&req)?;

    let content = req
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing or invalid 'content'".into()))?
        .to_string();

    let embedding = parse_embedding(&req, false)?;

    let tags: Vec<String> = match req.get("tags").and_then(|v| v.as_array()) {
        Some(arr) => arr.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        None => vec![],
    };

    let attributes = parse_attributes(req.get("attributes"));

    let space_id = match req.get("space_id") {
        Some(v) => v
            .as_str()
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or_else(|| ApiError::BadRequest("invalid 'space_id'".into()))?,
        None => Uuid::nil(),
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    let id = Uuid::new_v4();

    let salience = req.get("salience").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
    let confidence = req.get("confidence").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

    let node = MemoryNode {
        id,
        agent_id,
        memory_type,
        embedding,
        content,
        created_at: now,
        accessed_at: now,
        access_count: 0,
        salience,
        confidence,
        space_id,
        attributes,
        tags,
    };

    let mut db = state.db.write().await;
    db.store(node).map_err(|e| {
        error!("store failed: {e}");
        ApiError::Internal(format!("store failed: {e}"))
    })?;

    Ok((StatusCode::CREATED, Json(json!({ "id": id.to_string(), "status": "stored" }))))
}

// ---------------------------------------------------------------------------
// GET /v1/memories/:id
// ---------------------------------------------------------------------------

pub async fn get_memory(
    State(state): State<Arc<AppState>>,
    Path(id_str): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let id = Uuid::parse_str(&id_str)
        .map_err(|_| ApiError::BadRequest("invalid memory ID".into()))?;

    // PointLookup exists in QueryPlan but is not reachable via MQL syntax,
    // so we scan and filter client-side until MenteDb exposes a public get(id).
    let mut db = state.db.write().await;
    let window = db.recall("RECALL memories LIMIT 1000").map_err(|e| {
        error!("recall failed: {e}");
        ApiError::NotFound(format!("memory {id} not found"))
    })?;

    for block in &window.blocks {
        for scored in &block.memories {
            if scored.memory.id == id {
                return Ok(Json(memory_node_to_json(&scored.memory)));
            }
        }
    }

    Err(ApiError::NotFound(format!("memory {id} not found")))
}

// ---------------------------------------------------------------------------
// DELETE /v1/memories/:id
// ---------------------------------------------------------------------------

pub async fn forget_memory(
    State(state): State<Arc<AppState>>,
    Path(id_str): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let id = Uuid::parse_str(&id_str)
        .map_err(|_| ApiError::BadRequest("invalid memory ID".into()))?;

    let mut db = state.db.write().await;
    db.forget(id).map_err(|e| {
        error!("forget failed: {e}");
        ApiError::Internal(format!("forget failed: {e}"))
    })?;

    Ok(Json(json!({ "status": "deleted" })))
}

// ---------------------------------------------------------------------------
// POST /v1/recall
// ---------------------------------------------------------------------------

pub async fn recall_memories(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let query = req
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing 'query' field".into()))?;

    let mut db = state.db.write().await;
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

pub async fn search_similar(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let embedding = parse_embedding(&req, true)?;
    let k = req.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let mut db = state.db.write().await;
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

pub async fn create_edge(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let source = parse_uuid(&req, "source")?;
    let target = parse_uuid(&req, "target")?;

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

    let edge = MemoryEdge {
        source,
        target,
        edge_type,
        weight,
        created_at: now,
    };

    let mut db = state.db.write().await;
    db.relate(edge).map_err(|e| {
        error!("relate failed: {e}");
        ApiError::Internal(format!("relate failed: {e}"))
    })?;

    Ok((StatusCode::CREATED, Json(json!({ "status": "created" }))))
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
                let f = v
                    .as_f64()
                    .ok_or_else(|| ApiError::BadRequest("embedding values must be numbers".into()))?;
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
