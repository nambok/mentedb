//! HTTP endpoint handlers for the MenteDB REST API.

use std::collections::HashMap;
use std::sync::Arc;

use mentedb::MenteDb;
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::{AttributeValue, MemoryType};
use mentedb_core::{MemoryEdge, MemoryNode};
use serde_json::{Map, Value, json};
use tokio::sync::Mutex;
use tracing::error;
use uuid::Uuid;

use crate::router::Response;

type Db = Arc<Mutex<MenteDb>>;

pub async fn health() -> Response {
    Response::ok(json!({ "status": "ok", "version": "0.1.0" }))
}

pub async fn store_memory(db: Db, body: &[u8]) -> Response {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return Response::bad_request(&format!("invalid JSON: {e}")),
    };

    let agent_id = match parse_uuid(&req, "agent_id") {
        Ok(id) => id,
        Err(r) => return r,
    };

    let memory_type = match parse_memory_type(&req) {
        Ok(mt) => mt,
        Err(r) => return r,
    };

    let content = match req.get("content").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return Response::bad_request("missing or invalid 'content'"),
    };

    let embedding: Vec<f32> = match req.get("embedding").and_then(|v| v.as_array()) {
        Some(arr) => {
            let mut emb = Vec::with_capacity(arr.len());
            for v in arr {
                match v.as_f64() {
                    Some(f) => emb.push(f as f32),
                    None => return Response::bad_request("embedding values must be numbers"),
                }
            }
            emb
        }
        None => vec![],
    };

    let tags: Vec<String> = match req.get("tags").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        None => vec![],
    };

    let attributes = parse_attributes(req.get("attributes"));

    let space_id = match req.get("space_id") {
        Some(v) => match v.as_str().and_then(|s| Uuid::parse_str(s).ok()) {
            Some(id) => id,
            None => return Response::bad_request("invalid 'space_id'"),
        },
        None => Uuid::nil(),
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    let id = Uuid::new_v4();

    let salience = req
        .get("salience")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;

    let confidence = req
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

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

    let mut db = db.lock().await;
    match db.store(node) {
        Ok(()) => Response::created(json!({ "id": id.to_string(), "status": "stored" })),
        Err(e) => {
            error!("store failed: {e}");
            Response::internal_error(&format!("store failed: {e}"))
        }
    }
}

pub async fn get_memory(db: Db, id_str: &str) -> Response {
    let id = match Uuid::parse_str(id_str) {
        Ok(id) => id,
        Err(_) => return Response::bad_request("invalid memory ID"),
    };

    // Use MQL recall with a tag-based fallback scan, then filter by ID.
    // PointLookup exists in QueryPlan but is not reachable via MQL syntax,
    // so we scan and filter client-side. This is inefficient but correct
    // until MenteDb exposes a public get(id) method.
    let query = format!("RECALL memories LIMIT 1000");
    let mut db = db.lock().await;
    match db.recall(&query) {
        Ok(window) => {
            for block in &window.blocks {
                for scored in &block.memories {
                    if scored.memory.id == id {
                        return Response::ok(memory_node_to_json(&scored.memory));
                    }
                }
            }
            Response::not_found(&format!("memory {id} not found"))
        }
        Err(e) => {
            error!("recall failed: {e}");
            Response::not_found(&format!("memory {id} not found"))
        }
    }
}

pub async fn recall_memories(db: Db, body: &[u8]) -> Response {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return Response::bad_request(&format!("invalid JSON: {e}")),
    };

    let query = match req.get("query").and_then(|v| v.as_str()) {
        Some(q) => q,
        None => return Response::bad_request("missing 'query' field"),
    };

    let mut db = db.lock().await;
    match db.recall(query) {
        Ok(window) => {
            let memory_count: usize = window.blocks.iter().map(|b| b.memories.len()).sum();
            Response::ok(json!({
                "context": window.format,
                "total_tokens": window.total_tokens,
                "memory_count": memory_count,
            }))
        }
        Err(e) => {
            error!("recall failed: {e}");
            Response::internal_error(&format!("recall failed: {e}"))
        }
    }
}

pub async fn search_similar(db: Db, body: &[u8]) -> Response {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return Response::bad_request(&format!("invalid JSON: {e}")),
    };

    let embedding: Vec<f32> = match req.get("embedding").and_then(|v| v.as_array()) {
        Some(arr) => {
            let mut emb = Vec::with_capacity(arr.len());
            for v in arr {
                match v.as_f64() {
                    Some(f) => emb.push(f as f32),
                    None => return Response::bad_request("embedding values must be numbers"),
                }
            }
            emb
        }
        None => return Response::bad_request("missing 'embedding' array"),
    };

    let k = req.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let mut db = db.lock().await;
    match db.recall_similar(&embedding, k) {
        Ok(results) => {
            let items: Vec<Value> = results
                .iter()
                .map(|(id, score)| json!({ "id": id.to_string(), "score": score }))
                .collect();
            Response::ok(json!({ "results": items }))
        }
        Err(e) => {
            error!("search failed: {e}");
            Response::internal_error(&format!("search failed: {e}"))
        }
    }
}

pub async fn create_edge(db: Db, body: &[u8]) -> Response {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return Response::bad_request(&format!("invalid JSON: {e}")),
    };

    let source = match parse_uuid(&req, "source") {
        Ok(id) => id,
        Err(r) => return r,
    };
    let target = match parse_uuid(&req, "target") {
        Ok(id) => id,
        Err(r) => return r,
    };

    let edge_type = match req.get("edge_type").and_then(|v| v.as_str()) {
        Some(s) => match parse_edge_type(s) {
            Some(et) => et,
            None => return Response::bad_request(&format!("unknown edge_type: {s}")),
        },
        None => return Response::bad_request("missing 'edge_type'"),
    };

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

    let mut db = db.lock().await;
    match db.relate(edge) {
        Ok(()) => Response::created(json!({ "status": "created" })),
        Err(e) => {
            error!("relate failed: {e}");
            Response::internal_error(&format!("relate failed: {e}"))
        }
    }
}

pub async fn forget_memory(db: Db, id_str: &str) -> Response {
    let id = match Uuid::parse_str(id_str) {
        Ok(id) => id,
        Err(_) => return Response::bad_request("invalid memory ID"),
    };

    let mut db = db.lock().await;
    match db.forget(id) {
        Ok(()) => Response::ok(json!({ "status": "deleted" })),
        Err(e) => {
            error!("forget failed: {e}");
            Response::internal_error(&format!("forget failed: {e}"))
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_uuid(val: &Value, field: &str) -> Result<Uuid, Response> {
    match val.get(field).and_then(|v| v.as_str()) {
        Some(s) => Uuid::parse_str(s).map_err(|_| Response::bad_request(&format!("invalid UUID for '{field}'"))),
        None => Err(Response::bad_request(&format!("missing '{field}'"))),
    }
}

fn parse_memory_type(val: &Value) -> Result<MemoryType, Response> {
    match val.get("memory_type").and_then(|v| v.as_str()) {
        Some(s) => match s.to_lowercase().as_str() {
            "episodic" => Ok(MemoryType::Episodic),
            "semantic" => Ok(MemoryType::Semantic),
            "procedural" => Ok(MemoryType::Procedural),
            "antipattern" | "anti_pattern" => Ok(MemoryType::AntiPattern),
            "reasoning" => Ok(MemoryType::Reasoning),
            "correction" => Ok(MemoryType::Correction),
            _ => Err(Response::bad_request(&format!("unknown memory_type: {s}"))),
        },
        None => Err(Response::bad_request("missing 'memory_type'")),
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
