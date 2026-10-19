//! Memory feature endpoints: hybrid search, heads-up context, history, graph, conflicts, pins, profile, entities, batch store.
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use axum::extract::{Path, Query};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use mentedb::injection::{InjectionQuery, SelectionReason};
use mentedb_core::edge::EdgeType;
use mentedb_core::error::MenteError;
use mentedb_core::types::{AgentId, MemoryId, SpaceId, UserId};
use mentedb_core::{MemoryEdge, MemoryNode};
use serde_json::{Value, json};
use tracing::error;
use uuid::Uuid;

use crate::auth::AuthenticatedAgent;
use crate::databases::Db;
use crate::error::ApiError;
use crate::handlers::{
    check_body_agent, check_owner, embed_query, memory_id_param, memory_node_to_json,
    parse_attributes, parse_edge_type, parse_embedding, parse_memory_type, parse_optional_uuid,
    parse_uuid,
};
use crate::state::AppState;

/// Tag the injection path treats as pinned: always injected, never ranked out.
const PIN_TAG: &str = "scope:always";
/// Tag marking a pin made by hand, which standing-rule pruning keeps.
const MANUAL_TAG: &str = "source:manual";
/// Tag prefix of entity memories.
const ENTITY_TAG_PREFIX: &str = "entity:";

/// Request size and result size limits for these endpoints.
#[derive(Clone, Debug)]
pub struct MemoryApiLimits {
    /// Results returned by hybrid search when `k` is not given.
    pub default_k: usize,
    /// Most results one hybrid search may ask for.
    pub max_k: usize,
    /// Hops walked by the neighbors endpoint when `depth` is not given.
    pub default_neighbor_depth: usize,
    /// Most hops the neighbors endpoint walks.
    pub max_neighbor_depth: usize,
    /// Most memories the neighbors endpoint returns.
    pub max_neighbors: usize,
    /// Most predecessors the history endpoint walks.
    pub max_history_depth: usize,
    /// Items returned by list endpoints when `limit` is not given.
    pub default_list: usize,
    /// Most items one list request may ask for.
    pub max_list: usize,
    /// Most memories one batch store may carry.
    pub max_batch: usize,
    /// Ranked context items (beyond pins) when `max_items` is not given.
    pub default_context_items: usize,
    /// Most ranked context items one request may ask for.
    pub max_context_items: usize,
    /// Verbatim episodic items within the context when not given.
    pub default_context_episodic: usize,
}

impl Default for MemoryApiLimits {
    fn default() -> Self {
        Self {
            default_k: 10,
            max_k: 200,
            default_neighbor_depth: 1,
            max_neighbor_depth: 4,
            max_neighbors: 500,
            max_history_depth: 64,
            default_list: 50,
            max_list: 1000,
            max_batch: 1000,
            default_context_items: 10,
            max_context_items: 100,
            default_context_episodic: 3,
        }
    }
}

type Limits = Extension<Arc<MemoryApiLimits>>;
type Caller = Option<Extension<AuthenticatedAgent>>;

/// Routes on one database, mounted under `/v1` and `/v1/db/{db}`.
pub fn routes() -> Router<Arc<AppState>> {
    routes_with_limits(MemoryApiLimits::default())
}

/// [`routes`] with custom limits.
pub fn routes_with_limits(limits: MemoryApiLimits) -> Router<Arc<AppState>> {
    Router::new()
        .route("/search/hybrid", post(search_hybrid))
        .route("/context", post(context))
        .route("/memories/batch", post(store_batch))
        .route("/memories/{id}/history", get(history))
        .route("/memories/{id}/neighbors", get(neighbors))
        .route("/memories/{id}/pin", post(pin).delete(unpin))
        .route("/pins", get(list_pins))
        .route("/conflicts", get(conflicts))
        .route("/profile", get(profile))
        .route("/entities", get(entities))
        .route("/entities/{name}/memories", get(entity_memories))
        .route("/recent", get(recent))
        .layer(Extension(Arc::new(limits)))
}

// ---------------------------------------------------------------------------
// POST /v1/search/hybrid
// ---------------------------------------------------------------------------

/// Hybrid recall: vector similarity fused with BM25 keyword matching, at a
/// point in time, optionally scoped to an agent and a user.
async fn search_hybrid(
    agent: Caller,
    Extension(limits): Limits,
    db: Db,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let k = bounded(body_usize(&req, "k")?, limits.default_k, limits.max_k);
    let query_text = body_str(&req, "query");
    let embedding = query_embedding(&db, &req, query_text.as_deref()).await?;
    let at = match req.get("as_of") {
        Some(v) => v
            .as_u64()
            .ok_or_else(|| ApiError::BadRequest("'as_of' must be microseconds".into()))?,
        None => now_micros(),
    };
    let time_range = match req.get("time_range") {
        Some(v) => {
            let pair = v
                .as_array()
                .filter(|a| a.len() == 2)
                .and_then(|a| Some((a[0].as_u64()?, a[1].as_u64()?)))
                .ok_or_else(|| {
                    ApiError::BadRequest(
                        "'time_range' must be [from, until] in microseconds".into(),
                    )
                })?;
            Some(pair)
        }
        None => None,
    };
    let tags = string_array(&req, "tags")?;
    let tags_or = match req.get("tags_mode").and_then(|v| v.as_str()) {
        None | Some("and") => false,
        Some("or") => true,
        Some(other) => {
            return Err(ApiError::BadRequest(format!(
                "unknown tags_mode '{other}': use 'and' or 'or'"
            )));
        }
    };
    let agent_scope = scope_agent(&agent, parse_optional_uuid(&req, "agent_id")?)?;
    let user_scope = parse_optional_uuid(&req, "user_id")?.map(UserId);
    let project = body_str(&req, "project");

    let handle = db.handle.clone();
    let results = tokio::task::spawn_blocking(move || {
        let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
        let hits = handle.recall_hybrid_scoped_at_mode(
            &embedding,
            query_text.as_deref(),
            k,
            at,
            (!tag_refs.is_empty()).then_some(tag_refs.as_slice()),
            tags_or,
            time_range,
            agent_scope,
            user_scope,
            project.as_deref(),
        )?;
        Ok::<_, MenteError>(
            hits.into_iter()
                .filter_map(|(id, score)| {
                    let node = handle.get_memory(id).ok()?;
                    Some(with_fields(&node, json!({ "score": score })))
                })
                .collect::<Vec<Value>>(),
        )
    })
    .await
    .map_err(task_failed)?
    .map_err(|e| engine_error("hybrid search", e))?;

    Ok(Json(json!({
        "database": db.name,
        "as_of": at,
        "results": results,
    })))
}

// ---------------------------------------------------------------------------
// POST /v1/context
// ---------------------------------------------------------------------------

/// Heads-up context for a prompt: pinned memories plus the relevant ones,
/// each with its score and the reason it was selected.
async fn context(
    agent: Caller,
    Extension(limits): Limits,
    db: Db,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let query_text = body_str(&req, "query");
    let embedding = query_embedding(&db, &req, query_text.as_deref()).await?;
    let max_items = bounded(
        body_usize(&req, "max_items")?,
        limits.default_context_items,
        limits.max_context_items,
    );
    let max_episodic = body_usize(&req, "max_episodic")?
        .unwrap_or(limits.default_context_episodic)
        .min(max_items);
    let session_id = body_str(&req, "session_id");
    let project = body_str(&req, "project");
    let exclude_ids = string_array(&req, "exclude_ids")?
        .iter()
        .map(|s| {
            s.parse::<MemoryId>()
                .map_err(|_| ApiError::BadRequest(format!("invalid id in 'exclude_ids': {s}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let agent_scope = scope_agent(&agent, parse_optional_uuid(&req, "agent_id")?)?;
    let user_scope = parse_optional_uuid(&req, "user_id")?.map(UserId);

    let handle = db.handle.clone();
    let selected = tokio::task::spawn_blocking(move || {
        let query = InjectionQuery {
            embedding: &embedding,
            query_text: query_text.as_deref(),
            session_id: session_id.as_deref(),
            exclude_ids: &exclude_ids,
            max_items,
            max_episodic,
            agent_id: agent_scope,
            user_id: user_scope,
            current_project: project.as_deref(),
        };
        handle.recall_for_injection(&query)
    })
    .await
    .map_err(task_failed)?
    .map_err(|e| engine_error("context recall", e))?;

    let memories: Vec<Value> = selected
        .iter()
        .map(|c| {
            let reason = match c.reason {
                SelectionReason::Pinned => "pinned",
                SelectionReason::Relevant => "relevant",
                SelectionReason::ModeRule => "mode_rule",
            };
            with_fields(&c.node, json!({ "score": c.score, "reason": reason }))
        })
        .collect();
    Ok(Json(json!({ "database": db.name, "memories": memories })))
}

// ---------------------------------------------------------------------------
// GET /v1/memories/{id}/history
// ---------------------------------------------------------------------------

/// The chain of older facts this memory replaced, newest predecessor first.
async fn history(
    agent: Caller,
    Extension(limits): Limits,
    db: Db,
    Path(params): Path<HashMap<String, String>>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let id = memory_id_param(&params)?;
    let depth = bounded(
        query_usize(&query, "depth")?,
        limits.max_history_depth,
        limits.max_history_depth,
    );
    let node = load_owned(&agent, &db, id)?;
    let handle = db.handle.clone();
    let chain = tokio::task::spawn_blocking(move || handle.supersession_history(id, depth))
        .await
        .map_err(task_failed)?
        .map_err(|e| engine_error("history", e))?;
    Ok(Json(json!({
        "memory": memory_node_to_json(&node),
        "history": chain.iter().map(memory_node_to_json).collect::<Vec<_>>(),
    })))
}

// ---------------------------------------------------------------------------
// GET /v1/memories/{id}/neighbors
// ---------------------------------------------------------------------------

/// Memories within `depth` hops of one memory in the graph, with the edges
/// between them. `edge_types` (comma separated) limits which edges are walked;
/// `direction` is `out`, `in` or `both` (the default).
async fn neighbors(
    agent: Caller,
    Extension(limits): Limits,
    db: Db,
    Path(params): Path<HashMap<String, String>>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let id = memory_id_param(&params)?;
    let depth = bounded(
        query_usize(&query, "depth")?,
        limits.default_neighbor_depth,
        limits.max_neighbor_depth,
    );
    let edge_types: Vec<EdgeType> = match query.get("edge_types").filter(|s| !s.is_empty()) {
        Some(list) => list
            .split(',')
            .map(|s| {
                parse_edge_type(s.trim())
                    .ok_or_else(|| ApiError::BadRequest(format!("unknown edge_type: {s}")))
            })
            .collect::<Result<_, _>>()?,
        None => Vec::new(),
    };
    let (outgoing, incoming) = match query.get("direction").map(String::as_str) {
        None | Some("both") => (true, true),
        Some("out") => (true, false),
        Some("in") => (false, true),
        Some(other) => {
            return Err(ApiError::BadRequest(format!(
                "unknown direction '{other}': use 'out', 'in' or 'both'"
            )));
        }
    };
    let max_nodes = limits.max_neighbors;
    let owner = token_agent(&agent)?;
    let center = load_owned(&agent, &db, id)?;

    let handle = db.handle.clone();
    let (nodes, edges) = tokio::task::spawn_blocking(move || {
        let wanted = |t: EdgeType| edge_types.is_empty() || edge_types.contains(&t);
        // Walk under the graph read lock, then release it before loading
        // memories: the engine's lock order takes the page map first.
        let (reached, edges) = {
            let graph = handle.graph().graph();
            let mut seen: HashSet<MemoryId> = HashSet::from([id]);
            let mut reached: Vec<(MemoryId, usize)> = Vec::new();
            let mut edges: Vec<MemoryEdge> = Vec::new();
            let mut queue = VecDeque::from([(id, 0usize)]);
            while let Some((node, hops)) = queue.pop_front() {
                if hops >= depth {
                    continue;
                }
                let mut steps = Vec::new();
                if outgoing {
                    for (to, e) in graph.outgoing(node) {
                        steps.push((to, stored_edge(node, to, &e)));
                    }
                }
                if incoming {
                    for (from, e) in graph.incoming(node) {
                        steps.push((from, stored_edge(from, node, &e)));
                    }
                }
                for (next, edge) in steps {
                    if !wanted(edge.edge_type) {
                        continue;
                    }
                    if seen.contains(&next) {
                        edges.push(edge);
                        continue;
                    }
                    if reached.len() >= max_nodes {
                        continue;
                    }
                    seen.insert(next);
                    reached.push((next, hops + 1));
                    edges.push(edge);
                    queue.push_back((next, hops + 1));
                }
            }
            (reached, edges)
        };
        let mut dedup = HashSet::new();
        let edges: Vec<MemoryEdge> = edges
            .into_iter()
            .filter(|e| dedup.insert((e.source, e.target, e.edge_type)))
            .collect();
        let nodes: Vec<(MemoryNode, usize)> = reached
            .into_iter()
            .filter_map(|(nid, hops)| handle.get_memory(nid).ok().map(|n| (n, hops)))
            .filter(|(n, _)| visible(owner, n))
            .collect();
        (nodes, edges)
    })
    .await
    .map_err(task_failed)?;

    let shown: HashSet<MemoryId> = nodes
        .iter()
        .map(|(n, _)| n.id)
        .chain(std::iter::once(id))
        .collect();
    Ok(Json(json!({
        "memory": memory_node_to_json(&center),
        "depth": depth,
        "neighbors": nodes
            .iter()
            .map(|(n, hops)| with_fields(n, json!({ "hops": hops })))
            .collect::<Vec<_>>(),
        "edges": edges
            .iter()
            .filter(|e| shown.contains(&e.source) && shown.contains(&e.target))
            .map(edge_to_json)
            .collect::<Vec<_>>(),
    })))
}

// ---------------------------------------------------------------------------
// GET /v1/conflicts
// ---------------------------------------------------------------------------

/// Contradiction and supersession edges, with both memories of each pair.
async fn conflicts(
    agent: Caller,
    Extension(limits): Limits,
    db: Db,
    Query(query): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let limit = list_limit(&query, &limits)?;
    let owner = token_agent(&agent)?;
    let handle = db.handle.clone();
    let (counts, pairs) = tokio::task::spawn_blocking(move || {
        let counts = handle.conflict_edge_counts();
        let mut seen = HashSet::new();
        let mut pairs = Vec::new();
        for (source, target, edge_type, weight) in handle.conflict_edges() {
            if pairs.len() >= limit {
                break;
            }
            // A contradiction is undirected; show each pair once.
            let key = if edge_type == EdgeType::Contradicts && target < source {
                (target, source, edge_type)
            } else {
                (source, target, edge_type)
            };
            if !seen.insert(key) {
                continue;
            }
            let (Ok(a), Ok(b)) = (handle.get_memory(source), handle.get_memory(target)) else {
                continue;
            };
            if visible(owner, &a) && visible(owner, &b) {
                pairs.push(json!({
                    "edge_type": edge_type_name(edge_type),
                    "weight": weight,
                    "source": memory_node_to_json(&a),
                    "target": memory_node_to_json(&b),
                }));
            }
        }
        (counts, pairs)
    })
    .await
    .map_err(task_failed)?;
    Ok(Json(json!({
        "contradicts": counts.0,
        "supersedes": counts.1,
        "conflicts": pairs,
    })))
}

// ---------------------------------------------------------------------------
// Pins: GET /v1/pins, POST and DELETE /v1/memories/{id}/pin
// ---------------------------------------------------------------------------

/// Pinned memories, which every heads-up context includes.
async fn list_pins(
    agent: Caller,
    Extension(limits): Limits,
    db: Db,
    Query(query): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let limit = list_limit(&query, &limits)?;
    let owner = token_agent(&agent)?;
    let handle = db.handle.clone();
    let pins = tokio::task::spawn_blocking(move || {
        let ids = handle.memory_ids_with_tag(PIN_TAG);
        newest_visible(&handle, ids, owner, limit)
    })
    .await
    .map_err(task_failed)?;
    Ok(Json(json!({ "pins": pins })))
}

/// Pin a memory so every heads-up context includes it.
async fn pin(
    agent: Caller,
    db: Db,
    Path(params): Path<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    set_pin(agent, db, params, true).await
}

/// Remove a memory's pin. The memory itself is kept.
async fn unpin(
    agent: Caller,
    db: Db,
    Path(params): Path<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    set_pin(agent, db, params, false).await
}

async fn set_pin(
    agent: Caller,
    db: Db,
    params: HashMap<String, String>,
    pinned: bool,
) -> Result<Json<Value>, ApiError> {
    let id = memory_id_param(&params)?;
    let mut node = load_owned(&agent, &db, id)?;
    if node.tags.iter().any(|t| t == PIN_TAG) != pinned {
        if pinned {
            node.tags.push(PIN_TAG.to_string());
            if !node.tags.iter().any(|t| t == MANUAL_TAG) {
                node.tags.push(MANUAL_TAG.to_string());
            }
        } else {
            node.tags.retain(|t| t != PIN_TAG);
        }
        // Re-storing under the same id replaces the node and re-indexes its
        // tags, the engine's own path for pinning and un-pinning.
        let handle = db.handle.clone();
        tokio::task::spawn_blocking(move || handle.store(node))
            .await
            .map_err(task_failed)?
            .map_err(|e| engine_error("pin", e))?;
    }
    let now = db
        .get_memory(id)
        .map_err(|_| ApiError::NotFound(format!("memory {id} not found")))?;
    let is_pinned = now.tags.iter().any(|t| t == PIN_TAG);
    if is_pinned != pinned {
        // The engine skips a re-store that duplicates another memory of the
        // same owner, such as an identical standing rule already pinned.
        return Err(ApiError::Conflict(format!(
            "memory {id} duplicates another memory of the same owner; its pin was not changed"
        )));
    }
    Ok(Json(with_fields(&now, json!({ "pinned": is_pinned }))))
}

// ---------------------------------------------------------------------------
// GET /v1/profile
// ---------------------------------------------------------------------------

/// The user profile and the facts it is built from, for one owner pair.
async fn profile(
    agent: Caller,
    db: Db,
    Query(query): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let user = query_uuid(&query, "user_id")?
        .map(UserId)
        .unwrap_or_else(UserId::nil);
    let owner = scope_agent(&agent, query_uuid(&query, "agent_id")?)?.unwrap_or_else(AgentId::nil);
    let handle = db.handle.clone();
    let (profile, facts) = tokio::task::spawn_blocking(move || {
        (
            handle.user_profile_for(user, owner),
            handle.profile_facts(user, owner),
        )
    })
    .await
    .map_err(task_failed)?;
    Ok(Json(json!({
        "user_id": user.to_string(),
        "agent_id": owner.to_string(),
        "profile": profile.as_ref().map(memory_node_to_json),
        "facts": facts,
    })))
}

// ---------------------------------------------------------------------------
// GET /v1/entities, GET /v1/entities/{name}/memories
// ---------------------------------------------------------------------------

/// Entities named by entity memories, with how many memories carry each,
/// plus the names the entity resolver knows.
async fn entities(
    agent: Caller,
    Extension(limits): Limits,
    db: Db,
    Query(query): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let limit = list_limit(&query, &limits)?;
    let owner = token_agent(&agent)?;
    let handle = db.handle.clone();
    let (counts, known) = tokio::task::spawn_blocking(move || {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for node in handle.entity_memories() {
            if !visible(owner, &node) {
                continue;
            }
            for tag in &node.tags {
                if let Some(name) = tag.strip_prefix(ENTITY_TAG_PREFIX) {
                    *counts.entry(name.to_string()).or_default() += 1;
                }
            }
        }
        (counts, handle.known_entities())
    })
    .await
    .map_err(task_failed)?;
    let mut names: Vec<(String, usize)> = counts.into_iter().collect();
    names.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    names.truncate(limit);
    Ok(Json(json!({
        "entities": names
            .iter()
            .map(|(name, n)| json!({ "name": name, "memory_count": n }))
            .collect::<Vec<_>>(),
        // The resolver is shared by the whole database, so a caller bound to
        // one agent does not see it.
        "resolver": if owner.is_none() { known } else { Vec::new() },
    })))
}

/// Memories tagged with an entity, after resolving the name to its canonical
/// form.
async fn entity_memories(
    agent: Caller,
    Extension(limits): Limits,
    db: Db,
    Path(params): Path<HashMap<String, String>>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let name = params
        .get("name")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::BadRequest("missing entity name".into()))?;
    let limit = list_limit(&query, &limits)?;
    let owner = token_agent(&agent)?;
    let resolved = db.resolve_entity(&name);
    let mut tags = vec![format!("{ENTITY_TAG_PREFIX}{}", name.to_lowercase())];
    let canonical_tag = format!(
        "{ENTITY_TAG_PREFIX}{}",
        resolved.canonical.to_lowercase().trim()
    );
    if !tags.contains(&canonical_tag) {
        tags.push(canonical_tag);
    }
    let handle = db.handle.clone();
    let memories = tokio::task::spawn_blocking(move || {
        let mut ids: Vec<MemoryId> = Vec::new();
        for tag in &tags {
            for id in handle.memory_ids_with_tag(tag) {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        newest_visible(&handle, ids, owner, limit)
    })
    .await
    .map_err(task_failed)?;
    Ok(Json(json!({
        "name": name,
        "canonical": resolved.canonical,
        "confidence": resolved.confidence,
        "source": format!("{:?}", resolved.source).to_lowercase(),
        "memories": memories,
    })))
}

// ---------------------------------------------------------------------------
// POST /v1/memories/batch
// ---------------------------------------------------------------------------

/// Store many memories in one call. Every item is validated before any is
/// stored. Items without an `embedding` are embedded by the server's embedder
/// when one is configured.
async fn store_batch(
    agent: Caller,
    Extension(limits): Limits,
    db: Db,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let items = req
        .get("memories")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ApiError::BadRequest("missing 'memories' array".into()))?;
    if items.is_empty() {
        return Err(ApiError::BadRequest("'memories' is empty".into()));
    }
    if items.len() > limits.max_batch {
        return Err(ApiError::BadRequest(format!(
            "batch of {} exceeds the limit of {}",
            items.len(),
            limits.max_batch
        )));
    }
    let now = now_micros();
    let nodes = items
        .iter()
        .enumerate()
        .map(|(i, item)| parse_node(&agent, item, now).map_err(|e| at_index(i, e)))
        .collect::<Result<Vec<_>, _>>()?;

    let handle = db.handle.clone();
    let ids = tokio::task::spawn_blocking(move || {
        let mut nodes = nodes;
        let missing: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.embedding.is_empty())
            .map(|(i, _)| i)
            .collect();
        if !missing.is_empty() {
            let texts: Vec<&str> = missing.iter().map(|&i| nodes[i].content.as_str()).collect();
            if let Some(vectors) = handle.embed_batch(&texts)? {
                for (i, v) in missing.into_iter().zip(vectors) {
                    nodes[i].embedding = v;
                }
            }
        }
        handle.store_batch(nodes)
    })
    .await
    .map_err(task_failed)?
    .map_err(|e| match e {
        MenteError::EmbeddingDimensionMismatch { .. } => ApiError::BadRequest(e.to_string()),
        e => engine_error("batch store", e),
    })?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "ids": ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
            "stored": ids.len(),
        })),
    ))
}

/// One batch item as a memory node, with the same fields as `POST /memories`.
fn parse_node(agent: &Caller, req: &Value, now: u64) -> Result<MemoryNode, ApiError> {
    let agent_id = AgentId(parse_uuid(req, "agent_id")?);
    check_body_agent(agent, agent_id)?;
    let memory_type = parse_memory_type(req)?;
    let content = req
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing or invalid 'content'".into()))?
        .to_string();
    let space_id = match req.get("space_id") {
        Some(v) => v
            .as_str()
            .and_then(|s| s.parse::<SpaceId>().ok())
            .ok_or_else(|| ApiError::BadRequest("invalid 'space_id'".into()))?,
        None => SpaceId::nil(),
    };
    let optional_u64 = |field: &str| match req.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| ApiError::BadRequest(format!("'{field}' must be microseconds"))),
    };
    Ok(MemoryNode {
        id: MemoryId::new(),
        agent_id,
        user_id: parse_optional_uuid(req, "user_id")?
            .map(UserId)
            .unwrap_or_else(UserId::nil),
        memory_type,
        embedding: parse_embedding(req, false)?,
        content,
        created_at: now,
        accessed_at: now,
        access_count: 0,
        salience: req.get("salience").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32,
        confidence: req
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32,
        space_id,
        attributes: parse_attributes(req.get("attributes")),
        tags: string_array(req, "tags")?,
        valid_from: optional_u64("valid_from")?,
        valid_until: optional_u64("valid_until")?,
        context: None,
    })
}

fn at_index(i: usize, e: ApiError) -> ApiError {
    match e {
        ApiError::BadRequest(m) => ApiError::BadRequest(format!("memories[{i}]: {m}")),
        ApiError::Forbidden(m) => ApiError::Forbidden(format!("memories[{i}]: {m}")),
        other => other,
    }
}

// ---------------------------------------------------------------------------
// GET /v1/recent
// ---------------------------------------------------------------------------

/// The newest memories, newest first.
async fn recent(
    agent: Caller,
    Extension(limits): Limits,
    db: Db,
    Query(query): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let limit = list_limit(&query, &limits)?;
    let owner = token_agent(&agent)?;
    let handle = db.handle.clone();
    let memories = tokio::task::spawn_blocking(move || {
        handle
            .recall_recent(limit)
            .iter()
            .filter(|n| visible(owner, n))
            .map(memory_node_to_json)
            .collect::<Vec<_>>()
    })
    .await
    .map_err(task_failed)?;
    Ok(Json(json!({ "memories": memories })))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

fn task_failed(e: tokio::task::JoinError) -> ApiError {
    ApiError::Internal(format!("engine task failed: {e}"))
}

fn engine_error(what: &str, e: MenteError) -> ApiError {
    error!("{what} failed: {e}");
    ApiError::Internal(format!("{what} failed: {e}"))
}

/// `value` when given (at least 1), else `default`, never above `max`.
fn bounded(value: Option<usize>, default: usize, max: usize) -> usize {
    value.unwrap_or(default).clamp(1, max.max(1))
}

fn list_limit(
    query: &HashMap<String, String>,
    limits: &MemoryApiLimits,
) -> Result<usize, ApiError> {
    Ok(bounded(
        query_usize(query, "limit")?,
        limits.default_list,
        limits.max_list,
    ))
}

fn query_usize(query: &HashMap<String, String>, field: &str) -> Result<Option<usize>, ApiError> {
    query
        .get(field)
        .map(|s| {
            s.parse()
                .map_err(|_| ApiError::BadRequest(format!("'{field}' must be a whole number")))
        })
        .transpose()
}

fn query_uuid(query: &HashMap<String, String>, field: &str) -> Result<Option<Uuid>, ApiError> {
    query
        .get(field)
        .filter(|s| !s.is_empty())
        .map(|s| {
            Uuid::parse_str(s)
                .map_err(|_| ApiError::BadRequest(format!("invalid UUID for '{field}'")))
        })
        .transpose()
}

fn body_usize(req: &Value, field: &str) -> Result<Option<usize>, ApiError> {
    req.get(field)
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| ApiError::BadRequest(format!("'{field}' must be a whole number")))
        })
        .transpose()
}

fn body_str(req: &Value, field: &str) -> Option<String> {
    req.get(field)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from)
}

fn string_array(req: &Value, field: &str) -> Result<Vec<String>, ApiError> {
    match req.get(field) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(String::from)
                    .ok_or_else(|| ApiError::BadRequest(format!("'{field}' must hold strings")))
            })
            .collect(),
        Some(_) => Err(ApiError::BadRequest(format!("'{field}' must be an array"))),
    }
}

/// The query vector: the raw `embedding` when sent, else the `query` text
/// embedded by the server.
async fn query_embedding(db: &Db, req: &Value, text: Option<&str>) -> Result<Vec<f32>, ApiError> {
    match (req.get("embedding"), text) {
        (Some(_), _) => parse_embedding(req, true),
        (None, Some(text)) => embed_query(db, text).await,
        (None, None) => Err(ApiError::BadRequest(
            "send 'query' text or an 'embedding' vector".into(),
        )),
    }
}

/// The agent the token is bound to, unless it is an admin token.
fn token_agent(agent: &Caller) -> Result<Option<AgentId>, ApiError> {
    match agent {
        Some(Extension(authed)) if !authed.admin => authed.agent(),
        _ => Ok(None),
    }
}

/// The agent a read is scoped to: the requested one (which must match a
/// token bound to an agent), else the token's agent, else none.
fn scope_agent(agent: &Caller, requested: Option<Uuid>) -> Result<Option<AgentId>, ApiError> {
    match requested {
        Some(id) => {
            check_body_agent(agent, AgentId(id))?;
            Ok(Some(AgentId(id)))
        }
        None => token_agent(agent),
    }
}

/// Whether a caller bound to `owner` may see `node`: its own memories and
/// shared ones owned by no agent.
fn visible(owner: Option<AgentId>, node: &MemoryNode) -> bool {
    owner.is_none_or(|a| node.agent_id == a || node.agent_id.is_nil())
}

/// Load a memory and check the caller may use it.
fn load_owned(agent: &Caller, db: &Db, id: MemoryId) -> Result<MemoryNode, ApiError> {
    let node = db
        .get_memory(id)
        .map_err(|_| ApiError::NotFound(format!("memory {id} not found")))?;
    check_owner(agent, &node)?;
    Ok(node)
}

/// Load `ids`, keep what the caller may see, newest first, at most `limit`.
fn newest_visible(
    db: &mentedb::MenteDb,
    ids: Vec<MemoryId>,
    owner: Option<AgentId>,
    limit: usize,
) -> Vec<Value> {
    let mut nodes: Vec<MemoryNode> = ids
        .into_iter()
        .filter_map(|id| db.get_memory(id).ok())
        .filter(|n| visible(owner, n))
        .collect();
    nodes.sort_by_key(|n| std::cmp::Reverse(n.created_at));
    nodes.iter().take(limit).map(memory_node_to_json).collect()
}

/// A memory as JSON with extra fields merged in.
fn with_fields(node: &MemoryNode, extra: Value) -> Value {
    let mut out = memory_node_to_json(node);
    if let (Value::Object(o), Value::Object(e)) = (&mut out, extra) {
        o.extend(e);
    }
    out
}

fn stored_edge(source: MemoryId, target: MemoryId, e: &mentedb::graph::StoredEdge) -> MemoryEdge {
    MemoryEdge {
        source,
        target,
        edge_type: e.edge_type,
        weight: e.weight,
        created_at: e.created_at,
        valid_from: e.valid_from,
        valid_until: e.valid_until,
        label: e.label.clone(),
    }
}

fn edge_type_name(t: EdgeType) -> String {
    format!("{t:?}").to_lowercase()
}

fn edge_to_json(e: &MemoryEdge) -> Value {
    json!({
        "source": e.source.to_string(),
        "target": e.target.to_string(),
        "edge_type": edge_type_name(e.edge_type),
        "weight": e.weight,
        "created_at": e.created_at,
        "valid_from": e.valid_from,
        "valid_until": e.valid_until,
        "label": e.label,
    })
}
