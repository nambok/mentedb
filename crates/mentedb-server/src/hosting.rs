//! Endpoints and middleware for running the server as a shared, multi tenant
//! service:
//!
//! Per database (mounted under `/v1` and `/v1/db/{db}`):
//!
//! - `GET /memories`: a paged, token scoped list, narrowed by tag through the
//!   bitmap index, in a stable order.
//! - `PATCH /memories/{id}`: update attributes, tags and `valid_until` in
//!   place, keeping the id.
//! - `POST /memories/{id}/supersede`: replace a memory with a new one, closing
//!   the old one and linking the two with a Supersedes edge.
//! - `POST /recall_for_action`: memories that bear on an action about to
//!   happen.
//! - `GET /consolidation_candidates`: clusters of near duplicates (read only).
//! - `POST /consolidate`: merge exactly the given memories.
//!
//! Global:
//!
//! - `DELETE /v1/db/{db}`: drop a database (admin, or a maintenance token for
//!   it).
//! - `POST /v1/batch/search`: one query across several databases.
//! - `GET /v1/live`: liveness, touching nothing.
//!
//! Middleware: request deadlines (`x-mentedb-deadline-ms` and
//! `MENTEDB_REQUEST_TIMEOUT_MS`). Also the bounded enrichment worker pool.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use mentedb::injection::InjectionQuery;
use mentedb::{HybridOptions, MemoryPatch};
use mentedb_core::MemoryNode;
use mentedb_core::error::MenteError;
use mentedb_core::types::{AgentId, MemoryId, UserId};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::auth::AuthenticatedAgent;
use crate::databases::{DEFAULT_DB, Db, validate_name};
use crate::error::ApiError;
use crate::handlers::{
    check_body_agent, check_owner, embed_query, memory_id_param, memory_node_to_json,
    memory_node_to_json_with, parse_attributes, parse_embedding, parse_memory_type,
    parse_optional_uuid, wants_embedding,
};
use crate::state::AppState;

type Caller = Option<Extension<AuthenticatedAgent>>;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Settings for the hosting endpoints and middleware.
#[derive(Clone, Debug)]
pub struct HostingConfig {
    /// Page size of `GET /memories` when `limit` is not given.
    pub list_default_limit: usize,
    /// Largest page `GET /memories` returns (`MENTEDB_LIST_MAX_LIMIT`).
    pub list_max_limit: usize,
    /// Most databases one batch search may name (`MENTEDB_BATCH_MAX_DATABASES`).
    pub batch_max_databases: usize,
    /// Results per database when a batch search gives no `k`.
    pub batch_default_k: usize,
    /// Most results per database one batch search may ask for.
    pub batch_max_k: usize,
    /// Results of recall_for_action when no `k` is given.
    pub action_default_k: usize,
    /// Most results one recall_for_action may ask for.
    pub action_max_k: usize,
    /// Upper bound on every request (`MENTEDB_REQUEST_TIMEOUT_MS`); a
    /// request's own deadline header can only shorten it. `None`: no bound.
    pub request_timeout: Option<Duration>,
    /// Threads in the enrichment pool, which is also the most enrichment runs
    /// at once (`MENTEDB_ENRICH_WORKERS`).
    pub enrich_workers: usize,
    /// Named database `/v1/ready` opens (and so locks) to prove databases can
    /// be opened (`MENTEDB_READY_PROBE_DB`; empty turns the check off).
    pub ready_probe_db: Option<String>,
}

impl Default for HostingConfig {
    fn default() -> Self {
        Self {
            list_default_limit: 50,
            list_max_limit: 1000,
            batch_max_databases: 32,
            batch_default_k: 10,
            batch_max_k: 200,
            action_default_k: 10,
            action_max_k: 100,
            request_timeout: None,
            enrich_workers: 1,
            ready_probe_db: Some("readiness-probe".to_string()),
        }
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

impl HostingConfig {
    /// Defaults overridden by the environment.
    pub fn from_env() -> Self {
        let mut c = Self::default();
        if let Some(n) = env_usize("MENTEDB_LIST_MAX_LIMIT").filter(|n| *n > 0) {
            c.list_max_limit = n;
        }
        if let Some(n) = env_usize("MENTEDB_BATCH_MAX_DATABASES").filter(|n| *n > 0) {
            c.batch_max_databases = n;
        }
        if let Some(n) = env_usize("MENTEDB_REQUEST_TIMEOUT_MS") {
            c.request_timeout = (n > 0).then(|| Duration::from_millis(n as u64));
        }
        if let Some(n) = env_usize("MENTEDB_ENRICH_WORKERS").filter(|n| *n > 0) {
            c.enrich_workers = n;
        }
        if let Ok(v) = std::env::var("MENTEDB_READY_PROBE_DB") {
            let v = v.trim().to_string();
            c.ready_probe_db = (!v.is_empty()).then_some(v);
        }
        c
    }
}

/// The hosting settings, read from the environment once.
pub fn hosting_config() -> &'static HostingConfig {
    static CONFIG: OnceLock<HostingConfig> = OnceLock::new();
    CONFIG.get_or_init(HostingConfig::from_env)
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/// Routes on one database that are new paths. `GET /memories` and
/// `PATCH /memories/{id}` are added to the existing paths in `routes.rs`.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/memories/{id}/supersede", post(supersede_memory))
        .route("/recall_for_action", post(recall_for_action))
        .route("/consolidation_candidates", get(consolidation_candidates))
        .route("/consolidate", post(consolidate))
}

/// Routes outside any one database.
pub fn global_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/db/{db}", axum::routing::delete(drop_database))
        .route("/v1/batch/search", post(batch_search))
        .route("/v1/live", get(live))
}

// ---------------------------------------------------------------------------
// Deadlines
// ---------------------------------------------------------------------------

/// Header carrying the caller's budget for a request, in milliseconds.
pub const DEADLINE_HEADER: &str = "x-mentedb-deadline-ms";

/// When the current request must be answered by. Long handlers check it
/// between steps and stop with 504 once it has passed.
#[derive(Clone, Copy, Debug)]
pub struct Deadline(pub Instant);

impl Deadline {
    /// 504 once the deadline has passed.
    pub fn check(&self) -> Result<(), ApiError> {
        if Instant::now() >= self.0 {
            Err(ApiError::GatewayTimeout(
                "the request deadline passed before the work finished".into(),
            ))
        } else {
            Ok(())
        }
    }
}

fn check_deadline(deadline: &Option<Extension<Deadline>>) -> Result<(), ApiError> {
    match deadline {
        Some(Extension(d)) => d.check(),
        None => Ok(()),
    }
}

/// Bound every request by the smaller of the server's request timeout (the
/// middleware state) and the caller's `x-mentedb-deadline-ms`. Past it the
/// request is answered with 504; the handler sees the [`Deadline`] and its
/// long loops stop early. A malformed header is a 400.
pub async fn deadline_middleware(
    State(server_timeout): State<Option<Duration>>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    let requested = match request.headers().get(DEADLINE_HEADER) {
        None => None,
        Some(v) => match v.to_str().ok().and_then(|s| s.trim().parse::<u64>().ok()) {
            Some(ms) => Some(Duration::from_millis(ms)),
            None => {
                return ApiError::BadRequest(format!(
                    "{DEADLINE_HEADER} must be a whole number of milliseconds"
                ))
                .into_response();
            }
        },
    };
    let budget = match (requested, server_timeout) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    let Some(budget) = budget else {
        return next.run(request).await;
    };
    request
        .extensions_mut()
        .insert(Deadline(Instant::now() + budget));
    match tokio::time::timeout(budget, next.run(request)).await {
        Ok(response) => response,
        Err(_) => ApiError::GatewayTimeout(format!("no answer within {} ms", budget.as_millis()))
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

async fn blocking<T, F>(what: &'static str, f: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, ApiError> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| ApiError::Internal(format!("{what} task failed: {e}")))?
}

fn engine(what: &'static str) -> impl FnOnce(MenteError) -> ApiError {
    move |e| match e {
        MenteError::MemoryNotFound(id) => ApiError::NotFound(format!("memory {id} not found")),
        other => ApiError::Internal(format!("{what} failed: {other}")),
    }
}

/// The agent a token is bound to, unless it is an admin token.
fn token_agent(agent: &Caller) -> Result<Option<AgentId>, ApiError> {
    match agent {
        Some(Extension(a)) if !a.admin => a.agent(),
        _ => Ok(None),
    }
}

/// The agent a read is scoped to: the requested one (which must match a token
/// bound to an agent), else the token's agent, else none.
fn scope_agent(agent: &Caller, requested: Option<Uuid>) -> Result<Option<AgentId>, ApiError> {
    match requested {
        Some(id) => {
            check_body_agent(agent, AgentId(id))?;
            Ok(Some(AgentId(id)))
        }
        None => token_agent(agent),
    }
}

fn visible(owner: Option<AgentId>, node: &MemoryNode) -> bool {
    owner.is_none_or(|a| node.agent_id == a || node.agent_id.is_nil())
}

fn bounded(value: Option<usize>, default: usize, max: usize) -> usize {
    value.unwrap_or(default).clamp(1, max.max(1))
}

fn body_usize(req: &Value, field: &str) -> Result<Option<usize>, ApiError> {
    req.get(field)
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| ApiError::BadRequest(format!("'{field}' must be a whole number")))
        })
        .transpose()
}

fn query_usize(q: &HashMap<String, String>, field: &str) -> Result<Option<usize>, ApiError> {
    q.get(field)
        .map(|s| {
            s.trim()
                .parse()
                .map_err(|_| ApiError::BadRequest(format!("'{field}' must be a whole number")))
        })
        .transpose()
}

fn query_bool(q: &HashMap<String, String>, field: &str) -> Result<bool, ApiError> {
    match q.get(field).map(|s| s.trim().to_lowercase()) {
        None => Ok(false),
        Some(v) if v == "1" || v == "true" => Ok(true),
        Some(v) if v == "0" || v == "false" || v.is_empty() => Ok(false),
        Some(_) => Err(ApiError::BadRequest(format!("'{field}' must be 0 or 1"))),
    }
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

fn load_owned(agent: &Caller, db: &Db, id: MemoryId) -> Result<MemoryNode, ApiError> {
    let node = db
        .get_memory(id)
        .map_err(|_| ApiError::NotFound(format!("memory {id} not found")))?;
    check_owner(agent, &node)?;
    Ok(node)
}

// ---------------------------------------------------------------------------
// GET /memories
// ---------------------------------------------------------------------------

/// A page of the caller's memories. Query: `limit`, `offset`, `tag`,
/// `include_inactive` (0 or 1), `fields=embedding`. A token bound to an agent
/// sees its own memories and shared ones owned by no agent.
pub async fn list_memories(
    agent: Caller,
    deadline: Option<Extension<Deadline>>,
    db: Db,
    Query(q): Query<HashMap<String, String>>,
) -> Result<Json<Value>, ApiError> {
    let cfg = hosting_config();
    let limit = bounded(
        query_usize(&q, "limit")?,
        cfg.list_default_limit,
        cfg.list_max_limit,
    );
    let offset = query_usize(&q, "offset")?.unwrap_or(0);
    let include_inactive = query_bool(&q, "include_inactive")?;
    let tag = q.get("tag").filter(|t| !t.is_empty()).cloned();
    let owner = token_agent(&agent)?;
    let embedding = wants_embedding(&q);
    check_deadline(&deadline)?;
    let handle = db.handle.clone();
    let page = blocking("list memories", move || {
        Ok(handle.list_memories_page(
            tag.as_deref(),
            owner,
            include_inactive,
            now_micros(),
            offset,
            limit,
        ))
    })
    .await?;
    check_deadline(&deadline)?;
    let memories: Vec<Value> = page
        .memories
        .iter()
        .map(|n| memory_node_to_json_with(n, embedding))
        .collect();
    let next_offset = page.has_more.then_some(offset + memories.len());
    Ok(Json(json!({
        "database": db.name,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
        "memories": memories,
    })))
}

// ---------------------------------------------------------------------------
// PATCH /memories/{id}
// ---------------------------------------------------------------------------

/// Update a memory in place. Body: `attributes` (merged), `tags` (replaces),
/// `valid_until` (microseconds, or null to clear). The id is kept.
pub async fn patch_memory(
    agent: Caller,
    db: Db,
    Path(params): Path<HashMap<String, String>>,
    Json(req): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let id = memory_id_param(&params)?;
    load_owned(&agent, &db, id)?;
    let mut patch = MemoryPatch::default();
    match req.get("attributes") {
        None | Some(Value::Null) => {}
        Some(v @ Value::Object(_)) => patch.attributes = parse_attributes(Some(v)),
        Some(_) => {
            return Err(ApiError::BadRequest(
                "'attributes' must be an object".into(),
            ));
        }
    }
    if req.get("tags").is_some_and(|v| !v.is_null()) {
        patch.tags = Some(string_array(&req, "tags")?);
    }
    if let Some(v) = req.get("valid_until") {
        patch.valid_until = Some(match v {
            Value::Null => None,
            v => Some(v.as_u64().ok_or_else(|| {
                ApiError::BadRequest("'valid_until' must be microseconds or null".into())
            })?),
        });
    }
    if patch.attributes.is_empty() && patch.tags.is_none() && patch.valid_until.is_none() {
        return Err(ApiError::BadRequest(
            "nothing to update: send attributes, tags or valid_until".into(),
        ));
    }
    let handle = db.handle.clone();
    let node = blocking("update memory", move || {
        handle
            .update_memory_fields(id, &patch)
            .map_err(engine("update memory"))
    })
    .await?;
    Ok(Json(memory_node_to_json(&node)))
}

// ---------------------------------------------------------------------------
// POST /memories/{id}/supersede
// ---------------------------------------------------------------------------

/// Replace a memory. Body: `content` (required), `memory_type` (defaults to
/// the old memory's), `attributes`, `tags`, `embedding`. The new memory keeps
/// the old one's owners and space; the old one is closed now and linked from
/// the new one by a Supersedes edge.
pub async fn supersede_memory(
    agent: Caller,
    db: Db,
    Path(params): Path<HashMap<String, String>>,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let old_id = memory_id_param(&params)?;
    let old = load_owned(&agent, &db, old_id)?;
    let content = req
        .get("content")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::BadRequest("missing 'content'".into()))?
        .to_string();
    let memory_type = if req.get("memory_type").is_some() {
        parse_memory_type(&req)?
    } else {
        old.memory_type
    };
    let mut node = MemoryNode::new(
        old.agent_id,
        memory_type,
        content,
        parse_embedding(&req, false)?,
    )
    .with_user_id(old.user_id);
    node.space_id = old.space_id;
    node.attributes = parse_attributes(req.get("attributes"));
    node.tags = string_array(&req, "tags")?;
    let new_id = node.id;
    let handle = db.handle.clone();
    let at = now_micros();
    let stored = blocking("supersede memory", move || {
        handle
            .supersede_memory(old_id, node, at)
            .map_err(|e| match e {
                MenteError::MemoryNotFound(id) => {
                    ApiError::NotFound(format!("memory {id} not found"))
                }
                MenteError::Storage(msg) => ApiError::Conflict(msg),
                other => ApiError::Internal(format!("supersede failed: {other}")),
            })
    })
    .await?;
    debug_assert_eq!(stored, new_id);
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": stored.to_string(),
            "supersedes": old_id.to_string(),
            "superseded_at": at,
        })),
    ))
}

// ---------------------------------------------------------------------------
// POST /recall_for_action
// ---------------------------------------------------------------------------

/// Memories that bear on an action about to happen. Body: `action` (text),
/// optional `embedding` (else the action is embedded), `k`, `agent_id`,
/// `user_id`.
pub async fn recall_for_action(
    agent: Caller,
    db: Db,
    Json(req): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let cfg = hosting_config();
    let action = req
        .get("action")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from);
    let embedding = match (req.get("embedding"), &action) {
        (Some(_), _) => parse_embedding(&req, true)?,
        (None, Some(text)) => embed_query(&db, text).await?,
        (None, None) => {
            return Err(ApiError::BadRequest(
                "send 'action' text or an 'embedding' vector".into(),
            ));
        }
    };
    let k = bounded(
        body_usize(&req, "k")?,
        cfg.action_default_k,
        cfg.action_max_k,
    );
    let agent_scope = scope_agent(&agent, parse_optional_uuid(&req, "agent_id")?)?;
    let user_scope = parse_optional_uuid(&req, "user_id")?.map(UserId);
    let handle = db.handle.clone();
    let nodes = blocking("recall for action", move || {
        handle
            .recall_for_action(&embedding, action.as_deref(), agent_scope, user_scope, k)
            .map_err(engine("recall for action"))
    })
    .await?;
    let memories: Vec<Value> = nodes.iter().map(memory_node_to_json).collect();
    Ok(Json(json!({ "database": db.name, "memories": memories })))
}

// ---------------------------------------------------------------------------
// GET /consolidation_candidates, POST /consolidate
// ---------------------------------------------------------------------------

/// Clusters of near duplicate memories that could be merged. Read only.
/// Query: `min_cluster` (default from the server's consolidation settings),
/// `similarity` (0 to 1). A token bound to an agent sees only clusters made
/// entirely of memories it may see.
pub async fn consolidation_candidates(
    agent: Caller,
    deadline: Option<Extension<Deadline>>,
    db: Db,
    Query(q): Query<HashMap<String, String>>,
) -> Result<Json<Value>, ApiError> {
    let ops = crate::handlers_ops::ops_config();
    let min_cluster = query_usize(&q, "min_cluster")?
        .unwrap_or(ops.consolidation_min_cluster_size)
        .max(2);
    let similarity = match q.get("similarity") {
        None => ops.consolidation_similarity,
        Some(s) => s
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|v| (0.0..=1.0).contains(v))
            .ok_or_else(|| ApiError::BadRequest("'similarity' must be between 0 and 1".into()))?,
    };
    let owner = token_agent(&agent)?;
    check_deadline(&deadline)?;
    let handle = db.handle.clone();
    let clusters = blocking("consolidation candidates", move || {
        let found = handle
            .find_consolidation_candidates(min_cluster, similarity)
            .map_err(engine("consolidation candidates"))?;
        Ok(found
            .into_iter()
            .filter_map(|c| {
                let nodes: Vec<MemoryNode> = c
                    .memories
                    .iter()
                    .filter_map(|id| handle.get_memory(*id).ok())
                    .collect();
                (nodes.len() == c.memories.len() && nodes.iter().all(|n| visible(owner, n))).then(
                    || {
                        json!({
                            "topic": c.topic,
                            "avg_similarity": c.avg_similarity,
                            "memories": nodes.iter().map(memory_node_to_json).collect::<Vec<_>>(),
                        })
                    },
                )
            })
            .collect::<Vec<Value>>())
    })
    .await?;
    check_deadline(&deadline)?;
    Ok(Json(json!({
        "database": db.name,
        "min_cluster": min_cluster,
        "similarity": similarity,
        "clusters": clusters,
    })))
}

/// Merge exactly the given memories. Body: `ids` (two or more). The sources
/// are closed, not deleted, and a new semantic memory links back to them.
pub async fn consolidate(
    agent: Caller,
    db: Db,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let ids: Vec<MemoryId> = string_array(&req, "ids")?
        .iter()
        .map(|s| {
            s.parse()
                .map_err(|_| ApiError::BadRequest(format!("invalid id in 'ids': {s}")))
        })
        .collect::<Result<_, _>>()?;
    let mut unique = ids.clone();
    unique.sort();
    unique.dedup();
    if unique.len() < 2 {
        return Err(ApiError::BadRequest(
            "'ids' must name at least two different memories".into(),
        ));
    }
    for id in &unique {
        load_owned(&agent, &db, *id)?;
    }
    let handle = db.handle.clone();
    let merged = blocking("consolidate", move || {
        handle
            .consolidate_cluster(&unique)
            .map_err(engine("consolidate"))
    })
    .await?;
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "database": db.name,
            "id": merged.to_string(),
            "consolidated": ids.iter().map(|i| i.to_string()).collect::<Vec<_>>(),
        })),
    ))
}

// ---------------------------------------------------------------------------
// DELETE /v1/db/{db}
// ---------------------------------------------------------------------------

/// Whether the request carries the configured admin key.
fn has_admin_key(headers: &HeaderMap, state: &AppState) -> bool {
    crate::auth::admin_authorized(headers, &state.admin_key).is_ok()
}

/// Allow a database maintenance operation (drop, backup) on `name`: the
/// admin key, an admin token, or a token with the maintenance scope whose
/// `dbs` names the database. With auth off, everything is allowed.
pub fn authorize_maintenance(
    state: &AppState,
    headers: &HeaderMap,
    agent: Option<&AuthenticatedAgent>,
    name: &str,
) -> Result<(), ApiError> {
    if has_admin_key(headers, state) {
        return Ok(());
    }
    match agent {
        Some(a) => a.authorize_maintenance(name),
        None if state.verifier.is_none() && state.admin_key.is_none() => Ok(()),
        None => Err(ApiError::Unauthorized(
            "send the admin key, or a token with the maintenance scope for this database".into(),
        )),
    }
}

/// Close and permanently delete a named database.
pub async fn drop_database(
    State(state): State<Arc<AppState>>,
    agent: Caller,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<Json<Value>, ApiError> {
    if name == DEFAULT_DB {
        return Err(ApiError::BadRequest(
            "the default database cannot be deleted".into(),
        ));
    }
    validate_name(&name)?;
    authorize_maintenance(&state, &headers, agent.as_ref().map(|e| &e.0), &name)?;
    let dbs = state.databases.clone();
    let target = name.clone();
    blocking("drop database", move || dbs.delete(&target)).await?;
    Ok(Json(json!({ "name": name, "status": "deleted" })))
}

// ---------------------------------------------------------------------------
// POST /v1/batch/search
// ---------------------------------------------------------------------------

/// Hybrid search across several databases in one call, the query embedded
/// once. Body: `databases` (names), `query` and/or `embedding`, `k`,
/// `context` (also return each database's heads up context), `max_items`
/// (context size), `agent_id`, `user_id`. Every database must be readable by
/// the token, or the whole request is refused. A database that does not exist
/// is reported under `errors` without stopping the others.
pub async fn batch_search(
    State(state): State<Arc<AppState>>,
    agent: Caller,
    deadline: Option<Extension<Deadline>>,
    Json(req): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let cfg = hosting_config();
    let names = string_array(&req, "databases")?;
    if names.is_empty() {
        return Err(ApiError::BadRequest(
            "'databases' must name at least one".into(),
        ));
    }
    if names.len() > cfg.batch_max_databases {
        return Err(ApiError::BadRequest(format!(
            "at most {} databases per batch",
            cfg.batch_max_databases
        )));
    }
    for name in &names {
        if name != DEFAULT_DB {
            validate_name(name)?;
        }
        if let Some(Extension(a)) = &agent {
            a.authorize_db(name, false)?;
        }
    }
    let k = bounded(body_usize(&req, "k")?, cfg.batch_default_k, cfg.batch_max_k);
    let with_context = req
        .get("context")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let max_items = bounded(body_usize(&req, "max_items")?, 10, 100);
    let query_text = req
        .get("query")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from);
    let agent_scope = scope_agent(&agent, parse_optional_uuid(&req, "agent_id")?)?;
    let user_scope = parse_optional_uuid(&req, "user_id")?.map(UserId);

    // One embedding for every database: they share the server's embedder.
    let embedding = match (req.get("embedding"), &query_text) {
        (Some(_), _) => parse_embedding(&req, true)?,
        (None, Some(text)) => {
            let embedder = state.databases.embedder().ok_or_else(|| {
                ApiError::BadRequest(
                    "this server has no embedder; send an 'embedding' vector instead".into(),
                )
            })?;
            let text = text.clone();
            blocking("embed query", move || {
                embedder
                    .embed(&text)
                    .map_err(|e| ApiError::ServiceUnavailable(format!("embedding failed: {e}")))
            })
            .await?
        }
        (None, None) => {
            return Err(ApiError::BadRequest(
                "send 'query' text or an 'embedding' vector".into(),
            ));
        }
    };
    let embedding = Arc::new(embedding);

    let mut results = Vec::new();
    let mut errors = serde_json::Map::new();
    for name in names {
        check_deadline(&deadline)?;
        let dbs = state.databases.clone();
        let emb = embedding.clone();
        let text = query_text.clone();
        let lookup = name.clone();
        let outcome = blocking("batch search", move || {
            if !dbs.exists(&lookup) {
                return Ok(Err(format!("database '{lookup}' not found")));
            }
            let handle = dbs.get(&lookup, false)?;
            let hits = handle
                .recall_hybrid_scoped_with(
                    &emb,
                    text.as_deref(),
                    k,
                    now_micros(),
                    None,
                    false,
                    None,
                    agent_scope,
                    user_scope,
                    None,
                    &HybridOptions {
                        record_access: true,
                        ..HybridOptions::default()
                    },
                )
                .map_err(engine("hybrid search"))?;
            let found: Vec<Value> = hits
                .into_iter()
                .filter_map(|(id, score)| {
                    let node = handle.get_memory(id).ok()?;
                    let mut j = memory_node_to_json(&node);
                    if let Value::Object(m) = &mut j {
                        m.insert("score".into(), json!(score));
                    }
                    Some(j)
                })
                .collect();
            let context = if with_context {
                let selected = handle
                    .recall_for_injection(&InjectionQuery {
                        embedding: &emb,
                        query_text: text.as_deref(),
                        session_id: None,
                        exclude_ids: &[],
                        max_items,
                        max_episodic: max_items.min(3),
                        agent_id: agent_scope,
                        user_id: user_scope,
                        current_project: None,
                    })
                    .map_err(engine("context recall"))?;
                Some(
                    selected
                        .iter()
                        .map(|c| {
                            let mut j = memory_node_to_json(&c.node);
                            if let Value::Object(m) = &mut j {
                                m.insert("score".into(), json!(c.score));
                            }
                            j
                        })
                        .collect::<Vec<Value>>(),
                )
            } else {
                None
            };
            Ok(Ok((found, context)))
        })
        .await?;
        match outcome {
            Ok((found, context)) => {
                let mut entry = json!({ "database": name, "results": found });
                if let (Some(ctx), Value::Object(m)) = (context, &mut entry) {
                    m.insert("context".into(), json!(ctx));
                }
                results.push(entry);
            }
            Err(msg) => {
                errors.insert(name, json!(msg));
            }
        }
    }
    Ok(Json(json!({ "results": results, "errors": errors })))
}

// ---------------------------------------------------------------------------
// GET /v1/live
// ---------------------------------------------------------------------------

/// Liveness: the process answers. Touches no database, disk or embedder, so a
/// slow dependency never gets a healthy process restarted. Point liveness
/// probes here and readiness probes at `/v1/ready`.
pub async fn live() -> Json<Value> {
    Json(json!({ "status": "alive" }))
}

// ---------------------------------------------------------------------------
// Enrichment pool
// ---------------------------------------------------------------------------

/// Enrichment runs on its own small set of threads with bounded concurrency,
/// so a long LLM pipeline never occupies the threads that serve reads.
struct EnrichmentPool {
    runtime: tokio::runtime::Runtime,
    permits: Arc<tokio::sync::Semaphore>,
}

static ENRICHMENT_POOL: OnceLock<Result<EnrichmentPool, String>> = OnceLock::new();

fn enrichment_pool() -> Result<&'static EnrichmentPool, ApiError> {
    ENRICHMENT_POOL
        .get_or_init(|| {
            let workers = hosting_config().enrich_workers.max(1);
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(workers)
                .thread_name("mentedb-enrich")
                .enable_all()
                .build()
                .map(|runtime| EnrichmentPool {
                    runtime,
                    permits: Arc::new(tokio::sync::Semaphore::new(workers)),
                })
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| ApiError::Internal(format!("enrichment pool unavailable: {e}")))
}

/// Run `job` on the enrichment pool, waiting for a free slot first, and
/// return its output.
pub async fn run_on_enrichment_pool<F, T>(job: F) -> Result<T, ApiError>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let pool = enrichment_pool()?;
    let permits = pool.permits.clone();
    pool.runtime
        .spawn(async move {
            // The semaphore is never closed, so acquire cannot fail; a closed
            // one would just run the job unbounded.
            let _slot = permits.acquire_owned().await.ok();
            job.await
        })
        .await
        .map_err(|e| ApiError::Internal(format!("enrichment task failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn query_bool_accepts_zero_and_one() {
        let q = |v: &str| HashMap::from([("x".to_string(), v.to_string())]);
        assert!(query_bool(&q("1"), "x").unwrap());
        assert!(!query_bool(&q("0"), "x").unwrap());
        assert!(!query_bool(&HashMap::new(), "x").unwrap());
        assert!(query_bool(&q("yes please"), "x").is_err());
    }

    #[tokio::test]
    async fn enrichment_pool_bounds_concurrency_and_runs_off_the_caller() {
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut jobs = Vec::new();
        for _ in 0..4 {
            let (running, peak) = (running.clone(), peak.clone());
            jobs.push(tokio::spawn(run_on_enrichment_pool(async move {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(30)).await;
                running.fetch_sub(1, Ordering::SeqCst);
                std::thread::current().name().map(String::from)
            })));
        }
        for j in jobs {
            let name = j.await.unwrap().unwrap();
            assert_eq!(name.as_deref(), Some("mentedb-enrich"));
        }
        assert_eq!(
            peak.load(Ordering::SeqCst),
            hosting_config().enrich_workers.max(1)
        );
    }
}
