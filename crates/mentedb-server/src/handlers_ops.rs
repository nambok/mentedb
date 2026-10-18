//! Operations endpoints: maintenance, enrichment, documents, backups, database management, readiness.
//!
//! Per database (mounted under `/v1` and `/v1/db/{db}`):
//!
//! - `POST /maintenance/decay`, `/maintenance/archive`, `/maintenance/consolidate`,
//!   `/maintenance/sweep`: the rule based upkeep jobs, on demand.
//! - `POST /enrich`: the LLM enrichment pipeline (extraction, entity linking,
//!   communities, user model).
//! - `POST /documents`, `DELETE /documents/{document_id}`: load a long text as
//!   overlapping chunks, and forget it again.
//! - `POST /backup`, `GET /backups` (admin): snapshot the database into a
//!   single backup file and list those snapshots.
//!
//! Global:
//!
//! - `GET`/`POST /v1/databases`, `DELETE /v1/databases/{name}` (admin).
//! - `GET /v1/ready`: readiness, separate from `/v1/health` liveness.
//!
//! There is deliberately no restore endpoint: replacing the files of a live,
//! open database is unsafe. Restore is offline: stop the server, then call
//! `mentedb::storage::backup::BackupManager::restore_backup(file, dir)` with
//! an empty target directory (the data directory root for the default
//! database, `<data>/dbs/<name>` for a named one) and start the server again.

use std::collections::HashMap;
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{FromRequestParts, Path, State};
use axum::http::StatusCode;
use axum::http::request::Parts;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Extension, Json, Router};
use mentedb::consolidation::archival::ArchivalDecision;
use mentedb::storage::backup::BackupManager;
use mentedb_core::MemoryNode;
use mentedb_core::memory::{AttributeValue, MemoryType};
use mentedb_core::types::{AgentId, MemoryId, SpaceId, UserId};
use mentedb_embedding::provider::EmbeddingProvider;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::auth::AuthenticatedAgent;
use crate::databases::{DEFAULT_DB, Db, validate_name};
use crate::error::ApiError;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How documents are split into chunks.
#[derive(Clone, Debug)]
pub struct DocumentConfig {
    /// Target chunk length in characters.
    pub chunk_chars: usize,
    /// Characters carried from the end of one chunk to the start of the next.
    pub overlap_chars: usize,
    /// Smallest `chunk_chars` a request may ask for.
    pub min_chunk_chars: usize,
    /// Largest `chunk_chars` a request may ask for.
    pub max_chunk_chars: usize,
    /// Largest overlap as a fraction of the chunk length.
    pub max_overlap_ratio: f32,
    /// Most chunks one document may produce.
    pub max_chunks: usize,
    /// Characters that end a sentence when a paragraph must be split.
    pub sentence_terminators: Vec<char>,
}

impl Default for DocumentConfig {
    fn default() -> Self {
        Self {
            chunk_chars: 1500,
            overlap_chars: 200,
            min_chunk_chars: 100,
            max_chunk_chars: 8000,
            max_overlap_ratio: 0.5,
            max_chunks: 10_000,
            sentence_terminators: vec!['.', '!', '?', '\u{3002}', '\u{ff01}', '\u{ff1f}'],
        }
    }
}

/// How `/v1/ready` probes its dependencies.
#[derive(Clone, Debug)]
pub struct ReadinessConfig {
    /// How long a successful or failed embed check is reused before the
    /// embedder is asked again, so frequent probes stay cheap.
    pub embed_check_interval: Duration,
    /// Text embedded by the probe.
    pub probe_text: String,
    /// Longest any single check may take before it counts as failed.
    pub check_timeout: Duration,
}

impl Default for ReadinessConfig {
    fn default() -> Self {
        Self {
            embed_check_interval: Duration::from_secs(30),
            probe_text: "readiness probe".to_string(),
            check_timeout: Duration::from_secs(5),
        }
    }
}

/// Settings for the operations endpoints.
#[derive(Clone, Debug)]
pub struct OpsConfig {
    /// Where backups are written (`MENTEDB_BACKUP_DIR`). `None` means
    /// `<data_dir>/backups`. Each database gets its own subdirectory.
    pub backup_dir: Option<PathBuf>,
    /// Smallest cluster of near duplicates worth consolidating.
    pub consolidation_min_cluster_size: usize,
    /// Cosine similarity at which memories are grouped for consolidation.
    pub consolidation_similarity: f32,
    /// Memory ids handed to one paraphrase dedup pass, so a large database
    /// is swept in bounded steps.
    pub dedup_batch_size: usize,
    /// Most memories one archive run may forget.
    pub max_forgets_per_archive: usize,
    pub documents: DocumentConfig,
    pub readiness: ReadinessConfig,
}

impl Default for OpsConfig {
    fn default() -> Self {
        Self {
            backup_dir: None,
            consolidation_min_cluster_size: crate::maintenance::MIN_CLUSTER_SIZE,
            consolidation_similarity: crate::maintenance::SIMILARITY_THRESHOLD,
            dedup_batch_size: 1000,
            max_forgets_per_archive: crate::maintenance::MAX_FORGETS_PER_SWEEP,
            documents: DocumentConfig::default(),
            readiness: ReadinessConfig::default(),
        }
    }
}

impl OpsConfig {
    /// Defaults plus `MENTEDB_BACKUP_DIR`.
    pub fn from_env() -> Self {
        Self {
            backup_dir: std::env::var_os("MENTEDB_BACKUP_DIR")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
            ..Self::default()
        }
    }
}

/// The operations settings, read from the environment once.
pub fn ops_config() -> &'static OpsConfig {
    static CONFIG: OnceLock<OpsConfig> = OnceLock::new();
    CONFIG.get_or_init(OpsConfig::from_env)
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/// Routes on one database, mounted under `/v1` and `/v1/db/{db}`.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/maintenance/decay", post(maintenance_decay))
        .route("/maintenance/archive", post(maintenance_archive))
        .route("/maintenance/consolidate", post(maintenance_consolidate))
        .route("/maintenance/sweep", post(maintenance_sweep))
        .route("/enrich", post(enrich))
        .route("/documents", post(store_document))
        .route("/documents/{document_id}", delete(delete_document))
        .route("/backup", post(create_backup))
        .route("/backups", get(list_backups))
}

/// Routes outside any one database (database management, readiness).
pub fn global_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/databases", get(list_databases).post(create_database))
        .route("/v1/databases/{name}", delete(delete_database))
        .route("/v1/ready", get(ready))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Admits only admins: a token with the admin claim, or the configured admin
/// key in `x-api-key` / `Authorization: Bearer`. Listed before [`Db`] in a
/// handler so a refused request never opens or creates a database.
pub struct AdminGuard;

impl FromRequestParts<Arc<AppState>> for AdminGuard {
    type Rejection = ApiError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &Arc<AppState>,
    ) -> Result<Self, ApiError> {
        if parts
            .extensions
            .get::<AuthenticatedAgent>()
            .is_some_and(|a| a.admin)
        {
            return Ok(Self);
        }
        crate::auth::admin_authorized(&parts.headers, &state.admin_key)?;
        Ok(Self)
    }
}

/// Run blocking engine work off the async workers.
async fn blocking<T, F>(what: &'static str, f: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, ApiError> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| ApiError::Internal(format!("{what} task failed: {e}")))?
}

fn internal(what: &str) -> impl FnOnce(mentedb_core::MenteError) -> ApiError + '_ {
    move |e| ApiError::Internal(format!("{what} failed: {e}"))
}

fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

// ---------------------------------------------------------------------------
// POST /maintenance/*
// ---------------------------------------------------------------------------

/// Age salience. The engine derives decay on read, so this persists nothing
/// and reports what the engine reports.
async fn maintenance_decay(db: Db) -> Result<Json<Value>, ApiError> {
    let handle = db.handle.clone();
    let decayed = blocking("decay", move || {
        handle.apply_decay_global().map_err(internal("decay"))
    })
    .await?;
    Ok(Json(json!({ "database": db.name, "decayed": decayed })))
}

/// Evaluate every memory for archival and forget the ones flagged for
/// deletion, capped per run, the same way the maintenance sweep does.
/// Memories flagged for archiving or consolidation are only counted.
async fn maintenance_archive(db: Db) -> Result<Json<Value>, ApiError> {
    let handle = db.handle.clone();
    let cap = ops_config().max_forgets_per_archive;
    let report = blocking("archive", move || {
        let decisions = handle
            .evaluate_archival_global()
            .map_err(internal("archival"))?;
        let evaluated = decisions.len();
        let (mut keep, mut archive, mut consolidate, mut delete) = (0usize, 0usize, 0, 0);
        let mut forgotten = 0usize;
        for (id, decision) in decisions {
            match decision {
                ArchivalDecision::Keep => keep += 1,
                ArchivalDecision::Archive => archive += 1,
                ArchivalDecision::Consolidate(_) => consolidate += 1,
                ArchivalDecision::Delete => {
                    delete += 1;
                    if forgotten < cap && handle.forget(id).is_ok() {
                        forgotten += 1;
                    }
                }
            }
        }
        handle.flush().map_err(internal("flush"))?;
        Ok(json!({
            "evaluated": evaluated,
            "keep": keep,
            "flagged_archive": archive,
            "flagged_consolidate": consolidate,
            "flagged_delete": delete,
            "forgotten": forgotten,
            "capped": delete > forgotten,
        }))
    })
    .await?;
    Ok(Json(with_database(&db.name, report)))
}

/// Merge duplicates: first the paraphrase dedup pass (the newer restatement
/// is folded into the older memory), then the sweep's cluster consolidation
/// (near duplicates merged into one memory, sources invalidated and linked).
async fn maintenance_consolidate(db: Db) -> Result<Json<Value>, ApiError> {
    let handle = db.handle.clone();
    let cfg = ops_config();
    let (batch, min_cluster, similarity) = (
        cfg.dedup_batch_size.max(1),
        cfg.consolidation_min_cluster_size,
        cfg.consolidation_similarity,
    );
    let report = blocking("consolidate", move || {
        let ids = handle.memory_ids();
        let mut duplicates_merged = 0usize;
        for chunk in ids.chunks(batch) {
            duplicates_merged += handle.dedup_sweep_ids(chunk).map_err(internal("dedup"))?;
        }
        let candidates = handle
            .find_consolidation_candidates(min_cluster, similarity)
            .map_err(internal("consolidation"))?;
        let clusters_found = candidates.len();
        let clusters_consolidated = candidates
            .iter()
            .filter(|c| handle.consolidate_cluster(&c.memories).is_ok())
            .count();
        handle.flush().map_err(internal("flush"))?;
        Ok(json!({
            "duplicates_merged": duplicates_merged,
            "clusters_found": clusters_found,
            "clusters_consolidated": clusters_consolidated,
        }))
    })
    .await?;
    Ok(Json(with_database(&db.name, report)))
}

/// The full maintenance sweep the scheduler runs.
async fn maintenance_sweep(db: Db) -> Result<Json<Value>, ApiError> {
    let handle = db.handle.clone();
    let report = blocking("sweep", move || Ok(crate::maintenance::run_sweep(&handle))).await?;
    Ok(Json(json!({
        "database": db.name,
        "consolidated": report.consolidated,
        "decayed": report.decayed,
        "forgotten": report.forgotten,
    })))
}

fn with_database(name: &str, mut report: Value) -> Value {
    if let Value::Object(map) = &mut report {
        map.insert("database".into(), Value::String(name.to_string()));
    }
    report
}

// ---------------------------------------------------------------------------
// POST /enrich
// ---------------------------------------------------------------------------

/// Run the enrichment pipeline over this database's unenriched episodic
/// memories, then entity linking, communities and the user model.
async fn enrich(State(state): State<Arc<AppState>>, db: Db) -> Result<Json<Value>, ApiError> {
    let config = state.extraction_config.clone().ok_or_else(|| {
        ApiError::ServiceUnavailable(
            "enrichment needs an LLM: start the server with --llm-provider or \
             MENTEDB_LLM_PROVIDER"
                .into(),
        )
    })?;
    let embedder = state.databases.embedder().ok_or_else(|| {
        ApiError::ServiceUnavailable(
            "enrichment needs an embedder: start the server with an embedding provider".into(),
        )
    })?;
    let provider = mentedb_extraction::HttpExtractionProvider::new(config.clone())
        .map_err(|e| ApiError::ServiceUnavailable(format!("LLM provider unavailable: {e}")))?;
    let llm = mentedb::cognitive::CognitiveLlmService::new(
        mentedb_extraction::ExtractionLlmJudge::new(provider),
    );
    let handle = db.handle.clone();
    let runtime = tokio::runtime::Handle::current();
    let result = blocking("enrichment", move || {
        // The engine compares this watermark with memory creation times, so
        // the next run only considers memories stored after this one.
        let watermark = now_micros();
        let result = runtime.block_on(mentedb::enrichment::run_enrichment(
            &handle,
            config,
            &*embedder,
            Some(&llm),
            watermark,
            false,
        ));
        handle.flush().map_err(internal("flush"))?;
        Ok(result)
    })
    .await?;
    Ok(Json(json!({
        "database": db.name,
        "memories_stored": result.memories_stored,
        "edges_created": result.edges_created,
        "entities_extracted": result.entities_extracted,
        "duplicates_skipped": result.duplicates_skipped,
        "contradictions_found": result.contradictions_found,
        "sync_linked": result.sync_linked,
        "llm_linked": result.llm_linked,
        "communities_created": result.communities_created,
        "user_model_updated": result.user_model_updated,
    })))
}

// ---------------------------------------------------------------------------
// POST /documents, DELETE /documents/{document_id}
// ---------------------------------------------------------------------------

/// Tag carried by every chunk of a document.
fn document_tag(document_id: &Uuid) -> String {
    format!("doc:{document_id}")
}

fn optional_str<'a>(req: &'a Value, field: &str) -> Result<Option<&'a str>, ApiError> {
    match req.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(_) => Err(ApiError::BadRequest(format!("'{field}' must be a string"))),
    }
}

fn optional_usize(req: &Value, field: &str) -> Result<Option<usize>, ApiError> {
    match req.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(|n| Some(n as usize))
            .ok_or_else(|| ApiError::BadRequest(format!("'{field}' must be a whole number"))),
    }
}

/// Load a document into memory as overlapping chunks.
///
/// Body: `content` (required), optional `title`, `source`, `agent_id`,
/// `user_id`, `tags`, `memory_type` (default `semantic`), `chunk_chars`,
/// `overlap_chars`.
async fn store_document(
    agent: Option<Extension<AuthenticatedAgent>>,
    db: Db,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let cfg = &ops_config().documents;
    let content = req
        .get("content")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| ApiError::BadRequest("missing or empty 'content'".into()))?;
    let title = optional_str(&req, "title")?.map(String::from);
    let source = optional_str(&req, "source")?.map(String::from);

    let agent_id = match crate::handlers::parse_optional_uuid(&req, "agent_id")? {
        Some(id) => {
            let id = AgentId(id);
            crate::handlers::check_body_agent(&agent, id)?;
            id
        }
        None => match &agent {
            Some(Extension(authed)) => authed.agent()?.unwrap_or_else(AgentId::nil),
            None => AgentId::nil(),
        },
    };
    let user_id = crate::handlers::parse_optional_uuid(&req, "user_id")?
        .map(UserId)
        .unwrap_or_else(UserId::nil);
    let memory_type = if req.get("memory_type").is_some_and(|v| !v.is_null()) {
        crate::handlers::parse_memory_type(&req)?
    } else {
        MemoryType::Semantic
    };
    let mut tags: Vec<String> = match req.get("tags") {
        None | Some(Value::Null) => vec![],
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| {
                v.as_str()
                    .map(String::from)
                    .ok_or_else(|| ApiError::BadRequest("'tags' must be strings".into()))
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(ApiError::BadRequest("'tags' must be an array".into())),
    };

    let chunk_chars = optional_usize(&req, "chunk_chars")?.unwrap_or(cfg.chunk_chars);
    let overlap_chars = optional_usize(&req, "overlap_chars")?.unwrap_or(cfg.overlap_chars);
    if chunk_chars < cfg.min_chunk_chars || chunk_chars > cfg.max_chunk_chars {
        return Err(ApiError::BadRequest(format!(
            "'chunk_chars' must be between {} and {}",
            cfg.min_chunk_chars, cfg.max_chunk_chars
        )));
    }
    let max_overlap = (chunk_chars as f32 * cfg.max_overlap_ratio) as usize;
    if overlap_chars > max_overlap {
        return Err(ApiError::BadRequest(format!(
            "'overlap_chars' must be at most {max_overlap} for chunks of {chunk_chars}"
        )));
    }

    let chunks = chunk_text(
        content,
        chunk_chars,
        overlap_chars,
        &cfg.sentence_terminators,
    );
    if chunks.len() > cfg.max_chunks {
        return Err(ApiError::BadRequest(format!(
            "document splits into {} chunks, more than the limit of {}",
            chunks.len(),
            cfg.max_chunks
        )));
    }

    let document_id = Uuid::new_v4();
    let doc_tag = document_tag(&document_id);
    if !tags.contains(&doc_tag) {
        tags.push(doc_tag);
    }
    let handle = db.handle.clone();
    let chunk_count = chunks.len();
    let ids = blocking("store document", move || {
        let texts: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let embeddings = handle
            .embed_batch(&texts)
            .map_err(|e| ApiError::ServiceUnavailable(format!("embedding failed: {e}")))?;
        let now = now_micros();
        let nodes: Vec<MemoryNode> = chunks
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let mut attributes = HashMap::new();
                attributes.insert(
                    "document_id".to_string(),
                    AttributeValue::String(document_id.to_string()),
                );
                attributes.insert("chunk_index".to_string(), AttributeValue::Integer(i as i64));
                attributes.insert(
                    "chunk_count".to_string(),
                    AttributeValue::Integer(chunk_count as i64),
                );
                if let Some(t) = &title {
                    attributes.insert("title".to_string(), AttributeValue::String(t.clone()));
                }
                if let Some(s) = &source {
                    attributes.insert("source".to_string(), AttributeValue::String(s.clone()));
                }
                MemoryNode {
                    id: MemoryId::new(),
                    agent_id,
                    user_id,
                    memory_type,
                    embedding: embeddings
                        .as_ref()
                        .and_then(|e| e.get(i).cloned())
                        .unwrap_or_default(),
                    content: text.clone(),
                    created_at: now,
                    accessed_at: now,
                    access_count: 0,
                    salience: 0.5,
                    confidence: 1.0,
                    space_id: SpaceId::nil(),
                    attributes,
                    tags: tags.clone(),
                    valid_from: None,
                    valid_until: None,
                    context: None,
                }
            })
            .collect();
        let ids = handle.store_batch(nodes).map_err(internal("store"))?;
        handle.flush().map_err(internal("flush"))?;
        Ok(ids)
    })
    .await?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "database": db.name,
            "document_id": document_id.to_string(),
            "chunk_count": ids.len(),
            "chunk_ids": ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
        })),
    ))
}

/// Forget every chunk of a document.
async fn delete_document(
    agent: Option<Extension<AuthenticatedAgent>>,
    db: Db,
    Path(params): Path<HashMap<String, String>>,
) -> Result<Json<Value>, ApiError> {
    let document_id: Uuid = params
        .get("document_id")
        .ok_or_else(|| ApiError::BadRequest("missing document id".into()))?
        .parse()
        .map_err(|_| ApiError::BadRequest("invalid document id".into()))?;
    let handle = db.handle.clone();
    let deleted = blocking("delete document", move || {
        let ids = handle.memory_ids_with_tag(&document_tag(&document_id));
        if ids.is_empty() {
            return Err(ApiError::NotFound(format!(
                "document {document_id} not found"
            )));
        }
        // Check every chunk before forgetting any, so a refused request
        // leaves the document whole.
        if agent.is_some() {
            for id in &ids {
                if let Ok(node) = handle.get_memory(*id) {
                    crate::handlers::check_owner(&agent, &node)?;
                }
            }
        }
        for id in &ids {
            handle.forget(*id).map_err(internal("forget"))?;
        }
        handle.flush().map_err(internal("flush"))?;
        Ok(ids.len())
    })
    .await?;
    Ok(Json(json!({
        "database": db.name,
        "document_id": document_id.to_string(),
        "deleted": deleted,
    })))
}

/// Split text into chunks of at most `chunk_chars` characters. Paragraphs
/// (separated by blank lines) are kept whole when they fit; longer ones are
/// split on sentence boundaries, and sentences longer than a chunk on word
/// boundaries. Each chunk after the first starts with up to `overlap_chars`
/// characters from the end of the previous one, cut at a word boundary.
pub fn chunk_text(
    text: &str,
    chunk_chars: usize,
    overlap_chars: usize,
    terminators: &[char],
) -> Vec<String> {
    let chunk_chars = chunk_chars.max(1);
    let overlap_chars = overlap_chars.min(chunk_chars / 2);
    // Room left for new text once the overlap and a separator are in place.
    let max_unit = chunk_chars.saturating_sub(overlap_chars + 2).max(1);

    // (unit, starts a new paragraph)
    let mut units: Vec<(String, bool)> = Vec::new();
    for paragraph in paragraphs(text) {
        if char_len(&paragraph) <= max_unit {
            units.push((paragraph, true));
            continue;
        }
        let mut first = true;
        for sentence in sentences(&paragraph, terminators) {
            for piece in split_long(&sentence, max_unit) {
                units.push((piece, first));
                first = false;
            }
        }
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    for (unit, new_paragraph) in units {
        let sep = if new_paragraph { "\n\n" } else { " " };
        if current.is_empty() {
            current = unit;
        } else if char_len(&current) + sep.len() + char_len(&unit) <= chunk_chars {
            current.push_str(sep);
            current.push_str(&unit);
        } else {
            let tail = overlap_tail(&current, overlap_chars);
            chunks.push(std::mem::take(&mut current));
            if !tail.is_empty() {
                current.push_str(&tail);
                current.push_str(sep);
            }
            current.push_str(&unit);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Paragraphs separated by blank lines, trimmed, empty ones dropped.
fn paragraphs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut lines: Vec<&str> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !lines.is_empty() {
                out.push(lines.join("\n").trim().to_string());
                lines.clear();
            }
        } else {
            lines.push(line);
        }
    }
    if !lines.is_empty() {
        out.push(lines.join("\n").trim().to_string());
    }
    out.retain(|p| !p.is_empty());
    out
}

/// Sentences: a terminator followed by whitespace ends one.
fn sentences(paragraph: &str, terminators: &[char]) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = paragraph.chars().peekable();
    while let Some(c) = chars.next() {
        current.push(c);
        if terminators.contains(&c) && chars.peek().is_none_or(|n| n.is_whitespace()) {
            let s = current.trim();
            if !s.is_empty() {
                out.push(s.to_string());
            }
            current.clear();
        }
    }
    let s = current.trim();
    if !s.is_empty() {
        out.push(s.to_string());
    }
    out
}

/// Pieces of at most `max` characters, broken at whitespace when possible.
fn split_long(text: &str, max: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let mut word = word.to_string();
        // A single word longer than a piece is cut by characters.
        while char_len(&word) > max {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            let head: String = word.chars().take(max).collect();
            word = word.chars().skip(max).collect();
            out.push(head);
        }
        if word.is_empty() {
            continue;
        }
        if current.is_empty() {
            current = word;
        } else if char_len(&current) + 1 + char_len(&word) <= max {
            current.push(' ');
            current.push_str(&word);
        } else {
            out.push(std::mem::replace(&mut current, word));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// The last `overlap` characters of `chunk`, starting at a word boundary.
fn overlap_tail(chunk: &str, overlap: usize) -> String {
    if overlap == 0 {
        return String::new();
    }
    let total = char_len(chunk);
    if total <= overlap {
        return chunk.trim().to_string();
    }
    let start = chunk
        .char_indices()
        .nth(total - overlap)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let tail = &chunk[start..];
    // Drop a partial leading word unless the tail is one single word.
    let preceded_by_space = chunk[..start].ends_with(char::is_whitespace);
    let tail = if preceded_by_space {
        tail
    } else {
        match tail.find(char::is_whitespace) {
            Some(i) => &tail[i..],
            None => tail,
        }
    };
    tail.trim().to_string()
}

// ---------------------------------------------------------------------------
// POST /backup, GET /backups
// ---------------------------------------------------------------------------

/// File extension the storage layer lists as a backup.
const BACKUP_EXTENSION: &str = "mentebackup";

fn backup_root(state: &AppState) -> PathBuf {
    ops_config()
        .backup_dir
        .clone()
        .unwrap_or_else(|| state.databases.root().join("backups"))
}

/// `YYYYMMDDTHHMMSS.mmmZ` in UTC.
fn utc_stamp(at: SystemTime) -> String {
    let since = at.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = since.as_secs();
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}.{:03}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60,
        since.subsec_millis()
    )
}

/// Proleptic Gregorian date for days since 1970-01-01 (Howard Hinnant's
/// civil_from_days).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Copy `src` into `dst`, skipping any path under one of `exclude`.
fn copy_tree(src: &FsPath, dst: &FsPath, exclude: &[PathBuf]) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        if exclude.iter().any(|x| canonical.starts_with(x)) {
            continue;
        }
        let target = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&path, &target, exclude)?;
        } else {
            std::fs::copy(&path, &target)?;
        }
    }
    Ok(())
}

/// Flush the database and write a full backup file to
/// `<backup_dir>/<database>/<database>-<UTC time>.mentebackup`.
///
/// Writes that land while the files are read may or may not be included;
/// for a snapshot at an exact point, pause writers first.
async fn create_backup(
    _admin: AdminGuard,
    State(state): State<Arc<AppState>>,
    db: Db,
) -> Result<impl IntoResponse, ApiError> {
    let db_dir = state.databases.path_of(&db.name)?;
    let root = backup_root(&state);
    let data_root = state.databases.root().to_path_buf();
    let handle = db.handle.clone();
    let name = db.name.clone();
    let (file, manifest) = blocking("backup", move || {
        handle.flush_full().map_err(internal("flush"))?;
        let io = |what: &'static str| {
            move |e: std::io::Error| ApiError::Internal(format!("{what}: {e}"))
        };
        let dir = root.join(&name);
        std::fs::create_dir_all(&dir).map_err(io("create backup directory"))?;
        let root_c = root
            .canonicalize()
            .map_err(io("resolve backup directory"))?;
        let db_c = db_dir
            .canonicalize()
            .map_err(io("resolve database directory"))?;
        if db_c.starts_with(&root_c) {
            return Err(ApiError::Internal(
                "the backup directory must not contain the database directory".into(),
            ));
        }
        let file = format!("{name}-{}.{BACKUP_EXTENSION}", utc_stamp(SystemTime::now()));
        let path = dir.join(&file);
        if path.exists() {
            return Err(ApiError::Conflict(format!(
                "backup {file} already exists; retry"
            )));
        }

        // The default database's directory also holds the named databases and,
        // by default, the backups; those are left out of its backup.
        let mut exclude = vec![root_c];
        if name == DEFAULT_DB {
            let dbs = data_root.join("dbs");
            exclude.push(dbs.canonicalize().unwrap_or(dbs));
        }
        exclude.retain(|x| x.starts_with(&db_c));

        let manifest = if exclude.is_empty() {
            BackupManager::create_backup(&db_dir, &path).map_err(internal("backup"))?
        } else {
            let staging = dir.join(format!(".staging-{}", Uuid::new_v4()));
            let staged = copy_tree(&db_dir, &staging, &exclude)
                .map_err(io("stage backup"))
                .and_then(|_| {
                    BackupManager::create_backup(&staging, &path).map_err(internal("backup"))
                });
            let _ = std::fs::remove_dir_all(&staging);
            staged?
        };
        Ok((file, manifest))
    })
    .await?;
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "database": db.name,
            "file": file,
            "manifest": manifest,
        })),
    ))
}

/// Backups of this database, newest first.
async fn list_backups(
    _admin: AdminGuard,
    State(state): State<Arc<AppState>>,
    db: Db,
) -> Result<Json<Value>, ApiError> {
    let dir = backup_root(&state).join(&db.name);
    let backups = blocking("list backups", move || {
        let mut found = Vec::new();
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(found),
            Err(e) => return Err(ApiError::Internal(format!("list backups: {e}"))),
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(BACKUP_EXTENSION) {
                continue;
            }
            if let Ok(manifest) = BackupManager::read_manifest(&path) {
                found.push((entry.file_name().to_string_lossy().into_owned(), manifest));
            }
        }
        found.sort_by_key(|(_, m)| std::cmp::Reverse(m.created_at));
        Ok(found)
    })
    .await?;
    let backups: Vec<Value> = backups
        .into_iter()
        .map(|(file, manifest)| json!({ "file": file, "manifest": manifest }))
        .collect();
    Ok(Json(json!({ "database": db.name, "backups": backups })))
}

// ---------------------------------------------------------------------------
// /v1/databases
// ---------------------------------------------------------------------------

/// The default database and every named one on disk.
async fn list_databases(
    _admin: AdminGuard,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Value>, ApiError> {
    let dbs = state.databases.clone();
    let (names, open) = blocking("list databases", move || {
        let open: Vec<String> = dbs.open_handles().into_iter().map(|(n, _)| n).collect();
        Ok((dbs.list(), open))
    })
    .await?;
    Ok(Json(json!({
        "default": DEFAULT_DB,
        "databases": names,
        "open": open,
    })))
}

/// Create a named database. Body: `{"name": "..."}`.
async fn create_database(
    _admin: AdminGuard,
    State(state): State<Arc<AppState>>,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, ApiError> {
    let name = req
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::BadRequest("missing 'name'".into()))?
        .to_string();
    validate_name(&name)?;
    let dbs = state.databases.clone();
    let created = name.clone();
    blocking("create database", move || {
        if dbs.exists(&created) {
            return Err(ApiError::Conflict(format!(
                "database '{created}' already exists"
            )));
        }
        dbs.get(&created, true).map(|_| ())
    })
    .await?;
    Ok((
        StatusCode::CREATED,
        Json(json!({ "name": name, "status": "created" })),
    ))
}

/// Close and permanently delete a named database.
async fn delete_database(
    _admin: AdminGuard,
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<Value>, ApiError> {
    validate_name(&name)?;
    let dbs = state.databases.clone();
    let target = name.clone();
    blocking("delete database", move || dbs.delete(&target)).await?;
    Ok(Json(json!({ "name": name, "status": "deleted" })))
}

// ---------------------------------------------------------------------------
// GET /v1/ready
// ---------------------------------------------------------------------------

/// The last embed check for one embedder. The `Weak` keeps the embedder's
/// allocation from being reused, so a pointer match is always the same one.
struct EmbedProbe {
    embedder: Weak<dyn EmbeddingProvider>,
    checked_at: Instant,
    result: Result<(), String>,
}

static EMBED_PROBES: Mutex<Vec<EmbedProbe>> = Mutex::new(Vec::new());

async fn check_embedder(
    embedder: Arc<dyn EmbeddingProvider>,
    cfg: &ReadinessConfig,
) -> Result<(), String> {
    let weak = Arc::downgrade(&embedder);
    {
        let mut probes = EMBED_PROBES.lock().unwrap_or_else(|e| e.into_inner());
        probes.retain(|p| p.embedder.strong_count() > 0);
        if let Some(p) = probes.iter().find(|p| Weak::ptr_eq(&p.embedder, &weak))
            && p.checked_at.elapsed() < cfg.embed_check_interval
        {
            return p.result.clone();
        }
    }
    let text = cfg.probe_text.clone();
    let probe = tokio::task::spawn_blocking(move || embedder.embed(&text));
    let result = match tokio::time::timeout(cfg.check_timeout, probe).await {
        Ok(Ok(Ok(v))) if !v.is_empty() => Ok(()),
        Ok(Ok(Ok(_))) => Err("embedder returned an empty vector".to_string()),
        Ok(Ok(Err(e))) => Err(format!("embed failed: {e}")),
        Ok(Err(e)) => Err(format!("embed task failed: {e}")),
        Err(_) => Err(format!("embed timed out after {:?}", cfg.check_timeout)),
    };
    let mut probes = EMBED_PROBES.lock().unwrap_or_else(|e| e.into_inner());
    probes.retain(|p| !Weak::ptr_eq(&p.embedder, &weak));
    probes.push(EmbedProbe {
        embedder: weak,
        checked_at: Instant::now(),
        result: result.clone(),
    });
    result
}

/// Readiness: 200 when the default database answers and, if an embedder is
/// configured, it embeds (checked at most once per configured interval).
/// 503 naming the failing check otherwise.
async fn ready(State(state): State<Arc<AppState>>) -> Response {
    let cfg = &ops_config().readiness;
    let mut checks = serde_json::Map::new();
    let mut failed: Vec<&str> = Vec::new();

    let db = state.databases.default_db();
    let probe = tokio::task::spawn_blocking(move || db.memory_count());
    match tokio::time::timeout(cfg.check_timeout, probe).await {
        Ok(Ok(_)) => {
            checks.insert("database".into(), json!("ok"));
        }
        Ok(Err(e)) => {
            checks.insert("database".into(), json!(format!("error: {e}")));
            failed.push("database");
        }
        Err(_) => {
            checks.insert(
                "database".into(),
                json!(format!("error: timed out after {:?}", cfg.check_timeout)),
            );
            failed.push("database");
        }
    }

    match state.databases.embedder() {
        None => {
            checks.insert("embedder".into(), json!("not_configured"));
        }
        Some(embedder) => match check_embedder(embedder, cfg).await {
            Ok(()) => {
                checks.insert("embedder".into(), json!("ok"));
            }
            Err(e) => {
                checks.insert("embedder".into(), json!(format!("error: {e}")));
                failed.push("embedder");
            }
        },
    }

    if failed.is_empty() {
        (
            StatusCode::OK,
            Json(json!({ "status": "ready", "checks": checks })),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "not_ready", "failed": failed, "checks": checks })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TERMS: &[char] = &['.', '!', '?'];

    #[test]
    fn short_text_is_one_chunk() {
        let chunks = chunk_text("One paragraph. Two sentences.", 100, 20, TERMS);
        assert_eq!(chunks, vec!["One paragraph. Two sentences.".to_string()]);
    }

    #[test]
    fn chunks_respect_the_limit_and_overlap() {
        let sentence = "The quick brown fox jumps over the lazy dog again.";
        let paragraph = [sentence; 6].join(" ");
        let text = [paragraph.as_str(); 4].join("\n\n");
        let chunks = chunk_text(&text, 200, 40, TERMS);
        assert!(chunks.len() > 4);
        for c in &chunks {
            assert!(char_len(c) <= 200, "{} chars: {c}", char_len(c));
        }
        for pair in chunks.windows(2) {
            let tail_word = pair[0].split_whitespace().last().unwrap();
            assert!(pair[1].contains(tail_word));
        }
    }

    #[test]
    fn a_word_longer_than_a_chunk_is_cut() {
        let text = "x".repeat(450);
        let chunks = chunk_text(&text, 200, 0, TERMS);
        assert!(chunks.iter().all(|c| char_len(c) <= 200));
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn utc_stamp_formats_known_instants() {
        assert_eq!(utc_stamp(UNIX_EPOCH), "19700101T000000.000Z");
        let t = UNIX_EPOCH + Duration::from_millis(1_709_210_096_789);
        assert_eq!(utc_stamp(t), "20240229T123456.789Z");
    }
}
