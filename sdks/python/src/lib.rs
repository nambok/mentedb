use std::path::Path;

use mentedb::MenteDb;
use mentedb_cognitive::pain::{PainRegistry as RustPainRegistry, PainSignal};
use mentedb_cognitive::stream::{CognitionStream as RustCognitionStream, StreamAlert as RustStreamAlert};
use mentedb_cognitive::trajectory::{
    DecisionState, TrajectoryNode, TrajectoryTracker as RustTrajectoryTracker,
};
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{AgentId, Embedding, MemoryId, Timestamp};
use mentedb_core::{MemoryEdge, MemoryNode};
use mentedb_extraction::{
    ExtractionConfig, ExtractionPipeline, HttpExtractionProvider, LlmProvider,
    map_extraction_type_to_memory_type,
};
use mentedb_embedding::candle_provider::CandleEmbeddingProvider;
use mentedb_embedding::hash_provider::HashEmbeddingProvider;
use mentedb_embedding::http_provider::HttpEmbeddingConfig;
use mentedb_embedding::http_provider::HttpEmbeddingProvider;
use mentedb_embedding::provider::EmbeddingProvider;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn parse_uuid(s: &str) -> PyResult<Uuid> {
    Uuid::parse_str(s).map_err(|e| PyRuntimeError::new_err(format!("invalid UUID: {e}")))
}

fn parse_agent_id(s: &str) -> PyResult<AgentId> {
    Ok(AgentId(parse_uuid(s)?))
}

fn parse_memory_id(s: &str) -> PyResult<MemoryId> {
    Ok(MemoryId(parse_uuid(s)?))
}

fn parse_memory_type(s: &str) -> PyResult<MemoryType> {
    match s {
        "episodic" => Ok(MemoryType::Episodic),
        "semantic" => Ok(MemoryType::Semantic),
        "procedural" => Ok(MemoryType::Procedural),
        "anti_pattern" => Ok(MemoryType::AntiPattern),
        "reasoning" => Ok(MemoryType::Reasoning),
        "correction" => Ok(MemoryType::Correction),
        other => Err(PyRuntimeError::new_err(format!(
            "unknown memory type: {other}"
        ))),
    }
}

fn parse_edge_type(s: &str) -> PyResult<EdgeType> {
    match s {
        "caused" => Ok(EdgeType::Caused),
        "before" => Ok(EdgeType::Before),
        "related" => Ok(EdgeType::Related),
        "contradicts" => Ok(EdgeType::Contradicts),
        "supports" => Ok(EdgeType::Supports),
        "supersedes" => Ok(EdgeType::Supersedes),
        "derived" => Ok(EdgeType::Derived),
        "part_of" => Ok(EdgeType::PartOf),
        other => Err(PyRuntimeError::new_err(format!(
            "unknown edge type: {other}"
        ))),
    }
}

fn now_us() -> Timestamp {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as Timestamp
}

// ---------------------------------------------------------------------------
// MenteDB
// ---------------------------------------------------------------------------

#[pyclass]
struct MenteDB {
    db: Option<MenteDb>,
    embedder: Option<Box<dyn EmbeddingProvider>>,
}

#[pymethods]
impl MenteDB {
    #[new]
    #[pyo3(signature = (data_dir, embedding_provider=None, embedding_api_key=None, embedding_model=None))]
    fn new(
        data_dir: &str,
        embedding_provider: Option<&str>,
        embedding_api_key: Option<&str>,
        embedding_model: Option<&str>,
    ) -> PyResult<Self> {
        let embedder: Option<Box<dyn EmbeddingProvider>> = match embedding_provider {
            Some("openai") => {
                let key = embedding_api_key
                    .ok_or_else(|| PyRuntimeError::new_err("openai provider requires embedding_api_key"))?;
                let model = embedding_model.unwrap_or("text-embedding-3-small");
                let config = HttpEmbeddingConfig::openai(key, model);
                Some(Box::new(HttpEmbeddingProvider::new(config)))
            }
            Some("cohere") => {
                let key = embedding_api_key
                    .ok_or_else(|| PyRuntimeError::new_err("cohere provider requires embedding_api_key"))?;
                let model = embedding_model.unwrap_or("embed-english-v3.0");
                let config = HttpEmbeddingConfig::cohere(key, model);
                Some(Box::new(HttpEmbeddingProvider::new(config)))
            }
            Some("voyage") => {
                let key = embedding_api_key
                    .ok_or_else(|| PyRuntimeError::new_err("voyage provider requires embedding_api_key"))?;
                let model = embedding_model.unwrap_or("voyage-2");
                let config = HttpEmbeddingConfig::voyage(key, model);
                Some(Box::new(HttpEmbeddingProvider::new(config)))
            }
            Some("candle") | Some("local") => {
                let cache_dir = std::path::PathBuf::from(format!("{data_dir}/.candle-cache"));
                match CandleEmbeddingProvider::with_cache_dir(cache_dir) {
                    Ok(p) => Some(Box::new(p)),
                    Err(e) => return Err(PyRuntimeError::new_err(format!("candle init failed: {e}"))),
                }
            }
            Some("hash") | None => {
                Some(Box::new(HashEmbeddingProvider::new(384)))
            }
            Some(other) => {
                return Err(PyRuntimeError::new_err(format!(
                    "unknown embedding provider: {other}. Use 'openai', 'candle', 'cohere', 'voyage', or 'hash'"
                )));
            }
        };

        let mut db = MenteDb::open(Path::new(data_dir)).map_err(to_pyerr)?;
        if let Some(ref e) = embedder {
            db.set_embedder(Box::new(HashEmbeddingProvider::new(e.dimensions())));
        }
        Ok(Self { db: Some(db), embedder })
    }

    /// Store a memory and return its UUID string.
    #[pyo3(signature = (content, memory_type, embedding=None, agent_id=None, tags=None))]
    fn store(
        &mut self,
        content: &str,
        memory_type: &str,
        embedding: Option<Vec<f32>>,
        agent_id: Option<&str>,
        tags: Option<Vec<String>>,
    ) -> PyResult<String> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        let aid = match agent_id {
            Some(s) => parse_agent_id(s)?,
            None => AgentId::new(),
        };

        let mt = parse_memory_type(memory_type)?;

        let emb: Embedding = match embedding {
            Some(e) if !e.is_empty() => e,
            _ => {
                if let Some(ref embedder) = self.embedder {
                    embedder.embed(content).map_err(to_pyerr)?
                } else {
                    hash_embedding(content, 384)
                }
            }
        };

        let mut node = MemoryNode::new(aid, mt, content.to_string(), emb);
        if let Some(t) = tags {
            node.tags = t;
        }

        let id = node.id;
        db.store(node).map_err(to_pyerr)?;
        Ok(id.to_string())
    }

    /// Recall memories using an MQL query string.
    fn recall(&mut self, query: &str) -> PyResult<RecallResult> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        let window = db.recall(query).map_err(to_pyerr)?;

        // Build a text representation from the context blocks.
        let mut text = String::new();
        let mut memory_count: usize = 0;
        for block in &window.blocks {
            for sm in &block.memories {
                if !text.is_empty() {
                    text.push_str("\n---\n");
                }
                text.push_str(&sm.memory.content);
                memory_count += 1;
            }
        }

        Ok(RecallResult {
            text,
            total_tokens: window.total_tokens,
            memory_count,
        })
    }

    /// Vector similarity search. Returns a list of SearchResult.
    fn search(&mut self, embedding: Vec<f32>, k: usize) -> PyResult<Vec<SearchResult>> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        let hits = db.recall_similar(&embedding, k).map_err(to_pyerr)?;
        Ok(hits
            .into_iter()
            .map(|(id, score)| SearchResult {
                id: id.to_string(),
                score,
            })
            .collect())
    }

    /// Text-based similarity search using the configured embedding provider.
    /// Uses OpenAI/Cohere/Voyage if configured, falls back to hash embedding.
    #[pyo3(signature = (query, k=10, tags=None, after=None, before=None))]
    fn search_text(
        &mut self,
        query: &str,
        k: usize,
        tags: Option<Vec<String>>,
        after: Option<u64>,
        before: Option<u64>,
    ) -> PyResult<Vec<SearchResult>> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        let embedding = if let Some(ref embedder) = self.embedder {
            embedder.embed(query).map_err(to_pyerr)?
        } else {
            hash_embedding(query, 384)
        };

        let tag_strs: Option<Vec<&str>> = tags.as_ref().map(|t| t.iter().map(|s| s.as_str()).collect());
        let tag_refs: Option<&[&str]> = tag_strs.as_deref();
        let time_range = match (after, before) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        };

        let hits = db
            .recall_similar_filtered(&embedding, k, tag_refs, time_range)
            .map_err(to_pyerr)?;
        Ok(hits
            .into_iter()
            .map(|(id, score)| SearchResult {
                id: id.to_string(),
                score,
            })
            .collect())
    }

    /// Multi-query search with Reciprocal Rank Fusion.
    ///
    /// Takes multiple query strings, embeds each, and merges results via RRF
    /// for broader recall across different semantic aspects.
    #[pyo3(signature = (queries, k=10))]
    fn search_multi(&mut self, queries: Vec<String>, k: usize) -> PyResult<Vec<SearchResult>> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        let mut embeddings = Vec::with_capacity(queries.len());
        for q in &queries {
            let emb = if let Some(ref embedder) = self.embedder {
                embedder.embed(q).map_err(to_pyerr)?
            } else {
                hash_embedding(q, 384)
            };
            embeddings.push(emb);
        }

        let hits = db.recall_similar_multi(&embeddings, k).map_err(to_pyerr)?;
        Ok(hits
            .into_iter()
            .map(|(id, score)| SearchResult {
                id: id.to_string(),
                score,
            })
            .collect())
    }

    /// Expanded search: uses the engine's LLM to decompose a query into
    /// sub-queries, then runs multi-query RRF search.
    ///
    /// This is the engine-native way to get broad recall — the LLM call
    /// happens inside the engine, not in the benchmark or application layer.
    #[pyo3(signature = (query, k=10, provider=None))]
    fn search_expanded(
        &mut self,
        query: &str,
        k: usize,
        provider: Option<&str>,
    ) -> PyResult<Vec<SearchResult>> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        // Build LLM config for query expansion
        let config = build_extraction_config_from_env(provider)?;
        let http_provider = HttpExtractionProvider::new(config).map_err(to_pyerr)?;

        // Expand query via LLM
        let rt = tokio::runtime::Runtime::new().map_err(to_pyerr)?;
        let sub_queries = rt
            .block_on(http_provider.expand_query(query))
            .unwrap_or_default();

        // Build embedding list: original query + sub-queries
        let mut all_queries = vec![query.to_string()];
        all_queries.extend(sub_queries);

        let mut embeddings = Vec::with_capacity(all_queries.len());
        for q in &all_queries {
            let emb = if let Some(ref embedder) = self.embedder {
                embedder.embed(q).map_err(to_pyerr)?
            } else {
                hash_embedding(q, 384)
            };
            embeddings.push(emb);
        }

        let hits = db.recall_similar_multi(&embeddings, k).map_err(to_pyerr)?;
        Ok(hits
            .into_iter()
            .map(|(id, score)| SearchResult {
                id: id.to_string(),
                score,
            })
            .collect())
    }

    /// Add a typed, weighted edge between two memories.
    fn relate(
        &mut self,
        source: &str,
        target: &str,
        edge_type: &str,
        weight: f32,
        valid_from: Option<u64>,
        valid_until: Option<u64>,
    ) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        let edge = MemoryEdge {
            source: parse_memory_id(source)?,
            target: parse_memory_id(target)?,
            edge_type: parse_edge_type(edge_type)?,
            weight,
            created_at: now_us(),
            valid_from,
            valid_until,
        };

        db.relate(edge).map_err(to_pyerr)
    }

    /// Remove a memory by its UUID string.
    fn forget(&mut self, memory_id: &str) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        let id = parse_memory_id(memory_id)?;
        db.forget(id).map_err(to_pyerr)
    }

    /// Retrieve a memory by its UUID string.
    /// Returns a dict with id, content, memory_type, tags, created_at.
    fn get_memory(&mut self, memory_id: &str) -> PyResult<PyObject> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        let id = parse_memory_id(memory_id)?;
        let node = db.get_memory(id).map_err(to_pyerr)?;
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", node.id.to_string())?;
            dict.set_item("content", &node.content)?;
            dict.set_item("memory_type", format!("{:?}", node.memory_type))?;
            let tags: Vec<&str> = node.tags.iter().map(|s| s.as_str()).collect();
            dict.set_item("tags", tags)?;
            dict.set_item("created_at", node.created_at)?;
            Ok(dict.into())
        })
    }

    /// Extract memories from a conversation and store them.
    ///
    /// Requires `MENTEDB_LLM_PROVIDER` and `MENTEDB_LLM_API_KEY` env vars to be set,
    /// or pass `provider` ("openai", "anthropic", "ollama") explicitly.
    #[pyo3(signature = (conversation, provider=None, agent_id=None))]
    fn ingest(
        &mut self,
        conversation: &str,
        provider: Option<&str>,
        agent_id: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self
            .db
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

        let config = build_extraction_config_from_env(provider)?;

        let http_provider =
            HttpExtractionProvider::new(config.clone()).map_err(to_pyerr)?;
        let pipeline = ExtractionPipeline::new(http_provider, config.clone());

        let rt = tokio::runtime::Runtime::new().map_err(to_pyerr)?;
        let all_memories = rt
            .block_on(pipeline.extract_from_conversation(conversation))
            .map_err(to_pyerr)?;

        let total = all_memories.len();
        let quality_passed = pipeline.filter_quality(&all_memories);
        let rejected_low_quality = total - quality_passed.len();

        let aid = match agent_id {
            Some(s) => parse_agent_id(s)?,
            None => AgentId::new(),
        };

        let mut stored_ids = Vec::new();
        for memory in &quality_passed {
            let mt = map_extraction_type_to_memory_type(&memory.memory_type);
            // Fact-augmented embedding: include entities and tags in the vector
            let embed_text = memory.embedding_key();
            let emb = if let Some(ref embedder) = self.embedder {
                embedder.embed(&embed_text).map_err(to_pyerr)?
            } else {
                hash_embedding(&embed_text, 384)
            };
            let mut node = MemoryNode::new(aid, mt, memory.content.clone(), emb);
            node.tags = memory.tags.clone();
            node.salience = memory.confidence;
            node.confidence = memory.confidence;
            let id = node.id;
            db.store(node).map_err(to_pyerr)?;
            stored_ids.push(id.to_string());
        }

        // Turn-level decomposition: store individual user turns as episodic memories
        // for finer-grained retrieval alongside the LLM-extracted facts.
        for line in conversation.lines() {
            let trimmed = line.trim();
            let user_content = if let Some(rest) = trimmed.strip_prefix("User:") {
                Some(rest.trim())
            } else if let Some(rest) = trimmed.strip_prefix("user:") {
                Some(rest.trim())
            } else {
                None
            };

            if let Some(content) = user_content {
                if content.len() > 30 {
                    let emb = if let Some(ref embedder) = self.embedder {
                        embedder.embed(content).map_err(to_pyerr)?
                    } else {
                        hash_embedding(content, 384)
                    };
                    let mut node = MemoryNode::new(
                        aid,
                        MemoryType::Episodic,
                        content.to_string(),
                        emb,
                    );
                    node.tags = vec!["turn".to_string()];
                    node.salience = 0.4;
                    let id = node.id;
                    db.store(node).map_err(to_pyerr)?;
                    stored_ids.push(id.to_string());
                }
            }
        }

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("memories_stored", stored_ids.len())?;
            dict.set_item("rejected_low_quality", rejected_low_quality)?;
            dict.set_item("rejected_duplicate", 0)?;
            dict.set_item("contradictions", 0)?;
            dict.set_item("stored_ids", stored_ids)?;
            Ok(dict.into())
        })
    }

    /// Flush and close the database.
    fn close(&mut self) -> PyResult<()> {
        if let Some(mut db) = self.db.take() {
            db.close().map_err(to_pyerr)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RecallResult / SearchResult
// ---------------------------------------------------------------------------

#[pyclass(get_all)]
#[derive(Clone)]
struct RecallResult {
    text: String,
    total_tokens: usize,
    memory_count: usize,
}

#[pymethods]
impl RecallResult {
    fn __repr__(&self) -> String {
        format!(
            "RecallResult(tokens={}, memories={})",
            self.total_tokens, self.memory_count
        )
    }
}

#[pyclass(get_all)]
#[derive(Clone)]
struct SearchResult {
    id: String,
    score: f32,
}

#[pymethods]
impl SearchResult {
    fn __repr__(&self) -> String {
        format!("SearchResult(id='{}', score={:.4})", self.id, self.score)
    }
}

// ---------------------------------------------------------------------------
// CognitionStream
// ---------------------------------------------------------------------------

#[pyclass]
struct CognitionStream {
    inner: RustCognitionStream,
}

#[pymethods]
impl CognitionStream {
    #[new]
    #[pyo3(signature = (buffer_size=None))]
    fn new(buffer_size: Option<usize>) -> Self {
        let size = buffer_size.unwrap_or(1000);
        Self {
            inner: RustCognitionStream::new(size),
        }
    }

    /// Feed a single token into the stream buffer.
    fn feed_token(&self, token: &str) {
        self.inner.feed_token(token);
    }

    /// Check for contradictions and other alerts against known facts.
    /// Each fact is a (memory_id_str, fact_text) tuple.
    fn check_alerts(&self, known_facts: Vec<(String, String)>) -> PyResult<Vec<StreamAlertPy>> {
        let facts: Vec<(MemoryId, String)> = known_facts
            .into_iter()
            .map(|(id_str, text)| {
                let id = parse_memory_id(&id_str)?;
                Ok((id, text))
            })
            .collect::<PyResult<Vec<_>>>()?;

        let fact_refs: Vec<(MemoryId, String)> = facts;
        let ref_slice: Vec<(MemoryId, String)> = fact_refs;
        let as_refs: Vec<(MemoryId, String)> = ref_slice;
        // CognitionStream::check_alerts takes &[(MemoryId, String)]
        let alerts = self.inner.check_alerts(
            &as_refs
                .iter()
                .map(|(id, s)| (*id, s.clone()))
                .collect::<Vec<_>>(),
        );
        Ok(alerts.into_iter().map(StreamAlertPy::from_rust).collect())
    }

    /// Drain and return the accumulated buffer content.
    fn drain_buffer(&self) -> String {
        self.inner.drain_buffer()
    }
}

// ---------------------------------------------------------------------------
// StreamAlert
// ---------------------------------------------------------------------------

#[pyclass(get_all)]
#[derive(Clone)]
struct StreamAlertPy {
    alert_type: String,
    memory_id: String,
    message: String,
}

#[pymethods]
impl StreamAlertPy {
    fn __repr__(&self) -> String {
        format!(
            "StreamAlert(type='{}', memory_id='{}')",
            self.alert_type, self.memory_id
        )
    }
}

impl StreamAlertPy {
    fn from_rust(alert: RustStreamAlert) -> Self {
        match alert {
            RustStreamAlert::Contradiction {
                memory_id,
                ai_said,
                stored,
            } => Self {
                alert_type: "contradiction".to_string(),
                memory_id: memory_id.to_string(),
                message: format!("AI said: {ai_said}; stored: {stored}"),
            },
            RustStreamAlert::Forgotten {
                memory_id,
                summary,
            } => Self {
                alert_type: "forgotten".to_string(),
                memory_id: memory_id.to_string(),
                message: summary,
            },
            RustStreamAlert::Correction {
                memory_id,
                old,
                new,
            } => Self {
                alert_type: "correction".to_string(),
                memory_id: memory_id.to_string(),
                message: format!("old: {old}; new: {new}"),
            },
            RustStreamAlert::Reinforcement { memory_id } => Self {
                alert_type: "reinforcement".to_string(),
                memory_id: memory_id.to_string(),
                message: String::new(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// TrajectoryTracker
// ---------------------------------------------------------------------------

#[pyclass]
struct TrajectoryTracker {
    inner: RustTrajectoryTracker,
}

#[pymethods]
impl TrajectoryTracker {
    #[new]
    #[pyo3(signature = (max_turns=None))]
    fn new(max_turns: Option<usize>) -> Self {
        let mt = max_turns.unwrap_or(100);
        Self {
            inner: RustTrajectoryTracker::new(mt),
        }
    }

    /// Record a conversation turn.
    fn record_turn(
        &mut self,
        topic: &str,
        decision_state: &str,
        open_questions: Vec<String>,
    ) -> PyResult<()> {
        let state = match decision_state {
            "investigating" => DecisionState::Investigating,
            "decided" => DecisionState::Decided("decided".to_string()),
            "interrupted" => DecisionState::Interrupted,
            "completed" => DecisionState::Completed,
            other => {
                if let Some(rest) = other.strip_prefix("narrowed_to:") {
                    DecisionState::NarrowedTo(rest.trim().to_string())
                } else if let Some(rest) = other.strip_prefix("decided:") {
                    DecisionState::Decided(rest.trim().to_string())
                } else {
                    DecisionState::Investigating
                }
            }
        };

        let turn_count = self.inner.get_trajectory().len() as u64;
        let node = TrajectoryNode {
            turn_id: turn_count,
            topic_embedding: vec![],
            topic_summary: topic.to_string(),
            decision_state: state,
            open_questions,
            timestamp: now_us(),
        };

        self.inner.record_turn(node);
        Ok(())
    }

    /// Get a resume context string for the current conversation trajectory.
    fn get_resume_context(&self) -> Option<String> {
        self.inner.get_resume_context()
    }

    /// Predict likely next topics based on the trajectory so far.
    fn predict_next_topics(&self) -> Vec<String> {
        self.inner.predict_next_topics()
    }
}

// ---------------------------------------------------------------------------
// PainRegistry
// ---------------------------------------------------------------------------

#[pyclass]
struct PainRegistry {
    inner: RustPainRegistry,
}

#[pymethods]
impl PainRegistry {
    #[new]
    #[pyo3(signature = (max_warnings=None))]
    fn new(max_warnings: Option<usize>) -> Self {
        let mw = max_warnings.unwrap_or(5);
        Self {
            inner: RustPainRegistry::new(mw),
        }
    }

    /// Record a pain signal associated with a memory.
    fn record_pain(
        &mut self,
        memory_id: &str,
        intensity: f32,
        triggers: Vec<String>,
        description: &str,
    ) -> PyResult<()> {
        let mid = parse_memory_id(memory_id)?;
        let signal = PainSignal {
            id: MemoryId::new(),
            memory_id: mid,
            intensity,
            trigger_keywords: triggers,
            description: description.to_string(),
            created_at: now_us(),
            decay_rate: 0.1,
        };
        self.inner.record_pain(signal);
        Ok(())
    }

    /// Check if any recorded pain signals match the given keywords.
    /// Returns formatted warning text.
    fn check_triggers(&self, keywords: Vec<String>) -> String {
        let signals = self.inner.get_pain_for_context(&keywords);
        self.inner.format_pain_warnings(&signals)
    }
}

// ---------------------------------------------------------------------------
// Simple hash embedding (fallback when no embedding is provided)
// ---------------------------------------------------------------------------

fn hash_embedding(text: &str, dims: usize) -> Vec<f32> {
    let mut emb = vec![0.0f32; dims];
    for (i, b) in text.bytes().enumerate() {
        emb[i % dims] += (b as f32 - 96.0) / 32.0;
    }
    // L2 normalize
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut emb {
            *v /= norm;
        }
    }
    emb
}

// ---------------------------------------------------------------------------
// Extraction config helper
// ---------------------------------------------------------------------------

fn build_extraction_config_from_env(provider_override: Option<&str>) -> PyResult<ExtractionConfig> {
    let provider_str = match provider_override {
        Some(p) => p.to_string(),
        None => std::env::var("MENTEDB_LLM_PROVIDER").unwrap_or_default(),
    };

    let provider = match provider_str.to_lowercase().as_str() {
        "openai" => LlmProvider::OpenAI,
        "anthropic" => LlmProvider::Anthropic,
        "ollama" => LlmProvider::Ollama,
        _ => {
            return Err(PyRuntimeError::new_err(
                "LLM provider not configured. Set MENTEDB_LLM_PROVIDER env var or pass provider argument.",
            ));
        }
    };

    let api_key = std::env::var("MENTEDB_LLM_API_KEY").ok();
    let api_url = std::env::var("MENTEDB_LLM_BASE_URL")
        .unwrap_or_else(|_| provider.default_url().to_string());
    let model = std::env::var("MENTEDB_LLM_MODEL")
        .unwrap_or_else(|_| provider.default_model().to_string());
    let quality_threshold = std::env::var("MENTEDB_EXTRACTION_QUALITY_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.7);

    Ok(ExtractionConfig {
        provider,
        api_key,
        api_url,
        model,
        max_extractions_per_conversation: 50,
        quality_threshold,
        deduplication_threshold: 0.85,
        enable_contradiction_check: true,
        enable_deduplication: true,
    })
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _mentedb_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MenteDB>()?;
    m.add_class::<RecallResult>()?;
    m.add_class::<SearchResult>()?;
    m.add_class::<CognitionStream>()?;
    m.add_class::<StreamAlertPy>()?;
    m.add_class::<TrajectoryTracker>()?;
    m.add_class::<PainRegistry>()?;
    Ok(())
}
