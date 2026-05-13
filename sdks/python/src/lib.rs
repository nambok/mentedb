use std::path::Path;
use std::sync::Mutex;

use mentedb::CognitiveConfig;
use mentedb::MenteDb;
use mentedb::process_turn::ProcessTurnInput;
use mentedb_cognitive::pain::{PainRegistry as RustPainRegistry, PainSignal};
use mentedb_cognitive::stream::{
    CognitionStream as RustCognitionStream, StreamAlert as RustStreamAlert,
};
use mentedb_cognitive::trajectory::{
    DecisionState, TrajectoryNode, TrajectoryTracker as RustTrajectoryTracker,
};
use mentedb_context::DeltaTracker;
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{AgentId, Embedding, MemoryId, Timestamp};
use mentedb_core::{MemoryEdge, MemoryNode};
use mentedb_embedding::candle_provider::CandleEmbeddingProvider;
use mentedb_embedding::hash_provider::HashEmbeddingProvider;
use mentedb_embedding::http_provider::HttpEmbeddingConfig;
use mentedb_embedding::http_provider::HttpEmbeddingProvider;
use mentedb_embedding::provider::EmbeddingProvider;
use mentedb_extraction::{
    ExtractionConfig, ExtractionPipeline, HttpExtractionProvider, LlmProvider,
    map_extraction_type_to_memory_type,
};
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

/// Format a microsecond timestamp as "YYYY-MM-DD" for human-readable evidence.
fn format_timestamp_date(us: u64) -> Option<String> {
    if us == 0 {
        return None;
    }
    let secs = (us / 1_000_000) as i64;
    // Manual date calculation from unix timestamp (no chrono dependency)
    let days = secs / 86400;
    // Civil date from days since epoch (algorithm from Howard Hinnant)
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    Some(format!("{:04}-{:02}-{:02}", y, m, d))
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
    delta_tracker: Mutex<DeltaTracker>,
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
                let key = embedding_api_key.ok_or_else(|| {
                    PyRuntimeError::new_err("openai provider requires embedding_api_key")
                })?;
                let model = embedding_model.unwrap_or("text-embedding-3-small");
                let config = HttpEmbeddingConfig::openai(key, model);
                Some(Box::new(HttpEmbeddingProvider::new(config)))
            }
            Some("cohere") => {
                let key = embedding_api_key.ok_or_else(|| {
                    PyRuntimeError::new_err("cohere provider requires embedding_api_key")
                })?;
                let model = embedding_model.unwrap_or("embed-english-v3.0");
                let config = HttpEmbeddingConfig::cohere(key, model);
                Some(Box::new(HttpEmbeddingProvider::new(config)))
            }
            Some("voyage") => {
                let key = embedding_api_key.ok_or_else(|| {
                    PyRuntimeError::new_err("voyage provider requires embedding_api_key")
                })?;
                let model = embedding_model.unwrap_or("voyage-2");
                let config = HttpEmbeddingConfig::voyage(key, model);
                Some(Box::new(HttpEmbeddingProvider::new(config)))
            }
            Some("candle") | Some("local") => {
                let cache_dir = std::path::PathBuf::from(format!("{data_dir}/.candle-cache"));
                match CandleEmbeddingProvider::with_cache_dir(cache_dir) {
                    Ok(p) => Some(Box::new(p)),
                    Err(e) => {
                        return Err(PyRuntimeError::new_err(format!("candle init failed: {e}")));
                    }
                }
            }
            Some("hash") | None => Some(Box::new(HashEmbeddingProvider::new(384))),
            Some(other) => {
                return Err(PyRuntimeError::new_err(format!(
                    "unknown embedding provider: {other}. Use 'openai', 'candle', 'cohere', 'voyage', or 'hash'"
                )));
            }
        };

        let mut db = MenteDb::open_with_config(Path::new(data_dir), CognitiveConfig::default())
            .map_err(to_pyerr)?;
        if let Some(ref e) = embedder {
            db.set_embedder(Box::new(HashEmbeddingProvider::new(e.dimensions())));
        }
        Ok(Self {
            db: Some(db),
            embedder,
            delta_tracker: Mutex::new(DeltaTracker::new()),
        })
    }

    /// Store a memory and return its UUID string.
    #[pyo3(signature = (content, memory_type, embedding=None, agent_id=None, tags=None, created_at=None))]
    fn store(
        &self,
        content: &str,
        memory_type: &str,
        embedding: Option<Vec<f32>>,
        agent_id: Option<&str>,
        tags: Option<Vec<String>>,
        created_at: Option<u64>,
    ) -> PyResult<String> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

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
        if let Some(ts) = created_at {
            node.created_at = ts;
            node.accessed_at = ts;
            node.valid_from = Some(ts);
        }

        let id = node.id;
        db.store(node).map_err(to_pyerr)?;
        Ok(id.to_string())
    }

    /// Recall memories using an MQL query string.
    fn recall(&self, query: &str) -> PyResult<RecallResult> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

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
    fn search(&self, embedding: Vec<f32>, k: usize) -> PyResult<Vec<SearchResult>> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

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
        &self,
        query: &str,
        k: usize,
        tags: Option<Vec<String>>,
        after: Option<u64>,
        before: Option<u64>,
    ) -> PyResult<Vec<SearchResult>> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

        let embedding = if let Some(ref embedder) = self.embedder {
            embedder.embed(query).map_err(to_pyerr)?
        } else {
            hash_embedding(query, 384)
        };

        let tag_strs: Option<Vec<&str>> = tags
            .as_ref()
            .map(|t| t.iter().map(|s| s.as_str()).collect());
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
    #[pyo3(signature = (queries, k=10, tags=None, before=None))]
    fn search_multi(
        &self,
        queries: Vec<String>,
        k: usize,
        tags: Option<Vec<String>>,
        before: Option<u64>,
    ) -> PyResult<Vec<SearchResult>> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

        let mut embeddings = Vec::with_capacity(queries.len());
        for q in &queries {
            let emb = if let Some(ref embedder) = self.embedder {
                embedder.embed(q).map_err(to_pyerr)?
            } else {
                hash_embedding(q, 384)
            };
            embeddings.push(emb);
        }

        let tag_strs: Option<Vec<&str>> = tags
            .as_ref()
            .map(|t| t.iter().map(|s| s.as_str()).collect());
        let tag_refs: Option<&[&str]> = tag_strs.as_deref();
        let time_range = before.map(|b| (0u64, b));

        let hits = db
            .recall_hybrid_multi(&embeddings, Some(&queries), k, tag_refs, time_range)
            .map_err(to_pyerr)?;
        Ok(hits
            .into_iter()
            .map(|(id, score)| SearchResult {
                id: id.to_string(),
                score,
            })
            .collect())
    }

    /// Expanded search: uses the engine's LLM to decompose a query into
    /// sub-queries, then runs adaptive multi-pass retrieval with escalating K.
    ///
    /// Pass 1 (K=10): LLM query expansion + hybrid vector/BM25 search
    /// Pass 2 (K=30): Direct text search with original query for keyword coverage
    /// Pass 3 (K=50): LLM-extracted key nouns for incidental mention recall
    /// Pass 4 (K=100, counting only): Exhaustive BM25 sweep with LLM-generated
    ///   category synonyms — catches long-tail mentions that similarity misses
    ///
    /// All passes merge via RRF (Reciprocal Rank Fusion), so results that
    /// appear in multiple strategies rank highest. This mirrors human memory:
    /// instant recall → active search → deep dig → exhaustive category scan.
    #[pyo3(signature = (query, k=10, provider=None, tags=None, tags_or=None, before=None))]
    fn search_expanded(
        &self,
        query: &str,
        k: usize,
        provider: Option<&str>,
        tags: Option<Vec<String>>,
        tags_or: Option<bool>,
        before: Option<u64>,
    ) -> PyResult<Vec<SearchResult>> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

        let config = build_extraction_config_from_env(provider)?;
        let http_provider = HttpExtractionProvider::new(config.clone()).map_err(to_pyerr)?;

        // Synthesis model — configurable via MENTEDB_SYNTHESIS_MODEL, defaults to same as extraction
        let synth_model =
            std::env::var("MENTEDB_SYNTHESIS_MODEL").unwrap_or_else(|_| config.model.clone());
        let mut synth_config = config.clone();
        synth_config.model = synth_model.clone();
        let synth_provider = HttpExtractionProvider::new(synth_config).map_err(to_pyerr)?;

        let rt = tokio::runtime::Runtime::new().map_err(to_pyerr)?;

        let debug = std::env::var("MENTEDB_DEBUG").is_ok();
        if debug {
            eprintln!("[search_expanded] synthesis model: {}", synth_model);
        }

        let use_tags_or = tags_or.unwrap_or(false);

        // Adaptive K values for escalating retrieval depth
        // With pre-filtered tag search, higher k is cheap — let it flow through
        let k1 = k; // instant recall (multi-query fusion)
        let k2 = k * 2; // active search (BM25 + vector)
        let k3 = k * 3; // deep dig (noun-based)

        let tag_strs: Option<Vec<&str>> = tags
            .as_ref()
            .map(|t| t.iter().map(|s| s.as_str()).collect());
        let tag_refs: Option<&[&str]> = tag_strs.as_deref();
        // Don't use `before` as a hard time filter for search — memory timestamps
        // may reflect extraction time, not original conversation time.
        // `before` is used ONLY as temporal reference for date math computations.
        let time_range: Option<(u64, u64)> = None;
        let temporal_reference = before;

        // --- Pass 1: LLM query expansion + hybrid search (instant recall) ---
        let sub_queries = rt
            .block_on(http_provider.expand_query(query))
            .unwrap_or_default();

        let mut all_queries = vec![query.to_string()];
        // Detect counting intent and extract category keywords (two-tier)
        let mut item_keywords: Option<String> = None;
        let mut broad_keywords: Option<String> = None;
        let mut is_counting = false;
        let query_lower = query.to_lowercase();
        let is_temporal = query_lower.contains("when")
            || query_lower.contains("how long")
            || query_lower.contains("before")
            || query_lower.contains("after")
            || query_lower.contains("first")
            || query_lower.contains("most recent")
            || query_lower.contains("days")
            || query_lower.contains("weeks")
            || query_lower.contains("months ago")
            || query_lower.contains("order")
            || query_lower.contains("ago")
            || query_lower.contains("last saturday")
            || query_lower.contains("last sunday")
            || query_lower.contains("last monday")
            || query_lower.contains("last tuesday")
            || query_lower.contains("last wednesday")
            || query_lower.contains("last thursday")
            || query_lower.contains("last friday")
            || (query_lower.contains("last")
                && (query_lower.contains("week") || query_lower.contains("month")));
        let is_temporal_ordering = is_temporal
            && (query_lower.contains("order")
                || query_lower.contains("earliest")
                || query_lower.contains("latest")
                || query_lower.contains("first")
                || query_lower.contains("most recent"));
        let is_preference = query_lower.contains("suggest")
            || query_lower.contains("recommend")
            || query_lower.contains("any tips")
            || query_lower.contains("any advice")
            || query_lower.contains("what should")
            || query_lower.contains("do you think")
            || query_lower.contains("can you help me find");
        // Knowledge-update detection (moved early for recency boosting)
        let is_knowledge_update = query_lower.contains("current")
            || query_lower.contains("latest")
            || query_lower.contains("now")
            || query_lower.contains("updated")
            || query_lower.contains("changed")
            || query_lower.contains("still")
            || query_lower.contains("most recent")
            || query_lower.contains("switch")
            || query_lower.contains("previously")
            || query_lower.contains("how often")
            || query_lower.contains("how many followers")
            || query_lower.contains("new");
        for sq in &sub_queries {
            if sq.starts_with("ITEM_KEYWORDS:") {
                item_keywords = Some(sq.trim_start_matches("ITEM_KEYWORDS:").trim().to_string());
            } else if sq.starts_with("BROAD_KEYWORDS:") {
                broad_keywords = Some(sq.trim_start_matches("BROAD_KEYWORDS:").trim().to_string());
            } else if sq.starts_with("KEYWORDS:") {
                // Legacy fallback — treat all as item keywords
                item_keywords = Some(sq.trim_start_matches("KEYWORDS:").trim().to_string());
            } else if sq.eq_ignore_ascii_case("COUNTING") {
                is_counting = true;
            } else {
                all_queries.push(sq.clone());
            }
        }
        if debug {
            eprintln!(
                "[search_expanded] is_counting={}, item_kw={:?}, broad_kw={:?}",
                is_counting, item_keywords, broad_keywords
            );
        }

        let mut embeddings = Vec::with_capacity(all_queries.len());
        for q in &all_queries {
            let emb = if let Some(ref embedder) = self.embedder {
                embedder.embed(q).map_err(to_pyerr)?
            } else {
                hash_embedding(q, 384)
            };
            embeddings.push(emb);
        }

        let pass1_hits = db
            .recall_hybrid_multi_mode(
                &embeddings,
                Some(&all_queries),
                k1,
                tag_refs,
                use_tags_or,
                time_range,
            )
            .map_err(to_pyerr)?;

        // --- Pass 2: Direct text search with original query (active search) ---
        // BM25 keyword matching catches literal terms that semantic search misses
        let query_emb = if let Some(ref embedder) = self.embedder {
            embedder.embed(query).map_err(to_pyerr)?
        } else {
            hash_embedding(query, 384)
        };
        let pass2_hits = db
            .recall_hybrid_at_mode(
                &query_emb,
                Some(query),
                k2,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64,
                tag_refs,
                use_tags_or,
                time_range,
            )
            .map_err(to_pyerr)?;

        // --- Pass 3: LLM-extracted key nouns for incidental mentions (deep dig) ---
        let key_nouns_prompt = format!(
            "Extract the 2-3 most important nouns from this question. \
             Return ONLY the nouns separated by spaces, nothing else.\n\
             Question: {}",
            query
        );
        let noun_system =
            "You extract key nouns from questions. Return only nouns, space-separated.";
        let nouns = rt
            .block_on(http_provider.call_text_with_retry(&key_nouns_prompt, noun_system))
            .unwrap_or_default();

        let pass3_hits = if !nouns.trim().is_empty() {
            let noun_emb = if let Some(ref embedder) = self.embedder {
                embedder.embed(nouns.trim()).map_err(to_pyerr)?
            } else {
                hash_embedding(nouns.trim(), 384)
            };
            db.recall_hybrid_at_mode(
                &noun_emb,
                Some(nouns.trim()),
                k3,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64,
                tag_refs,
                use_tags_or,
                time_range,
            )
            .map_err(to_pyerr)?
        } else {
            Vec::new()
        };

        // --- Pass 4: Counting sweep (only for counting/aggregation queries) ---
        // Two-tier individual keyword search. Each keyword gets its own hybrid search
        // to avoid signal dilution. Item keywords (specific subtypes) get full RRF weight;
        // broad keywords (category terms) get reduced weight to prevent noise boosting.
        let mut pass4_item_hits: Vec<(mentedb_core::types::MemoryId, f32)> = Vec::new();
        let mut pass4_broad_hits: Vec<(mentedb_core::types::MemoryId, f32)> = Vec::new();

        if is_counting {
            let k4_per = 15; // results per keyword (increased for exhaustive retrieval)

            // Search each item keyword individually (specific subtypes)
            if let Some(ref kw_str) = item_keywords {
                let terms: Vec<&str> = kw_str
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();
                if debug {
                    eprintln!(
                        "[pass4] Searching {} item keywords individually",
                        terms.len()
                    );
                }
                for term in &terms {
                    let kw_emb = if let Some(ref embedder) = self.embedder {
                        embedder.embed(term).map_err(to_pyerr)?
                    } else {
                        hash_embedding(term, 384)
                    };
                    let hits = db
                        .recall_hybrid_at_mode(
                            &kw_emb,
                            Some(term),
                            k4_per,
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_micros() as u64,
                            tag_refs,
                            use_tags_or,
                            time_range,
                        )
                        .map_err(to_pyerr)?;
                    pass4_item_hits.extend(hits);
                }
            }

            // Search each broad keyword individually (category terms)
            if let Some(ref kw_str) = broad_keywords {
                let terms: Vec<&str> = kw_str
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();
                if debug {
                    eprintln!(
                        "[pass4] Searching {} broad keywords individually",
                        terms.len()
                    );
                }
                for term in &terms {
                    let kw_emb = if let Some(ref embedder) = self.embedder {
                        embedder.embed(term).map_err(to_pyerr)?
                    } else {
                        hash_embedding(term, 384)
                    };
                    let hits = db
                        .recall_hybrid_at_mode(
                            &kw_emb,
                            Some(term),
                            k4_per,
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_micros() as u64,
                            tag_refs,
                            use_tags_or,
                            time_range,
                        )
                        .map_err(to_pyerr)?;
                    pass4_broad_hits.extend(hits);
                }
            }
        }

        // --- Pass 4.5: Categorical retrieval (counting queries) ---
        // Like a human activating a mental category: "health devices" → find everything
        // tagged context:health_device. Searches both entity nodes (by category attribute)
        // and facts (by context: tags propagated from entities).
        let mut categorical_hits: Vec<(MemoryId, f32)> = Vec::new();
        if is_counting {
            // Normalize broad keywords into category-style terms
            let mut category_terms: Vec<String> = Vec::new();
            if let Some(ref kw_str) = broad_keywords {
                for term in kw_str
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                {
                    let normalized = term
                        .to_lowercase()
                        .replace(' ', "_")
                        .trim_end_matches('s')
                        .to_string();
                    category_terms.push(normalized);
                    category_terms.push(term.to_lowercase().replace(' ', "_"));
                }
            }
            if let Some(ref kw_str) = item_keywords {
                for term in kw_str
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                {
                    let normalized = term
                        .to_lowercase()
                        .replace(' ', "_")
                        .trim_end_matches('s')
                        .to_string();
                    category_terms.push(normalized);
                }
            }
            category_terms.sort();
            category_terms.dedup();

            if !category_terms.is_empty() {
                // Search ALL memories (not just graph nodes) for context tags
                let all_mem_ids: Vec<MemoryId> = db.memory_ids().to_vec();
                let before_us_filter = time_range.map(|(_, b)| b);
                let mut found_entities = 0;
                let mut found_facts = 0;
                for mid in &all_mem_ids {
                    if let Ok(node) = db.get_memory(*mid) {
                        // Time filter: skip memories created after question date
                        if let Some(before) = before_us_filter
                            && node.created_at > before
                        {
                            continue;
                        }
                        // Check entity nodes by category attribute
                        if node.tags.iter().any(|t| t.starts_with("entity_name:")) {
                            if let Some(mentedb_core::memory::AttributeValue::String(cat)) =
                                node.attributes.get("category")
                            {
                                let cat_lower = cat.to_lowercase();
                                let cat_parts: Vec<&str> =
                                    cat_lower.split(',').map(|s| s.trim()).collect();
                                for ct in &category_terms {
                                    if cat_parts.contains(&ct.as_str()) {
                                        categorical_hits.push((*mid, 1.0));
                                        found_entities += 1;
                                        break;
                                    }
                                }
                            }
                            continue;
                        }
                        // Check facts by context: tags
                        for ct in &category_terms {
                            let ctx_tag = format!("context:{}", ct);
                            if node.tags.contains(&ctx_tag) {
                                categorical_hits.push((*mid, 1.2)); // Higher weight for direct fact match
                                found_facts += 1;
                                break;
                            }
                        }
                    }
                }
                if debug {
                    eprintln!(
                        "[pass4.5-categorical] Found {} entities + {} facts matching categories {:?}",
                        found_entities, found_facts, category_terms
                    );
                }
            }
        }

        // --- Spreading activation: BFS from seed entities ---
        // Find entity nodes matching query concepts, BFS depth 2, decay 0.85/hop
        let mut activation_hits: Vec<(MemoryId, f32)> = Vec::new();
        {
            let seed_embedding = if let Some(ref embedder) = self.embedder {
                embedder.embed(query).ok()
            } else {
                None
            };
            if let Some(ref seed_emb) = seed_embedding {
                // Find top entity nodes by embedding similarity
                let all_node_ids: Vec<MemoryId> = db.graph().graph().node_ids().to_vec();
                let mut entity_sims: Vec<(MemoryId, f32)> = Vec::new();
                for nid in &all_node_ids {
                    if let Ok(node) = db.get_memory(*nid) {
                        // Time filter: skip nodes created after question date
                        if let Some(ref tr) = time_range
                            && node.created_at > tr.1
                        {
                            continue;
                        }
                        if node.tags.iter().any(|t| t.starts_with("entity_name:")) {
                            let sim = cosine_similarity(seed_emb, &node.embedding);
                            if sim > 0.3 {
                                entity_sims.push((*nid, sim));
                            }
                        }
                    }
                }
                entity_sims
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                entity_sims.truncate(5); // Top 5 seed entities

                // BFS depth 2 with activation decay
                let decay = 0.85f32;
                let threshold = 0.3f32;
                let mut visited: std::collections::HashSet<MemoryId> =
                    std::collections::HashSet::new();
                let mut frontier: Vec<(MemoryId, f32)> =
                    entity_sims.iter().map(|(id, sim)| (*id, *sim)).collect();

                for _depth in 0..2 {
                    let mut next_frontier: Vec<(MemoryId, f32)> = Vec::new();
                    for (nid, activation) in &frontier {
                        if !visited.insert(*nid) {
                            continue;
                        }
                        let propagated = activation * decay;
                        if propagated < threshold {
                            continue;
                        }
                        activation_hits.push((*nid, *activation));

                        let neighbors = db.graph().graph().outgoing(*nid);
                        for (target, _data) in neighbors {
                            if !visited.contains(&target) {
                                next_frontier.push((target, propagated));
                            }
                        }
                        let in_neighbors = db.graph().graph().incoming(*nid);
                        for (source, _data) in in_neighbors {
                            if !visited.contains(&source) {
                                next_frontier.push((source, propagated));
                            }
                        }
                    }
                    frontier = next_frontier;
                }

                if debug && !activation_hits.is_empty() {
                    eprintln!(
                        "[search] Spreading activation found {} nodes",
                        activation_hits.len()
                    );
                }
            }
        }

        // --- Pass 6: Temporal window retrieval (for temporal queries only) ---
        // For queries with relative time references ("X days/weeks ago", "last Saturday"),
        // search a narrow time window around the target date to find the right event.
        let mut temporal_window_hits: Vec<(mentedb_core::types::MemoryId, f32)> = Vec::new();
        let mut temporal_target_us: Option<u64> = None; // Target date in microseconds

        if is_temporal {
            let before_us = temporal_reference.unwrap_or(0);

            if before_us > 0 {
                // Parse temporal offset from query using regex-like matching
                let day_us: u64 = 86_400_000_000; // 1 day in microseconds
                let week_us: u64 = 7 * day_us;

                // Extract "X days/weeks/months ago" patterns
                let mut offset_us: Option<u64> = None;
                let words: Vec<&str> = query_lower.split_whitespace().collect();
                for i in 0..words.len() {
                    if let Ok(n) = words[i].parse::<u64>()
                        && i + 1 < words.len()
                    {
                        let unit = words[i + 1].trim_end_matches('?');
                        if unit.starts_with("day") {
                            offset_us = Some(n * day_us);
                        } else if unit.starts_with("week") {
                            offset_us = Some(n * week_us);
                        } else if unit.starts_with("month") {
                            offset_us = Some(n * 30 * day_us);
                        }
                    }
                }

                // "last Saturday/Sunday/etc." → compute exact weekday offset from before_us
                if offset_us.is_none() && query_lower.contains("last") {
                    // Compute weekday of before_us (question date)
                    // Unix epoch (1970-01-01) was a Thursday (weekday 3, where Mon=0)
                    let question_day_num = (before_us / day_us) as i64; // days since epoch
                    let question_weekday = ((question_day_num + 3) % 7) as u64; // Mon=0..Sun=6

                    let day_targets: &[(&str, u64)] = &[
                        ("saturday", 5),
                        ("sunday", 6),
                        ("monday", 0),
                        ("tuesday", 1),
                        ("wednesday", 2),
                        ("thursday", 3),
                        ("friday", 4),
                    ];
                    for (day_name, target_weekday) in day_targets {
                        if query_lower.contains(day_name) {
                            let mut days_back = (question_weekday as i64 - *target_weekday as i64)
                                .rem_euclid(7)
                                as u64;
                            if days_back == 0 {
                                days_back = 7;
                            }
                            offset_us = Some(days_back * day_us);
                            if debug {
                                eprintln!(
                                    "[temporal] 'last {}': question_weekday={}, target={}, days_back={}",
                                    day_name, question_weekday, target_weekday, days_back
                                );
                            }
                            break;
                        }
                    }
                }

                // Comprehensive relative time expressions
                if offset_us.is_none() {
                    let time_patterns: &[(&str, u64)] = &[
                        ("a week ago", week_us),
                        ("one week ago", week_us),
                        ("two weeks ago", 2 * week_us),
                        ("three weeks ago", 3 * week_us),
                        ("four weeks ago", 4 * week_us),
                        ("a month ago", 30 * day_us),
                        ("one month ago", 30 * day_us),
                        ("two months ago", 60 * day_us),
                        ("three months ago", 90 * day_us),
                        ("four months ago", 120 * day_us),
                        ("six months ago", 180 * day_us),
                        ("a year ago", 365 * day_us),
                        ("last week", week_us),
                        ("last month", 30 * day_us),
                    ];
                    for (pattern, offset) in time_patterns {
                        if query_lower.contains(pattern) {
                            offset_us = Some(*offset);
                            break;
                        }
                    }
                }

                // Numeric N weeks/months/days ago pattern: "5 weeks ago", "3 months ago"
                if offset_us.is_none() {
                    let words: Vec<&str> = query_lower.split_whitespace().collect();
                    for i in 0..words.len().saturating_sub(2) {
                        if words.get(i + 2) == Some(&"ago") {
                            let unit = words[i + 1];
                            let num: Option<u64> =
                                words[i].parse().ok().or_else(|| match words[i] {
                                    "one" => Some(1),
                                    "two" => Some(2),
                                    "three" => Some(3),
                                    "four" => Some(4),
                                    "five" => Some(5),
                                    "six" => Some(6),
                                    "seven" => Some(7),
                                    "eight" => Some(8),
                                    "nine" => Some(9),
                                    "ten" => Some(10),
                                    _ => None,
                                });
                            if let Some(n) = num {
                                let multiplier = if unit.starts_with("week") {
                                    Some(week_us)
                                } else if unit.starts_with("month") {
                                    Some(30 * day_us)
                                } else if unit.starts_with("day") {
                                    Some(day_us)
                                } else if unit.starts_with("year") {
                                    Some(365 * day_us)
                                } else {
                                    None
                                };
                                if let Some(mult) = multiplier {
                                    offset_us = Some(n * mult);
                                    if debug {
                                        eprintln!(
                                            "[temporal] Parsed '{} {} ago' = {} days",
                                            n,
                                            unit,
                                            n * mult / day_us
                                        );
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }

                if let Some(off) = offset_us {
                    let target = before_us.saturating_sub(off);
                    temporal_target_us = Some(target);
                    // Use tighter window (±2 days) for exact day-of-week calculations,
                    // wider (±5 days) for approximate offsets like "3 months ago"
                    let is_exact_day = off <= 7 * day_us; // Week or less = exact weekday math
                    let window_margin = if is_exact_day { 2 * day_us } else { 5 * day_us };
                    let window_start = target.saturating_sub(window_margin);
                    let window_end = std::cmp::min(target + window_margin, before_us);
                    let window_range = Some((window_start, window_end));

                    if debug {
                        eprintln!(
                            "[temporal_window] target={}, window=±{} days (exact={})",
                            target,
                            if is_exact_day { 2 } else { 5 },
                            is_exact_day
                        );
                    }

                    // Semantic search within the time window
                    let window_emb = if let Some(ref embedder) = self.embedder {
                        embedder.embed(query).map_err(to_pyerr)?
                    } else {
                        hash_embedding(query, 384)
                    };
                    let window_hits = db
                        .recall_hybrid_at_mode(
                            &window_emb,
                            Some(query),
                            20,
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_micros() as u64,
                            tag_refs,
                            use_tags_or,
                            window_range,
                        )
                        .map_err(to_pyerr)?;
                    temporal_window_hits = window_hits;
                    if debug {
                        eprintln!(
                            "[temporal_window] Found {} memories in window",
                            temporal_window_hits.len()
                        );
                    }
                }

                // For ordering queries with "past X months" pattern, do a wide window search
                if is_temporal_ordering && temporal_target_us.is_none() {
                    // Try to extract time range for ordering ("past three months", "past two months")
                    let mut range_months: Option<u64> = None;
                    for i in 0..words.len() {
                        if (words[i] == "past" || words[i] == "last") && i + 2 < words.len() {
                            let num_word = words[i + 1];
                            let n = match num_word {
                                "two" | "2" => Some(2u64),
                                "three" | "3" => Some(3),
                                "four" | "4" => Some(4),
                                "five" | "5" => Some(5),
                                "six" | "6" => Some(6),
                                _ => num_word.parse().ok(),
                            };
                            if let Some(months) = n
                                && words[i + 2].starts_with("month")
                            {
                                range_months = Some(months);
                            }
                        }
                    }

                    if let Some(months) = range_months {
                        let wide_start = before_us.saturating_sub(months * 30 * day_us);
                        let wide_range = Some((wide_start, before_us));
                        let wide_emb = if let Some(ref embedder) = self.embedder {
                            embedder.embed(query).map_err(to_pyerr)?
                        } else {
                            hash_embedding(query, 384)
                        };
                        let wide_hits = db
                            .recall_hybrid_at_mode(
                                &wide_emb,
                                Some(query),
                                30,
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_micros() as u64,
                                tag_refs,
                                use_tags_or,
                                wide_range,
                            )
                            .map_err(to_pyerr)?;
                        temporal_window_hits.extend(wide_hits);
                        if debug {
                            eprintln!(
                                "[temporal_window] Ordering wide search: {} months, {} hits",
                                months,
                                temporal_window_hits.len()
                            );
                        }
                    }
                }
            }
        }

        // --- Temporal target date injection ---
        // If we computed a temporal target (e.g., "10 days ago" → specific date),
        // convert it to human-readable date strings and add as additional search queries.
        // This helps BM25 find memories that mention the target date explicitly.
        let mut temporal_date_hits: Vec<(mentedb_core::types::MemoryId, f32)> = Vec::new();
        if let Some(target_us) = temporal_target_us {
            // Convert microseconds to date components
            let target_secs = (target_us / 1_000_000) as i64;
            let days_since_epoch = target_secs / 86400;
            // Simple date calculation from days since unix epoch
            let (year, month, day) = {
                let mut y = 1970i32;
                let mut remaining = days_since_epoch;
                loop {
                    let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                        366
                    } else {
                        365
                    };
                    if remaining < days_in_year {
                        break;
                    }
                    remaining -= days_in_year;
                    y += 1;
                }
                let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
                let days_in_months = [
                    31,
                    if leap { 29 } else { 28 },
                    31,
                    30,
                    31,
                    30,
                    31,
                    31,
                    30,
                    31,
                    30,
                    31,
                ];
                let mut m = 0u32;
                for dm in days_in_months.iter() {
                    if remaining < *dm {
                        break;
                    }
                    remaining -= dm;
                    m += 1;
                }
                (y, m + 1, remaining as u32 + 1)
            };
            let month_names = [
                "",
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ];
            let month_name = month_names[month as usize];
            // Search for memories mentioning this date in various formats
            let date_queries = vec![
                format!("{} {}, {}", month_name, day, year),
                format!("{} {}", month_name, day),
                format!("{}/{:02}/{:02}", year, month, day),
            ];
            if debug {
                eprintln!(
                    "[temporal_date_inject] Target date: {} {}, {} — searching",
                    month_name, day, year
                );
            }
            for dq in &date_queries {
                let date_emb = if let Some(ref embedder) = self.embedder {
                    embedder.embed(dq).map_err(to_pyerr)?
                } else {
                    hash_embedding(dq, 384)
                };
                let date_hits = db
                    .recall_hybrid_at_mode(
                        &date_emb,
                        Some(dq),
                        15,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_micros() as u64,
                        tag_refs,
                        use_tags_or,
                        None,
                    )
                    .map_err(to_pyerr)?;
                temporal_date_hits.extend(date_hits);
            }
            if debug {
                eprintln!(
                    "[temporal_date_inject] Found {} hits for target date",
                    temporal_date_hits.len()
                );
            }
        }

        // --- Pass 7: Contextual inference (multi-hop follow-up) ---
        // For questions requiring cross-memory reasoning (who/where/what about an event),
        // extract key entities from top results and do follow-up searches.
        let mut contextual_hits: Vec<(mentedb_core::types::MemoryId, f32)> = Vec::new();
        let is_contextual = query_lower.contains("who did")
            || query_lower.contains("who was")
            || query_lower.contains("who accompanied")
            || query_lower.contains("who joined")
            || query_lower.contains("who came")
            || query_lower.contains("where did we")
            || query_lower.contains("what happened")
            || query_lower.contains("what did we do")
            || query_lower.contains("who else")
            || query_lower.contains("who went");

        if is_contextual && !pass1_hits.is_empty() {
            // Extract key content snippets from top-5 primary results
            let mut context_keywords: Vec<String> = Vec::new();
            let top_n = std::cmp::min(pass1_hits.len(), 5);
            for (mid, _) in pass1_hits.iter().take(top_n) {
                if let Ok(node) = db.get_memory(*mid) {
                    // Extract proper nouns and key phrases (simple heuristic: capitalized words > 3 chars)
                    for word in node.content.split_whitespace() {
                        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                        if clean.len() > 3
                            && clean.chars().next().is_some_and(|c| c.is_uppercase())
                            && ![
                                "The", "This", "That", "With", "From", "About", "When", "What",
                                "Where", "Which", "User", "They", "Their", "Have", "Been", "Will",
                                "Would", "Could", "Should",
                            ]
                            .contains(&clean)
                        {
                            context_keywords.push(clean.to_string());
                        }
                    }
                }
            }

            // Deduplicate and limit
            context_keywords.sort();
            context_keywords.dedup();
            context_keywords.truncate(10);

            if debug && !context_keywords.is_empty() {
                eprintln!(
                    "[contextual] Extracted keywords from top results: {:?}",
                    context_keywords
                );
            }

            // Build a combined follow-up query from extracted keywords
            if !context_keywords.is_empty() {
                let followup_query = context_keywords.join(" ");
                let followup_emb = if let Some(ref embedder) = self.embedder {
                    embedder.embed(&followup_query).map_err(to_pyerr)?
                } else {
                    hash_embedding(&followup_query, 384)
                };
                let followup_hits = db
                    .recall_hybrid_at_mode(
                        &followup_emb,
                        Some(&followup_query),
                        20,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_micros() as u64,
                        tag_refs,
                        use_tags_or,
                        time_range,
                    )
                    .map_err(to_pyerr)?;
                contextual_hits = followup_hits;
                if debug {
                    eprintln!(
                        "[contextual] Follow-up search found {} hits",
                        contextual_hits.len()
                    );
                }
            }
        }

        // --- Pass 8: Preference retrieval (broader search for preference queries) ---
        // Preference questions fail when relevant memories don't match semantically.
        // Search with preference-specific terms to find scattered preference signals.
        let mut preference_hits: Vec<(mentedb_core::types::MemoryId, f32)> = Vec::new();
        if is_preference {
            // Extract topic from query for targeted preference mining
            let topic_words: Vec<&str> = query_lower
                .split_whitespace()
                .filter(|w| {
                    ![
                        "can",
                        "you",
                        "suggest",
                        "recommend",
                        "a",
                        "an",
                        "the",
                        "for",
                        "my",
                        "me",
                        "i",
                        "do",
                        "any",
                        "some",
                        "good",
                        "best",
                        "upcoming",
                        "trip",
                        "to",
                        "hotel",
                        "what",
                        "should",
                    ]
                    .contains(w)
                })
                .collect();
            let topic = topic_words.join(" ");
            let pref_queries = [
                format!("{} preferences interests likes", query),
                format!("user prefers likes enjoys {}", topic),
                format!("user {} favorite", topic),
                "user likes enjoys prefers favorite hobby interests".to_string(),
            ];
            for pq in &pref_queries {
                let pref_emb = if let Some(ref embedder) = self.embedder {
                    embedder.embed(pq).map_err(to_pyerr)?
                } else {
                    hash_embedding(pq, 384)
                };
                let hits = db
                    .recall_hybrid_at_mode(
                        &pref_emb,
                        Some(pq),
                        40,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_micros() as u64,
                        tag_refs,
                        use_tags_or,
                        time_range,
                    )
                    .map_err(to_pyerr)?;
                preference_hits.extend(hits);
            }
            if debug {
                eprintln!(
                    "[preference] Broad preference search found {} hits",
                    preference_hits.len()
                );
            }
        }

        // --- Pass 9: Knowledge recency retrieval (for knowledge-update queries) ---
        // For "what is my current X", "do I still Y", "what's my latest Z" type questions,
        // search specifically for the most recent memories about the topic.
        let mut recency_hits: Vec<(mentedb_core::types::MemoryId, f32)> = Vec::new();
        if is_knowledge_update {
            // Search with the query but bias heavily towards recency
            let recency_emb = if let Some(ref embedder) = self.embedder {
                embedder.embed(query).map_err(to_pyerr)?
            } else {
                hash_embedding(query, 384)
            };
            // Get a large pool and then we'll sort by recency in post-processing
            let hits = db
                .recall_hybrid_at_mode(
                    &recency_emb,
                    Some(query),
                    40,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_micros() as u64,
                    tag_refs,
                    use_tags_or,
                    time_range,
                )
                .map_err(to_pyerr)?;
            // Sort by created_at descending (most recent first)
            let mut hits_with_ts: Vec<(mentedb_core::types::MemoryId, u64)> = Vec::new();
            for (mid, _score) in &hits {
                if let Ok(node) = db.get_memory(*mid) {
                    hits_with_ts.push((*mid, node.created_at));
                }
            }
            hits_with_ts.sort_by(|a, b| b.1.cmp(&a.1));
            // Take top 15 most recent as high-priority results
            for (mid, _ts) in hits_with_ts.iter().take(15) {
                recency_hits.push((*mid, 1.0));
            }
            if debug {
                eprintln!(
                    "[knowledge-recency] Found {} recent memories for knowledge-update",
                    recency_hits.len()
                );
            }
        }

        use std::collections::HashMap;
        let rrf_k: f32 = 60.0;
        let mut rrf_scores: HashMap<String, f32> = HashMap::new();

        for (rank, (id, _)) in pass1_hits.iter().enumerate() {
            *rrf_scores.entry(id.to_string()).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
        }
        for (rank, (id, _)) in pass2_hits.iter().enumerate() {
            *rrf_scores.entry(id.to_string()).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
        }
        for (rank, (id, _)) in pass3_hits.iter().enumerate() {
            *rrf_scores.entry(id.to_string()).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
        }
        // Pass 4 item keywords: full weight (specific countable subtypes)
        for (rank, (id, _)) in pass4_item_hits.iter().enumerate() {
            *rrf_scores.entry(id.to_string()).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
        }
        // Pass 4 broad keywords: reduced weight (noisy category terms)
        for (rank, (id, _)) in pass4_broad_hits.iter().enumerate() {
            *rrf_scores.entry(id.to_string()).or_insert(0.0) += 0.4 / (rrf_k + rank as f32);
        }
        // Spreading activation: graph-discovered nodes weighted by activation level
        for (rank, (id, _activation)) in activation_hits.iter().enumerate() {
            *rrf_scores.entry(id.to_string()).or_insert(0.0) += 0.6 / (rrf_k + rank as f32);
        }
        // Pass 4.5 categorical: high weight — direct category match bypasses embedding
        for (rank, (id, _)) in categorical_hits.iter().enumerate() {
            *rrf_scores.entry(id.to_string()).or_insert(0.0) += 1.5 / (rrf_k + rank as f32);
        }
        // Pass 6 temporal window: high weight — time-targeted retrieval
        for (rank, (id, _)) in temporal_window_hits.iter().enumerate() {
            *rrf_scores.entry(id.to_string()).or_insert(0.0) += 1.8 / (rrf_k + rank as f32);
        }
        // Pass 7 contextual inference: follow-up search for cross-memory reasoning
        for (rank, (id, _)) in contextual_hits.iter().enumerate() {
            *rrf_scores.entry(id.to_string()).or_insert(0.0) += 1.2 / (rrf_k + rank as f32);
        }
        // Pass 8 preference: broader preference-specific retrieval (boosted weight)
        for (rank, (id, _)) in preference_hits.iter().enumerate() {
            *rrf_scores.entry(id.to_string()).or_insert(0.0) += 1.5 / (rrf_k + rank as f32);
        }
        // Pass 9 knowledge recency: very high weight — most recent memories for knowledge-update
        for (rank, (id, _)) in recency_hits.iter().enumerate() {
            *rrf_scores.entry(id.to_string()).or_insert(0.0) += 2.0 / (rrf_k + rank as f32);
        }
        // Temporal date injection: high weight — BM25 matches on computed target date
        for (rank, (id, _)) in temporal_date_hits.iter().enumerate() {
            *rrf_scores.entry(id.to_string()).or_insert(0.0) += 2.0 / (rrf_k + rank as f32);
        }

        let mut merged: Vec<(String, f32)> = rrf_scores.into_iter().collect();
        merged.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // --- Recency/proximity boost for knowledge-update and temporal queries ---
        if is_knowledge_update || is_temporal {
            let before_us = time_range.map(|(_, b)| b).unwrap_or(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64,
            );

            for (id_str, score) in merged.iter_mut() {
                if let Ok(mem_id) = parse_memory_id(id_str)
                    && let Ok(node) = db.get_memory(mem_id)
                {
                    if is_knowledge_update {
                        // Knowledge-update: strongly prefer most recent version of a fact
                        // Recency score: linear from 0 (oldest) to 1 (newest)
                        let recency = if before_us > 0 {
                            (node.created_at as f64 / before_us as f64).min(1.0) as f32
                        } else {
                            0.0
                        };
                        // Strong boost: recency can contribute up to 0.10 to RRF score
                        // This ensures the most recent memory about a topic outranks older ones
                        *score += recency * 0.10;
                    }

                    // Temporal proximity: if we have a target date, boost memories near it
                    if let Some(target) = temporal_target_us {
                        let dist = node.created_at.abs_diff(target);
                        let day_us: u64 = 86_400_000_000;
                        let days_away = dist as f64 / day_us as f64;
                        // Gaussian-like proximity: max boost at 0 days, decays to ~0 at 7 days
                        let proximity = (-days_away * days_away / 25.0).exp() as f32;
                        *score += proximity * 0.04;
                    }
                }
            }
            // Re-sort after boost
            merged.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            if debug {
                eprintln!(
                    "[proximity] Applied temporal proximity boost (target={:?})",
                    temporal_target_us
                );
            }
        }

        // Counting and temporal queries need more results to ensure completeness
        let final_k = if is_counting || is_temporal {
            std::cmp::max(k, 120)
        } else {
            k
        };
        merged.truncate(final_k);

        // --- Entity graph expansion ---
        // For entity nodes in the results, traverse PartOf edges to include
        // related attribute memories (e.g., entity MAX → breed: Golden Retriever)
        let mut expanded: Vec<(String, f32)> = Vec::with_capacity(merged.len() * 2);
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (id_str, score) in &merged {
            if seen.contains(id_str) {
                continue;
            }
            seen.insert(id_str.clone());
            expanded.push((id_str.clone(), *score));

            // Check if this is an entity node — if so, traverse graph
            if let Ok(mem_id) = parse_memory_id(id_str)
                && let Ok(node) = db.get_memory(mem_id)
            {
                let is_entity = node.tags.iter().any(|t| t.starts_with("entity_name:"));
                if is_entity {
                    // Get subgraph (depth 1) to find PartOf neighbors
                    let (neighbor_ids, edges) = db.graph().get_context_subgraph(mem_id, 1);
                    // Hebbian learning: strengthen traversed edges
                    for edge in &edges {
                        db.graph().strengthen_edge(edge.source, edge.target, 0.02);
                    }
                    for nid in neighbor_ids {
                        let nid_str = nid.to_string();
                        if !seen.contains(&nid_str) {
                            // Time filter: skip neighbors created after question date
                            if let Some(ref tr) = time_range
                                && let Ok(nnode) = db.get_memory(nid)
                                && nnode.created_at > tr.1
                            {
                                continue;
                            }
                            seen.insert(nid_str.clone());
                            // Entity neighbors get a slightly lower score
                            expanded.push((nid_str, score * 0.9));
                        }
                    }
                }
            }
        }
        // Re-sort after expansion and truncate
        expanded
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        expanded.truncate(final_k);

        // --- Pass 5: Iterative gap analysis (counting queries only) ---
        // Inspired by Iter-RetGen (2023) and IRCoT (ACL 2023): use Round 1 results
        // to inform a targeted Round 2 retrieval. The LLM examines what was found
        // and generates specific keywords for items that might be missing.
        if (is_counting || is_temporal) && !expanded.is_empty() {
            // Collect top-20 memory contents for the LLM to analyze
            let gap_limit = std::cmp::min(expanded.len(), 30);
            let mut found_items: Vec<String> = Vec::new();
            for (id_str, _) in expanded.iter().take(gap_limit) {
                if let Ok(mem_id) = parse_memory_id(id_str)
                    && let Ok(node) = db.get_memory(mem_id)
                {
                    let is_entity = node.tags.iter().any(|t| t.starts_with("entity_name:"));
                    let is_community = node.tags.iter().any(|t| t == "community_summary");
                    if is_entity || is_community {
                        continue;
                    }
                    found_items.push(node.content.clone());
                }
            }

            if !found_items.is_empty() {
                let items_text = found_items
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let snip = &c[..c.floor_char_boundary(std::cmp::min(c.len(), 200))];
                        format!("[{}] {}", i, snip)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let gap_prompt = format!(
                    "Question: {}\n\n\
                     Here are the top memories found so far:\n{}\n\n\
                     Based on these results, what specific items or instances might be MISSING \
                     that would be needed for a complete answer?\n\
                     Return a JSON object with:\n\
                     - \"found\": brief list of distinct items already found\n\
                     - \"gap_keywords\": comma-separated specific search terms for items that \
                       might exist but weren't found yet (be specific — use actual item names/types, \
                       not categories)\n\n\
                     If the results look complete, return empty gap_keywords.\n\
                     Example: {{\"found\": [\"emerald earrings\", \"engagement ring\"], \
                     \"gap_keywords\": \"necklace, bracelet, pendant, brooch, anklet\"}}",
                    query, items_text
                );
                let gap_system = "You analyze search results for completeness. Identify what specific items might be missing. Return JSON only.";

                match rt.block_on(http_provider.call_with_retry(&gap_prompt, gap_system)) {
                    Ok(response) => {
                        let cleaned = response
                            .trim()
                            .trim_start_matches("```json")
                            .trim_end_matches("```")
                            .trim();
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(cleaned) {
                            if let Some(gap_kw) = json.get("gap_keywords").and_then(|v| v.as_str())
                            {
                                let gap_terms: Vec<&str> = gap_kw
                                    .split(',')
                                    .map(|s| s.trim())
                                    .filter(|s| !s.is_empty())
                                    .collect();
                                if debug {
                                    let found = json
                                        .get("found")
                                        .map(|v| v.to_string())
                                        .unwrap_or_default();
                                    eprintln!("[pass5-gap] Found: {}", found);
                                    eprintln!("[pass5-gap] Gap keywords: {:?}", gap_terms);
                                }

                                if !gap_terms.is_empty() {
                                    // Round 2: search each gap keyword individually
                                    let mut pass5_hits: Vec<(String, f32)> = Vec::new();
                                    let k5_per = 8;
                                    for term in &gap_terms {
                                        let kw_emb = if let Some(ref embedder) = self.embedder {
                                            embedder.embed(term).map_err(to_pyerr)?
                                        } else {
                                            hash_embedding(term, 384)
                                        };
                                        let hits = db
                                            .recall_hybrid_at_mode(
                                                &kw_emb,
                                                Some(term),
                                                k5_per,
                                                std::time::SystemTime::now()
                                                    .duration_since(std::time::UNIX_EPOCH)
                                                    .unwrap_or_default()
                                                    .as_micros()
                                                    as u64,
                                                tag_refs,
                                                use_tags_or,
                                                time_range,
                                            )
                                            .map_err(to_pyerr)?;
                                        for (id, score) in hits {
                                            let id_str = id.to_string();
                                            pass5_hits.push((id_str, score));
                                        }
                                    }

                                    if debug {
                                        eprintln!(
                                            "[pass5-gap] Round 2 retrieved {} hits from {} gap keywords",
                                            pass5_hits.len(),
                                            gap_terms.len()
                                        );
                                    }

                                    // Merge Round 2 into expanded results with full RRF weight
                                    // (these are targeted, high-signal searches)
                                    let mut existing_scores: HashMap<String, f32> =
                                        expanded.iter().map(|(id, s)| (id.clone(), *s)).collect();
                                    for (rank, (id_str, _)) in pass5_hits.iter().enumerate() {
                                        let rrf_contrib = 1.0 / (rrf_k + rank as f32);
                                        *existing_scores.entry(id_str.clone()).or_insert(0.0) +=
                                            rrf_contrib;
                                    }

                                    // Rebuild expanded from merged scores
                                    expanded = existing_scores.into_iter().collect();
                                    expanded.sort_unstable_by(|a, b| {
                                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                                    });
                                    expanded.truncate(final_k);
                                }
                            }
                        } else if debug {
                            eprintln!(
                                "[pass5-gap] Failed to parse gap analysis response: {}",
                                cleaned
                            );
                        }
                    }
                    Err(e) => {
                        if debug {
                            eprintln!("[pass5-gap] LLM call failed: {e}");
                        }
                    }
                }
            }
        }

        // Track the current synthesis ID to avoid filtering it out later
        let mut current_synth_id: Option<String> = None;

        // --- Cognitive re-ranking + reconstructive synthesis for counting queries ---
        // Two-phase approach inspired by RankGPT (EMNLP 2023) and Chain-of-Noting:
        //
        // Phase 1 (Re-ranking): Score memories for relevance. DOES NOT modify retrieval
        //   scores — only used to select what goes into synthesis. This avoids the
        //   "coin flip" problem where LLM non-determinism buries relevant results.
        //
        // Phase 2 (Synthesis): Feed a UNION of re-ranker picks + retrieval top-K to
        //   a synthesis LLM that produces a structured summary. This summary becomes
        //   position [0] in the results — the reader's primary evidence.
        //
        // The original retrieval order is preserved for positions [1..N].
        if !expanded.is_empty() {
            // --- Phase 1: Score memories for synthesis selection ---
            if debug {
                eprintln!(
                    "[rerank] Starting cognitive re-ranking for {} results",
                    expanded.len()
                );
            }
            let rerank_limit = std::cmp::min(expanded.len(), 30);
            let mut memory_contents: Vec<(String, String)> = Vec::new(); // (id, content)
            for (id_str, _score) in expanded.iter().take(rerank_limit) {
                if let Ok(mem_id) = parse_memory_id(id_str)
                    && let Ok(node) = db.get_memory(mem_id)
                {
                    // Skip entity/community/old-synthesis nodes from reranker
                    let is_entity = node.tags.iter().any(|t| t.starts_with("entity_name:"));
                    let is_community = node.tags.iter().any(|t| t == "community_summary");
                    let is_old_synth = node.tags.iter().any(|t| t == "synthesis:true");
                    if is_entity || is_community || is_old_synth {
                        continue;
                    }
                    memory_contents.push((id_str.clone(), node.content.clone()));
                }
            }

            // Collect IDs that the re-ranker considers relevant (score >= 1)
            let mut reranker_picks: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            if !memory_contents.is_empty() {
                let mut memories_text = String::new();
                for (i, (_id, content)) in memory_contents.iter().enumerate() {
                    memories_text.push_str(&format!("[{}] {}\n", i, content));
                }

                let rerank_prompt = if is_counting {
                    format!(
                        "Question: {}\n\n\
                         Memories:\n{}\n\
                         This is a COUNTING question. Score each memory 0-10:\n\
                         - 10 = directly mentions a specific item/instance that should be counted\n\
                         - 7 = mentions a class, activity, item, or event in the relevant category\n\
                         - 3 = tangentially related but not a countable instance\n\
                         - 0 = clearly unrelated\n\n\
                         IMPORTANT: Score {} memories. Any memory mentioning a specific named item \
                         in the category being counted MUST score >= 7.\n\n\
                         Return ONLY a JSON array of {} integer scores, e.g. [8, 2, 10, 0, ...]",
                        query,
                        memories_text,
                        memory_contents.len(),
                        memory_contents.len()
                    )
                } else {
                    format!(
                        "Question: {}\n\n\
                         Memories:\n{}\n\
                         Score each memory 0-10 for how relevant it is to answering the question.\n\
                         - 10 = directly answers or provides key evidence\n\
                         - 5 = provides useful supporting context\n\
                         - 0 = clearly unrelated\n\n\
                         Return ONLY a JSON array of {} integer scores, e.g. [8, 2, 10, 0, ...]",
                        query,
                        memories_text,
                        memory_contents.len()
                    )
                };
                let rerank_system = "You are a relevance scorer. Return ONLY a JSON array of integer scores 0-10, one score per memory in order. No explanation, no text outside the array.";

                match rt.block_on(http_provider.call_text_with_retry(&rerank_prompt, rerank_system))
                {
                    Ok(response) => {
                        if debug {
                            eprintln!("[rerank] LLM response: {}", response.trim());
                        }
                        let trimmed = response.trim();
                        let array_str = if let Some(start) = trimmed.find('[') {
                            if let Some(end) = trimmed.rfind(']') {
                                &trimmed[start..=end]
                            } else {
                                trimmed
                            }
                        } else {
                            trimmed
                        };

                        match serde_json::from_str::<Vec<f32>>(array_str) {
                            Ok(mut scores) => {
                                // Pad or truncate to match memory count
                                scores.resize(memory_contents.len(), 0.0);
                                if debug {
                                    eprintln!(
                                        "[rerank] Parsed {} scores (adjusted to {}): {:?}",
                                        scores.len(),
                                        memory_contents.len(),
                                        scores
                                    );
                                }
                                for (i, (id, content)) in memory_contents.iter().enumerate() {
                                    let relevance = scores.get(i).copied().unwrap_or(0.0);
                                    let snip = {
                                        let max = std::cmp::min(content.len(), 80);
                                        let end = content.floor_char_boundary(max);
                                        &content[..end]
                                    };
                                    if debug {
                                        eprintln!(
                                            "[rerank]   [{}] score={} | {}",
                                            i, relevance, snip
                                        );
                                    }
                                    if relevance >= 1.0 {
                                        reranker_picks.insert(id.clone());
                                    }
                                }
                                if debug {
                                    eprintln!(
                                        "[rerank] Selected {} memories for synthesis",
                                        reranker_picks.len()
                                    );
                                }
                            }
                            Err(e) => {
                                if debug {
                                    eprintln!("[rerank] Failed to parse scores: {e}");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if debug {
                            eprintln!("[rerank] LLM call failed: {e}");
                        }
                    }
                }
            }

            // --- Phase 2: Reconstructive synthesis ---
            // Feed UNION of: (a) re-ranker picks + (b) top-K by original retrieval score.
            // This ensures nothing is lost even if the re-ranker misscores items.
            let retrieval_top_k = if is_temporal {
                std::cmp::min(expanded.len(), 50)
            } else if is_counting {
                std::cmp::min(expanded.len(), 40)
            } else if is_preference {
                std::cmp::min(expanded.len(), 35)
            } else if is_knowledge_update {
                std::cmp::min(expanded.len(), 30)
            } else {
                std::cmp::min(expanded.len(), 20)
            };
            let mut synth_ids: Vec<String> = Vec::new();
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

            // First: add all re-ranker picks (scored >= 1)
            for (id_str, _score) in expanded.iter() {
                if reranker_picks.contains(id_str) && seen.insert(id_str.clone()) {
                    synth_ids.push(id_str.clone());
                }
            }
            // Then: add top-K by retrieval score as safety net
            for (id_str, _score) in expanded.iter().take(retrieval_top_k) {
                if seen.insert(id_str.clone()) {
                    synth_ids.push(id_str.clone());
                }
            }

            if debug {
                eprintln!(
                    "[synthesis] Feeding {} memories ({} from reranker + {} from retrieval top-{})",
                    synth_ids.len(),
                    reranker_picks.len(),
                    synth_ids.len() - reranker_picks.len(),
                    retrieval_top_k
                );
            }

            // --- Detect multi-session and question category ---
            let mut session_set: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            for id_str in &synth_ids {
                if let Ok(mem_id) = parse_memory_id(id_str)
                    && let Ok(node) = db.get_memory(mem_id)
                {
                    for t in &node.tags {
                        if let Some(s) = t.strip_prefix("session:") {
                            session_set.insert(s.to_string());
                        }
                    }
                }
            }
            let is_multi_session = session_set.len() > 3;

            // Per-category evidence budget (max items to feed to reader)
            let evidence_budget: usize = if is_multi_session {
                40
            } else if is_temporal {
                50
            } else if is_preference {
                35
            } else if is_knowledge_update {
                25
            } else {
                20
            };

            if debug {
                eprintln!(
                    "[synthesis] multi_session={}, temporal={}, ku={}, budget={}",
                    is_multi_session, is_temporal, is_knowledge_update, evidence_budget
                );
            }

            // Build evidence with session date labels and assistant filtering
            #[allow(dead_code)]
            struct EvidenceItem {
                session: String,
                date: String,
                content: String,
                is_assistant_turn: bool,
            }
            let mut evidence_items: Vec<EvidenceItem> = Vec::new();
            for id_str in &synth_ids {
                if let Ok(mem_id) = parse_memory_id(id_str)
                    && let Ok(node) = db.get_memory(mem_id)
                {
                    let is_entity = node.tags.iter().any(|t| t.starts_with("entity_name:"));
                    let is_community = node.tags.iter().any(|t| t == "community_summary");
                    if is_entity || is_community {
                        continue;
                    }

                    let is_assistant_turn = node.tags.iter().any(|t| t == "turn:assistant");

                    // For multi-session: filter out assistant turns (they drown out user facts)
                    if is_multi_session && is_assistant_turn {
                        continue;
                    }

                    // Time filter: skip memories created after the question date
                    if let Some(ref tr) = time_range
                        && node.created_at > tr.1
                        && !node.tags.iter().any(|t| t.starts_with("date:"))
                    {
                        continue;
                    }

                    let session = node
                        .tags
                        .iter()
                        .find_map(|t| t.strip_prefix("session:").map(|s| s.to_string()))
                        .unwrap_or_default();
                    let raw_date = node
                        .tags
                        .iter()
                        .find_map(|t| t.strip_prefix("date:").map(|s| s.to_string()))
                        .or_else(|| format_timestamp_date(node.created_at))
                        .unwrap_or_default();
                    // Suppress dates that are obviously wrong (after question date)
                    let date = if let Some(ref tr) = time_range {
                        if node.created_at > tr.1
                            && !node.tags.iter().any(|t| t.starts_with("date:"))
                        {
                            String::new() // Don't show phantom future dates
                        } else {
                            raw_date
                        }
                    } else {
                        raw_date
                    };

                    evidence_items.push(EvidenceItem {
                        session,
                        date,
                        content: node.content.clone(),
                        is_assistant_turn,
                    });
                }
            }

            // Sort chronologically for temporal/KU queries so the LLM sees events in time order
            if is_temporal || is_knowledge_update {
                evidence_items.sort_by(|a, b| a.date.cmp(&b.date));
            }

            // Deduplicate near-identical content (saves token budget)
            // Two memories with >90% character overlap are likely duplicates from extraction
            let mut deduped_items: Vec<EvidenceItem> = Vec::new();
            for item in evidence_items {
                let dominated = deduped_items.iter().any(|existing| {
                    // Quick length check first
                    let len_a = existing.content.len();
                    let len_b = item.content.len();
                    if len_a.abs_diff(len_b) * 10 > std::cmp::max(len_a, len_b) {
                        return false; // Length differs by >10%, not a dupe
                    }
                    // Compare first 100 chars
                    let prefix_a = &existing.content[..existing
                        .content
                        .floor_char_boundary(std::cmp::min(100, len_a))];
                    let prefix_b = &item.content
                        [..item.content.floor_char_boundary(std::cmp::min(100, len_b))];
                    prefix_a == prefix_b
                });
                if !dominated {
                    deduped_items.push(item);
                }
            }
            let evidence_items = deduped_items;

            if debug && evidence_items.len() < evidence_budget {
                eprintln!(
                    "[synthesis] Deduped to {} items (budget={})",
                    evidence_items.len(),
                    evidence_budget
                );
            }

            // Apply evidence budget
            let evidence_items: Vec<EvidenceItem> =
                evidence_items.into_iter().take(evidence_budget).collect();

            // Group by session and inject session date headers
            let mut synth_contents: Vec<String> = Vec::new();
            let mut current_session = String::new();
            for item in &evidence_items {
                if !item.session.is_empty() && item.session != current_session {
                    current_session = item.session.clone();
                    if !item.date.is_empty() {
                        synth_contents.push(format!("--- [Session: {}] ---", item.date));
                    }
                }
                let prefix = if !item.date.is_empty() {
                    format!("[{}] ", item.date)
                } else {
                    String::new()
                };
                synth_contents.push(format!("{}{}", prefix, item.content));
            }

            if !synth_contents.is_empty() {
                let evidence = synth_contents
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("[{}] {}", i, c))
                    .collect::<Vec<_>>()
                    .join("\n");

                // --- Graph-informed synthesis for counting queries ---
                // Query entity nodes to build structured context: entity names, relationships, categories.
                // If graph has entities with category/relationship data, use structured context.
                // Otherwise fall back to flat evidence.
                let mut graph_entities: Vec<(String, String, String)> = Vec::new(); // (name, relationship, details)
                let mut graph_count: Option<usize> = None;

                if is_counting {
                    // Scan entity nodes in our result set for structured attributes
                    for id_str in &synth_ids {
                        if let Ok(mem_id) = parse_memory_id(id_str)
                            && let Ok(node) = db.get_memory(mem_id)
                        {
                            let is_entity = node.tags.iter().any(|t| t.starts_with("entity_name:"));
                            if !is_entity {
                                continue;
                            }

                            let name = node
                                .attributes
                                .get("entity_name")
                                .and_then(|v| match v {
                                    mentedb_core::memory::AttributeValue::String(s) => {
                                        Some(s.clone())
                                    }
                                    _ => None,
                                })
                                .unwrap_or_default();
                            let relationship = node
                                .attributes
                                .get("relationship")
                                .and_then(|v| match v {
                                    mentedb_core::memory::AttributeValue::String(s) => {
                                        Some(s.clone())
                                    }
                                    _ => None,
                                })
                                .unwrap_or_else(|| "unknown".to_string());
                            let etype = node
                                .attributes
                                .get("entity_type")
                                .and_then(|v| match v {
                                    mentedb_core::memory::AttributeValue::String(s) => {
                                        Some(s.clone())
                                    }
                                    _ => None,
                                })
                                .unwrap_or_default();

                            // Build details from other attributes
                            let mut details = Vec::new();
                            for (k, v) in &node.attributes {
                                if k == "entity_name"
                                    || k == "entity_type"
                                    || k == "relationship"
                                    || k == "category"
                                    || k == "relationship_owner"
                                {
                                    continue;
                                }
                                if let mentedb_core::memory::AttributeValue::String(s) = v {
                                    details.push(format!("{}: {}", k, s));
                                }
                            }
                            let owner_info =
                                node.attributes
                                    .get("relationship_owner")
                                    .and_then(|v| match v {
                                        mentedb_core::memory::AttributeValue::String(s) => {
                                            Some(s.clone())
                                        }
                                        _ => None,
                                    });

                            let detail_str = if details.is_empty() {
                                etype
                            } else {
                                format!("{}, {}", etype, details.join(", "))
                            };
                            let rel_str = if let Some(owner) = owner_info {
                                format!("{} ({})", relationship, owner)
                            } else {
                                relationship.clone()
                            };

                            if !name.is_empty() {
                                graph_entities.push((name, rel_str, detail_str));
                            }
                        }
                    }

                    // Also scan the full graph for entity nodes we might have missed
                    // by looking at entity nodes connected to found memories via PartOf edges
                    let already_found: std::collections::HashSet<String> = graph_entities
                        .iter()
                        .map(|(n, _, _)| n.to_lowercase())
                        .collect();
                    for id_str in &synth_ids {
                        if let Ok(mem_id) = parse_memory_id(id_str) {
                            // Check outgoing edges for PartOf connections to entity nodes
                            let neighbors = db.graph().graph().outgoing(mem_id);
                            for (neighbor_id, stored_edge) in neighbors {
                                if stored_edge.edge_type != EdgeType::PartOf {
                                    continue;
                                }
                                if let Ok(neighbor_node) = db.get_memory(neighbor_id) {
                                    let is_entity = neighbor_node
                                        .tags
                                        .iter()
                                        .any(|t| t.starts_with("entity_name:"));
                                    if !is_entity {
                                        continue;
                                    }
                                    let name = neighbor_node
                                        .attributes
                                        .get("entity_name")
                                        .and_then(|v| match v {
                                            mentedb_core::memory::AttributeValue::String(s) => {
                                                Some(s.clone())
                                            }
                                            _ => None,
                                        })
                                        .unwrap_or_default();
                                    if name.is_empty()
                                        || already_found.contains(&name.to_lowercase())
                                    {
                                        continue;
                                    }
                                    let relationship = neighbor_node
                                        .attributes
                                        .get("relationship")
                                        .and_then(|v| match v {
                                            mentedb_core::memory::AttributeValue::String(s) => {
                                                Some(s.clone())
                                            }
                                            _ => None,
                                        })
                                        .unwrap_or_else(|| "unknown".to_string());
                                    let etype = neighbor_node
                                        .attributes
                                        .get("entity_type")
                                        .and_then(|v| match v {
                                            mentedb_core::memory::AttributeValue::String(s) => {
                                                Some(s.clone())
                                            }
                                            _ => None,
                                        })
                                        .unwrap_or_default();
                                    graph_entities.push((name, relationship, etype));
                                }
                            }
                        }
                    }

                    if !graph_entities.is_empty() {
                        // Compute graph-based count: items where user directly owns/uses/attends
                        let user_owned: Vec<&(String, String, String)> = graph_entities
                            .iter()
                            .filter(|(_, rel, _)| {
                                let r = rel.to_lowercase();
                                !r.contains("someone_else")
                                    && !r.contains("considering")
                                    && !r.contains("wants")
                                    && !r.contains("previously")
                                    && (r.contains("owns")
                                        || r.contains("uses")
                                        || r.contains("attends")
                                        || r.contains("plays")
                                        || r.contains("member")
                                        || r == "unknown")
                            })
                            .collect();
                        graph_count = Some(user_owned.len());
                        if debug {
                            eprintln!(
                                "[graph-synthesis] Found {} entities, {} user-owned:",
                                graph_entities.len(),
                                user_owned.len()
                            );
                            for (name, rel, det) in &graph_entities {
                                eprintln!("[graph-synthesis]   {} — {} ({})", name, rel, det);
                            }
                        }
                    }
                }

                // Build the synthesis prompt — use graph structure if available
                let synth_prompt = if let Some(count) = graph_count {
                    // Graph-informed: structured entity list + evidence for verification
                    let entity_list = graph_entities
                        .iter()
                        .enumerate()
                        .map(|(i, (name, rel, det))| {
                            format!("{}. {} ({}) — {}", i + 1, name, det, rel)
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    format!(
                        "Question: {}\n\n\
                         Entities found in memory graph:\n{}\n\n\
                         Evidence from memory:\n{}\n\n\
                         The graph found {} distinct entities. Using the evidence, verify and answer.\n\n\
                         RULES:\n\
                         - ONLY use facts from the evidence above — do not invent items\n\
                         - Quote numbers and prices EXACTLY as stated. Add them up when the question asks for a total.\n\
                         - Dates must come from evidence — do not confuse months\n\
                         - If no evidence matches the question's time period, say so\n\
                         - When counting: include every item the user HAS or USES, even occasionally\n\
                         - When counting: exclude items belonging to others, or items only being considered/planned\n\
                         - When in doubt about whether to include an item, INCLUDE it\n\n\
                          TEMPORAL REASONING (for time-related questions):\n\
                         - Evidence is sorted CHRONOLOGICALLY — earlier events appear first\n\
                         - Extract the EXACT date for each relevant event, citing [N] for each\n\
                         - For 'how many days/weeks/months between X and Y': find the exact date of EACH event, then COUNT step by step\n\
                         - For 'which happened first/most recently': compare the extracted dates directly\n\
                         - For 'how many weeks/months ago': calculate from the question's context date to the event date\n\
                         - For ordering questions: list ALL events of the requested type found in evidence, do NOT skip any\n\
                         - SHOW YOUR DATE MATH: state each date, compute the difference, then VERIFY your arithmetic\n\
                         - If your calculation contradicts your stated answer, TRUST THE CALCULATION\n\
                         - For 'last Saturday/Sunday/etc': resolve the relative day using the session dates in the evidence\n\
                         - Each piece of evidence may come from a DIFFERENT date — do not assume they happened on the same day\n\n\
                         - For questions about current/latest values: report the MOST RECENT value from evidence (latest date overrides earlier ones)\n\
                         - If you see the same fact with different values at different dates, use the newest one\n\n\
                         ABSTENTION RULES:\n\
                         - If the topic is NEVER mentioned in evidence, say 'I don't have information about [topic] in our conversations'\n\
                         - Absence of evidence is NOT evidence of zero — do NOT say 'Total: 0' if the topic simply wasn't discussed\n\
                         - 'Planning to' or 'thinking about' acquiring X does NOT mean they have X\n\
                         - If evidence mentions someone ELSE doing X, that does NOT answer whether the USER did X\n\
                         - For ordering/sequence questions: if you can find dates for all items, you CAN answer even if uncertain about other things\n\n\
                         PREFERENCE / RECOMMENDATION QUESTIONS:\n\
                         - For 'suggest/recommend' questions: use the user's stated preferences, interests, and habits from evidence to give a personalized answer\n\
                         - Do NOT abstain on recommendation questions — instead use whatever preferences and context you find in evidence\n\
                         - Combine multiple preference signals (e.g., likes outdoor activities + prefers mornings → suggest morning hike)\n\n\
                         List each item with a citation [N] to the evidence entry that supports it, then state the total.",
                        query, entity_list, evidence, count
                    )
                } else {
                    // Fallback: flat evidence (no graph structure available)
                    // Build temporal context header if we have temporal info
                    let temporal_context = if let Some(ref_us) = temporal_reference.filter(|_| is_temporal) {
                        let ref_secs = (ref_us / 1_000_000) as i64;
                        let ref_days = ref_secs / 86400;
                        let (ry, rm, rd) = {
                            let mut y = 1970i32;
                            let mut remaining = ref_days;
                            loop {
                                let diy = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                                    366
                                } else {
                                    365
                                };
                                if remaining < diy {
                                    break;
                                }
                                remaining -= diy;
                                y += 1;
                            }
                            let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
                            let dim = [
                                31,
                                if leap { 29 } else { 28 },
                                31,
                                30,
                                31,
                                30,
                                31,
                                31,
                                30,
                                31,
                                30,
                                31,
                            ];
                            let mut m = 0u32;
                            for d in dim.iter() {
                                if remaining < *d {
                                    break;
                                }
                                remaining -= d;
                                m += 1;
                            }
                            (y, m + 1, remaining as u32 + 1)
                        };
                        let month_names = [
                            "",
                            "January",
                            "February",
                            "March",
                            "April",
                            "May",
                            "June",
                            "July",
                            "August",
                            "September",
                            "October",
                            "November",
                            "December",
                        ];
                        let mut ctx = format!(
                            "\n[TEMPORAL CONTEXT] The question is being asked on: {} {}, {}\n",
                            month_names[rm as usize], rd, ry
                        );
                        if let Some(target) = temporal_target_us {
                            let t_secs = (target / 1_000_000) as i64;
                            let t_days = t_secs / 86400;
                            let (ty, tm, td) = {
                                let mut y = 1970i32;
                                let mut remaining = t_days;
                                loop {
                                    let diy = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                                        366
                                    } else {
                                        365
                                    };
                                    if remaining < diy {
                                        break;
                                    }
                                    remaining -= diy;
                                    y += 1;
                                }
                                let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
                                let dim = [
                                    31,
                                    if leap { 29 } else { 28 },
                                    31,
                                    30,
                                    31,
                                    30,
                                    31,
                                    31,
                                    30,
                                    31,
                                    30,
                                    31,
                                ];
                                let mut m = 0u32;
                                for d in dim.iter() {
                                    if remaining < *d {
                                        break;
                                    }
                                    remaining -= d;
                                    m += 1;
                                }
                                (y, m + 1, remaining as u32 + 1)
                            };
                            ctx.push_str(&format!("[COMPUTED TARGET DATE] The referenced time resolves to: {} {}, {}\n", month_names[tm as usize], td, ty));
                        }
                        ctx
                    } else {
                        String::new()
                    };
                    format!(
                        "Question: {}\n{}\n\
                         Evidence from memory:\n{}\n\n\
                         Answer the question using ONLY the evidence above.\n\n\
                         RULES:\n\
                         - ONLY use facts from the evidence — do not invent items\n\
                         - Quote numbers and prices EXACTLY as stated. Add them up when the question asks for a total.\n\
                         - Dates must come from evidence — do not confuse months\n\
                         - If no evidence matches the question's time period, say so\n\
                         - When counting: include every item the user HAS or USES, even occasionally\n\
                         - When counting: exclude items belonging to others, or items only being considered/planned\n\
                         - When in doubt about whether to include an item, INCLUDE it\n\n\
                         TEMPORAL REASONING (for time-related questions):\n\
                         - Evidence is sorted CHRONOLOGICALLY — earlier events appear first\n\
                         - Extract the EXACT date for each relevant event, citing [N] for each\n\
                         - For 'how many days/weeks/months between X and Y': find the exact date of EACH event, then COUNT step by step\n\
                         - For 'which happened first/most recently': compare the extracted dates directly\n\
                         - For 'how many weeks/months ago': calculate from the question's context date to the event date\n\
                         - For ordering questions: list ALL events of the requested type found in evidence, do NOT skip any\n\
                         - SHOW YOUR DATE MATH: state each date, compute the difference, then VERIFY your arithmetic\n\
                         - If your calculation contradicts your stated answer, TRUST THE CALCULATION\n\
                         - For 'last Saturday/Sunday/etc': resolve the relative day using the session dates in the evidence\n\
                         - Each piece of evidence may come from a DIFFERENT date — do not assume they happened on the same day\n\n\
                         KNOWLEDGE UPDATES:\n\
                         - For questions about current/latest values: report the MOST RECENT value from evidence (latest date overrides earlier ones)\n\
                         - If you see the same fact with different values at different dates, ALWAYS use the newest one — even if older values appear more often\n\
                         - For 'previously vs now' questions: identify the CHANGE by finding the old value AND the new value with their dates\n\
                         - Frequency changes (e.g., 'every week' → 'every other week'): state both the old and new frequency\n\n\
                         DURATION AGGREGATION:\n\
                         - For 'how many weeks/months total spent on X, Y, and Z': find the INDIVIDUAL duration for each item, then ADD them\n\
                         - Do NOT conflate duration with time-since — 'spent 2 weeks reading' means duration=2 weeks, not when it happened\n\
                         - State each duration separately, then sum: e.g., '2 + 4 + 2 = 8 weeks total'\n\n\
                         ABSTENTION RULES:\n\
                         - If the topic is NEVER mentioned in evidence, say 'I don't have information about [topic] in our conversations'\n\
                         - Absence of evidence is NOT evidence of zero — do NOT say 'Total: 0' if the topic simply wasn't discussed\n\
                         - 'Planning to' or 'thinking about' acquiring X does NOT mean they have X\n\
                         - If evidence mentions someone ELSE doing X, that does NOT answer whether the USER did X\n\
                         - For ordering/sequence questions: if you can find dates for all items, you CAN answer even if uncertain about other things\n\n\
                         PREFERENCE / RECOMMENDATION QUESTIONS:\n\
                         - For 'suggest/recommend' questions: use the user's stated preferences, interests, and habits from evidence to give a personalized answer\n\
                         - Do NOT abstain on recommendation questions — instead use whatever preferences and context you find in evidence\n\
                         - Combine multiple preference signals (e.g., likes outdoor activities + prefers mornings → suggest morning hike)\n\n\
                         List each item with a citation [N] to the evidence entry that supports it, then state the total.",
                        query, temporal_context, evidence
                    )
                };
                let synth_system = "You recall facts from memory evidence. Be thorough — list every relevant item. When in doubt, include it. NEVER invent facts not in the evidence. Quote numbers exactly, add them up for totals. For temporal questions: use the [TEMPORAL CONTEXT] dates provided — they are pre-computed and correct. Find the date of each event in evidence, show your calculation, then answer. For knowledge-update questions: report the MOST RECENT value. For preference/recommendation questions: you MUST provide a recommendation based on user's interests and habits — NEVER abstain on these. If the topic was never discussed, say so — do not guess.";

                match rt.block_on(synth_provider.call_text_with_retry(&synth_prompt, synth_system))
                {
                    Ok(synthesis) => {
                        let mut final_synthesis = synthesis.trim().to_string();

                        // --- Temporal date computation pass ---
                        // For "how many days/weeks between X and Y" or "X weeks/months ago" questions,
                        // extract dates from evidence and compute the answer in code (LLMs are bad at date math).
                        let is_temporal_math = query_lower.contains("how many days")
                            || query_lower.contains("how many weeks")
                            || query_lower.contains("how many months")
                            || query_lower.contains("how long")
                            || (query_lower.contains("weeks ago")
                                && query_lower.contains("how many"))
                            || (query_lower.contains("months ago")
                                && query_lower.contains("how many"));

                        let mut temporal_math_succeeded = false;
                        if is_temporal_math {
                            // Ask LLM to extract dates from evidence as structured JSON
                            let date_extract_prompt = format!(
                                "Question: {}\n\n\
                                 Evidence:\n{}\n\n\
                                 Extract the specific dates mentioned in the evidence that are relevant to this question.\n\
                                 Return a JSON object with:\n\
                                 - \"events\": array of {{\"description\": \"...\", \"date\": \"YYYY-MM-DD\", \"evidence_idx\": N}}\n\
                                 - \"reference_date\": the date from which to calculate (if question says 'ago', use the latest conversation date; otherwise null)\n\n\
                                 RULES:\n\
                                 - Extract ONLY dates explicitly stated in evidence (e.g., 'on March 15, 2023' → '2023-03-15')\n\
                                 - If a day of week is mentioned with a session date context, resolve to the exact date\n\
                                 - If evidence says 'last Tuesday' in a session from May 10, compute the actual date\n\
                                 - Return ONLY valid JSON",
                                query, evidence
                            );
                            let date_system = "You extract dates from evidence. Return only valid JSON with exact dates in YYYY-MM-DD format.";

                            if let Ok(date_response) = rt.block_on(
                                synth_provider
                                    .call_text_with_retry(&date_extract_prompt, date_system),
                            ) {
                                let cleaned = date_response
                                    .trim()
                                    .trim_start_matches("```json")
                                    .trim_end_matches("```")
                                    .trim();
                                if let Ok(date_json) =
                                    serde_json::from_str::<serde_json::Value>(cleaned)
                                    && let Some(events) =
                                        date_json.get("events").and_then(|v| v.as_array())
                                {
                                        // Parse dates and compute differences
                                        let mut parsed_dates: Vec<(String, i64)> = Vec::new(); // (description, days_since_epoch)
                                        for event in events {
                                            if let (Some(desc), Some(date_str)) = (
                                                event.get("description").and_then(|v| v.as_str()),
                                                event.get("date").and_then(|v| v.as_str()),
                                            ) {
                                                // Parse YYYY-MM-DD to days since epoch
                                                let parts: Vec<&str> =
                                                    date_str.split('-').collect();
                                                if parts.len() == 3
                                                    && let (Ok(y), Ok(m), Ok(d)) = (
                                                        parts[0].parse::<i64>(),
                                                        parts[1].parse::<i64>(),
                                                        parts[2].parse::<i64>(),
                                                    )
                                                {
                                                        // Days since epoch (approximate but accurate enough for differences)
                                                        let days = (y - 1970) * 365
                                                            + (y - 1969) / 4
                                                            - (y - 1901) / 100
                                                            + (y - 1601) / 400
                                                            + (367 * m - 362) / 12
                                                            + d
                                                            - 1
                                                            + if m > 2 {
                                                                if y % 4 == 0
                                                                    && (y % 100 != 0
                                                                        || y % 400 == 0)
                                                                {
                                                                    -1
                                                                } else {
                                                                    -2
                                                                }
                                                            } else {
                                                                0
                                                            };
                                                        parsed_dates.push((desc.to_string(), days));
                                                    }
                                                }
                                        }
                                        if parsed_dates.len() >= 2 {
                                            // Compute difference between first two events
                                            let diff_days = (parsed_dates[1].1 - parsed_dates[0].1)
                                                .unsigned_abs();
                                            let diff_weeks = diff_days / 7;
                                            let diff_months = diff_days / 30; // approximate

                                            let computed_answer = if query_lower
                                                .contains("how many days")
                                            {
                                                format!(
                                                    "[VERIFIED COMPUTATION]\n\
                                                     Event 1: {}\n\
                                                     Event 2: {}\n\
                                                     Date arithmetic: {} days between the two events.\n\
                                                     ANSWER: {} days.",
                                                    parsed_dates[0].0,
                                                    parsed_dates[1].0,
                                                    diff_days,
                                                    diff_days
                                                )
                                            } else if query_lower.contains("how many weeks") {
                                                format!(
                                                    "[VERIFIED COMPUTATION]\n\
                                                     Event 1: {}\n\
                                                     Event 2: {}\n\
                                                     Date arithmetic: {} days = {} weeks between the two events.\n\
                                                     ANSWER: {} weeks.",
                                                    parsed_dates[0].0,
                                                    parsed_dates[1].0,
                                                    diff_days,
                                                    diff_weeks,
                                                    diff_weeks
                                                )
                                            } else if query_lower.contains("how many months") || query_lower.contains("how long") {
                                                format!(
                                                    "[VERIFIED COMPUTATION]\n\
                                                     Event 1: {}\n\
                                                     Event 2: {}\n\
                                                     Date arithmetic: {} days = approximately {} months between the two events.\n\
                                                     ANSWER: {} months.",
                                                    parsed_dates[0].0,
                                                    parsed_dates[1].0,
                                                    diff_days,
                                                    diff_months,
                                                    diff_months
                                                )
                                            } else {
                                                format!(
                                                    "[VERIFIED COMPUTATION]\n\
                                                     Event 1: {}\n\
                                                     Event 2: {}\n\
                                                     Date arithmetic: {} days ({} weeks, ~{} months) between the two events.\n\
                                                     ANSWER: {} days.",
                                                    parsed_dates[0].0,
                                                    parsed_dates[1].0,
                                                    diff_days,
                                                    diff_weeks,
                                                    diff_months,
                                                    diff_days
                                                )
                                            };
                                            if debug {
                                                eprintln!(
                                                    "[temporal-compute] Computed date difference: {} days between '{}' and '{}'",
                                                    diff_days, parsed_dates[0].0, parsed_dates[1].0
                                                );
                                            }
                                            final_synthesis = computed_answer;
                                            temporal_math_succeeded = true;
                                        } else if parsed_dates.len() == 1
                                            && query_lower.contains("ago")
                                        {
                                            // "How many weeks ago did X happen?"
                                            // Compute from reference date
                                            let event_days = parsed_dates[0].1;
                                            let local_day_us: u64 = 86_400_000_000;
                                            let local_before_us = temporal_reference.unwrap_or(
                                                std::time::SystemTime::now()
                                                    .duration_since(std::time::UNIX_EPOCH)
                                                    .unwrap_or_default()
                                                    .as_micros()
                                                    as u64,
                                            );
                                            let ref_days = (local_before_us / local_day_us) as i64;
                                            let diff_days = (ref_days - event_days).unsigned_abs();
                                            let diff_weeks = diff_days / 7;

                                            let diff_months = diff_days / 30;
                                            let computed_answer = if query_lower
                                                .contains("how many weeks")
                                            {
                                                format!(
                                                    "[VERIFIED COMPUTATION]\n\
                                                     Event: {}\n\
                                                     Date arithmetic: {} days = {} weeks ago from reference date.\n\
                                                     ANSWER: {} weeks ago.",
                                                    parsed_dates[0].0,
                                                    diff_days,
                                                    diff_weeks,
                                                    diff_weeks
                                                )
                                            } else if query_lower.contains("how many months") {
                                                format!(
                                                    "[VERIFIED COMPUTATION]\n\
                                                     Event: {}\n\
                                                     Date arithmetic: {} days = approximately {} months ago from reference date.\n\
                                                     ANSWER: {} months ago.",
                                                    parsed_dates[0].0,
                                                    diff_days,
                                                    diff_months,
                                                    diff_months
                                                )
                                            } else if query_lower.contains("how many days") {
                                                format!(
                                                    "[VERIFIED COMPUTATION]\n\
                                                     Event: {}\n\
                                                     Date arithmetic: {} days ago from reference date.\n\
                                                     ANSWER: {} days ago.",
                                                    parsed_dates[0].0,
                                                    diff_days,
                                                    diff_days
                                                )
                                            } else {
                                                format!(
                                                    "[VERIFIED COMPUTATION]\n\
                                                     Event: {}\n\
                                                     Date arithmetic: {} days (~{} months, ~{} weeks) ago from reference date.\n\
                                                     ANSWER: {} days ago.",
                                                    parsed_dates[0].0,
                                                    diff_days,
                                                    diff_months,
                                                    diff_weeks,
                                                    diff_days
                                                )
                                            };
                                            if debug {
                                                eprintln!(
                                                    "[temporal-compute] Event '{}' was {} days ago",
                                                    parsed_dates[0].0, diff_days
                                                );
                                            }
                                            final_synthesis = computed_answer;
                                            temporal_math_succeeded = true;
                                        }
                                    }
                                }
                            }

                        // --- Temporal ordering engine ---
                        // For "what is the order of X from earliest to latest" questions,
                        // extract events with dates and sort them chronologically in code.
                        // This bypasses LLM ordering errors.
                        if is_temporal_ordering && !temporal_math_succeeded {
                            let ordering_prompt = format!(
                                "Question: {}\n\n\
                                 Evidence:\n{}\n\n\
                                 Extract ALL events/items mentioned in the evidence that are relevant to this ordering question.\n\
                                 Return a JSON object with:\n\
                                 - \"events\": array of {{\"description\": \"short description of event/item\", \"date\": \"YYYY-MM-DD\"}}\n\n\
                                 RULES:\n\
                                 - Include EVERY relevant event mentioned in evidence\n\
                                 - Use the session date or explicit date from evidence for each event\n\
                                 - If exact date unknown but session context gives it, use that\n\
                                 - Return ONLY valid JSON",
                                query, evidence
                            );
                            let ordering_system = "You extract events with dates from evidence for chronological ordering. Return only valid JSON.";

                            if let Ok(ordering_response) = rt.block_on(
                                synth_provider.call_text_with_retry(&ordering_prompt, ordering_system),
                            ) {
                                let cleaned = ordering_response
                                    .trim()
                                    .trim_start_matches("```json")
                                    .trim_end_matches("```")
                                    .trim();
                                if let Ok(ordering_json) =
                                    serde_json::from_str::<serde_json::Value>(cleaned)
                                    && let Some(events) =
                                        ordering_json.get("events").and_then(|v| v.as_array())
                                {
                                    let mut dated_events: Vec<(String, i64)> = Vec::new();
                                    for event in events {
                                        if let (Some(desc), Some(date_str)) = (
                                            event.get("description").and_then(|v| v.as_str()),
                                            event.get("date").and_then(|v| v.as_str()),
                                        ) {
                                            let parts: Vec<&str> = date_str.split('-').collect();
                                            if parts.len() == 3
                                                && let (Ok(y), Ok(m), Ok(d)) = (
                                                    parts[0].parse::<i64>(),
                                                    parts[1].parse::<i64>(),
                                                    parts[2].parse::<i64>(),
                                                )
                                            {
                                                let days = (y - 1970) * 365
                                                    + (y - 1969) / 4
                                                    - (y - 1901) / 100
                                                    + (y - 1601) / 400
                                                    + (367 * m - 362) / 12
                                                    + d
                                                    - 1
                                                    + if m > 2 {
                                                        if y % 4 == 0
                                                            && (y % 100 != 0 || y % 400 == 0)
                                                        {
                                                            -1
                                                        } else {
                                                            -2
                                                        }
                                                    } else {
                                                        0
                                                    };
                                                dated_events.push((desc.to_string(), days));
                                            }
                                        }
                                    }

                                    if dated_events.len() >= 2 {
                                        // Sort by date (earliest first)
                                        dated_events.sort_by_key(|(_, d)| *d);

                                        let mut ordered_list = String::new();
                                        for (i, (desc, _)) in dated_events.iter().enumerate() {
                                            ordered_list.push_str(&format!("{}. {}\n", i + 1, desc));
                                        }

                                        let computed_answer = format!(
                                            "[VERIFIED COMPUTATION]\n\
                                             Chronological ordering (earliest to latest):\n\
                                             {}\n\
                                             ANSWER: The order from earliest to latest is: {}",
                                            ordered_list,
                                            dated_events.iter().map(|(d, _)| d.as_str()).collect::<Vec<_>>().join(", then ")
                                        );

                                        if debug {
                                            eprintln!(
                                                "[temporal-ordering] Sorted {} events chronologically",
                                                dated_events.len()
                                            );
                                        }
                                        final_synthesis = computed_answer;
                                        temporal_math_succeeded = true;
                                    }
                                }
                            }
                        }

                        // --- Chain-of-enumeration for SUM queries only ---
                        // For sum/aggregation queries (money, hours, days): LLM enumerates items,
                        // code parses amounts and sums them. This is deterministic and reliable.
                        // We do NOT run enumeration for pure COUNT queries — the LLM enumeration
                        // is unreliable and can override correct synthesis with wrong counts.
                        // Skip if temporal math already computed the answer (e.g., "how many days between X and Y")
                        let is_sum_query = query_lower.contains("how much")
                            && (query_lower.contains("money")
                                || query_lower.contains("spend")
                                || query_lower.contains("spent")
                                || query_lower.contains("cost")
                                || query_lower.contains("paid")
                                || query_lower.contains("save")
                                || query_lower.contains("earn")
                                || query_lower.contains("raised"))
                            || (query_lower.contains("total")
                                && (query_lower.contains("money")
                                    || query_lower.contains("spend")
                                    || query_lower.contains("spent")
                                    || query_lower.contains("cost")
                                    || query_lower.contains("hours")
                                    || query_lower.contains("days")
                                    || query_lower.contains("miles")
                                    || query_lower.contains("raised")))
                            || (query_lower.contains("how many")
                                && (query_lower.contains("spend") || query_lower.contains("spent"))
                                && (query_lower.contains("hours")
                                    || query_lower.contains("days")
                                    || query_lower.contains("miles")));

                        if is_sum_query && !temporal_math_succeeded {
                            let enum_prompt = format!(
                                "Question: {}\n\n\
                                 Evidence from memory:\n{}\n\n\
                                 List EVERY item/occurrence relevant to this question as a JSON array.\n\
                                 IMPORTANT RULES:\n\
                                 - ONLY include items explicitly mentioned in the evidence — never infer or invent items\n\
                                 - Quote amounts/numbers EXACTLY as stated in evidence\n\
                                 - If the question asks about a specific time period, only include items from that period\n\
                                 - If the question asks about frequency (per week/month), list each OCCURRENCE separately\n\
                                 - \"Plan to\" or \"looking to\" acquiring X does NOT mean they have X — exclude unless question asks about plans\n\
                                 - DEDUPLICATE: if the same event/item appears in multiple evidence entries, count it ONLY ONCE\n\
                                 - Each unique session may mention the same item — that is ONE item, not multiple\n\
                                 - READ ALL EVIDENCE carefully — items may appear in later entries that you missed\n\
                                 For each item, include:\n\
                                 - \"name\": the item name (be specific)\n\
                                 - \"qualifies\": true if it should be counted, false if not\n\
                                 - \"reason\": cite which evidence entry [N] supports this item\n\
                                 - \"session\": which session/date this came from (if identifiable)\n\
                                 - \"amount\": if the question involves money, quote the EXACT amount from evidence\n\n\
                                 Return ONLY valid JSON. Example:\n\
                                 [{{\"name\": \"Zumba - Tuesday\", \"qualifies\": true, \"reason\": \"Evidence [3] states user attends Tuesdays\", \"session\": \"Session 2\"}}]",
                                query, evidence
                            );
                            let enum_system = "You enumerate items from evidence as JSON. ONLY include items with explicit evidence support. Quote all numbers exactly. DEDUPLICATE: same item from multiple evidence entries = ONE item. Return ONLY a JSON array.";

                            match rt.block_on(
                                synth_provider.call_text_with_retry(&enum_prompt, enum_system),
                            ) {
                                Ok(enum_response) => {
                                    let trimmed = enum_response.trim();
                                    let array_str = if let Some(start) = trimmed.find('[') {
                                        if let Some(end) = trimmed.rfind(']') {
                                            &trimmed[start..=end]
                                        } else {
                                            trimmed
                                        }
                                    } else {
                                        trimmed
                                    };

                                    match serde_json::from_str::<Vec<serde_json::Value>>(array_str)
                                    {
                                        Ok(items) => {
                                            let qualifying: Vec<&serde_json::Value> = items
                                                .iter()
                                                .filter(|item| {
                                                    item.get("qualifies")
                                                        .and_then(|v| v.as_bool())
                                                        .unwrap_or(false)
                                                })
                                                .collect();

                                            // Dedup qualifying items by normalized name
                                            // Removes duplicates where the same item appears with slightly different descriptions
                                            let mut deduped: Vec<&serde_json::Value> = Vec::new();
                                            let mut seen_names: Vec<String> = Vec::new();
                                            for item in &qualifying {
                                                let name = item
                                                    .get("name")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("")
                                                    .to_lowercase();
                                                // Normalize: remove common prefixes, articles, extra whitespace
                                                let normalized = name
                                                    .replace("the ", "")
                                                    .replace("a ", "")
                                                    .replace("an ", "")
                                                    .replace("my ", "")
                                                    .trim()
                                                    .to_string();
                                                // Check for substring matches or high overlap
                                                let is_dup = seen_names.iter().any(|existing| {
                                                    existing.contains(&normalized)
                                                        || normalized.contains(existing.as_str())
                                                        || {
                                                            // Jaccard similarity on words
                                                            let a_words: std::collections::HashSet<
                                                                &str,
                                                            > = normalized
                                                                .split_whitespace()
                                                                .collect();
                                                            let b_words: std::collections::HashSet<
                                                                &str,
                                                            > = existing
                                                                .split_whitespace()
                                                                .collect();
                                                            let intersection = a_words
                                                                .intersection(&b_words)
                                                                .count();
                                                            let union =
                                                                a_words.union(&b_words).count();
                                                            union > 0
                                                                && (intersection as f32
                                                                    / union as f32)
                                                                    > 0.6
                                                        }
                                                });
                                                if !is_dup && !normalized.is_empty() {
                                                    seen_names.push(normalized);
                                                    deduped.push(item);
                                                } else if debug && is_dup {
                                                    eprintln!(
                                                        "[chain-enum-dedup] Removed duplicate: '{}'",
                                                        name
                                                    );
                                                }
                                            }
                                            let qualifying = deduped;
                                            let enum_count = qualifying.len();

                                            if debug {
                                                eprintln!(
                                                    "[chain-enum] Enumerated {} items, {} qualifying:",
                                                    items.len(),
                                                    enum_count
                                                );
                                                for item in &items {
                                                    let name = item
                                                        .get("name")
                                                        .and_then(|v| v.as_str())
                                                        .unwrap_or("?");
                                                    let q = item
                                                        .get("qualifies")
                                                        .and_then(|v| v.as_bool())
                                                        .unwrap_or(false);
                                                    let reason = item
                                                        .get("reason")
                                                        .and_then(|v| v.as_str())
                                                        .unwrap_or("");
                                                    eprintln!(
                                                        "[chain-enum]   {} — qualifies={} ({})",
                                                        name, q, reason
                                                    );
                                                }
                                            }

                                            // If enumeration found 0 items AND no items at all were extracted,
                                            // this is likely an abstention case (topic never discussed)
                                            if enum_count == 0 && items.is_empty() {
                                                if debug {
                                                    eprintln!(
                                                        "[chain-enum] Zero items found — abstaining instead of reporting 0"
                                                    );
                                                }
                                                let query_topic = query
                                                    .to_lowercase()
                                                    .replace("how many", "")
                                                    .replace("how much", "")
                                                    .replace("do i", "")
                                                    .replace("did i", "")
                                                    .replace("have i", "")
                                                    .replace("?", "")
                                                    .trim()
                                                    .to_string();
                                                final_synthesis = format!(
                                                    "I don't have information about {} in our conversations.",
                                                    if query_topic.is_empty() {
                                                        "that topic".to_string()
                                                    } else {
                                                        query_topic
                                                    }
                                                );
                                            } else {
                                                // Since we only run enumeration for sum queries,
                                                // always try to compute the sum
                                                let mut total = 0.0_f64;
                                                let mut found_any = false;
                                                for item in &qualifying {
                                                    if let Some(amt) =
                                                        item.get("amount").and_then(|v| v.as_str())
                                                    {
                                                        let cleaned: String = amt
                                                            .chars()
                                                            .filter(|c| {
                                                                c.is_ascii_digit()
                                                                    || *c == '.'
                                                                    || *c == '-'
                                                            })
                                                            .collect();
                                                        if let Ok(val) = cleaned.parse::<f64>()
                                                            && val > 0.0
                                                        {
                                                            total += val;
                                                            found_any = true;
                                                        }
                                                    } else if let Some(amt) =
                                                        item.get("amount").and_then(|v| v.as_f64())
                                                    {
                                                        total += amt;
                                                        found_any = true;
                                                    }
                                                }

                                                // Build item list with amounts
                                                let item_list = qualifying
                                                    .iter()
                                                    .enumerate()
                                                    .map(|(i, item)| {
                                                        let name = item
                                                            .get("name")
                                                            .and_then(|v| v.as_str())
                                                            .unwrap_or("unknown");
                                                        let reason = item
                                                            .get("reason")
                                                            .and_then(|v| v.as_str())
                                                            .unwrap_or("");
                                                        let amt = item
                                                            .get("amount")
                                                            .and_then(|v| v.as_str())
                                                            .unwrap_or("unknown amount");
                                                        format!(
                                                            "{}. {} — {} ({})",
                                                            i + 1,
                                                            name,
                                                            amt,
                                                            reason
                                                        )
                                                    })
                                                    .collect::<Vec<_>>()
                                                    .join("\n");

                                                if found_any {
                                                    let is_money =
                                                        query_lower.contains("money")
                                                            || query_lower.contains("spend")
                                                            || query_lower.contains("spent")
                                                            || query_lower.contains("cost")
                                                            || query_lower.contains("paid")
                                                            || query_lower.contains("save")
                                                            || query_lower.contains("earn")
                                                            || query_lower.contains("raised");
                                                    let formatted_total = if is_money {
                                                        if total == total.floor() {
                                                            format!("${}", total as i64)
                                                        } else {
                                                            format!("${:.2}", total)
                                                        }
                                                    } else if query_lower.contains("hours") {
                                                        format!("{} hours", total)
                                                    } else if query_lower.contains("days") {
                                                        format!("{} days", total)
                                                    } else if query_lower.contains("miles") {
                                                        format!("{} miles", total)
                                                    } else {
                                                        format!("{}", total)
                                                    };
                                                    if debug {
                                                        eprintln!(
                                                            "[chain-enum] SUM query: {} items, total = {}",
                                                            enum_count, formatted_total
                                                        );
                                                    }
                                                    final_synthesis = format!(
                                                        "[VERIFIED COMPUTATION]\nBased on the evidence, the total is {}.\n\n{}\n\nTotal: {}",
                                                        formatted_total, item_list, formatted_total
                                                    );
                                                }
                                                // If no amounts found, keep original synthesis

                                                if debug && let Some(gc) = graph_count {
                                                    if gc != enum_count {
                                                        eprintln!(
                                                            "[dual-path] DISAGREEMENT: graph={} enum={}",
                                                            gc, enum_count
                                                        );
                                                    } else {
                                                        eprintln!(
                                                            "[dual-path] AGREEMENT: graph={} enum={}",
                                                            gc, enum_count
                                                        );
                                                    }
                                                }
                                            } // close the else branch of enum_count == 0 check
                                        }
                                        Err(e) => {
                                            if debug {
                                                eprintln!("[chain-enum] Failed to parse JSON: {e}");
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    if debug {
                                        eprintln!("[chain-enum] LLM call failed: {e}");
                                    }
                                }
                            }
                        }

                        if debug {
                            eprintln!(
                                "[synthesis] Generated: {}",
                                &final_synthesis[..final_synthesis.floor_char_boundary(
                                    std::cmp::min(final_synthesis.len(), 200)
                                )]
                            );
                        }

                        let synth_emb = if let Some(ref embedder) = self.embedder {
                            embedder.embed(&final_synthesis).map_err(to_pyerr)?
                        } else {
                            hash_embedding(&final_synthesis, 384)
                        };
                        let mut synth_node = MemoryNode::new(
                            mentedb_core::types::AgentId::new(),
                            MemoryType::Semantic,
                            final_synthesis,
                            synth_emb,
                        );
                        synth_node.tags =
                            vec!["synthesis:true".to_string(), "ephemeral:true".to_string()];
                        let synth_id = synth_node.id;
                        db.store(synth_node).map_err(to_pyerr)?;
                        current_synth_id = Some(synth_id.to_string());

                        // Prepend synthesis as first result — original order preserved after it
                        let top_score = expanded.first().map(|(_, s)| *s).unwrap_or(1.0);
                        expanded.insert(0, (synth_id.to_string(), top_score * 1.5));
                    }
                    Err(e) => {
                        if debug {
                            eprintln!("[synthesis] LLM call failed: {e}");
                        }
                    }
                }
            }
        }

        // Filter entity nodes and old synthesis results from final results.
        // Keep community summaries — they're unique aggregated indexes that bridge
        // semantic gaps (e.g., "Health Device Summary" links hearing aids to health).
        // Keep the current synthesis (just created above) but filter out old ones
        // that would pollute evidence with potentially wrong previous answers.
        let mut filtered_results: Vec<(String, f32)> = Vec::new();
        for (id_str, score) in expanded {
            if let Ok(mem_id) = parse_memory_id(&id_str)
                && let Ok(node) = db.get_memory(mem_id)
            {
                let is_entity = node.tags.iter().any(|t| t.starts_with("entity_name:"));
                let is_old_synthesis = node.tags.iter().any(|t| t == "synthesis:true")
                    && Some(id_str.as_str()) != current_synth_id.as_deref();
                if is_entity || is_old_synthesis {
                    continue;
                }
            }
            filtered_results.push((id_str, score));
        }

        Ok(filtered_results
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect())
    }

    /// Add a typed, weighted edge between two memories.
    fn relate(
        &self,
        source: &str,
        target: &str,
        edge_type: &str,
        weight: f32,
        valid_from: Option<u64>,
        valid_until: Option<u64>,
    ) -> PyResult<()> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

        let edge = MemoryEdge {
            source: parse_memory_id(source)?,
            target: parse_memory_id(target)?,
            edge_type: parse_edge_type(edge_type)?,
            weight,
            created_at: now_us(),
            valid_from,
            valid_until,
            label: None,
        };

        db.relate(edge).map_err(to_pyerr)
    }

    /// Remove a memory by its UUID string.
    fn forget(&self, memory_id: &str) -> PyResult<()> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

        let id = parse_memory_id(memory_id)?;
        db.forget(id).map_err(to_pyerr)
    }

    /// Retrieve a memory by its UUID string.
    /// Returns a dict with id, content, memory_type, tags, created_at.
    fn get_memory(&self, memory_id: &str) -> PyResult<PyObject> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

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
        &self,
        conversation: &str,
        provider: Option<&str>,
        agent_id: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

        let config = build_extraction_config_from_env(provider)?;

        let http_provider = HttpExtractionProvider::new(config.clone()).map_err(to_pyerr)?;
        let pipeline = ExtractionPipeline::new(http_provider, config.clone());

        let rt = tokio::runtime::Runtime::new().map_err(to_pyerr)?;
        let extraction_result = rt
            .block_on(pipeline.extract_full(conversation))
            .map_err(to_pyerr)?;
        let all_memories = extraction_result.memories;
        let extracted_entities = extraction_result.entities;

        let total = all_memories.len();
        let quality_passed = pipeline.filter_quality(&all_memories);
        let rejected_low_quality = total - quality_passed.len();

        let aid = match agent_id {
            Some(s) => parse_agent_id(s)?,
            None => AgentId::new(),
        };

        // Collect all texts to embed (extracted memories + user turns)
        let mut embed_texts: Vec<String> = Vec::new();
        struct MemEntry {
            content: String,
            mt: MemoryType,
            tags: Vec<String>,
            salience: f32,
            confidence: f32,
        }
        let mut entries: Vec<MemEntry> = Vec::new();

        // Build entity name → categories map for semantic enrichment of fact embeddings
        let mut entity_categories: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for entity in &extracted_entities {
            if let Some(cat) = entity.attributes.get("category") {
                let cats: Vec<String> = cat
                    .split(',')
                    .map(|c| c.trim().to_lowercase().replace('_', " "))
                    .filter(|c| !c.is_empty())
                    .collect();
                if !cats.is_empty() {
                    entity_categories.insert(entity.name.to_lowercase(), cats);
                }
            }
        }

        for memory in &quality_passed {
            let mt = map_extraction_type_to_memory_type(&memory.memory_type);
            let mut embed_key = memory.embedding_key();

            // Enrich fact embeddings with categories from mentioned entities
            // "User has Phonak hearing aids" + entity has category "health_device"
            // → embed key becomes "User has Phonak hearing aids [context: health device]"
            if !entity_categories.is_empty() {
                let mut injected_cats: Vec<String> = Vec::new();
                for entity_name in &memory.entities {
                    if let Some(cats) = entity_categories.get(&entity_name.to_lowercase()) {
                        injected_cats.extend(cats.iter().cloned());
                    }
                }
                if !injected_cats.is_empty() {
                    injected_cats.sort();
                    injected_cats.dedup();
                    embed_key.push_str(&format!(" [context: {}]", injected_cats.join(", ")));
                }
            }

            embed_texts.push(embed_key);
            // Merge context categories as context: tags for categorical retrieval
            let mut tags = memory.tags.clone();
            for ctx in &memory.context {
                let ctx_tag = format!("context:{}", ctx.to_lowercase().replace(' ', "_"));
                if !tags.contains(&ctx_tag) {
                    tags.push(ctx_tag);
                }
            }
            entries.push(MemEntry {
                content: memory.content.clone(),
                mt,
                tags,
                salience: memory.confidence,
                confidence: memory.confidence,
            });
        }

        for line in conversation.lines() {
            let trimmed = line.trim();
            let (content, role_tag) = if let Some(rest) = trimmed
                .strip_prefix("User:")
                .or_else(|| trimmed.strip_prefix("user:"))
            {
                (rest.trim(), "turn:user")
            } else if let Some(rest) = trimmed
                .strip_prefix("Assistant:")
                .or_else(|| trimmed.strip_prefix("assistant:"))
            {
                (rest.trim(), "turn:assistant")
            } else {
                continue;
            };
            if content.len() > 15 {
                embed_texts.push(content.to_string());
                entries.push(MemEntry {
                    content: content.to_string(),
                    mt: MemoryType::Episodic,
                    tags: vec!["turn".to_string(), role_tag.to_string()],
                    salience: 0.75,
                    confidence: 0.8,
                });
            }
        }

        // Batch embed in chunks (OpenAI has payload limits)
        let embed_refs: Vec<&str> = embed_texts.iter().map(|s| s.as_str()).collect();
        let embeddings = if let Some(ref embedder) = self.embedder {
            let mut all_embs = Vec::with_capacity(embed_refs.len());
            for chunk in embed_refs.chunks(100) {
                let mut batch = embedder.embed_batch(chunk).map_err(to_pyerr)?;
                all_embs.append(&mut batch);
            }
            all_embs
        } else {
            embed_refs.iter().map(|k| hash_embedding(k, 384)).collect()
        };

        // Store all memories with pre-computed embeddings
        let mut stored_ids = Vec::with_capacity(entries.len());
        for (entry, emb) in entries.into_iter().zip(embeddings.into_iter()) {
            let mut node = MemoryNode::new(aid, entry.mt, entry.content, emb);
            node.tags = entry.tags;
            node.salience = entry.salience;
            node.confidence = entry.confidence;
            let id = node.id;
            db.store(node).map_err(to_pyerr)?;
            stored_ids.push(id.to_string());
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

    /// Extract memories from a conversation without storing them.
    /// Returns a list of dicts with content, memory_type, tags, confidence.
    /// Use with store_extracted() for parallel extraction workflows.
    #[pyo3(signature = (conversation, provider=None))]
    fn extract(
        &self,
        py: Python<'_>,
        conversation: String,
        provider: Option<String>,
    ) -> PyResult<PyObject> {
        let config = build_extraction_config_from_env(provider.as_deref())?;
        let http_provider = HttpExtractionProvider::new(config.clone()).map_err(to_pyerr)?;
        let pipeline = ExtractionPipeline::new(http_provider, config.clone());

        // Release the GIL during the HTTP call so other threads can run
        let extraction_result = py.allow_threads(|| {
            let rt = tokio::runtime::Runtime::new().map_err(to_pyerr)?;
            let result = rt
                .block_on(pipeline.extract_full(&conversation))
                .map_err(to_pyerr)?;
            Ok::<_, PyErr>(result)
        })?;

        let quality_passed = pipeline.filter_quality(&extraction_result.memories);

        let results = pyo3::types::PyList::empty(py);
        for memory in &quality_passed {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("content", &memory.content)?;
            dict.set_item("memory_type", &memory.memory_type)?;
            dict.set_item("tags", &memory.tags)?;
            dict.set_item("context", &memory.context)?;
            dict.set_item("confidence", memory.confidence)?;
            dict.set_item("embedding_key", memory.embedding_key())?;
            results.append(dict)?;
        }

        // Include extracted entities as special dicts with _entity marker
        for entity in &extraction_result.entities {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("content", entity.to_content())?;
            dict.set_item("memory_type", "entity")?;
            dict.set_item(
                "tags",
                vec![
                    format!("entity_type:{}", entity.entity_type),
                    format!("entity_name:{}", entity.name),
                ],
            )?;
            dict.set_item("confidence", 0.85)?;
            dict.set_item("embedding_key", entity.embedding_key())?;
            // Store entity attributes as a nested dict
            let attrs = pyo3::types::PyDict::new(py);
            for (k, v) in &entity.attributes {
                attrs.set_item(k, v)?;
            }
            dict.set_item("entity_attributes", attrs)?;
            dict.set_item("entity_name", &entity.name)?;
            dict.set_item("entity_type", &entity.entity_type)?;
            results.append(dict)?;
        }
        // Also include user AND assistant turns for episodic storage
        for line in conversation.lines() {
            let trimmed = line.trim();
            let (content, role_tag) = if let Some(rest) = trimmed
                .strip_prefix("User:")
                .or_else(|| trimmed.strip_prefix("user:"))
            {
                (rest.trim(), "turn:user")
            } else if let Some(rest) = trimmed
                .strip_prefix("Assistant:")
                .or_else(|| trimmed.strip_prefix("assistant:"))
            {
                (rest.trim(), "turn:assistant")
            } else {
                continue;
            };
            if content.len() > 15 {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("content", content)?;
                dict.set_item("memory_type", "episodic")?;
                let tags: Vec<String> = vec!["turn".to_string(), role_tag.to_string()];
                dict.set_item("tags", tags)?;
                dict.set_item("confidence", 0.8)?;
                dict.set_item("embedding_key", content)?;
                results.append(dict)?;
            }
        }
        Ok(results.into())
    }

    /// Store pre-extracted memories (from extract()) into the database.
    /// Uses batch embedding (1 API call for all memories) for speed.
    #[pyo3(signature = (memories, agent_id=None))]
    fn store_extracted(
        &self,
        memories: Vec<Py<pyo3::types::PyDict>>,
        agent_id: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let debug = std::env::var("MENTEDB_DEBUG").is_ok();

        let aid = match agent_id {
            Some(s) => parse_agent_id(s)?,
            None => AgentId::new(),
        };

        // Phase 1: Parse all dicts and collect embed keys
        struct ParsedMemory {
            content: String,
            memory_type: String,
            tags: Vec<String>,
            confidence: f32,
            embed_key: String,
            // Optional timestamp override (microseconds since epoch)
            created_at: Option<u64>,
            // Entity-specific fields (None for regular memories)
            entity_name: Option<String>,
            entity_type: Option<String>,
            entity_attributes: Option<std::collections::HashMap<String, String>>,
        }
        let mut parsed = Vec::with_capacity(memories.len());

        Python::with_gil(|py| {
            for mem_ref in &memories {
                let mem_dict = mem_ref.bind(py);
                let content: String = mem_dict
                    .get_item("content")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();
                let memory_type: String = mem_dict
                    .get_item("memory_type")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_else(|| "semantic".to_string());
                let mut tags: Vec<String> = mem_dict
                    .get_item("tags")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();
                // Parse context categories and merge as context: tags
                let context: Vec<String> = mem_dict
                    .get_item("context")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();
                for ctx in &context {
                    let ctx_tag = format!("context:{}", ctx.to_lowercase().replace(' ', "_"));
                    if !tags.contains(&ctx_tag) {
                        tags.push(ctx_tag);
                    }
                }
                let confidence: f32 = mem_dict
                    .get_item("confidence")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.5);
                let embed_key: String = mem_dict
                    .get_item("embedding_key")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_else(|| content.clone());

                // Parse entity-specific fields if present
                let entity_name: Option<String> = mem_dict
                    .get_item("entity_name")?
                    .map(|v| v.extract())
                    .transpose()?;
                let entity_type: Option<String> = mem_dict
                    .get_item("entity_type")?
                    .map(|v| v.extract())
                    .transpose()?;
                let entity_attributes: Option<std::collections::HashMap<String, String>> = mem_dict
                    .get_item("entity_attributes")?
                    .map(|v| v.extract())
                    .transpose()?;

                // Optional timestamp override (microseconds since epoch)
                let created_at: Option<u64> = mem_dict
                    .get_item("created_at")?
                    .and_then(|v| v.extract().ok());

                parsed.push(ParsedMemory {
                    content,
                    memory_type,
                    tags,
                    confidence,
                    embed_key,
                    created_at,
                    entity_name,
                    entity_type,
                    entity_attributes,
                });
            }
            Ok::<_, PyErr>(())
        })?;

        if parsed.is_empty() {
            return Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("stored_ids", Vec::<String>::new())?;
                Ok(dict.into())
            });
        }

        // Phase 1.5: Enrich embed keys with entity categories for semantic discoverability.
        // Build entity name → categories map from entity entries in this batch.
        {
            let mut entity_cats: std::collections::HashMap<String, Vec<String>> =
                std::collections::HashMap::new();
            for mem in &parsed {
                if let (Some(name), Some(attrs)) = (&mem.entity_name, &mem.entity_attributes)
                    && let Some(cat) = attrs.get("category")
                {
                    let cats: Vec<String> = cat
                        .split(',')
                        .map(|c| c.trim().to_lowercase().replace('_', " "))
                        .filter(|c| !c.is_empty())
                        .collect();
                    if !cats.is_empty() {
                        entity_cats.insert(name.to_lowercase(), cats);
                    }
                }
            }
            // For entity entries, ensure their embed_key includes categories
            if !entity_cats.is_empty() {
                for mem in &mut parsed {
                    if let Some(ref name) = mem.entity_name
                        && let Some(cats) = entity_cats.get(&name.to_lowercase())
                        && !mem.embed_key.contains("[categories:")
                    {
                        mem.embed_key
                            .push_str(&format!(" [categories: {}]", cats.join(", ")));
                    }
                }
            }
        }

        // Phase 2: Batch embed in chunks (OpenAI has payload limits)
        let embed_start = std::time::Instant::now();
        let embed_keys: Vec<&str> = parsed.iter().map(|p| p.embed_key.as_str()).collect();
        let embeddings = if let Some(ref embedder) = self.embedder {
            let mut all_embs = Vec::with_capacity(embed_keys.len());
            for chunk in embed_keys.chunks(100) {
                let mut batch = embedder.embed_batch(chunk).map_err(to_pyerr)?;
                all_embs.append(&mut batch);
            }
            all_embs
        } else {
            embed_keys.iter().map(|k| hash_embedding(k, 384)).collect()
        };
        let embed_ms = embed_start.elapsed().as_millis();

        // Phase 3: Store all memories with pre-computed embeddings.
        // Entity memories get attributes populated and graph edges created.
        // Entity resolution: merge attributes into existing entity if name+type match.
        let mut stored_ids = Vec::with_capacity(parsed.len());
        let mut entity_ids: Vec<(String, String, MemoryId)> = Vec::new(); // (name, type, id)

        // Build an index of existing entity nodes for resolution (only if batch has entities)
        let has_entities = parsed.iter().any(|m| m.entity_name.is_some());
        let mut existing_entities: std::collections::HashMap<String, MemoryId> =
            std::collections::HashMap::new();
        if has_entities {
            for mid in db.memory_ids() {
                if let Ok(node) = db.get_memory(mid)
                    && node.tags.iter().any(|t| t.starts_with("entity_name:"))
                {
                    let key = node
                        .tags
                        .iter()
                        .filter(|t| t.starts_with("entity_name:") || t.starts_with("entity_type:"))
                        .map(|t| t.to_lowercase())
                        .collect::<Vec<_>>()
                        .join("|");
                    existing_entities.insert(key, mid);
                }
            }
        }

        // Separate batch-storable nodes from entity-resolution nodes
        struct PendingNode {
            node: MemoryNode,
            entity_name: Option<String>,
            entity_type: Option<String>,
        }
        let mut batch_nodes: Vec<PendingNode> = Vec::with_capacity(parsed.len());

        for (mem, emb) in parsed.into_iter().zip(embeddings.into_iter()) {
            let is_entity = mem.entity_name.is_some() && mem.entity_attributes.is_some();

            // Entity resolution: check if this entity already exists
            if is_entity {
                let entity_key = [
                    format!(
                        "entity_name:{}",
                        mem.entity_name.as_ref().unwrap().to_lowercase()
                    ),
                    format!(
                        "entity_type:{}",
                        mem.entity_type
                            .as_ref()
                            .unwrap_or(&"unknown".to_string())
                            .to_lowercase()
                    ),
                ]
                .join("|");

                if let Some(&existing_id) = existing_entities.get(&entity_key) {
                    // Merge attributes into existing entity
                    if let Ok(mut existing_node) = db.get_memory(existing_id) {
                        if let Some(ref attrs) = mem.entity_attributes {
                            for (k, v) in attrs {
                                existing_node.attributes.insert(
                                    k.clone(),
                                    mentedb_core::memory::AttributeValue::String(v.clone()),
                                );
                            }
                        }
                        let _ = db.store(existing_node);
                        stored_ids.push(existing_id.to_string());
                        entity_ids.push((
                            mem.entity_name.unwrap_or_default(),
                            mem.entity_type.unwrap_or_default(),
                            existing_id,
                        ));
                        continue;
                    }
                }
            }

            let mt = map_extraction_type_to_memory_type(&mem.memory_type);
            let mut node = MemoryNode::new(aid, mt, mem.content, emb);
            node.tags = mem.tags;
            node.salience = mem.confidence;
            node.confidence = mem.confidence;
            // Override timestamp if caller provided one (e.g., session date)
            if let Some(ts) = mem.created_at {
                node.created_at = ts;
                node.accessed_at = ts;
            } else {
                // Fallback: derive timestamp from date: tag if present
                for tag in &node.tags {
                    if let Some(date_str) = tag.strip_prefix("date:") {
                        // Parse "YYYY/MM/DD" or "YYYY/MM/DD (Day) HH:MM" format
                        let parts: Vec<&str> = date_str.split('/').collect();
                        if parts.len() >= 3
                            && let (Ok(y), Ok(m), Ok(d)) = (
                                parts[0].parse::<i64>(),
                                parts[1].parse::<i64>(),
                                parts[2]
                                    .split_whitespace()
                                    .next()
                                    .unwrap_or("1")
                                    .split('(')
                                    .next()
                                    .unwrap_or("1")
                                    .trim()
                                    .parse::<i64>(),
                            )
                        {
                            // Approximate: days since epoch * microseconds per day
                            let days = (y - 1970) * 365
                                + (y - 1969) / 4
                                + match m {
                                    1 => 0,
                                    2 => 31,
                                    3 => 59,
                                    4 => 90,
                                    5 => 120,
                                    6 => 151,
                                    7 => 181,
                                    8 => 212,
                                    9 => 243,
                                    10 => 273,
                                    11 => 304,
                                    12 => 334,
                                    _ => 0,
                                }
                                + d
                                - 1;
                            let ts = (days as u64) * 86_400_000_000;
                            node.created_at = ts;
                            node.accessed_at = ts;
                        }
                        break;
                    }
                }
            }

            // If this is an entity, populate its attributes
            if let Some(ref attrs) = mem.entity_attributes {
                for (k, v) in attrs {
                    node.attributes.insert(
                        k.clone(),
                        mentedb_core::memory::AttributeValue::String(v.clone()),
                    );
                }
                if let Some(ref name) = mem.entity_name {
                    node.attributes.insert(
                        "entity_name".to_string(),
                        mentedb_core::memory::AttributeValue::String(name.clone()),
                    );
                }
                if let Some(ref etype) = mem.entity_type {
                    node.attributes.insert(
                        "entity_type".to_string(),
                        mentedb_core::memory::AttributeValue::String(etype.clone()),
                    );
                }
            }

            let _id = node.id;
            batch_nodes.push(PendingNode {
                node,
                entity_name: mem.entity_name,
                entity_type: mem.entity_type,
            });
        }

        // Batch store all collected nodes in a single transaction
        let store_start = std::time::Instant::now();
        {
            let nodes_vec: Vec<MemoryNode> = batch_nodes.iter().map(|p| p.node.clone()).collect();
            db.store_batch(nodes_vec).map_err(to_pyerr)?;
        }
        let store_ms = store_start.elapsed().as_millis();

        if debug {
            eprintln!(
                "[store_extracted] embed={}ms store={}ms count={}",
                embed_ms,
                store_ms,
                batch_nodes.len()
            );
        }

        // Update tracking structures after batch store
        for pending in &batch_nodes {
            let id = pending.node.id;
            stored_ids.push(id.to_string());
            if let (Some(name), Some(etype)) =
                (pending.entity_name.as_ref(), pending.entity_type.as_ref())
            {
                existing_entities.insert(
                    [
                        format!("entity_name:{}", name.to_lowercase()),
                        format!("entity_type:{}", etype.to_lowercase()),
                    ]
                    .join("|"),
                    id,
                );
                entity_ids.push((name.clone(), etype.clone(), id));
            }
        }

        // Phase 4: Create PartOf edges linking regular memories to their entities.
        // For each entity, link all memories that mention it.
        // Also propagate entity categories as context tags to linked facts,
        // enabling direct tag-based retrieval (e.g., context:health_device).
        // Use the entity's "relationship" attribute as the edge label.

        // Pre-load entity categories for tag propagation
        let mut entity_categories: std::collections::HashMap<MemoryId, Vec<String>> =
            std::collections::HashMap::new();
        for (_name, _etype, entity_id) in &entity_ids {
            if let Ok(node) = db.get_memory(*entity_id)
                && let Some(mentedb_core::memory::AttributeValue::String(cat)) =
                    node.attributes.get("category")
            {
                let cats: Vec<String> = cat
                    .split(',')
                    .map(|s| s.trim().to_lowercase().replace(' ', "_"))
                    .filter(|s| !s.is_empty())
                    .collect();
                entity_categories.insert(*entity_id, cats);
            }
        }

        for (entity_name, _etype, entity_id) in &entity_ids {
            let entity_name_lower = entity_name.to_lowercase();
            // Build name variants for flexible matching:
            // "Phonak BTE hearing aids" → also try "hearing aids", individual significant words
            let name_words: Vec<&str> = entity_name_lower.split_whitespace().collect();
            let mut match_patterns: Vec<String> = vec![entity_name_lower.clone()];
            // Add last N words as a pattern (e.g., "hearing aids" from "Phonak BTE hearing aids")
            if name_words.len() > 2 {
                for start in 1..name_words.len().saturating_sub(1) {
                    match_patterns.push(name_words[start..].join(" "));
                }
            }

            let edge_label = db.get_memory(*entity_id).ok().and_then(|node| {
                node.attributes.get("relationship").and_then(|v| match v {
                    mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()),
                    _ => None,
                })
            });
            let cats = entity_categories
                .get(entity_id)
                .cloned()
                .unwrap_or_default();

            for sid in &stored_ids {
                let mem_id = parse_memory_id(sid)?;
                if mem_id == *entity_id {
                    continue;
                }
                if let Ok(mut mem_node) = db.get_memory(mem_id) {
                    // Skip other entity nodes
                    if mem_node.tags.iter().any(|t| t.starts_with("entity_name:")) {
                        continue;
                    }

                    let content_lower = mem_node.content.to_lowercase();
                    let mentions = match_patterns
                        .iter()
                        .any(|p| content_lower.contains(p.as_str()));
                    if mentions {
                        let edge = MemoryEdge {
                            source: mem_id,
                            target: *entity_id,
                            edge_type: EdgeType::PartOf,
                            weight: 0.8,
                            created_at: now_us(),
                            valid_from: None,
                            valid_until: None,
                            label: edge_label.clone(),
                        };
                        let _ = db.relate(edge);

                        // Propagate entity categories as context tags
                        let mut tags_changed = false;
                        for cat in &cats {
                            let ctx_tag = format!("context:{}", cat);
                            if !mem_node.tags.contains(&ctx_tag) {
                                mem_node.tags.push(ctx_tag);
                                tags_changed = true;
                            }
                        }
                        if tags_changed {
                            let _ = db.store(mem_node);
                        }
                    }
                }
            }
        }

        // Phase 5: Create entity-to-entity edges for entities sharing a category.
        // This creates the associative connections that make the graph mind-like.
        // Categories can be comma-separated (e.g., "health_device, wearable, fitness_tracker")
        // so we split and index each category individually.
        {
            // Build category → entity_id index (split comma-separated categories)
            let mut category_index: std::collections::HashMap<String, Vec<MemoryId>> =
                std::collections::HashMap::new();
            for (_name, _etype, eid) in &entity_ids {
                if let Ok(node) = db.get_memory(*eid)
                    && let Some(mentedb_core::memory::AttributeValue::String(cat)) =
                        node.attributes.get("category")
                {
                    for single_cat in cat.split(',') {
                        let trimmed = single_cat.trim().to_lowercase();
                        if !trimmed.is_empty() {
                            category_index.entry(trimmed).or_default().push(*eid);
                        }
                    }
                }
            }
            // Connect entities within the same category
            for (category, ids) in &category_index {
                if ids.len() < 2 {
                    continue;
                }
                let label = format!("same_category:{}", category);
                for i in 0..ids.len() {
                    for j in (i + 1)..ids.len() {
                        let edge = MemoryEdge {
                            source: ids[i],
                            target: ids[j],
                            edge_type: EdgeType::Related,
                            weight: 0.7,
                            created_at: now_us(),
                            valid_from: None,
                            valid_until: None,
                            label: Some(label.clone()),
                        };
                        let _ = db.relate(edge);
                    }
                }
                if debug {
                    eprintln!(
                        "[store_extracted] Connected {} entities in category '{}'",
                        ids.len(),
                        category
                    );
                }
            }
        }

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("stored_ids", stored_ids)?;
            Ok(dict.into())
        })
    }

    /// Run consolidation: gist extraction for entities with many connected memories.
    /// Creates Derived summary nodes linked to entities via Derived edges.
    fn consolidate(&self, min_memories: usize) -> PyResult<Vec<String>> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let debug = std::env::var("MENTEDB_DEBUG").is_ok();
        let rt =
            tokio::runtime::Runtime::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Set up LLM provider for gist generation
        let config = build_extraction_config_from_env(None)?;
        let http_provider =
            mentedb_extraction::provider::HttpExtractionProvider::new(config.clone())
                .map_err(to_pyerr)?;

        let min_memories = if min_memories == 0 { 5 } else { min_memories };
        let mut gist_ids = Vec::new();

        // Find all entity nodes by scanning graph node IDs
        let all_node_ids: Vec<MemoryId> = db.graph().graph().node_ids().to_vec();
        let entity_nodes: Vec<MemoryNode> = all_node_ids
            .iter()
            .filter_map(|id| db.get_memory(*id).ok())
            .filter(|m| m.tags.iter().any(|t| t.starts_with("entity_name:")))
            .filter(|m| !m.tags.iter().any(|t| t == "gist"))
            .collect();

        if debug {
            eprintln!("[consolidate] Found {} entity nodes", entity_nodes.len());
        }

        for entity in &entity_nodes {
            let entity_name = entity
                .attributes
                .get("entity_name")
                .and_then(|v| match v {
                    mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();

            // Get connected memories via PartOf edges
            let (neighbor_ids, _edges) = db.graph().get_context_subgraph(entity.id, 1);
            let connected_memories: Vec<String> = neighbor_ids
                .iter()
                .filter(|nid| **nid != entity.id)
                .filter_map(|nid| db.get_memory(*nid).ok())
                .filter(|m| !m.tags.iter().any(|t| t.starts_with("entity_name:")))
                .map(|m| m.content.clone())
                .collect();

            if connected_memories.len() < min_memories {
                continue;
            }

            // Check if gist already exists for this entity
            let has_gist = neighbor_ids.iter().any(|nid| {
                db.get_memory(*nid)
                    .ok()
                    .map(|m| m.tags.iter().any(|t| t == "gist"))
                    .unwrap_or(false)
            });
            if has_gist {
                continue;
            }

            if debug {
                eprintln!(
                    "[consolidate] Creating gist for '{}' ({} memories)",
                    entity_name,
                    connected_memories.len()
                );
            }

            let gist_prompt = format!(
                "Summarize everything known about '{}' based on these memories. Be factual and complete. \
                 Include all specific details (names, dates, amounts, relationships).\n\n{}",
                entity_name,
                connected_memories
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("[{}] {}", i, c))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            let gist_system = "You create factual summaries from memory evidence. Be comprehensive — include every detail. One to three sentences.";

            match rt.block_on(http_provider.call_text_with_retry(&gist_prompt, gist_system)) {
                Ok(gist_text) => {
                    let gist_content = gist_text.trim().to_string();
                    if debug {
                        eprintln!(
                            "[consolidate] Gist: {}",
                            &gist_content[..gist_content
                                .floor_char_boundary(std::cmp::min(gist_content.len(), 100))]
                        );
                    }

                    let gist_emb = if let Some(ref embedder) = self.embedder {
                        embedder.embed(&gist_content).map_err(to_pyerr)?
                    } else {
                        hash_embedding(&gist_content, 384)
                    };

                    let mut gist_node = MemoryNode::new(
                        entity.agent_id,
                        MemoryType::Semantic,
                        gist_content,
                        gist_emb,
                    );
                    gist_node.tags = vec![
                        "gist".to_string(),
                        format!("gist_for:{}", entity_name.to_lowercase()),
                    ];
                    gist_node.confidence = 0.9;
                    let gist_id = gist_node.id;
                    db.store(gist_node).map_err(to_pyerr)?;

                    // Link gist → entity via Derived edge
                    let edge = MemoryEdge {
                        source: gist_id,
                        target: entity.id,
                        edge_type: EdgeType::Derived,
                        weight: 0.9,
                        created_at: now_us(),
                        valid_from: None,
                        valid_until: None,
                        label: Some("gist_summary".to_string()),
                    };
                    let _ = db.relate(edge);
                    gist_ids.push(gist_id.to_string());
                }
                Err(e) => {
                    if debug {
                        eprintln!("[consolidate] LLM call failed for '{}': {e}", entity_name);
                    }
                }
            }
        }

        if debug {
            eprintln!("[consolidate] Created {} gist nodes", gist_ids.len());
        }
        Ok(gist_ids)
    }

    /// State resolution: for entities with conflicting edge labels, keep latest and mark older edges.
    /// E.g., if user moved from NYC to SF, the "lives_in:NYC" edge gets valid_until set.
    fn resolve_states(&self) -> PyResult<usize> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let debug = std::env::var("MENTEDB_DEBUG").is_ok();
        let mut resolved_count = 0usize;

        let all_node_ids: Vec<MemoryId> = db.graph().graph().node_ids().to_vec();
        let entity_nodes: Vec<MemoryNode> = all_node_ids
            .iter()
            .filter_map(|id| db.get_memory(*id).ok())
            .filter(|m| m.tags.iter().any(|t| t.starts_with("entity_name:")))
            .collect();

        for entity in &entity_nodes {
            // Get all outgoing edges from this entity
            let outgoing = db.graph().graph().outgoing(entity.id);

            // Group edges by their relationship label prefix (e.g., "lives_in", "works_at")
            let mut label_groups: std::collections::HashMap<String, Vec<(MemoryId, u64, f32)>> =
                std::collections::HashMap::new();
            for (target_id, data) in &outgoing {
                if let Some(ref label) = data.label {
                    // Extract relationship type (before the colon or full label)
                    let rel_type = if let Some(pos) = label.find(':') {
                        &label[..pos]
                    } else {
                        label.as_str()
                    };
                    // Only resolve state-like relationships
                    let stateful = matches!(
                        rel_type,
                        "lives_in"
                            | "works_at"
                            | "has_role"
                            | "married_to"
                            | "dating"
                            | "owns"
                            | "uses"
                            | "drives"
                            | "studies_at"
                            | "located_in"
                            | "employed_by"
                            | "salary"
                            | "title"
                            | "status"
                    );
                    if stateful {
                        label_groups.entry(rel_type.to_string()).or_default().push((
                            *target_id,
                            data.created_at,
                            data.weight,
                        ));
                    }
                }
            }

            // For each relationship type with multiple edges, keep latest
            for (rel_type, mut edges) in label_groups {
                if edges.len() < 2 {
                    continue;
                }
                edges.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by created_at descending
                let latest_time = edges[0].1;

                // Mark all but the latest with valid_until
                for &(target_id, created_at, _weight) in &edges[1..] {
                    // Create a superseding edge with valid_until set
                    let superseded = MemoryEdge {
                        source: entity.id,
                        target: target_id,
                        edge_type: EdgeType::PartOf,
                        weight: 0.1, // Downweight superseded edges
                        created_at,
                        valid_from: None,
                        valid_until: Some(latest_time),
                        label: Some(format!("{}:superseded", rel_type)),
                    };
                    let _ = db.relate(superseded);
                    resolved_count += 1;
                    if debug {
                        let entity_name = entity
                            .attributes
                            .get("entity_name")
                            .and_then(|v| match v {
                                mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()),
                                _ => None,
                            })
                            .unwrap_or_default();
                        eprintln!(
                            "[resolve_states] Superseded '{}' edge for '{}'",
                            rel_type, entity_name
                        );
                    }
                }
            }
        }

        if debug {
            eprintln!(
                "[resolve_states] Resolved {} conflicting states",
                resolved_count
            );
        }
        Ok(resolved_count)
    }

    /// Embedding-based entity resolution: merge near-duplicate entities.
    /// Cosine >0.85 auto-merge, 0.7-0.85 merge if same entity_type, <0.7 keep separate.
    fn resolve_entities(&self) -> PyResult<usize> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let debug = std::env::var("MENTEDB_DEBUG").is_ok();
        let mut merge_count = 0usize;

        let all_node_ids: Vec<MemoryId> = db.graph().graph().node_ids().to_vec();
        let entity_nodes: Vec<MemoryNode> = all_node_ids
            .iter()
            .filter_map(|id| db.get_memory(*id).ok())
            .filter(|m| m.tags.iter().any(|t| t.starts_with("entity_name:")))
            .collect();

        if entity_nodes.len() < 2 {
            return Ok(0);
        }

        // Build list of (id, name, type, embedding)
        let entities: Vec<(MemoryId, String, String, Vec<f32>)> = entity_nodes
            .iter()
            .map(|m| {
                let name = m
                    .attributes
                    .get("entity_name")
                    .and_then(|v| match v {
                        mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let etype = m
                    .attributes
                    .get("entity_type")
                    .and_then(|v| match v {
                        mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                (m.id, name, etype, m.embedding.clone())
            })
            .collect();

        let mut merged_into: std::collections::HashMap<MemoryId, MemoryId> =
            std::collections::HashMap::new();

        for i in 0..entities.len() {
            if merged_into.contains_key(&entities[i].0) {
                continue;
            }
            for j in (i + 1)..entities.len() {
                if merged_into.contains_key(&entities[j].0) {
                    continue;
                }

                let sim = cosine_similarity(&entities[i].3, &entities[j].3);
                let should_merge =
                    sim > 0.85 || (sim > 0.7 && entities[i].2.eq_ignore_ascii_case(&entities[j].2));

                if should_merge {
                    // Merge j into i: redirect j's edges to i
                    let j_outgoing = db.graph().graph().outgoing(entities[j].0);
                    for (target, data) in j_outgoing {
                        if target == entities[i].0 {
                            continue;
                        }
                        let edge = MemoryEdge {
                            source: entities[i].0,
                            target,
                            edge_type: data.edge_type,
                            weight: data.weight,
                            created_at: data.created_at,
                            valid_from: data.valid_from,
                            valid_until: data.valid_until,
                            label: data.label.clone(),
                        };
                        let _ = db.relate(edge);
                    }
                    let j_incoming = db.graph().graph().incoming(entities[j].0);
                    for (source, data) in j_incoming {
                        if source == entities[i].0 {
                            continue;
                        }
                        let edge = MemoryEdge {
                            source,
                            target: entities[i].0,
                            edge_type: data.edge_type,
                            weight: data.weight,
                            created_at: data.created_at,
                            valid_from: data.valid_from,
                            valid_until: data.valid_until,
                            label: data.label.clone(),
                        };
                        let _ = db.relate(edge);
                    }

                    // Mark merged entity with tag
                    merged_into.insert(entities[j].0, entities[i].0);
                    merge_count += 1;

                    if debug {
                        eprintln!(
                            "[resolve_entities] Merged '{}' into '{}' (sim={:.3})",
                            entities[j].1, entities[i].1, sim
                        );
                    }
                }
            }
        }

        if debug {
            eprintln!(
                "[resolve_entities] Merged {} duplicate entities",
                merge_count
            );
        }
        Ok(merge_count)
    }

    /// Generate community summaries for clusters of related entities.
    /// Groups entities by shared categories, creates a summary node per cluster.
    /// These summaries are searchable at a higher level of abstraction.
    fn build_communities(&self) -> PyResult<Vec<String>> {
        // Allow skipping via env var for benchmarking
        if std::env::var("MENTEDB_SKIP_COMMUNITIES").is_ok() {
            return Ok(vec![]);
        }
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let debug = std::env::var("MENTEDB_DEBUG").is_ok();
        let rt =
            tokio::runtime::Runtime::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let config = build_extraction_config_from_env(None)?;
        let http_provider =
            mentedb_extraction::provider::HttpExtractionProvider::new(config.clone())
                .map_err(to_pyerr)?;

        // Build category → entity list index
        let all_node_ids: Vec<MemoryId> = db.graph().graph().node_ids().to_vec();
        let mut category_entities: std::collections::HashMap<
            String,
            Vec<(String, String, String)>,
        > = std::collections::HashMap::new();

        for nid in &all_node_ids {
            if let Ok(node) = db.get_memory(*nid) {
                if !node.tags.iter().any(|t| t.starts_with("entity_name:")) {
                    continue;
                }
                if node.tags.iter().any(|t| t == "community_summary") {
                    continue;
                }

                let name = node
                    .attributes
                    .get("entity_name")
                    .and_then(|v| match v {
                        mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                let relationship = node
                    .attributes
                    .get("relationship")
                    .and_then(|v| match v {
                        mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "unknown".to_string());
                let etype = node
                    .attributes
                    .get("entity_type")
                    .and_then(|v| match v {
                        mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();

                if let Some(mentedb_core::memory::AttributeValue::String(cat)) =
                    node.attributes.get("category")
                {
                    for single_cat in cat.split(',') {
                        let trimmed = single_cat.trim().to_lowercase();
                        if !trimmed.is_empty() {
                            category_entities.entry(trimmed).or_default().push((
                                name.clone(),
                                relationship.clone(),
                                etype.clone(),
                            ));
                        }
                    }
                }
            }
        }

        let mut community_ids = Vec::new();

        for (category, entities) in &category_entities {
            if entities.len() < 2 {
                continue;
            }

            // Check if community summary already exists for this category
            let community_tag = format!("community:{}", category);
            let already_exists = all_node_ids.iter().any(|nid| {
                db.get_memory(*nid)
                    .ok()
                    .map(|m| m.tags.iter().any(|t| t == &community_tag))
                    .unwrap_or(false)
            });
            if already_exists {
                continue;
            }

            let entity_list = entities
                .iter()
                .map(|(name, rel, etype)| {
                    format!("- {} ({}) — user relationship: {}", name, etype, rel)
                })
                .collect::<Vec<_>>()
                .join("\n");
            let readable_cat = category.replace('_', " ");

            let prompt = format!(
                "Create a brief summary of the user's {} based on these entities:\n{}\n\n\
                 Write 1-2 sentences listing ALL items and the user's relationship to each. \
                 Be specific — include every entity name. This summary should be findable \
                 when someone searches for '{}'.",
                readable_cat, entity_list, readable_cat
            );
            let system =
                "You create concise factual summaries. List every item mentioned. Be complete.";

            match rt.block_on(http_provider.call_text_with_retry(&prompt, system)) {
                Ok(summary) => {
                    let summary_text = summary.trim().to_string();
                    if debug {
                        eprintln!(
                            "[community] {}: {}",
                            category,
                            &summary_text[..summary_text
                                .floor_char_boundary(std::cmp::min(summary_text.len(), 120))]
                        );
                    }

                    let emb = if let Some(ref embedder) = self.embedder {
                        embedder.embed(&summary_text).map_err(to_pyerr)?
                    } else {
                        hash_embedding(&summary_text, 384)
                    };

                    let mut node = MemoryNode::new(
                        mentedb_core::types::AgentId::new(),
                        MemoryType::Semantic,
                        summary_text,
                        emb,
                    );
                    node.tags = vec!["community_summary".to_string(), community_tag];
                    node.confidence = 0.95;
                    let node_id = node.id;
                    db.store(node).map_err(to_pyerr)?;
                    community_ids.push(node_id.to_string());
                }
                Err(e) => {
                    if debug {
                        eprintln!("[community] LLM failed for '{}': {e}", category);
                    }
                }
            }
        }

        if debug {
            eprintln!(
                "[community] Created {} community summaries",
                community_ids.len()
            );
        }
        Ok(community_ids)
    }

    /// Process a conversation turn through the full cognitive pipeline.
    ///
    /// Returns a dict with: context, stored_ids, episodic_id, pain_warnings,
    /// cache_hit, inference_actions, detected_actions, proactive_recalls,
    /// correction_id, sentiment, phantom_count, contradiction_count,
    /// predicted_topics, facts_extracted, edges_created.
    #[pyo3(signature = (user_message, assistant_response=None, turn_id=0, project_context=None, agent_id=None))]
    fn process_turn(
        &self,
        py: Python<'_>,
        user_message: &str,
        assistant_response: Option<String>,
        turn_id: u64,
        project_context: Option<String>,
        agent_id: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

        let aid = match agent_id {
            Some(s) => Some(parse_uuid(s)?),
            None => None,
        };

        let input = ProcessTurnInput {
            user_message: user_message.to_string(),
            assistant_response,
            turn_id,
            project_context,
            agent_id: aid,
        };

        let mut delta = self
            .delta_tracker
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("lock poisoned: {e}")))?;

        let result = db.process_turn(&input, &mut delta).map_err(to_pyerr)?;

        let dict = pyo3::types::PyDict::new(py);
        let context_list: Vec<PyObject> = result
            .context
            .iter()
            .map(|sm| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("id", sm.memory.id.to_string()).unwrap();
                d.set_item("content", &sm.memory.content).unwrap();
                d.set_item("score", sm.score).unwrap();
                d.into_any().unbind()
            })
            .collect();
        dict.set_item("context", context_list)?;
        dict.set_item(
            "stored_ids",
            result
                .stored_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>(),
        )?;
        dict.set_item("episodic_id", result.episodic_id.map(|id| id.to_string()))?;
        let pain_list: Vec<PyObject> = result
            .pain_warnings
            .iter()
            .map(|pw| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("signal_id", pw.signal_id.to_string()).unwrap();
                d.set_item("intensity", pw.intensity).unwrap();
                d.set_item("description", &pw.description).unwrap();
                d.into_any().unbind()
            })
            .collect();
        dict.set_item("pain_warnings", pain_list)?;
        dict.set_item("cache_hit", result.cache_hit)?;
        dict.set_item("inference_actions", result.inference_actions)?;
        let actions_list: Vec<PyObject> = result
            .detected_actions
            .iter()
            .map(|a| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("action_type", &a.action_type).unwrap();
                d.set_item("detail", &a.detail).unwrap();
                d.into_any().unbind()
            })
            .collect();
        dict.set_item("detected_actions", actions_list)?;
        let recalls_list: Vec<PyObject> = result
            .proactive_recalls
            .iter()
            .map(|pr| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("memory_id", pr.memory_id.to_string()).unwrap();
                d.set_item("content", &pr.content).unwrap();
                d.set_item("relevance", pr.relevance).unwrap();
                d.set_item("action_type", &pr.action_type).unwrap();
                d.into_any().unbind()
            })
            .collect();
        dict.set_item("proactive_recalls", recalls_list)?;
        dict.set_item(
            "correction_id",
            result.correction_id.map(|id| id.to_string()),
        )?;
        dict.set_item("sentiment", result.sentiment)?;
        dict.set_item("phantom_count", result.phantom_count)?;
        dict.set_item("contradiction_count", result.contradiction_count)?;
        dict.set_item("predicted_topics", result.predicted_topics)?;
        dict.set_item("facts_extracted", result.facts_extracted)?;
        dict.set_item("edges_created", result.edges_created)?;
        dict.set_item(
            "delta_added",
            result
                .delta_added
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "delta_removed",
            result
                .delta_removed
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>(),
        )?;
        dict.set_item("enrichment_pending", result.enrichment_pending)?;

        Ok(dict.into_any().unbind())
    }

    /// Flush and close the database.
    fn close(&mut self) -> PyResult<()> {
        if let Some(db) = self.db.take() {
            db.close().map_err(to_pyerr)?;
        }
        Ok(())
    }

    /// Rebuild all indexes from storage. Use after index corruption.
    fn rebuild_indexes(&self) -> PyResult<usize> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        db.rebuild_indexes().map_err(to_pyerr)
    }

    /// Check if enrichment is pending.
    fn needs_enrichment(&self) -> PyResult<bool> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        Ok(db.needs_enrichment())
    }

    /// Get the turn ID of the last completed enrichment.
    fn last_enrichment_turn(&self) -> PyResult<u64> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        Ok(db.last_enrichment_turn())
    }

    /// Manually request enrichment on the next check.
    fn request_enrichment(&self) -> PyResult<()> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        db.request_enrichment();
        Ok(())
    }

    /// Get episodic memories that need enrichment.
    fn enrichment_candidates(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let candidates = db.enrichment_candidates();
        let result: Vec<PyObject> = candidates
            .iter()
            .map(|m| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("id", m.id.to_string()).unwrap();
                d.set_item("content", &m.content).unwrap();
                d.set_item("memory_type", format!("{:?}", m.memory_type))
                    .unwrap();
                d.set_item("created_at", m.created_at).unwrap();
                d.into_any().unbind()
            })
            .collect();
        Ok(result)
    }

    /// Mark enrichment as complete at the given turn.
    fn mark_enrichment_complete(&self, turn_id: u64) -> PyResult<()> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        db.mark_enrichment_complete(turn_id);
        Ok(())
    }

    /// Link entities across sessions by name + embedding similarity.
    ///
    /// Returns a dict with `linked`, `ambiguous`, and `edges_created` counts.
    fn link_entities(&self, py: Python<'_>) -> PyResult<PyObject> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let result = db.link_entities().map_err(to_pyerr)?;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("linked", result.linked).unwrap();
        d.set_item("ambiguous", result.ambiguous).unwrap();
        d.set_item("edges_created", result.edges_created).unwrap();
        Ok(d.into_any().unbind())
    }

    /// Get all entity memory nodes (memories tagged with `entity:{name}`).
    fn entity_memories(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let entities = db.entity_memories();
        let result: Vec<PyObject> = entities
            .iter()
            .map(|m| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("id", m.id.to_string()).unwrap();
                d.set_item("content", &m.content).unwrap();
                d.set_item("memory_type", format!("{:?}", m.memory_type))
                    .unwrap();
                d.set_item("tags", &m.tags).unwrap();
                d.set_item("created_at", m.created_at).unwrap();
                d.into_any().unbind()
            })
            .collect();
        Ok(result)
    }

    /// Run the full 4-phase sleeptime enrichment pipeline.
    ///
    /// Phases:
    ///   1. Batch LLM extraction — episodic memories → semantic + entities
    ///   2. Entity linking — rule-based + LLM resolution
    ///   3. Community detection — category clustering + LLM summaries
    ///   4. User model — always-scoped profile from all knowledge
    ///
    /// Requires MENTEDB_LLM_PROVIDER and MENTEDB_LLM_API_KEY env vars.
    /// Returns a dict with enrichment statistics.
    #[pyo3(signature = (provider=None, current_turn=0, skip_extraction=false))]
    fn run_enrichment(
        &self,
        py: Python<'_>,
        provider: Option<&str>,
        current_turn: u64,
        skip_extraction: bool,
    ) -> PyResult<PyObject> {
        let db = self
            .db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let embedder = self
            .embedder
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no embedding provider configured"))?;

        let config = build_extraction_config_from_env(provider)?;
        let http_provider = HttpExtractionProvider::new(config.clone()).map_err(to_pyerr)?;
        let judge = mentedb_extraction::cognitive_adapter::ExtractionLlmJudge::new(http_provider);
        let cognitive_llm = mentedb_cognitive::CognitiveLlmService::new(judge);

        let rt = tokio::runtime::Runtime::new().map_err(to_pyerr)?;
        let enrichment_result = py.allow_threads(|| {
            rt.block_on(mentedb::enrichment::run_enrichment(
                db,
                config,
                embedder.as_ref(),
                Some(&cognitive_llm),
                current_turn,
                skip_extraction,
            ))
        });

        let d = pyo3::types::PyDict::new(py);
        d.set_item("memories_stored", enrichment_result.memories_stored)?;
        d.set_item("edges_created", enrichment_result.edges_created)?;
        d.set_item("entities_extracted", enrichment_result.entities_extracted)?;
        d.set_item("duplicates_skipped", enrichment_result.duplicates_skipped)?;
        d.set_item(
            "contradictions_found",
            enrichment_result.contradictions_found,
        )?;
        d.set_item("sync_linked", enrichment_result.sync_linked)?;
        d.set_item("llm_linked", enrichment_result.llm_linked)?;
        d.set_item("communities_created", enrichment_result.communities_created)?;
        d.set_item("user_model_updated", enrichment_result.user_model_updated)?;
        Ok(d.into_any().unbind())
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
            RustStreamAlert::Forgotten { memory_id, summary } => Self {
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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
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
    let model =
        std::env::var("MENTEDB_LLM_MODEL").unwrap_or_else(|_| provider.default_model().to_string());
    let quality_threshold = std::env::var("MENTEDB_EXTRACTION_QUALITY_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.7);
    let extraction_passes = std::env::var("MENTEDB_EXTRACTION_PASSES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1usize);

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
        extraction_passes,
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
