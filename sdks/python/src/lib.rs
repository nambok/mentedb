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
    #[pyo3(signature = (content, memory_type, embedding=None, agent_id=None, tags=None, created_at=None))]
    fn store(
        &mut self,
        content: &str,
        memory_type: &str,
        embedding: Option<Vec<f32>>,
        agent_id: Option<&str>,
        tags: Option<Vec<String>>,
        created_at: Option<u64>,
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
    #[pyo3(signature = (queries, k=10, tags=None, before=None))]
    fn search_multi(&mut self, queries: Vec<String>, k: usize, tags: Option<Vec<String>>, before: Option<u64>) -> PyResult<Vec<SearchResult>> {
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

        let tag_strs: Option<Vec<&str>> = tags.as_ref().map(|t| t.iter().map(|s| s.as_str()).collect());
        let tag_refs: Option<&[&str]> = tag_strs.as_deref();
        let time_range = before.map(|b| (0u64, b));

        let hits = db.recall_hybrid_multi(&embeddings, Some(&queries), k, tag_refs, time_range).map_err(to_pyerr)?;
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
    #[pyo3(signature = (query, k=10, provider=None, tags=None, before=None))]
    fn search_expanded(
        &mut self,
        query: &str,
        k: usize,
        provider: Option<&str>,
        tags: Option<Vec<String>>,
        before: Option<u64>,
    ) -> PyResult<Vec<SearchResult>> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        let config = build_extraction_config_from_env(provider)?;
        let http_provider = HttpExtractionProvider::new(config).map_err(to_pyerr)?;

        let rt = tokio::runtime::Runtime::new().map_err(to_pyerr)?;

        let debug = std::env::var("MENTEDB_DEBUG").is_ok();

        // Adaptive K values for escalating retrieval depth
        let k1 = std::cmp::min(k, 10);  // instant recall
        let k2 = std::cmp::min(k * 3, 30); // active search
        let k3 = std::cmp::min(k * 5, 50); // deep dig

        let tag_strs: Option<Vec<&str>> = tags.as_ref().map(|t| t.iter().map(|s| s.as_str()).collect());
        let tag_refs: Option<&[&str]> = tag_strs.as_deref();
        let time_range = before.map(|b| (0u64, b));

        // --- Pass 1: LLM query expansion + hybrid search (instant recall) ---
        let sub_queries = rt
            .block_on(http_provider.expand_query(query))
            .unwrap_or_default();

        let mut all_queries = vec![query.to_string()];
        // Detect counting intent and extract category keywords (two-tier)
        let mut item_keywords: Option<String> = None;
        let mut broad_keywords: Option<String> = None;
        let mut is_counting = false;
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
        if debug { eprintln!("[search_expanded] is_counting={}, item_kw={:?}, broad_kw={:?}",
            is_counting, item_keywords, broad_keywords); }

        let mut embeddings = Vec::with_capacity(all_queries.len());
        for q in &all_queries {
            let emb = if let Some(ref embedder) = self.embedder {
                embedder.embed(q).map_err(to_pyerr)?
            } else {
                hash_embedding(q, 384)
            };
            embeddings.push(emb);
        }

        let pass1_hits = db.recall_hybrid_multi(
            &embeddings, Some(&all_queries), k1, tag_refs, time_range
        ).map_err(to_pyerr)?;

        // --- Pass 2: Direct text search with original query (active search) ---
        // BM25 keyword matching catches literal terms that semantic search misses
        let query_emb = if let Some(ref embedder) = self.embedder {
            embedder.embed(query).map_err(to_pyerr)?
        } else {
            hash_embedding(query, 384)
        };
        let pass2_hits = db.recall_hybrid_at(
            &query_emb, Some(query), k2,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            tag_refs, time_range
        ).map_err(to_pyerr)?;

        // --- Pass 3: LLM-extracted key nouns for incidental mentions (deep dig) ---
        let key_nouns_prompt = format!(
            "Extract the 2-3 most important nouns from this question. \
             Return ONLY the nouns separated by spaces, nothing else.\n\
             Question: {}", query
        );
        let noun_system = "You extract key nouns from questions. Return only nouns, space-separated.";
        let nouns = rt
            .block_on(http_provider.call_text_with_retry(&key_nouns_prompt, noun_system))
            .unwrap_or_default();

        let pass3_hits = if !nouns.trim().is_empty() {
            let noun_emb = if let Some(ref embedder) = self.embedder {
                embedder.embed(nouns.trim()).map_err(to_pyerr)?
            } else {
                hash_embedding(nouns.trim(), 384)
            };
            db.recall_hybrid_at(
                &noun_emb, Some(nouns.trim()), k3,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64,
                tag_refs, time_range
            ).map_err(to_pyerr)?
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
            let k4_per = 5; // results per keyword — small but sufficient

            // Search each item keyword individually (specific subtypes)
            if let Some(ref kw_str) = item_keywords {
                let terms: Vec<&str> = kw_str.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
                if debug { eprintln!("[pass4] Searching {} item keywords individually", terms.len()); }
                for term in &terms {
                    let kw_emb = if let Some(ref embedder) = self.embedder {
                        embedder.embed(term).map_err(to_pyerr)?
                    } else {
                        hash_embedding(term, 384)
                    };
                    let hits = db.recall_hybrid_at(
                        &kw_emb, Some(term), k4_per,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_micros() as u64,
                        tag_refs, time_range
                    ).map_err(to_pyerr)?;
                    pass4_item_hits.extend(hits);
                }
            }

            // Search each broad keyword individually (category terms)
            if let Some(ref kw_str) = broad_keywords {
                let terms: Vec<&str> = kw_str.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
                if debug { eprintln!("[pass4] Searching {} broad keywords individually", terms.len()); }
                for term in &terms {
                    let kw_emb = if let Some(ref embedder) = self.embedder {
                        embedder.embed(term).map_err(to_pyerr)?
                    } else {
                        hash_embedding(term, 384)
                    };
                    let hits = db.recall_hybrid_at(
                        &kw_emb, Some(term), k4_per,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_micros() as u64,
                        tag_refs, time_range
                    ).map_err(to_pyerr)?;
                    pass4_broad_hits.extend(hits);
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
                        if node.tags.iter().any(|t| t.starts_with("entity_name:")) {
                            let sim = cosine_similarity(seed_emb, &node.embedding);
                            if sim > 0.3 {
                                entity_sims.push((*nid, sim));
                            }
                        }
                    }
                }
                entity_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                entity_sims.truncate(5); // Top 5 seed entities

                // BFS depth 2 with activation decay
                let decay = 0.85f32;
                let threshold = 0.3f32;
                let mut visited: std::collections::HashSet<MemoryId> = std::collections::HashSet::new();
                let mut frontier: Vec<(MemoryId, f32)> = entity_sims.iter()
                    .map(|(id, sim)| (*id, *sim))
                    .collect();

                for _depth in 0..2 {
                    let mut next_frontier: Vec<(MemoryId, f32)> = Vec::new();
                    for (nid, activation) in &frontier {
                        if !visited.insert(*nid) { continue; }
                        let propagated = activation * decay;
                        if propagated < threshold { continue; }
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
                    eprintln!("[search] Spreading activation found {} nodes", activation_hits.len());
                }
            }
        }

        // --- Merge all passes with RRF ---
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

        let mut merged: Vec<(String, f32)> = rrf_scores.into_iter().collect();
        merged.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Counting queries need more results to ensure completeness
        let final_k = if is_counting { std::cmp::max(k, 80) } else { k };
        merged.truncate(final_k);

        // --- Entity graph expansion ---
        // For entity nodes in the results, traverse PartOf edges to include
        // related attribute memories (e.g., entity MAX → breed: Golden Retriever)
        let mut expanded: Vec<(String, f32)> = Vec::with_capacity(merged.len() * 2);
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (id_str, score) in &merged {
            if seen.contains(id_str) { continue; }
            seen.insert(id_str.clone());
            expanded.push((id_str.clone(), *score));

            // Check if this is an entity node — if so, traverse graph
            if let Ok(mem_id) = parse_memory_id(id_str) {
                if let Ok(node) = db.get_memory(mem_id) {
                    let is_entity = node.tags.iter().any(|t| t.starts_with("entity_name:"));
                    if is_entity {
                        // Get subgraph (depth 1) to find PartOf neighbors
                        let (neighbor_ids, edges) = db.graph().get_context_subgraph(mem_id, 1);
                        // Hebbian learning: strengthen traversed edges
                        for edge in &edges {
                            db.graph_mut().strengthen_edge(edge.source, edge.target, 0.02);
                        }
                        for nid in neighbor_ids {
                            let nid_str = nid.to_string();
                            if !seen.contains(&nid_str) {
                                seen.insert(nid_str.clone());
                                // Entity neighbors get a slightly lower score
                                expanded.push((nid_str, score * 0.9));
                            }
                        }
                    }
                }
            }
        }
        // Re-sort after expansion and truncate
        expanded.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        expanded.truncate(final_k);

        // --- Pass 5: Iterative gap analysis (counting queries only) ---
        // Inspired by Iter-RetGen (2023) and IRCoT (ACL 2023): use Round 1 results
        // to inform a targeted Round 2 retrieval. The LLM examines what was found
        // and generates specific keywords for items that might be missing.
        if is_counting && !expanded.is_empty() {
            // Collect top-20 memory contents for the LLM to analyze
            let gap_limit = std::cmp::min(expanded.len(), 20);
            let mut found_items: Vec<String> = Vec::new();
            for (id_str, _) in expanded.iter().take(gap_limit) {
                if let Ok(mem_id) = parse_memory_id(id_str) {
                    if let Ok(node) = db.get_memory(mem_id) {
                        found_items.push(node.content.clone());
                    }
                }
            }

            if !found_items.is_empty() {
                let items_text = found_items.iter().enumerate()
                    .map(|(i, c)| {
                        let snip = &c[..std::cmp::min(c.len(), 200)];
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
                        let cleaned = response.trim().trim_start_matches("```json").trim_end_matches("```").trim();
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(cleaned) {
                            if let Some(gap_kw) = json.get("gap_keywords").and_then(|v| v.as_str()) {
                                let gap_terms: Vec<&str> = gap_kw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
                                if debug {
                                    let found = json.get("found").map(|v| v.to_string()).unwrap_or_default();
                                    eprintln!("[pass5-gap] Found: {}", found);
                                    eprintln!("[pass5-gap] Gap keywords: {:?}", gap_terms);
                                }

                                if !gap_terms.is_empty() {
                                    // Round 2: search each gap keyword individually
                                    let mut pass5_hits: Vec<(String, f32)> = Vec::new();
                                    let k5_per = 5;
                                    for term in &gap_terms {
                                        let kw_emb = if let Some(ref embedder) = self.embedder {
                                            embedder.embed(term).map_err(to_pyerr)?
                                        } else {
                                            hash_embedding(term, 384)
                                        };
                                        let hits = db.recall_hybrid_at(
                                            &kw_emb, Some(term), k5_per,
                                            std::time::SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .unwrap_or_default()
                                                .as_micros() as u64,
                                            tag_refs, time_range
                                        ).map_err(to_pyerr)?;
                                        for (id, score) in hits {
                                            let id_str = id.to_string();
                                            pass5_hits.push((id_str, score));
                                        }
                                    }

                                    if debug { eprintln!("[pass5-gap] Round 2 retrieved {} hits from {} gap keywords", pass5_hits.len(), gap_terms.len()); }

                                    // Merge Round 2 into expanded results with full RRF weight
                                    // (these are targeted, high-signal searches)
                                    let mut existing_scores: HashMap<String, f32> = expanded.iter()
                                        .map(|(id, s)| (id.clone(), *s))
                                        .collect();
                                    for (rank, (id_str, _)) in pass5_hits.iter().enumerate() {
                                        let rrf_contrib = 1.0 / (rrf_k + rank as f32);
                                        *existing_scores.entry(id_str.clone()).or_insert(0.0) += rrf_contrib;
                                    }

                                    // Rebuild expanded from merged scores
                                    expanded = existing_scores.into_iter().collect();
                                    expanded.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                                    expanded.truncate(final_k);
                                }
                            }
                        } else if debug {
                            eprintln!("[pass5-gap] Failed to parse gap analysis response: {}", cleaned);
                        }
                    }
                    Err(e) => if debug { eprintln!("[pass5-gap] LLM call failed: {e}"); },
                }
            }
        }

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
        if is_counting && !expanded.is_empty() {
            // --- Phase 1: Score memories for synthesis selection ---
            if debug { eprintln!("[rerank] Starting cognitive re-ranking for {} results", expanded.len()); }
            let rerank_limit = std::cmp::min(expanded.len(), 30);
            let mut memory_contents: Vec<(String, String)> = Vec::new(); // (id, content)
            for (id_str, _score) in expanded.iter().take(rerank_limit) {
                if let Ok(mem_id) = parse_memory_id(id_str) {
                    if let Ok(node) = db.get_memory(mem_id) {
                        memory_contents.push((id_str.clone(), node.content.clone()));
                    }
                }
            }

            // Collect IDs that the re-ranker considers relevant (score >= 1)
            let mut reranker_picks: std::collections::HashSet<String> = std::collections::HashSet::new();

            if !memory_contents.is_empty() {
                let mut memories_text = String::new();
                for (i, (_id, content)) in memory_contents.iter().enumerate() {
                    memories_text.push_str(&format!("[{}] {}\n", i, content));
                }

                let rerank_prompt = format!(
                    "Question: {}\n\n\
                     Memories:\n{}\n\
                     Score each memory 0-10 for how relevant it is to answering the question.\n\
                     Think broadly — any memory that mentions a countable item, fact, or detail \
                     that could contribute to the answer should score at least 5.\n\
                     A score of 10 means this memory directly mentions something that should be counted.\n\
                     A score of 0 means it is clearly unrelated.\n\n\
                     Return ONLY a JSON array of scores in order, e.g. [8, 2, 10, 0, ...]",
                    query, memories_text
                );
                let rerank_system = "You are a relevance scorer. Return ONLY a JSON array of integer scores 0-10. No explanation.";

                match rt.block_on(http_provider.call_text_with_retry(&rerank_prompt, rerank_system)) {
                    Ok(response) => {
                        if debug { eprintln!("[rerank] LLM response: {}", response.trim()); }
                        let trimmed = response.trim();
                        let array_str = if let Some(start) = trimmed.find('[') {
                            if let Some(end) = trimmed.rfind(']') {
                                &trimmed[start..=end]
                            } else { trimmed }
                        } else { trimmed };

                        match serde_json::from_str::<Vec<f32>>(array_str) {
                            Ok(scores) => {
                                if debug { eprintln!("[rerank] Parsed {} scores: {:?}", scores.len(), scores); }
                                for (i, (id, content)) in memory_contents.iter().enumerate() {
                                    let relevance = scores.get(i).copied().unwrap_or(0.0);
                                    let snip = &content[..std::cmp::min(content.len(), 80)];
                                    if debug { eprintln!("[rerank]   [{}] score={} | {}", i, relevance, snip); }
                                    if relevance >= 1.0 {
                                        reranker_picks.insert(id.clone());
                                    }
                                }
                                if debug { eprintln!("[rerank] Selected {} memories for synthesis", reranker_picks.len()); }
                            }
                            Err(e) => if debug { eprintln!("[rerank] Failed to parse scores: {e}"); },
                        }
                    }
                    Err(e) => if debug { eprintln!("[rerank] LLM call failed: {e}"); },
                }
            }

            // --- Phase 2: Reconstructive synthesis ---
            // Feed UNION of: (a) re-ranker picks + (b) top-K by original retrieval score.
            // This ensures nothing is lost even if the re-ranker misscores items.
            let retrieval_top_k = std::cmp::min(expanded.len(), 20);
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

            if debug { eprintln!("[synthesis] Feeding {} memories ({} from reranker + {} from retrieval top-{})",
                synth_ids.len(), reranker_picks.len(), synth_ids.len() - reranker_picks.len(), retrieval_top_k); }

            let mut synth_contents: Vec<String> = Vec::new();
            for id_str in &synth_ids {
                if let Ok(mem_id) = parse_memory_id(id_str) {
                    if let Ok(node) = db.get_memory(mem_id) {
                        synth_contents.push(node.content.clone());
                    }
                }
            }

            if !synth_contents.is_empty() {
                let evidence = synth_contents.iter().enumerate()
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
                        if let Ok(mem_id) = parse_memory_id(id_str) {
                            if let Ok(node) = db.get_memory(mem_id) {
                                let is_entity = node.tags.iter().any(|t| t.starts_with("entity_name:"));
                                if !is_entity { continue; }

                                let name = node.attributes.get("entity_name")
                                    .and_then(|v| match v { mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()), _ => None })
                                    .unwrap_or_default();
                                let relationship = node.attributes.get("relationship")
                                    .and_then(|v| match v { mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()), _ => None })
                                    .unwrap_or_else(|| "unknown".to_string());
                                let etype = node.attributes.get("entity_type")
                                    .and_then(|v| match v { mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()), _ => None })
                                    .unwrap_or_default();

                                // Build details from other attributes
                                let mut details = Vec::new();
                                for (k, v) in &node.attributes {
                                    if k == "entity_name" || k == "entity_type" || k == "relationship" || k == "category" || k == "relationship_owner" { continue; }
                                    if let mentedb_core::memory::AttributeValue::String(s) = v {
                                        details.push(format!("{}: {}", k, s));
                                    }
                                }
                                let owner_info = node.attributes.get("relationship_owner")
                                    .and_then(|v| match v { mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()), _ => None });

                                let detail_str = if details.is_empty() { etype } else { format!("{}, {}", etype, details.join(", ")) };
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
                    }

                    // Also scan the full graph for entity nodes we might have missed
                    // by looking at entity nodes connected to found memories via PartOf edges
                    let already_found: std::collections::HashSet<String> = graph_entities.iter().map(|(n, _, _)| n.to_lowercase()).collect();
                    for id_str in &synth_ids {
                        if let Ok(mem_id) = parse_memory_id(id_str) {
                            // Check outgoing edges for PartOf connections to entity nodes
                            let neighbors = db.graph().graph().outgoing(mem_id);
                            for (neighbor_id, stored_edge) in neighbors {
                                if stored_edge.edge_type != EdgeType::PartOf { continue; }
                                if let Ok(neighbor_node) = db.get_memory(neighbor_id) {
                                    let is_entity = neighbor_node.tags.iter().any(|t| t.starts_with("entity_name:"));
                                    if !is_entity { continue; }
                                    let name = neighbor_node.attributes.get("entity_name")
                                        .and_then(|v| match v { mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()), _ => None })
                                        .unwrap_or_default();
                                    if name.is_empty() || already_found.contains(&name.to_lowercase()) { continue; }
                                    let relationship = neighbor_node.attributes.get("relationship")
                                        .and_then(|v| match v { mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()), _ => None })
                                        .unwrap_or_else(|| "unknown".to_string());
                                    let etype = neighbor_node.attributes.get("entity_type")
                                        .and_then(|v| match v { mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()), _ => None })
                                        .unwrap_or_default();
                                    graph_entities.push((name, relationship, etype));
                                }
                            }
                        }
                    }

                    if !graph_entities.is_empty() {
                        // Compute graph-based count: items where user directly owns/uses/attends
                        let user_owned: Vec<&(String, String, String)> = graph_entities.iter()
                            .filter(|(_, rel, _)| {
                                let r = rel.to_lowercase();
                                !r.contains("someone_else") && !r.contains("considering") &&
                                !r.contains("wants") && !r.contains("previously") &&
                                (r.contains("owns") || r.contains("uses") || r.contains("attends") ||
                                 r.contains("plays") || r.contains("member") || r == "unknown")
                            })
                            .collect();
                        graph_count = Some(user_owned.len());
                        if debug {
                            eprintln!("[graph-synthesis] Found {} entities, {} user-owned:", graph_entities.len(), user_owned.len());
                            for (name, rel, det) in &graph_entities {
                                eprintln!("[graph-synthesis]   {} — {} ({})", name, rel, det);
                            }
                        }
                    }
                }

                // Build the synthesis prompt — use graph structure if available
                let synth_prompt = if let Some(count) = graph_count {
                    // Graph-informed: structured entity list + evidence for verification
                    let entity_list = graph_entities.iter().enumerate()
                        .map(|(i, (name, rel, det))| format!("{}. {} ({}) — {}", i + 1, name, det, rel))
                        .collect::<Vec<_>>()
                        .join("\n");

                    format!(
                        "Question: {}\n\n\
                         Entities found in memory graph:\n{}\n\n\
                         Evidence from memory (for verification):\n{}\n\n\
                         The graph found {} distinct entities. Using the evidence, verify and answer.\n\n\
                         IMPORTANT counting rules:\n\
                         - If the question asks about frequency (e.g., \"how many X per week/month\"), count OCCURRENCES not unique items.\n\
                           Example: \"Zumba on Tuesdays and Thursdays\" = 2 classes per week, not 1.\n\
                         - If the question asks about distinct items (e.g., \"how many pets\"), count unique items.\n\
                         - Items the user CURRENTLY HAS (even if planning to sell/donate) = count them\n\
                         - Items the user is CONSIDERING getting (planning to buy, thinking of trying) = do NOT count\n\
                         - Items belonging to someone else = do NOT count\n\n\
                         List each qualifying item/occurrence with details, then state the total count.",
                        query, entity_list, evidence, count
                    )
                } else {
                    // Fallback: flat evidence (no graph structure available)
                    format!(
                        "Question: {}\n\n\
                         Evidence from memory:\n{}\n\n\
                         Answer the question by listing every qualifying item. Think like a human recalling from memory:\n\
                         - Read through ALL the evidence\n\
                         - Identify every distinct item relevant to the question\n\
                         - For each item, state it clearly with key details (dates, amounts, names)\n\
                         - Be definitive: state the final count confidently\n\n\
                         IMPORTANT counting rules:\n\
                         - If the question asks about frequency (e.g., \"how many X per week/month\"), count OCCURRENCES not unique items.\n\
                           Example: \"Zumba on Tuesdays and Thursdays\" = 2 classes per week, not 1.\n\
                         - If the question asks about distinct items (e.g., \"how many pets\"), count unique items.\n\
                         - Items the user CURRENTLY HAS (even if planning to sell/donate) = count them\n\
                         - Items the user is CONSIDERING getting (planning to buy, thinking of trying) = do NOT count\n\
                         - Items belonging to someone else = do NOT count\n\n\
                         Format: numbered list of qualifying items/occurrences, then state the total.",
                        query, evidence
                    )
                };
                let synth_system = "You recall and organize facts from memory evidence. Be thorough — list every qualifying item. Be precise — don't count items the user only plans to acquire or that belong to others.";

                match rt.block_on(http_provider.call_text_with_retry(&synth_prompt, synth_system)) {
                    Ok(synthesis) => {
                        let mut final_synthesis = synthesis.trim().to_string();

                        // --- Chain-of-enumeration + dual-path verification ---
                        // For counting queries: LLM enumerates items as JSON, code counts.
                        // Compare with the synthesis answer. If they disagree, use union + verify.
                        if is_counting {
                            let enum_prompt = format!(
                                "Question: {}\n\n\
                                 Evidence from memory:\n{}\n\n\
                                 List EVERY item/occurrence relevant to this question as a JSON array.\n\
                                 IMPORTANT: If the question asks about frequency (per week/month), list each OCCURRENCE separately.\n\
                                 Example: \"Zumba on Tuesdays and Thursdays\" = two entries (one for Tuesday, one for Thursday).\n\
                                 For each item, include:\n\
                                 - \"name\": the item name (include day/time if frequency question)\n\
                                 - \"qualifies\": true if it should be counted (user currently has/does it), false if not (considering, someone else's, historical)\n\
                                 - \"reason\": brief reason for qualification decision\n\n\
                                 Return ONLY valid JSON. Example:\n\
                                 [{{\"name\": \"Zumba - Tuesday 7pm\", \"qualifies\": true, \"reason\": \"user attends weekly\"}},\n\
                                  {{\"name\": \"Zumba - Thursday 7pm\", \"qualifies\": true, \"reason\": \"user attends weekly\"}}]",
                                query, evidence
                            );
                            let enum_system = "You enumerate items/occurrences from evidence as JSON. For frequency questions, list each occurrence separately. Return ONLY a JSON array. No explanation.";

                            match rt.block_on(http_provider.call_text_with_retry(&enum_prompt, enum_system)) {
                                Ok(enum_response) => {
                                    let trimmed = enum_response.trim();
                                    let array_str = if let Some(start) = trimmed.find('[') {
                                        if let Some(end) = trimmed.rfind(']') {
                                            &trimmed[start..=end]
                                        } else { trimmed }
                                    } else { trimmed };

                                    match serde_json::from_str::<Vec<serde_json::Value>>(array_str) {
                                        Ok(items) => {
                                            let qualifying: Vec<&serde_json::Value> = items.iter()
                                                .filter(|item| item.get("qualifies").and_then(|v| v.as_bool()).unwrap_or(false))
                                                .collect();
                                            let enum_count = qualifying.len();

                                            if debug {
                                                eprintln!("[chain-enum] Enumerated {} items, {} qualifying:", items.len(), enum_count);
                                                for item in &items {
                                                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                                                    let q = item.get("qualifies").and_then(|v| v.as_bool()).unwrap_or(false);
                                                    let reason = item.get("reason").and_then(|v| v.as_str()).unwrap_or("");
                                                    eprintln!("[chain-enum]   {} — qualifies={} ({})", name, q, reason);
                                                }
                                            }

                                            // Build verified synthesis with code-counted result
                                            let item_list = qualifying.iter().enumerate()
                                                .map(|(i, item)| {
                                                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
                                                    let reason = item.get("reason").and_then(|v| v.as_str()).unwrap_or("");
                                                    format!("{}. {} ({})", i + 1, name, reason)
                                                })
                                                .collect::<Vec<_>>()
                                                .join("\n");

                                            // Append the code-counted enumeration to synthesis
                                            final_synthesis = format!(
                                                "{}\n\n---\nVerified enumeration ({} items):\n{}",
                                                final_synthesis, enum_count, item_list
                                            );

                                            if debug {
                                                if let Some(gc) = graph_count {
                                                    if gc != enum_count {
                                                        eprintln!("[dual-path] DISAGREEMENT: graph={} enum={}", gc, enum_count);
                                                    } else {
                                                        eprintln!("[dual-path] AGREEMENT: graph={} enum={}", gc, enum_count);
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => if debug { eprintln!("[chain-enum] Failed to parse JSON: {e}"); },
                                    }
                                }
                                Err(e) => if debug { eprintln!("[chain-enum] LLM call failed: {e}"); },
                            }
                        }

                        if debug { eprintln!("[synthesis] Generated: {}", &final_synthesis[..std::cmp::min(final_synthesis.len(), 200)]); }

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
                        synth_node.tags = vec!["synthesis:true".to_string(), "ephemeral:true".to_string()];
                        let synth_id = synth_node.id;
                        db.store(synth_node).map_err(to_pyerr)?;

                        // Prepend synthesis as first result — original order preserved after it
                        let top_score = expanded.first().map(|(_, s)| *s).unwrap_or(1.0);
                        expanded.insert(0, (synth_id.to_string(), top_score * 1.5));
                    }
                    Err(e) => if debug { eprintln!("[synthesis] LLM call failed: {e}"); },
                }
            }
        }

        Ok(expanded
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
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
            label: None,
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

        for memory in &quality_passed {
            let mt = map_extraction_type_to_memory_type(&memory.memory_type);
            embed_texts.push(memory.embedding_key());
            entries.push(MemEntry {
                content: memory.content.clone(),
                mt,
                tags: memory.tags.clone(),
                salience: memory.confidence,
                confidence: memory.confidence,
            });
        }

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
                    embed_texts.push(content.to_string());
                    entries.push(MemEntry {
                        content: content.to_string(),
                        mt: MemoryType::Episodic,
                        tags: vec!["turn".to_string()],
                        salience: 0.4,
                        confidence: 0.4,
                    });
                }
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
        let http_provider =
            HttpExtractionProvider::new(config.clone()).map_err(to_pyerr)?;
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
            dict.set_item("confidence", memory.confidence)?;
            dict.set_item("embedding_key", memory.embedding_key())?;
            results.append(dict)?;
        }

        // Include extracted entities as special dicts with _entity marker
        for entity in &extraction_result.entities {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("content", entity.to_content())?;
            dict.set_item("memory_type", "entity")?;
            dict.set_item("tags", vec![
                format!("entity_type:{}", entity.entity_type),
                format!("entity_name:{}", entity.name),
            ])?;
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
        // Also include user turns for episodic storage
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
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("content", content)?;
                    dict.set_item("memory_type", "episodic")?;
                    let tags: Vec<String> = vec!["turn".to_string()];
                    dict.set_item("tags", tags)?;
                    dict.set_item("confidence", 0.4)?;
                    dict.set_item("embedding_key", content)?;
                    results.append(dict)?;
                }
            }
        }
        Ok(results.into())
    }

    /// Store pre-extracted memories (from extract()) into the database.
    /// Uses batch embedding (1 API call for all memories) for speed.
    #[pyo3(signature = (memories, agent_id=None))]
    fn store_extracted(
        &mut self,
        memories: Vec<Py<pyo3::types::PyDict>>,
        agent_id: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self
            .db
            .as_mut()
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
            // Entity-specific fields (None for regular memories)
            entity_name: Option<String>,
            entity_type: Option<String>,
            entity_attributes: Option<std::collections::HashMap<String, String>>,
        }
        let mut parsed = Vec::with_capacity(memories.len());

        Python::with_gil(|py| {
            for mem_ref in &memories {
                let mem_dict = mem_ref.bind(py);
                let content: String = mem_dict.get_item("content")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();
                let memory_type: String = mem_dict.get_item("memory_type")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_else(|| "semantic".to_string());
                let tags: Vec<String> = mem_dict.get_item("tags")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();
                let confidence: f32 = mem_dict.get_item("confidence")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.5);
                let embed_key: String = mem_dict.get_item("embedding_key")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_else(|| content.clone());

                // Parse entity-specific fields if present
                let entity_name: Option<String> = mem_dict.get_item("entity_name")?
                    .map(|v| v.extract())
                    .transpose()?;
                let entity_type: Option<String> = mem_dict.get_item("entity_type")?
                    .map(|v| v.extract())
                    .transpose()?;
                let entity_attributes: Option<std::collections::HashMap<String, String>> =
                    mem_dict.get_item("entity_attributes")?
                        .map(|v| v.extract())
                        .transpose()?;

                parsed.push(ParsedMemory {
                    content, memory_type, tags, confidence, embed_key,
                    entity_name, entity_type, entity_attributes,
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

        // Phase 2: Batch embed in chunks (OpenAI has payload limits)
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

        // Phase 3: Store all memories with pre-computed embeddings.
        // Entity memories get attributes populated and graph edges created.
        // Entity resolution: merge attributes into existing entity if name+type match.
        let mut stored_ids = Vec::with_capacity(parsed.len());
        let mut entity_ids: Vec<(String, String, MemoryId)> = Vec::new(); // (name, type, id)

        // Build an index of existing entity nodes for resolution
        let mut existing_entities: std::collections::HashMap<String, MemoryId> = std::collections::HashMap::new();
        for mid in db.memory_ids() {
            if let Ok(node) = db.get_memory(mid) {
                if node.tags.iter().any(|t| t.starts_with("entity_name:")) {
                    let key = node.tags.iter()
                        .filter(|t| t.starts_with("entity_name:") || t.starts_with("entity_type:"))
                        .map(|t| t.to_lowercase())
                        .collect::<Vec<_>>()
                        .join("|");
                    existing_entities.insert(key, mid);
                }
            }
        }

        for (mem, emb) in parsed.into_iter().zip(embeddings.into_iter()) {
            let is_entity = mem.entity_name.is_some() && mem.entity_attributes.is_some();

            // Entity resolution: check if this entity already exists
            if is_entity {
                let entity_key = vec![
                    format!("entity_name:{}", mem.entity_name.as_ref().unwrap().to_lowercase()),
                    format!("entity_type:{}", mem.entity_type.as_ref().unwrap_or(&"unknown".to_string()).to_lowercase()),
                ].join("|");

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
                        // Re-store the updated node
                        let _ = db.store(existing_node);
                        stored_ids.push(existing_id.to_string());
                        entity_ids.push((
                            mem.entity_name.unwrap_or_default(),
                            mem.entity_type.unwrap_or_default(),
                            existing_id,
                        ));
                        continue; // Skip creating a new node
                    }
                }
            }

            let mt = map_extraction_type_to_memory_type(&mem.memory_type);
            let mut node = MemoryNode::new(aid, mt, mem.content, emb);
            node.tags = mem.tags;
            node.salience = mem.confidence;
            node.confidence = mem.confidence;

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

            let id = node.id;
            db.store(node).map_err(to_pyerr)?;
            stored_ids.push(id.to_string());

            // Track entity nodes for graph edge creation
            if let (Some(name), Some(etype)) = (mem.entity_name, mem.entity_type) {
                existing_entities.insert(
                    vec![
                        format!("entity_name:{}", name.to_lowercase()),
                        format!("entity_type:{}", etype.to_lowercase()),
                    ].join("|"),
                    id,
                );
                entity_ids.push((name, etype, id));
            }
        }

        // Phase 4: Create PartOf edges linking regular memories to their entities.
        // For each entity, link all memories that mention it.
        // Use the entity's "relationship" attribute as the edge label.
        for (entity_name, _etype, entity_id) in &entity_ids {
            let entity_name_lower = entity_name.to_lowercase();
            // Extract the relationship label from the entity node's attributes
            let edge_label = db.get_memory(*entity_id).ok().and_then(|node| {
                node.attributes.get("relationship").and_then(|v| match v {
                    mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()),
                    _ => None,
                })
            });
            for sid in &stored_ids {
                let mem_id = parse_memory_id(sid)?;
                if mem_id == *entity_id { continue; } // Don't self-link
                // Check if this memory mentions the entity
                if let Ok(mem_node) = db.get_memory(mem_id) {
                    if mem_node.content.to_lowercase().contains(&entity_name_lower) {
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
                        let _ = db.relate(edge); // Best effort
                    }
                }
            }
        }

        // Phase 5: Create entity-to-entity edges for entities sharing a category.
        // This creates the associative connections that make the graph mind-like.
        {
            // Build category → entity_id index
            let mut category_index: std::collections::HashMap<String, Vec<MemoryId>> = std::collections::HashMap::new();
            for (_name, _etype, eid) in &entity_ids {
                if let Ok(node) = db.get_memory(*eid) {
                    if let Some(mentedb_core::memory::AttributeValue::String(cat)) = node.attributes.get("category") {
                        if !cat.is_empty() {
                            category_index.entry(cat.to_lowercase()).or_default().push(*eid);
                        }
                    }
                }
            }
            // Connect entities within the same category
            for (category, ids) in &category_index {
                if ids.len() < 2 { continue; }
                let label = format!("same_category:{}", category);
                for i in 0..ids.len() {
                    for j in (i+1)..ids.len() {
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
                if debug { eprintln!("[store_extracted] Connected {} entities in category '{}'", ids.len(), category); }
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
    fn consolidate(&mut self, min_memories: usize) -> PyResult<Vec<String>> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let debug = std::env::var("MENTEDB_DEBUG").is_ok();
        let rt = tokio::runtime::Runtime::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Set up LLM provider for gist generation
        let config = build_extraction_config_from_env(None)?;
        let http_provider = mentedb_extraction::provider::HttpExtractionProvider::new(config.clone())
            .map_err(to_pyerr)?;

        let min_memories = if min_memories == 0 { 5 } else { min_memories };
        let mut gist_ids = Vec::new();

        // Find all entity nodes by scanning graph node IDs
        let all_node_ids: Vec<MemoryId> = db.graph().graph().node_ids().to_vec();
        let entity_nodes: Vec<MemoryNode> = all_node_ids.iter()
            .filter_map(|id| db.get_memory(*id).ok())
            .filter(|m| m.tags.iter().any(|t| t.starts_with("entity_name:")))
            .filter(|m| !m.tags.iter().any(|t| t == "gist"))
            .collect();

        if debug { eprintln!("[consolidate] Found {} entity nodes", entity_nodes.len()); }

        for entity in &entity_nodes {
            let entity_name = entity.attributes.get("entity_name")
                .and_then(|v| match v { mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()), _ => None })
                .unwrap_or_default();

            // Get connected memories via PartOf edges
            let (neighbor_ids, _edges) = db.graph().get_context_subgraph(entity.id, 1);
            let connected_memories: Vec<String> = neighbor_ids.iter()
                .filter(|nid| **nid != entity.id)
                .filter_map(|nid| db.get_memory(*nid).ok())
                .filter(|m| !m.tags.iter().any(|t| t.starts_with("entity_name:")))
                .map(|m| m.content.clone())
                .collect();

            if connected_memories.len() < min_memories { continue; }

            // Check if gist already exists for this entity
            let has_gist = neighbor_ids.iter().any(|nid| {
                db.get_memory(*nid).ok().map(|m| m.tags.iter().any(|t| t == "gist")).unwrap_or(false)
            });
            if has_gist { continue; }

            if debug { eprintln!("[consolidate] Creating gist for '{}' ({} memories)", entity_name, connected_memories.len()); }

            let gist_prompt = format!(
                "Summarize everything known about '{}' based on these memories. Be factual and complete. \
                 Include all specific details (names, dates, amounts, relationships).\n\n{}",
                entity_name,
                connected_memories.iter().enumerate()
                    .map(|(i, c)| format!("[{}] {}", i, c))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            let gist_system = "You create factual summaries from memory evidence. Be comprehensive — include every detail. One to three sentences.";

            match rt.block_on(http_provider.call_text_with_retry(&gist_prompt, gist_system)) {
                Ok(gist_text) => {
                    let gist_content = gist_text.trim().to_string();
                    if debug { eprintln!("[consolidate] Gist: {}", &gist_content[..std::cmp::min(gist_content.len(), 100)]); }

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
                    gist_node.tags = vec!["gist".to_string(), format!("gist_for:{}", entity_name.to_lowercase())];
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
                Err(e) => if debug { eprintln!("[consolidate] LLM call failed for '{}': {e}", entity_name); },
            }
        }

        if debug { eprintln!("[consolidate] Created {} gist nodes", gist_ids.len()); }
        Ok(gist_ids)
    }

    /// State resolution: for entities with conflicting edge labels, keep latest and mark older edges.
    /// E.g., if user moved from NYC to SF, the "lives_in:NYC" edge gets valid_until set.
    fn resolve_states(&mut self) -> PyResult<usize> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let debug = std::env::var("MENTEDB_DEBUG").is_ok();
        let mut resolved_count = 0usize;

        let all_node_ids: Vec<MemoryId> = db.graph().graph().node_ids().to_vec();
        let entity_nodes: Vec<MemoryNode> = all_node_ids.iter()
            .filter_map(|id| db.get_memory(*id).ok())
            .filter(|m| m.tags.iter().any(|t| t.starts_with("entity_name:")))
            .collect();

        for entity in &entity_nodes {
            // Get all outgoing edges from this entity
            let outgoing = db.graph().graph().outgoing(entity.id);
            
            // Group edges by their relationship label prefix (e.g., "lives_in", "works_at")
            let mut label_groups: std::collections::HashMap<String, Vec<(MemoryId, u64, f32)>> = std::collections::HashMap::new();
            for (target_id, data) in &outgoing {
                if let Some(ref label) = data.label {
                    // Extract relationship type (before the colon or full label)
                    let rel_type = if let Some(pos) = label.find(':') {
                        &label[..pos]
                    } else {
                        label.as_str()
                    };
                    // Only resolve state-like relationships
                    let stateful = matches!(rel_type,
                        "lives_in" | "works_at" | "has_role" | "married_to" | "dating" |
                        "owns" | "uses" | "drives" | "studies_at" | "located_in" |
                        "employed_by" | "salary" | "title" | "status"
                    );
                    if stateful {
                        label_groups.entry(rel_type.to_string()).or_default()
                            .push((*target_id, data.created_at, data.weight));
                    }
                }
            }

            // For each relationship type with multiple edges, keep latest
            for (rel_type, mut edges) in label_groups {
                if edges.len() < 2 { continue; }
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
                        let entity_name = entity.attributes.get("entity_name")
                            .and_then(|v| match v { mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()), _ => None })
                            .unwrap_or_default();
                        eprintln!("[resolve_states] Superseded '{}' edge for '{}'", rel_type, entity_name);
                    }
                }
            }
        }

        if debug { eprintln!("[resolve_states] Resolved {} conflicting states", resolved_count); }
        Ok(resolved_count)
    }

    /// Embedding-based entity resolution: merge near-duplicate entities.
    /// Cosine >0.85 auto-merge, 0.7-0.85 merge if same entity_type, <0.7 keep separate.
    fn resolve_entities(&mut self) -> PyResult<usize> {
        let db = self.db.as_mut().ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let debug = std::env::var("MENTEDB_DEBUG").is_ok();
        let mut merge_count = 0usize;

        let all_node_ids: Vec<MemoryId> = db.graph().graph().node_ids().to_vec();
        let entity_nodes: Vec<MemoryNode> = all_node_ids.iter()
            .filter_map(|id| db.get_memory(*id).ok())
            .filter(|m| m.tags.iter().any(|t| t.starts_with("entity_name:")))
            .collect();

        if entity_nodes.len() < 2 { return Ok(0); }

        // Build list of (id, name, type, embedding)
        let entities: Vec<(MemoryId, String, String, Vec<f32>)> = entity_nodes.iter()
            .map(|m| {
                let name = m.attributes.get("entity_name")
                    .and_then(|v| match v { mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()), _ => None })
                    .unwrap_or_default();
                let etype = m.attributes.get("entity_type")
                    .and_then(|v| match v { mentedb_core::memory::AttributeValue::String(s) => Some(s.clone()), _ => None })
                    .unwrap_or_default();
                (m.id, name, etype, m.embedding.clone())
            })
            .collect();

        let mut merged_into: std::collections::HashMap<MemoryId, MemoryId> = std::collections::HashMap::new();

        for i in 0..entities.len() {
            if merged_into.contains_key(&entities[i].0) { continue; }
            for j in (i+1)..entities.len() {
                if merged_into.contains_key(&entities[j].0) { continue; }

                let sim = cosine_similarity(&entities[i].3, &entities[j].3);
                let should_merge = sim > 0.85
                    || (sim > 0.7 && entities[i].2.eq_ignore_ascii_case(&entities[j].2));

                if should_merge {
                    // Merge j into i: redirect j's edges to i
                    let j_outgoing = db.graph().graph().outgoing(entities[j].0);
                    for (target, data) in j_outgoing {
                        if target == entities[i].0 { continue; }
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
                        if source == entities[i].0 { continue; }
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
                        eprintln!("[resolve_entities] Merged '{}' into '{}' (sim={:.3})",
                            entities[j].1, entities[i].1, sim);
                    }
                }
            }
        }

        if debug { eprintln!("[resolve_entities] Merged {} duplicate entities", merge_count); }
        Ok(merge_count)
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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
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
