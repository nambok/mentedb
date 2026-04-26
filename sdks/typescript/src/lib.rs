use std::path::Path;
use std::sync::Mutex;

use mentedb::MenteDb;
use mentedb::process_turn::ProcessTurnInput;
use mentedb::CognitiveConfig;
use mentedb_cognitive::stream::CognitionStream as RustCognitionStream;
use mentedb_cognitive::trajectory::{
    DecisionState, TrajectoryNode, TrajectoryTracker as RustTrajectoryTracker,
};
use mentedb_context::DeltaTracker;
use mentedb_core::MemoryEdge;
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::{MemoryNode, MemoryType};
use mentedb_core::types::{AgentId, MemoryId};
use mentedb_extraction::{
    ExtractionConfig, ExtractionPipeline, HttpExtractionProvider, LlmProvider,
    map_extraction_type_to_memory_type,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_memory_type(s: &str) -> Result<MemoryType> {
    match s {
        "episodic" => Ok(MemoryType::Episodic),
        "semantic" => Ok(MemoryType::Semantic),
        "procedural" => Ok(MemoryType::Procedural),
        "anti_pattern" => Ok(MemoryType::AntiPattern),
        "reasoning" => Ok(MemoryType::Reasoning),
        "correction" => Ok(MemoryType::Correction),
        other => Err(Error::from_reason(format!("Unknown memory type: {other}"))),
    }
}

fn parse_edge_type(s: &str) -> Result<EdgeType> {
    match s {
        "caused" => Ok(EdgeType::Caused),
        "before" => Ok(EdgeType::Before),
        "related" => Ok(EdgeType::Related),
        "contradicts" => Ok(EdgeType::Contradicts),
        "supports" => Ok(EdgeType::Supports),
        "supersedes" => Ok(EdgeType::Supersedes),
        "derived" => Ok(EdgeType::Derived),
        "part_of" => Ok(EdgeType::PartOf),
        other => Err(Error::from_reason(format!("Unknown edge type: {other}"))),
    }
}

fn parse_uuid(s: &str) -> Result<Uuid> {
    Uuid::parse_str(s).map_err(|e| Error::from_reason(format!("Invalid UUID: {e}")))
}

fn mente_err(e: mentedb_core::MenteError) -> Error {
    Error::from_reason(format!("{e}"))
}

// ---------------------------------------------------------------------------
// RecallResult
// ---------------------------------------------------------------------------

#[napi(object)]
pub struct RecallResult {
    pub text: String,
    pub total_tokens: u32,
    pub memory_count: u32,
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

#[napi(object)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
}

// ---------------------------------------------------------------------------
// IngestResult
// ---------------------------------------------------------------------------

#[napi(object)]
pub struct IngestResult {
    pub memories_stored: u32,
    pub rejected_low_quality: u32,
    pub rejected_duplicate: u32,
    pub contradictions: u32,
    pub stored_ids: Vec<String>,
}

// ---------------------------------------------------------------------------
// ProcessTurn result types
// ---------------------------------------------------------------------------

#[napi(object)]
pub struct JsContextItem {
    pub id: String,
    pub content: String,
    pub score: f64,
}

#[napi(object)]
pub struct JsPainWarning {
    pub signal_id: String,
    pub intensity: f64,
    pub description: String,
}

#[napi(object)]
pub struct JsDetectedAction {
    pub action_type: String,
    pub detail: String,
}

#[napi(object)]
pub struct JsProactiveRecall {
    pub memory_id: String,
    pub content: String,
    pub relevance: f64,
    pub action_type: String,
}

#[napi(object)]
pub struct JsProcessTurnResult {
    pub context: Vec<JsContextItem>,
    pub stored_ids: Vec<String>,
    pub episodic_id: Option<String>,
    pub pain_warnings: Vec<JsPainWarning>,
    pub cache_hit: bool,
    pub inference_actions: u32,
    pub detected_actions: Vec<JsDetectedAction>,
    pub proactive_recalls: Vec<JsProactiveRecall>,
    pub correction_id: Option<String>,
    pub sentiment: f64,
    pub phantom_count: u32,
    pub contradiction_count: u32,
    pub predicted_topics: Vec<String>,
    pub facts_extracted: u32,
    pub edges_created: u32,
    pub enrichment_pending: bool,
    pub delta_added: Vec<String>,
    pub delta_removed: Vec<String>,
}

#[napi(object)]
pub struct JsEntityLinkResult {
    pub linked: u32,
    pub ambiguous: u32,
    pub edges_created: u32,
}

// ---------------------------------------------------------------------------
// MenteDB
// ---------------------------------------------------------------------------

#[napi]
pub struct MenteDB {
    inner: MenteDb,
    delta_tracker: Mutex<DeltaTracker>,
}

#[napi]
impl MenteDB {
    /// Open (or create) a MenteDB instance backed by the given directory.
    #[napi(constructor)]
    pub fn new(data_dir: String) -> Result<Self> {
        let db = MenteDb::open_with_config(Path::new(&data_dir), CognitiveConfig::default())
            .map_err(mente_err)?;
        Ok(Self {
            inner: db,
            delta_tracker: Mutex::new(DeltaTracker::new()),
        })
    }

    /// Store a memory and return its UUID.
    #[napi]
    pub fn store(
        &self,
        content: String,
        memory_type: String,
        embedding: Vec<f64>,
        agent_id: Option<String>,
        tags: Option<Vec<String>>,
    ) -> Result<String> {
        let mt = parse_memory_type(&memory_type)?;
        let emb: Vec<f32> = embedding.iter().map(|&v| v as f32).collect();

        let aid = match agent_id {
            Some(ref s) => AgentId(parse_uuid(s)?),
            None => AgentId::nil(),
        };

        let mut node = MemoryNode::new(aid, mt, content, emb);
        if let Some(t) = tags {
            node.tags = t;
        }
        let id = node.id;

        self.inner.store(node).map_err(mente_err)?;
        Ok(id.to_string())
    }

    /// Recall memories using an MQL query string.
    #[napi]
    pub fn recall(&self, query: String) -> Result<RecallResult> {
        let window = self.inner.recall(&query).map_err(mente_err)?;
        let text = window
            .blocks
            .iter()
            .flat_map(|b| b.memories.iter())
            .map(|m| m.memory.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let memory_count: u32 = window.blocks.iter().map(|b| b.memories.len() as u32).sum();

        Ok(RecallResult {
            text,
            total_tokens: window.total_tokens as u32,
            memory_count,
        })
    }

    /// Vector similarity search returning the top-k results.
    #[napi]
    pub fn search(&self, embedding: Vec<f64>, k: u32) -> Result<Vec<SearchResult>> {
        let emb: Vec<f32> = embedding.iter().map(|&v| v as f32).collect();
        let hits = self
            .inner
            .recall_similar(&emb, k as usize)
            .map_err(mente_err)?;
        Ok(hits
            .into_iter()
            .map(|(id, score)| SearchResult {
                id: id.to_string(),
                score: score as f64,
            })
            .collect())
    }

    /// Create a typed, weighted edge between two memories.
    #[napi]
    pub fn relate(
        &self,
        source: String,
        target: String,
        edge_type: String,
        weight: f64,
        valid_from: Option<i64>,
        valid_until: Option<i64>,
    ) -> Result<()> {
        let src = MemoryId(parse_uuid(&source)?);
        let tgt = MemoryId(parse_uuid(&target)?);
        let et = parse_edge_type(&edge_type)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let edge = MemoryEdge {
            source: src,
            target: tgt,
            edge_type: et,
            weight: weight as f32,
            created_at: now,
            valid_from: valid_from.map(|v| v as u64),
            valid_until: valid_until.map(|v| v as u64),
            label: None,
        };
        self.inner.relate(edge).map_err(mente_err)
    }

    /// Remove a memory by ID.
    #[napi]
    pub fn forget(&self, memory_id: String) -> Result<()> {
        let id = MemoryId(parse_uuid(&memory_id)?);
        self.inner.forget(id).map_err(mente_err)
    }

    /// Extract memories from a conversation and store them.
    ///
    /// Requires `MENTEDB_LLM_PROVIDER` and `MENTEDB_LLM_API_KEY` env vars to be set,
    /// or pass `provider` ("openai", "anthropic", "ollama") explicitly.
    #[napi]
    pub fn ingest(
        &self,
        conversation: String,
        provider: Option<String>,
        agent_id: Option<String>,
    ) -> Result<IngestResult> {
        let config = build_ts_extraction_config(provider.as_deref())?;

        let http_provider = HttpExtractionProvider::new(config.clone())
            .map_err(|e| Error::from_reason(format!("{e}")))?;
        let pipeline = ExtractionPipeline::new(http_provider, config.clone());

        let rt = tokio::runtime::Runtime::new().map_err(|e| Error::from_reason(format!("{e}")))?;
        let all_memories = rt
            .block_on(pipeline.extract_from_conversation(&conversation))
            .map_err(|e| Error::from_reason(format!("{e}")))?;

        let total = all_memories.len();
        let quality_passed = pipeline.filter_quality(&all_memories);
        let rejected_low_quality = (total - quality_passed.len()) as u32;

        let aid = match agent_id {
            Some(ref s) => AgentId(parse_uuid(s)?),
            None => AgentId::nil(),
        };

        let mut stored_ids = Vec::new();
        for memory in &quality_passed {
            let mt = map_extraction_type_to_memory_type(&memory.memory_type);
            let mut node = MemoryNode::new(aid, mt, memory.content.clone(), vec![]);
            node.tags = memory.tags.clone();
            node.salience = memory.confidence;
            node.confidence = memory.confidence;
            let id = node.id;
            self.inner.store(node).map_err(mente_err)?;
            stored_ids.push(id.to_string());
        }

        Ok(IngestResult {
            memories_stored: stored_ids.len() as u32,
            rejected_low_quality,
            rejected_duplicate: 0,
            contradictions: 0,
            stored_ids,
        })
    }

    /// Process a conversation turn through the full cognitive pipeline.
    ///
    /// Returns context, stored memories, pain warnings, detected actions,
    /// sentiment, phantom count, predictions, and more.
    #[napi]
    pub fn process_turn(
        &self,
        user_message: String,
        assistant_response: Option<String>,
        turn_id: u32,
        project_context: Option<String>,
        agent_id: Option<String>,
    ) -> Result<JsProcessTurnResult> {
        let aid = match agent_id {
            Some(ref s) => Some(parse_uuid(s)?),
            None => None,
        };

        let input = ProcessTurnInput {
            user_message,
            assistant_response,
            turn_id: turn_id as u64,
            project_context,
            agent_id: aid,
        };

        let mut delta = self
            .delta_tracker
            .lock()
            .map_err(|e| Error::from_reason(format!("lock poisoned: {e}")))?;

        let result = self.inner.process_turn(&input, &mut delta).map_err(mente_err)?;

        Ok(JsProcessTurnResult {
            context: result
                .context
                .iter()
                .map(|sm| JsContextItem {
                    id: sm.memory.id.to_string(),
                    content: sm.memory.content.clone(),
                    score: sm.score as f64,
                })
                .collect(),
            stored_ids: result.stored_ids.iter().map(|id| id.to_string()).collect(),
            episodic_id: result.episodic_id.map(|id| id.to_string()),
            pain_warnings: result
                .pain_warnings
                .iter()
                .map(|pw| JsPainWarning {
                    signal_id: pw.signal_id.to_string(),
                    intensity: pw.intensity as f64,
                    description: pw.description.clone(),
                })
                .collect(),
            cache_hit: result.cache_hit,
            inference_actions: result.inference_actions,
            detected_actions: result
                .detected_actions
                .iter()
                .map(|a| JsDetectedAction {
                    action_type: a.action_type.clone(),
                    detail: a.detail.clone(),
                })
                .collect(),
            proactive_recalls: result
                .proactive_recalls
                .iter()
                .map(|pr| JsProactiveRecall {
                    memory_id: pr.memory_id.to_string(),
                    content: pr.content.clone(),
                    relevance: pr.relevance as f64,
                    action_type: pr.action_type.clone(),
                })
                .collect(),
            correction_id: result.correction_id.map(|id| id.to_string()),
            sentiment: result.sentiment as f64,
            phantom_count: result.phantom_count as u32,
            contradiction_count: result.contradiction_count as u32,
            predicted_topics: result.predicted_topics,
            facts_extracted: result.facts_extracted as u32,
            edges_created: result.edges_created,
            enrichment_pending: result.enrichment_pending,
            delta_added: result.delta_added.iter().map(|id| id.to_string()).collect(),
            delta_removed: result.delta_removed.iter().map(|id| id.to_string()).collect(),
        })
    }

    /// Flush and close the database.
    #[napi]
    pub fn close(&self) -> Result<()> {
        self.inner.close().map_err(mente_err)
    }

    /// Check if enrichment is pending.
    #[napi]
    pub fn needs_enrichment(&self) -> Result<bool> {
        Ok(self.inner.needs_enrichment())
    }

    /// Get the turn ID of the last completed enrichment.
    #[napi]
    pub fn last_enrichment_turn(&self) -> Result<u32> {
        Ok(self.inner.last_enrichment_turn() as u32)
    }

    /// Manually request enrichment on the next check.
    #[napi]
    pub fn request_enrichment(&self) -> Result<()> {
        self.inner.request_enrichment();
        Ok(())
    }

    /// Get episodic memories that need enrichment.
    #[napi]
    pub fn enrichment_candidates(&self) -> Result<Vec<JsContextItem>> {
        let candidates = self.inner.enrichment_candidates();
        Ok(candidates
            .iter()
            .map(|m| JsContextItem {
                id: m.id.to_string(),
                content: m.content.clone(),
                score: 1.0,
            })
            .collect())
    }

    /// Mark enrichment as complete at the given turn.
    #[napi]
    pub fn mark_enrichment_complete(&self, turn_id: u32) -> Result<()> {
        self.inner.mark_enrichment_complete(turn_id as u64);
        Ok(())
    }

    /// Link entities across sessions by name + embedding similarity.
    #[napi]
    pub fn link_entities(&self) -> Result<JsEntityLinkResult> {
        let result = self.inner.link_entities().map_err(mente_err)?;
        Ok(JsEntityLinkResult {
            linked: result.linked as u32,
            ambiguous: result.ambiguous as u32,
            edges_created: result.edges_created as u32,
        })
    }

    /// Get all entity memory nodes (memories tagged with `entity:{name}`).
    #[napi]
    pub fn entity_memories(&self) -> Result<Vec<JsContextItem>> {
        let entities = self.inner.entity_memories();
        Ok(entities
            .iter()
            .map(|m| JsContextItem {
                id: m.id.to_string(),
                content: m.content.clone(),
                score: 1.0,
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Extraction config helper
// ---------------------------------------------------------------------------

fn build_ts_extraction_config(provider_override: Option<&str>) -> Result<ExtractionConfig> {
    let provider_str = match provider_override {
        Some(p) => p.to_string(),
        None => std::env::var("MENTEDB_LLM_PROVIDER").unwrap_or_default(),
    };

    let provider = match provider_str.to_lowercase().as_str() {
        "openai" => LlmProvider::OpenAI,
        "anthropic" => LlmProvider::Anthropic,
        "ollama" => LlmProvider::Ollama,
        _ => {
            return Err(Error::from_reason(
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
        extraction_passes: 1,
    })
}

// ---------------------------------------------------------------------------
// CognitionStream
// ---------------------------------------------------------------------------

#[napi]
pub struct JsCognitionStream {
    inner: RustCognitionStream,
}

#[napi]
impl JsCognitionStream {
    #[napi(constructor)]
    pub fn new(buffer_size: Option<u32>) -> Self {
        let size = buffer_size.unwrap_or(1000) as usize;
        Self {
            inner: RustCognitionStream::new(size),
        }
    }

    /// Push a token into the stream buffer.
    #[napi]
    pub fn feed_token(&self, token: String) {
        self.inner.feed_token(&token);
    }

    /// Drain the accumulated buffer and return its content.
    #[napi]
    pub fn drain_buffer(&self) -> String {
        self.inner.drain_buffer()
    }
}

// ---------------------------------------------------------------------------
// TrajectoryTracker
// ---------------------------------------------------------------------------

#[napi]
pub struct JsTrajectoryTracker {
    inner: RustTrajectoryTracker,
    next_turn_id: u64,
}

#[napi]
impl JsTrajectoryTracker {
    #[napi(constructor)]
    pub fn new(max_turns: Option<u32>) -> Self {
        let mt = max_turns.unwrap_or(100) as usize;
        Self {
            inner: RustTrajectoryTracker::new(mt),
            next_turn_id: 0,
        }
    }

    /// Record a conversation turn.
    #[napi]
    pub fn record_turn(
        &mut self,
        topic: String,
        decision_state: String,
        open_questions: Vec<String>,
    ) {
        let state = match decision_state.as_str() {
            "investigating" => DecisionState::Investigating,
            "interrupted" => DecisionState::Interrupted,
            "completed" => DecisionState::Completed,
            s if s.starts_with("narrowed:") => {
                DecisionState::NarrowedTo(s.trim_start_matches("narrowed:").to_string())
            }
            s if s.starts_with("decided:") => {
                DecisionState::Decided(s.trim_start_matches("decided:").to_string())
            }
            _ => DecisionState::Investigating,
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let node = TrajectoryNode {
            turn_id: self.next_turn_id,
            topic_embedding: vec![],
            topic_summary: topic,
            decision_state: state,
            open_questions,
            timestamp: now,
        };
        self.next_turn_id += 1;
        self.inner.record_turn(node);
    }

    /// Get a resume context string for the current trajectory.
    #[napi]
    pub fn get_resume_context(&self) -> Option<String> {
        self.inner.get_resume_context()
    }

    /// Predict the next likely topics based on trajectory.
    #[napi]
    pub fn predict_next_topics(&self) -> Vec<String> {
        self.inner.predict_next_topics()
    }
}
