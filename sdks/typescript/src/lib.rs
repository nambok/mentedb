use std::path::Path;

use mentedb::MenteDb;
use mentedb_cognitive::stream::CognitionStream as RustCognitionStream;
use mentedb_cognitive::trajectory::{
    DecisionState, TrajectoryNode, TrajectoryTracker as RustTrajectoryTracker,
};
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::{MemoryNode, MemoryType};
use mentedb_core::types::MemoryId;
use mentedb_core::MemoryEdge;
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
        other => Err(Error::from_reason(format!(
            "Unknown memory type: {other}"
        ))),
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
        other => Err(Error::from_reason(format!(
            "Unknown edge type: {other}"
        ))),
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
// MenteDB
// ---------------------------------------------------------------------------

#[napi]
pub struct MenteDB {
    inner: MenteDb,
}

#[napi]
impl MenteDB {
    /// Open (or create) a MenteDB instance backed by the given directory.
    #[napi(constructor)]
    pub fn new(data_dir: String) -> Result<Self> {
        let db = MenteDb::open(Path::new(&data_dir)).map_err(mente_err)?;
        Ok(Self { inner: db })
    }

    /// Store a memory and return its UUID.
    #[napi]
    pub fn store(
        &mut self,
        content: String,
        memory_type: String,
        embedding: Vec<f64>,
        agent_id: Option<String>,
        tags: Option<Vec<String>>,
    ) -> Result<String> {
        let mt = parse_memory_type(&memory_type)?;
        let emb: Vec<f32> = embedding.iter().map(|&v| v as f32).collect();

        let aid = match agent_id {
            Some(ref s) => parse_uuid(s)?,
            None => Uuid::nil(),
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
    pub fn recall(&mut self, query: String) -> Result<RecallResult> {
        let window = self.inner.recall(&query).map_err(mente_err)?;
        let text = window
            .blocks
            .iter()
            .flat_map(|b| b.memories.iter())
            .map(|m| m.memory.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let memory_count: u32 = window
            .blocks
            .iter()
            .map(|b| b.memories.len() as u32)
            .sum();

        Ok(RecallResult {
            text,
            total_tokens: window.total_tokens as u32,
            memory_count,
        })
    }

    /// Vector similarity search returning the top-k results.
    #[napi]
    pub fn search(&mut self, embedding: Vec<f64>, k: u32) -> Result<Vec<SearchResult>> {
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
        &mut self,
        source: String,
        target: String,
        edge_type: String,
        weight: f64,
    ) -> Result<()> {
        let src = parse_uuid(&source)?;
        let tgt = parse_uuid(&target)?;
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
        };
        self.inner.relate(edge).map_err(mente_err)
    }

    /// Remove a memory by ID.
    #[napi]
    pub fn forget(&mut self, memory_id: String) -> Result<()> {
        let id: MemoryId = parse_uuid(&memory_id)?;
        self.inner.forget(id).map_err(mente_err)
    }

    /// Flush and close the database.
    #[napi]
    pub fn close(&mut self) -> Result<()> {
        self.inner.close().map_err(mente_err)
    }
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
