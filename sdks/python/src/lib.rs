use std::path::Path;

use mentedb::MenteDb;
use mentedb_cognitive::pain::{PainRegistry as RustPainRegistry, PainSignal};
use mentedb_cognitive::stream::{CognitionStream as RustCognitionStream, StreamAlert as RustStreamAlert};
use mentedb_cognitive::trajectory::{
    DecisionState, TrajectoryNode, TrajectoryTracker as RustTrajectoryTracker,
};
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{Embedding, MemoryId, Timestamp};
use mentedb_core::{MemoryEdge, MemoryNode};
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
}

#[pymethods]
impl MenteDB {
    #[new]
    fn new(data_dir: &str) -> PyResult<Self> {
        let db = MenteDb::open(Path::new(data_dir)).map_err(to_pyerr)?;
        Ok(Self { db: Some(db) })
    }

    /// Store a memory and return its UUID string.
    #[pyo3(signature = (content, memory_type, embedding, agent_id=None, tags=None))]
    fn store(
        &mut self,
        content: &str,
        memory_type: &str,
        embedding: Vec<f32>,
        agent_id: Option<&str>,
        tags: Option<Vec<String>>,
    ) -> PyResult<String> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        let aid = match agent_id {
            Some(s) => parse_uuid(s)?,
            None => Uuid::new_v4(),
        };

        let mt = parse_memory_type(memory_type)?;

        let emb: Embedding = if embedding.is_empty() {
            // Generate a deterministic hash embedding so the index has something.
            hash_embedding(content, 384)
        } else {
            embedding
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

    /// Add a typed, weighted edge between two memories.
    fn relate(
        &mut self,
        source: &str,
        target: &str,
        edge_type: &str,
        weight: f32,
    ) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        let edge = MemoryEdge {
            source: parse_uuid(source)?,
            target: parse_uuid(target)?,
            edge_type: parse_edge_type(edge_type)?,
            weight,
            created_at: now_us(),
        };

        db.relate(edge).map_err(to_pyerr)
    }

    /// Remove a memory by its UUID string.
    fn forget(&mut self, memory_id: &str) -> PyResult<()> {
        let db = self.db.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("database is closed")
        })?;

        let id = parse_uuid(memory_id)?;
        db.forget(id).map_err(to_pyerr)
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
                let id = parse_uuid(&id_str)?;
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
        let mid = parse_uuid(memory_id)?;
        let signal = PainSignal {
            id: Uuid::new_v4(),
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
