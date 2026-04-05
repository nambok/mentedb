//! MemoryNode: the fundamental unit of storage in MenteDB.

use serde::{Deserialize, Serialize};

use crate::types::*;
use crate::types::{AgentId, MemoryId, SpaceId};

/// The type classification of a memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// A specific event or interaction.
    Episodic,
    /// A general fact or concept.
    Semantic,
    /// A learned skill or procedure.
    Procedural,
    /// What NOT to do: learned from failures.
    AntiPattern,
    /// A pre-computed reasoning chain or decision template.
    Reasoning,
    /// A correction record: "was X, actually Y because Z".
    Correction,
}

/// A memory node: the atomic unit of knowledge in MenteDB.
///
/// Combines vector embeddings, graph connections, temporal properties,
/// and flexible attributes in a single primitive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    /// Unique identifier.
    pub id: MemoryId,
    /// The agent that owns this memory.
    pub agent_id: AgentId,
    /// Memory type classification.
    pub memory_type: MemoryType,
    /// Embedding vector for semantic similarity search.
    pub embedding: Embedding,
    /// Human-readable content.
    pub content: String,
    /// When this memory was created.
    pub created_at: Timestamp,
    /// When this memory was last accessed.
    pub accessed_at: Timestamp,
    /// How many times this memory has been accessed.
    pub access_count: u32,
    /// Current salience score (decays over time).
    pub salience: Salience,
    /// Confidence in this memory's accuracy.
    pub confidence: Confidence,
    /// The memory space this belongs to.
    pub space_id: SpaceId,
    /// Flexible key-value attributes.
    pub attributes: std::collections::HashMap<String, AttributeValue>,
    /// Tags for bitmap indexing.
    pub tags: Vec<String>,
}

impl MemoryNode {
    /// Create a new memory node with the given content and embedding.
    pub fn new(
        agent_id: AgentId,
        memory_type: MemoryType,
        content: String,
        embedding: Embedding,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        Self {
            id: MemoryId::new(),
            agent_id,
            memory_type,
            embedding,
            content,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            salience: 1.0,
            confidence: 1.0,
            space_id: SpaceId::nil(),
            attributes: std::collections::HashMap::new(),
            tags: Vec::new(),
        }
    }
}

/// Flexible attribute values for memory metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    /// A string value.
    String(String),
    /// An integer value.
    Integer(i64),
    /// A floating-point value.
    Float(f64),
    /// A boolean value.
    Boolean(bool),
    /// Raw bytes.
    Bytes(Vec<u8>),
}
