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
    /// When this fact became true in the real world.
    /// None means valid since creation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<Timestamp>,
    /// When this fact stopped being true.
    /// None means still valid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<Timestamp>,
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
            valid_from: None,
            valid_until: None,
        }
    }
}

impl MemoryNode {
    /// Returns true if this memory is temporally valid at the given timestamp.
    pub fn is_valid_at(&self, at: Timestamp) -> bool {
        let from = self.valid_from.unwrap_or(0);
        match self.valid_until {
            Some(until) => at >= from && at < until,
            None => at >= from,
        }
    }

    /// Mark this memory as no longer valid.
    pub fn invalidate(&mut self, at: Timestamp) {
        self.valid_until = Some(at);
    }

    /// Returns true if this memory has been invalidated.
    pub fn is_invalidated(&self) -> bool {
        self.valid_until.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentId;

    #[test]
    fn new_memory_has_no_temporal_bounds() {
        let node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            "test".to_string(),
            vec![1.0],
        );
        assert_eq!(node.valid_from, None);
        assert_eq!(node.valid_until, None);
        assert!(!node.is_invalidated());
    }

    #[test]
    fn invalidate_memory() {
        let mut node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            "Alice works at Acme".to_string(),
            vec![1.0],
        );
        assert!(node.is_valid_at(node.created_at));

        node.invalidate(node.created_at + 1_000_000);
        assert!(node.is_invalidated());
        assert!(node.is_valid_at(node.created_at));
        assert!(!node.is_valid_at(node.created_at + 1_000_000));
    }

    #[test]
    fn serde_backward_compatible() {
        let json = r#"{"id":"00000000-0000-0000-0000-000000000001","agent_id":"00000000-0000-0000-0000-000000000002","memory_type":"Semantic","embedding":[1.0],"content":"test","created_at":1000,"accessed_at":1000,"access_count":0,"salience":1.0,"confidence":1.0,"space_id":"00000000-0000-0000-0000-000000000000","attributes":{},"tags":[]}"#;
        let node: MemoryNode = serde_json::from_str(json).unwrap();
        assert_eq!(node.valid_from, None);
        assert_eq!(node.valid_until, None);
        assert!(node.is_valid_at(5000));
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
