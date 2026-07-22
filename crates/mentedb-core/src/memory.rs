//! MemoryNode: the fundamental unit of storage in MenteDB.

use serde::{Deserialize, Serialize};

use crate::types::*;
use crate::types::{AgentId, MemoryId, SpaceId, UserId};

/// Default `user_id` for memories deserialized from data written before the
/// `user_id` axis existed: the nil (shared/global) user. This must be an
/// explicit function, not `#[serde(default)]`, because `UserId::default()`
/// mints a fresh random id, which would make old memories look owned by a
/// random user instead of shared.
fn default_user_id() -> UserId {
    UserId::nil()
}

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
    /// The end user that owns this memory (orthogonal to `agent_id`).
    ///
    /// A memory belongs to both a user and an agent; a scoped query sees it
    /// only when it is visible on both axes. Defaults to the nil (shared)
    /// user, both for `new()` and for data written before this field existed.
    #[serde(default = "default_user_id")]
    pub user_id: UserId,
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
    /// Optional retrieval context, prepended to the content when the memory is
    /// indexed (BM25) and embedded, but NOT stored as part of the content the
    /// caller reads back. This is the contextual-retrieval hook: a short
    /// situating blurb the caller generates (for example "From a thread about the
    /// billing migration") that makes an otherwise-ambiguous memory findable by
    /// terms it never literally contains. None indexes the content as-is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
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
            user_id: UserId::nil(),
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
            context: None,
        }
    }

    /// Set the owning end user, returning the node (builder style).
    ///
    /// Parallels the `agent_id` constructor argument on the orthogonal user
    /// axis. Leave unset (nil) for shared/global memories.
    pub fn with_user_id(mut self, user_id: UserId) -> Self {
        self.user_id = user_id;
        self
    }

    /// Attach retrieval context (builder style). See [`MemoryNode::context`].
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        let ctx = context.into();
        self.context = if ctx.is_empty() { None } else { Some(ctx) };
        self
    }

    /// The text to index and embed: the context (when present) prefixed to the
    /// content, so retrieval matches on both while the stored `content` stays
    /// exactly what the caller wrote.
    pub fn indexed_text(&self) -> std::borrow::Cow<'_, str> {
        match &self.context {
            Some(ctx) if !ctx.is_empty() => {
                std::borrow::Cow::Owned(format!("{ctx}\n{}", self.content))
            }
            _ => std::borrow::Cow::Borrowed(&self.content),
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
    use crate::types::{AgentId, UserId};

    #[test]
    fn context_prefixes_indexed_text_but_not_content() {
        let n = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            "the migration plan".to_string(),
            vec![1.0],
        )
        .with_context("From a thread about billing");
        // The stored content the caller reads back is untouched.
        assert_eq!(n.content, "the migration plan");
        // What gets indexed/embedded carries the context prefix.
        assert_eq!(
            n.indexed_text(),
            "From a thread about billing\nthe migration plan"
        );

        // No context (and empty context) indexes the content as-is.
        let plain = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            "x".to_string(),
            vec![1.0],
        );
        assert_eq!(plain.indexed_text(), "x");
        assert_eq!(plain.context, None);
        assert_eq!(plain.with_context("").context, None);
    }

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
        // Data written before the user_id axis existed must deserialize as the
        // nil (shared) user, never a random one. A random default would make
        // legacy memories look privately owned and vanish from every scoped
        // query.
        assert!(
            node.user_id.is_nil(),
            "legacy memory (no user_id) must default to the nil user"
        );
    }

    #[test]
    fn with_user_id_sets_owner() {
        let user = UserId::new();
        let node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            "test".to_string(),
            vec![1.0],
        )
        .with_user_id(user);
        assert_eq!(node.user_id, user);
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
