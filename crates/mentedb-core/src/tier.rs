//! Memory tiers — cognitive-inspired storage hierarchy.

use serde::{Deserialize, Serialize};

/// Memory tiers inspired by human cognitive architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum MemoryTier {
    /// Active conversation context. Ultra-fast, limited capacity.
    /// Analogy: human working memory (~7 items).
    Working = 0,
    /// Recent interactions and events. Fast retrieval with temporal decay.
    /// Analogy: human episodic memory.
    Episodic = 1,
    /// Consolidated facts, concepts, and relationships.
    /// Analogy: human semantic memory.
    Semantic = 2,
    /// Learned procedures, workflows, and skills.
    /// Analogy: human procedural/muscle memory.
    Procedural = 3,
    /// Long-term storage. Compressed, immutable, sealed.
    /// Analogy: human long-term memory archives.
    Archival = 4,
}
