//! MenteDB Consolidation: memory lifecycle and intelligence pipeline.
//!
//! This crate manages the long term health of stored memories:
//!
//! - [`consolidation`]: Merges similar memories into unified summaries
//! - [`compression`]: Reduces memory content to key facts
//! - [`decay`]: Applies time based salience decay
//! - [`archival`]: Moves cold memories to archival storage
//! - [`extraction`]: Extracts structured facts from memory content
//! - [`forget`]: GDPR compliant memory deletion with audit trails

/// Archival pipeline for moving cold memories to long term storage.
pub mod archival;
/// Memory content compression and key fact extraction.
pub mod compression;
/// Consolidation engine that merges similar memories.
pub mod consolidation;
/// Time based salience decay.
pub mod decay;
/// Structured fact extraction from memory content.
pub mod extraction;
/// GDPR compliant memory deletion with audit trails.
pub mod forget;

pub use archival::{ArchivalConfig, ArchivalDecision, ArchivalPipeline};
pub use compression::{CompressedMemory, MemoryCompressor};
pub use consolidation::{
    ConsolidatedMemory, ConsolidationCandidate, ConsolidationEngine, cosine_similarity,
};
pub use decay::{DecayConfig, DecayEngine};
pub use extraction::{ExtractedFact, FactExtractor};
pub use forget::{ForgetEngine, ForgetRequest, ForgetResult};

/// Test helper utilities (available in tests only).
#[cfg(test)]
pub(crate) mod test_helpers {
    use mentedb_core::MemoryNode;
    use mentedb_core::memory::MemoryType;
    use mentedb_core::types::Embedding;
    use std::collections::HashMap;

    /// Create a simple test memory with given content and embedding.
    pub fn make_memory(content: &str, embedding: Embedding) -> MemoryNode {
        MemoryNode {
            id: uuid::Uuid::new_v4(),
            agent_id: uuid::Uuid::new_v4(),
            memory_type: MemoryType::Episodic,
            embedding,
            content: content.to_string(),
            created_at: 0,
            accessed_at: 0,
            access_count: 0,
            salience: 0.5,
            confidence: 0.8,
            space_id: uuid::Uuid::new_v4(),
            attributes: HashMap::new(),
            tags: Vec::new(),
        }
    }
}
