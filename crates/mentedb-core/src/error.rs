//! Error types for MenteDB.

use thiserror::Error;

use crate::types::MemoryId;

/// Top-level error type for MenteDB operations.
#[derive(Debug, Error)]
pub enum MenteError {
    /// The requested memory was not found.
    #[error("memory not found: {0}")]
    MemoryNotFound(MemoryId),

    /// A storage-layer error occurred.
    #[error("storage error: {0}")]
    Storage(String),

    /// An index-layer error occurred.
    #[error("index error: {0}")]
    Index(String),

    /// A query-layer error occurred.
    #[error("query error: {0}")]
    Query(String),

    /// A serialization or deserialization error occurred.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// A capacity limit was exceeded.
    #[error("capacity exceeded: {0}")]
    CapacityExceeded(String),

    /// The embedding dimension does not match the expected dimension.
    #[error("embedding dimension mismatch: got {got}, expected {expected}")]
    EmbeddingDimensionMismatch {
        /// Actual dimension provided.
        got: usize,
        /// Expected dimension.
        expected: usize,
    },

    /// The agent does not have permission to access the memory space.
    #[error("permission denied: agent {agent_id} cannot access space {space_id}")]
    PermissionDenied {
        /// The agent that was denied access.
        agent_id: crate::types::AgentId,
        /// The space that was inaccessible.
        space_id: crate::types::SpaceId,
    },

    /// A resource limit has been exhausted.
    #[error("resource exhausted: {0}")]
    ResourceExhausted(String),

    /// An I/O error occurred.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience result type for MenteDB operations.
pub type MenteResult<T> = Result<T, MenteError>;
