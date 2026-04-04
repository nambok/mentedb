//! Common type aliases used throughout MenteDB.

use uuid::Uuid;

/// Unique identifier for a memory node.
pub type MemoryId = Uuid;

/// Unique identifier for an agent.
pub type AgentId = Uuid;

/// Unique identifier for a memory space.
pub type SpaceId = Uuid;

/// Embedding vector type.
pub type Embedding = Vec<f32>;

/// Timestamp in microseconds since epoch.
pub type Timestamp = u64;

/// Salience score (0.0 to 1.0).
pub type Salience = f32;

/// Confidence score (0.0 to 1.0).
pub type Confidence = f32;
