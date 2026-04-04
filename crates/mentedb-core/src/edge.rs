//! MemoryEdge — typed, weighted relationships between memories.

use serde::{Deserialize, Serialize};

use crate::types::*;

/// The type of relationship between two memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Causal: A caused B.
    Caused,
    /// Temporal: A happened before B.
    Before,
    /// Semantic: A is related to B.
    Related,
    /// Contradicts: A conflicts with B.
    Contradicts,
    /// Supports: A provides evidence for B.
    Supports,
    /// Supersedes: A replaces B (newer information).
    Supersedes,
    /// Derived: A was derived/inferred from B.
    Derived,
    /// Part of: A is a component of B.
    PartOf,
}

/// A directed, typed, weighted edge between two memory nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEdge {
    /// Source memory ID.
    pub source: MemoryId,
    /// Target memory ID.
    pub target: MemoryId,
    /// Relationship type.
    pub edge_type: EdgeType,
    /// Strength of the relationship (0.0 to 1.0).
    pub weight: f32,
    /// When this edge was created.
    pub created_at: Timestamp,
}
