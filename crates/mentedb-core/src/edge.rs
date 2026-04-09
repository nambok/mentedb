//! MemoryEdge: typed, weighted relationships between memories.

use serde::{Deserialize, Serialize};

use crate::types::MemoryId;
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
    /// When this relationship became valid in the real world.
    /// None means valid since creation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<Timestamp>,
    /// When this relationship stopped being valid.
    /// None means still valid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<Timestamp>,
    /// Semantic label describing the relationship (e.g. "owns", "attends", "uses daily").
    /// None for edges without a specific semantic meaning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl MemoryEdge {
    /// Returns true if this edge is temporally valid at the given timestamp.
    /// An edge with no validity bounds is always valid.
    pub fn is_valid_at(&self, at: Timestamp) -> bool {
        let from = self.valid_from.unwrap_or(0);
        match self.valid_until {
            Some(until) => at >= from && at < until,
            None => at >= from,
        }
    }

    /// Mark this edge as no longer valid, setting valid_until to the given timestamp.
    pub fn invalidate(&mut self, at: Timestamp) {
        self.valid_until = Some(at);
    }

    /// Returns true if this edge has been invalidated.
    pub fn is_invalidated(&self) -> bool {
        self.valid_until.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_edge() -> MemoryEdge {
        MemoryEdge {
            source: MemoryId::new(),
            target: MemoryId::new(),
            edge_type: EdgeType::Related,
            weight: 0.8,
            created_at: 1000,
            valid_from: None,
            valid_until: None,
            label: None,
        }
    }

    #[test]
    fn unbounded_edge_always_valid() {
        let edge = make_edge();
        assert!(edge.is_valid_at(0));
        assert!(edge.is_valid_at(u64::MAX));
        assert!(!edge.is_invalidated());
    }

    #[test]
    fn edge_with_valid_from() {
        let mut edge = make_edge();
        edge.valid_from = Some(5000);
        assert!(!edge.is_valid_at(4999));
        assert!(edge.is_valid_at(5000));
        assert!(edge.is_valid_at(99999));
    }

    #[test]
    fn edge_with_valid_until() {
        let mut edge = make_edge();
        edge.valid_until = Some(8000);
        assert!(edge.is_valid_at(0));
        assert!(edge.is_valid_at(7999));
        assert!(!edge.is_valid_at(8000));
        assert!(edge.is_invalidated());
    }

    #[test]
    fn edge_with_both_bounds() {
        let mut edge = make_edge();
        edge.valid_from = Some(1000);
        edge.valid_until = Some(5000);
        assert!(!edge.is_valid_at(999));
        assert!(edge.is_valid_at(1000));
        assert!(edge.is_valid_at(3000));
        assert!(!edge.is_valid_at(5000));
    }

    #[test]
    fn invalidate_sets_valid_until() {
        let mut edge = make_edge();
        assert!(!edge.is_invalidated());
        edge.invalidate(9000);
        assert!(edge.is_invalidated());
        assert_eq!(edge.valid_until, Some(9000));
        assert!(edge.is_valid_at(8999));
        assert!(!edge.is_valid_at(9000));
    }

    #[test]
    fn serde_roundtrip_with_no_bounds() {
        let edge = make_edge();
        let json = serde_json::to_string(&edge).unwrap();
        assert!(!json.contains("valid_from"));
        assert!(!json.contains("valid_until"));
        let deserialized: MemoryEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.valid_from, None);
        assert_eq!(deserialized.valid_until, None);
    }

    #[test]
    fn serde_roundtrip_with_bounds() {
        let mut edge = make_edge();
        edge.valid_from = Some(1000);
        edge.valid_until = Some(5000);
        let json = serde_json::to_string(&edge).unwrap();
        assert!(json.contains("valid_from"));
        assert!(json.contains("valid_until"));
        let deserialized: MemoryEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.valid_from, Some(1000));
        assert_eq!(deserialized.valid_until, Some(5000));
    }

    #[test]
    fn deserialize_old_format_without_temporal_fields() {
        // Simulates loading data saved before temporal fields existed
        let json = r#"{"source":"00000000-0000-0000-0000-000000000001","target":"00000000-0000-0000-0000-000000000002","edge_type":"Related","weight":0.8,"created_at":1000}"#;
        let edge: MemoryEdge = serde_json::from_str(json).unwrap();
        assert_eq!(edge.valid_from, None);
        assert_eq!(edge.valid_until, None);
        assert_eq!(edge.label, None);
        assert!(edge.is_valid_at(5000));
    }

    #[test]
    fn edge_with_label() {
        let mut edge = make_edge();
        edge.label = Some("owns".to_string());
        let json = serde_json::to_string(&edge).unwrap();
        assert!(json.contains("\"label\":\"owns\""));
        let deserialized: MemoryEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.label, Some("owns".to_string()));
    }
}
