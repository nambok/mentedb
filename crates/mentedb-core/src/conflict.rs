//! Conflict Resolution — detect and resolve concurrent-write conflicts.

use serde::{Deserialize, Serialize};

use crate::types::{AgentId, MemoryId, Timestamp};

/// One agent's version of a memory that may be in conflict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictVersion {
    pub agent_id: AgentId,
    pub content: String,
    pub confidence: f32,
    pub timestamp: Timestamp,
}

/// How a conflict was (or should be) resolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Resolution {
    KeepLatest,
    KeepHighestConfidence,
    Merge(String),
    Manual(String),
}

/// A detected conflict on a single memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    pub memory_id: MemoryId,
    pub versions: Vec<ConflictVersion>,
    pub resolution: Option<Resolution>,
}

/// Detects and resolves multi-agent write conflicts.
#[derive(Debug, Default)]
pub struct ConflictResolver;

/// Threshold in microseconds — versions within this window count as concurrent.
const CONFLICT_WINDOW_US: Timestamp = 1_000_000; // 1 second

impl ConflictResolver {
    pub fn new() -> Self {
        Self
    }

    /// Detect a conflict: two or more versions by different agents within 1 second.
    pub fn detect_conflict(
        &self,
        memory_id: MemoryId,
        versions: &[ConflictVersion],
    ) -> Option<Conflict> {
        if versions.len() < 2 {
            return None;
        }

        // Collect versions whose timestamps are within the conflict window of any other
        // version written by a different agent.
        let mut dominated = vec![false; versions.len()];
        for i in 0..versions.len() {
            for j in (i + 1)..versions.len() {
                let dt = versions[i].timestamp.abs_diff(versions[j].timestamp);
                if dt <= CONFLICT_WINDOW_US && versions[i].agent_id != versions[j].agent_id {
                    dominated[i] = true;
                    dominated[j] = true;
                }
            }
        }

        let conflicting: Vec<ConflictVersion> = versions
            .iter()
            .zip(dominated.iter())
            .filter(|&(_, d)| *d)
            .map(|(v, _)| v.clone())
            .collect();

        if conflicting.len() >= 2 {
            Some(Conflict {
                memory_id,
                versions: conflicting,
                resolution: None,
            })
        } else {
            None
        }
    }

    /// Resolve a conflict using the given strategy, returning the winning version.
    pub fn auto_resolve(
        &self,
        conflict: &Conflict,
        strategy: Resolution,
    ) -> ConflictVersion {
        match &strategy {
            Resolution::KeepLatest => self.resolve_keep_latest(conflict),
            Resolution::KeepHighestConfidence => self.resolve_keep_highest_confidence(conflict),
            Resolution::Merge(merged) => ConflictVersion {
                agent_id: conflict.versions[0].agent_id,
                content: merged.clone(),
                confidence: conflict
                    .versions
                    .iter()
                    .map(|v| v.confidence)
                    .fold(0.0_f32, f32::max),
                timestamp: conflict
                    .versions
                    .iter()
                    .map(|v| v.timestamp)
                    .max()
                    .unwrap_or(0),
            },
            Resolution::Manual(text) => ConflictVersion {
                agent_id: conflict.versions[0].agent_id,
                content: text.clone(),
                confidence: 1.0,
                timestamp: conflict
                    .versions
                    .iter()
                    .map(|v| v.timestamp)
                    .max()
                    .unwrap_or(0),
            },
        }
    }

    /// Pick the version with the highest timestamp.
    pub fn resolve_keep_latest(&self, conflict: &Conflict) -> ConflictVersion {
        conflict
            .versions
            .iter()
            .max_by_key(|v| v.timestamp)
            .cloned()
            .expect("conflict must have at least one version")
    }

    /// Pick the version with the highest confidence score.
    pub fn resolve_keep_highest_confidence(&self, conflict: &Conflict) -> ConflictVersion {
        conflict
            .versions
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .cloned()
            .expect("conflict must have at least one version")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_version(agent: AgentId, content: &str, confidence: f32, ts: Timestamp) -> ConflictVersion {
        ConflictVersion {
            agent_id: agent,
            content: content.to_string(),
            confidence,
            timestamp: ts,
        }
    }

    #[test]
    fn no_conflict_single_version() {
        let r = ConflictResolver::new();
        let mid = Uuid::new_v4();
        let v = make_version(Uuid::new_v4(), "a", 0.9, 100);
        assert!(r.detect_conflict(mid, &[v]).is_none());
    }

    #[test]
    fn no_conflict_same_agent() {
        let r = ConflictResolver::new();
        let mid = Uuid::new_v4();
        let a = Uuid::new_v4();
        let v1 = make_version(a, "a", 0.9, 100);
        let v2 = make_version(a, "b", 0.8, 200);
        assert!(r.detect_conflict(mid, &[v1, v2]).is_none());
    }

    #[test]
    fn detect_conflict_different_agents() {
        let r = ConflictResolver::new();
        let mid = Uuid::new_v4();
        let a1 = Uuid::new_v4();
        let a2 = Uuid::new_v4();
        let v1 = make_version(a1, "v1", 0.8, 1_000_000);
        let v2 = make_version(a2, "v2", 0.9, 1_500_000);
        let conflict = r.detect_conflict(mid, &[v1, v2]);
        assert!(conflict.is_some());
        assert_eq!(conflict.unwrap().versions.len(), 2);
    }

    #[test]
    fn resolve_keep_latest() {
        let r = ConflictResolver::new();
        let a1 = Uuid::new_v4();
        let a2 = Uuid::new_v4();
        let conflict = Conflict {
            memory_id: Uuid::new_v4(),
            versions: vec![
                make_version(a1, "old", 0.9, 100),
                make_version(a2, "new", 0.5, 200),
            ],
            resolution: None,
        };
        let winner = r.auto_resolve(&conflict, Resolution::KeepLatest);
        assert_eq!(winner.content, "new");
    }

    #[test]
    fn resolve_keep_highest_confidence() {
        let r = ConflictResolver::new();
        let a1 = Uuid::new_v4();
        let a2 = Uuid::new_v4();
        let conflict = Conflict {
            memory_id: Uuid::new_v4(),
            versions: vec![
                make_version(a1, "confident", 0.95, 100),
                make_version(a2, "unsure", 0.3, 200),
            ],
            resolution: None,
        };
        let winner = r.auto_resolve(&conflict, Resolution::KeepHighestConfidence);
        assert_eq!(winner.content, "confident");
    }
}
