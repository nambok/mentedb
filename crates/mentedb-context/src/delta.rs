//! Delta-aware serving: track what changed between turns to save tokens.

use ahash::AHashSet;
use mentedb_core::MemoryNode;
use mentedb_core::types::MemoryId;

/// Result of computing a delta between two context sets.
#[derive(Debug, Clone)]
pub struct DeltaResult {
    pub added: Vec<MemoryId>,
    pub removed: Vec<MemoryId>,
    pub unchanged: Vec<MemoryId>,
}

/// Tracks context served across turns for delta computation.
#[derive(Debug, Clone)]
pub struct DeltaTracker {
    pub last_served: AHashSet<MemoryId>,
    pub last_turn_id: u64,
}

impl DeltaTracker {
    pub fn new() -> Self {
        Self {
            last_served: AHashSet::new(),
            last_turn_id: 0,
        }
    }

    /// Compute the delta between current memories and previously served set.
    pub fn compute_delta(
        &self,
        current: &[MemoryId],
        previous: &AHashSet<MemoryId>,
    ) -> DeltaResult {
        let current_set: AHashSet<MemoryId> = current.iter().copied().collect();

        let added: Vec<MemoryId> = current
            .iter()
            .filter(|id| !previous.contains(id))
            .copied()
            .collect();
        let removed: Vec<MemoryId> = previous
            .iter()
            .filter(|id| !current_set.contains(id))
            .copied()
            .collect();
        let unchanged: Vec<MemoryId> = current
            .iter()
            .filter(|id| previous.contains(id))
            .copied()
            .collect();

        DeltaResult {
            added,
            removed,
            unchanged,
        }
    }

    /// Update tracking state after serving context.
    pub fn update(&mut self, served_ids: &[MemoryId]) {
        self.last_served = served_ids.iter().copied().collect();
        self.last_turn_id += 1;
    }

    /// Format a human-readable delta context string.
    pub fn format_delta_context(
        added: &[&MemoryNode],
        removed_summaries: &[String],
        unchanged_count: usize,
    ) -> String {
        let mut parts = Vec::new();

        for mem in added {
            parts.push(format!("[NEW] {}", mem.content));
        }

        if !removed_summaries.is_empty() {
            if removed_summaries.len() == 1 {
                parts.push(format!("[REMOVED] {}", removed_summaries[0]));
            } else {
                parts.push(format!(
                    "[REMOVED] {} memories no longer relevant",
                    removed_summaries.len()
                ));
            }
        }

        if unchanged_count > 0 {
            parts.push(format!(
                "[UNCHANGED] {} memories from previous turn",
                unchanged_count
            ));
        }

        parts.join("\n")
    }
}

impl Default for DeltaTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::types::AgentId;

    #[test]
    fn test_compute_delta_all_new() {
        let tracker = DeltaTracker::new();
        let ids = vec![MemoryId::new(), MemoryId::new()];
        let delta = tracker.compute_delta(&ids, &tracker.last_served);
        assert_eq!(delta.added.len(), 2);
        assert!(delta.removed.is_empty());
        assert!(delta.unchanged.is_empty());
    }

    #[test]
    fn test_compute_delta_mixed() {
        let kept = MemoryId::new();
        let old = MemoryId::new();
        let new = MemoryId::new();

        let mut previous = AHashSet::new();
        previous.insert(kept);
        previous.insert(old);

        let tracker = DeltaTracker::new();
        let current = vec![kept, new];
        let delta = tracker.compute_delta(&current, &previous);

        assert_eq!(delta.added, vec![new]);
        assert_eq!(delta.removed, vec![old]);
        assert_eq!(delta.unchanged, vec![kept]);
    }

    #[test]
    fn test_update_advances_turn() {
        let mut tracker = DeltaTracker::new();
        assert_eq!(tracker.last_turn_id, 0);
        tracker.update(&[MemoryId::new()]);
        assert_eq!(tracker.last_turn_id, 1);
        assert_eq!(tracker.last_served.len(), 1);
    }

    #[test]
    fn test_format_delta_context() {
        use mentedb_core::memory::MemoryType;

        let mem = mentedb_core::MemoryNode::new(
            AgentId::new(),
            MemoryType::Episodic,
            "user switched to MySQL on March 15".to_string(),
            vec![],
        );
        let result = DeltaTracker::format_delta_context(
            &[&mem],
            &[
                "old memory 1".into(),
                "old memory 2".into(),
                "old memory 3".into(),
            ],
            12,
        );
        assert!(result.contains("[NEW] user switched to MySQL on March 15"));
        assert!(result.contains("[REMOVED] 3 memories no longer relevant"));
        assert!(result.contains("[UNCHANGED] 12 memories from previous turn"));
    }
}
