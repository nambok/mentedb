use mentedb_core::MemoryNode;
use mentedb_core::types::{MemoryId, Timestamp};
use serde::{Deserialize, Serialize};

/// Decision on what to do with a memory during archival evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArchivalDecision {
    Keep,
    Archive,
    Delete,
    Consolidate(Vec<MemoryId>),
}

/// Configuration for the archival pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivalConfig {
    /// Minimum age in microseconds before considering archival.
    pub min_age_us: u64,
    /// Maximum salience for archival consideration.
    pub max_salience: f32,
    /// Minimum access count to avoid deletion.
    pub min_access_count: u32,
}

impl Default for ArchivalConfig {
    fn default() -> Self {
        Self {
            min_age_us: 7 * 24 * 3600 * 1_000_000, // 7 days
            max_salience: 0.1,
            min_access_count: 2,
        }
    }
}

const THIRTY_DAYS_US: u64 = 30 * 24 * 3600 * 1_000_000;
const SEVEN_DAYS_US: u64 = 7 * 24 * 3600 * 1_000_000;

/// Pipeline that evaluates memories for archival, deletion, or consolidation.
pub struct ArchivalPipeline {
    pub config: ArchivalConfig,
}

impl ArchivalPipeline {
    pub fn new(config: ArchivalConfig) -> Self {
        Self { config }
    }

    /// Evaluate a single memory for archival using its stored salience.
    ///
    /// Prefer `evaluate_effective` from the facade, which passes the salience
    /// decayed to `now`. The stored `salience` field is the base value as of the
    /// last reinforcement, not the current one, so judging archival on it alone
    /// is only correct when the caller has no decay engine.
    pub fn evaluate(&self, memory: &MemoryNode, now: Timestamp) -> ArchivalDecision {
        self.evaluate_effective(memory.salience, memory.created_at, memory.access_count, now)
    }

    /// Evaluate archival against an externally computed effective (decayed)
    /// salience, so lifecycle decisions reflect the memory's current strength
    /// rather than a stored base that decay never re-reads. Takes only the three
    /// fields the decision needs, so the caller need not clone the whole node.
    pub fn evaluate_effective(
        &self,
        effective_salience: f32,
        created_at: Timestamp,
        access_count: u32,
        now: Timestamp,
    ) -> ArchivalDecision {
        let age = now.saturating_sub(created_at);

        // Delete: very low salience, very old, rarely accessed
        if effective_salience < 0.05 && age > THIRTY_DAYS_US && access_count < 2 {
            return ArchivalDecision::Delete;
        }

        // Archive: low salience and older than 7 days
        if effective_salience < 0.1 && age > SEVEN_DAYS_US {
            return ArchivalDecision::Archive;
        }

        ArchivalDecision::Keep
    }

    /// Evaluate a batch of memories.
    pub fn evaluate_batch(
        &self,
        memories: &[MemoryNode],
        now: Timestamp,
    ) -> Vec<(MemoryId, ArchivalDecision)> {
        memories
            .iter()
            .map(|m| (m.id, self.evaluate(m, now)))
            .collect()
    }
}

impl Default for ArchivalPipeline {
    fn default() -> Self {
        Self::new(ArchivalConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_memory;

    const DAY_US: u64 = 24 * 3600 * 1_000_000;

    #[test]
    fn test_keep_recent_memory() {
        let pipeline = ArchivalPipeline::default();
        let now = 100 * DAY_US;
        let mut m = make_memory("recent", vec![1.0]);
        m.salience = 0.8;
        m.created_at = now - DAY_US;
        assert_eq!(pipeline.evaluate(&m, now), ArchivalDecision::Keep);
    }

    #[test]
    fn test_archive_old_low_salience() {
        let pipeline = ArchivalPipeline::default();
        let now = 100 * DAY_US;
        let mut m = make_memory("old", vec![1.0]);
        m.salience = 0.05;
        m.created_at = now - 10 * DAY_US;
        m.access_count = 5;
        assert_eq!(pipeline.evaluate(&m, now), ArchivalDecision::Archive);
    }

    #[test]
    fn test_delete_very_old_unused() {
        let pipeline = ArchivalPipeline::default();
        let now = 100 * DAY_US;
        let mut m = make_memory("ancient", vec![1.0]);
        m.salience = 0.01;
        m.created_at = now - 40 * DAY_US;
        m.access_count = 1;
        assert_eq!(pipeline.evaluate(&m, now), ArchivalDecision::Delete);
    }

    #[test]
    fn test_evaluate_batch() {
        let pipeline = ArchivalPipeline::default();
        let now = 100 * DAY_US;
        let mut m1 = make_memory("keep", vec![1.0]);
        m1.salience = 0.9;
        m1.created_at = now - DAY_US;

        let mut m2 = make_memory("archive", vec![1.0]);
        m2.salience = 0.05;
        m2.created_at = now - 10 * DAY_US;
        m2.access_count = 5;

        let results = pipeline.evaluate_batch(&[m1, m2], now);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1, ArchivalDecision::Keep);
        assert_eq!(results[1].1, ArchivalDecision::Archive);
    }

    // Regression: archival must judge on the salience decayed to `now`, not on a
    // stored base that decay no longer materializes. A full-strength memory a
    // week old is still strong (~0.45) and must be kept; a month-old one has
    // decayed past the floor and should go. Before the fix, a compounding decay
    // pass drove the stored base under 0.1 within a week and force-forgot it.
    #[test]
    fn effective_salience_governs_archival_not_stored_base() {
        use crate::decay::DecayEngine;
        let decay = DecayEngine::default();
        let pipe = ArchivalPipeline::default();

        // 8 days old, base 1.0, never reinforced. Derived ~0.45 > 0.1 archive
        // threshold, so keep, even though it is older than the 7-day min age.
        let now = 30 * DAY_US;
        let created = now - 8 * DAY_US;
        let eff = decay.compute_decay(1.0, created, created, 0, now);
        assert!(
            eff > 0.1,
            "derived salience {eff} should clear archive threshold"
        );
        assert_eq!(
            pipe.evaluate_effective(eff, created, 0, now),
            ArchivalDecision::Keep,
        );

        // 40 days old, never reinforced: derived salience decays under the floor
        // and, being 30d+ old and barely accessed, is deleted.
        let now = 60 * DAY_US;
        let created = now - 40 * DAY_US;
        let eff = decay.compute_decay(1.0, created, created, 0, now);
        assert!(
            eff < 0.05,
            "derived salience {eff} should fall under delete threshold"
        );
        assert_eq!(
            pipe.evaluate_effective(eff, created, 0, now),
            ArchivalDecision::Delete,
        );
    }
}
