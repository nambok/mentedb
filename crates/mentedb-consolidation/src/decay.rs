use mentedb_core::types::Timestamp;
use mentedb_core::MemoryNode;
use serde::{Deserialize, Serialize};

/// Configuration for salience decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayConfig {
    /// Half-life in microseconds (default: 7 days).
    pub half_life_us: u64,
    /// Minimum salience floor.
    pub min_salience: f32,
    /// Boost per access.
    pub access_boost: f32,
    /// Maximum salience cap.
    pub max_salience: f32,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            half_life_us: 7 * 24 * 3600 * 1_000_000, // 7 days in µs
            min_salience: 0.01,
            access_boost: 0.1,
            max_salience: 1.0,
        }
    }
}

/// Engine that applies time-based salience decay to memories.
pub struct DecayEngine {
    pub config: DecayConfig,
}

impl DecayEngine {
    pub fn new(config: DecayConfig) -> Self {
        Self { config }
    }

    /// Compute decayed salience.
    ///
    /// Formula: `salience * 2^(-time_since_access / half_life) + access_boost * ln(1 + access_count)`
    pub fn compute_decay(
        &self,
        original_salience: f32,
        _created_at: Timestamp,
        accessed_at: Timestamp,
        access_count: u32,
        now: Timestamp,
    ) -> f32 {
        let time_since_access = now.saturating_sub(accessed_at) as f64;
        let half_life = self.config.half_life_us as f64;

        let decay_factor = (-time_since_access / half_life).exp2() as f32;
        let access_bonus =
            self.config.access_boost * ((1.0 + access_count as f64).ln() as f32);

        let result = original_salience * decay_factor + access_bonus;
        result.clamp(self.config.min_salience, self.config.max_salience)
    }

    /// Apply decay to all memories in a batch.
    pub fn apply_decay_batch(&self, memories: &mut [MemoryNode], now: Timestamp) {
        for m in memories.iter_mut() {
            m.salience = self.compute_decay(
                m.salience,
                m.created_at,
                m.accessed_at,
                m.access_count,
                now,
            );
        }
    }

    /// Whether a memory's salience is below the archival threshold.
    pub fn needs_archival(memory: &MemoryNode, threshold: f32) -> bool {
        memory.salience < threshold
    }
}

impl Default for DecayEngine {
    fn default() -> Self {
        Self::new(DecayConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_memory;

    #[test]
    fn test_no_decay_when_just_accessed() {
        let engine = DecayEngine::default();
        let now = 1_000_000_000_000u64;
        let salience = engine.compute_decay(0.8, now - 1000, now, 0, now);
        // Decay factor ≈ 1.0, access_bonus = 0.1 * ln(1) = 0
        assert!((salience - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_decay_after_half_life() {
        let engine = DecayEngine::default();
        let half_life = engine.config.half_life_us;
        let now = 2 * half_life;
        let salience = engine.compute_decay(1.0, 0, 0, 0, now);
        // After 2 half-lives: 1.0 * 2^(-2) = 0.25, plus ln(1)*0.1 = 0
        assert!((salience - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_access_boost() {
        let engine = DecayEngine::default();
        let now = 1_000_000_000_000u64;
        let s_no_access = engine.compute_decay(0.5, now, now, 0, now);
        let s_with_access = engine.compute_decay(0.5, now, now, 10, now);
        assert!(s_with_access > s_no_access);
    }

    #[test]
    fn test_apply_decay_batch() {
        let mut engine = DecayEngine::default();
        engine.config.half_life_us = 1_000_000;
        let now = 10_000_000u64;
        let mut memories = vec![
            {
                let mut m = make_memory("a", vec![1.0]);
                m.salience = 0.9;
                m.accessed_at = 0;
                m
            },
            {
                let mut m = make_memory("b", vec![1.0]);
                m.salience = 0.9;
                m.accessed_at = now; // just accessed
                m
            },
        ];
        engine.apply_decay_batch(&mut memories, now);
        assert!(memories[0].salience < memories[1].salience);
    }

    #[test]
    fn test_needs_archival() {
        let mut m = make_memory("x", vec![1.0]);
        m.salience = 0.005;
        assert!(DecayEngine::needs_archival(&m, 0.01));
        m.salience = 0.5;
        assert!(!DecayEngine::needs_archival(&m, 0.01));
    }
}
