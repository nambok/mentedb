//! Attention-aware context layout using U-curve optimization.
//!
//! LLMs attend best to content at the START and END of context,
//! with degradation in the middle. This module arranges memories
//! to exploit that attention pattern.

use mentedb_core::MemoryNode;
use mentedb_core::memory::MemoryType;

use crate::budget::estimate_tokens;

/// A memory with an associated relevance score.
#[derive(Debug, Clone)]
pub struct ScoredMemory {
    pub memory: MemoryNode,
    pub score: f32,
}

/// Attention zones based on U-curve optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionZone {
    /// Zone 1: Anti-patterns, corrections — AI sees first.
    Opening,
    /// Zone 2: Direct answers, highest salience.
    Critical,
    /// Zone 3: Supporting context, related memories.
    Primary,
    /// Zone 4: Background, lower salience.
    Supporting,
    /// Zone 5: Summary, open questions — AI sees last (reinforcement).
    Closing,
}

impl AttentionZone {
    /// Returns all zones in layout order.
    pub fn all_ordered() -> &'static [AttentionZone] {
        &[
            AttentionZone::Opening,
            AttentionZone::Critical,
            AttentionZone::Primary,
            AttentionZone::Supporting,
            AttentionZone::Closing,
        ]
    }
}

/// A block of memories assigned to a specific attention zone.
#[derive(Debug, Clone)]
pub struct ContextBlock {
    pub zone: AttentionZone,
    pub memories: Vec<ScoredMemory>,
    pub estimated_tokens: usize,
}

/// Thresholds for classifying memories into attention zones.
#[derive(Debug, Clone)]
pub struct ZoneThresholds {
    /// Minimum score to place a memory in the Critical zone (default: 0.8).
    pub critical_score: f32,
    /// Minimum salience to place a memory in the Critical zone (default: 0.7).
    pub critical_salience: f32,
    /// Minimum score to place a memory in the Primary zone (default: 0.5).
    pub primary_score: f32,
    /// Minimum score to place a memory in the Supporting zone (default: 0.2).
    pub supporting_score: f32,
}

impl Default for ZoneThresholds {
    fn default() -> Self {
        Self {
            critical_score: 0.8,
            critical_salience: 0.7,
            primary_score: 0.5,
            supporting_score: 0.2,
        }
    }
}

/// Arranges memories into attention zones following the U-curve pattern.
#[derive(Debug)]
pub struct ContextLayout {
    thresholds: ZoneThresholds,
}

impl ContextLayout {
    pub fn new(thresholds: ZoneThresholds) -> Self {
        Self { thresholds }
    }

    /// Arrange scored memories into attention-optimized zones.
    pub fn arrange(&self, memories: Vec<ScoredMemory>) -> Vec<ContextBlock> {
        let mut opening = Vec::new();
        let mut critical = Vec::new();
        let mut primary = Vec::new();
        let mut supporting = Vec::new();
        let mut closing = Vec::new();

        for sm in memories {
            let zone = self.classify(&sm);
            match zone {
                AttentionZone::Opening => opening.push(sm),
                AttentionZone::Critical => critical.push(sm),
                AttentionZone::Primary => primary.push(sm),
                AttentionZone::Supporting => supporting.push(sm),
                AttentionZone::Closing => closing.push(sm),
            }
        }

        // Sort each zone by score descending
        for group in [
            &mut opening,
            &mut critical,
            &mut primary,
            &mut supporting,
            &mut closing,
        ] {
            group.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let zones = [
            (AttentionZone::Opening, opening),
            (AttentionZone::Critical, critical),
            (AttentionZone::Primary, primary),
            (AttentionZone::Supporting, supporting),
            (AttentionZone::Closing, closing),
        ];

        zones
            .into_iter()
            .map(|(zone, memories)| {
                let estimated_tokens = Self::estimate_block_tokens(&memories);
                ContextBlock {
                    zone,
                    memories,
                    estimated_tokens,
                }
            })
            .collect()
    }

    /// Classify a memory into an attention zone based on its type, salience, and score.
    fn classify(&self, sm: &ScoredMemory) -> AttentionZone {
        let mem = &sm.memory;

        // Anti-patterns and corrections go to Opening (highest attention)
        match mem.memory_type {
            MemoryType::AntiPattern | MemoryType::Correction => return AttentionZone::Opening,
            _ => {}
        }

        // High salience + high score -> Critical
        if sm.score >= self.thresholds.critical_score
            && mem.salience >= self.thresholds.critical_salience
        {
            return AttentionZone::Critical;
        }

        // Moderate score -> Primary
        if sm.score >= self.thresholds.primary_score {
            return AttentionZone::Primary;
        }

        // Low score but still included
        if sm.score >= self.thresholds.supporting_score {
            return AttentionZone::Supporting;
        }

        // Very low score -> Closing (reinforcement zone)
        AttentionZone::Closing
    }

    fn estimate_block_tokens(memories: &[ScoredMemory]) -> usize {
        memories
            .iter()
            .map(|sm| estimate_tokens(&sm.memory.content))
            .sum()
    }
}

impl Default for ContextLayout {
    fn default() -> Self {
        Self::new(ZoneThresholds::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::MemoryNode;
    use mentedb_core::memory::MemoryType;

    fn make_memory(content: &str, memory_type: MemoryType, salience: f32) -> MemoryNode {
        let mut m = MemoryNode::new(
            uuid::Uuid::new_v4(),
            memory_type,
            content.to_string(),
            vec![],
        );
        m.salience = salience;
        m
    }

    #[test]
    fn test_antipattern_goes_to_opening() {
        let layout = ContextLayout::default();
        let memories = vec![ScoredMemory {
            memory: make_memory("never use eval", MemoryType::AntiPattern, 0.9),
            score: 0.95,
        }];
        let blocks = layout.arrange(memories);
        let opening = blocks
            .iter()
            .find(|b| b.zone == AttentionZone::Opening)
            .unwrap();
        assert_eq!(opening.memories.len(), 1);
    }

    #[test]
    fn test_high_score_goes_to_critical() {
        let layout = ContextLayout::default();
        let memories = vec![ScoredMemory {
            memory: make_memory("user prefers dark mode", MemoryType::Semantic, 0.9),
            score: 0.85,
        }];
        let blocks = layout.arrange(memories);
        let critical = blocks
            .iter()
            .find(|b| b.zone == AttentionZone::Critical)
            .unwrap();
        assert_eq!(critical.memories.len(), 1);
    }

    #[test]
    fn test_low_score_goes_to_supporting() {
        let layout = ContextLayout::default();
        let memories = vec![ScoredMemory {
            memory: make_memory("background info", MemoryType::Episodic, 0.3),
            score: 0.3,
        }];
        let blocks = layout.arrange(memories);
        let supporting = blocks
            .iter()
            .find(|b| b.zone == AttentionZone::Supporting)
            .unwrap();
        assert_eq!(supporting.memories.len(), 1);
    }

    #[test]
    fn test_arrange_produces_all_zones() {
        let layout = ContextLayout::default();
        let blocks = layout.arrange(vec![]);
        assert_eq!(blocks.len(), 5);
    }
}
