//! Context assembler: the main entry point for context assembly.

use mentedb_core::MemoryEdge;

use crate::budget::TokenBudget;
use crate::delta::DeltaTracker;
use crate::layout::{ContextBlock, ContextLayout, ScoredMemory};
use crate::serializer::{CompactFormat, ContextSerializer, DeltaFormat, StructuredFormat};
/// Output format for context serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Compact,
    Structured,
    Delta,
}

/// Configuration for context assembly.
#[derive(Debug, Clone)]
pub struct AssemblyConfig {
    pub token_budget: usize,
    pub format: OutputFormat,
    pub include_edges: bool,
    pub include_metadata: bool,
}

impl Default for AssemblyConfig {
    fn default() -> Self {
        Self {
            token_budget: 4096,
            format: OutputFormat::Structured,
            include_edges: false,
            include_metadata: true,
        }
    }
}

/// Metadata about the assembly result.
#[derive(Debug, Clone)]
pub struct AssemblyMetadata {
    pub total_candidates: usize,
    pub included_count: usize,
    pub excluded_count: usize,
    pub edges_included: usize,
    pub zones_used: usize,
}

/// The assembled context window ready for LLM consumption.
#[derive(Debug, Clone)]
pub struct ContextWindow {
    pub blocks: Vec<ContextBlock>,
    pub total_tokens: usize,
    pub format: String,
    pub metadata: AssemblyMetadata,
}

/// Main entry point for context assembly.
#[derive(Debug)]
pub struct ContextAssembler;

impl ContextAssembler {
    /// Assemble memories and edges into a context window.
    pub fn assemble(
        memories: Vec<ScoredMemory>,
        edges: Vec<MemoryEdge>,
        config: &AssemblyConfig,
    ) -> ContextWindow {
        let total_candidates = memories.len();

        // 1. Sort by score descending
        let mut sorted = memories;
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 2. Apply token budget — greedily include memories that fit
        let mut budget = TokenBudget::new(config.token_budget);
        let mut included = Vec::new();

        for sm in sorted {
            if budget.can_fit(&sm.memory.content) {
                budget.consume(&sm.memory.content);
                included.push(sm);
            }
        }

        let included_count = included.len();
        let excluded_count = total_candidates - included_count;

        // 3. Arrange into attention zones
        let layout = ContextLayout::default();
        let blocks = layout.arrange(included);

        // 4. Optionally append edge info to format
        let edge_section = if config.include_edges && !edges.is_empty() {
            let mut lines = vec!["\n## 🔗 Relationships".to_string()];
            for edge in &edges {
                lines.push(format!(
                    "- {} --[{:?} w={:.2}]--> {}",
                    &edge.source.to_string()[..8],
                    edge.edge_type,
                    edge.weight,
                    &edge.target.to_string()[..8],
                ));
            }
            lines.join("\n")
        } else {
            String::new()
        };

        // 5. Serialize
        let serialized = Self::serialize_blocks(&blocks, config);
        let total_tokens = budget.used_tokens;

        let format_output = if edge_section.is_empty() {
            serialized
        } else {
            format!("{serialized}\n{edge_section}")
        };

        let zones_used = blocks.iter().filter(|b| !b.memories.is_empty()).count();

        ContextWindow {
            blocks,
            total_tokens,
            format: format_output,
            metadata: AssemblyMetadata {
                total_candidates,
                included_count,
                excluded_count,
                edges_included: if config.include_edges { edges.len() } else { 0 },
                zones_used,
            },
        }
    }

    /// Assemble with delta tracking: only sends changes from the previous turn.
    pub fn assemble_delta(
        current_memories: Vec<ScoredMemory>,
        edges: Vec<MemoryEdge>,
        delta_tracker: &mut DeltaTracker,
        config: &AssemblyConfig,
    ) -> ContextWindow {
        let current_ids: Vec<_> = current_memories.iter().map(|sm| sm.memory.id).collect();
        let delta = delta_tracker.compute_delta(&current_ids, &delta_tracker.last_served.clone());

        // Build lookup for added memories
        let added_memories: Vec<ScoredMemory> = current_memories
            .into_iter()
            .filter(|sm| delta.added.contains(&sm.memory.id))
            .collect();

        let removed_summaries: Vec<String> = delta
            .removed
            .iter()
            .map(|id| format!("memory {}", &id.to_string()[..8]))
            .collect();

        let delta_header = DeltaTracker::format_delta_context(
            &added_memories
                .iter()
                .map(|sm| &sm.memory)
                .collect::<Vec<_>>(),
            &removed_summaries,
            delta.unchanged.len(),
        );

        // Assemble only the new memories
        let total_candidates = added_memories.len() + delta.unchanged.len();
        let mut budget = TokenBudget::new(config.token_budget);

        // Reserve tokens for delta header
        budget.consume(&delta_header);

        let mut sorted = added_memories;
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut included = Vec::new();
        for sm in sorted {
            if budget.can_fit(&sm.memory.content) {
                budget.consume(&sm.memory.content);
                included.push(sm);
            }
        }

        let included_count = included.len();
        let layout = ContextLayout::default();
        let blocks = layout.arrange(included);
        let total_tokens = budget.used_tokens;

        let fmt = DeltaFormat::new(delta_header);
        let format_output = fmt.serialize(&blocks);

        // Update tracker
        delta_tracker.update(&current_ids);

        let zones_used = blocks.iter().filter(|b| !b.memories.is_empty()).count();

        ContextWindow {
            blocks,
            total_tokens,
            format: format_output,
            metadata: AssemblyMetadata {
                total_candidates,
                included_count,
                excluded_count: total_candidates.saturating_sub(included_count),
                edges_included: if config.include_edges { edges.len() } else { 0 },
                zones_used,
            },
        }
    }

    fn serialize_blocks(blocks: &[ContextBlock], config: &AssemblyConfig) -> String {
        match config.format {
            OutputFormat::Compact => CompactFormat.serialize(blocks),
            OutputFormat::Structured => StructuredFormat.serialize(blocks),
            OutputFormat::Delta => {
                // Delta without tracker context — fall back to structured
                StructuredFormat.serialize(blocks)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::ScoredMemory;
    use mentedb_core::MemoryNode;
    use mentedb_core::memory::MemoryType;
    use mentedb_core::types::AgentId;

    fn make_scored(content: &str, score: f32, salience: f32, mem_type: MemoryType) -> ScoredMemory {
        let mut m = MemoryNode::new(AgentId::new(), mem_type, content.to_string(), vec![]);
        m.salience = salience;
        ScoredMemory { memory: m, score }
    }

    #[test]
    fn test_assemble_basic() {
        let memories = vec![
            make_scored("high priority fact", 0.95, 0.9, MemoryType::Semantic),
            make_scored("low priority note", 0.3, 0.4, MemoryType::Episodic),
        ];
        let config = AssemblyConfig::default();
        let window = ContextAssembler::assemble(memories, vec![], &config);

        assert_eq!(window.metadata.total_candidates, 2);
        assert_eq!(window.metadata.included_count, 2);
        assert!(!window.format.is_empty());
    }

    #[test]
    fn test_assemble_respects_budget() {
        // Tiny budget to force exclusion
        let memories = vec![
            make_scored(
                "a very important memory with lots of words",
                0.9,
                0.9,
                MemoryType::Semantic,
            ),
            make_scored(
                "another memory with many words in it",
                0.8,
                0.8,
                MemoryType::Episodic,
            ),
        ];
        let config = AssemblyConfig {
            token_budget: 10,
            ..Default::default()
        };
        let window = ContextAssembler::assemble(memories, vec![], &config);
        // At least one should be included, possibly not both
        assert!(window.metadata.included_count <= 2);
        assert!(window.total_tokens <= 10);
    }

    #[test]
    fn test_assemble_compact_format() {
        let memories = vec![make_scored("compact test", 0.9, 0.9, MemoryType::Semantic)];
        let config = AssemblyConfig {
            format: OutputFormat::Compact,
            ..Default::default()
        };
        let window = ContextAssembler::assemble(memories, vec![], &config);
        assert!(window.format.contains("M|Semantic|"));
    }

    #[test]
    fn test_assemble_delta() {
        let mut tracker = DeltaTracker::new();
        let m1 = make_scored("first fact", 0.9, 0.9, MemoryType::Semantic);
        let m2 = make_scored("second fact", 0.8, 0.8, MemoryType::Episodic);

        let config = AssemblyConfig::default();

        // First turn — all new
        let window = ContextAssembler::assemble_delta(
            vec![m1.clone(), m2.clone()],
            vec![],
            &mut tracker,
            &config,
        );
        assert!(window.format.contains("[NEW]"));

        // Second turn — same memories, should see UNCHANGED
        let window2 = ContextAssembler::assemble_delta(vec![m1, m2], vec![], &mut tracker, &config);
        assert!(window2.format.contains("[UNCHANGED]"));
    }
}
