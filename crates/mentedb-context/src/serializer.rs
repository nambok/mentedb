//! Token-efficient serialization formats for context output.

use crate::layout::{AttentionZone, ContextBlock};

/// Trait for serializing context blocks into a string.
pub trait ContextSerializer {
    fn serialize(&self, blocks: &[ContextBlock]) -> String;
}

/// Compressed notation using ~3x fewer tokens than JSON.
/// Format: `M|<type>|<salience>|<content>|tags:<comma-separated>`
#[derive(Debug, Clone, Copy)]
pub struct CompactFormat;

impl ContextSerializer for CompactFormat {
    fn serialize(&self, blocks: &[ContextBlock]) -> String {
        let mut lines = Vec::new();

        for block in blocks {
            if block.memories.is_empty() {
                continue;
            }
            lines.push(format!("# {}", zone_label(block.zone)));
            for sm in &block.memories {
                let m = &sm.memory;
                let tags = if m.tags.is_empty() {
                    String::new()
                } else {
                    format!("|tags:{}", m.tags.join(","))
                };
                lines.push(format!(
                    "M|{:?}|{:.2}|{}{}",
                    m.memory_type, m.salience, m.content, tags
                ));
            }
        }

        lines.join("\n")
    }
}

/// Markdown-like structured format with headers and bullet points.
#[derive(Debug, Clone, Copy)]
pub struct StructuredFormat;

impl ContextSerializer for StructuredFormat {
    fn serialize(&self, blocks: &[ContextBlock]) -> String {
        let mut parts = Vec::new();

        for block in blocks {
            if block.memories.is_empty() {
                continue;
            }
            parts.push(format!("## {}", zone_label(block.zone)));
            for sm in &block.memories {
                let m = &sm.memory;
                let mut line = format!(
                    "- **[{:?}]** (salience: {:.2}) {}",
                    m.memory_type, m.salience, m.content
                );
                if !m.tags.is_empty() {
                    line.push_str(&format!(" [{}]", m.tags.join(", ")));
                }
                parts.push(line);
            }
            parts.push(String::new());
        }

        parts.join("\n")
    }
}

/// Delta format: only changes since last turn.
#[derive(Debug, Clone)]
pub struct DeltaFormat {
    pub delta_header: String,
}

impl DeltaFormat {
    pub fn new(delta_header: String) -> Self {
        Self { delta_header }
    }
}

impl ContextSerializer for DeltaFormat {
    fn serialize(&self, blocks: &[ContextBlock]) -> String {
        let mut parts = vec![self.delta_header.clone()];
        parts.push(String::new());

        // Only serialize non-empty blocks for new content
        for block in blocks {
            if block.memories.is_empty() {
                continue;
            }
            parts.push(format!("## {}", zone_label(block.zone)));
            for sm in &block.memories {
                parts.push(format!(
                    "- [NEW] {:?} | {}",
                    sm.memory.memory_type, sm.memory.content
                ));
            }
        }

        parts.join("\n")
    }
}

fn zone_label(zone: AttentionZone) -> &'static str {
    match zone {
        AttentionZone::Opening => "⚠️ Warnings & Corrections",
        AttentionZone::Critical => "🎯 Critical Context",
        AttentionZone::Primary => "📋 Primary Context",
        AttentionZone::Supporting => "📎 Supporting Context",
        AttentionZone::Closing => "🔁 Summary & Reinforcement",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{AttentionZone, ContextBlock, ScoredMemory};
    use mentedb_core::MemoryNode;
    use mentedb_core::memory::MemoryType;

    fn make_block(zone: AttentionZone, content: &str, mem_type: MemoryType) -> ContextBlock {
        let mut m = MemoryNode::new(uuid::Uuid::new_v4(), mem_type, content.to_string(), vec![]);
        m.salience = 0.9;
        m.tags = vec!["test".to_string()];
        ContextBlock {
            zone,
            memories: vec![ScoredMemory {
                memory: m,
                score: 0.9,
            }],
            estimated_tokens: 10,
        }
    }

    #[test]
    fn test_compact_format() {
        let blocks = vec![make_block(
            AttentionZone::Critical,
            "user likes Rust",
            MemoryType::Semantic,
        )];
        let output = CompactFormat.serialize(&blocks);
        assert!(output.contains("M|Semantic|0.90|user likes Rust|tags:test"));
        assert!(output.contains("🎯 Critical Context"));
    }

    #[test]
    fn test_structured_format() {
        let blocks = vec![make_block(
            AttentionZone::Opening,
            "avoid eval",
            MemoryType::AntiPattern,
        )];
        let output = StructuredFormat.serialize(&blocks);
        assert!(output.contains("## ⚠️ Warnings & Corrections"));
        assert!(output.contains("**[AntiPattern]**"));
        assert!(output.contains("avoid eval"));
    }

    #[test]
    fn test_delta_format() {
        let blocks = vec![make_block(
            AttentionZone::Critical,
            "new info",
            MemoryType::Episodic,
        )];
        let fmt = DeltaFormat::new("[UNCHANGED] 5 memories from previous turn".to_string());
        let output = fmt.serialize(&blocks);
        assert!(output.contains("[UNCHANGED] 5 memories"));
        assert!(output.contains("[NEW] Episodic | new info"));
    }

    #[test]
    fn test_empty_blocks_skipped() {
        let blocks = vec![ContextBlock {
            zone: AttentionZone::Supporting,
            memories: vec![],
            estimated_tokens: 0,
        }];
        let output = CompactFormat.serialize(&blocks);
        assert!(output.is_empty());
    }
}
