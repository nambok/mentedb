use mentedb_core::edge::EdgeType;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::MemoryId;
use mentedb_core::MemoryNode;

#[derive(Debug, Clone)]
pub enum InferredAction {
    FlagContradiction {
        existing: MemoryId,
        new: MemoryId,
        reason: String,
    },
    MarkObsolete {
        memory: MemoryId,
        superseded_by: MemoryId,
    },
    CreateEdge {
        source: MemoryId,
        target: MemoryId,
        edge_type: EdgeType,
        weight: f32,
    },
    UpdateConfidence {
        memory: MemoryId,
        new_confidence: f32,
    },
    PropagateBeliefChange {
        root: MemoryId,
        delta: f32,
    },
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

pub struct WriteInferenceEngine;

impl WriteInferenceEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn infer_on_write(
        &self,
        new_memory: &MemoryNode,
        existing_memories: &[MemoryNode],
        existing_edges: &[(MemoryId, MemoryId, EdgeType)],
    ) -> Vec<InferredAction> {
        let _ = existing_edges; // reserved for future graph-aware inference
        let mut actions = Vec::new();

        for existing in existing_memories {
            if existing.id == new_memory.id {
                continue;
            }

            let sim = cosine_similarity(&new_memory.embedding, &existing.embedding);

            // Very high similarity: potential duplicate or contradiction
            if sim > 0.95 {
                if existing.agent_id == new_memory.agent_id
                    && existing.content != new_memory.content
                {
                    actions.push(InferredAction::FlagContradiction {
                        existing: existing.id,
                        new: new_memory.id,
                        reason: format!(
                            "High embedding similarity ({:.3}) with different content from same agent",
                            sim
                        ),
                    });
                }
            }

            // High similarity: mark older as obsolete if newer timestamp
            if sim > 0.85 && new_memory.created_at > existing.created_at {
                actions.push(InferredAction::MarkObsolete {
                    memory: existing.id,
                    superseded_by: new_memory.id,
                });
            }

            // Moderate similarity: create Related edge
            if sim > 0.6 && sim <= 0.85 {
                actions.push(InferredAction::CreateEdge {
                    source: new_memory.id,
                    target: existing.id,
                    edge_type: EdgeType::Related,
                    weight: sim,
                });
            }
        }

        // Correction type: find the most similar existing memory and supersede it
        if new_memory.memory_type == MemoryType::Correction {
            if let Some(original) = existing_memories
                .iter()
                .filter(|m| m.id != new_memory.id)
                .max_by(|a, b| {
                    cosine_similarity(&new_memory.embedding, &a.embedding)
                        .partial_cmp(&cosine_similarity(&new_memory.embedding, &b.embedding))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                let sim = cosine_similarity(&new_memory.embedding, &original.embedding);
                if sim > 0.5 {
                    actions.push(InferredAction::CreateEdge {
                        source: new_memory.id,
                        target: original.id,
                        edge_type: EdgeType::Supersedes,
                        weight: 1.0,
                    });
                    actions.push(InferredAction::UpdateConfidence {
                        memory: original.id,
                        new_confidence: (original.confidence * 0.5).max(0.1),
                    });
                }
            }
        }

        actions
    }
}

impl Default for WriteInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::memory::MemoryType;

    fn make_memory(content: &str, embedding: Vec<f32>, mem_type: MemoryType) -> MemoryNode {
        let mut m = MemoryNode::new(
            uuid::Uuid::new_v4(),
            mem_type,
            content.to_string(),
            embedding,
        );
        m.created_at = 1000;
        m
    }

    #[test]
    fn test_flag_contradiction() {
        let agent = uuid::Uuid::new_v4();
        let mut existing = make_memory("uses PostgreSQL", vec![1.0, 0.0, 0.0], MemoryType::Semantic);
        existing.agent_id = agent;

        let mut new_mem = make_memory("uses MySQL", vec![0.99, 0.01, 0.0], MemoryType::Semantic);
        new_mem.agent_id = agent;
        new_mem.created_at = 2000;

        let engine = WriteInferenceEngine::new();
        let actions = engine.infer_on_write(&new_mem, &[existing], &[]);
        assert!(
            actions.iter().any(|a| matches!(a, InferredAction::FlagContradiction { .. })),
            "Expected FlagContradiction, got: {:?}",
            actions
        );
    }

    #[test]
    fn test_moderate_similarity_creates_edge() {
        let existing = make_memory("topic A", vec![1.0, 0.0, 0.0], MemoryType::Semantic);
        // ~0.7 similarity
        let new_mem = make_memory("topic B", vec![0.7, 0.714, 0.0], MemoryType::Semantic);

        let engine = WriteInferenceEngine::new();
        let actions = engine.infer_on_write(&new_mem, &[existing], &[]);
        assert!(
            actions.iter().any(|a| matches!(a, InferredAction::CreateEdge { edge_type: EdgeType::Related, .. })),
            "Expected CreateEdge Related, got: {:?}",
            actions
        );
    }
}
