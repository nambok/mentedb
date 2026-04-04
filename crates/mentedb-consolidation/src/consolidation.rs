use mentedb_core::memory::MemoryType;
use mentedb_core::types::{MemoryId, Timestamp};
use mentedb_core::MemoryNode;
use serde::{Deserialize, Serialize};

/// A group of memories that are candidates for consolidation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationCandidate {
    pub memories: Vec<MemoryId>,
    pub topic: String,
    pub avg_similarity: f32,
}

/// The result of consolidating a cluster of memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedMemory {
    pub summary: String,
    pub source_memories: Vec<MemoryId>,
    pub new_type: MemoryType,
    pub combined_confidence: f32,
    pub combined_embedding: Vec<f32>,
}

/// Engine that processes episodic memories and produces semantic summaries.
pub struct ConsolidationEngine;

impl ConsolidationEngine {
    pub fn new() -> Self {
        Self
    }

    /// Group memories by embedding similarity using union-find clustering.
    pub fn find_candidates(
        &self,
        memories: &[MemoryNode],
        min_cluster_size: usize,
        similarity_threshold: f32,
    ) -> Vec<ConsolidationCandidate> {
        let n = memories.len();
        if n == 0 {
            return Vec::new();
        }

        // Union-Find
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra == rb {
                return;
            }
            if rank[ra] < rank[rb] {
                parent[ra] = rb;
            } else if rank[ra] > rank[rb] {
                parent[rb] = ra;
            } else {
                parent[rb] = ra;
                rank[ra] += 1;
            }
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(&memories[i].embedding, &memories[j].embedding);
                if sim > similarity_threshold {
                    union(&mut parent, &mut rank, i, j);
                }
            }
        }

        // Group by root
        let mut clusters: ahash::HashMap<usize, Vec<usize>> = ahash::HashMap::default();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(i);
        }

        clusters
            .into_values()
            .filter(|c| c.len() >= min_cluster_size)
            .map(|indices| {
                let mem_ids: Vec<MemoryId> = indices.iter().map(|&i| memories[i].id).collect();

                // Compute average pairwise similarity
                let mut total_sim = 0.0_f32;
                let mut count = 0u32;
                for (a_idx, &i) in indices.iter().enumerate() {
                    for &j in &indices[a_idx + 1..] {
                        total_sim +=
                            cosine_similarity(&memories[i].embedding, &memories[j].embedding);
                        count += 1;
                    }
                }
                let avg_similarity = if count > 0 {
                    total_sim / count as f32
                } else {
                    1.0
                };

                // Use first memory's content start as topic
                let topic = memories[indices[0]]
                    .content
                    .split_whitespace()
                    .take(5)
                    .collect::<Vec<_>>()
                    .join(" ");

                ConsolidationCandidate {
                    memories: mem_ids,
                    topic,
                    avg_similarity,
                }
            })
            .collect()
    }

    /// Merge a cluster of memories into a single semantic memory.
    pub fn consolidate(&self, cluster: &[MemoryNode]) -> ConsolidatedMemory {
        if cluster.is_empty() {
            return ConsolidatedMemory {
                summary: String::new(),
                source_memories: Vec::new(),
                new_type: MemoryType::Semantic,
                combined_confidence: 0.0,
                combined_embedding: Vec::new(),
            };
        }

        let source_memories: Vec<MemoryId> = cluster.iter().map(|m| m.id).collect();

        // Deduplicated summary
        let mut seen_sentences: Vec<String> = Vec::new();
        for m in cluster {
            for sentence in m.content.split('.') {
                let trimmed = sentence.trim().to_lowercase();
                if !trimmed.is_empty() && !seen_sentences.contains(&trimmed) {
                    seen_sentences.push(trimmed);
                }
            }
        }
        let summary = seen_sentences.join(". ");

        // Max confidence
        let combined_confidence = cluster
            .iter()
            .map(|m| m.confidence)
            .fold(0.0_f32, f32::max);

        // Mean embedding, normalized
        let dim = cluster[0].embedding.len();
        let mut combined_embedding = vec![0.0_f32; dim];
        for m in cluster {
            if m.embedding.len() == dim {
                for (i, &v) in m.embedding.iter().enumerate() {
                    combined_embedding[i] += v;
                }
            }
        }
        let n = cluster.len() as f32;
        for v in &mut combined_embedding {
            *v /= n;
        }
        // Normalize
        let norm: f32 = combined_embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut combined_embedding {
                *v /= norm;
            }
        }

        ConsolidatedMemory {
            summary,
            source_memories,
            new_type: MemoryType::Semantic,
            combined_confidence,
            combined_embedding,
        }
    }

    /// Whether an episodic memory is ready for consolidation.
    /// Must be episodic, older than 24 hours, and accessed more than 2 times.
    pub fn should_consolidate(memory: &MemoryNode, current_time: Timestamp) -> bool {
        const DAY_US: u64 = 86_400_000_000;
        memory.memory_type == MemoryType::Episodic
            && current_time.saturating_sub(memory.created_at) > DAY_US
            && memory.access_count > 2
    }
}

impl Default for ConsolidationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_memory;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_should_consolidate() {
        let now = 200_000_000_000u64;
        let mut m = make_memory("test content", vec![1.0]);
        m.memory_type = MemoryType::Episodic;
        m.created_at = now - 100_000_000_000; // well over 24h ago
        m.access_count = 5;
        assert!(ConsolidationEngine::should_consolidate(&m, now));
    }

    #[test]
    fn test_should_not_consolidate_recent() {
        let now = 100_000_000_000u64;
        let mut m = make_memory("test", vec![1.0]);
        m.memory_type = MemoryType::Episodic;
        m.created_at = now - 1_000_000; // 1 second ago
        m.access_count = 5;
        assert!(!ConsolidationEngine::should_consolidate(&m, now));
    }

    #[test]
    fn test_find_candidates_and_consolidate() {
        let engine = ConsolidationEngine::new();
        let m1 = make_memory("topic A info", vec![1.0, 0.0, 0.0]);
        let m2 = make_memory("topic A data", vec![0.99, 0.1, 0.0]);
        let m3 = make_memory("topic B info", vec![0.0, 0.0, 1.0]);

        let candidates = engine.find_candidates(&[m1.clone(), m2.clone(), m3], 2, 0.9);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].memories.len(), 2);

        let consolidated = engine.consolidate(&[m1, m2]);
        assert_eq!(consolidated.new_type, MemoryType::Semantic);
        assert!(!consolidated.combined_embedding.is_empty());
    }
}
