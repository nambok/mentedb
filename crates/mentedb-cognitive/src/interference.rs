use mentedb_core::MemoryNode;
use mentedb_core::types::{MemoryId};

#[derive(Debug, Clone)]
pub struct InterferencePair {
    pub memory_a: MemoryId,
    pub memory_b: MemoryId,
    pub similarity: f32,
    pub disambiguation: String,
}

pub struct InterferenceDetector {
    similarity_threshold: f32,
    truncation_length: usize,
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
    if denom == 0.0 { 0.0 } else { dot / denom }
}

fn truncate_content(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..s.floor_char_boundary(max_len)]
    }
}

impl InterferenceDetector {
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold,
            truncation_length: 80,
        }
    }

    pub fn with_truncation_length(mut self, truncation_length: usize) -> Self {
        self.truncation_length = truncation_length;
        self
    }

    pub fn detect_interference(&self, memories: &[MemoryNode]) -> Vec<InterferencePair> {
        let mut pairs = Vec::new();
        for i in 0..memories.len() {
            for j in (i + 1)..memories.len() {
                let sim = cosine_similarity(&memories[i].embedding, &memories[j].embedding);
                if sim > self.similarity_threshold && memories[i].content != memories[j].content {
                    pairs.push(InterferencePair {
                        memory_a: memories[i].id,
                        memory_b: memories[j].id,
                        similarity: sim,
                        disambiguation: self.generate_disambiguation(&memories[i], &memories[j]),
                    });
                }
            }
        }
        pairs
    }

    pub fn generate_disambiguation(&self, a: &MemoryNode, b: &MemoryNode) -> String {
        let a_content = truncate_content(&a.content, self.truncation_length);
        let b_content = truncate_content(&b.content, self.truncation_length);
        format!(
            "Note: Memory A: \"{}\" (created {}), Memory B: \"{}\" (created {}). Do not confuse.",
            a_content, a.created_at, b_content, b.created_at
        )
    }

    /// Reorder memories so interference pairs are never adjacent
    pub fn arrange_with_separation(
        memories: Vec<MemoryId>,
        pairs: &[InterferencePair],
    ) -> Vec<MemoryId> {
        if memories.is_empty() || pairs.is_empty() {
            return memories;
        }

        let conflicts: ahash::AHashSet<(MemoryId, MemoryId)> = pairs
            .iter()
            .flat_map(|p| [(p.memory_a, p.memory_b), (p.memory_b, p.memory_a)])
            .collect();

        let mut result: Vec<MemoryId> = Vec::with_capacity(memories.len());
        let mut remaining: std::collections::VecDeque<MemoryId> = memories.into();

        // Greedy: pick the first non-conflicting memory from remaining
        while let Some(first) = remaining.pop_front() {
            if result.is_empty() {
                result.push(first);
                continue;
            }

            let last = *result.last().unwrap();
            if !conflicts.contains(&(last, first)) {
                result.push(first);
            } else {
                // Find a non-conflicting one
                let mut found = false;
                for i in 0..remaining.len() {
                    if !conflicts.contains(&(last, remaining[i])) {
                        let item = remaining.remove(i).unwrap();
                        result.push(item);
                        remaining.push_front(first); // put conflicting back
                        found = true;
                        break;
                    }
                }
                if !found {
                    // No choice, just place it
                    result.push(first);
                }
            }
        }

        result
    }
}

impl Default for InterferenceDetector {
    fn default() -> Self {
        Self::new(0.8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::memory::MemoryType;

    fn make_memory(content: &str, embedding: Vec<f32>) -> MemoryNode {
        MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            content.to_string(),
            embedding,
        )
    }

    #[test]
    fn test_detect_interference() {
        let a = make_memory("Project Alpha uses React", vec![1.0, 0.0, 0.0]);
        let b = make_memory("Project Beta uses Vue", vec![0.99, 0.1, 0.0]);
        let c = make_memory("Cooking recipe for pasta", vec![0.0, 0.0, 1.0]);

        let detector = InterferenceDetector::default();
        let pairs = detector.detect_interference(&[a, b, c]);
        // a and b should interfere (high similarity, different content)
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].disambiguation.contains("Do not confuse"));
    }

    #[test]
    fn test_arrange_separation() {
        let ids: Vec<MemoryId> = (0..4).map(|_| AgentId::new()).collect();
        let pairs = vec![InterferencePair {
            memory_a: ids[0],
            memory_b: ids[1],
            similarity: 0.9,
            disambiguation: String::new(),
        }];

        let arranged = InterferenceDetector::arrange_with_separation(ids.clone(), &pairs);
        // ids[0] and ids[1] should not be adjacent
        for w in arranged.windows(2) {
            assert!(
                !(w[0] == ids[0] && w[1] == ids[1]) && !(w[0] == ids[1] && w[1] == ids[0]),
                "Interference pair should not be adjacent"
            );
        }
    }
}
