use mentedb_core::MemoryNode;
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::MemoryId;

use crate::llm::{
    CognitiveLlmService, ContradictionVerdict, InvalidationVerdict, LlmJudge, MemorySummary,
};

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
    /// Set valid_until on the old memory instead of deleting it.
    InvalidateMemory {
        memory: MemoryId,
        superseded_by: MemoryId,
        valid_until: u64,
    },
    /// A byte-identical re-save: invalidate the duplicate and link it to the
    /// surviving copy with a `Derived` edge (deduplication lineage), NOT a
    /// `Supersedes` edge. A Supersedes edge here reads as a memory superseding an
    /// exact copy of itself, which only shows up as noise in the conflict view.
    DeduplicateExact {
        duplicate: MemoryId,
        keeper: MemoryId,
    },
    /// Update the content of an existing memory with merged information.
    UpdateContent {
        memory: MemoryId,
        new_content: String,
        reason: String,
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
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Configuration for write-time inference thresholds.
#[derive(Debug, Clone)]
pub struct WriteInferenceConfig {
    /// Similarity above which two memories may contradict (default: 0.95).
    pub contradiction_threshold: f32,
    /// Similarity above which an older memory is marked obsolete (default: 0.85).
    pub obsolete_threshold: f32,
    /// Minimum similarity for creating a Related edge (default: 0.6).
    pub related_min: f32,
    /// Maximum similarity for creating a Related edge (default: 0.85).
    pub related_max: f32,
    /// Minimum similarity for a Correction to supersede (default: 0.5).
    pub correction_threshold: f32,
    /// Multiplier applied to original confidence on correction (default: 0.5).
    pub confidence_decay_factor: f32,
    /// Minimum confidence after decay (default: 0.1).
    pub confidence_floor: f32,
    /// Whether the value-update rule runs: a new memory that shares an older
    /// same-owner memory's sentence frame (long common token prefix) but ends
    /// in a different value supersedes it, so "favorite coffee is a cortado"
    /// followed by "favorite coffee is a flat white" keeps only the correction.
    /// This is the cheap, LLM-free slice of contradiction handling; unlike bare
    /// cosine (which cannot tell paraphrase from contradiction and was removed
    /// at ~0% precision), the shared-frame test only fires on the update shape.
    pub value_update_enabled: bool,
    /// Minimum embedding similarity for the value-update rule (default: 0.88).
    pub value_update_min_similarity: f32,
    /// Minimum fraction of the longer memory's tokens that must be a shared
    /// prefix for the pair to count as the same sentence frame (default: 0.6).
    pub value_update_prefix_share: f32,
    /// Maximum differing tail tokens for a value update (default: 4). Larger
    /// tails mean genuinely different statements, not a value change.
    pub value_update_max_tail: usize,
}

impl Default for WriteInferenceConfig {
    fn default() -> Self {
        Self {
            contradiction_threshold: 0.95,
            obsolete_threshold: 0.85,
            related_min: 0.6,
            related_max: 0.85,
            correction_threshold: 0.5,
            confidence_decay_factor: 0.5,
            confidence_floor: 0.1,
            value_update_enabled: true,
            value_update_min_similarity: 0.88,
            value_update_prefix_share: 0.6,
            value_update_max_tail: 4,
        }
    }
}

/// Lowercased alphanumeric tokens of a memory's content, the unit of the
/// value-update frame comparison.
fn frame_tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// True when `new` and `old` share the same sentence frame but end in a
/// different value: a long common token prefix with a short, non-identical
/// tail. "the user's favorite coffee order is a cortado" vs "... is a flat
/// white" fires; "my dog is allergic to chicken" vs "my cat is allergic to
/// chicken" does not (the difference is at the head, a different subject).
fn is_value_update(new: &str, old: &str, prefix_share: f32, max_tail: usize) -> bool {
    let nt = frame_tokens(new);
    let ot = frame_tokens(old);
    if nt.is_empty() || ot.is_empty() || nt == ot {
        return false;
    }
    let prefix = nt.iter().zip(ot.iter()).take_while(|(a, b)| a == b).count();
    let longest = nt.len().max(ot.len());
    let tail = longest - prefix;
    prefix as f32 / longest as f32 >= prefix_share && tail > 0 && tail <= max_tail
}

pub struct WriteInferenceEngine {
    config: WriteInferenceConfig,
}

impl WriteInferenceEngine {
    pub fn new() -> Self {
        Self {
            config: WriteInferenceConfig::default(),
        }
    }

    pub fn with_config(config: WriteInferenceConfig) -> Self {
        Self { config }
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

            // Cosine similarity alone cannot tell a reworded duplicate from a
            // genuine contradiction: paraphrases ("uses Postgres" / "uses
            // PostgreSQL") and opposites ("prefers tabs" / "prefers spaces")
            // both land in the high-similarity band. So the cheap write-time
            // heuristic makes only the two judgments similarity CAN support:
            //   1. byte-identical content -> a true duplicate, supersede it
            //   2. moderate similarity    -> a Related edge
            // Every semantic decision (merge, paraphrase dedup, real
            // contradiction, which-fact-wins supersession) is deferred to the
            // LLM paths (`consolidate_memories`, `detect_conflicts_with_llm`),
            // which read the actual text. Flagging contradictions from bare
            // similarity here produced ~0% precision (every reworded re-save
            // looked like a contradiction), so it was removed.
            if existing.content == new_memory.content
                && sim > self.config.obsolete_threshold
                && new_memory.created_at > existing.created_at
            {
                actions.push(InferredAction::DeduplicateExact {
                    duplicate: existing.id,
                    keeper: new_memory.id,
                });
            } else if self.config.value_update_enabled
                && sim >= self.config.value_update_min_similarity
                && new_memory.created_at > existing.created_at
                && existing.agent_id == new_memory.agent_id
                && existing.user_id == new_memory.user_id
                && is_value_update(
                    &new_memory.content,
                    &existing.content,
                    self.config.value_update_prefix_share,
                    self.config.value_update_max_tail,
                )
            {
                // Same sentence frame, new value: the newer memory supersedes
                // the older one, so recall serves only the corrected fact.
                actions.push(InferredAction::CreateEdge {
                    source: new_memory.id,
                    target: existing.id,
                    edge_type: EdgeType::Supersedes,
                    weight: sim,
                });
                actions.push(InferredAction::UpdateConfidence {
                    memory: existing.id,
                    new_confidence: (existing.confidence * self.config.confidence_decay_factor)
                        .max(self.config.confidence_floor),
                });
            } else if sim > self.config.related_min && sim <= self.config.related_max {
                actions.push(InferredAction::CreateEdge {
                    source: new_memory.id,
                    target: existing.id,
                    edge_type: EdgeType::Related,
                    weight: sim,
                });
            }
        }

        // Correction type: find the most similar existing memory and supersede it
        if new_memory.memory_type == MemoryType::Correction
            && let Some(original) = existing_memories
                .iter()
                .filter(|m| m.id != new_memory.id)
                .max_by(|a, b| {
                    cosine_similarity(&new_memory.embedding, &a.embedding)
                        .partial_cmp(&cosine_similarity(&new_memory.embedding, &b.embedding))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        {
            let sim = cosine_similarity(&new_memory.embedding, &original.embedding);
            if sim > self.config.correction_threshold {
                actions.push(InferredAction::CreateEdge {
                    source: new_memory.id,
                    target: original.id,
                    edge_type: EdgeType::Supersedes,
                    weight: 1.0,
                });
                actions.push(InferredAction::UpdateConfidence {
                    memory: original.id,
                    new_confidence: (original.confidence * self.config.confidence_decay_factor)
                        .max(self.config.confidence_floor),
                });
            }
        }

        actions
    }

    /// LLM enhanced write inference. Uses the CognitiveLlmService when available
    /// to make smarter invalidation and contradiction decisions. Falls back to
    /// cosine similarity heuristics for moderate similarity (Related edges) and
    /// correction handling.
    pub async fn infer_on_write_with_llm<J: LlmJudge>(
        &self,
        new_memory: &MemoryNode,
        existing_memories: &[MemoryNode],
        existing_edges: &[(MemoryId, MemoryId, EdgeType)],
        llm: &CognitiveLlmService<J>,
    ) -> Vec<InferredAction> {
        let _ = existing_edges;
        let mut actions = Vec::new();

        let new_summary = memory_to_summary(new_memory);

        for existing in existing_memories {
            if existing.id == new_memory.id {
                continue;
            }

            let sim = cosine_similarity(&new_memory.embedding, &existing.embedding);

            // Only consult LLM for memories with moderate+ similarity (> 0.5)
            // to avoid burning tokens on completely unrelated pairs
            if sim > 0.5 && existing.agent_id == new_memory.agent_id {
                let old_summary = memory_to_summary(existing);

                // LLM invalidation check
                if let Ok(verdict) = llm.judge_invalidation(&old_summary, &new_summary).await {
                    match verdict {
                        InvalidationVerdict::Invalidate { reason: _ } => {
                            // InvalidateMemory sets valid_until and creates the
                            // Supersedes edge; no separate MarkObsolete/CreateEdge.
                            actions.push(InferredAction::InvalidateMemory {
                                memory: existing.id,
                                superseded_by: new_memory.id,
                                valid_until: new_memory.created_at,
                            });
                            actions.push(InferredAction::UpdateConfidence {
                                memory: existing.id,
                                new_confidence: (existing.confidence
                                    * self.config.confidence_decay_factor)
                                    .max(self.config.confidence_floor),
                            });
                            // Skip contradiction check since we already know the relationship
                            continue;
                        }
                        InvalidationVerdict::Update {
                            merged_content,
                            reason,
                        } => {
                            actions.push(InferredAction::UpdateContent {
                                memory: existing.id,
                                new_content: merged_content,
                                reason,
                            });
                            continue;
                        }
                        InvalidationVerdict::Keep { .. } => {
                            // Fall through to contradiction check
                        }
                    }
                }

                // LLM contradiction check for high similarity pairs that weren't invalidated
                if sim > 0.7
                    && existing.content != new_memory.content
                    && let Ok(verdict) = llm.detect_contradiction(&old_summary, &new_summary).await
                {
                    match verdict {
                        ContradictionVerdict::Contradicts { reason } => {
                            actions.push(InferredAction::FlagContradiction {
                                existing: existing.id,
                                new: new_memory.id,
                                reason,
                            });
                        }
                        ContradictionVerdict::Supersedes { winner, reason: _ } => {
                            let winner_is_new = winner == new_memory.id.to_string();
                            let (obsolete, superseder) = if winner_is_new {
                                (existing.id, new_memory.id)
                            } else {
                                (new_memory.id, existing.id)
                            };
                            actions.push(InferredAction::InvalidateMemory {
                                memory: obsolete,
                                superseded_by: superseder,
                                valid_until: new_memory.created_at,
                            });
                        }
                        ContradictionVerdict::Compatible { .. } => {}
                    }
                }
            }

            // Moderate similarity: create Related edge (heuristic, no LLM needed)
            if sim > self.config.related_min && sim <= self.config.related_max {
                actions.push(InferredAction::CreateEdge {
                    source: new_memory.id,
                    target: existing.id,
                    edge_type: EdgeType::Related,
                    weight: sim,
                });
            }
        }

        // Correction type still uses heuristic (find most similar and supersede)
        if new_memory.memory_type == MemoryType::Correction
            && let Some(original) = existing_memories
                .iter()
                .filter(|m| m.id != new_memory.id)
                .max_by(|a, b| {
                    cosine_similarity(&new_memory.embedding, &a.embedding)
                        .partial_cmp(&cosine_similarity(&new_memory.embedding, &b.embedding))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        {
            let sim = cosine_similarity(&new_memory.embedding, &original.embedding);
            if sim > self.config.correction_threshold {
                actions.push(InferredAction::InvalidateMemory {
                    memory: original.id,
                    superseded_by: new_memory.id,
                    valid_until: new_memory.created_at,
                });
                actions.push(InferredAction::UpdateConfidence {
                    memory: original.id,
                    new_confidence: (original.confidence * self.config.confidence_decay_factor)
                        .max(self.config.confidence_floor),
                });
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

fn memory_to_summary(m: &MemoryNode) -> MemorySummary {
    MemorySummary {
        id: m.id,
        content: m.content.clone(),
        memory_type: m.memory_type,
        confidence: m.confidence,
        created_at: m.created_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::memory::MemoryType;
    use mentedb_core::types::AgentId;

    use crate::llm::MockLlmJudge;

    fn make_memory(content: &str, embedding: Vec<f32>, mem_type: MemoryType) -> MemoryNode {
        let mut m = MemoryNode::new(AgentId::new(), mem_type, content.to_string(), embedding);
        m.created_at = 1000;
        m
    }

    #[test]
    fn test_heuristic_does_not_flag_contradiction_from_similarity() {
        // Two facts with high embedding similarity but different content look
        // identical to cosine similarity whether they are paraphrases or
        // opposites. The cheap heuristic must NOT guess "contradiction" here
        // (that produced ~0% precision in production); real contradiction
        // detection is the LLM path's job. High-sim different-content yields no
        // action from the heuristic.
        let agent = AgentId::new();
        let mut existing =
            make_memory("uses PostgreSQL", vec![1.0, 0.0, 0.0], MemoryType::Semantic);
        existing.agent_id = agent;

        let mut new_mem = make_memory("uses MySQL", vec![0.99, 0.01, 0.0], MemoryType::Semantic);
        new_mem.agent_id = agent;
        new_mem.created_at = 2000;

        let engine = WriteInferenceEngine::new();
        let actions = engine.infer_on_write(&new_mem, &[existing], &[]);
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, InferredAction::FlagContradiction { .. })),
            "Heuristic must not flag contradictions from bare similarity, got: {:?}",
            actions
        );
    }

    #[test]
    fn value_update_supersedes_the_old_fact() {
        // The canonical correction shape: same sentence frame, new value at the
        // tail. The newer memory must supersede the older one so recall serves
        // only the corrected fact (live bug: both coffee facts coexisted).
        let agent = AgentId::new();
        let mut existing = make_memory(
            "The user's favorite coffee order is a cortado",
            vec![1.0, 0.0, 0.0],
            MemoryType::Semantic,
        );
        existing.agent_id = agent;
        let mut new_mem = make_memory(
            "The user's favorite coffee order is a flat white",
            vec![0.98, 0.02, 0.0],
            MemoryType::Semantic,
        );
        new_mem.agent_id = agent;
        new_mem.created_at = 2000;

        let engine = WriteInferenceEngine::new();
        let actions = engine.infer_on_write(&new_mem, &[existing.clone()], &[]);
        assert!(
            actions.iter().any(|a| matches!(
                a,
                InferredAction::CreateEdge { source, target, edge_type: EdgeType::Supersedes, .. }
                    if *source == new_mem.id && *target == existing.id
            )),
            "value update must create a Supersedes edge, got: {actions:?}"
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, InferredAction::UpdateConfidence { memory, .. } if *memory == existing.id)),
            "superseded fact's confidence must decay"
        );
    }

    #[test]
    fn subject_change_is_not_a_value_update() {
        // "my dog ..." vs "my cat ..." differ at the head: two facts about two
        // subjects, both must survive.
        let agent = AgentId::new();
        let mut existing = make_memory(
            "My dog is allergic to chicken",
            vec![1.0, 0.0, 0.0],
            MemoryType::Semantic,
        );
        existing.agent_id = agent;
        let mut new_mem = make_memory(
            "My cat is allergic to chicken",
            vec![0.99, 0.01, 0.0],
            MemoryType::Semantic,
        );
        new_mem.agent_id = agent;
        new_mem.created_at = 2000;

        let engine = WriteInferenceEngine::new();
        let actions = engine.infer_on_write(&new_mem, &[existing], &[]);
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                InferredAction::CreateEdge {
                    edge_type: EdgeType::Supersedes,
                    ..
                }
            )),
            "a subject change must not supersede, got: {actions:?}"
        );
    }

    #[test]
    fn value_update_respects_owner_and_config() {
        // Cross-owner pairs never supersede, and the rule can be disabled.
        let mut existing = make_memory(
            "The user's favorite coffee order is a cortado",
            vec![1.0, 0.0, 0.0],
            MemoryType::Semantic,
        );
        existing.agent_id = AgentId::new();
        let mut new_mem = make_memory(
            "The user's favorite coffee order is a flat white",
            vec![0.98, 0.02, 0.0],
            MemoryType::Semantic,
        );
        new_mem.agent_id = AgentId::new(); // different owner
        new_mem.created_at = 2000;

        let engine = WriteInferenceEngine::new();
        let actions = engine.infer_on_write(&new_mem, &[existing.clone()], &[]);
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                InferredAction::CreateEdge {
                    edge_type: EdgeType::Supersedes,
                    ..
                }
            )),
            "cross-owner memories must never supersede each other"
        );

        // Same owner but rule disabled: no supersession either.
        let agent = AgentId::new();
        existing.agent_id = agent;
        new_mem.agent_id = agent;
        let engine_off = WriteInferenceEngine::with_config(WriteInferenceConfig {
            value_update_enabled: false,
            ..WriteInferenceConfig::default()
        });
        let actions = engine_off.infer_on_write(&new_mem, &[existing], &[]);
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                InferredAction::CreateEdge {
                    edge_type: EdgeType::Supersedes,
                    ..
                }
            )),
            "disabled rule must not supersede"
        );
    }

    #[test]
    fn is_value_update_frame_analysis() {
        // Positive: long shared prefix, short differing tail.
        assert!(is_value_update(
            "my phone number is 4321",
            "my phone number is 1234",
            0.6,
            4
        ));
        // Negative: head differs (different subject).
        assert!(!is_value_update(
            "my cat is allergic to chicken",
            "my dog is allergic to chicken",
            0.6,
            4
        ));
        // Negative: identical content is not an update.
        assert!(!is_value_update("same fact", "same fact", 0.6, 4));
        // Negative: tail too long (a genuinely different statement).
        assert!(!is_value_update(
            "the deploy runs every friday and sarah reviews it after standup",
            "the deploy runs every friday unless the release train is frozen for the quarter end audit",
            0.6,
            4
        ));
    }

    #[test]
    fn test_byte_identical_duplicate_is_deduplicated() {
        // Identical text is deduplicated (invalidate + Derived lineage), not a
        // supersession, so it never surfaces as a conflict.
        let agent = AgentId::new();
        let mut existing = make_memory(
            "Ran command: ls -la",
            vec![1.0, 0.0, 0.0],
            MemoryType::Episodic,
        );
        existing.agent_id = agent;

        let mut new_mem = make_memory(
            "Ran command: ls -la",
            vec![1.0, 0.0, 0.0],
            MemoryType::Episodic,
        );
        new_mem.agent_id = agent;
        new_mem.created_at = 2000;

        let engine = WriteInferenceEngine::new();
        let actions = engine.infer_on_write(&new_mem, &[existing], &[]);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, InferredAction::DeduplicateExact { .. })),
            "Expected identical duplicate to be deduplicated, got: {:?}",
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
            actions.iter().any(|a| matches!(
                a,
                InferredAction::CreateEdge {
                    edge_type: EdgeType::Related,
                    ..
                }
            )),
            "Expected CreateEdge Related, got: {:?}",
            actions
        );
    }

    #[tokio::test]
    async fn test_llm_invalidation_emits_temporal_actions() {
        let agent = AgentId::new();
        let mut existing = make_memory(
            "Alice works at Acme",
            vec![0.8, 0.6, 0.0],
            MemoryType::Semantic,
        );
        existing.agent_id = agent;

        let mut new_mem = make_memory(
            "Alice joined Google last week",
            vec![0.75, 0.65, 0.1],
            MemoryType::Semantic,
        );
        new_mem.agent_id = agent;
        new_mem.created_at = 2000;

        let judge =
            MockLlmJudge::new(r#"{"verdict": "invalidate", "reason": "Alice changed jobs"}"#);
        let llm = CognitiveLlmService::new(judge);
        let engine = WriteInferenceEngine::new();
        let actions = engine
            .infer_on_write_with_llm(&new_mem, &[existing.clone()], &[], &llm)
            .await;

        assert!(
            actions
                .iter()
                .any(|a| matches!(a, InferredAction::InvalidateMemory { .. })),
            "Expected InvalidateMemory from LLM verdict, got: {:?}",
            actions
        );
        // Exactly one invalidation action: InvalidateMemory both sets
        // valid_until and creates the Supersedes edge, so a MarkObsolete
        // alongside it would duplicate the edge.
        let invalidation_count = actions
            .iter()
            .filter(|a| {
                matches!(
                    a,
                    InferredAction::InvalidateMemory { .. } | InferredAction::MarkObsolete { .. }
                )
            })
            .count();
        assert_eq!(
            invalidation_count, 1,
            "Expected a single invalidation action, got: {:?}",
            actions
        );
    }

    #[tokio::test]
    async fn test_llm_update_emits_update_content() {
        let agent = AgentId::new();
        let mut existing = make_memory(
            "Project uses React",
            vec![0.8, 0.6, 0.0],
            MemoryType::Semantic,
        );
        existing.agent_id = agent;

        let mut new_mem = make_memory(
            "Project migrated from React to Vue",
            vec![0.75, 0.65, 0.1],
            MemoryType::Semantic,
        );
        new_mem.agent_id = agent;
        new_mem.created_at = 2000;

        let judge = MockLlmJudge::new(
            r#"{"verdict": "update", "merged_content": "Project migrated from React to Vue in Q2", "reason": "adds temporal context"}"#,
        );
        let llm = CognitiveLlmService::new(judge);
        let engine = WriteInferenceEngine::new();
        let actions = engine
            .infer_on_write_with_llm(&new_mem, &[existing.clone()], &[], &llm)
            .await;

        assert!(
            actions
                .iter()
                .any(|a| matches!(a, InferredAction::UpdateContent { .. })),
            "Expected UpdateContent from LLM update verdict, got: {:?}",
            actions
        );
    }

    #[tokio::test]
    async fn test_llm_keep_falls_through_to_contradiction_check() {
        let agent = AgentId::new();
        // High similarity pair where LLM says keep but content differs
        let mut existing = make_memory("Prefers tabs", vec![0.9, 0.44, 0.0], MemoryType::Semantic);
        existing.agent_id = agent;

        let mut new_mem = make_memory(
            "Prefers spaces",
            vec![0.88, 0.47, 0.0],
            MemoryType::Semantic,
        );
        new_mem.agent_id = agent;
        new_mem.created_at = 2000;

        let judge = MockLlmJudge::new(
            r#"{"verdict": "compatible", "reason": "different formatting preferences"}"#,
        );
        let llm = CognitiveLlmService::new(judge);
        let engine = WriteInferenceEngine::new();
        let actions = engine
            .infer_on_write_with_llm(&new_mem, &[existing.clone()], &[], &llm)
            .await;

        // MockLlmJudge returns the same response for all calls, so the keep
        // from invalidation falls through, then contradiction returns compatible
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, InferredAction::FlagContradiction { .. })),
            "Should not flag contradiction when LLM says compatible",
        );
    }
}
