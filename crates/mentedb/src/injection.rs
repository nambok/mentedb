//! Engine-native injection attention: retrieval shaped for context
//! injection rather than raw search.
//!
//! Plain top-k recall is the wrong contract for injecting memories into a
//! model's context: the most similar memories to a conversation are echoes
//! of that conversation, top-k always returns k items however weak the
//! tail, and near-duplicates (a distilled fact and the turn it came from)
//! co-rank. This module owns the selection policy that client hooks
//! previously approximated with heuristics:
//!
//! - Session provenance: memories originating from the querying session are
//!   never returned, they are in that context by construction.
//! - A working-memory ledger (`exclude_ids`) supplied by the client, holding
//!   what was already delivered within the current context lifetime.
//! - A relevance knee instead of a fixed floor: the candidate list is cut at
//!   the largest score gap, so weak tails vanish without a magic constant.
//! - Maximal Marginal Relevance over embeddings, so each selected item adds
//!   information the previous ones did not.
//! - Type quotas: distilled knowledge first, verbatim episodic capped,
//!   action notes never (they exist for distillation and session resume).
//! - User-pinned `scope:always` memories bypass every quality gate.
//! - Outcome learning: `record_injection_outcome` tracks shown vs used per
//!   memory; chronically ignored memories are demoted at selection time and
//!   used ones are reinforced.

use std::collections::HashSet;

use mentedb_core::error::MenteResult;
use mentedb_core::memory::{AttributeValue, MemoryNode, MemoryType};
use mentedb_core::types::{AgentId, MemoryId};

use crate::MenteDb;

/// Attribute key: how many times this memory was injected into a context.
pub const ATTR_INJECTION_SHOWN: &str = "injection_shown";
/// Attribute key: how many times an injected memory was drawn on by the
/// reply that followed.
pub const ATTR_INJECTION_USED: &str = "injection_used";

/// Tunables for injection attention selection.
#[derive(Debug, Clone)]
pub struct InjectionConfig {
    /// Retrieval pool fanned out before selection.
    pub candidate_pool: usize,
    /// MMR relevance weight; the remainder weighs redundancy.
    pub mmr_lambda: f32,
    /// Consecutive score ratio treated as the relevance knee.
    pub knee_gap_ratio: f32,
    /// Shown at least this often with zero uses = demoted at selection.
    pub demotion_shown_min: i64,
    /// Score multiplier applied to chronically ignored memories.
    pub demotion_factor: f32,
    /// Reply similarity above which a shown memory counts as used.
    pub used_similarity: f32,
    /// Salience reinforcement applied when a memory is actually used.
    pub use_reinforcement: f32,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            candidate_pool: 48,
            mmr_lambda: 0.7,
            knee_gap_ratio: 2.0,
            demotion_shown_min: 5,
            demotion_factor: 0.5,
            used_similarity: 0.6,
            use_reinforcement: 0.05,
        }
    }
}

/// A request for injection-ready context.
pub struct InjectionQuery<'a> {
    /// Embedding of the prompt (plus any conversational blend).
    pub embedding: &'a [f32],
    /// Raw query text for the lexical half of hybrid recall.
    pub query_text: Option<&'a str>,
    /// The session issuing the query. Memories tagged `session:<id>` with
    /// this ID are excluded: they are already in that session's context.
    pub session_id: Option<&'a str>,
    /// Memory IDs already delivered within the current context lifetime.
    pub exclude_ids: &'a [MemoryId],
    /// Maximum items returned beyond pinned memories.
    pub max_items: usize,
    /// Maximum verbatim episodic items within `max_items`.
    pub max_episodic: usize,
    /// Restrict recall to this agent's memories plus shared (nil owned)
    /// knowledge. None recalls globally.
    pub agent_id: Option<AgentId>,
}

/// Why an item was selected, for introspection and client display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionReason {
    /// User-pinned scope:always memory: bypasses every quality gate.
    Pinned,
    /// Survived the relevance knee, MMR, and quotas.
    Relevant,
}

/// One injection-ready memory with its selection metadata.
pub struct InjectionCandidate {
    pub node: MemoryNode,
    /// Retrieval score after demotion adjustment (0.0 for pinned items,
    /// which are not score-ranked).
    pub score: f32,
    pub reason: SelectionReason,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

fn attr_count(node: &MemoryNode, key: &str) -> i64 {
    match node.attributes.get(key) {
        Some(AttributeValue::Integer(n)) => *n,
        _ => 0,
    }
}

fn bump_attr(node: &mut MemoryNode, key: &str) {
    let next = attr_count(node, key) + 1;
    node.attributes
        .insert(key.to_string(), AttributeValue::Integer(next));
}

fn has_tag(node: &MemoryNode, tag: &str) -> bool {
    node.tags.iter().any(|t| t == tag)
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// Cut a descending score list at its largest relative gap (the relevance
/// knee), when that gap is decisive. Returns how many items to keep.
fn knee_cutoff(scores: &[f32], gap_ratio: f32) -> usize {
    if scores.len() <= 1 {
        return scores.len();
    }
    let mut best_ratio = 0.0f32;
    let mut cut = scores.len();
    for i in 0..scores.len() - 1 {
        let hi = scores[i];
        let lo = scores[i + 1].max(f32::EPSILON);
        let ratio = hi / lo;
        if ratio > best_ratio {
            best_ratio = ratio;
            cut = i + 1;
        }
    }
    if best_ratio >= gap_ratio {
        cut
    } else {
        scores.len()
    }
}

impl MenteDb {
    /// Injection-ready context selection. See the module docs for the
    /// policy; this is the API client hooks should prefer over raw recall.
    pub fn recall_for_injection(
        &self,
        query: &InjectionQuery<'_>,
    ) -> MenteResult<Vec<InjectionCandidate>> {
        let cfg = self.cognitive_config.injection_config.clone();
        let excluded: HashSet<MemoryId> = query.exclude_ids.iter().copied().collect();
        let session_tag = query.session_id.map(|s| format!("session:{s}"));

        // Candidate pool from hybrid recall.
        let hits = self
            .recall_hybrid_scoped_at_mode(
                query.embedding,
                query.query_text,
                cfg.candidate_pool,
                now_us(),
                None,
                false,
                None,
                query.agent_id,
            )
            .unwrap_or_default();

        let mut scored: Vec<(MemoryNode, f32)> = Vec::new();
        for (id, score) in hits {
            if excluded.contains(&id) {
                continue;
            }
            let Ok(node) = self.get_memory(id) else {
                continue;
            };
            if has_tag(&node, "action")
                || has_tag(&node, "scope:always")
                || has_tag(&node, "ghost-memory")
            {
                // Actions never inject; pinned items are handled separately.
                // Ghost memories are speculative working material for the
                // trajectory tracker, not confirmed knowledge; injecting
                // "Unconfirmed: ..." as if it were a fact misleads.
                continue;
            }
            if let Some(ref st) = session_tag
                && has_tag(&node, st)
            {
                continue;
            }
            // Chronically ignored memories fall back in the ranking until
            // usage or decay resolves them.
            let shown = attr_count(&node, ATTR_INJECTION_SHOWN);
            let used = attr_count(&node, ATTR_INJECTION_USED);
            let adjusted = if shown >= cfg.demotion_shown_min && used == 0 {
                score * cfg.demotion_factor
            } else {
                score
            };
            scored.push((node, adjusted));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let scores: Vec<f32> = scored.iter().map(|(_, s)| *s).collect();
        scored.truncate(knee_cutoff(&scores, cfg.knee_gap_ratio));

        // MMR selection with type quotas.
        let top_score = scored
            .first()
            .map(|(_, s)| *s)
            .unwrap_or(1.0)
            .max(f32::EPSILON);
        let mut remaining: Vec<(MemoryNode, f32)> = scored;
        let mut selected: Vec<InjectionCandidate> = Vec::new();
        let mut episodic_count = 0usize;

        while selected.len() < query.max_items && !remaining.is_empty() {
            let mut best_idx: Option<usize> = None;
            let mut best_value = f32::NEG_INFINITY;
            for (idx, (node, score)) in remaining.iter().enumerate() {
                if node.memory_type == MemoryType::Episodic && episodic_count >= query.max_episodic
                {
                    continue;
                }
                let redundancy = selected
                    .iter()
                    .map(|s| cosine_similarity(&node.embedding, &s.node.embedding))
                    .fold(0.0f32, f32::max);
                let value =
                    cfg.mmr_lambda * (score / top_score) - (1.0 - cfg.mmr_lambda) * redundancy;
                if value > best_value {
                    best_value = value;
                    best_idx = Some(idx);
                }
            }
            let Some(idx) = best_idx else {
                break;
            };
            let (node, score) = remaining.remove(idx);
            if node.memory_type == MemoryType::Episodic {
                episodic_count += 1;
            }
            selected.push(InjectionCandidate {
                node,
                score,
                reason: SelectionReason::Relevant,
            });
        }

        // Pinned memories bypass every gate except the ledger, and lead the
        // result. The client's ledger reset (at context loss) governs
        // re-delivery.
        let mut result: Vec<InjectionCandidate> = Vec::new();
        let page_ids: Vec<_> = {
            let pm = self.page_map.read();
            pm.values().copied().collect()
        };
        for pid in page_ids {
            if let Ok(node) = self.storage.load_memory(pid)
                && has_tag(&node, "scope:always")
                && !excluded.contains(&node.id)
                && crate::agent_visible(node.agent_id, query.agent_id)
            {
                result.push(InjectionCandidate {
                    node,
                    score: 0.0,
                    reason: SelectionReason::Pinned,
                });
            }
        }
        result.extend(selected);
        Ok(result)
    }

    /// Record the outcome of an injection: every shown memory's exposure
    /// count rises, and memories the reply actually drew on (embedding
    /// similarity against the reply) are counted as used and reinforced.
    /// Returns (shown_updated, used_count).
    pub fn record_injection_outcome(
        &self,
        shown: &[MemoryId],
        reply_embedding: Option<&[f32]>,
    ) -> MenteResult<(usize, usize)> {
        let cfg = self.cognitive_config.injection_config.clone();
        let mut updated = 0usize;
        let mut used_total = 0usize;
        let now = now_us();

        for id in shown {
            let pid = {
                let pm = self.page_map.read();
                match pm.get(id) {
                    Some(p) => *p,
                    None => continue,
                }
            };
            let Ok(mut node) = self.storage.load_memory(pid) else {
                continue;
            };
            bump_attr(&mut node, ATTR_INJECTION_SHOWN);

            let used = reply_embedding
                .map(|re| cosine_similarity(&node.embedding, re) >= cfg.used_similarity)
                .unwrap_or(false);
            if used {
                bump_attr(&mut node, ATTR_INJECTION_USED);
                node.access_count = node.access_count.saturating_add(1);
                node.accessed_at = now;
                node.salience = (node.salience + cfg.use_reinforcement).min(1.0);
                used_total += 1;
            }

            // Counter updates must not re-run write inference; write the
            // node in place.
            self.storage.update_memory(pid, &node)?;
            updated += 1;
        }
        Ok((updated, used_total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::types::AgentId;
    use uuid::Uuid;

    #[test]
    fn knee_cuts_at_largest_gap() {
        let ratio = InjectionConfig::default().knee_gap_ratio;
        assert_eq!(knee_cutoff(&[1.0, 0.9, 0.8, 0.1, 0.09], ratio), 3);
        // No decisive gap: keep everything.
        assert_eq!(knee_cutoff(&[1.0, 0.9, 0.8, 0.7], ratio), 4);
        assert_eq!(knee_cutoff(&[1.0], ratio), 1);
        assert_eq!(knee_cutoff(&[], ratio), 0);
    }

    #[test]
    fn attr_counters_roundtrip() {
        let mut node = MemoryNode::new(
            AgentId(Uuid::nil()),
            MemoryType::Semantic,
            "x".into(),
            vec![0.0; 4],
        );
        assert_eq!(attr_count(&node, ATTR_INJECTION_SHOWN), 0);
        bump_attr(&mut node, ATTR_INJECTION_SHOWN);
        bump_attr(&mut node, ATTR_INJECTION_SHOWN);
        assert_eq!(attr_count(&node, ATTR_INJECTION_SHOWN), 2);
    }
}
