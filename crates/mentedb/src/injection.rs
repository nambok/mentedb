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
use mentedb_core::types::{AgentId, MemoryId, UserId};

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
    /// Shown at least this often with zero uses = excluded from injection
    /// entirely (still reachable through ordinary recall). Demotion halves a
    /// score; this is the terminal stage for memories the model never draws on.
    pub demotion_drop_shown: i64,
    /// Head size exempt from the semantic tail gate. The top RRF ranks are
    /// trusted as-is because a pure keyword hit (exact identifier, phone
    /// number) can carry a low cosine yet be exactly right.
    pub tail_min_keep: usize,
    /// Beyond the head, a candidate must reach this fraction of the best
    /// query cosine among candidates or it is dropped. Relative to the query's
    /// own score distribution, never an absolute cosine, so it holds across
    /// embedders. This is what keeps a weak tail silent instead of letting
    /// selection fill its quota with plausible-but-irrelevant memories.
    pub tail_floor_fraction: f32,
    /// Whether episodic memories tagged to another project may inject when a
    /// current project is set. Episodic turns are working context of the
    /// project they happened in; injecting them elsewhere is the classic
    /// cross-context contamination users disable memory over.
    pub cross_project_episodic: bool,
    /// A non-episodic memory tagged to another project must reach this
    /// fraction of the best query cosine to inject cross-project. Facts and
    /// preferences may travel, but only when they are among the most relevant
    /// candidates for the query, not as tail filler.
    pub cross_project_floor_fraction: f32,
    /// Reply similarity above which a shown memory counts as used.
    pub used_similarity: f32,
    /// Salience reinforcement applied when a memory is actually used.
    pub use_reinforcement: f32,
    /// Associative recall: how many 1-hop graph neighbors of the top vector hits
    /// to fold into the candidate pool, so a memory linked to a strong hit can
    /// surface even when it is not itself vector-similar to the query. 0 (the
    /// default) disables expansion, leaving pure vector/hybrid recall unchanged.
    pub graph_expansion_max: usize,
    /// Score a neighbor inherits from the hit it was reached through, as a
    /// fraction of that hit's score times the edge weight. Keeps neighbors below
    /// direct hits so they only win a slot when there is room after them.
    pub graph_expansion_decay: f32,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            candidate_pool: 48,
            mmr_lambda: 0.7,
            knee_gap_ratio: 2.0,
            demotion_shown_min: 5,
            demotion_factor: 0.5,
            demotion_drop_shown: 12,
            tail_min_keep: 3,
            tail_floor_fraction: 0.55,
            cross_project_episodic: false,
            cross_project_floor_fraction: 0.8,
            used_similarity: 0.6,
            use_reinforcement: 0.05,
            graph_expansion_max: 0,
            graph_expansion_decay: 0.5,
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
    /// Restrict recall to this user's memories plus shared (nil owned)
    /// knowledge, orthogonal to `agent_id`. None recalls globally on the user
    /// axis. A memory is injectable only when visible on BOTH axes.
    pub user_id: Option<UserId>,
    /// Current project (a `scope:project:<name>` value without the prefix). When
    /// set, memories from other projects are weighted down in recall so injection
    /// favors the active project instead of dumping cross-project noise. None
    /// recalls across all projects.
    pub current_project: Option<&'a str>,
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

/// The project a memory is scoped to, from its `scope:project:<name>` tag.
fn project_of(node: &MemoryNode) -> Option<&str> {
    node.tags
        .iter()
        .find_map(|t| t.strip_prefix("scope:project:"))
}

/// Cross-project policy for a candidate when a current project is set.
/// Untagged (global) memories and same-project memories always pass. Episodic
/// memories from another project pass only if configured to. Other types from
/// another project must be among the most semantically relevant candidates
/// (cosine relative to the query's best), never tail filler.
fn cross_project_allowed(
    node: &MemoryNode,
    cos: f32,
    best_cos: f32,
    current_project: Option<&str>,
    cfg: &InjectionConfig,
) -> bool {
    let Some(current) = current_project else {
        return true;
    };
    let Some(mem_project) = project_of(node) else {
        return true;
    };
    if mem_project == current {
        return true;
    }
    if node.memory_type == MemoryType::Episodic {
        return cfg.cross_project_episodic;
    }
    cos >= best_cos * cfg.cross_project_floor_fraction
}

/// Semantic tail gate over a descending-ranked candidate list: the head
/// (`tail_min_keep`) is trusted as ranked, the tail must reach
/// `tail_floor_fraction` of the best query cosine to stay. Operates on
/// (index, cosine) so RRF rank compression does not blunt it. Degenerate
/// inputs (best cosine 0, e.g. mid-migration dimension mismatch) pass
/// everything rather than silencing injection entirely.
fn passes_tail_gate(idx: usize, cos: f32, best_cos: f32, cfg: &InjectionConfig) -> bool {
    if idx < cfg.tail_min_keep {
        return true;
    }
    cos >= best_cos * cfg.tail_floor_fraction
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
                query.user_id,
                query.current_project,
            )
            .unwrap_or_default();

        let mut candidates: Vec<(MemoryNode, f32, f32)> = Vec::new();
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
            // usage or decay resolves them; past the drop threshold they are
            // excluded from injection outright (ordinary recall still finds
            // them). Exposure without a single use is the strongest negative
            // signal the system has.
            let shown = attr_count(&node, ATTR_INJECTION_SHOWN);
            let used = attr_count(&node, ATTR_INJECTION_USED);
            if shown >= cfg.demotion_drop_shown && used == 0 {
                continue;
            }
            let adjusted = if shown >= cfg.demotion_shown_min && used == 0 {
                score * cfg.demotion_factor
            } else {
                score
            };
            let cos = cosine_similarity(query.embedding, &node.embedding);
            candidates.push((node, adjusted, cos));
        }

        // Gates are relative to the query's own best cosine, never an absolute
        // threshold, so they hold across embedders (RRF scores are rank
        // compressed and useless for this; the raw cosine is not).
        let best_cos = candidates.iter().map(|(_, _, c)| *c).fold(0.0f32, f32::max);
        candidates.retain(|(node, _, cos)| {
            cross_project_allowed(node, *cos, best_cos, query.current_project, &cfg)
        });

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let scores: Vec<f32> = candidates.iter().map(|(_, s, _)| *s).collect();
        candidates.truncate(knee_cutoff(&scores, cfg.knee_gap_ratio));

        // Semantic tail gate: beyond the trusted head, weak-cosine candidates
        // vanish instead of being carried to MMR, which would otherwise fill
        // max_items however irrelevant the tail. Silence beats plausible noise.
        let mut idx = 0usize;
        candidates.retain(|(_, _, cos)| {
            let keep = passes_tail_gate(idx, *cos, best_cos, &cfg);
            idx += 1;
            keep
        });
        let mut scored: Vec<(MemoryNode, f32)> =
            candidates.into_iter().map(|(n, s, _)| (n, s)).collect();

        // Associative recall: after the vector-tail knee, fold in 1-hop graph
        // neighbors of the surviving hits, so a memory linked to a relevant hit
        // can surface even when it is not itself vector-similar to the query.
        // Runs after the knee (which is for weak vector tails, a different signal
        // than an explicit edge) so a neighbor's decayed score is not mistaken
        // for a tail and cut. Disabled by default (graph_expansion_max == 0),
        // leaving pure vector/hybrid recall intact. Neighbors inherit a decayed
        // share of the hit's score so they never outrank direct hits, pass the
        // same gates, and are bounded by graph_expansion_max; MMR then picks the
        // final set within max_items.
        if cfg.graph_expansion_max > 0 && !scored.is_empty() {
            let now = now_us();
            let mut present: HashSet<MemoryId> = scored.iter().map(|(n, _)| n.id).collect();
            present.extend(excluded.iter().copied());
            let seeds: Vec<(MemoryId, f32)> = scored.iter().map(|(n, s)| (n.id, *s)).collect();

            let g = self.graph.graph();
            let mut added = 0usize;
            'seeds: for (seed_id, seed_score) in seeds {
                let mut edges = g.outgoing(seed_id);
                edges.extend(g.incoming(seed_id));
                for (nbr_id, edge) in edges {
                    if added >= cfg.graph_expansion_max {
                        break 'seeds;
                    }
                    if present.contains(&nbr_id) {
                        continue;
                    }
                    let Ok(node) = self.get_memory(nbr_id) else {
                        continue;
                    };
                    if has_tag(&node, "action")
                        || has_tag(&node, "scope:always")
                        || has_tag(&node, "ghost-memory")
                        || !node.is_valid_at(now)
                        || !crate::agent_visible(node.agent_id, query.agent_id)
                        || !crate::user_visible(node.user_id, query.user_id)
                    {
                        continue;
                    }
                    if let Some(ref st) = session_tag
                        && has_tag(&node, st)
                    {
                        continue;
                    }
                    let score = seed_score * edge.weight * cfg.graph_expansion_decay;
                    present.insert(nbr_id);
                    scored.push((node, score));
                    added += 1;
                }
            }
        }

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
        // re-delivery. Fetched from the tag index (scope:always is capped
        // near-zero) so this loads only the handful of matches; the old path
        // scanned every page on every injection call, which thrashed the
        // buffer pool against disk on large accounts.
        let mut result: Vec<InjectionCandidate> = Vec::new();
        for mid in self.index.bitmap.query_tag("scope:always") {
            if excluded.contains(&mid) {
                continue;
            }
            if let Ok(node) = self.get_memory(mid)
                && crate::agent_visible(node.agent_id, query.agent_id)
                && crate::user_visible(node.user_id, query.user_id)
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

            // Retrieval reinforcement: being surfaced at all is an access. Refresh
            // the decay clock and bump the access count for every shown memory, not
            // only the ones the reply echoed, so anything that keeps getting
            // recalled stays alive instead of decaying to the forget threshold while
            // it is actively in use. Memories that are never recalled still get no
            // touch here, so they decay and are forgotten as intended; only what
            // retrieval keeps surfacing survives.
            node.access_count = node.access_count.saturating_add(1);
            node.accessed_at = now;

            let used = reply_embedding
                .map(|re| cosine_similarity(&node.embedding, re) >= cfg.used_similarity)
                .unwrap_or(false);
            if used {
                // The reply actually drew on it: a stronger signal than mere
                // exposure, so add a salience boost on top of the access refresh.
                bump_attr(&mut node, ATTR_INJECTION_USED);
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

    fn node_with_tags(memory_type: MemoryType, tags: &[&str]) -> MemoryNode {
        let mut node = MemoryNode::new(AgentId(Uuid::nil()), memory_type, "x".into(), vec![0.0; 4]);
        node.tags = tags.iter().map(|t| t.to_string()).collect();
        node
    }

    #[test]
    fn tail_gate_trusts_head_and_prunes_weak_tail() {
        let cfg = InjectionConfig::default();
        let best = 0.8;
        // Head positions pass regardless of cosine (keyword hits).
        assert!(passes_tail_gate(0, 0.05, best, &cfg));
        assert!(passes_tail_gate(2, 0.05, best, &cfg));
        // Tail passes only near the query's own best cosine.
        assert!(passes_tail_gate(3, 0.6, best, &cfg));
        assert!(!passes_tail_gate(3, 0.3, best, &cfg));
        // Degenerate best cosine (mid-migration): everything passes.
        assert!(passes_tail_gate(5, 0.0, 0.0, &cfg));
    }

    #[test]
    fn cross_project_policy() {
        let cfg = InjectionConfig::default();
        let best = 0.8;
        let foreign_epi = node_with_tags(MemoryType::Episodic, &["scope:project:other"]);
        let foreign_sem = node_with_tags(MemoryType::Semantic, &["scope:project:other"]);
        let same_epi = node_with_tags(MemoryType::Episodic, &["scope:project:mine"]);
        let global_sem = node_with_tags(MemoryType::Semantic, &[]);

        // No current project: everything passes.
        assert!(cross_project_allowed(&foreign_epi, 0.1, best, None, &cfg));
        // Same project and untagged always pass.
        assert!(cross_project_allowed(
            &same_epi,
            0.1,
            best,
            Some("mine"),
            &cfg
        ));
        assert!(cross_project_allowed(
            &global_sem,
            0.1,
            best,
            Some("mine"),
            &cfg
        ));
        // Foreign episodic never passes by default.
        assert!(!cross_project_allowed(
            &foreign_epi,
            0.79,
            best,
            Some("mine"),
            &cfg
        ));
        // Foreign semantic passes only when among the most relevant.
        assert!(cross_project_allowed(
            &foreign_sem,
            0.7,
            best,
            Some("mine"),
            &cfg
        ));
        assert!(!cross_project_allowed(
            &foreign_sem,
            0.5,
            best,
            Some("mine"),
            &cfg
        ));
    }

    #[test]
    fn project_of_reads_scope_tag() {
        let node = node_with_tags(MemoryType::Semantic, &["session:s1", "scope:project:apex"]);
        assert_eq!(project_of(&node), Some("apex"));
        let untagged = node_with_tags(MemoryType::Semantic, &["session:s1"]);
        assert_eq!(project_of(&untagged), None);
    }
}
