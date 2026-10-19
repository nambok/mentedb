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

use std::collections::{HashMap, HashSet};

use mentedb_core::error::MenteResult;
use mentedb_core::memory::{AttributeValue, MemoryNode, MemoryType};
use mentedb_core::types::{AgentId, MemoryId, UserId};
use mentedb_storage::PageId;

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
    /// Cluster completion: when one section tag dominates the selected
    /// head (at least this fraction of it, minimum two items), the query
    /// lives inside that cluster, and the lowest ranked non cluster picks
    /// are swapped for the section's best remaining siblings. A dense
    /// cluster of sibling rules then arrives together instead of the top
    /// sibling crowding the rest out. 0 disables. Off by default until the
    /// offline sweep settles interactions with MMR and the knee and tail
    /// gates; the mechanism is pinned by tests at 0.34.
    pub cluster_dominant_fraction: f32,
    /// Cap on siblings swapped in by cluster completion.
    pub cluster_fill_max: usize,
    /// Sections larger than this never count as clusters: a catch all
    /// section holding half the file dominates any head trivially and its
    /// siblings are arbitrary, not coherent.
    pub cluster_max_span: usize,
    /// Whether injection's hybrid recall includes the lexical (BM25) leg.
    /// On rule stores the lexical leg can drown the vector signal: generic
    /// task words match everywhere and rank fusion buries the semantically
    /// right rule. Off sends vector only retrieval into the same pipeline.
    pub use_lexical_leg: bool,
    /// Score multiplier for memories owned by the querying agent. On an
    /// agent scoped query the agent's own knowledge (its ingested rules,
    /// its facts) must not be crowded out of the candidate ranking by the
    /// account's shared nil owned memories; measured on a busy account,
    /// shared memories otherwise flood the relevant slots. 1.0 disables.
    pub own_agent_boost: f32,
    /// Score multiplier for a memory tagged `escalated:violation`: a standing
    /// rule the agent was shown and then broke. Escalation exists so a rule
    /// that was violated once surfaces at the FRONT the next time its context
    /// recurs, not buried mid-list where it was ignored. Without this the
    /// escalation only bumps salience, which injection does not rank on, so the
    /// correction would not stick. 1.0 disables.
    pub escalation_boost: f32,
    /// The boost applies only to an escalated rule whose cosine is at least
    /// this fraction of the query's best cosine, so escalation reorders rules
    /// that are genuinely on topic and never drags an unrelated escalated rule
    /// into the context on the strength of the boost alone.
    pub escalation_min_relevance: f32,
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
    /// Mode rules: standing directives bound to an activity (working on
    /// code, composing a reply) carry `trigger:` tags, and a trigger is
    /// active for a turn when the turn's embedding lands within this cosine
    /// of one of that trigger's own rules. Self defining and cognitive,
    /// the rules are their own anchors, no pattern matching anywhere.
    /// Embedder coupled like every absolute gate. 0 disables the channel.
    pub mode_trigger_gate: f32,
    /// Maximum mode rules injected per turn across all active triggers.
    pub mode_rules_max: usize,
    /// Most mode probes one injection recall may load from storage on cache
    /// misses. Probes persist across reopen, so a miss is normally a rule
    /// written since the last snapshot; this caps the cost when the persisted
    /// probes are missing (first open after an upgrade, a deleted file), where
    /// a large rule set would otherwise load every rule inside one request.
    /// Rules past the cap are skipped for that request and load on later ones.
    pub mode_probe_load_budget: usize,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            candidate_pool: 48,
            cluster_dominant_fraction: 0.0,
            cluster_fill_max: 4,
            cluster_max_span: 12,
            use_lexical_leg: true,
            own_agent_boost: 1.25,
            escalation_boost: 2.5,
            escalation_min_relevance: 0.5,
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
            mode_trigger_gate: 0.40,
            mode_rules_max: 4,
            mode_probe_load_budget: 256,
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
    /// A standing directive whose activity mode is active for this turn:
    /// the turn's embedding landed within the mode gate of the rule.
    ModeRule,
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

        // Phase timings, logged when a call is slow, so a slow recall in
        // production names the phase and how many memories it loaded.
        let t_start = std::time::Instant::now();
        // Candidate pool from hybrid recall.
        let hits = self
            .recall_hybrid_scoped_at_mode(
                query.embedding,
                query.query_text.filter(|_| cfg.use_lexical_leg),
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

        let t_hybrid = t_start.elapsed();
        let hybrid_hits = hits.len();
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
                || node.tags.iter().any(|t| t.starts_with("mode-exemplar:"))
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
            let mut adjusted = if shown >= cfg.demotion_shown_min && used == 0 {
                score * cfg.demotion_factor
            } else {
                score
            };
            if let Some(agent) = query.agent_id
                && node.agent_id == agent
                && !agent.0.is_nil()
            {
                adjusted *= cfg.own_agent_boost;
            }
            let cos = cosine_similarity(query.embedding, &node.embedding);
            candidates.push((node, adjusted, cos));
        }

        // Pre gate snapshot of section tagged candidates: cluster completion
        // may need siblings the knee and tail gates are about to drop.
        let cluster_pool: Vec<(MemoryNode, f32)> = if cfg.cluster_dominant_fraction > 0.0 {
            candidates
                .iter()
                .filter(|(node, _, _)| node.tags.iter().any(|t| t.starts_with("section:")))
                .map(|(node, _, cos)| (node.clone(), *cos))
                .collect()
        } else {
            Vec::new()
        };

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

        // Escalation re-ranks AMONG the relevance survivors: a rule the agent
        // was shown and then broke leads the next time its context recurs. It
        // runs here, after the cosine gates, so the boost can never drag an
        // irrelevant escalated rule into the context, only reorder rules that
        // already earned their place.
        if cfg.escalation_boost != 1.0
            && candidates
                .iter()
                .any(|(n, _, _)| has_tag(n, "escalated:violation"))
        {
            let relevance_floor = best_cos * cfg.escalation_min_relevance;
            for (node, adjusted, cos) in candidates.iter_mut() {
                if has_tag(node, "escalated:violation") && *cos >= relevance_floor {
                    *adjusted *= cfg.escalation_boost;
                }
            }
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

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

        // Cluster dominance is read off the score ranking BEFORE MMR: MMR
        // deliberately de clusters the head, hiding the very signal cluster
        // completion needs.
        let dominant_section: Option<String> = if cfg.cluster_dominant_fraction > 0.0 {
            let head = scored.len().min(query.max_items.max(3));
            let mut counts: HashMap<String, usize> = HashMap::new();
            for (node, _) in &scored[..head] {
                for tag in &node.tags {
                    if let Some(sec) = tag.strip_prefix("section:") {
                        *counts.entry(sec.to_string()).or_insert(0) += 1;
                    }
                }
            }
            // A section wider than cluster_max_span across the whole pool is
            // a catch all, not a coherent cluster; it dominates any head
            // trivially and its siblings are arbitrary.
            let mut span: HashMap<&str, usize> = HashMap::new();
            for (node, _) in &cluster_pool {
                for tag in &node.tags {
                    if let Some(sec) = tag.strip_prefix("section:") {
                        *span.entry(sec).or_insert(0) += 1;
                    }
                }
            }
            let need = ((cfg.cluster_dominant_fraction * head as f32).ceil() as usize).max(2);
            counts
                .into_iter()
                .filter(|(sec, n)| {
                    *n >= need
                        && span.get(sec.as_str()).copied().unwrap_or(0) <= cfg.cluster_max_span
                })
                .max_by_key(|(_, n)| *n)
                .map(|(sec, _)| sec)
        } else {
            None
        };

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

        // Cluster completion: when one section dominates the selected head,
        // the turn lives inside that cluster of sibling rules, and embedding
        // rank underestimates the rest of the cluster (the harness measured
        // exactly this on a real 22KB agent file: two governing rules ranked
        // below k=24 behind their own section siblings). Swap the weakest
        // non cluster tail picks for the section's best remaining siblings,
        // gates overridden deliberately, bounded by cluster_fill_max.
        if selected.len() >= 3
            && let Some(sec) = dominant_section
        {
            {
                let sec_tag = format!("section:{sec}");
                let mut siblings: Vec<(MemoryNode, f32)> = cluster_pool
                    .into_iter()
                    .filter(|(node, _)| {
                        node.tags.iter().any(|t| t == &sec_tag)
                            && !selected.iter().any(|c| c.node.id == node.id)
                    })
                    .collect();
                siblings.sort_by(|a, b| b.1.total_cmp(&a.1));
                siblings.truncate(cfg.cluster_fill_max);
                // Evict the weakest non cluster picks, never the top pick.
                // The span cap above keeps this to coherent clusters, where
                // the cluster hypothesis says sibling relevance is
                // underestimated by embedding rank.
                let mut evict: Vec<usize> = (1..selected.len())
                    .filter(|i| !selected[*i].node.tags.iter().any(|t| t == &sec_tag))
                    .collect();
                evict.sort_by(|a, b| {
                    selected[*a]
                        .score
                        .partial_cmp(&selected[*b].score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for (pos, (node, cos)) in evict.into_iter().zip(siblings) {
                    selected[pos] = InjectionCandidate {
                        node,
                        score: cos,
                        reason: SelectionReason::Relevant,
                    };
                }
            }
        }

        // Pinned memories bypass every gate except the ledger, and lead the
        // result. The client's ledger reset (at context loss) governs
        // re-delivery. Fetched from the tag index (scope:always is capped
        // near-zero) so this loads only the handful of matches; the old path
        // scanned every page on every injection call, which thrashed the
        // buffer pool against disk on large accounts.
        let t_selected = t_start.elapsed();
        let mut pinned_loads = 0usize;
        let mut result: Vec<InjectionCandidate> = Vec::new();
        for mid in self.index.bitmap.query_tag("scope:always") {
            if excluded.contains(&mid) {
                continue;
            }
            pinned_loads += 1;
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
        // Mode rules: standing directives bound to an activity rather than a
        // topic or a tool moment. "Longterm fixes only" must fire for "fix
        // the login timeout" even though they share no words; the turn's
        // embedding landing near the rule's own stored embedding is the
        // activation, cognitive and self defining, dot products only.
        let t_pinned = t_start.elapsed();
        let mut mode_tags = 0usize;
        let mut mode_loads = 0usize;
        if cfg.mode_trigger_gate > 0.0 && !query.embedding.is_empty() {
            let now = now_us();
            let mut mode_rules: Vec<(f32, MemoryNode)> = Vec::new();
            let mut mode_candidates: Vec<(f32, MemoryId, String)> = Vec::new();
            for tag in self.index.bitmap.tags_with_prefix("trigger:") {
                mode_tags += 1;
                let trigger = tag.trim_start_matches("trigger:");
                // Activation signal: the trigger's exemplar turns when it
                // has any (model emitted examples of requests that enter
                // this mode; a turn matching a turn is the strong signal,
                // measured), otherwise the rules' own embeddings.
                let exemplar_tag = format!("mode-exemplar:{trigger}");
                let mut activation = f32::MIN;
                let exemplar_ids = self.index.bitmap.query_tag(&exemplar_tag);
                if exemplar_ids.is_empty() {
                    for mid in self.index.bitmap.query_tag(&tag) {
                        if let Some(probe) = self.mode_probe(mid, &mut mode_loads)
                            && probe.is_valid_at(now)
                            && !probe.embedding.is_empty()
                            && crate::agent_visible(probe.agent_id, query.agent_id)
                            && crate::user_visible(probe.user_id, query.user_id)
                        {
                            let sim = cosine_similarity(query.embedding, &probe.embedding);
                            activation = activation.max(sim);
                        }
                    }
                } else {
                    for mid in exemplar_ids {
                        if let Some(probe) = self.mode_probe(mid, &mut mode_loads)
                            && probe.is_valid_at(now)
                            && !probe.embedding.is_empty()
                            && crate::agent_visible(probe.agent_id, query.agent_id)
                            && crate::user_visible(probe.user_id, query.user_id)
                        {
                            let sim = cosine_similarity(query.embedding, &probe.embedding);
                            activation = activation.max(sim);
                        }
                    }
                }
                if activation < cfg.mode_trigger_gate {
                    continue;
                }
                // The mode is active: its RULES (never exemplars) become
                // candidates. Validity and owner come from the cached probes;
                // full memories are loaded below only for the few kept.
                for mid in self.index.bitmap.query_tag(&tag) {
                    if excluded.contains(&mid)
                        || result.iter().any(|c| c.node.id == mid)
                        || selected.iter().any(|c| c.node.id == mid)
                        || mode_candidates.iter().any(|(_, id, _)| *id == mid)
                    {
                        continue;
                    }
                    let Some(probe) = self.mode_probe(mid, &mut mode_loads) else {
                        continue;
                    };
                    if !probe.is_valid_at(now)
                        || !crate::agent_visible(probe.agent_id, query.agent_id)
                        || !crate::user_visible(probe.user_id, query.user_id)
                    {
                        continue;
                    }
                    mode_candidates.push((activation, mid, tag.clone()));
                }
            }
            // Strongest mode first (stable, so rules keep their index order
            // within a mode), then load only until the cap is filled. A rule
            // whose stored tags no longer carry the trigger is skipped, as
            // before.
            mode_candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (activation, mid, tag) in mode_candidates {
                if mode_rules.len() >= cfg.mode_rules_max {
                    break;
                }
                mode_loads += 1;
                let Ok(node) = self.get_memory(mid) else {
                    continue;
                };
                if node.tags.iter().any(|t| t == &tag) {
                    mode_rules.push((activation, node));
                }
            }
            for (sim, node) in mode_rules {
                result.push(InjectionCandidate {
                    node,
                    score: sim,
                    reason: SelectionReason::ModeRule,
                });
            }
        }

        result.extend(selected);
        let total = t_start.elapsed();
        if total >= std::time::Duration::from_millis(self.cognitive_config.slow_recall_log_ms) {
            tracing::info!(
                total_ms = total.as_millis() as u64,
                hybrid_ms = t_hybrid.as_millis() as u64,
                select_ms = (t_selected - t_hybrid).as_millis() as u64,
                pinned_ms = (t_pinned - t_selected).as_millis() as u64,
                mode_ms = (total - t_pinned).as_millis() as u64,
                hybrid_hits,
                pinned_loads,
                mode_tags,
                mode_loads,
                "slow injection recall"
            );
        }
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
            self.update_node(pid, &node)?;
            updated += 1;
        }
        Ok((updated, used_total))
    }
}

/// What mode activation needs from a `trigger:` rule or `mode-exemplar:` turn:
/// its embedding, validity window and owner. Cached per memory on the
/// database so activation never loads the full memory from storage.
pub(crate) struct ModeProbe {
    pub(crate) embedding: Vec<f32>,
    pub(crate) agent_id: AgentId,
    pub(crate) user_id: UserId,
    valid_from: Option<u64>,
    valid_until: Option<u64>,
    /// The memory's page and storage write stamp, read BEFORE the memory was
    /// loaded. The probe may be newer than the stamp, never older, so a
    /// reopen that finds the same page and stamp knows no write landed on the
    /// memory since and can trust the probe without loading it. None when
    /// storage had no stamp (the probe is then never persisted).
    stamp: Option<(PageId, u64)>,
}

impl ModeProbe {
    fn from_node(node: &MemoryNode, stamp: Option<(PageId, u64)>) -> Self {
        Self {
            embedding: node.embedding.clone(),
            agent_id: node.agent_id,
            user_id: node.user_id,
            valid_from: node.valid_from,
            valid_until: node.valid_until,
            stamp,
        }
    }

    /// Same rule as `MemoryNode::is_valid_at`.
    pub(crate) fn is_valid_at(&self, at: u64) -> bool {
        let from = self.valid_from.unwrap_or(0);
        match self.valid_until {
            Some(until) => at >= from && at < until,
            None => at >= from,
        }
    }
}

impl MenteDb {
    /// The cached mode probe for `id`, loading the memory once on a miss
    /// (counted in `loads`). The slot is marked pending while loading; a write
    /// to this memory removes the slot, so a copy read before that write is
    /// returned but never cached. Writes to other memories do not interfere.
    pub(crate) fn mode_probe(
        &self,
        id: MemoryId,
        loads: &mut usize,
    ) -> Option<std::sync::Arc<ModeProbe>> {
        if let Some(ProbeSlot::Ready(p)) = self.mode_probes.read().get(&id) {
            return Some(std::sync::Arc::clone(p));
        }
        // Past the per request budget a miss is skipped, not loaded: it loads
        // on a later request, so one request never pays for every rule.
        if *loads
            >= self
                .cognitive_config
                .injection_config
                .mode_probe_load_budget
        {
            return None;
        }
        let token = {
            let mut cache = self.mode_probes.write();
            match cache.get(&id) {
                Some(ProbeSlot::Ready(p)) => return Some(std::sync::Arc::clone(p)),
                Some(ProbeSlot::Loading(_)) => None,
                None => {
                    let token = NEXT_PROBE_TOKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    cache.insert(id, ProbeSlot::Loading(token));
                    Some(token)
                }
            }
        };
        *loads += 1;
        // Stamp first, then load: see `ModeProbe::stamp`.
        let loaded = self.page_of(id).and_then(|page| {
            let stamp = self
                .storage
                .memory_stamp(page)
                .filter(|(sid, _)| *sid == id)
                .map(|(_, s)| (page, s));
            self.memory_loads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.storage
                .load_memory(page)
                .ok()
                .filter(|n| n.id == id)
                .map(|n| std::sync::Arc::new(ModeProbe::from_node(&n, stamp)))
        });
        if let Some(token) = token {
            let mut cache = self.mode_probes.write();
            if matches!(cache.get(&id), Some(ProbeSlot::Loading(t)) if *t == token) {
                match &loaded {
                    Some(p) => {
                        cache.insert(id, ProbeSlot::Ready(std::sync::Arc::clone(p)));
                    }
                    None => {
                        cache.remove(&id);
                    }
                }
            }
        }
        loaded
    }
}

/// A mode probe cache slot: being loaded (by the holder of this token) or ready.
pub(crate) enum ProbeSlot {
    Loading(u64),
    Ready(std::sync::Arc<ModeProbe>),
}

static NEXT_PROBE_TOKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// File (in the database directory) holding the persisted mode probes.
pub(crate) const MODE_PROBES_FILE: &str = "mode_probes.bin";
const PROBES_MAGIC: &[u8; 8] = b"MDBPROB1";

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_opt(buf: &mut Vec<u8>, v: Option<u64>) {
    buf.push(u8::from(v.is_some()));
    put_u64(buf, v.unwrap_or(0));
}

struct ProbeReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl ProbeReader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len())?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Some(s)
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn uuid(&mut self) -> Option<uuid::Uuid> {
        uuid::Uuid::from_slice(self.take(16)?).ok()
    }
    fn opt(&mut self) -> Option<Option<u64>> {
        let flag = self.take(1)?[0];
        let v = self.u64()?;
        Some((flag == 1).then_some(v))
    }
}

/// Encode stamped probes: magic, count, records, CRC32 of everything before.
fn encode_probes(probes: &[(MemoryId, std::sync::Arc<ModeProbe>)]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(PROBES_MAGIC);
    let stamped: Vec<_> = probes
        .iter()
        .filter_map(|(id, p)| p.stamp.map(|s| (id, p, s)))
        .collect();
    put_u64(&mut buf, stamped.len() as u64);
    for (id, p, (page, stamp)) in stamped {
        buf.extend_from_slice(id.0.as_bytes());
        put_u64(&mut buf, page.0);
        put_u64(&mut buf, stamp);
        buf.extend_from_slice(p.agent_id.0.as_bytes());
        buf.extend_from_slice(p.user_id.0.as_bytes());
        put_opt(&mut buf, p.valid_from);
        put_opt(&mut buf, p.valid_until);
        put_u64(&mut buf, p.embedding.len() as u64);
        for x in &p.embedding {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf
}

/// Decode a probe file; None when it is malformed in any way.
fn decode_probes(bytes: &[u8]) -> Option<Vec<(MemoryId, ModeProbe)>> {
    let (body, crc) = bytes.split_at(bytes.len().checked_sub(4)?);
    if crc32fast::hash(body) != u32::from_le_bytes(crc.try_into().ok()?) {
        return None;
    }
    let mut r = ProbeReader { buf: body, pos: 0 };
    if r.take(8)? != PROBES_MAGIC {
        return None;
    }
    let n = r.u64()?;
    let mut out = Vec::new();
    for _ in 0..n {
        let id = MemoryId(r.uuid()?);
        let page = PageId(r.u64()?);
        let stamp = r.u64()?;
        let agent_id = AgentId(r.uuid()?);
        let user_id = UserId(r.uuid()?);
        let valid_from = r.opt()?;
        let valid_until = r.opt()?;
        let dim = usize::try_from(r.u64()?).ok()?;
        let raw = r.take(dim.checked_mul(4)?)?;
        let embedding = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        out.push((
            id,
            ModeProbe {
                embedding,
                agent_id,
                user_id,
                valid_from,
                valid_until,
                stamp: Some((page, stamp)),
            },
        ));
    }
    (r.pos == body.len()).then_some(out)
}

impl MenteDb {
    /// Persist the ready mode probes, each with the page and write stamp it
    /// was loaded at, so a reopen restores them without loading the rules.
    /// Atomic (temp file, fsync, rename).
    pub(crate) fn save_mode_probes(&self) -> MenteResult<()> {
        let ready: Vec<(MemoryId, std::sync::Arc<ModeProbe>)> = self
            .mode_probes
            .read()
            .iter()
            .filter_map(|(id, slot)| match slot {
                ProbeSlot::Ready(p) => Some((*id, std::sync::Arc::clone(p))),
                ProbeSlot::Loading(_) => None,
            })
            .collect();
        let bytes = encode_probes(&ready);
        let path = self.path.join(MODE_PROBES_FILE);
        let tmp = self.path.join(format!("{MODE_PROBES_FILE}.tmp"));
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// The persisted probes still current for `mapped` (the memory map just
    /// loaded on open): a probe is restored only when its memory is on the
    /// same page with the same write stamp, i.e. nothing wrote the memory
    /// since the probe was loaded. Anything else is dropped and loads on use.
    pub(crate) fn load_mode_probes(
        dir: &std::path::Path,
        mapped: &[mentedb_storage::MappedMemory],
    ) -> HashMap<MemoryId, ProbeSlot> {
        let mut out = HashMap::new();
        let Ok(bytes) = std::fs::read(dir.join(MODE_PROBES_FILE)) else {
            return out;
        };
        let Some(probes) = decode_probes(&bytes) else {
            tracing::warn!("mode probe snapshot unreadable, probes will load on use");
            return out;
        };
        let current: HashMap<MemoryId, (PageId, u64)> =
            mapped.iter().map(|m| (m.id, (m.page, m.stamp))).collect();
        for (id, probe) in probes {
            if probe.stamp.is_some() && current.get(&id).copied() == probe.stamp {
                out.insert(id, ProbeSlot::Ready(std::sync::Arc::new(probe)));
            }
        }
        out
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

#[cfg(test)]
mod mode_probe_tests {
    use super::*;
    use mentedb_embedding::hash_provider::HashEmbeddingProvider;

    #[test]
    fn probes_load_once_and_reload_after_a_write() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open_with_embedder(dir.path(), Box::new(HashEmbeddingProvider::new(16)))
            .unwrap();
        let emb = db.embed_text("rule").unwrap().unwrap();
        let mut node = MemoryNode::new(AgentId::nil(), MemoryType::Procedural, "rule".into(), emb);
        node.tags = vec!["trigger:x".into()];
        let id = node.id;
        db.store(node).unwrap();

        let mut loads = 0;
        assert!(db.mode_probe(id, &mut loads).is_some());
        assert!(db.mode_probe(id, &mut loads).is_some());
        assert_eq!(loads, 1, "second lookup must come from the cache");

        // Writes to other memories leave this probe cached.
        for _ in 0..10 {
            db.note_write(MemoryId::new());
        }
        assert!(db.mode_probe(id, &mut loads).is_some());
        assert_eq!(loads, 1, "writes to other memories must not evict it");

        db.note_write(id);
        assert!(db.mode_probe(id, &mut loads).is_some());
        assert_eq!(loads, 2, "a write must force a reload");
        assert!(db.mode_probe(id, &mut loads).is_some());
        assert_eq!(loads, 2, "and the reload is cached again");
    }

    /// The production failure: an account writing constantly (a sync backlog
    /// draining) while recall fills the cache. Writes to other memories during
    /// a load must not stop that probe from being cached.
    #[test]
    fn probes_cache_while_other_memories_are_written() {
        let dir = tempfile::tempdir().unwrap();
        let db = std::sync::Arc::new(
            MenteDb::open_with_embedder(dir.path(), Box::new(HashEmbeddingProvider::new(16)))
                .unwrap(),
        );
        let mut ids = Vec::new();
        for i in 0..200 {
            let text = format!("rule {i}");
            let emb = db.embed_text(&text).unwrap().unwrap();
            let mut node = MemoryNode::new(AgentId::nil(), MemoryType::Procedural, text, emb);
            node.tags = vec!["trigger:x".into()];
            ids.push(node.id);
            db.store(node).unwrap();
        }
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writer = {
            let db = std::sync::Arc::clone(&db);
            let stop = std::sync::Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    db.note_write(MemoryId::new());
                }
            })
        };
        let mut first = 0;
        for id in &ids {
            db.mode_probe(*id, &mut first);
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        writer.join().unwrap();
        assert_eq!(first, ids.len());
        let mut second = 0;
        for id in &ids {
            db.mode_probe(*id, &mut second);
        }
        assert_eq!(second, 0, "every probe should be cached after one pass");
    }
}
