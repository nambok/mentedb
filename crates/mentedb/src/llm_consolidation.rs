//! LLM-driven memory consolidation.
//!
//! Folds newly stored memories into existing near-duplicates the extractor
//! could not catch (semantic dedup, e.g. "uses Rust" and "prefers Rust for
//! systems work" become one memory). The LLM judgment is provided through the
//! [`LlmJudge`] trait, so the engine stays LLM-optional: callers with no judge
//! simply do not invoke this, and the rule-based [`MenteDb::consolidate_cluster`]
//! remains available.
//!
//! All applies are non-destructive: sources are invalidated (recall-hidden via
//! `valid_until`, retained on disk) and linked to the surviving memory by a
//! `Derived` edge, mirroring [`MenteDb::consolidate_cluster`].

use crate::MenteDb;
use mentedb_cognitive::llm::{
    ClusterMember, CognitiveLlmService, ConsolidationDecision, ContradictionVerdict, LlmJudge,
    MemorySummary,
};
use mentedb_core::edge::EdgeType;
use mentedb_core::error::MenteResult;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{AgentId, MemoryId, UserId};
use mentedb_core::{MemoryEdge, MemoryNode};

/// Tunables for LLM consolidation. Defaults mirror the contradiction sweep's
/// dedup band and a conservative information-loss guard.
#[derive(Debug, Clone)]
pub struct ConsolidationParams {
    /// Only existing memories at least this cosine-similar are candidates.
    pub similarity_floor: f32,
    /// Nearest existing memories fetched per new memory.
    pub top_k: usize,
    /// Reject a merge covering less than this fraction of any source's
    /// significant words (guards against dropped information).
    pub coverage_min: f32,
}

impl Default for ConsolidationParams {
    fn default() -> Self {
        Self {
            similarity_floor: 0.80,
            top_k: 6,
            coverage_min: 0.6,
        }
    }
}

/// Tunables for LLM conflict detection (contradictions and supersessions).
#[derive(Debug, Clone)]
pub struct ConflictDetectionParams {
    /// Minimum cosine similarity for two memories to be worth an LLM check.
    /// Below this they are unlikely to concern the same subject, so judging
    /// them would only burn tokens.
    pub check_floor: f32,
    /// At or above this similarity the pair is a near-identical duplicate and is
    /// left to `consolidate_memories` (merge/dedup) rather than flagged as a
    /// conflict. Genuine contradictions live in the band between the two.
    pub near_identical: f32,
    /// Nearest neighbors fetched per candidate memory.
    pub top_k: usize,
}

impl Default for ConflictDetectionParams {
    fn default() -> Self {
        Self {
            check_floor: 0.72,
            near_identical: 0.995,
            top_k: 6,
        }
    }
}

fn mtype_name(mt: MemoryType) -> &'static str {
    match mt {
        MemoryType::Episodic => "Episodic",
        MemoryType::Semantic => "Semantic",
        MemoryType::Procedural => "Procedural",
        MemoryType::AntiPattern => "AntiPattern",
        MemoryType::Reasoning => "Reasoning",
        MemoryType::Correction => "Correction",
    }
}

fn mtype_from(s: &str) -> MemoryType {
    match s.trim().to_ascii_lowercase().as_str() {
        "episodic" => MemoryType::Episodic,
        "procedural" => MemoryType::Procedural,
        "antipattern" | "anti_pattern" => MemoryType::AntiPattern,
        "reasoning" => MemoryType::Reasoning,
        "correction" => MemoryType::Correction,
        _ => MemoryType::Semantic,
    }
}

/// The scope tag that gates whether two memories may merge: a global fact and a
/// project fact are different assertions and must never fold together. Returns
/// the distinguishing scope tag, or None when the memory carries no scope tag.
fn scope_key(tags: &[String]) -> Option<&str> {
    tags.iter()
        .map(|t| t.as_str())
        .find(|t| *t == "scope:global" || t.starts_with("scope:project:"))
}

fn significant_words(s: &str) -> std::collections::HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 3)
        .map(|w| w.to_string())
        .collect()
}

/// Guard against information loss: every source's significant words must be
/// mostly present in the surviving text. Cheap defense in depth on top of the
/// consolidation prompt's "preserve all important information" constraint.
fn entailment_ok(surviving: &str, sources: &[&str], coverage_min: f32) -> bool {
    let sw_merged = significant_words(surviving);
    sources.iter().all(|src| {
        let sw = significant_words(src);
        if sw.is_empty() {
            return true;
        }
        let covered = sw.iter().filter(|w| sw_merged.contains(*w)).count();
        (covered as f32) >= (sw.len() as f32) * coverage_min
    })
}

/// Union of every source's tags, so a merged memory stays visible in every
/// scope its inputs were visible in.
fn merged_tags(sources: &[&MemoryNode]) -> Vec<String> {
    let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for s in sources {
        set.extend(s.tags.iter().cloned());
    }
    set.into_iter().collect()
}

fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

fn parse_id(s: &str) -> Option<MemoryId> {
    s.parse::<uuid::Uuid>().ok().map(MemoryId)
}

impl MenteDb {
    /// Consolidate newly stored memories against existing near-duplicates using
    /// an LLM judge.
    ///
    /// For each new memory, gathers same-agent same-scope neighbors above the
    /// similarity floor, asks the judge how to combine the cluster, and applies
    /// the MERGE/DEDUPLICATE decision non-destructively: sources are invalidated
    /// (recall-hidden via `valid_until`, retained on disk) and linked to the
    /// surviving memory by a `Derived` edge. Returns the number of clusters
    /// consolidated.
    ///
    /// Fails soft: a judge or parse error on one cluster skips that cluster, not
    /// the rest. The engine stays LLM-optional, pass any [`LlmJudge`]; with none,
    /// do not call this and consolidation simply does not run.
    pub async fn consolidate_memories<J: LlmJudge>(
        &self,
        new_ids: &[MemoryId],
        judge: J,
        params: &ConsolidationParams,
    ) -> MenteResult<usize> {
        let svc = CognitiveLlmService::new(judge);
        let new_set: std::collections::HashSet<MemoryId> = new_ids.iter().copied().collect();
        let mut consolidated = 0usize;

        for &new_id in new_ids {
            let Ok(new_node) = self.get_memory(new_id) else {
                continue;
            };
            if new_node.is_invalidated() || new_node.embedding.is_empty() {
                continue;
            }
            let new_scope = scope_key(&new_node.tags).map(|s| s.to_string());

            // Gather same-agent, same-scope neighbors above the floor, plus the
            // new memory itself, into one cluster for the judge.
            let hits = self
                .recall_similar(&new_node.embedding, params.top_k)
                .unwrap_or_default();
            let mut cluster: Vec<MemoryNode> = vec![new_node.clone()];
            let mut seen: std::collections::HashSet<MemoryId> = std::iter::once(new_id).collect();
            for (cid, sim) in hits {
                if sim < params.similarity_floor || new_set.contains(&cid) || !seen.insert(cid) {
                    continue;
                }
                let Ok(cand) = self.get_memory(cid) else {
                    continue;
                };
                if cand.agent_id != new_node.agent_id
                    || cand.user_id != new_node.user_id
                    || cand.is_invalidated()
                    || scope_key(&cand.tags).map(|s| s.to_string()) != new_scope
                {
                    continue;
                }
                cluster.push(cand);
            }
            if cluster.len() < 2 {
                continue;
            }

            let members: Vec<ClusterMember> = cluster
                .iter()
                .map(|n| ClusterMember {
                    id: n.id.to_string(),
                    content: n.content.clone(),
                    memory_type: mtype_name(n.memory_type).to_string(),
                    confidence: 1.0,
                    created_at: n.created_at,
                })
                .collect();

            let Ok(decision) = svc.consolidate(&members).await else {
                continue; // fail soft on judge/parse error
            };
            if self.apply_consolidation(
                &decision,
                &cluster,
                new_node.agent_id,
                new_node.user_id,
                params,
            ) {
                consolidated += 1;
            }
        }
        Ok(consolidated)
    }

    /// Apply one consolidation decision non-destructively against the cluster it
    /// was made for. Only ids the judge actually saw (present in `cluster`) can
    /// be touched. Returns true if anything changed.
    fn apply_consolidation(
        &self,
        decision: &ConsolidationDecision,
        cluster: &[MemoryNode],
        agent: AgentId,
        user: UserId,
        params: &ConsolidationParams,
    ) -> bool {
        let in_cluster = |id: &str| -> Option<&MemoryNode> {
            let mid = parse_id(id)?;
            cluster.iter().find(|n| n.id == mid)
        };
        let now = now_micros();

        match decision {
            ConsolidationDecision::KeepAll { .. } => false,

            ConsolidationDecision::Merge {
                merged_content,
                merged_type,
                remove_ids,
                ..
            } => {
                let merged_content = merged_content.trim();
                if merged_content.is_empty() {
                    return false;
                }
                // Resolve + revalidate sources: in the cluster, same agent AND
                // same user, still valid. Closes the window if a concurrent
                // write invalidated one.
                let sources: Vec<&MemoryNode> = remove_ids
                    .iter()
                    .filter_map(|id| in_cluster(id))
                    .filter(|n| n.agent_id == agent && n.user_id == user && !n.is_invalidated())
                    .collect();
                if sources.len() < 2 {
                    return false;
                }
                let src_texts: Vec<&str> = sources.iter().map(|n| n.content.as_str()).collect();
                if !entailment_ok(merged_content, &src_texts, params.coverage_min) {
                    return false;
                }

                let Ok(Some(embedding)) = self.embed_text(merged_content) else {
                    return false;
                };
                if embedding.is_empty() {
                    return false;
                }
                let mut merged = MemoryNode::new(
                    agent,
                    mtype_from(merged_type),
                    merged_content.to_string(),
                    embedding,
                )
                .with_user_id(user);
                merged.tags = merged_tags(&sources);
                let merged_id = merged.id;
                if self.store(merged).is_err() {
                    return false;
                }
                for s in &sources {
                    self.hide_into(merged_id, s.id, now);
                }
                true
            }

            ConsolidationDecision::Deduplicate {
                keep_id,
                remove_ids,
                ..
            } => {
                let Some(keep) = in_cluster(keep_id) else {
                    return false;
                };
                if keep.agent_id != agent || keep.user_id != user || keep.is_invalidated() {
                    return false;
                }
                let removes: Vec<&MemoryNode> = remove_ids
                    .iter()
                    .filter_map(|id| in_cluster(id))
                    .filter(|n| {
                        n.agent_id == agent
                            && n.user_id == user
                            && !n.is_invalidated()
                            && n.id != keep.id
                    })
                    .collect();
                if removes.is_empty() {
                    return false;
                }
                // Dedup must not drop information the kept memory lacks.
                let removed_texts: Vec<&str> = removes.iter().map(|n| n.content.as_str()).collect();
                if !entailment_ok(&keep.content, &removed_texts, params.coverage_min) {
                    return false;
                }
                for r in &removes {
                    self.hide_into(keep.id, r.id, now);
                }
                true
            }
        }
    }

    /// Invalidate `source` (recall-hidden, retained on disk) and record that
    /// `survivor` was derived from it. Best effort: edge/invalidation failures
    /// do not abort the consolidation.
    fn hide_into(&self, survivor: MemoryId, source: MemoryId, now: u64) {
        let _ = self.invalidate_memory(source, now);
        let _ = self.relate(MemoryEdge {
            source: survivor,
            target: source,
            edge_type: EdgeType::Derived,
            weight: 1.0,
            created_at: now,
            valid_from: None,
            valid_until: None,
            label: None,
        });
    }

    /// Detect genuine contradictions and supersessions between semantically
    /// similar memories using an LLM judge, recording them as typed edges.
    ///
    /// This is the correct path for conflict detection. Cosine similarity cannot
    /// tell a reworded duplicate from a real contradiction (both land in the
    /// high-similarity band), so the write-time heuristic no longer guesses; it
    /// produced ~0% precision. Here, for each candidate the engine gathers
    /// same-agent, same-scope neighbors in the "similar but not identical" band
    /// and asks the judge to read both texts:
    ///   - `Contradicts` becomes a `Contradicts` edge (both kept, flagged).
    ///   - `Supersedes` invalidates the loser and records a `Supersedes` edge.
    ///   - `Compatible` (or any judge/parse error) does nothing.
    ///
    /// Near-identical duplicates are skipped and left to `consolidate_memories`.
    ///
    /// Bounds work to `max_checks` LLM calls, fails soft per pair, and never
    /// re-judges a pair that already carries a conflict edge. Returns the number
    /// of conflict edges recorded.
    pub async fn detect_conflicts_with_llm<J: LlmJudge>(
        &self,
        candidate_ids: &[MemoryId],
        judge: J,
        params: &ConflictDetectionParams,
        max_checks: usize,
    ) -> MenteResult<usize> {
        if max_checks == 0 {
            return Ok(0);
        }

        // Phase 1: gather unordered candidate pairs. Read-only; no lock is held
        // across the later awaits because get_memory/recall_similar/graph each
        // take and release their own lock.
        let mut pairs: Vec<(MemoryNode, MemoryNode)> = Vec::new();
        let mut seen: std::collections::HashSet<(MemoryId, MemoryId)> =
            std::collections::HashSet::new();
        'outer: for &cid in candidate_ids {
            let Ok(node) = self.get_memory(cid) else {
                continue;
            };
            if node.is_invalidated() || node.embedding.is_empty() {
                continue;
            }
            let scope = scope_key(&node.tags).map(|s| s.to_string());
            let hits = self
                .recall_similar(&node.embedding, params.top_k)
                .unwrap_or_default();
            for (nid, sim) in hits {
                if nid == cid || sim < params.check_floor || sim >= params.near_identical {
                    continue;
                }
                let key = if cid < nid { (cid, nid) } else { (nid, cid) };
                if !seen.insert(key) {
                    continue;
                }
                let Ok(other) = self.get_memory(nid) else {
                    continue;
                };
                // Same agent + same user + same scope + not already a duplicate
                // string, and not already judged (a conflict edge exists).
                if other.is_invalidated()
                    || other.agent_id != node.agent_id
                    || other.user_id != node.user_id
                    || other.content == node.content
                    || scope_key(&other.tags).map(|s| s.to_string()) != scope
                    || self.has_conflict_edge(cid, nid)
                {
                    continue;
                }
                pairs.push((node.clone(), other));
                if pairs.len() >= max_checks {
                    break 'outer;
                }
            }
        }
        if pairs.is_empty() {
            return Ok(0);
        }

        // Phase 2: ask the judge to read each pair (no locks held).
        let svc = CognitiveLlmService::new(judge);
        enum Act {
            Contradict {
                a: MemoryId,
                b: MemoryId,
                reason: String,
            },
            Supersede {
                winner: MemoryId,
                loser: MemoryId,
            },
        }
        let mut acts: Vec<Act> = Vec::new();
        for (a, b) in &pairs {
            match svc
                .detect_contradiction(&conflict_summary(a), &conflict_summary(b))
                .await
            {
                Ok(ContradictionVerdict::Contradicts { reason }) => acts.push(Act::Contradict {
                    a: a.id,
                    b: b.id,
                    reason,
                }),
                Ok(ContradictionVerdict::Supersedes { winner, .. }) => {
                    let (w, l) = if winner == a.id.to_string() {
                        (a.id, b.id)
                    } else {
                        (b.id, a.id)
                    };
                    acts.push(Act::Supersede {
                        winner: w,
                        loser: l,
                    });
                }
                _ => {} // Compatible, or judge/parse error: nothing.
            }
        }

        // Phase 3: apply edges and invalidations.
        let now = now_micros();
        let mut recorded = 0usize;
        for act in acts {
            let ok = match act {
                Act::Contradict { a, b, reason } => self
                    .relate(MemoryEdge {
                        source: a,
                        target: b,
                        edge_type: EdgeType::Contradicts,
                        weight: 1.0,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: Some(reason),
                    })
                    .is_ok(),
                Act::Supersede { winner, loser } => {
                    let _ = self.invalidate_memory(loser, now);
                    self.relate(MemoryEdge {
                        source: winner,
                        target: loser,
                        edge_type: EdgeType::Supersedes,
                        weight: 1.0,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: None,
                    })
                    .is_ok()
                }
            };
            if ok {
                recorded += 1;
            }
        }
        Ok(recorded)
    }

    /// One-time cleanup: remove the conflict edges the old write-time heuristic
    /// created from bare cosine similarity (which flagged reworded duplicates as
    /// contradictions at ~0% precision). Removes every `Contradicts` and
    /// `Supersedes` edge, compacts, and persists a fresh graph snapshot so the
    /// edge log cannot re-add them on reopen.
    ///
    /// Does NOT alter memory validity: near-duplicate copies the heuristic hid
    /// stay hidden, and `detect_conflicts_with_llm` re-adds only genuine
    /// conflicts going forward. Returns `(contradicts_removed, supersedes_removed)`.
    pub fn purge_inferred_conflicts(&self) -> MenteResult<(usize, usize)> {
        let contradicts = self.graph().remove_edges_of_types(&[EdgeType::Contradicts]);
        let supersedes = self.graph().remove_edges_of_types(&[EdgeType::Supersedes]);
        self.graph().compact();
        self.flush_full()?;
        Ok((contradicts, supersedes))
    }

    /// True if a Contradicts or Supersedes edge already links `a` and `b` in
    /// either direction, so the conflict sweep never re-judges the same pair.
    fn has_conflict_edge(&self, a: MemoryId, b: MemoryId) -> bool {
        let gm = self.graph();
        let csr = gm.graph();
        let is_conflict = |et: EdgeType| matches!(et, EdgeType::Contradicts | EdgeType::Supersedes);
        if csr.contains_node(a) {
            for (t, e) in csr.outgoing(a) {
                if t == b && is_conflict(e.edge_type) {
                    return true;
                }
            }
        }
        if csr.contains_node(b) {
            for (t, e) in csr.outgoing(b) {
                if t == a && is_conflict(e.edge_type) {
                    return true;
                }
            }
        }
        false
    }
}

fn conflict_summary(n: &MemoryNode) -> MemorySummary {
    MemorySummary {
        id: n.id,
        content: n.content.clone(),
        memory_type: n.memory_type,
        confidence: n.confidence,
        created_at: n.created_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_cognitive::llm::MockLlmJudge;
    use mentedb_embedding::hash_provider::HashEmbeddingProvider;

    fn open_db(tag: &str) -> (MenteDb, std::path::PathBuf) {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "mentedb_llmcons_{}_{}_{}",
            tag,
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_dir_all(&path);
        let db = MenteDb::open_with_embedder(&path, Box::new(HashEmbeddingProvider::new(256)))
            .expect("open db");
        (db, path)
    }

    fn store(db: &MenteDb, agent: AgentId, content: &str, tags: &[&str]) -> MemoryId {
        let emb = db.embed_text(content).unwrap().unwrap();
        let mut node = MemoryNode::new(agent, MemoryType::Semantic, content.to_string(), emb);
        node.tags = tags.iter().map(|t| t.to_string()).collect();
        let id = node.id;
        db.store(node).unwrap();
        id
    }

    fn cluster_of(db: &MenteDb, ids: &[MemoryId]) -> Vec<MemoryNode> {
        ids.iter().map(|id| db.get_memory(*id).unwrap()).collect()
    }

    // Hash embeddings are not semantic, so real near-duplicates fall below any
    // meaningful production floor. A zero floor lets the end-to-end tests
    // exercise the full collect -> judge -> apply path deterministically; the
    // floor itself is a tunable, covered separately by the scope filter test.
    fn loose_floor() -> ConsolidationParams {
        ConsolidationParams {
            similarity_floor: 0.0,
            top_k: 6,
            coverage_min: 0.6,
        }
    }

    #[test]
    fn apply_merge_is_non_destructive() {
        let (db, path) = open_db("merge");
        let agent = AgentId(uuid::Uuid::new_v4());
        let e = store(&db, agent, "uses Rust", &["scope:global"]);
        let n = store(
            &db,
            agent,
            "prefers Rust for systems work",
            &["scope:global"],
        );
        let merged_text = "Uses Rust and prefers Rust for systems work";
        let decision = ConsolidationDecision::Merge {
            merged_content: merged_text.to_string(),
            merged_type: "Semantic".to_string(),
            keep_ids: vec![],
            remove_ids: vec![e.to_string(), n.to_string()],
            reason: "same preference".to_string(),
        };
        let cluster = cluster_of(&db, &[e, n]);
        assert!(db.apply_consolidation(
            &decision,
            &cluster,
            agent,
            UserId::nil(),
            &ConsolidationParams::default()
        ));

        // Sources retained with original content, invalidated.
        assert_eq!(db.get_memory(e).unwrap().content, "uses Rust");
        assert!(db.get_memory(e).unwrap().is_invalidated());
        assert!(db.get_memory(n).unwrap().is_invalidated());

        // Merged memory recallable (identical text is cosine 1.0), sources not.
        let q = db.embed_text(merged_text).unwrap().unwrap();
        let hits = db.recall_similar(&q, 10).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(db.get_memory(hits[0].0).unwrap().content, merged_text);
        assert!(hits.iter().all(|(id, _)| *id != e && *id != n));

        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn apply_dedup_keeps_one_hides_rest() {
        let (db, path) = open_db("dedup");
        let agent = AgentId(uuid::Uuid::new_v4());
        let keep = store(
            &db,
            agent,
            "the user deploys with Terraform to AWS",
            &["scope:global"],
        );
        let dupe = store(&db, agent, "deploys Terraform AWS", &["scope:global"]);
        let decision = ConsolidationDecision::Deduplicate {
            keep_id: keep.to_string(),
            remove_ids: vec![dupe.to_string()],
            reason: "redundant".to_string(),
        };
        let cluster = cluster_of(&db, &[keep, dupe]);
        assert!(db.apply_consolidation(
            &decision,
            &cluster,
            agent,
            UserId::nil(),
            &ConsolidationParams::default()
        ));
        assert!(!db.get_memory(keep).unwrap().is_invalidated());
        assert!(db.get_memory(dupe).unwrap().is_invalidated());
        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn apply_keep_all_is_noop() {
        let (db, path) = open_db("keepall");
        let agent = AgentId(uuid::Uuid::new_v4());
        let a = store(&db, agent, "likes tea", &["scope:global"]);
        let b = store(&db, agent, "uses Rust", &["scope:global"]);
        let decision = ConsolidationDecision::KeepAll {
            reason: "distinct".to_string(),
        };
        let cluster = cluster_of(&db, &[a, b]);
        assert!(!db.apply_consolidation(
            &decision,
            &cluster,
            agent,
            UserId::nil(),
            &ConsolidationParams::default()
        ));
        assert!(!db.get_memory(a).unwrap().is_invalidated());
        assert!(!db.get_memory(b).unwrap().is_invalidated());
        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn apply_merge_rejects_information_loss() {
        let (db, path) = open_db("entail");
        let agent = AgentId(uuid::Uuid::new_v4());
        let e = store(
            &db,
            agent,
            "deploys with Terraform to AWS",
            &["scope:global"],
        );
        let n = store(
            &db,
            agent,
            "uses Terraform for infrastructure",
            &["scope:global"],
        );
        let decision = ConsolidationDecision::Merge {
            merged_content: "does some deployment things".to_string(),
            merged_type: "Semantic".to_string(),
            keep_ids: vec![],
            remove_ids: vec![e.to_string(), n.to_string()],
            reason: "x".to_string(),
        };
        let cluster = cluster_of(&db, &[e, n]);
        assert!(!db.apply_consolidation(
            &decision,
            &cluster,
            agent,
            UserId::nil(),
            &ConsolidationParams::default()
        ));
        assert!(!db.get_memory(e).unwrap().is_invalidated());
        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[tokio::test]
    async fn consolidate_memories_end_to_end_with_mock_judge() {
        let (db, path) = open_db("e2e");
        let agent = AgentId(uuid::Uuid::new_v4());
        // Near-identical text so the hash embedder clusters them above the floor.
        let e = store(
            &db,
            agent,
            "deploys services to AWS using Terraform",
            &["scope:global"],
        );
        let n = store(
            &db,
            agent,
            "deploys services to AWS using Terraform daily",
            &["scope:global"],
        );
        let merged = "Deploys services to AWS using Terraform daily";
        let response = format!(
            r#"{{"action":"merge","merged_content":"{merged}","merged_type":"Semantic","keep_ids":[],"remove_ids":["{e}","{n}"],"reason":"same"}}"#
        );
        let judge = MockLlmJudge::new(response);
        let count = db
            .consolidate_memories(&[n], judge, &loose_floor())
            .await
            .unwrap();
        assert_eq!(count, 1, "the cluster should consolidate");
        assert!(db.get_memory(e).unwrap().is_invalidated());
        assert!(db.get_memory(n).unwrap().is_invalidated());
        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[tokio::test]
    async fn consolidate_memories_refuses_cross_scope() {
        let (db, path) = open_db("scope");
        let agent = AgentId(uuid::Uuid::new_v4());
        // Same text but different scope: a global fact and a project fact must
        // never cluster, so there is nothing for the judge to merge.
        store(
            &db,
            agent,
            "deploys services to AWS using Terraform",
            &["scope:global"],
        );
        let n = store(
            &db,
            agent,
            "deploys services to AWS using Terraform",
            &["scope:project:apex"],
        );
        let judge = MockLlmJudge::new(r#"{"action":"keep_all","reason":"n/a"}"#);
        let count = db
            .consolidate_memories(&[n], judge, &loose_floor())
            .await
            .unwrap();
        assert_eq!(count, 0);
        assert!(!db.get_memory(n).unwrap().is_invalidated());
        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[tokio::test]
    async fn detect_conflicts_flags_contradiction_via_judge() {
        let (db, path) = open_db("conflict");
        let agent = AgentId(uuid::Uuid::new_v4());
        let a = store(
            &db,
            agent,
            "User prefers tabs for indentation",
            &["scope:global"],
        );
        let b = store(
            &db,
            agent,
            "User prefers spaces for indentation",
            &["scope:global"],
        );

        // Loose bands so the hash-embedding neighbor qualifies as a candidate
        // (hash similarity is not semantic). The judge says they contradict.
        let params = ConflictDetectionParams {
            check_floor: 0.0,
            near_identical: 1.0,
            top_k: 6,
        };
        let judge = MockLlmJudge::new(r#"{"verdict":"contradicts","reason":"tabs vs spaces"}"#);
        let recorded = db
            .detect_conflicts_with_llm(&[b], judge, &params, 8)
            .await
            .unwrap();
        assert_eq!(recorded, 1, "one contradiction edge expected");

        let has_edge = {
            let g = db.graph().read_graph();
            g.outgoing(b)
                .iter()
                .any(|(t, e)| *t == a && e.edge_type == EdgeType::Contradicts)
        };
        assert!(has_edge, "expected a Contradicts edge b -> a");

        // Idempotent: the pair already carries a conflict edge, so a second sweep
        // records nothing.
        let judge2 = MockLlmJudge::new(r#"{"verdict":"contradicts","reason":"x"}"#);
        let again = db
            .detect_conflicts_with_llm(&[b], judge2, &params, 8)
            .await
            .unwrap();
        assert_eq!(again, 0, "existing conflict edge must not be re-judged");

        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn purge_removes_conflict_edges_keeps_related() {
        let (db, path) = open_db("purge");
        let agent = AgentId(uuid::Uuid::new_v4());
        let a = store(&db, agent, "fact A", &["scope:global"]);
        let b = store(&db, agent, "fact B", &["scope:global"]);

        let now = now_micros();
        let edge = |et: EdgeType| MemoryEdge {
            source: a,
            target: b,
            edge_type: et,
            weight: 1.0,
            created_at: now,
            valid_from: None,
            valid_until: None,
            label: None,
        };
        db.relate(edge(EdgeType::Contradicts)).unwrap();
        db.relate(edge(EdgeType::Related)).unwrap();
        db.relate(edge(EdgeType::Supersedes)).unwrap();

        let (c, s) = db.purge_inferred_conflicts().unwrap();
        assert_eq!((c, s), (1, 1), "one contradicts + one supersedes removed");

        let g = db.graph().read_graph();
        let out = g.outgoing(a);
        assert!(
            out.iter().all(|(_, e)| e.edge_type != EdgeType::Contradicts
                && e.edge_type != EdgeType::Supersedes),
            "conflict edges must be gone"
        );
        assert!(
            out.iter()
                .any(|(t, e)| *t == b && e.edge_type == EdgeType::Related),
            "the Related sibling must survive the purge"
        );
        drop(g);
        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }
}
