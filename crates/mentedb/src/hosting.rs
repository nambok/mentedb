//! Primitives a hosting server needs on top of store and recall: field
//! updates that keep a memory's id, explicit supersession, a stable paged
//! listing narrowed by the bitmap index, per call hybrid recall options,
//! batched access statistics, enrichment provenance, and the persisted
//! enrichment watermark.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;

use mentedb_core::edge::EdgeType;
use mentedb_core::error::MenteResult;
use mentedb_core::memory::AttributeValue;
use mentedb_core::types::{AgentId, MemoryId, Timestamp};
use mentedb_core::{MemoryEdge, MemoryNode, MenteError};
use tracing::warn;

use crate::reranker::Reranker;
use crate::{MenteDb, StoreOutcome};

/// Attribute every derived memory carries, with the value [`ORIGIN_DERIVED`].
pub const ORIGIN_ATTRIBUTE: &str = "origin";
/// Value of [`ORIGIN_ATTRIBUTE`] on memories enrichment derived.
pub const ORIGIN_DERIVED: &str = "derived";
/// Attribute naming what kind of derived memory this is (see [`DerivedKind`]).
pub const DERIVED_KIND_ATTRIBUTE: &str = "derived_kind";
/// File under a database directory holding the enrichment watermark.
const WATERMARK_FILE: &str = "enrichment_watermark";

/// Per call options for [`MenteDb::recall_hybrid_scoped_with`].
#[derive(Clone, Copy, Default)]
pub struct HybridOptions<'a> {
    /// MMR diversity weight for this call, overriding
    /// `CognitiveConfig::mmr_lambda`. `1.0` disables the pass.
    pub mmr_lambda: Option<f32>,
    /// Reranker for this call, overriding any installed one. Applied only when
    /// the call carries query text.
    pub reranker: Option<&'a dyn Reranker>,
    /// Choose the top k on relevance alone, then blend in decayed salience to
    /// order just those. By default the blend is applied before the cut, so a
    /// faded but relevant memory can lose its place to a fresher, weaker one.
    pub relevance_first: bool,
    /// Count the hits as accesses when `CognitiveConfig::refresh_access_on_search`
    /// is on. Set by callers serving a user's search; internal recalls leave it
    /// off.
    pub record_access: bool,
}

/// Changes applied by [`MenteDb::update_memory_fields`]. Unset fields are
/// left as they are.
#[derive(Clone, Debug, Default)]
pub struct MemoryPatch {
    /// Merged into the memory's attributes: listed keys are set, others kept.
    pub attributes: HashMap<String, AttributeValue>,
    /// Replaces the memory's tags when set.
    pub tags: Option<Vec<String>>,
    /// Sets (`Some(Some(t))`) or clears (`Some(None)`) the end of the
    /// validity window.
    pub valid_until: Option<Option<Timestamp>>,
}

/// One page of [`MenteDb::list_memories_page`].
#[derive(Clone, Debug, Default)]
pub struct MemoryPage {
    /// Matching memories, oldest first.
    pub memories: Vec<MemoryNode>,
    /// Whether more matches follow this page.
    pub has_more: bool,
}

/// What a derived memory is, recorded in [`DERIVED_KIND_ATTRIBUTE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerivedKind {
    Fact,
    Entity,
    Community,
    Profile,
}

impl DerivedKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::Entity => "entity",
            Self::Community => "community",
            Self::Profile => "profile",
        }
    }
}

/// Stamp a derived memory with its provenance: inherited source attributes
/// (never overwriting a value the memory already has), `origin=derived`, and
/// `derived_kind`.
pub(crate) fn mark_derived(
    node: &mut MemoryNode,
    kind: DerivedKind,
    inherited: &HashMap<String, AttributeValue>,
) {
    for (k, v) in inherited {
        node.attributes
            .entry(k.clone())
            .or_insert_with(|| v.clone());
    }
    node.attributes.insert(
        ORIGIN_ATTRIBUTE.to_string(),
        AttributeValue::String(ORIGIN_DERIVED.to_string()),
    );
    node.attributes.insert(
        DERIVED_KIND_ATTRIBUTE.to_string(),
        AttributeValue::String(kind.as_str().to_string()),
    );
}

/// The enrichment watermark persisted in `dir`, or 0 when none was written.
pub(crate) fn read_enrichment_watermark(dir: &Path) -> u64 {
    std::fs::read_to_string(dir.join(WATERMARK_FILE))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Persist the enrichment watermark: written to a temporary file, synced, and
/// renamed over the old one, so a crash leaves either value, never a torn one.
pub(crate) fn write_enrichment_watermark(dir: &Path, value: u64) -> std::io::Result<()> {
    let tmp = dir.join(format!("{WATERMARK_FILE}.tmp"));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(value.to_string().as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, dir.join(WATERMARK_FILE))
}

fn now_micros() -> Timestamp {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

impl MenteDb {
    /// Attribute values the configured `copy_source_attributes` keys take
    /// from `sources`, the first source carrying a key winning.
    pub(crate) fn inherited_attributes(
        &self,
        sources: &[MemoryId],
    ) -> HashMap<String, AttributeValue> {
        let keys = &self
            .cognitive_config
            .enrichment_config
            .copy_source_attributes;
        let mut out = HashMap::new();
        if keys.is_empty() {
            return out;
        }
        for id in sources {
            if out.len() == keys.len() {
                break;
            }
            let Ok(src) = self.get_memory(*id) else {
                continue;
            };
            for key in keys {
                if !out.contains_key(key)
                    && let Some(v) = src.attributes.get(key)
                {
                    out.insert(key.clone(), v.clone());
                }
            }
        }
        out
    }

    /// Update a stored memory in place, keeping its id: merge attributes,
    /// replace tags, set or clear `valid_until`. The write goes through the
    /// WAL like any store, and the tag index follows a tag change.
    pub fn update_memory_fields(
        &self,
        id: MemoryId,
        patch: &MemoryPatch,
    ) -> MenteResult<MemoryNode> {
        let page = self.page_of(id).ok_or(MenteError::MemoryNotFound(id))?;
        let old = self.storage.load_memory(page)?;
        let mut node = old.clone();
        for (k, v) in &patch.attributes {
            node.attributes.insert(k.clone(), v.clone());
        }
        if let Some(tags) = &patch.tags {
            node.tags = tags.clone();
        }
        if let Some(until) = patch.valid_until {
            node.valid_until = until;
        }
        self.update_node(page, &node)?;
        if node.tags != old.tags {
            self.index.reindex_tags(&node);
            if self
                .scope_counts_ready
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                let hidden = &self.cognitive_config.hidden_count_tags;
                let mut counts = self.scope_counts.write();
                counts.apply(&old, -1, hidden);
                counts.apply(&node, 1, hidden);
            }
        }
        Ok(node)
    }

    /// Replace memory `old` with `new`: store `new`, close `old`'s validity
    /// window at `at`, and record a `Supersedes` edge from `new` to `old`. The
    /// old memory stays stored for history and point in time recall. Returns
    /// the new memory's id.
    ///
    /// Fails when `old` does not exist, or when `new` is skipped as a
    /// duplicate of an existing memory (nothing would replace `old`).
    pub fn supersede_memory(
        &self,
        old: MemoryId,
        new: MemoryNode,
        at: Timestamp,
    ) -> MenteResult<MemoryId> {
        let previous = self.get_memory(old)?;
        let new_id = new.id;
        match self.store_with_outcome(new)? {
            StoreOutcome::Stored => {}
            StoreOutcome::Duplicate { of } => {
                return Err(MenteError::Storage(format!(
                    "the replacement duplicates memory {of}; nothing was stored"
                )));
            }
        }
        // Write inference may already have closed it at the store above.
        if previous.valid_until.is_none_or(|u| u > at) {
            self.update_memory_fields(
                old,
                &MemoryPatch {
                    valid_until: Some(Some(at)),
                    ..MemoryPatch::default()
                },
            )?;
        }
        let already_linked = self
            .graph
            .graph()
            .outgoing(new_id)
            .iter()
            .any(|(t, e)| *t == old && e.edge_type == EdgeType::Supersedes);
        if !already_linked {
            self.relate(MemoryEdge {
                source: new_id,
                target: old,
                edge_type: EdgeType::Supersedes,
                weight: 1.0,
                created_at: at,
                valid_from: Some(at),
                valid_until: None,
                label: None,
            })?;
        }
        Ok(new_id)
    }

    /// Whether an active `Supersedes` edge points at `id` at time `at`.
    fn superseded_at(&self, id: MemoryId, at: Timestamp) -> bool {
        self.graph
            .graph()
            .incoming(id)
            .iter()
            .any(|(_, e)| e.edge_type == EdgeType::Supersedes && e.is_valid_at(at))
    }

    /// Whether a memory is current at `at`: inside its validity window and not
    /// superseded.
    pub fn is_active_at(&self, node: &MemoryNode, at: Timestamp) -> bool {
        node.is_valid_at(at) && !self.superseded_at(node.id, at)
    }

    /// A page of memories in a stable order (creation time, then insertion).
    ///
    /// `tag` and `agent` narrow through the bitmap index before anything is
    /// loaded; `agent` admits that agent's memories and shared ones owned by
    /// no agent. Without `include_inactive`, memories outside their validity
    /// window at `at` or superseded are skipped. `offset` counts matching
    /// memories. With `include_inactive` set, skipping to `offset` loads
    /// nothing; otherwise each skipped memory is loaded to check it is active.
    pub fn list_memories_page(
        &self,
        tag: Option<&str>,
        agent: Option<AgentId>,
        include_inactive: bool,
        at: Timestamp,
        offset: usize,
        limit: usize,
    ) -> MemoryPage {
        let mut allowed: Option<HashSet<MemoryId>> =
            tag.map(|t| self.index.bitmap.query_tag(t).into_iter().collect());
        if let Some(a) = agent {
            let keys = [
                mentedb_index::manager::agent_index_key(a),
                mentedb_index::manager::agent_index_key(AgentId::nil()),
            ];
            let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
            let owned: HashSet<MemoryId> =
                self.index.bitmap.query_tags_or(&refs).into_iter().collect();
            allowed = Some(match allowed {
                Some(prev) => prev.intersection(&owned).copied().collect(),
                None => owned,
            });
        }
        let ordered = self
            .index
            .temporal
            .range(0, Timestamp::MAX)
            .into_iter()
            .filter(|id| allowed.as_ref().is_none_or(|a| a.contains(id)));

        let mut page = MemoryPage::default();
        let mut skipped = 0usize;
        for id in ordered {
            if include_inactive && skipped < offset {
                skipped += 1;
                continue;
            }
            let Ok(node) = self.get_memory(id) else {
                continue;
            };
            if !include_inactive && !self.is_active_at(&node, at) {
                continue;
            }
            if skipped < offset {
                skipped += 1;
                continue;
            }
            if page.memories.len() == limit {
                page.has_more = true;
                break;
            }
            page.memories.push(node);
        }
        page
    }

    /// Record search hits as accesses. Held in memory and written at the next
    /// flush, so a burst of searches costs no write per hit.
    pub fn note_access(&self, ids: impl IntoIterator<Item = MemoryId>) {
        let now = now_micros();
        let mut pending = self.pending_access.lock();
        for id in ids {
            let entry = pending.entry(id).or_insert((now, 0));
            entry.0 = entry.0.max(now);
            entry.1 = entry.1.saturating_add(1);
        }
    }

    /// Accesses recorded by [`Self::note_access`] and not yet written.
    pub fn pending_access_count(&self) -> usize {
        self.pending_access.lock().len()
    }

    /// Write the batched access statistics. Called by flush.
    pub(crate) fn apply_pending_access(&self) {
        let pending = std::mem::take(&mut *self.pending_access.lock());
        for (id, (at, hits)) in pending {
            let Some(page) = self.page_of(id) else {
                continue;
            };
            let Ok(mut node) = self.storage.load_memory(page) else {
                continue;
            };
            node.accessed_at = node.accessed_at.max(at);
            node.access_count = node.access_count.saturating_add(hits);
            if let Err(e) = self.update_node(page, &node) {
                warn!(%id, error = %e, "writing access statistics failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CognitiveConfig;
    use mentedb_core::memory::MemoryType;

    fn db(dir: &Path) -> MenteDb {
        MenteDb::open_with_config(
            dir,
            CognitiveConfig {
                write_inference: false,
                memory_dedup: false,
                ..CognitiveConfig::default()
            },
        )
        .unwrap()
    }

    fn node(agent: AgentId, content: &str, tags: &[&str]) -> MemoryNode {
        let mut n = MemoryNode::new(agent, MemoryType::Semantic, content.into(), vec![]);
        n.tags = tags.iter().map(|t| t.to_string()).collect();
        n
    }

    #[test]
    fn update_merges_attributes_replaces_tags_and_reindexes() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(dir.path());
        let mut n = node(AgentId::nil(), "the sky is blue", &["a"]);
        n.attributes
            .insert("keep".into(), AttributeValue::String("x".into()));
        let id = n.id;
        db.store(n).unwrap();

        let mut patch = MemoryPatch {
            tags: Some(vec!["b".into()]),
            ..MemoryPatch::default()
        };
        patch
            .attributes
            .insert("added".into(), AttributeValue::Integer(7));
        let updated = db.update_memory_fields(id, &patch).unwrap();
        assert_eq!(updated.id, id);
        assert!(updated.attributes.contains_key("keep"));
        assert!(updated.attributes.contains_key("added"));
        assert!(db.memory_ids_with_tag("a").is_empty());
        assert_eq!(db.memory_ids_with_tag("b"), vec![id]);

        // Durable: survives a reopen.
        db.close().unwrap();
        drop(db);
        let db = super::tests::db(dir.path());
        let back = db.get_memory(id).unwrap();
        assert_eq!(back.tags, vec!["b".to_string()]);
        assert!(matches!(
            back.attributes.get("added"),
            Some(AttributeValue::Integer(7))
        ));
    }

    #[test]
    fn supersede_closes_the_old_memory_and_links_the_new_one() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(dir.path());
        let old = node(AgentId::nil(), "phone number is 1234", &[]);
        let old_id = old.id;
        db.store(old).unwrap();
        let at = now_micros();
        let new = node(AgentId::nil(), "phone number is 4321", &[]);
        let new_id = db.supersede_memory(old_id, new, at).unwrap();
        assert_ne!(new_id, old_id);
        let closed = db.get_memory(old_id).unwrap();
        assert_eq!(closed.valid_until, Some(at));
        let history = db.supersession_history(new_id, 4).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].id, old_id);
        assert!(!db.is_active_at(&closed, at + 1));
        assert!(
            db.supersede_memory(MemoryId::new(), node(AgentId::nil(), "x", &[]), at)
                .is_err()
        );
    }

    #[test]
    fn pages_are_stable_and_narrowed_by_tag_agent_and_activity() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(dir.path());
        let a = AgentId::new();
        let b = AgentId::new();
        let mut ids = Vec::new();
        for i in 0..10 {
            let owner = if i % 2 == 0 { a } else { b };
            let mut n = node(
                owner,
                &format!("memory {i}"),
                if i < 6 { &["t"] } else { &[] },
            );
            n.created_at = 1_000 + i as u64;
            ids.push(n.id);
            db.store(n).unwrap();
        }
        let shared = node(AgentId::nil(), "shared", &["t"]);
        let shared_id = shared.id;
        db.store(shared).unwrap();
        // Close one of agent a's tagged memories.
        db.invalidate_memory(ids[2], 1).unwrap();
        let now = now_micros();

        let all = db.list_memories_page(None, None, true, now, 0, 100);
        assert_eq!(all.memories.len(), 11);
        let order: Vec<MemoryId> = all.memories.iter().map(|m| m.id).collect();
        assert_eq!(&order[..10], &ids[..], "creation order");

        let tagged_a = db.list_memories_page(Some("t"), Some(a), false, now, 0, 100);
        let got: Vec<MemoryId> = tagged_a.memories.iter().map(|m| m.id).collect();
        assert_eq!(got, vec![ids[0], ids[4], shared_id]);

        let with_inactive = db.list_memories_page(Some("t"), Some(a), true, now, 0, 100);
        assert_eq!(with_inactive.memories.len(), 4);

        // Paging walks the same order without gaps or repeats.
        let p1 = db.list_memories_page(None, None, true, now, 0, 4);
        let p2 = db.list_memories_page(None, None, true, now, 4, 4);
        let p3 = db.list_memories_page(None, None, true, now, 8, 4);
        assert!(p1.has_more && p2.has_more && !p3.has_more);
        let walked: Vec<MemoryId> = [p1, p2, p3]
            .into_iter()
            .flat_map(|p| p.memories.into_iter().map(|m| m.id))
            .collect();
        assert_eq!(walked, order);
    }

    #[test]
    fn access_statistics_are_batched_until_flush() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(dir.path());
        let n = node(AgentId::nil(), "x", &[]);
        let id = n.id;
        db.store(n).unwrap();
        let before = db.get_memory(id).unwrap();
        db.note_access([id, id]);
        assert_eq!(db.pending_access_count(), 1);
        assert_eq!(db.get_memory(id).unwrap().access_count, before.access_count);
        db.flush().unwrap();
        assert_eq!(db.pending_access_count(), 0);
        assert_eq!(
            db.get_memory(id).unwrap().access_count,
            before.access_count + 2
        );
    }

    #[test]
    fn watermark_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_enrichment_watermark(dir.path()), 0);
        write_enrichment_watermark(dir.path(), 42).unwrap();
        assert_eq!(read_enrichment_watermark(dir.path()), 42);
    }

    #[test]
    fn enrichment_watermark_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(dir.path());
        db.mark_enrichment_complete(12_345);
        db.close().unwrap();
        drop(db);
        let db = super::tests::db(dir.path());
        assert_eq!(db.last_enrichment_turn(), 12_345);
    }

    #[test]
    fn decay_exemption_by_attribute() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(dir.path());
        let mut n = node(AgentId::nil(), "x", &[]);
        assert!(!db.is_decay_exempt(&n));
        n.attributes
            .insert("decay_exempt".into(), AttributeValue::Boolean(true));
        assert!(db.is_decay_exempt(&n));
        n.attributes
            .insert("decay_exempt".into(), AttributeValue::Boolean(false));
        assert!(!db.is_decay_exempt(&n));
    }

    #[test]
    fn derived_memories_carry_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = CognitiveConfig {
            write_inference: false,
            memory_dedup: false,
            ..CognitiveConfig::default()
        };
        cfg.enrichment_config.copy_source_attributes = vec!["doc".into(), "absent".into()];
        let db = MenteDb::open_with_config(dir.path(), cfg).unwrap();
        let mut src = node(AgentId::nil(), "source turn", &[]);
        src.memory_type = MemoryType::Episodic;
        src.attributes
            .insert("doc".into(), AttributeValue::String("d-1".into()));
        src.attributes
            .insert("private".into(), AttributeValue::String("p".into()));
        let src_id = src.id;
        db.store(src).unwrap();

        let fact = node(AgentId::nil(), "a fact", &[]);
        let fact_id = fact.id;
        let entity = node(AgentId::nil(), "an entity", &["entity:x"]);
        let entity_id = entity.id;
        db.store_enrichment_memories(vec![fact, entity], &[src_id])
            .unwrap();
        let fact = db.get_memory(fact_id).unwrap();
        fn s<'a>(n: &'a MemoryNode, key: &str) -> Option<&'a str> {
            match n.attributes.get(key) {
                Some(AttributeValue::String(v)) => Some(v.as_str()),
                _ => None,
            }
        }
        assert_eq!(s(&fact, "doc"), Some("d-1"));
        assert!(!fact.attributes.contains_key("private"));
        assert_eq!(s(&fact, ORIGIN_ATTRIBUTE), Some("derived"));
        assert_eq!(s(&fact, DERIVED_KIND_ATTRIBUTE), Some("fact"));
        let entity = db.get_memory(entity_id).unwrap();
        assert_eq!(s(&entity, DERIVED_KIND_ATTRIBUTE), Some("entity"));

        let profile = db
            .store_user_profile(
                "likes tea",
                mentedb_core::types::UserId::nil(),
                AgentId::nil(),
            )
            .unwrap();
        let profile = db.get_memory(profile).unwrap();
        assert_eq!(s(&profile, DERIVED_KIND_ATTRIBUTE), Some("profile"));
    }

    #[test]
    fn enrichment_candidate_types_are_configurable() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = CognitiveConfig {
            write_inference: false,
            memory_dedup: false,
            ..CognitiveConfig::default()
        };
        let semantic = node(AgentId::nil(), "a semantic note", &[]);
        let mut episodic = node(AgentId::nil(), "a turn", &[]);
        episodic.memory_type = MemoryType::Episodic;
        {
            let db = MenteDb::open_with_config(dir.path(), cfg.clone()).unwrap();
            db.store(semantic).unwrap();
            db.store(episodic).unwrap();
            let c = db.enrichment_candidates(mentedb_core::types::UserId::nil(), AgentId::nil());
            assert_eq!(c.len(), 1, "episodic only by default");
            db.close().unwrap();
        }
        cfg.enrichment_config.candidate_types = vec![MemoryType::Episodic, MemoryType::Semantic];
        let db = MenteDb::open_with_config(dir.path(), cfg).unwrap();
        let c = db.enrichment_candidates(mentedb_core::types::UserId::nil(), AgentId::nil());
        assert_eq!(c.len(), 2);
    }
}
