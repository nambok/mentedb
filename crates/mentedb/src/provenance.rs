//! Write time provenance on the facade.
//!
//! [`MemoryNode::provenance`] is stamped when an id is first written and kept
//! unchanged on every later write of it: [`MenteDb::store`],
//! [`MenteDb::store_batch`] and every in place update load the stored
//! provenance and write it back, whatever the caller's node carries. Memories
//! the engine derives (extraction, consolidation, entity, community and
//! profile nodes) get source `derived`, their origin ids, and the session
//! their origins share.
//!
//! Session and source are indexed as reserved bitmap keys, so
//! [`MenteDb::memory_ids_in_session`] and the filtered list and recall paths
//! cost O(matches), and [`MenteDb::erase_session`] finds what to erase
//! without a scan.

use std::collections::HashSet;

use mentedb_core::error::MenteResult;
use mentedb_core::provenance::{Provenance, SourceKind};
use mentedb_core::types::{AgentId, MemoryId};
use mentedb_core::{MemoryNode, MenteError};
use mentedb_index::manager::{session_index_key, source_index_key};
use mentedb_storage::PageId;
use serde::{Deserialize, Serialize};

use crate::MenteDb;

/// Narrow a listing or a recall to memories by provenance. Unset fields do
/// not narrow; set fields must all match.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProvenanceFilter {
    /// Only memories written in this session.
    pub session_id: Option<String>,
    /// Only memories from this kind of source.
    pub source: Option<SourceKind>,
}

impl ProvenanceFilter {
    /// A filter on one session.
    pub fn session(session_id: impl Into<String>) -> Self {
        Self {
            session_id: Some(session_id.into()),
            source: None,
        }
    }

    /// A filter on one source kind.
    pub fn source(source: SourceKind) -> Self {
        Self {
            session_id: None,
            source: Some(source),
        }
    }

    /// True when the filter narrows nothing.
    pub fn is_empty(&self) -> bool {
        self.session_id.is_none() && self.source.is_none()
    }

    /// Whether `node` passes the filter. A memory with no provenance passes
    /// only an empty filter.
    pub fn matches(&self, node: &MemoryNode) -> bool {
        if self.is_empty() {
            return true;
        }
        let Some(p) = &node.provenance else {
            return false;
        };
        if let Some(s) = &self.session_id
            && p.session_id.as_ref() != Some(s)
        {
            return false;
        }
        if let Some(k) = &self.source
            && &p.source != k
        {
            return false;
        }
        true
    }

    /// The reserved bitmap keys that must all be present.
    fn index_keys(&self) -> Vec<String> {
        let mut keys = Vec::with_capacity(2);
        if let Some(s) = &self.session_id {
            keys.push(session_index_key(s));
        }
        if let Some(k) = &self.source {
            keys.push(source_index_key(k.as_str()));
        }
        keys
    }
}

impl MenteDb {
    /// Settle the provenance a write of `node` stores. A re-store of an
    /// existing id (`existing` holds its page) takes the stored provenance,
    /// including none for a record written before provenance existed. A first
    /// write keeps the caller's, or stamps source `unspecified` written now by
    /// the owning agent when the caller gave none.
    pub(crate) fn settle_provenance(
        &self,
        node: &mut MemoryNode,
        existing: Option<PageId>,
    ) -> MenteResult<()> {
        match existing {
            Some(page) => {
                let stored = self.storage.load_memory(page)?;
                node.provenance = stored.provenance;
            }
            None => {
                if node.provenance.is_none() {
                    node.provenance = Some(Provenance::new(SourceKind::Unspecified, node.agent_id));
                }
            }
        }
        Ok(())
    }

    /// The stored provenance of the record at `page`, when it differs from
    /// the one `node` carries (so the write must restore it); None when they
    /// already agree.
    pub(crate) fn stored_provenance_differs(
        &self,
        page: PageId,
        node: &MemoryNode,
    ) -> MenteResult<Option<Option<Provenance>>> {
        let stored = self.storage.load_memory(page)?;
        if stored.id != node.id {
            return Err(MenteError::Storage(format!(
                "update of {} addressed the record of {}",
                node.id, stored.id
            )));
        }
        Ok((stored.provenance != node.provenance).then_some(stored.provenance))
    }

    /// Provenance for a memory the engine derives now from `origins`: source
    /// `derived`, the origin ids, and the session every origin shares (none
    /// when they differ, or any has none). Origins that no longer exist count
    /// as having no session.
    pub fn derived_provenance(&self, written_by: AgentId, origins: &[MemoryId]) -> Provenance {
        let sessions: Vec<Option<String>> = origins
            .iter()
            .map(|id| {
                self.get_memory(*id)
                    .ok()
                    .and_then(|n| n.provenance.and_then(|p| p.session_id))
            })
            .collect();
        Provenance::derived(written_by, origins, sessions.iter().map(|s| s.as_deref()))
    }

    /// Stamp `node` (about to be written for the first time) as derived from
    /// `origins`, keeping a source reference it already carries.
    pub(crate) fn stamp_derived(&self, node: &mut MemoryNode, origins: &[MemoryId]) {
        let source_ref = node.provenance.take().and_then(|p| p.source_ref);
        let mut p = self.derived_provenance(node.agent_id, origins);
        p.source_ref = source_ref;
        node.provenance = Some(p);
    }

    /// The provenance stored for memory `id` (None for a memory written
    /// before provenance existed).
    pub fn provenance_of(&self, id: MemoryId) -> MenteResult<Option<Provenance>> {
        Ok(self.get_memory(id)?.provenance)
    }

    /// Every stored memory written in `session_id`, from the bitmap index.
    pub fn memory_ids_in_session(&self, session_id: &str) -> Vec<MemoryId> {
        self.index.bitmap.query_tag(&session_index_key(session_id))
    }

    /// Every stored memory from `source`, from the bitmap index.
    pub fn memory_ids_from_source(&self, source: &SourceKind) -> Vec<MemoryId> {
        self.index
            .bitmap
            .query_tag(&source_index_key(source.as_str()))
    }

    /// Every stored memory matching `filter`, from the bitmap index; None
    /// when the filter is empty (everything matches).
    pub fn memory_ids_matching(&self, filter: &ProvenanceFilter) -> Option<HashSet<MemoryId>> {
        if filter.is_empty() {
            return None;
        }
        let keys = filter.index_keys();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        Some(
            self.index
                .bitmap
                .query_tags_and(&refs)
                .into_iter()
                .collect(),
        )
    }

    /// The session ids any stored memory was written in.
    pub fn provenance_sessions(&self) -> Vec<String> {
        let prefix = session_index_key("");
        let mut out: Vec<String> = self
            .index
            .bitmap
            .tags_with_prefix(&prefix)
            .into_iter()
            // A forget clears the bits but can leave the key behind.
            .filter(|k| !self.index.bitmap.query_tag(k).is_empty())
            .filter_map(|k| k.strip_prefix(&prefix).map(str::to_string))
            .collect();
        out.sort();
        out
    }
}
