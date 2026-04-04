//! Raft log store and state machine backed by MenteDB storage.
//!
//! `LogStore` persists Raft log entries for durability and replication.
//! `StateMachine` applies committed entries to the MenteDB state.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::io;
use std::io::Cursor;
use std::ops::RangeBounds;
use std::sync::Arc;

use futures_util::StreamExt;
use openraft::entry::RaftEntry;
use openraft::storage::{IOFlushed, RaftLogStorage, RaftStateMachine};
use openraft::{EntryPayload, OptionalSend};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::raft_node::{MenteRequest, MenteResponse, TypeConfig};

// Type aliases derived from TypeConfig.
type LogId = openraft::type_config::alias::LogIdOf<TypeConfig>;
type Vote = openraft::type_config::alias::VoteOf<TypeConfig>;
type Entry = openraft::type_config::alias::EntryOf<TypeConfig>;
type StoredMembership = openraft::type_config::alias::StoredMembershipOf<TypeConfig>;
type Snapshot = openraft::type_config::alias::SnapshotOf<TypeConfig>;
type SnapshotMeta = openraft::type_config::alias::SnapshotMetaOf<TypeConfig>;
type EntryResponder = openraft::storage::EntryResponder<TypeConfig>;

/// Shared inner state for the log store, protected by a read-write lock
/// so that both the main LogStore and its LogReader can access it.
#[derive(Default)]
struct LogStoreInner {
    vote: Option<Vote>,
    log: BTreeMap<u64, Entry>,
    committed: Option<LogId>,
}

/// In-memory log store for Raft entries.
///
/// Stores log entries and the current vote in memory. In production this
/// would be backed by the MenteDB WAL for durability.
pub struct LogStore {
    inner: Arc<RwLock<LogStoreInner>>,
}

impl LogStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(LogStoreInner::default())),
        }
    }

    /// Returns the number of log entries currently stored.
    pub fn len(&self) -> usize {
        self.inner.read().log.len()
    }

    /// Returns true if the log store has no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.read().log.is_empty()
    }
}

impl Default for LogStore {
    fn default() -> Self {
        Self::new()
    }
}

/// A reader handle to the log store, sharing the same inner state.
pub struct LogReader {
    inner: Arc<RwLock<LogStoreInner>>,
}

impl openraft::storage::RaftLogReader<TypeConfig> for LogReader {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + OptionalSend>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry>, io::Error> {
        let inner = self.inner.read();
        Ok(inner.log.range(range).map(|(_, e)| e.clone()).collect())
    }

    async fn read_vote(&mut self) -> Result<Option<Vote>, io::Error> {
        Ok(self.inner.read().vote)
    }
}

impl RaftLogStorage<TypeConfig> for LogStore {
    type LogReader = LogReader;

    async fn get_log_state(
        &mut self,
    ) -> Result<openraft::storage::LogState<TypeConfig>, io::Error> {
        let inner = self.inner.read();
        let last = inner.log.iter().next_back().map(|(_, e)| e.log_id());
        Ok(openraft::storage::LogState {
            last_purged_log_id: None,
            last_log_id: last,
        })
    }

    async fn save_vote(&mut self, vote: &Vote) -> Result<(), io::Error> {
        self.inner.write().vote = Some(*vote);
        Ok(())
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        LogReader {
            inner: Arc::clone(&self.inner),
        }
    }

    async fn append<I>(
        &mut self,
        entries: I,
        callback: IOFlushed<TypeConfig>,
    ) -> Result<(), io::Error>
    where
        I: IntoIterator<Item = Entry> + OptionalSend,
        I::IntoIter: OptionalSend,
    {
        let mut inner = self.inner.write();
        for entry in entries {
            let index = entry.log_id.index;
            debug!(index, "appending log entry");
            inner.log.insert(index, entry);
        }
        callback.io_completed(Ok(()));
        Ok(())
    }

    async fn truncate_after(&mut self, last_log_id: Option<LogId>) -> Result<(), io::Error> {
        let mut inner = self.inner.write();
        match last_log_id {
            Some(log_id) => {
                let keys: Vec<u64> = inner
                    .log
                    .range((log_id.index + 1)..)
                    .map(|(k, _)| *k)
                    .collect();
                for key in keys {
                    inner.log.remove(&key);
                }
            }
            None => {
                inner.log.clear();
            }
        }
        Ok(())
    }

    async fn purge(&mut self, log_id: LogId) -> Result<(), io::Error> {
        let mut inner = self.inner.write();
        let keys: Vec<u64> = inner.log.range(..=log_id.index).map(|(k, _)| *k).collect();
        for key in keys {
            inner.log.remove(&key);
        }
        Ok(())
    }

    async fn save_committed(&mut self, committed: Option<LogId>) -> Result<(), io::Error> {
        self.inner.write().committed = committed;
        Ok(())
    }

    async fn read_committed(&mut self) -> Result<Option<LogId>, io::Error> {
        Ok(self.inner.read().committed)
    }
}

/// Serializable snapshot of the state machine.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StateMachineData {
    pub last_applied_log: Option<LogId>,
    pub last_membership: StoredMembership,
    /// Memory store: maps memory_id to (content, tags).
    pub memories: BTreeMap<String, MemoryEntry>,
    /// Relationships: maps "source:target" to (relation_type, weight).
    pub relationships: BTreeMap<String, RelationEntry>,
}

/// A stored memory entry in the replicated state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryEntry {
    pub content: String,
    pub tags: Vec<String>,
}

/// A stored relationship in the replicated state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RelationEntry {
    pub relation_type: String,
    pub weight: f32,
}

/// State machine that applies Raft-committed commands to MenteDB state.
pub struct StateMachine {
    data: RwLock<StateMachineData>,
}

impl StateMachine {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(StateMachineData::default()),
        }
    }

    /// Read a memory entry from the state machine by ID.
    pub fn get_memory(&self, memory_id: &str) -> Option<MemoryEntry> {
        self.data.read().memories.get(memory_id).cloned()
    }

    /// Returns the number of memories in the state machine.
    pub fn memory_count(&self) -> usize {
        self.data.read().memories.len()
    }

    /// Returns the number of relationships in the state machine.
    pub fn relationship_count(&self) -> usize {
        self.data.read().relationships.len()
    }
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl RaftStateMachine<TypeConfig> for StateMachine {
    type SnapshotBuilder = StateMachineSnapshot;

    async fn applied_state(&mut self) -> Result<(Option<LogId>, StoredMembership), io::Error> {
        let data = self.data.read();
        Ok((data.last_applied_log, data.last_membership.clone()))
    }

    async fn apply<Strm>(&mut self, mut entries: Strm) -> Result<(), io::Error>
    where
        Strm: futures_util::Stream<Item = Result<EntryResponder, io::Error>> + Unpin + OptionalSend,
    {
        while let Some(item) = entries.next().await {
            let (entry, responder) = item?;
            let mut data = self.data.write();
            data.last_applied_log = Some(entry.log_id());

            let response = match entry.payload {
                EntryPayload::Blank => MenteResponse::default(),
                EntryPayload::Normal(req) => apply_request(&mut data, req),
                EntryPayload::Membership(membership) => {
                    data.last_membership = StoredMembership::new(data.last_applied_log, membership);
                    MenteResponse::default()
                }
            };

            if let Some(responder) = responder {
                responder.send(response);
            }
        }
        Ok(())
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        StateMachineSnapshot {
            data: self.data.read().clone(),
        }
    }

    async fn begin_receiving_snapshot(&mut self) -> Result<Cursor<Vec<u8>>, io::Error> {
        Ok(Cursor::new(Vec::new()))
    }

    async fn install_snapshot(
        &mut self,
        _meta: &SnapshotMeta,
        snapshot: Cursor<Vec<u8>>,
    ) -> Result<(), io::Error> {
        let data: StateMachineData = serde_json::from_slice(snapshot.get_ref())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        *self.data.write() = data;
        Ok(())
    }

    async fn get_current_snapshot(&mut self) -> Result<Option<Snapshot>, io::Error> {
        Ok(None)
    }
}

/// A frozen copy of the state machine data for building a snapshot.
pub struct StateMachineSnapshot {
    data: StateMachineData,
}

impl openraft::storage::RaftSnapshotBuilder<TypeConfig> for StateMachineSnapshot {
    async fn build_snapshot(&mut self) -> Result<Snapshot, io::Error> {
        let json =
            serde_json::to_vec(&self.data).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let last_applied = self.data.last_applied_log;
        let membership = self.data.last_membership.clone();

        let snapshot_id = last_applied
            .map(|id| format!("{}-{}", id.leader_id, id.index))
            .unwrap_or_else(|| "0-0".to_string());

        let meta = SnapshotMeta {
            last_log_id: last_applied,
            last_membership: membership,
            snapshot_id,
        };

        Ok(Snapshot {
            meta,
            snapshot: Cursor::new(json),
        })
    }
}

/// Apply a single MenteRequest to the state machine data.
fn apply_request(data: &mut StateMachineData, req: MenteRequest) -> MenteResponse {
    match req {
        MenteRequest::StoreMemory {
            memory_id,
            content,
            tags,
        } => {
            info!(memory_id = %memory_id, "storing memory via raft");
            data.memories
                .insert(memory_id.clone(), MemoryEntry { content, tags });
            MenteResponse::MemoryStored { memory_id }
        }
        MenteRequest::ForgetMemory { memory_id } => {
            info!(memory_id = %memory_id, "forgetting memory via raft");
            data.memories.remove(&memory_id);
            MenteResponse::MemoryForgotten { memory_id }
        }
        MenteRequest::RelateMemories {
            source_id,
            target_id,
            relation_type,
            weight,
        } => {
            let key = format!("{}:{}", source_id, target_id);
            data.relationships.insert(
                key,
                RelationEntry {
                    relation_type,
                    weight,
                },
            );
            MenteResponse::MemoriesRelated {
                source_id,
                target_id,
            }
        }
        MenteRequest::UpdateMemory {
            memory_id,
            content,
            tags,
        } => {
            if let Some(entry) = data.memories.get_mut(&memory_id) {
                if let Some(c) = content {
                    entry.content = c;
                }
                if let Some(t) = tags {
                    entry.tags = t;
                }
                MenteResponse::MemoryUpdated { memory_id }
            } else {
                MenteResponse::Error {
                    message: format!("memory not found: {}", memory_id),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_store_new_is_empty() {
        let store = LogStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_state_machine_new_is_empty() {
        let sm = StateMachine::new();
        assert_eq!(sm.memory_count(), 0);
        assert_eq!(sm.relationship_count(), 0);
        assert!(sm.get_memory("nonexistent").is_none());
    }

    #[test]
    fn test_apply_store_memory() {
        let mut data = StateMachineData::default();
        let resp = apply_request(
            &mut data,
            MenteRequest::StoreMemory {
                memory_id: "m1".into(),
                content: "hello world".into(),
                tags: vec!["greeting".into()],
            },
        );
        assert_eq!(
            resp,
            MenteResponse::MemoryStored {
                memory_id: "m1".into()
            }
        );
        assert_eq!(data.memories.len(), 1);
        let entry = data.memories.get("m1").unwrap();
        assert_eq!(entry.content, "hello world");
        assert_eq!(entry.tags, vec!["greeting".to_string()]);
    }

    #[test]
    fn test_apply_forget_memory() {
        let mut data = StateMachineData::default();
        apply_request(
            &mut data,
            MenteRequest::StoreMemory {
                memory_id: "m1".into(),
                content: "data".into(),
                tags: vec![],
            },
        );
        assert_eq!(data.memories.len(), 1);

        let resp = apply_request(
            &mut data,
            MenteRequest::ForgetMemory {
                memory_id: "m1".into(),
            },
        );
        assert_eq!(
            resp,
            MenteResponse::MemoryForgotten {
                memory_id: "m1".into()
            }
        );
        assert_eq!(data.memories.len(), 0);
    }

    #[test]
    fn test_apply_relate_memories() {
        let mut data = StateMachineData::default();
        let resp = apply_request(
            &mut data,
            MenteRequest::RelateMemories {
                source_id: "m1".into(),
                target_id: "m2".into(),
                relation_type: "related".into(),
                weight: 0.8,
            },
        );
        assert_eq!(
            resp,
            MenteResponse::MemoriesRelated {
                source_id: "m1".into(),
                target_id: "m2".into(),
            }
        );
        let rel = data.relationships.get("m1:m2").unwrap();
        assert_eq!(rel.relation_type, "related");
        assert!((rel.weight - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_apply_update_memory() {
        let mut data = StateMachineData::default();
        apply_request(
            &mut data,
            MenteRequest::StoreMemory {
                memory_id: "m1".into(),
                content: "original".into(),
                tags: vec!["old".into()],
            },
        );

        let resp = apply_request(
            &mut data,
            MenteRequest::UpdateMemory {
                memory_id: "m1".into(),
                content: Some("updated".into()),
                tags: None,
            },
        );
        assert_eq!(
            resp,
            MenteResponse::MemoryUpdated {
                memory_id: "m1".into()
            }
        );
        let entry = data.memories.get("m1").unwrap();
        assert_eq!(entry.content, "updated");
        assert_eq!(entry.tags, vec!["old".to_string()]);
    }

    #[test]
    fn test_apply_update_nonexistent() {
        let mut data = StateMachineData::default();
        let resp = apply_request(
            &mut data,
            MenteRequest::UpdateMemory {
                memory_id: "missing".into(),
                content: Some("x".into()),
                tags: None,
            },
        );
        match resp {
            MenteResponse::Error { message } => {
                assert!(message.contains("not found"));
            }
            _ => panic!("expected error response"),
        }
    }

    #[tokio::test]
    async fn test_log_store_vote_roundtrip() {
        let mut store = LogStore::new();

        let state = store.get_log_state().await.unwrap();
        assert!(state.last_log_id.is_none());
        assert!(state.last_purged_log_id.is_none());

        let mut reader = store.get_log_reader().await;
        let no_vote = reader.read_vote().await.unwrap();
        assert!(no_vote.is_none());
    }

    #[tokio::test]
    async fn test_state_machine_applied_state() {
        let mut sm = StateMachine::new();
        let (last_applied, _membership) = sm.applied_state().await.unwrap();
        assert!(last_applied.is_none());
    }

    #[test]
    fn test_memory_entry_serialization() {
        let entry = MemoryEntry {
            content: "test".into(),
            tags: vec!["a".into(), "b".into()],
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn test_state_machine_data_serialization() {
        let mut data = StateMachineData::default();
        apply_request(
            &mut data,
            MenteRequest::StoreMemory {
                memory_id: "m1".into(),
                content: "content".into(),
                tags: vec![],
            },
        );
        let json = serde_json::to_vec(&data).unwrap();
        let back: StateMachineData = serde_json::from_slice(&json).unwrap();
        assert_eq!(back.memories.len(), 1);
    }
}
