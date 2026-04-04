//! MVCC Version Tracking — simple multi-version concurrency control for memories.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::types::{AgentId, MemoryId, Timestamp};

/// A single versioned snapshot of a memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version {
    pub version_id: u64,
    pub memory_id: MemoryId,
    pub agent_id: AgentId,
    pub timestamp: Timestamp,
    pub data_hash: u64,
}

/// Tracks version history for all memories.
#[derive(Debug)]
pub struct VersionStore {
    versions: HashMap<MemoryId, Vec<Version>>,
    next_version: AtomicU64,
}

impl Default for VersionStore {
    fn default() -> Self {
        Self {
            versions: HashMap::new(),
            next_version: AtomicU64::new(1),
        }
    }
}

fn now_micros() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as Timestamp
}

impl VersionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a write and return the new version ID.
    pub fn record_write(
        &mut self,
        memory_id: MemoryId,
        agent_id: AgentId,
        data_hash: u64,
    ) -> u64 {
        let vid = self.next_version.fetch_add(1, Ordering::Relaxed);
        let version = Version {
            version_id: vid,
            memory_id,
            agent_id,
            timestamp: now_micros(),
            data_hash,
        };
        self.versions.entry(memory_id).or_default().push(version);
        vid
    }

    /// Get the latest version for a memory.
    pub fn get_latest(&self, memory_id: MemoryId) -> Option<&Version> {
        self.versions.get(&memory_id).and_then(|v| v.last())
    }

    /// Get the full version history for a memory.
    pub fn get_history(&self, memory_id: MemoryId) -> Vec<&Version> {
        self.versions
            .get(&memory_id)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get the version that was current at or before `timestamp`.
    pub fn get_version_at(&self, memory_id: MemoryId, timestamp: Timestamp) -> Option<&Version> {
        self.versions
            .get(&memory_id)
            .and_then(|v| v.iter().rev().find(|ver| ver.timestamp <= timestamp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn record_and_get_latest() {
        let mut store = VersionStore::new();
        let mid = Uuid::new_v4();
        let aid = Uuid::new_v4();
        store.record_write(mid, aid, 111);
        store.record_write(mid, aid, 222);
        assert_eq!(store.get_latest(mid).unwrap().data_hash, 222);
    }

    #[test]
    fn version_ids_increment() {
        let mut store = VersionStore::new();
        let mid = Uuid::new_v4();
        let aid = Uuid::new_v4();
        let v1 = store.record_write(mid, aid, 1);
        let v2 = store.record_write(mid, aid, 2);
        assert_eq!(v2, v1 + 1);
    }

    #[test]
    fn get_history() {
        let mut store = VersionStore::new();
        let mid = Uuid::new_v4();
        let aid = Uuid::new_v4();
        store.record_write(mid, aid, 10);
        store.record_write(mid, aid, 20);
        store.record_write(mid, aid, 30);
        assert_eq!(store.get_history(mid).len(), 3);
    }

    #[test]
    fn get_version_at() {
        let mut store = VersionStore::new();
        let mid = Uuid::new_v4();
        let aid = Uuid::new_v4();
        store.record_write(mid, aid, 1);
        // The latest write should be findable at a very large timestamp.
        let ver = store.get_version_at(mid, u64::MAX);
        assert!(ver.is_some());
    }

    #[test]
    fn empty_history() {
        let store = VersionStore::new();
        assert!(store.get_latest(Uuid::new_v4()).is_none());
        assert!(store.get_history(Uuid::new_v4()).is_empty());
    }
}
