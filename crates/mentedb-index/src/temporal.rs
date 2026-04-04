//! BTreeMap-based temporal index for timestamp range queries.

use std::collections::BTreeMap;

use ahash::HashMap;
use parking_lot::RwLock;

use mentedb_core::types::{MemoryId, Timestamp};

/// Temporal index using a BTreeMap for efficient range queries on timestamps.
pub struct TemporalIndex {
    inner: RwLock<TemporalInner>,
}

struct TemporalInner {
    /// Timestamp → set of MemoryIds at that timestamp.
    tree: BTreeMap<Timestamp, Vec<MemoryId>>,
    /// Reverse lookup for fast removal.
    id_to_ts: HashMap<MemoryId, Timestamp>,
}

impl TemporalIndex {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(TemporalInner {
                tree: BTreeMap::new(),
                id_to_ts: HashMap::default(),
            }),
        }
    }

    /// Insert a memory with its timestamp.
    pub fn insert(&self, id: MemoryId, timestamp: Timestamp) {
        let mut inner = self.inner.write();
        inner.tree.entry(timestamp).or_default().push(id);
        inner.id_to_ts.insert(id, timestamp);
    }

    /// Query memories within [start, end] inclusive.
    pub fn range(&self, start: Timestamp, end: Timestamp) -> Vec<MemoryId> {
        let inner = self.inner.read();
        inner
            .tree
            .range(start..=end)
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect()
    }

    /// Get the `n` most recent memories (by timestamp, descending).
    pub fn latest(&self, n: usize) -> Vec<MemoryId> {
        let inner = self.inner.read();
        let mut results = Vec::with_capacity(n);
        for (_, ids) in inner.tree.iter().rev() {
            for &id in ids.iter().rev() {
                results.push(id);
                if results.len() >= n {
                    return results;
                }
            }
        }
        results
    }

    /// Remove a memory by id and its known timestamp.
    pub fn remove(&self, id: MemoryId, timestamp: Timestamp) {
        let mut inner = self.inner.write();
        if let Some(ids) = inner.tree.get_mut(&timestamp) {
            ids.retain(|&i| i != id);
            if ids.is_empty() {
                inner.tree.remove(&timestamp);
            }
        }
        inner.id_to_ts.remove(&id);
    }

    /// Remove a memory by id (looks up the timestamp internally).
    pub fn remove_by_id(&self, id: MemoryId) {
        let mut inner = self.inner.write();
        if let Some(ts) = inner.id_to_ts.remove(&id) {
            if let Some(ids) = inner.tree.get_mut(&ts) {
                ids.retain(|&i| i != id);
                if ids.is_empty() {
                    inner.tree.remove(&ts);
                }
            }
        }
    }

    /// Get the timestamp for a given memory id (if indexed).
    pub fn get_timestamp(&self, id: MemoryId) -> Option<Timestamp> {
        let inner = self.inner.read();
        inner.id_to_ts.get(&id).copied()
    }
}

impl Default for TemporalIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_insert_and_range() {
        let idx = TemporalIndex::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();

        idx.insert(a, 100);
        idx.insert(b, 200);
        idx.insert(c, 300);

        let result = idx.range(100, 200);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&a));
        assert!(result.contains(&b));
    }

    #[test]
    fn test_latest() {
        let idx = TemporalIndex::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();

        idx.insert(a, 100);
        idx.insert(b, 200);
        idx.insert(c, 300);

        let latest = idx.latest(2);
        assert_eq!(latest.len(), 2);
        assert_eq!(latest[0], c);
        assert_eq!(latest[1], b);
    }

    #[test]
    fn test_remove() {
        let idx = TemporalIndex::new();
        let a = Uuid::new_v4();
        idx.insert(a, 100);
        idx.remove(a, 100);

        assert!(idx.range(0, 1000).is_empty());
    }

    #[test]
    fn test_remove_by_id() {
        let idx = TemporalIndex::new();
        let a = Uuid::new_v4();
        idx.insert(a, 500);
        idx.remove_by_id(a);

        assert!(idx.range(0, 1000).is_empty());
    }

    #[test]
    fn test_empty_range() {
        let idx = TemporalIndex::new();
        assert!(idx.range(0, 100).is_empty());
    }
}
