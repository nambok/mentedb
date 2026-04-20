//! Sorted salience index for top-k retrieval by importance score.

use std::collections::BTreeMap;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;

/// An f32 wrapper that provides total ordering via bit representation.
/// This converts f32 to a sortable u32 key, handling NaN/negative/positive correctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct OrderedF32(u32);

impl OrderedF32 {
    fn from_f32(v: f32) -> Self {
        let bits = v.to_bits();
        // Flip sign bit, and if negative, flip all bits for correct ordering.
        let key = if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        };
        Self(key)
    }

    fn to_f32(self) -> f32 {
        let bits = if self.0 & 0x8000_0000 != 0 {
            self.0 ^ 0x8000_0000
        } else {
            !self.0
        };
        f32::from_bits(bits)
    }
}

/// Salience index: a BTreeMap sorted by salience score for efficient top-k retrieval.
pub struct SalienceIndex {
    inner: RwLock<SalienceInner>,
}

struct SalienceInner {
    tree: BTreeMap<OrderedF32, Vec<MemoryId>>,
}

impl SalienceIndex {
    /// Creates a new empty salience index.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(SalienceInner {
                tree: BTreeMap::new(),
            }),
        }
    }

    /// Insert a memory with its salience score.
    pub fn insert(&self, id: MemoryId, salience: f32) {
        let mut inner = self.inner.write();
        let key = OrderedF32::from_f32(salience);
        inner.tree.entry(key).or_default().push(id);
    }

    /// Update the salience score for a memory.
    pub fn update(&self, id: MemoryId, old: f32, new: f32) {
        self.remove(id, old);
        self.insert(id, new);
    }

    /// Get the top-k memories by salience (highest first).
    pub fn top_k(&self, k: usize) -> Vec<(MemoryId, f32)> {
        let inner = self.inner.read();
        let mut results = Vec::with_capacity(k);
        for (&key, ids) in inner.tree.iter().rev() {
            let salience = key.to_f32();
            for &id in ids.iter().rev() {
                results.push((id, salience));
                if results.len() >= k {
                    return results;
                }
            }
        }
        results
    }

    /// Remove a memory with a known salience score.
    pub fn remove(&self, id: MemoryId, salience: f32) {
        let mut inner = self.inner.write();
        let key = OrderedF32::from_f32(salience);
        if let Some(ids) = inner.tree.get_mut(&key) {
            ids.retain(|&i| i != id);
            if ids.is_empty() {
                inner.tree.remove(&key);
            }
        }
    }

    /// Get the salience score for a memory (linear scan: use sparingly).
    pub fn get_salience(&self, id: MemoryId) -> Option<f32> {
        let inner = self.inner.read();
        for (&key, ids) in &inner.tree {
            if ids.contains(&id) {
                return Some(key.to_f32());
            }
        }
        None
    }
}

/// Serializable snapshot of the salience index data.
#[derive(Serialize, Deserialize)]
struct SalienceSnapshot {
    /// Entries as (f32 salience, Vec<MemoryId>).
    entries: Vec<(f32, Vec<MemoryId>)>,
}

impl SalienceIndex {
    /// Save the salience index to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> MenteResult<()> {
        let inner = self.inner.read();
        let snapshot = SalienceSnapshot {
            entries: inner
                .tree
                .iter()
                .map(|(&k, v)| (k.to_f32(), v.clone()))
                .collect(),
        };
        let data =
            bincode::serialize(&snapshot).map_err(|e| MenteError::Serialization(e.to_string()))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load the salience index from a file (bincode, with JSON fallback for migration).
    pub fn load(path: &std::path::Path) -> MenteResult<Self> {
        let data = std::fs::read(path)?;
        let snapshot: SalienceSnapshot = bincode::deserialize(&data)
            .or_else(|_| serde_json::from_slice(&data))
            .map_err(|e| MenteError::Serialization(e.to_string()))?;

        let mut tree = BTreeMap::new();
        for (salience, ids) in snapshot.entries {
            tree.insert(OrderedF32::from_f32(salience), ids);
        }

        Ok(Self {
            inner: RwLock::new(SalienceInner { tree }),
        })
    }
}

impl Default for SalienceIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_top_k() {
        let idx = SalienceIndex::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        idx.insert(a, 0.3);
        idx.insert(b, 0.9);
        idx.insert(c, 0.6);

        let top = idx.top_k(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, b);
        assert_eq!(top[1].0, c);
    }

    #[test]
    fn test_update() {
        let idx = SalienceIndex::new();
        let a = MemoryId::new();
        idx.insert(a, 0.5);
        idx.update(a, 0.5, 0.9);

        let top = idx.top_k(1);
        assert_eq!(top[0].0, a);
        assert!((top[0].1 - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_remove() {
        let idx = SalienceIndex::new();
        let a = MemoryId::new();
        idx.insert(a, 0.5);
        idx.remove(a, 0.5);

        assert!(idx.top_k(10).is_empty());
    }

    #[test]
    fn test_ordered_f32_roundtrip() {
        let values = [0.0f32, 0.5, 1.0, -1.0, 0.001, 0.999];
        for &v in &values {
            let key = super::OrderedF32::from_f32(v);
            let back = key.to_f32();
            assert!((v - back).abs() < f32::EPSILON, "failed roundtrip for {v}");
        }
    }

    #[test]
    fn test_ordering() {
        // Verify that ordering through OrderedF32 preserves f32 ordering
        let vals = [-1.0f32, 0.0, 0.1, 0.5, 0.9, 1.0];
        let keys: Vec<_> = vals
            .iter()
            .map(|&v| super::OrderedF32::from_f32(v))
            .collect();
        let mut sorted_keys = keys.clone();
        sorted_keys.sort();
        assert_eq!(keys, sorted_keys);
    }
}
