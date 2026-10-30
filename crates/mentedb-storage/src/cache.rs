//! Record cache: decoded memory records, bounded by bytes, CLOCK eviction.
//!
//! The cache holds decoded [`MemoryNode`]s keyed by record locator, so a hit
//! costs neither a read nor a decode. Capacity is in bytes (each entry is
//! charged its encoded record size), which lets a host with spare memory size
//! it to hold an entire database: every scan after the first is then served
//! from memory. Entries are always clean; writes go to the file and the WAL
//! before the cache is updated.

use std::sync::atomic::{AtomicU64, Ordering};

use ahash::AHashMap;
use mentedb_core::MemoryNode;
use parking_lot::Mutex;

/// A snapshot of cache activity, for metrics. `hits / (hits + misses)` is the
/// hit ratio, the primary read performance signal.
#[derive(Debug, Clone, Copy, Default)]
pub struct BufferStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Records currently cached.
    pub resident_pages: u64,
    /// Bytes charged by the cached records.
    pub resident_bytes: u64,
    /// Configured capacity in bytes.
    pub capacity_bytes: u64,
}

struct Slot {
    locator: u64,
    node: MemoryNode,
    charge: u64,
    referenced: bool,
}

struct Inner {
    slots: Vec<Option<Slot>>,
    index: AHashMap<u64, usize>,
    vacant: Vec<usize>,
    hand: usize,
    used: u64,
    capacity: u64,
}

impl Inner {
    fn remove(&mut self, locator: u64) {
        if let Some(i) = self.index.remove(&locator)
            && let Some(slot) = self.slots[i].take()
        {
            self.used -= slot.charge;
            self.vacant.push(i);
        }
    }

    /// Evict until `need` more bytes fit. Every entry is evictable (no pins),
    /// so two sweeps always free enough when the cache is not empty.
    fn make_room(&mut self, need: u64, evictions: &AtomicU64) {
        let mut budget = self.slots.len() * 2 + 1;
        while self.used + need > self.capacity && !self.index.is_empty() && budget > 0 {
            budget -= 1;
            let i = self.hand;
            self.hand = (self.hand + 1) % self.slots.len();
            let evict = match self.slots[i].as_mut() {
                None => false,
                Some(slot) if slot.referenced => {
                    slot.referenced = false;
                    false
                }
                Some(_) => true,
            };
            if evict && let Some(slot) = self.slots[i].take() {
                self.index.remove(&slot.locator);
                self.used -= slot.charge;
                self.vacant.push(i);
                evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Thread safe, byte bounded record cache.
pub struct RecordCache {
    inner: Mutex<Inner>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl RecordCache {
    /// A cache holding up to `capacity` bytes of records. Zero disables it.
    pub fn new(capacity: u64) -> Self {
        Self {
            inner: Mutex::new(Inner {
                slots: Vec::new(),
                index: AHashMap::new(),
                vacant: Vec::new(),
                hand: 0,
                used: 0,
                capacity,
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// The cached node for `locator`, counting a hit or a miss.
    pub fn get(&self, locator: u64) -> Option<MemoryNode> {
        let mut inner = self.inner.lock();
        let found = inner.index.get(&locator).copied();
        match found.and_then(|i| inner.slots[i].as_mut()) {
            Some(slot) => {
                slot.referenced = true;
                let node = slot.node.clone();
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(node)
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Cache `node` for `locator`, charged `charge` bytes, replacing any
    /// previous entry. A record larger than the whole cache is not cached.
    pub fn insert(&self, locator: u64, node: MemoryNode, charge: u64) {
        let mut inner = self.inner.lock();
        inner.remove(locator);
        if charge > inner.capacity {
            return;
        }
        inner.make_room(charge, &self.evictions);
        if inner.used + charge > inner.capacity {
            return;
        }
        let slot = Slot {
            locator,
            node,
            charge,
            referenced: false,
        };
        let i = match inner.vacant.pop() {
            Some(i) => {
                inner.slots[i] = Some(slot);
                i
            }
            None => {
                inner.slots.push(Some(slot));
                inner.slots.len() - 1
            }
        };
        inner.index.insert(locator, i);
        inner.used += charge;
    }

    /// Drop the entry for `locator`, if cached.
    pub fn invalidate(&self, locator: u64) {
        self.inner.lock().remove(locator);
    }

    /// Change the capacity, evicting as needed to fit the new bound.
    pub fn set_capacity(&self, capacity: u64) {
        let mut inner = self.inner.lock();
        inner.capacity = capacity;
        inner.make_room(0, &self.evictions);
    }

    /// Configured capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.inner.lock().capacity
    }

    /// Activity counters and residency.
    pub fn stats(&self) -> BufferStats {
        let inner = self.inner.lock();
        BufferStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            resident_pages: inner.index.len() as u64,
            resident_bytes: inner.used,
            capacity_bytes: inner.capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::memory::MemoryType;
    use mentedb_core::types::AgentId;

    fn node(s: &str) -> MemoryNode {
        MemoryNode::new(AgentId::new(), MemoryType::Semantic, s.into(), vec![])
    }

    #[test]
    fn bounded_by_bytes() {
        let c = RecordCache::new(1000);
        for i in 0..10 {
            c.insert(i, node(&i.to_string()), 300);
        }
        let s = c.stats();
        assert!(s.resident_bytes <= 1000);
        assert_eq!(s.resident_pages, 3);
        assert!(s.evictions >= 7);
        // The newest entry is resident.
        assert_eq!(c.get(9).unwrap().content, "9");
    }

    #[test]
    fn referenced_entries_survive_a_sweep() {
        let c = RecordCache::new(300);
        c.insert(1, node("a"), 100);
        c.insert(2, node("b"), 100);
        c.insert(3, node("c"), 100);
        c.get(1);
        c.insert(4, node("d"), 100);
        assert!(c.get(1).is_some(), "recently used entry kept");
        assert!(c.get(2).is_none(), "unreferenced entry evicted");
    }

    #[test]
    fn replace_invalidate_resize() {
        let c = RecordCache::new(1000);
        c.insert(1, node("a"), 400);
        c.insert(1, node("b"), 500);
        assert_eq!(c.stats().resident_bytes, 500);
        assert_eq!(c.get(1).unwrap().content, "b");
        c.invalidate(1);
        assert!(c.get(1).is_none());
        c.insert(2, node("x"), 2000);
        assert!(c.get(2).is_none(), "larger than the cache");
        c.insert(3, node("y"), 600);
        c.set_capacity(100);
        assert_eq!(c.stats().resident_bytes, 0);
        let off = RecordCache::new(0);
        off.insert(1, node("a"), 1);
        assert!(off.get(1).is_none());
    }
}
