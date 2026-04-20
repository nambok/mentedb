//! Roaring bitmap tag index for fast set-based tag filtering.

use std::io::Cursor;

use ahash::HashMap;
use parking_lot::RwLock;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;

/// Roaring-bitmap-backed tag index.
///
/// Maps tag strings to roaring bitmaps for fast AND/OR/NOT set operations.
/// Uses an internal u32 offset for each MemoryId.
pub struct BitmapIndex {
    inner: RwLock<BitmapInner>,
}

struct BitmapInner {
    /// Tag name → bitmap of u32 offsets.
    tag_bitmaps: HashMap<String, RoaringBitmap>,
    /// MemoryId → internal u32 offset.
    id_to_offset: HashMap<MemoryId, u32>,
    /// Reverse: u32 offset → MemoryId.
    offset_to_id: Vec<MemoryId>,
}

impl BitmapIndex {
    /// Creates a new empty bitmap index.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(BitmapInner {
                tag_bitmaps: HashMap::default(),
                id_to_offset: HashMap::default(),
                offset_to_id: Vec::new(),
            }),
        }
    }

    /// Get or create the internal u32 offset for a MemoryId.
    fn ensure_offset(inner: &mut BitmapInner, id: MemoryId) -> u32 {
        if let Some(&offset) = inner.id_to_offset.get(&id) {
            return offset;
        }
        let offset = inner.offset_to_id.len() as u32;
        inner.id_to_offset.insert(id, offset);
        inner.offset_to_id.push(id);
        offset
    }

    /// Add a tag for the given memory id.
    pub fn add_tag(&self, id: MemoryId, tag: &str) {
        let mut inner = self.inner.write();
        let offset = Self::ensure_offset(&mut inner, id);
        inner
            .tag_bitmaps
            .entry(tag.to_string())
            .or_default()
            .insert(offset);
    }

    /// Remove a tag for the given memory id.
    pub fn remove_tag(&self, id: MemoryId, tag: &str) {
        let mut inner = self.inner.write();
        if let Some(&offset) = inner.id_to_offset.get(&id)
            && let Some(bm) = inner.tag_bitmaps.get_mut(tag)
        {
            bm.remove(offset);
        }
    }

    /// Get all memory ids that have the given tag.
    pub fn query_tag(&self, tag: &str) -> Vec<MemoryId> {
        let inner = self.inner.read();
        match inner.tag_bitmaps.get(tag) {
            Some(bm) => bm
                .iter()
                .filter_map(|offset| inner.offset_to_id.get(offset as usize).copied())
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get memory ids that have ALL given tags (intersection).
    pub fn query_tags_and(&self, tags: &[&str]) -> Vec<MemoryId> {
        let inner = self.inner.read();
        if tags.is_empty() {
            return Vec::new();
        }

        let mut result: Option<RoaringBitmap> = None;
        for tag in tags {
            match inner.tag_bitmaps.get(*tag) {
                Some(bm) => {
                    result = Some(match result {
                        Some(r) => r & bm,
                        None => bm.clone(),
                    });
                }
                None => return Vec::new(),
            }
        }

        match result {
            Some(bm) => bm
                .iter()
                .filter_map(|offset| inner.offset_to_id.get(offset as usize).copied())
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get memory ids that have ANY of the given tags (union).
    pub fn query_tags_or(&self, tags: &[&str]) -> Vec<MemoryId> {
        let inner = self.inner.read();
        if tags.is_empty() {
            return Vec::new();
        }

        let mut result = RoaringBitmap::new();
        for tag in tags {
            if let Some(bm) = inner.tag_bitmaps.get(*tag) {
                result |= bm;
            }
        }

        result
            .iter()
            .filter_map(|offset| inner.offset_to_id.get(offset as usize).copied())
            .collect()
    }

    /// Remove all tags for a given memory id.
    pub fn remove_all(&self, id: MemoryId) {
        let mut inner = self.inner.write();
        if let Some(&offset) = inner.id_to_offset.get(&id) {
            for bm in inner.tag_bitmaps.values_mut() {
                bm.remove(offset);
            }
        }
    }
}

/// Serializable snapshot of the bitmap index data.
#[derive(Serialize, Deserialize)]
struct BitmapSnapshot {
    /// Tag name → serialized RoaringBitmap bytes.
    tag_bitmaps: Vec<(String, Vec<u8>)>,
    id_to_offset: Vec<(MemoryId, u32)>,
    offset_to_id: Vec<MemoryId>,
}

impl BitmapIndex {
    /// Save the bitmap index to a binary file.
    pub fn save(&self, path: &std::path::Path) -> MenteResult<()> {
        let inner = self.inner.read();
        let mut tag_bitmaps = Vec::new();
        for (tag, bm) in &inner.tag_bitmaps {
            let mut buf = Vec::new();
            bm.serialize_into(&mut buf)
                .map_err(|e| MenteError::Serialization(e.to_string()))?;
            tag_bitmaps.push((tag.clone(), buf));
        }
        let snapshot = BitmapSnapshot {
            tag_bitmaps,
            id_to_offset: inner.id_to_offset.iter().map(|(&k, &v)| (k, v)).collect(),
            offset_to_id: inner.offset_to_id.clone(),
        };
        let data =
            bincode::serialize(&snapshot).map_err(|e| MenteError::Serialization(e.to_string()))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load the bitmap index from a file (bincode, with JSON fallback for migration).
    pub fn load(path: &std::path::Path) -> MenteResult<Self> {
        let data = std::fs::read(path)?;
        let snapshot: BitmapSnapshot = bincode::deserialize(&data)
            .or_else(|_| serde_json::from_slice(&data))
            .map_err(|e| MenteError::Serialization(e.to_string()))?;

        let mut tag_bitmaps = HashMap::default();
        for (tag, bytes) in snapshot.tag_bitmaps {
            let bm = RoaringBitmap::deserialize_from(&mut Cursor::new(bytes))
                .map_err(|e| MenteError::Serialization(e.to_string()))?;
            tag_bitmaps.insert(tag, bm);
        }

        let mut id_to_offset = HashMap::default();
        for (id, offset) in snapshot.id_to_offset {
            id_to_offset.insert(id, offset);
        }

        Ok(Self {
            inner: RwLock::new(BitmapInner {
                tag_bitmaps,
                id_to_offset,
                offset_to_id: snapshot.offset_to_id,
            }),
        })
    }
}

impl Default for BitmapIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_query_tag() {
        let idx = BitmapIndex::new();
        let id = MemoryId::new();
        idx.add_tag(id, "important");

        let results = idx.query_tag("important");
        assert_eq!(results, vec![id]);
    }

    #[test]
    fn test_query_empty_tag() {
        let idx = BitmapIndex::new();
        assert!(idx.query_tag("nonexistent").is_empty());
    }

    #[test]
    fn test_remove_tag() {
        let idx = BitmapIndex::new();
        let id = MemoryId::new();
        idx.add_tag(id, "foo");
        idx.remove_tag(id, "foo");

        assert!(idx.query_tag("foo").is_empty());
    }

    #[test]
    fn test_tags_and() {
        let idx = BitmapIndex::new();
        let a = MemoryId::new();
        let b = MemoryId::new();

        idx.add_tag(a, "x");
        idx.add_tag(a, "y");
        idx.add_tag(b, "x");

        let both = idx.query_tags_and(&["x", "y"]);
        assert_eq!(both, vec![a]);
    }

    #[test]
    fn test_tags_or() {
        let idx = BitmapIndex::new();
        let a = MemoryId::new();
        let b = MemoryId::new();

        idx.add_tag(a, "x");
        idx.add_tag(b, "y");

        let either = idx.query_tags_or(&["x", "y"]);
        assert_eq!(either.len(), 2);
    }

    #[test]
    fn test_remove_all() {
        let idx = BitmapIndex::new();
        let id = MemoryId::new();
        idx.add_tag(id, "a");
        idx.add_tag(id, "b");
        idx.remove_all(id);

        assert!(idx.query_tag("a").is_empty());
        assert!(idx.query_tag("b").is_empty());
    }
}
