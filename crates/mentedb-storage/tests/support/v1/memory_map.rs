//! Persisted memory map: which page holds which memory.
//!
//! Opening a database used to rebuild the memory id to page id map by loading
//! every page, which on network storage costs minutes for large databases. The
//! map is instead persisted at checkpoint time and replayed forward from the
//! WAL on open, so a cold open reads only the pages the WAL touched.
//!
//! # Anchoring
//!
//! Each checkpoint appends a `Checkpoint` WAL entry whose payload is a random
//! 16 byte token, then writes the snapshot stamped with the same token, then
//! truncates the WAL to that entry. The snapshot is the map as of its
//! checkpoint entry, so the map now is the snapshot plus every WAL entry after
//! the entry carrying its token. A snapshot whose token is not in the WAL
//! (the WAL was truncated by a checkpoint or recovery that did not write a
//! snapshot, the snapshot write failed, or the file is from another database)
//! can never be matched again, because tokens are never reused for a changed
//! map. On any doubt (missing, corrupt, unmatched, inconsistent page count)
//! the caller falls back to a full page scan.
//!
//! The table is keyed by page, mirroring exactly what a full scan observes:
//! a page holds a memory when its last write decodes as one, and a freed page
//! holds none. Each entry also carries a stamp, a random value refreshed on
//! every write applied to that page, which lets derived caches (mode probes)
//! persist alongside and detect, without loading the page, that the memory
//! changed since they were captured.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;

use mentedb_core::MemoryNode;
use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;

use super::page::{PAGE_DATA_SIZE, PageId};
use super::wal::{WalEntry, WalEntryType};

/// File name of the persisted memory map inside the database directory.
pub const MEMORY_MAP_FILE: &str = "memory_map.bin";

const MAGIC: &[u8; 8] = b"MDBMMAP1";
const FORMAT_VERSION: u32 = 1;

/// Checkpoint token carried in the `Checkpoint` WAL entry payload.
pub(crate) type Token = [u8; 16];

/// A memory the map knows about: its id, its page, and the page's write stamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedMemory {
    pub id: MemoryId,
    pub page: PageId,
    /// Changes whenever the page is written. Equal stamps for the same page
    /// mean no write was applied to it in between.
    pub stamp: u64,
}

/// Decode a memory node from a page's data region, with exactly the rules
/// [`super::StorageEngine::load_memory`] applies after the page type check.
/// `data` may be shorter than a page; the missing tail reads as zeros.
pub(crate) fn decode_memory(data: &[u8]) -> MenteResult<MemoryNode> {
    let data = &data[..data.len().min(PAGE_DATA_SIZE)];
    let mut prefix = [0u8; 4];
    let n = data.len().min(4);
    prefix[..n].copy_from_slice(&data[..n]);
    let len = u32::from_le_bytes(prefix) as usize;
    if len == 0 || len + 4 > PAGE_DATA_SIZE {
        return Err(MenteError::Storage(format!(
            "invalid memory node length prefix: {len}"
        )));
    }
    if 4 + len > data.len() {
        // The page tail is zero filled, and JSON never ends in NUL bytes, so
        // the page would not decode either.
        return Err(MenteError::Serialization(
            "memory node runs past the written data".into(),
        ));
    }
    serde_json::from_slice(&data[4..4 + len]).map_err(|e| MenteError::Serialization(e.to_string()))
}

/// A fresh random stamp or token half.
fn random_u64() -> u64 {
    let (a, b) = uuid::Uuid::new_v4().as_u64_pair();
    a ^ b.rotate_left(17)
}

pub(crate) fn new_token() -> Token {
    *uuid::Uuid::new_v4().as_bytes()
}

/// The stamp of the WAL entry at `lsn` written while `token` anchored the
/// map. Deterministic, so the stamp a write gets live, the stamp a checkpoint
/// gives it rolling the map forward, and the stamp a reopen gives it replaying
/// the WAL are the same value. LSNs only grow while a token anchors the map,
/// so (token, lsn) never repeats.
pub(crate) fn entry_stamp(token: &Token, lsn: u64) -> u64 {
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    let mut a = [0u8; 8];
    let mut b = [0u8; 8];
    a.copy_from_slice(&token[..8]);
    b.copy_from_slice(&token[8..]);
    mix(u64::from_le_bytes(a) ^ mix(u64::from_le_bytes(b) ^ mix(lsn)))
}

/// The page level view of which memories live where.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MemoryTable {
    /// page -> (memory id, write stamp), for pages holding a decodable memory.
    pages: BTreeMap<u64, (MemoryId, u64)>,
    /// Pages whose last logged operation freed them.
    free: BTreeSet<u64>,
}

impl MemoryTable {
    /// Record that `page` now holds `id` (or no memory when `None`), written
    /// with `stamp`.
    pub(crate) fn set_page(&mut self, page: u64, id: Option<MemoryId>, stamp: u64) {
        self.free.remove(&page);
        match id {
            Some(id) => {
                self.pages.insert(page, (id, stamp));
            }
            None => {
                self.pages.remove(&page);
            }
        }
    }

    /// Record that `page` was freed.
    pub(crate) fn set_free(&mut self, page: u64) {
        self.pages.remove(&page);
        self.free.insert(page);
    }

    /// Record a page found by a full scan. There is no WAL entry to derive a
    /// stamp from, so it gets a fresh random one.
    pub(crate) fn set_scanned(&mut self, page: u64, id: MemoryId) {
        self.set_page(page, Some(id), random_u64());
    }

    /// Apply one WAL entry written under `token`. Returns true when the entry
    /// can change the map.
    pub(crate) fn apply(&mut self, entry: &WalEntry, token: &Token) -> bool {
        match entry.entry_type {
            WalEntryType::PageWrite => {
                let id = decode_memory(&entry.data).ok().map(|n| n.id);
                self.set_page(entry.page_id, id, entry_stamp(token, entry.lsn));
                true
            }
            WalEntryType::PageFree => {
                self.set_free(entry.page_id);
                true
            }
            WalEntryType::Checkpoint | WalEntryType::Commit => false,
        }
    }

    /// Pages whose last logged operation freed them.
    pub(crate) fn free_pages(&self) -> &BTreeSet<u64> {
        &self.free
    }

    /// Highest page id referenced, if any.
    fn max_page(&self) -> Option<u64> {
        let a = self.pages.keys().next_back().copied();
        let b = self.free.iter().next_back().copied();
        a.max(b)
    }

    /// Every mapped memory in page order, the order a full scan returns.
    pub(crate) fn memories(&self) -> Vec<MappedMemory> {
        self.pages
            .iter()
            .map(|(&page, &(id, stamp))| MappedMemory {
                id,
                page: PageId(page),
                stamp,
            })
            .collect()
    }

    /// The memory and stamp recorded for `page`.
    pub(crate) fn get(&self, page: u64) -> Option<(MemoryId, u64)> {
        self.pages.get(&page).copied()
    }
}

/// The memory table as of the checkpoint entry carrying `token`.
#[derive(Debug, Clone)]
pub(crate) struct MapAnchor {
    pub(crate) token: Token,
    /// Page count when the snapshot was taken; the page file never shrinks,
    /// so a smaller count on open means the files do not belong together.
    pub(crate) page_count: u64,
    pub(crate) table: MemoryTable,
    /// Writes since the anchor's checkpoint, recorded as they happen (page ->
    /// memory and stamp, None when freed or not a memory), so stamp lookups
    /// see the current write. Not persisted: the WAL holds the same writes and
    /// rolling forward reproduces these exact stamps. A write path that does
    /// not record here only makes lookups older, which callers treat as stale.
    pub(crate) pending: std::collections::HashMap<u64, Option<(MemoryId, u64)>>,
}

impl MapAnchor {
    /// Index of the last `Checkpoint` entry carrying this anchor's token.
    fn position(&self, entries: &[WalEntry]) -> Option<usize> {
        entries
            .iter()
            .rposition(|e| e.entry_type == WalEntryType::Checkpoint && e.data == self.token)
    }

    /// Roll the table forward over the entries after this anchor's checkpoint.
    /// Returns None when the anchor's checkpoint is not in `entries` (the map
    /// cannot be derived), else whether any entry changed the table.
    pub(crate) fn advance(&mut self, entries: &[WalEntry]) -> Option<bool> {
        let pos = self.position(entries)?;
        let mut changed = false;
        for e in &entries[pos + 1..] {
            changed |= self.table.apply(e, &self.token);
        }
        self.pending.clear();
        Some(changed)
    }

    /// Record a write logged at `lsn` under this anchor.
    pub(crate) fn note_write(&mut self, page: u64, id: Option<MemoryId>, lsn: u64) {
        let stamp = entry_stamp(&self.token, lsn);
        self.pending.insert(page, id.map(|i| (i, stamp)));
    }

    /// The memory and stamp `page` holds now.
    pub(crate) fn current(&self, page: u64) -> Option<(MemoryId, u64)> {
        match self.pending.get(&page) {
            Some(v) => *v,
            None => self.table.get(page),
        }
    }

    /// Whether the snapshot is internally consistent with a page file of
    /// `page_count` pages.
    pub(crate) fn fits(&self, page_count: u64) -> bool {
        self.page_count <= page_count
            && self.table.max_page().is_none_or(|m| m < self.page_count)
            && !self.table.pages.contains_key(&0)
            && !self.table.free.contains(&0)
    }
}

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn encode(anchor: &MapAnchor) -> Vec<u8> {
    let t = &anchor.table;
    let mut buf = Vec::with_capacity(64 + t.pages.len() * 32 + t.free.len() * 8);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    buf.extend_from_slice(&anchor.token);
    put_u64(&mut buf, anchor.page_count);
    put_u64(&mut buf, t.pages.len() as u64);
    for (&page, &(id, stamp)) in &t.pages {
        put_u64(&mut buf, page);
        buf.extend_from_slice(id.0.as_bytes());
        put_u64(&mut buf, stamp);
    }
    put_u64(&mut buf, t.free.len() as u64);
    for &page in &t.free {
        put_u64(&mut buf, page);
    }
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> MenteResult<&[u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| MenteError::Storage("memory map snapshot truncated".into()))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u64(&mut self) -> MenteResult<u64> {
        let mut b = [0u8; 8];
        b.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(b))
    }
}

fn decode(bytes: &[u8]) -> MenteResult<MapAnchor> {
    let bad = |what: &str| MenteError::Storage(format!("memory map snapshot {what}"));
    if bytes.len() < MAGIC.len() + 4 + 16 + 8 + 8 + 8 + 4 {
        return Err(bad("too short"));
    }
    let (body, crc_bytes) = bytes.split_at(bytes.len() - 4);
    let mut crc = [0u8; 4];
    crc.copy_from_slice(crc_bytes);
    if crc32fast::hash(body) != u32::from_le_bytes(crc) {
        return Err(bad("checksum mismatch"));
    }
    let mut r = Reader { buf: body, pos: 0 };
    if r.take(8)? != MAGIC {
        return Err(bad("magic mismatch"));
    }
    let mut v = [0u8; 4];
    v.copy_from_slice(r.take(4)?);
    if u32::from_le_bytes(v) != FORMAT_VERSION {
        return Err(bad("version mismatch"));
    }
    let mut token = [0u8; 16];
    token.copy_from_slice(r.take(16)?);
    let page_count = r.u64()?;
    let n_pages = r.u64()?;
    let mut table = MemoryTable::default();
    for _ in 0..n_pages {
        let page = r.u64()?;
        let id = uuid::Uuid::from_slice(r.take(16)?).map_err(|e| bad(&e.to_string()))?;
        let stamp = r.u64()?;
        table.pages.insert(page, (MemoryId(id), stamp));
    }
    let n_free = r.u64()?;
    for _ in 0..n_free {
        table.free.insert(r.u64()?);
    }
    if r.pos != body.len() {
        return Err(bad("has trailing bytes"));
    }
    Ok(MapAnchor {
        token,
        page_count,
        table,
        pending: Default::default(),
    })
}

/// Read the snapshot in `dir`. `Ok(None)` when there is none.
pub(crate) fn read_snapshot(dir: &Path) -> MenteResult<Option<MapAnchor>> {
    match std::fs::read(dir.join(MEMORY_MAP_FILE)) {
        Ok(bytes) => decode(&bytes).map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Write the snapshot atomically: a reader sees the old file or the complete
/// new one, never a mix (temp file, fsync, rename).
pub(crate) fn write_snapshot(dir: &Path, anchor: &MapAnchor) -> MenteResult<()> {
    let path = dir.join(MEMORY_MAP_FILE);
    let tmp = dir.join(format!("{MEMORY_MAP_FILE}.tmp"));
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&encode(anchor))?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Remove the snapshot, ignoring a missing file.
pub(crate) fn remove_snapshot(dir: &Path) {
    let _ = std::fs::remove_file(dir.join(MEMORY_MAP_FILE));
}
