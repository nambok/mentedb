//! Record map: which extent holds which memory, persisted at checkpoints.
//!
//! The engine keeps the whole locator to extent table in memory; every read
//! resolves its locator through it. The table is persisted at checkpoint time
//! and replayed forward from the WAL on open, so a cold open reads no records.
//!
//! # Anchoring
//!
//! Each checkpoint that changed the table appends a `Checkpoint` WAL entry
//! whose payload is a random 16 byte token, then writes the snapshot stamped
//! with the same token, then truncates the WAL to that entry. The table now is
//! the snapshot plus every WAL entry after the entry carrying its token. A
//! snapshot whose token is not in the WAL can never be matched again, because
//! tokens are never reused for a changed table. On any doubt (missing,
//! corrupt, unmatched, inconsistent with the file) open falls back to a scan of
//! the record chain.
//!
//! Each entry also carries a stamp, a value refreshed on every write to the
//! record, which lets derived caches (mode probes, salience bounds) persist
//! alongside and detect, without loading the record, that it changed since
//! they were captured.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;

use crate::record::{Extent, FreeSpace, PageId, RecordOp, le_u32, le_u64};

/// File name of the persisted record map inside the database directory.
pub const MEMORY_MAP_FILE: &str = "records_map.bin";

const MAGIC: &[u8; 8] = b"MDBRMAP1";
const FORMAT_VERSION: u32 = 1;

/// Checkpoint token carried in the `Checkpoint` WAL entry payload.
pub(crate) type Token = [u8; 16];

/// A memory the map knows about: its id, its locator, and its write stamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedMemory {
    pub id: MemoryId,
    pub page: PageId,
    /// Changes whenever the record is written. Equal stamps for the same
    /// locator mean no write was applied to it in between.
    pub stamp: u64,
}

/// A fresh random stamp.
pub(crate) fn random_u64() -> u64 {
    let (a, b) = uuid::Uuid::new_v4().as_u64_pair();
    a ^ b.rotate_left(17)
}

pub(crate) fn new_token() -> Token {
    *uuid::Uuid::new_v4().as_bytes()
}

/// The stamp of the WAL entry at `lsn` written while `token` anchored the
/// map. Deterministic, so the stamp a write gets live and the stamp a reopen
/// gives it replaying the WAL are the same value. LSNs only grow while a
/// token anchors the map, so (token, lsn) never repeats.
pub(crate) fn entry_stamp(token: &Token, lsn: u64) -> u64 {
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    let a = le_u64(&token[..8]);
    let b = le_u64(&token[8..]);
    mix(a ^ mix(b ^ mix(lsn)))
}

/// Where one record lives and what it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecordEntry {
    pub(crate) id: MemoryId,
    pub(crate) stamp: u64,
    pub(crate) extent: Extent,
}

/// The locator to extent table.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RecordTable {
    pub(crate) records: BTreeMap<u64, RecordEntry>,
    /// Next locator to hand out; above every locator ever used.
    pub(crate) next_locator: u64,
}

impl RecordTable {
    /// Apply one logged operation with the given stamp.
    pub(crate) fn apply(&mut self, op: &RecordOp, stamp: u64) {
        match op {
            RecordOp::Write {
                locator,
                id,
                extent,
                ..
            } => {
                self.records.insert(
                    *locator,
                    RecordEntry {
                        id: *id,
                        stamp,
                        extent: *extent,
                    },
                );
            }
            RecordOp::Free { locator, .. } => {
                self.records.remove(locator);
            }
        }
        self.next_locator = self.next_locator.max(op.locator() + 1);
    }

    /// Every mapped memory in locator order.
    pub(crate) fn memories(&self) -> Vec<MappedMemory> {
        self.records
            .iter()
            .map(|(&page, e)| MappedMemory {
                id: e.id,
                page: PageId(page),
                stamp: e.stamp,
            })
            .collect()
    }

    /// Highest block any record occupies, exclusive (1 when empty).
    pub(crate) fn max_end(&self) -> u64 {
        self.records
            .values()
            .map(|e| e.extent.end())
            .max()
            .unwrap_or(1)
    }

    /// The free space of a file of `end` blocks holding these records.
    pub(crate) fn free_space(&self, end: u64) -> MenteResult<FreeSpace> {
        FreeSpace::from_live(self.records.values().map(|e| e.extent).collect(), end)
    }
}

/// A persisted table and the checkpoint token it is anchored on.
#[derive(Debug, Clone)]
pub(crate) struct Snapshot {
    pub(crate) token: Token,
    pub(crate) table: RecordTable,
}

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn encode(token: &Token, t: &RecordTable) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64 + t.records.len() * 56);
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    buf.extend_from_slice(token);
    put_u64(&mut buf, t.next_locator);
    put_u64(&mut buf, t.records.len() as u64);
    for (&locator, e) in &t.records {
        put_u64(&mut buf, locator);
        buf.extend_from_slice(e.id.0.as_bytes());
        put_u64(&mut buf, e.stamp);
        put_u64(&mut buf, e.extent.start);
        put_u64(&mut buf, e.extent.blocks);
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
            .ok_or_else(|| MenteError::Storage("record map snapshot truncated".into()))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u64(&mut self) -> MenteResult<u64> {
        Ok(le_u64(self.take(8)?))
    }
}

pub(crate) fn decode(bytes: &[u8]) -> MenteResult<Snapshot> {
    let bad = |what: &str| MenteError::Storage(format!("record map snapshot {what}"));
    if bytes.len() < MAGIC.len() + 4 + 16 + 8 + 8 + 4 {
        return Err(bad("too short"));
    }
    let (body, crc_bytes) = bytes.split_at(bytes.len() - 4);
    if crc32fast::hash(body) != le_u32(crc_bytes) {
        return Err(bad("checksum mismatch"));
    }
    let mut r = Reader { buf: body, pos: 0 };
    if r.take(8)? != MAGIC {
        return Err(bad("magic mismatch"));
    }
    if le_u32(r.take(4)?) != FORMAT_VERSION {
        return Err(bad("version mismatch"));
    }
    let mut token = [0u8; 16];
    token.copy_from_slice(r.take(16)?);
    let next_locator = r.u64()?;
    let n = r.u64()?;
    let mut table = RecordTable {
        records: BTreeMap::new(),
        next_locator,
    };
    for _ in 0..n {
        let locator = r.u64()?;
        let id = uuid::Uuid::from_slice(r.take(16)?).map_err(|e| bad(&e.to_string()))?;
        let stamp = r.u64()?;
        let start = r.u64()?;
        let blocks = r.u64()?;
        if locator >= next_locator {
            return Err(bad("has a locator past its counter"));
        }
        table.records.insert(
            locator,
            RecordEntry {
                id: MemoryId(id),
                stamp,
                extent: Extent { start, blocks },
            },
        );
    }
    if r.pos != body.len() {
        return Err(bad("has trailing bytes"));
    }
    Ok(Snapshot { token, table })
}

/// Read the snapshot in `dir`. `Ok(None)` when there is none.
pub(crate) fn read_snapshot(dir: &Path) -> MenteResult<Option<Snapshot>> {
    match std::fs::read(dir.join(MEMORY_MAP_FILE)) {
        Ok(bytes) => decode(&bytes).map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Write the snapshot atomically: a reader sees the old file or the complete
/// new one, never a mix (temp file, fsync, rename).
pub(crate) fn write_snapshot(dir: &Path, token: &Token, table: &RecordTable) -> MenteResult<()> {
    let path = dir.join(MEMORY_MAP_FILE);
    let tmp = dir.join(format!("{MEMORY_MAP_FILE}.tmp"));
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&encode(token, table))?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Remove the snapshot, ignoring a missing file.
pub(crate) fn remove_snapshot(dir: &Path) {
    let _ = std::fs::remove_file(dir.join(MEMORY_MAP_FILE));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap() -> Snapshot {
        let mut table = RecordTable::default();
        for (loc, start) in [(1u64, 1u64), (3, 4)] {
            table.apply(
                &RecordOp::Write {
                    locator: loc,
                    id: MemoryId::new(),
                    extent: Extent { start, blocks: 3 },
                    payload: vec![],
                    remainder: None,
                    freed: None,
                },
                random_u64(),
            );
        }
        Snapshot {
            token: new_token(),
            table,
        }
    }

    #[test]
    fn snapshot_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let a = snap();
        assert_eq!(a.table.next_locator, 4);
        write_snapshot(dir.path(), &a.token, &a.table).unwrap();
        let b = read_snapshot(dir.path()).unwrap().unwrap();
        assert_eq!(a.token, b.token);
        assert_eq!(a.table, b.table);
        assert!(!dir.path().join(format!("{MEMORY_MAP_FILE}.tmp")).exists());
    }

    #[test]
    fn corrupt_snapshot_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let s = snap();
        write_snapshot(dir.path(), &s.token, &s.table).unwrap();
        let path = dir.path().join(MEMORY_MAP_FILE);
        let mut bytes = std::fs::read(&path).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        assert!(read_snapshot(dir.path()).is_err());
        std::fs::write(&path, &bytes[..10]).unwrap();
        assert!(read_snapshot(dir.path()).is_err());
        std::fs::remove_file(&path).unwrap();
        assert!(read_snapshot(dir.path()).unwrap().is_none());
    }

    #[test]
    fn free_space_follows_the_table() {
        let s = snap();
        assert_eq!(s.table.max_end(), 7);
        let f = s.table.free_space(10).unwrap();
        assert_eq!(f.free_blocks(), 3);
        assert!(s.table.free_space(6).is_err(), "extent past the end");
    }
}
