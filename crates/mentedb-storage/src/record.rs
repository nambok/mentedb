//! Record file: variable length memory records packed into 512 byte blocks.
//!
//! # Layout
//!
//! ```text
//! block 0            file header (magic, version, block size, CRC)
//! block 1..end       extents, back to back
//! ```
//!
//! An extent is a run of contiguous blocks that starts with a 48 byte header:
//!
//! ```text
//! [magic "MREC"][kind u8][pad 3][payload len u32][blocks u32]
//! [locator u64][memory id 16][payload crc u32][header crc u32]
//! ```
//!
//! A live extent holds one memory record (header plus encoded payload, padded
//! to whole blocks); a free extent holds nothing but its header. Extents tile
//! the file exactly: the extent at block `b` is followed by the extent at
//! `b + blocks`. Every write keeps that chain intact (a new record carved from
//! a free extent writes a free header for the remainder) and every write is
//! WAL logged as the exact bytes it puts on disk, so after replay a scan can
//! always walk the chain from block 1 without ever looking inside free space,
//! where stale headers of dead records may linger.
//!
//! A record's size on disk is its encoded size rounded up to the next block,
//! so a typical memory (1024 dimension embedding plus a few hundred bytes of
//! text) occupies about 5 KB instead of a whole 64 KB page.
//!
//! # Locators
//!
//! A record is addressed by its locator ([`PageId`]), a stable id that never
//! changes for the life of the record. The engine keeps a locator to extent
//! table in memory (persisted as the record map snapshot), so a record that
//! outgrows its extent moves without its locator, and every cache or snapshot
//! keyed by locator stays valid.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;

use mentedb_core::MemoryNode;
use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;

/// Allocation unit of the record file. Records are rounded up to whole blocks,
/// so the average waste is half a block per record.
pub const BLOCK_SIZE: usize = 512;

/// Bytes of the header at the start of every extent.
pub const EXTENT_HEADER_SIZE: usize = 48;

/// File name of the record file inside the database directory.
pub const DATA_FILE: &str = "records.db";

/// File name of the record WAL inside the database directory.
pub const WAL_FILE: &str = "records.wal";

const FILE_MAGIC: &[u8; 8] = b"MDBREC01";
const FILE_VERSION: u32 = 1;
const EXTENT_MAGIC: &[u8; 4] = b"MREC";

/// Payload encoding tag: binary f32 embedding followed by the node as JSON
/// with an empty embedding.
const PAYLOAD_V1: u8 = 1;

/// Stable locator of a memory record.
///
/// Historically this was the index of the 64 KB page holding the memory, and
/// databases migrated from that layout keep those numbers, so every persisted
/// reference to a page id stays valid. It is now an opaque id: records are
/// addressed through the engine's locator table, not by file offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId(pub u64);

/// A run of `blocks` contiguous blocks starting at block `start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Extent {
    pub(crate) start: u64,
    pub(crate) blocks: u64,
}

impl Extent {
    pub(crate) fn end(&self) -> u64 {
        self.start + self.blocks
    }

    pub(crate) fn offset(&self) -> u64 {
        self.start * BLOCK_SIZE as u64
    }

    pub(crate) fn byte_len(&self) -> usize {
        self.blocks as usize * BLOCK_SIZE
    }
}

/// Blocks needed for a record whose payload is `payload_len` bytes.
pub(crate) fn blocks_for(payload_len: usize) -> u64 {
    (EXTENT_HEADER_SIZE + payload_len).div_ceil(BLOCK_SIZE) as u64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum ExtentKind {
    Live = 1,
    Free = 2,
}

/// A decoded extent header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExtentHeader {
    pub(crate) kind: ExtentKind,
    pub(crate) len: u32,
    pub(crate) blocks: u32,
    pub(crate) locator: u64,
    pub(crate) id: MemoryId,
    pub(crate) payload_crc: u32,
}

impl ExtentHeader {
    pub(crate) fn encode(&self) -> [u8; EXTENT_HEADER_SIZE] {
        let mut b = [0u8; EXTENT_HEADER_SIZE];
        b[0..4].copy_from_slice(EXTENT_MAGIC);
        b[4] = self.kind as u8;
        b[8..12].copy_from_slice(&self.len.to_le_bytes());
        b[12..16].copy_from_slice(&self.blocks.to_le_bytes());
        b[16..24].copy_from_slice(&self.locator.to_le_bytes());
        b[24..40].copy_from_slice(self.id.0.as_bytes());
        b[40..44].copy_from_slice(&self.payload_crc.to_le_bytes());
        let crc = crc32fast::hash(&b[..44]);
        b[44..48].copy_from_slice(&crc.to_le_bytes());
        b
    }

    /// Decode a header, or None when the bytes are not a valid header.
    pub(crate) fn decode(b: &[u8]) -> Option<Self> {
        if b.len() < EXTENT_HEADER_SIZE || &b[0..4] != EXTENT_MAGIC {
            return None;
        }
        if crc32fast::hash(&b[..44]) != le_u32(&b[44..48]) {
            return None;
        }
        let kind = match b[4] {
            1 => ExtentKind::Live,
            2 => ExtentKind::Free,
            _ => return None,
        };
        let blocks = le_u32(&b[12..16]);
        if blocks == 0 {
            return None;
        }
        let id = uuid::Uuid::from_slice(&b[24..40]).ok()?;
        Some(Self {
            kind,
            len: le_u32(&b[8..12]),
            blocks,
            locator: le_u64(&b[16..24]),
            id: MemoryId(id),
            payload_crc: le_u32(&b[40..44]),
        })
    }

    pub(crate) fn free(blocks: u64) -> Self {
        Self {
            kind: ExtentKind::Free,
            len: 0,
            blocks: blocks as u32,
            locator: 0,
            id: MemoryId(uuid::Uuid::nil()),
            payload_crc: 0,
        }
    }
}

pub(crate) fn le_u32(b: &[u8]) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(&b[..4]);
    u32::from_le_bytes(a)
}

pub(crate) fn le_u64(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    u64::from_le_bytes(a)
}

/// The full on-disk image of a live extent: header, payload, zero padding.
pub(crate) fn live_image(locator: u64, id: MemoryId, payload: &[u8], blocks: u64) -> Vec<u8> {
    let header = ExtentHeader {
        kind: ExtentKind::Live,
        len: payload.len() as u32,
        blocks: blocks as u32,
        locator,
        id,
        payload_crc: crc32fast::hash(payload),
    };
    let mut buf = vec![0u8; blocks as usize * BLOCK_SIZE];
    buf[..EXTENT_HEADER_SIZE].copy_from_slice(&header.encode());
    buf[EXTENT_HEADER_SIZE..EXTENT_HEADER_SIZE + payload.len()].copy_from_slice(payload);
    buf
}

/// Validate a live extent read from disk and return its payload.
pub(crate) fn live_payload(buf: &[u8], locator: u64) -> MenteResult<&[u8]> {
    let header = ExtentHeader::decode(buf)
        .ok_or_else(|| MenteError::Storage(format!("record {locator}: invalid extent header")))?;
    if header.kind != ExtentKind::Live || header.locator != locator {
        return Err(MenteError::Storage(format!(
            "record {locator}: extent holds {:?} record {}",
            header.kind, header.locator
        )));
    }
    let end = EXTENT_HEADER_SIZE + header.len as usize;
    if end > buf.len() {
        return Err(MenteError::Storage(format!(
            "record {locator}: payload runs past its extent"
        )));
    }
    let payload = &buf[EXTENT_HEADER_SIZE..end];
    if crc32fast::hash(payload) != header.payload_crc {
        return Err(MenteError::Storage(format!(
            "record {locator}: payload checksum mismatch"
        )));
    }
    Ok(payload)
}

// ---- payload codec ----

/// Encode a memory node as a record payload.
///
/// The embedding is stored as raw little endian f32 values (4 bytes per
/// dimension, exact) instead of JSON text (about 12 bytes per dimension), and
/// the rest of the node as JSON, so new fields keep round tripping.
pub(crate) fn encode_node(node: &MemoryNode) -> MenteResult<Vec<u8>> {
    let mut rest = node.clone();
    let embedding = std::mem::take(&mut rest.embedding);
    let json = serde_json::to_vec(&rest).map_err(|e| MenteError::Serialization(e.to_string()))?;
    let mut buf = Vec::with_capacity(1 + 4 + embedding.len() * 4 + json.len());
    buf.push(PAYLOAD_V1);
    buf.extend_from_slice(&(embedding.len() as u32).to_le_bytes());
    for v in &embedding {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.extend_from_slice(&json);
    Ok(buf)
}

/// Decode a record payload written by [`encode_node`].
pub(crate) fn decode_node(payload: &[u8]) -> MenteResult<MemoryNode> {
    let bad = |what: &str| MenteError::Serialization(format!("record payload {what}"));
    if payload.len() < 5 {
        return Err(bad("too short"));
    }
    if payload[0] != PAYLOAD_V1 {
        return Err(bad("has an unknown encoding"));
    }
    let dim = le_u32(&payload[1..5]) as usize;
    let emb_end = dim
        .checked_mul(4)
        .and_then(|n| n.checked_add(5))
        .filter(|&e| e <= payload.len())
        .ok_or_else(|| bad("embedding runs past the payload"))?;
    let mut node: MemoryNode = serde_json::from_slice(&payload[emb_end..])
        .map_err(|e| MenteError::Serialization(e.to_string()))?;
    node.embedding = payload[5..emb_end]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Ok(node)
}

// ---- file header ----

/// The block 0 image of a new record file. Written once at creation and never
/// rewritten, so it cannot be torn by a later crash.
pub(crate) fn file_header() -> Vec<u8> {
    let mut b = vec![0u8; BLOCK_SIZE];
    b[0..8].copy_from_slice(FILE_MAGIC);
    b[8..12].copy_from_slice(&FILE_VERSION.to_le_bytes());
    b[12..16].copy_from_slice(&(BLOCK_SIZE as u32).to_le_bytes());
    let crc = crc32fast::hash(&b[..16]);
    b[16..20].copy_from_slice(&crc.to_le_bytes());
    b
}

/// Check a record file header.
pub(crate) fn check_file_header(b: &[u8]) -> MenteResult<()> {
    if b.len() < 20 || &b[0..8] != FILE_MAGIC {
        return Err(MenteError::Storage("invalid record file magic".into()));
    }
    if crc32fast::hash(&b[..16]) != le_u32(&b[16..20]) {
        return Err(MenteError::Storage(
            "record file header checksum mismatch".into(),
        ));
    }
    let version = le_u32(&b[8..12]);
    if version != FILE_VERSION {
        return Err(MenteError::Storage(format!(
            "unsupported record file version {version}"
        )));
    }
    let block = le_u32(&b[12..16]) as usize;
    if block != BLOCK_SIZE {
        return Err(MenteError::Storage(format!(
            "record file block size {block} does not match {BLOCK_SIZE}"
        )));
    }
    Ok(())
}

// ---- positional I/O ----

/// Read exactly `buf.len()` bytes at `offset` without moving a shared cursor,
/// so concurrent readers need no lock around the file.
pub(crate) fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
    }
    #[cfg(windows)]
    {
        let mut done = 0usize;
        while done < buf.len() {
            let n = std::os::windows::fs::FileExt::seek_read(
                file,
                &mut buf[done..],
                offset + done as u64,
            )?;
            if n == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            done += n;
        }
        Ok(())
    }
}

/// Write all of `buf` at `offset`.
pub(crate) fn write_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::write_all_at(file, buf, offset)
    }
    #[cfg(windows)]
    {
        let mut done = 0usize;
        while done < buf.len() {
            let n = std::os::windows::fs::FileExt::seek_write(
                file,
                &buf[done..],
                offset + done as u64,
            )?;
            if n == 0 {
                return Err(std::io::ErrorKind::WriteZero.into());
            }
            done += n;
        }
        Ok(())
    }
}

/// Fsync a directory so renames and creations inside it are durable.
pub(crate) fn sync_dir(dir: &std::path::Path) -> MenteResult<()> {
    #[cfg(unix)]
    {
        File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
    Ok(())
}

// ---- logged operations ----

/// One logged change to the record file. The WAL entry carries everything
/// needed to redo it byte for byte, so replay needs no other state.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RecordOp {
    /// Write record `locator` (memory `id`) into `extent`. `remainder` is the
    /// free tail of the free extent the record was carved from, and `freed` is
    /// the record's previous extent when it moved to a larger one.
    Write {
        locator: u64,
        id: MemoryId,
        extent: Extent,
        payload: Vec<u8>,
        remainder: Option<Extent>,
        freed: Option<Extent>,
    },
    /// Delete record `locator`, freeing `extent`.
    Free { locator: u64, extent: Extent },
}

const OP_VERSION: u8 = 1;

fn put_extent(buf: &mut Vec<u8>, e: Option<Extent>) {
    match e {
        Some(e) => {
            buf.push(1);
            buf.extend_from_slice(&e.start.to_le_bytes());
            buf.extend_from_slice(&e.blocks.to_le_bytes());
        }
        None => buf.push(0),
    }
}

struct OpReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl OpReader<'_> {
    fn take(&mut self, n: usize) -> MenteResult<&[u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or_else(|| MenteError::Storage("record WAL entry truncated".into()))?;
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u64(&mut self) -> MenteResult<u64> {
        Ok(le_u64(self.take(8)?))
    }

    fn extent(&mut self) -> MenteResult<Option<Extent>> {
        match self.take(1)?[0] {
            0 => Ok(None),
            _ => {
                let start = self.u64()?;
                let blocks = self.u64()?;
                Ok(Some(Extent { start, blocks }))
            }
        }
    }
}

impl RecordOp {
    pub(crate) fn locator(&self) -> u64 {
        match self {
            RecordOp::Write { locator, .. } | RecordOp::Free { locator, .. } => *locator,
        }
    }

    /// Serialize for a WAL entry payload (the locator travels as the entry's
    /// page id).
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut buf = vec![OP_VERSION];
        match self {
            RecordOp::Write {
                id,
                extent,
                payload,
                remainder,
                freed,
                ..
            } => {
                buf.extend_from_slice(id.0.as_bytes());
                put_extent(&mut buf, Some(*extent));
                put_extent(&mut buf, *remainder);
                put_extent(&mut buf, *freed);
                buf.extend_from_slice(&(payload.len() as u64).to_le_bytes());
                buf.extend_from_slice(payload);
            }
            RecordOp::Free { extent, .. } => {
                put_extent(&mut buf, Some(*extent));
            }
        }
        buf
    }

    /// Decode a WAL entry of type RecordWrite (`write`) or RecordFree.
    pub(crate) fn decode(write: bool, locator: u64, data: &[u8]) -> MenteResult<Self> {
        let mut r = OpReader { b: data, pos: 0 };
        if r.take(1)?[0] != OP_VERSION {
            return Err(MenteError::Storage(
                "record WAL entry has an unknown version".into(),
            ));
        }
        let missing = || MenteError::Storage("record WAL entry without an extent".into());
        if write {
            let id = uuid::Uuid::from_slice(r.take(16)?)
                .map_err(|e| MenteError::Storage(e.to_string()))?;
            let extent = r.extent()?.ok_or_else(missing)?;
            let remainder = r.extent()?;
            let freed = r.extent()?;
            let len = r.u64()? as usize;
            let payload = r.take(len)?.to_vec();
            Ok(RecordOp::Write {
                locator,
                id: MemoryId(id),
                extent,
                payload,
                remainder,
                freed,
            })
        } else {
            let extent = r.extent()?.ok_or_else(missing)?;
            Ok(RecordOp::Free { locator, extent })
        }
    }

    /// Apply the operation's bytes to the record file. Idempotent, so WAL
    /// replay can redo any suffix of the log in order.
    pub(crate) fn redo(&self, file: &File) -> MenteResult<()> {
        match self {
            RecordOp::Write {
                locator,
                id,
                extent,
                payload,
                remainder,
                freed,
            } => {
                let image = live_image(*locator, *id, payload, extent.blocks);
                write_at(file, &image, extent.offset())?;
                if let Some(r) = remainder {
                    write_at(file, &ExtentHeader::free(r.blocks).encode(), r.offset())?;
                }
                if let Some(f) = freed {
                    write_at(file, &ExtentHeader::free(f.blocks).encode(), f.offset())?;
                }
            }
            RecordOp::Free { extent, .. } => {
                write_at(
                    file,
                    &ExtentHeader::free(extent.blocks).encode(),
                    extent.offset(),
                )?;
            }
        }
        Ok(())
    }
}

// ---- free space ----

/// Free extents, coalesced, indexed by start (for merging neighbours) and by
/// size (for best fit allocation).
#[derive(Debug, Default, Clone)]
pub(crate) struct FreeSpace {
    by_start: BTreeMap<u64, u64>,
    by_size: BTreeSet<(u64, u64)>,
}

/// Where a new extent of a given size will go, decided before the WAL append
/// and committed after it, so a failed append leaves the allocator untouched.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Placement {
    pub(crate) extent: Extent,
    /// The free tail left in the free extent the record was carved from.
    pub(crate) remainder: Option<Extent>,
    /// The free extent consumed, or None when the file grows.
    source: Option<Extent>,
}

impl FreeSpace {
    /// Free blocks in total.
    pub(crate) fn free_blocks(&self) -> u64 {
        self.by_start.values().sum()
    }

    /// Return an extent to free space, merging it with free neighbours.
    pub(crate) fn insert(&mut self, extent: Extent) {
        let mut start = extent.start;
        let mut blocks = extent.blocks;
        if let Some((&ps, &pb)) = self.by_start.range(..start).next_back()
            && ps + pb == start
        {
            self.remove(ps, pb);
            start = ps;
            blocks += pb;
        }
        if let Some(&nb) = self.by_start.get(&(start + blocks)) {
            self.remove(start + blocks, nb);
            blocks += nb;
        }
        self.by_start.insert(start, blocks);
        self.by_size.insert((blocks, start));
    }

    fn remove(&mut self, start: u64, blocks: u64) {
        self.by_start.remove(&start);
        self.by_size.remove(&(blocks, start));
    }

    /// Decide where `blocks` blocks go: the smallest free extent that fits,
    /// else the end of the file.
    pub(crate) fn place(&self, blocks: u64, end: u64) -> Placement {
        match self.by_size.range((blocks, 0)..).next() {
            Some(&(size, start)) => Placement {
                extent: Extent { start, blocks },
                remainder: (size > blocks).then_some(Extent {
                    start: start + blocks,
                    blocks: size - blocks,
                }),
                source: Some(Extent {
                    start,
                    blocks: size,
                }),
            },
            None => Placement {
                extent: Extent { start: end, blocks },
                remainder: None,
                source: None,
            },
        }
    }

    /// Commit a placement made by [`Self::place`]; returns the new file end.
    pub(crate) fn commit(&mut self, p: &Placement, end: u64) -> u64 {
        match p.source {
            Some(src) => {
                self.remove(src.start, src.blocks);
                if let Some(r) = p.remainder {
                    self.by_start.insert(r.start, r.blocks);
                    self.by_size.insert((r.blocks, r.start));
                }
                end
            }
            None => end.max(p.extent.end()),
        }
    }

    /// Free space implied by a set of live extents in a file of `end` blocks:
    /// every gap between them. Errors when two extents overlap or one runs
    /// past the end, which means the table does not match the file.
    pub(crate) fn from_live(mut live: Vec<Extent>, end: u64) -> MenteResult<Self> {
        live.sort_by_key(|e| e.start);
        let mut free = FreeSpace::default();
        let mut pos = 1u64;
        for e in live {
            if e.start < pos || e.blocks == 0 {
                return Err(MenteError::Storage(format!(
                    "record extents overlap at block {}",
                    e.start
                )));
            }
            if e.start > pos {
                free.insert(Extent {
                    start: pos,
                    blocks: e.start - pos,
                });
            }
            pos = e.end();
        }
        if pos > end {
            return Err(MenteError::Storage(format!(
                "record extent ends at block {pos}, past the file end {end}"
            )));
        }
        if end > pos {
            free.insert(Extent {
                start: pos,
                blocks: end - pos,
            });
        }
        Ok(free)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::memory::MemoryType;
    use mentedb_core::types::AgentId;

    #[test]
    fn header_roundtrip_and_corruption() {
        let h = ExtentHeader {
            kind: ExtentKind::Live,
            len: 1234,
            blocks: 4,
            locator: 99,
            id: MemoryId::new(),
            payload_crc: 7,
        };
        let mut b = h.encode();
        assert_eq!(ExtentHeader::decode(&b), Some(h));
        b[10] ^= 1;
        assert_eq!(ExtentHeader::decode(&b), None);
    }

    #[test]
    fn payload_roundtrip_is_exact_and_compact() {
        let mut node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            "x".repeat(300),
            (0..1024).map(|i| (i as f32).sin() / 7.0).collect(),
        );
        node.embedding[3] = f32::NAN;
        let p = encode_node(&node).unwrap();
        let json = serde_json::to_vec(&{
            let mut n = node.clone();
            n.embedding[3] = 0.0;
            n
        })
        .unwrap();
        assert!(
            p.len() * 2 < json.len(),
            "{} vs json {}",
            p.len(),
            json.len()
        );
        let back = decode_node(&p).unwrap();
        assert_eq!(back.id, node.id);
        assert_eq!(back.content, node.content);
        assert_eq!(back.embedding.len(), 1024);
        for (a, b) in back.embedding.iter().zip(&node.embedding) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn op_roundtrip() {
        let op = RecordOp::Write {
            locator: 5,
            id: MemoryId::new(),
            extent: Extent {
                start: 10,
                blocks: 3,
            },
            payload: vec![1, 2, 3],
            remainder: Some(Extent {
                start: 13,
                blocks: 2,
            }),
            freed: None,
        };
        assert_eq!(RecordOp::decode(true, 5, &op.encode()).unwrap(), op);
        let free = RecordOp::Free {
            locator: 5,
            extent: Extent {
                start: 10,
                blocks: 3,
            },
        };
        assert_eq!(RecordOp::decode(false, 5, &free.encode()).unwrap(), free);
    }

    #[test]
    fn free_space_best_fit_and_coalesce() {
        let mut f = FreeSpace::default();
        f.insert(Extent {
            start: 10,
            blocks: 4,
        });
        f.insert(Extent {
            start: 20,
            blocks: 2,
        });
        // Best fit picks the 2 block hole for 2 blocks.
        let p = f.place(2, 100);
        assert_eq!(p.extent.start, 20);
        assert!(p.remainder.is_none());
        assert_eq!(f.commit(&p, 100), 100);
        // 3 blocks come from the 4 block hole, leaving 1.
        let p = f.place(3, 100);
        assert_eq!(p.extent.start, 10);
        assert_eq!(
            p.remainder,
            Some(Extent {
                start: 13,
                blocks: 1
            })
        );
        f.commit(&p, 100);
        // Nothing fits 5: the file grows.
        let p = f.place(5, 100);
        assert_eq!(p.extent.start, 100);
        assert_eq!(f.commit(&p, 100), 105);
        // Neighbours merge.
        f.insert(Extent {
            start: 11,
            blocks: 2,
        });
        f.insert(Extent {
            start: 10,
            blocks: 1,
        });
        assert_eq!(f.by_start.get(&10), Some(&4));
        assert_eq!(f.free_blocks(), 4);
    }

    #[test]
    fn free_space_from_live_rejects_overlap() {
        let e = |start, blocks| Extent { start, blocks };
        let f = FreeSpace::from_live(vec![e(5, 2), e(1, 2)], 10).unwrap();
        assert_eq!(f.free_blocks(), 5);
        assert!(FreeSpace::from_live(vec![e(1, 3), e(2, 2)], 10).is_err());
        assert!(FreeSpace::from_live(vec![e(8, 3)], 10).is_err());
    }
}
