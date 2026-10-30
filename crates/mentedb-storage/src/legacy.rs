//! Migration from the page file layout (format 1) to the record file.
//!
//! Format 1 stored every memory in its own 64 KB page (`pages.db`), with a
//! WAL of full page images (`wal.log`) and a page keyed memory map snapshot
//! (`memory_map.bin`). A typical memory used under a tenth of its page.
//!
//! Migration reads that state without modifying a single byte of it, writes
//! the complete record database under new file names, and commits with one
//! atomic rename:
//!
//! 1. Read the old state: page file, plus the old WAL replayed in memory
//!    (last operation per page wins, exactly as the old recovery did), plus
//!    the old snapshot's write stamps when it is still anchored.
//! 2. Write every memory into `records.db.tmp`, keeping its page number as its
//!    locator; fsync.
//! 3. Write the new WAL (`records.wal`, a single checkpoint entry) and the new
//!    record map (`records_map.bin`) anchored on it; fsync both.
//! 4. Rename `records.db.tmp` to `records.db` and fsync the directory. This
//!    is the commit point: the new database exists exactly when `records.db`
//!    does.
//! 5. Move the old files aside (`pages.db.v1`, `wal.log.v1`,
//!    `memory_map.bin.v1`), kept as a copy of the database as it was before
//!    migration until an operator deletes them.
//!
//! A crash before step 4 leaves the old database untouched (open migrates it
//! again from scratch, discarding the partial new files); a crash after it
//! leaves a complete new database whose leftover old files are removed on the
//! next open. Engines older than this layout refuse a directory holding
//! `records.db`, so they never open the moved files as a fresh database.
//! Locators and stamps are preserved, so every snapshot keyed by
//! page and stamp (salience bounds, mode probes) stays valid.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

use mentedb_core::MemoryNode;
use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;
use tracing::{info, warn};

use crate::memory_map::{self, RecordEntry, RecordTable, Token, entry_stamp};
use crate::record::{self, DATA_FILE, Extent, WAL_FILE, le_u32, le_u64};
use crate::wal::{Wal, WalEntry, WalEntryType};

/// Page size of format 1.
pub const LEGACY_PAGE_SIZE: usize = 64 * 1024;
/// Page header size of format 1.
const LEGACY_HEADER_SIZE: usize = 32;
/// Usable bytes per format 1 page.
const LEGACY_DATA_SIZE: usize = LEGACY_PAGE_SIZE - LEGACY_HEADER_SIZE;
const LEGACY_MAGIC: u64 = 0x4D454E_5445444231;
const LEGACY_VERSION: u32 = 1;
const LEGACY_DATA_PAGE: u8 = 1;

/// Format 1 page file.
pub const LEGACY_PAGE_FILE: &str = "pages.db";
/// Format 1 WAL.
pub const LEGACY_WAL_FILE: &str = "wal.log";
/// Format 1 memory map snapshot.
pub const LEGACY_MAP_FILE: &str = "memory_map.bin";

const LEGACY_MAP_MAGIC: &[u8; 8] = b"MDBMMAP1";

/// Where an injected fault stops a migration, to test crash safety.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationFault {
    /// After part of the new record file is written, before its fsync.
    MidDataWrite,
    /// After the new record file is written and synced.
    AfterDataSync,
    /// After the new WAL is written.
    AfterWal,
    /// After the new record map snapshot is written, right before commit.
    BeforeCommit,
    /// After the commit rename, before the old files are moved aside.
    AfterCommit,
}

/// What a migration did.
#[derive(Debug, Clone, Default)]
pub struct MigrationReport {
    /// Memories carried over.
    pub memories: u64,
    /// Pages in the old file (including the header page).
    pub legacy_pages: u64,
    /// Old pages that held no decodable memory (free, empty, or corrupt).
    pub skipped_pages: u64,
    /// Old WAL operations applied during the migration.
    pub wal_ops: u64,
    /// Write stamps carried over from the old memory map snapshot.
    pub stamps_kept: u64,
    /// Bytes of the old page file, WAL and snapshot.
    pub bytes_before: u64,
    /// Bytes of the new record file, WAL and snapshot.
    pub bytes_after: u64,
    /// Wall time of the migration.
    pub elapsed_ms: u64,
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Whether `dir` holds a format 1 database and no record database.
pub(crate) fn is_legacy(dir: &Path) -> bool {
    !dir.join(DATA_FILE).exists() && file_len(&dir.join(LEGACY_PAGE_FILE)) > 0
}

/// Suffix of the format 1 files kept after a migration.
pub const LEGACY_KEPT_SUFFIX: &str = ".v1";

/// Settle format 1 files after a committed migration (also run on every open
/// while `records.db` exists, which finishes a migration that crashed after
/// its commit). The old page file, WAL and snapshot are moved aside with the
/// `.v1` suffix rather than deleted, so the pre-migration database survives
/// a migration bug until an operator removes it; temporary files are removed.
pub(crate) fn remove_leftovers(dir: &Path) -> MenteResult<()> {
    let mut kept = Vec::new();
    for name in [LEGACY_PAGE_FILE, LEGACY_WAL_FILE, LEGACY_MAP_FILE] {
        let p = dir.join(name);
        if p.exists() {
            std::fs::rename(&p, dir.join(format!("{name}{LEGACY_KEPT_SUFFIX}")))?;
            kept.push(name);
        }
    }
    let mut removed = Vec::new();
    for name in ["wal.log.tmp", "memory_map.bin.tmp", "records.db.tmp"] {
        let p = dir.join(name);
        if p.exists() {
            std::fs::remove_file(&p)?;
            removed.push(name);
        }
    }
    if !kept.is_empty() || !removed.is_empty() {
        record::sync_dir(dir)?;
        info!(
            ?kept,
            ?removed,
            suffix = LEGACY_KEPT_SUFFIX,
            "settled format 1 files after migration"
        );
    }
    Ok(())
}

/// Decode a memory from a format 1 page data region: `[len u32][json]`.
fn decode_legacy_memory(data: &[u8]) -> Option<MemoryNode> {
    let data = &data[..data.len().min(LEGACY_DATA_SIZE)];
    if data.len() < 4 {
        return None;
    }
    let len = le_u32(&data[..4]) as usize;
    if len == 0 || len + 4 > LEGACY_DATA_SIZE || 4 + len > data.len() {
        return None;
    }
    serde_json::from_slice(&data[4..4 + len]).ok()
}

/// A format 1 page decoded from its on-disk bytes: the memory it holds, if
/// any. A checksum mismatch is reported as an error.
fn decode_legacy_page(page_id: u64, buf: &[u8]) -> MenteResult<Option<MemoryNode>> {
    let header = &buf[..LEGACY_HEADER_SIZE];
    let data = &buf[LEGACY_HEADER_SIZE..LEGACY_PAGE_SIZE];
    let checksum = le_u32(&header[16..20]);
    if checksum != 0 {
        let mut h = crc32fast::Hasher::new();
        h.update(&header[0..8]); // page_id
        h.update(&header[8..16]); // lsn
        h.update(&header[20..22]); // free_space
        h.update(&header[22..24]); // num_slots
        h.update(&header[24..25]); // page_type
        h.update(data);
        if h.finalize() != checksum {
            return Err(MenteError::Storage(format!(
                "legacy page {page_id} checksum mismatch"
            )));
        }
    }
    if header[24] != LEGACY_DATA_PAGE {
        return Ok(None);
    }
    Ok(decode_legacy_memory(data))
}

/// The write stamps the old memory map recorded for each page, when its
/// snapshot is intact and still anchored in the old WAL.
fn legacy_stamps(
    dir: &Path,
    entries: &[WalEntry],
) -> Option<std::collections::HashMap<u64, (MemoryId, u64)>> {
    let bytes = std::fs::read(dir.join(LEGACY_MAP_FILE)).ok()?;
    if bytes.len() < 8 + 4 + 16 + 8 + 8 + 8 + 4 {
        return None;
    }
    let (body, crc) = bytes.split_at(bytes.len() - 4);
    if crc32fast::hash(body) != le_u32(crc) || &body[..8] != LEGACY_MAP_MAGIC {
        return None;
    }
    if le_u32(&body[8..12]) != 1 {
        return None;
    }
    let mut token: Token = [0; 16];
    token.copy_from_slice(&body[12..28]);
    let mut pos = 28 + 8; // token, page_count
    let take = |pos: &mut usize, n: usize| -> Option<&[u8]> {
        let s = body.get(*pos..*pos + n)?;
        *pos += n;
        Some(s)
    };
    let n = le_u64(take(&mut pos, 8)?);
    let mut stamps = std::collections::HashMap::new();
    for _ in 0..n {
        let page = le_u64(take(&mut pos, 8)?);
        let id = uuid::Uuid::from_slice(take(&mut pos, 16)?).ok()?;
        let stamp = le_u64(take(&mut pos, 8)?);
        stamps.insert(page, (MemoryId(id), stamp));
    }
    // Roll forward over the WAL after the anchoring checkpoint, with the
    // rules the old memory map applied.
    let at = entries
        .iter()
        .rposition(|e| e.entry_type == WalEntryType::Checkpoint && e.data == token)?;
    for e in &entries[at + 1..] {
        match e.entry_type {
            WalEntryType::PageWrite => match decode_legacy_memory(&e.data) {
                Some(node) => {
                    stamps.insert(e.page_id, (node.id, entry_stamp(&token, e.lsn)));
                }
                None => {
                    stamps.remove(&e.page_id);
                }
            },
            WalEntryType::PageFree => {
                stamps.remove(&e.page_id);
            }
            _ => {}
        }
    }
    Some(stamps)
}

/// Read the format 1 file header: page count.
fn legacy_page_count(file: &mut File) -> MenteResult<u64> {
    let mut hdr = [0u8; 32];
    file.read_exact(&mut hdr)?;
    if le_u64(&hdr[0..8]) != LEGACY_MAGIC {
        return Err(MenteError::Storage("invalid page file magic number".into()));
    }
    let version = le_u32(&hdr[8..12]);
    if version != LEGACY_VERSION {
        return Err(MenteError::Storage(format!(
            "unsupported page file version: {version}"
        )));
    }
    Ok(le_u64(&hdr[16..24]))
}

fn fault(at: Option<MigrationFault>, here: MigrationFault) -> MenteResult<()> {
    if at == Some(here) {
        return Err(MenteError::Storage(format!(
            "injected migration fault: {here:?}"
        )));
    }
    Ok(())
}

/// Migrate the format 1 database in `dir` to the record layout. The caller
/// holds the directory lock.
pub(crate) fn migrate(dir: &Path, at: Option<MigrationFault>) -> MenteResult<MigrationReport> {
    let started = Instant::now();
    let pages_path = dir.join(LEGACY_PAGE_FILE);
    let tmp_path = dir.join(format!("{DATA_FILE}.tmp"));
    let wal_path = dir.join(WAL_FILE);

    // Partial output of an earlier attempt that crashed before its commit.
    for p in [
        tmp_path.clone(),
        wal_path.clone(),
        dir.join(format!("{WAL_FILE}.tmp")),
        dir.join(memory_map::MEMORY_MAP_FILE),
        dir.join(format!("{}.tmp", memory_map::MEMORY_MAP_FILE)),
    ] {
        if p.exists() {
            std::fs::remove_file(&p)?;
        }
    }

    let mut report = MigrationReport {
        bytes_before: [LEGACY_PAGE_FILE, LEGACY_WAL_FILE, LEGACY_MAP_FILE]
            .iter()
            .map(|n| file_len(&dir.join(n)))
            .sum(),
        ..Default::default()
    };
    info!(
        dir = %dir.display(),
        bytes = report.bytes_before,
        "migrating page file database to the record layout"
    );

    // Old state: page file plus the old WAL, last operation per page wins.
    let entries = Wal::read_entries(&dir.join(LEGACY_WAL_FILE))?;
    let mut last_op: std::collections::HashMap<u64, &WalEntry> = Default::default();
    for e in &entries {
        if matches!(
            e.entry_type,
            WalEntryType::PageWrite | WalEntryType::PageFree
        ) {
            last_op.insert(e.page_id, e);
        }
    }
    report.wal_ops = last_op.len() as u64;
    let stamps = legacy_stamps(dir, &entries);

    let mut pages_file = File::open(&pages_path)?;
    let header_count = legacy_page_count(&mut pages_file)?;
    let file_pages = file_len(&pages_path) / LEGACY_PAGE_SIZE as u64;
    let wal_max = last_op.keys().max().map_or(0, |m| m + 1);
    let page_count = header_count.max(wal_max);
    report.legacy_pages = page_count;

    // Sequential, buffered reads of the whole old file: one pass, large
    // requests, which is what network file systems serve best.
    let mut reader = BufReader::with_capacity(64 * LEGACY_PAGE_SIZE, pages_file);
    let mut skip = vec![0u8; LEGACY_PAGE_SIZE - 32];
    reader.read_exact(&mut skip)?; // rest of page 0

    let mut out = BufWriter::with_capacity(4 << 20, File::create(&tmp_path)?);
    out.write_all(&record::file_header())?;
    let mut block = 1u64;
    let mut table = RecordTable::default();
    let mut page = vec![0u8; LEGACY_PAGE_SIZE];
    for page_id in 1..page_count {
        let on_disk = page_id < file_pages;
        if on_disk {
            reader.read_exact(&mut page)?;
        }
        let node = match last_op.get(&page_id) {
            Some(e) if e.entry_type == WalEntryType::PageWrite => decode_legacy_memory(&e.data),
            Some(_) => None,
            None if on_disk => match decode_legacy_page(page_id, &page) {
                Ok(n) => n,
                Err(e) => {
                    warn!(page_id, error = %e, "skipping unreadable legacy page");
                    None
                }
            },
            None => None,
        };
        let Some(node) = node else {
            report.skipped_pages += 1;
            continue;
        };
        let payload = record::encode_node(&node)?;
        let blocks = record::blocks_for(payload.len());
        out.write_all(&record::live_image(page_id, node.id, &payload, blocks))?;
        let stamp = match stamps.as_ref().and_then(|s| s.get(&page_id)) {
            Some(&(id, stamp)) if id == node.id => {
                report.stamps_kept += 1;
                stamp
            }
            _ => memory_map::random_u64(),
        };
        table.records.insert(
            page_id,
            RecordEntry {
                id: node.id,
                stamp,
                extent: Extent {
                    start: block,
                    blocks,
                },
            },
        );
        block += blocks;
        report.memories += 1;
        if report.memories == 1 {
            fault(at, MigrationFault::MidDataWrite)?;
        }
    }
    table.next_locator = page_count.max(1);
    let out = out
        .into_inner()
        .map_err(|e| MenteError::Storage(format!("record file write failed: {e}")))?;
    out.sync_all()?;
    drop(out);
    fault(at, MigrationFault::AfterDataSync)?;

    // New WAL holding only the checkpoint the snapshot is anchored on.
    let token = memory_map::new_token();
    {
        let mut wal = Wal::open_file(&wal_path)?;
        wal.append(WalEntryType::Checkpoint, 0, &token)?;
        wal.sync()?;
    }
    fault(at, MigrationFault::AfterWal)?;
    memory_map::write_snapshot(dir, &token, &table)?;
    record::sync_dir(dir)?;
    fault(at, MigrationFault::BeforeCommit)?;

    // Commit.
    std::fs::rename(&tmp_path, dir.join(DATA_FILE))?;
    record::sync_dir(dir)?;
    fault(at, MigrationFault::AfterCommit)?;

    remove_leftovers(dir)?;
    report.bytes_after = [DATA_FILE, WAL_FILE, memory_map::MEMORY_MAP_FILE]
        .iter()
        .map(|n| file_len(&dir.join(n)))
        .sum();
    report.elapsed_ms = started.elapsed().as_millis() as u64;
    info!(
        dir = %dir.display(),
        memories = report.memories,
        legacy_pages = report.legacy_pages,
        skipped_pages = report.skipped_pages,
        wal_ops = report.wal_ops,
        stamps_kept = report.stamps_kept,
        bytes_before = report.bytes_before,
        bytes_after = report.bytes_after,
        elapsed_ms = report.elapsed_ms,
        "migrated page file database to the record layout"
    );
    Ok(report)
}
