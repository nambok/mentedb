//! Storage Engine: facade over the record file, the WAL, the locator table
//! and the record cache.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use mentedb_core::MemoryNode;
use mentedb_core::config::StorageConfig;
use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;

use parking_lot::{Mutex, RwLock};
use tracing::{info, warn};

use crate::cache::{BufferStats, RecordCache};
use crate::legacy::{self, MigrationFault};
use crate::memory_map::{
    self, MappedMemory, RecordEntry, RecordTable, Token, entry_stamp, new_token,
};
use crate::record::{
    self, BLOCK_SIZE, DATA_FILE, EXTENT_HEADER_SIZE, ExtentHeader, ExtentKind, FreeSpace, PageId,
    RecordOp, WAL_FILE,
};
use crate::wal::{Wal, WalEntry, WalEntryType};

/// Mutable engine state, guarded by one lock. Writers hold it exclusively for
/// the whole transaction (WAL append, fsync, file write, table update);
/// readers hold it shared while they read a record, so they never see an
/// extent mid write.
struct State {
    wal: Wal,
    table: RecordTable,
    free: FreeSpace,
    /// File length in blocks.
    end: u64,
    /// Token of the last checkpoint; stamps of later writes derive from it.
    token: Token,
    /// Whether the record map snapshot on disk is anchored on `token`.
    anchored: bool,
    /// Whether the table changed since the last checkpoint.
    dirty: bool,
}

/// The unified storage engine for MenteDB.
///
/// Memories are stored as variable length records packed into 512 byte
/// blocks of one file (see [`crate::record`]), so the file is about the size
/// of the data. Every record is addressed by a stable locator ([`PageId`])
/// resolved through an in memory table that is persisted at checkpoints and
/// replayed forward from the WAL on open.
///
/// Concurrency model:
/// - **One process per database.** Open takes an exclusive lock on a LOCK
///   file in the data directory (fcntl OFD lock, enforced across hosts on
///   NFSv4) and holds it for the engine's lifetime, so in memory state never
///   goes stale and needs no reload from disk on each write.
/// - **Writes are serialized** by the state lock: WAL append and fsync, then
///   the record write, then the table update.
/// - **Reads run concurrently** with each other: a cache hit takes only the
///   cache lock; a miss reads its extent with a positional read under the
///   shared state lock.
pub struct StorageEngine {
    dir: PathBuf,
    file: File,
    state: RwLock<State>,
    cache: RecordCache,
    /// Held exclusively for the lifetime of this engine; released on close
    /// or process exit.
    process_lock: Mutex<Option<File>>,
    wal_checkpoint_bytes: u64,
    max_record_bytes: u64,
    /// Memory loads served since open, cached or not (metrics and tests).
    memory_loads: AtomicU64,
    /// Reads issued to the record file since open (cache misses and scans).
    disk_reads: AtomicU64,
    /// Bytes read from the record file since open.
    bytes_read: AtomicU64,
}

fn lock_dir(path: &Path) -> MenteResult<File> {
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path.join("LOCK"))?;
    match crate::lock::try_lock_exclusive(&lock_file) {
        Ok(true) => Ok(lock_file),
        Ok(false) => Err(MenteError::Storage(format!(
            "database directory {} is locked by another process",
            path.display()
        ))),
        Err(e) => Err(MenteError::Storage(format!(
            "failed to lock database directory {}: {e}",
            path.display()
        ))),
    }
}

/// Create an empty record file atomically (temp file, fsync, rename).
fn create_data_file(dir: &Path) -> MenteResult<()> {
    for stale in [WAL_FILE, memory_map::MEMORY_MAP_FILE] {
        let p = dir.join(stale);
        if p.exists() {
            std::fs::remove_file(&p)?;
        }
    }
    let tmp = dir.join(format!("{DATA_FILE}.tmp"));
    {
        let f = File::create(&tmp)?;
        record::write_at(&f, &record::file_header(), 0)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, dir.join(DATA_FILE))?;
    record::sync_dir(dir)?;
    info!(dir = %dir.display(), "created record file");
    Ok(())
}

impl StorageEngine {
    /// Open (or create) a storage engine rooted at `path` with default
    /// settings. See [`Self::open_with_config`].
    ///
    /// ```no_run
    /// use mentedb_storage::StorageEngine;
    ///
    /// let engine = StorageEngine::open("/tmp/mentedb".as_ref())?;
    /// // engine is ready: WAL recovery already ran if needed
    /// # Ok::<(), mentedb_core::error::MenteError>(())
    /// ```
    pub fn open(path: &Path) -> MenteResult<Self> {
        Self::open_with_config(path, &StorageConfig::default())
    }

    /// Open (or create) a storage engine rooted at `path`.
    ///
    /// `path` must be a directory; it is created if missing. A database in the
    /// old page file layout is migrated first (see [`crate::legacy`]); then
    /// any WAL entries are replayed. `config.cache_bytes` bounds the record
    /// cache; size it to the database to serve every scan from memory.
    pub fn open_with_config(path: &Path, config: &StorageConfig) -> MenteResult<Self> {
        Self::open_inner(path, config, None)
    }

    /// Open with a migration fault injected, to test crash safety. The
    /// migration stops at `fault` with an error, as a crash there would.
    #[doc(hidden)]
    pub fn open_with_migration_fault(
        path: &Path,
        config: &StorageConfig,
        fault: MigrationFault,
    ) -> MenteResult<Self> {
        Self::open_inner(path, config, Some(fault))
    }

    fn open_inner(
        path: &Path,
        config: &StorageConfig,
        fault: Option<MigrationFault>,
    ) -> MenteResult<Self> {
        std::fs::create_dir_all(path)?;
        let lock_file = lock_dir(path)?;

        if path.join(DATA_FILE).exists() {
            legacy::remove_leftovers(path)?;
        } else if legacy::is_legacy(path) {
            legacy::migrate(path, fault)?;
        } else {
            create_data_file(path)?;
        }

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.join(DATA_FILE))?;
        let mut header = [0u8; BLOCK_SIZE];
        record::read_at(&file, &mut header, 0)?;
        record::check_file_header(&header)?;
        let wal = Wal::open_file(&path.join(WAL_FILE))?;

        let engine = Self {
            dir: path.to_path_buf(),
            file,
            state: RwLock::new(State {
                wal,
                table: RecordTable::default(),
                free: FreeSpace::default(),
                end: 1,
                token: new_token(),
                anchored: false,
                dirty: false,
            }),
            cache: RecordCache::new(config.cache_bytes),
            process_lock: Mutex::new(Some(lock_file)),
            wal_checkpoint_bytes: config.wal_checkpoint_bytes,
            max_record_bytes: config.max_record_bytes,
            memory_loads: AtomicU64::new(0),
            disk_reads: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        };

        let replayed = engine.recover()?;
        let st = engine.state.read();
        info!(
            ?path,
            replayed,
            records = st.table.records.len(),
            bytes = st.end * BLOCK_SIZE as u64,
            "storage engine opened"
        );
        drop(st);
        Ok(engine)
    }

    /// Replay the WAL into the record file, then load the locator table from
    /// the snapshot rolled forward over the WAL, or by scanning the record
    /// chain when there is no usable snapshot. Returns the operations replayed.
    fn recover(&self) -> MenteResult<usize> {
        let mut st = self.state.write();
        let entries = st.wal.iterate()?;

        // Redo every logged operation in log order. Each one rewrites the
        // exact bytes it wrote, so redoing operations already on disk is
        // harmless and the file ends in the logged state.
        let mut ops: Vec<(u64, RecordOp)> = Vec::new();
        for e in &entries {
            let op = match e.entry_type {
                WalEntryType::RecordWrite => RecordOp::decode(true, e.page_id, &e.data)?,
                WalEntryType::RecordFree => RecordOp::decode(false, e.page_id, &e.data)?,
                WalEntryType::Checkpoint | WalEntryType::Commit => continue,
                WalEntryType::PageWrite | WalEntryType::PageFree => {
                    return Err(MenteError::Storage(
                        "record WAL holds a page file entry".into(),
                    ));
                }
            };
            op.redo(&self.file)?;
            ops.push((e.lsn, op));
        }
        if !ops.is_empty() {
            self.file.sync_data()?;
        }
        let file_end = self
            .file
            .metadata()?
            .len()
            .div_ceil(BLOCK_SIZE as u64)
            .max(1);

        let (table, token, anchored) = match self.load_snapshot(&entries, file_end) {
            Some((table, token)) => (table, token, true),
            None => (self.scan_chain(file_end)?, new_token(), false),
        };
        st.end = file_end.max(table.max_end());
        st.free = table.free_space(st.end)?;
        st.table = table;
        st.token = token;
        st.anchored = anchored;
        st.dirty = !ops.is_empty();

        if !anchored || !ops.is_empty() || st.wal.has_torn_tail() {
            self.checkpoint_locked(&mut st, true)?;
        }
        if !ops.is_empty() {
            info!(count = ops.len(), "WAL recovery replayed entries");
        }
        Ok(ops.len())
    }

    /// The persisted table rolled forward over `entries`, or None when it is
    /// missing, unreadable, not anchored in this WAL, or inconsistent with a
    /// file of `file_end` blocks. Never an error: the fallback is a scan.
    fn load_snapshot(&self, entries: &[WalEntry], file_end: u64) -> Option<(RecordTable, Token)> {
        let snap = match memory_map::read_snapshot(&self.dir) {
            Ok(Some(s)) => s,
            Ok(None) => return None,
            Err(e) => {
                warn!(error = %e, "record map snapshot unreadable, open will scan the record file");
                return None;
            }
        };
        let Some(at) = entries
            .iter()
            .rposition(|e| e.entry_type == WalEntryType::Checkpoint && e.data == snap.token)
        else {
            info!("record map snapshot predates the WAL, open will scan the record file");
            return None;
        };
        let mut table = snap.table;
        for e in &entries[at + 1..] {
            let write = match e.entry_type {
                WalEntryType::RecordWrite => true,
                WalEntryType::RecordFree => false,
                _ => continue,
            };
            let op = RecordOp::decode(write, e.page_id, &e.data).ok()?;
            table.apply(&op, entry_stamp(&snap.token, e.lsn));
        }
        if let Err(e) = table.free_space(file_end) {
            warn!(error = %e, "record map snapshot does not fit the record file, open will scan");
            return None;
        }
        Some((table, snap.token))
    }

    /// Build the table by walking the extent chain from block 1. Reads only
    /// extent headers; records get fresh stamps, since no log entry derives
    /// them.
    fn scan_chain(&self, file_end: u64) -> MenteResult<RecordTable> {
        let mut table = RecordTable::default();
        let mut reader = BufReader::with_capacity(1 << 20, self.file.try_clone()?);
        reader.seek(SeekFrom::Start(BLOCK_SIZE as u64))?;
        let mut at = 1u64;
        let mut header = [0u8; EXTENT_HEADER_SIZE];
        while at < file_end {
            reader.read_exact(&mut header)?;
            self.disk_reads.fetch_add(1, Ordering::Relaxed);
            let h = ExtentHeader::decode(&header)
                .ok_or_else(|| MenteError::Storage(format!("record chain broken at block {at}")))?;
            let extent = record::Extent {
                start: at,
                blocks: h.blocks as u64,
            };
            if h.kind == ExtentKind::Live {
                table.records.insert(
                    h.locator,
                    RecordEntry {
                        id: h.id,
                        stamp: memory_map::random_u64(),
                        extent,
                    },
                );
                table.next_locator = table.next_locator.max(h.locator + 1);
            }
            at = extent.end();
            reader.seek_relative((extent.byte_len() - EXTENT_HEADER_SIZE) as i64)?;
        }
        self.bytes_read
            .fetch_add(file_end * BLOCK_SIZE as u64, Ordering::Relaxed);
        table.next_locator = table.next_locator.max(1);
        info!(
            records = table.records.len(),
            "record map rebuilt by scanning the record file"
        );
        Ok(table)
    }

    /// Gracefully shut down: sync the record file and the WAL, release the
    /// directory lock.
    pub fn close(&self) -> MenteResult<()> {
        let mut st = self.state.write();
        self.file.sync_data()?;
        st.wal.sync()?;
        drop(st);
        if let Some(lock_file) = self.process_lock.lock().take() {
            let _ = crate::lock::unlock(&lock_file);
        }
        info!("storage engine closed");
        Ok(())
    }

    /// Release the process lock without flushing anything.
    ///
    /// Test support for simulating a process crash: a real crash releases
    /// the OS file lock but persists nothing beyond what was already synced.
    #[doc(hidden)]
    pub fn release_process_lock(&self) {
        if let Some(lock_file) = self.process_lock.lock().take() {
            let _ = crate::lock::unlock(&lock_file);
        }
    }

    // ---- metrics and settings ----

    /// Record cache activity, for metrics (hit ratio, evictions, residency).
    pub fn buffer_stats(&self) -> BufferStats {
        self.cache.stats()
    }

    /// Change the record cache capacity in bytes. A host with spare memory
    /// can raise it to the database size so every scan is served from memory.
    pub fn set_cache_capacity(&self, bytes: u64) {
        self.cache.set_capacity(bytes);
    }

    /// Record cache capacity in bytes.
    pub fn cache_capacity(&self) -> u64 {
        self.cache.capacity()
    }

    /// Blocks in the record file ([`BLOCK_SIZE`] bytes each).
    pub fn page_count(&self) -> u64 {
        self.state.read().end
    }

    /// Bytes of the record file.
    pub fn data_bytes(&self) -> u64 {
        self.state.read().end * BLOCK_SIZE as u64
    }

    /// Bytes of the record file not holding any record (reusable holes).
    pub fn free_bytes(&self) -> u64 {
        self.state.read().free.free_blocks() * BLOCK_SIZE as u64
    }

    /// Memory loads served since open, whether from the cache or from disk.
    pub fn memory_loads(&self) -> u64 {
        self.memory_loads.load(Ordering::Relaxed)
    }

    /// Reads issued to the record file since open (cache misses, plus one per
    /// extent header when open had to scan).
    pub fn page_reads(&self) -> u64 {
        self.disk_reads.load(Ordering::Relaxed)
    }

    /// Bytes read from the record file since open.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    // ---- memory operations ----

    fn encode(&self, node: &MemoryNode) -> MenteResult<Vec<u8>> {
        let payload = record::encode_node(node)?;
        if payload.len() as u64 > self.max_record_bytes {
            return Err(MenteError::CapacityExceeded(format!(
                "memory node encoded to {} bytes (max {})",
                payload.len(),
                self.max_record_bytes
            )));
        }
        Ok(payload)
    }

    /// Checkpoint when the WAL has grown past the configured size.
    fn maybe_checkpoint(&self, wal_bytes: u64) {
        if wal_bytes > self.wal_checkpoint_bytes
            && let Err(e) = self.checkpoint()
        {
            warn!("auto-checkpoint failed: {e}");
        }
    }

    /// Store a [`MemoryNode`] as a new record and return its locator.
    ///
    /// ```no_run
    /// use mentedb_storage::StorageEngine;
    /// use mentedb_core::{MemoryNode, memory::MemoryType, types::AgentId};
    ///
    /// let engine = StorageEngine::open("/tmp/mentedb".as_ref())?;
    /// let node = MemoryNode::new(
    ///     AgentId::new(),
    ///     MemoryType::Semantic,
    ///     "User likes dark mode".to_string(),
    ///     vec![0.1, 0.2],
    /// );
    /// let page_id = engine.store_memory(&node)?;
    /// # Ok::<(), mentedb_core::error::MenteError>(())
    /// ```
    pub fn store_memory(&self, node: &MemoryNode) -> MenteResult<PageId> {
        let ids = self.store_memory_batch(std::slice::from_ref(node))?;
        Ok(ids[0])
    }

    /// Store several [`MemoryNode`]s with a single WAL fsync.
    pub fn store_memory_batch(&self, nodes: &[MemoryNode]) -> MenteResult<Vec<PageId>> {
        let payloads = nodes
            .iter()
            .map(|n| self.encode(n))
            .collect::<MenteResult<Vec<_>>>()?;

        let mut st = self.state.write();
        // Plan on copies so a failed append leaves the allocator untouched.
        let mut free = st.free.clone();
        let mut end = st.end;
        let first = st.table.next_locator;
        let mut ops = Vec::with_capacity(nodes.len());
        for (i, (node, payload)) in nodes.iter().zip(payloads).enumerate() {
            let place = free.place(record::blocks_for(payload.len()), end);
            end = free.commit(&place, end);
            ops.push(RecordOp::Write {
                locator: first + i as u64,
                id: node.id,
                extent: place.extent,
                payload,
                remainder: place.remainder,
                freed: None,
            });
        }
        let mut lsns = Vec::with_capacity(ops.len());
        for op in &ops {
            lsns.push(
                st.wal
                    .append(WalEntryType::RecordWrite, op.locator(), &op.encode())?,
            );
        }
        st.wal.sync()?;
        st.free = free;
        st.end = end;

        let mut ids = Vec::with_capacity(ops.len());
        for ((op, lsn), node) in ops.iter().zip(lsns).zip(nodes) {
            op.redo(&self.file)?;
            let stamp = entry_stamp(&st.token, lsn);
            st.table.apply(op, stamp);
            if let RecordOp::Write { payload, .. } = op {
                self.cache
                    .insert(op.locator(), node.clone(), payload.len() as u64);
            }
            ids.push(PageId(op.locator()));
        }
        st.dirty = true;
        let wal_bytes = st.wal.file_size();
        drop(st);

        self.maybe_checkpoint(wal_bytes);
        Ok(ids)
    }

    /// Rewrite the record at `page_id` with `node`. The locator never
    /// changes: a record that still fits its extent is rewritten in place,
    /// one that outgrew it moves to a larger extent and the old one is freed,
    /// both in one logged operation.
    pub fn update_memory(&self, page_id: PageId, node: &MemoryNode) -> MenteResult<()> {
        let payload = self.encode(node)?;
        let need = record::blocks_for(payload.len());
        let len = payload.len() as u64;

        let mut st = self.state.write();
        let entry =
            *st.table.records.get(&page_id.0).ok_or_else(|| {
                MenteError::Storage(format!("page {} holds no memory", page_id.0))
            })?;
        let place = (need > entry.extent.blocks).then(|| st.free.place(need, st.end));
        let op = RecordOp::Write {
            locator: page_id.0,
            id: node.id,
            extent: place.map_or(entry.extent, |p| p.extent),
            payload,
            remainder: place.and_then(|p| p.remainder),
            freed: place.map(|_| entry.extent),
        };
        let lsn = st
            .wal
            .append(WalEntryType::RecordWrite, page_id.0, &op.encode())?;
        st.wal.sync()?;
        if let Some(p) = place {
            let end = st.end;
            st.end = st.free.commit(&p, end);
            st.free.insert(entry.extent);
        }
        op.redo(&self.file)?;
        let stamp = entry_stamp(&st.token, lsn);
        st.table.apply(&op, stamp);
        st.dirty = true;
        self.cache.insert(page_id.0, node.clone(), len);
        let wal_bytes = st.wal.file_size();
        drop(st);

        self.maybe_checkpoint(wal_bytes);
        Ok(())
    }

    /// Delete the record at `page_id`, freeing its extent for reuse. Logged
    /// before it is applied, so the memory does not resurrect after a crash.
    pub fn delete_memory(&self, page_id: PageId) -> MenteResult<()> {
        let mut st = self.state.write();
        let entry =
            *st.table.records.get(&page_id.0).ok_or_else(|| {
                MenteError::Storage(format!("page {} holds no memory", page_id.0))
            })?;
        let op = RecordOp::Free {
            locator: page_id.0,
            extent: entry.extent,
        };
        let _lsn = st
            .wal
            .append(WalEntryType::RecordFree, page_id.0, &op.encode())?;
        st.wal.sync()?;
        st.free.insert(entry.extent);
        op.redo(&self.file)?;
        st.table.apply(&op, 0);
        st.dirty = true;
        self.cache.invalidate(page_id.0);
        let wal_bytes = st.wal.file_size();
        drop(st);

        self.maybe_checkpoint(wal_bytes);
        Ok(())
    }

    /// Load the [`MemoryNode`] stored at `page_id`.
    ///
    /// ```no_run
    /// # use mentedb_storage::{StorageEngine, PageId};
    /// # let engine = StorageEngine::open("/tmp/mentedb".as_ref())?;
    /// let node = engine.load_memory(PageId(1))?;
    /// println!("memory: {}", node.content);
    /// # Ok::<(), mentedb_core::error::MenteError>(())
    /// ```
    pub fn load_memory(&self, page_id: PageId) -> MenteResult<MemoryNode> {
        self.memory_loads.fetch_add(1, Ordering::Relaxed);
        if let Some(node) = self.cache.get(page_id.0) {
            return Ok(node);
        }
        // The shared lock keeps writers out while the extent is read and the
        // result cached, so the cache never holds an older version.
        let st = self.state.read();
        let entry =
            st.table.records.get(&page_id.0).ok_or_else(|| {
                MenteError::Storage(format!("page {} holds no memory", page_id.0))
            })?;
        let mut buf = vec![0u8; entry.extent.byte_len()];
        record::read_at(&self.file, &mut buf, entry.extent.offset())?;
        self.disk_reads.fetch_add(1, Ordering::Relaxed);
        self.bytes_read
            .fetch_add(buf.len() as u64, Ordering::Relaxed);
        let payload = record::live_payload(&buf, page_id.0)?;
        let node = record::decode_node(payload)?;
        self.cache
            .insert(page_id.0, node.clone(), payload.len() as u64);
        Ok(node)
    }

    // ---- durability ----

    /// Checkpoint: sync the record file, persist the record map, and truncate
    /// the WAL. A no-op when nothing changed since the last checkpoint.
    pub fn checkpoint(&self) -> MenteResult<()> {
        let mut st = self.state.write();
        self.checkpoint_locked(&mut st, false)
    }

    fn checkpoint_locked(&self, st: &mut State, force: bool) -> MenteResult<()> {
        if !force && !st.dirty && st.anchored {
            return Ok(());
        }
        // Records must be durable before the WAL entries that redo them go.
        self.file.sync_data()?;
        let token = new_token();
        let lsn = st.wal.append(WalEntryType::Checkpoint, 0, &token)?;
        st.wal.sync()?;
        // The snapshot is anchored on the entry just appended; a crash before
        // the WAL is cut leaves that entry with the snapshot, a crash before
        // the snapshot leaves the previous anchor and all entries after it.
        match memory_map::write_snapshot(&self.dir, &token, &st.table) {
            Ok(()) => st.anchored = true,
            Err(e) => {
                warn!(error = %e, "record map snapshot write failed, next open will scan");
                memory_map::remove_snapshot(&self.dir);
                st.anchored = false;
            }
        }
        st.token = token;
        st.dirty = false;
        st.wal.reset_to(lsn, WalEntryType::Checkpoint, 0, &token)?;
        info!(lsn, records = st.table.records.len(), "checkpoint complete");
        Ok(())
    }

    /// Every memory in storage with its locator and write stamp, in locator
    /// order. Served from the in memory table; reads nothing.
    pub fn memory_map(&self) -> MenteResult<Vec<MappedMemory>> {
        Ok(self.state.read().table.memories())
    }

    /// The memory and write stamp at `page`, or None when it holds no memory.
    pub fn memory_stamp(&self, page: PageId) -> Option<(MemoryId, u64)> {
        self.state
            .read()
            .table
            .records
            .get(&page.0)
            .map(|e| (e.id, e.stamp))
    }

    /// Read and decode every record in the file, independent of the locator
    /// table, and return (memory id, locator) pairs in locator order. The
    /// ground truth of what is on disk; used by tests and diagnostics.
    pub fn scan_all_memories(&self) -> Vec<(MemoryId, PageId)> {
        let st = self.state.read();
        let end = st.end;
        let Ok(file) = self.file.try_clone() else {
            return Vec::new();
        };
        let mut reader = BufReader::with_capacity(1 << 20, file);
        if reader.seek(SeekFrom::Start(BLOCK_SIZE as u64)).is_err() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut at = 1u64;
        let mut header = [0u8; EXTENT_HEADER_SIZE];
        while at < end {
            if reader.read_exact(&mut header).is_err() {
                break;
            }
            self.disk_reads.fetch_add(1, Ordering::Relaxed);
            let Some(h) = ExtentHeader::decode(&header) else {
                break;
            };
            let mut rest = vec![0u8; h.blocks as usize * BLOCK_SIZE - EXTENT_HEADER_SIZE];
            if reader.read_exact(&mut rest).is_err() {
                break;
            }
            if h.kind == ExtentKind::Live {
                let mut buf = header.to_vec();
                buf.extend_from_slice(&rest);
                if let Ok(node) =
                    record::live_payload(&buf, h.locator).and_then(record::decode_node)
                {
                    out.push((node.id, PageId(h.locator)));
                }
            }
            at += h.blocks as u64;
        }
        drop(st);
        out.sort_by_key(|(_, p)| *p);
        out
    }
}

#[cfg(test)]
mod tests;
