//! Storage Engine: facade that ties the page manager, WAL, and buffer pool together.

use std::fs::File;
use std::path::{Path, PathBuf};

use mentedb_core::MemoryNode;
use mentedb_core::error::{MenteError, MenteResult};

use parking_lot::Mutex;
use tracing::info;

use crate::buffer::BufferPool;
use crate::memory_map::{self, MapAnchor, MappedMemory, MemoryTable};
use crate::page::{PAGE_DATA_SIZE, Page, PageId, PageManager, PageType};
use crate::wal::{Wal, WalEntryType};
/// Default number of page frames in the buffer pool.
const DEFAULT_BUFFER_POOL_SIZE: usize = 1024;

/// Auto-checkpoint when WAL file exceeds this size (8 MB).
const WAL_AUTO_CHECKPOINT_BYTES: u64 = 8 * 1024 * 1024;

/// The unified storage engine for MenteDB.
///
/// Coordinates page allocation, caching, and write-ahead logging to provide
/// crash-safe, page-oriented storage for memory nodes.
///
/// Concurrency model (inspired by WAL-mode databases):
/// - **Reads are lock-free**: `read_page` only touches the buffer pool and page
///   manager — no file locks, no WAL access.
/// - **Writes are fully serialized** via a blocking `flock(2)` on the WAL file.
///   The entire write transaction (page allocation + WAL append + page write +
///   fsync) executes under a single flock, ensuring correctness across multiple
///   processes sharing the same data directory.
/// - **State is refreshed from disk** under the flock: page count is re-read
///   from the file header and LSN is re-read from the WAL tail, so no process
///   can act on stale in-memory state.
/// - **One process per database.** Open acquires an exclusive lock on a LOCK
///   file in the data directory and holds it for the engine's lifetime. The
///   per operation flock protects individual write transactions, but layers
///   above the storage engine cache state across operations (buffer pool,
///   page map, index snapshots), and a second concurrent process flushing
///   those caches rolls the database back to its own open time snapshot.
///   A second open therefore fails fast with a "locked" error instead of
///   silently corrupting state; callers retry until the owner exits.
pub struct StorageEngine {
    page_manager: Mutex<PageManager>,
    buffer_pool: BufferPool,
    wal: Mutex<Wal>,
    /// Held exclusively for the lifetime of this engine; released on close
    /// or process exit. Guards against concurrent multi process opens.
    process_lock: Mutex<Option<File>>,
    /// Database directory (holds the memory map snapshot).
    dir: PathBuf,
    /// The memory map as of the checkpoint entry carrying its token, when
    /// known. None until a valid snapshot is loaded or a full scan builds one;
    /// every checkpoint rolls it forward and persists it. Lock order: after
    /// the WAL and page manager mutexes.
    map_anchor: Mutex<Option<MapAnchor>>,
    /// Memory loads served since open, cached or not (metrics and tests).
    memory_loads: std::sync::atomic::AtomicU64,
}

impl StorageEngine {
    /// Open (or create) a storage engine rooted at `path`.
    ///
    /// `path` must be a directory; it will be created if it does not exist.
    /// After opening, any uncommitted WAL entries are replayed for crash recovery.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mentedb_storage::StorageEngine;
    ///
    /// let engine = StorageEngine::open("/tmp/mentedb".as_ref())?;
    /// // engine is ready — WAL recovery already ran if needed
    /// # Ok::<(), mentedb_core::error::MenteError>(())
    /// ```
    pub fn open(path: &Path) -> MenteResult<Self> {
        std::fs::create_dir_all(path)?;

        // Exclusive process lock, held until close or process exit. A second
        // process opening the same directory would interleave stale cached
        // state with ours and roll the database back, so refuse it up front.
        let lock_path = path.join("LOCK");
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)?;
        // Cross-host lock: fcntl OFD lock on Linux (enforced across hosts by the
        // NFSv4 lock manager), flock elsewhere. This makes single-writer safe
        // when the directory lives on a network filesystem mounted by more than
        // one host.
        match crate::lock::try_lock_exclusive(&lock_file) {
            Ok(true) => {}
            Ok(false) => {
                return Err(MenteError::Storage(format!(
                    "database directory {} is locked by another process",
                    path.display()
                )));
            }
            Err(e) => {
                return Err(MenteError::Storage(format!(
                    "failed to lock database directory {}: {e}",
                    path.display()
                )));
            }
        }

        let page_manager = PageManager::open(path)?;
        let buffer_pool = BufferPool::new(DEFAULT_BUFFER_POOL_SIZE);
        let wal = Wal::open(path)?;

        let engine = Self {
            page_manager: Mutex::new(page_manager),
            buffer_pool,
            wal: Mutex::new(wal),
            process_lock: Mutex::new(Some(lock_file)),
            dir: path.to_path_buf(),
            map_anchor: Mutex::new(None),
            memory_loads: std::sync::atomic::AtomicU64::new(0),
        };

        let recovered = engine.recover()?;
        if recovered > 0 {
            info!(recovered, ?path, "storage engine opened with WAL recovery");
        } else {
            info!(?path, "storage engine opened");
        }

        Ok(engine)
    }

    /// Replay WAL entries to recover writes that were not checkpointed.
    ///
    /// For each `PageWrite` entry the serialized data is written back to its page.
    /// After replay the WAL is truncated. Returns the number of entries replayed.
    pub fn recover(&self) -> MenteResult<usize> {
        let mut wal = self.wal.lock();
        wal.lock_exclusive()?;
        let result = self.recover_locked(&mut wal);
        let unlocked = wal.unlock();
        let count = result?;
        unlocked?;
        Ok(count)
    }

    /// Load the persisted memory map and roll it forward over `entries`, or
    /// None when it is missing, unreadable, or does not belong to this WAL
    /// and page file. Never an error: the fallback is a full scan.
    fn load_anchor(&self, entries: &[crate::wal::WalEntry], page_count: u64) -> Option<MapAnchor> {
        let mut anchor = match memory_map::read_snapshot(&self.dir) {
            Ok(Some(a)) => a,
            Ok(None) => return None,
            Err(e) => {
                tracing::warn!(error = %e, "memory map snapshot unreadable, open will scan pages");
                return None;
            }
        };
        if !anchor.fits(page_count) {
            tracing::warn!(
                snapshot_pages = anchor.page_count,
                page_count,
                "memory map snapshot does not fit the page file, open will scan pages"
            );
            return None;
        }
        if anchor.advance(entries).is_none() {
            info!("memory map snapshot predates the WAL, open will scan pages");
            return None;
        }
        Some(anchor)
    }

    fn recover_locked(&self, wal: &mut Wal) -> MenteResult<usize> {
        let entries = wal.iterate()?;
        let mut count = 0usize;
        let mut pm = self.page_manager.lock();

        // Refresh page count from disk — another process may have written pages.
        pm.reload_header()?;

        // Roll the persisted memory map forward before replay may truncate
        // the WAL entries it is anchored on.
        let anchor = self.load_anchor(&entries, pm.page_count());

        // Replay with last-op-wins per page: every PageWrite carries a full page
        // image and PageFree discards the page, so only the final entry for each
        // page matters. This is what makes free-then-reuse sequences safe: a
        // PageFree followed by a later PageWrite for the same page must not
        // leave the page on the free list.
        let mut last_op: std::collections::HashMap<u64, &crate::wal::WalEntry> = Default::default();
        let mut order: Vec<u64> = Vec::new();
        for entry in &entries {
            match entry.entry_type {
                WalEntryType::PageWrite | WalEntryType::PageFree => {
                    if !last_op.contains_key(&entry.page_id) {
                        order.push(entry.page_id);
                    }
                    last_op.insert(entry.page_id, entry);
                }
                WalEntryType::Checkpoint | WalEntryType::Commit => {}
            }
        }

        for page_id_raw in order {
            let entry = last_op[&page_id_raw];
            let page_id = PageId(entry.page_id);
            match entry.entry_type {
                WalEntryType::PageWrite => {
                    while pm.page_count() <= entry.page_id {
                        pm.allocate_page()?;
                    }

                    let mut page = pm.read_page(page_id)?;
                    let copy_len = entry.data.len().min(PAGE_DATA_SIZE);
                    page.data[..copy_len].copy_from_slice(&entry.data[..copy_len]);
                    if copy_len < PAGE_DATA_SIZE {
                        page.data[copy_len..].fill(0);
                    }
                    page.header.page_id = entry.page_id;
                    page.header.lsn = entry.lsn;
                    page.header.page_type = PageType::Data as u8;
                    page.header.free_space = (PAGE_DATA_SIZE - copy_len) as u16;
                    page.header.checksum = page.compute_checksum();

                    pm.write_page(page_id, &page)?;
                    count += 1;
                }
                WalEntryType::PageFree => {
                    // Mark the page Free without touching the free list; the
                    // list is rebuilt from page types after replay so it stays
                    // consistent with WAL-derived page states.
                    if entry.page_id < pm.page_count() {
                        let mut page = Page::zeroed();
                        page.header.page_id = entry.page_id;
                        page.header.page_type = PageType::Free as u8;
                        pm.write_page(page_id, &page)?;
                        self.buffer_pool.invalidate(page_id);
                        count += 1;
                    }
                }
                WalEntryType::Checkpoint | WalEntryType::Commit => {}
            }
        }

        if count > 0 {
            // Header updates (free list head, page count) are not fsynced on
            // every write, so after a crash the free list can disagree with the
            // replayed page states. Rebuild it from page types: the WAL is the
            // sole source of truth for which pages are Free vs Data. With a
            // memory map the free pages are known, so only those are read
            // (and verified); otherwise every page is scanned.
            let rebuilt = match &anchor {
                Some(a) => pm.rebuild_free_list_from(a.table.free_pages())?,
                None => false,
            };
            if !rebuilt {
                pm.rebuild_free_list()?;
            }
            pm.sync()?;
            match anchor {
                // The replayed pages are now durable, which makes this a
                // checkpoint: anchor the rolled forward map on a fresh entry
                // before truncating the entries it was derived from.
                Some(anchor) => {
                    self.anchor_checkpoint(wal, &pm, anchor.table)?;
                }
                None => {
                    let next_lsn = wal.next_lsn();
                    wal.truncate(next_lsn)?;
                }
            }
            info!(count, "WAL recovery replayed entries");
        } else if let Some(anchor) = anchor {
            // Nothing to replay, so no entry after the anchor changed the map.
            *self.map_anchor.lock() = Some(anchor);
        }

        Ok(count)
    }

    /// Append a checkpoint entry with a fresh token, persist `table` as the map
    /// as of that entry, and truncate the WAL to it. The page file must already
    /// be durable. When the snapshot cannot be written the anchor is dropped
    /// (the next open scans) but the checkpoint still completes.
    fn anchor_checkpoint(
        &self,
        wal: &mut Wal,
        pm: &PageManager,
        table: MemoryTable,
    ) -> MenteResult<u64> {
        let token = memory_map::new_token();
        let lsn = wal.append(WalEntryType::Checkpoint, 0, &token)?;
        wal.sync()?;
        let anchor = MapAnchor {
            token,
            page_count: pm.page_count(),
            table,
            pending: Default::default(),
        };
        // The file is written before the slot is locked, so stamp lookups on
        // the recall path never wait on snapshot I/O.
        let written = memory_map::write_snapshot(&self.dir, &anchor);
        match written {
            Ok(()) => *self.map_anchor.lock() = Some(anchor),
            Err(e) => {
                tracing::warn!(error = %e, "memory map snapshot write failed, next open will scan pages");
                memory_map::remove_snapshot(&self.dir);
                *self.map_anchor.lock() = None;
            }
        }
        wal.truncate(lsn)?;
        Ok(lsn)
    }

    /// Gracefully shut down: flush dirty pages, sync files.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mentedb_storage::StorageEngine;
    /// # let engine = StorageEngine::open("/tmp/mentedb".as_ref())?;
    /// engine.close()?;
    /// # Ok::<(), mentedb_core::error::MenteError>(())
    /// ```
    pub fn close(&self) -> MenteResult<()> {
        let mut pm = self.page_manager.lock();
        self.buffer_pool.flush_all(&mut pm)?;
        pm.sync()?;
        self.wal.lock().sync()?;
        // Release the process lock so the directory can be reopened, by this
        // process or another, without waiting for us to exit.
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

    // ---- low-level page operations ----

    /// Allocate a fresh page (for internal/test use).
    ///
    /// **WARNING**: In multi-process scenarios, prefer `store_memory` which
    /// allocates under the WAL flock. This method does NOT acquire the flock.
    pub fn allocate_page(&self) -> MenteResult<PageId> {
        self.page_manager.lock().allocate_page()
    }

    /// Read a page through the buffer pool (lock-free — no WAL access).
    pub fn read_page(&self, page_id: PageId) -> MenteResult<Box<Page>> {
        self.buffer_pool
            .fetch_page(page_id, &mut self.page_manager.lock())
    }

    /// Buffer-pool cache activity, for metrics (hit ratio, evictions, residency).
    pub fn buffer_stats(&self) -> crate::buffer::BufferStats {
        self.buffer_pool.stats()
    }

    /// Total pages in the store; multiply by `PAGE_SIZE` for on-disk bytes.
    pub fn page_count(&self) -> u64 {
        self.page_manager.lock().page_count()
    }

    /// Write data into an already-allocated page with WAL protection.
    ///
    /// Acquires the WAL flock for the duration of the write transaction.
    /// For new pages, prefer `store_memory` which allocates + writes atomically.
    pub fn write_page(&self, page_id: PageId, data: &[u8]) -> MenteResult<()> {
        let id = memory_map::decode_memory(data).ok().map(|n| n.id);
        self.write_page_as(page_id, data, id)
    }

    /// `write_page` for data known to decode as memory `id` (or as none).
    fn write_page_as(
        &self,
        page_id: PageId,
        data: &[u8],
        id: Option<mentedb_core::types::MemoryId>,
    ) -> MenteResult<()> {
        let lsn = {
            let mut wal = self.wal.lock();
            wal.lock_exclusive()?;
            wal.reload_lsn()?;
            let lsn = wal.append(WalEntryType::PageWrite, page_id.0, data)?;
            wal.sync()?;
            self.note_map_write(page_id, id, lsn);
            wal.unlock()?;
            lsn
        };

        self.apply_page_write(page_id, data, lsn)
    }

    /// Apply a page write to the buffer pool and page manager (after WAL).
    fn apply_page_write(&self, page_id: PageId, data: &[u8], lsn: u64) -> MenteResult<()> {
        let mut pm = self.page_manager.lock();
        let mut page = self.buffer_pool.fetch_page(page_id, &mut pm)?;
        drop(pm);

        let copy_len = data.len().min(PAGE_DATA_SIZE);
        page.data[..copy_len].copy_from_slice(&data[..copy_len]);
        if copy_len < PAGE_DATA_SIZE {
            page.data[copy_len..].fill(0);
        }
        page.header.lsn = lsn;
        page.header.page_type = PageType::Data as u8;
        page.header.free_space = (PAGE_DATA_SIZE - copy_len) as u16;
        page.header.checksum = page.compute_checksum();

        if self.buffer_pool.update_page(page_id, &page).is_err() {
            self.page_manager.lock().write_page(page_id, &page)?;
        }
        self.buffer_pool.unpin_page(page_id, true).ok();

        Ok(())
    }

    // ---- high-level memory operations ----

    /// Serialize and store a [`MemoryNode`] into a single page.
    ///
    /// The entire operation — page allocation, WAL append, page write — executes
    /// under a single WAL flock, making it safe across multiple processes.
    ///
    /// # Example
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
        let serialized =
            serde_json::to_vec(node).map_err(|e| MenteError::Serialization(e.to_string()))?;

        if serialized.len() + 4 > PAGE_DATA_SIZE {
            return Err(MenteError::CapacityExceeded(format!(
                "memory node serialized to {} bytes (max {})",
                serialized.len(),
                PAGE_DATA_SIZE - 4,
            )));
        }

        let mut buf = Vec::with_capacity(4 + serialized.len());
        buf.extend_from_slice(&(serialized.len() as u32).to_le_bytes());
        buf.extend_from_slice(&serialized);

        // Atomic write transaction: allocate + WAL + page write under one flock
        let (page_id, lsn) = {
            let mut wal = self.wal.lock();
            let mut pm = self.page_manager.lock();

            // Acquire flock and refresh state from disk
            wal.lock_exclusive()?;
            pm.reload_header()?;
            wal.reload_lsn()?;

            // Allocate page (using fresh page_count from disk)
            let page_id = pm.allocate_page()?;

            // WAL append + sync (WAL fsync guarantees durability;
            // page data is written but not fsynced — checkpoint handles that)
            let lsn = wal.append(WalEntryType::PageWrite, page_id.0, &buf)?;
            wal.sync()?;
            self.note_map_write(page_id, Some(node.id), lsn);

            // Write page data to disk (no fsync — recoverable from WAL)
            let mut page = Page::zeroed();
            page.header.page_id = page_id.0;
            let copy_len = buf.len().min(PAGE_DATA_SIZE);
            page.data[..copy_len].copy_from_slice(&buf[..copy_len]);
            page.header.lsn = lsn;
            page.header.page_type = PageType::Data as u8;
            page.header.free_space = (PAGE_DATA_SIZE - copy_len) as u16;
            page.header.checksum = page.compute_checksum();
            pm.write_page(page_id, &page)?;

            // Release flock — other processes can now write
            wal.unlock()?;

            (page_id, lsn)
        };

        // Drop any stale cached copy (the page may have been reused from the
        // free list); the next read loads the fresh data from disk.
        let _ = lsn;
        self.buffer_pool.invalidate(page_id);

        // Auto-checkpoint when WAL exceeds threshold to prevent unbounded growth.
        // This keeps reload_lsn() fast for subsequent writes.
        if self.wal.lock().file_size() > WAL_AUTO_CHECKPOINT_BYTES
            && let Err(e) = self.checkpoint()
        {
            tracing::warn!("auto-checkpoint failed: {e}");
        }

        info!(
            page_id = page_id.0,
            bytes = serialized.len(),
            "stored memory node"
        );
        Ok(page_id)
    }

    /// Store multiple [`MemoryNode`]s in a single locked transaction.
    ///
    /// Acquires the WAL flock once, writes all nodes, then releases. This avoids
    /// the per-write overhead of `reload_header` / `reload_lsn` for bulk inserts.
    /// Auto-checkpoints after the batch if the WAL exceeds the threshold.
    pub fn store_memory_batch(&self, nodes: &[MemoryNode]) -> MenteResult<Vec<PageId>> {
        // Phase 1: serialize all nodes upfront (no locks held)
        let mut bufs = Vec::with_capacity(nodes.len());
        for node in nodes {
            let serialized =
                serde_json::to_vec(node).map_err(|e| MenteError::Serialization(e.to_string()))?;
            if serialized.len() + 4 > PAGE_DATA_SIZE {
                return Err(MenteError::CapacityExceeded(format!(
                    "memory node serialized to {} bytes (max {})",
                    serialized.len(),
                    PAGE_DATA_SIZE - 4,
                )));
            }
            let mut buf = Vec::with_capacity(4 + serialized.len());
            buf.extend_from_slice(&(serialized.len() as u32).to_le_bytes());
            buf.extend_from_slice(&serialized);
            bufs.push(buf);
        }

        // Phase 2: single locked transaction for all writes
        let page_ids = {
            let mut wal = self.wal.lock();
            let mut pm = self.page_manager.lock();

            wal.lock_exclusive()?;
            pm.reload_header()?;
            wal.reload_lsn()?;

            let mut ids = Vec::with_capacity(bufs.len());
            for (buf, node) in bufs.iter().zip(nodes) {
                let page_id = pm.allocate_page()?;
                let lsn = wal.append(WalEntryType::PageWrite, page_id.0, buf)?;
                self.note_map_write(page_id, Some(node.id), lsn);

                let mut page = Page::zeroed();
                page.header.page_id = page_id.0;
                let copy_len = buf.len().min(PAGE_DATA_SIZE);
                page.data[..copy_len].copy_from_slice(&buf[..copy_len]);
                page.header.lsn = lsn;
                page.header.page_type = PageType::Data as u8;
                page.header.free_space = (PAGE_DATA_SIZE - copy_len) as u16;
                page.header.checksum = page.compute_checksum();
                pm.write_page(page_id, &page)?;

                ids.push(page_id);
            }

            // WAL fsync only — page data is recoverable from WAL on crash.
            // Checkpoint handles page file fsync.
            wal.sync()?;
            wal.unlock()?;

            ids
        };

        // Drop stale cached copies for pages reused from the free list.
        for page_id in &page_ids {
            self.buffer_pool.invalidate(*page_id);
        }

        // Auto-checkpoint if WAL grew too large
        if self.wal.lock().file_size() > WAL_AUTO_CHECKPOINT_BYTES
            && let Err(e) = self.checkpoint()
        {
            tracing::warn!("auto-checkpoint failed: {e}");
        }

        info!(count = page_ids.len(), "stored memory batch");
        Ok(page_ids)
    }

    /// Update a [`MemoryNode`] in place on its existing page.
    ///
    /// The write goes through the WAL-protected `write_page` path, so it is
    /// crash-durable and keeps the buffer pool coherent. Unlike storing to a
    /// fresh page, this never orphans the old copy.
    pub fn update_memory(&self, page_id: PageId, node: &MemoryNode) -> MenteResult<()> {
        let serialized =
            serde_json::to_vec(node).map_err(|e| MenteError::Serialization(e.to_string()))?;

        if serialized.len() + 4 > PAGE_DATA_SIZE {
            return Err(MenteError::CapacityExceeded(format!(
                "memory node serialized to {} bytes (max {})",
                serialized.len(),
                PAGE_DATA_SIZE - 4,
            )));
        }

        let mut buf = Vec::with_capacity(4 + serialized.len());
        buf.extend_from_slice(&(serialized.len() as u32).to_le_bytes());
        buf.extend_from_slice(&serialized);

        self.write_page_as(page_id, &buf, Some(node.id))
    }

    /// Delete a memory by returning its page to the free list.
    ///
    /// The deletion is WAL-logged before the page is freed, so it survives a
    /// crash: recovery replays the `PageFree` entry and the memory does not
    /// resurrect on reopen. The freed page is reused by later allocations.
    pub fn delete_memory(&self, page_id: PageId) -> MenteResult<()> {
        {
            let mut wal = self.wal.lock();
            let mut pm = self.page_manager.lock();

            wal.lock_exclusive()?;
            pm.reload_header()?;
            wal.reload_lsn()?;

            // WAL fsync guarantees the deletion is durable before the page
            // is touched; the free-list update itself is recoverable.
            let lsn = wal.append(WalEntryType::PageFree, page_id.0, &[])?;
            wal.sync()?;
            self.note_map_write(page_id, None, lsn);

            pm.free_page(page_id)?;
            wal.unlock()?;
        }

        // A stale cached copy must never be served or flushed back.
        self.buffer_pool.invalidate(page_id);

        info!(page_id = page_id.0, "deleted memory node");
        Ok(())
    }

    /// Load and deserialize a [`MemoryNode`] from the given page.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mentedb_storage::{StorageEngine, PageId};
    /// # let engine = StorageEngine::open("/tmp/mentedb".as_ref())?;
    /// let node = engine.load_memory(PageId(1))?;
    /// println!("memory: {}", node.content);
    /// # Ok::<(), mentedb_core::error::MenteError>(())
    /// ```
    pub fn load_memory(&self, page_id: PageId) -> MenteResult<MemoryNode> {
        self.memory_loads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let page = self.read_page(page_id)?;
        self.buffer_pool.unpin_page(page_id, false).ok();

        if PageType::from(page.header.page_type) != PageType::Data {
            return Err(MenteError::Storage(format!(
                "page {} is not a data page",
                page_id.0
            )));
        }

        memory_map::decode_memory(&page.data)
    }

    // ---- durability ----

    /// Checkpoint: flush all dirty pages, sync to disk, and truncate the WAL.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mentedb_storage::StorageEngine;
    /// # let engine = StorageEngine::open("/tmp/mentedb".as_ref())?;
    /// // After a batch of writes, checkpoint to reclaim WAL space
    /// engine.checkpoint()?;
    /// # Ok::<(), mentedb_core::error::MenteError>(())
    /// ```
    pub fn checkpoint(&self) -> MenteResult<()> {
        let mut wal = self.wal.lock();
        let mut pm = self.page_manager.lock();

        wal.lock_exclusive()?;
        let result = self.checkpoint_locked(&mut wal, &mut pm, None);
        let unlocked = wal.unlock();
        let lsn = result?;
        unlocked?;

        info!(lsn, "checkpoint complete");
        Ok(())
    }

    /// Checkpoint under the WAL flock. `fresh` is a memory table known to be
    /// current (a full scan just built it); otherwise the anchored table is
    /// rolled forward over the WAL. The snapshot is rewritten only when the
    /// map changed; an unchanged map keeps its token, so the new checkpoint
    /// entry still anchors the existing snapshot.
    fn checkpoint_locked(
        &self,
        wal: &mut Wal,
        pm: &mut PageManager,
        fresh: Option<MemoryTable>,
    ) -> MenteResult<u64> {
        wal.reload_lsn()?;

        self.buffer_pool.flush_all(pm)?;
        pm.sync()?;

        let table = match fresh {
            Some(t) => Some(t),
            None => {
                let current = self.map_anchor.lock().take();
                match current {
                    Some(mut anchor) => {
                        let entries = wal.iterate()?;
                        match anchor.advance(&entries) {
                            Some(true) => Some(anchor.table),
                            Some(false) => {
                                // Unchanged: re-anchor on the same token, the
                                // snapshot on disk already holds this map.
                                let lsn = wal.append(WalEntryType::Checkpoint, 0, &anchor.token)?;
                                wal.sync()?;
                                *self.map_anchor.lock() = Some(anchor);
                                wal.truncate(lsn)?;
                                return Ok(lsn);
                            }
                            None => {
                                tracing::warn!(
                                    "memory map anchor left the WAL, next open will scan pages"
                                );
                                None
                            }
                        }
                    }
                    None => None,
                }
            }
        };

        match table {
            Some(table) => self.anchor_checkpoint(wal, pm, table),
            None => {
                // No map to persist. A fresh token keeps any old snapshot from
                // ever matching this WAL.
                let lsn = wal.append(WalEntryType::Checkpoint, 0, &memory_map::new_token())?;
                wal.sync()?;
                wal.truncate(lsn)?;
                Ok(lsn)
            }
        }
    }

    /// Every memory in storage with its page and write stamp, in page order:
    /// the same memories, in the same order, as [`Self::scan_all_memories`].
    ///
    /// Served from the persisted memory map rolled forward over the WAL, which
    /// reads no pages. When there is no usable map this falls back to a full
    /// page scan and then checkpoints, persisting the map so the next call
    /// (and the next open) is fast. Holds the WAL lock throughout, so writers
    /// wait; it is meant for open, before the database serves traffic.
    pub fn memory_map(&self) -> MenteResult<Vec<MappedMemory>> {
        let mut wal = self.wal.lock();
        wal.lock_exclusive()?;
        let result = self.memory_map_locked(&mut wal);
        let unlocked = wal.unlock();
        let map = result?;
        unlocked?;
        Ok(map)
    }

    fn memory_map_locked(&self, wal: &mut Wal) -> MenteResult<Vec<MappedMemory>> {
        let entries = wal.iterate()?;
        {
            let mut slot = self.map_anchor.lock();
            if let Some(anchor) = slot.as_ref() {
                let mut current = anchor.clone();
                if current.advance(&entries).is_some() {
                    return Ok(current.table.memories());
                }
                *slot = None;
            }
        }

        let mut pm = self.page_manager.lock();
        let table = self.scan_table(&mut pm)?;
        let memories = table.memories();
        info!(
            memories = memories.len(),
            "memory map rebuilt by full page scan"
        );
        self.checkpoint_locked(wal, &mut pm, Some(table))?;
        Ok(memories)
    }

    /// Build the memory table by reading every page.
    fn scan_table(&self, pm: &mut PageManager) -> MenteResult<MemoryTable> {
        pm.reload_header()?;
        let count = pm.page_count();
        let mut table = MemoryTable::default();
        for i in 1..count {
            let page_id = PageId(i);
            let Ok(page) = self.buffer_pool.fetch_page(page_id, pm) else {
                continue;
            };
            self.buffer_pool.unpin_page(page_id, false).ok();
            match PageType::from(page.header.page_type) {
                PageType::Data => {
                    if let Ok(node) = memory_map::decode_memory(&page.data) {
                        table.set_scanned(i, node.id);
                    }
                }
                PageType::Free => table.set_free(i),
                _ => {}
            }
        }
        Ok(table)
    }

    /// The memory and write stamp the persisted map records for `page`, as of
    /// the last checkpoint. None when the map is not anchored or the page holds
    /// no memory.
    pub fn memory_stamp(&self, page: PageId) -> Option<(mentedb_core::types::MemoryId, u64)> {
        self.map_anchor.lock().as_ref()?.current(page.0)
    }

    /// Record a logged write in the anchored map's pending set. Called with
    /// the WAL lock held, right after the append, so it is ordered with the
    /// checkpoint that later rolls the same entry into the table.
    fn note_map_write(&self, page: PageId, id: Option<mentedb_core::types::MemoryId>, lsn: u64) {
        if let Some(anchor) = self.map_anchor.lock().as_mut() {
            anchor.note_write(page.0, id, lsn);
        }
    }

    /// Memory loads served since open, whether the page came from the buffer
    /// pool or from disk. Counts what a caller asked for, so a test can bound
    /// the work of an operation independent of cache state.
    pub fn memory_loads(&self) -> u64 {
        self.memory_loads.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Pages read from disk since open (buffer pool misses and direct reads).
    pub fn page_reads(&self) -> u64 {
        self.page_manager.lock().reads()
    }

    /// Scan all pages and return (MemoryId, PageId) pairs for every valid memory node.
    ///
    /// Refreshes the page count from disk before scanning so pages written by
    /// other processes are included. Used to rebuild the page map on startup.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mentedb_storage::StorageEngine;
    /// # let engine = StorageEngine::open("/tmp/mentedb".as_ref())?;
    /// let memories = engine.scan_all_memories();
    /// for (memory_id, page_id) in &memories {
    ///     println!("{memory_id} -> page {}", page_id.0);
    /// }
    /// # Ok::<(), mentedb_core::error::MenteError>(())
    /// ```
    pub fn scan_all_memories(&self) -> Vec<(mentedb_core::types::MemoryId, PageId)> {
        let mut pm = self.page_manager.lock();
        // Refresh from disk to see pages written by other processes
        let _ = pm.reload_header();
        let count = pm.page_count();
        drop(pm);

        let mut results = Vec::new();
        for i in 1..count {
            let page_id = PageId(i);
            if let Ok(node) = self.load_memory(page_id) {
                results.push((node.id, page_id));
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_map::MappedMemory;
    use mentedb_core::memory::MemoryType;
    use mentedb_core::types::AgentId;

    fn setup() -> (tempfile::TempDir, StorageEngine) {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(dir.path()).unwrap();
        (dir, engine)
    }

    #[test]
    fn test_allocate_write_read() {
        let (_dir, engine) = setup();

        let pid = engine.allocate_page().unwrap();
        engine.write_page(pid, b"hello storage engine").unwrap();

        let page = engine.read_page(pid).unwrap();
        assert_eq!(&page.data[..20], b"hello storage engine");
        engine.buffer_pool.unpin_page(pid, false).ok();
    }

    #[test]
    fn test_store_and_load_memory() {
        let (_dir, engine) = setup();

        let node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Episodic,
            "The user prefers Rust over Go".to_string(),
            vec![0.1, 0.2, 0.3, 0.4],
        );

        let page_id = engine.store_memory(&node).unwrap();
        let loaded = engine.load_memory(page_id).unwrap();

        assert_eq!(node.id, loaded.id);
        assert_eq!(node.content, loaded.content);
        assert_eq!(node.embedding, loaded.embedding);
        assert_eq!(node.memory_type, loaded.memory_type);
    }

    #[test]
    fn test_checkpoint() {
        let (_dir, engine) = setup();

        let node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            "checkpoint test".to_string(),
            vec![1.0, 2.0],
        );

        let pid = engine.store_memory(&node).unwrap();
        engine.checkpoint().unwrap();

        let loaded = engine.load_memory(pid).unwrap();
        assert_eq!(loaded.content, "checkpoint test");
    }

    #[test]
    fn test_close_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let pid;
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let node = MemoryNode::new(
                AgentId::new(),
                MemoryType::Procedural,
                "persist across close".to_string(),
                vec![0.5],
            );
            pid = engine.store_memory(&node).unwrap();
            engine.close().unwrap();
        }
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let loaded = engine.load_memory(pid).unwrap();
            assert_eq!(loaded.content, "persist across close");
        }
    }

    #[test]
    fn test_crash_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let mut ids = Vec::new();
        let mut contents = Vec::new();
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            for i in 0..3 {
                let content = format!("crash-recovery-{i}");
                let node = MemoryNode::new(
                    AgentId::new(),
                    MemoryType::Episodic,
                    content.clone(),
                    vec![i as f32],
                );
                let pid = engine.store_memory(&node).unwrap();
                ids.push(pid);
                contents.push(content);
            }
            // Simulate crash: sync the WAL but do NOT call close/checkpoint.
            engine.wal.lock().sync().unwrap();
        }
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            for (pid, expected) in ids.iter().zip(contents.iter()) {
                let loaded = engine.load_memory(*pid).unwrap();
                assert_eq!(&loaded.content, expected);
            }
        }
    }

    #[test]
    fn test_recovery_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let pid;
        let content = "idempotent-check".to_string();
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let node = MemoryNode::new(
                AgentId::new(),
                MemoryType::Semantic,
                content.clone(),
                vec![1.0, 2.0],
            );
            pid = engine.store_memory(&node).unwrap();
            engine.checkpoint().unwrap();
            engine.close().unwrap();
        }
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let loaded = engine.load_memory(pid).unwrap();
            assert_eq!(loaded.content, content);
        }
    }

    #[test]
    fn test_partial_write_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let mut ids = Vec::new();
        let mut contents = Vec::new();
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            for i in 0..3 {
                let content = format!("checkpointed-{i}");
                let node = MemoryNode::new(
                    AgentId::new(),
                    MemoryType::Semantic,
                    content.clone(),
                    vec![i as f32],
                );
                let pid = engine.store_memory(&node).unwrap();
                ids.push(pid);
                contents.push(content);
            }
            engine.checkpoint().unwrap();

            for i in 3..5 {
                let content = format!("unckeckpointed-{i}");
                let node = MemoryNode::new(
                    AgentId::new(),
                    MemoryType::Episodic,
                    content.clone(),
                    vec![i as f32],
                );
                let pid = engine.store_memory(&node).unwrap();
                ids.push(pid);
                contents.push(content);
            }
            // Simulate crash — sync WAL but don't close.
            engine.wal.lock().sync().unwrap();
        }
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            for (pid, expected) in ids.iter().zip(contents.iter()) {
                let loaded = engine.load_memory(*pid).unwrap();
                assert_eq!(&loaded.content, expected);
            }
        }
    }

    #[test]
    fn test_delete_memory_durable() {
        let dir = tempfile::tempdir().unwrap();
        let pid;
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let node = MemoryNode::new(
                AgentId::new(),
                MemoryType::Semantic,
                "to be deleted".to_string(),
                vec![1.0],
            );
            pid = engine.store_memory(&node).unwrap();
            engine.delete_memory(pid).unwrap();
            assert!(engine.load_memory(pid).is_err());
            assert!(engine.scan_all_memories().is_empty());
            engine.close().unwrap();
        }
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            assert!(
                engine.load_memory(pid).is_err(),
                "deleted memory must not resurrect on reopen"
            );
            assert!(engine.scan_all_memories().is_empty());
        }
    }

    #[test]
    fn test_delete_survives_crash() {
        let dir = tempfile::tempdir().unwrap();
        let pid;
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let node = MemoryNode::new(
                AgentId::new(),
                MemoryType::Semantic,
                "crash delete".to_string(),
                vec![1.0],
            );
            pid = engine.store_memory(&node).unwrap();
            engine.delete_memory(pid).unwrap();
            // Simulate crash: no close, no checkpoint.
        }
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            assert!(
                engine.load_memory(pid).is_err(),
                "deletion must survive a crash via WAL replay"
            );
            assert!(engine.scan_all_memories().is_empty());
        }
    }

    #[test]
    fn test_deleted_page_reused() {
        let (_dir, engine) = setup();

        let a = MemoryNode::new(AgentId::new(), MemoryType::Semantic, "a".into(), vec![1.0]);
        let pid_a = engine.store_memory(&a).unwrap();
        engine.delete_memory(pid_a).unwrap();

        let b = MemoryNode::new(AgentId::new(), MemoryType::Semantic, "b".into(), vec![2.0]);
        let pid_b = engine.store_memory(&b).unwrap();
        assert_eq!(pid_a, pid_b, "freed page should be reused");

        let loaded = engine.load_memory(pid_b).unwrap();
        assert_eq!(loaded.content, "b");
    }

    #[test]
    fn test_delete_reuse_crash_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let pid;
        let b_id;
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let a = MemoryNode::new(AgentId::new(), MemoryType::Semantic, "a".into(), vec![1.0]);
            pid = engine.store_memory(&a).unwrap();
            engine.delete_memory(pid).unwrap();
            let b = MemoryNode::new(AgentId::new(), MemoryType::Semantic, "b".into(), vec![2.0]);
            let pid_b = engine.store_memory(&b).unwrap();
            assert_eq!(pid, pid_b);
            b_id = b.id;
            // Simulate crash: the WAL now holds PageFree(p) then PageWrite(p).
        }
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let loaded = engine.load_memory(pid).unwrap();
            assert_eq!(loaded.content, "b", "later write must win over the free");
            assert_eq!(loaded.id, b_id);
            // The page must NOT be on the free list: a fresh allocation must
            // not clobber b.
            let c = MemoryNode::new(AgentId::new(), MemoryType::Semantic, "c".into(), vec![3.0]);
            let pid_c = engine.store_memory(&c).unwrap();
            assert_ne!(pid_c, pid, "recovered free list must exclude reused page");
            assert_eq!(engine.load_memory(pid).unwrap().content, "b");
        }
    }

    #[test]
    fn test_update_memory_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let pid;
        let id;
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let mut node = MemoryNode::new(
                AgentId::new(),
                MemoryType::Semantic,
                "original".to_string(),
                vec![1.0],
            );
            pid = engine.store_memory(&node).unwrap();
            id = node.id;

            node.content = "updated".to_string();
            engine.update_memory(pid, &node).unwrap();

            let loaded = engine.load_memory(pid).unwrap();
            assert_eq!(loaded.content, "updated");
            // No orphan copy: exactly one entry in a full scan.
            let scanned = engine.scan_all_memories();
            assert_eq!(scanned.len(), 1);
            engine.close().unwrap();
        }
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let loaded = engine.load_memory(pid).unwrap();
            assert_eq!(loaded.content, "updated");
            assert_eq!(loaded.id, id);
            assert_eq!(engine.scan_all_memories().len(), 1);
        }
    }

    #[test]
    fn test_concurrent_open_is_rejected() {
        let dir = tempfile::tempdir().unwrap();

        // The per operation flock serializes individual write transactions,
        // but layers above the storage engine cache state across operations,
        // so a second concurrent engine rolls the database back to its own
        // open time snapshot. The process lock must refuse it.
        let engine1 = StorageEngine::open(dir.path()).unwrap();
        let second = StorageEngine::open(dir.path());
        assert!(second.is_err(), "second concurrent open must fail");
        let msg = second.err().unwrap().to_string();
        assert!(msg.contains("locked"), "error names the lock: {msg}");

        // Close releases the lock; the directory can be opened again.
        engine1.close().unwrap();
        let engine2 = StorageEngine::open(dir.path()).unwrap();
        engine2.close().unwrap();
    }

    #[test]
    fn test_concurrent_writes_from_threads() {
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let engine = Arc::new(StorageEngine::open(dir.path()).unwrap());

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let eng = Arc::clone(&engine);
                std::thread::spawn(move || {
                    let node = MemoryNode::new(
                        AgentId::new(),
                        MemoryType::Episodic,
                        format!("thread-{i}"),
                        vec![i as f32],
                    );
                    eng.store_memory(&node).unwrap()
                })
            })
            .collect();

        let pids: Vec<PageId> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All 10 writes succeeded and are readable
        for (i, pid) in pids.iter().enumerate() {
            let loaded = engine.load_memory(*pid).unwrap();
            assert_eq!(loaded.content, format!("thread-{i}"));
        }
    }

    // ---- persisted memory map ----

    fn node(content: &str) -> MemoryNode {
        MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            content.to_string(),
            vec![1.0, 2.0],
        )
    }

    /// (memory id, page) pairs, the part of the map a full scan can confirm.
    fn ids(map: &[MappedMemory]) -> Vec<(mentedb_core::types::MemoryId, PageId)> {
        map.iter().map(|m| (m.id, m.page)).collect()
    }

    /// Open a second, independent engine over the same files after `engine`
    /// crashed, and return what a full page scan sees there.
    fn scan_truth(dir: &Path) -> Vec<(mentedb_core::types::MemoryId, PageId)> {
        let engine = StorageEngine::open(dir).unwrap();
        let scanned = engine.scan_all_memories();
        engine.release_process_lock();
        scanned
    }

    #[test]
    fn open_after_checkpoint_reads_no_pages() {
        let dir = tempfile::tempdir().unwrap();
        let n = 300;
        let mut stored = Vec::new();
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            for i in 0..n {
                let m = node(&format!("m{i}"));
                let pid = engine.store_memory(&m).unwrap();
                stored.push((m.id, pid));
            }
            // First map request scans once and persists the map.
            assert_eq!(ids(&engine.memory_map().unwrap()), stored);
            engine.checkpoint().unwrap();
            engine.close().unwrap();
        }
        let engine = StorageEngine::open(dir.path()).unwrap();
        let map = engine.memory_map().unwrap();
        assert_eq!(ids(&map), stored);
        assert_eq!(
            engine.page_reads(),
            0,
            "open must serve the map from the snapshot, not by reading {n} pages"
        );
    }

    #[test]
    fn open_replays_only_wal_pages_after_crash() {
        let dir = tempfile::tempdir().unwrap();
        let mut stored = Vec::new();
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            for i in 0..200 {
                let m = node(&format!("base{i}"));
                stored.push((m.id, engine.store_memory(&m).unwrap()));
            }
            engine.memory_map().unwrap();
            engine.checkpoint().unwrap();
            for i in 0..5 {
                let m = node(&format!("after{i}"));
                stored.push((m.id, engine.store_memory(&m).unwrap()));
            }
            // Crash: nothing flushed or checkpointed beyond the WAL fsync.
            engine.release_process_lock();
        }
        let engine = StorageEngine::open(dir.path()).unwrap();
        let map = engine.memory_map().unwrap();
        assert_eq!(ids(&map), stored, "uncheckpointed stores survive the crash");
        assert_eq!(
            engine.page_reads(),
            5,
            "replay reads the 5 WAL pages only, not the 200 checkpointed ones"
        );
        engine.release_process_lock();
        assert_eq!(scan_truth(dir.path()), stored);
    }

    #[test]
    fn deletes_and_updates_after_checkpoint_are_replayed() {
        let dir = tempfile::tempdir().unwrap();
        let (keep, gone, updated);
        let updated_pid;
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let k = node("keep");
            let g = node("gone");
            let mut u = node("old");
            let kp = engine.store_memory(&k).unwrap();
            let gp = engine.store_memory(&g).unwrap();
            updated_pid = engine.store_memory(&u).unwrap();
            engine.memory_map().unwrap();
            engine.checkpoint().unwrap();
            engine.delete_memory(gp).unwrap();
            u.content = "new".into();
            engine.update_memory(updated_pid, &u).unwrap();
            keep = (k.id, kp);
            gone = g.id;
            updated = u.id;
            engine.release_process_lock();
        }
        let engine = StorageEngine::open(dir.path()).unwrap();
        let map = engine.memory_map().unwrap();
        assert!(
            map.iter().all(|m| m.id != gone),
            "deleted memory stays deleted"
        );
        assert!(ids(&map).contains(&keep));
        assert!(ids(&map).contains(&(updated, updated_pid)));
        assert_eq!(engine.load_memory(updated_pid).unwrap().content, "new");
        assert_eq!(map.len(), 2);
        engine.release_process_lock();
        assert_eq!(scan_truth(dir.path()), ids(&map));
    }

    #[test]
    fn free_list_reuse_after_checkpoint_maps_the_new_memory() {
        let dir = tempfile::tempdir().unwrap();
        let (a_id, b_id, pid);
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let a = node("a");
            pid = engine.store_memory(&a).unwrap();
            engine.store_memory(&node("other")).unwrap();
            engine.memory_map().unwrap();
            engine.checkpoint().unwrap();
            engine.delete_memory(pid).unwrap();
            let b = node("b");
            assert_eq!(engine.store_memory(&b).unwrap(), pid, "freed page reused");
            a_id = a.id;
            b_id = b.id;
            engine.release_process_lock();
        }
        let engine = StorageEngine::open(dir.path()).unwrap();
        let map = engine.memory_map().unwrap();
        assert!(
            map.iter().all(|m| m.id != a_id),
            "no stale id for the reused page"
        );
        assert!(ids(&map).contains(&(b_id, pid)));
        // The reused page is not on the rebuilt free list.
        let c = node("c");
        assert_ne!(engine.store_memory(&c).unwrap(), pid);
        assert_eq!(engine.load_memory(pid).unwrap().id, b_id);
        engine.release_process_lock();
        let truth = scan_truth(dir.path());
        assert!(truth.contains(&(b_id, pid)));
        assert!(truth.iter().all(|(id, _)| *id != a_id));
    }

    #[test]
    fn free_list_rebuilt_from_map_after_crash() {
        let dir = tempfile::tempdir().unwrap();
        let mut freed = Vec::new();
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let mut pids = Vec::new();
            for i in 0..50 {
                pids.push(engine.store_memory(&node(&format!("m{i}"))).unwrap());
            }
            engine.delete_memory(pids[3]).unwrap();
            freed.push(pids[3]);
            engine.memory_map().unwrap();
            engine.checkpoint().unwrap();
            engine.delete_memory(pids[7]).unwrap();
            freed.push(pids[7]);
            engine.release_process_lock();
        }
        let engine = StorageEngine::open(dir.path()).unwrap();
        // One replayed free plus verification reads of the two free pages,
        // not a scan of all 50.
        assert!(engine.page_reads() <= 3, "reads {}", engine.page_reads());
        let a = engine.store_memory(&node("x")).unwrap();
        let b = engine.store_memory(&node("y")).unwrap();
        let mut got = vec![a, b];
        got.sort_by_key(|p| p.0);
        assert_eq!(got, freed, "both freed pages are reusable after recovery");
        let c = engine.store_memory(&node("z")).unwrap();
        assert_eq!(c, PageId(51), "then allocation grows the file");
    }

    #[test]
    fn missing_corrupt_or_stale_snapshot_falls_back_to_scan() {
        let dir = tempfile::tempdir().unwrap();
        let mut stored = Vec::new();
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            for i in 0..40 {
                let m = node(&format!("m{i}"));
                stored.push((m.id, engine.store_memory(&m).unwrap()));
            }
            engine.memory_map().unwrap();
            engine.checkpoint().unwrap();
            engine.close().unwrap();
        }
        let snap = dir.path().join(memory_map::MEMORY_MAP_FILE);
        let good = std::fs::read(&snap).unwrap();

        let check = |label: &str| {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let map = engine.memory_map().unwrap();
            assert_eq!(ids(&map), stored, "{label}: same result as a scan");
            assert!(
                engine.page_reads() >= 40,
                "{label}: must have scanned, read {}",
                engine.page_reads()
            );
            engine.close().unwrap();
            // The scan re-persisted a valid map.
            let engine = StorageEngine::open(dir.path()).unwrap();
            assert_eq!(ids(&engine.memory_map().unwrap()), stored);
            assert_eq!(engine.page_reads(), 0, "{label}: next open is fast again");
            engine.close().unwrap();
        };

        std::fs::remove_file(&snap).unwrap();
        check("missing");

        let mut bad = std::fs::read(&snap).unwrap();
        let mid = bad.len() / 2;
        bad[mid] ^= 0x55;
        std::fs::write(&snap, &bad).unwrap();
        check("corrupt");

        // Stale: a snapshot from an earlier checkpoint whose token has left
        // the WAL, even though the file itself is intact.
        std::fs::write(&snap, &good).unwrap();
        check("stale");
    }

    #[test]
    fn stale_snapshot_with_unchanged_token_reuse_is_never_matched() {
        // A checkpoint whose snapshot write is skipped (no map) must not let an
        // older snapshot match afterwards: deletes after it would resurrect.
        let dir = tempfile::tempdir().unwrap();
        let (a, b);
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let na = node("a");
            let nb = node("b");
            a = engine.store_memory(&na).unwrap();
            b = (nb.id, engine.store_memory(&nb).unwrap());
            engine.memory_map().unwrap();
            engine.checkpoint().unwrap();
            // Drop the anchor as if the map were never loaded, then change the
            // map and checkpoint without persisting it.
            *engine.map_anchor.lock() = None;
            engine.delete_memory(a).unwrap();
            engine.checkpoint().unwrap();
            engine.close().unwrap();
        }
        let engine = StorageEngine::open(dir.path()).unwrap();
        assert_eq!(ids(&engine.memory_map().unwrap()), vec![b]);
    }

    #[test]
    fn handoff_between_handles_sees_writes_and_checkpoints() {
        let dir = tempfile::tempdir().unwrap();
        let mut stored = Vec::new();
        // Handle 1 builds the map and exits cleanly.
        {
            let e1 = StorageEngine::open(dir.path()).unwrap();
            for i in 0..20 {
                let m = node(&format!("h1-{i}"));
                stored.push((m.id, e1.store_memory(&m).unwrap()));
            }
            e1.memory_map().unwrap();
            e1.checkpoint().unwrap();
            e1.close().unwrap();
        }
        // Handle 2 (another process taking over) writes, checkpoints, writes
        // more, then dies.
        {
            let e2 = StorageEngine::open(dir.path()).unwrap();
            e2.memory_map().unwrap();
            for i in 0..10 {
                let m = node(&format!("h2-{i}"));
                stored.push((m.id, e2.store_memory(&m).unwrap()));
            }
            e2.checkpoint().unwrap();
            e2.delete_memory(stored[0].1).unwrap();
            stored.remove(0);
            let m = node("h2-last");
            stored.push((m.id, e2.store_memory(&m).unwrap()));
            e2.release_process_lock();
        }
        // Handle 3 reopens and sees all of it.
        let e3 = StorageEngine::open(dir.path()).unwrap();
        let mut got = ids(&e3.memory_map().unwrap());
        got.sort_by_key(|(_, p)| p.0);
        stored.sort_by_key(|(_, p)| p.0);
        assert_eq!(got, stored);
        assert!(e3.page_reads() < 10, "reads {}", e3.page_reads());
        e3.release_process_lock();
        assert_eq!(scan_truth(dir.path()), stored);
    }

    #[test]
    fn live_stamps_survive_checkpoint_and_crash_replay() {
        let dir = tempfile::tempdir().unwrap();
        let (b_page, b_live);
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            engine.store_memory(&node("a")).unwrap();
            engine.memory_map().unwrap();
            engine.checkpoint().unwrap();
            // A write after the checkpoint has its stamp before any checkpoint.
            let c = engine.store_memory(&node("c")).unwrap();
            let c_live = engine.memory_stamp(c).unwrap();
            engine.checkpoint().unwrap();
            assert_eq!(
                engine.memory_stamp(c).unwrap(),
                c_live,
                "checkpoint keeps it"
            );
            b_page = engine.store_memory(&node("b")).unwrap();
            b_live = engine.memory_stamp(b_page).unwrap();
            engine.release_process_lock();
        }
        let engine = StorageEngine::open(dir.path()).unwrap();
        let map = engine.memory_map().unwrap();
        let b = map.iter().find(|m| m.page == b_page).unwrap();
        assert_eq!(
            (b.id, b.stamp),
            b_live,
            "WAL replay reproduces the live stamp"
        );
    }

    #[test]
    fn unchanged_checkpoints_keep_the_snapshot_and_stamps() {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(dir.path()).unwrap();
        let pid = engine.store_memory(&node("a")).unwrap();
        engine.memory_map().unwrap();
        engine.checkpoint().unwrap();
        let snap = dir.path().join(memory_map::MEMORY_MAP_FILE);
        let before = std::fs::metadata(&snap).unwrap().modified().unwrap();
        let stamp = engine.memory_stamp(pid).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        engine.checkpoint().unwrap();
        engine.checkpoint().unwrap();
        assert_eq!(
            std::fs::metadata(&snap).unwrap().modified().unwrap(),
            before
        );
        assert_eq!(engine.memory_stamp(pid).unwrap(), stamp);
        // A write to the page changes its stamp at the next checkpoint.
        let mut n = engine.load_memory(pid).unwrap();
        n.content = "b".into();
        engine.update_memory(pid, &n).unwrap();
        engine.checkpoint().unwrap();
        assert_ne!(engine.memory_stamp(pid).unwrap().1, stamp.1);
        engine.close().unwrap();
        let engine = StorageEngine::open(dir.path()).unwrap();
        let map = engine.memory_map().unwrap();
        assert_eq!(map[0].stamp, engine.memory_stamp(pid).unwrap().1);
    }
}
