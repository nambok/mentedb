//! Storage Engine: facade that ties the page manager, WAL, and buffer pool together.

use std::path::Path;

use mentedb_core::MemoryNode;
use mentedb_core::error::{MenteError, MenteResult};

use parking_lot::Mutex;
use tracing::info;

use crate::buffer::BufferPool;
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
/// - **No DB-level lock on open.** Multiple processes can open the same database
///   simultaneously.
pub struct StorageEngine {
    page_manager: Mutex<PageManager>,
    buffer_pool: BufferPool,
    wal: Mutex<Wal>,
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

        let page_manager = PageManager::open(path)?;
        let buffer_pool = BufferPool::new(DEFAULT_BUFFER_POOL_SIZE);
        let wal = Wal::open(path)?;

        let engine = Self {
            page_manager: Mutex::new(page_manager),
            buffer_pool,
            wal: Mutex::new(wal),
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
        let entries = wal.iterate()?;
        let mut count = 0usize;
        let mut pm = self.page_manager.lock();

        // Refresh page count from disk — another process may have written pages.
        pm.reload_header()?;

        for entry in &entries {
            match entry.entry_type {
                WalEntryType::PageWrite => {
                    let page_id = PageId(entry.page_id);

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
                WalEntryType::Checkpoint | WalEntryType::Commit => {}
            }
        }

        if count > 0 {
            pm.sync()?;
            let next_lsn = wal.next_lsn();
            wal.truncate(next_lsn)?;
            info!(count, "WAL recovery replayed entries");
        }

        wal.unlock()?;
        Ok(count)
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
        info!("storage engine closed");
        Ok(())
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

    /// Write data into an already-allocated page with WAL protection.
    ///
    /// Acquires the WAL flock for the duration of the write transaction.
    /// For new pages, prefer `store_memory` which allocates + writes atomically.
    pub fn write_page(&self, page_id: PageId, data: &[u8]) -> MenteResult<()> {
        let lsn = {
            let mut wal = self.wal.lock();
            wal.lock_exclusive()?;
            wal.reload_lsn()?;
            let lsn = wal.append(WalEntryType::PageWrite, page_id.0, data)?;
            wal.sync()?;
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

        // Update buffer pool outside the flock (optional optimization)
        let _ = lsn; // buffer pool update uses the page already written to disk

        // Auto-checkpoint when WAL exceeds threshold to prevent unbounded growth.
        // This keeps reload_lsn() fast for subsequent writes.
        if self.wal.lock().file_size() > WAL_AUTO_CHECKPOINT_BYTES {
            if let Err(e) = self.checkpoint() {
                tracing::warn!("auto-checkpoint failed: {e}");
            }
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
            for buf in &bufs {
                let page_id = pm.allocate_page()?;
                let lsn = wal.append(WalEntryType::PageWrite, page_id.0, buf)?;

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

        // Auto-checkpoint if WAL grew too large
        if self.wal.lock().file_size() > WAL_AUTO_CHECKPOINT_BYTES {
            if let Err(e) = self.checkpoint() {
                tracing::warn!("auto-checkpoint failed: {e}");
            }
        }

        info!(count = page_ids.len(), "stored memory batch");
        Ok(page_ids)
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
        let page = self.read_page(page_id)?;
        self.buffer_pool.unpin_page(page_id, false).ok();

        let len = u32::from_le_bytes(page.data[..4].try_into().unwrap()) as usize;
        if len == 0 || len + 4 > PAGE_DATA_SIZE {
            return Err(MenteError::Storage(format!(
                "invalid memory node length prefix: {len}"
            )));
        }

        serde_json::from_slice(&page.data[4..4 + len])
            .map_err(|e| MenteError::Serialization(e.to_string()))
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
        wal.reload_lsn()?;

        self.buffer_pool.flush_all(&mut pm)?;
        pm.sync()?;

        let lsn = wal.append(WalEntryType::Checkpoint, 0, &[])?;
        wal.sync()?;
        wal.truncate(lsn)?;
        wal.unlock()?;

        info!(lsn, "checkpoint complete");
        Ok(())
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
    fn test_concurrent_open_no_lock_conflict() {
        let dir = tempfile::tempdir().unwrap();

        // Two engines open the same directory simultaneously — should succeed
        // now that we no longer hold an exclusive DB-level flock.
        let engine1 = StorageEngine::open(dir.path()).unwrap();
        let engine2 = StorageEngine::open(dir.path()).unwrap();

        // Both can write (serialized by WAL file lock)
        let node1 = MemoryNode::new(
            AgentId::new(),
            MemoryType::Episodic,
            "from engine 1".to_string(),
            vec![1.0],
        );
        let node2 = MemoryNode::new(
            AgentId::new(),
            MemoryType::Episodic,
            "from engine 2".to_string(),
            vec![2.0],
        );

        let pid1 = engine1.store_memory(&node1).unwrap();
        let pid2 = engine2.store_memory(&node2).unwrap();

        // Each engine can read what it wrote
        let loaded1 = engine1.load_memory(pid1).unwrap();
        assert_eq!(loaded1.content, "from engine 1");

        let loaded2 = engine2.load_memory(pid2).unwrap();
        assert_eq!(loaded2.content, "from engine 2");
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
}
