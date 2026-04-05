//! Storage Engine: facade that ties the page manager, WAL, and buffer pool together.

use std::path::Path;

use mentedb_core::MemoryNode;
use mentedb_core::error::{MenteError, MenteResult};

use tracing::info;

use crate::buffer::BufferPool;
use crate::page::{PAGE_DATA_SIZE, Page, PageId, PageManager, PageType};
use crate::wal::{Wal, WalEntryType};
/// Default number of page frames in the buffer pool.
const DEFAULT_BUFFER_POOL_SIZE: usize = 1024;

/// The unified storage engine for MenteDB.
///
/// Coordinates page allocation, caching, and write-ahead logging to provide
/// crash-safe, page-oriented storage for memory nodes.
pub struct StorageEngine {
    page_manager: PageManager,
    buffer_pool: BufferPool,
    wal: Wal,
}

impl StorageEngine {
    /// Open (or create) a storage engine rooted at `path`.
    ///
    /// `path` must be a directory; it will be created if it does not exist.
    /// After opening, any uncommitted WAL entries are replayed for crash recovery.
    pub fn open(path: &Path) -> MenteResult<Self> {
        std::fs::create_dir_all(path)?;

        let page_manager = PageManager::open(path)?;
        let buffer_pool = BufferPool::new(DEFAULT_BUFFER_POOL_SIZE);
        let wal = Wal::open(path)?;

        let mut engine = Self {
            page_manager,
            buffer_pool,
            wal,
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
    pub fn recover(&mut self) -> MenteResult<usize> {
        let entries = self.wal.iterate()?;
        let mut count = 0usize;

        for entry in &entries {
            match entry.entry_type {
                WalEntryType::PageWrite => {
                    let page_id = PageId(entry.page_id);

                    // Ensure the page file is large enough for this page id.
                    while self.page_manager.page_count() <= entry.page_id {
                        self.page_manager.allocate_page()?;
                    }

                    let mut page = self.page_manager.read_page(page_id)?;
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

                    self.page_manager.write_page(page_id, &page)?;
                    count += 1;
                }
                WalEntryType::Checkpoint | WalEntryType::Commit => {
                    // No page data to replay for these entry types.
                }
            }
        }

        if count > 0 {
            self.page_manager.sync()?;
            // Truncate the entire WAL — all entries have been applied.
            let next_lsn = self.wal.next_lsn();
            self.wal.truncate(next_lsn)?;
            info!(count, "WAL recovery replayed entries");
        }

        Ok(count)
    }

    /// Gracefully shut down: flush dirty pages, sync files.
    pub fn close(&mut self) -> MenteResult<()> {
        self.buffer_pool.flush_all(&mut self.page_manager)?;
        self.page_manager.sync()?;
        self.wal.sync()?;
        info!("storage engine closed");
        Ok(())
    }

    // ---- low-level page operations ----

    /// Allocate a fresh page.
    pub fn allocate_page(&mut self) -> MenteResult<PageId> {
        self.page_manager.allocate_page()
    }

    /// Read a page through the buffer pool.
    pub fn read_page(&mut self, page_id: PageId) -> MenteResult<Box<Page>> {
        self.buffer_pool.fetch_page(page_id, &mut self.page_manager)
    }

    /// Write data into a page with WAL protection.
    pub fn write_page(&mut self, page_id: PageId, data: &[u8]) -> MenteResult<()> {
        // WAL-first: log before modifying the page.
        let lsn = self.wal.append(WalEntryType::PageWrite, page_id.0, data)?;

        // Load the page into the buffer pool (or get cached copy).
        let mut page = self
            .buffer_pool
            .fetch_page(page_id, &mut self.page_manager)?;

        let copy_len = data.len().min(PAGE_DATA_SIZE);
        page.data[..copy_len].copy_from_slice(&data[..copy_len]);
        // Zero out remaining space if data is shorter than existing content.
        if copy_len < PAGE_DATA_SIZE {
            page.data[copy_len..].fill(0);
        }
        page.header.lsn = lsn;
        page.header.page_type = PageType::Data as u8;
        page.header.free_space = (PAGE_DATA_SIZE - copy_len) as u16;
        page.header.checksum = page.compute_checksum();

        // Push modified page back into the buffer pool.
        if self.buffer_pool.update_page(page_id, &page).is_err() {
            // Not cached (shouldn't happen after fetch, but be safe).
            self.page_manager.write_page(page_id, &page)?;
        }
        self.buffer_pool.unpin_page(page_id, true).ok();

        Ok(())
    }

    // ---- high-level memory operations ----

    /// Serialize and store a [`MemoryNode`] into a single page.
    ///
    /// Returns the [`PageId`] where the node was stored.
    pub fn store_memory(&mut self, node: &MemoryNode) -> MenteResult<PageId> {
        let serialized =
            serde_json::to_vec(node).map_err(|e| MenteError::Serialization(e.to_string()))?;

        // 4 bytes for the length prefix.
        if serialized.len() + 4 > PAGE_DATA_SIZE {
            return Err(MenteError::CapacityExceeded(format!(
                "memory node serialized to {} bytes (max {})",
                serialized.len(),
                PAGE_DATA_SIZE - 4,
            )));
        }

        let page_id = self.allocate_page()?;

        // Layout: [length: u32 LE][JSON bytes]
        let mut buf = Vec::with_capacity(4 + serialized.len());
        buf.extend_from_slice(&(serialized.len() as u32).to_le_bytes());
        buf.extend_from_slice(&serialized);

        self.write_page(page_id, &buf)?;

        info!(
            page_id = page_id.0,
            bytes = serialized.len(),
            "stored memory node"
        );
        Ok(page_id)
    }

    /// Load and deserialize a [`MemoryNode`] from the given page.
    pub fn load_memory(&mut self, page_id: PageId) -> MenteResult<MemoryNode> {
        let page = self.read_page(page_id)?;
        // Unpin immediately — we copy the data we need.
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
    pub fn checkpoint(&mut self) -> MenteResult<()> {
        self.buffer_pool.flush_all(&mut self.page_manager)?;
        self.page_manager.sync()?;

        let lsn = self.wal.append(WalEntryType::Checkpoint, 0, &[])?;
        self.wal.sync()?;
        self.wal.truncate(lsn)?;

        info!(lsn, "checkpoint complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::memory::MemoryType;
    
    fn setup() -> (tempfile::TempDir, StorageEngine) {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(dir.path()).unwrap();
        (dir, engine)
    }

    #[test]
    fn test_allocate_write_read() {
        let (_dir, mut engine) = setup();

        let pid = engine.allocate_page().unwrap();
        engine.write_page(pid, b"hello storage engine").unwrap();

        let page = engine.read_page(pid).unwrap();
        assert_eq!(&page.data[..20], b"hello storage engine");
        engine.buffer_pool.unpin_page(pid, false).ok();
    }

    #[test]
    fn test_store_and_load_memory() {
        let (_dir, mut engine) = setup();

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
        let (_dir, mut engine) = setup();

        let node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            "checkpoint test".to_string(),
            vec![1.0, 2.0],
        );

        let pid = engine.store_memory(&node).unwrap();
        engine.checkpoint().unwrap();

        // Data should still be readable after checkpoint.
        let loaded = engine.load_memory(pid).unwrap();
        assert_eq!(loaded.content, "checkpoint test");
    }

    #[test]
    fn test_close_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let pid;
        {
            let mut engine = StorageEngine::open(dir.path()).unwrap();
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
            let mut engine = StorageEngine::open(dir.path()).unwrap();
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
            let mut engine = StorageEngine::open(dir.path()).unwrap();
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
            engine.wal.sync().unwrap();
            // Drop without close — dirty pages may not be flushed.
        }
        {
            // Reopen — WAL replay should recover the writes.
            let mut engine = StorageEngine::open(dir.path()).unwrap();
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
            let mut engine = StorageEngine::open(dir.path()).unwrap();
            let node = MemoryNode::new(
                AgentId::new(),
                MemoryType::Semantic,
                content.clone(),
                vec![1.0, 2.0],
            );
            pid = engine.store_memory(&node).unwrap();
            // Proper shutdown — checkpoint flushes pages and truncates WAL.
            engine.checkpoint().unwrap();
            engine.close().unwrap();
        }
        {
            // Reopen after clean shutdown — WAL should be empty, no duplicate data.
            let mut engine = StorageEngine::open(dir.path()).unwrap();
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
            let mut engine = StorageEngine::open(dir.path()).unwrap();
            // Store 3 memories then checkpoint.
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

            // Store 2 more without checkpoint (will only be in WAL).
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
            engine.wal.sync().unwrap();
        }
        {
            let mut engine = StorageEngine::open(dir.path()).unwrap();
            // All 5 memories should be present: 3 from checkpoint, 2 from WAL replay.
            for (pid, expected) in ids.iter().zip(contents.iter()) {
                let loaded = engine.load_memory(*pid).unwrap();
                assert_eq!(&loaded.content, expected);
            }
        }
    }
}
