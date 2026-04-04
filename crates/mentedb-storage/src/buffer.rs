//! Buffer Pool — in-memory page cache with CLOCK eviction.
//!
//! Pages are loaded from disk into fixed-capacity frame slots. Pin counts
//! prevent eviction of pages currently in use. The CLOCK algorithm sweeps
//! frames looking for an unpinned, unreferenced victim when the pool is full.

use ahash::AHashMap;
use parking_lot::Mutex;

use crate::page::{Page, PageId, PageManager};
use mentedb_core::error::{MenteError, MenteResult};
use tracing::{debug, trace};

type FrameId = usize;

/// A single frame in the buffer pool.
struct Frame {
    page: Box<Page>,
    page_id: Option<PageId>,
    pin_count: u32,
    dirty: bool,
    /// CLOCK reference bit — set on access, cleared by the sweep hand.
    reference: bool,
}

impl Frame {
    fn new() -> Self {
        Self {
            page: Box::new(Page::zeroed()),
            page_id: None,
            pin_count: 0,
            dirty: false,
            reference: false,
        }
    }
}

struct BufferPoolInner {
    frames: Vec<Frame>,
    page_table: AHashMap<PageId, FrameId>,
    clock_hand: usize,
    capacity: usize,
}

/// Thread-safe buffer pool with CLOCK eviction.
pub struct BufferPool {
    inner: Mutex<BufferPoolInner>,
}

impl BufferPool {
    /// Create a buffer pool with `capacity` frame slots.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "buffer pool capacity must be > 0");
        let frames = (0..capacity).map(|_| Frame::new()).collect();
        Self {
            inner: Mutex::new(BufferPoolInner {
                frames,
                page_table: AHashMap::with_capacity(capacity),
                clock_hand: 0,
                capacity,
            }),
        }
    }

    /// Fetch a page into the pool (loading from disk if necessary).
    ///
    /// The page is automatically pinned (pin_count incremented).
    /// Caller must call [`unpin_page`] when done.
    pub fn fetch_page(&self, page_id: PageId, pm: &mut PageManager) -> MenteResult<Box<Page>> {
        let mut inner = self.inner.lock();

        // Cache hit
        if let Some(&frame_id) = inner.page_table.get(&page_id) {
            let frame = &mut inner.frames[frame_id];
            frame.pin_count += 1;
            frame.reference = true;
            trace!(page_id = page_id.0, frame_id, "buffer pool hit");
            return Ok(frame.page.clone());
        }

        // Cache miss — find a victim frame.
        let frame_id = Self::find_victim(&mut inner)?;

        // Flush dirty victim if needed.
        if inner.frames[frame_id].dirty {
            if let Some(old_pid) = inner.frames[frame_id].page_id {
                pm.write_page(old_pid, &inner.frames[frame_id].page)?;
                debug!(page_id = old_pid.0, frame_id, "flushed dirty victim");
            }
        }

        // Remove old mapping.
        if let Some(old_pid) = inner.frames[frame_id].page_id {
            inner.page_table.remove(&old_pid);
        }

        // Load the requested page from disk.
        let page = pm.read_page(page_id)?;
        {
            let frame = &mut inner.frames[frame_id];
            *frame.page = *page;
            frame.page_id = Some(page_id);
            frame.pin_count = 1;
            frame.dirty = false;
            frame.reference = true;
        }

        inner.page_table.insert(page_id, frame_id);
        trace!(page_id = page_id.0, frame_id, "loaded page into buffer pool");

        Ok(inner.frames[frame_id].page.clone())
    }

    /// Increment the pin count of a page already in the pool.
    pub fn pin_page(&self, page_id: PageId) -> MenteResult<()> {
        let mut inner = self.inner.lock();
        match inner.page_table.get(&page_id) {
            Some(&fid) => {
                inner.frames[fid].pin_count += 1;
                Ok(())
            }
            None => Err(MenteError::Storage(format!("page {} not in buffer pool", page_id.0))),
        }
    }

    /// Decrement pin count and optionally mark the page dirty.
    pub fn unpin_page(&self, page_id: PageId, dirty: bool) -> MenteResult<()> {
        let mut inner = self.inner.lock();
        match inner.page_table.get(&page_id) {
            Some(&fid) => {
                let frame = &mut inner.frames[fid];
                if frame.pin_count > 0 {
                    frame.pin_count -= 1;
                }
                if dirty {
                    frame.dirty = true;
                }
                Ok(())
            }
            None => Err(MenteError::Storage(format!("page {} not in buffer pool", page_id.0))),
        }
    }

    /// Replace the cached copy of a page and mark it dirty.
    pub fn update_page(&self, page_id: PageId, page: &Page) -> MenteResult<()> {
        let mut inner = self.inner.lock();
        match inner.page_table.get(&page_id) {
            Some(&fid) => {
                let frame = &mut inner.frames[fid];
                *frame.page = page.clone();
                frame.dirty = true;
                Ok(())
            }
            None => Err(MenteError::Storage(format!("page {} not in buffer pool", page_id.0))),
        }
    }

    /// Flush a single dirty page to disk.
    pub fn flush_page(&self, page_id: PageId, pm: &mut PageManager) -> MenteResult<()> {
        let mut inner = self.inner.lock();
        match inner.page_table.get(&page_id) {
            Some(&fid) => {
                let frame = &mut inner.frames[fid];
                if frame.dirty {
                    pm.write_page(page_id, &frame.page)?;
                    frame.dirty = false;
                    debug!(page_id = page_id.0, "flushed page");
                }
                Ok(())
            }
            None => Err(MenteError::Storage(format!("page {} not in buffer pool", page_id.0))),
        }
    }

    /// Flush all dirty pages to disk.
    pub fn flush_all(&self, pm: &mut PageManager) -> MenteResult<()> {
        let mut inner = self.inner.lock();
        for frame in &mut inner.frames {
            if frame.dirty {
                if let Some(pid) = frame.page_id {
                    pm.write_page(pid, &frame.page)?;
                    frame.dirty = false;
                }
            }
        }
        debug!("flushed all dirty pages");
        Ok(())
    }

    /// CLOCK eviction: find an unpinned, unreferenced frame.
    fn find_victim(inner: &mut BufferPoolInner) -> MenteResult<FrameId> {
        let cap = inner.capacity;

        // Prefer an empty frame first.
        for i in 0..cap {
            if inner.frames[i].page_id.is_none() {
                return Ok(i);
            }
        }

        // CLOCK sweep — at most 2 full rotations.
        let max_sweeps = cap * 2;
        for _ in 0..max_sweeps {
            let idx = inner.clock_hand;
            inner.clock_hand = (inner.clock_hand + 1) % cap;

            let frame = &mut inner.frames[idx];
            if frame.pin_count == 0 {
                if !frame.reference {
                    return Ok(idx);
                }
                frame.reference = false;
            }
        }

        Err(MenteError::Storage("buffer pool full: all pages are pinned".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::Page;

    fn setup() -> (tempfile::TempDir, PageManager) {
        let dir = tempfile::tempdir().unwrap();
        let pm = PageManager::open(dir.path()).unwrap();
        (dir, pm)
    }

    #[test]
    fn test_fetch_and_cache_hit() {
        let (_dir, mut pm) = setup();
        let pool = BufferPool::new(4);

        let pid = pm.allocate_page().unwrap();
        let mut page = Page::zeroed();
        page.header.page_id = pid.0;
        page.data[0..3].copy_from_slice(b"abc");
        pm.write_page(pid, &page).unwrap();

        // First fetch — cache miss, loads from disk.
        let p1 = pool.fetch_page(pid, &mut pm).unwrap();
        assert_eq!(&p1.data[0..3], b"abc");

        // Unpin.
        pool.unpin_page(pid, false).unwrap();

        // Second fetch — cache hit.
        let p2 = pool.fetch_page(pid, &mut pm).unwrap();
        assert_eq!(&p2.data[0..3], b"abc");
        pool.unpin_page(pid, false).unwrap();
    }

    #[test]
    fn test_dirty_flush() {
        let (_dir, mut pm) = setup();
        let pool = BufferPool::new(4);

        let pid = pm.allocate_page().unwrap();

        let mut page = Page::zeroed();
        page.header.page_id = pid.0;
        page.data[0] = 42;
        pm.write_page(pid, &page).unwrap();

        // Fetch, modify, mark dirty.
        let _ = pool.fetch_page(pid, &mut pm).unwrap();
        let mut modified = Page::zeroed();
        modified.header.page_id = pid.0;
        modified.data[0] = 99;
        pool.update_page(pid, &modified).unwrap();
        pool.unpin_page(pid, true).unwrap();

        // Flush to disk.
        pool.flush_page(pid, &mut pm).unwrap();

        // Read directly from disk to verify.
        let on_disk = pm.read_page(pid).unwrap();
        assert_eq!(on_disk.data[0], 99);
    }

    #[test]
    fn test_eviction() {
        let (_dir, mut pm) = setup();
        let pool = BufferPool::new(2); // tiny pool

        // Allocate 3 pages.
        let p1 = pm.allocate_page().unwrap();
        let p2 = pm.allocate_page().unwrap();
        let p3 = pm.allocate_page().unwrap();

        for pid in [p1, p2, p3] {
            let mut page = Page::zeroed();
            page.header.page_id = pid.0;
            page.data[0] = pid.0 as u8;
            pm.write_page(pid, &page).unwrap();
        }

        // Fill pool with p1, p2.
        let _ = pool.fetch_page(p1, &mut pm).unwrap();
        pool.unpin_page(p1, false).unwrap();
        let _ = pool.fetch_page(p2, &mut pm).unwrap();
        pool.unpin_page(p2, false).unwrap();

        // Fetching p3 must evict one of the above.
        let page3 = pool.fetch_page(p3, &mut pm).unwrap();
        assert_eq!(page3.data[0], p3.0 as u8);
        pool.unpin_page(p3, false).unwrap();
    }

    #[test]
    fn test_all_pinned_error() {
        let (_dir, mut pm) = setup();
        let pool = BufferPool::new(2);

        let p1 = pm.allocate_page().unwrap();
        let p2 = pm.allocate_page().unwrap();
        let p3 = pm.allocate_page().unwrap();

        for pid in [p1, p2, p3] {
            let mut page = Page::zeroed();
            page.header.page_id = pid.0;
            pm.write_page(pid, &page).unwrap();
        }

        // Pin both frames (don't unpin).
        let _ = pool.fetch_page(p1, &mut pm).unwrap();
        let _ = pool.fetch_page(p2, &mut pm).unwrap();

        // p3 should fail — no victim available.
        assert!(pool.fetch_page(p3, &mut pm).is_err());
    }

    #[test]
    fn test_flush_all() {
        let (_dir, mut pm) = setup();
        let pool = BufferPool::new(4);

        let p1 = pm.allocate_page().unwrap();
        let p2 = pm.allocate_page().unwrap();

        for pid in [p1, p2] {
            let mut page = Page::zeroed();
            page.header.page_id = pid.0;
            pm.write_page(pid, &page).unwrap();
        }

        let _ = pool.fetch_page(p1, &mut pm).unwrap();
        let _ = pool.fetch_page(p2, &mut pm).unwrap();

        let mut mod1 = Page::zeroed();
        mod1.data[0] = 0xAA;
        pool.update_page(p1, &mod1).unwrap();

        let mut mod2 = Page::zeroed();
        mod2.data[0] = 0xBB;
        pool.update_page(p2, &mod2).unwrap();

        pool.unpin_page(p1, true).unwrap();
        pool.unpin_page(p2, true).unwrap();

        pool.flush_all(&mut pm).unwrap();

        let d1 = pm.read_page(p1).unwrap();
        let d2 = pm.read_page(p2).unwrap();
        assert_eq!(d1.data[0], 0xAA);
        assert_eq!(d2.data[0], 0xBB);
    }
}
