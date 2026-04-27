//! Page Manager: file-backed page storage with free list management.
//!
//! Pages are 16KB fixed-size blocks used as the fundamental I/O unit.
//! The page file layout:
//! - Page 0: file header (magic, version, page count, free list head)
//! - Page 1..N: data pages

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use mentedb_core::error::{MenteError, MenteResult};
use tracing::{debug, info, trace};

/// Page size: 64KB to accommodate large embedding vectors and long content.
pub const PAGE_SIZE: usize = 64 * 1024;

/// Magic number identifying a MenteDB page file ("MENTEDB1").
const MAGIC: u64 = 0x4D454E_5445444231;

/// File format version.
const VERSION: u32 = 1;

/// A page identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u64);

/// Page type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageType {
    Free = 0,
    Data = 1,
    Index = 2,
    Overflow = 3,
}

impl From<u8> for PageType {
    fn from(v: u8) -> Self {
        match v {
            1 => PageType::Data,
            2 => PageType::Index,
            3 => PageType::Overflow,
            _ => PageType::Free,
        }
    }
}

/// Fixed-layout page header stored at the start of every page.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PageHeader {
    /// Which page this is.
    pub page_id: u64,
    /// Log Sequence Number: links to the WAL.
    pub lsn: u64,
    /// CRC-32 checksum of the page content.
    pub checksum: u32,
    /// Remaining free bytes in the data portion.
    pub free_space: u16,
    /// Number of occupied slots.
    pub num_slots: u16,
    /// Page type tag.
    pub page_type: u8,
    /// Padding for stable layout.
    pub _padding: [u8; 7],
}

/// Size of the page header in bytes.
pub const HEADER_SIZE: usize = std::mem::size_of::<PageHeader>();

/// Usable data bytes per page (total page size minus header).
pub const PAGE_DATA_SIZE: usize = PAGE_SIZE - HEADER_SIZE;

/// A fixed-size, 4KB-aligned page.
#[repr(C, align(4096))]
pub struct Page {
    /// The page header containing metadata (page ID, type, checksum, etc.).
    pub header: PageHeader,
    /// Raw page data payload.
    pub data: [u8; PAGE_DATA_SIZE],
}

impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("header", &self.header)
            .field("data_len", &self.data.len())
            .finish()
    }
}

impl Clone for Page {
    fn clone(&self) -> Self {
        let mut new_page = Page::zeroed();
        new_page.header = self.header;
        new_page.data.copy_from_slice(&self.data);
        new_page
    }
}

impl Page {
    /// Create a zero-initialized page.
    pub fn zeroed() -> Self {
        // Safety: Page is #[repr(C)] with only primitive types; all-zeros is valid.
        unsafe { std::mem::zeroed() }
    }

    /// View the raw bytes of this page.
    fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        // Safety: Page is #[repr(C)] with size == PAGE_SIZE.
        unsafe { &*(self as *const Page as *const [u8; PAGE_SIZE]) }
    }

    /// Reconstruct a Page from a raw byte buffer.
    fn from_bytes(bytes: &[u8; PAGE_SIZE]) -> Self {
        // Safety: Page is #[repr(C)]; any bit pattern in data is valid.
        unsafe { std::ptr::read(bytes.as_ptr() as *const Page) }
    }

    /// Compute CRC-32 checksum over header fields (excluding `checksum`) and data.
    pub fn compute_checksum(&self) -> u32 {
        let mut h = crc32fast::Hasher::new();
        h.update(&self.header.page_id.to_le_bytes());
        h.update(&self.header.lsn.to_le_bytes());
        h.update(&self.header.free_space.to_le_bytes());
        h.update(&self.header.num_slots.to_le_bytes());
        h.update(&[self.header.page_type]);
        h.update(&self.data);
        h.finalize()
    }
}

/// On-disk file header occupying the first bytes of page 0.
#[repr(C)]
struct FileHeader {
    magic: u64,
    version: u32,
    _pad: u32,
    page_count: u64,
    free_list_head: u64,
}

/// Manages a file-backed page store with free-list allocation.
pub struct PageManager {
    file: File,
    page_count: u64,
    free_list_head: u64,
}

impl PageManager {
    /// Open (or create) a page file at `dir_path/pages.db`.
    pub fn open(dir_path: &Path) -> MenteResult<Self> {
        let file_path = dir_path.join("pages.db");
        let exists = file_path.exists()
            && std::fs::metadata(&file_path)
                .map(|m| m.len() > 0)
                .unwrap_or(false);

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&file_path)?;

        if exists {
            let mut buf = [0u8; std::mem::size_of::<FileHeader>()];
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut buf)?;
            let hdr: FileHeader = unsafe { std::ptr::read(buf.as_ptr() as *const FileHeader) };

            if hdr.magic != MAGIC {
                return Err(MenteError::Storage("invalid page file magic number".into()));
            }
            if hdr.version != VERSION {
                return Err(MenteError::Storage(format!(
                    "unsupported page file version: {}",
                    hdr.version
                )));
            }

            info!(page_count = hdr.page_count, "opened existing page file");
            Ok(Self {
                file,
                page_count: hdr.page_count,
                free_list_head: hdr.free_list_head,
            })
        } else {
            let mut pm = Self {
                file,
                page_count: 1,
                free_list_head: 0,
            };
            // Write header page (page 0) — reserves first PAGE_SIZE bytes.
            let mut header_page = Page::zeroed();
            header_page.header.page_id = 0;
            pm.write_page_raw(PageId(0), &header_page)?;
            pm.write_file_header()?;
            info!("created new page file");
            Ok(pm)
        }
    }

    /// Re-read the file header from disk to pick up changes made by another process.
    pub fn reload_header(&mut self) -> MenteResult<()> {
        let mut buf = [0u8; std::mem::size_of::<FileHeader>()];
        self.file.seek(SeekFrom::Start(0))?;
        self.file.read_exact(&mut buf)?;
        let hdr: FileHeader = unsafe { std::ptr::read(buf.as_ptr() as *const FileHeader) };
        if hdr.magic != MAGIC {
            return Err(MenteError::Storage(
                "invalid page file magic on reload".into(),
            ));
        }
        self.page_count = hdr.page_count;
        self.free_list_head = hdr.free_list_head;
        debug!(page_count = self.page_count, "reloaded page file header");
        Ok(())
    }

    /// Persist the file header into the beginning of page 0.
    fn write_file_header(&mut self) -> MenteResult<()> {
        let hdr = FileHeader {
            magic: MAGIC,
            version: VERSION,
            _pad: 0,
            page_count: self.page_count,
            free_list_head: self.free_list_head,
        };
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &hdr as *const FileHeader as *const u8,
                std::mem::size_of::<FileHeader>(),
            )
        };
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(bytes)?;
        self.file.flush()?;
        Ok(())
    }

    /// Allocate a new page, reusing from the free list when possible.
    pub fn allocate_page(&mut self) -> MenteResult<PageId> {
        if self.free_list_head != 0 {
            let page_id = PageId(self.free_list_head);
            let page = self.read_page(page_id)?;
            let next_free = u64::from_le_bytes(page.data[..8].try_into().unwrap());
            self.free_list_head = next_free;
            self.write_file_header()?;
            debug!(page_id = page_id.0, "allocated page from free list");
            return Ok(page_id);
        }

        let page_id = PageId(self.page_count);
        self.page_count += 1;

        let mut page = Page::zeroed();
        page.header.page_id = page_id.0;
        page.header.page_type = PageType::Data as u8;
        page.header.free_space = PAGE_DATA_SIZE as u16;
        self.write_page_raw(page_id, &page)?;
        self.write_file_header()?;

        debug!(page_id = page_id.0, "allocated new page");
        Ok(page_id)
    }

    /// Read a page from disk.
    pub fn read_page(&mut self, page_id: PageId) -> MenteResult<Box<Page>> {
        if page_id.0 >= self.page_count {
            return Err(MenteError::Storage(format!(
                "page {} out of range (count={})",
                page_id.0, self.page_count
            )));
        }

        let offset = page_id.0 * PAGE_SIZE as u64;
        let mut buf = [0u8; PAGE_SIZE];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut buf)?;

        let page = Page::from_bytes(&buf);
        if page.header.checksum != 0 {
            let expected = page.compute_checksum();
            if page.header.checksum != expected {
                return Err(MenteError::Storage(format!(
                    "page {} checksum mismatch (stored={:#x}, computed={:#x})",
                    page_id.0, page.header.checksum, expected
                )));
            }
        }

        trace!(page_id = page_id.0, "read page from disk");
        Ok(Box::new(page))
    }

    /// Write a page to disk.
    pub fn write_page(&mut self, page_id: PageId, page: &Page) -> MenteResult<()> {
        self.write_page_raw(page_id, page)
    }

    fn write_page_raw(&mut self, page_id: PageId, page: &Page) -> MenteResult<()> {
        let offset = page_id.0 * PAGE_SIZE as u64;
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(page.as_bytes())?;
        trace!(page_id = page_id.0, "wrote page to disk");
        Ok(())
    }

    /// Return a page to the free list.
    pub fn free_page(&mut self, page_id: PageId) -> MenteResult<()> {
        let mut page = Page::zeroed();
        page.header.page_id = page_id.0;
        page.header.page_type = PageType::Free as u8;
        // Store the current free list head as a forward pointer.
        page.data[..8].copy_from_slice(&self.free_list_head.to_le_bytes());

        self.write_page(page_id, &page)?;
        self.free_list_head = page_id.0;
        self.write_file_header()?;

        debug!(page_id = page_id.0, "freed page");
        Ok(())
    }

    /// Total number of pages (including the header page).
    pub fn page_count(&self) -> u64 {
        self.page_count
    }

    /// Sync page file data to disk.
    pub fn sync(&mut self) -> MenteResult<()> {
        self.file.sync_data()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (tempfile::TempDir, PageManager) {
        let dir = tempfile::tempdir().unwrap();
        let pm = PageManager::open(dir.path()).unwrap();
        (dir, pm)
    }

    #[test]
    fn test_allocate_and_read_write() {
        let (_dir, mut pm) = setup();

        let pid = pm.allocate_page().unwrap();
        assert_eq!(pid.0, 1); // page 0 is file header

        let mut page = Page::zeroed();
        page.header.page_id = pid.0;
        page.header.page_type = PageType::Data as u8;
        page.data[0..5].copy_from_slice(b"hello");
        pm.write_page(pid, &page).unwrap();

        let loaded = pm.read_page(pid).unwrap();
        assert_eq!(&loaded.data[0..5], b"hello");
    }

    #[test]
    fn test_free_and_reuse() {
        let (_dir, mut pm) = setup();

        let p1 = pm.allocate_page().unwrap();
        let p2 = pm.allocate_page().unwrap();
        assert_eq!(p1.0, 1);
        assert_eq!(p2.0, 2);

        // Free p1, then allocate — should get p1 back.
        pm.free_page(p1).unwrap();
        let p3 = pm.allocate_page().unwrap();
        assert_eq!(p3.0, p1.0);

        // Next allocation should be brand new.
        let p4 = pm.allocate_page().unwrap();
        assert_eq!(p4.0, 3);
    }

    #[test]
    fn test_multiple_free_reuse() {
        let (_dir, mut pm) = setup();

        let p1 = pm.allocate_page().unwrap();
        let p2 = pm.allocate_page().unwrap();
        let _p3 = pm.allocate_page().unwrap();

        // Free in order p1, p2 — free list becomes p2 -> p1.
        pm.free_page(p1).unwrap();
        pm.free_page(p2).unwrap();

        // Allocations should come back in LIFO order: p2, p1.
        let a1 = pm.allocate_page().unwrap();
        let a2 = pm.allocate_page().unwrap();
        assert_eq!(a1.0, p2.0);
        assert_eq!(a2.0, p1.0);

        // p3 is still allocated; next new page is 4.
        let a3 = pm.allocate_page().unwrap();
        assert_eq!(a3.0, 4);
    }

    #[test]
    fn test_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let pid;
        {
            let mut pm = PageManager::open(dir.path()).unwrap();
            pid = pm.allocate_page().unwrap();
            let mut page = Page::zeroed();
            page.header.page_id = pid.0;
            page.data[0..4].copy_from_slice(b"test");
            pm.write_page(pid, &page).unwrap();
            pm.sync().unwrap();
        }
        {
            let mut pm = PageManager::open(dir.path()).unwrap();
            let page = pm.read_page(pid).unwrap();
            assert_eq!(&page.data[0..4], b"test");
        }
    }

    #[test]
    fn test_out_of_range() {
        let (_dir, mut pm) = setup();
        assert!(pm.read_page(PageId(999)).is_err());
    }

    #[test]
    fn test_checksum() {
        let mut page = Page::zeroed();
        page.header.page_id = 42;
        page.data[0] = 0xFF;
        let c1 = page.compute_checksum();
        page.data[0] = 0x00;
        let c2 = page.compute_checksum();
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_checksum_verified_on_read() {
        let dir = tempfile::tempdir().unwrap();
        let pid;
        {
            let mut pm = PageManager::open(dir.path()).unwrap();
            pid = pm.allocate_page().unwrap();
            let mut page = Page::zeroed();
            page.header.page_id = pid.0;
            page.header.page_type = PageType::Data as u8;
            page.data[0..5].copy_from_slice(b"valid");
            page.header.checksum = page.compute_checksum();
            pm.write_page(pid, &page).unwrap();
            pm.sync().unwrap();
        }
        {
            // Valid checksum — should read fine
            let mut pm = PageManager::open(dir.path()).unwrap();
            let page = pm.read_page(pid).unwrap();
            assert_eq!(&page.data[0..5], b"valid");
        }
        {
            // Corrupt the data on disk
            let data_path = dir.path().join("pages.db");
            let mut raw = std::fs::read(&data_path).unwrap();
            let offset = pid.0 as usize * PAGE_SIZE;
            // Flip a byte in the data section (after header)
            raw[offset + std::mem::size_of::<PageHeader>()] ^= 0xFF;
            std::fs::write(&data_path, &raw).unwrap();

            let mut pm = PageManager::open(dir.path()).unwrap();
            let result = pm.read_page(pid);
            assert!(result.is_err(), "corrupted page should fail checksum");
        }
    }
}
