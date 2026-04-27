//! Write-Ahead Log: append-only log for crash recovery.
//!
//! WAL entry format on disk:
//! ```text
//! [length: u32][lsn: u64][type: u8][page_id: u64][compressed_data: ...][crc32: u32]
//! ```
//!
//! - `length`: byte count of the payload (lsn + type + page_id + compressed_data).
//! - `compressed_data`: the data portion compressed with LZ4.
//! - `crc32`: checksum over the entire payload.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use mentedb_core::error::{MenteError, MenteResult};
use tracing::{debug, info, trace};

/// Log Sequence Number.
pub type Lsn = u64;

/// WAL entry type discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalEntryType {
    PageWrite = 1,
    /// Reserved for future transaction support. Not currently emitted.
    Commit = 2,
    Checkpoint = 3,
}

impl TryFrom<u8> for WalEntryType {
    type Error = MenteError;
    fn try_from(v: u8) -> MenteResult<Self> {
        match v {
            1 => Ok(Self::PageWrite),
            2 => Ok(Self::Commit),
            3 => Ok(Self::Checkpoint),
            _ => Err(MenteError::Storage(format!("invalid WAL entry type: {v}"))),
        }
    }
}

/// A single WAL entry (in-memory representation).
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Log sequence number.
    pub lsn: u64,
    /// The type of WAL operation.
    pub entry_type: WalEntryType,
    /// The page affected by this entry.
    pub page_id: u64,
    /// Serialized payload.
    pub data: Vec<u8>,
    /// CRC32 checksum for integrity verification.
    pub checksum: u32,
}

/// Append-only write-ahead log.
pub struct Wal {
    file: File,
    dir_path: std::path::PathBuf,
    next_lsn: u64,
}

/// Minimum payload size: lsn(8) + type(1) + page_id(8).
const MIN_PAYLOAD: usize = 17;

impl Wal {
    /// Open or create a WAL file at `dir_path/wal.log`.
    pub fn open(dir_path: &Path) -> MenteResult<Self> {
        let wal_path = dir_path.join("wal.log");
        let exists = wal_path.exists()
            && std::fs::metadata(&wal_path)
                .map(|m| m.len() > 0)
                .unwrap_or(false);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&wal_path)?;

        let mut wal = Self {
            file,
            dir_path: dir_path.to_path_buf(),
            next_lsn: 1,
        };

        if exists {
            let entries = wal.read_all_entries()?;
            if let Some(last) = entries.last() {
                wal.next_lsn = last.lsn + 1;
            }
            info!(
                next_lsn = wal.next_lsn,
                entries = entries.len(),
                "opened existing WAL"
            );
        } else {
            info!("created new WAL");
        }

        Ok(wal)
    }

    /// Acquire a blocking exclusive file lock on the WAL file.
    ///
    /// Uses `flock(2)` (via fs2) which works across processes on the same host
    /// and across NFS/EFS mounts. Blocks until the lock is available — callers
    /// should hold it only for the duration of append + fsync.
    pub fn lock_exclusive(&self) -> MenteResult<()> {
        use fs2::FileExt;
        self.file
            .lock_exclusive()
            .map_err(|e| MenteError::Storage(format!("WAL flock failed: {e}")))
    }

    /// Release the file lock on the WAL file.
    pub fn unlock(&self) -> MenteResult<()> {
        fs2::FileExt::unlock(&self.file)
            .map_err(|e| MenteError::Storage(format!("WAL unlock failed: {e}")))
    }

    /// Append an entry to the WAL and return its LSN.
    pub fn append(
        &mut self,
        entry_type: WalEntryType,
        page_id: u64,
        data: &[u8],
    ) -> MenteResult<Lsn> {
        let lsn = self.next_lsn;
        self.next_lsn += 1;

        let compressed = lz4_flex::compress_prepend_size(data);

        // Build the payload: lsn + type + page_id + compressed_data
        let payload_len = 8 + 1 + 8 + compressed.len();
        let mut payload = Vec::with_capacity(payload_len);
        payload.extend_from_slice(&lsn.to_le_bytes());
        payload.push(entry_type as u8);
        payload.extend_from_slice(&page_id.to_le_bytes());
        payload.extend_from_slice(&compressed);

        let crc = {
            let mut h = crc32fast::Hasher::new();
            h.update(&payload);
            h.finalize()
        };

        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&(payload_len as u32).to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.file.write_all(&crc.to_le_bytes())?;

        trace!(lsn, page_id, "appended WAL entry");
        Ok(lsn)
    }

    /// Flush the WAL to durable storage (fdatasync).
    pub fn sync(&mut self) -> MenteResult<()> {
        self.file.sync_data()?;
        debug!("WAL synced");
        Ok(())
    }

    /// Read all valid entries from the WAL for recovery.
    pub fn iterate(&mut self) -> MenteResult<Vec<WalEntry>> {
        self.read_all_entries()
    }

    /// Truncate all entries with LSN **less than** `before_lsn`.
    ///
    /// Uses atomic write-to-temp-then-rename to avoid data loss on crash.
    pub fn truncate(&mut self, before_lsn: Lsn) -> MenteResult<()> {
        let entries = self.read_all_entries()?;
        let to_keep: Vec<&WalEntry> = entries.iter().filter(|e| e.lsn >= before_lsn).collect();

        let wal_path = self.dir_path.join("wal.log");
        let tmp_path = self.dir_path.join("wal.log.tmp");

        {
            let mut tmp_file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;

            for entry in to_keep {
                let compressed = lz4_flex::compress_prepend_size(&entry.data);

                let payload_len = 8 + 1 + 8 + compressed.len();
                let mut payload = Vec::with_capacity(payload_len);
                payload.extend_from_slice(&entry.lsn.to_le_bytes());
                payload.push(entry.entry_type as u8);
                payload.extend_from_slice(&entry.page_id.to_le_bytes());
                payload.extend_from_slice(&compressed);

                let crc = {
                    let mut h = crc32fast::Hasher::new();
                    h.update(&payload);
                    h.finalize()
                };

                tmp_file.write_all(&(payload_len as u32).to_le_bytes())?;
                tmp_file.write_all(&payload)?;
                tmp_file.write_all(&crc.to_le_bytes())?;
            }

            tmp_file.sync_data()?;
        }

        std::fs::rename(&tmp_path, &wal_path)?;

        // Reopen the renamed file
        self.file = OpenOptions::new().read(true).write(true).open(&wal_path)?;

        debug!(before_lsn, "WAL truncated (atomic)");
        Ok(())
    }

    /// Current next LSN (useful for external callers).
    pub fn next_lsn(&self) -> Lsn {
        self.next_lsn
    }

    // ---- internal helpers ----

    fn read_all_entries(&mut self) -> MenteResult<Vec<WalEntry>> {
        self.file.seek(SeekFrom::Start(0))?;
        let file_len = self.file.metadata()?.len();
        let mut offset: u64 = 0;
        let mut entries = Vec::new();

        while offset + 4 <= file_len {
            // Read length
            let mut len_buf = [0u8; 4];
            if self.file.read_exact(&mut len_buf).is_err() {
                break;
            }
            let payload_len = u32::from_le_bytes(len_buf) as usize;
            offset += 4;

            if payload_len < MIN_PAYLOAD || offset + payload_len as u64 + 4 > file_len {
                break;
            }

            // Read payload
            let mut payload = vec![0u8; payload_len];
            if self.file.read_exact(&mut payload).is_err() {
                break;
            }
            offset += payload_len as u64;

            // Read CRC
            let mut crc_buf = [0u8; 4];
            if self.file.read_exact(&mut crc_buf).is_err() {
                break;
            }
            let stored_crc = u32::from_le_bytes(crc_buf);
            offset += 4;

            // Verify CRC
            let computed_crc = {
                let mut h = crc32fast::Hasher::new();
                h.update(&payload);
                h.finalize()
            };
            if computed_crc != stored_crc {
                break; // Corruption — stop here.
            }

            // Parse
            let lsn = u64::from_le_bytes(payload[0..8].try_into().unwrap());
            let entry_type = match WalEntryType::try_from(payload[8]) {
                Ok(t) => t,
                Err(_) => break,
            };
            let page_id = u64::from_le_bytes(payload[9..17].try_into().unwrap());
            let compressed_data = &payload[17..];

            let data = lz4_flex::decompress_size_prepended(compressed_data)
                .map_err(|e| MenteError::Storage(format!("LZ4 decompress failed: {e}")))?;

            entries.push(WalEntry {
                lsn,
                entry_type,
                page_id,
                data,
                checksum: stored_crc,
            });
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (tempfile::TempDir, Wal) {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::open(dir.path()).unwrap();
        (dir, wal)
    }

    #[test]
    fn test_append_and_iterate() {
        let (_dir, mut wal) = setup();

        let lsn1 = wal.append(WalEntryType::PageWrite, 1, b"hello").unwrap();
        let lsn2 = wal.append(WalEntryType::PageWrite, 2, b"world").unwrap();
        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);

        let entries = wal.iterate().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].lsn, 1);
        assert_eq!(entries[0].data, b"hello");
        assert_eq!(entries[1].lsn, 2);
        assert_eq!(entries[1].data, b"world");
    }

    #[test]
    fn test_sync() {
        let (_dir, mut wal) = setup();
        wal.append(WalEntryType::Commit, 0, b"").unwrap();
        wal.sync().unwrap(); // should not panic
    }

    #[test]
    fn test_truncate() {
        let (_dir, mut wal) = setup();

        wal.append(WalEntryType::PageWrite, 1, b"a").unwrap();
        wal.append(WalEntryType::PageWrite, 2, b"b").unwrap();
        wal.append(WalEntryType::Checkpoint, 0, b"").unwrap();

        // Truncate everything before LSN 3.
        wal.truncate(3).unwrap();

        let entries = wal.iterate().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].lsn, 3);
    }

    #[test]
    fn test_recovery_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut wal = Wal::open(dir.path()).unwrap();
            wal.append(WalEntryType::PageWrite, 10, b"recovery-data")
                .unwrap();
            wal.sync().unwrap();
        }
        {
            let mut wal = Wal::open(dir.path()).unwrap();
            assert_eq!(wal.next_lsn(), 2);
            let entries = wal.iterate().unwrap();
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].page_id, 10);
            assert_eq!(entries[0].data, b"recovery-data");
        }
    }

    #[test]
    fn test_empty_data_entry() {
        let (_dir, mut wal) = setup();
        let lsn = wal.append(WalEntryType::Checkpoint, 0, b"").unwrap();
        let entries = wal.iterate().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].lsn, lsn);
        assert!(entries[0].data.is_empty());
    }

    #[test]
    fn test_large_data_compression() {
        let (_dir, mut wal) = setup();
        let big_data = vec![0xABu8; 8192];
        wal.append(WalEntryType::PageWrite, 5, &big_data).unwrap();

        let entries = wal.iterate().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].data, big_data);
    }

    #[test]
    fn test_append_then_sync_is_durable() {
        // append() alone does not fsync — callers must call sync() for durability.
        // This matches the group-commit pattern: batch appends, sync once.
        let dir = tempfile::tempdir().unwrap();
        {
            let mut wal = Wal::open(dir.path()).unwrap();
            wal.append(WalEntryType::PageWrite, 1, b"batch1").unwrap();
            wal.append(WalEntryType::PageWrite, 2, b"batch2").unwrap();
            wal.sync().unwrap();
        }
        {
            let mut wal = Wal::open(dir.path()).unwrap();
            let entries = wal.iterate().unwrap();
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].data, b"batch1");
            assert_eq!(entries[1].data, b"batch2");
        }
    }

    #[test]
    fn test_truncate_atomic_preserves_kept_entries() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut wal = Wal::open(dir.path()).unwrap();
            wal.append(WalEntryType::PageWrite, 1, b"old1").unwrap();
            wal.append(WalEntryType::PageWrite, 2, b"old2").unwrap();
            wal.append(WalEntryType::PageWrite, 3, b"keep1").unwrap();
            wal.append(WalEntryType::PageWrite, 4, b"keep2").unwrap();

            wal.truncate(3).unwrap();

            let entries = wal.iterate().unwrap();
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].data, b"keep1");
            assert_eq!(entries[1].data, b"keep2");
        }
        // Verify survives reopen
        {
            let mut wal = Wal::open(dir.path()).unwrap();
            let entries = wal.iterate().unwrap();
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].lsn, 3);
            assert_eq!(entries[1].lsn, 4);
        }
    }

    #[test]
    fn test_truncate_no_temp_file_left_behind() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(dir.path()).unwrap();
        wal.append(WalEntryType::PageWrite, 1, b"a").unwrap();
        wal.truncate(2).unwrap();

        // Temp file should not exist after truncation
        assert!(!dir.path().join("wal.log.tmp").exists());
    }

    #[test]
    fn test_append_after_truncate_works() {
        let (_dir, mut wal) = setup();
        wal.append(WalEntryType::PageWrite, 1, b"before").unwrap();
        wal.truncate(2).unwrap();

        // Should be able to append after truncation (file handle is valid)
        wal.append(WalEntryType::PageWrite, 10, b"after").unwrap();
        let entries = wal.iterate().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].data, b"after");
    }
}
