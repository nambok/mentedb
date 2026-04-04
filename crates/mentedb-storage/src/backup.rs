//! Backup and restore for MenteDB storage.
//!
//! Uses a simple custom format: a JSON manifest header followed by
//! length-prefixed file entries. No external compression dependencies needed.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use mentedb_core::error::MenteResult;
use mentedb_core::types::Timestamp;
use serde::{Deserialize, Serialize};

/// Metadata about a backup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    /// When the backup was created (microseconds since epoch).
    pub created_at: Timestamp,
    /// Total size of all backed-up files in bytes.
    pub size_bytes: u64,
    /// Number of files included in the backup.
    pub memory_count: u64,
    /// Format version string.
    pub version: String,
}

/// Manages backup creation and restoration.
pub struct BackupManager;

const BACKUP_VERSION: &str = "mentedb-backup-v1";

impl BackupManager {
    /// Create a full backup of `data_dir` into a single file at `backup_path`.
    pub fn create_backup(data_dir: &Path, backup_path: &Path) -> MenteResult<BackupManifest> {
        let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        Self::collect_files(data_dir, data_dir, &mut files)?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as Timestamp;

        let total_bytes: u64 = files.values().map(|v| v.len() as u64).sum();

        let manifest = BackupManifest {
            created_at: now,
            size_bytes: total_bytes,
            memory_count: files.len() as u64,
            version: BACKUP_VERSION.to_string(),
        };

        let manifest_json = serde_json::to_vec(&manifest)
            .map_err(|e| mentedb_core::MenteError::Serialization(e.to_string()))?;

        let mut out = fs::File::create(backup_path)?;

        // Write manifest length + manifest
        out.write_all(&(manifest_json.len() as u32).to_le_bytes())?;
        out.write_all(&manifest_json)?;

        // Write each file entry
        for (name, data) in &files {
            let name_bytes = name.as_bytes();
            out.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            out.write_all(name_bytes)?;
            out.write_all(&(data.len() as u64).to_le_bytes())?;
            out.write_all(data)?;
        }

        out.flush()?;
        Ok(manifest)
    }

    /// Restore a backup file into `target_dir`.
    pub fn restore_backup(backup_path: &Path, target_dir: &Path) -> MenteResult<BackupManifest> {
        let mut file = fs::File::open(backup_path)?;

        // Read manifest
        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)?;
        let manifest_len = u32::from_le_bytes(len_buf) as usize;

        let mut manifest_buf = vec![0u8; manifest_len];
        file.read_exact(&mut manifest_buf)?;

        let manifest: BackupManifest = serde_json::from_slice(&manifest_buf)
            .map_err(|e| mentedb_core::MenteError::Serialization(e.to_string()))?;

        fs::create_dir_all(target_dir)?;

        // Read file entries
        for _ in 0..manifest.memory_count {
            let mut name_len_buf = [0u8; 4];
            file.read_exact(&mut name_len_buf)?;
            let name_len = u32::from_le_bytes(name_len_buf) as usize;

            let mut name_buf = vec![0u8; name_len];
            file.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)
                .map_err(|e| mentedb_core::MenteError::Serialization(e.to_string()))?;

            let mut data_len_buf = [0u8; 8];
            file.read_exact(&mut data_len_buf)?;
            let data_len = u64::from_le_bytes(data_len_buf) as usize;

            let mut data = vec![0u8; data_len];
            file.read_exact(&mut data)?;

            let dest = target_dir.join(&name);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&dest, &data)?;
        }

        Ok(manifest)
    }

    /// List backup manifests found in `backup_dir` (files matching `*.mentebackup`).
    pub fn list_backups(backup_dir: &Path) -> MenteResult<Vec<BackupManifest>> {
        let mut manifests = Vec::new();

        if !backup_dir.exists() {
            return Ok(manifests);
        }

        for entry in fs::read_dir(backup_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("mentebackup")
                && let Ok(m) = Self::read_manifest(&path)
            {
                manifests.push(m);
            }
        }

        manifests.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(manifests)
    }

    /// Read only the manifest header from a backup file.
    fn read_manifest(backup_path: &Path) -> MenteResult<BackupManifest> {
        let mut file = fs::File::open(backup_path)?;
        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)?;
        let manifest_len = u32::from_le_bytes(len_buf) as usize;
        let mut manifest_buf = vec![0u8; manifest_len];
        file.read_exact(&mut manifest_buf)?;
        serde_json::from_slice(&manifest_buf)
            .map_err(|e| mentedb_core::MenteError::Serialization(e.to_string()))
    }

    /// Recursively collect all files under `base` into the map.
    fn collect_files(
        base: &Path,
        dir: &Path,
        files: &mut BTreeMap<String, Vec<u8>>,
    ) -> MenteResult<()> {
        if !dir.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::collect_files(base, &path, files)?;
            } else {
                let rel = path
                    .strip_prefix(base)
                    .map_err(|e| mentedb_core::MenteError::Storage(e.to_string()))?;
                let data = fs::read(&path)?;
                files.insert(rel.to_string_lossy().into_owned(), data);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn backup_restore_roundtrip() {
        let data_dir = TempDir::new().unwrap();
        let backup_dir = TempDir::new().unwrap();

        // Create some test files
        fs::write(data_dir.path().join("file1.dat"), b"hello world").unwrap();
        fs::create_dir_all(data_dir.path().join("sub")).unwrap();
        fs::write(data_dir.path().join("sub/file2.dat"), b"nested data").unwrap();

        let backup_path = backup_dir.path().join("test.mentebackup");
        let manifest = BackupManager::create_backup(data_dir.path(), &backup_path).unwrap();
        assert_eq!(manifest.memory_count, 2);
        assert_eq!(manifest.version, BACKUP_VERSION);

        // Restore into same directory structure
        let restore_dir = TempDir::new().unwrap();
        let restored = BackupManager::restore_backup(&backup_path, restore_dir.path()).unwrap();
        assert_eq!(restored.memory_count, 2);

        assert_eq!(
            fs::read_to_string(restore_dir.path().join("file1.dat")).unwrap(),
            "hello world"
        );
        assert_eq!(
            fs::read_to_string(restore_dir.path().join("sub/file2.dat")).unwrap(),
            "nested data"
        );
    }

    #[test]
    fn restore_into_different_directory() {
        let data_dir = TempDir::new().unwrap();
        fs::write(data_dir.path().join("data.bin"), vec![0u8; 1024]).unwrap();

        let backup_dir = TempDir::new().unwrap();
        let backup_path = backup_dir.path().join("backup.mentebackup");
        BackupManager::create_backup(data_dir.path(), &backup_path).unwrap();

        // Restore into a completely different location
        let alt_dir = TempDir::new().unwrap();
        let alt_target = alt_dir.path().join("deep/nested/restore");
        let manifest = BackupManager::restore_backup(&backup_path, &alt_target).unwrap();

        assert_eq!(manifest.memory_count, 1);
        assert_eq!(manifest.size_bytes, 1024);
        assert_eq!(
            fs::read(alt_target.join("data.bin")).unwrap().len(),
            1024
        );
    }
}
