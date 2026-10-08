//! Crash-safe snapshot writes.

use std::io::Write;
use std::path::Path;

use mentedb_core::error::MenteResult;

/// Write `data` to `path` so a reader sees either the old file or the complete
/// new one, never a truncated mix: write a sibling temp file, fsync it, then
/// rename it over `path`. A plain `fs::write` interrupted by a process kill
/// (a deploy stopping the task mid-flush) left an empty or partial index
/// snapshot that then failed every later open of the database.
pub(crate) fn write_atomic(path: &Path, data: &[u8]) -> MenteResult<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(data)?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_contents_and_leaves_no_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("snap.bin");
        write_atomic(&p, b"first").unwrap();
        write_atomic(&p, b"second").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"second");
        assert!(!dir.path().join("snap.bin.tmp").exists());
    }
}
