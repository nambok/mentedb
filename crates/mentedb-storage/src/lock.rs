//! Cross-host single-writer lock for a database directory.
//!
//! The engine permits exactly one writer per directory. On a single host
//! `flock` (fs2) is enough, but when a directory lives on a network filesystem
//! (NFSv4) mounted by more than one host, `flock` is host-local: a second host
//! does not observe the lock, so two processes could open the same directory and
//! corrupt it.
//!
//! POSIX `fcntl` byte-range locks are enforced ACROSS hosts by the NFSv4 lock
//! manager. On Linux we use open-file-description locks (`F_OFD_SETLK`), which
//! keep the per-open-file-description semantics `flock` has (a second open of the
//! same directory is refused even within one process, which the engine and its
//! tests rely on) while also being enforced across hosts. The lock releases when
//! the fd is closed or the process dies, so a crashed writer frees the directory
//! once the NFS lock lease expires; a merely paused writer keeps it, so there is
//! no zombie-write window.
//!
//! Non-Linux platforms (macOS, Windows) keep `flock`: single-host use, and it
//! gives the same per-open-file-description exclusion.

use std::fs::File;
use std::io;

/// Try to take an exclusive whole-file lock without blocking.
///
/// `Ok(true)` = acquired, `Ok(false)` = another holder has it, `Err` = a real
/// error trying to lock.
#[cfg(target_os = "linux")]
pub fn try_lock_exclusive(file: &File) -> io::Result<bool> {
    ofd_set(file, libc::F_WRLCK)
}

/// Release the lock held on this file descriptor.
#[cfg(target_os = "linux")]
pub fn unlock(file: &File) -> io::Result<()> {
    ofd_set(file, libc::F_UNLCK).map(|_| ())
}

#[cfg(target_os = "linux")]
fn ofd_set(file: &File, l_type: libc::c_int) -> io::Result<bool> {
    use std::os::unix::io::AsRawFd;
    // l_len == 0 means "to the end of the file", i.e. the whole file.
    let fl = libc::flock {
        l_type: l_type as libc::c_short,
        l_whence: libc::SEEK_SET as libc::c_short,
        l_start: 0,
        l_len: 0,
        l_pid: 0,
    };
    // F_OFD_SETLK never blocks; it returns EAGAIN/EACCES when the range is
    // already locked by a conflicting open file description (possibly on another
    // host, via the NFSv4 lock manager).
    let ret = unsafe {
        libc::fcntl(
            file.as_raw_fd(),
            libc::F_OFD_SETLK,
            &fl as *const libc::flock,
        )
    };
    if ret == 0 {
        return Ok(true);
    }
    let err = io::Error::last_os_error();
    match err.raw_os_error() {
        Some(libc::EACCES) | Some(libc::EAGAIN) => Ok(false),
        _ => Err(err),
    }
}

#[cfg(not(target_os = "linux"))]
pub fn try_lock_exclusive(file: &File) -> io::Result<bool> {
    use fs2::FileExt;
    match FileExt::try_lock_exclusive(file) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(false),
        Err(e) => Err(e),
    }
}

#[cfg(not(target_os = "linux"))]
pub fn unlock(file: &File) -> io::Result<()> {
    use fs2::FileExt;
    FileExt::unlock(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclusive_lock_excludes_a_second_open_of_the_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("LOCK");
        let a = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .unwrap();
        assert!(try_lock_exclusive(&a).unwrap(), "first lock acquires");

        // A second open (separate open file description) must be refused while
        // the first is held, even in this one process.
        let b = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        assert!(
            !try_lock_exclusive(&b).unwrap(),
            "second open must not acquire while the first holds the lock"
        );

        // After releasing the first, the second can acquire.
        unlock(&a).unwrap();
        assert!(
            try_lock_exclusive(&b).unwrap(),
            "lock is free after the holder releases"
        );
    }
}
