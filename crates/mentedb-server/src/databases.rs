//! Named databases: one MenteDB per tenant (a person, a team, an
//! organization), each in its own directory with its own lock, indexes and
//! files, so nothing is shared between them but the process.
//!
//! The legacy default database lives at the data directory root and backs the
//! unprefixed `/v1/...` routes. Named databases live under `<data>/dbs/<name>`
//! and back `/v1/db/{name}/...`. They open on first use, and the least recently
//! used idle handle is flushed and closed once more than `max_open` are open.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use mentedb::{CognitiveConfig, MenteDb};
use mentedb_core::error::MenteResult;
use mentedb_embedding::provider::EmbeddingProvider;

use crate::error::ApiError;

/// Name of the database at the data directory root.
pub const DEFAULT_DB: &str = "default";

/// Most characters a database name may have.
const MAX_NAME_LEN: usize = 128;

/// Limits for named databases.
#[derive(Clone, Debug)]
pub struct DatabasesConfig {
    /// Named databases kept open at once; idle ones beyond this are closed.
    pub max_open: usize,
    /// Create a named database on first use instead of returning 404.
    pub auto_create: bool,
}

impl Default for DatabasesConfig {
    fn default() -> Self {
        Self {
            max_open: 256,
            auto_create: true,
        }
    }
}

/// One embedder shared by every open database.
struct SharedEmbedder(Arc<dyn EmbeddingProvider>);

impl EmbeddingProvider for SharedEmbedder {
    fn embed(&self, text: &str) -> MenteResult<Vec<f32>> {
        self.0.embed(text)
    }
    fn embed_batch(&self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        self.0.embed_batch(texts)
    }
    fn dimensions(&self) -> usize {
        self.0.dimensions()
    }
    fn model_name(&self) -> &str {
        self.0.model_name()
    }
}

struct OpenDb {
    db: Arc<MenteDb>,
    last_used: Instant,
}

/// Registry of the default database and the named ones.
pub struct Databases {
    root: PathBuf,
    default: Arc<MenteDb>,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    cognitive: CognitiveConfig,
    config: DatabasesConfig,
    open: Mutex<HashMap<String, OpenDb>>,
}

impl Databases {
    /// Open the default database at `root` and prepare the registry.
    pub fn open(
        root: &Path,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
        cognitive: CognitiveConfig,
        config: DatabasesConfig,
    ) -> MenteResult<Self> {
        std::fs::create_dir_all(root)?;
        let default = Arc::new(open_at(root, embedder.clone(), cognitive.clone())?);
        Ok(Self {
            root: root.to_path_buf(),
            default,
            embedder,
            cognitive,
            config,
            open: Mutex::new(HashMap::new()),
        })
    }

    /// The database at the data directory root.
    pub fn default_db(&self) -> Arc<MenteDb> {
        self.default.clone()
    }

    /// The data directory root (the default database's directory).
    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    /// The shared embedder, when one is configured.
    pub fn embedder(&self) -> Option<Arc<dyn EmbeddingProvider>> {
        self.embedder.clone()
    }

    fn dir(&self, name: &str) -> PathBuf {
        self.root.join("dbs").join(name)
    }

    /// Whether a named database exists on disk.
    pub fn exists(&self, name: &str) -> bool {
        name == DEFAULT_DB || self.dir(name).join(".mentedb-db").exists()
    }

    /// Get a database by name, opening it if needed. Blocking: call from
    /// `spawn_blocking`. `create` allows making a new database.
    pub fn get(&self, name: &str, create: bool) -> Result<Arc<MenteDb>, ApiError> {
        if name == DEFAULT_DB {
            return Ok(self.default.clone());
        }
        validate_name(name)?;
        let mut open = self.open.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = open.get_mut(name) {
            entry.last_used = Instant::now();
            return Ok(entry.db.clone());
        }
        let dir = self.dir(name);
        let marker = dir.join(".mentedb-db");
        if !marker.exists() {
            if !(create || self.config.auto_create) {
                return Err(ApiError::NotFound(format!("database '{name}' not found")));
            }
            std::fs::create_dir_all(&dir)
                .map_err(|e| ApiError::Internal(format!("create database '{name}': {e}")))?;
            std::fs::write(&marker, name.as_bytes())
                .map_err(|e| ApiError::Internal(format!("create database '{name}': {e}")))?;
        }
        // Opening holds the registry lock so one name is never opened twice;
        // the engine's own file lock would refuse the second open anyway.
        let db = Arc::new(
            open_at(&dir, self.embedder.clone(), self.cognitive.clone())
                .map_err(|e| ApiError::Internal(format!("open database '{name}': {e}")))?,
        );
        open.insert(
            name.to_string(),
            OpenDb {
                db: db.clone(),
                last_used: Instant::now(),
            },
        );
        let evicted = take_idle_overflow(&mut open, self.config.max_open);
        drop(open);
        for (evicted_name, handle) in evicted {
            if let Err(e) = handle.close() {
                tracing::warn!(db = %evicted_name, error = %e, "closing idle database failed");
            }
        }
        Ok(db)
    }

    /// Named databases on disk, sorted.
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(self.root.join("dbs"))
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().join(".mentedb-db").exists())
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        names.sort();
        names
    }

    /// Close and permanently delete a named database. Blocking.
    pub fn delete(&self, name: &str) -> Result<(), ApiError> {
        if name == DEFAULT_DB {
            return Err(ApiError::BadRequest(
                "the default database cannot be deleted".into(),
            ));
        }
        validate_name(name)?;
        let mut open = self.open.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = open.get(name)
            && Arc::strong_count(&entry.db) > 1
        {
            return Err(ApiError::Conflict(format!(
                "database '{name}' is in use; retry"
            )));
        }
        if let Some(entry) = open.remove(name) {
            let _ = entry.db.close();
        }
        let dir = self.dir(name);
        if !dir.join(".mentedb-db").exists() {
            return Err(ApiError::NotFound(format!("database '{name}' not found")));
        }
        std::fs::remove_dir_all(&dir)
            .map_err(|e| ApiError::Internal(format!("delete database '{name}': {e}")))
    }

    /// Directory a database's files live in.
    pub fn path_of(&self, name: &str) -> Result<PathBuf, ApiError> {
        if name == DEFAULT_DB {
            return Ok(self.root.clone());
        }
        validate_name(name)?;
        Ok(self.dir(name))
    }

    /// Every currently open database, the default first.
    pub fn open_handles(&self) -> Vec<(String, Arc<MenteDb>)> {
        let mut all = vec![(DEFAULT_DB.to_string(), self.default.clone())];
        let open = self.open.lock().unwrap_or_else(|e| e.into_inner());
        all.extend(open.iter().map(|(n, e)| (n.clone(), e.db.clone())));
        all
    }

    /// Flush and close every open database. Used at shutdown.
    pub fn close_all(&self) {
        let handles: Vec<(String, Arc<MenteDb>)> = {
            let mut open = self.open.lock().unwrap_or_else(|e| e.into_inner());
            open.drain().map(|(n, e)| (n, e.db)).collect()
        };
        for (name, db) in handles {
            if let Err(e) = db.close() {
                tracing::warn!(db = %name, error = %e, "closing database failed");
            }
        }
        if let Err(e) = self.default.close() {
            tracing::warn!(error = %e, "closing default database failed");
        }
    }
}

fn open_at(
    dir: &Path,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    cognitive: CognitiveConfig,
) -> MenteResult<MenteDb> {
    match embedder {
        Some(e) => {
            MenteDb::open_with_embedder_and_config(dir, Box::new(SharedEmbedder(e)), cognitive)
        }
        None => MenteDb::open_with_config(dir, cognitive),
    }
}

/// Remove the least recently used handles nobody else holds until at most
/// `max_open` remain. Returned handles are closed by the caller off the lock.
fn take_idle_overflow(
    open: &mut HashMap<String, OpenDb>,
    max_open: usize,
) -> Vec<(String, Arc<MenteDb>)> {
    let mut evicted = Vec::new();
    while open.len() > max_open {
        let victim = open
            .iter()
            .filter(|(_, e)| Arc::strong_count(&e.db) == 1)
            .min_by_key(|(_, e)| e.last_used)
            .map(|(n, _)| n.clone());
        match victim {
            Some(name) => {
                if let Some(entry) = open.remove(&name) {
                    evicted.push((name, entry.db));
                }
            }
            // Every handle is in use; stay over the limit until one frees up.
            None => break,
        }
    }
    evicted
}

/// Database names are path segments: letters, digits, `-`, `_` and `.`, not
/// starting with `.`, at most 128 characters.
pub fn validate_name(name: &str) -> Result<(), ApiError> {
    let ok = !name.is_empty()
        && name.len() <= MAX_NAME_LEN
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if ok {
        Ok(())
    } else {
        Err(ApiError::BadRequest(format!(
            "invalid database name '{name}': use letters, digits, '-', '_' or '.', at most \
             {MAX_NAME_LEN} characters, not starting with '.'"
        )))
    }
}

/// Paths under a database that only read, so read-only tokens may POST them.
const READ_ONLY_POSTS: &[&str] = &["/recall", "/search", "/search/hybrid", "/context", "/query"];

/// The database a request targets: `{db}` in `/v1/db/{db}/...`, otherwise the
/// default. Checks the caller's token scope and opens the database.
pub struct Db {
    pub name: String,
    pub handle: Arc<MenteDb>,
}

impl std::ops::Deref for Db {
    type Target = MenteDb;
    fn deref(&self) -> &MenteDb {
        &self.handle
    }
}

impl axum::extract::FromRequestParts<Arc<crate::state::AppState>> for Db {
    type Rejection = ApiError;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &Arc<crate::state::AppState>,
    ) -> Result<Self, ApiError> {
        use axum::extract::RawPathParams;
        let params = RawPathParams::from_request_parts(parts, state)
            .await
            .map_err(|e| ApiError::BadRequest(e.to_string()))?;
        let name = params
            .iter()
            .find(|(k, _)| *k == "db")
            .map(|(_, v)| v.to_string())
            .unwrap_or_else(|| DEFAULT_DB.to_string());
        let path = parts.uri.path();
        let write = match parts.method {
            axum::http::Method::GET | axum::http::Method::HEAD => false,
            axum::http::Method::POST => !READ_ONLY_POSTS.iter().any(|p| path.ends_with(p)),
            _ => true,
        };
        if let Some(agent) = parts.extensions.get::<crate::auth::AuthenticatedAgent>() {
            agent.authorize_db(&name, write)?;
        }
        let dbs = state.databases.clone();
        let lookup = name.clone();
        let handle = tokio::task::spawn_blocking(move || dbs.get(&lookup, false))
            .await
            .map_err(|e| ApiError::Internal(format!("open database task failed: {e}")))??;
        Ok(Self { name, handle })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry(dir: &Path, max_open: usize) -> Databases {
        Databases::open(
            dir,
            None,
            CognitiveConfig::default(),
            DatabasesConfig {
                max_open,
                auto_create: true,
            },
        )
        .unwrap()
    }

    #[test]
    fn names_are_safe_path_segments() {
        for ok in ["alice", "team-7", "org_1", "u.123"] {
            assert!(validate_name(ok).is_ok(), "{ok}");
        }
        for bad in ["", "..", ".hidden", "a/b", "a\\b", "x y", &"a".repeat(129)] {
            assert!(validate_name(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn named_databases_are_isolated_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let dbs = registry(tmp.path(), 8);
        let a = dbs.get("alice", false).unwrap();
        let b = dbs.get("bob", false).unwrap();
        assert!(!Arc::ptr_eq(&a, &b));
        assert!(tmp.path().join("dbs/alice/.mentedb-db").exists());
        assert!(tmp.path().join("dbs/bob/.mentedb-db").exists());
        assert_eq!(dbs.list(), vec!["alice".to_string(), "bob".to_string()]);
        assert!(
            Arc::ptr_eq(&a, &dbs.get("alice", false).unwrap()),
            "cached handle"
        );
    }

    #[test]
    fn idle_databases_beyond_the_limit_are_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let dbs = registry(tmp.path(), 2);
        drop(dbs.get("a", false).unwrap());
        drop(dbs.get("b", false).unwrap());
        drop(dbs.get("c", false).unwrap());
        let open: Vec<String> = dbs.open_handles().into_iter().map(|(n, _)| n).collect();
        assert_eq!(open.len(), 3, "default plus two named: {open:?}");
        assert!(
            !open.contains(&"a".to_string()),
            "least recently used closed"
        );
        // A closed database reopens with its data intact.
        assert!(dbs.get("a", false).is_ok());
    }

    #[test]
    fn missing_database_is_404_without_auto_create() {
        let tmp = tempfile::tempdir().unwrap();
        let dbs = Databases::open(
            tmp.path(),
            None,
            CognitiveConfig::default(),
            DatabasesConfig {
                max_open: 4,
                auto_create: false,
            },
        )
        .unwrap();
        assert!(matches!(
            dbs.get("nobody", false),
            Err(ApiError::NotFound(_))
        ));
        assert!(dbs.get("nobody", true).is_ok(), "explicit create works");
    }

    #[test]
    fn delete_removes_the_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let dbs = registry(tmp.path(), 4);
        drop(dbs.get("gone", false).unwrap());
        dbs.delete("gone").unwrap();
        assert!(!tmp.path().join("dbs/gone").exists());
        assert!(dbs.delete(DEFAULT_DB).is_err());
    }
}
