//! Shared application state for the MenteDB server.

use std::sync::Arc;
use std::time::Instant;

use mentedb::MenteDb;
use tokio::sync::RwLock;

/// Application state shared across all request handlers.
#[derive(Clone)]
pub struct AppState {
    /// The shared database instance, protected by a read/write lock.
    pub db: Arc<RwLock<MenteDb>>,
    /// Optional JWT secret for authentication. When `None`, auth is disabled.
    pub jwt_secret: Option<String>,
    /// Server start time, used for uptime reporting.
    pub start_time: Instant,
}
