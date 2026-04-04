//! Shared application state for the MenteDB server.

use std::sync::Arc;
use std::time::Instant;

use mentedb::MenteDb;
use tokio::sync::RwLock;

/// Application state shared across all request handlers.
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<RwLock<MenteDb>>,
    pub jwt_secret: Option<String>,
    pub start_time: Instant,
}
