//! Shared application state for the MenteDB server.
use std::sync::Arc;
use std::time::Instant;
use mentedb::MenteDb;
use tokio::sync::RwLock;
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<RwLock<MenteDb>>,
        pub spaces: Arc<tokio::sync::RwLock<mentedb_core::SpaceManager>>,
    pub jwt_secret: Option<String>,
    pub admin_key: Option<String>,
    pub start_time: Instant,
}
