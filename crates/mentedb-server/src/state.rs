//! Shared application state for the MenteDB server.
use mentedb::MenteDb;
use mentedb_extraction::ExtractionConfig;
use std::sync::Arc;
use std::time::Instant;
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<MenteDb>,
    pub spaces: Arc<tokio::sync::RwLock<mentedb_core::SpaceManager>>,
    pub jwt_secret: Option<String>,
    pub admin_key: Option<String>,
    pub start_time: Instant,
    /// Extraction pipeline configuration. `None` means no LLM provider configured.
    pub extraction_config: Option<ExtractionConfig>,
    /// When true, storing a conversation-type memory auto-runs extraction.
    pub auto_extract: bool,
}
