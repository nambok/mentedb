//! Shared application state for the MenteDB server.
use mentedb::MenteDb;
use mentedb_extraction::ExtractionConfig;
use std::sync::Arc;
use std::time::Instant;

use crate::extraction_queue::ExtractionSender;

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
    /// Bounded channel sender for the background extraction worker.
    pub extraction_tx: Option<ExtractionSender>,
    /// Self-organizing sharding handle. `None` unless `MENTEDB_SHARDING` is set.
    pub cluster: Option<crate::cluster::Cluster>,
    /// Per-agent delta trackers for `/v1/process_turn`, so delta-aware context
    /// serving (only what changed since the agent's last turn) works across
    /// requests instead of resetting every call.
    pub turn_trackers:
        Arc<tokio::sync::Mutex<std::collections::HashMap<String, mentedb::context::DeltaTracker>>>,
}
