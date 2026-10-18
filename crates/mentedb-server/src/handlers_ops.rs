//! Operations endpoints: maintenance, enrichment, documents, backups, database management, readiness.
use std::sync::Arc;

use axum::Router;

use crate::state::AppState;

/// Routes on one database, mounted under `/v1` and `/v1/db/{db}`.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
}

/// Routes outside any one database (database management, readiness).
pub fn global_routes() -> Router<Arc<AppState>> {
    Router::new()
}
