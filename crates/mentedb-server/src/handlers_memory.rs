//! Memory feature endpoints: hybrid search, heads-up context, history, graph, conflicts, pins, profile, entities, batch store.
use std::sync::Arc;

use axum::Router;

use crate::state::AppState;

/// Routes on one database, mounted under `/v1` and `/v1/db/{db}`.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
}
