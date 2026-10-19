//! Route definitions for the MenteDB REST API.

use std::sync::Arc;

use axum::Router;
use axum::routing::{delete, get, post};

use crate::handlers;
use crate::state::AppState;
use crate::websocket;

/// Routes that act on one database. Mounted twice: under `/v1` for the
/// default database and under `/v1/db/{db}` for a named one.
fn database_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/memories", post(handlers::store_memory))
        .route(
            "/memories/{id}",
            get(handlers::get_memory).delete(handlers::forget_memory),
        )
        .route("/recall", post(handlers::recall_memories))
        .route("/search", post(handlers::search_similar))
        .route("/edges", post(handlers::create_edge))
        .route("/stats", get(handlers::stats))
        .route("/ingest", post(handlers::ingest_conversation))
        .route("/process_turn", post(handlers::process_turn))
        .route("/admin/memories", get(handlers::admin_list_memories))
        .route(
            "/admin/memories/{id}",
            delete(handlers::admin_delete_memory),
        )
        .route("/admin/mql", post(handlers::admin_run_mql))
        .merge(crate::handlers_memory::routes())
        .merge(crate::handlers_ops::routes())
}

/// Build the complete axum router with all v1 API routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/health", get(handlers::health))
        .route("/v1/auth/token", post(crate::auth::generate_token))
        .route(
            "/v1/spaces",
            post(handlers::create_space).get(handlers::list_spaces),
        )
        .route("/v1/spaces/{id}/grant", post(handlers::grant_space_access))
        .route("/v1/ws/stream", get(websocket::ws_handler))
        .route(
            crate::cluster::GOSSIP_PATH,
            post(crate::cluster::gossip_handler),
        )
        .route("/metrics", get(crate::metrics::handler))
        .route("/console", get(crate::console::handler))
        .merge(crate::handlers_ops::global_routes())
        .nest("/v1", database_routes())
        .nest("/v1/db/{db}", database_routes())
        .with_state(state)
}
