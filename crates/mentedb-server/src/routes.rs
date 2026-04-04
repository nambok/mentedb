//! Route definitions for the MenteDB REST API.

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};

use crate::handlers;
use crate::state::AppState;
use crate::websocket;

/// Build the complete axum router with all v1 API routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/health", get(handlers::health))
        .route("/v1/memories", post(handlers::store_memory))
        .route(
            "/v1/memories/{id}",
            get(handlers::get_memory).delete(handlers::forget_memory),
        )
        .route("/v1/recall", post(handlers::recall_memories))
        .route("/v1/search", post(handlers::search_similar))
        .route("/v1/edges", post(handlers::create_edge))
        .route("/v1/stats", get(handlers::stats))
        .route("/v1/auth/token", post(crate::auth::generate_token))
        .route("/v1/ws/stream", get(websocket::ws_handler))
        .with_state(state)
}
