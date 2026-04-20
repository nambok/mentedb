//! MenteDB Server: REST API, gRPC, WebSocket, and auth layer.
//!
//! This crate implements the network facing server for MenteDB:
//!
//! - [`auth`]: JWT token creation, validation, and middleware
//! - [`extraction_queue`]: Background extraction queue with bounded concurrency
//! - [`handlers`]: Axum request handlers for memory CRUD and search
//! - [`routes`]: Router construction with middleware stack
//! - [`state`]: Shared application state (database handle, config)
//! - [`websocket`]: Real time memory event streaming
//! - [`grpc`]: Protocol Buffers based memory and cognition services
//! - [`rate_limit`]: Token bucket rate limiter
//! - [`error`]: Unified API error type with HTTP status mapping

/// JWT authentication and authorization.
pub mod auth;
/// Unified API error type.
pub mod error;
/// Background extraction queue with bounded concurrency.
pub mod extraction_queue;
/// gRPC service implementations for memory and cognition.
pub mod grpc;
/// HTTP request handlers for the REST API.
pub mod handlers;
/// Token bucket rate limiting middleware.
pub mod rate_limit;
/// Axum router construction.
pub mod routes;
/// Shared application state.
pub mod state;
/// WebSocket handler for real time event streaming.
pub mod websocket;
