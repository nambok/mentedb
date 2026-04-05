//! MenteDB Server: axum based REST API with WebSocket, JWT auth, and observability.

mod auth;
mod error;
mod grpc;
mod handlers;
mod rate_limit;
mod routes;
mod state;
mod websocket;

use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use axum::middleware;
use mentedb::MenteDb;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tonic::transport::Server as TonicServer;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::grpc::pb::cognition_service_server::CognitionServiceServer;
use crate::grpc::pb::memory_service_server::MemoryServiceServer;
use crate::grpc::{CognitionServiceImpl, MemoryServiceImpl};
use crate::rate_limit::RateLimiter;
use crate::state::AppState;

struct ServerConfig { data_dir: PathBuf, port: u16, grpc_port: u16, jwt_secret: Option<String>, admin_key: Option<String>, require_auth: bool }
fn parse_args() -> ServerConfig {
    let args: Vec<String> = env::args().collect();
    let mut data_dir = PathBuf::from("./mentedb-data");
    let mut port: u16 = 6677;
    let mut grpc_port: u16 = 6678;
    let mut jwt_secret: Option<String> = None;
    let mut admin_key: Option<String> = None;
    let mut require_auth = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" => {
                if i + 1 < args.len() {
                    data_dir = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    eprintln!("Error: --data-dir requires a value");
                    std::process::exit(1);
                }
            }
            "--port" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: --port must be a valid number");
                        std::process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("Error: --port requires a value");
                    std::process::exit(1);
                }
            }
            "--grpc-port" => {
                if i + 1 < args.len() {
                    grpc_port = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: --grpc-port must be a valid number");
                        std::process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("Error: --grpc-port requires a value");
                    std::process::exit(1);
                }
            }
            "--jwt-secret" => {
                if i + 1 < args.len() {
                    jwt_secret = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --jwt-secret requires a value");
                    std::process::exit(1);
                }
            }
            "--admin-key" => { if i+1<args.len() { admin_key = Some(args[i+1].clone()); i+=2; } else { eprintln!("Error: --admin-key requires a value"); std::process::exit(1); } }
            "--require-auth" => { require_auth = true; i += 1; }
            _ => { eprintln!("Unknown argument: {}", args[i]); std::process::exit(1); }
        }
    }
    if jwt_secret.is_none() && let Ok(v) = env::var("MENTEDB_JWT_SECRET") { jwt_secret = Some(v); }
    if admin_key.is_none() && let Ok(v) = env::var("MENTEDB_ADMIN_KEY") { admin_key = Some(v); }
    ServerConfig { data_dir, port, grpc_port, jwt_secret, admin_key, require_auth }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = parse_args();
    let (data_dir, port, grpc_port, jwt_secret) = (config.data_dir, config.port, config.grpc_port, config.jwt_secret.clone());
    let admin_key = config.admin_key.clone();
    if config.require_auth && jwt_secret.is_none() { eprintln!("Error: --require-auth was set but no --jwt-secret provided"); std::process::exit(1); }
    if jwt_secret.is_none() { eprintln!("WARNING: Running without authentication. All endpoints are open. Set --jwt-secret for production use."); }

    std::fs::create_dir_all(&data_dir)?;

    let db = MenteDb::open(&data_dir)?;
    let db = Arc::new(RwLock::new(db));

    let auth_mode = if jwt_secret.is_some() {
        "enabled"
    } else {
        "disabled (development mode)"
    };

    let state = Arc::new(AppState {
        db: db.clone(),
        spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
        jwt_secret: jwt_secret.clone(),
        admin_key: admin_key.clone(),
        start_time: Instant::now(),
    });

    let app = routes::build_router(state.clone())
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .layer(RateLimiter::default())
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    println!("MenteDB v0.1.0");
    println!("Data directory: {}", data_dir.display());
    println!("REST API on: 0.0.0.0:{port}");
    println!("gRPC API on: 0.0.0.0:{grpc_port}");
    println!("Auth: {auth_mode}");
    println!();
    println!("REST Endpoints:");
    println!("  GET    /v1/health           health check");
    println!("  POST   /v1/memories         store a memory");
    println!("  GET    /v1/memories/:id      get a memory by ID");
    println!("  DELETE /v1/memories/:id      forget a memory");
    println!("  POST   /v1/recall           query via MQL");
    println!("  POST   /v1/search           vector similarity search");
    println!("  POST   /v1/edges            create an edge");
    println!("  GET    /v1/stats            database statistics");
    println!("  POST   /v1/auth/token       generate JWT");
    println!("  GET    /v1/ws/stream        WebSocket cognition stream");
    println!();
    println!("gRPC Services:");
    println!("  CognitionService.StreamCognition  bidirectional streaming");
    println!("  MemoryService.Store               store a memory");
    println!("  MemoryService.Recall              query via MQL");
    println!("  MemoryService.Search              vector similarity search");
    println!("  MemoryService.Forget              forget a memory");
    println!("  MemoryService.Relate              create an edge");

    let grpc_addr = SocketAddr::from(([0, 0, 0, 0], grpc_port));
    let grpc_state = state.clone();

    let grpc_handle = tokio::spawn(async move {
        info!("gRPC server listening on {grpc_addr}");
        TonicServer::builder()
            .add_service(CognitionServiceServer::new(CognitionServiceImpl {
                state: grpc_state.clone(),
            }))
            .add_service(MemoryServiceServer::new(MemoryServiceImpl {
                state: grpc_state,
            }))
            .serve(grpc_addr)
            .await
            .expect("gRPC server failed");
    });

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(addr).await?;

    info!("MenteDB REST server listening on {addr}");

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    grpc_handle.abort();

    let mut db = db.write().await;
    db.close()?;
    info!("MenteDB server stopped");

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl+c");
    info!("received ctrl+c, shutting down");
}
