//! MenteDB Server: axum based REST API with WebSocket, JWT auth, and observability.

mod auth;
mod error;
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
use tracing::info;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::rate_limit::RateLimiter;
use crate::state::AppState;

fn parse_args() -> (PathBuf, u16, Option<String>) {
    let args: Vec<String> = env::args().collect();
    let mut data_dir = PathBuf::from("./mentedb-data");
    let mut port: u16 = 6677;
    let mut jwt_secret: Option<String> = None;

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
            "--jwt-secret" => {
                if i + 1 < args.len() {
                    jwt_secret = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --jwt-secret requires a value");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    (data_dir, port, jwt_secret)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let (data_dir, port, jwt_secret) = parse_args();

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
        jwt_secret: jwt_secret.clone(),
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
    println!("Listening on: 0.0.0.0:{port}");
    println!("Auth: {auth_mode}");
    println!();
    println!("Endpoints:");
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

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(addr).await?;

    info!("MenteDB server listening on {addr}");

    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(shutdown_signal())
        .await?;

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
