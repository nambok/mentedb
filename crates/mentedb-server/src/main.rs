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
use mentedb_extraction::{ExtractionConfig, LlmProvider};
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

struct ServerConfig {
    data_dir: PathBuf,
    port: u16,
    grpc_port: u16,
    jwt_secret: Option<String>,
    admin_key: Option<String>,
    require_auth: bool,
    auto_extract: bool,
    llm_provider: Option<String>,
    llm_api_key: Option<String>,
    llm_model: Option<String>,
    llm_base_url: Option<String>,
    extraction_quality_threshold: Option<f32>,
    extraction_dedup_threshold: Option<f32>,
}
fn parse_args() -> ServerConfig {
    let args: Vec<String> = env::args().collect();
    let mut data_dir = PathBuf::from("./mentedb-data");
    let mut port: u16 = 6677;
    let mut grpc_port: u16 = 6678;
    let mut jwt_secret: Option<String> = None;
    let mut admin_key: Option<String> = None;
    let mut require_auth = false;
    let mut auto_extract = false;
    let mut llm_provider: Option<String> = None;
    let mut llm_api_key: Option<String> = None;
    let mut llm_model: Option<String> = None;
    let mut llm_base_url: Option<String> = None;
    let mut extraction_quality_threshold: Option<f32> = None;
    let mut extraction_dedup_threshold: Option<f32> = None;

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
            "--admin-key" => {
                if i + 1 < args.len() {
                    admin_key = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --admin-key requires a value");
                    std::process::exit(1);
                }
            }
            "--require-auth" => {
                require_auth = true;
                i += 1;
            }
            "--auto-extract" => {
                auto_extract = true;
                i += 1;
            }
            "--llm-provider" => {
                if i + 1 < args.len() {
                    llm_provider = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --llm-provider requires a value");
                    std::process::exit(1);
                }
            }
            "--llm-api-key" => {
                if i + 1 < args.len() {
                    llm_api_key = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --llm-api-key requires a value");
                    std::process::exit(1);
                }
            }
            "--llm-model" => {
                if i + 1 < args.len() {
                    llm_model = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --llm-model requires a value");
                    std::process::exit(1);
                }
            }
            "--llm-base-url" => {
                if i + 1 < args.len() {
                    llm_base_url = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --llm-base-url requires a value");
                    std::process::exit(1);
                }
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }
    if jwt_secret.is_none()
        && let Ok(v) = env::var("MENTEDB_JWT_SECRET")
    {
        jwt_secret = Some(v);
    }
    if admin_key.is_none()
        && let Ok(v) = env::var("MENTEDB_ADMIN_KEY")
    {
        admin_key = Some(v);
    }
    if llm_provider.is_none()
        && let Ok(v) = env::var("MENTEDB_LLM_PROVIDER")
    {
        llm_provider = Some(v);
    }
    if llm_api_key.is_none()
        && let Ok(v) = env::var("MENTEDB_LLM_API_KEY")
    {
        llm_api_key = Some(v);
    }
    if llm_model.is_none()
        && let Ok(v) = env::var("MENTEDB_LLM_MODEL")
    {
        llm_model = Some(v);
    }
    if llm_base_url.is_none()
        && let Ok(v) = env::var("MENTEDB_LLM_BASE_URL")
    {
        llm_base_url = Some(v);
    }
    if !auto_extract && let Ok(v) = env::var("MENTEDB_AUTO_EXTRACT") {
        auto_extract = v == "true" || v == "1";
    }
    if extraction_quality_threshold.is_none()
        && let Ok(v) = env::var("MENTEDB_EXTRACTION_QUALITY_THRESHOLD")
    {
        extraction_quality_threshold = v.parse().ok();
    }
    if extraction_dedup_threshold.is_none()
        && let Ok(v) = env::var("MENTEDB_EXTRACTION_DEDUP_THRESHOLD")
    {
        extraction_dedup_threshold = v.parse().ok();
    }
    ServerConfig {
        data_dir,
        port,
        grpc_port,
        jwt_secret,
        admin_key,
        require_auth,
        auto_extract,
        llm_provider,
        llm_api_key,
        llm_model,
        llm_base_url,
        extraction_quality_threshold,
        extraction_dedup_threshold,
    }
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

    // Build extraction config first (before moving fields out of config)
    let extraction_config = build_extraction_config(&config);
    let auto_extract = config.auto_extract;

    let (data_dir, port, grpc_port, jwt_secret) = (
        config.data_dir,
        config.port,
        config.grpc_port,
        config.jwt_secret.clone(),
    );
    let admin_key = config.admin_key.clone();
    if config.require_auth && jwt_secret.is_none() {
        eprintln!("Error: --require-auth was set but no --jwt-secret provided");
        std::process::exit(1);
    }
    if jwt_secret.is_none() {
        eprintln!(
            "WARNING: Running without authentication. All endpoints are open. Set --jwt-secret for production use."
        );
    }

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
        extraction_config,
        auto_extract,
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
    println!("  POST   /v1/ingest           extract & store from conversation");
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

/// Build an ExtractionConfig from the ServerConfig's LLM settings.
/// Returns `None` if no provider is configured (i.e., "none").
fn build_extraction_config(config: &ServerConfig) -> Option<ExtractionConfig> {
    let provider_str = config.llm_provider.as_deref().unwrap_or("none");
    let provider = match provider_str.to_lowercase().as_str() {
        "openai" => LlmProvider::OpenAI,
        "anthropic" => LlmProvider::Anthropic,
        "ollama" => LlmProvider::Ollama,
        "none" | "" => return None,
        other => {
            eprintln!("WARNING: Unknown LLM provider '{other}', extraction disabled");
            return None;
        }
    };

    let api_url = config
        .llm_base_url
        .clone()
        .unwrap_or_else(|| provider.default_url().to_string());
    let model = config
        .llm_model
        .clone()
        .unwrap_or_else(|| provider.default_model().to_string());

    Some(ExtractionConfig {
        provider,
        api_key: config.llm_api_key.clone(),
        api_url,
        model,
        max_extractions_per_conversation: 50,
        quality_threshold: config.extraction_quality_threshold.unwrap_or(0.7),
        deduplication_threshold: config.extraction_dedup_threshold.unwrap_or(0.85),
        enable_contradiction_check: true,
        enable_deduplication: true,
    })
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl+c");
    info!("received ctrl+c, shutting down");
}
