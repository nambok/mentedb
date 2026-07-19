//! MenteDB Server: axum based REST API with WebSocket, JWT auth, and observability.

mod auth;
mod cluster;
mod error;
mod extraction_queue;
mod grpc;
mod handlers;
mod maintenance;
mod metrics;
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
    /// AWS region for the Bedrock provider (MENTEDB_LLM_REGION / AWS_REGION).
    llm_region: Option<String>,
    extraction_quality_threshold: Option<f32>,
    extraction_dedup_threshold: Option<f32>,
    embedding_provider: Option<String>,
    maintenance_interval_hours: u64,
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
    let mut llm_region: Option<String> = None;
    let mut extraction_quality_threshold: Option<f32> = None;
    let mut extraction_dedup_threshold: Option<f32> = None;
    let mut embedding_provider: Option<String> = None;
    let mut maintenance_interval_hours: u64 = 24;

    let mut i = 1;
    // Skip a leading subcommand (e.g. `maintenance`) so flags still parse.
    if args.len() > 1 && !args[1].starts_with("--") {
        i = 2;
    }
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
            "--llm-region" => {
                if i + 1 < args.len() {
                    llm_region = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --llm-region requires a value");
                    std::process::exit(1);
                }
            }
            "--embedding-provider" => {
                if i + 1 < args.len() {
                    embedding_provider = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --embedding-provider requires a value (candle, hash, none)");
                    std::process::exit(1);
                }
            }
            "--maintenance-interval-hours" => {
                if i + 1 < args.len() {
                    maintenance_interval_hours = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Error: --maintenance-interval-hours must be a number");
                        std::process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("Error: --maintenance-interval-hours requires a value");
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
    // Bedrock region: MENTEDB_LLM_REGION takes precedence, then AWS_REGION.
    if llm_region.is_none()
        && let Ok(v) = env::var("MENTEDB_LLM_REGION").or_else(|_| env::var("AWS_REGION"))
    {
        llm_region = Some(v);
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
    if embedding_provider.is_none()
        && let Ok(v) = env::var("MENTEDB_EMBEDDING_PROVIDER")
    {
        embedding_provider = Some(v);
    }
    if let Ok(v) = env::var("MENTEDB_MAINTENANCE_INTERVAL_HOURS")
        && let Ok(n) = v.parse::<u64>()
    {
        maintenance_interval_hours = n;
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
        llm_region,
        extraction_quality_threshold,
        extraction_dedup_threshold,
        embedding_provider,
        maintenance_interval_hours,
    }
}

/// Build the embedding provider for the server.
///
/// Without an embedder the engine's semantic search, auto-linking, and
/// contradiction detection are inert (only keyword retrieval works), so the
/// server always wires one and is loud about quality degradation.
///
/// Selection: `candle` (semantic, requires the `local-embeddings` build
/// feature), `hash` (deterministic but non-semantic), `none` (disable).
/// Default: candle when compiled in, hash otherwise.
fn build_embedder(
    choice: Option<&str>,
) -> Option<Box<dyn mentedb_embedding::provider::EmbeddingProvider>> {
    let choice = choice.unwrap_or(if cfg!(feature = "local-embeddings") {
        "candle"
    } else {
        "hash"
    });

    match choice {
        "none" => {
            eprintln!(
                "WARNING: embeddings disabled (--embedding-provider none). Semantic search, \
                 auto-linking, and contradiction detection will not work; retrieval is \
                 keyword-only."
            );
            None
        }
        "candle" => {
            #[cfg(feature = "local-embeddings")]
            {
                match mentedb_embedding::CandleEmbeddingProvider::new() {
                    Ok(provider) => {
                        tracing::info!("using local Candle embeddings (all-MiniLM-L6-v2)");
                        return Some(Box::new(provider));
                    }
                    Err(e) => {
                        eprintln!(
                            "WARNING: failed to load Candle embedding model ({e}); falling back \
                             to non-semantic hash embeddings. Semantic search quality will be \
                             severely degraded."
                        );
                        return Some(Box::new(mentedb_embedding::HashEmbeddingProvider::new(384)));
                    }
                }
            }
            #[cfg(not(feature = "local-embeddings"))]
            {
                eprintln!(
                    "Error: --embedding-provider candle requires a build with \
                     --features local-embeddings"
                );
                std::process::exit(1);
            }
        }
        "hash" => {
            eprintln!(
                "WARNING: using non-semantic hash embeddings. Only byte-identical text matches; \
                 semantic search, auto-linking, and contradiction detection are effectively \
                 disabled. Rebuild with --features local-embeddings for real semantic search."
            );
            Some(Box::new(mentedb_embedding::HashEmbeddingProvider::new(384)))
        }
        other => {
            eprintln!("Error: unknown embedding provider '{other}' (expected candle, hash, none)");
            std::process::exit(1);
        }
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

    // One-shot maintenance sweep: `mentedb-server maintenance --data-dir ./data`
    // (for external cron). Runs consolidation, decay, archival, and cache
    // eviction once, then exits.
    if std::env::args().nth(1).as_deref() == Some("maintenance") {
        std::fs::create_dir_all(&config.data_dir)?;
        let db = match build_embedder(config.embedding_provider.as_deref()) {
            Some(embedder) => MenteDb::open_with_embedder(&config.data_dir, embedder)?,
            None => MenteDb::open(&config.data_dir)?,
        };
        let r = maintenance::run_sweep(&db);
        // When an LLM is configured, also give recent trajectory topics a
        // stable semantic label so the speculative cache and topic prediction
        // stop keying on raw message text.
        let extraction_config = build_extraction_config(&config);
        let canonicalized = maintenance::canonicalize_topics(&db, &extraction_config).await;
        println!(
            "maintenance sweep complete: {} consolidated, {} decayed, {} forgotten, {} topics canonicalized",
            r.consolidated, r.decayed, r.forgotten, canonicalized
        );
        return Ok(());
    }

    // Build extraction config first (before moving fields out of config)
    let extraction_config = build_extraction_config(&config);
    let auto_extract = config.auto_extract;
    let maintenance_interval_hours = config.maintenance_interval_hours;

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

    let db = match build_embedder(config.embedding_provider.as_deref()) {
        Some(embedder) => MenteDb::open_with_embedder(&data_dir, embedder)?,
        None => MenteDb::open(&data_dir)?,
    };
    let db = Arc::new(db);

    // Background maintenance sweep (self-hosted overnight jobs). Default every
    // 24h; 0 disables it (run the `maintenance` subcommand from cron instead).
    maintenance::spawn_scheduler(
        db.clone(),
        maintenance_interval_hours,
        extraction_config.clone(),
    );

    let auth_mode = if jwt_secret.is_some() {
        "enabled"
    } else {
        "disabled (development mode)"
    };

    // Spawn extraction worker if auto-extract is enabled
    let (extraction_tx, extraction_handle) = if auto_extract && extraction_config.is_some() {
        let (tx, handle) = extraction_queue::spawn_worker();
        (Some(tx), Some(handle))
    } else {
        (None, None)
    };

    // Self-organizing sharding: off unless MENTEDB_SHARDING is set. When on, the
    // fleet gossips membership and forwards each request to the node that owns the
    // agent, so N nodes shard themselves with no external coordinator.
    let cluster = cluster::Cluster::from_env();
    if let Some(c) = &cluster {
        c.spawn_gossip();
    }

    let state = Arc::new(AppState {
        db: db.clone(),
        spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
        jwt_secret: jwt_secret.clone(),
        admin_key: admin_key.clone(),
        start_time: Instant::now(),
        extraction_config,
        auto_extract,
        extraction_tx,
        cluster,
    });

    let app = routes::build_router(state.clone())
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth::auth_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            cluster::route,
        ))
        .layer(middleware::from_fn(metrics::track))
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

    // Drop the extraction sender to signal the worker to drain, then wait
    if let Some(handle) = extraction_handle {
        // Sender is inside AppState which is still alive via Arc, so drop the state first
        drop(state);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(10), handle).await;
    }

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
        #[cfg(feature = "bedrock")]
        "bedrock" => LlmProvider::Bedrock,
        #[cfg(not(feature = "bedrock"))]
        "bedrock" => {
            eprintln!(
                "WARNING: LLM provider 'bedrock' requires the server to be built with \
                 --features bedrock; extraction disabled"
            );
            return None;
        }
        "none" | "" => return None,
        other => {
            eprintln!(
                "WARNING: Unknown LLM provider '{other}' (expected one of: openai, anthropic, \
                 ollama, bedrock [needs --features bedrock], none), extraction disabled"
            );
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
        region: config.llm_region.clone(),
        max_extractions_per_conversation: 50,
        quality_threshold: config.extraction_quality_threshold.unwrap_or(0.7),
        deduplication_threshold: config.extraction_dedup_threshold.unwrap_or(0.85),
        enable_contradiction_check: true,
        enable_deduplication: true,
        extraction_passes: 1,
    })
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl+c");
    info!("received ctrl+c, shutting down");
}
