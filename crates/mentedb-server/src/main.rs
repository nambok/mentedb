//! MenteDB Server: axum based REST API with WebSocket, JWT auth, and observability.

// The binary links the library's modules instead of compiling its own copy,
// so library items the binary does not call are not dead code here and new
// library modules need no declaration in this file.
mod tls;

use mentedb_server::{
    auth, cluster, databases, extraction_queue, grpc, maintenance, metrics, rate_limit, routes,
    state,
};

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
    /// PEM certificate chain for HTTPS (--tls-cert / MENTEDB_TLS_CERT_FILE).
    tls_cert: Option<PathBuf>,
    /// PEM private key for HTTPS (--tls-key / MENTEDB_TLS_KEY_FILE).
    tls_key: Option<PathBuf>,
    /// How long in-flight HTTPS connections get to finish on shutdown.
    shutdown_drain_secs: u64,
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
    let mut tls_cert: Option<PathBuf> = None;
    let mut tls_key: Option<PathBuf> = None;
    let mut shutdown_drain_secs: u64 = 10;

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
                    eprintln!(
                        "Error: --embedding-provider requires a value (candle, bedrock, hash, none)"
                    );
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
            "--tls-cert" => {
                if i + 1 < args.len() {
                    tls_cert = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    eprintln!("Error: --tls-cert requires a PEM file path");
                    std::process::exit(1);
                }
            }
            "--tls-key" => {
                if i + 1 < args.len() {
                    tls_key = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    eprintln!("Error: --tls-key requires a PEM file path");
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
    if tls_cert.is_none()
        && let Ok(v) = env::var("MENTEDB_TLS_CERT_FILE")
        && !v.is_empty()
    {
        tls_cert = Some(PathBuf::from(v));
    }
    if tls_key.is_none()
        && let Ok(v) = env::var("MENTEDB_TLS_KEY_FILE")
        && !v.is_empty()
    {
        tls_key = Some(PathBuf::from(v));
    }
    if let Ok(v) = env::var("MENTEDB_SHUTDOWN_DRAIN_SECS")
        && let Ok(n) = v.parse::<u64>()
    {
        shutdown_drain_secs = n;
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
        tls_cert,
        tls_key,
        shutdown_drain_secs,
    }
}

/// Build the embedding provider for the server.
///
/// Without an embedder the engine's semantic search, auto-linking, and
/// contradiction detection are inert (only keyword retrieval works), so the
/// server always wires one and is loud about quality degradation.
///
/// Selection: `candle` (semantic, requires the `local-embeddings` build
/// feature), `bedrock` (AWS Bedrock, requires the `bedrock` build feature),
/// `hash` (deterministic but non-semantic), `none` (disable).
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
                        Some(Box::new(provider))
                    }
                    Err(e) => {
                        eprintln!(
                            "WARNING: failed to load Candle embedding model ({e}); falling back \
                             to non-semantic hash embeddings. Semantic search quality will be \
                             severely degraded."
                        );
                        Some(Box::new(mentedb_embedding::HashEmbeddingProvider::new(384)))
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
        "bedrock" => {
            #[cfg(feature = "bedrock")]
            {
                // Cohere v3 by default: measurably better recall than Titan
                // and the same model the cloud platform uses.
                let model = env::var("MENTEDB_EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "cohere.embed-english-v3".to_string());
                let region = env::var("MENTEDB_EMBEDDING_REGION")
                    .or_else(|_| env::var("AWS_REGION"))
                    .unwrap_or_else(|_| "us-east-1".to_string());
                let config = match mentedb_embedding::BedrockEmbeddingConfig::for_model(
                    region.clone(),
                    &model,
                ) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Error: --embedding-provider bedrock: {e}");
                        std::process::exit(1);
                    }
                };
                let provider = match mentedb_embedding::BedrockEmbeddingProvider::new(config) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("Error: --embedding-provider bedrock: {e}");
                        std::process::exit(1);
                    }
                };
                // One real call so a missing permission or model agreement
                // shows up in the startup log, not on the first write. A
                // failure here is logged, not fatal, so a brief Bedrock outage
                // does not crash loop the server.
                use mentedb_embedding::provider::EmbeddingProvider as _;
                match provider.embed("startup check") {
                    Ok(_) => tracing::info!(%model, %region, "using Bedrock embeddings"),
                    Err(e) => tracing::warn!(
                        %model,
                        %region,
                        error = %e,
                        "Bedrock embedding check failed; writes will fail until this is fixed"
                    ),
                }
                Some(Box::new(provider))
            }
            #[cfg(not(feature = "bedrock"))]
            {
                eprintln!(
                    "Error: --embedding-provider bedrock requires a build with --features bedrock"
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
            eprintln!(
                "Error: unknown embedding provider '{other}' (expected candle, bedrock, hash, none)"
            );
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
    let verifier = match build_verifier(&jwt_secret) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    if config.require_auth && verifier.is_none() {
        eprintln!(
            "Error: --require-auth was set but neither --jwt-secret nor \
             MENTEDB_JWT_PUBLIC_KEY_FILE is configured"
        );
        std::process::exit(1);
    }
    if verifier.is_none() {
        eprintln!(
            "WARNING: Running without authentication. All endpoints are open. Set --jwt-secret \
             or MENTEDB_JWT_PUBLIC_KEY_FILE for production use."
        );
    }

    // TLS: both files or neither. Load and validate before anything binds so
    // a bad certificate fails the start instead of the first handshake.
    let tls = match tls::resolve(config.tls_cert.clone(), config.tls_key.clone())
        .and_then(|paths| paths.as_ref().map(tls::load).transpose())
    {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    let cognitive_config = match cognitive_config_from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    std::fs::create_dir_all(&data_dir)?;

    let embedder: Option<Arc<dyn mentedb_embedding::provider::EmbeddingProvider>> =
        build_embedder(config.embedding_provider.as_deref()).map(Arc::from);
    let databases = Arc::new(databases::Databases::open(
        &data_dir,
        embedder,
        cognitive_config,
        databases_config_from_env(),
    )?);
    let db = databases.default_db();

    // Background maintenance sweep (self-hosted overnight jobs). Default every
    // 24h; 0 disables it (run the `maintenance` subcommand from cron instead).
    maintenance::spawn_scheduler(
        databases.clone(),
        maintenance_interval_hours,
        extraction_config.clone(),
    );

    let auth_mode = if verifier.is_some() {
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
        databases: databases.clone(),
        verifier: verifier.map(Arc::new),
        spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
        jwt_secret: jwt_secret.clone(),
        admin_key: admin_key.clone(),
        start_time: Instant::now(),
        extraction_config,
        auto_extract,
        extraction_tx,
        cluster,
        turn_trackers: Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
    });

    let scheme = if tls.is_some() { "https" } else { "http" };
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

    println!("MenteDB v{}", env!("CARGO_PKG_VERSION"));
    println!("Data directory: {}", data_dir.display());
    println!("REST API on: {scheme}://0.0.0.0:{port}");
    println!(
        "gRPC API on: 0.0.0.0:{grpc_port} ({})",
        if tls.is_some() { "TLS" } else { "plaintext" }
    );
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
    println!("  POST   /v1/process_turn     full cognitive turn pipeline");
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

    let mut grpc_builder = TonicServer::builder();
    if let Some(t) = &tls {
        let identity = tonic::transport::Identity::from_pem(&t.cert_pem, &t.key_pem);
        grpc_builder = match grpc_builder
            .tls_config(tonic::transport::ServerTlsConfig::new().identity(identity))
        {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Error: gRPC TLS configuration: {e}");
                std::process::exit(1);
            }
        };
    }
    let grpc_handle = tokio::spawn(async move {
        info!("gRPC server listening on {grpc_addr}");
        grpc_builder
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

    info!("MenteDB REST server listening on {scheme}://{addr}");

    tls::serve_rest(
        listener,
        app,
        tls.map(|t| t.rest),
        shutdown_signal(),
        std::time::Duration::from_secs(config.shutdown_drain_secs),
    )
    .await?;

    grpc_handle.abort();

    // Drop the extraction sender to signal the worker to drain, then wait
    if let Some(handle) = extraction_handle {
        // Sender is inside AppState which is still alive via Arc, so drop the state first
        drop(state);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(10), handle).await;
    }

    drop(db);
    databases.close_all();
    info!("MenteDB server stopped");

    Ok(())
}

/// Token verification from the HS256 secret or, when
/// `MENTEDB_JWT_PUBLIC_KEY_FILE` is set, a PEM public key (RS256, ES256 or
/// EdDSA) so tokens can be signed elsewhere and no secret lives on the server.
/// `MENTEDB_JWT_ISSUER` / `MENTEDB_JWT_AUDIENCE` add `iss` / `aud` checks.
fn build_verifier(jwt_secret: &Option<String>) -> Result<Option<auth::TokenVerifier>, String> {
    let issuer = env::var("MENTEDB_JWT_ISSUER")
        .ok()
        .filter(|s| !s.is_empty());
    let audience = env::var("MENTEDB_JWT_AUDIENCE")
        .ok()
        .filter(|s| !s.is_empty());
    let inline_key = env::var("MENTEDB_JWT_PUBLIC_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let key_file = env::var("MENTEDB_JWT_PUBLIC_KEY_FILE")
        .ok()
        .filter(|s| !s.is_empty());
    let public_key = inline_key.is_some() || key_file.is_some();
    let verifier = match (inline_key, key_file) {
        (Some(_), Some(_)) => {
            return Err(
                "set MENTEDB_JWT_PUBLIC_KEY or MENTEDB_JWT_PUBLIC_KEY_FILE, not both".to_string(),
            );
        }
        (Some(text), None) => Some(
            auth::TokenVerifier::public_key_text(&text)
                .map_err(|e| format!("MENTEDB_JWT_PUBLIC_KEY: {e}"))?,
        ),
        (None, Some(path)) => {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("reading MENTEDB_JWT_PUBLIC_KEY_FILE {path}: {e}"))?;
            Some(
                auth::TokenVerifier::public_key_text(&text)
                    .map_err(|e| format!("MENTEDB_JWT_PUBLIC_KEY_FILE {path}: {e}"))?,
            )
        }
        (None, None) => jwt_secret
            .as_deref()
            .map(auth::TokenVerifier::shared_secret),
    };
    let policy = token_policy_from(
        public_key,
        env::var("MENTEDB_ALLOW_UNSCOPED_TOKENS").ok().as_deref(),
        env::var("MENTEDB_JWT_DBS_SUB_PATTERN").ok().as_deref(),
    );
    Ok(verifier.map(|v| v.with_issuer_audience(issuer, audience).with_policy(policy)))
}

/// Token rules: with a public key configured (tokens minted elsewhere, often
/// one per tenant), a token must carry `dbs` unless
/// `MENTEDB_ALLOW_UNSCOPED_TOKENS` is on; `MENTEDB_JWT_DBS_SUB_PATTERN` binds
/// `dbs` to the token's `sub`.
fn token_policy_from(
    public_key: bool,
    allow_unscoped: Option<&str>,
    sub_pattern: Option<&str>,
) -> auth::TokenPolicy {
    let allow = allow_unscoped.is_some_and(env_flag);
    auth::TokenPolicy {
        require_dbs: public_key && !allow,
        dbs_sub_pattern: sub_pattern
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(String::from),
    }
}

/// `1`, `true`, `yes` or `on`, any case.
fn env_flag(v: &str) -> bool {
    matches!(
        v.trim().to_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Limits for named databases from `MENTEDB_MAX_OPEN_DATABASES` and
/// `MENTEDB_AUTO_CREATE_DATABASES` (default on).
fn databases_config_from_env() -> databases::DatabasesConfig {
    let mut cfg = databases::DatabasesConfig::default();
    if let Some(n) = env::var("MENTEDB_MAX_OPEN_DATABASES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
    {
        cfg.max_open = n;
    }
    if let Ok(v) = env::var("MENTEDB_AUTO_CREATE_DATABASES") {
        cfg.auto_create = !matches!(v.to_lowercase().as_str(), "0" | "false" | "no" | "off");
    }
    cfg
}

/// Engine settings every database opens with.
///
/// `MENTEDB_CONTRADICTION_POLICY` picks what write inference does when a new
/// memory contradicts an older one: `supersede` (default) closes the older
/// one, `flag` keeps both valid and records a Contradicts edge. Any other
/// value is a startup error.
fn cognitive_config_from_env() -> Result<mentedb::CognitiveConfig, String> {
    cognitive_config_from(env::var("MENTEDB_CONTRADICTION_POLICY").ok().as_deref())
}

fn cognitive_config_from(
    contradiction_policy: Option<&str>,
) -> Result<mentedb::CognitiveConfig, String> {
    let mut cfg = mentedb::CognitiveConfig::default();
    if let Some(v) = contradiction_policy.filter(|v| !v.trim().is_empty()) {
        cfg.inference_config.contradiction_policy = v
            .parse::<mentedb_cognitive::ContradictionPolicy>()
            .map_err(|e| format!("MENTEDB_CONTRADICTION_POLICY: {e}"))?;
    }
    Ok(cfg)
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

    Some(
        ExtractionConfig {
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
            ..ExtractionConfig::default()
        }
        .with_output_env(),
    )
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl+c");
    info!("received ctrl+c, shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_cognitive::ContradictionPolicy;

    #[test]
    fn token_policy_requires_dbs_only_with_a_public_key() {
        assert!(token_policy_from(true, None, None).require_dbs);
        assert!(!token_policy_from(true, Some("1"), None).require_dbs);
        assert!(!token_policy_from(false, None, None).require_dbs);
        assert_eq!(
            token_policy_from(false, None, Some("tenant.{sub}.*"))
                .dbs_sub_pattern
                .as_deref(),
            Some("tenant.{sub}.*")
        );
        assert!(
            token_policy_from(false, None, Some("  "))
                .dbs_sub_pattern
                .is_none()
        );
    }

    #[test]
    fn contradiction_policy_from_env_value() {
        let policy = |v| cognitive_config_from(v).map(|c| c.inference_config.contradiction_policy);
        assert_eq!(policy(None), Ok(ContradictionPolicy::Supersede));
        assert_eq!(policy(Some("")), Ok(ContradictionPolicy::Supersede));
        assert_eq!(
            policy(Some("supersede")),
            Ok(ContradictionPolicy::Supersede)
        );
        assert_eq!(policy(Some("FLAG")), Ok(ContradictionPolicy::Flag));
        assert!(policy(Some("ignore")).is_err());
    }
}
