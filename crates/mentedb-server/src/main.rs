//! MenteDB Server — JSON-over-HTTP REST API for the mind database.

mod handlers;
mod router;

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use mentedb::MenteDb;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tracing::{error, info};

fn parse_args() -> (PathBuf, u16) {
    let args: Vec<String> = env::args().collect();
    let mut data_dir = PathBuf::from("./mentedb-data");
    let mut port: u16 = 6677;

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
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    (data_dir, port)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let (data_dir, port) = parse_args();

    std::fs::create_dir_all(&data_dir)?;

    let db = MenteDb::open(&data_dir)?;
    let db = Arc::new(Mutex::new(db));

    println!("MenteDB v0.1.0");
    println!("Data directory: {}", data_dir.display());
    println!("Listening on: 0.0.0.0:{port}");
    println!();
    println!("Endpoints:");
    println!("  GET    /v1/health          — health check");
    println!("  POST   /v1/memories        — store a memory");
    println!("  GET    /v1/memories/:id     — get a memory by ID");
    println!("  DELETE /v1/memories/:id     — forget a memory");
    println!("  POST   /v1/recall          — query via MQL");
    println!("  POST   /v1/search          — vector similarity search");
    println!("  POST   /v1/edges           — create an edge");

    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr).await?;

    info!("MenteDB server listening on {addr}");

    let accept_loop = async {
        loop {
            match listener.accept().await {
                Ok((stream, peer_addr)) => {
                    info!("connection from {peer_addr}");
                    let db = db.clone();
                    tokio::spawn(async move {
                        router::handle_connection(stream, db).await;
                        info!("connection from {peer_addr} closed");
                    });
                }
                Err(e) => {
                    error!("failed to accept connection: {e}");
                }
            }
        }
    };

    tokio::select! {
        _ = accept_loop => {}
        _ = tokio::signal::ctrl_c() => {
            info!("received ctrl+c, shutting down...");
        }
    }

    let mut db = db.lock().await;
    db.close()?;
    info!("MenteDB server stopped");

    Ok(())
}
