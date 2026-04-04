//! MenteDB Server — TCP server for the mind database.

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
    println!("Listening on: 0.0.0.0:{}", port);

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await?;

    info!("MenteDB server listening on {}", addr);

    let accept_loop = async {
        loop {
            match listener.accept().await {
                Ok((socket, peer_addr)) => {
                    info!("New connection from {}", peer_addr);
                    let _db = db.clone();
                    tokio::spawn(async move {
                        // Protocol handling will be implemented in a future version.
                        drop(socket);
                        info!("Connection from {} closed", peer_addr);
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                }
            }
        }
    };

    tokio::select! {
        _ = accept_loop => {}
        _ = tokio::signal::ctrl_c() => {
            info!("Received Ctrl+C, shutting down...");
        }
    }

    // Graceful shutdown: flush and close the database.
    let mut db = db.lock().await;
    db.close()?;
    info!("MenteDB server stopped");

    Ok(())
}
