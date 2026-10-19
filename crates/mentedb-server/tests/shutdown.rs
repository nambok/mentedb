//! The server binary stops cleanly on SIGTERM: it exits with success after
//! draining, and writes it acknowledged before the signal are there after a
//! restart.

#![cfg(unix)]

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn start(data: &std::path::Path, port: u16) -> Child {
    Command::new(env!("CARGO_BIN_EXE_mentedb-server"))
        .args([
            "--data-dir",
            data.to_str().unwrap(),
            "--port",
            &port.to_string(),
            "--grpc-port",
            "0",
            "--embedding-provider",
            "hash",
            "--maintenance-interval-hours",
            "0",
        ])
        .env_remove("MENTEDB_JWT_SECRET")
        .env_remove("MENTEDB_JWT_PUBLIC_KEY")
        .env_remove("MENTEDB_JWT_PUBLIC_KEY_FILE")
        .env("MENTEDB_SHUTDOWN_DRAIN_SECS", "5")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
}

async fn wait_healthy(client: &reqwest::Client, base: &str) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Ok(r) = client.get(format!("{base}/v1/health")).send().await
            && r.status().is_success()
        {
            return;
        }
        assert!(Instant::now() < deadline, "server did not come up");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn wait_exit(child: &mut Child, within: Duration) -> std::process::ExitStatus {
    let deadline = Instant::now() + within;
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("server did not exit within {within:?} of SIGTERM");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn sigterm_drains_closes_and_keeps_acknowledged_writes() {
    let data = tempfile::tempdir().unwrap();
    let port = free_port();
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();

    let mut server = start(data.path(), port);
    wait_healthy(&client, &base).await;
    let mut ids = Vec::new();
    for i in 0..5 {
        let r = client
            .post(format!("{base}/v1/memories"))
            .json(&json!({
                "agent_id": "00000000-0000-0000-0000-000000000001",
                "memory_type": "semantic",
                "content": format!("acknowledged fact number {i}"),
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), reqwest::StatusCode::CREATED);
        let body: Value = r.json().await.unwrap();
        ids.push(body["id"].as_str().unwrap().to_string());
    }

    let killed = Command::new("kill")
        .args(["-TERM", &server.id().to_string()])
        .status()
        .unwrap();
    assert!(killed.success());
    let status = wait_exit(&mut server, Duration::from_secs(30));
    assert!(
        status.success(),
        "SIGTERM must end in a clean shutdown, got {status:?}"
    );

    // Restart on the same directory: every acknowledged write is there, and
    // the directory lock was released (the open would fail otherwise).
    let mut again = start(data.path(), port);
    wait_healthy(&client, &base).await;
    for id in &ids {
        let r = client
            .get(format!("{base}/v1/memories/{id}"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), reqwest::StatusCode::OK, "{id} survived");
    }
    let _ = Command::new("kill")
        .args(["-TERM", &again.id().to_string()])
        .status();
    let _ = wait_exit(&mut again, Duration::from_secs(30));
}
