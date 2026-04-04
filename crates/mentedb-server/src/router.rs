//! Minimal HTTP/1.1 request parser and router for MenteDB.

use std::sync::Arc;

use mentedb::MenteDb;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tracing::error;

use crate::handlers;

type Db = Arc<Mutex<MenteDb>>;

// ---------------------------------------------------------------------------
// HTTP response helper
// ---------------------------------------------------------------------------

pub struct Response {
    pub status: u16,
    pub reason: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    pub fn new(status: u16, reason: &'static str, body: Value) -> Self {
        Self {
            status,
            reason,
            body: serde_json::to_vec(&body).unwrap_or_default(),
        }
    }

    pub fn ok(body: Value) -> Self {
        Self::new(200, "OK", body)
    }
    pub fn created(body: Value) -> Self {
        Self::new(201, "Created", body)
    }
    pub fn bad_request(msg: &str) -> Self {
        Self::new(400, "Bad Request", json!({ "error": msg }))
    }
    pub fn not_found(msg: &str) -> Self {
        Self::new(404, "Not Found", json!({ "error": msg }))
    }
    pub fn method_not_allowed() -> Self {
        Self::new(405, "Method Not Allowed", json!({ "error": "method not allowed" }))
    }
    pub fn internal_error(msg: &str) -> Self {
        Self::new(500, "Internal Server Error", json!({ "error": msg }))
    }

    fn to_http(&self) -> Vec<u8> {
        let header = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.status, self.reason, self.body.len()
        );
        let mut buf = header.into_bytes();
        buf.extend_from_slice(&self.body);
        buf
    }
}

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

const MAX_HEADER_SIZE: usize = 8 * 1024;
const MAX_BODY_SIZE: usize = 16 * 1024 * 1024; // 16 MiB

async fn read_request(stream: &mut TcpStream) -> Result<Request, &'static str> {
    let mut buf = vec![0u8; MAX_HEADER_SIZE];
    let mut filled = 0;

    // Read until we find the header terminator \r\n\r\n.
    let header_end = loop {
        if filled >= MAX_HEADER_SIZE {
            return Err("request headers too large");
        }
        let n = stream
            .read(&mut buf[filled..])
            .await
            .map_err(|_| "read error")?;
        if n == 0 {
            return Err("connection closed before headers");
        }
        filled += n;
        if let Some(pos) = find_header_end(&buf[..filled]) {
            break pos;
        }
    };

    let header_bytes = &buf[..header_end];
    let header_str = std::str::from_utf8(header_bytes).map_err(|_| "invalid UTF-8 in headers")?;

    // Parse request line.
    let mut lines = header_str.lines();
    let request_line = lines.next().ok_or("empty request")?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().ok_or("missing method")?.to_string();
    let path = parts.next().ok_or("missing path")?.to_string();

    // Parse Content-Length from headers.
    let mut content_length: usize = 0;
    for line in lines {
        if let Some(val) = line.strip_prefix("Content-Length:").or_else(|| line.strip_prefix("content-length:")) {
            content_length = val.trim().parse().map_err(|_| "bad Content-Length")?;
        }
    }

    if content_length > MAX_BODY_SIZE {
        return Err("body too large");
    }

    // body_start is after the \r\n\r\n
    let body_start = header_end + 4;
    let already_read = filled.saturating_sub(body_start);

    let mut body = Vec::with_capacity(content_length);
    if content_length > 0 {
        // Copy any body bytes already in the header buffer.
        let copy_len = already_read.min(content_length);
        body.extend_from_slice(&buf[body_start..body_start + copy_len]);

        // Read remaining body bytes.
        while body.len() < content_length {
            let mut chunk = vec![0u8; (content_length - body.len()).min(8192)];
            let n = stream
                .read(&mut chunk)
                .await
                .map_err(|_| "read error on body")?;
            if n == 0 {
                return Err("connection closed before body complete");
            }
            body.extend_from_slice(&chunk[..n]);
        }
    }

    Ok(Request { method, path, body })
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub async fn handle_connection(mut stream: TcpStream, db: Db) {
    let req = match read_request(&mut stream).await {
        Ok(r) => r,
        Err(msg) => {
            let resp = Response::bad_request(msg);
            let _ = stream.write_all(&resp.to_http()).await;
            return;
        }
    };

    let resp = route(req, db).await;

    if let Err(e) = stream.write_all(&resp.to_http()).await {
        error!("failed to write response: {e}");
    }
}

async fn route(req: Request, db: Db) -> Response {
    let method = req.method.as_str();
    let path = req.path.as_str();

    // Strip query string for routing.
    let route_path = path.split('?').next().unwrap_or(path);

    match (method, route_path) {
        ("GET", "/v1/health") => handlers::health().await,

        ("POST", "/v1/memories") => handlers::store_memory(db, &req.body).await,

        ("POST", "/v1/recall") => handlers::recall_memories(db, &req.body).await,

        ("POST", "/v1/search") => handlers::search_similar(db, &req.body).await,

        ("POST", "/v1/edges") => handlers::create_edge(db, &req.body).await,

        _ if route_path.starts_with("/v1/memories/") => {
            let id_str = &route_path["/v1/memories/".len()..];
            if id_str.is_empty() {
                return Response::bad_request("missing memory ID in path");
            }
            match method {
                "GET" => handlers::get_memory(db, id_str).await,
                "DELETE" => handlers::forget_memory(db, id_str).await,
                _ => Response::method_not_allowed(),
            }
        }

        _ => Response::not_found("unknown endpoint"),
    }
}
