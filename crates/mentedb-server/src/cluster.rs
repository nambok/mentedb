//! Self-organizing sharding for mentedb-server.
//!
//! Running N nodes shards the fleet with no external coordinator: nodes gossip to
//! converge on the live set (engine `mentedb::sharding::gossip`), rendezvous
//! placement picks one owner per agent (every node computes the same owner), and a
//! request that lands on the wrong node is forwarded to the owner. Ownership is the
//! single-writer lock, so an agent's database is only ever written by one node.
//!
//! Off unless `MENTEDB_SHARDING` is set. Routing needs JWT auth (the agent id comes
//! from the token); without a `--jwt-secret` the middleware is a pass-through.

use std::sync::Arc;
use std::time::Duration;

use axum::{
    Json,
    body::Body,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use mentedb::sharding::gossip::{GossipMembership, GossipTransport, GossipView};
use mentedb::sharding::{NodeRegistry, placement};

use crate::state::AppState;

/// Path peers POST their gossip view to (and the reply carries ours back).
pub const GOSSIP_PATH: &str = "/v1/cluster/gossip";
/// Marks a request already forwarded once, so a placement disagreement cannot loop.
const FORWARDED: &str = "x-mentedb-forwarded";

/// reqwest-backed gossip transport: POST our view to a peer, get theirs back.
#[derive(Clone)]
pub struct HttpGossip {
    client: reqwest::Client,
}

impl GossipTransport for HttpGossip {
    async fn exchange(&self, peer_addr: &str, ours: GossipView) -> Result<GossipView, String> {
        let url = format!("{}{}", peer_addr.trim_end_matches('/'), GOSSIP_PATH);
        let resp = self
            .client
            .post(&url)
            .json(&ours)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        resp.json::<GossipView>().await.map_err(|e| e.to_string())
    }
}

type Membership = GossipMembership<HttpGossip>;

/// Cluster handle held in `AppState`; present only when sharding is enabled.
#[derive(Clone)]
pub struct Cluster {
    node_id: String,
    membership: Arc<Membership>,
    http: reqwest::Client,
}

impl Cluster {
    /// Build from the environment, or return `None` when sharding is off or
    /// misconfigured:
    /// - `MENTEDB_SHARDING` = `1`/`true` to enable.
    /// - `MENTEDB_NODE_ID` (defaults to `$HOSTNAME`, else `node-<pid>`).
    /// - `MENTEDB_NODE_ADDR` this node's base URL peers reach it at (required).
    /// - `MENTEDB_SEEDS` comma-separated peer base URLs to bootstrap from.
    pub fn from_env() -> Option<Self> {
        let enabled = std::env::var("MENTEDB_SHARDING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if !enabled {
            return None;
        }
        let node_id = std::env::var("MENTEDB_NODE_ID")
            .ok()
            .or_else(|| std::env::var("HOSTNAME").ok())
            .unwrap_or_else(|| format!("node-{}", std::process::id()));
        let node_addr = std::env::var("MENTEDB_NODE_ADDR").unwrap_or_default();
        if node_addr.is_empty() {
            tracing::warn!(
                "MENTEDB_SHARDING is set but MENTEDB_NODE_ADDR is empty; sharding disabled"
            );
            return None;
        }
        let seeds: Vec<String> = std::env::var("MENTEDB_SEEDS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let http = reqwest::Client::new();
        let membership = Arc::new(GossipMembership::new(
            node_id.clone(),
            node_addr,
            seeds,
            Duration::from_secs(15),
            HttpGossip {
                client: http.clone(),
            },
        ));
        tracing::info!(node = %node_id, "sharding enabled: self-organizing gossip fleet");
        Some(Self {
            node_id,
            membership,
            http,
        })
    }

    /// How many nodes the gossip layer currently considers live.
    pub async fn live_node_count(&self) -> usize {
        self.membership
            .live_nodes()
            .await
            .map(|n| n.len())
            .unwrap_or(0)
    }

    /// Run the gossip loop in the background: one anti-entropy round per interval.
    pub fn spawn_gossip(&self) {
        let membership = self.membership.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(3));
            loop {
                ticker.tick().await;
                membership.round().await;
            }
        });
    }
}

/// Handler for `GOSSIP_PATH`: merge a peer's view and answer with ours. 404 when
/// sharding is off.
pub async fn gossip_handler(
    State(state): State<Arc<AppState>>,
    Json(incoming): Json<GossipView>,
) -> Response {
    match &state.cluster {
        Some(cluster) => {
            cluster.membership.merge(incoming);
            Json(cluster.membership.snapshot()).into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Middleware: forward a request to the node that owns its agent, or serve it here.
/// A pass-through when sharding is off, the request carries no resolvable agent, or
/// we already own it.
pub async fn route(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let Some(cluster) = state.cluster.clone() else {
        return next.run(req).await;
    };
    // The gossip endpoint and already-forwarded requests are always served locally.
    if req.uri().path() == GOSSIP_PATH || req.headers().contains_key(FORWARDED) {
        return next.run(req).await;
    }
    // Identify the agent from the JWT as an owned value before any await, so no
    // reference to the request is held across one. No token, no per-agent affinity.
    let Some(agent) = agent_from_request(&req, &state) else {
        return next.run(req).await;
    };

    let nodes = cluster.membership.live_nodes().await.unwrap_or_default();
    let ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
    let owner = placement::owner(&agent, &ids)
        .unwrap_or(cluster.node_id.as_str())
        .to_string();
    if owner == cluster.node_id {
        return next.run(req).await;
    }
    match nodes.iter().find(|n| n.id == owner).map(|n| n.addr.clone()) {
        Some(addr) => forward(&cluster.http, &addr, req).await,
        // Owner has no known address (mid-convergence): serve locally rather than fail.
        None => next.run(req).await,
    }
}

/// The agent id a request belongs to, from its Bearer JWT. `None` without auth (no
/// secret, no token, or a token that does not validate).
fn agent_from_request(req: &Request, state: &AppState) -> Option<String> {
    let secret = state.jwt_secret.as_deref()?;
    let token = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))?;
    crate::auth::validate_token(secret, token)
        .ok()
        .map(|c| c.agent_id)
}

/// Reverse-proxy the request to the owning node and relay its response.
async fn forward(http: &reqwest::Client, addr: &str, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let path = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let url = format!("{}{}", addr.trim_end_matches('/'), path);

    let bytes = match axum::body::to_bytes(body, 16 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            return error_response(StatusCode::BAD_REQUEST, "request body too large to forward");
        }
    };

    let mut builder = http.request(parts.method, &url).body(bytes.to_vec());
    for (name, value) in parts.headers.iter() {
        if name != header::HOST {
            builder = builder.header(name, value);
        }
    }
    builder = builder.header(FORWARDED, "1");

    match builder.send().await {
        Ok(resp) => {
            let status = resp.status();
            let headers = resp.headers().clone();
            let body = resp.bytes().await.unwrap_or_default();
            let mut out = Response::builder().status(status);
            for (name, value) in headers.iter() {
                if name != header::TRANSFER_ENCODING && name != header::CONNECTION {
                    out = out.header(name, value);
                }
            }
            out.body(Body::from(body)).unwrap_or_else(|_| {
                error_response(StatusCode::BAD_GATEWAY, "bad upstream response")
            })
        }
        Err(e) => {
            tracing::warn!(error = %e, url = %url, "sharding: forward to owner failed");
            error_response(StatusCode::BAD_GATEWAY, "owner node unreachable")
        }
    }
}

fn error_response(status: StatusCode, msg: &str) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        serde_json::json!({ "error": msg }).to_string(),
    )
        .into_response()
}
