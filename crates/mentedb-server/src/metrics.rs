//! Prometheus metrics exposed at `GET /metrics` in the standard text format.
//!
//! Aggregate only, with no per-account or per-agent labels, so it is safe to
//! scrape without auth. It covers process health (CPU and memory on Linux) plus
//! MenteDB gauges (memories stored, uptime, live cluster nodes) and HTTP request
//! rate and latency, which together answer the "is it healthy and how hard is it
//! working" question a dashboard needs.

use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

use axum::{
    body::Body,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use prometheus::{
    Encoder, HistogramVec, IntCounterVec, IntGauge, Registry, TextEncoder, histogram_opts, opts,
};

use crate::state::AppState;

/// The metric set, held in a process-global registry.
pub struct Metrics {
    registry: Registry,
    uptime: IntGauge,
    memory_count: IntGauge,
    live_nodes: IntGauge,
    http_requests: IntCounterVec,
    http_duration: HistogramVec,
    // Engine metrics, refreshed from `db.metrics()` on each scrape. Cumulative
    // totals are exposed as gauges set to the running total (rate() still works).
    stores: IntGauge,
    recalls: IntGauge,
    bp_hits: IntGauge,
    bp_misses: IntGauge,
    bp_evictions: IntGauge,
    bp_pages: IntGauge,
    storage_bytes: IntGauge,
    pages: IntGauge,
    vector_index_size: IntGauge,
    graph_nodes: IntGauge,
    standing_rules: IntGauge,
}

static METRICS: OnceLock<Metrics> = OnceLock::new();

/// The process-global metric set, initialized on first use.
pub fn metrics() -> &'static Metrics {
    METRICS.get_or_init(Metrics::new)
}

impl Metrics {
    fn new() -> Self {
        let registry = Registry::new();

        // Process CPU and memory, populated on Linux via /proc.
        #[cfg(target_os = "linux")]
        {
            let collector = prometheus::process_collector::ProcessCollector::for_self();
            let _ = registry.register(Box::new(collector));
        }

        // `mentedb_up` is a constant 1; register it and let the registry own it.
        let up = IntGauge::new("mentedb_up", "1 if the server is serving").unwrap();
        up.set(1);
        registry.register(Box::new(up)).ok();

        let uptime = IntGauge::new("mentedb_uptime_seconds", "Seconds since start").unwrap();
        let memory_count =
            IntGauge::new("mentedb_memory_count", "Memories stored on this node").unwrap();
        let live_nodes = IntGauge::new(
            "mentedb_cluster_live_nodes",
            "Live nodes in the gossip cluster (1 when sharding is off)",
        )
        .unwrap();
        let http_requests = IntCounterVec::new(
            opts!(
                "mentedb_http_requests_total",
                "HTTP requests by method and status"
            ),
            &["method", "status"],
        )
        .unwrap();
        let http_duration = HistogramVec::new(
            histogram_opts!(
                "mentedb_http_request_duration_seconds",
                "HTTP request latency in seconds"
            ),
            &["method"],
        )
        .unwrap();

        // Engine gauges: create, register, and return in one step.
        let gauge = |name: &str, help: &str| {
            let g = IntGauge::new(name, help).unwrap();
            registry.register(Box::new(g.clone())).ok();
            g
        };
        let stores = gauge("mentedb_stores_total", "Memories written (writes)");
        let recalls = gauge("mentedb_recalls_total", "Recall/query operations (reads)");
        let bp_hits = gauge("mentedb_buffer_pool_hits_total", "Page cache hits");
        let bp_misses = gauge("mentedb_buffer_pool_misses_total", "Page cache misses");
        let bp_evictions = gauge(
            "mentedb_buffer_pool_evictions_total",
            "Page cache evictions",
        );
        let bp_pages = gauge("mentedb_buffer_pool_pages", "Frames holding a page");
        let storage_bytes = gauge("mentedb_storage_bytes", "On-disk data size in bytes");
        let pages = gauge("mentedb_pages_total", "Pages in the store");
        let vector_index_size = gauge("mentedb_vector_index_size", "Vectors in the HNSW index");
        let graph_nodes = gauge("mentedb_graph_nodes", "Nodes in the memory graph");
        let standing_rules = gauge(
            "mentedb_standing_rules",
            "Pinned standing rules (scope:always)",
        );

        for g in [&uptime, &memory_count, &live_nodes] {
            registry.register(Box::new(g.clone())).ok();
        }
        registry.register(Box::new(http_requests.clone())).ok();
        registry.register(Box::new(http_duration.clone())).ok();

        Self {
            registry,
            uptime,
            memory_count,
            live_nodes,
            http_requests,
            http_duration,
            stores,
            recalls,
            bp_hits,
            bp_misses,
            bp_evictions,
            bp_pages,
            storage_bytes,
            pages,
            vector_index_size,
            graph_nodes,
            standing_rules,
        }
    }
}

/// Middleware: time every request and count it by method and response status.
pub async fn track(req: Request, next: Next) -> Response {
    let method = req.method().as_str().to_string();
    let start = Instant::now();
    let resp = next.run(req).await;
    let m = metrics();
    let status = resp.status().as_u16().to_string();
    m.http_requests.with_label_values(&[&method, &status]).inc();
    m.http_duration
        .with_label_values(&[&method])
        .observe(start.elapsed().as_secs_f64());
    resp
}

/// `GET /metrics`: refresh the point-in-time gauges, then encode the registry.
pub async fn handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let m = metrics();
    m.uptime.set(state.start_time.elapsed().as_secs() as i64);
    let dm = state.db.metrics();
    m.memory_count.set(dm.memory_count as i64);
    m.stores.set(dm.stores as i64);
    m.recalls.set(dm.recalls as i64);
    m.bp_hits.set(dm.buffer_pool_hits as i64);
    m.bp_misses.set(dm.buffer_pool_misses as i64);
    m.bp_evictions.set(dm.buffer_pool_evictions as i64);
    m.bp_pages.set(dm.buffer_pool_pages as i64);
    m.storage_bytes.set(dm.storage_bytes as i64);
    m.pages.set(dm.page_count as i64);
    m.vector_index_size.set(dm.vector_index_size as i64);
    m.graph_nodes.set(dm.graph_nodes as i64);
    m.standing_rules.set(dm.standing_rules as i64);
    let live = match &state.cluster {
        Some(c) => c.live_node_count().await as i64,
        None => 1,
    };
    m.live_nodes.set(live);

    let mut buf = Vec::new();
    if TextEncoder::new()
        .encode(&m.registry.gather(), &mut buf)
        .is_err()
    {
        return (StatusCode::INTERNAL_SERVER_ERROR, "metric encode error").into_response();
    }
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        Body::from(buf),
    )
        .into_response()
}
