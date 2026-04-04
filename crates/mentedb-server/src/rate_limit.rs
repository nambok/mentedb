//! Token bucket rate limiter implemented as a tower Layer/Service.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::IntoResponse;
use serde_json::json;
use tokio::sync::Mutex;
use tower::{Layer, Service};

/// Per-IP token bucket.
struct Bucket {
    tokens: u32,
    last_refill: Instant,
    max_tokens: u32,
    refill_rate: u32,
}

impl Bucket {
    fn new(max_tokens: u32, refill_rate: u32) -> Self {
        Self {
            tokens: max_tokens,
            last_refill: Instant::now(),
            max_tokens,
            refill_rate,
        }
    }

    /// Refill tokens based on elapsed time and try to consume one.
    fn try_consume(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let refill = (elapsed.as_secs_f64() * f64::from(self.refill_rate)) as u32;
        if refill > 0 {
            self.tokens = (self.tokens + refill).min(self.max_tokens);
            self.last_refill = now;
        }
        if self.tokens > 0 {
            self.tokens -= 1;
            true
        } else {
            false
        }
    }
}

/// Shared rate limiter state.
#[derive(Clone)]
pub struct RateLimiter {
    buckets: Arc<Mutex<HashMap<IpAddr, Bucket>>>,
    max_tokens: u32,
    refill_rate: u32,
}

impl RateLimiter {
    /// Creates a new rate limiter with the given bucket capacity and refill rate.
    pub fn new(max_tokens: u32, refill_rate: u32) -> Self {
        Self {
            buckets: Arc::new(Mutex::new(HashMap::new())),
            max_tokens,
            refill_rate,
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(100, 10)
    }
}

// -- Tower Layer --

impl<S> Layer<S> for RateLimiter {
    type Service = RateLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RateLimitService {
            inner,
            limiter: self.clone(),
        }
    }
}

// -- Tower Service --

#[derive(Clone)]
pub struct RateLimitService<S> {
    inner: S,
    limiter: RateLimiter,
}

impl<S> Service<Request<Body>> for RateLimitService<S>
where
    S: Service<Request<Body>, Response = axum::response::Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = axum::response::Response;
    type Error = S::Error;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let limiter = self.limiter.clone();
        let mut inner = self.inner.clone();

        // Extract client IP from connection info or forwarded headers.
        let ip = req
            .extensions()
            .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
            .map(|ci| ci.0.ip())
            .unwrap_or(IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED));

        Box::pin(async move {
            let allowed = {
                let mut buckets = limiter.buckets.lock().await;
                let bucket = buckets
                    .entry(ip)
                    .or_insert_with(|| Bucket::new(limiter.max_tokens, limiter.refill_rate));
                bucket.try_consume()
            };

            if allowed {
                inner.call(req).await
            } else {
                let body = json!({ "error": "too many requests" });
                Ok((StatusCode::TOO_MANY_REQUESTS, axum::Json(body)).into_response())
            }
        })
    }
}
