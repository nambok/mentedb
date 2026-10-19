//! Token bucket rate limiter implemented as a tower Layer/Service.
//!
//! Requests are counted per caller: the token's `sub` (else `jti`, else its
//! agent) when the request is authenticated, otherwise the peer IP address.
//! The layer must sit inside the auth middleware so it sees the caller.

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

/// Default bucket size (`MENTEDB_RATE_LIMIT_BURST`).
pub const DEFAULT_BURST: u32 = 100;
/// Default refill per second (`MENTEDB_RATE_LIMIT_PER_SEC`).
pub const DEFAULT_PER_SEC: u32 = 10;
/// Buckets kept before idle (full) ones are dropped.
const MAX_TRACKED_KEYS: usize = 100_000;

/// What requests are counted under.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Key {
    Caller(String),
    Ip(IpAddr),
}

/// One caller's token bucket.
struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

/// Shared rate limiter state.
#[derive(Clone)]
pub struct RateLimiter {
    buckets: Arc<Mutex<HashMap<Key, Bucket>>>,
    max_tokens: u32,
    refill_rate: u32,
}

impl RateLimiter {
    /// A limiter allowing bursts of `max_tokens` refilled at `refill_rate` per
    /// second. Either at 0 turns limiting off.
    pub fn new(max_tokens: u32, refill_rate: u32) -> Self {
        Self {
            buckets: Arc::new(Mutex::new(HashMap::new())),
            max_tokens,
            refill_rate,
        }
    }

    /// `MENTEDB_RATE_LIMIT_BURST` and `MENTEDB_RATE_LIMIT_PER_SEC` (0 for
    /// either turns limiting off), defaulting to 100 and 10.
    pub fn from_env() -> Self {
        let read = |name: &str, default: u32| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok())
                .unwrap_or(default)
        };
        Self::new(
            read("MENTEDB_RATE_LIMIT_BURST", DEFAULT_BURST),
            read("MENTEDB_RATE_LIMIT_PER_SEC", DEFAULT_PER_SEC),
        )
    }

    fn enabled(&self) -> bool {
        self.max_tokens > 0 && self.refill_rate > 0
    }

    async fn allow(&self, key: Key) -> bool {
        let now = Instant::now();
        let max = f64::from(self.max_tokens);
        let rate = f64::from(self.refill_rate);
        let mut buckets = self.buckets.lock().await;
        if buckets.len() >= MAX_TRACKED_KEYS {
            // A full bucket carries no state worth keeping.
            buckets.retain(|_, b| {
                b.tokens + now.duration_since(b.last_refill).as_secs_f64() * rate < max
            });
        }
        let bucket = buckets.entry(key).or_insert(Bucket {
            tokens: max,
            last_refill: now,
        });
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * rate).min(max);
        bucket.last_refill = now;
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(DEFAULT_BURST, DEFAULT_PER_SEC)
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

        let key = req
            .extensions()
            .get::<crate::auth::AuthenticatedAgent>()
            .and_then(|a| a.rate_key())
            .map(Key::Caller)
            .unwrap_or_else(|| {
                Key::Ip(
                    req.extensions()
                        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
                        .map(|ci| ci.0.ip())
                        .unwrap_or(IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED)),
                )
            });

        Box::pin(async move {
            if !limiter.enabled() || limiter.allow(key).await {
                inner.call(req).await
            } else {
                let body = json!({ "error": "too many requests" });
                Ok((StatusCode::TOO_MANY_REQUESTS, axum::Json(body)).into_response())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn buckets_are_per_key_and_refill() {
        let l = RateLimiter::new(2, 1000);
        let a = Key::Caller("alice".into());
        let b = Key::Caller("bob".into());
        assert!(l.allow(a.clone()).await);
        assert!(l.allow(a.clone()).await);
        assert!(!l.allow(a.clone()).await, "alice's burst is spent");
        assert!(l.allow(b.clone()).await, "bob has his own bucket");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        assert!(l.allow(a).await, "refilled");
    }

    #[test]
    fn zero_turns_limiting_off() {
        assert!(!RateLimiter::new(0, 10).enabled());
        assert!(!RateLimiter::new(10, 0).enabled());
        assert!(RateLimiter::new(1, 1).enabled());
    }
}
