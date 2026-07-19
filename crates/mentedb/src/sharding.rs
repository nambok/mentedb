//! Lease-based elastic sharding.
//!
//! A different scaling model from the Raft cluster in `mentedb-replication`:
//! instead of replicating one dataset, it shards accounts across nodes so each
//! account's single-writer database lives on exactly one node. This module owns
//! the placement math and the coordination logic; the concrete lease and
//! membership backends are provided by the embedder (the engine takes no external
//! database dependency), via the [`LeaseStore`] and [`NodeRegistry`] traits.

use std::collections::HashMap;
use std::future::Future;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

pub mod gossip;
pub mod placement;

/// A held ownership lease. `epoch` is the fence token that must accompany writes,
/// so a node that lost ownership is rejected even if it does not notice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lease {
    pub key: String,
    pub node: String,
    pub epoch: u64,
    /// Unix seconds at which the lease lapses unless renewed.
    pub expiry: u64,
}

/// A live node and the base URL peers reach it at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub id: String,
    pub addr: String,
}

#[derive(Debug)]
pub enum LeaseError {
    /// Another node currently holds a valid lease on this key.
    Held { owner: String, expiry: u64 },
    /// We no longer own this lease (a renew or release found a different owner).
    Lost,
    /// A backend failure.
    Backend(String),
}

impl std::fmt::Display for LeaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LeaseError::Held { owner, expiry } => write!(f, "lease held by {owner} until {expiry}"),
            LeaseError::Lost => write!(f, "lease lost"),
            LeaseError::Backend(e) => write!(f, "lease backend error: {e}"),
        }
    }
}
impl std::error::Error for LeaseError {}

/// Where an account should be served.
#[derive(Debug, PartialEq, Eq)]
pub enum Resolution {
    /// This node owns it; serve locally. `epoch` fences writes.
    Local { epoch: u64 },
    /// Another node owns it; forward to this base URL.
    Remote { addr: String },
}

/// A backend that grants exactly-one-owner leases, fenced by an epoch and
/// expiring on a TTL. Implemented by the embedder (for example over DynamoDB
/// conditional writes) so the engine stays dependency-free.
pub trait LeaseStore: Send + Sync {
    /// Take ownership of `key` if it is free or expired, bumping the epoch; if we
    /// already hold a valid lease, return it unchanged.
    fn acquire(&self, key: &str) -> impl Future<Output = Result<Lease, LeaseError>> + Send;
    /// Extend a lease we hold, keeping its epoch; errors [`LeaseError::Lost`] if we
    /// no longer own it.
    fn renew(&self, lease: &Lease) -> impl Future<Output = Result<Lease, LeaseError>> + Send;
    /// Give up ownership so another node can take over immediately.
    fn release(&self, lease: &Lease) -> impl Future<Output = Result<(), LeaseError>> + Send;
    /// The current live lease for a key, if any.
    fn current(&self, key: &str) -> impl Future<Output = Result<Option<Lease>, LeaseError>> + Send;
}

/// A no-op lease store for self-coordinated fleets (gossip membership plus
/// deterministic placement). When every node agrees on the live set, they also
/// agree on the owner of each key with no shared lease, and the single-writer file
/// lock is the hard safety net during a handoff. `acquire` therefore grants a
/// local, monotonically increasing epoch without any cross-node round trip, so a
/// [`Coordinator`] can run without an external lease backend.
pub struct NoCoordLease {
    node: String,
    epochs: Mutex<HashMap<String, u64>>,
}

impl NoCoordLease {
    pub fn new(node: impl Into<String>) -> Self {
        Self {
            node: node.into(),
            epochs: Mutex::new(HashMap::new()),
        }
    }

    fn lease(&self, key: &str, epoch: u64) -> Lease {
        Lease {
            key: key.to_string(),
            node: self.node.clone(),
            epoch,
            expiry: u64::MAX,
        }
    }
}

impl LeaseStore for NoCoordLease {
    async fn acquire(&self, key: &str) -> Result<Lease, LeaseError> {
        let mut epochs = self.epochs.lock();
        let epoch = epochs.entry(key.to_string()).or_insert(0);
        *epoch += 1;
        Ok(self.lease(key, *epoch))
    }

    async fn renew(&self, lease: &Lease) -> Result<Lease, LeaseError> {
        Ok(lease.clone())
    }

    async fn release(&self, _lease: &Lease) -> Result<(), LeaseError> {
        Ok(())
    }

    async fn current(&self, key: &str) -> Result<Option<Lease>, LeaseError> {
        Ok(self.epochs.lock().get(key).map(|e| self.lease(key, *e)))
    }
}

/// A backend that tracks the live node set. Implemented by the embedder.
pub trait NodeRegistry: Send + Sync {
    fn heartbeat(&self) -> impl Future<Output = Result<(), String>> + Send;
    fn live_nodes(&self) -> impl Future<Output = Result<Vec<Node>, String>> + Send;
    fn node_id(&self) -> &str;
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Ties placement to leases and membership: decides whether this node serves a key
/// locally (fenced by the lease epoch) or forwards to its owner, and keeps the live
/// node set and held leases fresh. Generic over the backends so the engine owns the
/// logic while the embedder owns the storage.
pub struct Coordinator<L: LeaseStore, R: NodeRegistry> {
    enabled: bool,
    node: String,
    leases: L,
    registry: R,
    nodes: Mutex<Vec<Node>>,
    held: Mutex<HashMap<String, Lease>>,
}

impl<L: LeaseStore, R: NodeRegistry> Coordinator<L, R> {
    pub fn new(enabled: bool, node: impl Into<String>, leases: L, registry: R) -> Self {
        Self {
            enabled,
            node: node.into(),
            leases,
            registry,
            nodes: Mutex::new(Vec::new()),
            held: Mutex::new(HashMap::new()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Decide where `key` is served. When disabled, or when we are the only live
    /// node, resolves `Local` without contacting the lease store on the hot path.
    pub async fn resolve(&self, key: &str) -> Result<Resolution, LeaseError> {
        if !self.enabled {
            return Ok(Resolution::Local { epoch: 0 });
        }
        let nodes = self.nodes.lock().clone();
        let ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
        let owner = placement::owner(key, &ids)
            .map(str::to_string)
            .unwrap_or_else(|| self.node.clone());

        if owner == self.node {
            // Serve a cached, still-valid lease without a round trip. Clone out of
            // the guard so no lock is held across the await.
            let cached = self.held.lock().get(key).cloned();
            if let Some(l) = cached
                && l.expiry > now_secs() + 5
            {
                return Ok(Resolution::Local { epoch: l.epoch });
            }
            let lease = self.leases.acquire(key).await?;
            let epoch = lease.epoch;
            self.held.lock().insert(key.to_string(), lease);
            Ok(Resolution::Local { epoch })
        } else {
            let addr = nodes
                .iter()
                .find(|n| n.id == owner)
                .map(|n| n.addr.clone())
                .ok_or_else(|| LeaseError::Backend(format!("owner {owner} has no address")))?;
            Ok(Resolution::Remote { addr })
        }
    }

    /// Background upkeep: heartbeat membership, refresh the live node set, and renew
    /// (or drop) held leases. A no-op when disabled.
    pub async fn maintain(&self) {
        if !self.enabled {
            return;
        }
        if let Err(e) = self.registry.heartbeat().await {
            tracing::warn!(error = %e, "sharding: heartbeat failed");
        }
        match self.registry.live_nodes().await {
            Ok(live) => *self.nodes.lock() = live,
            Err(e) => tracing::warn!(error = %e, "sharding: live-nodes refresh failed"),
        }
        let held: Vec<Lease> = self.held.lock().values().cloned().collect();
        for lease in held {
            match self.leases.renew(&lease).await {
                Ok(fresh) => {
                    self.held.lock().insert(fresh.key.clone(), fresh);
                }
                Err(LeaseError::Lost) => {
                    tracing::info!(key = %lease.key, "sharding: lease lost, releasing");
                    self.held.lock().remove(&lease.key);
                }
                Err(e) => tracing::warn!(key = %lease.key, error = %e, "sharding: renew failed"),
            }
        }
    }

    /// Renew interval, comfortably shorter than the lease TTL.
    pub fn maintain_interval() -> Duration {
        Duration::from_secs(5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// In-memory lease store, enough to exercise the coordinator logic.
    struct MemLeases {
        node: String,
        rows: StdMutex<HashMap<String, Lease>>,
    }

    impl MemLeases {
        fn new(node: &str) -> Self {
            Self {
                node: node.to_string(),
                rows: StdMutex::new(HashMap::new()),
            }
        }
    }

    impl LeaseStore for MemLeases {
        async fn acquire(&self, key: &str) -> Result<Lease, LeaseError> {
            let mut rows = self.rows.lock().unwrap();
            match rows.get(key).cloned() {
                Some(l) if l.expiry > now_secs() && l.node != self.node => Err(LeaseError::Held {
                    owner: l.node,
                    expiry: l.expiry,
                }),
                Some(l) if l.expiry > now_secs() && l.node == self.node => Ok(l),
                other => {
                    let epoch = other.map(|l| l.epoch).unwrap_or(0) + 1;
                    let lease = Lease {
                        key: key.to_string(),
                        node: self.node.clone(),
                        epoch,
                        expiry: now_secs() + 30,
                    };
                    rows.insert(key.to_string(), lease.clone());
                    Ok(lease)
                }
            }
        }
        async fn renew(&self, lease: &Lease) -> Result<Lease, LeaseError> {
            let mut rows = self.rows.lock().unwrap();
            match rows.get(&lease.key) {
                Some(l) if l.node == self.node && l.epoch == lease.epoch => {
                    let fresh = Lease {
                        expiry: now_secs() + 30,
                        ..lease.clone()
                    };
                    rows.insert(lease.key.clone(), fresh.clone());
                    Ok(fresh)
                }
                _ => Err(LeaseError::Lost),
            }
        }
        async fn release(&self, lease: &Lease) -> Result<(), LeaseError> {
            self.rows.lock().unwrap().remove(&lease.key);
            Ok(())
        }
        async fn current(&self, key: &str) -> Result<Option<Lease>, LeaseError> {
            Ok(self.rows.lock().unwrap().get(key).cloned())
        }
    }

    struct MemRegistry {
        node: String,
        nodes: Vec<Node>,
    }
    impl NodeRegistry for MemRegistry {
        async fn heartbeat(&self) -> Result<(), String> {
            Ok(())
        }
        async fn live_nodes(&self) -> Result<Vec<Node>, String> {
            Ok(self.nodes.clone())
        }
        fn node_id(&self) -> &str {
            &self.node
        }
    }

    fn nodes() -> Vec<Node> {
        (0..3)
            .map(|i| Node {
                id: format!("node-{i}"),
                addr: format!("10.0.0.{i}:8080"),
            })
            .collect()
    }

    #[tokio::test]
    async fn disabled_always_resolves_local() {
        let c = Coordinator::new(
            false,
            "node-0",
            MemLeases::new("node-0"),
            MemRegistry {
                node: "node-0".into(),
                nodes: vec![],
            },
        );
        assert_eq!(
            c.resolve("acct").await.unwrap(),
            Resolution::Local { epoch: 0 }
        );
    }

    #[tokio::test]
    async fn owner_serves_local_and_takes_a_lease() {
        let ns = nodes();
        // Find an account this node owns.
        let owned = (0..1000)
            .map(|i| format!("acct-{i}"))
            .find(|a| {
                placement::owner(a, &ns.iter().map(|n| n.id.clone()).collect::<Vec<_>>())
                    == Some("node-0")
            })
            .unwrap();
        let c = Coordinator::new(
            true,
            "node-0",
            MemLeases::new("node-0"),
            MemRegistry {
                node: "node-0".into(),
                nodes: ns,
            },
        );
        c.maintain().await; // load the node set
        match c.resolve(&owned).await.unwrap() {
            Resolution::Local { epoch } => assert_eq!(epoch, 1),
            other => panic!("expected Local, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_owner_forwards_to_the_owning_node() {
        let ns = nodes();
        let ids: Vec<String> = ns.iter().map(|n| n.id.clone()).collect();
        // An account owned by some other node.
        let remote = (0..1000)
            .map(|i| format!("acct-{i}"))
            .find(|a| placement::owner(a, &ids) != Some("node-0"))
            .unwrap();
        let owner = placement::owner(&remote, &ids).unwrap().to_string();
        let want_addr = ns.iter().find(|n| n.id == owner).unwrap().addr.clone();
        let c = Coordinator::new(
            true,
            "node-0",
            MemLeases::new("node-0"),
            MemRegistry {
                node: "node-0".into(),
                nodes: ns,
            },
        );
        c.maintain().await;
        assert_eq!(
            c.resolve(&remote).await.unwrap(),
            Resolution::Remote { addr: want_addr }
        );
    }
}
