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
    /// Whether every node sees the same leases. Only a shared store can say
    /// which node holds a key, so only then does the coordinator route by lease
    /// holder; a node-local store routes by placement alone.
    fn is_shared(&self) -> bool {
        true
    }
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
    fn is_shared(&self) -> bool {
        false
    }
}

/// A backend that tracks the live node set. Implemented by the embedder.
pub trait NodeRegistry: Send + Sync {
    fn heartbeat(&self) -> impl Future<Output = Result<(), String>> + Send;
    fn live_nodes(&self) -> impl Future<Output = Result<Vec<Node>, String>> + Send;
    fn node_id(&self) -> &str;
}

fn addr_of(nodes: &[Node], id: &str) -> Option<String> {
    nodes.iter().find(|n| n.id == id).map(|n| n.addr.clone())
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
    /// Short-lived view of which node holds another node's lease, so routing a
    /// request we do not serve costs a lease read at most every few seconds per
    /// key instead of on every request. Valid until min(lease expiry, read + 5s).
    holders: Mutex<HashMap<String, (String, u64)>>,
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
            holders: Mutex::new(HashMap::new()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Decide where `key` is served.
    ///
    /// The lease holder serves, not the placement math. Placement only picks who
    /// takes a key that nobody holds. Once a node holds a valid lease it keeps
    /// serving that key even when the live set changes, and every other node
    /// forwards to it. Without this, a scale out moved placement to the new node
    /// while the old one still held the lease and the open database: the new node
    /// was refused the lease, fell back to opening the database itself, and the
    /// account hung on the single-writer lock until the old node let go, which it
    /// never did while it kept renewing. A key moves only when its holder releases
    /// it or dies and the lease lapses.
    ///
    /// When disabled, resolves `Local` without contacting the lease store.
    pub async fn resolve(&self, key: &str) -> Result<Resolution, LeaseError> {
        if !self.enabled {
            return Ok(Resolution::Local { epoch: 0 });
        }
        if !self.leases.is_shared() {
            return self.resolve_by_placement(key).await;
        }
        // Sticky: a lease we hold stays ours while it is valid. Clone out of the
        // guard so no lock is held across an await.
        let cached = self.held.lock().get(key).cloned();
        if let Some(l) = cached
            && l.expiry > now_secs() + 5
        {
            return Ok(Resolution::Local { epoch: l.epoch });
        }

        let nodes = self.nodes.lock().clone();
        let ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
        let owner = placement::owner(key, &ids)
            .map(str::to_string)
            .unwrap_or_else(|| self.node.clone());

        if owner == self.node {
            return match self.leases.acquire(key).await {
                Ok(lease) => {
                    let epoch = lease.epoch;
                    self.held.lock().insert(key.to_string(), lease);
                    Ok(Resolution::Local { epoch })
                }
                // Another live node still holds it (we were just placed here by
                // a membership change): it keeps serving until it lets go.
                Err(LeaseError::Held {
                    owner: holder,
                    expiry,
                }) => {
                    self.remember_holder(key, &holder, expiry);
                    match addr_of(&nodes, &holder) {
                        Some(addr) => Ok(Resolution::Remote { addr }),
                        None => Err(LeaseError::Held {
                            owner: holder,
                            expiry,
                        }),
                    }
                }
                Err(e) => Err(e),
            };
        }

        // Not ours by placement: forward to whoever actually holds the lease, and
        // only to the placement owner when the key is free.
        if let Some(holder) = self.holder(key).await {
            if holder == self.node {
                let lease = self.leases.acquire(key).await?;
                let epoch = lease.epoch;
                self.held.lock().insert(key.to_string(), lease);
                return Ok(Resolution::Local { epoch });
            }
            if let Some(addr) = addr_of(&nodes, &holder) {
                return Ok(Resolution::Remote { addr });
            }
            // The holder is not live; its lease lapses within the TTL. Fall
            // through to the placement owner, which takes it over.
        }
        let addr = addr_of(&nodes, &owner)
            .ok_or_else(|| LeaseError::Backend(format!("owner {owner} has no address")))?;
        Ok(Resolution::Remote { addr })
    }

    /// Whether this node should do background work (sweeps, migrations) on `key`:
    /// it holds the key's lease, or nobody does and placement names it. The same
    /// rule [`resolve`](Self::resolve) serves by, so background work never opens
    /// an account another node is serving. True when sharding is disabled or the
    /// node set is not yet known.
    pub async fn serves(&self, key: &str) -> bool {
        if !self.enabled {
            return true;
        }
        if !self.leases.is_shared() {
            return self.owns(key);
        }
        let cached = self.held.lock().get(key).cloned();
        if let Some(l) = cached
            && l.expiry > now_secs()
        {
            return true;
        }
        if self.nodes.lock().is_empty() {
            return true;
        }
        match self.holder(key).await {
            Some(holder) => holder == self.node,
            None => self.owns(key),
        }
    }

    /// Placement-only routing for a node-local lease store (gossip fleets), where
    /// no node can see another's leases: the placement owner serves, fenced by its
    /// local epoch, and the single-writer file lock is the safety net in handoff.
    async fn resolve_by_placement(&self, key: &str) -> Result<Resolution, LeaseError> {
        let nodes = self.nodes.lock().clone();
        let ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
        let owner = placement::owner(key, &ids)
            .map(str::to_string)
            .unwrap_or_else(|| self.node.clone());
        if owner == self.node {
            let lease = self.leases.acquire(key).await?;
            let epoch = lease.epoch;
            self.held.lock().insert(key.to_string(), lease);
            return Ok(Resolution::Local { epoch });
        }
        let addr = addr_of(&nodes, &owner)
            .ok_or_else(|| LeaseError::Backend(format!("owner {owner} has no address")))?;
        Ok(Resolution::Remote { addr })
    }

    /// The node holding a valid lease on `key`, if any, through the short-lived
    /// holder cache. A lease backend error reads as "no holder" so callers fall
    /// back to placement, the behavior before leases were consulted here.
    async fn holder(&self, key: &str) -> Option<String> {
        let now = now_secs();
        if let Some((node, valid_until)) = self.holders.lock().get(key).cloned()
            && valid_until > now
        {
            return Some(node);
        }
        match self.leases.current(key).await {
            Ok(Some(l)) if l.expiry > now => {
                self.remember_holder(key, &l.node, l.expiry);
                Some(l.node)
            }
            _ => {
                self.holders.lock().remove(key);
                None
            }
        }
    }

    fn remember_holder(&self, key: &str, node: &str, expiry: u64) {
        let valid_until = expiry.min(now_secs() + 5);
        self.holders
            .lock()
            .insert(key.to_string(), (node.to_string(), valid_until));
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

    /// Whether this node is the placement owner of `key` under the current live
    /// node set, WITHOUT acquiring a lease. A cheap, read-only check for
    /// background work (maintenance sweeps) that should run only on the node
    /// already serving an account, so a non-owner never opens it and contends for
    /// its single-writer lock. Unlike [`resolve`](Self::resolve) it has no lease
    /// side effect, so a sweep can test every account without pinning leases it
    /// would then have to renew. Returns true when sharding is disabled or the
    /// node set is not yet known (a lone or just-started node still does the
    /// work), so a single-node fleet sweeps everything exactly as before.
    pub fn owns(&self, key: &str) -> bool {
        if !self.enabled {
            return true;
        }
        let nodes = self.nodes.lock();
        if nodes.is_empty() {
            return true;
        }
        let ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
        placement::owner(key, &ids)
            .map(|owner| owner == self.node)
            .unwrap_or(true)
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
        rows: std::sync::Arc<StdMutex<HashMap<String, Lease>>>,
    }

    impl MemLeases {
        fn new(node: &str) -> Self {
            Self {
                node: node.to_string(),
                rows: Default::default(),
            }
        }
        /// A node's view of a lease table shared by the whole fleet, the way
        /// every gateway task sees the same DynamoDB table.
        fn shared(node: &str, rows: &std::sync::Arc<StdMutex<HashMap<String, Lease>>>) -> Self {
            Self {
                node: node.to_string(),
                rows: rows.clone(),
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

    #[tokio::test]
    async fn owns_partitions_keys_across_the_fleet_without_leasing() {
        let ns = nodes(); // node-0..2, same live set for every coordinator
        let ids: Vec<String> = ns.iter().map(|n| n.id.clone()).collect();

        let mut coords = Vec::new();
        for n in &ns {
            let c = Coordinator::new(
                true,
                n.id.clone(),
                MemLeases::new(&n.id),
                MemRegistry {
                    node: n.id.clone(),
                    nodes: ns.clone(),
                },
            );
            c.maintain().await; // load the node set
            coords.push((n.id.clone(), c));
        }

        for i in 0..500 {
            let key = format!("acct-{i}");
            let placement_owner = placement::owner(&key, &ids).unwrap();
            let owners: Vec<&String> = coords
                .iter()
                .filter(|(_, c)| c.owns(&key))
                .map(|(id, _)| id)
                .collect();
            // Exactly one node owns each key, and it is the placement owner: the
            // sweep on every other node skips it, so no two nodes open it at once.
            assert_eq!(owners.len(), 1, "key {key} owned by {owners:?}");
            assert_eq!(owners[0], placement_owner);
        }

        // owns() must not acquire a lease (that is resolve's job): no held leases.
        for (_, c) in &coords {
            assert!(c.held.lock().is_empty(), "owns() must not take a lease");
        }
    }

    #[tokio::test]
    async fn disabled_owns_everything() {
        let c = Coordinator::new(
            false,
            "node-0",
            MemLeases::new("node-0"),
            MemRegistry {
                node: "node-0".into(),
                nodes: vec![],
            },
        );
        assert!(c.owns("anything"));
    }

    fn fleet(n: usize) -> Vec<Node> {
        (0..n)
            .map(|i| Node {
                id: format!("node-{i}"),
                addr: format!("10.0.0.{i}:8080"),
            })
            .collect()
    }

    fn ids(ns: &[Node]) -> Vec<String> {
        ns.iter().map(|n| n.id.clone()).collect()
    }

    /// A key placed on node-0 in a two-node fleet that placement moves to node-2
    /// once node-2 joins: exactly the scale-out that stranded an account.
    fn key_moved_by_scale_out() -> String {
        (0..10_000)
            .map(|i| format!("acct-{i}"))
            .find(|k| {
                placement::owner(k, &ids(&fleet(2))) == Some("node-0")
                    && placement::owner(k, &ids(&fleet(3))) == Some("node-2")
            })
            .expect("some key moves to the new node")
    }

    async fn coordinator(
        id: &str,
        live: Vec<Node>,
        rows: &std::sync::Arc<StdMutex<HashMap<String, Lease>>>,
    ) -> Coordinator<MemLeases, MemRegistry> {
        let c = Coordinator::new(
            true,
            id,
            MemLeases::shared(id, rows),
            MemRegistry {
                node: id.into(),
                nodes: live,
            },
        );
        c.maintain().await;
        c
    }

    #[tokio::test]
    async fn scale_out_keeps_a_held_key_on_its_holder() {
        let key = key_moved_by_scale_out();
        let rows = Default::default();

        // Before the scale out, node-0 serves the key and holds its lease.
        let n0 = coordinator("node-0", fleet(2), &rows).await;
        assert_eq!(
            n0.resolve(&key).await.unwrap(),
            Resolution::Local { epoch: 1 }
        );

        // node-2 joins. Placement now names node-2, but node-0 still holds the
        // lease (and the open database), so it keeps serving and everyone else
        // forwards to it instead of opening the database themselves.
        let n1 = coordinator("node-1", fleet(3), &rows).await;
        let n2 = coordinator("node-2", fleet(3), &rows).await;
        let to_holder = Resolution::Remote {
            addr: "10.0.0.0:8080".into(),
        };
        assert_eq!(n2.resolve(&key).await.unwrap(), to_holder);
        assert_eq!(n1.resolve(&key).await.unwrap(), to_holder);
        assert_eq!(
            n0.resolve(&key).await.unwrap(),
            Resolution::Local { epoch: 1 }
        );

        // Background work follows the same rule: only the holder sweeps it.
        assert!(n0.serves(&key).await);
        assert!(!n1.serves(&key).await);
        assert!(!n2.serves(&key).await);
    }

    #[tokio::test]
    async fn released_key_moves_to_its_new_placement_owner() {
        let key = key_moved_by_scale_out();
        let rows: std::sync::Arc<StdMutex<HashMap<String, Lease>>> = Default::default();
        let n0 = coordinator("node-0", fleet(3), &rows).await;
        let n2 = coordinator("node-2", fleet(3), &rows).await;
        // node-0 took the key while it was the placement owner.
        let lease = n0.leases.acquire(&key).await.unwrap();
        n0.held.lock().insert(key.clone(), lease.clone());

        // node-0 lets go (shutdown or rebalance). Its next upkeep drops the lease,
        // and the placement owner takes over with a higher epoch.
        n0.leases.release(&lease).await.unwrap();
        n0.maintain().await;
        n2.holders.lock().clear();
        assert_eq!(
            n2.resolve(&key).await.unwrap(),
            Resolution::Local { epoch: 1 }
        );
        n0.holders.lock().clear();
        assert_eq!(
            n0.resolve(&key).await.unwrap(),
            Resolution::Remote {
                addr: "10.0.0.2:8080".into()
            }
        );
        assert!(n2.serves(&key).await);
        assert!(!n0.serves(&key).await);
    }

    #[tokio::test]
    async fn free_key_goes_to_its_placement_owner() {
        let key = key_moved_by_scale_out();
        let rows = Default::default();
        let n0 = coordinator("node-0", fleet(3), &rows).await;
        let n2 = coordinator("node-2", fleet(3), &rows).await;
        // Nobody holds it: a non-owner forwards to placement, which takes it.
        assert_eq!(
            n0.resolve(&key).await.unwrap(),
            Resolution::Remote {
                addr: "10.0.0.2:8080".into()
            }
        );
        assert_eq!(
            n2.resolve(&key).await.unwrap(),
            Resolution::Local { epoch: 1 }
        );
        assert!(n2.serves(&key).await);
        assert!(!n0.serves(&key).await);
    }

    #[tokio::test]
    async fn node_local_leases_route_by_placement_alone() {
        // Gossip fleets have no shared lease view, so nothing can be sticky: the
        // placement owner serves, as before.
        let key = key_moved_by_scale_out();
        let n2 = Coordinator::new(
            true,
            "node-2",
            NoCoordLease::new("node-2"),
            MemRegistry {
                node: "node-2".into(),
                nodes: fleet(3),
            },
        );
        n2.maintain().await;
        assert_eq!(
            n2.resolve(&key).await.unwrap(),
            Resolution::Local { epoch: 1 }
        );
        let n0 = Coordinator::new(
            true,
            "node-0",
            NoCoordLease::new("node-0"),
            MemRegistry {
                node: "node-0".into(),
                nodes: fleet(3),
            },
        );
        n0.maintain().await;
        // Even after node-0 served it locally once, placement wins.
        n0.leases.acquire(&key).await.unwrap();
        assert_eq!(
            n0.resolve(&key).await.unwrap(),
            Resolution::Remote {
                addr: "10.0.0.2:8080".into()
            }
        );
        assert!(!n0.serves(&key).await);
    }
}
