//! Gossip membership: nodes self-discover and converge on the live set with no
//! external coordinator, so a fleet self-organizes for sharding.
//!
//! This is the Cassandra/Dynamo model. Deterministic rendezvous placement (see
//! [`super::placement`]) needs only an agreed live node set to decide who owns a
//! key, and gossip is how the fleet agrees on that set without a database or a
//! consensus leader. Each round a node sends its whole view of the cluster to a
//! few peers and merges what they send back (anti-entropy push-pull). A node
//! advertises itself with a monotonic `incarnation` it bumps every round, so a
//! newer announcement always supersedes an older one; a node that stops bumping
//! for `failure_timeout` is considered gone.
//!
//! The network transport is a trait the embedder supplies (typically an HTTP POST
//! to `{peer}/cluster/gossip`) so the engine core keeps no HTTP-client dependency.
//! Seeds are a static list of peer addresses; resolving a headless DNS name to
//! that list is the embedder's job.

use std::collections::HashMap;
use std::future::Future;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

use super::{Node, NodeRegistry};

/// The number of peers contacted per gossip round. Small and constant so cost
/// does not grow with fleet size; anti-entropy still converges in O(log n) rounds.
const FANOUT: usize = 3;

/// One node's advertised state, as gossiped.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Member {
    pub id: String,
    pub addr: String,
    /// Per-node monotonic version. A higher incarnation always wins a merge, and a
    /// node bumps its own every round so peers can tell it is still alive.
    pub incarnation: u64,
    /// A tombstone: set on graceful shutdown so the departure propagates instead of
    /// waiting for the failure timeout.
    pub departed: bool,
}

/// A snapshot of a node's whole view, exchanged in one gossip round.
pub type GossipView = Vec<Member>;

/// The transport the embedder supplies: send our view to a peer and get theirs
/// back. Typically an HTTP POST to `{peer_addr}/cluster/gossip`.
pub trait GossipTransport: Send + Sync {
    fn exchange(
        &self,
        peer_addr: &str,
        ours: GossipView,
    ) -> impl Future<Output = Result<GossipView, String>> + Send;
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// What we track locally for each known node: its advertised state plus the local
/// time we last accepted a newer incarnation for it (used for failure detection).
#[derive(Clone, Debug)]
struct Entry {
    addr: String,
    incarnation: u64,
    departed: bool,
    last_update: u64,
}

/// Self-organizing membership over a gossip [`GossipTransport`]. Implements
/// [`NodeRegistry`] so a [`super::Coordinator`] can shard over it directly.
pub struct GossipMembership<T: GossipTransport> {
    id: String,
    addr: String,
    /// A node is live if the last incarnation bump we saw for it is within this
    /// many seconds.
    failure_timeout: u64,
    seeds: Vec<String>,
    members: Mutex<HashMap<String, Entry>>,
    transport: T,
}

impl<T: GossipTransport> GossipMembership<T> {
    /// `seeds` are peer base addresses to bootstrap from (resolve a headless DNS
    /// name to this list before calling). `failure_timeout` should comfortably
    /// exceed a few gossip intervals so a slow round does not evict a live node.
    pub fn new(
        id: impl Into<String>,
        addr: impl Into<String>,
        seeds: Vec<String>,
        failure_timeout: std::time::Duration,
        transport: T,
    ) -> Self {
        let id = id.into();
        let addr = addr.into();
        let mut members = HashMap::new();
        members.insert(
            id.clone(),
            Entry {
                addr: addr.clone(),
                incarnation: 0,
                departed: false,
                last_update: now_secs(),
            },
        );
        Self {
            id,
            addr,
            failure_timeout: failure_timeout.as_secs().max(1),
            seeds,
            members: Mutex::new(members),
            transport,
        }
    }

    /// Our current view as a wire snapshot.
    pub fn snapshot(&self) -> GossipView {
        self.members
            .lock()
            .iter()
            .map(|(id, e)| Member {
                id: id.clone(),
                addr: e.addr.clone(),
                incarnation: e.incarnation,
                departed: e.departed,
            })
            .collect()
    }

    /// Fold a peer's view into ours: a member with a strictly higher incarnation
    /// (or an equal incarnation that newly reports departure) wins and refreshes
    /// the failure clock. Our own entry is never overwritten by a peer.
    pub fn merge(&self, incoming: GossipView) {
        let now = now_secs();
        let mut members = self.members.lock();
        for m in incoming {
            if m.id == self.id {
                continue; // only we author our own state
            }
            match members.get_mut(&m.id) {
                Some(e) if m.incarnation > e.incarnation => {
                    e.addr = m.addr;
                    e.incarnation = m.incarnation;
                    e.departed = m.departed;
                    e.last_update = now;
                }
                Some(e) if m.incarnation == e.incarnation && m.departed && !e.departed => {
                    e.departed = true;
                    e.last_update = now;
                }
                Some(_) => {}
                None => {
                    members.insert(
                        m.id.clone(),
                        Entry {
                            addr: m.addr,
                            incarnation: m.incarnation,
                            departed: m.departed,
                            last_update: now,
                        },
                    );
                }
            }
        }
    }

    /// Advertise a fresh incarnation for ourselves so peers see us as alive.
    fn bump_self(&self) {
        let now = now_secs();
        let mut members = self.members.lock();
        let e = members.entry(self.id.clone()).or_insert(Entry {
            addr: self.addr.clone(),
            incarnation: 0,
            departed: false,
            last_update: now,
        });
        e.incarnation += 1;
        e.addr = self.addr.clone();
        e.last_update = now;
    }

    /// Peers to gossip with this round: live members plus seeds, minus ourselves,
    /// capped at [`FANOUT`].
    fn targets(&self) -> Vec<String> {
        let now = now_secs();
        let mut addrs: Vec<String> = {
            let members = self.members.lock();
            members
                .values()
                .filter(|e| !e.departed && now.saturating_sub(e.last_update) < self.failure_timeout)
                .map(|e| e.addr.clone())
                .filter(|a| *a != self.addr && !a.is_empty())
                .collect()
        };
        for s in &self.seeds {
            if *s != self.addr && !addrs.contains(s) {
                addrs.push(s.clone());
            }
        }
        addrs.truncate(FANOUT);
        addrs
    }

    /// One gossip round: advertise ourselves, then push-pull with a few peers.
    pub async fn round(&self) {
        self.bump_self();
        for addr in self.targets() {
            let ours = self.snapshot();
            match self.transport.exchange(&addr, ours).await {
                Ok(theirs) => self.merge(theirs),
                Err(e) => tracing::debug!(peer = %addr, error = %e, "gossip: exchange failed"),
            }
        }
    }

    /// Mark ourselves departed and gossip once, so the fleet re-homes our keys
    /// immediately on graceful shutdown instead of after the failure timeout.
    pub async fn leave(&self) {
        {
            let mut members = self.members.lock();
            if let Some(e) = members.get_mut(&self.id) {
                e.incarnation += 1;
                e.departed = true;
                e.last_update = now_secs();
            }
        }
        for addr in self.targets() {
            let _ = self.transport.exchange(&addr, self.snapshot()).await;
        }
    }
}

impl<T: GossipTransport> NodeRegistry for GossipMembership<T> {
    async fn heartbeat(&self) -> Result<(), String> {
        self.round().await;
        Ok(())
    }

    async fn live_nodes(&self) -> Result<Vec<Node>, String> {
        let now = now_secs();
        let members = self.members.lock();
        Ok(members
            .iter()
            .filter(|(_, e)| {
                !e.departed && now.saturating_sub(e.last_update) < self.failure_timeout
            })
            .map(|(id, e)| Node {
                id: id.clone(),
                addr: e.addr.clone(),
            })
            .collect())
    }

    fn node_id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as Map;
    use std::sync::{Arc, Weak};
    use std::time::Duration;

    /// The mergeable half of a node, type-erased so the [`Bus`] can hold peers
    /// without a circular `GossipMembership<Bus>` type.
    trait Peer: Send + Sync {
        fn peer_merge(&self, v: GossipView);
        fn peer_snapshot(&self) -> GossipView;
    }
    impl<T: GossipTransport> Peer for GossipMembership<T> {
        fn peer_merge(&self, v: GossipView) {
            self.merge(v);
        }
        fn peer_snapshot(&self) -> GossipView {
            self.snapshot()
        }
    }

    /// In-process transport modelling the real HTTP handler: `exchange` reaches the
    /// target node, has it merge our push, and returns its snapshot. `Weak` refs
    /// avoid an ownership cycle (each node owns a clone of the bus). Dropping a
    /// node's `Arc` models it going down (the `Weak` fails to upgrade).
    #[derive(Clone, Default)]
    struct Bus {
        peers: Arc<Mutex<Map<String, Weak<dyn Peer>>>>,
    }
    impl Bus {
        fn register(&self, addr: &str, peer: Weak<dyn Peer>) {
            self.peers.lock().insert(addr.to_string(), peer);
        }
    }
    impl GossipTransport for Bus {
        async fn exchange(&self, peer_addr: &str, ours: GossipView) -> Result<GossipView, String> {
            let peer = self.peers.lock().get(peer_addr).and_then(Weak::upgrade);
            match peer {
                Some(p) => {
                    p.peer_merge(ours);
                    Ok(p.peer_snapshot())
                }
                None => Err(format!("peer {peer_addr} unreachable")),
            }
        }
    }

    fn node(bus: &Bus, id: &str, seeds: Vec<String>) -> Arc<GossipMembership<Bus>> {
        let addr = format!("http://{id}:6677");
        let m = Arc::new(GossipMembership::new(
            id,
            addr.clone(),
            seeds,
            Duration::from_secs(30),
            bus.clone(),
        ));
        // The returned `m` keeps the allocation alive, so this Weak upgrades for
        // the test's duration; dropping `m` would model the node going down.
        let dynp: Arc<dyn Peer> = m.clone();
        bus.register(&addr, Arc::downgrade(&dynp));
        m
    }

    async fn ids(m: &GossipMembership<Bus>) -> Vec<String> {
        let mut v: Vec<String> = m
            .live_nodes()
            .await
            .unwrap()
            .into_iter()
            .map(|n| n.id)
            .collect();
        v.sort();
        v
    }

    #[tokio::test]
    async fn solo_node_is_its_own_member() {
        let bus = Bus::default();
        let a = node(&bus, "a", vec![]);
        assert_eq!(ids(&a).await, vec!["a"]);
    }

    #[tokio::test]
    async fn three_nodes_converge_through_a_seed() {
        let bus = Bus::default();
        // b and c only know the seed a; a knows no one. Everyone must still learn
        // the full set purely by gossiping.
        let seed = "http://a:6677".to_string();
        let a = node(&bus, "a", vec![]);
        let b = node(&bus, "b", vec![seed.clone()]);
        let c = node(&bus, "c", vec![seed.clone()]);

        for _ in 0..6 {
            a.round().await;
            b.round().await;
            c.round().await;
        }

        assert_eq!(ids(&a).await, vec!["a", "b", "c"]);
        assert_eq!(ids(&b).await, vec!["a", "b", "c"]);
        assert_eq!(ids(&c).await, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn a_departed_node_leaves_the_live_set() {
        let bus = Bus::default();
        let a = node(&bus, "a", vec![]);
        let b = node(&bus, "b", vec!["http://a:6677".to_string()]);
        for _ in 0..4 {
            a.round().await;
            b.round().await;
        }
        assert_eq!(ids(&a).await, vec!["a", "b"]);

        // b departs gracefully; its tombstone reaches a, which drops it.
        b.leave().await;
        a.round().await;
        assert_eq!(ids(&a).await, vec!["a"]);
    }

    #[tokio::test]
    async fn higher_incarnation_wins_a_merge() {
        let bus = Bus::default();
        let a = node(&bus, "a", vec![]);
        // Learn b at addr1, then a newer incarnation moves b to addr2.
        a.merge(vec![Member {
            id: "b".into(),
            addr: "http://old:1".into(),
            incarnation: 1,
            departed: false,
        }]);
        a.merge(vec![Member {
            id: "b".into(),
            addr: "http://new:2".into(),
            incarnation: 2,
            departed: false,
        }]);
        // A stale lower incarnation must not revert the address.
        a.merge(vec![Member {
            id: "b".into(),
            addr: "http://stale:0".into(),
            incarnation: 1,
            departed: false,
        }]);
        let addr = a
            .live_nodes()
            .await
            .unwrap()
            .into_iter()
            .find(|n| n.id == "b")
            .unwrap()
            .addr;
        assert_eq!(addr, "http://new:2");
    }
}
