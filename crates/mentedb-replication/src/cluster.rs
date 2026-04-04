//! Cluster management for MenteDB Raft replication.
//!
//! Provides `MenteCluster` for managing cluster membership, node discovery,
//! and leader forwarding.

use std::collections::BTreeMap;

use openraft::BasicNode;
use tracing::info;

use crate::config::ReplicationConfig;
use crate::network::MenteNetworkFactory;
use crate::raft_node::{MenteRaftNode, MenteRequest, MenteResponse};
use crate::store::{LogStore, StateMachine};

/// Manages a MenteDB Raft cluster including membership and leader forwarding.
pub struct MenteCluster {
    node: MenteRaftNode,
}

impl MenteCluster {
    /// Initialize a new cluster with a single node and bootstrap it as leader.
    pub async fn bootstrap_single(
        config: ReplicationConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let log_store = LogStore::new();
        let state_machine = StateMachine::new();
        let network = MenteNetworkFactory::new();

        let node_id = config.node_id;
        let node = MenteRaftNode::new(config, log_store, state_machine, network).await?;

        // Bootstrap with this node as the only member.
        let mut members = BTreeMap::new();
        members.insert(
            node_id,
            BasicNode {
                addr: "http://127.0.0.1:0".to_string(),
            },
        );
        node.raft.initialize(members).await?;

        info!(node_id, "bootstrapped single-node cluster");

        Ok(Self { node })
    }

    /// Create a cluster node without bootstrapping (joins an existing cluster).
    pub async fn new_node(config: ReplicationConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let log_store = LogStore::new();
        let state_machine = StateMachine::new();
        let network = MenteNetworkFactory::new();

        let node = MenteRaftNode::new(config, log_store, state_machine, network).await?;

        Ok(Self { node })
    }

    /// Submit a write request to the cluster leader.
    ///
    /// If this node is the leader, the request is handled directly.
    /// Otherwise, the caller should forward to the known leader.
    pub async fn write(
        &self,
        request: MenteRequest,
    ) -> Result<MenteResponse, Box<dyn std::error::Error>> {
        let response = self.node.client_write(request).await?;
        Ok(response)
    }

    /// Returns the current leader's node ID, if known.
    pub fn current_leader(&self) -> Option<u64> {
        self.node.metrics().current_leader
    }

    /// Returns whether this node believes it is the leader.
    pub fn is_leader(&self) -> bool {
        let metrics = self.node.metrics();
        metrics.current_leader == Some(self.node.config.node_id)
    }

    /// Returns the node ID of this cluster member.
    pub fn node_id(&self) -> u64 {
        self.node.config.node_id
    }

    /// Request the Raft node to add a new learner to the cluster.
    pub async fn add_learner(
        &self,
        node_id: u64,
        addr: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node = BasicNode { addr };
        self.node.raft.add_learner(node_id, node, true).await?;
        info!(node_id, "added learner to cluster");
        Ok(())
    }

    /// Promote learners to voters by changing membership.
    pub async fn change_membership(
        &self,
        member_ids: Vec<u64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let members: BTreeMap<u64, BasicNode> = member_ids
            .into_iter()
            .map(|id| {
                (
                    id,
                    BasicNode {
                        addr: String::new(),
                    },
                )
            })
            .collect();
        self.node.raft.change_membership(members, false).await?;
        info!("cluster membership changed");
        Ok(())
    }

    /// Get a reference to the underlying Raft node.
    pub fn raft_node(&self) -> &MenteRaftNode {
        &self.node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bootstrap_single_node() {
        let config = ReplicationConfig::single_node(1);
        let cluster = MenteCluster::bootstrap_single(config).await.unwrap();

        // After bootstrap, this node should become leader.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        assert_eq!(cluster.node_id(), 1);
        assert!(cluster.is_leader());
        assert_eq!(cluster.current_leader(), Some(1));
    }

    #[tokio::test]
    async fn test_single_node_write_read() {
        let config = ReplicationConfig::single_node(1);
        let cluster = MenteCluster::bootstrap_single(config).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let resp = cluster
            .write(MenteRequest::StoreMemory {
                memory_id: "test-mem-1".into(),
                content: "a test memory for raft replication".into(),
                tags: vec!["test".into(), "raft".into()],
            })
            .await
            .unwrap();

        assert_eq!(
            resp,
            MenteResponse::MemoryStored {
                memory_id: "test-mem-1".into()
            }
        );
    }

    #[tokio::test]
    async fn test_single_node_forget() {
        let config = ReplicationConfig::single_node(1);
        let cluster = MenteCluster::bootstrap_single(config).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        cluster
            .write(MenteRequest::StoreMemory {
                memory_id: "ephemeral".into(),
                content: "temporary".into(),
                tags: vec![],
            })
            .await
            .unwrap();

        let resp = cluster
            .write(MenteRequest::ForgetMemory {
                memory_id: "ephemeral".into(),
            })
            .await
            .unwrap();

        assert_eq!(
            resp,
            MenteResponse::MemoryForgotten {
                memory_id: "ephemeral".into()
            }
        );
    }

    #[tokio::test]
    async fn test_single_node_relate_memories() {
        let config = ReplicationConfig::single_node(1);
        let cluster = MenteCluster::bootstrap_single(config).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let resp = cluster
            .write(MenteRequest::RelateMemories {
                source_id: "m1".into(),
                target_id: "m2".into(),
                relation_type: "supports".into(),
                weight: 0.9,
            })
            .await
            .unwrap();

        assert_eq!(
            resp,
            MenteResponse::MemoriesRelated {
                source_id: "m1".into(),
                target_id: "m2".into(),
            }
        );
    }
}
