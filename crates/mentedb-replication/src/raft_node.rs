//! Raft node wrapper with MenteDB-specific types.

use std::fmt;
use std::io::Cursor;

use openraft::async_runtime::watch::WatchReceiver;
use openraft::{BasicNode, Raft};
use serde::{Deserialize, Serialize};

use crate::config::ReplicationConfig;
use crate::network::MenteNetworkFactory;
use crate::store::{LogStore, StateMachine};

// Type configuration for openraft parameterized with MenteDB types.
openraft::declare_raft_types!(
    pub TypeConfig:
        D = MenteRequest,
        R = MenteResponse,
        Node = BasicNode,
        NodeId = u64,
        SnapshotData = Cursor<Vec<u8>>,
);

/// A request that can be replicated through Raft consensus.
///
/// Each variant corresponds to a write operation on MenteDB state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MenteRequest {
    /// Store a new memory with the given content and tags.
    StoreMemory {
        memory_id: String,
        content: String,
        tags: Vec<String>,
    },
    /// Remove a memory by ID.
    ForgetMemory { memory_id: String },
    /// Create a typed relationship between two memories.
    RelateMemories {
        source_id: String,
        target_id: String,
        relation_type: String,
        weight: f32,
    },
    /// Update the content or tags of an existing memory.
    UpdateMemory {
        memory_id: String,
        content: Option<String>,
        tags: Option<Vec<String>>,
    },
}

impl fmt::Display for MenteRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StoreMemory { memory_id, .. } => write!(f, "StoreMemory({})", memory_id),
            Self::ForgetMemory { memory_id } => write!(f, "ForgetMemory({})", memory_id),
            Self::RelateMemories {
                source_id,
                target_id,
                ..
            } => write!(f, "RelateMemories({} -> {})", source_id, target_id),
            Self::UpdateMemory { memory_id, .. } => write!(f, "UpdateMemory({})", memory_id),
        }
    }
}

/// A response returned after applying a Raft-replicated request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MenteResponse {
    /// A memory was successfully stored.
    MemoryStored { memory_id: String },
    /// A memory was successfully removed.
    MemoryForgotten { memory_id: String },
    /// Two memories were successfully related.
    MemoriesRelated {
        source_id: String,
        target_id: String,
    },
    /// A memory was successfully updated.
    MemoryUpdated { memory_id: String },
    /// The operation failed with the given reason.
    Error { message: String },
}

impl Default for MenteResponse {
    fn default() -> Self {
        Self::Error {
            message: "no-op".to_string(),
        }
    }
}

/// The MenteDB Raft node, wrapping an openraft Raft instance.
pub struct MenteRaftNode {
    pub raft: Raft<TypeConfig, StateMachine>,
    pub config: ReplicationConfig,
}

impl MenteRaftNode {
    /// Create a new Raft node with the given configuration, log store,
    /// state machine, and network factory.
    pub async fn new(
        config: ReplicationConfig,
        log_store: LogStore,
        state_machine: StateMachine,
        network: MenteNetworkFactory,
    ) -> Result<Self, openraft::error::Fatal<TypeConfig>> {
        let raft_config = openraft::Config {
            heartbeat_interval: config.heartbeat_interval_ms,
            election_timeout_min: config.election_timeout_min_ms,
            election_timeout_max: config.election_timeout_max_ms,
            snapshot_policy: openraft::SnapshotPolicy::LogsSinceLast(
                config.snapshot_threshold as u64,
            ),
            ..Default::default()
        };

        let raft_config = raft_config.validate().expect("invalid raft config");
        let raft_config = std::sync::Arc::new(raft_config);

        let raft = Raft::new(
            config.node_id,
            raft_config,
            network,
            log_store,
            state_machine,
        )
        .await?;

        Ok(Self { raft, config })
    }

    /// Submit a client write request through Raft consensus.
    pub async fn client_write(
        &self,
        request: MenteRequest,
    ) -> Result<MenteResponse, Box<dyn std::error::Error>> {
        let response = self.raft.client_write(request).await?;
        Ok(response.response().clone())
    }

    /// Returns the current Raft metrics (leader, term, membership, etc.).
    pub fn metrics(&self) -> openraft::metrics::RaftMetrics<TypeConfig> {
        self.raft.metrics().borrow_watched().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() {
        let req = MenteRequest::StoreMemory {
            memory_id: "mem-1".to_string(),
            content: "test content".to_string(),
            tags: vec!["tag1".to_string()],
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: MenteRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, deserialized);
    }

    #[test]
    fn test_response_serialization() {
        let resp = MenteResponse::MemoryStored {
            memory_id: "mem-1".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: MenteResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, deserialized);
    }

    #[test]
    fn test_request_display() {
        let req = MenteRequest::StoreMemory {
            memory_id: "m1".into(),
            content: "c".into(),
            tags: vec![],
        };
        assert_eq!(format!("{}", req), "StoreMemory(m1)");
    }

    #[test]
    fn test_all_request_variants() {
        let variants: Vec<MenteRequest> = vec![
            MenteRequest::StoreMemory {
                memory_id: "m1".into(),
                content: "c".into(),
                tags: vec![],
            },
            MenteRequest::ForgetMemory {
                memory_id: "m2".into(),
            },
            MenteRequest::RelateMemories {
                source_id: "m1".into(),
                target_id: "m2".into(),
                relation_type: "related".into(),
                weight: 0.5,
            },
            MenteRequest::UpdateMemory {
                memory_id: "m1".into(),
                content: Some("new".into()),
                tags: None,
            },
        ];

        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: MenteRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(v, &back);
        }
    }

    #[test]
    fn test_all_response_variants() {
        let variants: Vec<MenteResponse> = vec![
            MenteResponse::MemoryStored {
                memory_id: "m1".into(),
            },
            MenteResponse::MemoryForgotten {
                memory_id: "m2".into(),
            },
            MenteResponse::MemoriesRelated {
                source_id: "m1".into(),
                target_id: "m2".into(),
            },
            MenteResponse::MemoryUpdated {
                memory_id: "m1".into(),
            },
            MenteResponse::Error {
                message: "fail".into(),
            },
        ];

        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: MenteResponse = serde_json::from_str(&json).unwrap();
            assert_eq!(v, &back);
        }
    }
}
