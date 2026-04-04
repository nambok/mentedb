//! Configuration for the MenteDB replication layer.

use serde::{Deserialize, Serialize};

/// Configuration for the distributed replication cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// Unique identifier for this cluster.
    pub cluster_id: String,
    /// Unique numeric ID for this node within the cluster.
    pub node_id: u64,
    /// HTTP addresses of peer nodes, keyed by node ID.
    pub peer_addresses: Vec<PeerAddress>,
    /// Heartbeat interval in milliseconds.
    pub heartbeat_interval_ms: u64,
    /// Minimum election timeout in milliseconds.
    pub election_timeout_min_ms: u64,
    /// Maximum election timeout in milliseconds.
    pub election_timeout_max_ms: u64,
    /// Number of committed log entries before triggering a snapshot.
    pub snapshot_threshold: u32,
}

/// Address entry for a peer node in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeerAddress {
    pub node_id: u64,
    pub addr: String,
}

fn default_heartbeat_interval_ms() -> u64 {
    500
}

fn default_election_timeout_min_ms() -> u64 {
    1500
}

fn default_election_timeout_max_ms() -> u64 {
    3000
}

fn default_snapshot_threshold() -> u32 {
    1000
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            cluster_id: "mentedb-cluster".to_string(),
            node_id: 0,
            peer_addresses: Vec::new(),
            heartbeat_interval_ms: default_heartbeat_interval_ms(),
            election_timeout_min_ms: default_election_timeout_min_ms(),
            election_timeout_max_ms: default_election_timeout_max_ms(),
            snapshot_threshold: default_snapshot_threshold(),
        }
    }
}

impl ReplicationConfig {
    /// Create a minimal single-node configuration for testing.
    pub fn single_node(node_id: u64) -> Self {
        Self {
            cluster_id: "mentedb-test".to_string(),
            node_id,
            peer_addresses: vec![],
            ..Default::default()
        }
    }

    /// Validate the configuration, returning an error message if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.cluster_id.is_empty() {
            return Err("cluster_id must not be empty".to_string());
        }
        if self.heartbeat_interval_ms == 0 {
            return Err("heartbeat_interval_ms must be > 0".to_string());
        }
        if self.election_timeout_min_ms <= self.heartbeat_interval_ms {
            return Err(
                "election_timeout_min_ms must be greater than heartbeat_interval_ms".to_string(),
            );
        }
        if self.election_timeout_max_ms <= self.election_timeout_min_ms {
            return Err(
                "election_timeout_max_ms must be greater than election_timeout_min_ms".to_string(),
            );
        }
        if self.snapshot_threshold == 0 {
            return Err("snapshot_threshold must be > 0".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let config = ReplicationConfig::default();
        assert_eq!(config.cluster_id, "mentedb-cluster");
        assert_eq!(config.node_id, 0);
        assert!(config.peer_addresses.is_empty());
        assert_eq!(config.heartbeat_interval_ms, 500);
        assert_eq!(config.election_timeout_min_ms, 1500);
        assert_eq!(config.election_timeout_max_ms, 3000);
        assert_eq!(config.snapshot_threshold, 1000);
    }

    #[test]
    fn test_single_node_config() {
        let config = ReplicationConfig::single_node(1);
        assert_eq!(config.node_id, 1);
        assert_eq!(config.cluster_id, "mentedb-test");
        assert!(config.peer_addresses.is_empty());
    }

    #[test]
    fn test_validate_defaults() {
        let config = ReplicationConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_cluster_id() {
        let mut config = ReplicationConfig::default();
        config.cluster_id = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_zero_heartbeat() {
        let mut config = ReplicationConfig::default();
        config.heartbeat_interval_ms = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_election_timeout_order() {
        let mut config = ReplicationConfig::default();
        config.election_timeout_min_ms = 100;
        config.election_timeout_max_ms = 50;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_election_less_than_heartbeat() {
        let mut config = ReplicationConfig::default();
        config.heartbeat_interval_ms = 500;
        config.election_timeout_min_ms = 200;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_zero_snapshot_threshold() {
        let mut config = ReplicationConfig::default();
        config.snapshot_threshold = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = ReplicationConfig {
            cluster_id: "test-cluster".to_string(),
            node_id: 42,
            peer_addresses: vec![
                PeerAddress {
                    node_id: 1,
                    addr: "http://node1:9100".to_string(),
                },
                PeerAddress {
                    node_id: 2,
                    addr: "http://node2:9100".to_string(),
                },
            ],
            heartbeat_interval_ms: 250,
            election_timeout_min_ms: 1000,
            election_timeout_max_ms: 2000,
            snapshot_threshold: 500,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ReplicationConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.cluster_id, "test-cluster");
        assert_eq!(deserialized.node_id, 42);
        assert_eq!(deserialized.peer_addresses.len(), 2);
        assert_eq!(deserialized.heartbeat_interval_ms, 250);
        assert_eq!(deserialized.snapshot_threshold, 500);
    }

    #[test]
    fn test_peer_address_equality() {
        let a = PeerAddress {
            node_id: 1,
            addr: "http://localhost:9100".into(),
        };
        let b = PeerAddress {
            node_id: 1,
            addr: "http://localhost:9100".into(),
        };
        assert_eq!(a, b);
    }
}
