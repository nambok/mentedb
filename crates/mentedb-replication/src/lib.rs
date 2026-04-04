//! Distributed replication layer for MenteDB using Raft consensus.
//!
//! This crate provides a Raft-based replication layer that ensures strong
//! consistency across a cluster of MenteDB nodes. It uses the `openraft`
//! library for the consensus protocol implementation.
//!
//! # Architecture
//!
//! - `MenteRaftNode` wraps the openraft Raft instance with MenteDB-specific types
//! - `MenteNetwork` provides HTTP-based node-to-node transport
//! - `LogStore` and `StateMachine` back the Raft log and state with MenteDB storage
//! - `MenteCluster` manages cluster membership and leader forwarding
//! - `ReplicationConfig` holds cluster and consensus configuration

/// Cluster membership management and leader forwarding.
pub mod cluster;
/// Replication and consensus configuration.
pub mod config;
/// HTTP based node to node transport layer.
pub mod network;
/// Raft node wrapper with MenteDB specific types.
pub mod raft_node;
/// Log store and state machine backed by MenteDB storage.
pub mod store;

pub use cluster::MenteCluster;
pub use config::ReplicationConfig;
pub use raft_node::{MenteRaftNode, MenteRequest, MenteResponse};
