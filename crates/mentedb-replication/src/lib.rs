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

pub mod cluster;
pub mod config;
pub mod network;
pub mod raft_node;
pub mod store;

pub use cluster::MenteCluster;
pub use config::ReplicationConfig;
pub use raft_node::{MenteRaftNode, MenteRequest, MenteResponse};
