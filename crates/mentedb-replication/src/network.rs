//! HTTP-based network transport for Raft node-to-node communication.
//!
//! Implements `RaftNetworkFactory` and `RaftNetworkV2` traits from openraft,
//! using reqwest for outbound HTTP calls. Provides endpoints for
//! AppendEntries, Vote, and full Snapshot transfer.

use std::future::Future;

use openraft::errors::{RPCError, ReplicationClosed, StreamingError, Unreachable};
use openraft::network::{RPCOption, RaftNetworkFactory, RaftNetworkV2};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, SnapshotResponse, VoteRequest, VoteResponse,
};
use openraft::{BasicNode, OptionalSend};
use serde::{Deserialize, Serialize};

use crate::raft_node::TypeConfig;

type SnapshotOf = openraft::type_config::alias::SnapshotOf<TypeConfig>;
type VoteOf = openraft::type_config::alias::VoteOf<TypeConfig>;
type MenteRPCError = RPCError<TypeConfig>;
type MenteStreamingError = StreamingError<TypeConfig>;
type MenteSnapshotResponse = SnapshotResponse<TypeConfig>;

/// Factory that creates network connections to peer nodes.
#[derive(Clone)]
pub struct MenteNetworkFactory {
    client: reqwest::Client,
}

impl MenteNetworkFactory {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for MenteNetworkFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl RaftNetworkFactory<TypeConfig> for MenteNetworkFactory {
    type Network = MenteNetwork;

    async fn new_client(&mut self, _target: u64, node: &BasicNode) -> Self::Network {
        MenteNetwork {
            target_addr: node.addr.clone(),
            client: self.client.clone(),
        }
    }
}

/// A network connection to a single Raft peer node.
///
/// Sends Raft RPCs over HTTP POST requests to the target node.
pub struct MenteNetwork {
    target_addr: String,
    client: reqwest::Client,
}

impl MenteNetwork {
    async fn post<Req, Resp>(&mut self, path: &str, req: &Req) -> Result<Resp, MenteRPCError>
    where
        Req: Serialize + Send + Sync,
        Resp: for<'de> Deserialize<'de>,
    {
        let url = format!("{}{}", self.target_addr, path);
        let resp = self
            .client
            .post(&url)
            .json(req)
            .send()
            .await
            .map_err(|e| RPCError::Unreachable(Unreachable::new(&e)))?;

        let body = resp
            .bytes()
            .await
            .map_err(|e| RPCError::Unreachable(Unreachable::new(&e)))?;

        serde_json::from_slice(&body).map_err(|e| RPCError::Unreachable(Unreachable::new(&e)))
    }
}

impl RaftNetworkV2<TypeConfig> for MenteNetwork {
    async fn append_entries(
        &mut self,
        req: AppendEntriesRequest<TypeConfig>,
        _option: RPCOption,
    ) -> Result<AppendEntriesResponse<TypeConfig>, MenteRPCError> {
        self.post("/raft/append-entries", &req).await
    }

    async fn full_snapshot(
        &mut self,
        vote: VoteOf,
        snapshot: SnapshotOf,
        _cancel: impl Future<Output = ReplicationClosed> + OptionalSend + 'static,
        _option: RPCOption,
    ) -> Result<MenteSnapshotResponse, MenteStreamingError> {
        #[derive(Serialize)]
        struct SnapshotPayload {
            vote: String,
            meta: String,
            data: Vec<u8>,
        }
        let payload = SnapshotPayload {
            vote: serde_json::to_string(&vote).unwrap_or_default(),
            meta: serde_json::to_string(&snapshot.meta).unwrap_or_default(),
            data: snapshot.snapshot.into_inner(),
        };

        let url = format!("{}/raft/snapshot", self.target_addr);
        let resp = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| StreamingError::Unreachable(Unreachable::new(&e)))?;

        let body = resp
            .bytes()
            .await
            .map_err(|e| StreamingError::Unreachable(Unreachable::new(&e)))?;

        serde_json::from_slice(&body)
            .map_err(|e| StreamingError::Unreachable(Unreachable::new(&e)))
    }

    async fn vote(
        &mut self,
        req: VoteRequest<TypeConfig>,
        _option: RPCOption,
    ) -> Result<VoteResponse<TypeConfig>, MenteRPCError> {
        self.post("/raft/vote", &req).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_factory_creation() {
        let _factory = MenteNetworkFactory::new();
    }

    #[test]
    fn test_network_factory_default() {
        let _factory = MenteNetworkFactory::default();
    }

    #[tokio::test]
    async fn test_network_client_creation() {
        let mut factory = MenteNetworkFactory::new();
        let node = BasicNode {
            addr: "http://127.0.0.1:9100".to_string(),
        };
        let network = factory.new_client(1, &node).await;
        assert_eq!(network.target_addr, "http://127.0.0.1:9100");
    }
}
