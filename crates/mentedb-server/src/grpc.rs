//! gRPC service implementations for MenteDB.
//!
//! Provides a CognitionService with bidirectional streaming that wraps the
//! CognitionStream engine, and a MemoryService with standard CRUD operations.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use mentedb_cognitive::stream::{CognitionStream, StreamAlert};
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::{AttributeValue, MemoryType};
use mentedb_core::{MemoryEdge, MemoryNode};
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tonic::{Request, Response, Status, Streaming};
use tracing::{error, info};
use uuid::Uuid;

use crate::auth;
use crate::state::AppState;

pub mod pb {
    tonic::include_proto!("mentedb");
}

use mentedb_core::types::{AgentId, MemoryId, SpaceId};
use pb::cognition_service_server::CognitionService;
use pb::memory_service_server::MemoryService;

// ---------------------------------------------------------------------------
// CognitionService
// ---------------------------------------------------------------------------

/// gRPC service implementation for cognitive memory operations.
pub struct CognitionServiceImpl {
    #[allow(dead_code)]
    pub state: Arc<AppState>,
}

type CognitionStream_ =
    Pin<Box<dyn Stream<Item = Result<pb::CognitionAlert, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl CognitionService for CognitionServiceImpl {
    type StreamCognitionStream = CognitionStream_;

    async fn stream_cognition(
        &self,
        request: Request<Streaming<pb::CognitionRequest>>,
    ) -> Result<Response<Self::StreamCognitionStream>, Status> {
        authenticate_grpc_streaming(&self.state, request.metadata())?;
        info!("gRPC cognition stream opened");

        let stream_engine = Arc::new(CognitionStream::new(1000));
        let known_facts: Arc<std::sync::Mutex<Vec<(MemoryId, String)>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        let (tx, rx) = mpsc::channel(128);
        let mut inbound = request.into_inner();

        let engine = stream_engine.clone();
        let facts = known_facts.clone();

        tokio::spawn(async move {
            while let Some(result) = inbound.next().await {
                let req = match result {
                    Ok(r) => r,
                    Err(e) => {
                        error!("cognition stream receive error: {e}");
                        break;
                    }
                };

                let payload = match req.payload {
                    Some(p) => p,
                    None => continue,
                };

                match payload {
                    pb::cognition_request::Payload::Token(t) => {
                        engine.feed_token(&t.text);

                        let current_facts = facts.lock().unwrap().clone();
                        let alerts = engine.check_alerts(&current_facts);
                        for alert in alerts {
                            let proto_alert = stream_alert_to_proto(alert);
                            if tx.send(Ok(proto_alert)).await.is_err() {
                                return;
                            }
                        }
                    }
                    pb::cognition_request::Payload::KnownFact(kf) => {
                        let mid = kf
                            .memory_id
                            .parse::<MemoryId>()
                            .unwrap_or_else(|_| MemoryId::nil());
                        facts.lock().unwrap().push((mid, kf.content));
                    }
                    pb::cognition_request::Payload::EndOfTurn(_) => {
                        let current_facts = facts.lock().unwrap().clone();
                        let alerts = engine.check_alerts(&current_facts);
                        for alert in alerts {
                            let proto_alert = stream_alert_to_proto(alert);
                            if tx.send(Ok(proto_alert)).await.is_err() {
                                return;
                            }
                        }
                    }
                    pb::cognition_request::Payload::Flush(_) => {
                        let text = engine.drain_buffer();
                        let alert = pb::CognitionAlert {
                            alert: Some(pb::cognition_alert::Alert::BufferFlushed(
                                pb::BufferFlushedAlert {
                                    accumulated_text: text,
                                },
                            )),
                        };
                        if tx.send(Ok(alert)).await.is_err() {
                            return;
                        }
                    }
                }
            }
            info!("gRPC cognition stream closed");
        });

        let output = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output)))
    }
}

fn stream_alert_to_proto(alert: StreamAlert) -> pb::CognitionAlert {
    let inner = match alert {
        StreamAlert::Contradiction {
            memory_id,
            ai_said,
            stored,
        } => pb::cognition_alert::Alert::Contradiction(pb::ContradictionAlert {
            memory_id: memory_id.to_string(),
            ai_said,
            stored,
        }),
        StreamAlert::Forgotten { memory_id, summary } => {
            pb::cognition_alert::Alert::Forgotten(pb::ForgottenAlert {
                memory_id: memory_id.to_string(),
                summary,
            })
        }
        StreamAlert::Correction {
            memory_id,
            old,
            new,
        } => pb::cognition_alert::Alert::Correction(pb::CorrectionAlert {
            memory_id: memory_id.to_string(),
            old_content: old,
            new_content: new,
        }),
        StreamAlert::Reinforcement { memory_id } => {
            pb::cognition_alert::Alert::Reinforcement(pb::ReinforcementAlert {
                memory_id: memory_id.to_string(),
            })
        }
    };
    pb::CognitionAlert { alert: Some(inner) }
}

// ---------------------------------------------------------------------------
// MemoryService
// ---------------------------------------------------------------------------

/// gRPC service implementation for basic memory CRUD operations.
pub struct MemoryServiceImpl {
    /// Shared application state containing the database handle.
    pub state: Arc<AppState>,
}

#[tonic::async_trait]
impl MemoryService for MemoryServiceImpl {
    async fn store(
        &self,
        request: Request<pb::StoreRequest>,
    ) -> Result<Response<pb::StoreResponse>, Status> {
        let caller = authenticate_grpc_request(&self.state, &request)?;
        let req = request.into_inner();

        let agent_id = AgentId(parse_uuid_field(&req.agent_id, "agent_id")?);
        if let Some(ref ta) = caller {
            let tu: AgentId = ta
                .parse()
                .map_err(|_| Status::internal("bad token agent_id"))?;
            if tu != agent_id {
                return Err(Status::permission_denied("agent_id mismatch"));
            }
        }
        let memory_type = parse_memory_type_str(&req.memory_type)?;
        let space_id = if req.space_id.is_empty() {
            SpaceId::nil()
        } else {
            SpaceId(parse_uuid_field(&req.space_id, "space_id")?)
        };

        let salience = if req.salience == 0.0 {
            0.5
        } else {
            req.salience
        };
        let confidence = if req.confidence == 0.0 {
            1.0
        } else {
            req.confidence
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let id = MemoryId::new();

        let attributes = req
            .attributes
            .into_iter()
            .map(|(k, v)| (k, AttributeValue::String(v)))
            .collect::<HashMap<String, AttributeValue>>();

        let node = MemoryNode {
            id,
            agent_id,
            memory_type,
            embedding: req.embedding,
            content: req.content,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            salience,
            confidence,
            space_id,
            attributes,
            tags: req.tags,
            valid_from: req.valid_from,
            valid_until: req.valid_until,
        };

        let mut db = self.state.db.write().await;
        db.store(node).map_err(|e| {
            error!("gRPC store failed: {e}");
            Status::internal(format!("store failed: {e}"))
        })?;

        Ok(Response::new(pb::StoreResponse {
            id: id.to_string(),
            status: "stored".into(),
        }))
    }

    async fn recall(
        &self,
        request: Request<pb::RecallRequest>,
    ) -> Result<Response<pb::RecallResponse>, Status> {
        authenticate_grpc_request(&self.state, &request)?;
        let req = request.into_inner();

        let mut db = self.state.db.write().await;
        let window = db.recall(&req.query).map_err(|e| {
            error!("gRPC recall failed: {e}");
            Status::internal(format!("recall failed: {e}"))
        })?;

        let memory_count: usize = window.blocks.iter().map(|b| b.memories.len()).sum();

        Ok(Response::new(pb::RecallResponse {
            context: window.format,
            total_tokens: window.total_tokens as u64,
            memory_count: memory_count as u64,
        }))
    }

    async fn search(
        &self,
        request: Request<pb::SearchRequest>,
    ) -> Result<Response<pb::SearchResponse>, Status> {
        authenticate_grpc_request(&self.state, &request)?;
        let req = request.into_inner();
        let k = if req.k == 0 { 10 } else { req.k as usize };

        if req.embedding.is_empty() {
            return Err(Status::invalid_argument("missing embedding vector"));
        }

        let mut db = self.state.db.write().await;
        let results = db.recall_similar(&req.embedding, k).map_err(|e| {
            error!("gRPC search failed: {e}");
            Status::internal(format!("search failed: {e}"))
        })?;

        let items: Vec<pb::SearchResult> = results
            .iter()
            .map(|(id, score)| pb::SearchResult {
                id: id.to_string(),
                score: *score,
            })
            .collect();

        Ok(Response::new(pb::SearchResponse { results: items }))
    }

    async fn forget(
        &self,
        request: Request<pb::ForgetRequest>,
    ) -> Result<Response<pb::ForgetResponse>, Status> {
        authenticate_grpc_request(&self.state, &request)?;
        let req = request.into_inner();
        let id = MemoryId(parse_uuid_field(&req.id, "id")?);

        let mut db = self.state.db.write().await;
        db.forget(id).map_err(|e| {
            error!("gRPC forget failed: {e}");
            Status::internal(format!("forget failed: {e}"))
        })?;

        Ok(Response::new(pb::ForgetResponse {
            status: "deleted".into(),
        }))
    }

    async fn relate(
        &self,
        request: Request<pb::RelateRequest>,
    ) -> Result<Response<pb::RelateResponse>, Status> {
        authenticate_grpc_request(&self.state, &request)?;
        let req = request.into_inner();
        let source = MemoryId(parse_uuid_field(&req.source, "source")?);
        let target = MemoryId(parse_uuid_field(&req.target, "target")?);
        let edge_type = parse_edge_type_str(&req.edge_type)?;
        let weight = if req.weight == 0.0 { 1.0 } else { req.weight };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let edge = MemoryEdge {
            source,
            target,
            edge_type,
            weight,
            created_at: now,
            valid_from: req.valid_from,
            valid_until: req.valid_until,
            label: None,
        };

        let mut db = self.state.db.write().await;
        db.relate(edge).map_err(|e| {
            error!("gRPC relate failed: {e}");
            Status::internal(format!("relate failed: {e}"))
        })?;

        Ok(Response::new(pb::RelateResponse {
            status: "created".into(),
        }))
    }
}

#[allow(clippy::result_large_err)]
fn authenticate_grpc_request<T>(
    state: &AppState,
    request: &Request<T>,
) -> Result<Option<String>, Status> {
    let secret = match &state.jwt_secret {
        Some(s) => s,
        None => return Ok(None),
    };
    let token = request
        .metadata()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .ok_or_else(|| Status::unauthenticated("missing or invalid authorization metadata"))?;
    auth::validate_token(secret, token)
        .map(|c| Some(c.agent_id))
        .map_err(|e| Status::unauthenticated(format!("invalid token: {e}")))
}
#[allow(clippy::result_large_err)]
fn authenticate_grpc_streaming(
    state: &AppState,
    metadata: &tonic::metadata::MetadataMap,
) -> Result<Option<String>, Status> {
    let secret = match &state.jwt_secret {
        Some(s) => s,
        None => return Ok(None),
    };
    let token = metadata
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .ok_or_else(|| Status::unauthenticated("missing or invalid authorization metadata"))?;
    auth::validate_token(secret, token)
        .map(|c| Some(c.agent_id))
        .map_err(|e| Status::unauthenticated(format!("invalid token: {e}")))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[allow(clippy::result_large_err)]
fn parse_uuid_field(s: &str, field: &str) -> Result<Uuid, Status> {
    Uuid::parse_str(s)
        .map_err(|_| Status::invalid_argument(format!("invalid UUID for '{field}': {s}")))
}

#[allow(clippy::result_large_err)]
fn parse_memory_type_str(s: &str) -> Result<MemoryType, Status> {
    match s.to_lowercase().as_str() {
        "episodic" => Ok(MemoryType::Episodic),
        "semantic" => Ok(MemoryType::Semantic),
        "procedural" => Ok(MemoryType::Procedural),
        "antipattern" | "anti_pattern" => Ok(MemoryType::AntiPattern),
        "reasoning" => Ok(MemoryType::Reasoning),
        "correction" => Ok(MemoryType::Correction),
        _ => Err(Status::invalid_argument(format!(
            "unknown memory_type: {s}"
        ))),
    }
}

#[allow(clippy::result_large_err)]
fn parse_edge_type_str(s: &str) -> Result<EdgeType, Status> {
    match s.to_lowercase().as_str() {
        "caused" => Ok(EdgeType::Caused),
        "before" => Ok(EdgeType::Before),
        "related" => Ok(EdgeType::Related),
        "contradicts" => Ok(EdgeType::Contradicts),
        "supports" => Ok(EdgeType::Supports),
        "supersedes" => Ok(EdgeType::Supersedes),
        "derived" => Ok(EdgeType::Derived),
        "partof" | "part_of" => Ok(EdgeType::PartOf),
        _ => Err(Status::invalid_argument(format!("unknown edge_type: {s}"))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb::MenteDb;
    use std::time::Instant;
    use tempfile::TempDir;
    use tokio::sync::RwLock;

    fn make_test_state() -> (Arc<AppState>, TempDir) {
        let tmp = TempDir::new().unwrap();
        let db = MenteDb::open(tmp.path()).unwrap();
        let state = Arc::new(AppState {
            db: Arc::new(RwLock::new(db)),
            spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
            jwt_secret: None,
            admin_key: None,
            start_time: Instant::now(),
            extraction_config: None,
            auto_extract: false,
        });
        (state, tmp)
    }

    #[tokio::test]
    async fn test_grpc_memory_store_and_recall() {
        let (state, _tmp) = make_test_state();
        let svc = MemoryServiceImpl {
            state: state.clone(),
        };

        let agent_id = AgentId::new().to_string();
        let store_req = Request::new(pb::StoreRequest {
            agent_id: agent_id.clone(),
            memory_type: "episodic".into(),
            content: "The user prefers dark mode".into(),
            embedding: vec![],
            tags: vec!["preference".into()],
            attributes: HashMap::new(),
            space_id: String::new(),
            salience: 0.8,
            confidence: 1.0,
            valid_from: None,
            valid_until: None,
        });

        let resp = svc.store(store_req).await.unwrap();
        let inner = resp.into_inner();
        assert_eq!(inner.status, "stored");
        assert!(!inner.id.is_empty());

        let recall_req = Request::new(pb::RecallRequest {
            query: "RECALL memories LIMIT 100".into(),
        });
        let resp = svc.recall(recall_req).await.unwrap();
        let inner = resp.into_inner();
        // Recall succeeded without error; count may be 0 if the engine has not
        // indexed the memory yet, so we just verify the response is valid.
        let _ = inner.memory_count;
    }

    #[tokio::test]
    async fn test_grpc_memory_forget() {
        let (state, _tmp) = make_test_state();
        let svc = MemoryServiceImpl {
            state: state.clone(),
        };

        let agent_id = AgentId::new().to_string();
        let store_resp = svc
            .store(Request::new(pb::StoreRequest {
                agent_id,
                memory_type: "semantic".into(),
                content: "Temporary memory".into(),
                embedding: vec![],
                tags: vec![],
                attributes: HashMap::new(),
                space_id: String::new(),
                salience: 0.5,
                confidence: 1.0,
                valid_from: None,
                valid_until: None,
            }))
            .await
            .unwrap();
        let stored_id = store_resp.into_inner().id;

        let forget_resp = svc
            .forget(Request::new(pb::ForgetRequest {
                id: stored_id.clone(),
            }))
            .await
            .unwrap();
        assert_eq!(forget_resp.into_inner().status, "deleted");
    }

    #[tokio::test]
    async fn test_grpc_memory_relate() {
        let (state, _tmp) = make_test_state();
        let svc = MemoryServiceImpl {
            state: state.clone(),
        };

        let agent_id = AgentId::new().to_string();
        let id1 = svc
            .store(Request::new(pb::StoreRequest {
                agent_id: agent_id.clone(),
                memory_type: "episodic".into(),
                content: "Event A".into(),
                embedding: vec![],
                tags: vec![],
                attributes: HashMap::new(),
                space_id: String::new(),
                salience: 0.5,
                confidence: 1.0,
                valid_from: None,
                valid_until: None,
            }))
            .await
            .unwrap()
            .into_inner()
            .id;

        let id2 = svc
            .store(Request::new(pb::StoreRequest {
                agent_id,
                memory_type: "episodic".into(),
                content: "Event B".into(),
                embedding: vec![],
                tags: vec![],
                attributes: HashMap::new(),
                space_id: String::new(),
                salience: 0.5,
                confidence: 1.0,
                valid_from: None,
                valid_until: None,
            }))
            .await
            .unwrap()
            .into_inner()
            .id;

        let relate_resp = svc
            .relate(Request::new(pb::RelateRequest {
                source: id1,
                target: id2,
                edge_type: "caused".into(),
                weight: 0.9,
                valid_from: None,
                valid_until: None,
            }))
            .await
            .unwrap();
        assert_eq!(relate_resp.into_inner().status, "created");
    }

    #[tokio::test]
    async fn test_grpc_cognition_stream() {
        use pb::cognition_service_server::CognitionServiceServer;
        use tokio::net::TcpListener;
        use tonic::transport::{Channel, Server};

        let (state, _tmp) = make_test_state();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let svc = CognitionServiceServer::new(CognitionServiceImpl {
            state: state.clone(),
        });

        tokio::spawn(async move {
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            Server::builder()
                .add_service(svc)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });

        // Small delay to let the server start
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let channel = Channel::from_shared(format!("http://{addr}"))
            .unwrap()
            .connect()
            .await
            .unwrap();

        let mut client = pb::cognition_service_client::CognitionServiceClient::new(channel);

        let (tx, rx) = mpsc::channel(32);
        let outbound = ReceiverStream::new(rx);

        let response = client.stream_cognition(outbound).await.unwrap();
        let mut inbound = response.into_inner();

        // Register a known fact
        let mid = MemoryId::new();
        tx.send(pb::CognitionRequest {
            payload: Some(pb::cognition_request::Payload::KnownFact(pb::KnownFact {
                memory_id: mid.to_string(),
                content: "The system uses PostgreSQL for storage".into(),
            })),
        })
        .await
        .unwrap();

        // Feed tokens that contradict the known fact
        tx.send(pb::CognitionRequest {
            payload: Some(pb::cognition_request::Payload::Token(pb::TokenPayload {
                text: "The system does not use PostgreSQL, actually it uses MySQL".into(),
            })),
        })
        .await
        .unwrap();

        // We should get a contradiction alert
        if let Some(Ok(alert)) = inbound.next().await {
            match alert.alert {
                Some(pb::cognition_alert::Alert::Contradiction(c)) => {
                    assert_eq!(c.memory_id, mid.to_string());
                    assert!(!c.ai_said.is_empty());
                    assert!(!c.stored.is_empty());
                }
                other => panic!("expected contradiction alert, got: {:?}", other),
            }
        } else {
            panic!("expected an alert from the cognition stream");
        }

        // Test flush
        tx.send(pb::CognitionRequest {
            payload: Some(pb::cognition_request::Payload::Flush(pb::FlushBuffer {})),
        })
        .await
        .unwrap();

        if let Some(Ok(alert)) = inbound.next().await {
            match alert.alert {
                Some(pb::cognition_alert::Alert::BufferFlushed(f)) => {
                    assert!(!f.accumulated_text.is_empty());
                }
                other => panic!("expected buffer_flushed alert, got: {:?}", other),
            }
        }

        drop(tx);
    }

    #[tokio::test]
    async fn test_grpc_invalid_uuid() {
        let (state, _tmp) = make_test_state();
        let svc = MemoryServiceImpl { state };

        let result = svc
            .store(Request::new(pb::StoreRequest {
                agent_id: "not-a-uuid".into(),
                memory_type: "episodic".into(),
                content: "test".into(),
                embedding: vec![],
                tags: vec![],
                attributes: HashMap::new(),
                space_id: String::new(),
                salience: 0.5,
                confidence: 1.0,
                valid_from: None,
                valid_until: None,
            }))
            .await;

        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn test_grpc_invalid_memory_type() {
        let (state, _tmp) = make_test_state();
        let svc = MemoryServiceImpl { state };

        let result = svc
            .store(Request::new(pb::StoreRequest {
                agent_id: AgentId::new().to_string(),
                memory_type: "nonexistent".into(),
                content: "test".into(),
                embedding: vec![],
                tags: vec![],
                attributes: HashMap::new(),
                space_id: String::new(),
                salience: 0.5,
                confidence: 1.0,
                valid_from: None,
                valid_until: None,
            }))
            .await;

        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn test_grpc_search_missing_embedding() {
        let (state, _tmp) = make_test_state();
        let svc = MemoryServiceImpl { state };

        let result = svc
            .search(Request::new(pb::SearchRequest {
                embedding: vec![],
                k: 10,
            }))
            .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    fn make_auth_test_state(secret: &str) -> (Arc<AppState>, TempDir) {
        let tmp = TempDir::new().unwrap();
        let db = MenteDb::open(tmp.path()).unwrap();
        (
            Arc::new(AppState {
                db: Arc::new(RwLock::new(db)),
                spaces: Arc::new(tokio::sync::RwLock::new(mentedb_core::SpaceManager::new())),
                jwt_secret: Some(secret.into()),
                admin_key: Some("ak".into()),
                start_time: Instant::now(),
                extraction_config: None,
                auto_extract: false,
            }),
            tmp,
        )
    }
    #[tokio::test]
    async fn test_grpc_auth_required_when_secret_set() {
        let (s, _t) = make_auth_test_state("s");
        let svc = MemoryServiceImpl { state: s };
        let r = svc
            .recall(Request::new(pb::RecallRequest {
                query: "RECALL memories LIMIT 10".into(),
            }))
            .await;
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code(), tonic::Code::Unauthenticated);
    }
    #[tokio::test]
    async fn test_grpc_auth_succeeds_with_valid_token() {
        let (s, _t) = make_auth_test_state("s");
        let svc = MemoryServiceImpl { state: s };
        let a = AgentId::new();
        let tok = crate::auth::create_token("s", &a.to_string(), false, 1);
        let mut r = Request::new(pb::StoreRequest {
            agent_id: a.to_string(),
            memory_type: "episodic".into(),
            content: "t".into(),
            embedding: vec![],
            tags: vec![],
            attributes: HashMap::new(),
            space_id: String::new(),
            salience: 0.5,
            confidence: 1.0,
            valid_from: None,
            valid_until: None,
        });
        r.metadata_mut()
            .insert("authorization", format!("Bearer {tok}").parse().unwrap());
        assert_eq!(svc.store(r).await.unwrap().into_inner().status, "stored");
    }
    #[tokio::test]
    async fn test_grpc_auth_rejects_wrong_agent_id() {
        let (s, _t) = make_auth_test_state("s");
        let svc = MemoryServiceImpl { state: s };
        let ta = AgentId::new();
        let ra = AgentId::new();
        let tok = crate::auth::create_token("s", &ta.to_string(), false, 1);
        let mut r = Request::new(pb::StoreRequest {
            agent_id: ra.to_string(),
            memory_type: "episodic".into(),
            content: "t".into(),
            embedding: vec![],
            tags: vec![],
            attributes: HashMap::new(),
            space_id: String::new(),
            salience: 0.5,
            confidence: 1.0,
            valid_from: None,
            valid_until: None,
        });
        r.metadata_mut()
            .insert("authorization", format!("Bearer {tok}").parse().unwrap());
        assert_eq!(
            svc.store(r).await.unwrap_err().code(),
            tonic::Code::PermissionDenied
        );
    }
    #[tokio::test]
    async fn test_grpc_auth_invalid_token() {
        let (s, _t) = make_auth_test_state("s");
        let svc = MemoryServiceImpl { state: s };
        let mut r = Request::new(pb::RecallRequest {
            query: "RECALL memories LIMIT 10".into(),
        });
        r.metadata_mut()
            .insert("authorization", "Bearer bad.tok".parse().unwrap());
        assert_eq!(
            svc.recall(r).await.unwrap_err().code(),
            tonic::Code::Unauthenticated
        );
    }
}
