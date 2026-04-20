//! Bounded extraction queue with backpressure and graceful shutdown.
//!
//! Instead of fire-and-forget `tokio::spawn` per extraction request,
//! this module provides a bounded channel + worker pool. Benefits:
//! - Backpressure: at most `QUEUE_CAPACITY` pending extractions
//! - Bounded concurrency: `MAX_CONCURRENT` parallel LLM calls
//! - Graceful drain: on shutdown, the worker finishes in-flight work

use std::sync::Arc;

use mentedb_core::types::{AgentId, SpaceId};
use mentedb_extraction::ExtractionConfig;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

const QUEUE_CAPACITY: usize = 64;
const MAX_CONCURRENT: usize = 4;

/// What we send into the queue.
pub struct ExtractionRequest {
    pub config: ExtractionConfig,
    pub content: String,
    pub agent_id: AgentId,
    pub space_id: SpaceId,
    pub db: Arc<mentedb::MenteDb>,
}

pub type ExtractionSender = mpsc::Sender<ExtractionRequest>;
type ExtractionReceiver = mpsc::Receiver<ExtractionRequest>;

/// Spawn the extraction worker. Returns the sender half and a join handle
/// that resolves when all in-flight work is drained.
pub fn spawn_worker() -> (ExtractionSender, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel(QUEUE_CAPACITY);
    let handle = tokio::spawn(worker_loop(rx));
    (tx, handle)
}

async fn worker_loop(mut rx: ExtractionReceiver) {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT));

    while let Some(req) = rx.recv().await {
        let permit = semaphore.clone().acquire_owned().await;
        if permit.is_err() {
            break; // semaphore closed
        }
        let permit = permit.unwrap();
        tokio::spawn(async move {
            if let Err(e) = process_extraction(req).await {
                tracing::warn!(error = %e, "background extraction failed");
            }
            drop(permit);
        });
    }

    // Channel closed — wait for all in-flight extractions to finish
    let _ = semaphore.acquire_many(MAX_CONCURRENT as u32).await;
    tracing::info!("extraction worker drained");
}

async fn process_extraction(req: ExtractionRequest) -> Result<(), String> {
    use mentedb_core::MemoryNode;
    use mentedb_core::types::MemoryId;
    use mentedb_extraction::{ExtractionPipeline, HttpExtractionProvider};

    let provider = HttpExtractionProvider::new(req.config.clone())
        .map_err(|e| format!("extraction provider init: {e}"))?;
    let pipeline = ExtractionPipeline::new(provider, req.config);

    let all_memories = pipeline
        .extract_from_conversation(&req.content)
        .await
        .map_err(|e| format!("extraction failed: {e}"))?;

    let quality_passed = pipeline.filter_quality(&all_memories);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    let mut stored = 0usize;
    for memory in &quality_passed {
        let memory_type =
            mentedb_extraction::map_extraction_type_to_memory_type(&memory.memory_type);
        let id = MemoryId::new();
        let node = MemoryNode {
            id,
            agent_id: req.agent_id,
            memory_type,
            embedding: vec![],
            content: memory.content.clone(),
            created_at: now,
            accessed_at: now,
            access_count: 0,
            salience: memory.confidence,
            confidence: memory.confidence,
            space_id: req.space_id,
            attributes: std::collections::HashMap::new(),
            tags: memory.tags.clone(),
            valid_from: None,
            valid_until: None,
        };
        match req.db.store(node) {
            Ok(()) => stored += 1,
            Err(e) => {
                tracing::warn!(error = %e, "failed to store extracted memory, skipping");
            }
        }
    }

    tracing::debug!(
        total = all_memories.len(),
        quality = quality_passed.len(),
        stored,
        "extraction complete"
    );
    Ok(())
}
