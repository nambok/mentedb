//! High-level knowledge graph manager.

use std::io::Write;
use std::path::{Path, PathBuf};

use mentedb_core::edge::MemoryEdge;
use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::belief::propagate_update;
use crate::contradiction::find_contradictions;
use crate::csr::CsrGraph;
use crate::traversal::extract_subgraph;

/// A single durable graph mutation, appended to the edge log as one JSON line.
#[derive(Debug, Serialize, Deserialize)]
enum LogRecord {
    AddNode(MemoryId),
    RemoveNode(MemoryId),
    AddEdge(MemoryEdge),
}

/// Append-only log of graph mutations since the last snapshot.
///
/// The snapshot (`graph.json`) is only written on `save()`; without this log
/// every edge created since the last flush would be lost on crash while the
/// memories themselves survive via the storage WAL.
struct EdgeLog {
    file: std::fs::File,
}

const EDGE_LOG_FILE: &str = "edges.jsonl";

impl EdgeLog {
    fn append(&mut self, record: &LogRecord) -> MenteResult<()> {
        let mut line =
            serde_json::to_vec(record).map_err(|e| MenteError::Serialization(e.to_string()))?;
        line.push(b'\n');
        self.file.write_all(&line)?;
        self.file.sync_data()?;
        Ok(())
    }
}

/// Owns a `CsrGraph` and provides high-level graph operations.
///
/// All methods take `&self` — internal `RwLock` handles concurrency.
pub struct GraphManager {
    graph: RwLock<CsrGraph>,
    /// Directory-backed managers log mutations for crash durability.
    /// `None` for purely in-memory managers created with `new()`.
    log: Mutex<Option<EdgeLog>>,
    dir: Option<PathBuf>,
}

impl GraphManager {
    /// Creates a new in-memory graph manager (no durability log).
    pub fn new() -> Self {
        Self {
            graph: RwLock::new(CsrGraph::new()),
            log: Mutex::new(None),
            dir: None,
        }
    }

    /// Open a directory-backed graph manager.
    ///
    /// Loads the last snapshot (`graph.json`) if present, replays the edge log
    /// (`edges.jsonl`) on top of it, and keeps the log open for appending, so
    /// graph mutations are crash-durable between snapshots.
    pub fn open(dir: &Path) -> MenteResult<Self> {
        std::fs::create_dir_all(dir)?;

        let snapshot_path = dir.join("graph.json");
        let mut graph = if snapshot_path.exists() {
            CsrGraph::load(&snapshot_path)?
        } else {
            CsrGraph::new()
        };

        let log_path = dir.join(EDGE_LOG_FILE);
        if log_path.exists() {
            let contents = std::fs::read_to_string(&log_path)?;
            let lines: Vec<&str> = contents.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<LogRecord>(line) {
                    Ok(LogRecord::AddNode(id)) => {
                        graph.add_node(id);
                    }
                    Ok(LogRecord::RemoveNode(id)) => {
                        graph.remove_node(id);
                    }
                    Ok(LogRecord::AddEdge(edge)) => {
                        graph.add_node(edge.source);
                        graph.add_node(edge.target);
                        graph.add_edge(&edge);
                    }
                    Err(e) => {
                        // A torn final line is expected after a crash mid-append;
                        // anything else is corruption worth surfacing.
                        if i + 1 == lines.len() {
                            warn!("dropping torn final edge log line: {e}");
                        } else {
                            return Err(MenteError::Serialization(format!(
                                "corrupt edge log line {}: {e}",
                                i + 1
                            )));
                        }
                    }
                }
            }
        }

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        Ok(Self {
            graph: RwLock::new(graph),
            log: Mutex::new(Some(EdgeLog { file })),
            dir: Some(dir.to_path_buf()),
        })
    }

    /// Save a graph snapshot to the given directory and truncate the edge log.
    pub fn save(&self, dir: &Path) -> MenteResult<()> {
        std::fs::create_dir_all(dir)?;
        self.graph.read().save(&dir.join("graph.json"))?;

        // The snapshot now contains everything the log recorded.
        let mut log = self.log.lock();
        if log.is_some() && self.dir.as_deref() == Some(dir) {
            let log_path = dir.join(EDGE_LOG_FILE);
            let file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&log_path)?;
            file.sync_data()?;
            drop(file);
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)?;
            *log = Some(EdgeLog { file });
        }
        Ok(())
    }

    /// Load the graph snapshot from the given directory (no durability log).
    ///
    /// Prefer `open()`, which also replays and maintains the edge log.
    pub fn load(dir: &Path) -> MenteResult<Self> {
        let graph = CsrGraph::load(&dir.join("graph.json"))?;
        Ok(Self {
            graph: RwLock::new(graph),
            log: Mutex::new(None),
            dir: None,
        })
    }

    fn log_record(&self, record: &LogRecord) {
        if let Some(log) = self.log.lock().as_mut()
            && let Err(e) = log.append(record)
        {
            warn!("failed to append to graph edge log: {e}");
        }
    }

    /// Register a memory node in the graph.
    pub fn add_memory(&self, id: MemoryId) {
        let is_new = {
            let mut g = self.graph.write();
            let existed = g.contains_node(id);
            g.add_node(id);
            !existed
        };
        if is_new {
            self.log_record(&LogRecord::AddNode(id));
        }
    }

    /// Remove a memory node and all its edges.
    pub fn remove_memory(&self, id: MemoryId) {
        self.graph.write().remove_node(id);
        self.log_record(&LogRecord::RemoveNode(id));
    }

    /// Add a relationship (edge) between two memory nodes.
    pub fn add_relationship(&self, edge: &MemoryEdge) -> MenteResult<()> {
        {
            let mut g = self.graph.write();
            if !g.contains_node(edge.source) {
                return Err(MenteError::MemoryNotFound(edge.source));
            }
            if !g.contains_node(edge.target) {
                return Err(MenteError::MemoryNotFound(edge.target));
            }
            g.add_edge(edge);
        }
        self.log_record(&LogRecord::AddEdge(edge.clone()));
        Ok(())
    }

    /// Extract a context subgraph around a center node.
    pub fn get_context_subgraph(
        &self,
        center: MemoryId,
        depth: usize,
    ) -> (Vec<MemoryId>, Vec<MemoryEdge>) {
        extract_subgraph(&self.graph.read(), center, depth)
    }

    /// Propagate a confidence change through the graph.
    pub fn propagate_belief_change(
        &self,
        id: MemoryId,
        new_confidence: f32,
    ) -> Vec<(MemoryId, f32)> {
        propagate_update(&self.graph.read(), id, new_confidence)
    }

    /// Find all nodes that contradict the given node.
    pub fn find_all_contradictions(&self, id: MemoryId) -> Vec<MemoryId> {
        find_contradictions(&self.graph.read(), id)
    }

    /// Merge the delta log into CSR/CSC compressed storage.
    pub fn compact(&self) {
        self.graph.write().compact();
    }

    /// Remove every edge whose type is in `types`, preserving other types.
    /// Returns the number removed.
    ///
    /// There is no per-removal log record, so this is not crash-durable on its
    /// own: the caller must `save()` afterward to persist a fresh snapshot and
    /// truncate the edge log (otherwise replaying the log would re-add the
    /// removed edges). Intended for one-time bulk cleanup.
    pub fn remove_edges_of_types(&self, types: &[mentedb_core::edge::EdgeType]) -> usize {
        self.graph.write().remove_edges_of_types(types)
    }

    /// Strengthen an edge weight (Hebbian learning: neurons that fire together wire together).
    pub fn strengthen_edge(&self, source: MemoryId, target: MemoryId, delta: f32) {
        self.graph.write().strengthen_edge(source, target, delta);
    }

    /// Access the underlying graph for read-only traversals.
    ///
    /// Returns a read guard — hold it only briefly to avoid blocking writers.
    pub fn read_graph(&self) -> parking_lot::RwLockReadGuard<'_, CsrGraph> {
        self.graph.read()
    }

    /// Alias for `read_graph()` — backward compatible access to the CsrGraph.
    pub fn graph(&self) -> parking_lot::RwLockReadGuard<'_, CsrGraph> {
        self.graph.read()
    }
}

impl Default for GraphManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::edge::EdgeType;

    fn make_edge(src: MemoryId, tgt: MemoryId, etype: EdgeType) -> MemoryEdge {
        MemoryEdge {
            source: src,
            target: tgt,
            edge_type: etype,
            weight: 0.8,
            created_at: 1000,
            valid_from: None,
            valid_until: None,
            label: None,
        }
    }

    #[test]
    fn test_add_memory_and_relationship() {
        let mgr = GraphManager::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        mgr.add_memory(a);
        mgr.add_memory(b);
        assert!(
            mgr.add_relationship(&make_edge(a, b, EdgeType::Caused))
                .is_ok()
        );
    }

    #[test]
    fn test_relationship_missing_node() {
        let mgr = GraphManager::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        mgr.add_memory(a);
        // b not added
        assert!(
            mgr.add_relationship(&make_edge(a, b, EdgeType::Caused))
                .is_err()
        );
    }

    #[test]
    fn test_context_subgraph() {
        let mgr = GraphManager::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        mgr.add_memory(a);
        mgr.add_memory(b);
        mgr.add_memory(c);
        mgr.add_relationship(&make_edge(a, b, EdgeType::Caused))
            .unwrap();
        mgr.add_relationship(&make_edge(b, c, EdgeType::Related))
            .unwrap();

        let (nodes, edges) = mgr.get_context_subgraph(a, 2);
        assert_eq!(nodes.len(), 3);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_compact() {
        let mgr = GraphManager::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        mgr.add_memory(a);
        mgr.add_memory(b);
        mgr.add_relationship(&make_edge(a, b, EdgeType::Caused))
            .unwrap();
        mgr.compact();

        let out = mgr.graph().outgoing(a);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_edge_log_replay_without_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        {
            let mgr = GraphManager::open(dir.path()).unwrap();
            mgr.add_memory(a);
            mgr.add_memory(b);
            mgr.add_relationship(&make_edge(a, b, EdgeType::Caused))
                .unwrap();
            // No save(): the mutations exist only in the edge log.
        }
        {
            let mgr = GraphManager::open(dir.path()).unwrap();
            assert!(mgr.read_graph().contains_node(a));
            let out = mgr.read_graph().outgoing(a);
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].0, b);
        }
    }

    #[test]
    fn test_save_truncates_edge_log() {
        let dir = tempfile::tempdir().unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        {
            let mgr = GraphManager::open(dir.path()).unwrap();
            mgr.add_memory(a);
            mgr.add_memory(b);
            mgr.add_relationship(&make_edge(a, b, EdgeType::Related))
                .unwrap();
            mgr.save(dir.path()).unwrap();
            let log_len = std::fs::metadata(dir.path().join(super::EDGE_LOG_FILE))
                .unwrap()
                .len();
            assert_eq!(log_len, 0, "save must truncate the edge log");
        }
        {
            // Snapshot alone restores everything; further mutations keep logging.
            let mgr = GraphManager::open(dir.path()).unwrap();
            assert_eq!(mgr.read_graph().outgoing(a).len(), 1);
            let c = MemoryId::new();
            mgr.add_memory(c);
            mgr.add_relationship(&make_edge(a, c, EdgeType::Supports))
                .unwrap();
        }
        {
            let mgr = GraphManager::open(dir.path()).unwrap();
            assert_eq!(
                mgr.read_graph().outgoing(a).len(),
                2,
                "post-save mutations must replay from the fresh log"
            );
        }
    }

    #[test]
    fn test_belief_propagation() {
        let mgr = GraphManager::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        mgr.add_memory(a);
        mgr.add_memory(b);
        mgr.add_relationship(&MemoryEdge {
            source: a,
            target: b,
            edge_type: EdgeType::Caused,
            weight: 1.0,
            created_at: 1000,
            valid_from: None,
            valid_until: None,
            label: None,
        })
        .unwrap();

        let results = mgr.propagate_belief_change(a, 0.5);
        assert!(results.len() >= 2);
    }
}
