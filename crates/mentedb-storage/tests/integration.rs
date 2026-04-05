//! Integration tests for the MenteDB storage engine.

use mentedb_core::MemoryNode;
use mentedb_core::memory::MemoryType;
use mentedb_storage::StorageEngine;
use mentedb_core::types::{AgentId};

#[test]
fn test_store_and_load_memory() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = StorageEngine::open(dir.path()).unwrap();

    let node = MemoryNode::new(
        AgentId::new(),
        MemoryType::Episodic,
        "The user prefers Rust over Go".to_string(),
        vec![0.1, 0.2, 0.3, 0.4],
    );

    let page_id = engine.store_memory(&node).unwrap();
    let loaded = engine.load_memory(page_id).unwrap();

    assert_eq!(node.id, loaded.id);
    assert_eq!(node.content, loaded.content);
    assert_eq!(node.embedding, loaded.embedding);
    assert_eq!(node.memory_type, loaded.memory_type);
    assert_eq!(node.agent_id, loaded.agent_id);
}

#[test]
fn test_multiple_memories() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = StorageEngine::open(dir.path()).unwrap();

    let nodes: Vec<MemoryNode> = (0..10)
        .map(|i| {
            MemoryNode::new(
                AgentId::new(),
                MemoryType::Semantic,
                format!("memory #{i}"),
                vec![i as f32; 4],
            )
        })
        .collect();

    let page_ids: Vec<_> = nodes
        .iter()
        .map(|n| engine.store_memory(n).unwrap())
        .collect();

    for (node, pid) in nodes.iter().zip(page_ids.iter()) {
        let loaded = engine.load_memory(*pid).unwrap();
        assert_eq!(node.id, loaded.id);
        assert_eq!(node.content, loaded.content);
        assert_eq!(node.embedding, loaded.embedding);
    }
}

#[test]
fn test_persist_across_reopen() {
    let dir = tempfile::tempdir().unwrap();

    let node = MemoryNode::new(
        AgentId::new(),
        MemoryType::Procedural,
        "persisted memory".to_string(),
        vec![3.14, 2.72],
    );
    let id = node.id;

    let page_id;
    {
        let mut engine = StorageEngine::open(dir.path()).unwrap();
        page_id = engine.store_memory(&node).unwrap();
        engine.close().unwrap();
    }
    {
        let mut engine = StorageEngine::open(dir.path()).unwrap();
        let loaded = engine.load_memory(page_id).unwrap();
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.content, "persisted memory");
    }
}

#[test]
fn test_checkpoint_and_reload() {
    let dir = tempfile::tempdir().unwrap();

    let node = MemoryNode::new(
        AgentId::new(),
        MemoryType::AntiPattern,
        "don't use global state".to_string(),
        vec![0.0, 1.0],
    );
    let id = node.id;

    let page_id;
    {
        let mut engine = StorageEngine::open(dir.path()).unwrap();
        page_id = engine.store_memory(&node).unwrap();
        engine.checkpoint().unwrap();
        engine.close().unwrap();
    }
    {
        let mut engine = StorageEngine::open(dir.path()).unwrap();
        let loaded = engine.load_memory(page_id).unwrap();
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.content, "don't use global state");
    }
}
