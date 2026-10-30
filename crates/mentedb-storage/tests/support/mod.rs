//! Shared helpers for the storage integration tests.
#![allow(dead_code)]

#[allow(clippy::all, unused_imports)]
pub mod v1;

use mentedb_core::MemoryNode;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::AgentId;

/// A typical production memory: a 1024 dimension embedding, about 300
/// characters of text, a couple of tags.
pub fn typical(i: usize) -> MemoryNode {
    let mut seed = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let embedding = (0..1024)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 20_000) as f32 / 10_000.0 - 1.0
        })
        .collect();
    let mut node = MemoryNode::new(
        AgentId::new(),
        MemoryType::Episodic,
        format!(
            "memory {i}: the user mentioned that the quarterly planning review moved to \
             Thursday afternoon, that the draft budget still needs the infrastructure line \
             items, and that they prefer short written summaries over meetings for status \
             updates. Follow up on the open questions about hiring and the launch date {}.",
            i % 97
        ),
        embedding,
    );
    node.tags = vec!["planning".into(), format!("thread:{}", i % 13)];
    node
}

/// Serialized size of a node as the page file engine stored it (JSON).
pub fn json_len(node: &MemoryNode) -> u64 {
    serde_json::to_vec(node).unwrap().len() as u64
}
