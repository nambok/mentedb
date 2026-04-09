//! Basic MenteDB usage: open, store, recall, relate, close.
//!
//! Run with: cargo run --example basic_usage

use std::path::Path;

use mentedb::prelude::*;
use mentedb_core::types::AgentId;

fn main() -> MenteResult<()> {
    // Open (or create) a database at a local directory.
    let mut db = MenteDb::open(Path::new("./example_data"))?;

    // Create an agent identity for this example.
    let agent_id = AgentId::new();

    // Store three memories with embeddings.
    // In a real application, embeddings come from a model like OpenAI's
    // text-embedding-3-small. Here we use hand crafted vectors for clarity.

    let mut m1 = MemoryNode::new(
        agent_id,
        MemoryType::Episodic,
        "User prefers dark mode in all editors".to_string(),
        vec![0.1, 0.9, 0.2, 0.0],
    );
    m1.tags = vec!["preferences".into(), "ui".into()];
    m1.salience = 0.8;
    let m1_id = m1.id;

    let mut m2 = MemoryNode::new(
        agent_id,
        MemoryType::Semantic,
        "Rust's ownership model prevents data races at compile time".to_string(),
        vec![0.8, 0.1, 0.7, 0.3],
    );
    m2.tags = vec!["rust".into(), "concurrency".into()];
    m2.salience = 0.9;
    let m2_id = m2.id;

    let mut m3 = MemoryNode::new(
        agent_id,
        MemoryType::Procedural,
        "To deploy, run cargo build --release then copy the binary to /usr/local/bin".to_string(),
        vec![0.3, 0.4, 0.8, 0.6],
    );
    m3.tags = vec!["deployment".into(), "rust".into()];
    m3.salience = 0.7;
    let m3_id = m3.id;

    db.store(m1)?;
    db.store(m2)?;
    db.store(m3)?;
    println!("Stored 3 memories");

    // Recall by vector similarity.
    // Search for memories similar to a "rust concurrency" query embedding.
    let query_embedding = vec![0.7, 0.2, 0.6, 0.3];
    let results = db.recall_similar(&query_embedding, 2)?;
    println!("\nTop 2 similar memories:");
    for (id, score) in &results {
        println!("  id={}, score={:.4}", id, score);
    }

    // Add edges to express relationships between memories.
    let edge = MemoryEdge {
        source: m2_id,
        target: m3_id,
        edge_type: EdgeType::Related,
        weight: 0.7,
        created_at: 0,
        valid_from: None,
        valid_until: None,
        label: None,
    };
    db.relate(edge)?;
    println!("\nRelated memory {} -> {}", m2_id, m3_id);

    // Add a "supports" relationship.
    let support_edge = MemoryEdge {
        source: m1_id,
        target: m2_id,
        edge_type: EdgeType::Supports,
        weight: 0.5,
        created_at: 0,
        valid_from: None,
        valid_until: None,
        label: None,
    };
    db.relate(support_edge)?;
    println!("Added Supports edge {} -> {}", m1_id, m2_id);

    // Close the database, flushing all data to disk.
    db.close()?;
    println!("\nDatabase closed successfully");

    // Clean up example data directory.
    let _ = std::fs::remove_dir_all("./example_data");

    Ok(())
}
