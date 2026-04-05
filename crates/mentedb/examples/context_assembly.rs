//! Demonstrates MenteDB's context assembly system: scored memories,
//! token budgets, attention zones, and output formats.
//!
//! Run with: cargo run --example context_assembly

use mentedb::context::{
    AssemblyConfig, ContextAssembler, OutputFormat, ScoredMemory,
    budget::{BudgetAllocation, TokenBudget, estimate_tokens},
};
use mentedb::prelude::*;
use mentedb_core::types::AgentId;

fn main() {
    println!("=== Token Budget ===\n");
    demonstrate_token_budget();

    println!("\n=== Context Assembly: Compact Format ===\n");
    demonstrate_assembly(OutputFormat::Compact);

    println!("\n=== Context Assembly: Structured Format ===\n");
    demonstrate_assembly(OutputFormat::Structured);

    println!("\n=== Context Assembly: Delta Format ===\n");
    demonstrate_assembly(OutputFormat::Delta);

    println!("\n=== Budget Zone Allocation ===\n");
    demonstrate_zone_allocation();
}

/// Show how token estimation and budgeting works.
fn demonstrate_token_budget() {
    let text = "The user prefers dark mode in all code editors and terminal applications";
    let tokens = estimate_tokens(text);
    println!("Text: '{}'", text);
    println!("Estimated tokens: {}", tokens);

    // Create a budget and consume text against it.
    let mut budget = TokenBudget::new(4096);
    println!("\nBudget: {} total, {} remaining", 4096, budget.remaining());

    let consumed = budget.consume(text);
    println!("Consumed {} tokens", consumed);
    println!("Remaining: {}", budget.remaining());

    let long_text = "x ".repeat(5000);
    println!("\nCan fit 5000 word text? {}", budget.can_fit(&long_text));
}

/// Create scored memories and assemble them with a given format.
fn demonstrate_assembly(format: OutputFormat) {
    let agent_id = AgentId::new();
    let memories = create_sample_memories(agent_id);

    let config = AssemblyConfig {
        token_budget: 2048,
        format,
        include_edges: false,
        include_metadata: true,
    };

    let window = ContextAssembler::assemble(memories, vec![], &config);

    println!("Total tokens used: {}", window.total_tokens);
    println!("Format: {}", window.format);
    println!(
        "Metadata: {} candidates, {} included, {} excluded, {} zones",
        window.metadata.total_candidates,
        window.metadata.included_count,
        window.metadata.excluded_count,
        window.metadata.zones_used,
    );
    println!("Blocks:");
    for block in &window.blocks {
        println!(
            "  Zone: {:?}, memories: {}, tokens: {}",
            block.zone,
            block.memories.len(),
            block.estimated_tokens,
        );
    }
}

/// Show how the budget is divided across attention zones.
fn demonstrate_zone_allocation() {
    let allocation = BudgetAllocation::from_total(4096);
    println!("Total budget: 4096 tokens");
    println!("  System zone:     {} tokens", allocation.system);
    println!("  Critical zone:   {} tokens", allocation.critical);
    println!("  Primary zone:    {} tokens", allocation.primary);
    println!("  Supporting zone: {} tokens", allocation.supporting);
    println!("  Reference zone:  {} tokens", allocation.reference);
    println!("  Sum:             {} tokens", allocation.total());
}

/// Build a set of scored memories spanning different relevance levels.
fn create_sample_memories(agent_id: AgentId) -> Vec<ScoredMemory> {
    // High score, high salience: will be placed in Critical zone.
    let mut m1 = MemoryNode::new(
        agent_id,
        MemoryType::Semantic,
        "The API rate limit is 100 requests per minute per key".to_string(),
        vec![0.9, 0.1, 0.2, 0.0],
    );
    m1.salience = 0.9;

    // Anti-pattern: will be placed in Opening zone (highest attention).
    let mut m2 = MemoryNode::new(
        agent_id,
        MemoryType::AntiPattern,
        "Never store API keys in source code or environment variables visible in logs".to_string(),
        vec![0.1, 0.8, 0.3, 0.1],
    );
    m2.salience = 0.95;

    // Medium score: will be placed in Primary zone.
    let mut m3 = MemoryNode::new(
        agent_id,
        MemoryType::Episodic,
        "User asked about caching strategies for the recommendation engine".to_string(),
        vec![0.4, 0.5, 0.6, 0.2],
    );
    m3.salience = 0.5;

    // Low score: will be placed in Supporting zone.
    let mut m4 = MemoryNode::new(
        agent_id,
        MemoryType::Episodic,
        "Mentioned using Redis for session storage in early conversations".to_string(),
        vec![0.2, 0.3, 0.1, 0.7],
    );
    m4.salience = 0.3;

    // Correction: will be placed in Opening zone alongside anti-patterns.
    let mut m5 = MemoryNode::new(
        agent_id,
        MemoryType::Correction,
        "Corrected: the timeout should be 30s not 300s for health checks".to_string(),
        vec![0.5, 0.5, 0.5, 0.5],
    );
    m5.salience = 0.85;

    vec![
        ScoredMemory {
            memory: m1,
            score: 0.92,
        },
        ScoredMemory {
            memory: m2,
            score: 0.88,
        },
        ScoredMemory {
            memory: m3,
            score: 0.55,
        },
        ScoredMemory {
            memory: m4,
            score: 0.25,
        },
        ScoredMemory {
            memory: m5,
            score: 0.80,
        },
    ]
}
