//! Integration tests for the full context assembly pipeline.

use mentedb_context::{
    AssemblyConfig, BudgetAllocation, CompactFormat, ContextAssembler, ContextLayout,
    ContextSerializer, DeltaTracker, OutputFormat, ScoredMemory, StructuredFormat, TokenBudget,
};
use mentedb_core::MemoryEdge;
use mentedb_core::MemoryNode;
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::AgentId;

fn make_memory(content: &str, mem_type: MemoryType, salience: f32) -> MemoryNode {
    let mut m = MemoryNode::new(AgentId::new(), mem_type, content.to_string(), vec![]);
    m.salience = salience;
    m.tags = vec!["test".to_string()];
    m
}

fn make_scored(content: &str, score: f32, salience: f32, mem_type: MemoryType) -> ScoredMemory {
    ScoredMemory {
        memory: make_memory(content, mem_type, salience),
        score,
    }
}

fn sample_memories() -> Vec<ScoredMemory> {
    vec![
        make_scored(
            "never use eval in production code",
            0.95,
            0.95,
            MemoryType::AntiPattern,
        ),
        make_scored(
            "user prefers dark mode for all editors",
            0.88,
            0.85,
            MemoryType::Semantic,
        ),
        make_scored(
            "switched database to PostgreSQL last week",
            0.75,
            0.7,
            MemoryType::Episodic,
        ),
        make_scored(
            "always run tests before committing",
            0.60,
            0.5,
            MemoryType::Procedural,
        ),
        make_scored("project uses MIT license", 0.30, 0.3, MemoryType::Semantic),
        make_scored("had coffee this morning", 0.10, 0.1, MemoryType::Episodic),
    ]
}

fn sample_edges(memories: &[ScoredMemory]) -> Vec<MemoryEdge> {
    if memories.len() < 3 {
        return vec![];
    }
    vec![MemoryEdge {
        source: memories[0].memory.id,
        target: memories[1].memory.id,
        edge_type: EdgeType::Related,
        weight: 0.8,
        created_at: 1000,
    }]
}

#[test]
fn test_full_pipeline_structured() {
    let memories = sample_memories();
    let edges = sample_edges(&memories);
    let config = AssemblyConfig {
        token_budget: 4096,
        format: OutputFormat::Structured,
        include_edges: true,
        include_metadata: true,
    };

    let window = ContextAssembler::assemble(memories, edges, &config);

    assert_eq!(window.metadata.total_candidates, 6);
    assert!(window.metadata.included_count > 0);
    assert!(!window.format.is_empty());
    assert!(window.format.contains("Warnings & Corrections"));
    assert!(window.format.contains("never use eval"));
    assert!(window.format.contains("Relationships"));
    assert!(window.total_tokens > 0);
}

#[test]
fn test_full_pipeline_compact() {
    let memories = sample_memories();
    let config = AssemblyConfig {
        token_budget: 4096,
        format: OutputFormat::Compact,
        include_edges: false,
        include_metadata: false,
    };

    let window = ContextAssembler::assemble(memories, vec![], &config);

    assert!(window.format.contains("M|AntiPattern|"));
    assert!(window.format.contains("M|Semantic|"));
    assert!(window.metadata.edges_included == 0);
}

#[test]
fn test_full_pipeline_delta() {
    let mut tracker = DeltaTracker::new();
    let memories = sample_memories();
    let config = AssemblyConfig {
        token_budget: 4096,
        format: OutputFormat::Delta,
        include_edges: false,
        include_metadata: true,
    };

    // Turn 1: everything is new
    let window1 = ContextAssembler::assemble_delta(memories.clone(), vec![], &mut tracker, &config);
    assert!(window1.format.contains("[NEW]"));
    assert_eq!(tracker.last_turn_id, 1);

    // Turn 2: same memories — all unchanged, nothing new
    let window2 = ContextAssembler::assemble_delta(memories.clone(), vec![], &mut tracker, &config);
    assert!(window2.format.contains("[UNCHANGED]"));
    assert_eq!(tracker.last_turn_id, 2);

    // Turn 3: add a new memory, drop the last one
    let mut new_set = memories[..5].to_vec();
    new_set.push(make_scored(
        "new discovery about API",
        0.9,
        0.85,
        MemoryType::Reasoning,
    ));
    let window3 = ContextAssembler::assemble_delta(new_set, vec![], &mut tracker, &config);
    assert!(window3.format.contains("[NEW]"));
    assert!(window3.format.contains("new discovery about API"));
    assert_eq!(tracker.last_turn_id, 3);
}

#[test]
fn test_budget_constrains_assembly() {
    let memories = sample_memories();
    let config = AssemblyConfig {
        token_budget: 15, // very tight budget
        format: OutputFormat::Compact,
        include_edges: false,
        include_metadata: false,
    };

    let window = ContextAssembler::assemble(memories, vec![], &config);
    assert!(window.metadata.excluded_count > 0);
    assert!(window.total_tokens <= 15);
}

#[test]
fn test_layout_zones_populated_correctly() {
    let memories = sample_memories();
    let blocks = ContextLayout::default().arrange(memories);

    assert_eq!(blocks.len(), 5);

    // AntiPattern should be in Opening zone
    let opening = &blocks[0];
    assert!(
        opening
            .memories
            .iter()
            .any(|sm| sm.memory.memory_type == MemoryType::AntiPattern)
    );
}

#[test]
fn test_token_budget_integration() {
    let mut budget = TokenBudget::new(100);
    let alloc = BudgetAllocation::from_total(100);

    assert_eq!(alloc.system, 10);
    assert_eq!(alloc.critical, 25);

    budget.consume("hello world test");
    assert!(budget.remaining() < 100);
    budget.reset();
    assert_eq!(budget.remaining(), 100);
}

#[test]
fn test_serializer_trait_dispatch() {
    let memories = vec![make_scored(
        "trait dispatch test",
        0.9,
        0.9,
        MemoryType::Semantic,
    )];
    let blocks = ContextLayout::default().arrange(memories);

    let compact_output = CompactFormat.serialize(&blocks);
    let structured_output = StructuredFormat.serialize(&blocks);

    assert!(compact_output.contains("M|Semantic|"));
    assert!(structured_output.contains("**[Semantic]**"));
    assert_ne!(compact_output, structured_output);
}
