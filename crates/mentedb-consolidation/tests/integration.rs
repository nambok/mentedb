use mentedb_consolidation::{
    ArchivalDecision, ArchivalPipeline, ConsolidationEngine, DecayConfig, DecayEngine,
    FactExtractor, ForgetEngine, ForgetRequest, MemoryCompressor,
};
use mentedb_core::MemoryNode;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{AgentId, MemoryId, SpaceId};
use std::collections::HashMap;

const DAY_US: u64 = 24 * 3600 * 1_000_000;

fn make_memory(content: &str, embedding: Vec<f32>) -> MemoryNode {
    MemoryNode {
        id: MemoryId::new(),
        agent_id: AgentId::new(),
        memory_type: MemoryType::Episodic,
        embedding,
        content: content.to_string(),
        created_at: 0,
        accessed_at: 0,
        access_count: 0,
        salience: 0.5,
        confidence: 0.8,
        space_id: SpaceId::new(),
        attributes: HashMap::new(),
        tags: Vec::new(),
    }
}

#[test]
fn test_full_pipeline() {
    let now = 100 * DAY_US;
    let agent_id = AgentId::new();
    let space_id = SpaceId::new();

    // Create test memories
    let mut memories: Vec<MemoryNode> = vec![
        {
            let mut m = make_memory(
                "The team uses Rust for the backend. They decided on PostgreSQL for storage",
                vec![0.9, 0.1, 0.0],
            );
            m.agent_id = agent_id;
            m.space_id = space_id;
            m.created_at = now - 50 * DAY_US;
            m.accessed_at = now - 20 * DAY_US;
            m.access_count = 5;
            m.salience = 0.3;
            m
        },
        {
            let mut m = make_memory(
                "Rust is preferred for performance. The team uses async Rust",
                vec![0.85, 0.15, 0.0],
            );
            m.agent_id = agent_id;
            m.space_id = space_id;
            m.created_at = now - 45 * DAY_US;
            m.accessed_at = now - 15 * DAY_US;
            m.access_count = 3;
            m.salience = 0.4;
            m
        },
        {
            let mut m = make_memory("Unrelated topic about weather", vec![0.0, 0.0, 1.0]);
            m.created_at = now - 40 * DAY_US;
            m.accessed_at = now - 35 * DAY_US;
            m.access_count = 0;
            m.salience = 0.02;
            m
        },
    ];

    // 1. Apply decay
    let decay_engine = DecayEngine::new(DecayConfig::default());
    decay_engine.apply_decay_batch(&mut memories, now);
    // Verify salience was updated
    for m in &memories {
        assert!(m.salience > 0.0);
        assert!(m.salience <= 1.0);
    }

    // 2. Find consolidation candidates
    let consolidation_engine = ConsolidationEngine::new();
    let candidates = consolidation_engine.find_candidates(&memories, 2, 0.9);
    // The first two memories have similar embeddings
    assert!(!candidates.is_empty() || memories.len() < 2);

    // 3. Consolidate if candidates found
    if !candidates.is_empty() {
        let cluster: Vec<MemoryNode> = candidates[0]
            .memories
            .iter()
            .filter_map(|id| memories.iter().find(|m| m.id == *id).cloned())
            .collect();
        let consolidated = consolidation_engine.consolidate(&cluster);
        assert_eq!(consolidated.new_type, MemoryType::Semantic);
        assert!(!consolidated.summary.is_empty());
    }

    // 4. Extract facts
    let extractor = FactExtractor::new();
    let mut all_facts = Vec::new();
    for m in &memories {
        let facts = extractor.extract_facts(m);
        all_facts.extend(facts);
    }
    let deduped = extractor.deduplicate_facts(&all_facts);
    assert!(!deduped.is_empty());

    // 5. Compress memories
    let compressor = MemoryCompressor::new();
    let compressed = compressor.compress_batch(&memories);
    assert_eq!(compressed.len(), memories.len());
    for c in &compressed {
        assert!(c.compression_ratio <= 1.0 || c.compression_ratio > 0.0);
    }

    // 6. Evaluate archival
    let pipeline = ArchivalPipeline::default();
    let decisions = pipeline.evaluate_batch(&memories, now);
    assert_eq!(decisions.len(), memories.len());
    // The third memory (very low salience, old, no access) should be archived or deleted
    let weather_decision = &decisions[2].1;
    assert!(
        *weather_decision == ArchivalDecision::Archive
            || *weather_decision == ArchivalDecision::Delete
    );

    // 7. Plan forget
    let forget_engine = ForgetEngine::new();
    let edges = vec![(memories[0].id, memories[1].id)];
    let request = ForgetRequest {
        agent_id: Some(agent_id),
        space_id: None,
        memory_ids: Vec::new(),
        reason: "GDPR deletion request".into(),
        requested_at: now,
    };
    let result = forget_engine.plan_forget(&request, &memories, &edges);
    assert_eq!(result.deleted_memories, 2); // two memories belong to agent_id
    assert_eq!(result.deleted_edges, 1);
    assert!(result.audit_log_entry.contains("FORGET REQUEST"));
}
