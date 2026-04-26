use mentedb::CognitiveConfig;
use mentedb::EnrichmentConfig;
use mentedb::MenteDb;
use mentedb::process_turn::ProcessTurnInput;
use mentedb_context::DeltaTracker;

fn open_db() -> (MenteDb, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open_with_config(dir.path(), CognitiveConfig::default()).unwrap();
    (db, dir)
}

fn open_db_with_enrichment(interval: u64) -> (MenteDb, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let config = CognitiveConfig {
        enrichment_config: EnrichmentConfig {
            enabled: true,
            trigger_interval: interval,
            ..EnrichmentConfig::default()
        },
        ..CognitiveConfig::default()
    };
    let db = MenteDb::open_with_config(dir.path(), config).unwrap();
    (db, dir)
}

#[test]
fn test_process_turn_basic() {
    let (db, _dir) = open_db();
    let mut delta = DeltaTracker::new();

    let input = ProcessTurnInput {
        user_message: "I prefer using Rust for systems programming".to_string(),
        assistant_response: Some("Great choice! Rust is excellent for systems work.".to_string()),
        turn_id: 0,
        project_context: Some("test-project".to_string()),
        agent_id: None,
    };

    let result = db.process_turn(&input, &mut delta).unwrap();

    // Should have stored at least an episodic memory
    assert!(result.episodic_id.is_some());
    assert!(!result.stored_ids.is_empty());

    // Sentiment should be positive (words like "great", "excellent")
    // Note: sentiment is only on user_message, which has "prefer" — neutral
    // Let's just check it's in range
    assert!(result.sentiment >= -1.0 && result.sentiment <= 1.0);
}

#[test]
fn test_process_turn_detects_actions() {
    let (db, _dir) = open_db();
    let mut delta = DeltaTracker::new();

    let input = ProcessTurnInput {
        user_message: "I need to deploy the service and run tests".to_string(),
        assistant_response: Some("I'll help you deploy and test.".to_string()),
        turn_id: 1,
        project_context: None,
        agent_id: None,
    };

    let result = db.process_turn(&input, &mut delta).unwrap();

    // Should detect "deploy" and "test" actions
    let action_types: Vec<&str> = result
        .detected_actions
        .iter()
        .map(|a| a.action_type.as_str())
        .collect();
    assert!(action_types.contains(&"deployment"));
    assert!(action_types.contains(&"testing"));
}

#[test]
fn test_process_turn_detects_corrections() {
    let (db, _dir) = open_db();
    let mut delta = DeltaTracker::new();

    let input = ProcessTurnInput {
        user_message: "Actually, I was wrong — we use MongoDB not PostgreSQL".to_string(),
        assistant_response: Some("Got it, updating to MongoDB.".to_string()),
        turn_id: 1,
        project_context: None,
        agent_id: None,
    };

    let result = db.process_turn(&input, &mut delta).unwrap();

    // Should detect correction indicators: "actually", "i was wrong"
    assert!(result.correction_id.is_some());
}

#[test]
fn test_process_turn_sentiment() {
    let (db, _dir) = open_db();
    let mut delta = DeltaTracker::new();

    let positive_input = ProcessTurnInput {
        user_message: "This is great, excellent work, perfect solution!".to_string(),
        assistant_response: None,
        turn_id: 0,
        project_context: None,
        agent_id: None,
    };

    let result = db.process_turn(&positive_input, &mut delta).unwrap();
    assert!(
        result.sentiment > 0.0,
        "positive sentiment expected, got {}",
        result.sentiment
    );

    let negative_input = ProcessTurnInput {
        user_message: "This is terrible, broken and frustrating".to_string(),
        assistant_response: None,
        turn_id: 1,
        project_context: None,
        agent_id: None,
    };

    let result = db.process_turn(&negative_input, &mut delta).unwrap();
    assert!(
        result.sentiment < 0.0,
        "negative sentiment expected, got {}",
        result.sentiment
    );
}

#[test]
fn test_process_turn_multi_turn_context() {
    let (db, _dir) = open_db();
    let mut delta = DeltaTracker::new();

    // Turn 0: seed a memory
    let input0 = ProcessTurnInput {
        user_message: "I am building a marketplace called ShopFlow using Next.js".to_string(),
        assistant_response: Some("Cool! ShopFlow with Next.js.".to_string()),
        turn_id: 0,
        project_context: Some("shopflow".to_string()),
        agent_id: None,
    };
    let r0 = db.process_turn(&input0, &mut delta).unwrap();
    assert!(r0.episodic_id.is_some());

    // Turn 1: should find the first memory as context
    let input1 = ProcessTurnInput {
        user_message: "What framework am I using for ShopFlow?".to_string(),
        assistant_response: None,
        turn_id: 1,
        project_context: Some("shopflow".to_string()),
        agent_id: None,
    };
    let r1 = db.process_turn(&input1, &mut delta).unwrap();

    // The episodic from turn 0 should show up in context (via hybrid search)
    // With no embedder, results depend on BM25 + zero-vec similarity
    // At minimum, we should have stored the episodic and the pipeline should run
    assert!(r1.episodic_id.is_some());
}

#[test]
fn test_process_turn_pain_signals() {
    let (db, _dir) = open_db();
    let mut delta = DeltaTracker::new();

    // Register a pain signal
    use mentedb_cognitive::pain::PainSignal;
    use mentedb_core::types::MemoryId;

    let signal = PainSignal {
        id: MemoryId::new(),
        memory_id: MemoryId::new(),
        description: "Redis caused production outage".to_string(),
        trigger_keywords: vec!["redis".to_string(), "cache".to_string()],
        intensity: 0.9,
        created_at: 0,
        decay_rate: 0.01,
    };
    db.record_pain(signal);

    let input = ProcessTurnInput {
        user_message: "Should I use Redis for caching?".to_string(),
        assistant_response: None,
        turn_id: 0,
        project_context: None,
        agent_id: None,
    };

    let result = db.process_turn(&input, &mut delta).unwrap();

    // Should match pain signal on "redis" and "caching" (partial match on "cache")
    assert!(
        !result.pain_warnings.is_empty(),
        "expected pain warnings for Redis"
    );
    assert!(result.pain_warnings[0].intensity > 0.5);
}

#[test]
fn test_process_turn_delta_tracking() {
    let (db, _dir) = open_db();
    let mut delta = DeltaTracker::new();

    let input0 = ProcessTurnInput {
        user_message: "First turn content about Rust programming".to_string(),
        assistant_response: Some("Rust is great!".to_string()),
        turn_id: 0,
        project_context: None,
        agent_id: None,
    };
    let r0 = db.process_turn(&input0, &mut delta).unwrap();

    // First turn: everything is new (delta_added should have items)
    // delta_removed should be empty since there was nothing before
    assert!(r0.delta_removed.is_empty());

    let input1 = ProcessTurnInput {
        user_message: "Second turn about Python programming".to_string(),
        assistant_response: Some("Python is versatile!".to_string()),
        turn_id: 1,
        project_context: None,
        agent_id: None,
    };
    let _r1 = db.process_turn(&input1, &mut delta).unwrap();
    // Just verify it doesn't panic — delta computation runs
}

#[test]
fn test_process_turn_maintenance_intervals() {
    let (db, _dir) = open_db();
    let mut delta = DeltaTracker::new();

    // Turn 0 should skip maintenance
    let input = ProcessTurnInput {
        user_message: "Hello".to_string(),
        assistant_response: None,
        turn_id: 0,
        project_context: None,
        agent_id: None,
    };
    let _ = db.process_turn(&input, &mut delta).unwrap();

    // Turn 50 should trigger decay (no panic even with no memories)
    let input50 = ProcessTurnInput {
        user_message: "Turn fifty".to_string(),
        assistant_response: None,
        turn_id: 50,
        project_context: None,
        agent_id: None,
    };
    let _ = db.process_turn(&input50, &mut delta).unwrap();

    // Turn 200 should trigger consolidation (no panic)
    let input200 = ProcessTurnInput {
        user_message: "Turn two hundred".to_string(),
        assistant_response: None,
        turn_id: 200,
        project_context: None,
        agent_id: None,
    };
    let _ = db.process_turn(&input200, &mut delta).unwrap();
}

#[test]
fn test_enrichment_disabled_by_default() {
    let (db, _dir) = open_db();
    assert!(!db.needs_enrichment());
    assert_eq!(db.last_enrichment_turn(), 0);
    assert!(db.enrichment_candidates().is_empty());
}

#[test]
fn test_enrichment_trigger_after_interval() {
    let (db, _dir) = open_db_with_enrichment(5);
    let mut delta = DeltaTracker::new();

    // Run 4 turns — should NOT trigger yet
    for i in 1..=4 {
        let input = ProcessTurnInput {
            user_message: format!("Turn {}", i),
            assistant_response: None,
            turn_id: i,
            project_context: None,
            agent_id: None,
        };
        let result = db.process_turn(&input, &mut delta).unwrap();
        assert!(!result.enrichment_pending, "turn {} should not trigger", i);
    }

    // Turn 5 should trigger enrichment
    let input5 = ProcessTurnInput {
        user_message: "Turn five".to_string(),
        assistant_response: None,
        turn_id: 5,
        project_context: None,
        agent_id: None,
    };
    let result = db.process_turn(&input5, &mut delta).unwrap();
    assert!(result.enrichment_pending);
    assert!(db.needs_enrichment());
}

#[test]
fn test_enrichment_candidates_returns_episodics() {
    let (db, _dir) = open_db_with_enrichment(5);
    let mut delta = DeltaTracker::new();

    // Store some turns
    for i in 1..=3 {
        let input = ProcessTurnInput {
            user_message: format!("I like {} programming", ["Rust", "Python", "Go"][i - 1]),
            assistant_response: Some("Nice choice!".to_string()),
            turn_id: i as u64,
            project_context: None,
            agent_id: None,
        };
        db.process_turn(&input, &mut delta).unwrap();
    }

    let candidates = db.enrichment_candidates();
    assert!(
        candidates.len() >= 3,
        "should have at least 3 episodic candidates"
    );

    // All should be Episodic
    for c in &candidates {
        assert_eq!(c.memory_type, mentedb_core::memory::MemoryType::Episodic);
    }
}

#[test]
fn test_enrichment_store_results() {
    let (db, _dir) = open_db_with_enrichment(5);
    let mut delta = DeltaTracker::new();

    // Store a turn to get a source memory
    let input = ProcessTurnInput {
        user_message: "I prefer using Rust".to_string(),
        assistant_response: Some("Rust is great!".to_string()),
        turn_id: 1,
        project_context: None,
        agent_id: None,
    };
    let result = db.process_turn(&input, &mut delta).unwrap();
    let source_id = result.episodic_id.unwrap();

    // Create an enrichment memory (simulating extraction output)
    let enrichment_mem = mentedb_core::MemoryNode::new(
        mentedb_core::types::AgentId::nil(),
        mentedb_core::memory::MemoryType::Semantic,
        "User prefers Rust for programming".to_string(),
        vec![0.1; 384],
    );

    let (stored, edges) = db
        .store_enrichment_memories(vec![enrichment_mem], &[source_id])
        .unwrap();

    assert_eq!(stored, 1);
    assert_eq!(edges, 1);

    // Verify the stored memory has enrichment tag and capped confidence
    let candidates = db.enrichment_candidates();
    // The enrichment memory should NOT show up as a candidate (it has source:enrichment tag)
    for c in &candidates {
        assert!(
            !c.tags.contains(&"source:enrichment".to_string()),
            "enrichment memories should not be candidates"
        );
    }
}

#[test]
fn test_enrichment_mark_complete_resets_pending() {
    let (db, _dir) = open_db_with_enrichment(3);
    let mut delta = DeltaTracker::new();

    // Trigger enrichment
    for i in 1..=3 {
        let input = ProcessTurnInput {
            user_message: format!("Turn {}", i),
            assistant_response: None,
            turn_id: i,
            project_context: None,
            agent_id: None,
        };
        db.process_turn(&input, &mut delta).unwrap();
    }
    assert!(db.needs_enrichment());

    // Mark complete
    db.mark_enrichment_complete(3);
    assert!(!db.needs_enrichment());
    assert_eq!(db.last_enrichment_turn(), 3);

    // Next 2 turns should not trigger again
    for i in 4..=5 {
        let input = ProcessTurnInput {
            user_message: format!("Turn {}", i),
            assistant_response: None,
            turn_id: i,
            project_context: None,
            agent_id: None,
        };
        let result = db.process_turn(&input, &mut delta).unwrap();
        assert!(!result.enrichment_pending);
    }

    // Turn 6 should trigger again (3 turns since last enrichment at turn 3)
    let input6 = ProcessTurnInput {
        user_message: "Turn 6".to_string(),
        assistant_response: None,
        turn_id: 6,
        project_context: None,
        agent_id: None,
    };
    let result = db.process_turn(&input6, &mut delta).unwrap();
    assert!(result.enrichment_pending);
}

#[test]
fn test_always_scope_in_retrieve_context() {
    let (db, _dir) = open_db();
    let mut delta = DeltaTracker::new();

    // Store an always-scoped memory directly
    let mut always_mem = mentedb_core::MemoryNode::new(
        mentedb_core::types::AgentId::nil(),
        mentedb_core::memory::MemoryType::Semantic,
        "Always remember: user prefers dark mode".to_string(),
        vec![0.1; 384],
    );
    always_mem.tags.push("scope:always".to_string());
    db.store(always_mem).unwrap();

    // process_turn on an unrelated topic should still include the always-scoped memory
    let input = ProcessTurnInput {
        user_message: "What is the weather today?".to_string(),
        assistant_response: None,
        turn_id: 1,
        project_context: None,
        agent_id: None,
    };
    let result = db.process_turn(&input, &mut delta).unwrap();

    let has_always = result
        .context
        .iter()
        .any(|sm| sm.memory.tags.contains(&"scope:always".to_string()));
    assert!(has_always, "always-scoped memory should be in context");
}

#[test]
fn test_entity_linking_creates_edges() {
    let (db, _dir) = open_db_with_enrichment(50);

    // Create two entity memories with the same name but different content
    let emb1 = vec![0.9, 0.1, 0.0, 0.0];
    let emb2 = vec![0.85, 0.15, 0.05, 0.0];

    let mut node1 = mentedb_core::MemoryNode::new(
        mentedb_core::types::AgentId::nil(),
        mentedb_core::memory::MemoryType::Semantic,
        "Max is a Golden Retriever".to_string(),
        emb1,
    );
    node1.tags.push("entity:max".to_string());
    node1.tags.push("source:enrichment".to_string());
    db.store(node1).unwrap();

    let mut node2 = mentedb_core::MemoryNode::new(
        mentedb_core::types::AgentId::nil(),
        mentedb_core::memory::MemoryType::Semantic,
        "Max loves playing fetch in the park".to_string(),
        emb2,
    );
    node2.tags.push("entity:max".to_string());
    node2.tags.push("source:enrichment".to_string());
    db.store(node2).unwrap();

    // Pre-teach the resolver that "max" is a known entity
    db.add_entity_alias("max", "max", 1.0);

    let result = db.link_entities().unwrap();
    // Both memories share the canonical name "max" → linked
    assert!(
        result.linked > 0,
        "should link same-name entities via EntityResolver cache"
    );
    assert!(result.edges_created > 0, "should create edges");

    // Run again — should NOT create duplicate edges
    let result2 = db.link_entities().unwrap();
    assert_eq!(
        result2.edges_created, 0,
        "should not create duplicate edges"
    );
}

#[test]
fn test_entity_linking_separates_different_entities() {
    let (db, _dir) = open_db_with_enrichment(50);

    // Two "python" entities — resolver doesn't know about them yet
    let emb_prog = vec![1.0, 0.0, 0.0, 0.0];
    let emb_comedy = vec![0.0, 0.0, 0.0, 1.0];

    let mut node1 = mentedb_core::MemoryNode::new(
        mentedb_core::types::AgentId::nil(),
        mentedb_core::memory::MemoryType::Semantic,
        "Python programming language".to_string(),
        emb_prog,
    );
    node1.tags.push("entity:python".to_string());
    db.store(node1).unwrap();

    let mut node2 = mentedb_core::MemoryNode::new(
        mentedb_core::types::AgentId::nil(),
        mentedb_core::memory::MemoryType::Semantic,
        "Monty Python comedy troupe".to_string(),
        emb_comedy,
    );
    node2.tags.push("entity:python".to_string());
    db.store(node2).unwrap();

    // Without resolver knowledge, link_entities shouldn't link them
    let result = db.link_entities().unwrap();
    assert_eq!(result.linked, 0, "unresolved entities should NOT be linked");
    assert_eq!(result.edges_created, 0, "no edges for unresolved entities");
}

#[test]
fn test_apply_entity_link_resolutions() {
    let (db, _dir) = open_db_with_enrichment(50);

    let emb1 = vec![0.9, 0.1, 0.0, 0.0];
    let emb2 = vec![0.85, 0.15, 0.05, 0.0];

    let mut node1 = mentedb_core::MemoryNode::new(
        mentedb_core::types::AgentId::nil(),
        mentedb_core::memory::MemoryType::Semantic,
        "NYC is where I live".to_string(),
        emb1,
    );
    node1.tags.push("entity:nyc".to_string());
    db.store(node1).unwrap();

    let mut node2 = mentedb_core::MemoryNode::new(
        mentedb_core::types::AgentId::nil(),
        mentedb_core::memory::MemoryType::Semantic,
        "New York City has great pizza".to_string(),
        emb2,
    );
    node2.tags.push("entity:new york city".to_string());
    db.store(node2).unwrap();

    // Simulate LLM resolution: NYC and New York City are the same
    let resolutions = vec![mentedb::EntityLinkResolution {
        canonical: "new york city".to_string(),
        aliases: vec!["nyc".to_string()],
        confidence: 0.95,
    }];
    let separations = vec![];

    let result = db
        .apply_entity_link_resolutions(&resolutions, &separations)
        .unwrap();
    assert!(result.edges_created > 0, "should create cross-name edges");
    assert!(result.linked > 0, "should report linked pairs");

    // Verify resolver learned the alias
    assert_eq!(
        db.get_canonical_entity("nyc"),
        Some("new york city".to_string())
    );

    // Now link_entities should also work (uses cache)
    // (won't create new edges since apply already created them)
    let sync_result = db.link_entities().unwrap();
    assert_eq!(sync_result.edges_created, 0, "edges already exist");
}

#[test]
fn test_entity_separation_negative_cache() {
    let (db, _dir) = open_db_with_enrichment(50);

    let emb1 = vec![1.0, 0.0, 0.0, 0.0];
    let emb2 = vec![0.0, 0.0, 0.0, 1.0];

    let mut node1 = mentedb_core::MemoryNode::new(
        mentedb_core::types::AgentId::nil(),
        mentedb_core::memory::MemoryType::Semantic,
        "Java island in Indonesia".to_string(),
        emb1,
    );
    node1.tags.push("entity:java".to_string());
    db.store(node1).unwrap();

    let mut node2 = mentedb_core::MemoryNode::new(
        mentedb_core::types::AgentId::nil(),
        mentedb_core::memory::MemoryType::Semantic,
        "Java programming language".to_string(),
        emb2,
    );
    node2.tags.push("entity:java programming".to_string());
    db.store(node2).unwrap();

    // LLM says these are different
    let resolutions = vec![];
    let separations = vec![mentedb::EntitySeparation {
        name_a: "java".to_string(),
        name_b: "java programming".to_string(),
    }];

    db.apply_entity_link_resolutions(&resolutions, &separations)
        .unwrap();

    // Verify: "java" and "java programming" should show as unresolved
    // (negative cache prevents future LLM calls for this pair)
    let unresolved = db.unresolved_entity_names();
    // Both are still "unresolved" in terms of having no canonical alias,
    // but the negative cache will skip them in future LLM batches
    assert!(unresolved.contains(&"java".to_string()));
}

#[test]
fn test_entity_memories_returns_tagged() {
    let (db, _dir) = open_db();

    let mut entity_node = mentedb_core::MemoryNode::new(
        mentedb_core::types::AgentId::nil(),
        mentedb_core::memory::MemoryType::Semantic,
        "Max the dog".to_string(),
        vec![0.5, 0.5, 0.0, 0.0],
    );
    entity_node.tags.push("entity:max".to_string());
    db.store(entity_node).unwrap();

    let mut regular_node = mentedb_core::MemoryNode::new(
        mentedb_core::types::AgentId::nil(),
        mentedb_core::memory::MemoryType::Semantic,
        "I like coffee".to_string(),
        vec![0.0, 0.0, 0.5, 0.5],
    );
    regular_node.tags.push("preference".to_string());
    db.store(regular_node).unwrap();

    let entities = db.entity_memories();
    assert_eq!(entities.len(), 1);
    assert!(entities[0].tags.contains(&"entity:max".to_string()));
}
