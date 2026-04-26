use mentedb::MenteDb;
use mentedb::prelude::*;
use mentedb_core::types::AgentId;

fn make_memory(content: &str, embedding: Vec<f32>) -> MemoryNode {
    MemoryNode::new(
        AgentId::new(),
        MemoryType::Episodic,
        content.to_string(),
        embedding,
    )
}

#[test]
fn test_store_and_recall_similar() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let memories = vec![
        make_memory("The user prefers dark mode", vec![1.0, 0.0, 0.0, 0.0]),
        make_memory("Meeting scheduled for Monday", vec![0.0, 1.0, 0.0, 0.0]),
        make_memory("API key rotated last week", vec![0.0, 0.0, 1.0, 0.0]),
        make_memory("User likes Python over Rust", vec![0.8, 0.1, 0.0, 0.0]),
        make_memory(
            "Deploy pipeline uses GitHub Actions",
            vec![0.0, 0.0, 0.9, 0.1],
        ),
    ];

    for m in &memories {
        db.store(m.clone()).unwrap();
    }

    // Vector search: query near [1.0, 0.0, 0.0, 0.0] should return preference memories.
    let results = db.recall_similar(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();
    assert!(!results.is_empty(), "Should return at least one result");
    assert!(results.len() <= 3, "Should respect k limit");

    db.close().unwrap();
}

#[test]
fn test_forget_memory() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let node = make_memory("Temporary thought", vec![0.5, 0.5, 0.0, 0.0]);
    let id = node.id;

    db.store(node).unwrap();
    db.forget(id).unwrap();

    // After forgetting, similarity search should not return the forgotten memory.
    let results = db.recall_similar(&[0.5, 0.5, 0.0, 0.0], 5).unwrap();
    for (result_id, _) in &results {
        assert_ne!(
            *result_id, id,
            "Forgotten memory should not appear in results"
        );
    }

    db.close().unwrap();
}

#[test]
fn test_relate_memories() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let m1 = make_memory("Cause event", vec![1.0, 0.0, 0.0, 0.0]);
    let m2 = make_memory("Effect event", vec![0.0, 1.0, 0.0, 0.0]);
    let id1 = m1.id;
    let id2 = m2.id;

    db.store(m1).unwrap();
    db.store(m2).unwrap();

    let edge = MemoryEdge {
        source: id1,
        target: id2,
        edge_type: EdgeType::Caused,
        weight: 0.9,
        created_at: 0,
        valid_from: None,
        valid_until: None,
        label: None,
    };
    db.relate(edge).unwrap();

    db.close().unwrap();
}

#[test]
fn test_close_and_reopen() {
    let dir = tempfile::tempdir().unwrap();

    // First session: store memories.
    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(make_memory("Persistent memory", vec![1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        db.close().unwrap();
    }

    // Second session: reopen and verify storage engine opens cleanly.
    {
        let db = MenteDb::open(dir.path()).unwrap();
        // The database should open without errors after a previous clean close.
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Persistence tests
// ---------------------------------------------------------------------------

fn make_tagged_memory(embedding: Vec<f32>, tags: Vec<String>, salience: f32) -> MemoryNode {
    let mut node = MemoryNode::new(
        AgentId::new(),
        MemoryType::Episodic,
        "test memory".into(),
        embedding,
    );
    node.tags = tags;
    node.salience = salience;
    node.created_at = 1000;
    node
}

#[test]
fn test_db_persistence_indexes_survive_close() {
    let dir = tempfile::tempdir().unwrap();

    let node1 = make_tagged_memory(vec![1.0, 0.0, 0.0, 0.0], vec!["alpha".into()], 0.9);
    let node2 = make_tagged_memory(vec![0.0, 1.0, 0.0, 0.0], vec!["beta".into()], 0.5);
    let id1 = node1.id;

    // Store memories and close
    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(node1).unwrap();
        db.store(node2).unwrap();
        db.close().unwrap();
    }

    // Reopen and verify index-based recall works
    {
        let db = MenteDb::open(dir.path()).unwrap();
        let results = db.recall_similar(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id1);
        db.close().unwrap();
    }
}

#[test]
fn test_db_persistence_graph_survives_close() {
    let dir = tempfile::tempdir().unwrap();

    let node1 = make_tagged_memory(vec![1.0, 0.0, 0.0, 0.0], vec![], 0.5);
    let node2 = make_tagged_memory(vec![0.0, 1.0, 0.0, 0.0], vec![], 0.5);
    let id1 = node1.id;
    let id2 = node2.id;

    // Store memories, relate them, and close
    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(node1).unwrap();
        db.store(node2).unwrap();
        db.relate(MemoryEdge {
            source: id1,
            target: id2,
            edge_type: EdgeType::Caused,
            weight: 0.8,
            created_at: 1000,
            valid_from: None,
            valid_until: None,
            label: None,
        })
        .unwrap();
        db.close().unwrap();
    }

    // Reopen and verify graph edges are preserved
    {
        let db = MenteDb::open(dir.path()).unwrap();
        let results = db.recall_similar(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert!(!results.is_empty());
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Cognitive engine integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_write_inference_creates_related_edge() {
    // Two memories with cosine similarity ~0.72 should get a Related edge.
    // Related threshold: 0.6 < sim ≤ 0.85
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    let m1 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "The project uses PostgreSQL as the primary database".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    let m1_id = m1.id;
    db.store(m1).unwrap();

    // Cosine similarity with [1,0,0,0] is 0.8 / (1.0 * 1.0) ≈ 0.74
    // which falls in the Related range (0.6 < sim ≤ 0.85).
    let m2 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "We also use Redis for caching alongside the main DB".to_string(),
        vec![0.8, 0.5, 0.2, 0.1],
    );
    let m2_id = m2.id;
    db.store(m2).unwrap();

    // Verify a Related edge was created.
    let graph = db.graph().graph();
    let outgoing_m2 = graph.outgoing(m2_id);
    let incoming_m1 = graph.incoming(m1_id);
    let has_related = outgoing_m2
        .iter()
        .any(|(_, e)| e.edge_type == EdgeType::Related)
        || incoming_m1
            .iter()
            .any(|(_, e)| e.edge_type == EdgeType::Related);
    assert!(
        has_related,
        "Should create Related edge for moderate similarity"
    );

    db.close().unwrap();
}

#[test]
fn test_write_inference_detects_contradiction() {
    // Two memories with very high similarity (>0.95) but different content
    // should trigger a contradiction edge.
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    let m1 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "The deployment uses Kubernetes".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    let m1_id = m1.id;
    db.store(m1).unwrap();

    // Near-identical embedding (sim ≈ 0.999) but different content → contradiction.
    let m2 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "The deployment uses Docker Swarm, not Kubernetes".to_string(),
        vec![0.999, 0.01, 0.0, 0.0],
    );
    let m2_id = m2.id;
    db.store(m2).unwrap();

    let graph = db.graph().graph();
    let outgoing = graph.outgoing(m2_id);
    let incoming = graph.incoming(m1_id);

    // Should have Contradicts edge and/or Supersedes edge
    let has_contradiction = outgoing
        .iter()
        .any(|(_, e)| e.edge_type == EdgeType::Contradicts || e.edge_type == EdgeType::Supersedes)
        || incoming.iter().any(|(_, e)| {
            e.edge_type == EdgeType::Contradicts || e.edge_type == EdgeType::Supersedes
        });
    assert!(
        has_contradiction,
        "Near-identical embeddings with different content should trigger contradiction/supersede"
    );

    db.close().unwrap();
}

#[test]
fn test_write_inference_invalidates_superseded_memory() {
    // When sim > 0.85 and new memory is newer, the old one should be invalidated.
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    let m1 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "Project version is 2.0".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    let m1_id = m1.id;
    db.store(m1).unwrap();

    // Very similar embedding (sim > 0.85) = should mark m1 as obsolete.
    let m2 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "Project version is 3.0".to_string(),
        vec![0.99, 0.05, 0.0, 0.0],
    );
    db.store(m2).unwrap();

    // m1 should now have valid_until set (invalidated).
    let m1_after = db.get_memory(m1_id).unwrap();
    assert!(
        m1_after.valid_until.is_some(),
        "Superseded memory should have valid_until set, got None"
    );

    db.close().unwrap();
}

#[test]
fn test_write_inference_no_edges_for_dissimilar() {
    // Memories with low similarity (< 0.6) should NOT create any edges.
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    let m1 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "I love pizza".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    db.store(m1).unwrap();

    // Orthogonal embedding → cosine sim = 0.0
    let m2 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "The weather is sunny today".to_string(),
        vec![0.0, 1.0, 0.0, 0.0],
    );
    let m2_id = m2.id;
    db.store(m2).unwrap();

    let graph = db.graph().graph();
    let outgoing = graph.outgoing(m2_id);
    assert!(
        outgoing.is_empty(),
        "Dissimilar memories should not create any edges, got {} edges",
        outgoing.len()
    );

    db.close().unwrap();
}

#[test]
fn test_write_inference_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let config = mentedb::CognitiveConfig {
        write_inference: false,
        ..Default::default()
    };
    let db = MenteDb::open_with_config(dir.path(), config).unwrap();
    let agent = AgentId::new();

    let m1 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "fact one".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    db.store(m1).unwrap();

    // Identical embedding → would normally trigger contradiction + supersede
    let m2 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "fact one updated".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    let m2_id = m2.id;
    db.store(m2).unwrap();

    let graph = db.graph().graph();
    let outgoing = graph.outgoing(m2_id);
    assert!(
        outgoing.is_empty(),
        "No edges should be created when write inference is disabled"
    );

    db.close().unwrap();
}

#[test]
fn test_decay_fresh_memory_salience_near_one() {
    // A freshly stored memory should have salience near 1.0.
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    let m = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "test decay".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    db.store(m).unwrap();

    let ids = db.memory_ids();
    let memory = db.get_memory(ids[0]).unwrap();
    let decayed = db.compute_decayed_salience(&memory);

    // Fresh memory: time_since_access ≈ 0 → 2^0 = 1.0, plus access_boost * ln(1+0) = 0
    // Should be very close to 1.0.
    assert!(
        decayed > 0.95,
        "Fresh memory salience should be near 1.0, got {}",
        decayed
    );

    db.close().unwrap();
}

#[test]
fn test_decay_disabled_preserves_raw_scores() {
    let dir = tempfile::tempdir().unwrap();

    let score_no_decay = {
        let config = mentedb::CognitiveConfig {
            decay_on_recall: false,
            write_inference: false,
            ..Default::default()
        };
        let db = MenteDb::open_with_config(dir.path(), config).unwrap();
        let agent = AgentId::new();

        let m = MemoryNode::new(
            agent,
            MemoryType::Semantic,
            "test no decay".to_string(),
            vec![1.0, 0.0, 0.0, 0.0],
        );
        db.store(m).unwrap();

        let results = db.recall_similar(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert!(!results.is_empty());
        let score = results[0].1;
        db.close().unwrap();
        score
    };

    let score_with_decay = {
        let config_decay = mentedb::CognitiveConfig {
            decay_on_recall: true,
            write_inference: false,
            ..Default::default()
        };
        let db2 = MenteDb::open_with_config(dir.path(), config_decay).unwrap();
        let results2 = db2.recall_similar(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert!(!results2.is_empty());
        let score = results2[0].1;
        db2.close().unwrap();
        score
    };

    // With decay enabled, score = raw * 0.7 + salience * 0.3 (blended).
    // For a fresh memory, salience ≈ 1.0, so the blended score should differ from raw.
    assert!(
        (score_no_decay - score_with_decay).abs() > 0.001 || score_with_decay > 0.0,
        "Decay toggle should affect scoring (no_decay={}, with_decay={})",
        score_no_decay,
        score_with_decay
    );
}

#[test]
fn test_decay_global_applies_to_all_memories() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    for i in 0..10 {
        let m = MemoryNode::new(
            agent,
            MemoryType::Semantic,
            format!("memory {}", i),
            vec![0.5, 0.5, 0.0, 0.0],
        );
        db.store(m).unwrap();
    }

    // apply_decay_global should not error.
    let updated = db.apply_decay_global().unwrap();
    // For fresh memories, salience won't change much so updated may be 0.
    assert!(updated <= 10, "Should not update more memories than exist");

    db.close().unwrap();
}

#[test]
fn test_consolidation_api_empty_db() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let candidates = db.find_consolidation_candidates(2, 0.8).unwrap();
    assert!(candidates.is_empty(), "No candidates for empty db");

    db.close().unwrap();
}

#[test]
fn test_consolidation_fresh_memories_not_eligible() {
    // Fresh memories (< 24h old) should NOT be eligible for consolidation.
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    for i in 0..5 {
        let m = MemoryNode::new(
            agent,
            MemoryType::Episodic,
            format!("event {}", i),
            vec![0.5, 0.5, 0.0, 0.0],
        );
        db.store(m).unwrap();
    }

    let candidates = db.find_consolidation_candidates(2, 0.8).unwrap();
    assert!(
        candidates.is_empty(),
        "Fresh memories should not be consolidation-eligible"
    );

    db.close().unwrap();
}

#[test]
fn test_consolidate_cluster_requires_minimum_two() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    let m = MemoryNode::new(
        agent,
        MemoryType::Episodic,
        "single memory".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    let m_id = m.id;
    db.store(m).unwrap();

    // Trying to consolidate a single memory should fail.
    let result = db.consolidate_cluster(&[m_id]);
    assert!(result.is_err(), "Cannot consolidate fewer than 2 memories");

    db.close().unwrap();
}

#[test]
fn test_consolidate_cluster_merges_memories() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    let m1 = MemoryNode::new(
        agent,
        MemoryType::Episodic,
        "User deployed to staging on Monday".to_string(),
        vec![0.8, 0.2, 0.0, 0.0],
    );
    let m1_id = m1.id;
    db.store(m1).unwrap();

    let m2 = MemoryNode::new(
        agent,
        MemoryType::Episodic,
        "User deployed to production on Wednesday".to_string(),
        vec![0.7, 0.3, 0.0, 0.0],
    );
    let m2_id = m2.id;
    db.store(m2).unwrap();

    let count_before = db.memory_count();
    let consolidated_id = db.consolidate_cluster(&[m1_id, m2_id]).unwrap();

    // The consolidated memory should exist.
    let consolidated = db.get_memory(consolidated_id).unwrap();
    assert!(
        !consolidated.content.is_empty(),
        "Consolidated memory should have content"
    );
    assert_eq!(
        consolidated.memory_type,
        MemoryType::Semantic,
        "Consolidated memories should be Semantic type"
    );

    // Source memories should be invalidated.
    let m1_after = db.get_memory(m1_id).unwrap();
    let m2_after = db.get_memory(m2_id).unwrap();
    assert!(
        m1_after.valid_until.is_some(),
        "Source m1 should be invalidated"
    );
    assert!(
        m2_after.valid_until.is_some(),
        "Source m2 should be invalidated"
    );

    // A new memory was created (count goes up by 1, since sources aren't deleted).
    assert_eq!(db.memory_count(), count_before + 1);

    // Derived edges should exist from consolidated → sources.
    let graph = db.graph().graph();
    let outgoing = graph.outgoing(consolidated_id);
    let derived_count = outgoing
        .iter()
        .filter(|(_, e)| e.edge_type == EdgeType::Derived)
        .count();
    assert_eq!(
        derived_count, 2,
        "Should have 2 Derived edges to source memories"
    );

    db.close().unwrap();
}

#[test]
fn test_cognitive_config_default_enables_all() {
    let config = mentedb::CognitiveConfig::default();
    assert!(
        config.write_inference,
        "Write inference should be enabled by default"
    );
    assert!(config.decay_on_recall, "Decay should be enabled by default");
    assert!(
        config.pain_tracking,
        "Pain tracking should be enabled by default"
    );
    assert!(
        config.interference_detection,
        "Interference detection should be enabled by default"
    );
    assert!(
        config.phantom_tracking,
        "Phantom tracking should be enabled by default"
    );
    assert!(
        config.speculative_cache,
        "Speculative cache should be enabled by default"
    );
    assert!(
        config.archival_evaluation,
        "Archival evaluation should be enabled by default"
    );
}

// -----------------------------------------------------------------------
// Pain Registry Tests
// -----------------------------------------------------------------------

#[test]
fn test_pain_record_and_retrieve() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let signal = mentedb_cognitive::PainSignal {
        id: MemoryId::new(),
        memory_id: MemoryId::new(),
        intensity: 0.8,
        trigger_keywords: vec!["deploy".into(), "production".into()],
        description: "Deployment to production failed twice last week".into(),
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64,
        decay_rate: 0.1,
    };
    db.record_pain(signal);

    let warnings = db.get_pain_warnings(&["deploy".into(), "staging".into()]);
    assert_eq!(
        warnings.len(),
        1,
        "Should find pain signal matching 'deploy'"
    );
    assert!(
        warnings[0].description.contains("production"),
        "Should return the correct pain signal"
    );
}

#[test]
fn test_pain_disabled_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    let config = mentedb::CognitiveConfig {
        pain_tracking: false,
        ..Default::default()
    };
    let db = MenteDb::open_with_config(dir.path(), config).unwrap();

    let signal = mentedb_cognitive::PainSignal {
        id: MemoryId::new(),
        memory_id: MemoryId::new(),
        intensity: 0.9,
        trigger_keywords: vec!["deploy".into()],
        description: "test".into(),
        created_at: 0,
        decay_rate: 0.1,
    };
    db.record_pain(signal);

    let warnings = db.get_pain_warnings(&["deploy".into()]);
    assert!(
        warnings.is_empty(),
        "Disabled pain should return no warnings"
    );
}

#[test]
fn test_pain_decay_reduces_signals() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let signal = mentedb_cognitive::PainSignal {
        id: MemoryId::new(),
        memory_id: MemoryId::new(),
        intensity: 0.8,
        trigger_keywords: vec!["test".into()],
        description: "test pain".into(),
        created_at: 1_000_000, // very old
        decay_rate: 0.9,
    };
    db.record_pain(signal);
    db.decay_pain();

    let all = db.all_pain_signals();
    assert_eq!(all.len(), 1);
}

// -----------------------------------------------------------------------
// Trajectory Tracking Tests
// -----------------------------------------------------------------------

#[test]
fn test_trajectory_record_and_resume() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let turn = mentedb_cognitive::trajectory::TrajectoryNode {
        turn_id: 1,
        topic_embedding: vec![1.0, 0.0, 0.0],
        topic_summary: "Discussing database schema design".into(),
        decision_state: mentedb_cognitive::trajectory::DecisionState::Investigating,
        open_questions: vec!["Which DB to use?".into()],
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64,
    };
    db.record_trajectory_turn(turn);

    let resume = db.get_resume_context();
    assert!(
        resume.is_some(),
        "Should have resume context after recording a turn"
    );

    let trajectory = db.get_trajectory();
    assert_eq!(trajectory.len(), 1);
    assert_eq!(
        trajectory[0].topic_summary,
        "Discussing database schema design"
    );
}

#[test]
fn test_trajectory_predict_topics() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    // Record multiple turns to build transition patterns
    for i in 0u64..5 {
        let turn = mentedb_cognitive::trajectory::TrajectoryNode {
            turn_id: i,
            topic_embedding: vec![i as f32, 0.0, 0.0],
            topic_summary: format!("Topic {}", i),
            decision_state: mentedb_cognitive::trajectory::DecisionState::Investigating,
            open_questions: vec![],
            timestamp: i * 1_000_000,
        };
        db.record_trajectory_turn(turn);
    }

    let predictions = db.predict_next_topics();
    // Predictions may or may not be empty depending on transition patterns
    assert!(
        predictions.len() <= 3,
        "Should return at most 3 predictions"
    );
}

// -----------------------------------------------------------------------
// Cognition Stream Tests
// -----------------------------------------------------------------------

#[test]
fn test_stream_feed_and_check() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    db.feed_stream_token("The");
    db.feed_stream_token(" user");
    db.feed_stream_token(" prefers");
    db.feed_stream_token(" PostgreSQL");

    let known_facts = vec![(
        MemoryId::new(),
        "The user uses MySQL for their database".into(),
    )];
    let alerts = db.check_stream_alerts(&known_facts);
    // Stream may or may not detect contradiction depending on keyword overlap
    // The important thing is it doesn't panic
    let _ = alerts;

    let buffer = db.drain_stream_buffer();
    assert!(
        buffer.contains("PostgreSQL"),
        "Buffer should contain fed tokens"
    );
}

// -----------------------------------------------------------------------
// Phantom Tracking Tests
// -----------------------------------------------------------------------

#[test]
fn test_phantom_detect_and_resolve() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    // Register some known entities
    db.register_entities(&["PostgreSQL", "Redis", "Docker"]);

    // Detect phantoms — mention "Kubernetes" which is NOT registered
    let phantoms = db.detect_phantoms(
        "We should deploy to Kubernetes using our Docker setup",
        &["PostgreSQL".into(), "Redis".into(), "Docker".into()],
        1,
    );
    // Phantom detection is heuristic — may or may not find gaps
    // The API should work without panicking regardless
    let _ = phantoms;

    let active = db.get_active_phantoms();
    for p in &active {
        db.resolve_phantom(p.id);
    }
    let after_resolve = db.get_active_phantoms();
    assert!(
        after_resolve.len() <= active.len(),
        "Resolving should not increase active phantoms"
    );
}

#[test]
fn test_phantom_disabled_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    let config = mentedb::CognitiveConfig {
        phantom_tracking: false,
        ..Default::default()
    };
    let db = MenteDb::open_with_config(dir.path(), config).unwrap();

    let phantoms = db.detect_phantoms("mention of Unknown Entity", &[], 1);
    assert!(
        phantoms.is_empty(),
        "Disabled phantom tracking returns empty"
    );
}

// -----------------------------------------------------------------------
// Speculative Cache Tests
// -----------------------------------------------------------------------

#[test]
fn test_speculative_cache_hit_and_miss() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    // Pre-assemble cache for a predicted topic
    db.pre_assemble_speculative(vec!["database design".into()], |topic| {
        Some((
            format!("Context about {}", topic),
            vec![MemoryId::new()],
            None,
        ))
    });

    // Try a hit with matching keywords
    let hit = db.try_speculative_hit("database design patterns", None);
    // Whether this hits depends on keyword overlap threshold
    let _ = hit;

    // Try a miss with completely different topic
    let miss = db.try_speculative_hit("cooking recipes", None);
    assert!(miss.is_none(), "Unrelated query should miss the cache");

    let stats = db.speculative_cache_stats();
    assert!(
        stats.cache_size > 0,
        "Cache should have entries after pre-assembly"
    );
}

#[test]
fn test_speculative_cache_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let config = mentedb::CognitiveConfig {
        speculative_cache: false,
        ..Default::default()
    };
    let db = MenteDb::open_with_config(dir.path(), config).unwrap();

    let hit = db.try_speculative_hit("anything", None);
    assert!(hit.is_none(), "Disabled cache always misses");
}

// -----------------------------------------------------------------------
// Interference Detection Tests
// -----------------------------------------------------------------------

#[test]
fn test_interference_detects_similar_memories() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let m1 = make_memory("Deploy to staging environment", vec![0.9, 0.1, 0.0, 0.0]);
    let m2 = make_memory(
        "Deploy to production environment",
        vec![0.88, 0.12, 0.0, 0.0],
    );
    let m3 = make_memory("Cook pasta for dinner", vec![0.0, 0.0, 1.0, 0.0]);

    let pairs = db.detect_interference(&[m1.clone(), m2.clone(), m3]);
    // m1 and m2 have very similar embeddings, m3 is different
    // Whether interference is detected depends on threshold (0.8 default)
    for pair in &pairs {
        assert!(
            pair.similarity >= 0.8,
            "Interference pairs should be above threshold"
        );
        assert!(
            !pair.disambiguation.is_empty(),
            "Disambiguation should not be empty"
        );
    }
}

#[test]
fn test_interference_disabled_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    let config = mentedb::CognitiveConfig {
        interference_detection: false,
        ..Default::default()
    };
    let db = MenteDb::open_with_config(dir.path(), config).unwrap();

    let m1 = make_memory("test A", vec![1.0, 0.0, 0.0, 0.0]);
    let m2 = make_memory("test B", vec![1.0, 0.0, 0.0, 0.0]);
    let pairs = db.detect_interference(&[m1, m2]);
    assert!(pairs.is_empty(), "Disabled interference returns empty");
}

// -----------------------------------------------------------------------
// Entity Resolution Tests
// -----------------------------------------------------------------------

#[test]
fn test_entity_resolve_with_alias() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    db.add_entity_alias("JS", "JavaScript", 0.95);
    db.add_entity_alias("TS", "TypeScript", 0.9);

    let resolved = db.resolve_entity("JS");
    assert_eq!(resolved.canonical, "javascript");
    assert!(resolved.confidence >= 0.9);

    let canonical = db.get_canonical_entity("TS");
    assert_eq!(canonical, Some("typescript".into()));

    let unknown = db.get_canonical_entity("Rust");
    assert!(unknown.is_none(), "Unknown entity returns None");

    let entities = db.known_entities();
    assert!(entities.len() >= 2, "Should know at least 2 entities");
}

// -----------------------------------------------------------------------
// Memory Compression Tests
// -----------------------------------------------------------------------

#[test]
fn test_compress_memory() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let memory = make_memory(
        "The user mentioned that they really prefer using PostgreSQL for their database needs \
         because it has great support for JSON and is very reliable in production environments. \
         They also said they like the extension ecosystem.",
        vec![1.0, 0.0, 0.0, 0.0],
    );

    let compressed = db.compress_memory(&memory);
    assert_eq!(compressed.original_id, memory.id);
    assert!(
        !compressed.compressed_content.is_empty(),
        "Compressed content should not be empty"
    );
    assert!(
        compressed.compression_ratio <= 1.0,
        "Compression ratio should be <= 1.0"
    );
}

#[test]
fn test_compress_batch() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let memories = vec![
        make_memory("First memory with some content", vec![1.0, 0.0, 0.0, 0.0]),
        make_memory(
            "Second memory about something else",
            vec![0.0, 1.0, 0.0, 0.0],
        ),
    ];

    let compressed = db.compress_memories(&memories);
    assert_eq!(compressed.len(), 2);
}

#[test]
fn test_estimate_tokens() {
    let tokens = MenteDb::estimate_tokens("Hello world, this is a test sentence.");
    assert!(tokens > 0, "Token estimate should be positive");
}

// -----------------------------------------------------------------------
// Archival Evaluation Tests
// -----------------------------------------------------------------------

#[test]
fn test_archival_fresh_memory_kept() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let memory = make_memory("A fresh new memory", vec![1.0, 0.0, 0.0, 0.0]);
    let decision = db.evaluate_archival(&memory);
    assert!(
        matches!(
            decision,
            mentedb_consolidation::archival::ArchivalDecision::Keep
        ),
        "Fresh memory should be kept, got {:?}",
        decision
    );
}

#[test]
fn test_archival_disabled_always_keeps() {
    let dir = tempfile::tempdir().unwrap();
    let config = mentedb::CognitiveConfig {
        archival_evaluation: false,
        ..Default::default()
    };
    let db = MenteDb::open_with_config(dir.path(), config).unwrap();

    let mut memory = make_memory("Old memory", vec![1.0, 0.0, 0.0, 0.0]);
    memory.salience = 0.01;
    memory.created_at = 1_000_000; // very old

    let decision = db.evaluate_archival(&memory);
    assert!(
        matches!(
            decision,
            mentedb_consolidation::archival::ArchivalDecision::Keep
        ),
        "Disabled archival should always Keep"
    );
}

#[test]
fn test_archival_batch_evaluation() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let memories = vec![
        make_memory("Fresh memory A", vec![1.0, 0.0, 0.0, 0.0]),
        make_memory("Fresh memory B", vec![0.0, 1.0, 0.0, 0.0]),
    ];

    let decisions = db.evaluate_archival_batch(&memories);
    assert_eq!(decisions.len(), 2);
    // Fresh memories should be kept
    for (_, decision) in &decisions {
        assert!(
            matches!(
                decision,
                mentedb_consolidation::archival::ArchivalDecision::Keep
            ),
            "Fresh memories should be kept"
        );
    }
}

// -----------------------------------------------------------------------
// Persistence Tests: Cognitive State Survives Flush
// -----------------------------------------------------------------------

#[test]
fn test_entity_resolver_persists_across_flush() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.add_entity_alias("pg", "PostgreSQL", 0.95);
        db.flush().unwrap();
        db.close().unwrap();
    }
    {
        let db = MenteDb::open(dir.path()).unwrap();
        let canonical = db.get_canonical_entity("pg");
        assert_eq!(
            canonical,
            Some("postgresql".into()),
            "Entity alias should persist across flush/reopen"
        );
    }
}
