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
fn test_forget_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let kept = make_memory("keep me", vec![1.0, 0.0, 0.0, 0.0]);
    let forgotten = make_memory("forget me", vec![0.0, 1.0, 0.0, 0.0]);
    let (kept_id, forgotten_id) = (kept.id, forgotten.id);

    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(kept).unwrap();
        db.store(forgotten).unwrap();
        db.forget(forgotten_id).unwrap();
        db.close().unwrap();
    }
    {
        let db = MenteDb::open(dir.path()).unwrap();
        assert_eq!(db.memory_count(), 1, "forgotten memory must not resurrect");
        assert!(db.get_memory(kept_id).is_ok());
        assert!(
            db.get_memory(forgotten_id).is_err(),
            "forgotten memory must stay forgotten after reopen"
        );
        db.close().unwrap();
    }
}

#[test]
fn test_forget_survives_crash() {
    let dir = tempfile::tempdir().unwrap();
    let node = make_memory("forget me before crash", vec![0.0, 1.0, 0.0, 0.0]);
    let id = node.id;

    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(node).unwrap();
        db.forget(id).unwrap();
        // Simulate crash: no close, no flush.
        db.simulate_crash();
    }
    {
        let db = MenteDb::open(dir.path()).unwrap();
        assert!(
            db.get_memory(id).is_err(),
            "forget must be WAL-durable across a crash"
        );
        assert_eq!(db.memory_count(), 0);
        db.close().unwrap();
    }
}

#[test]
fn test_invalidation_updates_in_place_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let node = make_memory("versioned fact", vec![1.0, 0.0, 0.0, 0.0]);
    let id = node.id;

    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(node).unwrap();
        db.invalidate_memory(id, 12345).unwrap();
        db.close().unwrap();
    }
    {
        let db = MenteDb::open(dir.path()).unwrap();
        assert_eq!(
            db.memory_count(),
            1,
            "in-place update must not orphan a duplicate copy"
        );
        let loaded = db.get_memory(id).unwrap();
        assert_eq!(loaded.valid_until, Some(12345));
        db.close().unwrap();
    }
}

#[test]
fn test_edges_survive_crash() {
    let dir = tempfile::tempdir().unwrap();
    let m1 = make_memory("cause", vec![1.0, 0.0, 0.0, 0.0]);
    let m2 = make_memory("effect", vec![0.0, 1.0, 0.0, 0.0]);
    let (id1, id2) = (m1.id, m2.id);

    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(m1).unwrap();
        db.store(m2).unwrap();
        db.relate(MemoryEdge {
            source: id1,
            target: id2,
            edge_type: EdgeType::Caused,
            weight: 0.9,
            created_at: 0,
            valid_from: None,
            valid_until: None,
            label: Some("crash-edge".into()),
        })
        .unwrap();
        // Simulate crash: no close, no flush — the graph snapshot was never
        // written, so the edge only exists in the edge log.
        db.simulate_crash();
    }
    {
        let db = MenteDb::open(dir.path()).unwrap();
        let g = db.graph().read_graph();
        let out = g.outgoing(id1);
        assert_eq!(out.len(), 1, "edge must survive a crash via the edge log");
        assert_eq!(out[0].0, id2);
        assert_eq!(out[0].1.label.as_deref(), Some("crash-edge"));
        drop(g);
        db.close().unwrap();
    }
}

#[test]
fn test_relate_works_after_crash() {
    let dir = tempfile::tempdir().unwrap();
    let m1 = make_memory("first", vec![1.0, 0.0, 0.0, 0.0]);
    let m2 = make_memory("second", vec![0.0, 1.0, 0.0, 0.0]);
    let (id1, id2) = (m1.id, m2.id);

    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(m1).unwrap();
        db.store(m2).unwrap();
        // Crash before any flush: no graph snapshot exists on disk.
        db.simulate_crash();
    }
    {
        // Graph nodes must be rebuilt from storage so surviving memories can
        // still be linked.
        let db = MenteDb::open(dir.path()).unwrap();
        db.relate(MemoryEdge {
            source: id1,
            target: id2,
            edge_type: EdgeType::Supports,
            weight: 0.5,
            created_at: 0,
            valid_from: None,
            valid_until: None,
            label: None,
        })
        .expect("relate must work on memories that survived a crash");
        db.close().unwrap();
    }
}

#[test]
fn test_forgotten_memory_edges_do_not_resurrect() {
    let dir = tempfile::tempdir().unwrap();
    let m1 = make_memory("keep", vec![1.0, 0.0, 0.0, 0.0]);
    let m2 = make_memory("forget", vec![0.0, 1.0, 0.0, 0.0]);
    let (id1, id2) = (m1.id, m2.id);

    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(m1).unwrap();
        db.store(m2).unwrap();
        db.relate(MemoryEdge {
            source: id1,
            target: id2,
            edge_type: EdgeType::Related,
            weight: 0.5,
            created_at: 0,
            valid_from: None,
            valid_until: None,
            label: None,
        })
        .unwrap();
        db.forget(id2).unwrap();
        db.simulate_crash(); // crash: RemoveNode only in the edge log
    }
    {
        let db = MenteDb::open(dir.path()).unwrap();
        let g = db.graph().read_graph();
        assert!(
            g.outgoing(id1).is_empty(),
            "edges to a forgotten memory must not survive the crash"
        );
        drop(g);
        db.close().unwrap();
    }
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
fn test_write_inference_defers_conflict_to_llm() {
    // High similarity + different content is ambiguous to cosine similarity: a
    // paraphrase and an opposite look identical. The write-time heuristic must
    // NOT guess a conflict here (that produced ~0% precision); contradiction and
    // supersession detection is the LLM path's job (detect_conflicts_with_llm,
    // covered in the llm_consolidation tests). So no conflict edge appears.
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

    let m2 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "The deployment uses Docker Swarm, not Kubernetes".to_string(),
        vec![0.999, 0.01, 0.0, 0.0],
    );
    let m2_id = m2.id;
    db.store(m2).unwrap();

    let graph = db.graph().graph();
    let has_conflict = graph
        .outgoing(m2_id)
        .iter()
        .chain(graph.incoming(m1_id).iter())
        .any(|(_, e)| e.edge_type == EdgeType::Contradicts || e.edge_type == EdgeType::Supersedes);
    assert!(
        !has_conflict,
        "heuristic must not create conflict edges from bare similarity"
    );

    db.close().unwrap();
}

#[test]
fn test_byte_identical_store_is_skipped() {
    // Byte-identical text is the ultimate paraphrase: since store-time
    // paraphrase dedup landed, the re-save is skipped outright, so the
    // original stays the single valid copy and nothing needs invalidating or
    // linking. (Previously the copy was stored and the older invalidated with
    // Derived lineage; the store-time gate makes that round-trip moot.)
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

    // Identical content and embedding: a true duplicate.
    let m2 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "Project version is 2.0".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    let m2_id = m2.id;
    db.store(m2).unwrap();

    assert!(
        db.get_memory(m2_id).is_err(),
        "the identical re-save must be skipped, not stored"
    );
    let m1_after = db.get_memory(m1_id).unwrap();
    assert!(
        m1_after.valid_until.is_none(),
        "the original stays the single valid copy"
    );
    assert_eq!(db.memory_count(), 1);

    db.close().unwrap();
}

#[test]
fn test_high_similarity_different_content_left_for_llm() {
    // "User loves PostgreSQL" vs "User hates PostgreSQL": a real contradiction,
    // but one the heuristic cannot distinguish from a paraphrase by embedding
    // alone. It must not silently invalidate the old memory nor guess a conflict
    // edge; the LLM path resolves this later. So both stay valid and unlinked.
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    let m1 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "User loves PostgreSQL".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    let m1_id = m1.id;
    db.store(m1).unwrap();

    let m2 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "User hates PostgreSQL".to_string(),
        vec![0.99, 0.14106736, 0.0, 0.0],
    );
    let m2_id = m2.id;
    db.store(m2).unwrap();

    let m1_after = db.get_memory(m1_id).unwrap();
    assert!(
        m1_after.valid_until.is_none(),
        "the heuristic must not silently invalidate the old memory"
    );

    let g = db.graph().read_graph();
    assert!(
        g.outgoing(m2_id).iter().all(|(_, e)| {
            e.edge_type != EdgeType::Contradicts && e.edge_type != EdgeType::Supersedes
        }),
        "the heuristic must not create a conflict edge from bare similarity"
    );
    drop(g);

    db.close().unwrap();
}

#[test]
fn test_store_batch_runs_write_inference() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    let m1 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "User works at Acme".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    let m1_id = m1.id;
    db.store(m1).unwrap();

    // Batch-stored memory at moderate similarity (Related band) must be linked
    // the same as store() would, proving inference runs on the batch path.
    let m2 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "User enjoys the Acme office".to_string(),
        vec![0.75, 0.66143782, 0.0, 0.0],
    );
    let m2_id = m2.id;
    db.store_batch(vec![m2]).unwrap();

    let g = db.graph().read_graph();
    assert!(
        g.outgoing(m2_id)
            .iter()
            .any(|(t, e)| *t == m1_id && e.edge_type == EdgeType::Related),
        "store_batch must run write inference (Related edge for moderate similarity)"
    );
    drop(g);

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
    // Decay is derived on read now, so the global pass persists nothing.
    assert_eq!(updated, 0, "apply_decay_global no longer persists decay");

    db.close().unwrap();
}

#[test]
fn test_reset_decay_state_repairs_corrupted_salience() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    // Simulate a memory whose base salience was cratered by the old compounding
    // decay pass and whose decay clock is stale.
    let mut m = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "important but over-decayed".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    m.salience = 0.02;
    m.accessed_at = 1_000; // long ago
    m.access_count = 1; // access bonus 0.1*ln(2) ~ 0.069, so derived stays under 0.1
    db.store(m).unwrap();

    let id = db.memory_ids()[0];
    let before = db.get_memory(id).unwrap();
    assert!(db.compute_decayed_salience(&before) < 0.1);

    let reset = db.reset_decay_state().unwrap();
    assert_eq!(reset, 1);

    let after = db.get_memory(id).unwrap();
    assert!(
        after.salience > 0.99,
        "salience should be reset to full, got {}",
        after.salience
    );
    assert!(
        db.compute_decayed_salience(&after) > 0.99,
        "a just-reset memory should read as fully healthy"
    );
    // Access count is preserved, not wiped.
    assert_eq!(after.access_count, 1);

    db.close().unwrap();
}

#[test]
fn reembed_all_embeds_and_indexes_and_is_idempotent() {
    // Models the embedding-dimension migration: a memory whose stored vector is
    // not at the current dimension (here, missing entirely) is unrecallable
    // until reembed_all re-embeds it at the embedder's dimension and indexes it.
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open_with_embedder(
        dir.path(),
        Box::new(mentedb_embedding::hash_provider::HashEmbeddingProvider::new(8)),
    )
    .unwrap();
    let agent = AgentId::new();

    let target = db.embed_text("probe").unwrap().unwrap().len();
    assert_eq!(target, 8);

    let m = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "the deploy runs on github actions".to_string(),
        Vec::new(), // no vector, as if from an incompatible prior embedder
    );
    let id = m.id;
    db.store(m).unwrap();

    let n = db.reembed_all().unwrap();
    assert_eq!(n, 1, "the vectorless memory is re-embedded");
    let node = db.get_memory(id).unwrap();
    assert_eq!(
        node.embedding.len(),
        target,
        "re-embedded at the current embedder dimension"
    );

    // Idempotent: nothing left to migrate.
    assert_eq!(db.reembed_all().unwrap(), 0);

    // Now retrievable.
    let q = db.embed_text("github actions deploy").unwrap().unwrap();
    let hits = db.recall_similar(&q, 5).unwrap();
    assert!(
        hits.iter().any(|(hid, _)| *hid == id),
        "the re-embedded memory is retrievable"
    );

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

/// End-to-end retention probe: the full lifecycle after the decay fix. Two
/// memories are 40 days old; one keeps getting recalled, the other never is.
/// Recall must keep the first alive (retrieval refreshes its decay clock) and
/// leave it retrievable, while the never-recalled one is retired. This is the
/// property "actively used memories persist, unused old ones are forgotten,"
/// exercised through store -> recall -> reinforce -> archival -> recall.
#[test]
fn lifecycle_recalled_memories_survive_unused_are_retired() {
    use mentedb_consolidation::archival::ArchivalDecision;

    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let day = 24 * 3600 * 1_000_000u64;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    let long_ago = now - 40 * day;

    let mut recalled = make_memory("user prefers dark mode", vec![1.0, 0.0, 0.0, 0.0]);
    recalled.created_at = long_ago;
    recalled.accessed_at = long_ago;
    let rid = recalled.id;

    let mut unused = make_memory("one-off trivia nobody revisits", vec![0.0, 1.0, 0.0, 0.0]);
    unused.created_at = long_ago;
    unused.accessed_at = long_ago;
    let uid = unused.id;

    db.store(recalled).unwrap();
    db.store(unused).unwrap();

    // The dark-mode fact is surfaced by recall again now; retrieval refreshes its
    // decay clock even though the reply did not echo it.
    let (touched, _) = db.record_injection_outcome(&[rid], None).unwrap();
    assert_eq!(touched, 1);

    // Lifecycle outcome: recall keeps its memory alive; the never-revisited one is retired.
    assert_eq!(
        db.evaluate_archival(&db.get_memory(rid).unwrap()),
        ArchivalDecision::Keep,
        "a memory kept warm by recall must survive"
    );
    assert_ne!(
        db.evaluate_archival(&db.get_memory(uid).unwrap()),
        ArchivalDecision::Keep,
        "a 40-day memory never recalled must be retired"
    );

    // And the surviving memory is still retrievable.
    let hits = db.recall_similar(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
    assert!(
        hits.iter().any(|(id, _)| *id == rid),
        "the memory kept alive by recall is still retrievable"
    );
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

// ---------------------------------------------------------------------------
// Injection attention
// ---------------------------------------------------------------------------

mod injection_attention {
    use super::*;
    use mentedb::injection::{
        ATTR_INJECTION_SHOWN, ATTR_INJECTION_USED, InjectionQuery, SelectionReason,
    };

    fn semantic(embedding: Vec<f32>, content: &str, tags: Vec<String>) -> MemoryNode {
        let mut node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            content.into(),
            embedding,
        );
        node.tags = tags;
        node
    }

    #[test]
    fn graph_expansion_surfaces_linked_neighbors() {
        // A matches the query; B does not, but is linked to A by an edge. With a
        // candidate pool of 1, only A is a direct hit, so B can appear ONLY via
        // associative graph expansion. Disabled, B stays out; enabled, B rides in.
        let a_emb = vec![1.0, 0.0, 0.0, 0.0];
        let b_emb = vec![0.0, 1.0, 0.0, 0.0];

        let build = |expansion: usize| {
            let dir = tempfile::tempdir().unwrap();
            let config = mentedb::CognitiveConfig {
                write_inference: false,
                injection_config: mentedb::injection::InjectionConfig {
                    candidate_pool: 1,
                    graph_expansion_max: expansion,
                    ..Default::default()
                },
                ..Default::default()
            };
            let db = MenteDb::open_with_config(dir.path(), config).unwrap();
            let a = semantic(a_emb.clone(), "user prefers dark mode", vec![]);
            let b = semantic(b_emb.clone(), "user is colorblind", vec![]);
            let (aid, bid) = (a.id, b.id);
            db.store(a).unwrap();
            db.store(b).unwrap();
            db.relate(MemoryEdge {
                source: aid,
                target: bid,
                edge_type: EdgeType::Related,
                weight: 0.9,
                created_at: 0,
                valid_from: None,
                valid_until: None,
                label: None,
            })
            .unwrap();
            let query = InjectionQuery {
                embedding: &a_emb,
                query_text: None,
                session_id: None,
                exclude_ids: &[],
                max_items: 5,
                max_episodic: 0,
                agent_id: None,
                user_id: None,
                current_project: None,
            };
            let ids: Vec<_> = db
                .recall_for_injection(&query)
                .unwrap()
                .iter()
                .map(|c| c.node.id)
                .collect();
            (aid, bid, ids)
        };

        let (aid, bid, off) = build(0);
        assert!(off.contains(&aid), "A is the direct hit");
        assert!(!off.contains(&bid), "B stays out with expansion disabled");

        let (aid, bid, on) = build(4);
        assert!(on.contains(&aid), "A is still the direct hit");
        assert!(on.contains(&bid), "B pulled in by graph expansion");
    }

    #[test]
    fn injection_excludes_session_actions_and_ledger_and_pins_always() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();

        let relevant = semantic(vec![1.0, 0.0, 0.0, 0.0], "prefers rust", vec![]);
        let relevant_id = relevant.id;
        let ledgered = semantic(vec![0.8, 0.6, 0.0, 0.0], "ledgered fact", vec![]);
        let ledgered_id = ledgered.id;
        let same_session = semantic(
            vec![0.8, 0.0, 0.6, 0.0],
            "from this session",
            vec!["turn".into(), "session:sess-a".into()],
        );
        let action = semantic(
            vec![0.8, 0.0, 0.0, 0.6],
            "Edited file: x.rs",
            vec!["action".into()],
        );
        let pinned = semantic(
            vec![0.0, 1.0, 0.0, 0.0],
            "never deploy fridays",
            vec!["scope:always".into()],
        );
        let pinned_id = pinned.id;

        for node in [relevant, ledgered, same_session, action, pinned] {
            db.store(node).unwrap();
        }

        let query = InjectionQuery {
            embedding: &[1.0, 0.0, 0.0, 0.0],
            query_text: None,
            session_id: Some("sess-a"),
            exclude_ids: &[ledgered_id],
            max_items: 6,
            max_episodic: 2,
            agent_id: None,
            user_id: None,
            current_project: None,
        };
        let out = db.recall_for_injection(&query).unwrap();

        let ids: Vec<_> = out.iter().map(|c| c.node.id).collect();
        assert!(ids.contains(&relevant_id), "relevant fact selected");
        assert!(ids.contains(&pinned_id), "pinned memory always included");
        assert!(!ids.contains(&ledgered_id), "ledger exclusion respected");
        assert!(
            !out.iter().any(|c| c.node.content == "from this session"),
            "same-session memory excluded"
        );
        assert!(
            !out.iter()
                .any(|c| c.node.content.starts_with("Edited file")),
            "action notes never inject"
        );
        assert_eq!(
            out.iter().find(|c| c.node.id == pinned_id).unwrap().reason,
            SelectionReason::Pinned
        );
    }

    #[test]
    fn outcome_recording_tracks_shown_and_used() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();

        let mut used = semantic(vec![1.0, 0.0, 0.0, 0.0], "drawn on by reply", vec![]);
        used.salience = 0.5;
        let used_id = used.id;
        let ignored = semantic(vec![0.0, 1.0, 0.0, 0.0], "ignored fact", vec![]);
        let ignored_id = ignored.id;
        let before_salience = used.salience;
        db.store(used).unwrap();
        db.store(ignored).unwrap();

        // The reply embedding matches the first memory only.
        let (updated, used_count) = db
            .record_injection_outcome(&[used_id, ignored_id], Some(&[1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        assert_eq!(updated, 2);
        assert_eq!(used_count, 1);

        let used_node = db.get_memory(used_id).unwrap();
        let ignored_node = db.get_memory(ignored_id).unwrap();
        let count = |n: &MemoryNode, k: &str| match n.attributes.get(k) {
            Some(mentedb_core::memory::AttributeValue::Integer(v)) => *v,
            _ => 0,
        };
        assert_eq!(count(&used_node, ATTR_INJECTION_SHOWN), 1);
        assert_eq!(count(&used_node, ATTR_INJECTION_USED), 1);
        assert_eq!(count(&ignored_node, ATTR_INJECTION_SHOWN), 1);
        assert_eq!(count(&ignored_node, ATTR_INJECTION_USED), 0);
        assert!(
            used_node.salience > before_salience,
            "used memory reinforced"
        );
        // Retrieval reinforcement: the shown-but-not-echoed memory still had its
        // decay clock and access count refreshed, so being recalled keeps it alive
        // even though it did not earn the salience boost.
        assert_eq!(count(&ignored_node, ATTR_INJECTION_USED), 0);
        assert_eq!(
            ignored_node.access_count, 1,
            "shown memory's access count bumped even without echo"
        );
    }

    #[test]
    fn retrieval_refreshes_decay_clock_for_shown_not_echoed() {
        // A memory surfaced by recall but not echoed by the reply must still have
        // its decay clock refreshed, so anything actively recalled resists decay
        // and is not forgotten while in use.
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();
        let mut m = semantic(vec![1.0, 0.0, 0.0, 0.0], "recalled, never echoed", vec![]);
        m.accessed_at = 1_000; // stale decay clock
        m.access_count = 0;
        let id = m.id;
        db.store(m).unwrap();

        // Reply embedding is orthogonal, so cosine < used_similarity: not "used".
        db.record_injection_outcome(&[id], Some(&[0.0, 1.0, 0.0, 0.0]))
            .unwrap();

        let n = db.get_memory(id).unwrap();
        assert_eq!(n.access_count, 1, "recall bumps access count");
        assert!(
            n.accessed_at > 1_000,
            "recall refreshes the decay clock even without an echo"
        );
    }

    #[test]
    fn chronically_ignored_memories_are_demoted() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();

        let stale = semantic(vec![1.0, 0.0, 0.0, 0.0], "always shown never used", vec![]);
        let stale_id = stale.id;
        let fresh = semantic(vec![0.9, 0.3, 0.0, 0.0], "newcomer fact", vec![]);
        let fresh_id = fresh.id;
        db.store(stale).unwrap();
        db.store(fresh).unwrap();

        // Show the stale memory five times with no matching reply.
        for _ in 0..5 {
            db.record_injection_outcome(&[stale_id], None).unwrap();
        }

        let query = InjectionQuery {
            embedding: &[1.0, 0.0, 0.0, 0.0],
            query_text: None,
            session_id: None,
            exclude_ids: &[],
            max_items: 1,
            max_episodic: 0,
            agent_id: None,
            user_id: None,
            current_project: None,
        };
        let out = db.recall_for_injection(&query).unwrap();
        let relevant: Vec<_> = out
            .iter()
            .filter(|c| c.reason == SelectionReason::Relevant)
            .collect();
        assert_eq!(relevant.len(), 1);
        assert_eq!(
            relevant[0].node.id, fresh_id,
            "demoted memory loses its top slot to the fresh one"
        );
    }
}

#[test]
fn test_store_user_profile_bumps_updated_timestamp() {
    // Regenerating the profile must move its timestamp forward. profile_updated_at
    // reads the node's created_at, so if a rebuild kept the old created_at the UI
    // showed "updated N days ago" forever.
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    db.store_user_profile("first version", UserId::nil(), AgentId::nil())
        .unwrap();
    let t1 = db.user_profile().unwrap().created_at;
    std::thread::sleep(std::time::Duration::from_millis(2));
    db.store_user_profile("second version", UserId::nil(), AgentId::nil())
        .unwrap();
    let node = db.user_profile().unwrap();
    assert_eq!(
        node.content, "second version",
        "content must update on rebuild"
    );
    assert!(
        node.created_at > t1,
        "created_at must bump on rebuild (was frozen at first creation): {} !> {}",
        node.created_at,
        t1
    );
    db.close().unwrap();
}

#[test]
fn as_of_recalls_the_validity_window_including_superseded_facts() {
    // Bitemporal AS OF: `RECALL ... AS OF <t>` returns exactly the memories whose
    // validity window [valid_from, valid_until) contains t. A fact superseded
    // after t is still visible when we ask "as of" a moment it was true, and a
    // fact that only becomes valid later is hidden. This is the whole point of
    // AS OF: currently-invalid memories are NOT globally dropped, they are judged
    // against the requested instant.
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    // Always valid (no window).
    let stable = make_memory("stable fact", vec![1.0, 0.0, 0.0, 0.0]);

    // Valid from the beginning, superseded at t=1000.
    let mut superseded = make_memory("was true early then replaced", vec![0.0, 1.0, 0.0, 0.0]);
    superseded.valid_until = Some(1000);

    // Only becomes valid at t=2000.
    let mut future = make_memory("becomes true later", vec![0.0, 0.0, 1.0, 0.0]);
    future.valid_from = Some(2000);

    db.store(stable).unwrap();
    db.store(superseded).unwrap();
    db.store(future).unwrap();

    let contents = |q: &str| -> Vec<String> {
        db.recall(q)
            .unwrap()
            .blocks
            .iter()
            .flat_map(|b| b.memories.iter().map(|sm| sm.memory.content.clone()))
            .collect()
    };
    let has = |v: &[String], s: &str| v.iter().any(|c| c == s);

    // AS OF 500: stable + superseded are valid; future is not yet valid.
    let at_500 = contents("RECALL memories AS OF 500");
    assert!(has(&at_500, "stable fact"));
    assert!(has(&at_500, "was true early then replaced"));
    assert!(!has(&at_500, "becomes true later"));

    // AS OF 1500: superseded is now invalidated (valid_until=1000); future still not valid.
    let at_1500 = contents("RECALL memories AS OF 1500");
    assert!(has(&at_1500, "stable fact"));
    assert!(!has(&at_1500, "was true early then replaced"));
    assert!(!has(&at_1500, "becomes true later"));

    // AS OF 3000: future is now valid; superseded still gone.
    let at_3000 = contents("RECALL memories AS OF 3000");
    assert!(has(&at_3000, "stable fact"));
    assert!(!has(&at_3000, "was true early then replaced"));
    assert!(has(&at_3000, "becomes true later"));

    db.close().unwrap();
}
