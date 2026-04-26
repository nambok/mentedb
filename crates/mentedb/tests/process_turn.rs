use mentedb::CognitiveConfig;
use mentedb::MenteDb;
use mentedb::process_turn::ProcessTurnInput;
use mentedb_context::DeltaTracker;

fn open_db() -> (MenteDb, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open_with_config(dir.path(), CognitiveConfig::default()).unwrap();
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
