use mentedb_cognitive::*;
use mentedb_core::memory::MemoryType;
use mentedb_core::MemoryNode;
use uuid::Uuid;

fn make_memory(content: &str, embedding: Vec<f32>, mem_type: MemoryType) -> MemoryNode {
    MemoryNode::new(Uuid::new_v4(), mem_type, content.to_string(), embedding)
}

#[test]
fn test_stream_cognition_contradiction() {
    let stream = CognitionStream::new(256);
    let mid = Uuid::new_v4();

    // Feed a sentence that contradicts a known fact
    stream.feed_token("The system actually uses MySQL instead of PostgreSQL now");

    let facts = vec![(mid, "The system uses PostgreSQL for data storage".to_string())];
    let alerts = stream.check_alerts(&facts);

    assert!(
        alerts.iter().any(|a| matches!(a, StreamAlert::Contradiction { .. })),
        "Expected contradiction alert when AI contradicts stored fact. Got: {:?}",
        alerts
    );

    // Drain should return all text
    let drained = stream.drain_buffer();
    assert!(drained.contains("PostgreSQL"));
}

#[test]
fn test_write_inference_contradiction() {
    let agent = Uuid::new_v4();

    let mut existing = make_memory("The backend uses PostgreSQL", vec![1.0, 0.0, 0.0], MemoryType::Semantic);
    existing.agent_id = agent;
    existing.created_at = 1000;

    let mut new_mem = make_memory("The backend uses MySQL", vec![0.99, 0.01, 0.0], MemoryType::Semantic);
    new_mem.agent_id = agent;
    new_mem.created_at = 2000;

    let engine = WriteInferenceEngine::new();
    let actions = engine.infer_on_write(&new_mem, &[existing], &[]);

    assert!(
        actions.iter().any(|a| matches!(a, InferredAction::FlagContradiction { .. })),
        "Expected FlagContradiction action. Got: {:?}",
        actions
    );
}

#[test]
fn test_trajectory_resume_context() {
    let mut tracker = TrajectoryTracker::default();

    tracker.record_turn(TrajectoryNode {
        turn_id: 1,
        topic_embedding: vec![1.0, 0.0],
        topic_summary: "JWT authentication design".to_string(),
        decision_state: DecisionState::Investigating,
        open_questions: vec![],
        timestamp: 1000,
    });

    tracker.record_turn(TrajectoryNode {
        turn_id: 2,
        topic_embedding: vec![0.9, 0.1],
        topic_summary: "Token refresh strategy".to_string(),
        decision_state: DecisionState::Decided("Short-lived access tokens (15min)".to_string()),
        open_questions: vec!["Where to store refresh tokens?".to_string()],
        timestamp: 2000,
    });

    tracker.record_turn(TrajectoryNode {
        turn_id: 3,
        topic_embedding: vec![0.8, 0.2],
        topic_summary: "Session storage approach".to_string(),
        decision_state: DecisionState::NarrowedTo("Redis vs Cookie".to_string()),
        open_questions: vec!["Security implications of each?".to_string()],
        timestamp: 3000,
    });

    let ctx = tracker.get_resume_context().unwrap();
    assert!(ctx.contains("Session storage"), "Resume context should mention current topic. Got: {}", ctx);
    assert!(ctx.contains("Redis vs Cookie"), "Resume context should mention narrowed choice. Got: {}", ctx);
}

#[test]
fn test_phantom_detection() {
    let mut tracker = PhantomTracker::default();
    let known = vec!["React".to_string(), "Node".to_string()];

    let phantoms = tracker.detect_gaps(
        "We need to configure the Kubernetes cluster with Istio service mesh",
        &known,
        1,
    );

    let refs: Vec<&str> = phantoms.iter().map(|p| p.source_reference.as_str()).collect();
    assert!(
        refs.iter().any(|r| r.contains("Kubernetes")),
        "Expected phantom for Kubernetes. Got: {:?}",
        refs
    );

    // Should have active phantoms
    assert!(!tracker.get_active_phantoms().is_empty());

    // Warnings should mention unknown entities
    let warnings = tracker.format_phantom_warnings();
    assert!(warnings.contains("WARNING"));
}

#[test]
fn test_pain_trigger_matching() {
    let mut registry = PainRegistry::default();

    registry.record_pain(PainSignal {
        id: Uuid::new_v4(),
        memory_id: Uuid::new_v4(),
        intensity: 0.95,
        trigger_keywords: vec!["mongodb".to_string(), "nosql".to_string()],
        description: "User strongly opposed MongoDB suggestion".to_string(),
        created_at: 1000,
        decay_rate: 0.0001,
    });

    let results = registry.get_pain_for_context(&["mongodb".to_string(), "database".to_string()]);
    assert_eq!(results.len(), 1);
    assert!(results[0].intensity > 0.9);

    let warnings = registry.format_pain_warnings(&results);
    assert!(warnings.contains("CAUTION"));
    assert!(warnings.contains("mongodb"));
}

#[test]
fn test_speculative_cache() {
    let mut cache = SpeculativeCache::default();

    cache.pre_assemble(
        vec![
            "database schema design".to_string(),
            "API endpoint structure".to_string(),
        ],
        |topic| {
            Some((
                format!("Pre-assembled context for: {}", topic),
                vec![Uuid::new_v4()],
            ))
        },
    );

    assert_eq!(cache.stats().cache_size, 2);

    // Should hit on related query
    let hit = cache.try_hit("database schema");
    assert!(hit.is_some(), "Expected cache hit for 'database schema'");
    assert!(hit.unwrap().context_text.contains("database schema design"));
    assert_eq!(cache.stats().hits, 1);

    // Should miss on unrelated query
    let miss = cache.try_hit("cooking recipes for dinner");
    assert!(miss.is_none(), "Expected cache miss for unrelated query");
    assert_eq!(cache.stats().misses, 1);
}

#[test]
fn test_interference_detection() {
    let a = make_memory("Project Alpha uses React framework", vec![1.0, 0.0, 0.0], MemoryType::Semantic);
    let b = make_memory("Project Beta uses Vue framework", vec![0.99, 0.1, 0.0], MemoryType::Semantic);

    let detector = InterferenceDetector::default();
    let pairs = detector.detect_interference(&[a, b]);

    assert!(!pairs.is_empty(), "Expected interference pair for similar but different memories");
    assert!(pairs[0].disambiguation.contains("Do not confuse"));
}
