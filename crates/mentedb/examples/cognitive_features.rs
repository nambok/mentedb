//! Demonstrates MenteDB's cognitive features: stream cognition,
//! trajectory tracking, phantom detection, and pain signals.
//!
//! Run with: cargo run --example cognitive_features

use mentedb_cognitive::{
    CognitionStream, PainRegistry, PainSignal, PhantomConfig, PhantomTracker, StreamAlert,
    TrajectoryNode, TrajectoryTracker, trajectory::DecisionState,
};
use mentedb_core::types::{MemoryId, Timestamp};

fn main() {
    println!("=== Stream Cognition ===\n");
    demonstrate_stream_cognition();

    println!("\n=== Trajectory Tracking ===\n");
    demonstrate_trajectory_tracking();

    println!("\n=== Phantom Memory Detection ===\n");
    demonstrate_phantom_detection();

    println!("\n=== Pain Signals ===\n");
    demonstrate_pain_signals();
}

/// Feed tokens into the cognition stream and check for alerts.
///
/// Stream cognition monitors the LLM's output in real time, comparing
/// tokens against known facts to detect contradictions, forgotten
/// knowledge, corrections, and reinforcements.
fn demonstrate_stream_cognition() {
    let stream = CognitionStream::new(256);

    // Simulate an LLM generating tokens.
    let tokens = ["The", "user", "prefers", "light", "mode", "in", "editors"];
    for token in &tokens {
        stream.feed_token(token);
    }

    // Check the buffered output against known facts.
    // The stored fact says "dark mode" but the LLM said "light mode."
    let known_facts: Vec<(MemoryId, String)> = vec![(
        MemoryId::new(),
        "User prefers dark mode in all editors".to_string(),
    )];

    let alerts = stream.check_alerts(&known_facts);
    println!("Alerts generated: {}", alerts.len());
    for alert in &alerts {
        match alert {
            StreamAlert::Contradiction {
                memory_id,
                ai_said,
                stored,
            } => {
                println!(
                    "  CONTRADICTION: AI said '{}', but stored fact is '{}'  (memory: {})",
                    ai_said, stored, memory_id
                );
            }
            StreamAlert::Reinforcement { memory_id } => {
                println!("  REINFORCEMENT: confirmed memory {}", memory_id);
            }
            StreamAlert::Forgotten { memory_id, summary } => {
                println!("  FORGOTTEN: {} (memory: {})", summary, memory_id);
            }
            StreamAlert::Correction {
                memory_id,
                old,
                new,
            } => {
                println!(
                    "  CORRECTION: '{}' -> '{}' (memory: {})",
                    old, new, memory_id
                );
            }
        }
    }

    // Drain the token buffer.
    let buffered = stream.drain_buffer();
    println!("Buffered text: '{}'", buffered);
}

/// Record conversation turns and extract resume context.
///
/// Trajectory tracking records the reasoning arc of a conversation
/// so it can be resumed after interruption.
fn demonstrate_trajectory_tracking() {
    let mut tracker = TrajectoryTracker::new(100);

    // Record a sequence of conversation turns.
    tracker.record_turn(TrajectoryNode {
        turn_id: 1,
        topic_embedding: vec![0.1, 0.9, 0.2],
        topic_summary: "Discussing database architecture options".to_string(),
        decision_state: DecisionState::Investigating,
        open_questions: vec!["Which storage engine to use?".to_string()],
        timestamp: 1000,
    });

    tracker.record_turn(TrajectoryNode {
        turn_id: 2,
        topic_embedding: vec![0.2, 0.8, 0.3],
        topic_summary: "Narrowed to custom page storage vs SQLite".to_string(),
        decision_state: DecisionState::NarrowedTo("custom storage".to_string()),
        open_questions: vec!["What page size?".to_string()],
        timestamp: 2000,
    });

    tracker.record_turn(TrajectoryNode {
        turn_id: 3,
        topic_embedding: vec![0.3, 0.7, 0.4],
        topic_summary: "Decided on 16KB pages with CLOCK eviction".to_string(),
        decision_state: DecisionState::Decided("16KB pages, CLOCK eviction".to_string()),
        open_questions: vec![],
        timestamp: 3000,
    });

    // Get the trajectory length.
    let trajectory = tracker.get_trajectory();
    println!("Recorded {} turns", trajectory.len());

    // Generate a resume context for continuing the conversation.
    if let Some(resume) = tracker.get_resume_context() {
        println!("Resume context:\n  {}", resume);
    }

    // Predict what topics might come next.
    let predictions = tracker.predict_next_topics();
    println!("Predicted next topics:");
    for topic in &predictions {
        println!("  {}", topic);
    }
}

/// Detect references to unknown entities (phantom memories).
///
/// When content references entities that are not in the knowledge base,
/// those gaps are flagged as phantom memories so the agent can decide
/// whether to acquire the missing knowledge.
fn demonstrate_phantom_detection() {
    let config = PhantomConfig::default();
    let mut tracker = PhantomTracker::new(config);

    // Content that references entities not in our known set.
    let content = "We should use the PostgreSQL wire protocol for compatibility. \
                   The MVCC implementation from Dr. Smith's paper handles \
                   write skew anomalies correctly.";

    let known_entities = vec!["MVCC".to_string(), "wire protocol".to_string()];

    let phantoms = tracker.detect_gaps(content, &known_entities, 1);
    println!("Detected {} phantom memories:", phantoms.len());
    for phantom in &phantoms {
        println!(
            "  [{}] gap: '{}', source: '{}'",
            format!("{:?}", phantom.priority),
            phantom.gap_description,
            phantom.source_reference
        );
    }

    // Check all active (unresolved) phantoms.
    let active = tracker.get_active_phantoms();
    println!("Active phantoms: {}", active.len());

    // Resolve a phantom once the knowledge is acquired.
    if let Some(first) = phantoms.first() {
        tracker.resolve(first.id.into());
        println!("Resolved phantom: {}", first.id);
    }

    let warnings = tracker.format_phantom_warnings();
    if !warnings.is_empty() {
        println!("Phantom warnings:\n{}", warnings);
    }
}

/// Record and retrieve pain signals (negative experiences).
///
/// Pain signals represent past mistakes or negative outcomes. They are
/// surfaced as warnings during context assembly so the agent avoids
/// repeating errors.
fn demonstrate_pain_signals() {
    let mut registry = PainRegistry::new(10);

    // Record a pain signal from a past mistake.
    let signal = PainSignal {
        id: MemoryId::new(),
        memory_id: MemoryId::new(),
        intensity: 0.9,
        trigger_keywords: vec!["deploy".into(), "production".into(), "rollback".into()],
        description: "Deployed untested code to production, required emergency rollback"
            .to_string(),
        created_at: 1000,
        decay_rate: 0.001,
    };
    registry.record_pain(signal);

    // Record another pain signal.
    let signal2 = PainSignal {
        id: MemoryId::new(),
        memory_id: MemoryId::new(),
        intensity: 0.6,
        trigger_keywords: vec!["cache".into(), "stale".into(), "invalidation".into()],
        description: "Cache invalidation bug caused stale data to be served for 2 hours"
            .to_string(),
        created_at: 2000,
        decay_rate: 0.002,
    };
    registry.record_pain(signal2);

    // Query pain signals relevant to a context.
    let context_keywords = vec!["deploy".into(), "production".into()];
    let relevant = registry.get_pain_for_context(&context_keywords);
    println!(
        "Pain signals matching 'deploy production': {}",
        relevant.len()
    );
    for pain in &relevant {
        println!(
            "  intensity={:.2}, description='{}'",
            pain.intensity, pain.description
        );
    }

    // Format warnings for inclusion in context.
    let warnings = registry.format_pain_warnings(&relevant);
    if !warnings.is_empty() {
        println!("\nPain warnings for context:\n{}", warnings);
    }

    // Decay all signals over time.
    let future_time: Timestamp = 1_000_000;
    registry.decay_all(future_time);
    println!("\nAfter time decay, re-checking...");

    let still_relevant = registry.get_pain_for_context(&context_keywords);
    println!("Pain signals still active: {}", still_relevant.len());
}
