//! Belief propagation, end to end: when write inference lowers a memory's
//! confidence (a value update supersedes it), the memories that depend on it
//! over `Caused` and `Supports` edges lose confidence too, damped per hop, on
//! both the `store` path and the `process_turn` path. The winner, and the
//! memories the loser once contradicted or superseded, are left alone.

use mentedb::process_turn::ProcessTurnInput;
use mentedb::{CognitiveConfig, MenteDb};
use mentedb_context::DeltaTracker;
use mentedb_core::edge::EdgeType;
use mentedb_core::error::MenteResult;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{AgentId, MemoryId};
use mentedb_core::{MemoryEdge, MemoryNode};
use mentedb_embedding::provider::EmbeddingProvider;

const EPS: f32 = 1e-4;

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

/// A unit vector along axis `i` of 8 dims, orthogonal to the coffee pair.
fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0; 8];
    v[i] = 1.0;
    v
}

fn node(agent: AgentId, content: &str, embedding: Vec<f32>, confidence: f32) -> MemoryNode {
    let mut n = MemoryNode::new(agent, MemoryType::Semantic, content.into(), embedding);
    n.created_at = now_us();
    n.confidence = confidence;
    n
}

fn edge(source: MemoryId, target: MemoryId, edge_type: EdgeType) -> MemoryEdge {
    MemoryEdge {
        source,
        target,
        edge_type,
        weight: 1.0,
        created_at: now_us(),
        valid_from: None,
        valid_until: None,
        label: None,
    }
}

fn confidence(db: &MenteDb, id: MemoryId) -> f32 {
    db.get_memory(id).unwrap().confidence
}

/// The losing fact: "cortado" at confidence 0.8. Superseding it drops it to
/// 0.4 (decay factor 0.5), a change of -0.4.
fn store_loser(db: &MenteDb, agent: AgentId) -> MemoryId {
    let old = node(
        agent,
        "The user's favorite coffee order is a cortado",
        vec![1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        0.8,
    );
    let id = old.id;
    db.store(old).unwrap();
    id
}

/// The winning correction, stored after the loser on the same sentence frame.
fn store_winner(db: &MenteDb, agent: AgentId) -> MemoryId {
    let mut new = node(
        agent,
        "The user's favorite coffee order is a flat white",
        vec![0.99, 0.08, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        0.9,
    );
    new.created_at = now_us() + 1;
    let id = new.id;
    db.store(new).unwrap();
    id
}

fn store_dependent(db: &MenteDb, agent: AgentId, content: &str, i: usize, conf: f32) -> MemoryId {
    let n = node(agent, content, axis(i), conf);
    let id = n.id;
    db.store(n).unwrap();
    id
}

fn open(dir: &std::path::Path, cfg: CognitiveConfig) -> MenteDb {
    MenteDb::open_with_config(dir, cfg).unwrap()
}

#[test]
fn contradiction_lowers_a_caused_chain_by_damped_amounts_and_persists() {
    let dir = tempfile::tempdir().unwrap();
    let agent = AgentId::new();
    let (a, b, c, w);
    {
        let db = open(dir.path(), CognitiveConfig::default());
        a = store_loser(&db, agent);
        b = store_dependent(
            &db,
            agent,
            "The office espresso machine was ordered",
            2,
            0.9,
        );
        c = store_dependent(
            &db,
            agent,
            "Facilities budgeted for milk deliveries",
            3,
            0.8,
        );
        db.relate(edge(a, b, EdgeType::Caused)).unwrap();
        db.relate(edge(b, c, EdgeType::Caused)).unwrap();

        w = store_winner(&db, agent);

        // The root ends at exactly the decayed value: no double delta.
        assert!(
            (confidence(&db, a) - 0.4).abs() < EPS,
            "{}",
            confidence(&db, a)
        );
        // B: 0.9 - 0.4 * 0.9 = 0.54. C: 0.8 - 0.36 * 0.9 = 0.476.
        assert!(
            (confidence(&db, b) - 0.54).abs() < EPS,
            "{}",
            confidence(&db, b)
        );
        assert!(
            (confidence(&db, c) - 0.476).abs() < EPS,
            "{}",
            confidence(&db, c)
        );
        assert!(
            (confidence(&db, w) - 0.9).abs() < EPS,
            "the winner is untouched"
        );
        db.close().unwrap();
    }
    let db = open(dir.path(), CognitiveConfig::default());
    assert!((confidence(&db, a) - 0.4).abs() < EPS);
    assert!(
        (confidence(&db, b) - 0.54).abs() < EPS,
        "{}",
        confidence(&db, b)
    );
    assert!(
        (confidence(&db, c) - 0.476).abs() < EPS,
        "{}",
        confidence(&db, c)
    );
    assert!((confidence(&db, w) - 0.9).abs() < EPS);
}

#[test]
fn supports_lowers_the_supported_memory() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), CognitiveConfig::default());
    let agent = AgentId::new();
    let a = store_loser(&db, agent);
    let s = store_dependent(
        &db,
        agent,
        "The user is a regular at the corner cafe",
        2,
        0.9,
    );
    db.relate(edge(a, s, EdgeType::Supports)).unwrap();

    store_winner(&db, agent);

    // 0.9 - 0.4 * 0.5 = 0.7.
    assert!(
        (confidence(&db, s) - 0.7).abs() < EPS,
        "{}",
        confidence(&db, s)
    );
}

#[test]
fn conflict_edges_out_of_the_loser_and_the_winner_are_not_lowered() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), CognitiveConfig::default());
    let agent = AgentId::new();
    let a = store_loser(&db, agent);
    // Memories the loser once contradicted and replaced. Walking these edges
    // from the weakened loser would move them on no new evidence.
    let x = store_dependent(&db, agent, "The team standup is on Mondays", 2, 0.9);
    let y = store_dependent(&db, agent, "The build agent runs nightly", 3, 0.9);
    db.relate(edge(a, x, EdgeType::Contradicts)).unwrap();
    db.relate(edge(a, y, EdgeType::Supersedes)).unwrap();

    let w = store_winner(&db, agent);

    assert!((confidence(&db, a) - 0.4).abs() < EPS);
    assert!(
        (confidence(&db, w) - 0.9).abs() < EPS,
        "the winner is untouched"
    );
    assert!(
        (confidence(&db, x) - 0.9).abs() < EPS,
        "{}",
        confidence(&db, x)
    );
    assert!(
        (confidence(&db, y) - 0.9).abs() < EPS,
        "{}",
        confidence(&db, y)
    );
}

#[test]
fn disable_flag_and_min_delta_are_respected() {
    let mut off = CognitiveConfig::default();
    off.inference_config.belief_propagation_enabled = false;
    let mut high_floor = CognitiveConfig::default();
    high_floor.inference_config.belief_propagation_min_delta = 0.5;

    for cfg in [off, high_floor] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path(), cfg);
        let agent = AgentId::new();
        let a = store_loser(&db, agent);
        let b = store_dependent(
            &db,
            agent,
            "The office espresso machine was ordered",
            2,
            0.9,
        );
        db.relate(edge(a, b, EdgeType::Caused)).unwrap();

        store_winner(&db, agent);

        assert!(
            (confidence(&db, a) - 0.4).abs() < EPS,
            "the root still drops"
        );
        assert!(
            (confidence(&db, b) - 0.9).abs() < EPS,
            "{}",
            confidence(&db, b)
        );
    }
}

#[test]
fn affected_memories_per_propagation_are_capped() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = CognitiveConfig::default();
    cfg.belief_propagation.max_affected = 2;
    let db = open(dir.path(), cfg);
    let agent = AgentId::new();
    let a = store_loser(&db, agent);
    let deps: Vec<MemoryId> = (0..5)
        .map(|i| {
            let id = store_dependent(
                &db,
                agent,
                &format!("Dependent fact number {i} about rooms"),
                2 + i,
                0.9,
            );
            db.relate(edge(a, id, EdgeType::Caused)).unwrap();
            id
        })
        .collect();

    store_winner(&db, agent);

    let lowered = deps
        .iter()
        .filter(|id| (confidence(&db, **id) - 0.9).abs() > EPS)
        .count();
    assert_eq!(lowered, 2);
}

/// Bag of words embedder, so two turns on the same sentence frame embed close
/// together and an unrelated fact does not.
struct BagOfWords;

impl EmbeddingProvider for BagOfWords {
    fn embed(&self, text: &str) -> MenteResult<Vec<f32>> {
        let mut v = vec![0.0f32; 64];
        for w in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
        {
            let h = w
                .to_lowercase()
                .bytes()
                .fold(7u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
            v[(h % 64) as usize] += 1.0;
        }
        Ok(v)
    }
    fn embed_batch(&self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
    fn dimensions(&self) -> usize {
        64
    }
    fn model_name(&self) -> &str {
        "bag-of-words-test"
    }
}

#[test]
fn process_turn_applies_the_propagation() {
    // Store time inference is off, so the turn's explicit write inference
    // pass is the only thing that can supersede the earlier turn and spread
    // the drop.
    let cfg = CognitiveConfig {
        write_inference: false,
        calibration_enabled: false,
        ..CognitiveConfig::default()
    };
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open_with_embedder_and_config(dir.path(), Box::new(BagOfWords), cfg).unwrap();
    let nil = AgentId(uuid::Uuid::nil());

    let mut earlier = MemoryNode::new(
        nil,
        MemoryType::Episodic,
        "User: my favorite coffee order is a cortado\nAssistant: noted".into(),
        Vec::new(),
    );
    earlier.created_at = now_us() - 1_000_000;
    earlier.confidence = 0.8;
    earlier.tags.push("turn".into());
    let earlier_id = earlier.id;
    db.store(earlier).unwrap();

    let mut dep = MemoryNode::new(
        nil,
        MemoryType::Semantic,
        "Facilities budgeted for milk deliveries".into(),
        Vec::new(),
    );
    dep.confidence = 0.9;
    let dep_id = dep.id;
    db.store(dep).unwrap();
    db.relate(edge(earlier_id, dep_id, EdgeType::Caused))
        .unwrap();

    let input = ProcessTurnInput {
        user_message: "my favorite coffee order is a flat white".into(),
        assistant_response: Some("noted".into()),
        turn_id: 1,
        project_context: None,
        agent_id: None,
        user_id: None,
        session_id: None,
    };
    db.process_turn(&input, &mut DeltaTracker::new()).unwrap();

    let earlier = db.get_memory(earlier_id).unwrap();
    assert!(
        earlier.valid_until.is_some(),
        "the earlier turn is superseded"
    );
    assert!(
        (earlier.confidence - 0.4).abs() < EPS,
        "{}",
        earlier.confidence
    );
    // 0.9 - 0.4 * 0.9 = 0.54.
    let dep = confidence(&db, dep_id);
    assert!(
        (dep - 0.54).abs() < EPS,
        "process_turn must apply the propagation, got {dep}"
    );
}
