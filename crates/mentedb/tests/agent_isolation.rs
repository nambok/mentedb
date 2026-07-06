//! Agent scoped retrieval: memories owned by one agent are invisible to
//! another agent's recall, injection, and turn context, while nil owned
//! (shared) memories stay visible to everyone. No scope means global
//! visibility, preserving single agent behavior.

use mentedb::CognitiveConfig;
use mentedb::MenteDb;
use mentedb::injection::InjectionQuery;
use mentedb::prelude::*;
use mentedb::process_turn::ProcessTurnInput;
use mentedb_context::DeltaTracker;
use uuid::Uuid;

fn open_db() -> (MenteDb, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open_with_config(dir.path(), CognitiveConfig::default()).unwrap();
    (db, dir)
}

fn vec_for(axis: usize) -> Vec<f32> {
    // Well separated directions: near coincident vectors can orphan HNSW
    // nodes (a known indexing quirk), and with only a handful of nodes any
    // reachable node is returned, so distance does not matter here.
    let mut v = vec![0.1_f32; 8];
    v[axis] = 1.0;
    v
}

fn store_owned(db: &MenteDb, agent: AgentId, content: &str, topic: usize) -> MemoryId {
    let node = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        content.to_string(),
        vec_for(topic),
    );
    let id = node.id;
    db.store(node).unwrap();
    id
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

#[test]
fn scoped_recall_isolates_agents() {
    let (db, _dir) = open_db();
    let coder = AgentId::new();
    let researcher = AgentId::new();

    let coder_mem = store_owned(&db, coder, "the payment service uses rust and axum", 0);
    let researcher_mem = store_owned(
        &db,
        researcher,
        "the survey paper covers rust memory models",
        1,
    );
    let shared = store_owned(
        &db,
        AgentId::nil(),
        "the team standup about rust happens at nine",
        2,
    );

    let query = vec_for(0);

    // Scoped to the coder: own memory plus shared, never the researcher's.
    let hits = db
        .recall_hybrid_scoped_at_mode(
            &query,
            Some("rust"),
            10,
            now_us(),
            None,
            false,
            None,
            Some(coder),
        )
        .unwrap();
    let ids: Vec<MemoryId> = hits.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&coder_mem), "agent must see its own memory");
    assert!(
        ids.contains(&shared),
        "agent must see shared nil owned memory"
    );
    assert!(
        !ids.contains(&researcher_mem),
        "agent must not see another agent's memory"
    );

    // No scope: global visibility, everything is recallable.
    let hits = db
        .recall_hybrid_scoped_at_mode(&query, Some("rust"), 10, now_us(), None, false, None, None)
        .unwrap();
    let ids: Vec<MemoryId> = hits.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&coder_mem));
    assert!(ids.contains(&researcher_mem));
    assert!(ids.contains(&shared));
}

#[test]
fn process_turn_context_is_agent_scoped() {
    let (db, _dir) = open_db();
    let mut delta = DeltaTracker::new();
    let coder = Uuid::new_v4();
    let researcher = Uuid::new_v4();

    // The coder learns something in its own turns.
    let input = ProcessTurnInput {
        user_message: "our deployment pipeline uses blue green releases".to_string(),
        assistant_response: Some("noted, blue green it is".to_string()),
        turn_id: 0,
        project_context: None,
        agent_id: Some(coder),
        session_id: None,
    };
    db.process_turn(&input, &mut delta).unwrap();

    // The researcher asks about the same topic under its own identity.
    let input = ProcessTurnInput {
        user_message: "what do we know about deployment pipelines".to_string(),
        assistant_response: None,
        turn_id: 1,
        project_context: None,
        agent_id: Some(researcher),
        session_id: None,
    };
    let result = db.process_turn(&input, &mut delta).unwrap();
    assert!(
        result
            .context
            .iter()
            .all(|c| !c.memory.content.contains("blue green")),
        "another agent's turns must not leak into context"
    );

    // An unscoped turn still sees everything.
    let input = ProcessTurnInput {
        user_message: "what do we know about deployment pipelines".to_string(),
        assistant_response: None,
        turn_id: 2,
        project_context: None,
        agent_id: None,
        session_id: None,
    };
    let result = db.process_turn(&input, &mut delta).unwrap();
    assert!(
        result
            .context
            .iter()
            .any(|c| c.memory.content.contains("blue green")),
        "unscoped turns keep global visibility"
    );
}

#[test]
fn injection_respects_agent_scope() {
    let (db, _dir) = open_db();
    let coder = AgentId::new();
    let researcher = AgentId::new();

    // A pinned rule owned by the researcher, one owned by nobody.
    let emb = vec_for(3);
    let mut theirs = MemoryNode::new(
        researcher,
        MemoryType::Procedural,
        "researcher only: cite every claim".to_string(),
        emb.clone(),
    );
    theirs.tags = vec!["scope:always".to_string()];
    db.store(theirs).unwrap();

    let mut shared = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Procedural,
        "everyone: never commit secrets".to_string(),
        emb,
    );
    shared.tags = vec!["scope:always".to_string()];
    db.store(shared).unwrap();

    let query_emb = vec_for(3);
    let selected = db
        .recall_for_injection(&InjectionQuery {
            embedding: &query_emb,
            query_text: Some("what are the rules"),
            session_id: None,
            exclude_ids: &[],
            max_items: 6,
            max_episodic: 2,
            agent_id: Some(coder),
        })
        .unwrap();

    let contents: Vec<&str> = selected.iter().map(|c| c.node.content.as_str()).collect();
    assert!(
        contents.iter().any(|c| c.contains("never commit secrets")),
        "shared pinned memories inject for every agent"
    );
    assert!(
        !contents.iter().any(|c| c.contains("researcher only")),
        "another agent's pinned memories must not inject"
    );
}
