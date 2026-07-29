//! Violated-rule escalation must change what gets INJECTED, not just bump a
//! salience number injection never reads. A rule the agent was shown and then
//! broke (tagged `escalated:violation`) must surface at the front the next
//! time its context recurs, so the correction sticks instead of being ignored
//! again from mid-list.

use mentedb::MenteDb;
use mentedb::injection::InjectionQuery;
use mentedb::prelude::*;
use mentedb_core::types::AgentId;

const DIM: usize = 8;

fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[i] = 1.0;
    v
}

fn inject<'a>(emb: &'a [f32]) -> InjectionQuery<'a> {
    InjectionQuery {
        embedding: emb,
        query_text: None,
        session_id: None,
        exclude_ids: &[],
        max_items: 6,
        max_episodic: 2,
        agent_id: None,
        user_id: None,
        current_project: None,
    }
}

#[test]
fn escalated_rule_leads_over_an_equally_similar_peer() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    // Two procedures with the SAME embedding as the query, so cosine cannot
    // separate them. Same type too, so type quotas cannot. Only the escalation
    // tag differs.
    let peer = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Procedural,
        "prefer conventional commit prefixes for messages".to_string(),
        axis(0),
    );
    db.store(peer).unwrap();

    let mut escalated = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Procedural,
        "read the run output before reporting a deploy as green".to_string(),
        axis(0),
    );
    escalated.tags = vec!["escalated:violation".to_string()];
    let escalated_id = escalated.id;
    db.store(escalated).unwrap();

    let q = axis(0);
    let out = db.recall_for_injection(&inject(&q)).unwrap();

    assert!(!out.is_empty(), "both rules should be injectable");
    assert_eq!(
        out[0].node.id,
        escalated_id,
        "the violated rule must lead, got order: {:?}",
        out.iter().map(|c| &c.node.content).collect::<Vec<_>>()
    );
}

#[test]
fn escalation_does_not_resurrect_an_irrelevant_rule() {
    // The boost re-ranks among relevant candidates; it must not drag an
    // escalated rule that is unrelated to the turn into the context.
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let mut escalated = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Procedural,
        "escalated rule about a completely unrelated activity".to_string(),
        axis(3),
    );
    escalated.tags = vec!["escalated:violation".to_string()];
    db.store(escalated).unwrap();

    let relevant = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Procedural,
        "the rule that actually matches this action".to_string(),
        axis(0),
    );
    let relevant_id = relevant.id;
    db.store(relevant).unwrap();

    let q = axis(0);
    let out = db.recall_for_injection(&inject(&q)).unwrap();
    assert_eq!(
        out.first().map(|c| c.node.id),
        Some(relevant_id),
        "an orthogonal escalated rule must not outrank the relevant one"
    );
}
