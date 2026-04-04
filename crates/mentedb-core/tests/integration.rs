//! Integration tests for multi-agent collaboration features.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use mentedb_core::agent::AgentRegistry;
use mentedb_core::conflict::{ConflictResolver, ConflictVersion, Resolution};
use mentedb_core::event::{EventBus, MenteEvent};
use mentedb_core::mvcc::VersionStore;
use mentedb_core::space::{Permission, SpaceManager};

#[test]
fn full_agent_space_workflow() {
    // 1. Register agents
    let mut agents = AgentRegistry::new();
    let alice = agents.register("alice");
    let bob = agents.register("bob");
    assert_eq!(agents.list().len(), 2);

    // 2. Create a shared space owned by Alice
    let mut spaces = SpaceManager::new();
    let space = spaces.create_space("shared-kb", alice.id);

    // Owner has admin
    assert!(spaces.check_access(space.id, alice.id, Permission::Admin));

    // Bob has no access yet
    assert!(!spaces.check_access(space.id, bob.id, Permission::Read));

    // 3. Grant Bob read-write access
    spaces.grant_access(space.id, bob.id, Permission::ReadWrite);
    assert!(spaces.check_access(space.id, bob.id, Permission::Read));
    assert!(spaces.check_access(space.id, bob.id, Permission::Write));
    assert!(!spaces.check_access(space.id, bob.id, Permission::Admin));

    // 4. List spaces for each agent
    assert_eq!(spaces.list_spaces_for_agent(alice.id).len(), 1);
    assert_eq!(spaces.list_spaces_for_agent(bob.id).len(), 1);
}

#[test]
fn version_tracking_across_agents() {
    let mut agents = AgentRegistry::new();
    let alice = agents.register("alice");
    let bob = agents.register("bob");

    let mut versions = VersionStore::new();
    let mid = uuid::Uuid::new_v4();

    let v1 = versions.record_write(mid, alice.id, 0xAA);
    let v2 = versions.record_write(mid, bob.id, 0xBB);
    assert!(v2 > v1);

    let latest = versions.get_latest(mid).unwrap();
    assert_eq!(latest.agent_id, bob.id);

    let history = versions.get_history(mid);
    assert_eq!(history.len(), 2);
}

#[test]
fn event_bus_integration() {
    let bus = EventBus::new();
    let event_count = Arc::new(AtomicUsize::new(0));

    let c = event_count.clone();
    bus.subscribe(move |_| {
        c.fetch_add(1, Ordering::Relaxed);
    });

    let mid = uuid::Uuid::new_v4();
    let aid = uuid::Uuid::new_v4();
    bus.publish(MenteEvent::MemoryCreated {
        id: mid,
        agent_id: aid,
    });
    bus.publish(MenteEvent::MemoryUpdated {
        id: mid,
        version: 1,
    });

    assert_eq!(event_count.load(Ordering::Relaxed), 2);
}

#[test]
fn conflict_detection_and_resolution() {
    let mut agents = AgentRegistry::new();
    let alice = agents.register("alice");
    let bob = agents.register("bob");

    let resolver = ConflictResolver::new();
    let mid = uuid::Uuid::new_v4();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;

    let versions = vec![
        ConflictVersion {
            agent_id: alice.id,
            content: "The sky is blue".into(),
            confidence: 0.9,
            timestamp: now,
        },
        ConflictVersion {
            agent_id: bob.id,
            content: "The sky is gray".into(),
            confidence: 0.7,
            timestamp: now + 500_000, // 0.5s later — within conflict window
        },
    ];

    let conflict = resolver
        .detect_conflict(mid, &versions)
        .expect("should detect conflict");
    assert_eq!(conflict.versions.len(), 2);

    // Resolve by highest confidence — Alice wins
    let winner = resolver.auto_resolve(&conflict, Resolution::KeepHighestConfidence);
    assert_eq!(winner.content, "The sky is blue");

    // Resolve by latest — Bob wins
    let winner = resolver.auto_resolve(&conflict, Resolution::KeepLatest);
    assert_eq!(winner.content, "The sky is gray");

    // Resolve by merge
    let winner = resolver.auto_resolve(
        &conflict,
        Resolution::Merge("The sky can be blue or gray".into()),
    );
    assert_eq!(winner.content, "The sky can be blue or gray");
}

#[test]
fn revoke_access_and_verify() {
    let mut agents = AgentRegistry::new();
    let owner = agents.register("owner");
    let guest = agents.register("guest");

    let mut spaces = SpaceManager::new();
    let space = spaces.create_space("private", owner.id);

    spaces.grant_access(space.id, guest.id, Permission::Read);
    assert!(spaces.check_access(space.id, guest.id, Permission::Read));

    spaces.revoke_access(space.id, guest.id);
    assert!(!spaces.check_access(space.id, guest.id, Permission::Read));
}
