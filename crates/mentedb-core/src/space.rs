//! Memory Spaces: isolated, permission-controlled namespaces for memories.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::types::{AgentId, SpaceId, Timestamp};

/// Access permission level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Permission {
    Read,
    Write,
    ReadWrite,
    Admin,
}

impl Permission {
    /// Returns `true` when `self` satisfies `required`.
    fn satisfies(self, required: Permission) -> bool {
        matches!(
            (self, required),
            (Permission::Admin, _)
                | (Permission::ReadWrite, Permission::Read)
                | (Permission::ReadWrite, Permission::Write)
                | (Permission::ReadWrite, Permission::ReadWrite)
                | (Permission::Read, Permission::Read)
                | (Permission::Write, Permission::Write)
        )
    }
}

/// One entry in a space's access-control list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessEntry {
    /// The agent granted access.
    pub agent_id: AgentId,
    /// The permission level granted.
    pub permission: Permission,
}

/// A namespace that groups memories and controls access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySpace {
    /// Unique identifier for this space.
    pub id: SpaceId,
    /// Human readable name.
    pub name: String,
    /// The agent that owns this space.
    pub owner: AgentId,
    /// Access control list for other agents.
    pub access_list: Vec<AccessEntry>,
    /// When this space was created.
    pub created_at: Timestamp,
    /// Optional capacity limit on stored memories.
    pub max_memories: Option<usize>,
    /// Current number of memories in this space.
    pub current_count: usize,
}

/// Manages the set of memory spaces.
#[derive(Debug, Default)]
pub struct SpaceManager {
    spaces: HashMap<SpaceId, MemorySpace>,
}

fn now_micros() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as Timestamp
}

impl SpaceManager {
    /// Creates a new empty space manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new space owned by `owner`.
    pub fn create_space(&mut self, name: &str, owner: AgentId) -> MemorySpace {
        let space = MemorySpace {
            id: SpaceId::new(),
            name: name.to_string(),
            owner,
            access_list: vec![AccessEntry {
                agent_id: owner,
                permission: Permission::Admin,
            }],
            created_at: now_micros(),
            max_memories: None,
            current_count: 0,
        };
        self.spaces.insert(space.id, space.clone());
        space
    }

    /// Returns a reference to the space with the given ID, if it exists.
    pub fn get_space(&self, id: SpaceId) -> Option<&MemorySpace> {
        self.spaces.get(&id)
    }

    /// Removes a space by ID.
    pub fn delete_space(&mut self, id: SpaceId) {
        self.spaces.remove(&id);
    }

    /// Grant `perm` to `agent` in the given space.
    pub fn grant_access(&mut self, space: SpaceId, agent: AgentId, perm: Permission) {
        if let Some(s) = self.spaces.get_mut(&space) {
            // Replace existing entry for this agent, or append.
            if let Some(entry) = s.access_list.iter_mut().find(|e| e.agent_id == agent) {
                entry.permission = perm;
            } else {
                s.access_list.push(AccessEntry {
                    agent_id: agent,
                    permission: perm,
                });
            }
        }
    }

    /// Remove all access for `agent` in the given space.
    pub fn revoke_access(&mut self, space: SpaceId, agent: AgentId) {
        if let Some(s) = self.spaces.get_mut(&space) {
            s.access_list.retain(|e| e.agent_id != agent);
        }
    }

    /// Check whether `agent` has at least `required` permission in the space.
    pub fn check_access(&self, space: SpaceId, agent: AgentId, required: Permission) -> bool {
        self.spaces.get(&space).is_some_and(|s| {
            if s.owner == agent {
                return true;
            }
            s.access_list
                .iter()
                .any(|e| e.agent_id == agent && e.permission.satisfies(required))
        })
    }

    /// List all spaces that `agent` owns or has an ACL entry in.
    pub fn list_spaces_for_agent(&self, agent: AgentId) -> Vec<&MemorySpace> {
        self.spaces
            .values()
            .filter(|s| s.owner == agent || s.access_list.iter().any(|e| e.agent_id == agent))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent() -> AgentId {
        AgentId::new()
    }

    #[test]
    fn create_and_get() {
        let mut mgr = SpaceManager::new();
        let owner = agent();
        let sp = mgr.create_space("test", owner);
        assert_eq!(mgr.get_space(sp.id).unwrap().name, "test");
    }

    #[test]
    fn owner_has_admin() {
        let mut mgr = SpaceManager::new();
        let owner = agent();
        let sp = mgr.create_space("s", owner);
        assert!(mgr.check_access(sp.id, owner, Permission::Admin));
    }

    #[test]
    fn grant_and_check() {
        let mut mgr = SpaceManager::new();
        let owner = agent();
        let reader = agent();
        let sp = mgr.create_space("s", owner);
        mgr.grant_access(sp.id, reader, Permission::Read);
        assert!(mgr.check_access(sp.id, reader, Permission::Read));
        assert!(!mgr.check_access(sp.id, reader, Permission::Write));
    }

    #[test]
    fn revoke_access() {
        let mut mgr = SpaceManager::new();
        let owner = agent();
        let a = agent();
        let sp = mgr.create_space("s", owner);
        mgr.grant_access(sp.id, a, Permission::ReadWrite);
        mgr.revoke_access(sp.id, a);
        assert!(!mgr.check_access(sp.id, a, Permission::Read));
    }

    #[test]
    fn list_spaces_for_agent() {
        let mut mgr = SpaceManager::new();
        let o1 = agent();
        let o2 = agent();
        mgr.create_space("s1", o1);
        mgr.create_space("s2", o2);
        assert_eq!(mgr.list_spaces_for_agent(o1).len(), 1);
    }
}
