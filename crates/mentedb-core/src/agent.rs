//! Agent Registry — tracks agents that participate in the memory system.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{AgentId, SpaceId, Timestamp};

/// An agent that can read/write memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    /// Unique agent identifier.
    pub id: AgentId,
    /// Human-readable name.
    pub name: String,
    /// When the agent was registered (µs since epoch).
    pub created_at: Timestamp,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
    /// The default space this agent operates in.
    pub default_space: SpaceId,
}

/// In-memory registry of all known agents.
#[derive(Debug, Default)]
pub struct AgentRegistry {
    agents: HashMap<AgentId, Agent>,
}

fn now_micros() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as Timestamp
}

impl AgentRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new agent and return it.
    pub fn register(&mut self, name: &str) -> Agent {
        let agent = Agent {
            id: Uuid::new_v4(),
            name: name.to_string(),
            created_at: now_micros(),
            metadata: HashMap::new(),
            default_space: Uuid::nil(),
        };
        self.agents.insert(agent.id, agent.clone());
        agent
    }

    /// Look up an agent by ID.
    pub fn get(&self, id: AgentId) -> Option<&Agent> {
        self.agents.get(&id)
    }

    /// Remove an agent from the registry.
    pub fn remove(&mut self, id: AgentId) {
        self.agents.remove(&id);
    }

    /// List all registered agents.
    pub fn list(&self) -> Vec<&Agent> {
        self.agents.values().collect()
    }

    /// Update a single metadata key for an agent.
    pub fn update_metadata(&mut self, id: AgentId, key: &str, value: &str) {
        if let Some(agent) = self.agents.get_mut(&id) {
            agent.metadata.insert(key.to_string(), value.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_get() {
        let mut reg = AgentRegistry::new();
        let a = reg.register("alice");
        assert_eq!(reg.get(a.id).unwrap().name, "alice");
    }

    #[test]
    fn remove_agent() {
        let mut reg = AgentRegistry::new();
        let a = reg.register("bob");
        reg.remove(a.id);
        assert!(reg.get(a.id).is_none());
    }

    #[test]
    fn list_agents() {
        let mut reg = AgentRegistry::new();
        reg.register("a1");
        reg.register("a2");
        assert_eq!(reg.list().len(), 2);
    }

    #[test]
    fn update_metadata() {
        let mut reg = AgentRegistry::new();
        let a = reg.register("meta");
        reg.update_metadata(a.id, "role", "planner");
        assert_eq!(reg.get(a.id).unwrap().metadata.get("role").unwrap(), "planner");
    }
}
