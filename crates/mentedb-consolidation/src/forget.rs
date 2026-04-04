use mentedb_core::types::{AgentId, MemoryId, SpaceId, Timestamp};
use mentedb_core::MemoryNode;
use serde::{Deserialize, Serialize};

/// A request to forget (delete) memories, for GDPR compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetRequest {
    pub agent_id: Option<AgentId>,
    pub space_id: Option<SpaceId>,
    pub memory_ids: Vec<MemoryId>,
    pub reason: String,
    pub requested_at: Timestamp,
}

/// Result of planning a forget operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetResult {
    pub deleted_memories: usize,
    pub deleted_edges: usize,
    pub deleted_facts: usize,
    pub audit_log_entry: String,
}

/// Engine that implements the right-to-deletion (GDPR forget).
pub struct ForgetEngine;

impl ForgetEngine {
    pub fn new() -> Self {
        Self
    }

    /// Plan what would be deleted for a given forget request.
    pub fn plan_forget(
        &self,
        request: &ForgetRequest,
        all_memories: &[MemoryNode],
        all_edges: &[(MemoryId, MemoryId)],
    ) -> ForgetResult {
        // Determine which memories to delete
        let mut to_delete: ahash::HashSet<MemoryId> = ahash::HashSet::default();

        // Specific memory IDs
        for &id in &request.memory_ids {
            to_delete.insert(id);
        }

        // By agent
        if let Some(agent_id) = request.agent_id {
            for m in all_memories {
                if m.agent_id == agent_id {
                    to_delete.insert(m.id);
                }
            }
        }

        // By space
        if let Some(space_id) = request.space_id {
            for m in all_memories {
                if m.space_id == space_id {
                    to_delete.insert(m.id);
                }
            }
        }

        // Count edges touching deleted memories
        let deleted_edges = all_edges
            .iter()
            .filter(|(src, tgt)| to_delete.contains(src) || to_delete.contains(tgt))
            .count();

        // Estimate deleted facts (roughly 1 per memory)
        let deleted_facts = to_delete.len();

        let audit_log_entry =
            self.generate_audit_log(request, &ForgetResult {
                deleted_memories: to_delete.len(),
                deleted_edges,
                deleted_facts,
                audit_log_entry: String::new(),
            });

        ForgetResult {
            deleted_memories: to_delete.len(),
            deleted_edges,
            deleted_facts,
            audit_log_entry,
        }
    }

    /// Generate a human-readable audit log entry.
    pub fn generate_audit_log(&self, request: &ForgetRequest, result: &ForgetResult) -> String {
        let mut log = format!(
            "FORGET REQUEST at timestamp {}\n",
            request.requested_at
        );
        log.push_str(&format!("Reason: {}\n", request.reason));

        if let Some(agent_id) = request.agent_id {
            log.push_str(&format!("Agent: {}\n", agent_id));
        }
        if let Some(space_id) = request.space_id {
            log.push_str(&format!("Space: {}\n", space_id));
        }
        if !request.memory_ids.is_empty() {
            log.push_str(&format!("Specific IDs: {}\n", request.memory_ids.len()));
        }

        log.push_str(&format!(
            "Result: {} memories, {} edges, {} facts deleted\n",
            result.deleted_memories, result.deleted_edges, result.deleted_facts
        ));

        log
    }
}

impl Default for ForgetEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_memory;

    #[test]
    fn test_forget_by_memory_ids() {
        let engine = ForgetEngine::new();
        let m1 = make_memory("a", vec![1.0]);
        let m2 = make_memory("b", vec![1.0]);
        let edges = vec![(m1.id, m2.id)];

        let request = ForgetRequest {
            agent_id: None,
            space_id: None,
            memory_ids: vec![m1.id],
            reason: "User request".into(),
            requested_at: 1000,
        };

        let result = engine.plan_forget(&request, &[m1.clone(), m2.clone()], &edges);
        assert_eq!(result.deleted_memories, 1);
        assert_eq!(result.deleted_edges, 1);
    }

    #[test]
    fn test_forget_by_agent_id() {
        let engine = ForgetEngine::new();
        let agent = uuid::Uuid::new_v4();
        let mut m1 = make_memory("a", vec![1.0]);
        m1.agent_id = agent;
        let mut m2 = make_memory("b", vec![1.0]);
        m2.agent_id = agent;
        let m3 = make_memory("c", vec![1.0]); // different agent

        let request = ForgetRequest {
            agent_id: Some(agent),
            space_id: None,
            memory_ids: Vec::new(),
            reason: "Agent decommissioned".into(),
            requested_at: 2000,
        };

        let result = engine.plan_forget(&request, &[m1, m2, m3], &[]);
        assert_eq!(result.deleted_memories, 2);
    }

    #[test]
    fn test_forget_by_space_id() {
        let engine = ForgetEngine::new();
        let space = uuid::Uuid::new_v4();
        let mut m1 = make_memory("a", vec![1.0]);
        m1.space_id = space;

        let request = ForgetRequest {
            agent_id: None,
            space_id: Some(space),
            memory_ids: Vec::new(),
            reason: "Space purge".into(),
            requested_at: 3000,
        };

        let result = engine.plan_forget(&request, &[m1], &[]);
        assert_eq!(result.deleted_memories, 1);
    }

    #[test]
    fn test_audit_log_generation() {
        let engine = ForgetEngine::new();
        let request = ForgetRequest {
            agent_id: None,
            space_id: None,
            memory_ids: vec![uuid::Uuid::new_v4()],
            reason: "GDPR request".into(),
            requested_at: 5000,
        };
        let result = ForgetResult {
            deleted_memories: 1,
            deleted_edges: 2,
            deleted_facts: 1,
            audit_log_entry: String::new(),
        };
        let log = engine.generate_audit_log(&request, &result);
        assert!(log.contains("FORGET REQUEST"));
        assert!(log.contains("GDPR request"));
        assert!(log.contains("1 memories"));
    }
}
