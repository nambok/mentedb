//! Resource limits and backpressure for MenteDB.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::error::{MenteError, MenteResult};
use crate::types::AgentId;

/// Configurable resource limits.
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum total memories (None = unlimited).
    pub max_memories: Option<usize>,
    /// Maximum total memory usage in bytes.
    pub max_memory_bytes: Option<u64>,
    /// Per-agent memory limit.
    pub max_memories_per_agent: Option<usize>,
    /// Maximum embedding dimensions (default: 4096).
    pub max_embedding_dimensions: usize,
    /// Maximum writes per second (None = unlimited).
    pub write_rate_limit: Option<u32>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memories: None,
            max_memory_bytes: None,
            max_memories_per_agent: None,
            max_embedding_dimensions: 4096,
            write_rate_limit: None,
        }
    }
}

/// Current resource usage snapshot.
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Total number of stored memories.
    pub total_memories: u64,
    /// Total bytes used.
    pub total_bytes: u64,
    /// Configured memory limit.
    pub limit_memories: Option<usize>,
    /// Configured byte limit.
    pub limit_bytes: Option<u64>,
    /// Overall utilization percentage (0.0–100.0).
    pub utilization_pct: f32,
}

/// Tracks resource usage and enforces limits.
pub struct ResourceTracker {
    limits: ResourceLimits,
    current_memory_count: AtomicU64,
    current_bytes: AtomicU64,
    agent_counts: RwLock<HashMap<AgentId, u64>>,
}

impl ResourceTracker {
    /// Create a new tracker with the given limits.
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            limits,
            current_memory_count: AtomicU64::new(0),
            current_bytes: AtomicU64::new(0),
            agent_counts: RwLock::new(HashMap::new()),
        }
    }

    /// Check whether a write is allowed given current usage.
    pub fn check_can_write(&self, agent_id: AgentId, estimated_bytes: u64) -> MenteResult<()> {
        let count = self.current_memory_count.load(Ordering::Relaxed);
        if let Some(max) = self.limits.max_memories
            && count as usize >= max
        {
            return Err(MenteError::ResourceExhausted(format!(
                "memory limit reached: {count}/{max}"
            )));
        }

        let bytes = self.current_bytes.load(Ordering::Relaxed);
        if let Some(max_bytes) = self.limits.max_memory_bytes
            && bytes + estimated_bytes > max_bytes
        {
            return Err(MenteError::ResourceExhausted(format!(
                "byte limit would be exceeded: {} + {} > {}",
                bytes, estimated_bytes, max_bytes
            )));
        }

        if let Some(per_agent) = self.limits.max_memories_per_agent {
            let agent_map = self.agent_counts.read();
            let agent_count = agent_map.get(&agent_id).copied().unwrap_or(0);
            if agent_count as usize >= per_agent {
                return Err(MenteError::ResourceExhausted(format!(
                    "per-agent limit reached for {agent_id}: {agent_count}/{per_agent}"
                )));
            }
        }

        Ok(())
    }

    /// Record that a write occurred.
    pub fn record_write(&self, agent_id: AgentId, bytes: u64) {
        self.current_memory_count.fetch_add(1, Ordering::Relaxed);
        self.current_bytes.fetch_add(bytes, Ordering::Relaxed);
        let mut agent_map = self.agent_counts.write();
        *agent_map.entry(agent_id).or_insert(0) += 1;
    }

    /// Record that a delete occurred.
    pub fn record_delete(&self, agent_id: AgentId, bytes: u64) {
        self.current_memory_count
            .fetch_sub(1.min(self.current_memory_count.load(Ordering::Relaxed)), Ordering::Relaxed);
        let current = self.current_bytes.load(Ordering::Relaxed);
        self.current_bytes
            .fetch_sub(bytes.min(current), Ordering::Relaxed);
        let mut agent_map = self.agent_counts.write();
        if let Some(count) = agent_map.get_mut(&agent_id) {
            *count = count.saturating_sub(1);
        }
    }

    /// Get a snapshot of current resource usage.
    pub fn usage(&self) -> ResourceUsage {
        let total_memories = self.current_memory_count.load(Ordering::Relaxed);
        let total_bytes = self.current_bytes.load(Ordering::Relaxed);

        let utilization_pct = match (self.limits.max_memories, self.limits.max_memory_bytes) {
            (Some(max_mem), Some(max_bytes)) => {
                let mem_pct = (total_memories as f32 / max_mem as f32) * 100.0;
                let byte_pct = (total_bytes as f32 / max_bytes as f32) * 100.0;
                mem_pct.max(byte_pct)
            }
            (Some(max_mem), None) => (total_memories as f32 / max_mem as f32) * 100.0,
            (None, Some(max_bytes)) => (total_bytes as f32 / max_bytes as f32) * 100.0,
            (None, None) => 0.0,
        };

        ResourceUsage {
            total_memories,
            total_bytes,
            limit_memories: self.limits.max_memories,
            limit_bytes: self.limits.max_memory_bytes,
            utilization_pct,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn within_limits_succeeds() {
        let tracker = ResourceTracker::new(ResourceLimits {
            max_memories: Some(10),
            max_memory_bytes: Some(10_000),
            ..Default::default()
        });
        let agent = Uuid::new_v4();
        assert!(tracker.check_can_write(agent, 100).is_ok());
        tracker.record_write(agent, 100);
        assert_eq!(tracker.usage().total_memories, 1);
        assert_eq!(tracker.usage().total_bytes, 100);
    }

    #[test]
    fn exceeding_limit_returns_error() {
        let tracker = ResourceTracker::new(ResourceLimits {
            max_memories: Some(2),
            ..Default::default()
        });
        let agent = Uuid::new_v4();

        tracker.record_write(agent, 50);
        tracker.record_write(agent, 50);
        let result = tracker.check_can_write(agent, 50);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("memory limit reached"));
    }

    #[test]
    fn per_agent_limit_works() {
        let tracker = ResourceTracker::new(ResourceLimits {
            max_memories_per_agent: Some(1),
            ..Default::default()
        });
        let agent_a = Uuid::new_v4();
        let agent_b = Uuid::new_v4();

        tracker.record_write(agent_a, 50);
        // agent_a is now at limit
        assert!(tracker.check_can_write(agent_a, 50).is_err());
        // agent_b can still write
        assert!(tracker.check_can_write(agent_b, 50).is_ok());
    }
}
