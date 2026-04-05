use std::collections::VecDeque;

use mentedb_core::types::{MemoryId};
use parking_lot::Mutex;

#[derive(Debug, Clone)]
pub enum TokenEvent {
    Token(String),
    EndOfTurn,
    Correction(MemoryId, String),
    Flush,
}

#[derive(Debug, Clone)]
pub enum StreamAlert {
    Contradiction {
        memory_id: MemoryId,
        ai_said: String,
        stored: String,
    },
    Forgotten {
        memory_id: MemoryId,
        summary: String,
    },
    Correction {
        memory_id: MemoryId,
        old: String,
        new: String,
    },
    Reinforcement {
        memory_id: MemoryId,
    },
}

/// Configuration for stream-based cognition alerts.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Keyword overlap ratio above which a potential contradiction is flagged (default: 0.5).
    pub contradiction_keyword_ratio: f32,
    /// Keyword overlap ratio above which a reinforcement alert is emitted (default: 0.8).
    pub reinforcement_threshold: f32,
    /// Maximum number of tokens held in the ring buffer (default: 1000).
    pub buffer_size: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            contradiction_keyword_ratio: 0.5,
            reinforcement_threshold: 0.8,
            buffer_size: 1000,
        }
    }
}

struct StreamState {
    buffer: VecDeque<String>,
    accumulated: String,
    buffer_size: usize,
}

pub struct CognitionStream {
    state: Mutex<StreamState>,
    config: StreamConfig,
}

impl CognitionStream {
    pub fn new(buffer_size: usize) -> Self {
        Self::with_config(StreamConfig {
            buffer_size,
            ..StreamConfig::default()
        })
    }

    pub fn with_config(config: StreamConfig) -> Self {
        Self {
            state: Mutex::new(StreamState {
                buffer: VecDeque::with_capacity(config.buffer_size),
                accumulated: String::new(),
                buffer_size: config.buffer_size,
            }),
            config,
        }
    }

    pub fn feed_token(&self, token: &str) {
        let mut state = self.state.lock();
        if state.buffer.len() >= state.buffer_size {
            // Drain oldest token into accumulated text before evicting
            if let Some(old) = state.buffer.pop_front() {
                state.accumulated.push_str(&old);
            }
        }
        state.buffer.push_back(token.to_string());
    }

    pub fn check_alerts(&self, known_facts: &[(MemoryId, String)]) -> Vec<StreamAlert> {
        let state = self.state.lock();
        let mut full_text = state.accumulated.clone();
        for t in &state.buffer {
            full_text.push_str(t);
        }
        let full_lower = full_text.to_lowercase();

        let mut alerts = Vec::new();
        for (memory_id, fact) in known_facts {
            let fact_lower = fact.to_lowercase();
            // Extract significant keywords (3+ chars) from the stored fact
            let keywords: Vec<&str> = fact_lower
                .split_whitespace()
                .filter(|w| w.len() >= 3)
                .collect();

            if keywords.is_empty() {
                continue;
            }

            let matched = keywords
                .iter()
                .filter(|kw| full_lower.contains(*kw))
                .count();
            let ratio = matched as f32 / keywords.len() as f32;

            // High keyword overlap but not identical text = potential contradiction
            if ratio > self.config.contradiction_keyword_ratio && !full_lower.contains(&fact_lower)
            {
                // Check for negation patterns that suggest contradiction
                let has_negation = full_lower.contains("not ")
                    || full_lower.contains("never ")
                    || full_lower.contains("don't ")
                    || full_lower.contains("doesn't ")
                    || full_lower.contains("isn't ")
                    || full_lower.contains("actually ")
                    || full_lower.contains("instead ");

                if has_negation {
                    alerts.push(StreamAlert::Contradiction {
                        memory_id: *memory_id,
                        ai_said: full_text.clone(),
                        stored: fact.clone(),
                    });
                } else if ratio > self.config.reinforcement_threshold {
                    alerts.push(StreamAlert::Reinforcement {
                        memory_id: *memory_id,
                    });
                }
            } else if ratio > self.config.reinforcement_threshold {
                alerts.push(StreamAlert::Reinforcement {
                    memory_id: *memory_id,
                });
            }
        }
        alerts
    }

    pub fn drain_buffer(&self) -> String {
        let mut state = self.state.lock();
        let mut result = std::mem::take(&mut state.accumulated);
        for t in state.buffer.drain(..) {
            result.push_str(&t);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feed_and_drain() {
        let stream = CognitionStream::new(100);
        stream.feed_token("hello ");
        stream.feed_token("world");
        assert_eq!(stream.drain_buffer(), "hello world");
        assert_eq!(stream.drain_buffer(), "");
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let stream = CognitionStream::new(2);
        stream.feed_token("a");
        stream.feed_token("b");
        stream.feed_token("c");
        let result = stream.drain_buffer();
        assert_eq!(result, "abc");
    }

    #[test]
    fn test_contradiction_alert() {
        let stream = CognitionStream::new(100);
        let mid = MemoryId::new();
        stream.feed_token("The system does not use PostgreSQL, actually it uses MySQL");

        let facts = vec![(mid, "The system uses PostgreSQL for storage".to_string())];
        let alerts = stream.check_alerts(&facts);
        assert!(
            alerts
                .iter()
                .any(|a| matches!(a, StreamAlert::Contradiction { .. })),
            "Expected contradiction alert, got: {:?}",
            alerts
        );
    }
}
