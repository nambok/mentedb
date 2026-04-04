use mentedb_core::types::Timestamp;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionState {
    Investigating,
    NarrowedTo(String),
    Decided(String),
    Interrupted,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryNode {
    pub turn_id: u64,
    pub topic_embedding: Vec<f32>,
    pub topic_summary: String,
    pub decision_state: DecisionState,
    pub open_questions: Vec<String>,
    pub timestamp: Timestamp,
}

const MAX_TURNS_DEFAULT: usize = 100;

pub struct TrajectoryTracker {
    trajectory: Vec<TrajectoryNode>,
    max_turns: usize,
}

impl TrajectoryTracker {
    pub fn new(max_turns: usize) -> Self {
        Self {
            trajectory: Vec::new(),
            max_turns,
        }
    }

    pub fn record_turn(&mut self, turn: TrajectoryNode) {
        if self.trajectory.len() >= self.max_turns {
            self.trajectory.remove(0);
        }
        self.trajectory.push(turn);
    }

    pub fn get_trajectory(&self) -> &[TrajectoryNode] {
        &self.trajectory
    }

    pub fn get_resume_context(&self) -> Option<String> {
        if self.trajectory.is_empty() {
            return None;
        }

        let mut parts = Vec::new();

        // Find the last non-completed topic
        if let Some(last) = self.trajectory.last() {
            parts.push(format!("You were working on: {}", last.topic_summary));

            match &last.decision_state {
                DecisionState::Investigating => {
                    parts.push("Status: Still investigating.".to_string());
                }
                DecisionState::NarrowedTo(choice) => {
                    parts.push(format!("You narrowed down to: {}", choice));
                }
                DecisionState::Decided(decision) => {
                    parts.push(format!("You decided on: {}", decision));
                }
                DecisionState::Interrupted => {
                    parts.push("Status: Was interrupted before completion.".to_string());
                }
                DecisionState::Completed => {
                    parts.push("Status: Completed.".to_string());
                }
            }

            if !last.open_questions.is_empty() {
                let qs: Vec<String> = last
                    .open_questions
                    .iter()
                    .map(|q| format!("- {}", q))
                    .collect();
                parts.push(format!("Open questions:\n{}", qs.join("\n")));
            }
        }

        // Add recent trajectory summary
        if self.trajectory.len() > 1 {
            let recent: Vec<String> = self
                .trajectory
                .iter()
                .rev()
                .skip(1)
                .take(3)
                .rev()
                .map(|t| t.topic_summary.clone())
                .collect();
            parts.push(format!("Recent trajectory: {}", recent.join(" → ")));
        }

        Some(parts.join(" "))
    }

    pub fn predict_next_topics(&self) -> Vec<String> {
        let mut predictions = Vec::new();

        if let Some(last) = self.trajectory.last() {
            // Open questions are the best predictors
            for q in &last.open_questions {
                predictions.push(q.clone());
                if predictions.len() >= 3 {
                    return predictions;
                }
            }

            // Last topic as continuation
            if predictions.len() < 3 {
                predictions.push(format!("{} (continued)", last.topic_summary));
            }

            // Look at trajectory pattern for related topics
            if predictions.len() < 3 && self.trajectory.len() >= 2 {
                let prev = &self.trajectory[self.trajectory.len() - 2];
                predictions.push(format!("{} (revisit)", prev.topic_summary));
            }
        }

        predictions.truncate(3);
        predictions
    }
}

impl Default for TrajectoryTracker {
    fn default() -> Self {
        Self::new(MAX_TURNS_DEFAULT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_turn(id: u64, summary: &str, state: DecisionState, questions: Vec<&str>) -> TrajectoryNode {
        TrajectoryNode {
            turn_id: id,
            topic_embedding: vec![0.0; 4],
            topic_summary: summary.to_string(),
            decision_state: state,
            open_questions: questions.into_iter().map(String::from).collect(),
            timestamp: id * 1000,
        }
    }

    #[test]
    fn test_record_and_resume() {
        let mut tracker = TrajectoryTracker::default();
        tracker.record_turn(make_turn(1, "JWT auth design", DecisionState::Investigating, vec![]));
        tracker.record_turn(make_turn(
            2,
            "Token refresh strategy",
            DecisionState::Decided("short-lived access tokens (15min)".into()),
            vec!["Where to store refresh tokens?"],
        ));

        let ctx = tracker.get_resume_context().unwrap();
        assert!(ctx.contains("Token refresh strategy"));
        assert!(ctx.contains("short-lived access tokens"));
        assert!(ctx.contains("refresh tokens"));
    }

    #[test]
    fn test_predict_topics() {
        let mut tracker = TrajectoryTracker::default();
        tracker.record_turn(make_turn(
            1,
            "Database schema",
            DecisionState::Decided("normalized".into()),
            vec!["How to handle migrations?", "Index strategy?"],
        ));

        let preds = tracker.predict_next_topics();
        assert!(!preds.is_empty());
        assert!(preds.iter().any(|p| p.contains("migrations")));
    }

    #[test]
    fn test_fifo_eviction() {
        let mut tracker = TrajectoryTracker::default();
        for i in 0..105 {
            tracker.record_turn(make_turn(i, &format!("turn {}", i), DecisionState::Investigating, vec![]));
        }
        assert_eq!(tracker.get_trajectory().len(), MAX_TURNS_DEFAULT);
        assert_eq!(tracker.get_trajectory()[0].turn_id, 5);
    }
}
