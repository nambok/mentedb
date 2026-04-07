use mentedb_core::types::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::path::Path;

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
const REINFORCEMENT_BONUS: u32 = 2;

/// Basic topic normalization: lowercase, collapse whitespace, trim.
/// This handles the easy cases (casing, extra spaces) without attempting
/// semantic canonicalization (tracked in #22).
fn normalize_topic(raw: &str) -> String {
    raw.split_whitespace()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Tracks topic transitions as a Markov chain. Maps
/// from_topic -> (to_topic -> frequency_count).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransitionMap {
    transitions: HashMap<String, HashMap<String, u32>>,
}

impl TransitionMap {
    pub fn record(&mut self, from: &str, to: &str) {
        let from = normalize_topic(from);
        let to = normalize_topic(to);
        *self
            .transitions
            .entry(from)
            .or_default()
            .entry(to)
            .or_insert(0) += 1;
    }

    pub fn reinforce(&mut self, from: &str, to: &str) {
        let from = normalize_topic(from);
        let to = normalize_topic(to);
        *self
            .transitions
            .entry(from)
            .or_default()
            .entry(to)
            .or_insert(0) += REINFORCEMENT_BONUS;
    }

    pub fn decay(&mut self, from: &str, to: &str) {
        let from = normalize_topic(from);
        let to = normalize_topic(to);
        if let Some(targets) = self.transitions.get_mut(&from) {
            if let Some(count) = targets.get_mut(&to) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    targets.remove(&to);
                }
            }
            if targets.is_empty() {
                self.transitions.remove(&from);
            }
        }
    }

    /// Returns the top N predicted topics from a given topic,
    /// sorted by frequency descending.
    pub fn predict_from(&self, topic: &str, limit: usize) -> Vec<(String, u32)> {
        let topic = normalize_topic(topic);
        let Some(targets) = self.transitions.get(&topic) else {
            return Vec::new();
        };
        let mut ranked: Vec<(String, u32)> = targets.iter().map(|(t, &c)| (t.clone(), c)).collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        ranked.truncate(limit);
        ranked
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn total_transitions(&self) -> usize {
        self.transitions.values().map(|t| t.len()).sum()
    }

    /// Save the transition map to a JSON file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let json = serde_json::to_string(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load a transition map from a JSON file, merging counts into the
    /// current map so that patterns accumulate across sessions.
    pub fn load(&mut self, path: &Path) -> io::Result<()> {
        let json = std::fs::read_to_string(path)?;
        let loaded: TransitionMap = serde_json::from_str(&json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        for (from, targets) in loaded.transitions {
            let entry = self.transitions.entry(from).or_default();
            for (to, count) in targets {
                *entry.entry(to).or_insert(0) += count;
            }
        }
        Ok(())
    }
}

pub struct TrajectoryTracker {
    trajectory: Vec<TrajectoryNode>,
    max_turns: usize,
    pub transitions: TransitionMap,
}

impl TrajectoryTracker {
    pub fn new(max_turns: usize) -> Self {
        Self {
            trajectory: Vec::new(),
            max_turns,
            transitions: TransitionMap::default(),
        }
    }

    pub fn record_turn(&mut self, turn: TrajectoryNode) {
        if let Some(prev) = self.trajectory.last() {
            self.transitions
                .record(&prev.topic_summary, &turn.topic_summary);
        }

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
        let mut seen = ahash::AHashSet::new();

        let Some(last) = self.trajectory.last() else {
            return predictions;
        };

        // Learned transitions are the strongest signal
        let learned = self.transitions.predict_from(&last.topic_summary, 3);
        for (topic, _count) in &learned {
            if seen.insert(topic.clone()) {
                predictions.push(topic.clone());
            }
        }

        // Open questions fill remaining slots
        for q in &last.open_questions {
            if predictions.len() >= 3 {
                break;
            }
            if seen.insert(q.clone()) {
                predictions.push(q.clone());
            }
        }

        // Continuation of current topic
        if predictions.len() < 3 {
            let cont = format!("{} (continued)", last.topic_summary);
            if seen.insert(cont.clone()) {
                predictions.push(cont);
            }
        }

        // Revisit previous topic
        if predictions.len() < 3 && self.trajectory.len() >= 2 {
            let prev = &self.trajectory[self.trajectory.len() - 2];
            let rev = format!("{} (revisit)", prev.topic_summary);
            if seen.insert(rev.clone()) {
                predictions.push(rev);
            }
        }

        predictions.truncate(3);
        predictions
    }

    /// Called when the speculative cache gets a hit. Reinforces the
    /// transition from the previous topic to the hit topic.
    pub fn reinforce_transition(&mut self, hit_topic: &str) {
        if let Some(last) = self.trajectory.last() {
            self.transitions.reinforce(&last.topic_summary, hit_topic);
        }
    }

    /// Called when the speculative cache misses. Slightly decays the
    /// transition from the previous topic to the predicted topic.
    pub fn decay_transition(&mut self, predicted_topic: &str) {
        if let Some(last) = self.trajectory.last() {
            self.transitions.decay(&last.topic_summary, predicted_topic);
        }
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

    fn make_turn(
        id: u64,
        summary: &str,
        state: DecisionState,
        questions: Vec<&str>,
    ) -> TrajectoryNode {
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
        tracker.record_turn(make_turn(
            1,
            "JWT auth design",
            DecisionState::Investigating,
            vec![],
        ));
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
            tracker.record_turn(make_turn(
                i,
                &format!("turn {}", i),
                DecisionState::Investigating,
                vec![],
            ));
        }
        assert_eq!(tracker.get_trajectory().len(), MAX_TURNS_DEFAULT);
        assert_eq!(tracker.get_trajectory()[0].turn_id, 5);
    }

    #[test]
    fn test_transition_recording() {
        let mut tracker = TrajectoryTracker::default();
        tracker.record_turn(make_turn(1, "auth", DecisionState::Investigating, vec![]));
        tracker.record_turn(make_turn(
            2,
            "database",
            DecisionState::Investigating,
            vec![],
        ));
        tracker.record_turn(make_turn(3, "auth", DecisionState::Investigating, vec![]));
        tracker.record_turn(make_turn(
            4,
            "database",
            DecisionState::Investigating,
            vec![],
        ));
        tracker.record_turn(make_turn(5, "auth", DecisionState::Investigating, vec![]));
        tracker.record_turn(make_turn(
            6,
            "deployment",
            DecisionState::Investigating,
            vec![],
        ));

        // auth -> database happened twice, auth -> deployment once
        let preds = tracker.transitions.predict_from("auth", 5);
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].0, "database");
        assert_eq!(preds[0].1, 2);
        assert_eq!(preds[1].0, "deployment");
        assert_eq!(preds[1].1, 1);
    }

    #[test]
    fn test_learned_predictions_take_priority() {
        let mut tracker = TrajectoryTracker::default();

        // Build a pattern: auth -> database (3 times)
        for _ in 0..3 {
            tracker.record_turn(make_turn(0, "auth", DecisionState::Investigating, vec![]));
            tracker.record_turn(make_turn(
                0,
                "database",
                DecisionState::Investigating,
                vec![],
            ));
        }

        // Now land on auth with an open question
        tracker.record_turn(make_turn(
            0,
            "auth",
            DecisionState::Investigating,
            vec!["how to handle JWT expiry?"],
        ));

        let preds = tracker.predict_next_topics();
        // Learned transition "database" should come first
        assert_eq!(preds[0], "database");
    }

    #[test]
    fn test_reinforce_and_decay() {
        let mut map = TransitionMap::default();
        map.record("auth", "database");
        map.record("auth", "database");
        assert_eq!(map.predict_from("auth", 1)[0].1, 2);

        // Reinforce adds bonus
        map.reinforce("auth", "database");
        assert_eq!(map.predict_from("auth", 1)[0].1, 4);

        // Decay subtracts 1
        map.decay("auth", "database");
        assert_eq!(map.predict_from("auth", 1)[0].1, 3);
    }

    #[test]
    fn test_decay_removes_zero_entries() {
        let mut map = TransitionMap::default();
        map.record("auth", "database");
        assert_eq!(map.total_transitions(), 1);

        map.decay("auth", "database");
        assert!(map.is_empty());
    }

    #[test]
    fn test_reinforce_via_tracker() {
        let mut tracker = TrajectoryTracker::default();
        tracker.record_turn(make_turn(1, "auth", DecisionState::Investigating, vec![]));
        tracker.record_turn(make_turn(
            2,
            "database",
            DecisionState::Investigating,
            vec![],
        ));

        // One natural transition recorded
        assert_eq!(tracker.transitions.predict_from("auth", 1)[0].1, 1);

        // Simulate cache hit reinforcement
        tracker.reinforce_transition("database");
        assert_eq!(
            tracker.transitions.predict_from("database", 1)[0].1,
            REINFORCEMENT_BONUS
        );
    }

    #[test]
    fn test_no_duplicate_predictions() {
        let mut tracker = TrajectoryTracker::default();

        // Build pattern: auth -> database
        tracker.record_turn(make_turn(1, "auth", DecisionState::Investigating, vec![]));
        tracker.record_turn(make_turn(
            2,
            "database",
            DecisionState::Investigating,
            vec![],
        ));

        // Land on auth with "database" as an open question too
        tracker.record_turn(make_turn(
            3,
            "auth",
            DecisionState::Investigating,
            vec!["database"],
        ));

        let preds = tracker.predict_next_topics();
        let unique: ahash::AHashSet<&String> = preds.iter().collect();
        assert_eq!(preds.len(), unique.len(), "predictions should be unique");
    }

    #[test]
    fn test_normalization_collapses_variants() {
        let mut map = TransitionMap::default();
        map.record("Auth Setup", "database");
        map.record("auth setup", "DATABASE");
        map.record("  auth   setup  ", "  database  ");

        // All three should collapse into one transition with count 3
        let preds = map.predict_from("AUTH SETUP", 1);
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].0, "database");
        assert_eq!(preds[0].1, 3);
    }

    #[test]
    fn test_transition_map_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transitions.json");

        let mut map = TransitionMap::default();
        map.record("auth", "database");
        map.record("auth", "database");
        map.record("auth", "deploy");
        map.save(&path).unwrap();

        // Load into a fresh map — counts should carry over
        let mut loaded = TransitionMap::default();
        loaded.load(&path).unwrap();
        let preds = loaded.predict_from("auth", 5);
        assert_eq!(preds[0].0, "database");
        assert_eq!(preds[0].1, 2);
        assert_eq!(preds[1].0, "deploy");
        assert_eq!(preds[1].1, 1);
    }

    #[test]
    fn test_transition_map_load_merges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transitions.json");

        let mut map = TransitionMap::default();
        map.record("auth", "database");
        map.save(&path).unwrap();

        // Load into a map that already has data — counts should add
        let mut existing = TransitionMap::default();
        existing.record("auth", "database");
        existing.record("auth", "testing");
        existing.load(&path).unwrap();

        let preds = existing.predict_from("auth", 5);
        // database: 1 existing + 1 loaded = 2
        assert_eq!(preds[0].0, "database");
        assert_eq!(preds[0].1, 2);
        // testing: 1 existing only
        assert_eq!(preds[1].0, "testing");
        assert_eq!(preds[1].1, 1);
    }
}
