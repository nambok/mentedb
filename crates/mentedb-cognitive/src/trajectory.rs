use crate::llm::{CognitiveLlmService, LlmJudge};
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
///
/// Also maintains a learned topic cache that maps raw user messages
/// to canonical topic labels, so repeated patterns skip the LLM.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransitionMap {
    transitions: HashMap<String, HashMap<String, u32>>,
    /// Learned mapping from raw topic strings to canonical labels.
    /// Built up over time via LLM canonicalization calls.
    #[serde(default)]
    topic_cache: HashMap<String, String>,
}

#[derive(Serialize, Deserialize)]
struct TransitionSnapshot {
    version: u32,
    transitions: HashMap<String, HashMap<String, u32>>,
    #[serde(default)]
    topic_cache: HashMap<String, String>,
}

const TRANSITION_SNAPSHOT_VERSION: u32 = 2;

impl TransitionMap {
    /// Record a transition, applying the topic cache if available.
    /// If a canonical label exists for a raw topic, uses that instead.
    pub fn record(&mut self, from: &str, to: &str) {
        let from = self.resolve_topic(from);
        let to = self.resolve_topic(to);
        *self
            .transitions
            .entry(from)
            .or_default()
            .entry(to)
            .or_insert(0) += 1;
    }

    pub fn reinforce(&mut self, from: &str, to: &str) {
        let from = self.resolve_topic(from);
        let to = self.resolve_topic(to);
        *self
            .transitions
            .entry(from)
            .or_default()
            .entry(to)
            .or_insert(0) += REINFORCEMENT_BONUS;
    }

    pub fn decay(&mut self, from: &str, to: &str) {
        let from = self.resolve_topic(from);
        let to = self.resolve_topic(to);
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
        let topic = self.resolve_topic(topic);
        let Some(targets) = self.transitions.get(&topic) else {
            return Vec::new();
        };
        let mut ranked: Vec<(String, u32)> = targets.iter().map(|(t, &c)| (t.clone(), c)).collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        ranked.truncate(limit);
        ranked
    }

    /// Resolve a raw topic to its canonical label.
    /// Checks the learned cache first, falls back to normalize_topic().
    fn resolve_topic(&self, raw: &str) -> String {
        let normalized = normalize_topic(raw);
        self.topic_cache
            .get(&normalized)
            .cloned()
            .unwrap_or(normalized)
    }

    /// Store a learned mapping from a raw topic to its canonical label.
    pub fn learn_canonical(&mut self, raw: &str, canonical: &str) {
        let normalized = normalize_topic(raw);
        let canonical = normalize_topic(canonical);
        if normalized != canonical {
            self.topic_cache.insert(normalized, canonical);
        }
    }

    /// Look up a cached canonical label for a raw topic.
    pub fn get_canonical(&self, raw: &str) -> Option<&String> {
        let normalized = normalize_topic(raw);
        self.topic_cache.get(&normalized)
    }

    /// Returns the list of known canonical topic labels (deduped).
    pub fn known_topics(&self) -> Vec<String> {
        let mut topics: Vec<String> = self.topic_cache.values().cloned().collect();
        // Also include transition keys and targets that aren't in the cache
        for (key, targets) in &self.transitions {
            if !topics.contains(key) {
                topics.push(key.clone());
            }
            for target in targets.keys() {
                if !topics.contains(target) {
                    topics.push(target.clone());
                }
            }
        }
        topics.sort();
        topics.dedup();
        topics
    }

    pub fn topic_cache_size(&self) -> usize {
        self.topic_cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn total_transitions(&self) -> usize {
        self.transitions.values().map(|t| t.len()).sum()
    }

    /// Save the transition map to a JSON file. Prunes transitions with
    /// count below `min_count` to keep the file from growing unbounded.
    /// Uses atomic write (temp file + rename) to avoid corruption.
    pub fn save(&self, path: &Path, min_count: u32) -> io::Result<()> {
        let pruned: HashMap<String, HashMap<String, u32>> = self
            .transitions
            .iter()
            .filter_map(|(from, targets)| {
                let kept: HashMap<String, u32> = targets
                    .iter()
                    .filter(|(_, c)| **c >= min_count)
                    .map(|(t, &c)| (t.clone(), c))
                    .collect();
                if kept.is_empty() {
                    None
                } else {
                    Some((from.clone(), kept))
                }
            })
            .collect();
        let snapshot = TransitionSnapshot {
            version: TRANSITION_SNAPSHOT_VERSION,
            transitions: pruned,
            topic_cache: self.topic_cache.clone(),
        };
        let json = serde_json::to_string(&snapshot)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)
    }

    /// Load a transition map from a JSON file, merging counts into the
    /// current map so that patterns accumulate across sessions.
    /// Supports both v1 (no topic_cache) and v2 (with topic_cache) formats.
    pub fn load(&mut self, path: &Path) -> io::Result<()> {
        let json = std::fs::read_to_string(path)?;
        let snapshot: TransitionSnapshot = serde_json::from_str(&json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        if snapshot.version > TRANSITION_SNAPSHOT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported transition snapshot version: {} (expected <= {})",
                    snapshot.version, TRANSITION_SNAPSHOT_VERSION
                ),
            ));
        }

        for (from, targets) in snapshot.transitions {
            let entry = self.transitions.entry(from).or_default();
            for (to, count) in targets {
                *entry.entry(to).or_insert(0) += count;
            }
        }

        // Merge topic cache (loaded values don't overwrite existing)
        for (raw, canonical) in snapshot.topic_cache {
            self.topic_cache.entry(raw).or_insert(canonical);
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

    /// Record a turn with LLM-powered topic canonicalization.
    ///
    /// Checks the learned topic cache first (no LLM call needed for known patterns).
    /// On cache miss, asks the LLM for a canonical label and stores it for future use.
    /// Falls back to the heuristic path on any LLM error.
    pub async fn record_turn_with_llm<J: LlmJudge>(
        &mut self,
        mut turn: TrajectoryNode,
        llm: &CognitiveLlmService<J>,
    ) {
        // Canonicalize the incoming topic
        let canonical = self.canonicalize_topic(&turn.topic_summary, llm).await;
        turn.topic_summary = canonical;

        if let Some(prev) = self.trajectory.last() {
            self.transitions
                .record(&prev.topic_summary, &turn.topic_summary);
        }

        if self.trajectory.len() >= self.max_turns {
            self.trajectory.remove(0);
        }
        self.trajectory.push(turn);
    }

    /// Resolve a topic to its canonical label, using cache or LLM.
    async fn canonicalize_topic<J: LlmJudge>(
        &mut self,
        raw: &str,
        llm: &CognitiveLlmService<J>,
    ) -> String {
        // Tier 3: Check learned cache first (free, no LLM call)
        if let Some(cached) = self.transitions.get_canonical(raw) {
            return cached.clone();
        }

        // Tier 2: Ask the LLM
        let existing = self.transitions.known_topics();
        match llm.canonicalize_topic(raw, &existing).await {
            Ok(label) => {
                self.transitions.learn_canonical(raw, &label.topic);
                normalize_topic(&label.topic)
            }
            Err(_) => {
                // Tier 1: Fallback to normalize + exact match
                normalize_topic(raw)
            }
        }
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
        map.save(&path, 1).unwrap();

        // Load into a fresh map — counts should carry over
        let mut loaded = TransitionMap::default();
        loaded.load(&path).unwrap();
        let preds = loaded.predict_from("auth", 5);
        assert_eq!(preds[0].0, "database");
        assert_eq!(preds[0].1, 2);
        // deploy has count 1, should be saved with min_count=1
        assert_eq!(preds[1].0, "deploy");
        assert_eq!(preds[1].1, 1);
    }

    #[test]
    fn test_transition_map_save_prunes_low_counts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transitions.json");

        let mut map = TransitionMap::default();
        map.record("auth", "database");
        map.record("auth", "database");
        map.record("auth", "deploy"); // count 1
        map.save(&path, 2).unwrap(); // only keep count >= 2

        let mut loaded = TransitionMap::default();
        loaded.load(&path).unwrap();
        let preds = loaded.predict_from("auth", 5);
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].0, "database");
        assert_eq!(preds[0].1, 2);
    }

    #[test]
    fn test_transition_map_load_merges() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transitions.json");

        let mut map = TransitionMap::default();
        map.record("auth", "database");
        map.save(&path, 1).unwrap();

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

    #[test]
    fn test_topic_cache_learn_and_resolve() {
        let mut map = TransitionMap::default();

        // Without cache, "auth setup" and "configure authentication" are different keys
        map.record("auth setup", "database");
        map.record("configure authentication", "database");
        // Two separate entries
        assert_eq!(map.predict_from("auth setup", 1)[0].1, 1);

        // Teach the cache
        map.learn_canonical("auth setup", "authentication");
        map.learn_canonical("configure authentication", "authentication");

        // Now both resolve to "authentication"
        map.record("auth setup", "database");
        map.record("configure authentication", "database");
        let preds = map.predict_from("authentication", 1);
        assert_eq!(preds[0].0, "database");
        assert_eq!(preds[0].1, 2);
    }

    #[test]
    fn test_topic_cache_persists_across_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transitions.json");

        let mut map = TransitionMap::default();
        map.learn_canonical("auth setup", "authentication");
        map.learn_canonical("db design", "database");
        map.record("auth setup", "db design");
        map.save(&path, 1).unwrap();

        let mut loaded = TransitionMap::default();
        loaded.load(&path).unwrap();

        // Cache should survive the roundtrip
        assert_eq!(
            loaded.get_canonical("auth setup"),
            Some(&"authentication".to_string())
        );
        assert_eq!(
            loaded.get_canonical("db design"),
            Some(&"database".to_string())
        );
        // Transition should use canonical keys
        let preds = loaded.predict_from("authentication", 1);
        assert_eq!(preds[0].0, "database");
    }

    #[test]
    fn test_known_topics_includes_cache_and_transitions() {
        let mut map = TransitionMap::default();
        map.learn_canonical("auth setup", "authentication");
        map.record("deployment", "testing");

        let topics = map.known_topics();
        assert!(topics.contains(&"authentication".to_string()));
        assert!(topics.contains(&"deployment".to_string()));
        assert!(topics.contains(&"testing".to_string()));
    }

    #[test]
    fn test_v1_snapshot_loads_without_topic_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transitions.json");

        // Write a v1 snapshot manually (no topic_cache field)
        let v1_json = r#"{"version":1,"transitions":{"auth":{"database":3}}}"#;
        std::fs::write(&path, v1_json).unwrap();

        let mut map = TransitionMap::default();
        map.load(&path).unwrap();

        // Transitions loaded, cache empty
        let preds = map.predict_from("auth", 1);
        assert_eq!(preds[0].0, "database");
        assert_eq!(preds[0].1, 3);
        assert_eq!(map.topic_cache_size(), 0);
    }

    use crate::llm::{CognitiveLlmService, MockLlmJudge};

    #[tokio::test]
    async fn test_record_turn_with_llm_canonicalizes() {
        let judge = MockLlmJudge::new(r#"{"topic": "authentication", "is_new": false}"#);
        let llm = CognitiveLlmService::new(judge);
        let mut tracker = TrajectoryTracker::default();

        tracker
            .record_turn_with_llm(
                make_turn(
                    1,
                    "auth setup question",
                    DecisionState::Investigating,
                    vec![],
                ),
                &llm,
            )
            .await;

        // Topic should be canonicalized to "authentication"
        assert_eq!(tracker.get_trajectory()[0].topic_summary, "authentication");
        // Cache should have the learned mapping
        assert_eq!(
            tracker.transitions.get_canonical("auth setup question"),
            Some(&"authentication".to_string())
        );
    }

    #[tokio::test]
    async fn test_record_turn_with_llm_uses_cache_on_repeat() {
        let judge = MockLlmJudge::new(r#"{"topic": "authentication", "is_new": false}"#);
        let llm = CognitiveLlmService::new(judge);
        let mut tracker = TrajectoryTracker::default();

        // First call learns the mapping
        tracker
            .record_turn_with_llm(
                make_turn(1, "auth setup", DecisionState::Investigating, vec![]),
                &llm,
            )
            .await;

        // Pre-teach a different canonical for the next topic
        // to prove the cache is being used (not the LLM)
        tracker
            .transitions
            .learn_canonical("configure auth", "authentication");

        // This should use cache, NOT the MockLlmJudge
        // (MockLlmJudge always returns "authentication" anyway, but the point is
        // the cache path is exercised — no LLM call needed)
        tracker
            .record_turn_with_llm(
                make_turn(2, "configure auth", DecisionState::Investigating, vec![]),
                &llm,
            )
            .await;

        assert_eq!(tracker.get_trajectory()[1].topic_summary, "authentication");

        // Both turns should contribute to the same transition key
        // Turn 1: "authentication" recorded (no prev)
        // Turn 2: "authentication" -> "authentication" (same topic)
        assert_eq!(tracker.get_trajectory().len(), 2);
    }

    #[tokio::test]
    async fn test_record_turn_with_llm_transitions_accumulate() {
        let judge = MockLlmJudge::new(r#"{"topic": "authentication", "is_new": false}"#);
        let llm = CognitiveLlmService::new(judge);
        let mut tracker = TrajectoryTracker::default();

        // Record first turn (no transition yet)
        tracker
            .record_turn_with_llm(
                make_turn(1, "auth setup", DecisionState::Investigating, vec![]),
                &llm,
            )
            .await;

        // Now record with a different mock response for "database"
        let judge2 = MockLlmJudge::new(r#"{"topic": "database", "is_new": false}"#);
        let llm2 = CognitiveLlmService::new(judge2);
        tracker
            .record_turn_with_llm(
                make_turn(2, "db schema design", DecisionState::Investigating, vec![]),
                &llm2,
            )
            .await;

        // Should have transition: authentication -> database
        let preds = tracker.transitions.predict_from("authentication", 3);
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].0, "database");
        assert_eq!(preds[0].1, 1);
    }
}
