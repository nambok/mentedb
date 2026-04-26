//! Unified `process_turn` orchestration.
//!
//! This module provides a single `MenteDb::process_turn()` method that
//! implements the full conversation-turn pipeline used by MCP servers,
//! cloud APIs, and any other client. The steps are:
//!
//! 1. Embed query + retrieve context (speculative cache → hybrid search)
//! 2. Check pain signals
//! 3. Store episodic memory (with write inference)
//! 4. Detect actions, proactive recall, corrections, sentiment
//! 5. Detect phantoms + stream contradiction check
//! 6. Update trajectory + ghost memories
//! 7. Update speculative cache
//! 8. Extract facts + link edges
//! 9. Auto-maintenance (decay / archival / consolidation on intervals)
//!
//! LLM-powered features (entity resolution, topic canonicalization,
//! contradiction verification, memory extraction) are **not** included
//! here — they are client-specific and should be layered on top of the
//! `ProcessTurnResult` by the caller.

use mentedb_cognitive::stream::StreamAlert;
use mentedb_cognitive::trajectory::{DecisionState, TrajectoryNode};
use mentedb_consolidation::FactExtractor;
use mentedb_context::{DeltaTracker, ScoredMemory};
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{AgentId, MemoryId, Timestamp};
use mentedb_core::{MemoryEdge, MemoryNode};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::MenteDb;

/// Input for a single conversation turn.
#[derive(Debug, Clone)]
pub struct ProcessTurnInput {
    /// The user's message text.
    pub user_message: String,
    /// The assistant's response (if available).
    pub assistant_response: Option<String>,
    /// Monotonically increasing turn number.
    pub turn_id: u64,
    /// Optional project/workspace scope for memory tagging.
    pub project_context: Option<String>,
    /// Agent UUID (defaults to nil if not provided).
    pub agent_id: Option<Uuid>,
}

/// A detected action from the conversation.
#[derive(Debug, Clone)]
pub struct DetectedAction {
    pub action_type: String,
    pub detail: String,
}

/// A pain warning matched against the current context.
#[derive(Debug, Clone)]
pub struct PainWarning {
    pub signal_id: MemoryId,
    pub intensity: f32,
    pub description: String,
}

/// A proactively recalled memory relevant to detected actions.
#[derive(Debug, Clone)]
pub struct ProactiveRecall {
    pub memory_id: MemoryId,
    pub content: String,
    pub relevance: f32,
    pub action_type: String,
}

/// Result of processing a conversation turn.
#[derive(Debug, Clone)]
pub struct ProcessTurnResult {
    /// Retrieved context memories (scored and ranked).
    pub context: Vec<ScoredMemory>,
    /// IDs of newly stored memories this turn.
    pub stored_ids: Vec<MemoryId>,
    /// The episodic memory ID for this turn.
    pub episodic_id: Option<MemoryId>,
    /// Pain warnings triggered by the user message.
    pub pain_warnings: Vec<PainWarning>,
    /// Whether context came from the speculative cache.
    pub cache_hit: bool,
    /// Number of write-inference actions applied.
    pub inference_actions: u32,
    /// Detected actions in the conversation.
    pub detected_actions: Vec<DetectedAction>,
    /// Proactively recalled memories.
    pub proactive_recalls: Vec<ProactiveRecall>,
    /// Auto-correction memory ID if a correction was detected.
    pub correction_id: Option<MemoryId>,
    /// Sentiment score (-1.0 to 1.0).
    pub sentiment: f32,
    /// Number of phantom (knowledge gap) entities detected.
    pub phantom_count: usize,
    /// Number of stream contradictions detected.
    pub contradiction_count: usize,
    /// Predicted next topics from trajectory analysis.
    pub predicted_topics: Vec<String>,
    /// Number of facts extracted and linked.
    pub facts_extracted: usize,
    /// Number of edges created from fact extraction.
    pub edges_created: u32,
    /// Delta: memory IDs added since last turn.
    pub delta_added: Vec<MemoryId>,
    /// Delta: memory IDs removed since last turn.
    pub delta_removed: Vec<MemoryId>,
}

// ── Action detection keywords ──

const ACTION_KEYWORDS: &[(&str, &str)] = &[
    ("deploy", "deployment"),
    ("release", "release"),
    ("migrate", "migration"),
    ("refactor", "refactoring"),
    ("debug", "debugging"),
    ("install", "installation"),
    ("configure", "configuration"),
    ("test", "testing"),
    ("build", "building"),
    ("commit", "version_control"),
    ("merge", "version_control"),
    ("review", "code_review"),
];

// ── Correction indicators ──

const CORRECTION_INDICATORS: &[&str] = &[
    "actually",
    "correction",
    "i was wrong",
    "that's not right",
    "let me correct",
    "i meant",
    "not quite",
    "to clarify",
    "i misspoke",
    "update:",
    "scratch that",
];

// ── Speculation indicators (for ghost memories) ──

const SPECULATION_INDICATORS: &[&str] = &[
    "might be",
    "probably",
    "seems like",
    "i think",
    "looks like",
    "considering",
    "planning to",
    "thinking about",
    "maybe",
];

// ── Sentiment keywords ──

const POSITIVE_WORDS: &[&str] = &[
    "great",
    "excellent",
    "perfect",
    "thanks",
    "awesome",
    "love",
    "good",
    "nice",
    "wonderful",
    "fantastic",
    "helpful",
    "amazing",
    "brilliant",
    "works",
    "solved",
];

const NEGATIVE_WORDS: &[&str] = &[
    "bad",
    "terrible",
    "awful",
    "hate",
    "wrong",
    "broken",
    "fail",
    "error",
    "bug",
    "frustrating",
    "annoying",
    "useless",
    "horrible",
    "worst",
    "disappointed",
];

impl MenteDb {
    /// Process a single conversation turn through the full cognitive pipeline.
    ///
    /// This is the unified entry point that replaces the duplicated orchestration
    /// in MCP and platform. It handles:
    /// - Context retrieval (speculative cache → hybrid search)
    /// - Pain signal checking
    /// - Episodic memory storage (with automatic write inference)
    /// - Action detection, proactive recall, corrections, sentiment
    /// - Phantom detection + stream contradiction checking
    /// - Trajectory tracking + ghost memory creation
    /// - Speculative cache pre-assembly
    /// - Fact extraction + edge linking
    /// - Auto-maintenance (decay / archival / consolidation)
    ///
    /// LLM-powered features (entity resolution, topic canonicalization,
    /// Bedrock extraction, etc.) should be layered on top by the caller.
    pub fn process_turn(
        &self,
        input: &ProcessTurnInput,
        delta_tracker: &mut DeltaTracker,
    ) -> crate::MenteResult<ProcessTurnResult> {
        let agent_id = AgentId(input.agent_id.unwrap_or(Uuid::nil()));
        let assistant_resp = input.assistant_response.as_deref().unwrap_or("");
        let conversation = format!(
            "User: {}\nAssistant: {}",
            input.user_message, assistant_resp
        );

        // §1: Embed query
        let query_embedding = self
            .embed_text(&input.user_message)?
            .unwrap_or_else(|| vec![0.0; 384]);

        // §1b: Try speculative cache, fall back to hybrid search
        let (context, current_ids, cache_hit) =
            self.retrieve_context(&input.user_message, &query_embedding, delta_tracker)?;

        // §2: Pain signals
        let pain_warnings = self.check_pain_signals(&input.user_message);

        // §2b: Delta tracking
        let delta = delta_tracker.compute_delta(&current_ids, &delta_tracker.last_served.clone());
        delta_tracker.update(&current_ids);

        // §3: Store episodic turn (write inference runs automatically inside store)
        let (stored_ids, episodic_id) =
            self.store_episodic(&conversation, agent_id, &input.project_context)?;

        // §4: Write inference on the episodic memory
        let inference_actions = if let Some(eid) = episodic_id {
            self.run_explicit_write_inference(eid)?
        } else {
            0
        };

        // §5: Extract facts + link edges
        let (facts_extracted, edges_created) = if let Some(eid) = episodic_id {
            self.extract_and_link_facts(eid)?
        } else {
            (0, 0)
        };

        // §6: Detect actions
        let combined_text = format!("{} {}", input.user_message, assistant_resp);
        let detected_actions = detect_actions(&combined_text);

        // §7: Proactive recall
        let proactive_recalls = self.proactive_recall(&detected_actions)?;

        // §8: Auto-detect corrections
        let correction_id = self.auto_detect_correction(
            &input.user_message,
            assistant_resp,
            agent_id,
            &input.project_context,
        )?;

        // §9: Sentiment analysis
        let sentiment = analyze_sentiment(&input.user_message);

        // §10: Phantom detection + stream contradiction check
        let (phantom_count, contradiction_count) = self.detect_phantoms_and_check_stream(
            &conversation,
            assistant_resp,
            &context,
            input.turn_id,
        );

        // §11: Trajectory + ghost memories
        let predicted_topics = self.update_trajectory(
            input,
            agent_id,
            &stored_ids,
            &detected_actions,
            &combined_text,
        )?;

        // §12: Speculative cache pre-assembly
        self.update_speculative_cache_from_predictions(&predicted_topics);

        // §13: Auto-maintenance
        self.maybe_run_maintenance(input.turn_id);

        Ok(ProcessTurnResult {
            context,
            stored_ids: stored_ids.clone(),
            episodic_id,
            pain_warnings,
            cache_hit,
            inference_actions,
            detected_actions,
            proactive_recalls,
            correction_id,
            sentiment,
            phantom_count,
            contradiction_count,
            predicted_topics,
            facts_extracted,
            edges_created,
            delta_added: delta.added,
            delta_removed: delta.removed,
        })
    }

    // ── Private helpers ──

    fn retrieve_context(
        &self,
        user_message: &str,
        query_embedding: &[f32],
        _delta_tracker: &DeltaTracker,
    ) -> crate::MenteResult<(Vec<ScoredMemory>, Vec<MemoryId>, bool)> {
        // Try speculative cache first
        if let Some(entry) = self.try_speculative_hit(user_message, Some(query_embedding)) {
            let matched: Vec<ScoredMemory> = entry
                .memory_ids
                .iter()
                .filter_map(|id| {
                    self.get_memory(*id).ok().map(|m| ScoredMemory {
                        memory: m,
                        score: 0.9,
                    })
                })
                .collect();
            if matched.len() >= entry.memory_ids.len() / 2 {
                let ids: Vec<MemoryId> = matched.iter().map(|sm| sm.memory.id).collect();
                return Ok((matched, ids, true));
            }
        }

        // Hybrid search fallback
        let now = now_us();
        let results =
            self.recall_hybrid_at(query_embedding, Some(user_message), 10, now, None, None)?;
        let scored: Vec<ScoredMemory> = results
            .iter()
            .filter_map(|(mid, score)| {
                self.get_memory(*mid).ok().map(|m| ScoredMemory {
                    memory: m,
                    score: *score,
                })
            })
            .collect();
        let ids: Vec<MemoryId> = scored.iter().map(|sm| sm.memory.id).collect();
        Ok((scored, ids, false))
    }

    fn check_pain_signals(&self, user_message: &str) -> Vec<PainWarning> {
        let context_words: Vec<String> = user_message
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        let all_signals = self.all_pain_signals();
        let mut scored: Vec<_> = all_signals
            .iter()
            .filter_map(|s| {
                let matched = s
                    .trigger_keywords
                    .iter()
                    .filter(|trigger| {
                        context_words
                            .iter()
                            .any(|ctx| ctx.contains(&trigger.to_lowercase()))
                    })
                    .count();
                if matched > 0 {
                    let relevance = matched as f32 / s.trigger_keywords.len().max(1) as f32;
                    let score = s.intensity * relevance;
                    Some((s, score))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .iter()
            .map(|(s, _)| PainWarning {
                signal_id: s.id,
                intensity: s.intensity,
                description: s.description.clone(),
            })
            .collect()
    }

    fn store_episodic(
        &self,
        conversation: &str,
        agent_id: AgentId,
        project_context: &Option<String>,
    ) -> crate::MenteResult<(Vec<MemoryId>, Option<MemoryId>)> {
        let embedding = self
            .embed_text(conversation)?
            .unwrap_or_else(|| vec![0.0; 384]);
        let mut node = MemoryNode::new(
            agent_id,
            MemoryType::Episodic,
            conversation.to_string(),
            embedding,
        );
        node.tags.push("turn".to_string());
        if let Some(ctx) = project_context {
            node.tags.push(format!("scope:project:{}", ctx));
        }
        let id = node.id;
        self.store(node)?;
        Ok((vec![id], Some(id)))
    }

    fn run_explicit_write_inference(&self, id: MemoryId) -> crate::MenteResult<u32> {
        let target = self.get_memory(id)?;
        let similar_ids = self
            .recall_similar(&target.embedding, 50)
            .unwrap_or_default();
        let existing: Vec<MemoryNode> = similar_ids
            .iter()
            .filter_map(|(mid, _)| {
                if *mid != id {
                    self.get_memory(*mid).ok()
                } else {
                    None
                }
            })
            .collect();

        let engine = crate::WriteInferenceEngine::new();
        let actions = engine.infer_on_write(&target, &existing, &[]);
        let now = now_us();
        let mut applied = 0u32;

        for action in &actions {
            match action {
                crate::InferredAction::FlagContradiction {
                    existing: ex, new, ..
                } => {
                    let edge = MemoryEdge {
                        source: *new,
                        target: *ex,
                        edge_type: EdgeType::Contradicts,
                        weight: 1.0,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: None,
                    };
                    let _ = self.relate(edge);
                    applied += 1;
                }
                crate::InferredAction::MarkObsolete {
                    memory,
                    superseded_by,
                } => {
                    let edge = MemoryEdge {
                        source: *superseded_by,
                        target: *memory,
                        edge_type: EdgeType::Supersedes,
                        weight: 1.0,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: None,
                    };
                    let _ = self.relate(edge);
                    applied += 1;
                }
                crate::InferredAction::CreateEdge {
                    source,
                    target,
                    edge_type,
                    weight,
                } => {
                    let edge = MemoryEdge {
                        source: *source,
                        target: *target,
                        edge_type: *edge_type,
                        weight: *weight,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: None,
                    };
                    let _ = self.relate(edge);
                    applied += 1;
                }
                crate::InferredAction::UpdateConfidence {
                    memory,
                    new_confidence,
                } => {
                    if let Ok(mut mem) = self.get_memory(*memory) {
                        mem.confidence = *new_confidence;
                        let _ = self.store(mem);
                        applied += 1;
                    }
                }
                crate::InferredAction::InvalidateMemory {
                    memory,
                    valid_until,
                    ..
                } => {
                    if let Ok(mut mem) = self.get_memory(*memory) {
                        mem.valid_until = Some(*valid_until);
                        let _ = self.store(mem);
                        applied += 1;
                    }
                }
                crate::InferredAction::UpdateContent {
                    memory,
                    new_content,
                    ..
                } => {
                    if let Ok(mut mem) = self.get_memory(*memory) {
                        mem.content = new_content.clone();
                        let _ = self.store(mem);
                        applied += 1;
                    }
                }
                crate::InferredAction::PropagateBeliefChange { .. } => {
                    applied += 1;
                }
            }
        }
        Ok(applied)
    }

    fn extract_and_link_facts(&self, id: MemoryId) -> crate::MenteResult<(usize, u32)> {
        let target = self.get_memory(id)?;
        let extractor = FactExtractor::new();
        let facts = extractor.extract_facts(&target);
        if facts.is_empty() {
            return Ok((0, 0));
        }

        let similar_ids = self
            .recall_similar(&target.embedding, 50)
            .unwrap_or_default();
        let nearby: Vec<MemoryNode> = similar_ids
            .iter()
            .filter_map(|(mid, _)| {
                if *mid != id {
                    self.get_memory(*mid).ok()
                } else {
                    None
                }
            })
            .collect();

        let now = now_us();
        let mut edges_created = 0u32;
        for fact in &facts {
            for other in &nearby {
                if other.content.contains(&fact.subject) || other.content.contains(&fact.object) {
                    let edge = MemoryEdge {
                        source: id,
                        target: other.id,
                        edge_type: EdgeType::Related,
                        weight: 0.5,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: None,
                    };
                    let _ = self.relate(edge);
                    edges_created += 1;
                }
            }
        }
        Ok((facts.len(), edges_created))
    }

    fn proactive_recall(
        &self,
        actions: &[DetectedAction],
    ) -> crate::MenteResult<Vec<ProactiveRecall>> {
        let mut recalls = Vec::new();
        for action in actions {
            let search_query = format!("{} {}", action.action_type, action.detail);
            let Some(emb) = self.embed_text(&search_query)? else {
                continue;
            };
            let results = self.recall_similar(&emb, 3).unwrap_or_default();
            for (mid, score) in &results {
                if let Ok(mem) = self.get_memory(*mid) {
                    recalls.push(ProactiveRecall {
                        memory_id: *mid,
                        content: mem.content.clone(),
                        relevance: *score,
                        action_type: action.action_type.clone(),
                    });
                }
            }
        }
        Ok(recalls)
    }

    fn auto_detect_correction(
        &self,
        user_message: &str,
        assistant_resp: &str,
        agent_id: AgentId,
        project_context: &Option<String>,
    ) -> crate::MenteResult<Option<MemoryId>> {
        let combined = format!("{} {}", user_message, assistant_resp).to_lowercase();
        let is_correction = CORRECTION_INDICATORS
            .iter()
            .any(|ind| combined.contains(ind));
        if !is_correction {
            return Ok(None);
        }

        let correction_content = format!("Correction: {}", user_message);
        let embedding = self
            .embed_text(&correction_content)?
            .unwrap_or_else(|| vec![0.0; 384]);
        let mut node = MemoryNode::new(
            agent_id,
            MemoryType::Semantic,
            correction_content,
            embedding,
        );
        node.tags.push("auto-correction".to_string());
        node.tags.push("anti_pattern".to_string());
        if let Some(ctx) = project_context {
            node.tags.push(format!("scope:project:{}", ctx));
        }
        let id = node.id;
        self.store(node)?;
        Ok(Some(id))
    }

    fn detect_phantoms_and_check_stream(
        &self,
        conversation: &str,
        assistant_resp: &str,
        context: &[ScoredMemory],
        turn_id: u64,
    ) -> (usize, usize) {
        // Phantom detection
        let known: Vec<String> = context
            .iter()
            .flat_map(|sm| {
                sm.memory
                    .content
                    .split_whitespace()
                    .map(|w| w.to_lowercase())
            })
            .collect();
        let phantoms = self.detect_phantoms(conversation, &known, turn_id);
        let phantom_count = phantoms.len();

        // Stream contradiction check
        let known_facts: Vec<(MemoryId, String)> = context
            .iter()
            .map(|sm| (sm.memory.id, sm.memory.content.clone()))
            .collect();
        let contradiction_count = if !known_facts.is_empty() {
            self.feed_stream_token(assistant_resp);
            let alerts = self.check_stream_alerts(&known_facts);
            alerts
                .iter()
                .filter(|a| matches!(a, StreamAlert::Contradiction { .. }))
                .count()
        } else {
            0
        };

        (phantom_count, contradiction_count)
    }

    fn update_trajectory(
        &self,
        input: &ProcessTurnInput,
        agent_id: AgentId,
        stored_ids: &[MemoryId],
        detected_actions: &[DetectedAction],
        combined_text: &str,
    ) -> crate::MenteResult<Vec<String>> {
        let decision_state = if stored_ids.is_empty() {
            DecisionState::Investigating
        } else {
            DecisionState::Completed
        };

        let raw_topic = if input.user_message.len() > 100 {
            format!("{}...", &input.user_message[..100])
        } else {
            input.user_message.clone()
        };

        let topic_embedding = self
            .embed_text(&raw_topic)?
            .unwrap_or_else(|| vec![0.0; 384]);

        let node = TrajectoryNode {
            turn_id: input.turn_id,
            topic_summary: raw_topic,
            topic_embedding,
            decision_state,
            open_questions: Vec::new(),
            timestamp: now_us(),
        };
        self.record_trajectory_turn(node);
        let predictions = self.predict_next_topics();

        // Ghost memory: store speculative content at low confidence
        let has_speculation = SPECULATION_INDICATORS
            .iter()
            .any(|s| combined_text.contains(s));
        if has_speculation && !detected_actions.is_empty() {
            let ghost_content = format!(
                "Unconfirmed: {}",
                input.user_message.chars().take(300).collect::<String>()
            );
            if let Ok(Some(ghost_emb)) = self.embed_text(&ghost_content) {
                let mut ghost_node =
                    MemoryNode::new(agent_id, MemoryType::Semantic, ghost_content, ghost_emb);
                ghost_node.confidence = 0.3;
                ghost_node.tags = vec!["ghost-memory".to_string(), "unconfirmed".to_string()];
                if let Some(ctx) = &input.project_context {
                    ghost_node.tags.push(format!("scope:project:{}", ctx));
                }
                let _ = self.store(ghost_node);
            }
        }

        Ok(predictions)
    }

    fn update_speculative_cache_from_predictions(&self, predictions: &[String]) {
        if predictions.is_empty() {
            return;
        }
        // We need to capture `self` for use in the closure, but pre_assemble_speculative
        // takes &self already. We build the closure to search via self.
        let predictions_owned = predictions.to_vec();
        self.pre_assemble_speculative(predictions_owned, |topic| {
            let topic_emb = self.embed_text(topic).ok()??;
            let similar_ids = self.recall_similar(&topic_emb, 5).ok()?;
            if similar_ids.is_empty() {
                return None;
            }
            let mut context_parts = Vec::new();
            let mut memory_ids = Vec::new();
            for (mid, _) in &similar_ids {
                if let Ok(mem) = self.get_memory(*mid) {
                    context_parts.push(mem.content.clone());
                    memory_ids.push(*mid);
                }
            }
            if memory_ids.is_empty() {
                return None;
            }
            Some((context_parts.join("\n---\n"), memory_ids, None))
        });
    }

    fn maybe_run_maintenance(&self, turn_id: u64) {
        if turn_id == 0 {
            return;
        }

        // Every 50 turns: salience decay
        if turn_id.is_multiple_of(50) {
            match self.apply_decay_global() {
                Ok(updated) => {
                    debug!(turn_id, updated, "auto-maintenance: decay applied");
                }
                Err(e) => {
                    warn!(turn_id, error = %e, "auto-maintenance: decay failed");
                }
            }
        }

        // Every 100 turns: archival evaluation
        if turn_id.is_multiple_of(100) {
            match self.evaluate_archival_global() {
                Ok(decisions) => {
                    let mut archived = 0u64;
                    for (id, decision) in &decisions {
                        if matches!(
                            decision,
                            crate::ArchivalDecision::Delete | crate::ArchivalDecision::Archive
                        ) {
                            let _ = self.forget(*id);
                            archived += 1;
                        }
                    }
                    debug!(turn_id, archived, "auto-maintenance: archival evaluated");
                }
                Err(e) => {
                    warn!(turn_id, error = %e, "auto-maintenance: archival failed");
                }
            }
        }

        // Every 200 turns: consolidation
        if turn_id.is_multiple_of(200) {
            match self.find_consolidation_candidates(2, 0.85) {
                Ok(candidates) => {
                    for candidate in &candidates {
                        let _ = self.consolidate_cluster(&candidate.memories);
                    }
                    debug!(
                        turn_id,
                        clusters = candidates.len(),
                        "auto-maintenance: consolidation"
                    );
                }
                Err(e) => {
                    warn!(turn_id, error = %e, "auto-maintenance: consolidation failed");
                }
            }
        }
    }
}

// ── Free functions ──

fn detect_actions(combined_text: &str) -> Vec<DetectedAction> {
    let lower = combined_text.to_lowercase();
    ACTION_KEYWORDS
        .iter()
        .filter(|(kw, _)| lower.contains(kw))
        .map(|(kw, action_type)| {
            let detail = extract_sentence_containing(&lower, kw);
            DetectedAction {
                action_type: action_type.to_string(),
                detail,
            }
        })
        .collect()
}

fn analyze_sentiment(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let total = words.len().max(1) as f32;

    let positive = words
        .iter()
        .filter(|w| POSITIVE_WORDS.iter().any(|p| w.contains(p)))
        .count() as f32;
    let negative = words
        .iter()
        .filter(|w| NEGATIVE_WORDS.iter().any(|n| w.contains(n)))
        .count() as f32;

    ((positive - negative) / total).clamp(-1.0, 1.0)
}

fn extract_sentence_containing(text: &str, keyword: &str) -> String {
    text.split('.')
        .find(|s| s.contains(keyword))
        .unwrap_or(keyword)
        .trim()
        .chars()
        .take(200)
        .collect()
}

fn now_us() -> Timestamp {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}
