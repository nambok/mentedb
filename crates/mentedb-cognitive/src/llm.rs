//! # CognitiveLlmService: LLM powered judgment for memory operations
//!
//! Provides intelligent decision making for temporal invalidation, contradiction
//! detection, entity resolution, memory consolidation, and topic canonicalization.
//! When no LLM is configured the caller falls back to heuristic paths.

use mentedb_core::memory::MemoryType;
use mentedb_core::types::MemoryId;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur during LLM judgment calls.
#[derive(Debug, thiserror::Error)]
pub enum LlmJudgeError {
    #[error("LLM provider returned an error: {0}")]
    ProviderError(String),
    #[error("failed to parse LLM response: {0}")]
    ParseError(String),
    #[error("LLM returned an unexpected verdict: {0}")]
    UnexpectedVerdict(String),
}

// ---------------------------------------------------------------------------
// LlmJudge trait
// ---------------------------------------------------------------------------

/// Minimal trait for sending a prompt to an LLM and receiving a text response.
///
/// Implementations exist for HTTP providers (OpenAI, Anthropic, Ollama, Custom)
/// via the adapter in `mentedb-extraction`, and for testing via [`MockLlmJudge`].
pub trait LlmJudge: Send + Sync {
    /// Send a system prompt and user prompt to the LLM, return the raw response text.
    fn complete(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> impl std::future::Future<Output = Result<String, LlmJudgeError>> + Send;
}

impl<T: LlmJudge> LlmJudge for std::sync::Arc<T> {
    fn complete(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> impl std::future::Future<Output = Result<String, LlmJudgeError>> + Send {
        (**self).complete(system_prompt, user_prompt)
    }
}

// ---------------------------------------------------------------------------
// Verdict types
// ---------------------------------------------------------------------------

/// Result of asking the LLM whether a new memory invalidates an existing one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum InvalidationVerdict {
    /// Both memories are valid and describe different facts.
    Keep { reason: String },
    /// The new memory makes the old one no longer true.
    Invalidate { reason: String },
    /// The old memory should be updated with new information.
    Update {
        merged_content: String,
        reason: String,
    },
}

/// Result of asking the LLM whether two memories contradict each other.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum ContradictionVerdict {
    /// The memories are compatible and can coexist.
    Compatible { reason: String },
    /// The memories directly contradict each other.
    Contradicts { reason: String },
    /// One memory supersedes the other (includes direction).
    Supersedes {
        #[serde(rename = "superseding_id")]
        winner: String,
        reason: String,
    },
}

/// A group of entity references that the LLM determined refer to the same thing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityMergeGroup {
    /// The canonical name for this entity.
    pub canonical: String,
    /// All alternative names that map to this entity.
    pub aliases: Vec<String>,
    /// How confident the LLM is in this grouping (0.0 to 1.0).
    pub confidence: f32,
}

/// An entity candidate passed to the resolution method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityCandidate {
    /// The raw entity reference string.
    pub name: String,
    /// Optional surrounding context to help disambiguation.
    pub context: Option<String>,
    /// The memory ID this entity was extracted from.
    pub memory_id: Option<MemoryId>,
}

/// What to do with a cluster of similar memories after consolidation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ConsolidationDecision {
    /// All memories in the cluster are distinct, keep them all.
    KeepAll { reason: String },
    /// Merge multiple memories into one with richer content.
    Merge {
        merged_content: String,
        merged_type: String,
        keep_ids: Vec<String>,
        remove_ids: Vec<String>,
        reason: String,
    },
    /// One memory is clearly the best, remove the rest.
    Deduplicate {
        keep_id: String,
        remove_ids: Vec<String>,
        reason: String,
    },
}

/// A canonical topic label returned by the LLM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopicLabel {
    /// The canonical 1-3 word topic label.
    pub topic: String,
    /// Whether this is a new topic or matched an existing one.
    pub is_new: bool,
}

/// LLM-generated community summary for a cluster of entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommunitySummary {
    pub summary: String,
}

/// LLM-generated user profile from accumulated knowledge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserProfile {
    pub profile: String,
}

// ---------------------------------------------------------------------------
// Compact memory representation for prompts
// ---------------------------------------------------------------------------

/// Lightweight memory summary used as input to judgment methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySummary {
    pub id: MemoryId,
    pub content: String,
    pub memory_type: MemoryType,
    pub confidence: f32,
    pub created_at: u64,
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

mod prompts {
    pub const INVALIDATION_SYSTEM: &str = r#"You are a memory validity judge for an AI agent's long term memory system.

Given an EXISTING memory and a NEW memory, determine the relationship:

- "keep": Both memories are valid. They describe different facts or the old one is still true.
- "invalidate": The new memory makes the old one no longer true. Example: old says "Alice works at Acme", new says "Alice joined Google" — the old employment fact is now outdated.
- "update": The old memory should be updated to incorporate new information. Example: old says "project uses React", new says "project migrated from React to Vue" — merge into one memory.

Respond with ONLY a JSON object. Examples:
{"verdict": "keep", "reason": "These describe different aspects of the same topic"}
{"verdict": "invalidate", "reason": "The user changed jobs, old employment is outdated"}
{"verdict": "update", "merged_content": "Project migrated from React to Vue in Q2", "reason": "The new memory adds temporal context to the old one"}"#;

    pub const CONTRADICTION_SYSTEM: &str = r#"You are a contradiction detector for an AI agent's long term memory system.

Given two memories A and B, determine their relationship:

- "compatible": They can both be true at the same time.
- "contradicts": They directly contradict each other and cannot both be true.
- "supersedes": One replaces the other due to a change over time (not a logical contradiction but a temporal update).

For "supersedes", include which memory wins using its ID.

Respond with ONLY a JSON object. Examples:
{"verdict": "compatible", "reason": "These describe different topics"}
{"verdict": "contradicts", "reason": "Cannot prefer both PostgreSQL and MySQL as primary database"}
{"verdict": "supersedes", "superseding_id": "<id of winner>", "reason": "Memory B reflects a more recent decision"}"#;

    pub const ENTITY_RESOLUTION_SYSTEM: &str = r#"You are an entity resolution system for an AI agent's long term memory.

Given a list of entity references (names, pronouns, descriptions), group the ones that refer to the same real world entity. Use the provided context to disambiguate.

Rules:
- Only group entities you are confident refer to the same thing
- "Python" (language) and "python" (the snake) are DIFFERENT entities if context makes that clear
- Pronouns like "my manager" can be grouped with a name if context confirms it
- Include a confidence score (0.0 to 1.0) for each group

Respond with ONLY a JSON object:
{"groups": [{"canonical": "Alice Smith", "aliases": ["Alice", "my manager"], "confidence": 0.9}]}"#;

    pub const CONSOLIDATION_SYSTEM: &str = r#"You are a memory consolidation system for an AI agent's long term memory.

Given a cluster of similar memories, decide how to combine them:

- "keep_all": The memories describe genuinely different facts despite being similar.
- "merge": Multiple memories should be combined into one richer memory. Provide the merged content.
- "deduplicate": One memory captures everything, the others are redundant. Keep the best one.

When merging, preserve all important information from each memory. The merged content should be strictly better than any individual memory.

Respond with ONLY a JSON object. Examples:
{"action": "keep_all", "reason": "Each memory describes a different API endpoint"}
{"action": "merge", "merged_content": "User prefers Rust for systems programming due to memory safety and zero cost abstractions", "merged_type": "Semantic", "keep_ids": [], "remove_ids": ["id1", "id2"], "reason": "Both memories express the same preference with complementary detail"}
{"action": "deduplicate", "keep_id": "id1", "remove_ids": ["id2", "id3"], "reason": "id1 is the most complete version"}"#;

    pub const TOPIC_SYSTEM: &str = r#"You are a topic labeler for a conversation tracking system.

Given a user message and a list of existing topic labels, either:
1. Assign the MOST appropriate existing label if the message fits
2. Create a new 1-3 word label if no existing label fits

Rules:
- Labels should be lowercase, concise (1-3 words max)
- Prefer existing labels when the topic clearly matches
- "deploy to staging" and "deploy to production" are the SAME topic: "deployment"
- "auth error" and "set up authentication" are the SAME topic: "authentication"
- Be general enough to group related messages but specific enough to be useful

Respond with ONLY a JSON object:
{"topic": "authentication", "is_new": false}"#;

    pub const COMMUNITY_SUMMARY_SYSTEM: &str = r#"You are a knowledge summarizer for an AI agent's long-term memory system.

Given a category name and a list of entities the user has interacted with, create a concise summary that captures the user's relationship to this cluster of entities.

Rules:
- Mention the key entities by name — prioritize the most significant ones
- Focus on the user's relationship to each (uses, likes, works with, etc.)
- Keep it to 2-3 sentences maximum
- The summary should be findable when someone searches for this topic
- Write in third person ("The user...")

Respond with ONLY a JSON object:
{"summary": "The user works with Python and Rust for systems programming, uses PostgreSQL for databases, and deploys on AWS."}"#;

    pub const USER_PROFILE_SYSTEM: &str = r#"You are a profile generator for an AI agent's long-term memory system.

Given a collection of facts, preferences, and patterns about a user, generate a concise profile that captures the most important information an AI assistant should know.

Rules:
- Prioritize: preferences > active projects > skills > relationships > habits
- Be factual — only include what the evidence supports
- Keep it under 200 words
- Write in third person ("The user...")
- Group related facts together
- Include specific names, tools, and preferences — not vague generalities
- This profile will be included in EVERY conversation, so only include persistently relevant facts

Respond with ONLY a JSON object:
{"profile": "The user is a software engineer who prefers Rust and Python. They are building MenteDB, a cognitive memory database..."}"#;
}

// ---------------------------------------------------------------------------
// CognitiveLlmService
// ---------------------------------------------------------------------------

/// The unified LLM intelligence layer for memory operations.
///
/// Wraps any [`LlmJudge`] implementation and provides typed judgment methods
/// with specialized prompts for each cognitive task.
pub struct CognitiveLlmService<J: LlmJudge> {
    judge: J,
}

impl<J: LlmJudge> CognitiveLlmService<J> {
    /// Create a new service backed by the given LLM judge.
    pub fn new(judge: J) -> Self {
        Self { judge }
    }

    /// Determine whether a new memory invalidates an existing one.
    ///
    /// This is the key method that replaces the cosine similarity heuristic
    /// for temporal invalidation. The LLM can understand that "Alice works at
    /// Acme" is invalidated by "Alice joined Google" even though the embeddings
    /// are quite different.
    pub async fn judge_invalidation(
        &self,
        old: &MemorySummary,
        new: &MemorySummary,
    ) -> Result<InvalidationVerdict, LlmJudgeError> {
        let user_prompt = format!(
            "EXISTING memory:\n{}\n\nNEW memory:\n{}",
            serde_json::to_string_pretty(old).unwrap_or_default(),
            serde_json::to_string_pretty(new).unwrap_or_default(),
        );

        let response = self
            .judge
            .complete(prompts::INVALIDATION_SYSTEM, &user_prompt)
            .await?;
        parse_json_response::<InvalidationVerdict>(&response)
    }

    /// Determine whether two memories contradict each other.
    pub async fn detect_contradiction(
        &self,
        a: &MemorySummary,
        b: &MemorySummary,
    ) -> Result<ContradictionVerdict, LlmJudgeError> {
        let user_prompt = format!(
            "Memory A:\n{}\n\nMemory B:\n{}",
            serde_json::to_string_pretty(a).unwrap_or_default(),
            serde_json::to_string_pretty(b).unwrap_or_default(),
        );

        let response = self
            .judge
            .complete(prompts::CONTRADICTION_SYSTEM, &user_prompt)
            .await?;
        parse_json_response::<ContradictionVerdict>(&response)
    }

    /// Resolve a batch of entity references into merge groups.
    pub async fn resolve_entities(
        &self,
        candidates: &[EntityCandidate],
    ) -> Result<Vec<EntityMergeGroup>, LlmJudgeError> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let user_prompt = format!(
            "Entity references to resolve:\n{}",
            serde_json::to_string_pretty(candidates).unwrap_or_default(),
        );

        let response = self
            .judge
            .complete(prompts::ENTITY_RESOLUTION_SYSTEM, &user_prompt)
            .await?;

        #[derive(Deserialize)]
        struct EntityResponse {
            groups: Vec<EntityMergeGroup>,
        }

        let parsed = parse_json_response::<EntityResponse>(&response)?;
        Ok(parsed.groups)
    }

    /// Decide how to consolidate a cluster of similar memories.
    pub async fn consolidate(
        &self,
        cluster: &[ClusterMember],
    ) -> Result<ConsolidationDecision, LlmJudgeError> {
        if cluster.is_empty() {
            return Ok(ConsolidationDecision::KeepAll {
                reason: "Empty cluster".to_string(),
            });
        }

        let user_prompt = format!(
            "Memory cluster to consolidate:\n{}",
            serde_json::to_string_pretty(cluster).unwrap_or_default(),
        );

        let response = self
            .judge
            .complete(prompts::CONSOLIDATION_SYSTEM, &user_prompt)
            .await?;
        parse_json_response::<ConsolidationDecision>(&response)
    }

    /// Map a raw user message to a canonical topic label.
    pub async fn canonicalize_topic(
        &self,
        message: &str,
        existing_topics: &[String],
    ) -> Result<TopicLabel, LlmJudgeError> {
        let user_prompt = format!(
            "User message: \"{}\"\n\nExisting topic labels: {:?}",
            message, existing_topics,
        );

        let response = self
            .judge
            .complete(prompts::TOPIC_SYSTEM, &user_prompt)
            .await?;
        parse_json_response::<TopicLabel>(&response)
    }

    /// Generate a summary for a community (cluster of related entities).
    pub async fn generate_community_summary(
        &self,
        category: &str,
        entities: &[(String, String)], // (name, context/relationship)
    ) -> Result<CommunitySummary, LlmJudgeError> {
        let entity_list = entities
            .iter()
            .map(|(name, ctx)| format!("- {} — {}", name, ctx))
            .collect::<Vec<_>>()
            .join("\n");

        let user_prompt = format!(
            "Category: {}\n\nEntities in this cluster:\n{}",
            category, entity_list
        );

        let response = self
            .judge
            .complete(prompts::COMMUNITY_SUMMARY_SYSTEM, &user_prompt)
            .await?;
        parse_json_response::<CommunitySummary>(&response)
    }

    /// Generate a user profile from accumulated knowledge.
    pub async fn generate_user_profile(
        &self,
        facts: &[String],
        community_summaries: &[String],
    ) -> Result<UserProfile, LlmJudgeError> {
        let mut sections = Vec::new();

        if !facts.is_empty() {
            sections.push(format!(
                "Known facts about the user:\n{}",
                facts
                    .iter()
                    .map(|f| format!("- {}", f))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }

        if !community_summaries.is_empty() {
            sections.push(format!(
                "Community summaries:\n{}",
                community_summaries
                    .iter()
                    .map(|s| format!("- {}", s))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }

        let user_prompt = sections.join("\n\n");

        if user_prompt.is_empty() {
            return Err(LlmJudgeError::ParseError(
                "no facts or summaries provided for profile generation".into(),
            ));
        }

        let response = self
            .judge
            .complete(prompts::USER_PROFILE_SYSTEM, &user_prompt)
            .await?;
        parse_json_response::<UserProfile>(&response)
    }
}

/// A memory in a consolidation cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMember {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub confidence: f32,
    pub created_at: u64,
}

// ---------------------------------------------------------------------------
// JSON parsing helper
// ---------------------------------------------------------------------------

fn parse_json_response<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, LlmJudgeError> {
    // Try direct parse first
    if let Ok(v) = serde_json::from_str::<T>(raw) {
        return Ok(v);
    }

    let trimmed = raw.trim();

    // LLMs sometimes wrap JSON in markdown code blocks
    let stripped = if trimmed.starts_with("```json") {
        trimmed
            .strip_prefix("```json")
            .and_then(|s| s.strip_suffix("```"))
            .unwrap_or(trimmed)
            .trim()
    } else if trimmed.starts_with("```") {
        trimmed
            .strip_prefix("```")
            .and_then(|s| s.strip_suffix("```"))
            .unwrap_or(trimmed)
            .trim()
    } else {
        trimmed
    };

    if let Ok(v) = serde_json::from_str::<T>(stripped) {
        return Ok(v);
    }

    // LLMs sometimes add explanatory text around the JSON.
    // Try to extract the first JSON object from the response.
    if let Some(start) = stripped.find('{')
        && let Some(end) = rfind_matching_brace(stripped, start)
    {
        let candidate = &stripped[start..=end];
        if let Ok(v) = serde_json::from_str::<T>(candidate) {
            return Ok(v);
        }
    }

    Err(LlmJudgeError::ParseError(format!(
        "could not parse LLM response as expected type. Raw response: {raw}"
    )))
}

/// Find the closing brace that matches the opening brace at `start`.
fn rfind_matching_brace(s: &str, start: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in s[start..].char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(start + i);
                }
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// MockLlmJudge for testing
// ---------------------------------------------------------------------------

/// A mock judge that returns a predetermined response. For testing.
pub struct MockLlmJudge {
    response: String,
}

impl MockLlmJudge {
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
        }
    }
}

impl LlmJudge for MockLlmJudge {
    async fn complete(
        &self,
        _system_prompt: &str,
        _user_prompt: &str,
    ) -> Result<String, LlmJudgeError> {
        Ok(self.response.clone())
    }
}

/// A mock judge that captures prompts for inspection.
#[cfg(test)]
pub struct CapturingJudge {
    response: String,
    captured: std::sync::Mutex<Vec<(String, String)>>,
}

#[cfg(test)]
impl CapturingJudge {
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
            captured: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<(String, String)> {
        self.captured.lock().unwrap().clone()
    }
}

#[cfg(test)]
impl LlmJudge for CapturingJudge {
    async fn complete(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, LlmJudgeError> {
        self.captured
            .lock()
            .unwrap()
            .push((system_prompt.to_string(), user_prompt.to_string()));
        Ok(self.response.clone())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::types::MemoryId;

    fn test_id() -> MemoryId {
        MemoryId::new()
    }

    fn mem(content: &str) -> MemorySummary {
        MemorySummary {
            id: test_id(),
            content: content.to_string(),
            memory_type: MemoryType::Semantic,
            confidence: 0.9,
            created_at: 1000,
        }
    }

    #[allow(dead_code)]
    fn mem_at(content: &str, created_at: u64) -> MemorySummary {
        MemorySummary {
            id: test_id(),
            content: content.to_string(),
            memory_type: MemoryType::Semantic,
            confidence: 0.9,
            created_at,
        }
    }

    #[tokio::test]
    async fn test_judge_invalidation_keep() {
        let judge = MockLlmJudge::new(
            r#"{"verdict": "keep", "reason": "These describe different people"}"#,
        );
        let svc = CognitiveLlmService::new(judge);

        let result = svc
            .judge_invalidation(&mem("Alice works at Acme"), &mem("Bob works at Google"))
            .await
            .unwrap();

        assert!(matches!(result, InvalidationVerdict::Keep { .. }));
    }

    #[tokio::test]
    async fn test_judge_invalidation_invalidate() {
        let judge =
            MockLlmJudge::new(r#"{"verdict": "invalidate", "reason": "Alice changed jobs"}"#);
        let svc = CognitiveLlmService::new(judge);

        let result = svc
            .judge_invalidation(&mem("Alice works at Acme"), &mem("Alice joined Google"))
            .await
            .unwrap();

        assert!(matches!(result, InvalidationVerdict::Invalidate { .. }));
    }

    #[tokio::test]
    async fn test_judge_invalidation_update() {
        let judge = MockLlmJudge::new(
            r#"{"verdict": "update", "merged_content": "Project migrated from React to Vue in Q2", "reason": "Temporal update"}"#,
        );
        let svc = CognitiveLlmService::new(judge);

        let result = svc
            .judge_invalidation(&mem("Project uses React"), &mem("Project migrated to Vue"))
            .await
            .unwrap();

        match result {
            InvalidationVerdict::Update { merged_content, .. } => {
                assert!(merged_content.contains("React"));
                assert!(merged_content.contains("Vue"));
            }
            other => panic!("Expected Update, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_detect_contradiction_compatible() {
        let judge = MockLlmJudge::new(
            r#"{"verdict": "compatible", "reason": "Different topics entirely"}"#,
        );
        let svc = CognitiveLlmService::new(judge);

        let result = svc
            .detect_contradiction(&mem("Likes Python"), &mem("Uses PostgreSQL"))
            .await
            .unwrap();

        assert!(matches!(result, ContradictionVerdict::Compatible { .. }));
    }

    #[tokio::test]
    async fn test_detect_contradiction_contradicts() {
        let judge =
            MockLlmJudge::new(r#"{"verdict": "contradicts", "reason": "Cannot prefer both"}"#);
        let svc = CognitiveLlmService::new(judge);

        let result = svc
            .detect_contradiction(&mem("Prefers tabs"), &mem("Prefers spaces"))
            .await
            .unwrap();

        assert!(matches!(result, ContradictionVerdict::Contradicts { .. }));
    }

    #[tokio::test]
    async fn test_detect_contradiction_supersedes() {
        let id_b = test_id();
        let response = format!(
            r#"{{"verdict": "supersedes", "superseding_id": "{}", "reason": "B is newer"}}"#,
            id_b
        );
        let judge = MockLlmJudge::new(response);
        let svc = CognitiveLlmService::new(judge);

        let mem_b = MemorySummary {
            id: id_b,
            content: "Migrated to Vue".into(),
            memory_type: MemoryType::Semantic,
            confidence: 0.9,
            created_at: 2000,
        };

        let result = svc
            .detect_contradiction(&mem("Uses React"), &mem_b)
            .await
            .unwrap();

        assert!(matches!(result, ContradictionVerdict::Supersedes { .. }));
    }

    #[tokio::test]
    async fn test_resolve_entities() {
        let judge = MockLlmJudge::new(
            r#"{"groups": [{"canonical": "Alice Smith", "aliases": ["Alice", "my manager"], "confidence": 0.92}]}"#,
        );
        let svc = CognitiveLlmService::new(judge);

        let candidates = vec![
            EntityCandidate {
                name: "Alice".into(),
                context: None,
                memory_id: None,
            },
            EntityCandidate {
                name: "my manager".into(),
                context: Some("my manager Alice told me".into()),
                memory_id: None,
            },
            EntityCandidate {
                name: "Alice Smith".into(),
                context: None,
                memory_id: None,
            },
        ];

        let groups = svc.resolve_entities(&candidates).await.unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].canonical, "Alice Smith");
        assert_eq!(groups[0].aliases.len(), 2);
    }

    #[tokio::test]
    async fn test_resolve_entities_empty() {
        let judge = MockLlmJudge::new("should not be called");
        let svc = CognitiveLlmService::new(judge);

        let groups = svc.resolve_entities(&[]).await.unwrap();
        assert!(groups.is_empty());
    }

    #[tokio::test]
    async fn test_consolidate_merge() {
        let judge = MockLlmJudge::new(
            r#"{"action": "merge", "merged_content": "User prefers Rust for systems programming due to safety and performance", "merged_type": "Semantic", "keep_ids": [], "remove_ids": ["a", "b"], "reason": "Complementary details"}"#,
        );
        let svc = CognitiveLlmService::new(judge);

        let cluster = vec![
            ClusterMember {
                id: "a".into(),
                content: "Uses Rust".into(),
                memory_type: "Semantic".into(),
                confidence: 0.9,
                created_at: 1000,
            },
            ClusterMember {
                id: "b".into(),
                content: "Prefers Rust for safety".into(),
                memory_type: "Semantic".into(),
                confidence: 0.85,
                created_at: 2000,
            },
        ];

        let decision = svc.consolidate(&cluster).await.unwrap();
        match decision {
            ConsolidationDecision::Merge {
                merged_content,
                remove_ids,
                ..
            } => {
                assert!(merged_content.contains("Rust"));
                assert_eq!(remove_ids.len(), 2);
            }
            other => panic!("Expected Merge, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_consolidate_empty_cluster() {
        let judge = MockLlmJudge::new("should not be called");
        let svc = CognitiveLlmService::new(judge);

        let decision = svc.consolidate(&[]).await.unwrap();
        assert!(matches!(decision, ConsolidationDecision::KeepAll { .. }));
    }

    #[tokio::test]
    async fn test_canonicalize_topic_existing() {
        let judge = MockLlmJudge::new(r#"{"topic": "authentication", "is_new": false}"#);
        let svc = CognitiveLlmService::new(judge);

        let label = svc
            .canonicalize_topic(
                "how do I configure the auth middleware",
                &[
                    "database".into(),
                    "authentication".into(),
                    "deployment".into(),
                ],
            )
            .await
            .unwrap();

        assert_eq!(label.topic, "authentication");
        assert!(!label.is_new);
    }

    #[tokio::test]
    async fn test_canonicalize_topic_new() {
        let judge = MockLlmJudge::new(r#"{"topic": "caching", "is_new": true}"#);
        let svc = CognitiveLlmService::new(judge);

        let label = svc
            .canonicalize_topic(
                "should we add Redis for caching",
                &["database".into(), "authentication".into()],
            )
            .await
            .unwrap();

        assert_eq!(label.topic, "caching");
        assert!(label.is_new);
    }

    #[tokio::test]
    async fn test_json_markdown_code_block_stripping() {
        let judge = MockLlmJudge::new(
            "```json\n{\"verdict\": \"keep\", \"reason\": \"wrapped in markdown\"}\n```",
        );
        let svc = CognitiveLlmService::new(judge);

        let result = svc.judge_invalidation(&mem("A"), &mem("B")).await.unwrap();

        assert!(matches!(result, InvalidationVerdict::Keep { .. }));
    }

    #[tokio::test]
    async fn test_json_surrounded_by_text() {
        let judge = MockLlmJudge::new(
            "Here is my analysis:\n{\"verdict\": \"invalidate\", \"reason\": \"job changed\"}\nLet me know if you need more detail.",
        );
        let svc = CognitiveLlmService::new(judge);

        let result = svc.judge_invalidation(&mem("A"), &mem("B")).await.unwrap();

        assert!(matches!(result, InvalidationVerdict::Invalidate { .. }));
    }

    #[tokio::test]
    async fn test_json_with_nested_braces_in_strings() {
        let judge = MockLlmJudge::new(
            "Sure! {\"verdict\": \"update\", \"merged_content\": \"uses {curly} braces\", \"reason\": \"test\"}",
        );
        let svc = CognitiveLlmService::new(judge);

        let result = svc.judge_invalidation(&mem("A"), &mem("B")).await.unwrap();

        match result {
            InvalidationVerdict::Update { merged_content, .. } => {
                assert!(merged_content.contains("{curly}"))
            }
            other => panic!("Expected Update, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_provider_error_propagates() {
        struct FailingJudge;
        impl LlmJudge for FailingJudge {
            async fn complete(&self, _: &str, _: &str) -> Result<String, LlmJudgeError> {
                Err(LlmJudgeError::ProviderError("connection refused".into()))
            }
        }

        let svc = CognitiveLlmService::new(FailingJudge);
        let result = svc.judge_invalidation(&mem("A"), &mem("B")).await;

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("connection refused")
        );
    }

    #[tokio::test]
    async fn test_prompt_includes_memory_content() {
        let judge = std::sync::Arc::new(CapturingJudge::new(
            r#"{"verdict": "keep", "reason": "test"}"#,
        ));
        let svc = CognitiveLlmService::new(judge.clone());

        let _ = svc
            .judge_invalidation(&mem("Alice works at Acme"), &mem("Alice joined Google"))
            .await;

        let calls = judge.calls();
        assert_eq!(calls.len(), 1);
        let (system, user) = &calls[0];
        assert!(system.contains("memory validity judge"));
        assert!(user.contains("Alice works at Acme"));
        assert!(user.contains("Alice joined Google"));
    }
}
