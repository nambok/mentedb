use mentedb_cognitive::write_inference::{InferredAction, WriteInferenceEngine};
use mentedb_core::MemoryNode;
use mentedb_core::types::{AgentId, MemoryId};
use mentedb_embedding::provider::EmbeddingProvider;

use crate::config::ExtractionConfig;
use crate::error::ExtractionError;
use crate::prompts::extraction_system_prompt;
use crate::provider::ExtractionProvider;
use crate::schema::{ExtractedMemory, ExtractionResult};

/// Findings from cognitive checks (contradiction detection).
#[derive(Debug, Clone)]
pub struct CognitiveFinding {
    /// What type of issue was found.
    pub finding_type: CognitiveFindingType,
    /// Human-readable description of the finding.
    pub description: String,
    /// ID of the existing memory involved, if any.
    pub related_memory_id: Option<MemoryId>,
}

/// Types of cognitive findings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveFindingType {
    Contradiction,
    Obsolescence,
    Related,
    ConfidenceUpdate,
}

/// Statistics from a full extraction pipeline run.
#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    pub total_extracted: usize,
    pub accepted: usize,
    pub rejected_quality: usize,
    pub rejected_duplicate: usize,
    pub contradictions_found: usize,
}

/// The complete result of running the extraction pipeline.
#[derive(Debug)]
pub struct ProcessedExtractionResult {
    /// Memories that passed all checks and should be stored.
    pub to_store: Vec<ExtractedMemory>,
    /// Memories rejected for low confidence scores.
    pub rejected_low_quality: Vec<ExtractedMemory>,
    /// Memories rejected as duplicates of existing memories.
    pub rejected_duplicate: Vec<ExtractedMemory>,
    /// Memories that contradict existing ones (stored anyway, with findings).
    pub contradictions: Vec<(ExtractedMemory, Vec<CognitiveFinding>)>,
    /// Summary statistics.
    pub stats: ExtractionStats,
}

/// The main extraction engine. Takes raw conversations, extracts structured
/// memories via an LLM, then filters and validates them before storage.
pub struct ExtractionPipeline<P: ExtractionProvider> {
    provider: P,
    config: ExtractionConfig,
}

impl<P: ExtractionProvider> ExtractionPipeline<P> {
    pub fn new(provider: P, config: ExtractionConfig) -> Self {
        Self { provider, config }
    }

    /// Call the LLM to extract memories from a conversation, then parse the
    /// response and filter by quality threshold.
    pub async fn extract_from_conversation(
        &self,
        conversation: &str,
    ) -> Result<Vec<ExtractedMemory>, ExtractionError> {
        let system_prompt = extraction_system_prompt();
        let raw_response = self.provider.extract(conversation, system_prompt).await?;

        let result = self.parse_extraction_response(&raw_response)?;

        let mut memories = result.memories;
        if memories.len() > self.config.max_extractions_per_conversation {
            tracing::warn!(
                extracted = memories.len(),
                max = self.config.max_extractions_per_conversation,
                "truncating extractions to configured maximum"
            );
            memories.truncate(self.config.max_extractions_per_conversation);
        }

        Ok(memories)
    }

    /// Parse the raw JSON response from the LLM into an ExtractionResult.
    /// Handles edge cases like markdown fences around JSON and preamble text.
    fn parse_extraction_response(&self, raw: &str) -> Result<ExtractionResult, ExtractionError> {
        let trimmed = raw.trim();

        // Empty response = no memories to extract
        if trimmed.is_empty() {
            return Ok(ExtractionResult { memories: vec![] });
        }

        // Strip markdown code fences if present
        let stripped = if trimmed.starts_with("```") {
            let without_prefix = trimmed
                .trim_start_matches("```json")
                .trim_start_matches("```");
            without_prefix.trim_end_matches("```").trim()
        } else {
            trimmed
        };

        // Find the outermost JSON object using brace-depth matching
        // that respects quoted strings (handles braces inside string values)
        let json_str = if let Some(start) = stripped.find('{') {
            let candidate = &stripped[start..];
            let mut depth = 0i32;
            let mut in_string = false;
            let mut escape_next = false;
            let mut end = candidate.len();
            for (i, ch) in candidate.char_indices() {
                if escape_next {
                    escape_next = false;
                    continue;
                }
                if in_string {
                    match ch {
                        '\\' => escape_next = true,
                        '"' => in_string = false,
                        _ => {}
                    }
                    continue;
                }
                match ch {
                    '"' => in_string = true,
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            &candidate[..end]
        } else {
            // No JSON object found — LLM returned plain text (e.g. "No memories to extract")
            return Ok(ExtractionResult { memories: vec![] });
        };

        // Parse with serde_json::Value first (tolerates duplicate keys — last one wins)
        // then convert to ExtractionResult. LLMs sometimes emit duplicate fields.
        let value: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
            tracing::error!(
                error = %e,
                response_preview = &json_str[..json_str.len().min(200)],
                "failed to parse LLM extraction response as JSON"
            );
            ExtractionError::ParseError(format!("Failed to parse extraction JSON: {e}"))
        })?;

        serde_json::from_value::<ExtractionResult>(value).map_err(|e| {
            tracing::error!(
                error = %e,
                "failed to deserialize extraction JSON into ExtractionResult"
            );
            ExtractionError::ParseError(format!("Failed to parse extraction JSON: {e}"))
        })
    }

    /// Remove memories below the configured confidence threshold.
    pub fn filter_quality(&self, memories: &[ExtractedMemory]) -> Vec<ExtractedMemory> {
        memories
            .iter()
            .filter(|m| m.confidence >= self.config.quality_threshold)
            .cloned()
            .collect()
    }

    /// Check a new extracted memory against existing memories for contradictions
    /// using the WriteInferenceEngine.
    pub fn check_contradictions(
        &self,
        new_memory: &ExtractedMemory,
        existing: &[MemoryNode],
        embedding_provider: &dyn EmbeddingProvider,
    ) -> Vec<CognitiveFinding> {
        if !self.config.enable_contradiction_check || existing.is_empty() {
            return Vec::new();
        }

        let embedding = match embedding_provider.embed(&new_memory.content) {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(error = %err, "failed to embed memory for contradiction check");
                return Vec::new();
            }
        };

        let memory_type = map_extraction_type_to_memory_type(&new_memory.memory_type);
        let temp_node = MemoryNode::new(
            AgentId::nil(),
            memory_type,
            new_memory.content.clone(),
            embedding,
        );

        let engine = WriteInferenceEngine::new();
        let actions = engine.infer_on_write(&temp_node, existing, &[]);

        let mut findings = Vec::new();
        for action in actions {
            match action {
                InferredAction::FlagContradiction {
                    existing: existing_id,
                    reason,
                    ..
                } => {
                    findings.push(CognitiveFinding {
                        finding_type: CognitiveFindingType::Contradiction,
                        description: reason,
                        related_memory_id: Some(existing_id),
                    });
                }
                InferredAction::MarkObsolete {
                    memory,
                    superseded_by: _,
                } => {
                    findings.push(CognitiveFinding {
                        finding_type: CognitiveFindingType::Obsolescence,
                        description: format!("Memory {memory} may be obsolete"),
                        related_memory_id: Some(memory),
                    });
                }
                InferredAction::UpdateConfidence {
                    memory,
                    new_confidence,
                } => {
                    findings.push(CognitiveFinding {
                        finding_type: CognitiveFindingType::ConfidenceUpdate,
                        description: format!(
                            "Confidence for {memory} should be updated to {new_confidence:.2}"
                        ),
                        related_memory_id: Some(memory),
                    });
                }
                InferredAction::CreateEdge { target, .. } => {
                    findings.push(CognitiveFinding {
                        finding_type: CognitiveFindingType::Related,
                        description: format!("Related to existing memory {target}"),
                        related_memory_id: Some(target),
                    });
                }
                _ => {}
            }
        }

        findings
    }

    /// Check if a new memory is too similar to any existing memory
    /// (above deduplication_threshold).
    pub fn check_duplicates(
        &self,
        new_memory: &ExtractedMemory,
        existing: &[MemoryNode],
        embedding_provider: &dyn EmbeddingProvider,
    ) -> bool {
        if !self.config.enable_deduplication || existing.is_empty() {
            return false;
        }

        let new_embedding = match embedding_provider.embed(&new_memory.content) {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(error = %err, "failed to embed memory for dedup check");
                return false;
            }
        };

        for mem in existing {
            let sim = cosine_similarity(&new_embedding, &mem.embedding);
            if sim >= self.config.deduplication_threshold {
                tracing::debug!(
                    similarity = sim,
                    threshold = self.config.deduplication_threshold,
                    existing_id = %mem.id,
                    "duplicate detected"
                );
                return true;
            }
        }

        false
    }

    /// Run the full extraction pipeline: extract -> filter quality ->
    /// check duplicates -> check contradictions.
    pub async fn process(
        &self,
        conversation: &str,
        existing_memories: &[MemoryNode],
        embedding_provider: &dyn EmbeddingProvider,
    ) -> Result<ProcessedExtractionResult, ExtractionError> {
        let all_memories = self.extract_from_conversation(conversation).await?;
        let total_extracted = all_memories.len();

        let quality_passed = self.filter_quality(&all_memories);
        let rejected_low_quality: Vec<ExtractedMemory> = all_memories
            .iter()
            .filter(|m| m.confidence < self.config.quality_threshold)
            .cloned()
            .collect();

        let mut to_store = Vec::new();
        let mut rejected_duplicate = Vec::new();
        let mut contradictions = Vec::new();

        for memory in quality_passed {
            if self.check_duplicates(&memory, existing_memories, embedding_provider) {
                rejected_duplicate.push(memory);
                continue;
            }

            let findings =
                self.check_contradictions(&memory, existing_memories, embedding_provider);
            let has_contradiction = findings
                .iter()
                .any(|f| f.finding_type == CognitiveFindingType::Contradiction);

            if has_contradiction {
                contradictions.push((memory, findings));
            } else {
                to_store.push(memory);
            }
        }

        let stats = ExtractionStats {
            total_extracted,
            accepted: to_store.len(),
            rejected_quality: rejected_low_quality.len(),
            rejected_duplicate: rejected_duplicate.len(),
            contradictions_found: contradictions.len(),
        };

        tracing::info!(
            total = stats.total_extracted,
            accepted = stats.accepted,
            rejected_quality = stats.rejected_quality,
            rejected_duplicate = stats.rejected_duplicate,
            contradictions = stats.contradictions_found,
            "extraction pipeline complete"
        );

        Ok(ProcessedExtractionResult {
            to_store,
            rejected_low_quality,
            rejected_duplicate,
            contradictions,
            stats,
        })
    }
}

/// Map extraction type strings to MemoryType enum variants.
pub fn map_extraction_type_to_memory_type(
    extraction_type: &str,
) -> mentedb_core::memory::MemoryType {
    use mentedb_core::memory::MemoryType;
    match extraction_type.to_lowercase().as_str() {
        "decision" | "preference" | "fact" | "entity" => MemoryType::Semantic,
        "correction" => MemoryType::Correction,
        "anti_pattern" => MemoryType::AntiPattern,
        _ => MemoryType::Episodic,
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}
