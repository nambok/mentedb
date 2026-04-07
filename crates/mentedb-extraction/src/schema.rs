use serde::{Deserialize, Serialize};

/// The complete result of an extraction call, as returned by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub memories: Vec<ExtractedMemory>,
}

/// A single memory extracted from a conversation by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMemory {
    /// The factual content of the memory.
    pub content: String,
    /// Classification: decision, preference, correction, fact, entity, anti_pattern.
    pub memory_type: String,
    /// How confident the LLM is that this is worth remembering (0.0 to 1.0).
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    /// Entities mentioned in this memory.
    #[serde(default)]
    pub entities: Vec<String>,
    /// Categorization tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Why this memory was extracted (for debugging and auditing).
    #[serde(default)]
    pub reasoning: String,
}

impl ExtractedMemory {
    /// Build an augmented text string for embedding generation.
    ///
    /// Concatenates the content with entities and tags to create a richer
    /// vector representation that matches on more search queries.
    pub fn embedding_key(&self) -> String {
        let mut key = self.content.clone();
        if !self.entities.is_empty() {
            key.push_str(" [entities: ");
            key.push_str(&self.entities.join(", "));
            key.push(']');
        }
        if !self.tags.is_empty() {
            key.push_str(" [topics: ");
            key.push_str(&self.tags.join(", "));
            key.push(']');
        }
        key
    }
}

fn default_confidence() -> f32 {
    0.5
}
