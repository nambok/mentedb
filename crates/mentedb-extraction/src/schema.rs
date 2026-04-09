use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

/// Deserialize a HashMap where values may be strings, numbers, or booleans,
/// coercing everything to String. LLMs sometimes return `"age": 10` instead of `"age": "10"`.
fn deserialize_string_map<'de, D>(deserializer: D) -> Result<HashMap<String, String>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw: HashMap<String, serde_json::Value> = HashMap::deserialize(deserializer)?;
    Ok(raw
        .into_iter()
        .map(|(k, v)| {
            let s = match v {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            };
            (k, s)
        })
        .collect())
}

/// The complete result of an extraction call, as returned by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub memories: Vec<ExtractedMemory>,
    /// Entities extracted from the conversation, with typed attributes.
    /// Each entity represents a person, pet, place, event, item, etc.
    #[serde(default)]
    pub entities: Vec<ExtractedEntity>,
}

/// An entity extracted from a conversation — a person, pet, place, event, or item
/// with structured attributes. Entities accumulate knowledge across mentions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Canonical name of the entity (e.g., "Max", "Serenity Yoga", "Love is in the Air Dinner")
    pub name: String,
    /// Type classification: person, pet, place, event, item, organization, account
    pub entity_type: String,
    /// Key-value attributes discovered about this entity.
    /// Keys are attribute names (e.g., "breed", "location", "date"),
    /// values are the attribute values (e.g., "Golden Retriever", "downtown", "February 14th").
    #[serde(default, deserialize_with = "deserialize_string_map")]
    pub attributes: HashMap<String, String>,
}

impl ExtractedEntity {
    /// Build a searchable text representation of this entity and its attributes.
    /// Includes categories for discoverability by abstract searches.
    pub fn embedding_key(&self) -> String {
        let mut key = format!("{} ({})", self.name, self.entity_type);
        // Prioritize category in embedding for abstract query matching
        if let Some(cat) = self.attributes.get("category") {
            if !cat.is_empty() {
                key.push_str(&format!(" [categories: {}]", cat));
            }
        }
        for (attr_name, attr_value) in &self.attributes {
            if attr_name == "category" { continue; } // already included above
            key.push_str(&format!(" {}: {}", attr_name, attr_value));
        }
        key
    }

    /// Build a rich content string for storage as a memory node.
    /// Includes category for searchability by abstract queries.
    pub fn to_content(&self) -> String {
        let mut content = String::new();
        // Prepend categories for semantic enrichment (makes entities findable by category search)
        if let Some(cat) = self.attributes.get("category") {
            if !cat.is_empty() {
                let cats: Vec<&str> = cat.split(',').map(|c| c.trim()).filter(|c| !c.is_empty()).collect();
                if !cats.is_empty() {
                    let readable: Vec<String> = cats.iter().map(|c| c.replace('_', " ")).collect();
                    content.push_str(&format!("{} — ", readable.join(", ")));
                }
            }
        }
        content.push_str(&format!("{} is a {}", self.name, self.entity_type));
        if !self.attributes.is_empty() {
            let attrs: Vec<String> = self
                .attributes
                .iter()
                .filter(|(k, _)| k.as_str() != "category") // already in prefix
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect();
            if !attrs.is_empty() {
                content.push_str(&format!(". Attributes: {}", attrs.join(", ")));
            }
        }
        content
    }
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
