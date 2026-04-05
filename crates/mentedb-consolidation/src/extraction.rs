use mentedb_core::MemoryNode;
use mentedb_core::types::MemoryId;
use serde::{Deserialize, Serialize};

/// A structured fact extracted from memory content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtractedFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
    pub source_memory: MemoryId,
}

/// Patterns for fact extraction: (verb/phrase, predicate label).
const PATTERNS: &[(&str, &str)] = &[
    (" is ", "is"),
    (" uses ", "uses"),
    (" prefers ", "prefers"),
    (" switched to ", "switched_to"),
    (" decided on ", "decided_on"),
];

/// Extracts structured facts from memory content using rule-based patterns.
pub struct FactExtractor;

impl FactExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract facts from a memory's content using simple pattern matching.
    pub fn extract_facts(&self, memory: &MemoryNode) -> Vec<ExtractedFact> {
        let mut facts = Vec::new();

        for sentence in memory.content.split('.') {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            for &(pattern, predicate) in PATTERNS {
                if let Some(pos) = sentence.to_lowercase().find(pattern) {
                    let subject = sentence[..pos].trim();
                    let object = sentence[pos + pattern.len()..].trim();

                    if !subject.is_empty() && !object.is_empty() {
                        facts.push(ExtractedFact {
                            subject: subject.to_string(),
                            predicate: predicate.to_string(),
                            object: object.to_string(),
                            confidence: memory.confidence,
                            source_memory: memory.id,
                        });
                    }
                }
            }
        }

        facts
    }

    /// Remove duplicate facts by (subject, predicate, object) tuple.
    pub fn deduplicate_facts(&self, facts: &[ExtractedFact]) -> Vec<ExtractedFact> {
        let mut seen: ahash::HashSet<(String, String, String)> = ahash::HashSet::default();
        let mut result = Vec::new();

        for fact in facts {
            let key = (
                fact.subject.to_lowercase(),
                fact.predicate.to_lowercase(),
                fact.object.to_lowercase(),
            );
            if seen.insert(key) {
                result.push(fact.clone());
            }
        }

        result
    }

    /// Find facts that contradict the new fact (same subject+predicate, different object).
    pub fn find_contradicting_facts<'a>(
        &self,
        new_fact: &ExtractedFact,
        existing: &'a [ExtractedFact],
    ) -> Vec<&'a ExtractedFact> {
        existing
            .iter()
            .filter(|f| {
                f.subject.to_lowercase() == new_fact.subject.to_lowercase()
                    && f.predicate.to_lowercase() == new_fact.predicate.to_lowercase()
                    && f.object.to_lowercase() != new_fact.object.to_lowercase()
            })
            .collect()
    }
}

impl Default for FactExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_memory;

    #[test]
    fn test_extract_is_pattern() {
        let extractor = FactExtractor::new();
        let m = make_memory("Rust is a systems language", vec![1.0]);
        let facts = extractor.extract_facts(&m);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].subject, "Rust");
        assert_eq!(facts[0].predicate, "is");
        assert_eq!(facts[0].object, "a systems language");
    }

    #[test]
    fn test_extract_uses_pattern() {
        let extractor = FactExtractor::new();
        let m = make_memory("The team uses Postgres for storage", vec![1.0]);
        let facts = extractor.extract_facts(&m);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].predicate, "uses");
    }

    #[test]
    fn test_extract_multiple_sentences() {
        let extractor = FactExtractor::new();
        let m = make_memory("Alice prefers Python. Bob uses Java", vec![1.0]);
        let facts = extractor.extract_facts(&m);
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn test_deduplicate_facts() {
        let extractor = FactExtractor::new();
        let id = MemoryId::new();
        let facts = vec![
            ExtractedFact {
                subject: "A".into(),
                predicate: "is".into(),
                object: "B".into(),
                confidence: 0.9,
                source_memory: id,
            },
            ExtractedFact {
                subject: "a".into(),
                predicate: "is".into(),
                object: "b".into(),
                confidence: 0.8,
                source_memory: id,
            },
        ];
        let deduped = extractor.deduplicate_facts(&facts);
        assert_eq!(deduped.len(), 1);
    }

    #[test]
    fn test_find_contradictions() {
        let extractor = FactExtractor::new();
        let id = MemoryId::new();
        let new_fact = ExtractedFact {
            subject: "Team".into(),
            predicate: "uses".into(),
            object: "Postgres".into(),
            confidence: 0.9,
            source_memory: id,
        };
        let existing = vec![
            ExtractedFact {
                subject: "Team".into(),
                predicate: "uses".into(),
                object: "MySQL".into(),
                confidence: 0.8,
                source_memory: id,
            },
            ExtractedFact {
                subject: "Team".into(),
                predicate: "prefers".into(),
                object: "Go".into(),
                confidence: 0.7,
                source_memory: id,
            },
        ];
        let contradictions = extractor.find_contradicting_facts(&new_fact, &existing);
        assert_eq!(contradictions.len(), 1);
        assert_eq!(contradictions[0].object, "MySQL");
    }
}
