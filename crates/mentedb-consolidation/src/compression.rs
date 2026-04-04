use mentedb_core::MemoryNode;
use mentedb_core::types::MemoryId;
use serde::{Deserialize, Serialize};

/// A compressed representation of a memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedMemory {
    pub original_id: MemoryId,
    pub compressed_content: String,
    pub compression_ratio: f32,
    pub key_facts: Vec<String>,
}

const KEY_WORDS: &[&str] = &[
    "decided",
    "uses",
    "prefers",
    "switched",
    "chose",
    "selected",
    "important",
    "critical",
    "must",
    "should",
    "always",
    "never",
];

const FILLER_WORDS: &[&str] = &[
    "actually",
    "basically",
    "honestly",
    "really",
    "very",
    "quite",
    "just",
    "simply",
    "perhaps",
    "maybe",
    "somewhat",
    "rather",
    "kind of",
    "sort of",
    "you know",
    "i think",
    "i mean",
];

/// Compresses verbose memories into token-efficient forms.
pub struct MemoryCompressor;

impl MemoryCompressor {
    pub fn new() -> Self {
        Self
    }

    /// Compress a single memory by extracting key sentences and removing filler.
    pub fn compress(&self, memory: &MemoryNode) -> CompressedMemory {
        let paragraphs: Vec<&str> = memory.content.split("\n\n").collect();
        let mut key_sentences: Vec<String> = Vec::new();

        for para in &paragraphs {
            let sentences: Vec<&str> = para
                .split('.')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            // Always include first sentence of each paragraph
            if let Some(&first) = sentences.first() {
                let cleaned = remove_filler(first);
                if !cleaned.is_empty() && !key_sentences.contains(&cleaned) {
                    key_sentences.push(cleaned);
                }
            }

            // Include sentences with keywords
            for &sentence in &sentences[1..] {
                let lower = sentence.to_lowercase();
                if KEY_WORDS.iter().any(|kw| lower.contains(kw)) {
                    let cleaned = remove_filler(sentence);
                    if !cleaned.is_empty() && !key_sentences.contains(&cleaned) {
                        key_sentences.push(cleaned);
                    }
                }
            }
        }

        let compressed_content = key_sentences.join(". ");
        let original_len = memory.content.len().max(1) as f32;
        let compressed_len = compressed_content.len() as f32;
        let compression_ratio = compressed_len / original_len;

        CompressedMemory {
            original_id: memory.id,
            compressed_content,
            compression_ratio,
            key_facts: key_sentences,
        }
    }

    /// Compress a batch of memories.
    pub fn compress_batch(&self, memories: &[MemoryNode]) -> Vec<CompressedMemory> {
        memories.iter().map(|m| self.compress(m)).collect()
    }

    /// Estimate token count for text (word_count * 1.3).
    pub fn estimate_tokens(text: &str) -> usize {
        let word_count = text.split_whitespace().count();
        (word_count as f64 * 1.3).ceil() as usize
    }
}

impl Default for MemoryCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Remove filler words from text.
fn remove_filler(text: &str) -> String {
    let mut result = text.to_string();
    for &filler in FILLER_WORDS {
        // Case-insensitive removal
        let lower = result.to_lowercase();
        if let Some(pos) = lower.find(filler) {
            let end = pos + filler.len();
            let actual = &result[pos..end];
            result = result.replacen(actual, "", 1);
        }
    }
    // Collapse whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_memory;

    #[test]
    fn test_compress_extracts_key_sentences() {
        let compressor = MemoryCompressor::new();
        let m = make_memory(
            "The team met today. They discussed various topics. They decided on Rust for the backend. It was a long meeting.",
            vec![1.0],
        );
        let compressed = compressor.compress(&m);
        assert!(compressed.compressed_content.contains("decided"));
        assert!(compressed.compression_ratio <= 1.0);
    }

    #[test]
    fn test_compress_removes_filler() {
        let compressor = MemoryCompressor::new();
        let m = make_memory("We basically decided to actually use Postgres", vec![1.0]);
        let compressed = compressor.compress(&m);
        assert!(!compressed.compressed_content.contains("basically"));
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(MemoryCompressor::estimate_tokens("hello world"), 3); // 2 * 1.3 = 2.6 -> 3
        assert_eq!(MemoryCompressor::estimate_tokens(""), 0);
    }

    #[test]
    fn test_compress_batch() {
        let compressor = MemoryCompressor::new();
        let memories = vec![
            make_memory("First memory content", vec![1.0]),
            make_memory("Second memory content", vec![1.0]),
        ];
        let results = compressor.compress_batch(&memories);
        assert_eq!(results.len(), 2);
    }
}
