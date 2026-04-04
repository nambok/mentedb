use serde::{Deserialize, Serialize};

/// Which LLM provider to use for extraction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmProvider {
    OpenAI,
    Anthropic,
    Ollama,
    Custom,
}

impl LlmProvider {
    /// Default API URL for this provider.
    pub fn default_url(&self) -> &str {
        match self {
            LlmProvider::OpenAI => "https://api.openai.com/v1/chat/completions",
            LlmProvider::Anthropic => "https://api.anthropic.com/v1/messages",
            LlmProvider::Ollama => "http://localhost:11434/api/chat",
            LlmProvider::Custom => "http://localhost:8080/v1/chat/completions",
        }
    }

    /// Default model name for this provider.
    pub fn default_model(&self) -> &str {
        match self {
            LlmProvider::OpenAI => "gpt-4o-mini",
            LlmProvider::Anthropic => "claude-sonnet-4-20250514",
            LlmProvider::Ollama => "llama3",
            LlmProvider::Custom => "default",
        }
    }
}

/// Configuration for the extraction pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Which LLM provider to use.
    pub provider: LlmProvider,
    /// API key for the provider (not needed for Ollama).
    pub api_key: Option<String>,
    /// API endpoint URL (defaults based on provider).
    pub api_url: String,
    /// Model name to use.
    pub model: String,
    /// Maximum number of memories to extract from a single conversation.
    pub max_extractions_per_conversation: usize,
    /// Minimum confidence score for a memory to be accepted (0.0 to 1.0).
    pub quality_threshold: f32,
    /// Embedding similarity above which a memory is considered a duplicate (0.0 to 1.0).
    pub deduplication_threshold: f32,
    /// Whether to check new memories against existing ones for contradictions.
    pub enable_contradiction_check: bool,
    /// Whether to check new memories against existing ones for duplicates.
    pub enable_deduplication: bool,
}

impl ExtractionConfig {
    /// Create a config for OpenAI with the given API key.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            provider: LlmProvider::OpenAI,
            api_key: Some(api_key.into()),
            api_url: LlmProvider::OpenAI.default_url().to_string(),
            model: LlmProvider::OpenAI.default_model().to_string(),
            ..Self::default()
        }
    }

    /// Create a config for Anthropic with the given API key.
    pub fn anthropic(api_key: impl Into<String>) -> Self {
        Self {
            provider: LlmProvider::Anthropic,
            api_key: Some(api_key.into()),
            api_url: LlmProvider::Anthropic.default_url().to_string(),
            model: LlmProvider::Anthropic.default_model().to_string(),
            ..Self::default()
        }
    }

    /// Create a config for a local Ollama instance.
    pub fn ollama() -> Self {
        Self {
            provider: LlmProvider::Ollama,
            api_key: None,
            api_url: LlmProvider::Ollama.default_url().to_string(),
            model: LlmProvider::Ollama.default_model().to_string(),
            ..Self::default()
        }
    }
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            provider: LlmProvider::OpenAI,
            api_key: None,
            api_url: LlmProvider::OpenAI.default_url().to_string(),
            model: LlmProvider::OpenAI.default_model().to_string(),
            max_extractions_per_conversation: 50,
            quality_threshold: 0.6,
            deduplication_threshold: 0.85,
            enable_contradiction_check: true,
            enable_deduplication: true,
        }
    }
}
