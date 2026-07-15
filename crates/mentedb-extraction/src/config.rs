use serde::{Deserialize, Serialize};

/// Which LLM provider to use for extraction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmProvider {
    OpenAI,
    Anthropic,
    Ollama,
    /// AWS Bedrock (Anthropic Claude on Bedrock) signed with SigV4. Uses AWS
    /// credentials from the environment, not an API key. The endpoint is built
    /// per-request from `ExtractionConfig::region` plus the model id, so
    /// `default_url` is empty for this provider. Requires the `bedrock` feature.
    Bedrock,
    Custom,
}

impl LlmProvider {
    /// Default API URL for this provider.
    ///
    /// Bedrock returns an empty string: its endpoint is region-specific and is
    /// built per-request from `ExtractionConfig::region` and the model id
    /// (`https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke`).
    pub fn default_url(&self) -> &str {
        match self {
            LlmProvider::OpenAI => "https://api.openai.com/v1/chat/completions",
            LlmProvider::Anthropic => "https://api.anthropic.com/v1/messages",
            LlmProvider::Ollama => "http://localhost:11434/api/chat",
            // Region-specific; built per-request in call_bedrock.
            LlmProvider::Bedrock => "",
            LlmProvider::Custom => "http://localhost:8080/v1/chat/completions",
        }
    }

    /// Default extraction model for this provider (cheap, high-volume).
    pub fn default_model(&self) -> &str {
        match self {
            LlmProvider::OpenAI => "gpt-4o-mini",
            LlmProvider::Anthropic => "claude-haiku-4-5",
            LlmProvider::Ollama => "llama3",
            LlmProvider::Bedrock => "us.anthropic.claude-haiku-4-5",
            LlmProvider::Custom => "default",
        }
    }

    /// Default reader model for this provider (smart, low-volume).
    pub fn default_reader_model(&self) -> &str {
        match self {
            LlmProvider::OpenAI => "gpt-4o",
            LlmProvider::Anthropic => "claude-sonnet-4-20250514",
            LlmProvider::Ollama => "llama3",
            LlmProvider::Bedrock => "us.anthropic.claude-sonnet-4-6",
            LlmProvider::Custom => "default",
        }
    }
}

/// Resolve the AWS region for Bedrock from the environment, defaulting to
/// `us-east-1`. Checks `MENTEDB_LLM_REGION` first (MenteDB-specific override),
/// then the standard `AWS_REGION`.
pub fn default_bedrock_region() -> String {
    std::env::var("MENTEDB_LLM_REGION")
        .or_else(|_| std::env::var("AWS_REGION"))
        .unwrap_or_else(|_| "us-east-1".to_string())
}

/// Configuration for the extraction pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Which LLM provider to use.
    pub provider: LlmProvider,
    /// API key for the provider (not needed for Ollama).
    pub api_key: Option<String>,
    /// API endpoint URL (defaults based on provider).
    ///
    /// Ignored for the Bedrock provider, whose endpoint is derived from
    /// `region` and `model` per-request.
    pub api_url: String,
    /// Model name to use.
    pub model: String,
    /// AWS region for the Bedrock provider (e.g. "us-east-1"). Ignored by all
    /// other providers. Defaults from `MENTEDB_LLM_REGION`/`AWS_REGION`.
    pub region: Option<String>,
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
    /// Number of extraction passes (1 = single pass, 2 = first pass + verification).
    pub extraction_passes: usize,
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

    /// Create a config for AWS Bedrock (Anthropic Claude on Bedrock) in the
    /// given region. Credentials are read from the AWS environment variables at
    /// call time, not stored here. Requires the `bedrock` feature to actually
    /// dispatch calls.
    pub fn bedrock(region: impl Into<String>) -> Self {
        Self {
            provider: LlmProvider::Bedrock,
            api_key: None,
            api_url: LlmProvider::Bedrock.default_url().to_string(),
            model: LlmProvider::Bedrock.default_model().to_string(),
            region: Some(region.into()),
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
            region: None,
            max_extractions_per_conversation: 50,
            quality_threshold: 0.6,
            deduplication_threshold: 0.85,
            enable_contradiction_check: true,
            enable_deduplication: true,
            extraction_passes: 1,
        }
    }
}
