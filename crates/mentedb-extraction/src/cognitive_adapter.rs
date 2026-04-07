//! Adapter that wraps [`HttpExtractionProvider`] as a [`LlmJudge`] for the
//! cognitive service. This bridges the extraction infrastructure (HTTP clients,
//! provider configs, retry logic) with the cognitive judgment layer.

use crate::ExtractionProvider;
use crate::provider::HttpExtractionProvider;
use mentedb_cognitive::LlmJudgeError;

/// Wraps an [`HttpExtractionProvider`] so it can be used as a [`LlmJudge`]
/// for the [`CognitiveLlmService`](mentedb_cognitive::CognitiveLlmService).
///
/// The extraction provider already handles OpenAI, Anthropic, Ollama, and
/// Custom endpoints with retry logic — this adapter simply maps the interface.
pub struct ExtractionLlmJudge {
    provider: HttpExtractionProvider,
}

impl ExtractionLlmJudge {
    /// Create a new adapter from an existing extraction provider.
    pub fn new(provider: HttpExtractionProvider) -> Self {
        Self { provider }
    }
}

impl mentedb_cognitive::LlmJudge for ExtractionLlmJudge {
    async fn complete(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, LlmJudgeError> {
        self.provider
            .extract(user_prompt, system_prompt)
            .await
            .map_err(|e| LlmJudgeError::ProviderError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ExtractionConfig;

    #[test]
    fn test_adapter_construction() {
        let config = ExtractionConfig::ollama();
        let provider = HttpExtractionProvider::new(config).unwrap();
        let _judge = ExtractionLlmJudge::new(provider);
    }
}
