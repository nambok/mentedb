use crate::config::{ExtractionConfig, LlmProvider};
use crate::error::ExtractionError;

/// Trait for LLM providers that can extract memories from conversation text.
pub trait ExtractionProvider: Send + Sync {
    /// Send a conversation to the LLM with the given system prompt and return
    /// the raw response text (expected to be JSON).
    fn extract(
        &self,
        conversation: &str,
        system_prompt: &str,
    ) -> impl std::future::Future<Output = Result<String, ExtractionError>> + Send;
}

/// HTTP-based extraction provider that calls OpenAI, Anthropic, or Ollama APIs.
pub struct HttpExtractionProvider {
    client: reqwest::Client,
    config: ExtractionConfig,
}

impl HttpExtractionProvider {
    pub fn new(config: ExtractionConfig) -> Result<Self, ExtractionError> {
        if config.provider != LlmProvider::Ollama && config.api_key.is_none() {
            return Err(ExtractionError::ConfigError(
                "API key is required for this provider".to_string(),
            ));
        }
        let client = reqwest::Client::new();
        Ok(Self { client, config })
    }

    /// Expand a search query into multiple sub-queries via LLM.
    ///
    /// Given a natural language question, extracts 2-3 search queries
    /// targeting key entities, names, places, dates, and topics.
    /// Returns the sub-queries (does NOT include the original).
    pub async fn expand_query(&self, query: &str) -> Result<Vec<String>, ExtractionError> {
        let system_prompt = "You extract search queries from questions. \
            Return 2-3 short search queries, one per line, no numbering. \
            Focus on key entities, names, places, dates, and topics.";
        let result = self.call_with_retry(query, system_prompt).await?;
        let queries: Vec<String> = result
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .take(3)
            .collect();
        Ok(queries)
    }

    async fn call_openai(
        &self,
        conversation: &str,
        system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        let body = serde_json::json!({
            "model": self.config.model,
            "response_format": { "type": "json_object" },
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": conversation }
            ]
        });

        let api_key = self.config.api_key.as_deref().unwrap_or_default();

        let resp = self
            .client
            .post(&self.config.api_url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(ExtractionError::ProviderError(format!(
                "OpenAI API returned {status}: {text}"
            )));
        }

        let parsed: serde_json::Value = serde_json::from_str(&text)?;
        parsed["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| {
                ExtractionError::ParseError("Missing content in OpenAI response".to_string())
            })
    }

    async fn call_anthropic(
        &self,
        conversation: &str,
        system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        let body = serde_json::json!({
            "model": self.config.model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [
                { "role": "user", "content": conversation }
            ]
        });

        let api_key = self.config.api_key.as_deref().unwrap_or_default();

        let resp = self
            .client
            .post(&self.config.api_url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(ExtractionError::ProviderError(format!(
                "Anthropic API returned {status}: {text}"
            )));
        }

        let parsed: serde_json::Value = serde_json::from_str(&text)?;

        // Anthropic may return multiple content blocks; find the first text block
        let content_text = parsed["content"]
            .as_array()
            .and_then(|blocks| {
                blocks.iter().find_map(|block| {
                    if block["type"].as_str() == Some("text") {
                        block["text"].as_str().map(|s| s.to_string())
                    } else {
                        None
                    }
                })
            })
            .or_else(|| {
                // Fallback: try the old path for backwards compat
                parsed["content"][0]["text"]
                    .as_str()
                    .map(|s| s.to_string())
            });

        match content_text {
            Some(t) if !t.trim().is_empty() => Ok(t),
            Some(_) => {
                tracing::warn!(
                    model = %self.config.model,
                    "Anthropic returned empty text content"
                );
                Ok("{\"memories\": []}".to_string())
            }
            None => {
                tracing::warn!(
                    model = %self.config.model,
                    response_preview = &text[..text.len().min(300)],
                    "No text block found in Anthropic response"
                );
                Ok("{\"memories\": []}".to_string())
            }
        }
    }

    async fn call_ollama(
        &self,
        conversation: &str,
        system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        let body = serde_json::json!({
            "model": self.config.model,
            "stream": false,
            "format": "json",
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": conversation }
            ]
        });

        let resp = self
            .client
            .post(&self.config.api_url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(ExtractionError::ProviderError(format!(
                "Ollama API returned {status}: {text}"
            )));
        }

        let parsed: serde_json::Value = serde_json::from_str(&text)?;
        parsed["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| {
                ExtractionError::ParseError("Missing content in Ollama response".to_string())
            })
    }

    /// Execute a request with retry logic for rate limits (HTTP 429).
    /// Uses exponential backoff: 1s, 2s, 4s.
    async fn call_with_retry(
        &self,
        conversation: &str,
        system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        let max_attempts = 3;
        let mut last_err = None;

        for attempt in 0..max_attempts {
            if attempt > 0 {
                let delay = std::time::Duration::from_secs(1 << attempt);
                tracing::warn!(
                    attempt,
                    delay_secs = delay.as_secs(),
                    "retrying after rate limit"
                );
                tokio::time::sleep(delay).await;
            }

            tracing::info!(
                provider = ?self.config.provider,
                model = %self.config.model,
                attempt = attempt + 1,
                "calling LLM extraction API"
            );

            let result = match self.config.provider {
                LlmProvider::OpenAI | LlmProvider::Custom => {
                    self.call_openai(conversation, system_prompt).await
                }
                LlmProvider::Anthropic => self.call_anthropic(conversation, system_prompt).await,
                LlmProvider::Ollama => self.call_ollama(conversation, system_prompt).await,
            };

            match result {
                Ok(text) => {
                    tracing::info!(response_len = text.len(), "LLM extraction complete");
                    return Ok(text);
                }
                Err(ExtractionError::ProviderError(ref msg)) if msg.contains("429") => {
                    tracing::warn!(attempt = attempt + 1, "rate limited by provider");
                    last_err = Some(result.unwrap_err());
                    continue;
                }
                Err(e) => {
                    tracing::error!(error = %e, "LLM extraction failed");
                    return Err(e);
                }
            }
        }

        match last_err {
            Some(e) => Err(e),
            None => Err(ExtractionError::RateLimitExceeded {
                attempts: max_attempts,
            }),
        }
    }
}

impl ExtractionProvider for HttpExtractionProvider {
    async fn extract(
        &self,
        conversation: &str,
        system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        self.call_with_retry(conversation, system_prompt).await
    }
}

/// Mock extraction provider for testing. Returns a predefined JSON response.
pub struct MockExtractionProvider {
    response: String,
}

impl MockExtractionProvider {
    /// Create a mock provider that always returns the given JSON string.
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
        }
    }

    /// Create a mock provider with a realistic extraction response.
    pub fn with_realistic_response() -> Self {
        let response = serde_json::json!({
            "memories": [
                {
                    "content": "The team decided to use PostgreSQL 15 as the primary database for the REST API project",
                    "memory_type": "decision",
                    "confidence": 0.95,
                    "entities": ["PostgreSQL", "REST API"],
                    "tags": ["database", "architecture"],
                    "reasoning": "Explicitly decided after comparing options"
                },
                {
                    "content": "REST endpoints should follow the /api/v1/ prefix convention",
                    "memory_type": "decision",
                    "confidence": 0.9,
                    "entities": ["REST API"],
                    "tags": ["api-design", "conventions"],
                    "reasoning": "Team agreed on URL structure"
                },
                {
                    "content": "User prefers Rust over Go for backend services due to memory safety guarantees",
                    "memory_type": "preference",
                    "confidence": 0.85,
                    "entities": ["Rust", "Go"],
                    "tags": ["language", "backend"],
                    "reasoning": "Explicitly stated preference with clear reasoning"
                },
                {
                    "content": "The initial plan to use MongoDB was incorrect; PostgreSQL is the right choice for relational data",
                    "memory_type": "correction",
                    "confidence": 0.9,
                    "entities": ["MongoDB", "PostgreSQL"],
                    "tags": ["database", "correction"],
                    "reasoning": "Corrected an earlier wrong assumption"
                },
                {
                    "content": "The project deadline is March 15, 2025",
                    "memory_type": "fact",
                    "confidence": 0.8,
                    "entities": ["REST API project"],
                    "tags": ["timeline"],
                    "reasoning": "Confirmed date mentioned in discussion"
                },
                {
                    "content": "Using global mutable state for database connections caused race conditions in testing",
                    "memory_type": "anti_pattern",
                    "confidence": 0.85,
                    "entities": [],
                    "tags": ["testing", "concurrency"],
                    "reasoning": "Documented failure pattern to avoid repeating"
                },
                {
                    "content": "Low confidence speculation about maybe using Redis",
                    "memory_type": "fact",
                    "confidence": 0.3,
                    "entities": ["Redis"],
                    "tags": ["cache"],
                    "reasoning": "Mentioned but not confirmed"
                }
            ]
        });
        Self::new(response.to_string())
    }
}

impl ExtractionProvider for MockExtractionProvider {
    async fn extract(
        &self,
        _conversation: &str,
        _system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        Ok(self.response.clone())
    }
}
