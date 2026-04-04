use thiserror::Error;

/// Errors that can occur during memory extraction.
#[derive(Debug, Error)]
pub enum ExtractionError {
    /// The LLM provider returned an error or was unreachable.
    #[error("provider error: {0}")]
    ProviderError(String),

    /// The LLM response could not be parsed as valid extraction output.
    #[error("parse error: {0}")]
    ParseError(String),

    /// The extraction configuration is invalid.
    #[error("config error: {0}")]
    ConfigError(String),

    /// An HTTP request failed.
    #[error("http error: {0}")]
    HttpError(String),

    /// An embedding operation failed.
    #[error("embedding error: {0}")]
    EmbeddingError(String),

    /// Rate limit exceeded after all retry attempts.
    #[error("rate limit exceeded after {attempts} attempts")]
    RateLimitExceeded { attempts: usize },
}

impl From<reqwest::Error> for ExtractionError {
    fn from(e: reqwest::Error) -> Self {
        ExtractionError::HttpError(e.to_string())
    }
}

impl From<serde_json::Error> for ExtractionError {
    fn from(e: serde_json::Error) -> Self {
        ExtractionError::ParseError(e.to_string())
    }
}
