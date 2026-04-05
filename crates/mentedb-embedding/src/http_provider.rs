//! Generic HTTP-based embedding provider for OpenAI, Cohere, Voyage, and other APIs.

use std::collections::HashMap;

use mentedb_core::MenteError;
use mentedb_core::error::MenteResult;
use serde::{Deserialize, Serialize};

use crate::provider::{AsyncEmbeddingProvider, EmbeddingProvider};

/// Configuration for an HTTP-based embedding API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpEmbeddingConfig {
    /// The API endpoint URL.
    pub api_url: String,
    /// The API key for authentication.
    pub api_key: String,
    /// The model name to request.
    pub model_name: String,
    /// The dimensionality of the returned embeddings.
    pub dimensions: usize,
    /// Additional headers to include in requests.
    pub headers: HashMap<String, String>,
}

impl HttpEmbeddingConfig {
    /// Create a configuration for OpenAI's embedding API.
    ///
    /// Default dimensions: 1536 for text-embedding-ada-002, 3072 for text-embedding-3-large.
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        let dimensions = match model.as_str() {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536,
        };

        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        Self {
            api_url: "https://api.openai.com/v1/embeddings".to_string(),
            api_key: api_key.into(),
            model_name: model,
            dimensions,
            headers,
        }
    }

    /// Create a configuration for Cohere's embedding API.
    ///
    /// Default dimensions: 1024 for embed-english-v3.0.
    pub fn cohere(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        let dimensions = match model.as_str() {
            "embed-english-v3.0" => 1024,
            "embed-multilingual-v3.0" => 1024,
            "embed-english-light-v3.0" => 384,
            "embed-multilingual-light-v3.0" => 384,
            _ => 1024,
        };

        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        Self {
            api_url: "https://api.cohere.ai/v1/embed".to_string(),
            api_key: api_key.into(),
            model_name: model,
            dimensions,
            headers,
        }
    }

    /// Create a configuration for Voyage AI's embedding API.
    ///
    /// Default dimensions: 1024 for voyage-2.
    pub fn voyage(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        let dimensions = match model.as_str() {
            "voyage-2" => 1024,
            "voyage-large-2" => 1536,
            "voyage-code-2" => 1536,
            "voyage-lite-02-instruct" => 1024,
            _ => 1024,
        };

        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());

        Self {
            api_url: "https://api.voyageai.com/v1/embeddings".to_string(),
            api_key: api_key.into(),
            model_name: model,
            dimensions,
            headers,
        }
    }

    /// Override the embedding dimensions.
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Add a custom header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }
}

/// HTTP-based embedding provider.
///
/// Currently requires an external HTTP client feature to function.
/// The structure and configuration are fully usable for setup and validation.
pub struct HttpEmbeddingProvider {
    config: HttpEmbeddingConfig,
}

impl HttpEmbeddingProvider {
    /// Create a new HTTP embedding provider with the given configuration.
    pub fn new(config: HttpEmbeddingConfig) -> Self {
        Self { config }
    }

    /// Get a reference to the provider's configuration.
    pub fn config(&self) -> &HttpEmbeddingConfig {
        &self.config
    }
}

impl AsyncEmbeddingProvider for HttpEmbeddingProvider {
    async fn embed(&self, _text: &str) -> MenteResult<Vec<f32>> {
        Err(MenteError::Storage(
            "HTTP embedding requires the 'http' feature for async, use sync EmbeddingProvider instead".to_string(),
        ))
    }

    async fn embed_batch(&self, _texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        Err(MenteError::Storage(
            "HTTP embedding requires the 'http' feature for async, use sync EmbeddingProvider instead".to_string(),
        ))
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn model_name(&self) -> &str {
        &self.config.model_name
    }
}

#[cfg(feature = "http")]
mod http_impl {
    use super::*;
    use serde_json::json;

    #[derive(Deserialize)]
    struct OpenAIEmbeddingResponse {
        data: Vec<OpenAIEmbeddingData>,
    }

    #[derive(Deserialize)]
    struct OpenAIEmbeddingData {
        embedding: Vec<f32>,
    }

    impl EmbeddingProvider for HttpEmbeddingProvider {
        fn embed(&self, text: &str) -> MenteResult<Vec<f32>> {
            let body = json!({
                "model": self.config.model_name,
                "input": text,
            });

            let mut req = ureq::post(&self.config.api_url)
                .header("Authorization", &format!("Bearer {}", self.config.api_key))
                .header("Content-Type", "application/json");

            for (k, v) in &self.config.headers {
                req = req.header(k, v);
            }

            let resp: OpenAIEmbeddingResponse = req
                .send_json(&body)
                .map_err(|e| MenteError::Storage(format!("HTTP embedding request failed: {}", e)))?
                .body_mut()
                .read_json()
                .map_err(|e| MenteError::Storage(format!("Failed to parse embedding response: {}", e)))?;

            resp.data
                .into_iter()
                .next()
                .map(|d| d.embedding)
                .ok_or_else(|| MenteError::Storage("Empty embedding response".to_string()))
        }

        fn embed_batch(&self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
            let body = json!({
                "model": self.config.model_name,
                "input": texts,
            });

            let mut req = ureq::post(&self.config.api_url)
                .header("Authorization", &format!("Bearer {}", self.config.api_key))
                .header("Content-Type", "application/json");

            for (k, v) in &self.config.headers {
                req = req.header(k, v);
            }

            let resp: OpenAIEmbeddingResponse = req
                .send_json(&body)
                .map_err(|e| MenteError::Storage(format!("HTTP embedding request failed: {}", e)))?
                .body_mut()
                .read_json()
                .map_err(|e| MenteError::Storage(format!("Failed to parse embedding response: {}", e)))?;

            Ok(resp.data.into_iter().map(|d| d.embedding).collect())
        }

        fn dimensions(&self) -> usize {
            self.config.dimensions
        }

        fn model_name(&self) -> &str {
            &self.config.model_name
        }
    }
}

#[cfg(not(feature = "http"))]
impl EmbeddingProvider for HttpEmbeddingProvider {
    fn embed(&self, _text: &str) -> MenteResult<Vec<f32>> {
        Err(MenteError::Storage(
            "HTTP embedding requires the 'http' feature. Enable it in Cargo.toml.".to_string(),
        ))
    }

    fn embed_batch(&self, _texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        Err(MenteError::Storage(
            "HTTP embedding requires the 'http' feature. Enable it in Cargo.toml.".to_string(),
        ))
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn model_name(&self) -> &str {
        &self.config.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_config() {
        let config = HttpEmbeddingConfig::openai("sk-test", "text-embedding-3-small");
        assert_eq!(config.api_url, "https://api.openai.com/v1/embeddings");
        assert_eq!(config.dimensions, 1536);
        assert_eq!(config.model_name, "text-embedding-3-small");
    }

    #[test]
    fn test_cohere_config() {
        let config = HttpEmbeddingConfig::cohere("key", "embed-english-v3.0");
        assert_eq!(config.api_url, "https://api.cohere.ai/v1/embed");
        assert_eq!(config.dimensions, 1024);
    }

    #[test]
    fn test_voyage_config() {
        let config = HttpEmbeddingConfig::voyage("key", "voyage-2");
        assert_eq!(config.api_url, "https://api.voyageai.com/v1/embeddings");
        assert_eq!(config.dimensions, 1024);
    }

    #[test]
    fn test_with_dimensions_override() {
        let config =
            HttpEmbeddingConfig::openai("key", "text-embedding-3-small").with_dimensions(256);
        assert_eq!(config.dimensions, 256);
    }
}
