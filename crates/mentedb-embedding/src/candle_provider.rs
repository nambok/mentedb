//! Local embedding provider using Candle (pure Rust ML framework).
//!
//! Downloads and caches a small transformer model (all-MiniLM-L6-v2) from
//! Hugging Face on first use. Generates 384-dimensional embeddings locally
//! with no API key required.

use std::path::PathBuf;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{Repo, RepoType, api::sync::Api};
use mentedb_core::MenteError;
use mentedb_core::error::MenteResult;
use tokenizers::Tokenizer;

use crate::provider::EmbeddingProvider;

/// Default model for local embeddings.
const DEFAULT_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Local embedding provider powered by Candle.
///
/// Uses a small BERT-based model to generate embeddings entirely on CPU,
/// with no external API calls. The model is downloaded from Hugging Face
/// on first use and cached in the HF cache directory.
pub struct CandleEmbeddingProvider {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimensions: usize,
    model_id: String,
}

impl CandleEmbeddingProvider {
    /// Create a new provider with the default model (all-MiniLM-L6-v2).
    ///
    /// Downloads the model on first use. Subsequent calls load from cache.
    pub fn new() -> MenteResult<Self> {
        Self::with_model(DEFAULT_MODEL_ID)
    }

    /// Create a new provider with a specific Hugging Face model ID.
    pub fn with_model(model_id: &str) -> MenteResult<Self> {
        Self::load(model_id, None)
    }

    /// Create a new provider with a custom cache directory.
    pub fn with_cache_dir(cache_dir: PathBuf) -> MenteResult<Self> {
        Self::load(DEFAULT_MODEL_ID, Some(cache_dir))
    }

    fn load(model_id: &str, cache_dir: Option<PathBuf>) -> MenteResult<Self> {
        let device = Device::Cpu;

        let api = match cache_dir {
            Some(dir) => {
                let cache = hf_hub::Cache::new(dir);
                hf_hub::api::sync::ApiBuilder::from_cache(cache)
                    .build()
                    .map_err(|e| {
                        MenteError::Storage(format!("Failed to create HF API with cache: {e}"))
                    })?
            }
            None => Api::new()
                .map_err(|e| MenteError::Storage(format!("Failed to create HF API: {e}")))?,
        };

        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        tracing::info!(model = model_id, "Loading local embedding model");

        // Download model files
        let config_path = repo
            .get("config.json")
            .map_err(|e| MenteError::Storage(format!("Failed to download config.json: {e}")))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| MenteError::Storage(format!("Failed to download tokenizer.json: {e}")))?;
        let weights_path = repo.get("model.safetensors").map_err(|e| {
            MenteError::Storage(format!("Failed to download model.safetensors: {e}"))
        })?;

        // Load config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| MenteError::Storage(format!("Failed to read config: {e}")))?;
        let config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| MenteError::Storage(format!("Failed to parse config: {e}")))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| MenteError::Storage(format!("Failed to load tokenizer: {e}")))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)
                .map_err(|e| MenteError::Storage(format!("Failed to load weights: {e}")))?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| MenteError::Storage(format!("Failed to load model: {e}")))?;

        let dimensions = config.hidden_size;

        tracing::info!(
            model = model_id,
            dimensions = dimensions,
            "Local embedding model loaded"
        );

        Ok(Self {
            model,
            tokenizer,
            device,
            dimensions,
            model_id: model_id.to_string(),
        })
    }

    /// Encode texts into embeddings using mean pooling over token embeddings.
    fn encode(&self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| MenteError::Storage(format!("Tokenization failed: {e}")))?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let mut all_input_ids: Vec<u32> = Vec::new();
        let mut all_attention_mask: Vec<u32> = Vec::new();
        let mut all_token_type_ids: Vec<u32> = Vec::new();

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();

            let pad_len = max_len - ids.len();

            all_input_ids.extend_from_slice(ids);
            all_input_ids.extend(std::iter::repeat_n(0u32, pad_len));

            all_attention_mask.extend_from_slice(mask);
            all_attention_mask.extend(std::iter::repeat_n(0u32, pad_len));

            all_token_type_ids.extend_from_slice(type_ids);
            all_token_type_ids.extend(std::iter::repeat_n(0u32, pad_len));
        }

        let batch_size = texts.len();

        let input_ids = Tensor::from_vec(all_input_ids, (batch_size, max_len), &self.device)
            .map_err(|e| MenteError::Storage(format!("Tensor creation failed: {e}")))?;

        let attention_mask = Tensor::from_vec(
            all_attention_mask.clone(),
            (batch_size, max_len),
            &self.device,
        )
        .map_err(|e| MenteError::Storage(format!("Tensor creation failed: {e}")))?;

        let token_type_ids =
            Tensor::from_vec(all_token_type_ids, (batch_size, max_len), &self.device)
                .map_err(|e| MenteError::Storage(format!("Tensor creation failed: {e}")))?;

        // Forward pass
        let output = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| MenteError::Storage(format!("Model forward pass failed: {e}")))?;

        // Mean pooling: average token embeddings, masked by attention
        let mask_f32 = Tensor::from_vec(
            all_attention_mask
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<_>>(),
            (batch_size, max_len),
            &self.device,
        )
        .map_err(|e| MenteError::Storage(format!("Mask tensor failed: {e}")))?;

        let mask_expanded = mask_f32
            .unsqueeze(2)
            .map_err(|e| MenteError::Storage(format!("Unsqueeze failed: {e}")))?
            .broadcast_as(output.shape())
            .map_err(|e| MenteError::Storage(format!("Broadcast failed: {e}")))?;

        let masked = output
            .mul(&mask_expanded)
            .map_err(|e| MenteError::Storage(format!("Mul failed: {e}")))?;

        let summed = masked
            .sum(1)
            .map_err(|e| MenteError::Storage(format!("Sum failed: {e}")))?;

        let counts = mask_expanded
            .sum(1)
            .map_err(|e| MenteError::Storage(format!("Count sum failed: {e}")))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| MenteError::Storage(format!("Clamp failed: {e}")))?;

        let mean_pooled = summed
            .div(&counts)
            .map_err(|e| MenteError::Storage(format!("Div failed: {e}")))?;

        // L2 normalize each embedding
        let norms = mean_pooled
            .sqr()
            .map_err(|e| MenteError::Storage(format!("Sqr failed: {e}")))?
            .sum(1)
            .map_err(|e| MenteError::Storage(format!("Norm sum failed: {e}")))?
            .sqrt()
            .map_err(|e| MenteError::Storage(format!("Sqrt failed: {e}")))?
            .clamp(1e-12, f64::MAX)
            .map_err(|e| MenteError::Storage(format!("Norm clamp failed: {e}")))?
            .unsqueeze(1)
            .map_err(|e| MenteError::Storage(format!("Norm unsqueeze failed: {e}")))?
            .broadcast_as(mean_pooled.shape())
            .map_err(|e| MenteError::Storage(format!("Norm broadcast failed: {e}")))?;

        let normalized = mean_pooled
            .div(&norms)
            .map_err(|e| MenteError::Storage(format!("Normalize failed: {e}")))?;

        // Extract embeddings
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let emb = normalized
                .get(i)
                .map_err(|e| MenteError::Storage(format!("Get embedding failed: {e}")))?
                .to_vec1::<f32>()
                .map_err(|e| MenteError::Storage(format!("To vec failed: {e}")))?;
            results.push(emb);
        }

        Ok(results)
    }
}

impl EmbeddingProvider for CandleEmbeddingProvider {
    fn embed(&self, text: &str) -> MenteResult<Vec<f32>> {
        let results = self.encode(&[text])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| MenteError::Storage("Empty embedding result".to_string()))
    }

    fn embed_batch(&self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        self.encode(texts)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_provider_loads() {
        let provider = CandleEmbeddingProvider::new();
        assert!(
            provider.is_ok(),
            "Failed to load model: {:?}",
            provider.err()
        );
    }

    #[test]
    fn test_candle_embed_single() {
        let provider = CandleEmbeddingProvider::new().unwrap();
        let emb = provider.embed("hello world").unwrap();
        assert_eq!(emb.len(), provider.dimensions());

        // Check normalized
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "Not normalized: {norm}");
    }

    #[test]
    fn test_candle_embed_batch() {
        let provider = CandleEmbeddingProvider::new().unwrap();
        let results = provider.embed_batch(&["hello", "world", "test"]).unwrap();
        assert_eq!(results.len(), 3);
        for emb in &results {
            assert_eq!(emb.len(), provider.dimensions());
        }
    }

    #[test]
    fn test_candle_semantic_similarity() {
        let provider = CandleEmbeddingProvider::new().unwrap();
        let e1 = provider.embed("PostgreSQL database").unwrap();
        let e2 = provider.embed("relational database system").unwrap();
        let e3 = provider.embed("chocolate cake recipe").unwrap();

        let sim_related: f32 = e1.iter().zip(e2.iter()).map(|(a, b)| a * b).sum();
        let sim_unrelated: f32 = e1.iter().zip(e3.iter()).map(|(a, b)| a * b).sum();

        assert!(
            sim_related > sim_unrelated,
            "Related texts should be more similar: related={sim_related}, unrelated={sim_unrelated}"
        );
    }
}
