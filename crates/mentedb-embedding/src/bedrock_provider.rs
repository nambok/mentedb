//! AWS Bedrock embedding provider (Amazon Titan and Cohere Embed) over the
//! InvokeModel REST API.
//!
//! Deliberately dependency light: a synchronous `ureq` call plus a small
//! hand-rolled SigV4 signer, so the engine gains native Bedrock support without
//! pulling in the full async AWS SDK. This matches the lean, no-heavy-deps
//! philosophy of the rest of the engine (see the `http` provider).
//!
//! Credentials come from [`AwsCredentialSource`]: static keys from the
//! environment, or the ECS task role / EKS Pod Identity endpoint, refreshed
//! before they expire.
//!
//! ```no_run
//! use mentedb_embedding::bedrock_provider::{BedrockEmbeddingConfig, BedrockEmbeddingProvider};
//! use mentedb_embedding::provider::EmbeddingProvider;
//!
//! let cfg = BedrockEmbeddingConfig::titan_v2("us-east-1")?; // reads creds from env
//! let provider = BedrockEmbeddingProvider::new(cfg)?;
//! let vector = provider.embed("the sky is blue")?;
//! # Ok::<(), mentedb_core::error::MenteError>(())
//! ```

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use mentedb_core::error::{MenteError, MenteResult};

use crate::aws_credentials::AwsCredentialSource;
use crate::provider::EmbeddingProvider;

type HmacSha256 = Hmac<Sha256>;

const SERVICE: &str = "bedrock";

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// SigV4 signing key: HMAC chain over date, region, service, "aws4_request".
fn signing_key(secret: &str, datestamp: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_sha256(format!("AWS4{secret}").as_bytes(), datestamp.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

/// URI-encode a single path segment per SigV4 rules (unreserved chars pass
/// through; everything else, including the `:` in a model id, is percent
/// encoded). The request URL and the signed canonical URI must match exactly.
fn uri_encode_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Most texts Cohere Embed accepts in one InvokeModel call.
const COHERE_MAX_BATCH: usize = 96;

/// Request and response shape of a Bedrock embedding model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BedrockEmbeddingFamily {
    /// Amazon Titan: one text per call.
    Titan,
    /// Cohere Embed: up to 96 texts per call.
    Cohere,
}

impl BedrockEmbeddingFamily {
    /// Family for a model id, by its provider prefix.
    pub fn for_model(model_id: &str) -> MenteResult<Self> {
        let bare = model_id.rsplit('/').next().unwrap_or(model_id);
        let bare = bare
            .strip_prefix("us.")
            .or_else(|| bare.strip_prefix("eu."))
            .or_else(|| bare.strip_prefix("apac."))
            .or_else(|| bare.strip_prefix("global."))
            .unwrap_or(bare);
        if bare.starts_with("amazon.titan-embed") {
            Ok(Self::Titan)
        } else if bare.starts_with("cohere.embed") {
            Ok(Self::Cohere)
        } else {
            Err(MenteError::Storage(format!(
                "unsupported Bedrock embedding model '{model_id}' (expected amazon.titan-embed-* \
                 or cohere.embed-*)"
            )))
        }
    }
}

/// Configuration for the Bedrock embedding provider.
#[derive(Clone)]
pub struct BedrockEmbeddingConfig {
    pub region: String,
    pub model_id: String,
    pub family: BedrockEmbeddingFamily,
    pub dimensions: usize,
    pub credentials: AwsCredentialSource,
}

impl BedrockEmbeddingConfig {
    /// Any supported model. `dimensions` must match what the model returns.
    pub fn new(
        region: impl Into<String>,
        model_id: impl Into<String>,
        dimensions: usize,
        credentials: AwsCredentialSource,
    ) -> MenteResult<Self> {
        let model_id = model_id.into();
        Ok(Self {
            region: region.into(),
            family: BedrockEmbeddingFamily::for_model(&model_id)?,
            model_id,
            dimensions,
            credentials,
        })
    }

    /// A known model with its native dimensions, credentials from the
    /// environment or the container role.
    pub fn for_model(region: impl Into<String>, model_id: &str) -> MenteResult<Self> {
        let dimensions = match model_id {
            "amazon.titan-embed-text-v2:0" => 1024,
            "amazon.titan-embed-text-v1" => 1536,
            "cohere.embed-english-v3" | "cohere.embed-multilingual-v3" => 1024,
            other => {
                return Err(MenteError::Storage(format!(
                    "unknown dimensions for Bedrock embedding model '{other}'; use \
                     BedrockEmbeddingConfig::new with explicit dimensions"
                )));
            }
        };
        Self::new(
            region,
            model_id,
            dimensions,
            AwsCredentialSource::from_env()?,
        )
    }

    /// Amazon Titan Text Embeddings V2 (1024 dimensions).
    pub fn titan_v2(region: impl Into<String>) -> MenteResult<Self> {
        Self::for_model(region, "amazon.titan-embed-text-v2:0")
    }

    /// Amazon Titan Text Embeddings V1 (1536 dimensions).
    pub fn titan_v1(region: impl Into<String>) -> MenteResult<Self> {
        Self::for_model(region, "amazon.titan-embed-text-v1")
    }

    /// Cohere Embed English v3 (1024 dimensions).
    pub fn cohere_english_v3(region: impl Into<String>) -> MenteResult<Self> {
        Self::for_model(region, "cohere.embed-english-v3")
    }
}

/// Synchronous Bedrock embedding provider.
pub struct BedrockEmbeddingProvider {
    config: BedrockEmbeddingConfig,
    agent: ureq::Agent,
}

impl BedrockEmbeddingProvider {
    pub fn new(config: BedrockEmbeddingConfig) -> MenteResult<Self> {
        Ok(Self {
            config,
            agent: ureq::Agent::new_with_defaults(),
        })
    }

    /// Embed up to one call's worth of texts.
    fn invoke(&self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        let body = match self.config.family {
            BedrockEmbeddingFamily::Titan => serde_json::json!({ "inputText": texts[0] }),
            // Stored memories and queries are embedded the same way, so the
            // document input type is used for both, as on the cloud platform.
            BedrockEmbeddingFamily::Cohere => serde_json::json!({
                "texts": texts,
                "input_type": "search_document",
            }),
        };
        let value = self.post(&body)?;
        let vectors = match self.config.family {
            BedrockEmbeddingFamily::Titan => {
                let r: TitanResponse = serde_json::from_value(value).map_err(|e| {
                    MenteError::Storage(format!("bedrock response parse failed: {e}"))
                })?;
                vec![r.embedding]
            }
            BedrockEmbeddingFamily::Cohere => {
                let r: CohereResponse = serde_json::from_value(value).map_err(|e| {
                    MenteError::Storage(format!("bedrock response parse failed: {e}"))
                })?;
                r.embeddings
            }
        };
        if vectors.len() != texts.len() {
            return Err(MenteError::Storage(format!(
                "bedrock returned {} embeddings for {} texts",
                vectors.len(),
                texts.len()
            )));
        }
        Ok(vectors)
    }

    /// Sign and send one InvokeModel request.
    fn post(&self, body: &serde_json::Value) -> MenteResult<serde_json::Value> {
        let cfg = &self.config;
        let creds = cfg.credentials.credentials()?;
        let host = format!("bedrock-runtime.{}.amazonaws.com", cfg.region);
        // The request is sent with the raw model id in the path, while the SigV4
        // canonical URI uses the percent-encoded form. AWS re-encodes the path it
        // receives the same way, so signing the encoded form while sending the raw
        // form matches, which is what the AWS SDKs do for model ids containing ':'.
        let canonical_uri = format!("/model/{}/invoke", uri_encode_segment(&cfg.model_id));
        let url = format!("https://{host}/model/{}/invoke", cfg.model_id);

        let body =
            serde_json::to_vec(body).map_err(|e| MenteError::Serialization(e.to_string()))?;
        let payload_hash = sha256_hex(&body);

        let now = chrono::Utc::now();
        let amzdate = now.format("%Y%m%dT%H%M%SZ").to_string();
        let datestamp = now.format("%Y%m%d").to_string();

        // Canonical (sorted, lowercase) headers that we sign and send.
        let mut headers: Vec<(String, String)> = vec![
            ("host".to_string(), host.clone()),
            ("x-amz-content-sha256".to_string(), payload_hash.clone()),
            ("x-amz-date".to_string(), amzdate.clone()),
        ];
        if let Some(token) = &creds.session_token {
            headers.push(("x-amz-security-token".to_string(), token.clone()));
        }
        headers.sort_by(|a, b| a.0.cmp(&b.0));
        let canonical_headers: String = headers.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
        let signed_headers = headers
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>()
            .join(";");

        let canonical_request = format!(
            "POST\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        );
        let scope = format!("{datestamp}/{}/{SERVICE}/aws4_request", cfg.region);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amzdate}\n{scope}\n{}",
            sha256_hex(canonical_request.as_bytes())
        );
        let key = signing_key(&creds.secret_access_key, &datestamp, &cfg.region, SERVICE);
        let signature = hex(&hmac_sha256(&key, string_to_sign.as_bytes()));
        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
            creds.access_key_id
        );

        let mut req = self
            .agent
            .post(&url)
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .header("x-amz-date", &amzdate)
            .header("x-amz-content-sha256", &payload_hash)
            .header("authorization", &authorization);
        if let Some(token) = &creds.session_token {
            req = req.header("x-amz-security-token", token);
        }

        let mut resp = req
            .send(body.as_slice())
            .map_err(|e| MenteError::Storage(format!("bedrock invoke failed: {e}")))?;
        resp.body_mut()
            .read_json()
            .map_err(|e| MenteError::Storage(format!("bedrock response parse failed: {e}")))
    }
}

#[derive(serde::Deserialize)]
struct TitanResponse {
    embedding: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct CohereResponse {
    embeddings: Vec<Vec<f32>>,
}

impl EmbeddingProvider for BedrockEmbeddingProvider {
    fn embed(&self, text: &str) -> MenteResult<Vec<f32>> {
        self.invoke(&[text])?
            .pop()
            .ok_or_else(|| MenteError::Storage("bedrock returned no embedding".to_string()))
    }

    fn embed_batch(&self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        let per_call = match self.config.family {
            BedrockEmbeddingFamily::Titan => 1,
            BedrockEmbeddingFamily::Cohere => COHERE_MAX_BATCH,
        };
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(per_call) {
            out.extend(self.invoke(chunk)?);
        }
        Ok(out)
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn model_name(&self) -> &str {
        &self.config.model_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the SigV4 signing chain against AWS's published worked example
    /// (docs: "Examples of the complete Signature Version 4 signing process").
    /// This proves the HMAC key derivation and final signature are correct
    /// without needing a live AWS call.
    #[test]
    fn sigv4_matches_aws_reference_vector() {
        let secret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
        let datestamp = "20150830";
        let region = "us-east-1";
        let service = "iam";
        let string_to_sign = "AWS4-HMAC-SHA256\n20150830T123600Z\n\
             20150830/us-east-1/iam/aws4_request\n\
             f536975d06c0309214f805bb90ccff089219ecd68b2577efef23edd43b7e1a59";
        let key = signing_key(secret, datestamp, region, service);
        let signature = hex(&hmac_sha256(&key, string_to_sign.as_bytes()));
        assert_eq!(
            signature,
            "5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7"
        );
    }

    #[test]
    fn model_id_path_segment_is_percent_encoded() {
        // The ':' in a Bedrock model id must be encoded so the request URL and
        // the signed canonical URI match.
        assert_eq!(
            uri_encode_segment("amazon.titan-embed-text-v2:0"),
            "amazon.titan-embed-text-v2%3A0"
        );
    }

    #[test]
    fn model_family_follows_the_provider_prefix() {
        use BedrockEmbeddingFamily::*;
        assert_eq!(
            BedrockEmbeddingFamily::for_model("amazon.titan-embed-text-v2:0").unwrap(),
            Titan
        );
        assert_eq!(
            BedrockEmbeddingFamily::for_model("cohere.embed-english-v3").unwrap(),
            Cohere
        );
        assert_eq!(
            BedrockEmbeddingFamily::for_model("us.cohere.embed-english-v3").unwrap(),
            Cohere
        );
        assert!(BedrockEmbeddingFamily::for_model("anthropic.claude-haiku-5-5").is_err());
    }

    #[test]
    fn sha256_hex_of_empty_is_known_constant() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// Live end-to-end call against real Bedrock. Ignored by default (needs AWS
    /// credentials in the environment). Run with:
    ///   AWS_REGION=us-east-1 cargo test -p mentedb-embedding --features bedrock \
    ///     live_titan_embed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn live_titan_embed() {
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
        let cfg = BedrockEmbeddingConfig::titan_v2(region).expect("credentials from env");
        let provider = BedrockEmbeddingProvider::new(cfg).expect("provider");
        let v = provider.embed("the sky is blue").expect("live embed");
        assert_eq!(v.len(), 1024);
        assert!(v.iter().any(|&x| x != 0.0), "embedding should be non-zero");
    }

    /// Live Cohere batch call. Ignored by default, like `live_titan_embed`.
    #[test]
    #[ignore]
    fn live_cohere_embed_batch() {
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
        let cfg = BedrockEmbeddingConfig::cohere_english_v3(region).expect("credentials");
        let provider = BedrockEmbeddingProvider::new(cfg).expect("provider");
        let texts: Vec<String> = (0..100).map(|i| format!("memory number {i}")).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let v = provider.embed_batch(&refs).expect("live batch");
        assert_eq!(v.len(), 100, "100 texts span two calls of at most 96");
        assert!(v.iter().all(|e| e.len() == 1024));
    }
}
