use crate::config::{ExtractionConfig, LlmProvider};
use crate::error::ExtractionError;

/// AWS Bedrock SigV4 signing for the Anthropic Messages API on Bedrock.
///
/// This ports the hand-rolled SigV4 signer from `mentedb-embedding`'s
/// `bedrock_provider` (which signs a synchronous `ureq` call) to produce the
/// headers for an asynchronous `reqwest` request. Credentials are reused from
/// `mentedb_embedding::AwsCredentials`. Only compiled with the `bedrock`
/// feature.
#[cfg(feature = "bedrock")]
mod bedrock_sig {
    use hmac::{Hmac, Mac};
    use mentedb_embedding::AwsCredentials;
    use sha2::{Digest, Sha256};

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
    /// encoded). The request URL and the signed canonical URI must match.
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

    /// A fully signed Bedrock request, ready to hand to `reqwest`.
    pub(super) struct SignedRequest {
        pub url: String,
        pub body: Vec<u8>,
        /// Header (name, value) pairs to set on the request. Includes
        /// `Authorization`, `X-Amz-Date`, `X-Amz-Content-Sha256`, and
        /// `X-Amz-Security-Token` when a session token is present.
        pub headers: Vec<(&'static str, String)>,
    }

    /// Build and SigV4-sign a Bedrock InvokeModel request for the given region,
    /// model, and JSON body. `amzdate`/`datestamp` are passed in (rather than
    /// read from the clock) so this is deterministically testable; the live
    /// caller passes the current UTC time.
    pub(super) fn build_signed_request(
        region: &str,
        model: &str,
        body: Vec<u8>,
        creds: &AwsCredentials,
        amzdate: &str,
        datestamp: &str,
    ) -> SignedRequest {
        let host = format!("bedrock-runtime.{region}.amazonaws.com");
        // Sign the percent-encoded path; send the raw path. AWS re-encodes the
        // received path the same way, so the signatures match (this is what the
        // AWS SDKs do for model ids containing ':').
        let canonical_uri = format!("/model/{}/invoke", uri_encode_segment(model));
        let url = format!("https://{host}/model/{model}/invoke");

        let payload_hash = sha256_hex(&body);

        let mut signed: Vec<(String, String)> = vec![
            ("host".to_string(), host.clone()),
            ("x-amz-content-sha256".to_string(), payload_hash.clone()),
            ("x-amz-date".to_string(), amzdate.to_string()),
        ];
        if let Some(token) = &creds.session_token {
            signed.push(("x-amz-security-token".to_string(), token.clone()));
        }
        signed.sort_by(|a, b| a.0.cmp(&b.0));
        let canonical_headers: String = signed.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
        let signed_headers = signed
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>()
            .join(";");

        let canonical_request = format!(
            "POST\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        );
        let scope = format!("{datestamp}/{region}/{SERVICE}/aws4_request");
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amzdate}\n{scope}\n{}",
            sha256_hex(canonical_request.as_bytes())
        );
        let key = signing_key(&creds.secret_access_key, datestamp, region, SERVICE);
        let signature = hex(&hmac_sha256(&key, string_to_sign.as_bytes()));
        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
            creds.access_key_id
        );

        let mut headers: Vec<(&'static str, String)> = vec![
            ("Authorization", authorization),
            ("X-Amz-Date", amzdate.to_string()),
            ("X-Amz-Content-Sha256", payload_hash),
        ];
        if let Some(token) = &creds.session_token {
            headers.push(("X-Amz-Security-Token", token.clone()));
        }

        SignedRequest { url, body, headers }
    }
}

/// Classify an HTTP error response into a specific ExtractionError variant.
fn classify_api_error(
    status: reqwest::StatusCode,
    body: &str,
    provider: &str,
    model: &str,
) -> ExtractionError {
    let code = status.as_u16();
    match code {
        401 => ExtractionError::AuthError(format!(
            "{provider} returned 401 Unauthorized. Check your API key (MENTEDB_LLM_API_KEY). \
             Current provider: {provider}, model: {model}"
        )),
        403 => ExtractionError::AuthError(format!(
            "{provider} returned 403 Forbidden. Your API key may lack permissions for model '{model}'."
        )),
        404 => ExtractionError::ModelNotFound(format!(
            "{provider} returned 404. Model '{model}' may not exist or is not available on your account."
        )),
        _ => ExtractionError::ProviderError(format!("{provider} API returned {status}: {body}")),
    }
}

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
        // Ollama needs no auth; Bedrock authenticates with AWS credentials from
        // the environment (verified below), not an api_key. Every other
        // provider requires an api_key.
        let needs_api_key = !matches!(config.provider, LlmProvider::Ollama | LlmProvider::Bedrock);
        if needs_api_key && config.api_key.is_none() {
            return Err(ExtractionError::ConfigError(
                "API key is required for this provider".to_string(),
            ));
        }
        if config.provider == LlmProvider::Bedrock {
            #[cfg(feature = "bedrock")]
            {
                // Fail fast with a clear message if AWS creds are missing,
                // rather than at the first extraction call.
                mentedb_embedding::AwsCredentials::from_env().map_err(|e| {
                    ExtractionError::ConfigError(format!(
                        "Bedrock requires AWS credentials in the environment \
                         (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, and \
                         AWS_SESSION_TOKEN for temporary/SSO credentials): {e}"
                    ))
                })?;
            }
            #[cfg(not(feature = "bedrock"))]
            {
                return Err(ExtractionError::ConfigError(
                    "bedrock support not compiled in (build with --features bedrock)".to_string(),
                ));
            }
        }
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| ExtractionError::ConfigError(format!("HTTP client error: {}", e)))?;
        Ok(Self { client, config })
    }

    /// Expand a search query into multiple sub-queries via LLM.
    ///
    /// Given a natural language question, identifies the expected answer type
    /// and extracts 2-3 targeted search queries. The first line of the response
    /// is the answer type (PLACE, DATE, NUMBER, NAME, PERSON, BRAND, etc.),
    /// followed by the search queries.
    ///
    /// For counting/aggregation/comparison queries, also generates comprehensive
    /// category synonyms for exhaustive BM25 sweep.
    pub async fn expand_query(&self, query: &str) -> Result<Vec<String>, ExtractionError> {
        let system_prompt = "You help search a memory database. Given a question, return a JSON object with:\n\
            - \"answer_type\": one of PLACE, DATE, TIME, NUMBER, NAME, PERSON, BRAND, ITEM, ACTIVITY, COUNTING, OTHER\n\
            - \"queries\": array of 2-3 short search queries\n\
            - For COUNTING only, also include:\n\
              - \"item_keywords\": comma-separated specific subtypes/instances that would be individually counted\n\
              - \"broad_keywords\": comma-separated category terms, action verbs, and general synonyms\n\n\
            Use COUNTING when the question requires COMPLETENESS — counting, listing, aggregating, totaling, \
            or comparing to find a superlative (most, least, best, worst, first, last, biggest, highest, lowest).\n\n\
            The distinction matters:\n\
            - item_keywords: specific things you would COUNT (types of the thing being asked about)\n\
            - broad_keywords: general terms that help FIND memories but aren't counted themselves\n\n\
            Examples:\n\
            Q: \"Where do I take yoga classes?\"\n\
            {\"answer_type\": \"PLACE\", \"queries\": [\"yoga studio name\", \"yoga class location\"]}\n\n\
            Q: \"How many doctors did I visit?\"\n\
            {\"answer_type\": \"COUNTING\", \"queries\": [\"doctor visits appointments\", \"medical specialist visits\"], \
            \"item_keywords\": \"doctor, Dr., physician, specialist, dermatologist, cardiologist, dentist, surgeon, pediatrician, orthopedist, ophthalmologist\", \
            \"broad_keywords\": \"medical, clinic, appointment, visit, diagnosed, prescribed, referred, checkup, exam\"}\n\n\
            Q: \"Which platform did I gain the most followers on?\"\n\
            {\"answer_type\": \"COUNTING\", \"queries\": [\"social media follower growth\", \"follower count increase\"], \
            \"item_keywords\": \"TikTok, Instagram, Twitter, YouTube, Facebook, LinkedIn, Snapchat, Reddit, Twitch\", \
            \"broad_keywords\": \"followers, follower count, gained, growth, platform, social media, increase, jumped, grew\"}";
        let result = self.call_with_retry(query, system_prompt).await?;

        // Parse JSON response (call_openai forces json_object response format)
        let mut lines: Vec<String> = Vec::new();
        let cleaned = result
            .trim()
            .trim_start_matches("```json")
            .trim_end_matches("```")
            .trim();
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(cleaned) {
            if let Some(answer_type) = json.get("answer_type").and_then(|v| v.as_str()) {
                lines.push(answer_type.to_string());
            }
            if let Some(queries) = json.get("queries").and_then(|v| v.as_array()) {
                for q in queries {
                    if let Some(s) = q.as_str() {
                        lines.push(s.to_string());
                    }
                }
            }
            if let Some(item_kw) = json.get("item_keywords").and_then(|v| v.as_str()) {
                lines.push(format!("ITEM_KEYWORDS: {}", item_kw));
            }
            if let Some(broad_kw) = json.get("broad_keywords").and_then(|v| v.as_str()) {
                lines.push(format!("BROAD_KEYWORDS: {}", broad_kw));
            }
            // Fallback: old single "keywords" field → treat all as item keywords
            if let Some(keywords) = json.get("keywords").and_then(|v| v.as_str())
                && json.get("item_keywords").is_none()
            {
                lines.push(format!("ITEM_KEYWORDS: {}", keywords));
            }
        } else {
            // Fallback: parse as plain text lines
            lines = result
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
        }
        if std::env::var("MENTEDB_DEBUG").is_ok() {
            eprintln!("[expand_query] input={:?} parsed={:?}", query, lines);
        }
        Ok(lines)
    }

    async fn call_openai(
        &self,
        conversation: &str,
        system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        let body = serde_json::json!({
            "model": self.config.model,
            "temperature": 0,
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
            return Err(classify_api_error(
                status,
                &text,
                "OpenAI",
                &self.config.model,
            ));
        }

        let parsed: serde_json::Value = serde_json::from_str(&text)?;
        parsed["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| {
                ExtractionError::ParseError("Missing content in OpenAI response".to_string())
            })
    }

    /// OpenAI call without forced JSON response format.
    /// Used for plain text outputs (synthesis, re-ranking, key noun extraction).
    async fn call_openai_text(
        &self,
        conversation: &str,
        system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        let body = serde_json::json!({
            "model": self.config.model,
            "temperature": 0,
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
            return Err(classify_api_error(
                status,
                &text,
                "OpenAI",
                &self.config.model,
            ));
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
            "temperature": 0,
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
            return Err(classify_api_error(
                status,
                &text,
                "Anthropic",
                &self.config.model,
            ));
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
                parsed["content"][0]["text"].as_str().map(|s| s.to_string())
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

    /// Call AWS Bedrock's Anthropic Messages API, signed with SigV4.
    ///
    /// The endpoint is built from `config.region` and `config.model`; the body
    /// uses the Bedrock Anthropic Messages format. Credentials are read from the
    /// AWS environment via `mentedb_embedding::AwsCredentials`. Compiled only
    /// with the `bedrock` feature; otherwise returns a clear ConfigError.
    #[cfg(feature = "bedrock")]
    async fn call_bedrock(
        &self,
        conversation: &str,
        system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        let region = self
            .config
            .region
            .clone()
            .unwrap_or_else(crate::config::default_bedrock_region);

        let body_json = serde_json::json!({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [
                { "role": "user", "content": conversation }
            ]
        });
        let body = serde_json::to_vec(&body_json)?;

        let creds = mentedb_embedding::AwsCredentials::from_env().map_err(|e| {
            ExtractionError::ConfigError(format!(
                "Bedrock requires AWS credentials in the environment \
                 (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, and AWS_SESSION_TOKEN \
                 for temporary/SSO credentials): {e}"
            ))
        })?;

        let now = chrono::Utc::now();
        let amzdate = now.format("%Y%m%dT%H%M%SZ").to_string();
        let datestamp = now.format("%Y%m%d").to_string();

        let signed = bedrock_sig::build_signed_request(
            &region,
            &self.config.model,
            body,
            &creds,
            &amzdate,
            &datestamp,
        );

        let mut req = self
            .client
            .post(&signed.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");
        for (name, value) in &signed.headers {
            req = req.header(*name, value);
        }

        let resp = req.body(signed.body).send().await?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(classify_api_error(
                status,
                &text,
                "Bedrock",
                &self.config.model,
            ));
        }

        let parsed: serde_json::Value = serde_json::from_str(&text)?;

        // Bedrock returns the Anthropic content-block shape: concatenate all
        // text blocks (mirrors call_anthropic's fallback behavior on empty).
        let content_text: String = parsed["content"]
            .as_array()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter(|block| block["type"].as_str() == Some("text"))
                    .filter_map(|block| block["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();

        if content_text.trim().is_empty() {
            tracing::warn!(
                model = %self.config.model,
                response_preview = &text[..text.len().min(300)],
                "No text block found in Bedrock response"
            );
            return Ok("{\"memories\": []}".to_string());
        }
        Ok(content_text)
    }

    /// Feature-disabled stub: when built without `--features bedrock`, selecting
    /// the Bedrock provider fails with a clear, actionable message instead of a
    /// panic.
    #[cfg(not(feature = "bedrock"))]
    async fn call_bedrock(
        &self,
        _conversation: &str,
        _system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        Err(ExtractionError::ConfigError(
            "bedrock support not compiled in (build with --features bedrock)".to_string(),
        ))
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
            return Err(classify_api_error(
                status,
                &text,
                "Ollama",
                &self.config.model,
            ));
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
    pub async fn call_with_retry(
        &self,
        conversation: &str,
        system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        self.call_with_retry_inner(conversation, system_prompt, true)
            .await
    }

    /// Like call_with_retry but without forcing JSON response format.
    /// Use for prompts that expect plain text output (synthesis, re-ranking, etc).
    pub async fn call_text_with_retry(
        &self,
        conversation: &str,
        system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        self.call_with_retry_inner(conversation, system_prompt, false)
            .await
    }

    async fn call_with_retry_inner(
        &self,
        conversation: &str,
        system_prompt: &str,
        force_json: bool,
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
                    if force_json {
                        self.call_openai(conversation, system_prompt).await
                    } else {
                        self.call_openai_text(conversation, system_prompt).await
                    }
                }
                LlmProvider::Anthropic => self.call_anthropic(conversation, system_prompt).await,
                // Bedrock (Anthropic on Bedrock) handles both the JSON and text
                // paths with one method, like the native Anthropic provider.
                LlmProvider::Bedrock => self.call_bedrock(conversation, system_prompt).await,
                LlmProvider::Ollama => self.call_ollama(conversation, system_prompt).await,
            };

            match result {
                Ok(text) => {
                    tracing::info!(response_len = text.len(), "LLM extraction complete");
                    return Ok(text);
                }
                Err(ExtractionError::ProviderError(ref msg))
                    if msg.contains("429")
                        || msg.contains("500")
                        || msg.contains("502")
                        || msg.contains("503")
                        || msg.contains("529")
                        || msg.contains("timeout")
                        || msg.contains("connection")
                        || msg.contains("overloaded") =>
                {
                    tracing::warn!(attempt = attempt + 1, error = %msg, "retrying transient LLM error");
                    last_err = Some(result.unwrap_err());
                    continue;
                }
                Err(e) => {
                    tracing::error!(error = %e, "LLM extraction failed (non-retryable)");
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

#[cfg(all(test, feature = "bedrock"))]
mod bedrock_tests {
    use super::*;
    use mentedb_embedding::AwsCredentials;

    /// Build the exact JSON body call_bedrock sends, so the test and the real
    /// code stay in sync on the wire format.
    fn bedrock_body(system: &str, user: &str) -> Vec<u8> {
        let body_json = serde_json::json!({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": system,
            "messages": [
                { "role": "user", "content": user }
            ]
        });
        serde_json::to_vec(&body_json).unwrap()
    }

    /// Construct a signed Bedrock request with fixed credentials and timestamp
    /// (no network call) and assert: the URL is the region/model
    /// bedrock-runtime path, the body is valid Anthropic-Bedrock JSON carrying
    /// the system + user message, and an `Authorization: AWS4-HMAC-SHA256`
    /// header is produced.
    #[test]
    fn signed_bedrock_request_has_expected_url_body_and_auth() {
        let creds = AwsCredentials {
            access_key_id: "AKIDEXAMPLE".to_string(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: None,
        };
        let region = "us-east-1";
        let model = "us.anthropic.claude-haiku-4-5";
        let system = "You extract memories.";
        let user = "I switched my database to PostgreSQL.";
        let body = bedrock_body(system, user);

        let signed = bedrock_sig::build_signed_request(
            region,
            model,
            body,
            &creds,
            "20150830T123600Z",
            "20150830",
        );

        // URL is the region/model bedrock-runtime InvokeModel path (raw model id).
        assert_eq!(
            signed.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-haiku-4-5/invoke"
        );

        // Body is valid Anthropic-Bedrock JSON with system + user message.
        let parsed: serde_json::Value = serde_json::from_slice(&signed.body).unwrap();
        assert_eq!(parsed["anthropic_version"], "bedrock-2023-05-31");
        assert_eq!(parsed["max_tokens"], 4096);
        assert_eq!(parsed["system"], system);
        assert_eq!(parsed["messages"][0]["role"], "user");
        assert_eq!(parsed["messages"][0]["content"], user);

        // An AWS4-HMAC-SHA256 Authorization header is produced.
        let auth = signed
            .headers
            .iter()
            .find(|(k, _)| *k == "Authorization")
            .map(|(_, v)| v.as_str())
            .expect("Authorization header present");
        assert!(
            auth.starts_with("AWS4-HMAC-SHA256 "),
            "unexpected auth scheme: {auth}"
        );
        assert!(auth.contains("Credential=AKIDEXAMPLE/20150830/us-east-1/bedrock/aws4_request"));
        assert!(auth.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date"));
        assert!(auth.contains("Signature="));

        // X-Amz-Date is set; no security token header without a session token.
        assert!(
            signed
                .headers
                .iter()
                .any(|(k, v)| *k == "X-Amz-Date" && v == "20150830T123600Z")
        );
        assert!(
            !signed
                .headers
                .iter()
                .any(|(k, _)| *k == "X-Amz-Security-Token")
        );
    }

    /// Signing is deterministic for fixed inputs, and a session token adds the
    /// X-Amz-Security-Token header (and includes it in SignedHeaders).
    #[test]
    fn session_token_adds_security_token_header() {
        let creds = AwsCredentials {
            access_key_id: "AKIDEXAMPLE".to_string(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: Some("FQoGZQ-token".to_string()),
        };
        let signed = bedrock_sig::build_signed_request(
            "us-west-2",
            "us.anthropic.claude-sonnet-4-6",
            bedrock_body("sys", "usr"),
            &creds,
            "20150830T123600Z",
            "20150830",
        );

        let token = signed
            .headers
            .iter()
            .find(|(k, _)| *k == "X-Amz-Security-Token")
            .map(|(_, v)| v.as_str());
        assert_eq!(token, Some("FQoGZQ-token"));

        let auth = signed
            .headers
            .iter()
            .find(|(k, _)| *k == "Authorization")
            .map(|(_, v)| v.as_str())
            .unwrap();
        // The session token participates in the signed headers.
        assert!(auth.contains("x-amz-security-token"));
        // The region flows into the endpoint host.
        assert!(
            signed
                .url
                .starts_with("https://bedrock-runtime.us-west-2.amazonaws.com/")
        );
    }

    /// The config defaults for the Bedrock provider are the Claude-on-Bedrock
    /// model ids and an empty (region-derived) default URL.
    #[test]
    fn bedrock_config_defaults() {
        let cfg = ExtractionConfig::bedrock("eu-central-1");
        assert_eq!(cfg.provider, LlmProvider::Bedrock);
        assert!(cfg.api_key.is_none());
        assert_eq!(cfg.region.as_deref(), Some("eu-central-1"));
        assert_eq!(cfg.model, "us.anthropic.claude-haiku-4-5");
        assert_eq!(LlmProvider::Bedrock.default_url(), "");
        assert_eq!(
            LlmProvider::Bedrock.default_reader_model(),
            "us.anthropic.claude-sonnet-4-6"
        );
    }
}
