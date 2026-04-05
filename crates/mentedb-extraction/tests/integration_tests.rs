use mentedb_core::MemoryNode;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::AgentId;
use mentedb_embedding::HashEmbeddingProvider;
use mentedb_embedding::provider::EmbeddingProvider;
use mentedb_extraction::{
    ExtractedMemory, ExtractionConfig, ExtractionPipeline, ExtractionStats, MockExtractionProvider,
};

fn make_embedding_provider() -> HashEmbeddingProvider {
    HashEmbeddingProvider::new(128)
}

fn make_existing_memory(content: &str, provider: &HashEmbeddingProvider) -> MemoryNode {
    let embedding = provider.embed(content).unwrap();
    MemoryNode::new(
        AgentId::nil(),
        MemoryType::Semantic,
        content.to_string(),
        embedding,
    )
}

fn sample_conversation() -> &'static str {
    r#"User: We need to decide on a database for the new REST API project.
Assistant: What are your requirements? Do you need relational data or document storage?
User: Relational. We have complex joins and transactions. I was thinking MongoDB initially but that was wrong.
Assistant: For relational data with complex joins, PostgreSQL would be a great choice.
User: Yes, let's go with PostgreSQL 15. Also, I prefer Rust over Go for the backend because of memory safety.
Assistant: Good choice. What about the API URL structure?
User: All REST endpoints should follow the /api/v1/ prefix convention. The deadline is March 15, 2025.
Assistant: Noted. Any patterns to avoid?
User: Yes, using global mutable state for database connections caused race conditions in our tests last time. Never again.
User: Maybe we should also use Redis for caching? I'm not sure yet though."#
}

#[tokio::test]
async fn test_extraction_pipeline_with_mock() {
    let provider = MockExtractionProvider::with_realistic_response();
    let config = ExtractionConfig::default();
    let pipeline = ExtractionPipeline::new(provider, config);

    let memories = pipeline
        .extract_from_conversation(sample_conversation())
        .await
        .unwrap();

    assert!(!memories.is_empty(), "should extract at least one memory");
    assert!(memories.len() <= 50, "should respect max extraction limit");

    for memory in &memories {
        assert!(!memory.content.is_empty(), "content should not be empty");
        assert!(
            memory.confidence >= 0.0 && memory.confidence <= 1.0,
            "confidence should be between 0 and 1"
        );
        assert!(
            !memory.memory_type.is_empty(),
            "memory_type should not be empty"
        );
    }

    let types: Vec<&str> = memories.iter().map(|m| m.memory_type.as_str()).collect();
    assert!(
        types.contains(&"decision"),
        "should extract at least one decision"
    );
    assert!(
        types.contains(&"preference"),
        "should extract at least one preference"
    );
}

#[tokio::test]
async fn test_quality_filter() {
    let provider = MockExtractionProvider::with_realistic_response();
    let config = ExtractionConfig {
        quality_threshold: 0.6,
        ..ExtractionConfig::default()
    };
    let pipeline = ExtractionPipeline::new(provider, config);

    let memories = pipeline
        .extract_from_conversation(sample_conversation())
        .await
        .unwrap();

    let filtered = pipeline.filter_quality(&memories);
    let rejected: Vec<&ExtractedMemory> = memories.iter().filter(|m| m.confidence < 0.6).collect();

    assert!(
        filtered.len() < memories.len(),
        "filter should remove at least one low-confidence memory"
    );
    assert!(
        !rejected.is_empty(),
        "there should be at least one rejected memory"
    );
    for m in &filtered {
        assert!(
            m.confidence >= 0.6,
            "all filtered memories should be above threshold"
        );
    }
}

#[tokio::test]
async fn test_deduplication() {
    let emb = make_embedding_provider();
    let existing = vec![make_existing_memory(
        "The team decided to use PostgreSQL 15 as the primary database for the REST API project",
        &emb,
    )];

    let provider = MockExtractionProvider::with_realistic_response();
    let config = ExtractionConfig {
        deduplication_threshold: 0.85,
        enable_deduplication: true,
        ..ExtractionConfig::default()
    };
    let pipeline = ExtractionPipeline::new(provider, config);

    let memories = pipeline
        .extract_from_conversation(sample_conversation())
        .await
        .unwrap();

    // The first extracted memory has nearly identical content to the existing one
    let first = &memories[0];
    let is_dup = pipeline.check_duplicates(first, &existing, &emb);

    // With hash embeddings, identical text produces identical embeddings
    let exact_dup = ExtractedMemory {
        content:
            "The team decided to use PostgreSQL 15 as the primary database for the REST API project"
                .to_string(),
        memory_type: "decision".to_string(),
        confidence: 0.95,
        entities: vec![],
        tags: vec![],
        reasoning: "test".to_string(),
    };
    let is_exact_dup = pipeline.check_duplicates(&exact_dup, &existing, &emb);
    assert!(
        is_exact_dup,
        "exact same content should be detected as duplicate"
    );
}

#[tokio::test]
async fn test_contradiction_detection() {
    let emb = make_embedding_provider();

    // Existing memory: "we use PostgreSQL"
    let existing = vec![make_existing_memory(
        "we use PostgreSQL for everything",
        &emb,
    )];

    let provider = MockExtractionProvider::with_realistic_response();
    let config = ExtractionConfig {
        enable_contradiction_check: true,
        ..ExtractionConfig::default()
    };
    let pipeline = ExtractionPipeline::new(provider, config);

    // Create a memory that should be flagged as related/contradictory
    let correction = ExtractedMemory {
        content: "we use PostgreSQL for everything".to_string(),
        memory_type: "correction".to_string(),
        confidence: 0.9,
        entities: vec!["PostgreSQL".to_string()],
        tags: vec![],
        reasoning: "test contradiction".to_string(),
    };

    let findings = pipeline.check_contradictions(&correction, &existing, &emb);
    // With identical content, the WriteInferenceEngine should detect high similarity
    // and produce actions (obsolescence or contradiction depending on agent_id match)
    assert!(
        !findings.is_empty(),
        "should detect findings for highly similar content"
    );
}

#[tokio::test]
async fn test_prompt_output_parsing() {
    // Valid JSON
    let provider = MockExtractionProvider::new(r#"{"memories": []}"#);
    let config = ExtractionConfig::default();
    let pipeline = ExtractionPipeline::new(provider, config);
    let result = pipeline.extract_from_conversation("test").await.unwrap();
    assert!(result.is_empty());

    // JSON with markdown fences
    let provider = MockExtractionProvider::new(
        "```json\n{\"memories\": [{\"content\": \"test\", \"memory_type\": \"fact\", \"confidence\": 0.8}]}\n```",
    );
    let config = ExtractionConfig::default();
    let pipeline = ExtractionPipeline::new(provider, config);
    let result = pipeline.extract_from_conversation("test").await.unwrap();
    assert_eq!(result.len(), 1);

    // Missing optional fields use defaults
    let provider = MockExtractionProvider::new(
        r#"{"memories": [{"content": "bare minimum", "memory_type": "fact"}]}"#,
    );
    let config = ExtractionConfig::default();
    let pipeline = ExtractionPipeline::new(provider, config);
    let result = pipeline.extract_from_conversation("test").await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].confidence, 0.5); // default
    assert!(result[0].entities.is_empty()); // default
    assert!(result[0].tags.is_empty()); // default

    // Malformed JSON should return a parse error
    let provider = MockExtractionProvider::new("not json at all");
    let config = ExtractionConfig::default();
    let pipeline = ExtractionPipeline::new(provider, config);
    let result = pipeline.extract_from_conversation("test").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_extraction_stats() {
    let emb = make_embedding_provider();

    // Create an existing memory that will be an exact duplicate of one extraction
    let existing = vec![make_existing_memory(
        "The team decided to use PostgreSQL 15 as the primary database for the REST API project",
        &emb,
    )];

    let provider = MockExtractionProvider::with_realistic_response();
    let config = ExtractionConfig {
        quality_threshold: 0.6,
        deduplication_threshold: 0.85,
        enable_deduplication: true,
        enable_contradiction_check: true,
        ..ExtractionConfig::default()
    };
    let pipeline = ExtractionPipeline::new(provider, config);

    let result = pipeline
        .process(sample_conversation(), &existing, &emb)
        .await
        .unwrap();

    let stats = &result.stats;
    assert_eq!(stats.total_extracted, 7, "mock provides 7 memories");
    assert!(
        stats.rejected_quality >= 1,
        "at least one memory should be below quality threshold (the 0.3 confidence one)"
    );
    assert_eq!(
        stats.total_extracted,
        stats.accepted
            + stats.rejected_quality
            + stats.rejected_duplicate
            + stats.contradictions_found,
        "all memories should be accounted for in stats"
    );

    // Verify rejected low quality memories are present
    assert!(
        !result.rejected_low_quality.is_empty(),
        "should have rejected low quality memories"
    );
    for m in &result.rejected_low_quality {
        assert!(
            m.confidence < 0.6,
            "rejected memories should be below threshold"
        );
    }
}
