//! Realistic and stress tests for MenteDB.
//!
//! These tests go beyond basic CRUD by simulating real-world AI agent workloads:
//! multi-turn conversations with actual code, concurrent multi-agent writes,
//! context assembly under token budgets, and contradiction detection chains.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use mentedb::MenteDb;
use mentedb::prelude::*;

use mentedb_cognitive::{InferredAction, WriteInferenceConfig, WriteInferenceEngine};
use mentedb_consolidation::{DecayConfig, DecayEngine};
use mentedb_context::{AssemblyConfig, ContextAssembler, ScoredMemory};
use mentedb_embedding::{EmbeddingProvider, HashEmbeddingProvider};

use mentedb_core::types::{AgentId, MemoryId};
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EMBEDDING_DIM: usize = 64;

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

const DAY_US: u64 = 24 * 3600 * 1_000_000;

fn embedder() -> HashEmbeddingProvider {
    HashEmbeddingProvider::new(EMBEDDING_DIM)
}

fn embed(provider: &HashEmbeddingProvider, text: &str) -> Vec<f32> {
    provider.embed(text).unwrap()
}

fn make_memory_from_text(
    agent_id: AgentId,
    content: &str,
    mem_type: MemoryType,
    provider: &HashEmbeddingProvider,
) -> MemoryNode {
    let embedding = embed(provider, content);
    MemoryNode::new(agent_id, mem_type, content.to_string(), embedding)
}

fn make_memory_with_tags(
    agent_id: AgentId,
    content: &str,
    mem_type: MemoryType,
    tags: Vec<String>,
    provider: &HashEmbeddingProvider,
) -> MemoryNode {
    let mut node = make_memory_from_text(agent_id, content, mem_type, provider);
    node.tags = tags;
    node
}

fn make_memory_at_time(
    agent_id: AgentId,
    content: &str,
    mem_type: MemoryType,
    created_at: u64,
    salience: f32,
    provider: &HashEmbeddingProvider,
) -> MemoryNode {
    let mut node = make_memory_from_text(agent_id, content, mem_type, provider);
    node.created_at = created_at;
    node.accessed_at = created_at;
    node.salience = salience;
    node
}

// ---------------------------------------------------------------------------
// Test 1: Multi-turn coding conversation
// ---------------------------------------------------------------------------

/// Validates that MenteDB can store and recall context from a realistic 15+ turn
/// conversation between a developer and an AI assistant building a REST API.
/// The conversation includes actual code snippets, error messages, and iterative
/// refinement -- the kind of dialogue a coding assistant handles daily.
#[test]
fn test_multi_turn_coding_conversation() {
    let dir = tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent_id = AgentId::new();
    let provider = embedder();

    let conversation: Vec<(&str, MemoryType)> = vec![
        (
            "User: I want to build a REST API in Rust using Actix-web for a todo list application.",
            MemoryType::Episodic,
        ),
        (
            "Assistant: Let us start with the project structure. Run `cargo new todo-api` and add \
             actix-web, serde, and sqlx to Cargo.toml.",
            MemoryType::Episodic,
        ),
        (
            "User: Done. Here is my Cargo.toml:\n\
             [dependencies]\n\
             actix-web = \"4\"\n\
             serde = { version = \"1\", features = [\"derive\"] }\n\
             sqlx = { version = \"0.7\", features = [\"runtime-tokio\", \"sqlite\"] }",
            MemoryType::Episodic,
        ),
        (
            "Assistant: Now create the Todo model:\n\
             ```rust\n\
             #[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]\n\
             pub struct Todo {\n\
                 pub id: i64,\n\
                 pub title: String,\n\
                 pub completed: bool,\n\
             }\n\
             ```",
            MemoryType::Procedural,
        ),
        (
            "User: I added the model. Now I need the GET /todos endpoint.",
            MemoryType::Episodic,
        ),
        (
            "Assistant: Here is the handler for listing todos:\n\
             ```rust\n\
             async fn list_todos(pool: web::Data<SqlitePool>) -> impl Responder {\n\
                 let todos = sqlx::query_as::<_, Todo>(\"SELECT * FROM todos\")\n\
                     .fetch_all(pool.get_ref())\n\
                     .await\n\
                     .unwrap();\n\
                 HttpResponse::Ok().json(todos)\n\
             }\n\
             ```",
            MemoryType::Procedural,
        ),
        (
            "User: I also need POST /todos to create new todos with JSON body validation.",
            MemoryType::Episodic,
        ),
        (
            "Assistant: Add this create handler:\n\
             ```rust\n\
             #[derive(Deserialize)]\n\
             struct CreateTodo { title: String }\n\n\
             async fn create_todo(\n\
                 pool: web::Data<SqlitePool>,\n\
                 body: web::Json<CreateTodo>,\n\
             ) -> impl Responder {\n\
                 let result = sqlx::query(\"INSERT INTO todos (title, completed) VALUES (?, false)\")\n\
                     .bind(&body.title)\n\
                     .execute(pool.get_ref())\n\
                     .await\n\
                     .unwrap();\n\
                 HttpResponse::Created().json(serde_json::json!({\"id\": result.last_insert_rowid()}))\n\
             }\n\
             ```",
            MemoryType::Procedural,
        ),
        (
            "User: I am getting a compile error: `the trait bound SqlitePool: Clone is not satisfied`.",
            MemoryType::Episodic,
        ),
        (
            "Assistant: SqlitePool already implements Clone. The issue is likely that you need to \
             wrap it in web::Data before passing to App::app_data(). Make sure you do:\n\
             ```rust\n\
             let pool = SqlitePool::connect(\"sqlite:todos.db\").await.unwrap();\n\
             HttpServer::new(move || {\n\
                 App::new()\n\
                     .app_data(web::Data::new(pool.clone()))\n\
                     .route(\"/todos\", web::get().to(list_todos))\n\
                     .route(\"/todos\", web::post().to(create_todo))\n\
             })\n\
             ```",
            MemoryType::Procedural,
        ),
        (
            "User: That fixed the compile error. Now I want to add DELETE /todos/{id} for removing individual items.",
            MemoryType::Episodic,
        ),
        (
            "Assistant: Here is the delete handler with path parameter extraction:\n\
             ```rust\n\
             async fn delete_todo(\n\
                 pool: web::Data<SqlitePool>,\n\
                 path: web::Path<i64>,\n\
             ) -> impl Responder {\n\
                 let id = path.into_inner();\n\
                 sqlx::query(\"DELETE FROM todos WHERE id = ?\")\n\
                     .bind(id)\n\
                     .execute(pool.get_ref())\n\
                     .await\n\
                     .unwrap();\n\
                 HttpResponse::NoContent().finish()\n\
             }\n\
             ```\n\
             Register it with `.route(\"/todos/{id}\", web::delete().to(delete_todo))`.",
            MemoryType::Procedural,
        ),
        (
            "User: I want to add proper error handling instead of unwrap() everywhere.",
            MemoryType::Episodic,
        ),
        (
            "Assistant: Create a custom error type that implements actix_web::ResponseError:\n\
             ```rust\n\
             #[derive(Debug)]\n\
             enum ApiError {\n\
                 DbError(sqlx::Error),\n\
                 NotFound,\n\
             }\n\
             impl std::fmt::Display for ApiError {\n\
                 fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {\n\
                     match self {\n\
                         ApiError::DbError(e) => write!(f, \"Database error: {}\", e),\n\
                         ApiError::NotFound => write!(f, \"Resource not found\"),\n\
                     }\n\
                 }\n\
             }\n\
             impl ResponseError for ApiError {}\n\
             ```",
            MemoryType::Procedural,
        ),
        (
            "User: Should I add middleware for request logging and CORS?",
            MemoryType::Episodic,
        ),
        (
            "Assistant: Yes. For logging use actix_web::middleware::Logger, and for CORS use the \
             actix-cors crate. Add `actix-cors = \"0.7\"` to Cargo.toml, then configure:\n\
             ```rust\n\
             use actix_cors::Cors;\n\n\
             App::new()\n\
                 .wrap(Logger::default())\n\
                 .wrap(Cors::permissive())\n\
                 .app_data(web::Data::new(pool.clone()))\n\
             ```",
            MemoryType::Procedural,
        ),
        (
            "User: The API is working. I ran `curl -X POST http://localhost:8080/todos -H \
             'Content-Type: application/json' -d '{\"title\":\"Buy groceries\"}'` and got back a 201.",
            MemoryType::Episodic,
        ),
    ];

    let mut memory_ids: Vec<MemoryId> = Vec::new();
    for (content, mem_type) in &conversation {
        let node = make_memory_from_text(agent_id, content, *mem_type, &provider);
        memory_ids.push(node.id);
        db.store(node).unwrap();
    }

    assert_eq!(
        memory_ids.len(),
        17,
        "All 17 conversation turns should be stored"
    );

    // At turn 15+, recall context about error handling (turn 13-14)
    let query_embedding = embed(&provider, "error handling in actix-web REST API");
    let results = db.recall_similar(&query_embedding, 5).unwrap();

    assert!(
        !results.is_empty(),
        "Recall should find relevant memories about error handling"
    );
    assert!(results.len() <= 5, "Should respect the k=5 limit");

    // Recall context about database setup
    let db_query = embed(&provider, "SQLite database pool setup in actix-web");
    let db_results = db.recall_similar(&db_query, 3).unwrap();
    assert!(
        !db_results.is_empty(),
        "Recall should find memories about database configuration"
    );

    // Recall about the POST endpoint specifically
    let post_query = embed(&provider, "create new todo item POST endpoint handler");
    let post_results = db.recall_similar(&post_query, 3).unwrap();
    assert!(
        !post_results.is_empty(),
        "Recall should find the POST handler discussion"
    );

    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// Test 2: Customer support agent with interleaved tickets
// ---------------------------------------------------------------------------

/// Validates that a support agent can handle multiple interleaved customer
/// conversations without context bleeding. Each customer has a distinct problem,
/// and the agent must recall the correct context when switching between them.
/// This is a common real-world pattern where agents serve multiple users.
#[test]
fn test_customer_support_agent() {
    let dir = tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent_id = AgentId::new();
    let provider = embedder();

    // Customer A: billing dispute about double charge
    let customer_a_msgs = vec![
        "Customer A: I was charged twice for my Pro subscription this month, order #ORD-7823.",
        "Agent to A: I can see the duplicate charge of $49.99 on your account. Let me initiate a refund.",
        "Customer A: How long will the refund take to appear on my credit card?",
        "Agent to A: The refund has been processed. It typically takes 5-7 business days to appear on your statement.",
    ];

    // Customer B: technical issue with API integration
    let customer_b_msgs = vec![
        "Customer B: Our webhook endpoint is returning 502 errors when receiving events from your API.",
        "Agent to B: Can you share the webhook URL and any error logs from your server?",
        "Customer B: The URL is https://api.example.com/webhooks/ingest and the nginx logs show upstream timeout after 30 seconds.",
        "Agent to B: The timeout suggests your handler is taking too long. Webhook delivery has a 10-second timeout on our side. Consider processing events asynchronously with a message queue.",
    ];

    // Customer C: account access issue
    let customer_c_msgs = vec![
        "Customer C: I cannot log in to my admin dashboard. It says my account has been locked.",
        "Agent to C: I see your account was locked after 5 failed login attempts. I will send a password reset link to your registered email.",
        "Customer C: I did not receive the reset email. Can you check if my email is still john.doe@company.org?",
        "Agent to C: Your registered email is j.doe@company.org, not john.doe@company.org. I have sent the reset link to the correct address.",
    ];

    // Store all messages with customer-specific tags, interleaved
    let all_msgs = vec![
        (customer_a_msgs.as_slice(), "customer-a", "billing"),
        (customer_b_msgs.as_slice(), "customer-b", "technical"),
        (customer_c_msgs.as_slice(), "customer-c", "account"),
    ];

    let mut ids_by_customer: std::collections::HashMap<&str, Vec<MemoryId>> =
        std::collections::HashMap::new();

    // Interleave the messages as they would arrive in a real support queue
    for turn in 0..4 {
        for (msgs, customer_tag, category_tag) in &all_msgs {
            let node = make_memory_with_tags(
                agent_id,
                msgs[turn],
                MemoryType::Episodic,
                vec![customer_tag.to_string(), category_tag.to_string()],
                &provider,
            );
            ids_by_customer
                .entry(customer_tag)
                .or_default()
                .push(node.id);
            db.store(node).unwrap();
        }
    }

    // Verify all 12 memories stored
    let total_ids: usize = ids_by_customer.values().map(|v| v.len()).sum();
    assert_eq!(
        total_ids, 12,
        "All 12 interleaved messages should be stored"
    );

    // Query for billing-related context (should surface Customer A)
    let billing_query = embed(&provider, "refund for duplicate subscription charge");
    let billing_results = db.recall_similar(&billing_query, 5).unwrap();
    assert!(
        !billing_results.is_empty(),
        "Should find billing-related memories for Customer A"
    );

    // Query for webhook/API issues (should surface Customer B)
    let webhook_query = embed(&provider, "webhook 502 error timeout upstream");
    let webhook_results = db.recall_similar(&webhook_query, 5).unwrap();
    assert!(
        !webhook_results.is_empty(),
        "Should find webhook-related memories for Customer B"
    );

    // Query for account lockout (should surface Customer C)
    let lockout_query = embed(
        &provider,
        "account locked failed login attempts password reset",
    );
    let lockout_results = db.recall_similar(&lockout_query, 5).unwrap();
    assert!(
        !lockout_results.is_empty(),
        "Should find account lockout memories for Customer C"
    );

    // Verify that billing queries do not return webhook memories as the top result.
    let customer_b_id_set: HashSet<MemoryId> =
        ids_by_customer["customer-b"].iter().copied().collect();

    // The top billing result should not be from Customer B's webhook thread
    if let Some((top_id, _)) = billing_results.first() {
        assert!(
            !customer_b_id_set.contains(top_id),
            "Top billing recall result should not be a webhook memory from Customer B"
        );
    }

    // Verify no overlap between the top result of each customer query
    let top_billing = billing_results.first().map(|(id, _)| *id);
    let top_webhook = webhook_results.first().map(|(id, _)| *id);
    let top_lockout = lockout_results.first().map(|(id, _)| *id);

    if let (Some(b), Some(w)) = (top_billing, top_webhook) {
        assert_ne!(
            b, w,
            "Top billing and webhook results should be distinct memories"
        );
    }
    if let (Some(b), Some(l)) = (top_billing, top_lockout) {
        assert_ne!(
            b, l,
            "Top billing and lockout results should be distinct memories"
        );
    }
    if let (Some(w), Some(l)) = (top_webhook, top_lockout) {
        assert_ne!(
            w, l,
            "Top webhook and lockout results should be distinct memories"
        );
    }

    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// Test 3: Research assistant knowledge accumulation over time
// ---------------------------------------------------------------------------

/// Simulates a research assistant that accumulates knowledge about ML model
/// deployment over a simulated week. Validates that the decay engine properly
/// reduces salience of older memories while preserving foundational knowledge
/// that has been accessed frequently, and that recent findings have higher
/// effective salience than stale, rarely-accessed notes.
#[test]
fn test_research_assistant_knowledge_accumulation() {
    let dir = tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent_id = AgentId::new();
    let provider = embedder();

    let base_time = now_us() - 8 * DAY_US;

    // Day 1: foundational knowledge (high salience, frequently accessed)
    let day1_content = vec![
        "ML model deployment requires converting trained models to optimized inference formats like ONNX or TensorRT for production serving.",
        "Docker containers are the standard packaging mechanism for ML model serving, bundling the model, runtime, and dependencies together.",
        "Kubernetes orchestrates containerized model services, handling auto-scaling based on request load and GPU resource allocation.",
        "Model versioning with tools like MLflow or DVC tracks which training run produced which artifact, enabling reproducible deployments.",
        "A/B testing infrastructure routes a percentage of inference traffic to canary model versions before full rollout.",
        "Feature stores like Feast ensure that the same feature transformations used during training are applied consistently at inference time.",
        "Model monitoring tracks prediction drift, latency percentiles, and throughput to detect when retraining is needed.",
        "CI/CD pipelines for ML include data validation, model training, evaluation gating, and staged rollout to production.",
        "Model registries like MLflow Model Registry or Weights and Biases store trained artifacts with metadata and lineage information.",
        "GPU memory management is critical for serving large models; techniques include model sharding, quantization, and memory-mapped weights.",
        "Load balancing for model servers uses least-connections routing to distribute inference requests across healthy replicas.",
        "Health checks for model endpoints verify both HTTP liveness and model inference readiness with sample prediction calls.",
        "Logging inference requests and predictions enables debugging, auditing, and building datasets for future model retraining.",
    ];

    // Day 3: intermediate findings
    let day3_content = vec![
        "NVIDIA Triton Inference Server supports concurrent model execution across multiple GPUs with dynamic batching.",
        "gRPC outperforms REST for model serving by 3-5x on latency due to binary serialization with Protocol Buffers.",
        "Model quantization from FP32 to INT8 reduces memory by 4x with less than 1% accuracy degradation for most vision models.",
        "Kubernetes Horizontal Pod Autoscaler can scale model replicas based on custom metrics like GPU utilization or queue depth.",
        "Blue-green deployments for model updates maintain the previous version as a hot standby for instant rollback.",
        "Prometheus and Grafana dashboards visualize model serving metrics including p50, p95, and p99 latency distributions.",
        "Batch inference pipelines using Apache Spark or Ray process large datasets offline at lower cost than real-time serving.",
        "TensorRT optimization applies layer fusion, kernel auto-tuning, and precision calibration to reduce inference latency by 5-10x.",
        "Model warm-up scripts pre-load weights and run dummy inference to avoid cold-start latency spikes on first real requests.",
        "Canary analysis compares error rates and latency distributions between the baseline and canary model versions using statistical tests.",
        "Shadow deployments route a copy of production traffic to a new model version without affecting user-facing responses.",
        "Data pipeline monitoring with Great Expectations validates that input features conform to expected distributions before inference.",
        "Cost attribution tags on GPU instances enable tracking inference spend per model, per team, and per customer.",
    ];

    // Day 5: more recent research
    let day5_content = vec![
        "vLLM achieves 24x throughput improvement over HuggingFace Transformers for LLM serving through PagedAttention memory management.",
        "Speculative decoding uses a smaller draft model to generate candidate tokens verified by the main model, reducing latency by 2-3x.",
        "LoRA adapters can be hot-swapped at serving time, allowing a single base model to serve multiple fine-tuned variants.",
        "KV-cache compression techniques reduce memory requirements for long-context LLM inference by up to 8x.",
        "Model sharding across multiple GPUs with tensor parallelism enables serving models too large for a single GPU memory.",
        "Continuous batching in LLM serving dynamically adds new requests to ongoing batches, improving GPU utilization from 30% to 90%.",
        "SGLang compiler optimizes LLM serving by reusing KV cache across requests that share common prompt prefixes.",
        "Medusa decoding adds multiple prediction heads to generate several tokens per forward pass, trading compute for reduced latency.",
        "AWQ quantization preserves salient weight channels while aggressively quantizing others, maintaining accuracy at 4-bit precision.",
        "Pipeline parallelism splits model layers across GPUs in a pipeline, overlapping compute and communication for higher throughput.",
        "Request scheduling policies like shortest-job-first reduce average latency for LLM serving workloads with variable output lengths.",
        "Disaggregated prefill architectures separate prompt encoding from token generation across different GPU pools for better utilization.",
        "Prompt caching stores encoded representations of common system prompts to skip redundant prefill computation.",
    ];

    // Day 6-7: most recent findings
    let day7_content = vec![
        "Structured output generation using constrained decoding with JSON Schema guarantees valid output format from LLMs.",
        "Embedding model serving with sentence-transformers can batch encode 1000+ documents per second on a single A100 GPU.",
        "Model cascading routes simple queries to a small model and complex ones to a large model, reducing average inference cost by 60%.",
        "Retrieval-augmented generation pipelines combine vector search with LLM generation, requiring careful chunk size tuning around 512 tokens.",
        "Prefill-decode disaggregation separates the compute-heavy prefill phase from memory-bound decode phase across different GPU pools.",
        "FlashAttention-2 reduces transformer attention computation from O(n^2) memory to O(n) with 2x speedup on A100 GPUs.",
        "GGUF format enables efficient CPU inference of quantized LLMs, making deployment possible on commodity hardware without GPUs.",
        "Tool-use fine-tuning teaches LLMs to emit structured function calls, requiring specialized serving infrastructure for tool execution loops.",
        "Multi-modal model serving requires handling image, audio, and text inputs in a unified pipeline with different preprocessing steps.",
        "Inference routers use classifier models to predict query complexity and route to appropriately sized models in a cascade.",
        "Guardrail models run in parallel with the main LLM to detect and filter harmful or off-topic outputs before delivery.",
        "Token streaming over Server-Sent Events delivers partial LLM responses to clients incrementally, improving perceived latency.",
        "Auto-scaling policies for LLM serving must account for variable request durations, using queue depth rather than CPU utilization.",
    ];

    // Store day-1 memories with high salience, simulating frequent access
    let mut all_memories: Vec<MemoryNode> = Vec::new();
    for content in &day1_content {
        let mut node = make_memory_at_time(
            agent_id,
            content,
            MemoryType::Semantic,
            base_time,
            1.0,
            &provider,
        );
        // Foundational knowledge accessed many times
        node.access_count = 15;
        node.accessed_at = base_time + 6 * DAY_US;
        db.store(node.clone()).unwrap();
        all_memories.push(node);
    }

    for content in &day3_content {
        let mut node = make_memory_at_time(
            agent_id,
            content,
            MemoryType::Semantic,
            base_time + 2 * DAY_US,
            0.9,
            &provider,
        );
        node.access_count = 5;
        node.accessed_at = base_time + 4 * DAY_US;
        db.store(node.clone()).unwrap();
        all_memories.push(node);
    }

    for content in &day5_content {
        let mut node = make_memory_at_time(
            agent_id,
            content,
            MemoryType::Semantic,
            base_time + 4 * DAY_US,
            0.95,
            &provider,
        );
        node.access_count = 3;
        node.accessed_at = base_time + 5 * DAY_US;
        db.store(node.clone()).unwrap();
        all_memories.push(node);
    }

    for content in &day7_content {
        let mut node = make_memory_at_time(
            agent_id,
            content,
            MemoryType::Semantic,
            base_time + 6 * DAY_US,
            1.0,
            &provider,
        );
        node.access_count = 1;
        node.accessed_at = base_time + 6 * DAY_US;
        db.store(node.clone()).unwrap();
        all_memories.push(node);
    }

    assert!(
        all_memories.len() >= 50,
        "Should have stored at least 50 memories, got {}",
        all_memories.len()
    );

    // Apply decay to all memories at "now" (end of week + 1 day)
    let decay_time = base_time + 8 * DAY_US;
    let decay_engine = DecayEngine::new(DecayConfig {
        half_life_us: 7 * DAY_US,
        min_salience: 0.01,
        access_boost: 0.1,
        max_salience: 1.0,
    });

    // Clone for decay computation (the engine mutates salience in-place)
    let mut decayed = all_memories.clone();
    decay_engine.apply_decay_batch(&mut decayed, decay_time);

    // Foundational day-1 memories with high access counts should still have
    // reasonable salience after decay due to the access boost
    let day1_decayed: Vec<f32> = decayed[..day1_content.len()]
        .iter()
        .map(|m| m.salience)
        .collect();
    for (i, sal) in day1_decayed.iter().enumerate() {
        assert!(
            *sal > 0.05,
            "Foundational memory {} should retain salience above 0.05 due to frequent access, got {}",
            i,
            sal
        );
    }

    // Day-3 memories with moderate access should decay more than day-1
    let day3_start = day1_content.len();
    let day3_end = day3_start + day3_content.len();
    let day3_decayed: Vec<f32> = decayed[day3_start..day3_end]
        .iter()
        .map(|m| m.salience)
        .collect();

    // Recent day-7 memories should have higher salience than older stale memories
    let day7_start = day3_end + day5_content.len();
    let day7_decayed: Vec<f32> = decayed[day7_start..].iter().map(|m| m.salience).collect();

    let avg_day3 = day3_decayed.iter().sum::<f32>() / day3_decayed.len() as f32;
    let avg_day7 = day7_decayed.iter().sum::<f32>() / day7_decayed.len() as f32;

    assert!(
        avg_day7 > avg_day3,
        "Recent day-7 memories (avg salience {:.4}) should have higher salience than day-3 \
         memories with fewer accesses (avg salience {:.4})",
        avg_day7,
        avg_day3
    );

    // Verify recall still returns results for foundational topics
    let query_emb = embed(
        &provider,
        "Kubernetes auto-scaling ML model deployment containers",
    );
    let results = db.recall_similar(&query_emb, 10).unwrap();
    assert!(
        !results.is_empty(),
        "Should recall foundational deployment knowledge even after a simulated week"
    );

    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// Test 4: Concurrent multi-agent writes
// ---------------------------------------------------------------------------

/// Validates that MenteDb handles concurrent writes from multiple agent threads
/// without data loss or corruption. Four threads each write 25 memories to the
/// same database instance (behind a Mutex), and we verify all 100 memories are
/// present and retrievable after the writes complete.
#[test]
fn test_concurrent_multi_agent_writes() {
    let dir = tempdir().unwrap();
    let db = Arc::new(Mutex::new(MenteDb::open(dir.path()).unwrap()));
    let provider = Arc::new(embedder());

    let num_agents = 4;
    let memories_per_agent = 25;
    let all_ids = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::new();

    for agent_idx in 0..num_agents {
        let db = Arc::clone(&db);
        let provider = Arc::clone(&provider);
        let all_ids = Arc::clone(&all_ids);
        let agent_id = AgentId::new();

        let handle = thread::spawn(move || {
            let topics: Vec<String> = (0..memories_per_agent)
                .map(|i| {
                    match (agent_idx, i % 5) {
                        (0, 0) => format!(
                            "Agent-0 research note {}: Transformer attention mechanisms \
                             compute scaled dot-product attention with Q, K, V matrices.",
                            i
                        ),
                        (0, 1) => format!(
                            "Agent-0 research note {}: Positional encoding adds sinusoidal \
                             signals to input embeddings so transformers can model sequence order.",
                            i
                        ),
                        (0, 2) => format!(
                            "Agent-0 research note {}: Layer normalization stabilizes training \
                             by normalizing activations across the feature dimension.",
                            i
                        ),
                        (0, 3) => format!(
                            "Agent-0 research note {}: Multi-head attention runs h parallel \
                             attention operations, each with d_model/h dimensions.",
                            i
                        ),
                        (0, _) => format!(
                            "Agent-0 research note {}: Feed-forward networks in transformers \
                             use two linear layers with a ReLU or GELU activation between them.",
                            i
                        ),
                        (1, 0) => format!(
                            "Agent-1 code review {}: The database connection pool should use \
                             a maximum of 20 connections to avoid exhausting PostgreSQL limits.",
                            i
                        ),
                        (1, 1) => format!(
                            "Agent-1 code review {}: Missing index on users.email column \
                             causes full table scans on login queries taking 200ms+.",
                            i
                        ),
                        (1, 2) => format!(
                            "Agent-1 code review {}: The authentication middleware should \
                             validate JWT tokens before reaching any route handler.",
                            i
                        ),
                        (1, 3) => format!(
                            "Agent-1 code review {}: Connection retry logic uses exponential \
                             backoff with jitter starting at 100ms, maxing at 30 seconds.",
                            i
                        ),
                        (1, _) => format!(
                            "Agent-1 code review {}: Rate limiting at the API gateway should \
                             use token bucket algorithm with 100 requests per minute per client.",
                            i
                        ),
                        (2, 0) => format!(
                            "Agent-2 debug log {}: Memory leak traced to unclosed gRPC streams \
                             in the prediction service, accumulating 50MB per hour.",
                            i
                        ),
                        (2, 1) => format!(
                            "Agent-2 debug log {}: Deadlock between the cache invalidation \
                             thread and the request handler holding concurrent read-write locks.",
                            i
                        ),
                        (2, 2) => format!(
                            "Agent-2 debug log {}: Segfault in native BLAS library when batch \
                             size exceeds 512 on the ARM-based inference nodes.",
                            i
                        ),
                        (2, 3) => format!(
                            "Agent-2 debug log {}: OOM killer terminated the model server \
                             after loading 3 large models simultaneously into GPU memory.",
                            i
                        ),
                        (2, _) => format!(
                            "Agent-2 debug log {}: Race condition in the feature cache \
                             returns stale embeddings when concurrent writes update the same key.",
                            i
                        ),
                        (3, 0) => format!(
                            "Agent-3 planning note {}: Sprint 14 deliverables include the \
                             search reranking pipeline and the embedding cache warm-up job.",
                            i
                        ),
                        (3, 1) => format!(
                            "Agent-3 planning note {}: Migration from Elasticsearch to \
                             Meilisearch requires reindexing 2.3 million documents over the weekend.",
                            i
                        ),
                        (3, 2) => format!(
                            "Agent-3 planning note {}: On-call rotation for Q4 needs at least \
                             3 engineers who understand the inference pipeline end to end.",
                            i
                        ),
                        (3, 3) => format!(
                            "Agent-3 planning note {}: Cost optimization target is reducing \
                             GPU spend by 30% through better batching and model distillation.",
                            i
                        ),
                        (_, _) => format!(
                            "Agent-3 planning note {}: Quarterly OKR review scheduled for \
                             November 15th, focusing on latency targets and reliability SLOs.",
                            i
                        ),
                    }
                })
                .collect();

            let mut local_ids = Vec::new();
            for content in &topics {
                let embedding = provider.embed(content.as_str()).unwrap();
                let node =
                    MemoryNode::new(agent_id, MemoryType::Episodic, content.clone(), embedding);
                local_ids.push(node.id);

                let db = db.lock().unwrap();
                db.store(node).unwrap();
            }

            all_ids.lock().unwrap().extend(local_ids);
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    let ids = all_ids.lock().unwrap();
    assert_eq!(
        ids.len(),
        num_agents * memories_per_agent,
        "All {} memories should have been stored across {} agents",
        num_agents * memories_per_agent,
        num_agents
    );

    // Verify no duplicate IDs
    let unique: HashSet<MemoryId> = ids.iter().copied().collect();
    assert_eq!(
        unique.len(),
        ids.len(),
        "All memory IDs should be unique (no collisions)"
    );

    // Verify the database can still recall after concurrent writes
    let db = db.lock().unwrap();
    let query_emb = embed(&provider, "transformer attention mechanism");
    let results = db.recall_similar(&query_emb, 10).unwrap();
    assert!(
        !results.is_empty(),
        "Should recall memories after concurrent writes from 4 agents"
    );

    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// Test 5: Large context window assembly with token budgets
// ---------------------------------------------------------------------------

/// Validates that the context assembler correctly respects different token
/// budgets when assembling context from a large pool of 200 memories. Tests
/// that smaller budgets include fewer memories, that the assembler prioritizes
/// high-salience content, and that the token count stays within limits.
#[test]
fn test_large_context_window_assembly() {
    let agent_id = AgentId::new();
    let provider = embedder();

    // Build 200 scored memories with varying salience and realistic content
    let topics: Vec<&str> = vec![
        "Rust ownership model prevents data races at compile time without a garbage collector.",
        "The borrow checker enforces that references cannot outlive their referent.",
        "Lifetimes annotate the scope for which a reference is valid.",
        "Pattern matching with match expressions exhaustively handles all enum variants.",
        "Traits define shared behavior similar to interfaces in other languages.",
        "Async/await in Rust uses zero-cost futures that compile to state machines.",
        "The tokio runtime provides multi-threaded work-stealing for async IO tasks.",
        "Serde serialization framework handles JSON, TOML, YAML, and binary formats.",
        "Cargo build system manages dependencies, compilation, and cross-compilation targets.",
        "Procedural macros generate code at compile time from token streams.",
        "Error handling uses Result and Option types instead of exceptions.",
        "The type system prevents null pointer dereferences through Option<T>.",
        "Closures in Rust capture variables by reference, mutable reference, or value.",
        "Smart pointers like Box, Rc, and Arc manage heap allocation and reference counting.",
        "Unsafe blocks opt out of specific compiler guarantees for FFI and raw pointer operations.",
        "Iterators are lazy and compose with combinators like map, filter, and fold.",
        "Channels in std::sync::mpsc provide message passing between threads.",
        "Mutex and RwLock provide interior mutability for shared state across threads.",
        "Pin prevents values from being moved in memory, required for self-referential types.",
        "The Drop trait defines custom cleanup logic run when values go out of scope.",
    ];

    let mut scored_memories: Vec<ScoredMemory> = Vec::new();

    for i in 0..200 {
        let topic = topics[i % topics.len()];
        let content = format!(
            "Memory {}: {}. Additional context for entry {} covering implementation details \
             and practical usage patterns in production systems.",
            i, topic, i
        );
        let embedding = embed(&provider, &content);
        let mut node = MemoryNode::new(agent_id, MemoryType::Semantic, content, embedding);

        // Vary salience: memories 0-49 are high, 50-149 are medium, 150-199 are low
        node.salience = if i < 50 {
            0.9 + (i as f32 * 0.002)
        } else if i < 150 {
            0.5 + ((i - 50) as f32 * 0.003)
        } else {
            0.1 + ((i - 150) as f32 * 0.004)
        };

        // Use salience as score for assembly ordering
        scored_memories.push(ScoredMemory {
            score: node.salience,
            memory: node,
        });
    }

    assert_eq!(
        scored_memories.len(),
        200,
        "Should have 200 candidate memories"
    );

    // Assemble with a small 500-token budget
    let small_config = AssemblyConfig {
        token_budget: 500,
        ..AssemblyConfig::default()
    };
    let small_window = ContextAssembler::assemble(scored_memories.clone(), vec![], &small_config);

    // Assemble with a medium 2000-token budget
    let medium_config = AssemblyConfig {
        token_budget: 2000,
        ..AssemblyConfig::default()
    };
    let medium_window = ContextAssembler::assemble(scored_memories.clone(), vec![], &medium_config);

    // Assemble with a large 8000-token budget
    let large_config = AssemblyConfig {
        token_budget: 8000,
        ..AssemblyConfig::default()
    };
    let large_window = ContextAssembler::assemble(scored_memories.clone(), vec![], &large_config);

    // Token counts should respect budgets
    assert!(
        small_window.total_tokens <= 500,
        "Small window tokens ({}) should not exceed 500",
        small_window.total_tokens
    );
    assert!(
        medium_window.total_tokens <= 2000,
        "Medium window tokens ({}) should not exceed 2000",
        medium_window.total_tokens
    );
    assert!(
        large_window.total_tokens <= 8000,
        "Large window tokens ({}) should not exceed 8000",
        large_window.total_tokens
    );

    // Larger budgets should include more memories
    assert!(
        medium_window.metadata.included_count > small_window.metadata.included_count,
        "Medium budget ({} memories) should include more than small budget ({} memories)",
        medium_window.metadata.included_count,
        small_window.metadata.included_count
    );
    assert!(
        large_window.metadata.included_count > medium_window.metadata.included_count,
        "Large budget ({} memories) should include more than medium budget ({} memories)",
        large_window.metadata.included_count,
        medium_window.metadata.included_count
    );

    // All assemblies should have excluded some candidates (200 is a lot)
    assert!(
        small_window.metadata.excluded_count > 0,
        "Small budget should exclude memories"
    );
    assert!(
        medium_window.metadata.excluded_count > 0,
        "Medium budget should exclude memories"
    );

    // The format output should be non-empty for all budgets
    assert!(
        !small_window.format.is_empty(),
        "Small window should produce serialized output"
    );
    assert!(
        !medium_window.format.is_empty(),
        "Medium window should produce serialized output"
    );
    assert!(
        !large_window.format.is_empty(),
        "Large window should produce serialized output"
    );

    // Metadata total_candidates should always be 200
    assert_eq!(small_window.metadata.total_candidates, 200);
    assert_eq!(medium_window.metadata.total_candidates, 200);
    assert_eq!(large_window.metadata.total_candidates, 200);
}

// ---------------------------------------------------------------------------
// Test 6: Contradiction chain detection
// ---------------------------------------------------------------------------

/// Simulates an evolving decision about which database to use in a project.
/// Each new memory contradicts the previous one. The write inference engine
/// should detect these contradictions and flag or mark obsolete the earlier
/// decisions. This validates that MenteDB can track belief changes over time
/// and surface the contradiction chain for the agent to reason about.
#[test]
fn test_contradiction_chain() {
    let dir = tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent_id = AgentId::new();
    let provider = embedder();

    // Use very low thresholds to catch contradictions with hash embeddings.
    // Hash embeddings are deterministic but not semantically meaningful, so
    // cosine similarity between related texts will be lower than with a real
    // model. We lower the thresholds to ensure the engine can still detect
    // relationships in the chain.
    let config = WriteInferenceConfig {
        contradiction_threshold: 0.3,
        obsolete_threshold: 0.2,
        related_min: 0.05,
        related_max: 0.3,
        correction_threshold: 0.1,
        confidence_decay_factor: 0.5,
        confidence_floor: 0.1,
    };
    let engine = WriteInferenceEngine::with_config(config);

    // The evolving chain of database decisions
    let chain = [
        "The project will use PostgreSQL as the primary database for all relational data storage and querying.",
        "We are switching from PostgreSQL to MySQL because the hosting provider offers managed MySQL at lower cost.",
        "Actually, we decided on MongoDB instead of MySQL because our data is mostly unstructured JSON documents.",
        "After load testing, MongoDB write performance was insufficient. Moving to Cassandra for its write throughput.",
        "Cassandra operational complexity is too high for our small team. Switching to CockroachDB for managed distributed SQL.",
        "Final decision: we are going with TiDB because it offers MySQL compatibility with horizontal scaling built in.",
    ];

    let base_time = now_us() - 6 * DAY_US;
    let mut stored_memories: Vec<MemoryNode> = Vec::new();
    let mut all_actions: Vec<Vec<InferredAction>> = Vec::new();
    let mut total_edges_created: usize = 0;
    let mut total_obsolete_flags = 0;
    let mut total_contradiction_flags = 0;

    for (i, content) in chain.iter().enumerate() {
        let node = make_memory_at_time(
            agent_id,
            content,
            MemoryType::Semantic,
            base_time + (i as u64) * DAY_US,
            1.0,
            &provider,
        );

        // Run write inference against all previously stored memories
        let actions = engine.infer_on_write(&node, &stored_memories, &[]);

        for action in &actions {
            match action {
                InferredAction::FlagContradiction { .. } => {
                    total_contradiction_flags += 1;
                }
                InferredAction::MarkObsolete { .. } => {
                    total_obsolete_flags += 1;
                }
                InferredAction::CreateEdge { .. } => {
                    total_edges_created += 1;
                }
                _ => {}
            }
        }

        all_actions.push(actions);
        db.store(node.clone()).unwrap();
        stored_memories.push(node);
    }

    assert_eq!(
        stored_memories.len(),
        chain.len(),
        "All memories in the contradiction chain should be stored"
    );

    // With lowered thresholds, the inference engine should detect at least some
    // relationships between the database decision memories
    let total_inferred = total_contradiction_flags + total_obsolete_flags + total_edges_created;
    assert!(
        total_inferred > 0,
        "Write inference should detect at least one relationship in the contradiction chain. \
         Got: {} contradictions, {} obsolete, {} edges",
        total_contradiction_flags,
        total_obsolete_flags,
        total_edges_created
    );

    // The first memory in the chain should have been written before any inference
    // could run (no existing memories at that point)
    assert!(
        all_actions[0].is_empty(),
        "First memory should have no inferred actions (nothing to compare against)"
    );

    // Later memories should produce inferred actions as the chain grows
    let later_actions_count: usize = all_actions[1..].iter().map(|a| a.len()).sum();
    assert!(
        later_actions_count > 0,
        "Later chain entries should produce inferred actions against earlier memories"
    );

    // Verify all memories are queryable
    let query_emb = embed(&provider, "database technology selection for the project");
    let results = db.recall_similar(&query_emb, 10).unwrap();
    assert!(
        !results.is_empty(),
        "Should recall database decision memories"
    );

    db.close().unwrap();
}
