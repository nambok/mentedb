//! Token Efficiency Benchmark
//!
//! Measures token consumption across MenteDB's serialization formats
//! (Compact, Structured, Raw JSON) and multi-turn delta serving.
//!
//! Run with: cargo run --example token_efficiency -p mentedb
//!
//! Outputs markdown tables suitable for README / documentation.

use mentedb::context::budget::estimate_tokens;
use mentedb::context::{
    AssemblyConfig, ContextAssembler, DeltaTracker, OutputFormat, ScoredMemory,
};
use mentedb::prelude::*;
use mentedb_core::types::AgentId;

/// Realistic memory content samples spanning different types and lengths.
const MEMORY_CONTENTS: &[(&str, MemoryType)] = &[
    // Semantic — preferences, facts, decisions
    (
        "User prefers Rust for backend services and TypeScript for frontend",
        MemoryType::Semantic,
    ),
    (
        "The production database is PostgreSQL 15 running on RDS with read replicas",
        MemoryType::Semantic,
    ),
    (
        "Authentication uses OAuth2 with Google as the identity provider",
        MemoryType::Semantic,
    ),
    (
        "The deployment target is AWS ECS Fargate with auto-scaling policies",
        MemoryType::Semantic,
    ),
    (
        "API versioning follows URL-based strategy with /v1/ and /v2/ prefixes",
        MemoryType::Semantic,
    ),
    (
        "Redis is used for caching with a 15 minute TTL and LRU eviction",
        MemoryType::Semantic,
    ),
    (
        "The CI pipeline uses GitHub Actions with parallel test execution across 4 runners",
        MemoryType::Semantic,
    ),
    (
        "Monitoring stack is Prometheus for metrics and Grafana for dashboards",
        MemoryType::Semantic,
    ),
    (
        "Background tasks use a custom job queue backed by SQS with dead letter queues",
        MemoryType::Semantic,
    ),
    (
        "The project follows trunk-based development with short-lived feature branches",
        MemoryType::Semantic,
    ),
    // Episodic — conversation events
    (
        "User asked about implementing WebSocket support for real-time notifications in the dashboard",
        MemoryType::Episodic,
    ),
    (
        "User debugged a race condition in the connection pool that caused intermittent 500 errors",
        MemoryType::Episodic,
    ),
    (
        "User decided to migrate from REST to gRPC for internal service communication",
        MemoryType::Episodic,
    ),
    (
        "User discussed implementing rate limiting at the API gateway level using token buckets",
        MemoryType::Episodic,
    ),
    (
        "User refactored the authentication middleware to support multiple identity providers",
        MemoryType::Episodic,
    ),
    // Procedural — how-to knowledge
    (
        "To deploy: run cargo build --release, docker build, push to ECR, update ECS service",
        MemoryType::Procedural,
    ),
    (
        "Database migrations: create migration with sqlx, test on staging, apply with --dry-run first",
        MemoryType::Procedural,
    ),
    (
        "To add a new API endpoint: define route, add handler, write integration test, update OpenAPI spec",
        MemoryType::Procedural,
    ),
    // AntiPattern — things to avoid
    (
        "Never store API keys in environment variables visible in logs or error messages",
        MemoryType::AntiPattern,
    ),
    (
        "Do not use SELECT * in production queries, always specify explicit column lists",
        MemoryType::AntiPattern,
    ),
    // Correction — updated facts
    (
        "Corrected: the health check timeout should be 30 seconds not 300 seconds",
        MemoryType::Correction,
    ),
    (
        "Corrected: the maximum batch size for SQS is 10 messages not 100",
        MemoryType::Correction,
    ),
    // Reasoning — derived insights
    (
        "The intermittent latency spikes correlate with garbage collection pauses in the JVM sidecar",
        MemoryType::Reasoning,
    ),
    (
        "Based on traffic patterns, the optimal auto-scaling threshold is 70% CPU utilization",
        MemoryType::Reasoning,
    ),
    (
        "The user's architecture follows a modified hexagonal pattern with ports and adapters",
        MemoryType::Reasoning,
    ),
];

fn main() {
    println!("# MenteDB Token Efficiency Benchmark\n");
    println!(
        "Measures token consumption across serialization formats and multi-turn delta serving.\n"
    );

    format_comparison();
    println!();
    scale_comparison();
    println!();
    delta_savings();
    println!();
    budget_density();
}

/// Compare token counts for the same memories across formats.
fn format_comparison() {
    println!("## Format Comparison\n");
    println!("Same 25 memories serialized in each format:\n");

    let memories = generate_memories(25);

    // Raw JSON baseline (what you get without an optimized serializer)
    let json_output = serialize_as_json(&memories);
    let json_tokens = estimate_tokens(&json_output);

    // MenteDB Compact
    let compact_config = AssemblyConfig {
        token_budget: 100_000, // no budget limit for this test
        format: OutputFormat::Compact,
        include_edges: false,
        include_metadata: true,
    };
    let compact_window = ContextAssembler::assemble(memories.clone(), vec![], &compact_config);
    let compact_tokens = estimate_tokens(&compact_window.format);

    // MenteDB Structured
    let structured_config = AssemblyConfig {
        token_budget: 100_000,
        format: OutputFormat::Structured,
        include_edges: false,
        include_metadata: true,
    };
    let structured_window =
        ContextAssembler::assemble(memories.clone(), vec![], &structured_config);
    let structured_tokens = estimate_tokens(&structured_window.format);

    println!("| Format | Tokens | vs Raw JSON |");
    println!("|--------|-------:|------------:|");
    println!("| Raw JSON | {json_tokens} | — |");
    println!(
        "| Structured (markdown) | {structured_tokens} | {:.1}x fewer |",
        json_tokens as f64 / structured_tokens as f64
    );
    println!(
        "| **Compact (pipe-delimited)** | **{compact_tokens}** | **{:.1}x fewer** |",
        json_tokens as f64 / compact_tokens as f64
    );

    println!("\n<details><summary>Sample output: Compact format</summary>\n");
    println!("```");
    // Show first 15 lines
    for line in compact_window.format.lines().take(15) {
        println!("{line}");
    }
    println!("...");
    println!("```\n</details>");
}

/// Measure how token savings scale with memory count.
fn scale_comparison() {
    println!("## Token Savings at Scale\n");
    println!("Compact format vs Raw JSON at increasing memory counts:\n");

    println!("| Memories | Raw JSON | Compact | Savings |");
    println!("|---------:|---------:|--------:|--------:|");

    for count in [10, 25, 50, 100, 250] {
        let memories = generate_memories(count);

        let json_output = serialize_as_json(&memories);
        let json_tokens = estimate_tokens(&json_output);

        let config = AssemblyConfig {
            token_budget: 1_000_000,
            format: OutputFormat::Compact,
            include_edges: false,
            include_metadata: true,
        };
        let window = ContextAssembler::assemble(memories, vec![], &config);
        let compact_tokens = estimate_tokens(&window.format);

        let savings = (1.0 - compact_tokens as f64 / json_tokens as f64) * 100.0;
        println!("| {count} | {json_tokens} | {compact_tokens} | {savings:.1}% |");
    }
}

/// Simulate multi-turn conversation and measure delta savings.
fn delta_savings() {
    println!("## Multi-Turn Delta Serving\n");
    println!(
        "Simulates a 20-turn conversation where 1-3 memories change per turn.\n\
         Full retrieval resends everything each turn. Delta only sends changes.\n"
    );

    let agent_id = AgentId::new();
    let num_turns = 20;

    // Build a pool of memories (simulating growing knowledge)
    let mut all_memories: Vec<ScoredMemory> = Vec::new();
    let mut tracker = DeltaTracker::new();

    let config = AssemblyConfig {
        token_budget: 100_000,
        format: OutputFormat::Structured,
        include_edges: false,
        include_metadata: true,
    };

    let mut total_full_tokens = 0usize;
    let mut total_delta_tokens = 0usize;
    let mut turn_data: Vec<(usize, usize, usize)> = Vec::new(); // (turn, full, delta)

    for turn in 0..num_turns {
        // Add 1-3 new memories per turn
        let new_count = 1 + (turn % 3);
        for j in 0..new_count {
            let idx = (turn * 3 + j) % MEMORY_CONTENTS.len();
            let (content, mem_type) = MEMORY_CONTENTS[idx];
            let mut node = MemoryNode::new(
                agent_id,
                mem_type,
                format!("[turn {turn}] {content}"),
                vec![],
            );
            node.salience = 0.5 + (0.5 * ((turn * 7 + j * 13) % 10) as f32 / 10.0);
            let score = 0.3 + (0.7 * ((turn * 11 + j * 3) % 10) as f32 / 10.0);
            all_memories.push(ScoredMemory {
                memory: node,
                score,
            });
        }

        // Full retrieval: assemble all current memories
        let full_window = ContextAssembler::assemble(all_memories.clone(), vec![], &config);
        let full_tokens = estimate_tokens(&full_window.format);
        total_full_tokens += full_tokens;

        // Delta retrieval: only send changes
        let delta_window =
            ContextAssembler::assemble_delta(all_memories.clone(), vec![], &mut tracker, &config);
        let delta_tokens = estimate_tokens(&delta_window.format);
        total_delta_tokens += delta_tokens;

        turn_data.push((turn + 1, full_tokens, delta_tokens));
    }

    println!("| Turn | Memories | Full (tokens) | Delta (tokens) | Turn Savings |");
    println!("|-----:|---------:|--------------:|---------------:|-------------:|");
    let mut mem_count = 0;
    for (i, (turn, full, delta)) in turn_data.iter().enumerate() {
        mem_count += 1 + (i % 3);
        let savings = if *full > 0 {
            (1.0 - *delta as f64 / *full as f64) * 100.0
        } else {
            0.0
        };
        println!("| {turn} | {mem_count} | {full} | {delta} | {savings:.1}% |");
    }

    let total_savings = if total_full_tokens > 0 {
        (1.0 - total_delta_tokens as f64 / total_full_tokens as f64) * 100.0
    } else {
        0.0
    };

    println!("\n**Cumulative over {num_turns} turns:**\n");
    println!("| Metric | Tokens |");
    println!("|--------|-------:|");
    println!("| Full retrieval (total) | {total_full_tokens} |");
    println!("| Delta serving (total) | {total_delta_tokens} |");
    println!("| **Total savings** | **{total_savings:.1}%** |");
}

/// Measure how many memories fit within common token budgets per format.
fn budget_density() {
    println!("## Memory Density per Token Budget\n");
    println!(
        "How many memories fit within a given **serialized output** token budget.\n\
         This measures what the LLM actually receives:\n"
    );

    println!("| Budget | Compact | Structured | Raw JSON |");
    println!("|-------:|--------:|-----------:|---------:|");

    for budget in [1_000, 2_000, 4_096, 8_192, 16_384] {
        let memories = generate_memories(500); // large pool

        let compact_count = serialized_fit_count(&memories, OutputFormat::Compact, budget);
        let structured_count = serialized_fit_count(&memories, OutputFormat::Structured, budget);
        let json_count = json_fit_count(&memories, budget);

        println!(
            "| {} | {} | {} | {} |",
            format_number(budget),
            compact_count,
            structured_count,
            json_count
        );
    }
}

// --- Helpers ---

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn generate_memories(count: usize) -> Vec<ScoredMemory> {
    let agent_id = AgentId::new();
    (0..count)
        .map(|i| {
            let idx = i % MEMORY_CONTENTS.len();
            let (content, mem_type) = MEMORY_CONTENTS[idx];
            let mut node = MemoryNode::new(
                agent_id,
                mem_type,
                if count > MEMORY_CONTENTS.len() {
                    format!("{content} (instance {i})")
                } else {
                    content.to_string()
                },
                vec![],
            );
            node.salience = 0.5 + (0.5 * ((i * 7) % 10) as f32 / 10.0);
            node.tags = vec!["project".to_string()];
            let score = 0.3 + (0.7 * ((i * 11) % 10) as f32 / 10.0);
            ScoredMemory {
                memory: node,
                score,
            }
        })
        .collect()
}

fn serialize_as_json(memories: &[ScoredMemory]) -> String {
    let items: Vec<serde_json::Value> = memories
        .iter()
        .map(|sm| {
            serde_json::json!({
                "id": sm.memory.id.to_string(),
                "type": format!("{:?}", sm.memory.memory_type),
                "content": sm.memory.content,
                "salience": sm.memory.salience,
                "score": sm.score,
                "tags": sm.memory.tags,
                "created_at": sm.memory.created_at,
            })
        })
        .collect();
    serde_json::to_string_pretty(&items).unwrap_or_default()
}

fn serialized_fit_count(memories: &[ScoredMemory], format: OutputFormat, budget: usize) -> usize {
    // Binary search: find the maximum number of memories whose serialized output fits in budget
    let mut sorted: Vec<ScoredMemory> = memories.to_vec();
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Use huge internal budget so the assembler doesn't limit us — we measure the output
    let config = AssemblyConfig {
        token_budget: 1_000_000,
        format,
        include_edges: false,
        include_metadata: true,
    };

    let mut lo = 0usize;
    let mut hi = sorted.len();
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        let subset: Vec<ScoredMemory> = sorted[..mid].to_vec();
        let window = ContextAssembler::assemble(subset, vec![], &config);
        let tokens = estimate_tokens(&window.format);
        if tokens <= budget {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

fn json_fit_count(memories: &[ScoredMemory], budget: usize) -> usize {
    let mut tokens_used = 0;
    let mut count = 0;
    // Sort by score descending like the assembler does
    let mut sorted: Vec<&ScoredMemory> = memories.iter().collect();
    sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for sm in sorted {
        let json = serde_json::to_string(&serde_json::json!({
            "id": sm.memory.id.to_string(),
            "type": format!("{:?}", sm.memory.memory_type),
            "content": sm.memory.content,
            "salience": sm.memory.salience,
            "tags": sm.memory.tags,
        }))
        .unwrap_or_default();
        let t = estimate_tokens(&json);
        if tokens_used + t > budget {
            break;
        }
        tokens_used += t;
        count += 1;
    }
    count
}
