//! Integration tests for CognitiveLlmService against a real Ollama instance.
//!
//! These tests verify that the prompts produce correct LLM verdicts for
//! curated scenarios. Run with:
//!
//!   cargo test -p mentedb-extraction --test llm_accuracy -- --ignored --nocapture
//!
//! Requires Ollama running locally with a model pulled:
//!   ollama pull llama3.2
//!
//! All tests are #[ignore] by default so they dont run in CI.

use mentedb_cognitive::llm::*;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::MemoryId;
use mentedb_extraction::cognitive_adapter::ExtractionLlmJudge;
use mentedb_extraction::config::{ExtractionConfig, LlmProvider};
use mentedb_extraction::provider::HttpExtractionProvider;

/// Build a CognitiveLlmService from environment variables.
///
/// Environment variables:
///   LLM_PROVIDER  - "ollama" (default), "openai", "anthropic", "custom"
///   LLM_API_KEY   - API key (required for openai/anthropic/custom)
///   LLM_MODEL     - Model name (defaults per provider)
///   LLM_API_URL   - Custom API URL (optional, uses provider defaults)
///
/// Examples:
///   # Ollama (default)
///   cargo test -p mentedb-extraction --test llm_accuracy -- --ignored --nocapture
///
///   # OpenAI
///   LLM_PROVIDER=openai LLM_API_KEY=sk-... cargo test -p mentedb-extraction --test llm_accuracy -- --ignored --nocapture
///
///   # Anthropic
///   LLM_PROVIDER=anthropic LLM_API_KEY=sk-ant-... cargo test -p mentedb-extraction --test llm_accuracy -- --ignored --nocapture
///
///   # Custom (e.g. Groq via OpenAI-compatible endpoint)
///   LLM_PROVIDER=custom LLM_API_KEY=gsk-... LLM_API_URL=https://api.groq.com/openai/v1/chat/completions LLM_MODEL=llama-3.1-8b-instant cargo test -p mentedb-extraction --test llm_accuracy -- --ignored --nocapture
fn ollama_service() -> CognitiveLlmService<ExtractionLlmJudge> {
    let provider_str = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "ollama".into());
    let api_key = std::env::var("LLM_API_KEY").ok();
    let model_override = std::env::var("LLM_MODEL").ok();
    let url_override = std::env::var("LLM_API_URL").ok();

    let mut config = match provider_str.to_lowercase().as_str() {
        "openai" => {
            let key = api_key.expect("LLM_API_KEY required for openai provider");
            ExtractionConfig::openai(key)
        }
        "anthropic" => {
            let key = api_key.expect("LLM_API_KEY required for anthropic provider");
            ExtractionConfig::anthropic(key)
        }
        "custom" => {
            let key = api_key.expect("LLM_API_KEY required for custom provider");
            let mut cfg = ExtractionConfig::openai(key);
            cfg.provider = LlmProvider::Custom;
            cfg
        }
        _ => ExtractionConfig::ollama(),
    };

    if let Some(model) = model_override {
        config.model = model;
    } else if config.provider == LlmProvider::Ollama {
        config.model = "llama3.1:8b".to_string();
    }

    if let Some(url) = url_override {
        config.api_url = url;
    }

    eprintln!(
        "\n  Provider: {:?} | Model: {} | URL: {}\n",
        config.provider, config.model, config.api_url
    );

    let provider = HttpExtractionProvider::new(config).expect("failed to create LLM provider");
    let judge = ExtractionLlmJudge::new(provider);
    CognitiveLlmService::new(judge)
}

fn mem(content: &str, created_at: u64) -> MemorySummary {
    MemorySummary {
        id: MemoryId::new(),
        content: content.to_string(),
        memory_type: MemoryType::Semantic,
        confidence: 0.9,
        created_at,
    }
}

// ============================================================================
// Invalidation tests
// ============================================================================

struct InvalidationCase {
    old: &'static str,
    new: &'static str,
    expected: &'static str, // "keep", "invalidate", or "update"
    description: &'static str,
}

const INVALIDATION_CASES: &[InvalidationCase] = &[
    // --- Keep cases ---
    InvalidationCase {
        old: "Alice works at Acme",
        new: "Bob works at Google",
        expected: "keep",
        description: "different people",
    },
    InvalidationCase {
        old: "Prefers Rust for backend",
        new: "Uses PostgreSQL for the database",
        expected: "keep",
        description: "different topics",
    },
    InvalidationCase {
        old: "Meeting scheduled for Monday",
        new: "Lunch planned for Tuesday",
        expected: "keep",
        description: "different events",
    },
    InvalidationCase {
        old: "Team uses Scrum methodology",
        new: "Backend is written in Java",
        expected: "keep",
        description: "process vs technology",
    },
    InvalidationCase {
        old: "Has a dog named Max",
        new: "Favorite color is blue",
        expected: "keep",
        description: "unrelated facts",
    },
    InvalidationCase {
        old: "Likes Thai food",
        new: "Enjoys hiking on weekends",
        expected: "keep",
        description: "different life facts",
    },
    InvalidationCase {
        old: "Uses VS Code",
        new: "Monitors with Datadog",
        expected: "keep",
        description: "different tools",
    },
    InvalidationCase {
        old: "Sprint starts Monday",
        new: "Demo is on Friday",
        expected: "keep",
        description: "different events in same sprint",
    },
    // --- Invalidate cases ---
    InvalidationCase {
        old: "Alice works at Acme Corp",
        new: "Alice just started at Google",
        expected: "invalidate",
        description: "job change",
    },
    InvalidationCase {
        old: "Project uses React for the frontend",
        new: "Team migrated the frontend to Vue last month",
        expected: "invalidate",
        description: "framework change",
    },
    InvalidationCase {
        old: "Team lead is Sarah",
        new: "Mike replaced Sarah as team lead",
        expected: "invalidate",
        description: "role change",
    },
    InvalidationCase {
        old: "Office is on the 5th floor",
        new: "We moved to the 3rd floor last week",
        expected: "invalidate",
        description: "location change",
    },
    InvalidationCase {
        old: "Using Python 3.9 for the project",
        new: "Upgraded the project to Python 3.12",
        expected: "invalidate",
        description: "version upgrade",
    },
    InvalidationCase {
        old: "Deployment target is AWS",
        new: "Migrated everything from AWS to GCP",
        expected: "invalidate",
        description: "cloud migration",
    },
    InvalidationCase {
        old: "Primary programming language is Java",
        new: "Team rewrote the codebase in Go",
        expected: "invalidate",
        description: "language rewrite",
    },
    InvalidationCase {
        old: "Database is MySQL",
        new: "Switched from MySQL to PostgreSQL",
        expected: "invalidate",
        description: "database change",
    },
    InvalidationCase {
        old: "Salary is $100k",
        new: "Got a raise to $130k",
        expected: "invalidate",
        description: "salary update",
    },
    InvalidationCase {
        old: "Lives in San Francisco",
        new: "Relocated to Austin last month",
        expected: "invalidate",
        description: "city change",
    },
    // --- Update cases ---
    InvalidationCase {
        old: "User prefers Rust",
        new: "User prefers Rust specifically for its memory safety and zero cost abstractions",
        expected: "update",
        description: "adds detail to preference",
    },
    InvalidationCase {
        old: "Project deadline is Q2",
        new: "Project deadline confirmed as June 30, end of Q2",
        expected: "update",
        description: "adds specific date",
    },
    InvalidationCase {
        old: "Uses Docker for deployment",
        new: "Uses Docker with Kubernetes orchestration in production",
        expected: "update",
        description: "adds orchestration detail",
    },
    InvalidationCase {
        old: "API follows REST",
        new: "API follows REST with OpenAPI 3.0 spec at /docs endpoint",
        expected: "update",
        description: "adds spec detail",
    },
    InvalidationCase {
        old: "Team has 5 engineers",
        new: "Team grew to 5 engineers after hiring 2 juniors last month",
        expected: "update",
        description: "adds hiring context",
    },
];

// ============================================================================
// Contradiction tests
// ============================================================================

struct ContradictionCase {
    a: &'static str,
    b: &'static str,
    expected: &'static str, // "compatible", "contradicts", or "supersedes"
    description: &'static str,
}

const CONTRADICTION_CASES: &[ContradictionCase] = &[
    // --- Compatible ---
    ContradictionCase {
        a: "Likes Python",
        b: "Also enjoys Rust",
        expected: "compatible",
        description: "can like multiple languages",
    },
    ContradictionCase {
        a: "Works from home",
        b: "Has an office desk for in person days",
        expected: "compatible",
        description: "hybrid work",
    },
    ContradictionCase {
        a: "Prefers dark mode",
        b: "Uses a 32 inch monitor",
        expected: "compatible",
        description: "unrelated preferences",
    },
    ContradictionCase {
        a: "Uses Git for version control",
        b: "Uses Jira for project tracking",
        expected: "compatible",
        description: "different tool categories",
    },
    ContradictionCase {
        a: "Drinks coffee in the morning",
        b: "Drinks tea in the afternoon",
        expected: "compatible",
        description: "time separated habits",
    },
    ContradictionCase {
        a: "Expert in backend development",
        b: "Learning frontend development",
        expected: "compatible",
        description: "growing skillset",
    },
    ContradictionCase {
        a: "Uses macOS for development",
        b: "Runs Linux on the server",
        expected: "compatible",
        description: "different machines",
    },
    ContradictionCase {
        a: "Likes reading fiction",
        b: "Enjoys non fiction books too",
        expected: "compatible",
        description: "both can be true",
    },
    // --- Contradicts ---
    ContradictionCase {
        a: "Prefers tabs for indentation",
        b: "Prefers spaces for indentation",
        expected: "contradicts",
        description: "mutually exclusive",
    },
    ContradictionCase {
        a: "Database must be NoSQL",
        b: "Database must be relational SQL",
        expected: "contradicts",
        description: "opposing requirements",
    },
    ContradictionCase {
        a: "Project is fully open source",
        b: "Project is proprietary and confidential",
        expected: "contradicts",
        description: "cannot be both",
    },
    ContradictionCase {
        a: "The deadline is absolutely firm",
        b: "The deadline is flexible and can be moved",
        expected: "contradicts",
        description: "opposing constraints",
    },
    ContradictionCase {
        a: "We must use a monorepo",
        b: "Each service needs its own repository",
        expected: "contradicts",
        description: "opposing architecture",
    },
    ContradictionCase {
        a: "All API calls must be synchronous",
        b: "All API calls must be asynchronous",
        expected: "contradicts",
        description: "opposing design",
    },
    ContradictionCase {
        a: "No third party dependencies allowed",
        b: "Use as many libraries as possible to save time",
        expected: "contradicts",
        description: "opposing policies",
    },
    ContradictionCase {
        a: "Testing is not a priority right now",
        b: "We need 100% test coverage before shipping",
        expected: "contradicts",
        description: "opposing priorities",
    },
    // --- Supersedes ---
    ContradictionCase {
        a: "Using React 17",
        b: "Upgraded to React 18 last week",
        expected: "supersedes",
        description: "version upgrade",
    },
    ContradictionCase {
        a: "Primary database is MySQL",
        b: "Migrated from MySQL to PostgreSQL this quarter",
        expected: "supersedes",
        description: "database migration",
    },
    ContradictionCase {
        a: "Team has 3 members",
        b: "Team grew to 7 after the new hires",
        expected: "supersedes",
        description: "team growth",
    },
    ContradictionCase {
        a: "CI pipeline uses Jenkins",
        b: "Replaced Jenkins with GitHub Actions",
        expected: "supersedes",
        description: "tool replacement",
    },
    ContradictionCase {
        a: "API is at version 1",
        b: "Launched version 2 of the API",
        expected: "supersedes",
        description: "API version bump",
    },
    ContradictionCase {
        a: "Office is downtown",
        b: "Company moved to the suburbs campus",
        expected: "supersedes",
        description: "location move",
    },
    ContradictionCase {
        a: "CEO is John",
        b: "Jane became CEO after John retired",
        expected: "supersedes",
        description: "leadership change",
    },
    ContradictionCase {
        a: "Monthly budget is $50k",
        b: "Budget increased to $75k starting this month",
        expected: "supersedes",
        description: "budget update",
    },
];

// ============================================================================
// Topic canonicalization tests
// ============================================================================

struct TopicCase {
    message: &'static str,
    existing: &'static [&'static str],
    expected_topic: &'static str,
    expected_new: bool,
    description: &'static str,
}

const TOPIC_CASES: &[TopicCase] = &[
    // Match existing
    TopicCase {
        message: "how do I set up authentication",
        existing: &["database", "authentication", "deployment"],
        expected_topic: "authentication",
        expected_new: false,
        description: "direct match",
    },
    TopicCase {
        message: "configure the auth middleware",
        existing: &["database", "authentication"],
        expected_topic: "authentication",
        expected_new: false,
        description: "auth synonym",
    },
    TopicCase {
        message: "the login endpoint returns 401",
        existing: &["authentication", "deployment"],
        expected_topic: "authentication",
        expected_new: false,
        description: "login is auth",
    },
    TopicCase {
        message: "deploy to staging",
        existing: &["authentication", "deployment"],
        expected_topic: "deployment",
        expected_new: false,
        description: "staging deploy",
    },
    TopicCase {
        message: "push to production",
        existing: &["deployment", "testing"],
        expected_topic: "deployment",
        expected_new: false,
        description: "prod deploy",
    },
    TopicCase {
        message: "run the test suite",
        existing: &["deployment", "testing"],
        expected_topic: "testing",
        expected_new: false,
        description: "direct test match",
    },
    TopicCase {
        message: "fix the failing unit tests",
        existing: &["testing", "database"],
        expected_topic: "testing",
        expected_new: false,
        description: "test failure",
    },
    TopicCase {
        message: "PostgreSQL connection pool config",
        existing: &["database", "testing"],
        expected_topic: "database",
        expected_new: false,
        description: "postgres is database",
    },
    TopicCase {
        message: "add an index to the users table",
        existing: &["database", "authentication"],
        expected_topic: "database",
        expected_new: false,
        description: "table index",
    },
    TopicCase {
        message: "the query on orders is slow",
        existing: &["database", "monitoring"],
        expected_topic: "database",
        expected_new: false,
        description: "slow query",
    },
    // New topics
    TopicCase {
        message: "set up Prometheus alerting",
        existing: &["database", "authentication"],
        expected_topic: "monitoring",
        expected_new: true,
        description: "new monitoring topic",
    },
    TopicCase {
        message: "add a Redis caching layer",
        existing: &["database", "authentication"],
        expected_topic: "caching",
        expected_new: true,
        description: "new caching topic",
    },
    TopicCase {
        message: "write API documentation for the endpoints",
        existing: &["testing", "deployment"],
        expected_topic: "documentation",
        expected_new: true,
        description: "new docs topic",
    },
    TopicCase {
        message: "set up the GitHub Actions CI pipeline",
        existing: &["testing", "deployment"],
        expected_topic: "ci",
        expected_new: true,
        description: "new CI topic",
    },
    TopicCase {
        message: "configure CORS and rate limiting",
        existing: &["authentication", "database"],
        expected_topic: "security",
        expected_new: true,
        description: "new security topic",
    },
];

// ============================================================================
// Test runner
// ============================================================================

#[tokio::test]
#[ignore]
async fn llm_accuracy_invalidation() {
    let svc = ollama_service();
    let mut pass = 0;
    let mut _fail = 0;
    let total = INVALIDATION_CASES.len();

    println!("\n======================================================================");
    println!("  INVALIDATION ACCURACY TEST ({total} cases)");
    println!("======================================================================\n");

    for case in INVALIDATION_CASES {
        let old = mem(case.old, 1000);
        let new = mem(case.new, 2000);

        let result = svc.judge_invalidation(&old, &new).await;
        let actual = match &result {
            Ok(InvalidationVerdict::Keep { .. }) => "keep",
            Ok(InvalidationVerdict::Invalidate { .. }) => "invalidate",
            Ok(InvalidationVerdict::Update { .. }) => "update",
            Err(e) => {
                println!(
                    "  FAIL [{}]: {} -> ERROR: {}",
                    case.description, case.expected, e
                );
                _fail += 1;
                continue;
            }
        };

        if actual == case.expected {
            println!(
                "  PASS [{}]: expected={}, got={}",
                case.description, case.expected, actual
            );
            pass += 1;
        } else {
            println!(
                "  FAIL [{}]: expected={}, got={} | old=\"{}\" new=\"{}\"",
                case.description, case.expected, actual, case.old, case.new
            );
            if let Ok(v) = &result {
                println!("        verdict detail: {:?}", v);
            }
            _fail += 1;
        }
    }

    let accuracy = (pass as f64 / total as f64) * 100.0;
    println!("\n  Results: {pass}/{total} passed ({accuracy:.1}%)\n");
    assert!(
        accuracy >= 70.0,
        "Invalidation accuracy {accuracy:.1}% below 70% threshold"
    );
}

#[tokio::test]
#[ignore]
async fn llm_accuracy_contradiction() {
    let svc = ollama_service();
    let mut pass = 0;
    let mut _fail = 0;
    let total = CONTRADICTION_CASES.len();

    println!("\n======================================================================");
    println!("  CONTRADICTION ACCURACY TEST ({total} cases)");
    println!("======================================================================\n");

    for case in CONTRADICTION_CASES {
        let a = mem(case.a, 1000);
        let b = mem(case.b, 2000);

        let result = svc.detect_contradiction(&a, &b).await;
        let actual = match &result {
            Ok(ContradictionVerdict::Compatible { .. }) => "compatible",
            Ok(ContradictionVerdict::Contradicts { .. }) => "contradicts",
            Ok(ContradictionVerdict::Supersedes { .. }) => "supersedes",
            Err(e) => {
                println!(
                    "  FAIL [{}]: {} -> ERROR: {}",
                    case.description, case.expected, e
                );
                _fail += 1;
                continue;
            }
        };

        if actual == case.expected {
            println!(
                "  PASS [{}]: expected={}, got={}",
                case.description, case.expected, actual
            );
            pass += 1;
        } else {
            println!(
                "  FAIL [{}]: expected={}, got={} | a=\"{}\" b=\"{}\"",
                case.description, case.expected, actual, case.a, case.b
            );
            if let Ok(v) = &result {
                println!("        verdict detail: {:?}", v);
            }
            _fail += 1;
        }
    }

    let accuracy = (pass as f64 / total as f64) * 100.0;
    println!("\n  Results: {pass}/{total} passed ({accuracy:.1}%)\n");
    assert!(
        accuracy >= 70.0,
        "Contradiction accuracy {accuracy:.1}% below 70% threshold"
    );
}

#[tokio::test]
#[ignore]
async fn llm_accuracy_topic_canonicalization() {
    let svc = ollama_service();
    let mut pass = 0;
    let mut _fail = 0;
    let total = TOPIC_CASES.len();

    println!("\n======================================================================");
    println!("  TOPIC CANONICALIZATION ACCURACY TEST ({total} cases)");
    println!("======================================================================\n");

    for case in TOPIC_CASES {
        let existing: Vec<String> = case.existing.iter().map(|s| s.to_string()).collect();
        let result = svc.canonicalize_topic(case.message, &existing).await;

        match &result {
            Ok(label) => {
                // For existing topics, check exact match
                // For new topics, check that is_new is true (topic name may vary)
                let topic_ok = if case.expected_new {
                    label.is_new
                } else {
                    label.topic.to_lowercase() == case.expected_topic.to_lowercase()
                };

                if topic_ok {
                    println!(
                        "  PASS [{}]: topic=\"{}\", is_new={}",
                        case.description, label.topic, label.is_new
                    );
                    pass += 1;
                } else {
                    println!(
                        "  FAIL [{}]: expected topic=\"{}\"/new={}, got topic=\"{}\"/new={} | msg=\"{}\"",
                        case.description,
                        case.expected_topic,
                        case.expected_new,
                        label.topic,
                        label.is_new,
                        case.message
                    );
                    _fail += 1;
                }
            }
            Err(e) => {
                println!(
                    "  FAIL [{}]: {} -> ERROR: {}",
                    case.description, case.expected_topic, e
                );
                _fail += 1;
            }
        }
    }

    let accuracy = (pass as f64 / total as f64) * 100.0;
    println!("\n  Results: {pass}/{total} passed ({accuracy:.1}%)\n");
    assert!(
        accuracy >= 70.0,
        "Topic canonicalization accuracy {accuracy:.1}% below 70% threshold"
    );
}

#[tokio::test]
#[ignore]
async fn llm_accuracy_full_report() {
    let svc = ollama_service();
    let mut total_pass = 0;
    let mut total_fail = 0;

    println!("\n======================================================================");
    println!("  FULL LLM ACCURACY REPORT");
    println!("======================================================================");

    // Invalidation
    println!(
        "\n--- Invalidation ({} cases) ---\n",
        INVALIDATION_CASES.len()
    );
    for case in INVALIDATION_CASES {
        let result = svc
            .judge_invalidation(&mem(case.old, 1000), &mem(case.new, 2000))
            .await;
        let actual = match &result {
            Ok(InvalidationVerdict::Keep { .. }) => "keep",
            Ok(InvalidationVerdict::Invalidate { .. }) => "invalidate",
            Ok(InvalidationVerdict::Update { .. }) => "update",
            Err(_) => "error",
        };
        let ok = actual == case.expected;
        if ok {
            total_pass += 1;
        } else {
            total_fail += 1;
        }
        println!(
            "  {} [{}]: expected={}, got={}",
            if ok { "PASS" } else { "FAIL" },
            case.description,
            case.expected,
            actual
        );
    }

    // Contradiction
    println!(
        "\n--- Contradiction ({} cases) ---\n",
        CONTRADICTION_CASES.len()
    );
    for case in CONTRADICTION_CASES {
        let result = svc
            .detect_contradiction(&mem(case.a, 1000), &mem(case.b, 2000))
            .await;
        let actual = match &result {
            Ok(ContradictionVerdict::Compatible { .. }) => "compatible",
            Ok(ContradictionVerdict::Contradicts { .. }) => "contradicts",
            Ok(ContradictionVerdict::Supersedes { .. }) => "supersedes",
            Err(_) => "error",
        };
        let ok = actual == case.expected;
        if ok {
            total_pass += 1;
        } else {
            total_fail += 1;
        }
        println!(
            "  {} [{}]: expected={}, got={}",
            if ok { "PASS" } else { "FAIL" },
            case.description,
            case.expected,
            actual
        );
    }

    // Topic canonicalization
    println!(
        "\n--- Topic Canonicalization ({} cases) ---\n",
        TOPIC_CASES.len()
    );
    for case in TOPIC_CASES {
        let existing: Vec<String> = case.existing.iter().map(|s| s.to_string()).collect();
        let result = svc.canonicalize_topic(case.message, &existing).await;
        let ok = match &result {
            Ok(label) if case.expected_new => label.is_new,
            Ok(label) => label.topic.to_lowercase() == case.expected_topic.to_lowercase(),
            Err(_) => false,
        };
        if ok {
            total_pass += 1;
        } else {
            total_fail += 1;
        }
        let got = match &result {
            Ok(l) => format!("topic=\"{}\", new={}", l.topic, l.is_new),
            Err(e) => format!("error: {e}"),
        };
        println!(
            "  {} [{}]: expected=\"{}\"/new={}, got {}",
            if ok { "PASS" } else { "FAIL" },
            case.description,
            case.expected_topic,
            case.expected_new,
            got
        );
    }

    let grand_total = total_pass + total_fail;
    let accuracy = (total_pass as f64 / grand_total as f64) * 100.0;
    println!("\n======================================================================");
    println!("  GRAND TOTAL: {total_pass}/{grand_total} passed ({accuracy:.1}%)");
    println!("======================================================================\n");

    assert!(
        accuracy >= 70.0,
        "Overall accuracy {accuracy:.1}% below 70% threshold"
    );
}
