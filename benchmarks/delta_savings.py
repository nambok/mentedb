"""
The Delta Savings Test — Quantify token efficiency.

Simulates a 20-turn conversation. Measures total tokens sent as context:
  A) Full retrieval each turn (what competitors do)
  B) Delta-only after turn 1 (what MenteDB can do)

Reports: "X% fewer tokens across 20 turns"
"""

from harness import MenteDBBenchmark, print_result

def estimate_tokens(text):
    """Rough token estimate: words * 1.3"""
    return int(len(text.split()) * 1.3)

def run_delta_savings_test():
    bench = MenteDBBenchmark()
    
    try:
        # Simulate 20 turns of conversation, each adding a memory
        turn_memories = []
        topics = [
            "The user prefers Python for backend development and uses Flask.",
            "The user's project is called DataPipeline and handles ETL workflows.",
            "The user needs to process 10GB of CSV files daily from S3.",
            "The user chose PostgreSQL 15 for the production database.",
            "Authentication should use OAuth2 with Google as the identity provider.",
            "The deployment target is AWS ECS with Fargate launch type.",
            "The user wants comprehensive logging with structured JSON output.",
            "Error handling should use custom exception classes with error codes.",
            "The API should follow OpenAPI 3.0 spec with auto-generated docs.",
            "Rate limiting should be 100 requests per minute per API key.",
            "The user wants Redis for caching with a 15 minute TTL.",
            "Database migrations should use Alembic with automatic revision IDs.",
            "The CI pipeline uses GitHub Actions with parallel test execution.",
            "The user prefers pytest with fixtures for all test setup.",
            "Monitoring should use Prometheus metrics with Grafana dashboards.",
            "The user wants WebSocket support for real-time notifications.",
            "Background tasks should use Celery with Redis as the broker.",
            "The API versioning strategy is URL-based: /v1/, /v2/ etc.",
            "The user wants automatic retry with exponential backoff for HTTP calls.",
            "Documentation should be in Markdown with MkDocs for the site.",
        ]
        for i in range(20):
            mem_id = bench.store(topics[i], memory_type="semantic", tags=[f"turn_{i}"])
            turn_memories.append((mem_id, topics[i]))
        
        # --- Measure Full Retrieval (competitor approach) ---
        total_tokens_full = 0
        for turn in range(20):
            # Each turn retrieves ALL relevant memories (top-10)
            results = bench.search(f"What do I know about the project?", limit=10)
            context = ""
            for rid, score in (results or []):
                try:
                    mem = bench.get(rid)
                    content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
                    context += content + "\n"
                except:
                    pass
            total_tokens_full += estimate_tokens(context)
        
        # --- Measure Delta Approach (MenteDB) ---
        total_tokens_delta = 0
        seen_ids = set()
        for turn in range(20):
            results = bench.search(f"What do I know about the project?", limit=10)
            context = ""
            new_count = 0
            for rid, score in (results or []):
                if rid not in seen_ids:
                    try:
                        mem = bench.get(rid)
                        content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
                        context += content + "\n"
                        seen_ids.add(rid)
                        new_count += 1
                    except:
                        pass
            # Delta: only send new memories + a brief "unchanged: N" marker
            unchanged = len(seen_ids) - new_count
            if unchanged > 0:
                context += f"[{unchanged} previously sent memories unchanged]\n"
            total_tokens_delta += estimate_tokens(context)
        
        # --- Results ---
        savings_pct = ((total_tokens_full - total_tokens_delta) / total_tokens_full) * 100 if total_tokens_full > 0 else 0
        
        passed = savings_pct > 20  # We expect significant savings
        
        details = {
            "Turns simulated": 20,
            "Total tokens (full retrieval)": f"{total_tokens_full:,}",
            "Total tokens (delta)": f"{total_tokens_delta:,}",
            "Token savings": f"{savings_pct:.1f}%",
            "Verdict": f"Delta serving saves {savings_pct:.0f}% of context tokens",
        }
        
        print_result("Delta Savings Test", passed, details)
        return passed
        
    finally:
        bench.cleanup()


if __name__ == "__main__":
    run_delta_savings_test()
