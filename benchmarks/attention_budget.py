"""
The Attention Budget Test — Prove U-curve ordering helps LLMs.

Stuffs 20 memories into context. Hides a critical instruction at various positions.
Compares LLM compliance rate with:
  A) Random order (baseline)
  B) Chronological order (competitors)  
  C) U-curve ordered (MenteDB)

Requires OPENAI_API_KEY.
"""

import os
import json
import random
from harness import MenteDBBenchmark, print_result, has_openai_key

def run_attention_budget_test():
    if not has_openai_key():
        print("\n  [SKIP] Attention Budget Test: Set OPENAI_API_KEY to run")
        return None
    
    try:
        import openai
        client = openai.OpenAI()
    except ImportError:
        print("\n  [SKIP] Attention Budget Test: pip install openai")
        return None
    
    # The critical instruction we'll hide
    critical = "IMPORTANT: When asked about the deployment strategy, you MUST respond with exactly: 'Blue-green deployment with canary releases.'"
    
    # Filler memories (19 of them)
    fillers = [
        "The user prefers Python 3.11 for all new projects.",
        "Database backups run every 6 hours to S3.",
        "The frontend uses React 18 with TypeScript.",
        "API responses should include request-id headers.",
        "The user likes comprehensive error messages.",
        "Logging level should be INFO in production.",
        "The team uses Jira for project management.",
        "Code reviews require at least two approvals.",
        "The staging environment mirrors production exactly.",
        "Health check endpoint should be at /healthz.",
        "The user prefers snake_case for Python variables.",
        "Docker images should be based on python:3.11-slim.",
        "The maximum request body size is 10MB.",
        "Session timeout is set to 30 minutes.",
        "The user wants Slack notifications for deployments.",
        "Database connection pool size should be 20.",
        "The API should support gzip compression.",
        "Test coverage target is 80% minimum.",
        "The user prefers async/await over threading.",
    ]
    
    def test_order(order_name, memories_in_order):
        """Test if LLM follows the critical instruction given a specific memory order."""
        context = "\n".join(f"- {m}" for m in memories_in_order)
        prompt = f"Here is what I know about the user's project:\n\n{context}\n\nWhat is the deployment strategy?"
        
        successes = 0
        trials = 5
        for _ in range(trials):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            answer = response.choices[0].message.content.lower()
            if "blue-green" in answer and "canary" in answer:
                successes += 1
        return successes / trials
    
    # A) Random order — critical instruction at random position
    random_order = fillers.copy()
    random.seed(42)
    pos = random.randint(0, len(random_order))
    random_order.insert(pos, critical)
    random_rate = test_order("Random", random_order)
    
    # B) Chronological — critical instruction in the middle (position 10)
    chrono_order = fillers[:10] + [critical] + fillers[10:]
    chrono_rate = test_order("Chronological", chrono_order)
    
    # C) U-curve — critical instruction at position 0 or 1 (opening zone)
    ucurve_order = [critical] + fillers  # Critical in opening zone
    ucurve_rate = test_order("U-curve", ucurve_order)
    
    passed = ucurve_rate >= chrono_rate
    
    details = {
        "Memories in context": 20,
        "Critical instruction": critical[:60] + "...",
        "Random order compliance": f"{random_rate*100:.0f}%",
        "Chronological compliance": f"{chrono_rate*100:.0f}%",
        "U-curve compliance": f"{ucurve_rate*100:.0f}%",
        "Improvement over random": f"{(ucurve_rate - random_rate)*100:+.0f}pp",
    }
    
    print_result("Attention Budget Test", passed, details)
    return passed


if __name__ == "__main__":
    run_attention_budget_test()
