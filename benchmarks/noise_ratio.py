"""
The Noise Ratio Test — Extraction quality comparison.

Ingests 10 sample conversations. Counts stored memories.
Uses LLM to evaluate what percentage are actually useful.

Requires OPENAI_API_KEY.
"""

import os
import json
from harness import MenteDBBenchmark, print_result, has_openai_key

SAMPLE_CONVERSATIONS = [
    """User: Hey, I'm building a REST API for my startup.
Assistant: Great! What tech stack are you considering?
User: I'm going with Python and FastAPI. We need to handle about 1000 requests per second.
Assistant: FastAPI is excellent for that throughput. Would you like help with the architecture?
User: Yes, and we need to use PostgreSQL for the database. Oh and the API needs to be versioned.
Assistant: I'd recommend URL-based versioning with /v1/ prefix. For PostgreSQL, consider using SQLAlchemy with async support.""",
    
    """User: We need to add authentication to our API.
Assistant: There are several approaches. What are your requirements?
User: OAuth2 with Google login. And we need role-based access control with admin and user roles.
Assistant: I'd suggest using python-jose for JWT tokens with Google OAuth2.
User: Sounds good. Also, rate limiting is important. 100 requests per minute per user.
Assistant: You could use slowapi or a Redis-based solution for distributed rate limiting.""",
    
    """User: I'm having issues with our deployment pipeline.
Assistant: What's your current setup?
User: GitHub Actions for CI, but deployments to AWS ECS keep failing.
Assistant: Common issues include Docker build caching and ECS task definition updates.
User: Actually the problem is our Docker images are too large. They're 2GB.
Assistant: Switch to multi-stage builds with python:3.11-slim as the final stage. Should get you under 200MB.""",
    
    """User: How should we handle background tasks?
Assistant: Celery is the most popular choice for Python.
User: We tried Celery but it's complex. Any simpler alternatives?
Assistant: Consider arq (async Redis queue) or dramatiq. Both are simpler than Celery.
User: Let's go with dramatiq. We already have Redis running.
Assistant: Good choice. Dramatiq with Redis is straightforward to set up.""",
    
    """User: We need to add real-time features.
Assistant: WebSockets or Server-Sent Events?
User: WebSockets for bidirectional communication. Using it for live notifications.
Assistant: FastAPI has built-in WebSocket support. You'll need a connection manager.
User: What about scaling WebSockets across multiple instances?
Assistant: Use Redis pub/sub as a message broker between instances.""",
]

def run_noise_ratio_test():
    if not has_openai_key():
        print("\n  [SKIP] Noise Ratio Test: Set OPENAI_API_KEY to run")
        return None
    
    try:
        import openai
        client = openai.OpenAI()
    except ImportError:
        print("\n  [SKIP] Noise Ratio Test: pip install openai")
        return None
    
    bench = MenteDBBenchmark()
    
    try:
        # Store each conversation turn as individual memories (naive approach)
        naive_memories = []
        for conv in SAMPLE_CONVERSATIONS:
            for line in conv.strip().split("\n"):
                line = line.strip()
                if line:
                    mid = bench.store(line, memory_type="episodic")
                    naive_memories.append((mid, line))
        
        # Also store extracted facts (MenteDB approach) 
        extracted_memories = []
        for conv in SAMPLE_CONVERSATIONS:
            # Use LLM to extract key facts
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Extract the key technical decisions and user preferences from this conversation as a JSON array of strings. Only include actionable facts, not chit-chat:\n\n{conv}"}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            try:
                facts = json.loads(response.choices[0].message.content)
                if isinstance(facts, dict):
                    facts = facts.get("facts", facts.get("decisions", list(facts.values())[0] if facts else []))
                for fact in facts:
                    if isinstance(fact, str) and len(fact) > 10:
                        mid = bench.store(fact, memory_type="semantic", tags=["extracted"])
                        extracted_memories.append((mid, fact))
            except:
                pass
        
        # Evaluate usefulness with LLM
        def evaluate_usefulness(memories):
            if not memories:
                return 0.0
            sample = memories[:30]  # Cap at 30
            items = "\n".join(f"{i+1}. {content[:100]}" for i, (_, content) in enumerate(sample))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Rate each memory below as USEFUL or NOISE for an AI assistant helping with this software project. A memory is USEFUL if it contains an actionable technical decision, user preference, or project requirement. Return a JSON object with 'useful_count' and 'total'.\n\nMemories:\n{items}"}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            try:
                result = json.loads(response.choices[0].message.content)
                useful = result.get("useful_count", 0)
                total = result.get("total", len(sample))
                return useful / total if total > 0 else 0.0
            except:
                return 0.5
        
        naive_ratio = evaluate_usefulness(naive_memories)
        extracted_ratio = evaluate_usefulness(extracted_memories)
        
        passed = extracted_ratio > naive_ratio
        
        details = {
            "Conversations processed": len(SAMPLE_CONVERSATIONS),
            "Naive approach (store every line)": f"{len(naive_memories)} memories, {naive_ratio*100:.0f}% useful",
            "MenteDB approach (extract facts)": f"{len(extracted_memories)} memories, {extracted_ratio*100:.0f}% useful",
            "Memory reduction": f"{(1 - len(extracted_memories)/len(naive_memories))*100:.0f}% fewer memories" if naive_memories else "N/A",
            "Quality improvement": f"{(extracted_ratio - naive_ratio)*100:+.0f}pp usefulness",
        }
        
        print_result("Noise Ratio Test", passed, details)
        return passed
        
    finally:
        bench.cleanup()


if __name__ == "__main__":
    run_noise_ratio_test()
