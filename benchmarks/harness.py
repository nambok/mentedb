"""Shared benchmark utilities for MenteDB quality benchmarks."""

import subprocess
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Path to the mentedb-mcp binary (build if needed)
MCP_BINARY = os.environ.get("MENTEDB_MCP", "mentedb-mcp")

def ensure_binary():
    """Build the MCP server if not found."""
    # Try to find it
    result = subprocess.run(["which", MCP_BINARY], capture_output=True)
    if result.returncode != 0:
        print("Building mentedb-mcp...")
        subprocess.run(["cargo", "build", "--release", "-p", "mentedb-mcp"], 
                       cwd=Path(__file__).parent.parent / "mentedb-mcp",
                       check=True)

class MenteDBBenchmark:
    """Benchmark harness that uses MenteDB via its Rust library directly."""
    
    def __init__(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="mentedb-bench-")
        try:
            import mentedb
            self.db = mentedb.MenteDB(self.tmp_dir)
        except ImportError:
            print("mentedb Python package not installed. Install with: pip install mentedb")
            print("Or build from source: cd sdks/python && maturin develop")
            sys.exit(1)
    
    def store(self, content, memory_type="semantic", tags=None, agent_id=None):
        """Store a memory and return its ID."""
        return self.db.store(content, memory_type=memory_type, tags=tags or [])
    
    def search(self, query, limit=10):
        """Search memories using the real HNSW engine via search_text.
        
        Uses hash embedding similarity search through the Rust engine.
        Superseded and contradicted memories are filtered out natively
        by the engine's graph-aware recall_similar.
        """
        try:
            results = self.db._db.search_text(query, limit)
            return [(r.id, r.score) for r in results]
        except Exception:
            return []
    
    def relate(self, from_id, to_id, edge_type, weight=1.0):
        """Create a relationship between memories."""
        return self.db.relate(from_id, to_id, edge_type, weight=weight)
    
    def get(self, memory_id):
        """Get a memory by ID from the engine."""
        try:
            return self.db._db.get_memory(memory_id)
        except Exception:
            return {"content": ""}
    
    def forget(self, memory_id):
        """Delete a memory."""
        return self.db.forget(memory_id)

    def cleanup(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


def print_result(test_name, passed, details):
    """Print a formatted benchmark result."""
    status = "PASS" if passed else "FAIL"
    print(f"\n{'='*60}")
    print(f"  {test_name}: {status}")
    print(f"{'='*60}")
    for key, value in details.items():
        print(f"  {key}: {value}")
    print()


def has_openai_key():
    return bool(os.environ.get("OPENAI_API_KEY"))


def has_anthropic_key():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def has_llm_key():
    return has_openai_key() or has_anthropic_key()


def get_llm_client():
    """Return (client, provider_name) for whichever LLM key is available.
    Prefers Anthropic if both are set."""
    if has_anthropic_key():
        try:
            import anthropic
            return anthropic.Anthropic(), "anthropic"
        except ImportError:
            print("  anthropic package not installed: pip install anthropic")
    if has_openai_key():
        try:
            import openai
            return openai.OpenAI(), "openai"
        except ImportError:
            print("  openai package not installed: pip install openai")
    return None, None


def llm_chat(client, provider, prompt, temperature=0.0, max_tokens=200, json_mode=False):
    """Unified chat completion across providers. Returns the response text."""
    if provider == "openai":
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    elif provider == "anthropic":
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["messages"][0]["content"] = prompt + "\n\nRespond with valid JSON only. No markdown fences."
        response = client.messages.create(**kwargs)
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0].strip()
        return text
    else:
        raise ValueError(f"Unknown provider: {provider}")
