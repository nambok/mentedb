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
        # We'll use the Python SDK directly
        try:
            import mentedb
            self.db = mentedb.MenteDb(self.tmp_dir)
        except ImportError:
            print("mentedb Python package not installed. Install with: pip install mentedb")
            print("Or build from source: cd sdks/python && maturin develop")
            sys.exit(1)
    
    def store(self, content, memory_type="semantic", tags=None, agent_id=None):
        """Store a memory and return its ID."""
        return self.db.store(content, memory_type=memory_type, tags=tags or [])
    
    def search(self, query, limit=10):
        """Search memories by semantic similarity."""
        return self.db.search(query, limit=limit)
    
    def relate(self, from_id, to_id, edge_type, weight=1.0):
        """Create a relationship between memories."""
        return self.db.relate(from_id, to_id, edge_type, weight=weight)
    
    def get(self, memory_id):
        """Get a memory by ID."""
        return self.db.get_memory(memory_id)
    
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
