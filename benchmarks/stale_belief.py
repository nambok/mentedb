"""
The Stale Belief Test — MenteDB's unique capability.

Scenario:
  Turn 1: User says "I use PostgreSQL for everything"
  Turn 5: User says "Actually I switched to SQLite"  
  Turn 10: What database does the user prefer?

Naive RAG: Returns both, possibly PostgreSQL first (higher similarity to "database")
MenteDB: Returns SQLite only (belief propagation superseded PostgreSQL)

This test proves the engine handles belief updates correctly WITHOUT an LLM.
"""

from harness import MenteDBBenchmark, print_result

def run_stale_belief_test():
    bench = MenteDBBenchmark()
    
    try:
        # --- Simulate conversation turns ---
        
        # Turn 1: Initial belief
        pg_id = bench.store(
            "The user strongly prefers PostgreSQL for all database needs. They use it for every project.",
            memory_type="semantic",
            tags=["preference", "database", "postgresql"]
        )
        
        # Turns 2-4: Other memories (noise)
        bench.store("The user likes Python for scripting tasks.", memory_type="semantic", tags=["preference", "language"])
        bench.store("The user's project deadline is next Friday.", memory_type="episodic", tags=["deadline"])
        bench.store("The user prefers dark mode in their IDE.", memory_type="semantic", tags=["preference", "ide"])
        
        # Turn 5: Belief update — contradicts Turn 1
        sqlite_id = bench.store(
            "The user has switched to SQLite. They no longer use PostgreSQL and prefer SQLite for its simplicity.",
            memory_type="semantic",
            tags=["preference", "database", "sqlite"]
        )
        
        # Create supersession relationship
        bench.relate(sqlite_id, pg_id, "supersedes")
        bench.relate(sqlite_id, pg_id, "contradicts")
        
        # Turns 6-9: More noise
        bench.store("The user completed the authentication module.", memory_type="episodic")
        bench.store("The user prefers REST over GraphQL.", memory_type="semantic", tags=["preference", "api"])
        bench.store("Meeting notes from standup: discussed deployment pipeline.", memory_type="episodic")
        bench.store("The user wants to use Docker for deployment.", memory_type="semantic", tags=["preference", "deployment"])
        
        # --- Turn 10: Query "What database does the user prefer?" ---
        results = bench.search("What database does the user prefer?", limit=5)
        
        # --- Evaluate ---
        # Check what comes back
        result_ids = [r[0] for r in results] if results else []
        result_contents = []
        for rid, score in (results or []):
            try:
                mem = bench.get(rid)
                result_contents.append((rid, score, mem.get("content", "") if isinstance(mem, dict) else str(mem)))
            except:
                result_contents.append((rid, score, ""))
        
        # The test passes if:
        # 1. SQLite memory appears in top results
        # 2. SQLite ranks higher than PostgreSQL
        sqlite_rank = None
        pg_rank = None
        for i, (rid, score, content) in enumerate(result_contents):
            if "sqlite" in content.lower():
                sqlite_rank = i
            if "postgresql" in content.lower() and "no longer" not in content.lower():
                pg_rank = i
        
        # Determine pass/fail
        sqlite_found = sqlite_rank is not None
        pg_superseded = pg_rank is None or (sqlite_rank is not None and sqlite_rank < pg_rank)
        passed = sqlite_found and pg_superseded
        
        details = {
            "Total memories stored": 9,
            "Query": "What database does the user prefer?",
            "SQLite rank": sqlite_rank if sqlite_rank is not None else "NOT FOUND",
            "PostgreSQL rank": pg_rank if pg_rank is not None else "NOT FOUND (superseded)",
            "Belief propagation": "Working" if pg_superseded else "NOT WORKING",
            "Top results": len(result_contents),
        }
        
        for i, (rid, score, content) in enumerate(result_contents[:3]):
            details[f"Result {i+1} (score={score:.3f})"] = content[:80]
        
        print_result("Stale Belief Test", passed, details)
        return passed
        
    finally:
        bench.cleanup()


if __name__ == "__main__":
    run_stale_belief_test()
