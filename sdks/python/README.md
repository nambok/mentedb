# MenteDB Python SDK

Python bindings for [MenteDB](https://github.com/nambok/mentedb), the mind
database for AI agents. Built with PyO3 and maturin for native Rust
performance from Python.

## Installation

### From PyPI

```bash
pip install mentedb
```

Prebuilt wheels are published for macOS arm64, Linux x86_64 and aarch64, and
Windows x86_64 (Python 3.9+).

### From source (development)

```bash
cd sdks/python
pip install maturin
maturin develop
```

On Debian and Ubuntu systems pip refuses system wide installs (PEP 668). Use a virtual environment or pipx:

```bash
python3 -m venv .venv && .venv/bin/pip install mentedb
# or: pipx install mentedb
```

## Quick start

```python
from mentedb import MenteDB, MemoryType, EdgeType

with MenteDB("./agent-memory") as db:
    # process_turn — the primary API, one call does everything
    result = db.process_turn(
        user_message="The deployment failed because the config was missing",
        assistant_response="I'll check the config setup.",
        turn_id=0,
    )
    # result.context — relevant memories for your prompt
    # result.facts_extracted — what was learned this turn
    # result.contradiction_count — conflicting beliefs detected
    # Optional kwargs: project_context, agent_id, session_id (tags stored
    # turns so injection recall can exclude the requesting session), user_id
    # (agent_id and user_id must be UUID strings)

    # Enrichment (LLM fact extraction, entity linking, community summaries,
    # user profile) is a separate call; see "Sleeptime enrichment" below.

    # Or use low-level APIs directly:

    # Store a memory
    mid = db.store(
        "The deployment failed because the config was missing",
        memory_type=MemoryType.EPISODIC,
        tags=["deployment", "config"],
    )

    # Recall memories with MQL
    result = db.recall("RECALL tag:deployment LIMIT 5")
    print(result.text)

    # Vector similarity search
    hits = db.search(embedding=[0.1] * 384, k=5)
    for hit in hits:
        print(f"{hit.id}: {hit.score:.4f}")

    # Relate memories
    mid2 = db.store("Always validate config before deploy", memory_type=MemoryType.PROCEDURAL)
    db.relate(mid, mid2, edge_type=EdgeType.CAUSED)

    # Forget a memory
    db.forget(mid)
```

## MenteDB Cloud (hosted)

Prefer a managed API with no engine to run? Use `MenteDBClient` with an `mdb_`
key from [app.mentedb.com](https://app.mentedb.com). The verbs mirror the
embedded client, so only the constructor changes.

```python
from mentedb import MenteDBClient

client = MenteDBClient(api_key="mdb_...")

# Turn 0: tell it something.
client.process_turn("I switched from Postgres to SQLite for side projects", "Noted.", 0)

# Turn 1: it remembers.
result = client.process_turn("what database am I using for side projects?", "", 1)
for memory in result.context:
    print(memory.content)

# Also available: client.search(...), client.store(...),
# client.store_multimodal(data, media_type=...), client.forget(memory_id)
```

`MenteDBClient` is pure standard library (no extra dependencies) and talks to
`https://api.mentedb.com`. `process_turn` uses the REST endpoint, while `store`,
`search`, `store_multimodal`, and `forget` use the hosted MCP tools.

## Cognitive features

The SDK also exposes MenteDB cognitive subsystems for real time stream
monitoring, conversation trajectory tracking, and pain signal management.

## Sleeptime Enrichment

The embedded SDK does not enrich on its own. `process_turn` marks when
enrichment is due; check `db.needs_enrichment()` and call
`db.run_enrichment()` (for example from a background worker). The pipeline
extracts semantic facts, links and deduplicates entities, groups them into
communities with summaries, and builds a user profile, all of which feed later
`process_turn` context retrieval.

Requires an LLM provider: set `MENTEDB_LLM_PROVIDER` (`openai`, `anthropic` or
`ollama`) and `MENTEDB_LLM_API_KEY`. Without them the engine works normally
and `run_enrichment` raises an error.

## Cognitive subsystems

```python
import uuid
from mentedb._mentedb_python import CognitionStream, TrajectoryTracker, PainRegistry

# Stream monitoring (fact ids must be UUID strings)
fact_id = str(uuid.uuid4())
stream = CognitionStream(buffer_size=500)
stream.feed_token("The")
stream.feed_token(" sky")
alerts = stream.check_alerts([(fact_id, "the sky is blue")])

# Trajectory tracking
tracker = TrajectoryTracker(max_turns=50)
tracker.record_turn("deployment", "investigating", ["which env?"])
context = tracker.get_resume_context()

# Pain registry
pain = PainRegistry(max_warnings=3)
pain.record_pain(str(uuid.uuid4()), 0.8, ["timeout", "deploy"], "deploy timed out")
warnings = pain.check_triggers(["deploy"])
```

## API reference

### MenteDB

| Method | Description |
|--------|-------------|
| `process_turn(user_message, assistant_response, turn_id, project_context, agent_id, session_id, user_id)` | **Primary API.** Process a conversation turn through the full cognitive pipeline |
| `store(content, memory_type, embedding, agent_id, tags, created_at)` | Store a memory, returns its UUID |
| `recall(query)` | Recall memories using MQL |
| `search(embedding, k)` | Vector similarity search |
| `search_text(query, k, tags, after, before)` | Similarity search with the configured embedding provider |
| `get_memory(memory_id)` | Fetch one memory as a dict |
| `relate(source, target, edge_type, weight, valid_from, valid_until)` | Add a relationship |
| `forget(memory_id)` | Remove a memory |
| `needs_enrichment()` / `run_enrichment(provider, current_turn)` | Check for and run sleeptime enrichment (needs an LLM) |
| `close()` | Flush and close the database |

Only one `MenteDB` instance can hold a data directory at a time; opening the
same directory again raises a "locked by another process" error.

### Types

`MemoryType`: episodic, semantic, procedural, anti_pattern, reasoning, correction

`EdgeType`: caused, before, related, contradicts, supports, supersedes, derived, part_of

## License

Apache 2.0. See the repository root LICENSE file for details.
