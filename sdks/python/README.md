# MenteDB Python SDK

Python bindings for [MenteDB](https://github.com/nambok/mentedb), the mind
database for AI agents. Built with PyO3 and maturin for native Rust
performance from Python.

## Installation

### From source (development)

```bash
cd sdks/python
pip install maturin
maturin develop
```

### From PyPI (once published)

```bash
pip install mentedb
```

## Quick start

```python
from mentedb import MenteDB

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

    # Sleeptime enrichment runs automatically after process_turn:
    # - Extracts semantic facts from conversations
    # - Links and deduplicates entities
    # - Builds community summaries and user profile
    # Requires LLM config: MENTEDB_OPENAI_API_KEY or MENTEDB_ANTHROPIC_API_KEY

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

## Cognitive features

The SDK also exposes MenteDB cognitive subsystems for real time stream
monitoring, conversation trajectory tracking, and pain signal management.

## Sleeptime Enrichment

MenteDB automatically enriches memories in the background after `process_turn`. The pipeline extracts semantic facts, links and deduplicates entities, groups them into communities with summaries, and builds a user profile — all feeding back into future `process_turn` context retrieval.

Requires an LLM provider: set `MENTEDB_OPENAI_API_KEY` or `MENTEDB_ANTHROPIC_API_KEY`. Without one, the engine works normally — enrichment just doesn't run.

## Cognitive subsystems

```python
from mentedb._mentedb_python import CognitionStream, TrajectoryTracker, PainRegistry

# Stream monitoring
stream = CognitionStream(buffer_size=500)
stream.feed_token("The")
stream.feed_token(" sky")
alerts = stream.check_alerts([("some-uuid", "the sky is blue")])

# Trajectory tracking
tracker = TrajectoryTracker(max_turns=50)
tracker.record_turn("deployment", "investigating", ["which env?"])
context = tracker.get_resume_context()

# Pain registry
pain = PainRegistry(max_warnings=3)
pain.record_pain("some-uuid", 0.8, ["timeout", "deploy"], "deploy timed out")
warnings = pain.check_triggers(["deploy"])
```

## API reference

### MenteDB

| Method | Description |
|--------|-------------|
| `process_turn(user_message, assistant_response, turn_id, project_context, agent_id)` | **Primary API.** Process a conversation turn through the full cognitive pipeline |
| `store(content, memory_type, embedding, agent_id, tags)` | Store a memory, returns its UUID |
| `recall(query)` | Recall memories using MQL |
| `search(embedding, k)` | Vector similarity search |
| `relate(source, target, edge_type, weight)` | Add a relationship |
| `forget(memory_id)` | Remove a memory |
| `close()` | Flush and close the database |

### Types

`MemoryType`: episodic, semantic, procedural, anti_pattern, reasoning, correction

`EdgeType`: caused, before, related, contradicts, supports, supersedes, derived, part_of

## License

Apache 2.0. See the repository root LICENSE file for details.
