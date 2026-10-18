# mentedb-crewai

MenteDB helpers for CrewAI and AutoGen style multi agent setups: a memory class
that stores and searches team memories in MenteDB, and a tool class that runs
MQL queries.

## Installation

```bash
pip install mentedb-crewai
```

## Status

`MenteDBCrewMemory` and `MenteDBTool` are plain Python classes. They do not
subclass CrewAI's memory or `BaseTool` types, so CrewAI rejects them when they
are passed directly as `Agent(memory=...)` or `Agent(tools=[...])` (checked
with CrewAI 1.15). Call them from your own code, or wrap them in a CrewAI tool
yourself, as shown below.

Only one MenteDB instance can open a data directory at a time. Give every
`MenteDBCrewMemory` and `MenteDBTool` its own `data_dir`, or share one object
between agents.

## Components

### MenteDBCrewMemory

Stores memories tagged with a `space` name (and the optional `agent_name`) and
searches within that space.

```python
from mentedb_crewai import MenteDBCrewMemory

memory = MenteDBCrewMemory(data_dir="./crew-memory", space="dev-team")

# Store context from agent work
memory.store("The team decided to use PostgreSQL for the main database.")

# Search for relevant memories in this space
results = memory.search("database decision", k=5)
# -> [{"content": ..., "score": ..., "id": ..., "memory_type": ...}]

# Text of memories matching the task (MQL recall, up to 10 memories)
context = memory.get_context("Plan the database migration")
```

Memory is stored on disk, so a later run that opens the same `data_dir` can
search what earlier runs stored.

### MenteDBTool

Runs an MQL (Mente Query Language) query and returns the assembled text.

```python
from mentedb_crewai import MenteDBTool

memory_tool = MenteDBTool(data_dir="./tool-memory")
print(memory_tool.run("RECALL similar(\"database decision\") LIMIT 5"))
```

### Using it from a CrewAI agent

Wrap the call in a CrewAI tool:

```python
from crewai import Agent
from crewai.tools import tool
from mentedb_crewai import MenteDBCrewMemory

memory = MenteDBCrewMemory(data_dir="./crew-memory", space="research-team")

@tool("Team memory search")
def team_memory(query: str) -> str:
    """Search what the team already knows about a topic."""
    return "\n".join(r["content"] for r in memory.search(query, k=5))

analyst = Agent(
    role="Analyst",
    goal="Analyze data using historical context",
    backstory="Knows the team's past decisions.",
    tools=[team_memory],
)
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `./mentedb-data` | Path to the MenteDB data directory |
| `space` | `default` | Tag used to group and filter memories (MenteDBCrewMemory) |
| `agent_name` | `None` | Extra tag added to stored memories (MenteDBCrewMemory) |

`get_context` accepts a `token_budget` argument, but it is currently ignored.

## License

Apache 2.0
