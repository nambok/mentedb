# mentedb-langchain

MenteDB helpers for LangChain and LangGraph: a conversation memory, a
retriever, and a chat history, each backed by a local MenteDB database.

## Installation

```bash
pip install mentedb-langchain
```

## Status

`MenteDBMemory`, `MenteDBRetriever` and `MenteDBChatHistory` follow the method
names of LangChain's memory, retriever and chat history interfaces, but they do
not subclass LangChain's base classes. LangChain rejects them when they are
passed directly to chains such as `ConversationChain(memory=...)` or
`RetrievalQA.from_chain_type(retriever=...)` (checked with LangChain 0.3).
Call their methods from your own code or graph nodes, as shown below.

Only one MenteDB instance can open a data directory at a time. Give each
component its own `data_dir`, or share one object. `agent_id`, when set, must
be a UUID string.

## Components

### MenteDBMemory

Stores each turn as an episodic memory and loads the most similar stored
memories (top 10 by similarity search) for the next input.

```python
from mentedb_langchain import MenteDBMemory

memory = MenteDBMemory(data_dir="./agent-memory")

memory.save_context(
    inputs={"input": "What database should I use for time series data?"},
    outputs={"output": "TimescaleDB is a good fit."},
)
context = memory.load_memory_variables({"input": "Tell me more about that database"})
# -> {"history": "User: ...\nAssistant: ..."}
```

### MenteDBRetriever

Similarity search with an optional tag filter (all listed tags must match).
Results are dicts shaped like LangChain documents.

```python
from mentedb_langchain import MenteDBRetriever

retriever = MenteDBRetriever(data_dir="./docs-memory", k=10, tags=["backend"])
docs = retriever.get_relevant_documents("database migration decisions")
# -> [{"page_content": ..., "metadata": {"id", "score", "memory_type", "tags"}}]
```

### MenteDBChatHistory

Stores every message in MenteDB as an episodic memory tagged with the session
id. `messages` returns the messages added through this object as
`{"role", "content"}` dicts; it does not reload earlier sessions from disk.

```python
from mentedb_langchain import MenteDBChatHistory

history = MenteDBChatHistory(session_id="session-123", data_dir="./chat-memory")

history.add_user_message("What database should I use?")
history.add_ai_message("I recommend PostgreSQL for your use case.")

messages = history.messages
```

## Usage with LangGraph

Use `MenteDBMemory` as a context source inside graph nodes:

```python
from mentedb_langchain import MenteDBMemory

memory = MenteDBMemory(data_dir="./graph-memory")

def plan_node(state):
    context = memory.load_memory_variables({"input": state["task"]})
    return {**state, "context": context}

def reflect_node(state):
    memory.save_context(
        inputs={"input": state["task"]},
        outputs={"output": state["result"]},
    )
    return state
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `./mentedb-data` | Path to the MenteDB data directory |
| `agent_id` | `None` | Optional agent UUID stored as the memory owner (not used to filter searches) |
| `token_budget` | `4096` | Accepted by MenteDBMemory but currently not applied |
| `k` | `10` | Number of results to return (MenteDBRetriever) |
| `tags` | `None` | Tag filter for retrieval (MenteDBRetriever) |

## License

Apache 2.0
