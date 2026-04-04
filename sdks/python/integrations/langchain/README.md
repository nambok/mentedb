# mentedb-langchain

MenteDB integration for LangChain and LangGraph. Gives your agents persistent, cognitive memory that goes beyond simple vector retrieval.

## Installation

```bash
pip install mentedb-langchain
```

## Components

### MenteDBMemory

A LangChain compatible memory backend that stores conversation context in MenteDB. Unlike buffer or summary memory, MenteDBMemory uses hybrid search (vector similarity, tag filtering, temporal decay) to assemble the most relevant context for each turn.

```python
from mentedb_langchain import MenteDBMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

memory = MenteDBMemory(
    data_dir="./agent-memory",
    agent_id="my-agent",
    token_budget=4096,
)

chain = ConversationChain(
    llm=ChatOpenAI(),
    memory=memory,
)

chain.predict(input="What database should I use for time series data?")
chain.predict(input="Tell me more about that recommendation")
```

### MenteDBRetriever

A LangChain compatible retriever that uses MenteDB hybrid search. Supports optional tag filtering and agent scoping to narrow results.

```python
from mentedb_langchain import MenteDBRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

retriever = MenteDBRetriever(
    data_dir="./agent-memory",
    k=10,
    tags=["backend", "architecture"],
)

chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever,
)

chain.invoke("What were the key decisions about our database migration?")
```

### MenteDBChatHistory

Persistent chat history with cognitive tracking. MenteDB stores messages alongside reasoning trajectories, knowledge gaps, and contradiction signals so the agent's memory improves over time.

```python
from mentedb_langchain import MenteDBChatHistory

history = MenteDBChatHistory(
    session_id="session-123",
    data_dir="./agent-memory",
)

history.add_user_message("What database should I use?")
history.add_ai_message("I recommend PostgreSQL for your use case.")

messages = history.messages
```

## Usage with LangGraph

MenteDB works naturally with LangGraph. Use `MenteDBMemory` as a checkpointer or context source within graph nodes:

```python
from mentedb_langchain import MenteDBMemory

memory = MenteDBMemory(data_dir="./graph-memory", agent_id="planner")

def plan_node(state):
    context = memory.load_memory_variables({"input": state["task"]})
    # Use context to inform planning
    return {**state, "context": context}

def reflect_node(state):
    memory.save_context(
        inputs={"input": state["task"]},
        outputs={"output": state["result"]},
    )
    return state
```

## Configuration

All components accept `data_dir` to specify where MenteDB stores its data. For multi agent setups, use `agent_id` to isolate each agent's memory space.

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `./mentedb-data` | Path to the MenteDB data directory |
| `agent_id` | `None` | Optional agent identifier for memory isolation |
| `token_budget` | `4096` | Maximum tokens for assembled context (MenteDBMemory) |
| `k` | `10` | Number of results to return (MenteDBRetriever) |
| `tags` | `None` | Tag filter for retrieval (MenteDBRetriever) |

## License

Apache 2.0
