# mentedb-crewai

MenteDB integration for CrewAI and AutoGen. Gives your multi agent teams persistent, cognitive memory that survives across sessions and tracks reasoning quality over time.

## Installation

```bash
pip install mentedb-crewai
```

## Components

### MenteDBCrewMemory

A memory backend for CrewAI agents that stores and retrieves memories through MenteDB. Each agent can have its own memory space, or agents on the same team can share a common space for collaborative recall.

```python
from crewai import Agent, Crew, Task
from mentedb_crewai import MenteDBCrewMemory

memory = MenteDBCrewMemory(
    data_dir="./crew-memory",
    space="research-team",
)

researcher = Agent(
    role="Senior Researcher",
    goal="Find comprehensive information on the topic",
    memory=memory,
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear documentation from research",
    memory=MenteDBCrewMemory(
        data_dir="./crew-memory",
        space="research-team",
        agent_name="writer",
    ),
)

task = Task(
    description="Research and document the latest trends in vector databases",
    agent=researcher,
)

crew = Crew(agents=[researcher, writer], tasks=[task])
crew.kickoff()
```

Memory is stored persistently, so the next time the crew runs it can recall findings from previous sessions.

### MenteDBTool

A tool that lets agents query MenteDB directly using MQL (Mente Query Language). Attach it to any agent so the agent can search and retrieve from the team's knowledge base during task execution.

```python
from crewai import Agent
from mentedb_crewai import MenteDBTool

memory_tool = MenteDBTool(data_dir="./crew-memory")

agent = Agent(
    role="Analyst",
    goal="Analyze data using historical context",
    tools=[memory_tool],
)
```

The agent can then invoke the tool during its reasoning to recall past decisions, find related context, or check what the team already knows about a topic.

## Usage with AutoGen

MenteDB works with AutoGen agents through the same memory and tool interfaces. Use `MenteDBCrewMemory` to give AutoGen agents persistent memory, or `MenteDBTool` to let them query the knowledge base.

```python
from mentedb_crewai import MenteDBCrewMemory, MenteDBTool

memory = MenteDBCrewMemory(data_dir="./autogen-memory", space="dev-team")
tool = MenteDBTool(data_dir="./autogen-memory")

# Store context from agent work
memory.store("The team decided to use PostgreSQL for the main database.")

# Search for relevant memories
results = memory.search("database decision", k=5)

# Get assembled context within a token budget
context = memory.get_context("Plan the database migration", token_budget=2000)
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `./mentedb-data` | Path to the MenteDB data directory |
| `space` | `default` | Memory space name for agent isolation or sharing |
| `agent_name` | `None` | Optional agent name for per agent memory scoping |

## License

Apache 2.0
