"""CrewAI memory backend powered by MenteDB."""
from typing import Any, Dict, List, Optional

from mentedb import MenteDB


class MenteDBCrewMemory:
    """Drop-in memory backend for CrewAI agents.

    Gives CrewAI agents persistent, cognitive memory with:
    - Cross-session recall
    - Multi-agent memory isolation via spaces
    - Semantic search over past context

    Usage:
        from crewai import Agent
        from mentedb_crewai import MenteDBCrewMemory

        memory = MenteDBCrewMemory(
            data_dir="./crew-memory",
            space="research-team",
        )

        agent = Agent(
            role="Researcher",
            memory=memory,
        )
    """

    def __init__(self, data_dir: str = "./mentedb-data", space: str = "default",
                 agent_name: Optional[str] = None):
        self.data_dir = data_dir
        self.space = space
        self.agent_name = agent_name
        self._db = MenteDB(data_dir)

    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory from the agent's work."""
        tags = [self.space]
        if self.agent_name:
            tags.append(self.agent_name)
        if metadata and "tags" in metadata:
            tags.extend(metadata["tags"])
        memory_type = (metadata or {}).get("memory_type", "episodic")
        return self._db.store(content, memory_type=memory_type, tags=tags)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        results = self._db.search_text(query, k=k, tags=[self.space])
        docs = []
        for r in results:
            mem = self._db.get_memory(r.id)
            if mem:
                docs.append({
                    "content": mem["content"],
                    "score": r.score,
                    "id": r.id,
                    "memory_type": mem.get("memory_type", ""),
                })
        return docs

    def get_context(self, task_description: str, token_budget: int = 2000) -> str:
        """Get assembled context for a task using MQL recall."""
        result = self._db.recall(
            f"RECALL memories WHERE content ~> \"{task_description}\" LIMIT 10"
        )
        return result.text if result else ""
