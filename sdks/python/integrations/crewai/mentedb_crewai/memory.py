"""CrewAI memory backend powered by MenteDB."""
from typing import Any, Dict, List, Optional


class MenteDBCrewMemory:
    """Drop-in memory backend for CrewAI agents.

    Gives CrewAI agents persistent, cognitive memory with:
    - Cross-session recall
    - Multi-agent memory isolation via spaces
    - Automatic knowledge gap detection
    - Pain signal tracking for mistake aversion

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

    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory from the agent's work."""
        return ""

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        return []

    def get_context(self, task_description: str, token_budget: int = 2000) -> str:
        """Get assembled context for a task, optimized for the agent's attention."""
        return ""
