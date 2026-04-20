"""CrewAI tool for querying MenteDB with MQL."""

from mentedb import MenteDB


class MenteDBTool:
    """A CrewAI tool that lets agents query their memory database directly.

    Usage:
        from crewai import Agent
        from mentedb_crewai import MenteDBTool

        memory_tool = MenteDBTool(data_dir="./crew-memory")

        agent = Agent(
            role="Researcher",
            tools=[memory_tool],
        )
    """

    name: str = "mentedb_query"
    description: str = (
        "Query the team's knowledge database using MQL (Mente Query Language). "
        "Use for recalling past decisions, finding related context, or checking "
        "what the team knows about a topic."
    )

    def __init__(self, data_dir: str = "./mentedb-data"):
        self.data_dir = data_dir
        self._db = MenteDB(data_dir)

    def run(self, query: str) -> str:
        """Execute an MQL query and return assembled context."""
        result = self._db.recall(query)
        return result.text if result else ""
