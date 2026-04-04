"""LangChain retriever backed by MenteDB vector search."""
from typing import List, Optional


class MenteDBRetriever:
    """LangChain compatible retriever using MenteDB hybrid search.

    Combines vector similarity, tag filtering, and temporal relevance
    in a single retrieval call.

    Usage:
        from mentedb_langchain import MenteDBRetriever

        retriever = MenteDBRetriever(
            data_dir="./agent-memory",
            k=10,
            tags=["backend", "architecture"],
        )

        docs = retriever.get_relevant_documents("database migration strategy")
    """

    def __init__(self, data_dir: str = "./mentedb-data", k: int = 10,
                 tags: Optional[List[str]] = None, agent_id: Optional[str] = None):
        self.data_dir = data_dir
        self.k = k
        self.tags = tags
        self.agent_id = agent_id

    def get_relevant_documents(self, query: str) -> list:
        """Retrieve relevant memories as LangChain Documents."""
        # Would build MQL query with optional tag filters
        # RECALL memories WHERE content ~> query AND tag IN tags LIMIT k
        return []

    async def aget_relevant_documents(self, query: str) -> list:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query)
