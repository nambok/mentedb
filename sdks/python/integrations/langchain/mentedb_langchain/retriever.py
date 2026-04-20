"""LangChain retriever backed by MenteDB vector search."""
from typing import List, Optional

from mentedb import MenteDB


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
        self._db = MenteDB(data_dir)

    def get_relevant_documents(self, query: str) -> list:
        """Retrieve relevant memories as dicts with id, content, and score."""
        results = self._db.search_text(query, k=self.k, tags=self.tags)
        docs = []
        for r in results:
            mem = self._db.get_memory(r.id)
            if mem:
                docs.append({
                    "page_content": mem["content"],
                    "metadata": {
                        "id": r.id,
                        "score": r.score,
                        "memory_type": mem.get("memory_type", ""),
                        "tags": mem.get("tags", []),
                    },
                })
        return docs

    async def aget_relevant_documents(self, query: str) -> list:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query)
