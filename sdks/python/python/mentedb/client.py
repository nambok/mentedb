"""High-level Python client for MenteDB."""

from __future__ import annotations

from typing import Optional

from mentedb._mentedb_python import MenteDB as _MenteDB
from mentedb.types import EdgeType, MemoryType


class MenteDB:
    """The mind database for AI agents.

    Use as a context manager or call ``close()`` explicitly when done::

        with MenteDB("./data") as db:
            mid = db.store("the sky is blue")

    For real semantic search, pass an embedding provider::

        with MenteDB("./data", embedding_provider="openai", embedding_api_key="sk-...") as db:
            mid = db.store("the sky is blue")
            results = db.search_text("sky color", k=5)
    """

    def __init__(
        self,
        data_dir: str = "./mentedb-data",
        embedding_provider: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model: str | None = None,
    ):
        self._db = _MenteDB(
            data_dir,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_model=embedding_model,
        )

    def store(
        self,
        content: str,
        memory_type: str | MemoryType = MemoryType.EPISODIC,
        embedding: list[float] | None = None,
        agent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Store a memory and return its UUID."""
        if isinstance(memory_type, MemoryType):
            memory_type = memory_type.value
        return self._db.store(content, memory_type, embedding, agent_id, tags)

    def recall(self, query: str):
        """Recall memories using an MQL query string."""
        return self._db.recall(query)

    def search(self, embedding: list[float], k: int = 10):
        """Vector similarity search. Returns a list of SearchResult."""
        return self._db.search(embedding, k)

    def search_text(self, query: str, k: int = 10):
        """Text-based similarity search using the configured embedding provider."""
        return self._db.search_text(query, k)

    def get_memory(self, memory_id: str):
        """Retrieve a single memory by its UUID."""
        return self._db.get_memory(memory_id)

    def relate(
        self,
        source: str,
        target: str,
        edge_type: str | EdgeType = EdgeType.RELATED,
        weight: float = 1.0,
    ):
        """Add a relationship between two memories."""
        if isinstance(edge_type, EdgeType):
            edge_type = edge_type.value
        self._db.relate(source, target, edge_type, weight)

    def forget(self, memory_id: str):
        """Remove a memory by its UUID."""
        self._db.forget(memory_id)

    def close(self):
        """Flush and close the database."""
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
