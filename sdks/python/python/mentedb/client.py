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
        created_at: int | None = None,
    ) -> str:
        """Store a memory and return its UUID.

        Optionally set created_at (microsecond timestamp) to place the memory
        at a specific point in time for temporal filtering.
        """
        if isinstance(memory_type, MemoryType):
            memory_type = memory_type.value
        return self._db.store(content, memory_type, embedding, agent_id, tags, created_at)

    def recall(self, query: str):
        """Recall memories using an MQL query string."""
        return self._db.recall(query)

    def search(self, embedding: list[float], k: int = 10):
        """Vector similarity search. Returns a list of SearchResult."""
        return self._db.search(embedding, k)

    def search_text(self, query: str, k: int = 10, tags: list[str] | None = None,
                    after: int | None = None, before: int | None = None):
        """Text-based similarity search using the configured embedding provider.

        Optionally filter by tags (AND) and/or time range (microsecond timestamps).
        """
        return self._db.search_text(query, k, tags, after, before)

    def search_multi(self, queries: list[str], k: int = 10, tags: list[str] | None = None,
                     before: int | None = None):
        """Multi-query search with Reciprocal Rank Fusion.

        Searches for each query separately and merges results via RRF
        for broader recall across different semantic aspects.
        Optionally filters by tags and/or time (before timestamp in microseconds).
        """
        return self._db.search_multi(queries, k, tags, before)

    def search_expanded(self, query: str, k: int = 10, provider: str | None = None,
                        tags: list[str] | None = None, before: int | None = None):
        """Expanded search with engine-native query decomposition.

        Uses the engine's LLM to decompose the query into sub-queries,
        then runs multi-query RRF search for broader recall.
        Optionally filters by tags and/or time (before timestamp in microseconds).
        """
        return self._db.search_expanded(query, k, provider, tags, before)

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

    def ingest(
        self,
        conversation: str,
        provider: str | None = None,
        agent_id: str | None = None,
    ) -> dict:
        """Extract memories from a conversation using the LLM pipeline and store them.

        Requires MENTEDB_LLM_PROVIDER and MENTEDB_LLM_API_KEY env vars,
        or pass provider ("openai", "anthropic", "ollama") explicitly.

        Returns a dict with memories_stored, rejected_low_quality, stored_ids, etc.
        """
        return self._db.ingest(conversation, provider, agent_id)

    def extract(
        self,
        conversation: str,
        provider: str | None = None,
    ) -> list[dict]:
        """Extract memories from a conversation without storing them.

        Returns a list of dicts with content, memory_type, tags, confidence,
        embedding_key. Use with store_extracted() for parallel workflows.
        The GIL is released during the HTTP call.
        """
        return self._db.extract(conversation, provider)

    def store_extracted(
        self,
        memories: list[dict],
        agent_id: str | None = None,
    ) -> dict:
        """Store pre-extracted memories (from extract()) into the database.

        Returns a dict with stored_ids.
        """
        return self._db.store_extracted(memories, agent_id)

    def close(self):
        """Flush and close the database."""
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
