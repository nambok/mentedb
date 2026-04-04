from enum import Enum


class MemoryType(str, Enum):
    """Types of memories that MenteDB can store."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    ANTI_PATTERN = "anti_pattern"
    REASONING = "reasoning"
    CORRECTION = "correction"


class EdgeType(str, Enum):
    """Types of edges (relationships) between memories."""

    CAUSED = "caused"
    BEFORE = "before"
    RELATED = "related"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    SUPERSEDES = "supersedes"
    DERIVED = "derived"
    PART_OF = "part_of"


class RecallResult:
    """Result of a recall query."""

    def __init__(self, text: str, total_tokens: int, memory_count: int):
        self.text = text
        self.total_tokens = total_tokens
        self.memory_count = memory_count

    def __repr__(self) -> str:
        return f"RecallResult(tokens={self.total_tokens}, memories={self.memory_count})"


class SearchResult:
    """A single vector similarity search hit."""

    def __init__(self, id: str, score: float):
        self.id = id
        self.score = score

    def __repr__(self) -> str:
        return f"SearchResult(id='{self.id}', score={self.score:.4f})"
