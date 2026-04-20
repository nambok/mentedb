from mentedb.types import MemoryType, EdgeType, RecallResult, SearchResult

try:
    from mentedb.client import MenteDB
except ImportError:
    MenteDB = None  # type: ignore[assignment,misc]

__version__ = "0.3.1"
__all__ = ["MenteDB", "MemoryType", "EdgeType", "RecallResult", "SearchResult"]
