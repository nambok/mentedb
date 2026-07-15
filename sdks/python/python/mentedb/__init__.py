from mentedb.types import MemoryType, EdgeType, RecallResult, SearchResult

try:
    from mentedb.client import MenteDB
except ImportError:
    MenteDB = None  # type: ignore[assignment,misc]

# Hosted client for MenteDB Cloud. Pure stdlib, so it imports without the native
# engine: self-host embedded uses MenteDB, the managed API uses MenteDBClient.
from mentedb.hosted import MenteDBClient, MenteDBError

__version__ = "0.3.1"
__all__ = [
    "MenteDB",
    "MenteDBClient",
    "MenteDBError",
    "MemoryType",
    "EdgeType",
    "RecallResult",
    "SearchResult",
]
