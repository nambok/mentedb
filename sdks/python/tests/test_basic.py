"""Basic MenteDB Python SDK tests.

These tests verify the Python binding layer compiles and the API surface is correct.
They require the native extension to be built with maturin first.
"""


def test_imports():
    """Verify all public types are importable."""
    from mentedb import MenteDB, MemoryType, EdgeType

    assert MemoryType.EPISODIC.value == "episodic"
    assert EdgeType.CAUSED.value == "caused"


def test_memory_types():
    from mentedb.types import MemoryType

    assert len(MemoryType) == 6
    expected = {
        "episodic",
        "semantic",
        "procedural",
        "anti_pattern",
        "reasoning",
        "correction",
    }
    assert {m.value for m in MemoryType} == expected


def test_edge_types():
    from mentedb.types import EdgeType

    assert len(EdgeType) == 8
    expected = {
        "caused",
        "before",
        "related",
        "contradicts",
        "supports",
        "supersedes",
        "derived",
        "part_of",
    }
    assert {e.value for e in EdgeType} == expected


def test_recall_result():
    from mentedb.types import RecallResult

    r = RecallResult(text="hello", total_tokens=10, memory_count=1)
    assert r.text == "hello"
    assert r.total_tokens == 10
    assert r.memory_count == 1
    assert "tokens=10" in repr(r)


def test_search_result():
    from mentedb.types import SearchResult

    s = SearchResult(id="abc-123", score=0.95)
    assert s.id == "abc-123"
    assert abs(s.score - 0.95) < 1e-6
    assert "abc-123" in repr(s)
