"""Cognitive feature tests for MenteDB Python SDK.

These tests verify the cognitive Python types and their string-based API.
Integration tests that exercise the native extension require maturin build first.
"""


def test_cognitive_imports():
    """Verify cognitive types are importable once the extension is built."""
    # Pure-Python types are always available.
    from mentedb.types import MemoryType

    assert MemoryType.ANTI_PATTERN.value == "anti_pattern"


def test_memory_type_is_str_enum():
    """MemoryType members can be used as plain strings."""
    from mentedb.types import MemoryType

    assert isinstance(MemoryType.EPISODIC, str)
    assert MemoryType.EPISODIC == "episodic"


def test_edge_type_is_str_enum():
    """EdgeType members can be used as plain strings."""
    from mentedb.types import EdgeType

    assert isinstance(EdgeType.RELATED, str)
    assert EdgeType.RELATED == "related"
