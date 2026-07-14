"""Integration tests for relate() against the native extension.

Guards against wrapper/native signature drift. The native binding requires
bi-temporal valid_from/valid_until arguments; the high-level wrapper must supply
them (defaulting to None = valid for all time). A regression here previously
raised `TypeError: relate() missing 2 required positional arguments` for every
caller of the wrapper. Skipped unless the extension is built (maturin develop).
"""

import tempfile

import pytest


def _open_db():
    try:
        from mentedb import MenteDB
    except Exception:  # native extension not built
        pytest.skip("native extension not built (run: maturin develop)")
    try:
        return MenteDB(tempfile.mkdtemp(prefix="mentedb-test-"))
    except Exception as e:
        pytest.skip(f"could not open native db: {e}")


def test_relate_without_temporal_args():
    """relate() must work without temporal args (the drift regression case)."""
    db = _open_db()
    a = db.store("The user prefers PostgreSQL.")
    b = db.store("The user prefers SQLite.")
    db.relate(b, a, "supersedes")
    db.relate(b, a, "contradicts", weight=0.9)


def test_relate_with_temporal_bounds():
    """relate() accepts explicit bi-temporal validity bounds."""
    db = _open_db()
    a = db.store("The user lives in Seattle.")
    b = db.store("The user lives in Austin.")
    db.relate(b, a, "supersedes", valid_from=1000)
    db.relate(b, a, "related", valid_from=1000, valid_until=2000)
