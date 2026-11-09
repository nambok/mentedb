"""Provenance through the Python SDK: written on store, returned on reads,
filtered on search, and erase by session.

Requires the native extension (``maturin develop``); skipped without it.
"""

import math

import pytest

mentedb = pytest.importorskip("mentedb._mentedb_python")


def _vec(seed: int, dim: int = 384) -> list[float]:
    # Distinct, non degenerate vectors: the erase scan looks for the erased
    # vector's bytes, and long runs of zeros in one hot vectors can line up
    # across neighbouring index entries.
    return [math.sin((i + 1) * (seed + 1.37)) for i in range(dim)]


def test_store_read_filter_and_erase_session(tmp_path):
    from mentedb import MenteDB

    db = MenteDB(str(tmp_path / "db"))
    in_session = db.store(
        "Written during the planning session",
        memory_type="semantic",
        embedding=_vec(1),
        session_id="plan-1",
        source="chat",
        source_ref="msg-9",
    )
    outside = db.store(
        "Written somewhere else entirely",
        memory_type="semantic",
        embedding=_vec(0),
    )

    p = db.get_memory(in_session)["provenance"]
    assert p["session_id"] == "plan-1"
    assert p["source"] == "chat"
    assert p["source_ref"] == "msg-9"
    assert p["derived_from"] == []
    assert db.get_memory(outside)["provenance"]["source"] == "api"
    assert db.memory_ids_in_session("plan-1") == [in_session]

    hits = db.search_text("planning", k=5, session_id="plan-1")
    assert [h.id for h in hits] == [in_session]

    receipt = db.erase_session("plan-1")
    assert receipt["session_id"] == "plan-1"
    assert receipt["requested"] == [in_session]
    assert receipt["verification"]["residual_found"] is False
    assert db.memory_ids_in_session("plan-1") == []
    assert db.get_memory(outside)["content"] == "Written somewhere else entirely"
    db.close()
