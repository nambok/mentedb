"""Tests for the memory tool backend helper (MemoryTool).

The helper is pure standard library, so these import it from source without the
native extension. The mocked tests check the request each call sends and how
replies become ``tool_result`` blocks. The round trip test runs every command
against a real ``mentedb-server`` when a built binary is found (set
``MENTEDB_SERVER_BIN``, or build it with ``cargo build -p mentedb-server``), and
is skipped otherwise.
"""

import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from mentedb.hosted import MenteDBClient, MenteDBError  # noqa: E402
from mentedb.memory_tool import MEMORY_TOOL_DEFINITION, MemoryTool  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class _FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _capture(reply):
    captured = []

    def fake_urlopen(req, timeout=None):
        captured.append(
            {
                "url": req.full_url,
                "method": req.get_method(),
                "headers": {k.lower(): v for k, v in req.header_items()},
                "body": json.loads(req.data.decode("utf-8")),
            }
        )
        return _FakeResp(reply)

    return captured, fake_urlopen


def test_tool_definition():
    assert MEMORY_TOOL_DEFINITION == {"type": "memory_20250818", "name": "memory"}
    assert MemoryTool.TOOL is MEMORY_TOOL_DEFINITION


def test_execute_posts_tool_input_with_scope():
    captured, fake = _capture({"content": "File created successfully at: /memories/a.md", "is_error": False})
    with mock.patch("urllib.request.urlopen", fake):
        tool = MemoryTool("http://localhost:6677/", "jwt-token", agent_id="agent-1", user_id="user-1")
        out = tool.execute({"command": "create", "path": "/memories/a.md", "file_text": "hi", "view_range": None})
    assert out == {"content": "File created successfully at: /memories/a.md", "is_error": False}
    req = captured[0]
    assert req["url"] == "http://localhost:6677/v1/memory_tool"
    assert req["method"] == "POST"
    assert req["headers"]["authorization"] == "Bearer jwt-token"
    assert req["body"] == {
        "command": "create",
        "path": "/memories/a.md",
        "file_text": "hi",
        "agent_id": "agent-1",
        "user_id": "user-1",
    }


def test_named_database_and_no_token():
    captured, fake = _capture({"content": "ok", "is_error": False})
    with mock.patch("urllib.request.urlopen", fake):
        MemoryTool(database="team a").execute({"command": "view", "path": "/memories"})
    assert captured[0]["url"] == "http://localhost:6677/v1/db/team%20a/memory_tool"
    assert "authorization" not in captured[0]["headers"]
    assert captured[0]["body"] == {"command": "view", "path": "/memories"}


def test_results_turns_memory_calls_into_tool_results():
    captured, fake = _capture({"content": "The path /memories/x does not exist. Please provide a valid path.", "is_error": True})
    content = [
        SimpleNamespace(type="text", text="Let me check my memory."),
        SimpleNamespace(type="tool_use", id="toolu_1", name="memory", input={"command": "view", "path": "/memories/x"}),
        {"type": "tool_use", "id": "toolu_2", "name": "other_tool", "input": {}},
    ]
    with mock.patch("urllib.request.urlopen", fake):
        blocks = MemoryTool().results(content)
    assert len(captured) == 1
    assert blocks == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "The path /memories/x does not exist. Please provide a valid path.",
            "is_error": True,
        }
    ]


def test_success_result_has_no_error_flag():
    _, fake = _capture({"content": "Successfully deleted /memories/a.md", "is_error": False})
    with mock.patch("urllib.request.urlopen", fake):
        block = MemoryTool().handle({"id": "toolu_9", "input": {"command": "delete", "path": "/memories/a.md"}})
    assert block == {"type": "tool_result", "tool_use_id": "toolu_9", "content": "Successfully deleted /memories/a.md"}


def test_http_errors_raise():
    def fake(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b'{"error":"token is read-only"}'))

    with mock.patch("urllib.request.urlopen", fake):
        try:
            MemoryTool().execute({"command": "create", "path": "/memories/a", "file_text": ""})
            assert False, "expected MenteDBError"
        except MenteDBError as exc:
            assert "403" in str(exc) and "read-only" in str(exc)


def test_cloud_client_hands_out_a_memory_tool():
    captured, fake = _capture({"content": "ok", "is_error": False})
    with mock.patch("urllib.request.urlopen", fake):
        MenteDBClient(api_key="mdb_test").memory_tool().execute({"command": "view", "path": "/memories"})
    assert captured[0]["url"] == "https://api.mentedb.com/v1/memory_tool"
    assert captured[0]["headers"]["authorization"] == "Bearer mdb_test"


# ---- real server round trip ------------------------------------------------


def _server_binary():
    explicit = os.environ.get("MENTEDB_SERVER_BIN")
    if explicit:
        return explicit
    for profile in ("debug", "release"):
        candidate = os.path.join(_REPO, "target", profile, "mentedb-server")
        if os.path.exists(candidate):
            return candidate
    return None


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(binary, data_dir):
    port, grpc_port = _free_port(), _free_port()
    proc = subprocess.Popen(
        [binary, "--data-dir", data_dir, "--port", str(port), "--grpc-port", str(grpc_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/v1/health", timeout=1):
                return proc, url
        except Exception:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("mentedb-server did not start")


def test_round_trip_against_a_real_server():
    binary = _server_binary()
    if binary is None:
        try:
            import pytest

            pytest.skip("no mentedb-server binary; set MENTEDB_SERVER_BIN")
        except ImportError:
            print("skip: no mentedb-server binary")
            return
    with tempfile.TemporaryDirectory() as data_dir:
        proc, url = _start_server(binary, data_dir)
        try:
            alice = MemoryTool(url, agent_id="00000000-0000-0000-0000-000000000001")
            bob = MemoryTool(url, agent_id="00000000-0000-0000-0000-000000000002")

            def call(tool, **tool_input):
                return tool.execute(tool_input)

            assert call(alice, command="create", path="/memories/notes.md", file_text="alpha\nbeta\n") == {
                "content": "File created successfully at: /memories/notes.md",
                "is_error": False,
            }
            assert call(alice, command="view", path="/memories/notes.md")["content"] == (
                "Here's the content of /memories/notes.md with line numbers:\n     1\talpha\n     2\tbeta"
            )
            assert not call(alice, command="str_replace", path="/memories/notes.md", old_str="beta", new_str="gamma")["is_error"]
            assert not call(alice, command="insert", path="/memories/notes.md", insert_line=0, insert_text="top")["is_error"]
            assert not call(alice, command="rename", old_path="/memories/notes.md", new_path="/memories/done/notes.md")["is_error"]
            assert call(alice, command="view", path="/memories/done/notes.md", view_range=[1, -1])["content"].endswith(
                "     1\ttop\n     2\talpha\n     3\tgamma"
            )
            listing = call(alice, command="view", path="/memories")["content"]
            assert listing.endswith("\t/memories/done/notes.md")

            # Another agent sees nothing of it.
            assert call(bob, command="view", path="/memories")["content"].endswith("\n0B\t/memories")
            assert call(bob, command="view", path="/memories/done/notes.md")["is_error"]

            # Traversal is refused as a tool error.
            bad = call(alice, command="create", path="/memories/../escape", file_text="x")
            assert bad["is_error"] and bad["content"].startswith("Error: Invalid path")

            # A tool loop turn: tool_use blocks in, tool_result blocks out.
            blocks = alice.results(
                [{"type": "tool_use", "id": "toolu_1", "name": "memory", "input": {"command": "delete", "path": "/memories/done"}}]
            )
            assert blocks == [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Successfully deleted /memories/done"}]
            assert call(alice, command="view", path="/memories")["content"].endswith("\n0B\t/memories")
        finally:
            proc.terminate()
            proc.wait(timeout=15)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok {_name}")
    print("all memory tool tests passed")
