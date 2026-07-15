"""Tests for the hosted MenteDB Cloud client (MenteDBClient).

The hosted client is pure standard library, so these load it directly from source
and mock urllib. No native extension or network is required. The mocked request and
response shapes match the gateway contract: POST /v1/process_turn for the turn loop,
and POST /mcp with a JSON-RPC tools/call whose result carries MCP text content.
"""

import importlib.util
import io
import json
import os
import urllib.error
from unittest import mock

_HOSTED = os.path.join(
    os.path.dirname(__file__), "..", "python", "mentedb", "hosted.py"
)
_spec = importlib.util.spec_from_file_location("mentedb_hosted", _HOSTED)
hosted = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hosted)

MenteDBClient = hosted.MenteDBClient
MenteDBError = hosted.MenteDBError


class _FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_requires_api_key():
    try:
        MenteDBClient(api_key="")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_process_turn_posts_to_rest_and_parses_context():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(
            {
                "ok": True,
                "context": [
                    {"content": "the sky is blue", "memory_type": "semantic"}
                ],
                "stored": 1,
            }
        )

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        client = MenteDBClient(api_key="mdb_test")
        result = client.process_turn("what color is the sky?", "", 1, project_context="u:1")

    assert captured["url"] == "https://api.mentedb.com/v1/process_turn"
    assert captured["method"] == "POST"
    assert captured["headers"]["authorization"] == "Bearer mdb_test"
    assert captured["body"]["user_message"] == "what color is the sky?"
    assert captured["body"]["turn_id"] == 1
    assert captured["body"]["project_context"] == "u:1"
    # context items are attribute-accessible, mirroring the embedded engine
    assert result.context[0].content == "the sky is blue"
    assert result.stored == 1


def test_call_tool_parses_json_text_block():
    def fake_urlopen(req, timeout=None):
        assert req.full_url == "https://api.mentedb.com/mcp"
        body = json.loads(req.data.decode("utf-8"))
        assert body["method"] == "tools/call"
        assert body["params"]["name"] == "store_memory"
        assert body["params"]["arguments"]["content"] == "hello world"
        return _FakeResp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [
                        {"type": "text", "text": '{"stored": true, "memory_id": "abc-123"}'}
                    ],
                    "isError": False,
                },
            }
        )

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        client = MenteDBClient(api_key="mdb_test")
        res = client.store("hello world", memory_type="semantic")

    assert res.stored is True
    assert res.memory_id == "abc-123"


def test_tool_error_raises():
    def fake_urlopen(req, timeout=None):
        return _FakeResp(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [{"type": "text", "text": "over memory limit"}],
                    "isError": True,
                },
            }
        )

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        client = MenteDBClient(api_key="mdb_test")
        try:
            client.store("x")
            assert False, "expected MenteDBError"
        except MenteDBError as exc:
            assert "over memory limit" in str(exc)


def test_http_error_raises_mentedb_error():
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}')
        )

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        client = MenteDBClient(api_key="mdb_bad")
        try:
            client.process_turn("hi")
            assert False, "expected MenteDBError"
        except MenteDBError as exc:
            assert "401" in str(exc)


if __name__ == "__main__":
    # Allow running without pytest installed.
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok {_name}")
    print("all hosted client tests passed")
