"""Hosted client for MenteDB Cloud (the managed service).

Talks to the managed service at https://api.mentedb.com with an ``mdb_`` API key.
No native engine, no local database, nothing to run. The verbs mirror the embedded
``MenteDB`` so code moves between self-hosted and hosted by swapping the constructor::

    from mentedb import MenteDBClient

    client = MenteDBClient(api_key="mdb_...")   # get a key at https://app.mentedb.com

    # Turn 0: tell it something.
    client.process_turn("I switched from Postgres to SQLite for side projects", "Noted.", 0)

    # Turn 1: it remembers.
    result = client.process_turn("what database am I using for side projects?", "", 1)
    for memory in result.context:
        print(memory.content)

Only the Python standard library is used, so this adds no dependencies to the package.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Optional

DEFAULT_BASE_URL = "https://api.mentedb.com"


class MenteDBError(RuntimeError):
    """Raised when the hosted API returns an error."""


def _ns(obj: Any) -> Any:
    """Recursively turn dicts into attribute-accessible namespaces.

    Keeps hosted results shaped like the embedded engine's, so ``result.context``
    items expose ``.content`` the same way in both.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


class MenteDBClient:
    """Client for MenteDB Cloud.

    Args:
        api_key: your ``mdb_`` key from https://app.mentedb.com.
        base_url: override the API host (defaults to https://api.mentedb.com).
        timeout: per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ):
        if not api_key:
            raise ValueError(
                "api_key is required (get one at https://app.mentedb.com)"
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---- internal ------------------------------------------------------

    def _post(self, path: str, payload: dict) -> Any:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise MenteDBError(f"mentedb {exc.code}: {detail}") from None

    def _call_tool(self, name: str, arguments: dict) -> Any:
        """Call an MCP tool over the hosted JSON-RPC endpoint.

        The tool result comes back as MCP text content; when it is JSON (as the
        memory tools return) it is parsed, otherwise the raw text is returned.
        """
        resp = self._post(
            "/mcp",
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        if isinstance(resp, dict) and resp.get("error"):
            raise MenteDBError(f"tool {name}: {resp['error']}")
        result = resp.get("result", {}) if isinstance(resp, dict) else {}
        blocks = result.get("content", []) if isinstance(result, dict) else []
        text = "\n".join(
            b.get("text", "") for b in blocks if b.get("type") == "text"
        )
        if isinstance(result, dict) and result.get("isError"):
            raise MenteDBError(f"tool {name}: {text}")
        try:
            return _ns(json.loads(text))
        except (ValueError, TypeError):
            return text

    # ---- primary API ---------------------------------------------------

    def process_turn(
        self,
        user_message: str,
        assistant_response: Optional[str] = None,
        turn_id: int = 0,
        project_context: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """Process one conversation turn through the managed cognitive pipeline.

        Mirrors the embedded ``MenteDB.process_turn``. One call stores the turn,
        runs hybrid recall, extracts facts, and returns attention-ordered
        ``context`` for your next prompt (each item exposes ``.content``), plus the
        other fields the service reports.

        ``agent_id`` and ``user_id`` are orthogonal owner scopes: recall for one
        (user_id, agent_id) never returns another owner's memories.
        """
        payload: dict[str, Any] = {
            "user_message": user_message,
            "assistant_response": assistant_response or "",
            "turn_id": turn_id,
        }
        if project_context is not None:
            payload["project_context"] = project_context
        if agent_id is not None:
            payload["agent_id"] = agent_id
        if session_id is not None:
            payload["session_id"] = session_id
        if user_id is not None:
            payload["user_id"] = user_id
        return _ns(self._post("/v1/process_turn", payload))

    def search(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
    ):
        """Semantic search over stored memories."""
        args: dict[str, Any] = {"query": query, "limit": limit}
        if memory_type is not None:
            args["memory_type"] = memory_type
        return self._call_tool("search_memories", args)

    def store(
        self,
        content: str,
        memory_type: str = "semantic",
        tags: Optional[list[str]] = None,
        scope: Optional[str] = None,
    ):
        """Store a single memory. The result includes its ``memory_id``."""
        args: dict[str, Any] = {"content": content, "memory_type": memory_type}
        if tags is not None:
            args["tags"] = tags
        if scope is not None:
            args["scope"] = scope
        return self._call_tool("store_memory", args)

    def store_multimodal(
        self,
        data: str,
        media_type: str,
        memory_type: str = "semantic",
        tags: Optional[list[str]] = None,
        scope: Optional[str] = None,
    ):
        """Store text extracted from a base64 image or PDF.

        ``media_type`` is one of image/png, image/jpeg, image/webp, image/gif,
        application/pdf. Only the extracted text is stored; the raw file is not.
        """
        args: dict[str, Any] = {
            "data": data,
            "media_type": media_type,
            "memory_type": memory_type,
        }
        if tags is not None:
            args["tags"] = tags
        if scope is not None:
            args["scope"] = scope
        return self._call_tool("store_memory_multimodal", args)

    def forget(self, memory_id: str, reason: Optional[str] = None):
        """Delete a memory by id."""
        args: dict[str, Any] = {"id": memory_id}
        if reason is not None:
            args["reason"] = reason
        return self._call_tool("forget_memory", args)
