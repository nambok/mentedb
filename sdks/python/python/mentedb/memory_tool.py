"""Backend for the client side memory tool (``memory_20250818``), stored in MenteDB.

The model sends file style commands (``view``, ``create``, ``str_replace``,
``insert``, ``delete``, ``rename``) against a ``/memories`` directory. Your
tool loop forwards each ``tool_use`` input to ``POST /v1/memory_tool`` and
copies the reply into a ``tool_result`` block. :class:`MemoryTool` does both
steps. The same contract is served by a self-hosted ``mentedb-server`` and by
MenteDB Cloud, so only the base URL and credential change::

    import anthropic
    from mentedb import MemoryTool

    memory = MemoryTool("http://localhost:6677", agent_id="...")
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": "Remember that I prefer tabs."}]
    while True:
        response = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=16000,
            tools=[MemoryTool.TOOL],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        messages.append({"role": "user", "content": memory.results(response.content)})

Only the Python standard library is used, so this adds no dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Mapping, Optional

from mentedb.hosted import MenteDBError

#: Default address of a self-hosted ``mentedb-server``.
DEFAULT_SERVER_URL = "http://localhost:6677"

#: Tool definition to pass in ``tools`` on a Messages API request.
MEMORY_TOOL_DEFINITION = {"type": "memory_20250818", "name": "memory"}

_TOOL_INPUT_FIELDS = (
    "command",
    "path",
    "view_range",
    "file_text",
    "old_str",
    "new_str",
    "insert_line",
    "insert_text",
    "old_path",
    "new_path",
)


def _field(obj: Any, name: str) -> Any:
    """Read ``name`` from a dict or from an SDK model object."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


class MemoryTool:
    """Runs memory tool calls against a MenteDB server.

    Args:
        base_url: the server, e.g. ``http://localhost:6677`` for a self-hosted
            ``mentedb-server`` or ``https://api.mentedb.com`` for MenteDB Cloud.
        token: bearer token (a server JWT, or an ``mdb_`` key for the cloud).
            Leave unset for a server running without auth.
        database: a named database on a self-hosted server. Unset uses the
            default database.
        agent_id: whose files to act on. A token bound to an agent always acts
            as that agent; this is for unbound or admin tokens and for servers
            without auth. Files of one agent are never visible to another.
        user_id: optionally split one agent's files further by user.
        timeout: per request timeout in seconds.
    """

    #: Tool definition to pass in ``tools`` on a Messages API request.
    TOOL = MEMORY_TOOL_DEFINITION
    #: The tool name the model uses in ``tool_use`` blocks.
    NAME = MEMORY_TOOL_DEFINITION["name"]

    def __init__(
        self,
        base_url: str = DEFAULT_SERVER_URL,
        token: Optional[str] = None,
        *,
        database: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.agent_id = agent_id
        self.user_id = user_id
        self.timeout = timeout
        if database:
            self.path = f"/v1/db/{urllib.parse.quote(database, safe='')}/memory_tool"
        else:
            self.path = "/v1/memory_tool"

    def execute(self, tool_input: Any) -> dict:
        """Run one command and return ``{"content": str, "is_error": bool}``.

        ``tool_input`` is the ``input`` of a memory ``tool_use`` block, as a
        dict or SDK object. Command failures the model can fix (a missing file,
        a bad range, an invalid path) come back with ``is_error`` set. Auth and
        server failures raise :class:`MenteDBError`.
        """
        if isinstance(tool_input, Mapping):
            payload = {k: v for k, v in tool_input.items() if v is not None}
        else:
            payload = {
                k: _field(tool_input, k)
                for k in _TOOL_INPUT_FIELDS
                if _field(tool_input, k) is not None
            }
        if self.agent_id is not None:
            payload["agent_id"] = self.agent_id
        if self.user_id is not None:
            payload["user_id"] = self.user_id

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(
            f"{self.base_url}{self.path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise MenteDBError(f"mentedb {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise MenteDBError(f"mentedb unreachable at {self.base_url}: {exc.reason}") from None
        if not isinstance(body, dict) or "content" not in body:
            raise MenteDBError(f"unexpected memory tool reply: {body!r}")
        return {"content": str(body["content"]), "is_error": bool(body.get("is_error"))}

    def handle(self, tool_use: Any) -> dict:
        """Turn one memory ``tool_use`` block into its ``tool_result`` block."""
        result = self.execute(_field(tool_use, "input") or {})
        block = {
            "type": "tool_result",
            "tool_use_id": _field(tool_use, "id"),
            "content": result["content"],
        }
        if result["is_error"]:
            block["is_error"] = True
        return block

    def results(self, content: Iterable[Any]) -> list:
        """``tool_result`` blocks for every memory call in an assistant turn.

        Pass ``response.content``; blocks that are not memory ``tool_use``
        calls are skipped. Return all results in one user message.
        """
        return [
            self.handle(block)
            for block in content
            if _field(block, "type") == "tool_use" and _field(block, "name") == self.NAME
        ]
