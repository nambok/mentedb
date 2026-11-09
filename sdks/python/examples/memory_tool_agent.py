"""A Claude tool loop whose memory tool files live in MenteDB.

Start a server (``mentedb-server --data-dir ./data``), then::

    pip install anthropic mentedb
    MENTEDB_URL=http://localhost:6677 python memory_tool_agent.py "Remember that I deploy on Fridays."

``MENTEDB_TOKEN`` is the server JWT (or an ``mdb_`` key with
``MENTEDB_URL=https://api.mentedb.com``). ``MENTEDB_AGENT_ID`` picks whose
``/memories`` tree to use when the token is not bound to an agent.
"""

import os
import sys

import anthropic

from mentedb import MemoryTool

memory = MemoryTool(
    os.environ.get("MENTEDB_URL", "http://localhost:6677"),
    os.environ.get("MENTEDB_TOKEN"),
    agent_id=os.environ.get("MENTEDB_AGENT_ID"),
)
client = anthropic.Anthropic()


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What do you remember about me?"
    messages = [{"role": "user", "content": prompt}]
    while True:
        response = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=16000,
            tools=[MemoryTool.TOOL],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            if response.stop_reason == "refusal":
                print("The request was declined.", file=sys.stderr)
            return
        # Every memory call in this turn, answered in one user message.
        messages.append({"role": "user", "content": memory.results(response.content)})


if __name__ == "__main__":
    main()
