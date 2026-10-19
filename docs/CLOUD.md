# MenteDB Cloud

The managed version. No servers, no cron, no LLM keys to wire up, and the
overnight maintenance runs for you. Three steps.

## 1. Get an API key

Sign up at [app.mentedb.com](https://app.mentedb.com), open
**API Keys**, and create one. It looks like `mdb_...` and is shown once, copy it
somewhere safe.

## 2. Set it

```bash
export MENTEDB_API_KEY=mdb_your_key_here
```

That is the only credential you need. The endpoint defaults to
`https://api.mentedb.com`; the MCP connector reads `MENTEDB_API_URL` if you
ever need to point it somewhere else.

## 3. Connect

**AI coding tools, one command:**

```bash
npx mentedb-mcp@latest setup claude-code   # or: cursor, copilot, claude
```

With `MENTEDB_API_KEY` set it connects to the cloud automatically. Your assistant
now calls `process_turn` every turn, persistent memory across sessions, no glue
code.

**Or call the API directly:**

```bash
curl -X POST https://api.mentedb.com/v1/process_turn \
  -H "Authorization: Bearer $MENTEDB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_message":"...","assistant_response":"...","turn_id":0}'
```

**From Python** (any HTTP client, no SDK needed):

```python
import os, requests

resp = requests.post(
    "https://api.mentedb.com/v1/process_turn",
    headers={"Authorization": f"Bearer {os.environ['MENTEDB_API_KEY']}"},
    json={
        "user_message": "I switched from Postgres to SQLite",
        "assistant_response": "Noted.",
        "turn_id": 0,
    },
)
print(resp.json())
```

**Building your own agent?** The snippets above are the primitive. See
[Build an agent](./BUILD_AN_AGENT.md) for the full turn-by-turn loop in Python
and Node, cloud and self-hosted.

## What runs for you

In the background, every few hours, the platform runs the upkeep a self-hoster
would have to schedule: it distills recent turns into facts, merges near
duplicates, decays salience, and forgets what has gone cold, and once a day it
rebuilds your profile from the facts that survive. You do nothing. That is the
whole point of cloud.

---

Prefer to run it yourself? See [Self-hosting](./SELF_HOSTING.md).
