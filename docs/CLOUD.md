# MenteDB Cloud

The managed version. No servers, no cron, no LLM keys to wire up, and the
overnight maintenance runs for you. Three steps.

## 1. Get an API key

Sign up at [app.mentedb.com](https://app.mentedb.com) (email or GitHub), open
**API Keys**, and create one. It looks like `mdb_...` and is shown once, copy it
somewhere safe.

## 2. Set it

```bash
export MENTEDB_API_KEY=mdb_your_key_here
```

That is the only credential you need. The endpoint defaults to
`https://api.mentedb.com`; override it with `MENTEDB_CLOUD_URL` if you ever need
to.

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
curl -X POST https://api.mentedb.com/mcp/v1/tools/call \
  -H "Authorization: Bearer $MENTEDB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"process_turn","arguments":{"user_message":"...","assistant_response":"...","turn_id":0}}'
```

Python and TypeScript SDKs are available too, see the [docs site](https://mentedb.com/docs).

## What runs for you

Every night the platform runs the same maintenance a self-hoster would have to
schedule, consolidation, decay, archival, and cache eviction, plus enrichment
and profile building. You do nothing. That is the whole point of cloud.

---

Prefer to run it yourself? See [Self-hosting](./SELF_HOSTING.md).
