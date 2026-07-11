# Build an agent with memory

MenteDB gives your agent long-term memory through one call: `process_turn`. Each
turn it stores what was said, retrieves what is relevant, and hands back the
memories to put in your next prompt. That loop is the whole integration.

This guide has a runnable example for every combination of **where MenteDB runs**
(cloud or self-hosted) and **your language** (Python, Node, Rust). Pick the row
that matches your stack and copy it.

## The loop

Every setup below is the same three lines wearing different clothes:

```text
each turn:
  1. build your prompt with the memories from last turn's `context`
  2. call your LLM, get a reply
  3. process_turn(user_message, reply)  ->  returns `context` for next turn
```

`process_turn` embeds the user message, runs hybrid recall (vector + keyword +
graph) over everything stored so far, saves the new turn, and returns the
relevant memories as `context`. On the first turn `context` is empty because
nothing has been learned yet. By turn two it carries the facts, preferences, and
corrections from earlier in the conversation, and across sessions.

Fact extraction and contradiction detection need an LLM. On **cloud** that is
wired up for you. When **self-hosting** you set `MENTEDB_LLM_PROVIDER` and
`MENTEDB_LLM_API_KEY`; without them you still get storage and recall, just no
automatic extraction. There is no `configure_llm()` call, it is environment or
platform config.

## Which example do I want?

| | Python | Node / TypeScript | Rust |
|---|---|---|---|
| **Cloud** (managed, no infra) | [REST + `requests`](#cloud--python) | [REST + `fetch`](#cloud--node) | [REST](#cloud--any-language) |
| **Self-hosted** (embedded, your disk) | [`pip install mentedb`](#self-hosted--python) | [`npm install mentedb`](#self-hosted--node) | [`cargo add mentedb`](#self-hosted--rust) |

Cloud talks to `https://api.mentedb.com` over REST with an `mdb_` API key, no SDK
to install and nothing to run. Self-hosting embeds the engine in your process
with data on local disk, you own the LLM key and the maintenance schedule (see
[Self-hosting](./SELF_HOSTING.md)). If your language is not listed, self-host and
call the server's REST API, shown [at the end](#self-hosted--any-language).

---

## Cloud + Python

No SDK. Get a key at [app.mentedb.com](https://app.mentedb.com), then call the
REST endpoint with any HTTP client.

```bash
export MENTEDB_API_KEY=mdb_your_key_here
pip install requests
```

```python
import os, requests

API = "https://api.mentedb.com/v1/process_turn"
HEADERS = {"Authorization": f"Bearer {os.environ['MENTEDB_API_KEY']}"}

def process_turn(user_message, assistant_response, turn_id):
    resp = requests.post(API, headers=HEADERS, json={
        "user_message": user_message,
        "assistant_response": assistant_response,
        "turn_id": turn_id,
    })
    resp.raise_for_status()
    # the response is the turn data directly
    return resp.json()

# Turn 0: tell it something.
process_turn("I switched from Postgres to SQLite for side projects", "Noted.", 0)

# Turn 1: it remembers.
result = process_turn("what database am I using for side projects?", "", 1)
for memory in result["context"]:
    print(memory["content"])
```

In a real agent, `result["context"]` is what you inject into your model prompt:

```python
history = []                                   # memories from the previous turn
for turn_id, user_message in enumerate(conversation):
    prompt = build_prompt(history, user_message)   # your prompt template
    reply = call_your_llm(prompt)                  # your model
    result = process_turn(user_message, reply, turn_id)
    history = result["context"]                    # feeds the next prompt
```

---

## Cloud + Node

Same REST endpoint, no SDK, use `fetch`.

```bash
export MENTEDB_API_KEY=mdb_your_key_here
```

```javascript
const API = "https://api.mentedb.com/v1/process_turn";

async function processTurn(userMessage, assistantResponse, turnId) {
  const resp = await fetch(API, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.MENTEDB_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_message: userMessage, assistant_response: assistantResponse, turn_id: turnId }),
  });
  if (!resp.ok) throw new Error(`mentedb ${resp.status}`);
  return await resp.json();
}

// Turn 0: tell it something. Turn 1: it remembers.
await processTurn("I switched from Postgres to SQLite for side projects", "Noted.", 0);
const result = await processTurn("what database am I using for side projects?", "", 1);
for (const memory of result.context) console.log(memory.content);
```

The agent loop is identical to Python: prepend `result.context` to your next
prompt, generate, call `processTurn`, repeat.

---

## Self-hosted + Python

The engine runs in your process. Data lives on local disk. Install the SDK:

```bash
pip install mentedb
# for automatic fact extraction, point it at an LLM:
export MENTEDB_LLM_PROVIDER=anthropic      # openai | anthropic | ollama
export MENTEDB_LLM_API_KEY=sk-ant-...
```

```python
from mentedb import MenteDB

db = MenteDB("./agent-memory")

# Turn 0: tell it something.
db.process_turn(
    user_message="I switched from Postgres to SQLite for side projects",
    assistant_response="Noted.",
    turn_id=0,
)

# Turn 1: it remembers.
result = db.process_turn(
    user_message="what database am I using for side projects?",
    turn_id=1,
)
for memory in result.context:
    print(memory.content)

# result.facts_extracted     -> facts learned this turn (needs an LLM)
# result.contradiction_count -> conflicts with existing beliefs
db.close()
```

Agent loop:

```python
history = []
for turn_id, user_message in enumerate(conversation):
    prompt = build_prompt(history, user_message)
    reply = call_your_llm(prompt)
    result = db.process_turn(user_message, reply, turn_id)
    history = result.context
```

---

## Self-hosted + Node

Same embedded engine, native Node binding. Install the SDK:

```bash
npm install mentedb
# for automatic fact extraction, point it at an LLM:
export MENTEDB_LLM_PROVIDER=anthropic      # openai | anthropic | ollama
export MENTEDB_LLM_API_KEY=sk-ant-...
```

```javascript
const { MenteDB } = require("mentedb");

const db = new MenteDB("./agent-memory");

// Turn 0: tell it something.
db.processTurn("I switched from Postgres to SQLite for side projects", "Noted.", 0);

// Turn 1: it remembers.
const result = db.processTurn("what database am I using for side projects?", undefined, 1);
for (const memory of result.context) console.log(memory.content);

// result.factsExtracted      -> facts learned this turn (needs an LLM)
// result.contradictionCount  -> conflicts with existing beliefs
db.close();
```

`processTurn(userMessage, assistantResponse?, turnId?, projectContext?, agentId?, sessionId?)`
is synchronous and returns the result object directly. Agent loop:

```javascript
let history = [];
for (let turnId = 0; turnId < conversation.length; turnId++) {
  const userMessage = conversation[turnId];
  const prompt = buildPrompt(history, userMessage);
  const reply = await callYourLlm(prompt);
  const result = db.processTurn(userMessage, reply, turnId);
  history = result.context;
}
```

---

## Self-hosted + Rust

Embed the engine directly, no server, no network.

```bash
cargo add mentedb
```

```rust
use mentedb::{MenteDb, process_turn::ProcessTurnInput};
use mentedb_context::DeltaTracker;

let db = MenteDb::open("./agent-memory")?;
let mut delta = DeltaTracker::default();

let result = db.process_turn(&ProcessTurnInput {
    user_message: "I switched from Postgres to SQLite for side projects".into(),
    assistant_response: Some("Noted.".into()),
    turn_id: 0,
    project_context: None,
    agent_id: None,
    session_id: None,
}, &mut delta)?;

for memory in &result.context {
    println!("{}", memory.content());
}
```

---

## Cloud + any language

The cloud API is plain REST. Any language that can POST JSON works. Call
`https://api.mentedb.com/v1/process_turn` with your `mdb_` key:

```bash
curl -X POST https://api.mentedb.com/v1/process_turn \
  -H "Authorization: Bearer $MENTEDB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_message":"I switched from Postgres to SQLite",
        "assistant_response":"Noted.",
        "turn_id":0}'
```

The response is the turn data directly as JSON:

```json
{ "context": [...], "contradictions": 0 }
```

## Self-hosted + any language

Run the server and call its REST API. Nothing to embed:

```bash
cargo install mentedb-server
mentedb-server --data-dir ./data          # REST on :6677
```

```bash
curl -X POST http://localhost:6677/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"user_message":"I switched from Postgres to SQLite","assistant_response":"Noted.","turn_id":0}'
```

See [Self-hosting](./SELF_HOSTING.md) for LLM config, embeddings, auth, and the
overnight maintenance the server runs to keep memory healthy.

## What `process_turn` returns

One call runs the full cognitive pipeline and returns everything it found. The
field you use most is `context`. The rest is signal you can act on or ignore.

| Field | What it is |
|-------|------------|
| `context` | Memories relevant to this turn, attention-ordered. Prepend to your next prompt. |
| `facts_extracted` | Facts learned this turn. Zero without an LLM configured. |
| `contradiction_count` | New information that conflicts with an existing belief. |
| `pain_warnings` | Things that went wrong in similar past contexts. |
| `stored_ids` | Memory IDs written this turn. |
| `predicted_topics` | Where the conversation is likely heading (from trajectory analysis). |

Node returns these in camelCase (`factsExtracted`, `contradictionCount`); Python
and the REST payload use snake_case.

## Next steps

- [Cloud](./CLOUD.md): keys, connecting AI coding tools, what runs for you.
- [Self-hosting](./SELF_HOSTING.md): LLM providers, embeddings, overnight maintenance, production auth.
- [MCP setup](https://github.com/nambok/mentedb-mcp): connect Claude Code, Cursor, Copilot, or Claude with one command instead of writing the loop yourself.
