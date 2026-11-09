# Memory tool backend

Claude's memory tool (`memory_20250818`) is a client side tool: the model
sends file style commands against a `/memories` directory, and your code runs
them and returns the text. MenteDB can be that backend. Each file is stored as
one memory, so the model's notes live in your database, scoped per agent, and
show up in search beside everything else the agent remembers.

`mentedb-server` serves the backend at `POST /v1/memory_tool`, with the same
request and response contract as MenteDB Cloud, so a tool loop works against
either one by changing only the base URL and the credential.

- [How it works](#how-it-works)
- [The endpoint](#the-endpoint)
- [Commands](#commands)
- [Scoping](#scoping)
- [Path rules](#path-rules)
- [Python](#python)
- [TypeScript](#typescript)
- [Other hosts](#other-hosts)

## How it works

1. You declare `{"type": "memory_20250818", "name": "memory"}` in `tools` on a
   Messages API request.
2. The model answers with a `tool_use` block named `memory`, for example
   `{"command": "view", "path": "/memories"}`.
3. Your loop posts that input to `/v1/memory_tool` and gets back
   `{"content": "...", "is_error": false}`.
4. You send a `tool_result` block with that `content` (and `is_error: true`
   when set), then call the model again.

Each file maps to one memory: the content is the file text, a
`memtool:path:<path>` tag holds the path, and a `source:memory_tool` tag marks
it. Files are found through the tag index, so a command costs time in the
number of memory tool files, not the size of the database. Edits keep the
memory's id and embed the new text; renames change only the path tag.

## The endpoint

`POST /v1/memory_tool`, or `POST /v1/db/{db}/memory_tool` for a named
database.

The body is the `tool_use` input as the model sent it, plus two optional
fields:

| Field | Type | Meaning |
|---|---|---|
| `command` | string | `view`, `create`, `str_replace`, `insert`, `delete` or `rename` |
| `path` | string | The file or directory, for every command except `rename` |
| `view_range` | `[start, end]` | `view` only: 1 indexed lines, `end` of `-1` means to the end |
| `file_text` | string | `create`: the whole file |
| `old_str`, `new_str` | string | `str_replace`: `old_str` must occur exactly once; leaving out `new_str` deletes it |
| `insert_line`, `insert_text` | integer, string | `insert`: the text goes after line `insert_line`, `0` is the top |
| `old_path`, `new_path` | string | `rename`: a file or a whole directory |
| `agent_id` | UUID | Optional. Whose files to use, see [Scoping](#scoping) |
| `user_id` | UUID | Optional. Splits one agent's files further by user |

The reply is always:

```json
{ "content": "File created successfully at: /memories/notes.md", "is_error": false }
```

Copy `content` into the `tool_result` block as is. Failures the model can fix
come back as HTTP 200 with `is_error: true` and a message it can read: a
missing file, a bad line range, an `old_str` that matches zero or several
times, an invalid path, an unknown command. HTTP errors are for the caller,
not the model: `400` for a body that is not a memory tool call (no `command`,
or an `agent_id` that is not a UUID), `401` and `403` for auth, `500` for an
engine fault.

```bash
curl -X POST http://localhost:6677/v1/memory_tool \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"command": "create", "path": "/memories/prefs.md", "file_text": "tabs, not spaces\n"}'
```

## Commands

| Command | Success text |
|---|---|
| `view` on a file | `Here's the content of {path} with line numbers:` then each line as a 6 wide number, a tab, and the text |
| `view` on a directory | A listing of files and subdirectories up to 2 levels deep with `du` style sizes. `/memories` always exists, even when empty |
| `create` | `File created successfully at: {path}`. Writing an existing path replaces its text |
| `str_replace` | `The memory file has been edited.` |
| `insert` | `The file {path} has been edited.` |
| `delete` | `Successfully deleted {path}`. A directory deletes every file under it |
| `rename` | `Successfully renamed {old_path} to {new_path}`. A directory moves every file under it. A destination that is an existing file or directory is refused |

`/memories` itself cannot be created, deleted or renamed. Directories are not
stored on their own: a directory exists while a file lives under it.

## Scoping

Files belong to an agent and a user. A command only sees the files of its own
scope, so one agent can never list, read, change, rename onto or delete
another agent's files, even at the same path. The scope comes from:

1. A token bound to an agent (the `agent_id` claim): always that agent. An
   `agent_id` in the body must match it, or the request gets `403`.
2. Otherwise (no auth, a token scoped to databases only, or an admin token):
   the `agent_id` in the body, or no agent when it is left out.
3. `user_id` in the body, or no user when it is left out.

Database access follows the usual token rules: the token's `dbs` claim must
name the database, and read only tokens may `view` but get `403` for every
other command. See [Authentication and access](./SERVER_API.md#authentication-and-access).

Mutating commands on one database run one at a time, so two creates of the
same new path store one file.

## Path rules

Every path must be `/memories` or live under `/memories/`. A trailing slash is
dropped. These are refused with `Error: Invalid path ...` before anything is
read or written:

- anything outside `/memories` (`/etc/passwd`, `/memoriesx`, `memories/x`)
- `.` and `..` segments, and empty segments (`/memories//x`)
- backslashes
- percent encoded dots, slashes, backslashes and percent signs (`%2e`, `%2f`,
  `%5c`, `%25`), so double encoding is refused too
- control characters, including newlines

Paths are only ever tag values, never filesystem paths, so nothing could leave
the database either way; the checks keep the store to what the tool spec
allows.

## Python

`MemoryTool` in the `mentedb` package (standard library only) posts each call
and builds the `tool_result` blocks:

```python
import anthropic
from mentedb import MemoryTool

memory = MemoryTool("http://localhost:6677", token, agent_id="...")
client = anthropic.Anthropic()

messages = [{"role": "user", "content": "Remember that I deploy on Fridays."}]
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
```

`memory.execute(tool_input)` runs one call and returns the reply dict;
`memory.handle(tool_use)` returns one `tool_result` block. Pass `database=` for
a named database. For MenteDB Cloud, `MenteDBClient(api_key=...).memory_tool()`
returns a `MemoryTool` for the account. Auth and server failures raise
`MenteDBError`. A full script is in
[`sdks/python/examples/memory_tool_agent.py`](../sdks/python/examples/memory_tool_agent.py).

## TypeScript

`MemoryTool` in the `mentedb` package uses the global `fetch`:

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { MemoryTool } from 'mentedb';

const memory = new MemoryTool({ baseUrl: 'http://localhost:6677', token, agentId: '...' });
const client = new Anthropic();

const messages: Anthropic.MessageParam[] = [{ role: 'user', content: 'Remember that I deploy on Fridays.' }];
while (true) {
  const response = await client.messages.create({
    model: 'claude-opus-5-5',
    max_tokens: 16000,
    tools: [MemoryTool.TOOL],
    messages,
  });
  messages.push({ role: 'assistant', content: response.content });
  if (response.stop_reason !== 'tool_use') break;
  messages.push({ role: 'user', content: await memory.results(response.content) });
}
```

`execute`, `handle` and the `database` option work as in Python. Auth and
server failures throw `MemoryToolError` with the HTTP `status`. A full script
is in
[`sdks/typescript/examples/memory_tool_agent.ts`](../sdks/typescript/examples/memory_tool_agent.ts).

## Other hosts

The command logic is in the engine crate, `mentedb::memory_tool`, so any host
that embeds `MenteDb` can serve the same contract:
`memory_tool::handle(&db, &scope, &request)` parses, validates and runs one
call and returns the reply plus the change in memory count. The host is
responsible for serializing mutating commands per scope.
