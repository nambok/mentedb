# mentedb

The mind database for AI agents. TypeScript/Node.js SDK powered by napi-rs.

MenteDB is a purpose built database engine for AI agent memory. It provides
vector similarity search, a typed knowledge graph, token budget aware context
assembly, and cognitive features such as contradiction detection and trajectory
tracking. This package delivers the Rust engine as a native Node.js addon with
no runtime npm dependencies.

## Install

```bash
npm install mentedb
```

The package ships prebuilt native binaries for macOS (arm64, x64), Linux
(x64, arm64, glibc) and Windows (x64). On any other platform, clone the
repository and build the addon from `sdks/typescript` (see "Building from
source"; requires Rust and [napi-rs](https://napi.rs)).

## Quick start

```typescript
import { MenteDB, MemoryType, EdgeType } from 'mentedb';

const db = new MenteDB('./my-agent-memory');

// Primary API — one call does everything (synchronous)
const result = db.processTurn(
  "I prefer using TypeScript for backend services",
  "Noted! TypeScript is great for type-safe backends.",
  0 // turn_id
  // optional: projectContext, agentId, sessionId (tags stored turns so
  // injection recall can exclude the requesting session)
);

console.log(result.context);     // relevant memories
console.log(result.storedIds);   // ids of the memories stored this turn

// Low-level APIs:

// Store a memory
const id = db.store({
  content: 'The deploy key rotates every 90 days',
  memoryType: MemoryType.Semantic,
  embedding: embeddingFromYourModel,
  tags: ['infra', 'security'],
});

// Vector similarity search
const hits = db.search(queryEmbedding, 5);

// MQL recall with token budget
const ctx = db.recall('RECALL similar("deploy key rotation") LIMIT 10');

// Relate memories
db.relate(id, otherId, EdgeType.Supersedes);

// Forget a memory
db.forget(id);

// Close the database
db.close();
```

## Cognitive features

### CognitionStream

Buffer an LLM token stream. The Node.js binding exposes only feeding and
draining the buffer; the alert checks against stored facts are available in
the Python SDK.

```typescript
import { CognitionStream } from 'mentedb';

const stream = new CognitionStream(1000);
for (const token of llmTokens) {
  stream.feedToken(token);
}
const text = stream.drainBuffer();
```

### TrajectoryTracker

Track the reasoning arc of a conversation and predict upcoming topics.

```typescript
import { TrajectoryTracker } from 'mentedb';

const tracker = new TrajectoryTracker();

tracker.recordTurn('JWT auth design', 'investigating', [
  'Which algorithm?',
  'Token lifetime?',
]);
tracker.recordTurn('Token lifetime', 'decided:15 minutes');

const resume = tracker.getResumeContext();
const next = tracker.predictNextTopics();
```

## LLM features

`processTurn` stores and recalls without an LLM. Sleeptime
enrichment (LLM fact extraction, entity linking, community summaries, user
profile) is not exposed by the Node.js package; use the Python SDK or the
server for it. `ingest` calls an LLM and needs `MENTEDB_LLM_PROVIDER`
(`openai`, `anthropic` or `ollama`) and `MENTEDB_LLM_API_KEY`.

## API reference

### `MenteDB`

| Method | Description |
|--------|-------------|
| `new MenteDB(dataDir)` | Open or create a database at the given path. |
| `store(options)` | Store a memory. Returns its UUID string. |
| `recall(query)` | Recall memories via an MQL query. Returns `RecallResult`. |
| `search(embedding, k)` | Vector similarity search. Returns `SearchResult[]`. |
| `relate(source, target, edgeType?, weight?)` | Create a typed edge between two memories. |
| `processTurn(userMessage, assistantResponse?, turnId?, projectContext?, agentId?, sessionId?)` | **Primary API.** Process a conversation turn through the full cognitive pipeline. Returns `ProcessTurnResult` (context, stored ids, actions, sentiment, predictions, and more). |
| `forget(memoryId)` | Remove a memory by ID. |
| `ingest(conversation, provider?, agentId?)` | **Deprecated** — use `processTurn` instead. Extract and store memories from a conversation via LLM. |
| `close()` | Flush and close the database. |

### `CognitionStream`

| Method | Description |
|--------|-------------|
| `new CognitionStream(bufferSize?)` | Create a token stream monitor. |
| `feedToken(token)` | Push a token into the ring buffer. |
| `drainBuffer()` | Drain and return the accumulated text. |

### `TrajectoryTracker`

| Method | Description |
|--------|-------------|
| `new TrajectoryTracker(maxTurns?)` | Create a trajectory tracker. |
| `recordTurn(topic, decisionState, openQuestions?)` | Record a conversation turn. |
| `getResumeContext()` | Build a resume context string. |
| `predictNextTopics()` | Predict the next likely topics. |

## Types

```typescript
enum MemoryType {
  Episodic, Semantic, Procedural, AntiPattern, Reasoning, Correction
}

enum EdgeType {
  Caused, Before, Related, Contradicts, Supports, Supersedes, Derived, PartOf
}

interface StoreOptions {
  content: string;
  memoryType?: MemoryType;
  embedding?: number[];
  agentId?: string;
  tags?: string[];
}

interface RecallResult {
  text: string;
  totalTokens: number;
  memoryCount: number;
}

interface SearchResult {
  id: string;
  score: number;
}
```

## Building from source

```bash
cd sdks/typescript
npm install -g @napi-rs/cli   # the napi CLI is not a devDependency
cargo check                   # verify Rust compiles
npm run build                 # build the native addon (napi build --release)
npx tsc                       # compile the npm/ wrapper
```

## License

Apache 2.0
