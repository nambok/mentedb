# mentedb

The mind database for AI agents. TypeScript/Node.js SDK powered by napi-rs.

MenteDB is a purpose built database engine for AI agent memory. It provides
vector similarity search, a typed knowledge graph, token budget aware context
assembly, and cognitive features such as contradiction detection and trajectory
tracking. This package delivers the full Rust engine as a native Node.js addon
with zero runtime dependencies.

## Install

```bash
npm install mentedb
```

The package ships a prebuilt native binary. If no prebuild is available for your
platform, `npm run build` compiles from source (requires Rust and
[napi-rs](https://napi.rs)).

## Quick start

```typescript
import { MenteDB, MemoryType, EdgeType } from 'mentedb';

const db = new MenteDB('./my-agent-memory');

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

Monitor an LLM token stream for contradictions and reinforcements against
stored facts.

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

## API reference

### `MenteDB`

| Method | Description |
|--------|-------------|
| `new MenteDB(dataDir)` | Open or create a database at the given path. |
| `store(options)` | Store a memory. Returns its UUID string. |
| `recall(query)` | Recall memories via an MQL query. Returns `RecallResult`. |
| `search(embedding, k)` | Vector similarity search. Returns `SearchResult[]`. |
| `relate(source, target, edgeType?, weight?)` | Create a typed edge between two memories. |
| `processTurn(userMessage, assistantResponse?, turnId?, projectContext?, agentId?)` | Process a conversation turn through the full cognitive pipeline. Returns context, actions, sentiment, predictions, and more. |
| `forget(memoryId)` | Remove a memory by ID. |
| `ingest(conversation, provider?, agentId?)` | Extract and store memories from a conversation via LLM. |
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
cargo check          # verify Rust compiles
npm run build        # build the native addon
npm test             # run tests
```

## License

Apache 2.0
