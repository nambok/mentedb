import {
  MemoryType,
  EdgeType,
  type StoreOptions,
  type RecallResult,
  type SearchResult,
  type ProcessTurnResult,
  type IngestResult,
  type ErasureReceipt,
  type SessionErasureReceipt,
  type Memory,
  type ProvenanceFilter,
} from './types';

import * as os from 'os';
import * as path from 'path';

const PLATFORM_TRIPLES: Record<string, string> = {
  'linux-x64': 'linux-x64-gnu',
    'linux-arm64': 'linux-arm64-gnu',
  'darwin-arm64': 'darwin-arm64',
  'darwin-x64': 'darwin-x64',
  'win32-x64': 'win32-x64-msvc',
};

function loadNativeBinding(): any {
  const key = `${os.platform()}-${os.arch()}`;
  const triple = PLATFORM_TRIPLES[key];

  // Try platform-specific binary first (from CI cross-compile)
  if (triple) {
    try {
      return require(path.join(__dirname, '..', `mentedb.${triple}.node`));
    } catch {}
  }

  // Fall back to unqualified binary (local `napi build --release`)
  try {
    return require(path.join(__dirname, '..', 'mentedb.node'));
  } catch {}

  const supported = Object.keys(PLATFORM_TRIPLES).join(', ');
  throw new Error(
    `No native binding found for ${key}. ` +
    `Supported platforms: ${supported}. ` +
    `Build locally with: cd sdks/typescript && npm run build`
  );
}

const nativeBinding = loadNativeBinding();

/**
 * MenteDB client. Wraps the native Rust database engine and exposes a
 * TypeScript-friendly API for storing, recalling, searching, relating, and
 * forgetting memories.
 */
export class MenteDB {
  private native: any;

  constructor(dataDir: string = './mentedb-data') {
    this.native = new nativeBinding.MenteDb(dataDir);
  }

  /**
   * Process one conversation turn through the full cognitive pipeline.
   *
   * This is the primary API. A single call embeds the message, runs hybrid
   * recall (vector + keyword + graph), stores the turn, extracts facts,
   * detects contradictions, and returns attention-ordered context plus what
   * was learned. The field you use most is `context`: inject those memories
   * into your next prompt.
   *
   * Fact extraction and contradiction detection require an LLM. Set
   * `MENTEDB_LLM_PROVIDER` and `MENTEDB_LLM_API_KEY` in the environment;
   * without them you still get storage and recall.
   */
  processTurn(
    userMessage: string,
    assistantResponse?: string,
    turnId: number = 0,
    projectContext?: string,
    agentId?: string,
    sessionId?: string,
  ): ProcessTurnResult {
    return this.native.processTurn(
      userMessage,
      assistantResponse ?? null,
      turnId,
      projectContext ?? null,
      agentId ?? null,
      sessionId ?? null,
    );
  }

  /**
   * Extract memories from a raw conversation transcript and store them.
   *
   * Requires an LLM. Set `MENTEDB_LLM_PROVIDER` and `MENTEDB_LLM_API_KEY`,
   * or pass `provider` ("openai", "anthropic", or "ollama") explicitly.
   */
  ingest(conversation: string, provider?: string, agentId?: string): IngestResult {
    return this.native.ingest(conversation, provider ?? null, agentId ?? null);
  }

  /**
   * Store a memory and return its UUID. `sessionId`, `source` and
   * `sourceRef` are recorded as its provenance on the first write.
   */
  store(options: StoreOptions): string {
    const {
      content,
      memoryType = MemoryType.Episodic,
      embedding = [],
      agentId,
      tags,
      sessionId,
      source,
      sourceRef,
    } = options;
    return this.native.store(
      content,
      memoryType,
      embedding,
      agentId,
      tags,
      sessionId ?? null,
      source ?? null,
      sourceRef ?? null,
    );
  }

  /** One stored memory with its provenance. Throws when the id is unknown. */
  getMemory(memoryId: string): Memory {
    return this.native.getMemory(memoryId);
  }

  /** Ids of every memory written in `sessionId`. */
  memoryIdsInSession(sessionId: string): string[] {
    return this.native.memoryIdsInSession(sessionId);
  }

  /** Recall memories using an MQL query string. */
  recall(query: string): RecallResult {
    return this.native.recall(query);
  }

  /**
   * Vector similarity search returning top-k results, optionally narrowed to
   * memories written in a session and/or from a kind of source.
   */
  search(embedding: number[], k: number = 10, filter: ProvenanceFilter = {}): SearchResult[] {
    return this.native.search(embedding, k, filter.sessionId ?? null, filter.source ?? null);
  }

  /** Create a typed, weighted edge between two memories. */
  relate(
    source: string,
    target: string,
    edgeType: EdgeType = EdgeType.Related,
    weight: number = 1.0,
  ): void {
    this.native.relate(source, target, edgeType, weight);
  }

  /** Remove a memory by ID. */
  forget(memoryId: string): void {
    this.native.forget(memoryId);
  }

  /**
   * Verified erase: remove the memory, everything derived from it, and every
   * copy of their bytes in the database directory. `includeSources` also
   * erases what the memory was derived from; `verify` scans every file
   * afterwards. Returns the erasure receipt (ids, SHA-256 hashes, the scrub
   * steps, and the scan result), never the content. See docs/FORGETTING.md.
   */
  erase(
    memoryId: string,
    options: { includeSources?: boolean; verify?: boolean } = {},
  ): ErasureReceipt {
    const json = this.native.erase(memoryId, options.includeSources, options.verify);
    return JSON.parse(json) as ErasureReceipt;
  }

  /**
   * Verified erase of every memory written in `sessionId` and everything
   * derived from them, in one erase with one receipt. A session with no
   * memories returns an empty receipt. See docs/PROVENANCE.md.
   */
  eraseSession(
    sessionId: string,
    options: { includeSources?: boolean; verify?: boolean } = {},
  ): SessionErasureReceipt {
    const json = this.native.eraseSession(sessionId, options.includeSources, options.verify);
    return JSON.parse(json) as SessionErasureReceipt;
  }

  /** Flush all data and close the database. */
  close(): void {
    this.native.close();
  }
}

/**
 * Monitors an LLM token stream for contradictions, forgotten facts,
 * corrections, and reinforcements.
 */
export class CognitionStream {
  private native: any;

  constructor(bufferSize: number = 1000) {
    this.native = new nativeBinding.JsCognitionStream(bufferSize);
  }

  /** Push a token into the stream buffer. */
  feedToken(token: string): void {
    this.native.feedToken(token);
  }

  /** Drain the accumulated buffer and return its content. */
  drainBuffer(): string {
    return this.native.drainBuffer();
  }
}

/**
 * Tracks the reasoning arc of a conversation and predicts next topics.
 */
export class TrajectoryTracker {
  private native: any;

  constructor(maxTurns: number = 100) {
    this.native = new nativeBinding.JsTrajectoryTracker(maxTurns);
  }

  /**
   * Record a conversation turn.
   *
   * `decisionState` accepts one of:
   *   - `"investigating"`
   *   - `"interrupted"`
   *   - `"completed"`
   *   - `"narrowed:<choice>"`
   *   - `"decided:<decision>"`
   */
  recordTurn(
    topic: string,
    decisionState: string,
    openQuestions: string[] = [],
  ): void {
    this.native.recordTurn(topic, decisionState, openQuestions);
  }

  /** Get a resume context string describing the current trajectory. */
  getResumeContext(): string | null {
    return this.native.getResumeContext() ?? null;
  }

  /** Predict the next likely topics based on trajectory. */
  predictNextTopics(): string[] {
    return this.native.predictNextTopics();
  }
}
