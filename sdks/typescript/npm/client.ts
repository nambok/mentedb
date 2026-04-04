import {
  MemoryType,
  EdgeType,
  type StoreOptions,
  type RecallResult,
  type SearchResult,
} from './types';

let nativeBinding: any;

try {
  nativeBinding = require('../mentedb.node');
} catch {
  nativeBinding = null;
}

function requireNative(): any {
  if (!nativeBinding) {
    throw new Error(
      'Native extension not loaded. Build with: npm run build'
    );
  }
  return nativeBinding;
}

/**
 * MenteDB client. Wraps the native Rust database engine and exposes a
 * TypeScript-friendly API for storing, recalling, searching, relating, and
 * forgetting memories.
 */
export class MenteDB {
  private native: any;

  constructor(dataDir: string = './mentedb-data') {
    const binding = requireNative();
    this.native = new binding.MenteDB(dataDir);
  }

  /** Store a memory and return its UUID. */
  store(options: StoreOptions): string {
    const {
      content,
      memoryType = MemoryType.Episodic,
      embedding = [],
      agentId,
      tags,
    } = options;
    return this.native.store(content, memoryType, embedding, agentId, tags);
  }

  /** Recall memories using an MQL query string. */
  recall(query: string): RecallResult {
    return this.native.recall(query);
  }

  /** Vector similarity search returning top-k results. */
  search(embedding: number[], k: number = 10): SearchResult[] {
    return this.native.search(embedding, k);
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
    const binding = requireNative();
    this.native = new binding.JsCognitionStream(bufferSize);
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
    const binding = requireNative();
    this.native = new binding.JsTrajectoryTracker(maxTurns);
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
