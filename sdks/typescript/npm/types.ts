export enum MemoryType {
  Episodic = 'episodic',
  Semantic = 'semantic',
  Procedural = 'procedural',
  AntiPattern = 'anti_pattern',
  Reasoning = 'reasoning',
  Correction = 'correction',
}

export enum EdgeType {
  Caused = 'caused',
  Before = 'before',
  Related = 'related',
  Contradicts = 'contradicts',
  Supports = 'supports',
  Supersedes = 'supersedes',
  Derived = 'derived',
  PartOf = 'part_of',
}

export interface RecallResult {
  text: string;
  totalTokens: number;
  memoryCount: number;
}

export interface SearchResult {
  id: string;
  score: number;
}

export interface StoreOptions {
  content: string;
  memoryType?: MemoryType;
  embedding?: number[];
  agentId?: string;
  tags?: string[];
  /** The session the memory is written in (its provenance). */
  sessionId?: string;
  /** chat, file, api, mcp, import, or another name. Defaults to api. */
  source?: string;
  /** A file path, URL or message id within the source. */
  sourceRef?: string;
}

/**
 * Where a memory came from, recorded when it was first written and never
 * changed afterwards. See docs/PROVENANCE.md.
 */
export interface Provenance {
  sessionId?: string | null;
  source: string;
  sourceRef?: string | null;
  writtenBy: string;
  /** Microseconds since the Unix epoch. */
  writtenAt: number;
  /** For a derived memory, the memories it was made from. */
  derivedFrom: string[];
}

/** One stored memory, as `getMemory` returns it. */
export interface Memory {
  id: string;
  content: string;
  memoryType: string;
  agentId: string;
  tags: string[];
  createdAt: number;
  /** Absent for a memory written before provenance existed. */
  provenance?: Provenance | null;
}

/** Narrow a search to memories by provenance. */
export interface ProvenanceFilter {
  sessionId?: string;
  source?: string;
}

/** A single memory selected for the working-context window. */
export interface ContextItem {
  id: string;
  content: string;
  score: number;
}

export interface PainWarning {
  signalId: string;
  intensity: number;
  description: string;
}

export interface DetectedAction {
  actionType: string;
  detail: string;
}

export interface ProactiveRecall {
  memoryId: string;
  content: string;
  relevance: number;
  actionType: string;
}

/**
 * Everything one `processTurn` call produced. The field you use most is
 * `context`: the attention-ordered memories to inject into your next prompt.
 */
export interface ProcessTurnResult {
  /** Attention-ordered memories relevant to this turn. Inject into your next prompt. */
  context: ContextItem[];
  /** IDs of the memories stored from this turn. */
  storedIds: string[];
  episodicId?: string;
  painWarnings: PainWarning[];
  cacheHit: boolean;
  inferenceActions: number;
  detectedActions: DetectedAction[];
  proactiveRecalls: ProactiveRecall[];
  correctionId?: string;
  sentiment: number;
  phantomCount: number;
  contradictionCount: number;
  predictedTopics: string[];
  /** Facts extracted this turn. Zero unless an LLM is configured (see MENTEDB_LLM_PROVIDER). */
  factsExtracted: number;
  edgesCreated: number;
  enrichmentPending: boolean;
  deltaAdded: string[];
  deltaRemoved: string[];
}

export interface IngestResult {
  memoriesStored: number;
  rejectedLowQuality: number;
  rejectedDuplicate: number;
  contradictions: number;
  storedIds: string[];
}

/** One erased memory: its id, why it went, and hashes of what it held. */
export interface ErasedMemory {
  id: string;
  reason: 'requested' | 'derived' | 'source' | 'identical';
  /** SHA-256 of the content, lowercase hex. */
  content_sha256: string;
  /** SHA-256 of the embedding as little endian f32 bytes, when it had one. */
  embedding_sha256: string | null;
}

/** A file in which the verification scan found erased bytes. */
export interface ResidualHit {
  file: string;
  memory: string;
  kind: 'content' | 'embedding';
}

/** Result of `MenteDB.erase`. Carries ids and hashes, never content. */
export interface ErasureReceipt extends ErasureOutcome {
  requested: string;
}

/**
 * Result of `MenteDB.eraseSession`: one erase over every memory written in
 * the session and everything derived from them.
 */
export interface SessionErasureReceipt extends ErasureOutcome {
  session_id: string;
  /** The memories written in the session. */
  requested: string[];
}

/** What every erase receipt carries. */
export interface ErasureOutcome {
  erased: ErasedMemory[];
  ancestors_kept: string[];
  profiles_to_review: string[];
  erased_at_us: number;
  steps: {
    records_zeroed: number;
    index_vectors_scrubbed: number;
    free_space_bytes_zeroed: number | null;
    wal_scrubbed: boolean;
    snapshots_rewritten: boolean;
    replaced_files_zeroed: number;
    legacy_files_removed: string[];
    cache_entries_dropped: number;
  };
  verification: {
    files_scanned: number;
    bytes_scanned: number;
    residual_found: boolean;
    hits: ResidualHit[];
    unreadable: string[];
    content_not_checked: string[];
  } | null;
}
