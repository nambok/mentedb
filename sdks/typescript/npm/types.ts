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
