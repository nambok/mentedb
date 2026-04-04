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
