export interface MemoryNode {
  id: string;
  agent_id: string;
  memory_type: MemoryType;
  embedding: number[];
  content: string;
  created_at: number;
  accessed_at: number;
  access_count: number;
  salience: number;
  confidence: number;
  space_id: string;
  attributes: Record<string, AttributeValue>;
  tags: string[];
}

export type MemoryType =
  | "episodic"
  | "semantic"
  | "procedural"
  | "antipattern"
  | "reasoning"
  | "correction";

export type EdgeType =
  | "caused"
  | "before"
  | "related"
  | "contradicts"
  | "supports"
  | "supersedes"
  | "derived"
  | "partof";

export type AttributeValue = string | number | boolean;

export interface MemoryEdge {
  source: string;
  target: string;
  edge_type: EdgeType;
  weight: number;
  created_at: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
}

export interface StatsResponse {
  memory_count: number;
  uptime_seconds: number;
}

export interface CreateMemoryRequest {
  agent_id: string;
  memory_type: string;
  content: string;
  embedding?: number[];
  tags?: string[];
  attributes?: Record<string, AttributeValue>;
  space_id?: string;
  salience?: number;
  confidence?: number;
}

export interface CreateMemoryResponse {
  id: string;
  status: string;
}

export interface SearchRequest {
  embedding: number[];
  k?: number;
}

export interface SearchResult {
  id: string;
  score: number;
}

export interface SearchResponse {
  results: SearchResult[];
}

export interface RecallRequest {
  query: string;
}

export interface RecallResponse {
  context: string;
  total_tokens: number;
  memory_count: number;
}

export interface CreateEdgeRequest {
  source: string;
  target: string;
  edge_type: string;
  weight?: number;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface GraphNode {
  id: string;
  content: string;
  memory_type: MemoryType;
  salience: number;
  tags: string[];
}

export interface GraphLink {
  source: string;
  target: string;
  edge_type: EdgeType;
  weight: number;
}

export interface CognitionAlert {
  id: string;
  type: "contradiction" | "reinforcement" | "decay" | "consolidation";
  message: string;
  memory_ids: string[];
  timestamp: number;
  severity: "low" | "medium" | "high";
}

export interface PainSignal {
  id: string;
  pattern: string;
  intensity: number;
  trigger_count: number;
  last_triggered: number;
  active: boolean;
}

export interface PhantomMemory {
  id: string;
  query_pattern: string;
  frequency: number;
  last_queried: number;
  suggested_content: string;
}

export interface TrajectoryPrediction {
  query: string;
  probability: number;
  based_on: string[];
}

export interface AgentSpace {
  id: string;
  name: string;
  agent_id: string;
  agent_name: string;
  memory_count: number;
  permission: "read" | "write" | "readwrite" | "admin";
}
