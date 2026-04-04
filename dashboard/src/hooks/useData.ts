import { useState, useEffect, useCallback, useRef } from "react";
import { api, ApiError } from "../api";
import type {
  HealthResponse,
  StatsResponse,
  MemoryNode,
  CognitionAlert,
  PainSignal,
  PhantomMemory,
  TrajectoryPrediction,
  AgentSpace,
  GraphData,
} from "../api/types";

export function useHealth(pollMs = 10000) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const data = await api.health();
        if (active) {
          setHealth(data);
          setError(null);
        }
      } catch (e) {
        if (active) setError(e instanceof ApiError ? e.message : "Connection failed");
      }
    };
    poll();
    const id = setInterval(poll, pollMs);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [pollMs]);

  return { health, error };
}

export function useStats(pollMs = 5000) {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const data = await api.stats();
        if (active) {
          setStats(data);
          setLoading(false);
        }
      } catch {
        if (active) setLoading(false);
      }
    };
    poll();
    const id = setInterval(poll, pollMs);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [pollMs]);

  return { stats, loading };
}

export function useMemories() {
  const [memories, setMemories] = useState<MemoryNode[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchMemory = useCallback(async (id: string) => {
    try {
      const mem = await api.getMemory(id);
      setMemories((prev) => {
        if (prev.find((m) => m.id === mem.id)) return prev;
        return [...prev, mem];
      });
      return mem;
    } catch {
      return null;
    }
  }, []);

  return { memories, setMemories, loading, setLoading, fetchMemory };
}

// Demo data generators for when the server is unavailable
function generateDemoMemories(): MemoryNode[] {
  const types: MemoryNode["memory_type"][] = [
    "episodic", "semantic", "procedural", "antipattern", "reasoning", "correction",
  ];
  const contents = [
    "User prefers concise responses under 100 words",
    "API rate limit is 1000 requests per minute",
    "Deploy sequence: build, test, stage, production",
    "Never expose database credentials in environment logs",
    "If user asks about pricing, check subscription tier first",
    "Previous response about timezone was incorrect -- UTC offset is +5, not +4",
    "Team standup is at 9:30 AM EST daily",
    "Error handling should use Result type, not exceptions",
    "Customer reported latency issues on EU-west endpoints",
    "Authentication flow uses JWT with 24h expiry",
    "Regression: caching layer broke after v2.3 update",
    "User session context should persist across page reloads",
    "Database migration requires downtime window of 15 minutes",
    "Successful pattern: chunked file uploads reduce timeouts",
    "Contradiction detected: doc says REST, implementation uses GraphQL",
    "Memory consolidation reduced working set by 34%",
  ];
  const now = Date.now() * 1000;
  return contents.map((content, i) => ({
    id: `demo-${i.toString().padStart(4, "0")}-0000-0000-${i.toString(16).padStart(12, "0")}`,
    agent_id: `agent-${(i % 3).toString().padStart(4, "0")}-0000-0000-000000000000`,
    memory_type: types[i % types.length],
    embedding: [],
    content,
    created_at: now - (contents.length - i) * 3600_000_000,
    accessed_at: now - i * 600_000_000,
    access_count: Math.floor(Math.random() * 50) + 1,
    salience: Math.round((0.3 + Math.random() * 0.7) * 100) / 100,
    confidence: Math.round((0.5 + Math.random() * 0.5) * 100) / 100,
    space_id: `space-${(i % 2).toString().padStart(4, "0")}-0000-0000-000000000000`,
    attributes: {},
    tags: [`tag-${i % 5}`, types[i % types.length]],
  }));
}

function generateDemoGraph(): GraphData {
  const memories = generateDemoMemories();
  const edgeTypes: GraphData["links"][0]["edge_type"][] = [
    "related", "supports", "contradicts", "caused", "derived", "before", "supersedes", "partof",
  ];
  const nodes = memories.map((m) => ({
    id: m.id,
    content: m.content,
    memory_type: m.memory_type,
    salience: m.salience,
    tags: m.tags,
  }));
  const links: GraphData["links"] = [];
  for (let i = 1; i < nodes.length; i++) {
    links.push({
      source: nodes[Math.floor(Math.random() * i)].id,
      target: nodes[i].id,
      edge_type: edgeTypes[i % edgeTypes.length],
      weight: Math.round((0.3 + Math.random() * 0.7) * 100) / 100,
    });
    if (Math.random() > 0.6 && i > 2) {
      links.push({
        source: nodes[Math.floor(Math.random() * i)].id,
        target: nodes[Math.floor(Math.random() * i)].id,
        edge_type: edgeTypes[(i + 3) % edgeTypes.length],
        weight: Math.round((0.2 + Math.random() * 0.5) * 100) / 100,
      });
    }
  }
  return { nodes, links };
}

function generateDemoAlerts(): CognitionAlert[] {
  const now = Date.now();
  return [
    { id: "a1", type: "contradiction", message: "Memory #0003 contradicts #0014 on API protocol", memory_ids: ["demo-0003", "demo-0014"], timestamp: now - 120000, severity: "high" },
    { id: "a2", type: "reinforcement", message: "Pattern confirmed: chunked uploads reduce failures (3 sources)", memory_ids: ["demo-0013"], timestamp: now - 300000, severity: "medium" },
    { id: "a3", type: "decay", message: "Memory #0006 salience decayed below threshold (0.12)", memory_ids: ["demo-0006"], timestamp: now - 600000, severity: "low" },
    { id: "a4", type: "consolidation", message: "5 episodic memories consolidated into semantic knowledge", memory_ids: ["demo-0000", "demo-0008"], timestamp: now - 900000, severity: "medium" },
    { id: "a5", type: "contradiction", message: "Timezone offset conflict between memory #0005 and external data", memory_ids: ["demo-0005"], timestamp: now - 1500000, severity: "high" },
  ];
}

function generateDemoPainSignals(): PainSignal[] {
  const now = Date.now();
  return [
    { id: "p1", pattern: "Exposing credentials in logs", intensity: 0.92, trigger_count: 7, last_triggered: now - 180000, active: true },
    { id: "p2", pattern: "Deploying without running tests", intensity: 0.78, trigger_count: 3, last_triggered: now - 3600000, active: true },
    { id: "p3", pattern: "Ignoring rate limit headers", intensity: 0.45, trigger_count: 2, last_triggered: now - 86400000, active: false },
  ];
}

function generateDemoPhantoms(): PhantomMemory[] {
  return [
    { id: "ph1", query_pattern: "database connection pooling config", frequency: 12, last_queried: Date.now() - 600000, suggested_content: "Connection pool settings and best practices" },
    { id: "ph2", query_pattern: "rollback procedure for failed deploys", frequency: 8, last_queried: Date.now() - 1800000, suggested_content: "Step-by-step rollback instructions" },
    { id: "ph3", query_pattern: "monitoring alert thresholds", frequency: 5, last_queried: Date.now() - 7200000, suggested_content: "Alert threshold configuration per service" },
  ];
}

function generateDemoTrajectory(): TrajectoryPrediction[] {
  return [
    { query: "deploy status check", probability: 0.82, based_on: ["demo-0002", "demo-0010"] },
    { query: "error rate metrics", probability: 0.65, based_on: ["demo-0008", "demo-0010"] },
    { query: "user session timeout config", probability: 0.48, based_on: ["demo-0011"] },
  ];
}

function generateDemoSpaces(): AgentSpace[] {
  return [
    { id: "space-0000", name: "Production Agent", agent_id: "agent-0000", agent_name: "prod-agent", memory_count: 847, permission: "admin" },
    { id: "space-0001", name: "Dev Agent", agent_id: "agent-0001", agent_name: "dev-agent", memory_count: 234, permission: "readwrite" },
    { id: "space-0002", name: "Monitor Agent", agent_id: "agent-0002", agent_name: "monitor-agent", memory_count: 1203, permission: "read" },
  ];
}

export function useDemoData() {
  const [memories] = useState(generateDemoMemories);
  const [graph] = useState(generateDemoGraph);
  const [alerts] = useState(generateDemoAlerts);
  const [painSignals] = useState(generateDemoPainSignals);
  const [phantoms] = useState(generateDemoPhantoms);
  const [trajectory] = useState(generateDemoTrajectory);
  const [spaces] = useState(generateDemoSpaces);

  return { memories, graph, alerts, painSignals, phantoms, trajectory, spaces };
}

export function useWebSocket(url?: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<unknown[]>([]);

  useEffect(() => {
    if (!url) return;
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onclose = () => setConnected(false);
      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          setMessages((prev) => [...prev.slice(-99), data]);
        } catch { /* ignore non-json */ }
      };
      return () => ws.close();
    } catch {
      return;
    }
  }, [url]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { connected, messages, send };
}
