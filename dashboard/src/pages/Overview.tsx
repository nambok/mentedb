import {
  Database,
  Network,
  Users,
  Layers,
  Clock,
  TrendingUp,
} from "lucide-react";
import StatCard from "../components/StatCard";
import { MemoryTypeBadge, timeAgo } from "../components/MemoryTypeBadge";
import { useHealth, useStats, useDemoData } from "../hooks/useData";

export default function Overview() {
  const { health, error } = useHealth();
  const { stats } = useStats();
  const { memories, graph } = useDemoData();

  const memoryCount = stats?.memory_count ?? memories.length;
  const edgeCount = graph.links.length;
  const agentCount = new Set(memories.map((m) => m.agent_id)).size;
  const spaceCount = new Set(memories.map((m) => m.space_id)).size;

  const recentMemories = [...memories]
    .sort((a, b) => b.created_at - a.created_at)
    .slice(0, 8);

  const typeCounts = memories.reduce(
    (acc, m) => {
      acc[m.memory_type] = (acc[m.memory_type] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100">Overview</h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            System status and memory statistics
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              error ? "bg-red-500" : health ? "bg-emerald-500" : "bg-zinc-600"
            }`}
          />
          <span className="text-xs text-zinc-500">
            {error
              ? "Disconnected"
              : health
                ? `v${health.version} -- up ${formatUptime(health.uptime_seconds)}`
                : "Connecting..."}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total Memories"
          value={memoryCount.toLocaleString()}
          icon={<Database className="w-4 h-4" />}
          variant="emerald"
        />
        <StatCard
          label="Edges"
          value={edgeCount.toLocaleString()}
          icon={<Network className="w-4 h-4" />}
        />
        <StatCard
          label="Agents"
          value={agentCount}
          icon={<Users className="w-4 h-4" />}
        />
        <StatCard
          label="Spaces"
          value={spaceCount}
          icon={<Layers className="w-4 h-4" />}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-zinc-900 border border-zinc-800 rounded-xl">
          <div className="px-5 py-4 border-b border-zinc-800 flex items-center gap-2">
            <Clock className="w-4 h-4 text-zinc-500" />
            <h2 className="text-sm font-medium text-zinc-300">
              Recent Activity
            </h2>
          </div>
          <div className="divide-y divide-zinc-800/50">
            {recentMemories.map((m) => (
              <div
                key={m.id}
                className="px-5 py-3 flex items-start gap-3 hover:bg-zinc-800/30 transition-colors"
              >
                <div className="pt-0.5 shrink-0">
                  <MemoryTypeBadge type={m.memory_type} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-zinc-300 truncate">
                    {m.content}
                  </p>
                  <p className="text-xs text-zinc-600 mt-0.5">
                    {timeAgo(m.created_at)} -- salience{" "}
                    {m.salience.toFixed(2)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-zinc-900 border border-zinc-800 rounded-xl">
          <div className="px-5 py-4 border-b border-zinc-800 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-zinc-500" />
            <h2 className="text-sm font-medium text-zinc-300">
              Memory Distribution
            </h2>
          </div>
          <div className="p-5 space-y-3">
            {Object.entries(typeCounts).map(([type, count]) => {
              const pct = Math.round((count / memories.length) * 100);
              return (
                <div key={type}>
                  <div className="flex items-center justify-between mb-1">
                    <MemoryTypeBadge
                      type={type as import("../api/types").MemoryType}
                    />
                    <span className="text-xs text-zinc-500">
                      {count} ({pct}%)
                    </span>
                  </div>
                  <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-emerald-500/60 transition-all"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86400)}d`;
}
