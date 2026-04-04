import { Database, Shield, BarChart3 } from "lucide-react";
import { useDemoData } from "../hooks/useData";

const permissionColors: Record<string, string> = {
  admin: "bg-emerald-500/15 text-emerald-400 border-emerald-500/25",
  readwrite: "bg-blue-500/15 text-blue-400 border-blue-500/25",
  write: "bg-amber-500/15 text-amber-400 border-amber-500/25",
  read: "bg-zinc-700/50 text-zinc-400 border-zinc-600",
};

export default function AgentSpaces() {
  const { spaces } = useDemoData();

  const totalMemories = spaces.reduce((s, sp) => s + sp.memory_count, 0);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-zinc-100">Agent Spaces</h1>
        <p className="text-sm text-zinc-500 mt-0.5">
          {spaces.length} spaces, {totalMemories.toLocaleString()} total
          memories
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {spaces.map((space) => (
          <div
            key={space.id}
            className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 hover:border-zinc-700 transition-colors"
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-sm font-medium text-zinc-200">
                  {space.name}
                </h3>
                <p className="text-xs text-zinc-500 mt-0.5 font-mono">
                  {space.agent_name}
                </p>
              </div>
              <span
                className={`text-[10px] font-medium px-2 py-0.5 rounded border ${permissionColors[space.permission]}`}
              >
                {space.permission.toUpperCase()}
              </span>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <Database className="w-4 h-4 text-zinc-600" />
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-zinc-500">Memories</span>
                    <span className="text-xs text-zinc-400">
                      {space.memory_count.toLocaleString()}
                    </span>
                  </div>
                  <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500/50 rounded-full"
                      style={{
                        width: `${Math.min((space.memory_count / 1500) * 100, 100)}%`,
                      }}
                    />
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <Shield className="w-4 h-4 text-zinc-600" />
                <div className="flex-1">
                  <span className="text-xs text-zinc-500">RBAC Status</span>
                  <div className="flex items-center gap-1.5 mt-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    <span className="text-xs text-zinc-400">Enforced</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <BarChart3 className="w-4 h-4 text-zinc-600" />
                <div className="flex-1">
                  <span className="text-xs text-zinc-500">Activity</span>
                  <div className="flex gap-0.5 mt-1.5">
                    {Array.from({ length: 14 }, (_, i) => {
                      const height = Math.random() * 20 + 4;
                      return (
                        <div
                          key={i}
                          className="flex-1 bg-emerald-500/30 rounded-sm"
                          style={{ height: `${height}px` }}
                        />
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-zinc-800/50 flex items-center justify-between">
              <span className="text-[10px] text-zinc-600 font-mono truncate max-w-[60%]">
                {space.id}
              </span>
              <span className="text-[10px] text-zinc-600 font-mono truncate">
                {space.agent_id}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
