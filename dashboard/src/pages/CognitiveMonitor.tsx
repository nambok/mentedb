import {
  AlertTriangle,
  ShieldAlert,
  HelpCircle,
  TrendingUp,
  Zap,
  RefreshCw,
  ArrowDownCircle,
  Layers,
} from "lucide-react";
import { useDemoData } from "../hooks/useData";
import type { CognitionAlert } from "../api/types";

const alertIcons: Record<CognitionAlert["type"], typeof Zap> = {
  contradiction: AlertTriangle,
  reinforcement: RefreshCw,
  decay: ArrowDownCircle,
  consolidation: Layers,
};

const alertColors: Record<CognitionAlert["type"], string> = {
  contradiction: "text-red-400 bg-red-500/10 border-red-500/20",
  reinforcement: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
  decay: "text-amber-400 bg-amber-500/10 border-amber-500/20",
  consolidation: "text-blue-400 bg-blue-500/10 border-blue-500/20",
};

const severityColors: Record<CognitionAlert["severity"], string> = {
  high: "bg-red-500",
  medium: "bg-amber-500",
  low: "bg-zinc-500",
};

function formatTime(ts: number): string {
  const seconds = Math.floor((Date.now() - ts) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

export default function CognitiveMonitor() {
  const { alerts, painSignals, phantoms, trajectory } = useDemoData();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-zinc-100">
          Cognitive Monitor
        </h1>
        <p className="text-sm text-zinc-500 mt-0.5">
          Real-time cognition stream, pain signals, and trajectory
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cognition Stream */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl">
          <div className="px-5 py-4 border-b border-zinc-800 flex items-center gap-2">
            <Zap className="w-4 h-4 text-emerald-400" />
            <h2 className="text-sm font-medium text-zinc-300">
              Cognition Stream
            </h2>
            <div className="ml-auto flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[10px] text-zinc-500">LIVE</span>
            </div>
          </div>
          <div className="divide-y divide-zinc-800/50 max-h-96 overflow-y-auto">
            {alerts.map((alert) => {
              const Icon = alertIcons[alert.type];
              return (
                <div
                  key={alert.id}
                  className="px-5 py-3 flex items-start gap-3 hover:bg-zinc-800/30 transition-colors"
                >
                  <div
                    className={`mt-0.5 p-1.5 rounded-lg border ${alertColors[alert.type]}`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-[10px] font-medium text-zinc-400 uppercase tracking-wider">
                        {alert.type}
                      </span>
                      <div
                        className={`w-1.5 h-1.5 rounded-full ${severityColors[alert.severity]}`}
                      />
                    </div>
                    <p className="text-sm text-zinc-300">{alert.message}</p>
                    <p className="text-xs text-zinc-600 mt-0.5">
                      {formatTime(alert.timestamp)}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Pain Signals */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl">
          <div className="px-5 py-4 border-b border-zinc-800 flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-red-400" />
            <h2 className="text-sm font-medium text-zinc-300">
              Pain Signals
            </h2>
            <span className="ml-auto text-xs text-zinc-600">
              {painSignals.filter((p) => p.active).length} active
            </span>
          </div>
          <div className="divide-y divide-zinc-800/50">
            {painSignals.map((signal) => (
              <div
                key={signal.id}
                className="px-5 py-4 hover:bg-zinc-800/30 transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-zinc-300">
                    {signal.pattern}
                  </span>
                  <span
                    className={`text-[10px] font-medium px-2 py-0.5 rounded ${
                      signal.active
                        ? "bg-red-500/15 text-red-400"
                        : "bg-zinc-800 text-zinc-500"
                    }`}
                  >
                    {signal.active ? "ACTIVE" : "INACTIVE"}
                  </span>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] text-zinc-500">
                        Intensity
                      </span>
                      <span className="text-[10px] text-zinc-500">
                        {(signal.intensity * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${signal.intensity * 100}%`,
                          backgroundColor:
                            signal.intensity > 0.7
                              ? "#ef4444"
                              : signal.intensity > 0.4
                                ? "#f59e0b"
                                : "#6b7280",
                        }}
                      />
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-zinc-500">
                      Triggered {signal.trigger_count}x
                    </p>
                    <p className="text-[10px] text-zinc-600">
                      {formatTime(signal.last_triggered)}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Phantom Memories */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl">
          <div className="px-5 py-4 border-b border-zinc-800 flex items-center gap-2">
            <HelpCircle className="w-4 h-4 text-violet-400" />
            <h2 className="text-sm font-medium text-zinc-300">
              Phantom Memories
            </h2>
            <span className="ml-auto text-xs text-zinc-600">
              Knowledge gaps detected
            </span>
          </div>
          <div className="divide-y divide-zinc-800/50">
            {phantoms.map((phantom) => (
              <div
                key={phantom.id}
                className="px-5 py-4 hover:bg-zinc-800/30 transition-colors"
              >
                <div className="flex items-center justify-between mb-1">
                  <p className="text-sm text-zinc-300 font-mono">
                    "{phantom.query_pattern}"
                  </p>
                  <span className="text-xs text-violet-400/80">
                    {phantom.frequency} queries
                  </span>
                </div>
                <p className="text-xs text-zinc-500">
                  Suggested: {phantom.suggested_content}
                </p>
                <p className="text-[10px] text-zinc-600 mt-1">
                  Last queried {formatTime(phantom.last_queried)}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Trajectory */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl">
          <div className="px-5 py-4 border-b border-zinc-800 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-amber-400" />
            <h2 className="text-sm font-medium text-zinc-300">
              Predicted Trajectory
            </h2>
            <span className="ml-auto text-xs text-zinc-600">
              Next likely queries
            </span>
          </div>
          <div className="divide-y divide-zinc-800/50">
            {trajectory.map((pred, i) => (
              <div
                key={i}
                className="px-5 py-4 hover:bg-zinc-800/30 transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm text-zinc-300">"{pred.query}"</p>
                  <span className="text-xs font-mono text-amber-400/80">
                    {(pred.probability * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden mb-2">
                  <div
                    className="h-full bg-amber-500/50 rounded-full"
                    style={{ width: `${pred.probability * 100}%` }}
                  />
                </div>
                <p className="text-[10px] text-zinc-600">
                  Based on {pred.based_on.length} memory{" "}
                  {pred.based_on.length === 1 ? "node" : "nodes"}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
