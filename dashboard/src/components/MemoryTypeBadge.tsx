import type { MemoryType } from "../api/types";

export const memoryTypeColors: Record<MemoryType, string> = {
  episodic: "#3b82f6",
  semantic: "#34d399",
  procedural: "#a78bfa",
  antipattern: "#f87171",
  reasoning: "#fbbf24",
  correction: "#f97316",
};

export const memoryTypeBadgeClass: Record<MemoryType, string> = {
  episodic: "bg-blue-500/15 text-blue-400 border-blue-500/25",
  semantic: "bg-emerald-500/15 text-emerald-400 border-emerald-500/25",
  procedural: "bg-violet-500/15 text-violet-400 border-violet-500/25",
  antipattern: "bg-red-500/15 text-red-400 border-red-500/25",
  reasoning: "bg-amber-500/15 text-amber-400 border-amber-500/25",
  correction: "bg-orange-500/15 text-orange-400 border-orange-500/25",
};

export function MemoryTypeBadge({ type }: { type: MemoryType }) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium border ${memoryTypeBadgeClass[type]}`}
    >
      {type}
    </span>
  );
}

export function formatTimestamp(microseconds: number): string {
  return new Date(microseconds / 1000).toLocaleString();
}

export function timeAgo(microseconds: number): string {
  const seconds = Math.floor((Date.now() - microseconds / 1000) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
