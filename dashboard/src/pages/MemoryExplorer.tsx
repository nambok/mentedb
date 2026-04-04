import { useState, useMemo } from "react";
import { Search, ChevronDown, ChevronUp, X } from "lucide-react";
import {
  MemoryTypeBadge,
  formatTimestamp,
  timeAgo,
} from "../components/MemoryTypeBadge";
import { useDemoData } from "../hooks/useData";
import type { MemoryNode, MemoryType } from "../api/types";

type SortField = "salience" | "created_at" | "accessed_at" | "access_count" | "confidence";
type SortDir = "asc" | "desc";

export default function MemoryExplorer() {
  const { memories } = useDemoData();
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<MemoryType | "all">("all");
  const [sortField, setSortField] = useState<SortField>("salience");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [expanded, setExpanded] = useState<string | null>(null);

  const filtered = useMemo(() => {
    let result = memories;
    if (typeFilter !== "all") {
      result = result.filter((m) => m.memory_type === typeFilter);
    }
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(
        (m) =>
          m.content.toLowerCase().includes(q) ||
          m.tags.some((t) => t.toLowerCase().includes(q)) ||
          m.id.includes(q)
      );
    }
    result = [...result].sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      return sortDir === "desc"
        ? (bVal as number) - (aVal as number)
        : (aVal as number) - (bVal as number);
    });
    return result;
  }, [memories, search, typeFilter, sortField, sortDir]);

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field)
      return <ChevronDown className="w-3 h-3 text-zinc-600" />;
    return sortDir === "desc" ? (
      <ChevronDown className="w-3 h-3 text-emerald-400" />
    ) : (
      <ChevronUp className="w-3 h-3 text-emerald-400" />
    );
  };

  const types: MemoryType[] = [
    "episodic", "semantic", "procedural", "antipattern", "reasoning", "correction",
  ];

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-zinc-100">
          Memory Explorer
        </h1>
        <p className="text-sm text-zinc-500 mt-0.5">
          {filtered.length} of {memories.length} memories
        </p>
      </div>

      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search memories..."
            className="w-full pl-9 pr-3 py-2 bg-zinc-900 border border-zinc-800 rounded-lg text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-emerald-500/50 transition-colors"
          />
          {search && (
            <button
              onClick={() => setSearch("")}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
        <div className="flex items-center gap-1 bg-zinc-900 border border-zinc-800 rounded-lg p-1">
          <button
            onClick={() => setTypeFilter("all")}
            className={`px-2.5 py-1 rounded text-xs transition-colors ${
              typeFilter === "all"
                ? "bg-zinc-700 text-zinc-100"
                : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            All
          </button>
          {types.map((t) => (
            <button
              key={t}
              onClick={() => setTypeFilter(t)}
              className={`px-2.5 py-1 rounded text-xs transition-colors ${
                typeFilter === t
                  ? "bg-zinc-700 text-zinc-100"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800">
              <th className="text-left px-4 py-3 text-xs font-medium text-zinc-500 uppercase tracking-wider">
                Type
              </th>
              <th className="text-left px-4 py-3 text-xs font-medium text-zinc-500 uppercase tracking-wider">
                Content
              </th>
              <th
                className="text-left px-4 py-3 text-xs font-medium text-zinc-500 uppercase tracking-wider cursor-pointer select-none"
                onClick={() => toggleSort("salience")}
              >
                <span className="flex items-center gap-1">
                  Salience <SortIcon field="salience" />
                </span>
              </th>
              <th
                className="text-left px-4 py-3 text-xs font-medium text-zinc-500 uppercase tracking-wider cursor-pointer select-none"
                onClick={() => toggleSort("confidence")}
              >
                <span className="flex items-center gap-1">
                  Confidence <SortIcon field="confidence" />
                </span>
              </th>
              <th
                className="text-left px-4 py-3 text-xs font-medium text-zinc-500 uppercase tracking-wider cursor-pointer select-none"
                onClick={() => toggleSort("created_at")}
              >
                <span className="flex items-center gap-1">
                  Created <SortIcon field="created_at" />
                </span>
              </th>
              <th
                className="text-left px-4 py-3 text-xs font-medium text-zinc-500 uppercase tracking-wider cursor-pointer select-none"
                onClick={() => toggleSort("access_count")}
              >
                <span className="flex items-center gap-1">
                  Accesses <SortIcon field="access_count" />
                </span>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {filtered.map((m) => (
              <MemoryRow
                key={m.id}
                memory={m}
                isExpanded={expanded === m.id}
                onToggle={() =>
                  setExpanded(expanded === m.id ? null : m.id)
                }
              />
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <div className="px-4 py-12 text-center text-zinc-600 text-sm">
            No memories match your search criteria
          </div>
        )}
      </div>
    </div>
  );
}

function MemoryRow({
  memory,
  isExpanded,
  onToggle,
}: {
  memory: MemoryNode;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr
        className="hover:bg-zinc-800/30 transition-colors cursor-pointer"
        onClick={onToggle}
      >
        <td className="px-4 py-3">
          <MemoryTypeBadge type={memory.memory_type} />
        </td>
        <td className="px-4 py-3 text-zinc-300 max-w-md truncate">
          {memory.content}
        </td>
        <td className="px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500/70 rounded-full"
                style={{ width: `${memory.salience * 100}%` }}
              />
            </div>
            <span className="text-xs text-zinc-500">
              {memory.salience.toFixed(2)}
            </span>
          </div>
        </td>
        <td className="px-4 py-3 text-xs text-zinc-400">
          {memory.confidence.toFixed(2)}
        </td>
        <td className="px-4 py-3 text-xs text-zinc-500">
          {timeAgo(memory.created_at)}
        </td>
        <td className="px-4 py-3 text-xs text-zinc-500">
          {memory.access_count}
        </td>
      </tr>
      {isExpanded && (
        <tr className="bg-zinc-800/20">
          <td colSpan={6} className="px-4 py-4">
            <div className="grid grid-cols-2 gap-4 text-xs">
              <div>
                <span className="text-zinc-500 block mb-1">Full Content</span>
                <p className="text-zinc-300">{memory.content}</p>
              </div>
              <div className="space-y-2">
                <div>
                  <span className="text-zinc-500">ID</span>
                  <p className="text-zinc-400 font-mono">{memory.id}</p>
                </div>
                <div>
                  <span className="text-zinc-500">Agent</span>
                  <p className="text-zinc-400 font-mono">{memory.agent_id}</p>
                </div>
                <div>
                  <span className="text-zinc-500">Space</span>
                  <p className="text-zinc-400 font-mono">{memory.space_id}</p>
                </div>
                <div>
                  <span className="text-zinc-500">Created</span>
                  <p className="text-zinc-400">
                    {formatTimestamp(memory.created_at)}
                  </p>
                </div>
                <div>
                  <span className="text-zinc-500">Last Accessed</span>
                  <p className="text-zinc-400">
                    {formatTimestamp(memory.accessed_at)}
                  </p>
                </div>
                {memory.tags.length > 0 && (
                  <div>
                    <span className="text-zinc-500">Tags</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {memory.tags.map((tag) => (
                        <span
                          key={tag}
                          className="px-1.5 py-0.5 text-[10px] bg-zinc-800 text-zinc-400 rounded"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
