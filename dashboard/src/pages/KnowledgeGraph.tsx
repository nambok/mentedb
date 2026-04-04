import { useCallback, useState, useRef, useEffect } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { X, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";
import { memoryTypeColors, MemoryTypeBadge } from "../components/MemoryTypeBadge";
import { useDemoData } from "../hooks/useData";
import type { GraphNode, MemoryType, EdgeType } from "../api/types";

const edgeTypeColors: Record<EdgeType, string> = {
  caused: "#f59e0b",
  before: "#6b7280",
  related: "#3b82f6",
  contradicts: "#ef4444",
  supports: "#22c55e",
  supersedes: "#a855f7",
  derived: "#06b6d4",
  partof: "#8b5cf6",
};

export default function KnowledgeGraph() {
  const { graph } = useDemoData();
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [filterType, setFilterType] = useState<MemoryType | "all">("all");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  const filteredGraph =
    filterType === "all"
      ? graph
      : {
          nodes: graph.nodes.filter((n) => n.memory_type === filterType),
          links: graph.links.filter((l) => {
            const nodeIds = new Set(
              graph.nodes
                .filter((n) => n.memory_type === filterType)
                .map((n) => n.id)
            );
            return nodeIds.has(l.source as string) && nodeIds.has(l.target as string);
          }),
        };

  const nodeColor = useCallback(
    (node: GraphNode) => memoryTypeColors[node.memory_type] || "#6b7280",
    []
  );

  const nodeSize = useCallback(
    (node: GraphNode) => 3 + node.salience * 8,
    []
  );

  const linkColor = useCallback(
    (link: { edge_type: EdgeType }) => edgeTypeColors[link.edge_type] || "#374151",
    []
  );

  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelected(node);
  }, []);

  const nodeLabel = useCallback(
    (node: GraphNode) => node.content.substring(0, 60),
    []
  );

  const types: MemoryType[] = [
    "episodic", "semantic", "procedural", "antipattern", "reasoning", "correction",
  ];

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100">
            Knowledge Graph
          </h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            {filteredGraph.nodes.length} nodes, {filteredGraph.links.length}{" "}
            edges
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 bg-zinc-900 border border-zinc-800 rounded-lg p-1">
            <button
              onClick={() => setFilterType("all")}
              className={`px-2.5 py-1 rounded text-xs transition-colors ${
                filterType === "all"
                  ? "bg-zinc-700 text-zinc-100"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              All
            </button>
            {types.map((t) => (
              <button
                key={t}
                onClick={() => setFilterType(t)}
                className={`px-2.5 py-1 rounded text-xs transition-colors ${
                  filterType === t
                    ? "bg-zinc-700 text-zinc-100"
                    : "text-zinc-500 hover:text-zinc-300"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1 ml-2">
            <button
              onClick={() => graphRef.current?.zoom(1.5, 300)}
              className="p-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-zinc-200 transition-colors"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={() => graphRef.current?.zoom(0.67, 300)}
              className="p-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-zinc-200 transition-colors"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              onClick={() => graphRef.current?.zoomToFit(400)}
              className="p-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-zinc-200 transition-colors"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 relative bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden" ref={containerRef}>
        <ForceGraph2D
          ref={graphRef}
          graphData={filteredGraph}
          width={dimensions.width}
          height={dimensions.height}
          nodeColor={nodeColor}
          nodeRelSize={1}
          nodeVal={nodeSize}
          nodeLabel={nodeLabel}
          linkColor={linkColor}
          linkWidth={(link: { weight: number }) => 0.5 + link.weight * 2}
          linkDirectionalArrowLength={3}
          linkDirectionalArrowRelPos={0.9}
          onNodeClick={handleNodeClick}
          backgroundColor="#09090b"
          cooldownTime={3000}
          enableNodeDrag={true}
        />

        <div className="absolute bottom-4 left-4 bg-zinc-900/90 border border-zinc-800 rounded-lg p-3 backdrop-blur-sm">
          <p className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">
            Node Types
          </p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            {types.map((t) => (
              <div key={t} className="flex items-center gap-1.5">
                <div
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: memoryTypeColors[t] }}
                />
                <span className="text-[11px] text-zinc-400">{t}</span>
              </div>
            ))}
          </div>
        </div>

        {selected && (
          <div className="absolute top-4 right-4 w-80 bg-zinc-900/95 border border-zinc-800 rounded-xl backdrop-blur-sm">
            <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
              <h3 className="text-sm font-medium text-zinc-200">
                Node Details
              </h3>
              <button
                onClick={() => setSelected(null)}
                className="text-zinc-500 hover:text-zinc-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="p-4 space-y-3">
              <div>
                <span className="text-[10px] text-zinc-500 uppercase tracking-wider">
                  Type
                </span>
                <div className="mt-1">
                  <MemoryTypeBadge type={selected.memory_type} />
                </div>
              </div>
              <div>
                <span className="text-[10px] text-zinc-500 uppercase tracking-wider">
                  Content
                </span>
                <p className="text-sm text-zinc-300 mt-1">
                  {selected.content}
                </p>
              </div>
              <div>
                <span className="text-[10px] text-zinc-500 uppercase tracking-wider">
                  Salience
                </span>
                <div className="flex items-center gap-2 mt-1">
                  <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full"
                      style={{ width: `${selected.salience * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-zinc-400">
                    {selected.salience.toFixed(2)}
                  </span>
                </div>
              </div>
              {selected.tags.length > 0 && (
                <div>
                  <span className="text-[10px] text-zinc-500 uppercase tracking-wider">
                    Tags
                  </span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {selected.tags.map((tag) => (
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
              <div>
                <span className="text-[10px] text-zinc-500 uppercase tracking-wider">
                  ID
                </span>
                <p className="text-xs text-zinc-500 mt-1 font-mono truncate">
                  {selected.id}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
