import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Overview from "./pages/Overview";
import KnowledgeGraph from "./pages/KnowledgeGraph";
import MemoryExplorer from "./pages/MemoryExplorer";
import CognitiveMonitor from "./pages/CognitiveMonitor";
import AgentSpaces from "./pages/AgentSpaces";
import Settings from "./pages/Settings";

export default function App() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <Sidebar />
      <main className="ml-60 p-6 min-h-screen">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/graph" element={<KnowledgeGraph />} />
          <Route path="/memories" element={<MemoryExplorer />} />
          <Route path="/cognitive" element={<CognitiveMonitor />} />
          <Route path="/spaces" element={<AgentSpaces />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  );
}
