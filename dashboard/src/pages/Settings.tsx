import { useState } from "react";
import { Save, RotateCcw } from "lucide-react";
import { api } from "../api";

export default function SettingsPage() {
  const [baseUrl, setBaseUrl] = useState("http://localhost:6677/v1");
  const [token, setToken] = useState("");
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    api.configure(baseUrl, token || undefined);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    setBaseUrl("http://localhost:6677/v1");
    setToken("");
    api.configure("/v1");
  };

  return (
    <div className="max-w-xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-zinc-100">Settings</h1>
        <p className="text-sm text-zinc-500 mt-0.5">
          Configure server connection and authentication
        </p>
      </div>

      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-4">
        <div>
          <label className="text-xs font-medium text-zinc-400 block mb-1.5">
            Server URL
          </label>
          <input
            type="text"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-sm text-zinc-200 font-mono focus:outline-none focus:border-emerald-500/50 transition-colors"
          />
          <p className="text-[10px] text-zinc-600 mt-1">
            Base URL for the MenteDB REST API (default: http://localhost:6677/v1)
          </p>
        </div>

        <div>
          <label className="text-xs font-medium text-zinc-400 block mb-1.5">
            Auth Token (optional)
          </label>
          <input
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="Bearer token for authenticated requests"
            className="w-full px-3 py-2 bg-zinc-950 border border-zinc-800 rounded-lg text-sm text-zinc-200 placeholder:text-zinc-700 focus:outline-none focus:border-emerald-500/50 transition-colors"
          />
        </div>

        <div className="flex items-center gap-3 pt-2">
          <button
            onClick={handleSave}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <Save className="w-3.5 h-3.5" />
            {saved ? "Saved" : "Save"}
          </button>
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-sm rounded-lg transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
        </div>
      </div>

      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
        <h2 className="text-sm font-medium text-zinc-300 mb-3">
          API Endpoints
        </h2>
        <div className="space-y-2 text-xs font-mono">
          {[
            ["GET", "/health", "Server health check"],
            ["GET", "/stats", "Database statistics"],
            ["POST", "/memories", "Store a new memory"],
            ["GET", "/memories/:id", "Retrieve a memory"],
            ["DELETE", "/memories/:id", "Delete a memory"],
            ["POST", "/search", "Vector similarity search"],
            ["POST", "/recall", "MQL context recall"],
            ["POST", "/edges", "Create a graph edge"],
            ["GET", "/ws/stream", "WebSocket cognition stream"],
          ].map(([method, path, desc]) => (
            <div key={path} className="flex items-center gap-3">
              <span
                className={`w-14 text-center text-[10px] font-semibold px-1.5 py-0.5 rounded ${
                  method === "GET"
                    ? "bg-blue-500/15 text-blue-400"
                    : method === "POST"
                      ? "bg-emerald-500/15 text-emerald-400"
                      : "bg-red-500/15 text-red-400"
                }`}
              >
                {method}
              </span>
              <span className="text-zinc-400">{path}</span>
              <span className="text-zinc-600 ml-auto">{desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
