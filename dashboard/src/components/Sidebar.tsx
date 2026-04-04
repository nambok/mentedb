import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Network,
  Database,
  Brain,
  Users,
  Settings,
  Activity,
} from "lucide-react";

const navItems = [
  { to: "/", icon: LayoutDashboard, label: "Overview" },
  { to: "/graph", icon: Network, label: "Knowledge Graph" },
  { to: "/memories", icon: Database, label: "Memory Explorer" },
  { to: "/cognitive", icon: Brain, label: "Cognitive Monitor" },
  { to: "/spaces", icon: Users, label: "Agent Spaces" },
];

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-screen w-60 bg-zinc-900 border-r border-zinc-800 flex flex-col z-50">
      <div className="p-5 border-b border-zinc-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center">
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-zinc-100">MenteDB</h1>
            <p className="text-[10px] text-zinc-500 tracking-wide uppercase">
              Dashboard
            </p>
          </div>
        </div>
      </div>

      <nav className="flex-1 py-3 px-2">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors mb-0.5 ${
                isActive
                  ? "bg-emerald-500/10 text-emerald-400"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
              }`
            }
          >
            <Icon className="w-4 h-4 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="p-3 border-t border-zinc-800">
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
              isActive
                ? "bg-emerald-500/10 text-emerald-400"
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
            }`
          }
        >
          <Settings className="w-4 h-4 shrink-0" />
          Settings
        </NavLink>
      </div>
    </aside>
  );
}
