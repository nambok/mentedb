import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: string | number;
  icon: ReactNode;
  change?: string;
  variant?: "default" | "emerald" | "amber" | "rose";
}

const variants = {
  default: "border-zinc-800 bg-zinc-900",
  emerald: "border-emerald-500/20 bg-emerald-500/5",
  amber: "border-amber-500/20 bg-amber-500/5",
  rose: "border-rose-500/20 bg-rose-500/5",
};

export default function StatCard({
  label,
  value,
  icon,
  change,
  variant = "default",
}: StatCardProps) {
  return (
    <div
      className={`rounded-xl border p-5 ${variants[variant]}`}
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
          {label}
        </span>
        <div className="text-zinc-500">{icon}</div>
      </div>
      <div className="text-2xl font-semibold text-zinc-100">{value}</div>
      {change && (
        <p className="text-xs text-zinc-500 mt-1">{change}</p>
      )}
    </div>
  );
}
