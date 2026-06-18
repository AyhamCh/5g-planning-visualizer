import { Link, useRouterState } from "@tanstack/react-router";
import { useEffect, useState, type ReactNode } from "react";
import {
  LayoutDashboard,
  Map as MapIcon,
  Radio,
  Cpu,
  Sparkles,
  FlaskConical,
  MessageSquare,
  FileText,
} from "lucide-react";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/map", label: "Network Map", icon: MapIcon },
  { to: "/sites", label: "Recommended Sites", icon: Radio },
  { to: "/agents", label: "Agents", icon: Cpu },
  { to: "/decision", label: "Decision", icon: Sparkles },
  { to: "/what-if", label: "What-If", icon: FlaskConical },
  { to: "/chat", label: "5G AI Assistant", icon: MessageSquare },
  { to: "/reports", label: "Reports", icon: FileText },
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  const path = useRouterState({ select: (s) => s.location.pathname });
  return (
    <div className="min-h-screen flex">
      <aside className="w-60 shrink-0 border-r border-border bg-card/40 backdrop-blur sticky top-0 h-screen flex flex-col">
        <Link to="/" className="flex items-center gap-2 px-5 h-16 border-b border-border">
          <span className="relative inline-flex">
            <span className="inline-block size-2.5 rounded-full bg-primary shadow-[0_0_16px_var(--color-primary)]" />
            <span className="absolute inset-0 rounded-full bg-primary/40 animate-ping" />
          </span>
          <div className="leading-tight">
            <div className="font-semibold tracking-tight">5G Planner</div>
            <div className="mono text-[10px] text-muted-foreground uppercase">AI multi-agent</div>
          </div>
        </Link>
        <nav className="flex-1 overflow-y-auto p-3 space-y-1">
          {NAV.map((n) => {
            const Icon = n.icon;
            const active = n.to === "/" ? path === "/" : path.startsWith(n.to);
            return (
              <Link
                key={n.to}
                to={n.to}
                className={[
                  "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-all",
                  active
                    ? "bg-primary/15 text-foreground border border-primary/30 shadow-[0_0_20px_-8px_var(--color-primary)]"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/60 border border-transparent",
                ].join(" ")}
              >
                <Icon className="size-4" />
                <span>{n.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="px-4 py-3 border-t border-border mono text-[10px] text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <span className="size-1.5 rounded-full bg-[var(--color-success)]" />
            Pipeline online
          </div>
          <div className="mt-1 opacity-70">final_merged_grid · v3 sites</div>
        </div>
      </aside>
      <main className="flex-1 min-w-0">{children}</main>
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  right,
}: {
  title: string;
  subtitle?: string;
  right?: ReactNode;
}) {
  return (
    <div className="flex items-end justify-between gap-6 mb-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

export function Card({
  title,
  children,
  className = "",
}: {
  title?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl border border-border bg-card/80 backdrop-blur p-4 shadow-[0_1px_0_0_rgba(255,255,255,0.04)_inset,0_8px_24px_-12px_rgba(0,0,0,0.5)] transition-all hover:border-primary/30 ${className}`}
    >
      {title && (
        <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground mb-3">
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

export function Stat({
  label,
  value,
  unit,
  tone = "default",
}: {
  label: string;
  value: string | number;
  unit?: string;
  tone?: "default" | "good" | "warn" | "bad";
}) {
  const toneCls =
    tone === "good"
      ? "text-[var(--color-success)]"
      : tone === "warn"
        ? "text-[var(--color-warning)]"
        : tone === "bad"
          ? "text-[var(--color-critical)]"
          : "text-foreground";
  return (
    <div className="rounded-xl border border-border bg-card/80 backdrop-blur p-4 transition-all hover:border-primary/30">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={`mt-2 text-2xl font-semibold mono ${toneCls}`}>
        {value}
        {unit && <span className="text-sm text-muted-foreground ml-1">{unit}</span>}
      </div>
    </div>
  );
}

export function DistributionBar({
  items,
  colors,
}: {
  items: Array<{ label: string; count: number; ratio: number }>;
  colors?: Record<string, string>;
}) {
  return (
    <div className="space-y-2">
      {items.map((it) => (
        <div key={it.label}>
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-foreground">{it.label}</span>
            <span className="mono text-muted-foreground">
              {it.count} · {(it.ratio * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${it.ratio * 100}%`,
                background: colors?.[it.label] ?? "var(--color-primary)",
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
