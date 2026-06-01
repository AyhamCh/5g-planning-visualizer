import { Link, useRouterState } from "@tanstack/react-router";
import type { ReactNode } from "react";

const NAV = [
  { to: "/", label: "Overview" },
  { to: "/map", label: "Network Map" },
  { to: "/sites", label: "Recommended Sites" },
  { to: "/agents", label: "Agents" },
  { to: "/decision", label: "Decision" },
  { to: "/what-if", label: "What-If" },
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  const path = useRouterState({ select: (s) => s.location.pathname });
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-border bg-card/60 backdrop-blur sticky top-0 z-40">
        <div className="mx-auto max-w-[1600px] px-6 h-14 flex items-center gap-8">
          <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="inline-block size-2 rounded-full bg-primary shadow-[0_0_12px_var(--color-primary)]" />
            <span>5G Planner</span>
            <span className="mono text-xs text-muted-foreground ml-1">AI multi-agent</span>
          </Link>
          <nav className="flex items-center gap-1">
            {NAV.map((n) => {
              const active = n.to === "/" ? path === "/" : path.startsWith(n.to);
              return (
                <Link
                  key={n.to}
                  to={n.to}
                  className={[
                    "px-3 py-1.5 rounded-md text-sm transition-colors",
                    active
                      ? "bg-secondary text-foreground"
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary/60",
                  ].join(" ")}
                >
                  {n.label}
                </Link>
              );
            })}
          </nav>
          <div className="ml-auto mono text-xs text-muted-foreground">
            data · final_merged_grid.gpkg · recommended_sites_v3.gpkg
          </div>
        </div>
      </header>
      <main className="flex-1">{children}</main>
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
    <div className={`rounded-lg border border-border bg-card p-4 ${className}`}>
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
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{label}</div>
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
              className="h-full rounded-full"
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
