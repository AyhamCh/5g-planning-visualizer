import { createFileRoute } from "@tanstack/react-router";
import { useSuspenseQuery } from "@tanstack/react-query";
import {
  CheckCircle2,
  ArrowRight,
  Database,
  Cpu,
  Sparkles,
  Gauge,
  FileText,
  MessageSquare,
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
} from "recharts";
import { gridQuery, sitesQuery } from "@/lib/queries";
import {
  computeKpis,
  distribution,
  fmt,
  URBAN_COLORS,
  COVERAGE_COLORS,
  DEMAND_COLORS,
  SITE_COLORS,
} from "@/lib/telco-data";
import { AppShell, Card, ClientOnly, PageHeader, Stat } from "@/components/AppShell";

export const Route = createFileRoute("/")({
  head: () => ({ meta: [{ title: "Overview · 5G Planner" }] }),
  component: Page,
});

const AGENTS = [
  "Urban Morphology (OSMnx)",
  "Terrain Environment",
  "Population Demand",
  "Coverage Agent",
  "Site Placement",
  "Decision Agent",
  "RAG Agent (ChromaDB · Qwen)",
  "LLM Reporting Agent",
];

const PIPELINE = [
  { label: "Input Data", icon: Database },
  { label: "Agents Analysis", icon: Cpu },
  { label: "Optimization", icon: Gauge },
  { label: "Decision", icon: Sparkles },
  { label: "PDF Report", icon: FileText },
];

function Page() {
  return (
    <AppShell>
      <div className="mx-auto max-w-[1600px] px-6 py-8">
        <PageHeader
          title="Network Overview"
          subtitle="Real outputs from the 5G AI multi-agent pipeline (osmnx · demand · terrain · coverage · decision · RAG · reporting)."
          right={
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-[var(--color-success)]/40 bg-[var(--color-success)]/10 text-xs">
              <span className="size-1.5 rounded-full bg-[var(--color-success)] animate-pulse" />
              <span className="mono text-[var(--color-success)]">Pipeline ready</span>
            </div>
          }
        />

        <ClientOnly
          fallback={
            <div className="rounded-xl border border-border bg-card/40 p-10 text-sm text-muted-foreground">
              Loading pipeline outputs…
            </div>
          }
        >
          <Dashboard />
        </ClientOnly>
      </div>
    </AppShell>
  );
}

function Dashboard() {
  const { data: grid } = useSuspenseQuery(gridQuery);
  const { data: sites } = useSuspenseQuery(sitesQuery);
  const k = computeKpis(grid);

  const urbanData = distribution(grid, "urban_class");
  const coverageData = distribution(grid, "coverage_class");
  const demandData = distribution(grid, "demand_class");
  const siteCounts = new Map<string, number>();
  for (const f of sites.features) {
    const t = (f.properties.site_type as string) ?? "macro_cell";
    siteCounts.set(t, (siteCounts.get(t) ?? 0) + 1);
  }
  const siteTotal = sites.features.length || 1;
  const siteData = [...siteCounts.entries()]
    .map(([label, count]) => ({ label, count, ratio: count / siteTotal }))
    .sort((a, b) => b.count - a.count);

  return (
    <>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
        <Stat label="Grid cells" value={fmt(k.cells, 0)} />
        <Stat label="Population" value={fmt(k.population, 0)} unit="hab" />
        <Stat label="Peak demand" value={fmt(k.totalPeakDemandGbps, 1)} unit="Gbps" />
        <Stat label="Capacity" value={fmt(k.totalCapacityGbps, 1)} unit="Gbps" />
        <Stat
          label="Total deficit"
          value={fmt(k.totalDeficitGbps, 1)}
          unit="Gbps"
          tone={k.totalDeficitGbps > 0 ? "bad" : "good"}
        />
        <Stat label="Recommended sites" value={fmt(sites.features.length, 0)} tone="good" />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
        <Stat
          label="Coverage ratio"
          value={`${(k.coverageRatio * 100).toFixed(1)}%`}
          tone={k.coverageRatio > 0.9 ? "good" : "warn"}
        />
        <Stat
          label="5G cells"
          value={`${(k.has5gRatio * 100).toFixed(1)}%`}
          tone={k.has5gRatio > 0.3 ? "good" : "warn"}
        />
        <Stat label="Hotspot risk" value={fmt(k.hotspotRiskCount, 0)} unit="cells" tone="warn" />
        <Stat
          label="Antennas"
          value={fmt(k.antennas, 0)}
          unit={`NR ${k.antennasNr} / LTE ${k.antennasLte}`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.3fr] gap-4 mb-8">
        <Card title="AI Agents Status">
          <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {AGENTS.map((a) => (
              <li
                key={a}
                className="flex items-center gap-2 px-3 py-2 rounded-md border border-border bg-secondary/40 text-sm"
              >
                <CheckCircle2 className="size-4 text-[var(--color-success)] shrink-0" />
                <span className="truncate">{a}</span>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Pipeline Execution">
          <div className="flex items-stretch gap-2 overflow-x-auto py-2">
            {PIPELINE.map((step, i) => {
              const Icon = step.icon;
              return (
                <div key={step.label} className="flex items-center gap-2 shrink-0">
                  <div className="flex flex-col items-center gap-2 min-w-[110px]">
                    <div className="size-12 rounded-xl bg-primary/15 border border-primary/30 flex items-center justify-center shadow-[0_0_24px_-8px_var(--color-primary)]">
                      <Icon className="size-5 text-primary" />
                    </div>
                    <div className="text-xs text-center text-foreground font-medium">
                      {step.label}
                    </div>
                  </div>
                  {i < PIPELINE.length - 1 && (
                    <ArrowRight className="size-4 text-muted-foreground shrink-0" />
                  )}
                </div>
              );
            })}
          </div>
          <div className="mt-4 flex items-center gap-2 text-xs text-muted-foreground">
            <MessageSquare className="size-3.5" />
            Le RAG Agent et le Reporting Agent sont accessibles depuis{" "}
            <span className="mono text-primary">/chat</span> et{" "}
            <span className="mono text-primary">/reports</span>.
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        <Card title="Urban classification">
          <PieDist data={urbanData} colors={URBAN_COLORS} />
        </Card>
        <Card title="Coverage classification">
          <PieDist data={coverageData} colors={COVERAGE_COLORS} />
        </Card>
        <Card title="Demand classification">
          <PieDist data={demandData} colors={DEMAND_COLORS} />
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="Recommended site types">
          <BarDist data={siteData} colors={SITE_COLORS} />
        </Card>
        <Card title="Capacity vs Peak demand (Gbps)">
          <BarDist
            data={[
              { label: "Capacity", count: Math.round(k.totalCapacityGbps), ratio: 1 },
              { label: "Peak demand", count: Math.round(k.totalPeakDemandGbps), ratio: 1 },
              { label: "Deficit", count: Math.round(k.totalDeficitGbps), ratio: 1 },
            ]}
            colors={{ Capacity: "#22c55e", "Peak demand": "#3b82f6", Deficit: "#ef4444" }}
          />
        </Card>
      </div>

      <div className="mt-6 text-xs text-muted-foreground mono">
        Source: results/final_merged_grid.gpkg ({k.cells} cells) ·
        results/recommended_sites_v3.gpkg ({sites.features.length} sites)
      </div>
    </>
  );
}

const PIE_PALETTE = ["#3b82f6", "#a855f7", "#ec4899", "#f97316", "#eab308", "#22c55e", "#06b6d4"];

function PieDist({
  data,
  colors,
}: {
  data: Array<{ label: string; count: number; ratio: number }>;
  colors?: Record<string, string>;
}) {
  if (!data.length) return <div className="text-sm text-muted-foreground">No data.</div>;
  return (
    <div style={{ width: "100%", height: 240 }}>
      <ResponsiveContainer>
        <PieChart>
          <Pie
            data={data}
            dataKey="count"
            nameKey="label"
            innerRadius={48}
            outerRadius={84}
            paddingAngle={2}
          >
            {data.map((d, i) => (
              <Cell
                key={d.label}
                fill={colors?.[d.label] ?? PIE_PALETTE[i % PIE_PALETTE.length]}
                stroke="rgba(0,0,0,0.4)"
              />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              background: "var(--color-card)",
              border: "1px solid var(--color-border)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

function BarDist({
  data,
  colors,
}: {
  data: Array<{ label: string; count: number; ratio: number }>;
  colors?: Record<string, string>;
}) {
  if (!data.length) return <div className="text-sm text-muted-foreground">No data.</div>;
  return (
    <div style={{ width: "100%", height: 240 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 12, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis dataKey="label" tick={{ fill: "var(--color-muted-foreground)", fontSize: 11 }} />
          <YAxis tick={{ fill: "var(--color-muted-foreground)", fontSize: 11 }} />
          <Tooltip
            contentStyle={{
              background: "var(--color-card)",
              border: "1px solid var(--color-border)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <Bar dataKey="count" radius={[6, 6, 0, 0]}>
            {data.map((d, i) => (
              <Cell
                key={d.label}
                fill={colors?.[d.label] ?? PIE_PALETTE[i % PIE_PALETTE.length]}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
