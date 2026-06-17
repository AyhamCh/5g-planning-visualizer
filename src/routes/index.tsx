import { createFileRoute } from "@tanstack/react-router";
import { useSuspenseQuery } from "@tanstack/react-query";
import { CheckCircle2, ArrowRight, Database, Cpu, Sparkles, Gauge, FileText, MessageSquare } from "lucide-react";
import { gridQuery, sitesQuery } from "@/lib/queries";
import { computeKpis, distribution, fmt, URBAN_COLORS, COVERAGE_COLORS, DEMAND_COLORS } from "@/lib/telco-data";
import { AppShell, Card, DistributionBar, PageHeader, Stat } from "@/components/AppShell";

export const Route = createFileRoute("/")({
  head: () => ({ meta: [{ title: "Overview · 5G Planner" }] }),
  component: Page,
});

const AGENTS = [
  "Urban Morphology Agent",
  "Terrain Environment Agent",
  "Population Demand Agent",
  "Coverage Agent",
  "Site Placement Agent",
  "Decision Agent",
  "RAG Agent",
  "Reporting Agent",
];

const PIPELINE = [
  { label: "Input Data", icon: Database },
  { label: "Agents Analysis", icon: Cpu },
  { label: "Optimization", icon: Gauge },
  { label: "Decision", icon: Sparkles },
  { label: "PDF Report", icon: FileText },
];

function Page() {
  const { data: grid } = useSuspenseQuery(gridQuery);
  const { data: sites } = useSuspenseQuery(sitesQuery);
  const k = computeKpis(grid);

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
          <Stat label="Antennas" value={fmt(k.antennas, 0)} unit={`NR ${k.antennasNr} / LTE ${k.antennasLte}`} />
        </div>

        {/* AI Agents Status + Pipeline Execution */}
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
              Le RAG Agent et le Reporting Agent sont accessibles depuis les onglets <span className="mono">/chat</span> et <span className="mono">/reports</span>.
            </div>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Card title="Urban classification">
            <DistributionBar items={distribution(grid, "urban_class")} colors={URBAN_COLORS} />
          </Card>
          <Card title="Coverage classification">
            <DistributionBar items={distribution(grid, "coverage_class")} colors={COVERAGE_COLORS} />
          </Card>
          <Card title="Demand classification">
            <DistributionBar items={distribution(grid, "demand_class")} colors={DEMAND_COLORS} />
          </Card>
          <Card title="Usage type">
            <DistributionBar items={distribution(grid, "usage_type")} />
          </Card>
          <Card title="Accessibility class">
            <DistributionBar items={distribution(grid, "accessibility_class")} />
          </Card>
          <Card title="Urban structure">
            <DistributionBar items={distribution(grid, "urban_structure")} />
          </Card>
        </div>

        <div className="mt-6 text-xs text-muted-foreground mono">
          Source: final_merged_grid.gpkg ({k.cells} cells, EPSG:32631 → 4326) ·
          recommended_sites_v3.gpkg ({sites.features.length} sites, EPSG:2154 → 4326)
        </div>
      </div>
    </AppShell>
  );
}
