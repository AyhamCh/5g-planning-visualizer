import { createFileRoute } from "@tanstack/react-router";
import { useSuspenseQuery } from "@tanstack/react-query";
import { gridQuery, sitesQuery } from "@/lib/queries";
import { computeKpis, distribution, fmt, URBAN_COLORS, COVERAGE_COLORS, DEMAND_COLORS } from "@/lib/telco-data";
import { AppShell, Card, DistributionBar, PageHeader, Stat } from "@/components/AppShell";

export const Route = createFileRoute("/")({
  head: () => ({ meta: [{ title: "Overview · 5G Planner" }] }),
  loader: ({ context }) => {
    context.queryClient.ensureQueryData(gridQuery);
    context.queryClient.ensureQueryData(sitesQuery);
  },
  component: Page,
});

function Page() {
  const { data: grid } = useSuspenseQuery(gridQuery);
  const { data: sites } = useSuspenseQuery(sitesQuery);
  const k = computeKpis(grid);

  return (
    <AppShell>
      <div className="mx-auto max-w-[1600px] px-6 py-8">
        <PageHeader
          title="Network Overview"
          subtitle="Real outputs from the 5G AI multi-agent pipeline (osmnx · demand · terrain · coverage · decision)."
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
