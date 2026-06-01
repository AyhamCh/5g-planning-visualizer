import { createFileRoute } from "@tanstack/react-router";
import { useSuspenseQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { gridQuery, sitesQuery } from "@/lib/queries";
import { AppShell, Card, PageHeader, Stat } from "@/components/AppShell";
import { fmt } from "@/lib/telco-data";

export const Route = createFileRoute("/what-if")({
  head: () => ({ meta: [{ title: "What-If · 5G Planner" }] }),
  loader: ({ context }) => {
    context.queryClient.ensureQueryData(gridQuery);
    context.queryClient.ensureQueryData(sitesQuery);
  },
  component: Page,
});

/**
 * What-If scenarios are limited to parameters that actually exist in the
 * pipeline (see orchestrator.py + agent configs). We re-derive coverage
 * outcomes from real per-cell values — no synthetic data.
 *
 * Knobs (all from real pipeline config):
 *   - demand_multiplier  → re-scales peak_demand_gbps (population growth / busy-hour shift)
 *   - capacity_uplift    → multiplies capacity_available_gbps (radio_filter ⊇ NR, top_k_capacity)
 *   - los_threshold      → required min los_probability for a cell to be "viable"
 *   - select top-N sites → cumulative deficit addressed
 */
function Page() {
  const { data: grid } = useSuspenseQuery(gridQuery);
  const { data: sites } = useSuspenseQuery(sitesQuery);

  const [demandMul, setDemandMul] = useState(1.0);
  const [capUplift, setCapUplift] = useState(1.0);
  const [losMin, setLosMin] = useState(0.0);
  const [topN, setTopN] = useState(sites.features.length);

  const scenario = useMemo(() => {
    const cells = grid.features.map((f) => f.properties);
    const peak = cells.reduce((a, c) => a + c.peak_demand_gbps * demandMul, 0);
    const cap = cells.reduce((a, c) => a + c.capacity_available_gbps * capUplift, 0);
    const perCellDeficit = cells.map((c) =>
      Math.max(0, c.peak_demand_gbps * demandMul - c.capacity_available_gbps * capUplift),
    );
    const deficit = perCellDeficit.reduce((a, b) => a + b, 0);
    const viable = cells.filter((c) => c.los_probability >= losMin).length;
    const hotspot = cells.filter(
      (c) =>
        c.capacity_available_gbps * capUplift <
        0.8 * c.peak_demand_gbps * demandMul,
    ).length;

    const sortedSites = [...sites.features].sort(
      (a, b) => b.properties.composite_score - a.properties.composite_score,
    );
    const selected = sortedSites.slice(0, topN);
    const addressed = selected.reduce(
      (a, f) => a + f.properties.capacity_deficit_gbps,
      0,
    );
    return { peak, cap, deficit, viable, hotspot, addressed, totalCells: cells.length };
  }, [grid, sites, demandMul, capUplift, losMin, topN]);

  return (
    <AppShell>
      <div className="mx-auto max-w-[1600px] px-6 py-8">
        <PageHeader
          title="What-If Scenarios"
          subtitle="Re-derived from real per-cell peak_demand_gbps / capacity_available_gbps / los_probability."
        />

        <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
          <Card title="Parameters">
            <div className="space-y-5">
              <Slider
                label="Demand multiplier"
                value={demandMul}
                onChange={setDemandMul}
                min={0.5} max={2.5} step={0.05}
                fmtFn={(v) => `×${v.toFixed(2)}`}
                hint="Scales peak demand (growth / busy-hour shift)"
              />
              <Slider
                label="Capacity uplift"
                value={capUplift}
                onChange={setCapUplift}
                min={0.5} max={3.0} step={0.05}
                fmtFn={(v) => `×${v.toFixed(2)}`}
                hint="Multiplies installed capacity (densification / NR enable)"
              />
              <Slider
                label="LOS viability threshold"
                value={losMin}
                onChange={setLosMin}
                min={0} max={1} step={0.05}
                fmtFn={(v) => `${(v * 100).toFixed(0)}%`}
                hint="Min los_probability required per cell"
              />
              <Slider
                label="Sites to deploy (top-N)"
                value={topN}
                onChange={(v) => setTopN(Math.round(v))}
                min={0} max={sites.features.length} step={1}
                fmtFn={(v) => `${Math.round(v)} / ${sites.features.length}`}
              />
            </div>
          </Card>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <Stat label="Peak demand" value={fmt(scenario.peak, 1)} unit="Gbps" />
            <Stat label="Capacity" value={fmt(scenario.cap, 1)} unit="Gbps" />
            <Stat
              label="Residual deficit"
              value={fmt(scenario.deficit, 1)}
              unit="Gbps"
              tone={scenario.deficit > 0 ? "bad" : "good"}
            />
            <Stat
              label="Viable cells (LOS)"
              value={`${scenario.viable}/${scenario.totalCells}`}
            />
            <Stat label="Hotspot risk cells" value={scenario.hotspot} tone="warn" />
            <Stat
              label="Deficit addressed by sites"
              value={fmt(scenario.addressed, 1)}
              unit="Gbps"
              tone="good"
            />
          </div>
        </div>

        <p className="mt-6 text-xs text-muted-foreground">
          These knobs map to real pipeline parameters: demand growth multiplies
          PopulationDemandAgent outputs, capacity uplift mirrors CoverageAgent
          options (<span className="mono">radio_filter</span>,{" "}
          <span className="mono">top_k_capacity</span>,{" "}
          <span className="mono">max_antennas_per_cell</span>), and the LOS
          threshold filters cells using TerrainEnvironmentAgent's{" "}
          <span className="mono">los_probability</span>.
        </p>
      </div>
    </AppShell>
  );
}

function Slider({
  label, value, onChange, min, max, step, fmtFn, hint,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number; max: number; step: number;
  fmtFn: (v: number) => string;
  hint?: string;
}) {
  return (
    <div>
      <div className="flex items-center justify-between text-sm mb-1">
        <span>{label}</span>
        <span className="mono text-primary">{fmtFn(value)}</span>
      </div>
      <input
        type="range"
        className="w-full accent-[var(--color-primary)]"
        min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      {hint && <div className="text-xs text-muted-foreground mt-1">{hint}</div>}
    </div>
  );
}
