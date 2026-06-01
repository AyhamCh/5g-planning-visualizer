import { createFileRoute } from "@tanstack/react-router";
import { useSuspenseQuery } from "@tanstack/react-query";
import { useState } from "react";
import { gridQuery, sitesQuery } from "@/lib/queries";
import { AppShell, Card, PageHeader } from "@/components/AppShell";
import { NetworkMap, type ColorBy } from "@/components/NetworkMap";
import { COVERAGE_COLORS, DEMAND_COLORS, SITE_COLORS, URBAN_COLORS } from "@/lib/telco-data";

export const Route = createFileRoute("/map")({
  head: () => ({ meta: [{ title: "Network Map · 5G Planner" }] }),
  loader: ({ context }) => {
    context.queryClient.ensureQueryData(gridQuery);
    context.queryClient.ensureQueryData(sitesQuery);
  },
  component: Page,
});

const MODES: Array<{ id: ColorBy; label: string }> = [
  { id: "coverage_class", label: "Coverage" },
  { id: "demand_class", label: "Demand" },
  { id: "urban_class", label: "Urban class" },
  { id: "los_probability", label: "LOS probability" },
];

function Legend({ mode }: { mode: ColorBy }) {
  if (mode === "los_probability") {
    return (
      <div>
        <div className="text-xs text-muted-foreground mb-1">LOS probability</div>
        <div className="h-2 rounded" style={{ background: "linear-gradient(to right, hsl(0 70% 50%), hsl(140 70% 50%))" }} />
        <div className="flex justify-between text-[10px] mono text-muted-foreground mt-1">
          <span>0%</span><span>100%</span>
        </div>
      </div>
    );
  }
  const map =
    mode === "coverage_class" ? COVERAGE_COLORS :
    mode === "demand_class" ? DEMAND_COLORS : URBAN_COLORS;
  return (
    <div className="space-y-1.5">
      {Object.entries(map).map(([k, c]) => (
        <div key={k} className="flex items-center gap-2 text-xs">
          <span className="inline-block size-3 rounded" style={{ background: c }} />
          <span>{k}</span>
        </div>
      ))}
    </div>
  );
}

function Page() {
  const { data: grid } = useSuspenseQuery(gridQuery);
  const { data: sites } = useSuspenseQuery(sitesQuery);
  const [mode, setMode] = useState<ColorBy>("coverage_class");
  const [showSites, setShowSites] = useState(true);

  return (
    <AppShell>
      <div className="mx-auto max-w-[1600px] px-6 py-6">
        <PageHeader
          title="Network Map"
          subtitle="Real cells from final_merged_grid.gpkg · real recommended sites from recommended_sites_v3.gpkg."
          right={
            <div className="flex items-center gap-2">
              {MODES.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setMode(m.id)}
                  className={[
                    "px-3 py-1.5 rounded-md text-xs border transition-colors",
                    mode === m.id
                      ? "bg-primary text-primary-foreground border-primary"
                      : "border-border text-muted-foreground hover:text-foreground",
                  ].join(" ")}
                >
                  {m.label}
                </button>
              ))}
              <label className="ml-2 flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={showSites}
                  onChange={(e) => setShowSites(e.target.checked)}
                />
                Show sites
              </label>
            </div>
          }
        />

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_260px] gap-4">
          <div className="rounded-lg border border-border bg-card overflow-hidden" style={{ height: "calc(100vh - 220px)" }}>
            <NetworkMap grid={grid} sites={sites} colorBy={mode} showSites={showSites} />
          </div>

          <div className="space-y-4">
            <Card title="Legend">
              <Legend mode={mode} />
            </Card>
            <Card title="Recommended sites">
              <div className="space-y-1.5">
                {Object.entries(SITE_COLORS).map(([k, c]) => (
                  <div key={k} className="flex items-center gap-2 text-xs">
                    <span className="inline-block size-3 rounded-full border border-white/60" style={{ background: c }} />
                    <span>{k}</span>
                  </div>
                ))}
                <div className="text-[10px] text-muted-foreground mt-2">
                  Marker radius ∝ composite_score
                </div>
              </div>
            </Card>
            <Card title="Source">
              <div className="text-xs mono text-muted-foreground space-y-1">
                <div>cells: {grid.features.length}</div>
                <div>sites: {sites.features.length}</div>
                <div>crs (export): EPSG:4326</div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
