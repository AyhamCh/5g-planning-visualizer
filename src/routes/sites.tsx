import { createFileRoute } from "@tanstack/react-router";
import { useSuspenseQuery } from "@tanstack/react-query";
import { sitesQuery } from "@/lib/queries";
import { AppShell, Card, PageHeader } from "@/components/AppShell";
import { SITE_COLORS, fmt } from "@/lib/telco-data";

export const Route = createFileRoute("/sites")({
  head: () => ({ meta: [{ title: "Recommended sites · 5G Planner" }] }),
  loader: ({ context }) => context.queryClient.ensureQueryData(sitesQuery),
  component: Page,
});

function Page() {
  const { data: sites } = useSuspenseQuery(sitesQuery);
  const rows = [...sites.features].sort(
    (a, b) => b.properties.composite_score - a.properties.composite_score,
  );

  return (
    <AppShell>
      <div className="mx-auto max-w-[1600px] px-6 py-8">
        <PageHeader
          title="Recommended Sites"
          subtitle={`Ranked by composite_score from SitePlacementAgent. Source: recommended_sites_v3.gpkg.`}
        />

        <Card>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase text-muted-foreground border-b border-border">
                  <th className="py-2 pr-3">Rank</th>
                  <th className="py-2 pr-3">Site</th>
                  <th className="py-2 pr-3">Type</th>
                  <th className="py-2 pr-3">Urban class</th>
                  <th className="py-2 pr-3 text-right">Score</th>
                  <th className="py-2 pr-3 text-right">Deficit (Gbps)</th>
                  <th className="py-2 pr-3 text-right">Peak demand</th>
                  <th className="py-2 pr-3 text-right">Pop. density</th>
                  <th className="py-2 pr-3 text-right">LOS</th>
                  <th className="py-2 pr-3 text-right">Attenuation</th>
                  <th className="py-2 pr-3 text-right">Coords (lat, lng)</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((f, i) => {
                  const p = f.properties;
                  const [lng, lat] = f.geometry.coordinates;
                  return (
                    <tr key={p.site_id} className="border-b border-border/60 hover:bg-secondary/40">
                      <td className="py-2 pr-3 mono">#{i + 1}</td>
                      <td className="py-2 pr-3 mono">#{p.site_id}</td>
                      <td className="py-2 pr-3">
                        <span className="inline-flex items-center gap-2">
                          <span className="size-2 rounded-full" style={{ background: SITE_COLORS[p.site_type] ?? "#a855f7" }} />
                          {p.site_type}
                        </span>
                      </td>
                      <td className="py-2 pr-3">{p.urban_class}</td>
                      <td className="py-2 pr-3 mono text-right">{p.composite_score.toFixed(4)}</td>
                      <td className="py-2 pr-3 mono text-right">{fmt(p.capacity_deficit_gbps, 2)}</td>
                      <td className="py-2 pr-3 mono text-right">{fmt(p.peak_demand_gbps, 2)}</td>
                      <td className="py-2 pr-3 mono text-right">{fmt(p.population_density, 0)}</td>
                      <td className="py-2 pr-3 mono text-right">{(p.los_probability * 100).toFixed(0)}%</td>
                      <td className="py-2 pr-3 mono text-right">{fmt(p.attenuation_factor, 1)} dB</td>
                      <td className="py-2 pr-3 mono text-right text-muted-foreground">
                        {lat.toFixed(5)}, {lng.toFixed(5)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
