import { createFileRoute } from "@tanstack/react-router";
import { useSuspenseQuery } from "@tanstack/react-query";
import { sitesQuery } from "@/lib/queries";
import { AppShell, Card, PageHeader } from "@/components/AppShell";
import { SITE_COLORS, fmt } from "@/lib/telco-data";

export const Route = createFileRoute("/decision")({
  head: () => ({ meta: [{ title: "Decision · 5G Planner" }] }),
  loader: ({ context }) => context.queryClient.ensureQueryData(sitesQuery),
  component: Page,
});

// Justification derived from SitePlacementAgent rules visible in the report:
//   capacity deficit, overload, population density, LOS limited.
function justify(p: {
  capacity_deficit_gbps: number;
  overload_score: number;
  population_density: number;
  los_probability: number;
}): string[] {
  const out: string[] = [];
  if (p.capacity_deficit_gbps >= 4) out.push(`critical capacity deficit (${p.capacity_deficit_gbps.toFixed(1)} Gbps)`);
  else if (p.capacity_deficit_gbps >= 2) out.push(`significant capacity deficit (${p.capacity_deficit_gbps.toFixed(1)} Gbps)`);
  else if (p.capacity_deficit_gbps >= 0.5) out.push(`moderate capacity deficit (${p.capacity_deficit_gbps.toFixed(2)} Gbps)`);
  if (p.overload_score >= 0.75) out.push(`critical overload risk (${p.overload_score.toFixed(2)})`);
  else if (p.overload_score >= 0.5) out.push(`high overload risk (${p.overload_score.toFixed(2)})`);
  if (p.population_density >= 20000) out.push(`high population density (${Math.round(p.population_density).toLocaleString()} hab/km²)`);
  if (p.los_probability < 0.45) out.push(`limited LOS (${Math.round(p.los_probability * 100)}%) → small cell recommended`);
  return out;
}

function Page() {
  const { data: sites } = useSuspenseQuery(sitesQuery);
  const rows = [...sites.features].sort(
    (a, b) => b.properties.composite_score - a.properties.composite_score,
  );
  const maxScore = Math.max(...rows.map((r) => r.properties.composite_score));

  return (
    <AppShell>
      <div className="mx-auto max-w-[1600px] px-6 py-8">
        <PageHeader
          title="Decision"
          subtitle="Ranking from SitePlacementAgent. composite_score is the agent-produced priority. Justification mirrors the agent rules."
        />

        <div className="space-y-3">
          {rows.map((f, i) => {
            const p = f.properties;
            const [lng, lat] = f.geometry.coordinates;
            const reasons = justify(p);
            return (
              <Card key={p.site_id}>
                <div className="flex items-start gap-4">
                  <div className="text-xs mono w-10 text-muted-foreground">#{i + 1}</div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3 flex-wrap">
                      <span className="font-semibold">Site #{p.site_id}</span>
                      <span
                        className="text-xs px-2 py-0.5 rounded-full"
                        style={{ background: `${SITE_COLORS[p.site_type] ?? "#a855f7"}33`, color: SITE_COLORS[p.site_type] ?? "#a855f7" }}
                      >
                        {p.site_type}
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-secondary text-muted-foreground">
                        {p.urban_class}
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-secondary text-muted-foreground mono">
                        {lat.toFixed(5)}, {lng.toFixed(5)}
                      </span>
                    </div>
                    {reasons.length > 0 && (
                      <ul className="mt-2 text-sm text-muted-foreground list-disc list-inside space-y-0.5">
                        {reasons.map((r) => <li key={r}>{r}</li>)}
                      </ul>
                    )}
                  </div>
                  <div className="w-56">
                    <div className="text-xs text-muted-foreground mb-1 flex justify-between">
                      <span>composite_score</span>
                      <span className="mono">{p.composite_score.toFixed(4)}</span>
                    </div>
                    <div className="h-2 rounded-full bg-secondary overflow-hidden">
                      <div
                        className="h-full"
                        style={{
                          width: `${(p.composite_score / maxScore) * 100}%`,
                          background: "var(--color-primary)",
                        }}
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
                      <div>
                        <div className="text-muted-foreground">deficit</div>
                        <div className="mono">{fmt(p.capacity_deficit_gbps, 2)} Gbps</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">peak demand</div>
                        <div className="mono">{fmt(p.peak_demand_gbps, 2)} Gbps</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">LOS</div>
                        <div className="mono">{(p.los_probability * 100).toFixed(0)}%</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">pop. density</div>
                        <div className="mono">{fmt(p.population_density, 0)}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      </div>
    </AppShell>
  );
}
