import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { FileText, Download, ExternalLink, RefreshCw } from "lucide-react";
import { AppShell, Card, PageHeader } from "@/components/AppShell";

export const Route = createFileRoute("/reports")({
  head: () => ({ meta: [{ title: "Reports · 5G Planner" }] }),
  component: ReportsPage,
});

type Report = { name: string; date: string; path: string; size?: number };

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "";

async function fetchReports(): Promise<Report[]> {
  try {
    const r = await fetch(`${API_BASE}/api/reports`);
    if (!r.ok) throw new Error(`${r.status}`);
    return await r.json();
  } catch {
    return [
      {
        name: "rapport_final_5g.pdf",
        date: new Date().toISOString(),
        path: "/api/reports/rapport_final_5g.pdf",
      },
    ];
  }
}

function ReportsPage() {
  const [reports, setReports] = useState<Report[]>([]);
  const [selected, setSelected] = useState<Report | null>(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    const list = await fetchReports();
    setReports(list);
    setSelected(list[0] ?? null);
    setLoading(false);
  }

  useEffect(() => {
    load();
  }, []);

  const pdfUrl = selected ? `${API_BASE}${selected.path}` : "";

  return (
    <AppShell>
      <div className="mx-auto max-w-[1600px] px-6 py-8">
        <PageHeader
          title="Reports"
          subtitle="Rapports PDF générés par le LLM Report Agent à partir des sorties du pipeline."
          right={
            <button
              onClick={load}
              className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm border border-border bg-secondary/60 hover:border-primary/30 transition"
            >
              <RefreshCw className="size-3.5" /> Refresh
            </button>
          }
        />

        <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-4">
          <Card title="Available reports">
            {loading ? (
              <div className="text-sm text-muted-foreground">Loading…</div>
            ) : reports.length === 0 ? (
              <div className="text-sm text-muted-foreground">
                Aucun rapport. Lance le pipeline puis llm_report_agent.py.
              </div>
            ) : (
              <ul className="space-y-2">
                {reports.map((r) => {
                  const active = selected?.name === r.name;
                  return (
                    <li key={r.name}>
                      <button
                        onClick={() => setSelected(r)}
                        className={`w-full text-left rounded-lg p-3 border transition ${
                          active
                            ? "border-primary/40 bg-primary/10"
                            : "border-border bg-card/60 hover:border-primary/30"
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          <div className="size-9 rounded-md bg-primary/15 border border-primary/30 flex items-center justify-center shrink-0">
                            <FileText className="size-4 text-primary" />
                          </div>
                          <div className="min-w-0">
                            <div className="text-sm font-medium truncate">{r.name}</div>
                            <div className="text-[11px] mono text-muted-foreground mt-0.5">
                              {new Date(r.date).toLocaleString()}
                            </div>
                          </div>
                        </div>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </Card>

          <Card className="!p-0 overflow-hidden">
            {selected ? (
              <>
                <div className="flex items-center justify-between gap-3 px-4 py-2.5 border-b border-border bg-card/60">
                  <div className="text-sm font-medium truncate">{selected.name}</div>
                  <div className="flex items-center gap-2">
                    <a
                      href={pdfUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs border border-border bg-secondary/60 hover:border-primary/30 transition"
                    >
                      <ExternalLink className="size-3.5" /> Open PDF
                    </a>
                    <a
                      href={pdfUrl}
                      download={selected.name}
                      className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs bg-primary text-primary-foreground hover:opacity-90 transition"
                    >
                      <Download className="size-3.5" /> Download PDF
                    </a>
                  </div>
                </div>
                <iframe
                  src={pdfUrl}
                  title={selected.name}
                  className="w-full h-[calc(100vh-220px)] bg-white"
                />
              </>
            ) : (
              <div className="p-8 text-sm text-muted-foreground">
                Sélectionne un rapport à gauche pour le visualiser.
              </div>
            )}
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
