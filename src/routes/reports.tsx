import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { FileText, FileType2, Download, ExternalLink, RefreshCw, AlertCircle } from "lucide-react";
import { AppShell, Card, ClientOnly, PageHeader } from "@/components/AppShell";

export const Route = createFileRoute("/reports")({
  head: () => ({ meta: [{ title: "Reports · 5G Planner" }] }),
  component: ReportsPage,
});

type Report = {
  name: string;
  date: string;
  path: string;
  size?: number;
  kind?: "pdf" | "txt" | "md";
};

const ENV_API = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "");
const API_BASE = ENV_API ?? (typeof window !== "undefined" ? "http://localhost:8000" : "");

async function fetchReports(): Promise<{ list: Report[]; live: boolean }> {
  try {
    const r = await fetch(`${API_BASE}/api/reports`);
    if (r.ok) return { list: (await r.json()) as Report[], live: true };
  } catch { /* fall through */ }
  // Preview fallback — static snapshot in /public/outputs/
  const STATIC: Report[] = [
    { name: "rapport_final_5g.pdf", date: new Date().toISOString(), path: "/outputs/rapport_final_5g.pdf", kind: "pdf" },
    { name: "agents_summary.txt", date: new Date().toISOString(), path: "/outputs/agents_summary.txt", kind: "txt" },
    { name: "site_placement_report.txt", date: new Date().toISOString(), path: "/outputs/site_placement_report.txt", kind: "txt" },
  ];
  return { list: STATIC, live: false };
}

function ReportsPage() {
  return (
    <AppShell>
      <div className="mx-auto max-w-[1600px] px-6 py-8">
        <PageHeader
          title="Reports & Artefacts"
          subtitle="PDF rapports + résumés texte produits dans outputs/ (rapport_final_5g.pdf, agents_summary.txt, …)."
        />
        <ClientOnly fallback={<div className="text-sm text-muted-foreground">Loading…</div>}>
          <ReportsBody />
        </ClientOnly>
      </div>
    </AppShell>
  );
}

function ReportsBody() {
  const [reports, setReports] = useState<Report[] | null>(null);
  const [live, setLive] = useState(true);
  const [selected, setSelected] = useState<Report | null>(null);
  const [loading, setLoading] = useState(true);
  const [textBody, setTextBody] = useState<string>("");

  async function load() {
    setLoading(true);
    const { list, live } = await fetchReports();
    setLive(live);
    setReports(list);
    setSelected(list.length ? list[0] : null);
    setLoading(false);
  }

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    if (!selected || selected.kind === "pdf") return setTextBody("");
    fetch(`${API_BASE}${selected.path}`)
      .then((r) => r.text())
      .then(setTextBody)
      .catch(() => setTextBody("Failed to load."));
  }, [selected]);

  const fileUrl = selected
    ? (live ? `${API_BASE}${selected.path}` : selected.path)
    : "";

  return (
    <>
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="text-xs mono text-muted-foreground">
          Source:{" "}
          <span className={live ? "text-[var(--color-success)]" : "text-[var(--color-warning)]"}>
            {live ? `live · ${API_BASE}` : "static preview snapshot"}
          </span>
        </div>
        <button
          onClick={load}
          className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm border border-border bg-secondary/60 hover:border-primary/30 transition"
        >
          <RefreshCw className="size-3.5" /> Refresh
        </button>
      </div>

      {!live && (
        <Card className="mb-4 !bg-[var(--color-warning)]/5 border-[var(--color-warning)]/30">
          <div className="flex gap-3 text-sm">
            <AlertCircle className="size-5 text-[var(--color-warning)] shrink-0 mt-0.5" />
            <div>
              Le backend FastAPI n&apos;est pas joignable sur{" "}
              <span className="mono text-primary">{API_BASE}</span>. Lance depuis la racine du
              projet&nbsp;:
              <pre className="mt-2 p-2 rounded bg-secondary/60 text-xs overflow-x-auto">
{`uvicorn main:app --reload --port 8000`}
              </pre>
              Les rapports lus se trouvent dans <span className="mono">outputs/</span>.
            </div>
          </div>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-4">
        <Card title="Available artefacts">
          {loading ? (
            <div className="text-sm text-muted-foreground">Loading…</div>
          ) : !reports || reports.length === 0 ? (
            <div className="text-sm text-muted-foreground">
              Aucun fichier dans <span className="mono">outputs/</span>. Lance le pipeline puis{" "}
              <span className="mono">python agents/llm_report_agent.py</span>.
            </div>
          ) : (
            <ul className="space-y-2">
              {reports.map((r) => {
                const active = selected?.name === r.name;
                const isPdf = r.kind === "pdf" || r.name.toLowerCase().endsWith(".pdf");
                const Icon = isPdf ? FileText : FileType2;
                const tone = isPdf
                  ? "bg-[#ef4444]/15 border-[#ef4444]/30 text-[#ef4444]"
                  : "bg-[#3b82f6]/15 border-[#3b82f6]/30 text-[#3b82f6]";
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
                        <div className={`size-9 rounded-md flex items-center justify-center shrink-0 border ${tone}`}>
                          <Icon className="size-4" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="text-sm font-medium truncate">{r.name}</div>
                          <div className="text-[11px] mono text-muted-foreground mt-0.5">
                            {new Date(r.date).toLocaleString()}
                            {r.size ? ` · ${(r.size / 1024).toFixed(1)} KB` : ""}
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

        <Card className="!p-0 overflow-hidden min-h-[480px]">
          {selected ? (
            <>
              <div className="flex items-center justify-between gap-3 px-4 py-2.5 border-b border-border bg-card/60">
                <div className="text-sm font-medium truncate">{selected.name}</div>
                <div className="flex items-center gap-2">
                  <a
                    href={fileUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs border border-border bg-secondary/60 hover:border-primary/30 transition"
                  >
                    <ExternalLink className="size-3.5" /> Open
                  </a>
                  <a
                    href={fileUrl}
                    download={selected.name}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs bg-primary text-primary-foreground hover:opacity-90 transition"
                  >
                    <Download className="size-3.5" /> Download
                  </a>
                </div>
              </div>
              {selected.kind === "pdf" || selected.name.toLowerCase().endsWith(".pdf") ? (
                <iframe
                  src={fileUrl}
                  title={selected.name}
                  className="w-full h-[calc(100vh-260px)] bg-white"
                />
              ) : (
                <pre className="p-5 text-xs leading-relaxed whitespace-pre-wrap font-mono text-foreground/90 overflow-auto h-[calc(100vh-260px)]">
                  {textBody || "…"}
                </pre>
              )}
            </>
          ) : (
            <div className="p-8 text-sm text-muted-foreground">
              Sélectionne un rapport à gauche pour le visualiser.
            </div>
          )}
        </Card>
      </div>
    </>
  );
}
