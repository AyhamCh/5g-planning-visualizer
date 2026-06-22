import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Send, Sparkles, User, Loader2, Settings2, CheckCircle2, XCircle } from "lucide-react";
import { AppShell, Card, PageHeader } from "@/components/AppShell";

export const Route = createFileRoute("/chat")({
  head: () => ({ meta: [{ title: "5G AI Assistant · 5G Planner" }] }),
  component: ChatPage,
});

type Msg = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
};

const ENV_API = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "");
const LS_KEY = "rag_api_base";

function getApiBase(): string {
  if (typeof window === "undefined") return ENV_API ?? "";
  const stored = window.localStorage.getItem(LS_KEY)?.replace(/\/$/, "");
  return stored || ENV_API || "http://localhost:8000";
}

const SUGGESTIONS = [
  "Quelles sont les normes 3GPP applicables à la 5G NR ?",
  "Comment justifie-t-on un site small_cell_dense en milieu hyper_dense ?",
  "Explique la logique de l'agent SitePlacement",
  "Que signifie un coverage_score < 0.4 dans le grid ?",
];

type AskResult = { ok: true; answer: string; sources?: string[] } | { ok: false; error: string };

async function askRag(apiBase: string, question: string): Promise<AskResult> {
  const url = `${apiBase}/api/chat`;
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!r.ok) {
      const body = await r.text().catch(() => "");
      return { ok: false, error: `HTTP ${r.status} sur ${url}\n${body.slice(0, 400)}` };
    }
    const data = await r.json();
    return { ok: true, answer: data.answer ?? "(réponse vide)", sources: data.sources };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return {
      ok: false,
      error: `Impossible de joindre ${url}\nRaison: ${msg}`,
    };
  }
}

function ChatPage() {
  const [messages, setMessages] = useState<Msg[]>([
    {
      role: "assistant",
      content:
        "Bonjour 👋 Je suis l'assistant 5G du projet (RAG + ChromaDB + Qwen via Ollama, défini dans `agents/rag_agent.py`). Posez-moi une question sur la 5G, les agents ou les résultats du pipeline.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [apiBase, setApiBase] = useState<string>("");
  const [showSettings, setShowSettings] = useState(false);
  const [draftBase, setDraftBase] = useState("");
  const [health, setHealth] = useState<"unknown" | "ok" | "down">("unknown");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const base = getApiBase();
    setApiBase(base);
    setDraftBase(base);
  }, []);

  useEffect(() => {
    if (!apiBase) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${apiBase}/api/health`, { method: "GET" }).catch(() =>
          fetch(`${apiBase}/`, { method: "GET" }),
        );
        if (!cancelled) setHealth(r && r.ok ? "ok" : "down");
      } catch {
        if (!cancelled) setHealth("down");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function saveBase() {
    const v = draftBase.trim().replace(/\/$/, "");
    if (v) window.localStorage.setItem(LS_KEY, v);
    else window.localStorage.removeItem(LS_KEY);
    setApiBase(v || ENV_API || "http://localhost:8000");
    setHealth("unknown");
    setShowSettings(false);
  }

  async function send(text: string) {
    const q = text.trim();
    if (!q || loading) return;
    setMessages((m) => [...m, { role: "user", content: q }]);
    setInput("");
    setLoading(true);
    const res = await askRag(apiBase, q);
    if (res.ok) {
      setMessages((m) => [...m, { role: "assistant", content: res.answer, sources: res.sources }]);
    } else {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            `⚠️ Backend RAG injoignable.\n\n${res.error}\n\n` +
            `Vérifications :\n` +
            `1. Lance le backend localement : \`uvicorn main:app --reload --port 8000\` (depuis la racine du projet)\n` +
            `2. Ce preview tourne sur lovable.app — il ne peut PAS joindre votre \`localhost\`. ` +
            `Pour le tester depuis le preview, expose ton backend via un tunnel :\n` +
            `   • \`ngrok http 8000\` → copie l'URL https générée\n` +
            `   • ou \`cloudflared tunnel --url http://localhost:8000\`\n` +
            `3. Ouvre ⚙️ Settings (en haut) et colle l'URL du tunnel.\n` +
            `4. Si tu lances le frontend en local (\`bun dev\` sur localhost:8080), \`http://localhost:8000\` fonctionne directement.\n\n` +
            `Le code RAG (\`agents/rag_agent.py\`) est correct ; le souci est purement réseau.`,
        },
      ]);
    }
    setLoading(false);
  }

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8 flex flex-col h-[calc(100vh-0px)]">
        <PageHeader
          title="5G Knowledge Assistant"
          subtitle="Posez vos questions à propos de la 5G, des agents et des résultats."
          right={
            <div className="flex items-center gap-3 text-xs mono">
              <div className="flex items-center gap-1.5 text-muted-foreground">
                {health === "ok" ? (
                  <>
                    <CheckCircle2 className="size-3.5 text-[var(--color-success)]" />
                    <span className="text-[var(--color-success)]">Backend OK</span>
                  </>
                ) : health === "down" ? (
                  <>
                    <XCircle className="size-3.5 text-red-400" />
                    <span className="text-red-400">Backend down</span>
                  </>
                ) : (
                  <>
                    <span className="size-1.5 rounded-full bg-muted-foreground animate-pulse" />
                    <span>checking…</span>
                  </>
                )}
              </div>
              <button
                onClick={() => setShowSettings((s) => !s)}
                className="flex items-center gap-1 px-2 py-1 rounded border border-border hover:border-primary/40"
              >
                <Settings2 className="size-3.5" /> API
              </button>
            </div>
          }
        />

        {showSettings && (
          <Card className="mb-4 !py-3">
            <div className="text-xs text-muted-foreground mb-2">
              URL du backend FastAPI (qui sert <code className="mono">/api/chat</code>). Pour exposer
              ton serveur local au preview Lovable, utilise un tunnel :{" "}
              <code className="mono">ngrok http 8000</code>.
            </div>
            <div className="flex gap-2">
              <input
                value={draftBase}
                onChange={(e) => setDraftBase(e.target.value)}
                placeholder="https://xxxx.ngrok-free.app  ou  http://localhost:8000"
                className="flex-1 bg-secondary/40 border border-border rounded px-3 py-2 text-sm mono outline-none focus:border-primary/50"
              />
              <button
                onClick={saveBase}
                className="px-3 py-2 rounded bg-primary text-primary-foreground text-sm"
              >
                Sauver
              </button>
            </div>
            <div className="mt-2 text-[11px] text-muted-foreground mono">
              Actuel : {apiBase || "(non défini)"}
            </div>
          </Card>
        )}

        <Card className="flex-1 flex flex-col min-h-0 !p-0 overflow-hidden">
          <div className="flex-1 overflow-y-auto p-6 space-y-5">
            {messages.map((m, i) => (
              <Message key={i} msg={m} />
            ))}
            {loading && (
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <div className="size-8 rounded-full bg-primary/15 border border-primary/30 flex items-center justify-center">
                  <Sparkles className="size-4 text-primary" />
                </div>
                <div className="flex items-center gap-2">
                  <Loader2 className="size-3.5 animate-spin" />
                  AI is thinking…
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>

          {messages.length <= 1 && (
            <div className="px-6 pb-3 flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="text-xs px-3 py-1.5 rounded-full border border-border bg-secondary/60 hover:border-primary/40 hover:bg-secondary transition"
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          <form
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
            className="border-t border-border p-3 flex items-end gap-2 bg-card/60"
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(input);
                }
              }}
              rows={1}
              placeholder="Pose une question sur la 5G, un agent, un résultat…"
              className="flex-1 resize-none bg-transparent outline-none px-3 py-2 text-sm placeholder:text-muted-foreground/60 max-h-40"
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className="size-9 rounded-md bg-primary text-primary-foreground flex items-center justify-center hover:opacity-90 disabled:opacity-40 transition"
            >
              <Send className="size-4" />
            </button>
          </form>
        </Card>
      </div>
    </AppShell>
  );
}

function Message({ msg }: { msg: Msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`size-8 shrink-0 rounded-full flex items-center justify-center border ${
          isUser ? "bg-secondary border-border" : "bg-primary/15 border-primary/30"
        }`}
      >
        {isUser ? <User className="size-4" /> : <Sparkles className="size-4 text-primary" />}
      </div>
      <div className={`max-w-[80%] ${isUser ? "items-end" : "items-start"} flex flex-col gap-1`}>
        <div
          className={`px-4 py-2.5 rounded-2xl text-sm whitespace-pre-wrap leading-relaxed ${
            isUser
              ? "bg-primary text-primary-foreground rounded-tr-sm"
              : "bg-secondary/70 text-foreground rounded-tl-sm border border-border"
          }`}
        >
          {msg.content}
        </div>
        {msg.sources && msg.sources.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-1">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
              Sources:
            </span>
            {msg.sources.map((s, i) => (
              <span
                key={i}
                className="text-[10px] mono px-1.5 py-0.5 rounded border border-border bg-card/60 text-muted-foreground"
              >
                {s}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
