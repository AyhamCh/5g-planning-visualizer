import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Send, Sparkles, User, Loader2 } from "lucide-react";
import { AppShell, Card, PageHeader } from "@/components/AppShell";

export const Route = createFileRoute("/chat")({
  head: () => ({ meta: [{ title: "5G AI Assistant · 5G Planify" }] }),
  component: ChatPage,
});

type Msg = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
};

const ENV_API = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "");
const API_BASE = ENV_API ?? (typeof window !== "undefined" ? "http://localhost:8000" : "");

const SUGGESTIONS = [
  "Quelles sont les normes 3GPP applicables à la 5G NR ?",
  "Comment justifie-t-on un site small_cell_dense en milieu hyper_dense ?",
  "Explique la logique de l'agent SitePlacement",
  "Que signifie un coverage_score < 0.4 dans le grid ?",
];

async function askRag(question: string): Promise<{ answer: string; sources?: string[] }> {
  try {
    const r = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!r.ok) throw new Error(`${r.status}`);
    return await r.json();
  } catch {
    return {
      answer:
        "⚠️ Le backend RAG n'est pas joignable. Démarre `uvicorn backend.main:app --port 8000` et configure `VITE_API_BASE`. " +
        "Le pipeline local utilise rag_agent.py (ChromaDB + Qwen via Ollama).",
      sources: [],
    };
  }
}

function ChatPage() {
  const [messages, setMessages] = useState<Msg[]>(() => {
    try {
      const stored = localStorage.getItem("5g_chat_history");
      if (stored) return JSON.parse(stored);
    } catch { /* ignore */ }
    return [
      {
        role: "assistant",
        content:
          "Bonjour 👋 Je suis l'assistant 5G du projet. Je peux répondre à vos questions sur les normes 5G, la planification réseau, l'implantation des sites, la couverture, l'architecture des agents et les documents techniques indexés.",
      },
    ];
  });
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    localStorage.setItem("5g_chat_history", JSON.stringify(messages));
  }, [messages]);

  async function send(text: string) {
    const q = text.trim();
    if (!q || loading) return;
    setMessages((m) => [...m, { role: "user", content: q }]);
    setInput("");
    setLoading(true);
    const res = await askRag(q);
    setMessages((m) => [...m, { role: "assistant", content: res.answer, sources: res.sources }]);
    setLoading(false);
  }

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8 flex flex-col h-[calc(100vh-0px)]">
        <PageHeader
          title="5G Knowledge Assistant"
          subtitle="Posez vos questions à propos de la 5G, des agents et des résultats."
        />

        <Card className="flex-1 flex flex-col min-h-0 !p-0 overflow-hidden">
          <div className="flex-1 min-h-0 overflow-y-auto p-6 space-y-5">
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
          isUser
            ? "bg-secondary border-border"
            : "bg-primary/15 border-primary/30"
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
