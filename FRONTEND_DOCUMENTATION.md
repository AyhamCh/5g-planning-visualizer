# 📘 Documentation Frontend — 5G Planner

Plateforme visuelle de planification 5G basée sur un système multi-agents IA. Ce document décrit **l'intégralité du développement du frontend** : stack technique, frameworks, librairies, architecture, modularité, conventions de code et flux de données.

---

## 1. 🎯 Objectif du frontend

Le frontend constitue la **couche d'interaction visuelle** du système AI multi-agents 5G. Il ne contient **aucune logique métier ni donnée synthétique** : il lit, transforme et visualise les sorties réelles produites par le pipeline Python (`orchestrator.py` + agents).

Sources de vérité consommées par l'UI :
- `results/final_merged_grid.gpkg` → exporté en `public/data/grid.geojson`
- `results/recommended_sites_v3.gpkg` → exporté en `public/data/sites.geojson`
- `outputs/rapport_final_5g.pdf` + `outputs/agents_summary.txt` (via `/api/reports`)
- Endpoint `/api/chat` (RAG ChromaDB + Qwen via Ollama)

---

## 2. 🧱 Stack technique

### Runtime & langages
| Élément | Version | Rôle |
|---|---|---|
| **Node / Bun** | LTS | Runtime de build & dev server |
| **TypeScript** | 5.8 (strict) | Typage statique, contrats forts |
| **React** | 19 | Couche UI déclarative |

### Framework applicatif
| Outil | Version | Rôle |
|---|---|---|
| **TanStack Start** | 1.167 | Framework fullstack React (SSR, server functions, routing) |
| **TanStack Router** | 1.168 | File-based routing typé bout-en-bout |
| **TanStack Query** | 5.83 | Cache serveur, `useSuspenseQuery` pour le data-fetching |
| **Vite** | 7 | Bundler / dev server (HMR) |

### Styling & design system
| Outil | Rôle |
|---|---|
| **Tailwind CSS v4** | Utilitaires CSS via `@import "tailwindcss"` dans `src/styles.css` |
| **Tokens CSS sémantiques** | `--color-primary`, `--color-card`, etc. (jamais de couleurs hardcodées) |
| **shadcn/ui** (`src/components/ui/*`) | Primitives accessibles (Radix UI) |
| **Radix UI** | Composants headless (Dialog, Tabs, Tooltip…) |
| **lucide-react** | Iconographie (Sparkles, Radio, Cpu…) |
| **tw-animate-css** | Animations utilitaires |

### Visualisation de données
| Lib | Rôle |
|---|---|
| **Leaflet + react-leaflet** | Cartographie SIG, fond CARTO dark, GeoJSON layers |
| **Recharts** | Pie / Bar charts (KPI, distributions) |

### Formulaires / utilitaires
| Lib | Rôle |
|---|---|
| **react-hook-form + Zod** | Formulaires typés, validation déclarative |
| **clsx + tailwind-merge** | Composition de classes Tailwind |
| **sonner** | Toasts |

---

## 3. 🗂️ Structure du projet

```
src/
├── routes/                  # File-based routing TanStack
│   ├── __root.tsx           # Layout racine (HTML shell, QueryClientProvider, Suspense)
│   ├── index.tsx            # /            — Overview (KPI, agents, pipeline)
│   ├── map.tsx              # /map         — Carte Leaflet (grid + sites)
│   ├── sites.tsx            # /sites       — Sites recommandés (table triée)
│   ├── agents.tsx           # /agents      — Indicateurs par agent
│   ├── decision.tsx         # /decision    — Décisions classées + justifications
│   ├── what-if.tsx          # /what-if     — Simulation par cellule
│   ├── chat.tsx             # /chat        — Assistant 5G (RAG)
│   └── reports.tsx          # /reports     — PDF & TXT générés
│
├── components/
│   ├── AppShell.tsx         # Sidebar + header NOC-style
│   ├── NetworkMap.tsx       # Wrapper client-only Leaflet
│   └── ui/                  # Primitives shadcn (button, card, dialog, …)
│
├── lib/
│   ├── telco-data.ts        # Types réels, fetch GeoJSON, KPI, helpers couleurs
│   ├── queries.ts           # queryOptions TanStack Query (gridQuery, sitesQuery)
│   ├── utils.ts             # cn(), helpers généraux
│   └── lovable-error-reporting.ts
│
├── hooks/
│   └── use-mobile.tsx
│
├── styles.css               # Tailwind v4 + tokens sémantiques + fonts
├── router.tsx               # Factory createRouter (queryClient en contexte)
├── routeTree.gen.ts         # ⚠️ Auto-généré — ne pas éditer
├── start.ts                 # Bootstrap TanStack Start
└── server.ts                # Entrée SSR

public/
├── data/
│   ├── grid.geojson         # Snapshot des cellules réelles
│   └── sites.geojson        # Snapshot des sites recommandés
└── outputs/
    ├── rapport_final_5g.pdf
    ├── agents_summary.txt
    └── site_placement_report.txt
```

---

## 4. 🔀 Architecture & flux de données

### 4.1 Routing (file-based)
Chaque fichier `src/routes/*.tsx` devient une URL. Le plugin Vite TanStack Router régénère automatiquement `routeTree.gen.ts`.

```tsx
// src/routes/map.tsx
export const Route = createFileRoute("/map")({
  head: () => ({ meta: [{ title: "Network Map · 5G Planner" }] }),
  component: MapPage,
});
```

### 4.2 Layout racine (`__root.tsx`)
- Fournit le shell HTML (`<html>`, `<head>`, `<body>`)
- Monte `QueryClientProvider` (contexte de cache global)
- Enveloppe `<Outlet />` dans un `<Suspense>` pour gérer le fallback SSR/hydration
- Définit `errorComponent` et `notFoundComponent` (boundaries obligatoires)

### 4.3 Data layer
Pattern canonique TanStack Start :

```ts
// src/lib/queries.ts
export const gridQuery = queryOptions({
  queryKey: ["grid"],
  queryFn: () => fetchGrid(),
});
```

```tsx
// dans une route
const { data: grid } = useSuspenseQuery(gridQuery);
```

`fetchGrid()` tente d'abord `${API_BASE}/api/grid` (FastAPI local), puis retombe sur `/data/grid.geojson` (snapshot statique). Cela permet à l'UI de fonctionner **sans backend** pour la démo, et **avec backend live** pour la production.

### 4.4 Schémas typés (source de vérité)
`src/lib/telco-data.ts` déclare les interfaces `GridProps` et `SiteProps` **strictement alignées sur les colonnes des GeoPackages** produits par les agents Python. Aucun champ inventé. Aucun mock.

### 4.5 Helpers de visualisation
- `computeKpis(grid)` → agrège population, demande pic, capacité, déficit, ratio couverture…
- `distribution(grid, field)` → calcule les parts par classe (urban, coverage, demand)
- `URBAN_COLORS`, `COVERAGE_COLORS`, `DEMAND_COLORS`, `SITE_COLORS` → palettes sémantiques partagées entre carte et charts

---

## 5. 🧩 Modularité

### Principes appliqués
1. **Séparation données / présentation** — `lib/` ne dépend jamais de React ; `components/` ne fetch jamais directement.
2. **Composants atomiques shadcn** — chaque primitive UI vit dans son fichier (`components/ui/card.tsx`, etc.) et est composable.
3. **Atoms de layout réutilisables** exportés depuis `AppShell.tsx` :
   - `<AppShell>` — sidebar + zone de contenu
   - `<PageHeader>` — titre + sous-titre + slot droit
   - `<Card>` — conteneur avec titre optionnel
   - `<Stat>` — métrique avec ton (`good` / `warn` / `bad`)
   - `<ClientOnly>` — évite l'hydration SSR pour Leaflet/Recharts
4. **Une responsabilité par route** — pas de page monolithique. Les graphes (`PieDist`, `BarDist`) sont colocalisés dans `index.tsx` car non réutilisés ailleurs.
5. **Cartographie isolée** — `NetworkMap.tsx` encapsule Leaflet (CSS, projections, popups) et n'est monté que côté client.

---

## 6. 🎨 Design system

- **Thème sombre NOC** (telecom Network Operations Center)
- Variables CSS exposées dans `src/styles.css` :
  ```css
  --color-primary: #3b82f6;
  --color-success: #22c55e;
  --color-card: #0f172a;
  --color-border: #1e293b;
  ```
- **Règle absolue** : jamais de `text-white`, `bg-[#xxx]` dans les composants. Toujours passer par les tokens.
- **Typographie** : sans-serif système + classe utilitaire `.mono` pour les valeurs techniques.
- **Effets signature** : glow `shadow-[0_0_24px_-8px_var(--color-primary)]`, ping animation sur indicateurs live, pulse sur status "Pipeline ready".

---

## 7. 🔌 Intégration backend

### Endpoints consommés
| Endpoint | Source | Page |
|---|---|---|
| `GET /api/grid` | `final_merged_grid.gpkg` | Overview, Map, Decision, What-If |
| `GET /api/sites` | `recommended_sites_v3.gpkg` | Sites, Map |
| `POST /api/chat` | `agents/rag_agent.py` | Chat |
| `GET /api/reports` | `outputs/*` | Reports |
| `GET /api/reports/{name}` | `outputs/{name}` | Reports (iframe PDF) |
| `POST /api/run-pipeline` | `orchestrator.py` | (optionnel) |

### Configuration
```bash
# .env.local
VITE_API_BASE=http://localhost:8000
```
Si absent → fallback automatique vers les snapshots `public/data/*` et `public/outputs/*`.

---

## 8. 🛠️ Workflow de développement

```bash
bun install            # installer les dépendances
bun run dev            # démarrer Vite (HMR) sur :8080
bun run build          # build production
bun run lint           # ESLint + prettier
```

Lancer le backend FastAPI (séparément) :
```bash
uvicorn main:app --reload --port 8000
```

### Conventions
- **TypeScript strict** activé : tout import doit résoudre, tout type doit être satisfait.
- **Pas de `useEffect` + `fetch`** pour le data initial → toujours `useSuspenseQuery`.
- **Pas de navigation `<a href>`** vers une route TanStack → toujours `<Link to=…>`.
- **Pas d'édition de `routeTree.gen.ts`** (régénéré).

---

## 9. ✅ Garanties de qualité

- 🔒 Types alignés sur les schémas réels des `.gpkg`
- ♿ Primitives Radix accessibles par défaut (focus, ARIA, keyboard)
- 🌗 Thème cohérent via tokens — pas de dérive visuelle
- ⚡ SSR + Suspense → premier paint rapide, pas de flash de contenu vide
- 🧪 Fallback gracieux quand le backend n'est pas joignable
- 📦 Code splitting automatique par route (TanStack Router)

---

## 10. 📚 Pour aller plus loin

- Ajouter une nouvelle page → créer `src/routes/<nom>.tsx`, ajouter une entrée dans `NAV` (`AppShell.tsx`).
- Ajouter une nouvelle métrique → enrichir `computeKpis` dans `telco-data.ts`, exposer via `<Stat>`.
- Ajouter un nouveau chart → utiliser Recharts (`PieDist` / `BarDist` comme références).
- Brancher un nouvel agent → exposer son output GPKG/JSON via FastAPI puis créer un `queryOptions` dans `lib/queries.ts`.

---

*Document maintenu en parallèle du code. Toute évolution majeure (route, dépendance, schéma) doit y être reflétée.*
