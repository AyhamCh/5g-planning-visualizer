# 5G Planner — Backend (FastAPI)

This FastAPI layer is a **thin visual API over the existing AI multi-agent pipeline**.
It does not fabricate data. It reads the real outputs produced by `orchestrator.py`
and the agents shipped in `backend/agents/`.

## Layout

```
backend/
├── main.py                  # FastAPI app
├── orchestrator.py          # the real pipeline orchestrator (unchanged)
├── agents/                  # the real agents (unchanged)
│   ├── osmnx_agent.py
│   ├── population_demand_agent.py
│   ├── terrain_agent.py
│   ├── coverage_agent.py
│   ├── decision_agent.py
│   ├── visual_planning_agent.py
│   └── llm_report_agent.py
├── data/                    # mounted pipeline outputs (real .gpkg)
│   ├── final_merged_grid.gpkg
│   ├── recommended_sites_v3.gpkg
│   ├── osmnx/        demand/        terrain/        coverage/
└── requirements.txt
```

`PIPELINE_DATA_DIR` (env) overrides `backend/data/`.
`PIPELINE_CONFIG`   (env) points to the `config.yaml` consumed by `orchestrator.py`.

## Run

```bash
pip install -r backend/requirements.txt
# also install whatever deps your agents need (osmnx, rasterio, …)
uvicorn backend.main:app --reload --port 8000
```

## Endpoints

| Method | Path                  | Purpose                                                         |
|--------|-----------------------|-----------------------------------------------------------------|
| GET    | `/api/grid`           | Real `final_merged_grid.gpkg` reprojected to EPSG:4326 (GeoJSON)|
| GET    | `/api/sites`          | Real `recommended_sites_v3.gpkg` (GeoJSON)                      |
| GET    | `/api/kpis`           | KPIs derived from the real grid + sites                         |
| GET    | `/api/coverage`       | Coverage metrics + class distribution                           |
| GET    | `/api/demand`         | Demand metrics + class distribution                             |
| GET    | `/api/terrain`        | Terrain metrics                                                 |
| GET    | `/api/agents`         | Per-agent artifact status (exists/size/mtime) + job status      |
| POST   | `/api/run-pipeline`   | Spawns `python orchestrator.py --config …` and refreshes cache  |
| GET    | `/api/health`         | Liveness                                                        |

## Pointing the frontend at this API

The frontend (TanStack Start) reads the real GeoJSON exports from
`/data/grid.geojson` and `/data/sites.geojson` by default. To switch it to
this FastAPI backend, set:

```
VITE_API_BASE=http://localhost:8000
```

and rebuild. The frontend hooks (`src/lib/telco-data.ts`) will then call
`/api/grid` / `/api/sites` instead of the static exports.

## Notes

- Cache is keyed by file mtime — re-running `orchestrator.py` (or hitting
  `/api/run-pipeline`) busts it automatically.
- `/api/run-pipeline` is a long-running shell-out; it is fire-and-forget and
  status is exposed via `/api/agents`.
- The agents and orchestrator are copied verbatim from the source you
  provided. No modifications.
