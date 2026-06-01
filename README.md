# 5G Planner — AI multi-agent decision platform

A professional visual layer over your existing 5G AI multi-agent pipeline.

- **Frontend**: TanStack Start (React 19, TypeScript, Tailwind v4, Leaflet)
  → reads the **real** outputs from your pipeline.
- **Backend**: FastAPI in `backend/` → exposes the real `.gpkg` outputs and
  can re-run the pipeline via `orchestrator.py`.
- **Source of truth**: `final_merged_grid.gpkg` and `recommended_sites_v3.gpkg`.
  All KPIs, classifications, rankings, and map layers are derived from these
  files. No mock data anywhere.

## Frontend data flow

```
public/data/grid.geojson     ←  final_merged_grid.gpkg   (reprojected to WGS84)
public/data/sites.geojson    ←  recommended_sites_v3.gpkg

OR (when VITE_API_BASE is set):

FastAPI /api/grid /api/sites  ←  same .gpkg files via GeoPandas
```

Switch to the API:
```bash
VITE_API_BASE=http://localhost:8000 bun run dev
```

## Pages

| Route        | What it shows                                                        |
|--------------|----------------------------------------------------------------------|
| `/`          | KPIs + class distributions derived from the real grid                |
| `/map`       | CARTO dark basemap + grid colored by coverage/demand/urban/LOS, sites|
| `/sites`     | Ranked table from `composite_score` (SitePlacementAgent)             |
| `/agents`    | Per-agent indicators (OSMnx · Demand · Terrain · Coverage · Sites)   |
| `/decision`  | Ranked decision cards with agent-rule-based justifications           |
| `/what-if`   | Real per-cell re-derivation under demand/capacity/LOS scenarios      |

## Refreshing data after a pipeline run

1. Run the pipeline (locally or via `POST /api/run-pipeline`).
2. Re-export the GeoJSON for the static frontend:
   ```python
   import geopandas as gpd
   gpd.read_file("outputs/final_merged_grid.gpkg").to_crs(4326)\
      .to_file("public/data/grid.geojson", driver="GeoJSON")
   gpd.read_file("outputs/recommended_sites_v3.gpkg").to_crs(4326)\
      .to_file("public/data/sites.geojson", driver="GeoJSON")
   ```
   …or simply set `VITE_API_BASE` and the frontend pulls fresh data from FastAPI.

See `backend/README.md` for endpoint reference.
