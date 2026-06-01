"""
backend/main.py — FastAPI layer over the real 5G AI multi-agent outputs.

Reads ONLY the real GeoPackage files produced by the pipeline. Does not
fabricate data. Endpoints:

  GET  /api/grid       → final_merged_grid.gpkg as GeoJSON (EPSG:4326)
  GET  /api/sites      → recommended_sites_v3.gpkg as GeoJSON (EPSG:4326)
  GET  /api/kpis       → KPIs derived from the real grid
  GET  /api/coverage   → coverage metrics + class distribution
  GET  /api/demand     → demand metrics + class distribution
  GET  /api/terrain    → terrain metrics
  GET  /api/agents     → agent execution status (from outputs on disk)
  POST /api/run-pipeline  → spawns orchestrator.py; returns job status

Run:
    pip install -r backend/requirements.txt
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# ── Paths ────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("PIPELINE_DATA_DIR", BACKEND_DIR / "data"))
GRID_PATH = DATA_DIR / "final_merged_grid.gpkg"
SITES_PATH = DATA_DIR / "recommended_sites_v3.gpkg"
ORCHESTRATOR = BACKEND_DIR / "orchestrator.py"
PIPELINE_CONFIG = Path(os.environ.get("PIPELINE_CONFIG", BACKEND_DIR / "config.yaml"))

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="5G Planner API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cache (file mtime keyed) ────────────────────────────────────────────────
_cache: dict[str, tuple[float, Any]] = {}

def _read_gdf(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise HTTPException(404, f"Missing output file: {path}")
    mtime = path.stat().st_mtime
    cached = _cache.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    gdf = gpd.read_file(path).to_crs(4326)
    # Stable id columns
    if path == GRID_PATH and "cell_id" not in gdf.columns:
        gdf = gdf.reset_index().rename(columns={"index": "cell_id"})
    if path == SITES_PATH and "site_id" not in gdf.columns:
        gdf = gdf.reset_index().rename(columns={"index": "site_id"})
        gdf["site_id"] = gdf["site_id"] + 1
    _cache[str(path)] = (mtime, gdf)
    return gdf

def _to_geojson(gdf: gpd.GeoDataFrame) -> dict:
    return json.loads(gdf.to_json())

def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/grid")
def get_grid():
    return _to_geojson(_read_gdf(GRID_PATH))

@app.get("/api/sites")
def get_sites():
    return _to_geojson(_read_gdf(SITES_PATH))

@app.get("/api/kpis")
def get_kpis():
    g = _read_gdf(GRID_PATH)
    s = _read_gdf(SITES_PATH)
    return {
        "cells": int(len(g)),
        "population": float(g["population"].sum()),
        "total_peak_demand_gbps": float(g["peak_demand_gbps"].sum()),
        "total_capacity_gbps": float(g["capacity_available_gbps"].sum()),
        "total_deficit_gbps": float(g["capacity_deficit_gbps"].sum()),
        "coverage_ratio": _ratio(int(g["is_covered"].sum()), len(g)),
        "has_5g_ratio": _ratio(int(g["has_5g"].sum()), len(g)),
        "hotspot_risk_count": int(g["is_hotspot_risk"].sum()),
        "antennas": int(g["antenna_count"].sum()),
        "antennas_nr": int(g["antenna_count_nr"].sum()),
        "antennas_lte": int(g["antenna_count_lte"].sum()),
        "sites_recommended": int(len(s)),
    }

def _dist(g: gpd.GeoDataFrame, col: str) -> list[dict]:
    vc = g[col].value_counts(dropna=False)
    n = len(g) or 1
    return [{"label": str(k), "count": int(v), "ratio": float(v) / n} for k, v in vc.items()]

@app.get("/api/coverage")
def get_coverage():
    g = _read_gdf(GRID_PATH)
    return {
        "coverage_ratio": _ratio(int(g["is_covered"].sum()), len(g)),
        "mean_coverage_score": float(g["coverage_score"].mean()),
        "mean_overload_score": float(g["overload_score"].mean()),
        "total_capacity_gbps": float(g["capacity_available_gbps"].sum()),
        "total_deficit_gbps": float(g["capacity_deficit_gbps"].sum()),
        "antennas": int(g["antenna_count"].sum()),
        "antennas_nr": int(g["antenna_count_nr"].sum()),
        "antennas_lte": int(g["antenna_count_lte"].sum()),
        "distribution_coverage_class": _dist(g, "coverage_class"),
        "distribution_nearest_radio": _dist(g, "nearest_radio"),
    }

@app.get("/api/demand")
def get_demand():
    g = _read_gdf(GRID_PATH)
    return {
        "population_total": float(g["population"].sum()),
        "mean_population_density": float(g["population_density"].mean()),
        "total_base_demand_gbps": float(g["base_demand_gbps"].sum()),
        "total_peak_demand_gbps": float(g["peak_demand_gbps"].sum()),
        "distribution_usage_type": _dist(g, "usage_type"),
        "distribution_demand_class": _dist(g, "demand_class"),
        "distribution_demand_category": _dist(g, "demand_category"),
    }

@app.get("/api/terrain")
def get_terrain():
    g = _read_gdf(GRID_PATH)
    return {
        "mean_elevation_m": float(g["elevation_mean"].mean()),
        "mean_elevation_range_m": float(g["elevation_range"].mean()),
        "mean_slope_norm": float(g["slope_norm"].mean()),
        "mean_los_probability": float(g["los_probability"].mean()),
        "mean_attenuation_db": float(g["attenuation_factor"].mean()),
        "mean_forest_ratio": float(g["forest_ratio"].mean()),
        "mean_water_ratio": float(g["water_ratio"].mean()),
        "mean_terrain_complexity": float(g["terrain_complexity"].mean()),
    }

# ── Agent execution status (from on-disk artifacts) ─────────────────────────
AGENT_FILES = {
    "osmnx":     DATA_DIR / "osmnx"    / "osmnx_grid.gpkg",
    "demand":    DATA_DIR / "demand"   / "demand_grid.gpkg",
    "terrain":   DATA_DIR / "terrain"  / "terrain_grid.gpkg",
    "coverage":  DATA_DIR / "coverage" / "coverage_grid.gpkg",
    "merge":     GRID_PATH,
    "decision":  SITES_PATH,
}

@app.get("/api/agents")
def get_agents():
    out = []
    for name, p in AGENT_FILES.items():
        exists = p.exists()
        out.append({
            "agent": name,
            "exists": exists,
            "path": str(p),
            "size_bytes": p.stat().st_size if exists else 0,
            "updated_at": p.stat().st_mtime if exists else None,
        })
    return {"agents": out, "job": _job_status()}

# ── Pipeline execution ──────────────────────────────────────────────────────
_job_lock = threading.Lock()
_job: dict | None = None

def _job_status() -> dict | None:
    with _job_lock:
        return None if _job is None else dict(_job)

def _run_pipeline_thread(only: list[str] | None, from_step: str | None) -> None:
    global _job
    cmd = [sys.executable, str(ORCHESTRATOR), "--config", str(PIPELINE_CONFIG)]
    if only:
        cmd += ["--only", *only]
    elif from_step:
        cmd += ["--from", from_step]
    with _job_lock:
        _job = {"status": "running", "cmd": cmd, "started_at": time.time(), "log": ""}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BACKEND_DIR))
        with _job_lock:
            _job = {
                "status": "ok" if proc.returncode == 0 else "failed",
                "cmd": cmd,
                "started_at": _job["started_at"],
                "finished_at": time.time(),
                "returncode": proc.returncode,
                "log": (proc.stdout + "\n" + proc.stderr)[-8000:],
            }
        # Bust the cache so the next /api/grid call reloads.
        _cache.clear()
    except Exception as e:  # pragma: no cover
        with _job_lock:
            _job = {"status": "failed", "cmd": cmd, "error": str(e), "finished_at": time.time()}

@app.post("/api/run-pipeline")
def run_pipeline(only: list[str] | None = None, from_step: str | None = None):
    if not PIPELINE_CONFIG.exists():
        raise HTTPException(400, f"Missing pipeline config: {PIPELINE_CONFIG}")
    with _job_lock:
        if _job and _job.get("status") == "running":
            return JSONResponse({"status": "already_running", "job": dict(_job)}, status_code=409)
    threading.Thread(
        target=_run_pipeline_thread, args=(only, from_step), daemon=True
    ).start()
    return {"status": "started"}

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "data_dir": str(DATA_DIR),
        "grid_exists": GRID_PATH.exists(),
        "sites_exists": SITES_PATH.exists(),
    }

# ── Root redirect for convenience ───────────────────────────────────────────
@app.get("/")
def root():
    return Response(
        content="5G Planner API — see /docs",
        media_type="text/plain",
    )
