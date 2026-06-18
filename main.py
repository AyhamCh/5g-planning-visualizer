"""
main.py — FastAPI layer for the 5G AI multi-agent system.

Project layout (at project root):
  agents/                 ← all agents (incl. rag_agent.py, llm_report_agent.py)
  orchestrator.py         ← entry point of the pipeline
  results/                ← GeoPackage outputs (final_merged_grid.gpkg, …)
  outputs/                ← human-readable artefacts (rapport_final_5g.pdf,
                            agents_summary.txt, site_placement_report.txt)
  chroma_db/              ← persistent vector store of the RAG
  config.yaml             ← pipeline config

Run locally (Ollama must be up with qwen3:8b):

    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import json, os, subprocess, sys, threading, time
from pathlib import Path
from typing import Any

import geopandas as gpd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse, PlainTextResponse
from pydantic import BaseModel

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
RESULTS_DIR   = Path(os.environ.get("PIPELINE_RESULTS_DIR", ROOT / "results"))
OUTPUTS_DIR   = Path(os.environ.get("PIPELINE_OUTPUTS_DIR", ROOT / "outputs"))
GRID_PATH     = RESULTS_DIR / "final_merged_grid.gpkg"
SITES_PATH    = RESULTS_DIR / "recommended_sites_v3.gpkg"
ORCHESTRATOR  = ROOT / "orchestrator.py"
PIPELINE_CFG  = Path(os.environ.get("PIPELINE_CONFIG", ROOT / "config.yaml"))

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="5G Planner API", version="2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── GeoPackage helpers (mtime cached) ───────────────────────────────────────
_cache: dict[str, tuple[float, Any]] = {}

def _read_gdf(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise HTTPException(404, f"Missing output file: {path}")
    mtime = path.stat().st_mtime
    cached = _cache.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    gdf = gpd.read_file(path).to_crs(4326)
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

def _dist(g: gpd.GeoDataFrame, col: str) -> list[dict]:
    vc = g[col].value_counts(dropna=False)
    n = len(g) or 1
    return [{"label": str(k), "count": int(v), "ratio": float(v) / n} for k, v in vc.items()]

# ── Spatial endpoints ────────────────────────────────────────────────────────
@app.get("/api/grid")
def get_grid():  return _to_geojson(_read_gdf(GRID_PATH))

@app.get("/api/sites")
def get_sites(): return _to_geojson(_read_gdf(SITES_PATH))

@app.get("/api/kpis")
def get_kpis():
    g = _read_gdf(GRID_PATH); s = _read_gdf(SITES_PATH)
    return {
        "cells": int(len(g)),
        "population": float(g["population"].sum()),
        "total_peak_demand_gbps": float(g["peak_demand_gbps"].sum()),
        "total_capacity_gbps":    float(g["capacity_available_gbps"].sum()),
        "total_deficit_gbps":     float(g["capacity_deficit_gbps"].sum()),
        "coverage_ratio": _ratio(int(g["is_covered"].sum()), len(g)),
        "has_5g_ratio":   _ratio(int(g["has_5g"].sum()),    len(g)),
        "hotspot_risk_count": int(g["is_hotspot_risk"].sum()),
        "antennas":     int(g["antenna_count"].sum()),
        "antennas_nr":  int(g["antenna_count_nr"].sum()),
        "antennas_lte": int(g["antenna_count_lte"].sum()),
        "sites_recommended": int(len(s)),
    }

@app.get("/api/coverage")
def get_coverage():
    g = _read_gdf(GRID_PATH)
    return {
        "coverage_ratio": _ratio(int(g["is_covered"].sum()), len(g)),
        "mean_coverage_score": float(g["coverage_score"].mean()),
        "mean_overload_score": float(g["overload_score"].mean()),
        "total_capacity_gbps": float(g["capacity_available_gbps"].sum()),
        "total_deficit_gbps":  float(g["capacity_deficit_gbps"].sum()),
        "antennas":      int(g["antenna_count"].sum()),
        "antennas_nr":   int(g["antenna_count_nr"].sum()),
        "antennas_lte":  int(g["antenna_count_lte"].sum()),
        "distribution_coverage_class": _dist(g, "coverage_class"),
        "distribution_nearest_radio":  _dist(g, "nearest_radio"),
    }

@app.get("/api/demand")
def get_demand():
    g = _read_gdf(GRID_PATH)
    return {
        "population_total": float(g["population"].sum()),
        "mean_population_density": float(g["population_density"].mean()),
        "total_base_demand_gbps": float(g["base_demand_gbps"].sum()),
        "total_peak_demand_gbps": float(g["peak_demand_gbps"].sum()),
        "distribution_usage_type":     _dist(g, "usage_type"),
        "distribution_demand_class":   _dist(g, "demand_class"),
        "distribution_demand_category":_dist(g, "demand_category"),
    }

# ── Agents status ────────────────────────────────────────────────────────────
AGENT_FILES = {
    "osmnx":    RESULTS_DIR / "osmnx"   / "osmnx_grid.gpkg",
    "demand":   RESULTS_DIR / "demand"  / "demand_grid.gpkg",
    "terrain":  RESULTS_DIR / "terrain" / "terrain_grid.gpkg",
    "coverage": RESULTS_DIR / "coverage"/ "coverage_grid.gpkg",
    "merge":    GRID_PATH,
    "decision": SITES_PATH,
}

@app.get("/api/agents")
def get_agents():
    out = []
    for name, p in AGENT_FILES.items():
        exists = p.exists()
        out.append({
            "agent": name, "exists": exists, "path": str(p),
            "size_bytes": p.stat().st_size if exists else 0,
            "updated_at": p.stat().st_mtime if exists else None,
        })
    return {"agents": out, "job": _job_status()}

# ── Pipeline execution ──────────────────────────────────────────────────────
_job_lock = threading.Lock()
_job: dict | None = None

def _job_status() -> dict | None:
    with _job_lock: return None if _job is None else dict(_job)

def _run_pipeline_thread(only, from_step):
    global _job
    cmd = [sys.executable, str(ORCHESTRATOR), "--config", str(PIPELINE_CFG)]
    if only:           cmd += ["--only", *only]
    elif from_step:    cmd += ["--from", from_step]
    with _job_lock:
        _job = {"status": "running", "cmd": cmd, "started_at": time.time(), "log": ""}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        with _job_lock:
            _job = {
                "status": "ok" if proc.returncode == 0 else "failed",
                "cmd": cmd, "started_at": _job["started_at"], "finished_at": time.time(),
                "returncode": proc.returncode,
                "log": (proc.stdout + "\n" + proc.stderr)[-8000:],
            }
        _cache.clear()
    except Exception as e:
        with _job_lock:
            _job = {"status": "failed", "cmd": cmd, "error": str(e), "finished_at": time.time()}

@app.post("/api/run-pipeline")
def run_pipeline(only: list[str] | None = None, from_step: str | None = None):
    with _job_lock:
        if _job and _job.get("status") == "running":
            return JSONResponse({"status": "already_running", "job": dict(_job)}, status_code=409)
    threading.Thread(target=_run_pipeline_thread, args=(only, from_step), daemon=True).start()
    return {"status": "started"}

@app.get("/api/health")
def health():
    return {
        "ok": True, "results_dir": str(RESULTS_DIR), "outputs_dir": str(OUTPUTS_DIR),
        "grid_exists":  GRID_PATH.exists(), "sites_exists": SITES_PATH.exists(),
    }

# ── RAG chat (delegates to agents/rag_agent.py) ─────────────────────────────
class ChatBody(BaseModel):
    question: str

@app.post("/api/chat")
def chat(body: ChatBody):
    try:
        from agents.rag_agent import answer_question  # type: ignore
        return answer_question(body.question)
    except Exception as e:
        return {
            "answer": (
                "Le moteur RAG local n'est pas joignable "
                f"({e.__class__.__name__}: {e}). Vérifie qu'Ollama tourne "
                "(`ollama serve` + `ollama pull qwen3:8b`) et que la collection "
                "ChromaDB existe (`python agents/rag_agent.py --build-only`)."
            ),
            "sources": [],
        }

# ── Reports ──────────────────────────────────────────────────────────────────
@app.get("/api/reports")
def list_reports():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(OUTPUTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_file(): continue
        if p.suffix.lower() not in (".pdf", ".txt", ".md"): continue
        st = p.stat()
        items.append({
            "name": p.name,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
            "path": f"/api/reports/{p.name}",
            "size": st.st_size,
            "kind": p.suffix.lower().lstrip("."),
        })
    return items

@app.get("/api/reports/{name}")
def get_report(name: str):
    if "/" in name or "\\" in name:
        raise HTTPException(400, "Invalid report name")
    p = OUTPUTS_DIR / name
    if not p.exists():
        raise HTTPException(404, f"Report not found: {name}")
    if p.suffix.lower() == ".pdf":
        return FileResponse(p, media_type="application/pdf", filename=name)
    return PlainTextResponse(p.read_text(encoding="utf-8", errors="ignore"))

@app.get("/")
def root():
    return Response("5G Planner API — see /docs", media_type="text/plain")
