"""
orchestrator.py — Pipeline 5G AI
==================================
Orchestre l'exécution complète du pipeline de recommandation de sites 5G.

DAG d'exécution :
    osmnx_agent
        ├──► population_demand_agent ──► coverage_agent ──┐
        └──► terrain_agent ───────────────────────────────┤
                                                           ▼
                                                  merge_grids
                                                           ▼
                                                  decision_agent
                                                           ▼
                                              visual_planning_agent

Usage :
    python orchestrator.py --config config.yaml
    python orchestrator.py --config config.yaml --from coverage  # reprendre depuis une étape
    python orchestrator.py --config config.yaml --only merge decision visual
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml
import geopandas as gpd

# ─────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")

# Toutes les étapes dans l'ordre
ALL_STEPS = ["osmnx", "demand", "terrain", "coverage", "merge", "decision", "visual"]


# ─────────────────────────────────────────────────────────────────
#  UTILITAIRES
# ─────────────────────────────────────────────────────────────────

def banner(title: str):
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def step_done(name: str, t0: float):
    elapsed = time.time() - t0
    logger.info(f"✓ [{name}] terminé en {elapsed:.1f}s")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_steps(args) -> list[str]:
    """Détermine la liste ordonnée des étapes à exécuter."""
    if args.only:
        steps = [s for s in ALL_STEPS if s in args.only]
    elif args.from_step:
        idx = ALL_STEPS.index(args.from_step)
        steps = ALL_STEPS[idx:]
    else:
        steps = ALL_STEPS
    return steps


def ensure_dirs(cfg: dict):
    out = cfg["outputs"]
    for key, path in out.items():
        if key == "merged_grid":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        else:
            Path(path).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  ÉTAPES DU PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_osmnx(cfg: dict) -> gpd.GeoDataFrame:
    """Étape 1 — OSMnxUrbanAgent : analyse urbaine sur la zone."""
    banner("ÉTAPE 1 — OSMnxUrbanAgent")
    t0 = time.time()

    from agents.osmnx_agent import OSMnxUrbanAgent

    bounds = cfg["bounds"]
    cell_size = cfg["cell_size_m"]
    out_dir = cfg["outputs"]["osmnx_dir"]

    # Génération de la grille depuis les bounds
    try:
        from core.generate_grid_from_bounds import generate_grid_from_bounds
    except ImportError:
        from shapely.geometry import box as _box
        import numpy as np

        def generate_grid_from_bounds(bounds_dict, cell_size=200):
            minx, miny = bounds_dict["minx"], bounds_dict["miny"]
            maxx, maxy = bounds_dict["maxx"], bounds_dict["maxy"]
            gdf = gpd.GeoDataFrame(
                {"geometry": [_box(minx, miny, maxx, maxy)]}, crs="EPSG:4326"
            ).to_crs("EPSG:3857")
            x0, y0, x1, y1 = gdf.total_bounds
            cells = [
                _box(x, y, x + cell_size, y + cell_size)
                for x in np.arange(x0, x1, cell_size)
                for y in np.arange(y0, y1, cell_size)
            ]
            return gpd.GeoDataFrame(
                {"geometry": cells}, crs="EPSG:3857"
            ).to_crs("EPSG:4326")

    logger.info(f"Génération grille — bounds: {bounds}, cell_size: {cell_size}m")
    bounds_tuple = (bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"])
    grid = generate_grid_from_bounds(bounds_tuple, cell_size=cell_size)
    logger.info(f"  {len(grid)} cellules générées")

    output_gpkg = str(Path(out_dir) / "osmnx_grid.gpkg")
    output_stats = str(Path(out_dir) / "osmnx_stats.txt")

    agent = OSMnxUrbanAgent(grid, cell_size_m=cell_size, output_path=output_gpkg)
    result = agent.run()

    # Stats
    stats_text = agent.summary()
    Path(output_stats).write_text(stats_text, encoding="utf-8")
    logger.info(f"  Stats → {output_stats}")

    step_done("osmnx", t0)
    return result


def run_demand(cfg: dict, osmnx_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Étape 2a — PopulationDemandAgent (parallèle avec terrain)."""
    banner("ÉTAPE 2a — PopulationDemandAgent")
    t0 = time.time()

    from agents.population_demand_agent import PopulationDemandAgent

    out_dir = cfg["outputs"]["demand_dir"]
    output_gpkg  = str(Path(out_dir) / "demand_grid.gpkg")
    output_stats = str(Path(out_dir) / "demand_stats.txt")

    agent = PopulationDemandAgent(
        osmnx_grid=osmnx_grid,
        pop_raster_path=cfg["data"]["pop_raster"],
        cell_size_m=cfg["cell_size_m"],
    )
    result = agent.estimate_demand()
    agent.save(output_gpkg)
    agent.export_statistics(output_stats)

    step_done("demand", t0)
    return result


def run_terrain(cfg: dict, osmnx_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Étape 2b — TerrainEnvironmentAgent (parallèle avec demand)."""
    banner("ÉTAPE 2b — TerrainEnvironmentAgent")
    t0 = time.time()

    from agents.terrain_agent import TerrainEnvironmentAgent

    out_dir = cfg["outputs"]["terrain_dir"]
    output_gpkg  = str(Path(out_dir) / "terrain_grid.gpkg")
    output_stats = str(Path(out_dir) / "terrain_stats.txt")

    agent = TerrainEnvironmentAgent(
        dem_path=cfg["data"]["dem"],
        landcover_dir=cfg["data"]["landcover_dir"],
    )
    result = agent.run(osmnx_grid)
    agent.save(output_gpkg)
    agent.export_statistics(output_stats)

    step_done("terrain", t0)
    return result


def run_coverage(cfg: dict, demand_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Étape 3 — CoverageAgent (nécessite demand_grid)."""
    banner("ÉTAPE 3 — CoverageAgent")
    t0 = time.time()

    from agents.coverage_agent import CoverageAgent

    out_dir = cfg["outputs"]["coverage_dir"]
    output_gpkg  = str(Path(out_dir) / "coverage_grid.gpkg")
    output_stats = str(Path(out_dir) / "coverage_stats.txt")

    # Sauvegarder demand_grid sur disque (CoverageAgent attend un path)
    demand_path = str(Path(cfg["outputs"]["demand_dir"]) / "demand_grid.gpkg")

    cov_cfg = cfg.get("coverage", {})
    agent = CoverageAgent(
        opencellid_path=cfg["data"]["opencellid"],
        grid_path=demand_path,
        config={
            "radio_filter":           cov_cfg.get("radio_filter", ["LTE", "NR"]),
            "min_samples":            cov_cfg.get("min_samples", 3),
            "buffer_km":              cov_cfg.get("buffer_km", 5),
            "max_antennas_per_cell":  cov_cfg.get("max_antennas_per_cell", 10),
            "top_k_capacity":         cov_cfg.get("top_k_capacity", 3),
        }
    )
    result = agent.compute_coverage()
    agent.save(output_gpkg)
    agent.export_statistics(output_stats)

    step_done("coverage", t0)
    return result


def run_merge(cfg: dict) -> gpd.GeoDataFrame:
    """Étape 4 — Fusion des 4 grilles en final_merged_grid.gpkg."""
    banner("ÉTAPE 4 — Merge grilles")
    t0 = time.time()

    TARGET_CRS  = "EPSG:32631"
    OUTPUT_PATH = cfg["outputs"]["merged_grid"]

    PATHS = {
        "osmnx":    str(Path(cfg["outputs"]["osmnx_dir"])    / "osmnx_grid.gpkg"),
        "demand":   str(Path(cfg["outputs"]["demand_dir"])   / "demand_grid.gpkg"),
        "coverage": str(Path(cfg["outputs"]["coverage_dir"]) / "coverage_grid.gpkg"),
        "terrain":  str(Path(cfg["outputs"]["terrain_dir"])  / "terrain_grid.gpkg"),
    }

    # Chargement
    gdfs = {}
    for name, path in PATHS.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"[merge] Fichier manquant : {path}")
        gdf = gpd.read_file(path)
        logger.info(f"  {name:10s}: {len(gdf)} cellules | {len(gdf.columns)} colonnes | CRS={gdf.crs}")
        gdfs[name] = gdf

    # Validation tailles
    lengths = {name: len(gdf) for name, gdf in gdfs.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"[merge] Tailles incohérentes : {lengths}")
    logger.info(f"  ✔ tailles cohérentes : {next(iter(lengths.values()))} cellules")

    # Harmonisation CRS
    for name in gdfs:
        if str(gdfs[name].crs) != TARGET_CRS:
            gdfs[name] = gdfs[name].to_crs(TARGET_CRS)

    # Merge colonne par colonne (sans doublons)
    base_name = "osmnx"
    gdf_final = gdfs[base_name].copy()

    for name, gdf in gdfs.items():
        if name == base_name:
            continue
        new_cols = [c for c in gdf.columns if c != "geometry" and c not in gdf_final.columns]
        logger.info(f"  + {name} : {len(new_cols)} nouvelles colonnes")
        gdf_final = gdf_final.merge(
            gdf[new_cols], left_index=True, right_index=True, how="left"
        )

    # Nettoyage
    gdf_final.replace([float("inf"), float("-inf")], 0, inplace=True)
    gdf_final.fillna(0, inplace=True)

    # Vérification doublons
    dup_cols = [c for c in gdf_final.columns if list(gdf_final.columns).count(c) > 1]
    if dup_cols:
        raise ValueError(f"[merge] Colonnes dupliquées : {set(dup_cols)}")

    logger.info(f"  → {len(gdf_final)} cellules | {len(gdf_final.columns)} colonnes")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    gdf_final.to_file(OUTPUT_PATH, driver="GPKG")
    logger.info(f"  → Sauvegardé : {OUTPUT_PATH}")

    step_done("merge", t0)
    return gdf_final


def run_decision(cfg: dict, merged_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Étape 5 — SitePlacementAgent."""
    banner("ÉTAPE 5 — SitePlacementAgent (decision)")
    t0 = time.time()

    from agents.decision_agent import SitePlacementAgent

    out_dir = cfg["outputs"]["decision_dir"]
    output_gpkg  = str(Path(out_dir) / "recommended_sites.gpkg")
    output_report = str(Path(out_dir) / "site_placement_report.txt")

    cell_size = cfg.get("decision", {}).get("cell_size_m", cfg["cell_size_m"])

    agent = SitePlacementAgent(
        cell_size_m=cell_size,
        report_path=output_report,
    )
    sites = agent.run(merged_grid)

    out_cols = [c for c in sites.columns if c != "geometry"] + ["geometry"]
    sites[out_cols].to_file(output_gpkg, driver="GPKG")
    logger.info(f"  → Sites : {output_gpkg}")

    step_done("decision", t0)
    return sites


def run_visual(cfg: dict):
    """Étape 6 — VisualPlanningAgent."""
    banner("ÉTAPE 6 — VisualPlanningAgent")
    t0 = time.time()

    from agents.visual_planning_agent import VisualPlanningAgent

    grid_path  = cfg["outputs"]["merged_grid"]
    sites_path = str(Path(cfg["outputs"]["decision_dir"]) / "recommended_sites.gpkg")
    output_html = str(Path(cfg["outputs"]["visual_dir"]) / "visual_planning.html")

    agent = VisualPlanningAgent(
        grid_path=grid_path,
        sites_path=sites_path,
        output_path=output_html,
    )
    agent.generate()
    logger.info(f"  → Carte : {output_html}")

    step_done("visual", t0)


# ─────────────────────────────────────────────────────────────────
#  ORCHESTRATEUR PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict, steps: list[str]):
    """
    Exécute le pipeline en respectant le DAG :
      osmnx → [demand ‖ terrain] → coverage → merge → decision → visual
    """
    total_t0 = time.time()
    banner(f"PIPELINE 5G AI — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Étapes : {steps}")

    ensure_dirs(cfg)

    # Résultats intermédiaires
    osmnx_grid   = None
    demand_grid  = None
    terrain_grid = None
    coverage_grid = None
    merged_grid  = None

    # ── ÉTAPE 1 : osmnx ──────────────────────────────────────
    if "osmnx" in steps:
        osmnx_grid = run_osmnx(cfg)
    else:
        # Charger depuis disque si l'étape est sautée
        path = str(Path(cfg["outputs"]["osmnx_dir"]) / "osmnx_grid.gpkg")
        if Path(path).exists():
            logger.info(f"  [osmnx] Chargement depuis disque : {path}")
            osmnx_grid = gpd.read_file(path)

    # ── ÉTAPES 2a + 2b : demand ‖ terrain (parallèle) ────────
    need_demand  = "demand"  in steps
    need_terrain = "terrain" in steps

    if need_demand or need_terrain:
        if osmnx_grid is None:
            raise RuntimeError(
                "osmnx_grid manquant — relancez avec --from osmnx "
                "ou incluez 'osmnx' dans --only"
            )

        if need_demand and need_terrain:
            logger.info("  Lancement demand + terrain en PARALLÈLE…")
            with ThreadPoolExecutor(max_workers=2) as executor:
                fut_demand  = executor.submit(run_demand,  cfg, osmnx_grid)
                fut_terrain = executor.submit(run_terrain, cfg, osmnx_grid)
                demand_grid  = fut_demand.result()
                terrain_grid = fut_terrain.result()
        elif need_demand:
            demand_grid = run_demand(cfg, osmnx_grid)
        else:
            terrain_grid = run_terrain(cfg, osmnx_grid)

    # Charger depuis disque si sautés
    if demand_grid is None:
        path = str(Path(cfg["outputs"]["demand_dir"]) / "demand_grid.gpkg")
        if Path(path).exists():
            logger.info(f"  [demand] Chargement depuis disque : {path}")
            demand_grid = gpd.read_file(path)

    if terrain_grid is None:
        path = str(Path(cfg["outputs"]["terrain_dir"]) / "terrain_grid.gpkg")
        if Path(path).exists():
            logger.info(f"  [terrain] Chargement depuis disque : {path}")
            terrain_grid = gpd.read_file(path)

    # ── ÉTAPE 3 : coverage ───────────────────────────────────
    if "coverage" in steps:
        if demand_grid is None:
            raise RuntimeError("demand_grid manquant pour coverage_agent.")
        coverage_grid = run_coverage(cfg, demand_grid)
    else:
        path = str(Path(cfg["outputs"]["coverage_dir"]) / "coverage_grid.gpkg")
        if Path(path).exists():
            logger.info(f"  [coverage] Chargement depuis disque : {path}")
            coverage_grid = gpd.read_file(path)

    # ── ÉTAPE 4 : merge ──────────────────────────────────────
    if "merge" in steps:
        merged_grid = run_merge(cfg)
    else:
        path = cfg["outputs"]["merged_grid"]
        if Path(path).exists():
            logger.info(f"  [merge] Chargement depuis disque : {path}")
            merged_grid = gpd.read_file(path)

    # ── ÉTAPE 5 : decision ───────────────────────────────────
    if "decision" in steps:
        if merged_grid is None:
            raise RuntimeError("merged_grid manquant pour decision_agent.")
        sites = run_decision(cfg, merged_grid)

    # ── ÉTAPE 6 : visual ─────────────────────────────────────
    if "visual" in steps:
        run_visual(cfg)

    # ── RÉSUMÉ ───────────────────────────────────────────────
    total_elapsed = time.time() - total_t0
    banner(f"PIPELINE TERMINÉ en {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"  Outputs dans : {cfg['outputs']['base_dir']}/")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrateur Pipeline 5G AI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Exemples :
  # Pipeline complet
  python orchestrator.py --config config.yaml

  # Reprendre depuis une étape (les étapes précédentes sont lues depuis disque)
  python orchestrator.py --config config.yaml --from coverage

  # Exécuter uniquement certaines étapes
  python orchestrator.py --config config.yaml --only merge decision visual

Étapes disponibles (dans l'ordre) :
  osmnx | demand | terrain | coverage | merge | decision | visual
        """,
    )
    parser.add_argument(
        "--config", required=True,
        help="Chemin vers config.yaml"
    )
    parser.add_argument(
        "--from", dest="from_step", default=None,
        choices=ALL_STEPS,
        help="Reprendre le pipeline depuis cette étape"
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        choices=ALL_STEPS,
        metavar="STEP",
        help="Exécuter uniquement ces étapes"
    )

    args = parser.parse_args()

    # Validation
    if args.from_step and args.only:
        parser.error("--from et --only sont mutuellement exclusifs.")

    cfg   = load_config(args.config)
    steps = resolve_steps(args)

    logger.info(f"Config chargée : {args.config}")
    logger.info(f"Étapes à exécuter : {steps}")

    try:
        run_pipeline(cfg, steps)
    except KeyboardInterrupt:
        logger.warning("\n⚠ Pipeline interrompu par l'utilisateur.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Erreur pipeline : {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()