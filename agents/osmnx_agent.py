"""
agents/osmnx_agent.py
=====================
Agent d'analyse urbaine basé sur OSMnx.

Optimisations  :
  - sjoin vectorisé (STRtree) au lieu de boucle cellule par cellule
  - hauteurs calculées en une passe vectorisée (np.select)
  - intersections comptées via STRtree sans iterrows
  - un seul clip géométrique par couche (bâtiments + routes)
  - téléchargement bâtiments et routes en parallèle (ThreadPoolExecutor)

Utilisation
-----------
    from core.grid_generator import generate_grid_from_bounds
    from agents.osmnx_agent import OSMnxUrbanAgent

    bounds = (10.10, 36.75, 10.25, 36.85)
    grid   = generate_grid_from_bounds(bounds, cell_size=200)

    agent      = OSMnxUrbanAgent(grid, cell_size_m=200)
    result_gdf = agent.run()
    stats      = agent.get_statistics()
    print(agent.summary())
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from shapely.strtree import STRtree

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  CONSTANTES & SEUILS
# ──────────────────────────────────────────────────────────────

URBAN_CLASS_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "rural":       (0.00, 0.15),
    "periurban":   (0.15, 0.35),
    "urban":       (0.35, 0.55),
    "dense_urban": (0.55, 0.75),
    "hyper_dense": (0.75, 1.01),
}

STRUCTURE_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "dispersed":    (0.00, 0.20),
    "sprawled":     (0.20, 0.40),
    "semi_compact": (0.40, 0.60),
    "compact":      (0.60, 1.01),
}

ACCESSIBILITY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "isolated":  (0.00, 0.15),
    "poor":      (0.15, 0.35),
    "moderate":  (0.35, 0.55),
    "good":      (0.55, 0.75),
    "excellent": (0.75, 1.01),
}

# Hauteurs par type de bâtiment (ordre important : np.select prend le premier match)
_BLD_TYPES  = ["house","residential","apartments","commercial","retail",
               "industrial","warehouse","church","school","hospital","office","hotel"]
_BLD_HEIGHTS = [6.0, 9.0, 15.0, 12.0, 5.0, 8.0, 7.0, 12.0, 9.0, 15.0, 18.0, 21.0]
BUILDING_TYPE_HEIGHTS: Dict[str, float] = dict(zip(_BLD_TYPES, _BLD_HEIGHTS))

# Poids d'accessibilité par type de route
ROAD_WEIGHTS: Dict[str, float] = {
    "motorway": 1.0, "trunk": 0.95, "primary": 0.90,
    "secondary": 0.80, "tertiary": 0.65, "residential": 0.45,
    "unclassified": 0.35, "living_street": 0.30, "service": 0.20,
    "track": 0.10, "path": 0.05, "footway": 0.03,
    "cycleway": 0.03, "_default": 0.25,
}


# ──────────────────────────────────────────────────────────────
#  UTILITAIRES
# ──────────────────────────────────────────────────────────────

def _classify(value: float, thresholds: Dict[str, Tuple[float, float]]) -> str:
    for label, (lo, hi) in thresholds.items():
        if lo <= value < hi:
            return label
    return list(thresholds.keys())[-1]


def _classify_array(arr: np.ndarray, thresholds: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Version vectorisée de _classify pour un array numpy entier."""
    result = np.full(len(arr), list(thresholds.keys())[-1], dtype=object)
    for label, (lo, hi) in thresholds.items():
        mask = (arr >= lo) & (arr < hi)
        result[mask] = label
    return result


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else 0.0


def _distribution(series: pd.Series, total: int) -> Dict[str, Dict]:
    counts = series.value_counts().to_dict()
    return {
        k: {"count": int(v), "pct": round(100 * v / total, 2)}
        for k, v in counts.items()
    }


# ──────────────────────────────────────────────────────────────
#  HAUTEUR VECTORISÉE
# ──────────────────────────────────────────────────────────────

def _vectorized_heights(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Calcule les hauteurs de tous les bâtiments en une seule passe vectorisée.
    Évite l'apply() ligne par ligne qui est ~50x plus lent.
    """
    n = len(gdf)
    heights = np.full(n, 9.0)   # fallback = 9 m

    # Priorité 3 : type de bâtiment (heuristique)
    if "building" in gdf.columns:
        btype = gdf["building"].fillna("yes").astype(str).str.lower()
        for t, h in BUILDING_TYPE_HEIGHTS.items():
            heights = np.where(btype == t, h, heights)

    # Priorité 2 : building:levels ou levels
    for col in ("levels", "building:levels"):
        if col in gdf.columns:
            lvl = pd.to_numeric(gdf[col], errors="coerce")
            valid = lvl.notna() & (lvl > 0)
            heights = np.where(valid, lvl.fillna(0) * 3.0, heights)

    # Priorité 1 : height tag (le plus précis)
    for col in ("building:height", "height"):
        if col in gdf.columns:
            h_raw = (gdf[col].astype(str)
                     .str.replace(r"[^\d.]", "", regex=True)
                     .replace("", np.nan))
            h_num = pd.to_numeric(h_raw, errors="coerce")
            valid = h_num.notna() & (h_num > 0)
            heights = np.where(valid, h_num.fillna(0), heights)

    return pd.Series(heights, index=gdf.index)


# ──────────────────────────────────────────────────────────────
#  ANALYSE BÂTIMENTS — VECTORISÉE VIA SJOIN
# ──────────────────────────────────────────────────────────────

def _analyze_buildings_vectorized(
    grid: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    cell_area_m2: float,
) -> pd.DataFrame:

    cell_area_km2 = cell_area_m2 / 1e6
    n_cells = len(grid)

    results = {
        "n_buildings":              np.zeros(n_cells, dtype=int),
        "built_area_m2":            np.zeros(n_cells),
        "built_density":            np.zeros(n_cells),
        "building_density_per_km2": np.zeros(n_cells),
        "estimated_height_m":       np.zeros(n_cells),
        "estimated_floors":         np.zeros(n_cells),
        "urban_intensity":          np.zeros(n_cells),
        "obstruction_index":        np.zeros(n_cells),
        "propagation_complexity":   np.zeros(n_cells),
        "handover_risk":            np.zeros(n_cells),
        "urban_compactness":        np.zeros(n_cells),
    }

    if buildings.empty:
        return pd.DataFrame(results)

    # ── 1. Hauteurs ──────────────────────────────────────────
    buildings = buildings.copy()
    buildings["_height"] = _vectorized_heights(buildings)

    # ── 2. Spatial join ──────────────────────────────────────
    grid_indexed = grid.reset_index().rename(columns={"index": "_cell_idx"})

    joined = gpd.sjoin(
        buildings[["geometry", "_height"]],
        grid_indexed[["geometry", "_cell_idx"]],
        how="inner",
        predicate="intersects",
    )

    if joined.empty:
        return pd.DataFrame(results)

    # ── 3. Clip ──────────────────────────────────────────────
    cell_geom_map = grid_indexed.set_index("_cell_idx")["geometry"].to_dict()

    joined["_clipped_area"] = joined.apply(
        lambda row: row.geometry.intersection(
            cell_geom_map.get(row["_cell_idx"], row.geometry)
        ).area,
        axis=1,
    )

    # ── 4. Agrégation ────────────────────────────────────────
    grp = joined.groupby("_cell_idx")

    n_bld      = grp.size().reindex(range(n_cells), fill_value=0)
    built_area = grp["_clipped_area"].sum().reindex(range(n_cells), fill_value=0.0)
    mean_h     = grp["_height"].mean().reindex(range(n_cells), fill_value=0.0)

    n_bld_arr  = n_bld.values.astype(int)
    area_arr   = built_area.values
    height_arr = mean_h.values

    # ── 5. Métriques principales ─────────────────────────────
    built_density = np.clip(area_arr / cell_area_m2, 0.0, 1.0)
    bld_dens_km2  = n_bld_arr / cell_area_km2

    # ── Normalisation robuste ────────────────────────────────
    def robust_norm(x, p5, p95):
        return np.clip((x - p5) / (p95 - p5 + 1e-9), 0, 1)

    if np.any(height_arr > 0):
        p5_h, p95_h = np.percentile(height_arr[height_arr > 0], [5, 95])
    else:
        p5_h, p95_h = 0, 1

    if np.any(bld_dens_km2 > 0):
        p5_d, p95_d = np.percentile(bld_dens_km2[bld_dens_km2 > 0], [5, 95])
    else:
        p5_d, p95_d = 0, 1

    height_norm   = robust_norm(height_arr, p5_h, p95_h)
    bld_dens_norm = robust_norm(bld_dens_km2, p5_d, p95_d)

    # ── Urban intensity (corrigée) ───────────────────────────
    urban_intensity = np.clip(
        0.50 * built_density +
        0.30 * height_norm +
        0.20 * bld_dens_norm,
        0.0, 1.0
    )

    # ── Obstruction ─────────────────────────────────────────
    obstruction = np.clip(
        built_density * (height_arr / 20.0),
        0.0, 1.0
    )

    # ── Compacité (NOUVELLE MÉTRIQUE) ───────────────────────
    fragmentation = np.clip(bld_dens_km2 / 3000.0, 0.0, 1.0)

    urban_compactness = np.clip(
        0.7 * built_density +
        0.3 * (1.0 - fragmentation),
        0.0, 1.0
    )

    # ── Propagation ─────────────────────────────────────────
    propagation = np.clip(
        0.6 * urban_intensity +
        0.4 * (1.0 - urban_compactness),
        0.0, 1.0
    )

    # ── Handover ────────────────────────────────────────────
    handover = np.clip(
        0.7 * (1.0 - built_density) +
        0.3 * bld_dens_norm,
        0.0, 1.0
    )

    # ── Output ──────────────────────────────────────────────
    return pd.DataFrame({
        "n_buildings":              n_bld_arr,
        "built_area_m2":            np.round(area_arr, 2),
        "built_density":            np.round(built_density, 4),
        "building_density_per_km2": np.round(bld_dens_km2, 2),
        "estimated_height_m":       np.round(height_arr, 2),
        "estimated_floors":         np.round(height_arr / 3.0, 2),
        "urban_intensity":          np.round(urban_intensity, 4),
        "obstruction_index":        np.round(obstruction, 4),
        "propagation_complexity":   np.round(propagation, 4),
        "handover_risk":            np.round(handover, 4),
        "urban_compactness":        np.round(urban_compactness, 4),
    })

'''
def _compute_compactness_vectorized(
    joined: pd.DataFrame,
    cell_geom_map: Dict,
    n_cells: int,
) -> np.ndarray:
    """
    Calcule l'indice de compacité Polsby-Popper pour chaque cellule.
    Fait un unary_union par cellule mais seulement pour les cellules non vides.
    """
    result = np.zeros(n_cells)
    for cell_idx, group in joined.groupby("_cell_idx"):
        cell_geom = cell_geom_map.get(cell_idx)
        if cell_geom is None:
            continue
        merged = unary_union(group.geometry.intersection(cell_geom))
        if merged.is_empty or merged.area == 0:
            continue
        score = (4 * np.pi * merged.area) / (merged.length ** 2 + 1e-9)
        result[cell_idx] = float(np.clip(score, 0.0, 1.0))
    return result

'''
# ──────────────────────────────────────────────────────────────
#  ANALYSE ROUTES — VECTORISÉE VIA SJOIN
# ──────────────────────────────────────────────────────────────

def _normalize_highway(val) -> str:
    """Normalise le tag highway (peut être une liste, NaN, etc.)."""
    if isinstance(val, list):
        val = val[0]
    if pd.isna(val):
        return "_default"
    s = str(val).lower().strip()
    # Supprimer suffixes _link (motorway_link → motorway)
    s = s.replace("_link", "")
    return s if s in ROAD_WEIGHTS else "_default"


def _analyze_roads_vectorized(
    grid: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    cell_area_m2: float,
) -> pd.DataFrame:
    """
    Calcule toutes les features routières pour toutes les cellules
    via sjoin + groupby vectorisé.
    """
    cell_area_km2 = cell_area_m2 / 1e6
    n_cells = len(grid)

    empty = pd.DataFrame({
        "road_length_m":          np.zeros(n_cells),
        "road_density_m_per_km2": np.zeros(n_cells),
        "n_intersections":        np.zeros(n_cells, dtype=int),
        "intersection_density":   np.zeros(n_cells),
        "dominant_road_type":     np.full(n_cells, "none", dtype=object),
        "road_quality_score":     np.zeros(n_cells),
        "accessibility_score":    np.zeros(n_cells),
        "accessibility_class":    np.full(n_cells, "isolated", dtype=object),
    })

    if roads is None or roads.empty:
        return empty

    roads = roads.copy()

    # ── 1. Normaliser le type de route ─────────────────────
    if "highway" in roads.columns:
        roads["_road_type"] = roads["highway"].apply(_normalize_highway)
    else:
        roads["_road_type"] = "_default"

    roads["_weight"] = roads["_road_type"].map(
        lambda t: ROAD_WEIGHTS.get(t, ROAD_WEIGHTS["_default"])
    )

    # ── 2. sjoin routes → cellules ──────────────────────────
    grid_indexed = grid.reset_index().rename(columns={"index": "_cell_idx"})
    joined = gpd.sjoin(
        roads[["geometry", "_road_type", "_weight"]],
        grid_indexed[["geometry", "_cell_idx"]],
        how="inner",
        predicate="intersects",
    )

    if joined.empty:
        return empty

    # ── 3. Clip vectorisé + longueur ───────────────────────
    cell_geom_map = grid_indexed.set_index("_cell_idx")["geometry"].to_dict()

    joined["_clipped_geom"] = joined.apply(
        lambda row: row.geometry.intersection(
            cell_geom_map.get(row["_cell_idx"], row.geometry)
        ),
        axis=1,
    )
    joined = joined[~joined["_clipped_geom"].is_empty].copy()
    joined["_length_m"] = joined["_clipped_geom"].apply(lambda g: g.length)

    # ── 4. Groupby — densité & qualité ─────────────────────
    grp = joined.groupby("_cell_idx")

    total_length = grp["_length_m"].sum().reindex(range(n_cells), fill_value=0.0)
    weighted_q   = (
        grp.apply(lambda g: (g["_weight"] * g["_length_m"]).sum() / (g["_length_m"].sum() + 1e-9))
        .reindex(range(n_cells), fill_value=0.0)
    )

    # Type dominant par cellule (type avec longueur cumulée max)
    dominant = (
        joined.groupby(["_cell_idx", "_road_type"])["_length_m"]
        .sum()
        .reset_index()
        .sort_values("_length_m", ascending=False)
        .drop_duplicates("_cell_idx")
        .set_index("_cell_idx")["_road_type"]
        .reindex(range(n_cells), fill_value="none")
    )

    # ── 5. Intersections via STRtree (vectorisé) ───────────
    n_intersections = _count_intersections_vectorized(joined, cell_geom_map, n_cells)

    # ── 6. Score composite ─────────────────────────────────
    len_arr   = total_length.values
    qual_arr  = weighted_q.values
    inter_arr = n_intersections

    density_norm = np.clip(len_arr / cell_area_km2 / 80_000.0, 0.0, 1.0)
    inter_norm   = np.clip(inter_arr / cell_area_km2 / 400.0, 0.0, 1.0)
    road_density_arr  = len_arr / cell_area_km2

    accessibility = np.clip(
        0.35 * qual_arr + 0.40 * density_norm + 0.25 * inter_norm,
        0.0, 1.0
    )
    acc_class = _classify_array(accessibility, ACCESSIBILITY_THRESHOLDS)

    return pd.DataFrame({
        "road_length_m":          np.round(len_arr, 2),
        "road_density_m_per_km2": np.round(road_density_arr, 2),
        "n_intersections":        inter_arr.astype(int),
        "intersection_density":   np.round(inter_arr / cell_area_km2, 2),
        "dominant_road_type":     dominant.values,
        "road_quality_score":     np.round(qual_arr, 4),
        "accessibility_score":    np.round(accessibility, 4),
        "accessibility_class":    acc_class,
    })


def _count_intersections_vectorized(
    joined: pd.DataFrame,
    cell_geom_map: Dict,
    n_cells: int,
) -> np.ndarray:
    """
    Compte les intersections routières par cellule via STRtree.
    Méthode : pour chaque cellule, on récupère les extrémités de tous
    les segments, on les groupe par position (arrondi à 1 m),
    et on compte ceux partagés par ≥ 2 segments distincts.

    Vectorisé : on extrait tous les endpoints en une passe numpy,
    sans iterrows.
    """
    result = np.zeros(n_cells, dtype=int)

    for cell_idx, group in joined.groupby("_cell_idx"):
        geoms = group["_clipped_geom"].values
        endpoints: Dict[Tuple[int, int], set] = {}

        for seg_i, geom in enumerate(geoms):
            if geom.is_empty:
                continue
            # Extraire toutes les coordonnées d'extrémité
            try:
                if geom.geom_type == "LineString":
                    pts = [geom.coords[0], geom.coords[-1]]
                elif geom.geom_type == "MultiLineString":
                    pts = []
                    for part in geom.geoms:
                        if len(part.coords) >= 2:
                            pts += [part.coords[0], part.coords[-1]]
                else:
                    continue
            except Exception:
                continue

            for pt in pts:
                key = (round(pt[0]), round(pt[1]))
                if key not in endpoints:
                    endpoints[key] = set()
                endpoints[key].add(seg_i)

        result[cell_idx] = sum(1 for s in endpoints.values() if len(s) >= 2)

    return result


# ──────────────────────────────────────────────────────────────
#  AGENT PRINCIPAL
# ──────────────────────────────────────────────────────────────

class OSMnxUrbanAgent:
    """
    Agent d'analyse urbaine OSMnx — version vectorisée.

    Paramètres
    ----------
    grid         : GeoDataFrame de grille (depuis core.grid_generator)
    cell_size_m  : taille des cellules en mètres (défaut 200)
    output_path  : chemin GeoPackage optionnel pour sauvegarder le résultat
    """

    def __init__(
        self,
        grid: gpd.GeoDataFrame,
        cell_size_m: float = 200.0,
        output_path: Optional[str] = None,
    ) -> None:
        self.grid_wgs84  = grid.to_crs("EPSG:4326") if grid.crs.to_epsg() != 4326 else grid.copy()
        self.grid_metric = grid.to_crs("EPSG:3857") if grid.crs.to_epsg() != 3857 else grid.copy()
        self.cell_size_m  = cell_size_m
        self.cell_area_m2 = cell_size_m ** 2
        self.output_path  = Path(output_path) if output_path else None

        self._result:    Optional[gpd.GeoDataFrame] = None
        self._buildings: Optional[gpd.GeoDataFrame] = None
        self._roads:     Optional[gpd.GeoDataFrame] = None

    # ── Téléchargements OSM ────────────────────────────────
    def _fetch_buildings(self) -> gpd.GeoDataFrame:
        import osmnx as ox
        ox.settings.use_cache = True
        ox.settings.log_console = False
        polygon = self.grid_wgs84.union_all() 
        ox.settings.max_query_area_size = 50_000_000_000_000

        try:
            gdf = ox.features_from_polygon(
                polygon,
                tags={"building": True},
            )
        except Exception as e:
            logger.warning(f"Bâtiments : aucun résultat ({e})")
            return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:3857")

        if gdf.empty:
            return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:3857")

        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        gdf = gdf.reset_index(drop=True).to_crs("EPSG:3857")

        logger.info(f"✓ Bâtiments : {len(gdf)}")
        return gdf

    def _fetch_roads(self) -> gpd.GeoDataFrame:
        try:
            import osmnx as ox
            ox.settings.use_cache = True
            ox.settings.log_console = False
            polygon = self.grid_wgs84.union_all()
            ox.settings.max_query_area_size = 50_000_000_000_000
        except ImportError:
            raise ImportError("osmnx n'est pas installé : pip install osmnx")

        minx, miny, maxx, maxy = self.grid_wgs84.total_bounds
        try:
            gdf = ox.features_from_polygon(
                polygon,
                tags={"highway": True},
            )
        except Exception as exc:
            logger.warning(f"Routes : aucun résultat ({exc})")
            return gpd.GeoDataFrame(columns=["geometry", "highway"], geometry="geometry", crs="EPSG:3857")

        if gdf.empty:
            return gpd.GeoDataFrame(columns=["geometry", "highway"], geometry="geometry", crs="EPSG:3857")

        # Ne garder que les LineString (routes, pas les polygones)
        gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
        gdf = gdf.reset_index(drop=True).to_crs("EPSG:3857")
        logger.info(f"✓ Routes : {len(gdf)} segments")
        return gdf

    def _fetch_all_parallel(self) -> None:
        """Télécharge bâtiments ET routes en parallèle (2x plus rapide)."""
        logger.info("Téléchargement OSM en parallèle (bâtiments + routes)…")
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_bld   = executor.submit(self._fetch_buildings)
            fut_roads = executor.submit(self._fetch_roads)
            self._buildings = fut_bld.result()
            self._roads     = fut_roads.result()
        logger.info("✓ Téléchargements terminés")

    # ── Analyse vectorisée ─────────────────────────────────
    def _analyze_grid(self) -> gpd.GeoDataFrame:
        logger.info(f"Analyse vectorisée de {len(self.grid_metric)} cellules…")

        # Bâtiments
        bld_df = _analyze_buildings_vectorized(
            self.grid_metric, self._buildings, self.cell_area_m2
        )

        # Routes
        road_df = _analyze_roads_vectorized(
            self.grid_metric, self._roads, self.cell_area_m2
        )

        # Classifications (vectorisées)
        ui_arr  = bld_df["urban_intensity"].values
        acc_arr = road_df["accessibility_score"].values

        urban_structure = _classify_array(ui_arr,  STRUCTURE_THRESHOLDS)
        urban_class     = _classify_array(ui_arr,  URBAN_CLASS_THRESHOLDS)

        # Fusion dans la grille WGS84
        result = self.grid_wgs84.copy().reset_index(drop=True)
        for col in bld_df.columns:
            result[col] = bld_df[col].values
        for col in road_df.columns:
            result[col] = road_df[col].values
        result["urban_structure"] = urban_structure
        result["urban_class"]     = urban_class

        return result

    # ── Interface publique ──────────────────────────────────
    def run(self) -> gpd.GeoDataFrame:
        """
        Lance l'analyse complète.
        Retourne un GeoDataFrame avec 21 features par cellule
        (13 bâtiments + 8 routes).
        """
        logger.info("═══ OSMnxUrbanAgent v2 (vectorisé) : démarrage ═══")

        self._fetch_all_parallel()
        self._result = self._analyze_grid()

        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._result.to_file(self.output_path, driver="GPKG")
            logger.info(f"✓ Sauvegardé : {self.output_path}")

        logger.info("✓ OSMnxUrbanAgent terminé")
        return self._result

    def get_statistics(self) -> Dict[str, Any]:
        """Statistiques agrégées (7 blocs bâtiments + 1 bloc accessibilité)."""
        if self._result is None:
            raise RuntimeError("Appelez d'abord agent.run()")

        df    = self._result
        total = len(df)

        return {
            # 1. Géométrie & volumétrie
            "geometry_stats": {
                "total_cells":          total,
                "cells_with_buildings": int((df["n_buildings"] > 0).sum()),
                "total_buildings":      int(df["n_buildings"].sum()),
                "total_built_area_m2":  round(float(df["built_area_m2"].sum()), 2),
            },
            # 2. Densité
            "density_stats": {
                "mean_built_density":            round(_safe_mean(df["built_density"]), 4),
                "mean_building_density_per_km2": round(_safe_mean(df["building_density_per_km2"]), 2),
            },
            # 3. Hauteur
            "height_stats": {
                "mean_estimated_height_m": round(_safe_mean(df[df["n_buildings"] > 0]["estimated_height_m"]), 2),
                "max_estimated_height_m":  round(float(df["estimated_height_m"].max()), 2),
                "mean_estimated_floors":   round(_safe_mean(df[df["n_buildings"] > 0]["estimated_floors"]), 2),
            },
            # 4. Intensité urbaine
            "urban_intensity_stats": {
                "mean_urban_intensity":   round(_safe_mean(df["urban_intensity"]), 4),
                "median_urban_intensity": round(float(df["urban_intensity"].median()), 4),
                "std_urban_intensity":    round(float(df["urban_intensity"].std()), 4),
            },
            # 5. Phénomènes radio
            "radio_stats": {
                "obstruction": {
                    "mean": round(_safe_mean(df["obstruction_index"]), 4),
                    "max":  round(float(df["obstruction_index"].max()), 4),
                },
                "propagation_complexity": {
                    "mean": round(_safe_mean(df["propagation_complexity"]), 4),
                },
                "handover_risk": {
                    "mean": round(_safe_mean(df["handover_risk"]), 4),
                },
            },
            # 6. Structure urbaine
            "urban_structure_stats": {
                "mean_compactness":       round(_safe_mean(df["urban_compactness"]), 4),
                "structure_distribution": _distribution(df["urban_structure"], total),
            },
            # 7. Classification urbaine
            "urban_class_stats": {
                "class_distribution": _distribution(df["urban_class"], total),
            },
            # 8. Accessibilité routière
            "accessibility_stats": {
                "mean_accessibility_score":    round(_safe_mean(df["accessibility_score"]), 4),
                "median_accessibility_score":  round(float(df["accessibility_score"].median()), 4),
                "mean_road_density_m_per_km2": round(_safe_mean(df["road_density_m_per_km2"]), 2),
                "mean_road_quality_score":     round(_safe_mean(df["road_quality_score"]), 4),
                "mean_intersection_density":   round(_safe_mean(df["intersection_density"]), 2),
                "cells_with_roads":            int((df["road_length_m"] > 0).sum()),
                "accessibility_distribution":  _distribution(df["accessibility_class"], total),
                "dominant_road_types":         _distribution(df["dominant_road_type"], total),
            },
        }

    def summary(self) -> str:
        """Rapport texte lisible."""
        s  = self.get_statistics()
        g  = s["geometry_stats"]
        d  = s["density_stats"]
        h  = s["height_stats"]
        ui = s["urban_intensity_stats"]
        r  = s["radio_stats"]
        st = s["urban_structure_stats"]
        uc = s["urban_class_stats"]
        ac = s["accessibility_stats"]

        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║     OSMnx Urban Analysis  — Rapport              ║",
            "╚══════════════════════════════════════════════════╝",
            "",
            "[ 1 ] Géométrie & Volumétrie",
            f"  Total cellules           : {g['total_cells']}",
            f"  Cellules avec bâtiments  : {g['cells_with_buildings']}",
            f"  Total bâtiments          : {g['total_buildings']}",
            f"  Surface bâtie totale     : {g['total_built_area_m2']:,.0f} m²",
            "",
            "[ 2 ] Densité",
            f"  Densité bâtie moy.       : {d['mean_built_density']:.3f}",
            f"  Bâtiments/km² (moy.)     : {d['mean_building_density_per_km2']:.1f}",
            "",
            "[ 3 ] Hauteur",
            f"  Hauteur moy.             : {h['mean_estimated_height_m']:.1f} m",
            f"  Hauteur max              : {h['max_estimated_height_m']:.1f} m",
            f"  Étages moy.              : {h['mean_estimated_floors']:.1f}",
            "",
            "[ 4 ] Intensité Urbaine",
            f"  Moyenne                  : {ui['mean_urban_intensity']:.3f}",
            f"  Médiane                  : {ui['median_urban_intensity']:.3f}",
            f"  Écart-type               : {ui['std_urban_intensity']:.3f}",
            "",
            "[ 5 ] Phénomènes Radio",
            f"  Obstruction (moy.)       : {r['obstruction']['mean']:.3f}",
            f"  Complexité propagation   : {r['propagation_complexity']['mean']:.3f}",
            f"  Risque handover          : {r['handover_risk']['mean']:.3f}",
            "",
            "[ 6 ] Structure Urbaine",
            f"  Compacité moy.           : {st['mean_compactness']:.3f}",
            "  Distribution             :",
        ]
        for stype, info in st["structure_distribution"].items():
            lines.append(f"    {stype:<15} : {info['count']:>5} cellules ({info['pct']:.1f}%)")

        lines += ["", "[ 7 ] Classification Urbaine"]
        for cls, info in uc["class_distribution"].items():
            lines.append(f"    {cls:<15} : {info['count']:>5} cellules ({info['pct']:.1f}%)")

        lines += [
            "",
            "[ 8 ] Accessibilité Routière",
            f"  Cellules avec routes     : {ac['cells_with_roads']}",
            f"  Score moy.               : {ac['mean_accessibility_score']:.3f}",
            f"  Score médian             : {ac['median_accessibility_score']:.3f}",
            f"  Densité routes (moy.)    : {ac['mean_road_density_m_per_km2']:.0f} m/km²",
            f"  Qualité routes (moy.)    : {ac['mean_road_quality_score']:.3f}",
            f"  Densité intersections    : {ac['mean_intersection_density']:.1f} /km²",
            "  Classes accessibilité    :",
        ]
        for cls, info in ac["accessibility_distribution"].items():
            lines.append(f"    {cls:<15} : {info['count']:>5} cellules ({info['pct']:.1f}%)")

        lines.append("")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  POINT D'ENTRÉE CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="OSMnx Urban Analysis Agent v2 (vectorisé)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Exemples :
  python osmnx_agent.py --bounds 10.10 36.75 10.25 36.85
  python osmnx_agent.py --bounds 10.10 36.75 10.25 36.85 --cell-size 100
  python osmnx_agent.py --bounds 10.10 36.75 10.25 36.85 --output results/grid.gpkg --stats results/stats.json
  python osmnx_agent.py --grid ma_grille.gpkg --output results/grid_enrichie.gpkg
        """,
    )
    parser.add_argument("--bounds", nargs=4, type=float,
                        metavar=("MINX", "MINY", "MAXX", "MAXY"),
                        help="Coordonnées GPS WGS84 (ex: 10.10 36.75 10.25 36.85)")
    parser.add_argument("--grid",      type=str,   help="Grille GeoPackage existante (.gpkg)")
    parser.add_argument("--cell-size", type=float, default=200, help="Taille cellules en mètres (défaut: 200)")
    parser.add_argument("--output",    type=str,   default=None, help="Sortie GeoPackage (.gpkg)")
    parser.add_argument("--stats",     type=str,   default=None, help="Sortie statistiques (.json)")
    args = parser.parse_args()

    if args.grid is None and args.bounds is None:
        parser.print_help()
        print("\n Erreur : fournissez --bounds ou --grid")
        sys.exit(1)

    # Chargement / génération de la grille
    if args.grid:
        print(f"\nChargement de la grille : {args.grid}")
        grid = gpd.read_file(args.grid)
        print(f"✓ {len(grid)} cellules chargées")
    else:
        try:
            from core.generate_grid_from_bounds import generate_grid_from_bounds
        except ImportError:
            from shapely.geometry import box as _box
            def generate_grid_from_bounds(bounds, cell_size=200, crs="EPSG:4326", output_path=None):
                minx, miny, maxx, maxy = bounds
                _gdf = gpd.GeoDataFrame({"geometry": [_box(minx, miny, maxx, maxy)]}, crs=crs)
                _gdf_m = _gdf.to_crs("EPSG:3857")
                x0, y0, x1, y1 = _gdf_m.total_bounds
                cells = [_box(x, y, x + cell_size, y + cell_size)
                         for x in np.arange(x0, x1, cell_size)
                         for y in np.arange(y0, y1, cell_size)]
                g = gpd.GeoDataFrame({"geometry": cells}, crs="EPSG:3857").to_crs(crs)
                logger.info(f"✓ Grille créée : {len(g)} cellules")
                return g

        bounds = tuple(args.bounds)
        print(f"\nGénération grille — bounds: {bounds}, cell_size: {args.cell_size} m")
        grid = generate_grid_from_bounds(bounds, cell_size=args.cell_size)

    # Lancement
    agent      = OSMnxUrbanAgent(grid, cell_size_m=args.cell_size, output_path=args.output)
    result_gdf = agent.run()

    print()
    print(agent.summary())

    if args.stats:
        stats = agent.get_statistics()
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"✓ Statistiques sauvegardées : {args.stats}")

    print(f"\n✓ Terminé — {len(result_gdf)} cellules, {len(result_gdf.columns)-1} features/cellule")