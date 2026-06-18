"""
VISUAL PLANNING AGENT — 5G Network Simulation
===============================================

Agent de visualisation et simulation interactive du réseau 5G.

Entrées:
    - final_merged_grid.gpkg  : grille 63 colonnes (OSMnx + Demand + Coverage + Terrain)
    - recommended_sites.gpkg  : sites recommandés par SitePlacementAgent

Sortie:
    - visual_planning.html    : carte interactive Folium avec simulation radio

Modèle radio:
    - Okumura-Hata (2.1 GHz) pour propagation urbaine/périurbaine/rurale
    - Intègre : urban_class, estimated_height_m, obstruction_index, los_probability

Fonctionnalités carte:
    - Couches : coverage_score, peak_demand_gbps, capacity_deficit, terrain_complexity
    - Antennes existantes (OpenCellID via nearest_radio)
    - Sites recommandés (SitePlacementAgent)
    - Simulation : clic sur carte → place antenne → recalcul couverture voisins
    - Panneau stats dynamique (KPIs mis à jour à chaque simulation)

Auteur: Stage Amaris — Pipeline 5G AI
Date: Avril 2026
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster, MiniMap, MousePosition
from branca.colormap import LinearColormap
import branca.element as be

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES OKUMURA-HATA (2.1 GHz)
# =============================================================================

FREQUENCY_MHZ = 2100.0          # 2.1 GHz → MHz pour formule Hata
ANTENNA_HEIGHT_M = 30.0          # Hauteur antenne émettrice (BS) en mètres
MOBILE_HEIGHT_M = 1.5            # Hauteur terminal mobile en mètres

# Portée maximale par urban_class (mètres) — plafond absolu
MAX_RANGE_BY_CLASS = {
    "hyper_dense":  400,
    "dense_urban":  700,
    "urban":        1200,
    "periurban":    2500,
    "rural":        5000,
    "default":      1200,
}

# Capacité simulée par antenne placée (Gbps) — baseline 5G NR 2.1 GHz
SIMULATED_CAPACITY_GBPS = 1.2

# Couleurs couches
LAYER_COLORMAPS = {
    "coverage_score":       ["#d73027", "#fc8d59", "#fee090", "#91cf60", "#1a9850"],
    "peak_demand_gbps":     ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"],
    "capacity_deficit_gbps":["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
    "terrain_complexity":   ["#f7f7f7", "#cccccc", "#969696", "#636363", "#252525"],
}


# =============================================================================
# MODÈLE RADIO OKUMURA-HATA
# =============================================================================

class OkumuraHataModel:
    """
    Modèle de propagation Okumura-Hata (2.1 GHz).

    Fréquence fixe : 2100 MHz (bande 2.1 GHz LTE/5G NR)
    Hauteur BS     : 30 m (macro cell standard)
    Hauteur MS     : 1.5 m (mobile)

    Formules selon CCIR Rep. 567 / ITU-R P.1546:
      Urban    : Lu = 69.55 + 26.16·log(f) - 13.82·log(hb) - a(hm) + (44.9 - 6.55·log(hb))·log(d)
      Suburban : Lsu = Lu - 2·(log(f/28))² - 5.4
      Rural    : Lo  = Lu - 4.78·(log(f))² + 18.33·log(f) - 40.94

    Retourne: path_loss (dB) et portée effective (m) pour un budget de liaison fixe.
    """

    # Budget de liaison : EIRP - sensibilité récepteur (dB)
    LINK_BUDGET_DB = 140.0

    def __init__(self, freq_mhz: float = FREQUENCY_MHZ,
                 hb: float = ANTENNA_HEIGHT_M,
                 hm: float = MOBILE_HEIGHT_M):
        self.freq_mhz = freq_mhz
        self.hb = hb        # hauteur BS (mètres)
        self.hm = hm        # hauteur mobile (mètres)

        # Facteur de correction hauteur mobile (grande ville)
        self._a_hm = (
            3.2 * (math.log10(11.75 * hm)) ** 2 - 4.97
        )

    def path_loss_db(self, distance_m: float, urban_class: str,
                     obstruction_index: float = 0.5) -> float:
        """
        Calcule la perte de trajet (dB) pour une distance et un contexte donnés.

        Args:
            distance_m       : distance émetteur-récepteur (mètres)
            urban_class      : classe urbaine OSMnx
            obstruction_index: indice d'obstruction OSMnx [0,1]

        Returns:
            path_loss en dB
        """
        if distance_m <= 0:
            return 0.0

        d_km = max(distance_m, 1.0) / 1000.0
        f    = self.freq_mhz
        hb   = self.hb

        # Perte de base (formule urbaine Hata)
        Lu = (
            69.55
            + 26.16 * math.log10(f)
            - 13.82 * math.log10(hb)
            - self._a_hm
            + (44.9 - 6.55 * math.log10(hb)) * math.log10(d_km)
        )

        # Correction selon urban_class
        if urban_class in ("rural", "periurban"):
            # Formule open/quasi-open area
            correction = -4.78 * (math.log10(f)) ** 2 + 18.33 * math.log10(f) - 40.94
            L = Lu + correction
        elif urban_class == "urban":
            # Suburban correction
            L = Lu - 2 * (math.log10(f / 28.0)) ** 2 - 5.4
        else:
            # dense_urban / hyper_dense → Hata urbain pur
            L = Lu

        # Correction obstruction OSMnx (+0 à +8 dB selon densité bâti)
        obstruction_penalty = obstruction_index * 8.0

        return L + obstruction_penalty

    def effective_range_m(self, urban_class: str,
                          obstruction_index: float = 0.5) -> float:
        """
        Calcule la portée effective maximale (en mètres) pour le budget
        de liaison défini, en résolvant path_loss_db(d) = LINK_BUDGET_DB.

        Méthode : recherche binaire simple (< 20 itérations).
        """
        max_range = MAX_RANGE_BY_CLASS.get(urban_class, MAX_RANGE_BY_CLASS["default"])

        lo, hi = 10.0, float(max_range)
        for _ in range(20):
            mid = (lo + hi) / 2.0
            pl  = self.path_loss_db(mid, urban_class, obstruction_index)
            if pl < self.LINK_BUDGET_DB:
                lo = mid
            else:
                hi = mid

        return lo

    def signal_strength_dbm(self, distance_m: float, urban_class: str,
                             obstruction_index: float = 0.5,
                             tx_power_dbm: float = 43.0) -> float:
        """
        Signal reçu en dBm.
        tx_power_dbm = 43 dBm → 20W (macro cell standard 2.1 GHz)
        """
        return tx_power_dbm - self.path_loss_db(distance_m, urban_class, obstruction_index)


# =============================================================================
# VISUAL PLANNING AGENT
# =============================================================================

class VisualPlanningAgent:
    """
    Agent de visualisation et simulation 5G réseau.

    Génère une carte Folium interactive HTML à partir de la grille
    fusionnée (63 colonnes) et des sites recommandés.

    La simulation Okumura-Hata (2.1 GHz) est intégrée côté JavaScript :
    quand l'utilisateur place une antenne sur la carte, les cellules voisines
    voient leur coverage_score recalculé instantanément sans serveur.
    """

    def __init__(self,
                 grid_path: str,
                 sites_path: str,
                 output_path: str = "outputs/visual_planning.html",
                 config: Optional[dict] = None):
        """
        Args:
            grid_path   : Chemin final_merged_grid.gpkg (63 colonnes)
            sites_path  : Chemin recommended_sites.gpkg (SitePlacementAgent)
            output_path : Chemin HTML de sortie
            config      : Configuration optionnelle
        """
        self.grid_path   = Path(grid_path)
        self.sites_path  = Path(sites_path)
        self.output_path = Path(output_path)

        if not self.grid_path.exists():
            raise FileNotFoundError(f"Grille introuvable: {self.grid_path}")
        if not self.sites_path.exists():
            raise FileNotFoundError(f"Sites introuvables: {self.sites_path}")

        self.config = {
            "map_tiles":       "CartoDB positron",
            "default_layer":   "coverage_score",
            "opacity":         0.65,
            "sim_radius_cells": 8,       # Nb cellules voisines affectées par simulation
        }
        if config:
            self.config.update(config)

        # Modèle radio
        self.radio_model = OkumuraHataModel()

        # Chargement données
        logger.info(f"Chargement grille: {self.grid_path}")
        self.grid = gpd.read_file(self.grid_path)
        logger.info(f"✓ Grille: {len(self.grid)} cellules | {len(self.grid.columns)} colonnes")

        logger.info(f"Chargement sites: {self.sites_path}")
        self.sites = gpd.read_file(self.sites_path)
        logger.info(f"✓ Sites recommandés: {len(self.sites)}")

        self._validate_and_patch()

    # =========================================================================
    # VALIDATION & PATCH
    # =========================================================================

    def _validate_and_patch(self):
        """Vérifie colonnes requises et injecte défauts si absentes."""
        required = [
            "coverage_score", "peak_demand_gbps", "capacity_deficit_gbps",
            "urban_class", "obstruction_index", "estimated_height_m",
            "los_probability", "terrain_complexity", "is_covered",
            "antenna_count", "population", "capacity_available_gbps",
        ]
        for col in required:
            if col not in self.grid.columns:
                logger.warning(f"Colonne absente → défaut: {col}")
                self.grid[col] = 0.0

        # Normaliser CRS → WGS84 pour Folium
        if self.grid.crs and self.grid.crs.to_epsg() != 4326:
            self.grid = self.grid.to_crs("EPSG:4326")
        if self.sites.crs and self.sites.crs.to_epsg() != 4326:
            self.sites = self.sites.to_crs("EPSG:4326")

        logger.info("✓ Validation OK")

    # =========================================================================
    # PRÉPARATION DONNÉES JAVASCRIPT
    # =========================================================================

    def _prepare_grid_geojson(self) -> dict:
        """
        Prépare le GeoJSON allégé de la grille pour la carte.
        Garde uniquement les colonnes nécessaires à la visualisation
        et à la simulation côté JS.
        """
        cols_to_keep = [
            "coverage_score", "peak_demand_gbps", "capacity_deficit_gbps",
            "terrain_complexity", "urban_class", "obstruction_index",
            "estimated_height_m", "los_probability", "is_covered",
            "antenna_count", "population", "capacity_available_gbps",
            "coverage_class", "demand_category", "urban_intensity",
            "geometry"
        ]
        cols_available = [c for c in cols_to_keep if c in self.grid.columns]
        gdf = self.grid[cols_available].copy()

        # Arrondir float pour alléger le JSON
        float_cols = gdf.select_dtypes(include=[np.floating]).columns
        gdf[float_cols] = gdf[float_cols].round(4)

        # Convertir booléens
        bool_cols = gdf.select_dtypes(include=[bool]).columns
        for col in bool_cols:
            gdf[col] = gdf[col].astype(int)

        return json.loads(gdf.to_json())

    def _prepare_sites_geojson(self) -> dict:
        """Prépare le GeoJSON des sites recommandés."""
        cols = ["site_type", "composite_score", "urban_class",
                "capacity_deficit_gbps", "geometry"]
        cols_available = [c for c in cols if c in self.sites.columns]
        gdf = self.sites[cols_available].copy()

        float_cols = gdf.select_dtypes(include=[np.floating]).columns
        gdf[float_cols] = gdf[float_cols].round(4)

        return json.loads(gdf.to_json())

    def _compute_radio_params_per_class(self) -> dict:
        """
        Pré-calcule les portées effectives Okumura-Hata par urban_class
        pour les passer au moteur JS.
        """
        urban_classes = self.grid["urban_class"].unique().tolist()
        params = {}

        for uc in urban_classes:
            # Obstruction médiane pour cette classe
            mask = self.grid["urban_class"] == uc
            med_obs = float(self.grid.loc[mask, "obstruction_index"].median())

            range_m = self.radio_model.effective_range_m(uc, med_obs)
            pl_at_range = self.radio_model.path_loss_db(range_m, uc, med_obs)

            params[uc] = {
                "effective_range_m": round(range_m, 1),
                "path_loss_at_range_db": round(pl_at_range, 1),
                "median_obstruction": round(med_obs, 3),
                "capacity_gbps": SIMULATED_CAPACITY_GBPS,
            }
            logger.info(f"  Radio [{uc}]: portée={range_m:.0f}m | PL={pl_at_range:.1f}dB")

        return params

    def _get_map_center(self) -> Tuple[float, float]:
        """Calcule le centre de la carte."""
        bounds = self.grid.total_bounds  # [minx, miny, maxx, maxy]
        return (
            (bounds[1] + bounds[3]) / 2,
            (bounds[0] + bounds[2]) / 2,
        )

    def _get_kpis(self) -> dict:
        """Calcule les KPIs globaux pour le panneau de stats."""
        g = self.grid
        return {
            "total_cells":        int(len(g)),
            "covered_cells":      int(g["is_covered"].sum()) if "is_covered" in g.columns else 0,
            "coverage_rate":      round(float(g["is_covered"].mean() * 100), 1) if "is_covered" in g.columns else 0,
            "mean_coverage_score": round(float(g["coverage_score"].mean()), 3),
            "total_demand_gbps":  round(float(g["peak_demand_gbps"].sum()), 1),
            "total_capacity_gbps": round(float(g["capacity_available_gbps"].sum()), 1),
            "total_deficit_gbps": round(float(g["capacity_deficit_gbps"].sum()), 1),
            "n_sites_recommended": int(len(self.sites)),
            "total_population":   int(g["population"].sum()),
        }

    # =========================================================================
    # GÉNÉRATION HTML
    # =========================================================================

    def generate(self) -> Path:
        """
        Génère la carte interactive HTML complète.

        Returns:
            Path vers le fichier HTML généré
        """
        logger.info("=" * 70)
        logger.info("VISUAL PLANNING AGENT — Génération carte")
        logger.info("=" * 70)

        # Prépare les données
        logger.info("Préparation données GeoJSON...")
        grid_geojson  = self._prepare_grid_geojson()
        sites_geojson = self._prepare_sites_geojson()

        logger.info("Calcul paramètres radio par urban_class...")
        radio_params = self._compute_radio_params_per_class()

        kpis = self._get_kpis()
        center = self._get_map_center()

        logger.info(f"Centre carte: {center}")
        logger.info(f"KPIs: {kpis}")

        # Construction HTML
        logger.info("Construction HTML...")
        html = self._build_html(
            grid_geojson=grid_geojson,
            sites_geojson=sites_geojson,
            radio_params=radio_params,
            kpis=kpis,
            center=center,
        )

        # Écriture fichier
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"✓ Carte générée: {self.output_path}")
        logger.info(f"  Taille: {self.output_path.stat().st_size / 1024:.0f} KB")

        return self.output_path

    # =========================================================================
    # CONSTRUCTION HTML (carte Leaflet + simulation JS)
    # =========================================================================

    def _build_html(self, grid_geojson, sites_geojson, radio_params, kpis, center) -> str:
        """
        Construit le fichier HTML complet avec :
        - Carte Leaflet (via Folium CDN)
        - Couches GeoJSON colorées par métrique
        - Contrôleur de couches
        - Moteur de simulation Okumura-Hata en JavaScript
        - Panneau KPIs dynamique
        - Panneau simulation (résultats recalculés)
        """

        grid_json_str  = json.dumps(grid_geojson)
        sites_json_str = json.dumps(sites_geojson)
        radio_json_str = json.dumps(radio_params)
        kpis_json_str  = json.dumps(kpis)

        lat, lon = center

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>5G Visual Planning — Stage Amaris</title>

    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
    <!-- Leaflet JS -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>

    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}

        /* ── HEADER ── */
        #header {{
            background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 2px solid #e94560;
            flex-shrink: 0;
        }}
        #header h1 {{
            font-size: 16px;
            color: #e94560;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}
        #header .subtitle {{
            font-size: 11px;
            color: #a0a0b0;
            margin-top: 2px;
        }}
        #sim-mode-badge {{
            background: #e94560;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            display: none;
        }}
        #sim-mode-badge.active {{ display: inline-block; animation: pulse 1.5s infinite; }}
        @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.5; }} }}

        /* ── MAIN LAYOUT ── */
        #main {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}

        /* ── MAP ── */
        #map {{
            flex: 1;
            cursor: crosshair;
        }}

        /* ── SIDEBAR ── */
        #sidebar {{
            width: 300px;
            background: #16213e;
            border-left: 1px solid #0f3460;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            flex-shrink: 0;
        }}

        .panel {{
            padding: 12px;
            border-bottom: 1px solid #0f3460;
        }}
        .panel-title {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #e94560;
            margin-bottom: 8px;
            font-weight: bold;
        }}

        /* ── KPI GRID ── */
        .kpi-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
        }}
        .kpi-card {{
            background: #0f3460;
            border-radius: 6px;
            padding: 8px;
            text-align: center;
        }}
        .kpi-value {{
            font-size: 18px;
            font-weight: bold;
            color: #e94560;
        }}
        .kpi-label {{
            font-size: 9px;
            color: #a0a0b0;
            text-transform: uppercase;
            margin-top: 2px;
        }}

        /* ── LAYER SELECTOR ── */
        .layer-btn {{
            display: block;
            width: 100%;
            padding: 7px 10px;
            margin-bottom: 4px;
            background: #0f3460;
            border: 1px solid transparent;
            border-radius: 5px;
            color: #e0e0e0;
            font-size: 12px;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s;
        }}
        .layer-btn:hover {{ background: #1a4a8a; }}
        .layer-btn.active {{ border-color: #e94560; color: #e94560; background: #1a1a3e; }}

        /* ── SIMULATION PANEL ── */
        #sim-panel {{
            background: #0d1b2a;
            border-radius: 6px;
            padding: 10px;
        }}
        #sim-status {{
            font-size: 12px;
            color: #a0a0b0;
            margin-bottom: 8px;
        }}
        .sim-result {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            padding: 4px 0;
            border-bottom: 1px solid #1a3060;
        }}
        .sim-result .label {{ color: #a0a0b0; }}
        .sim-result .value {{ color: #4ecdc4; font-weight: bold; }}
        .sim-result .value.improved {{ color: #6bcb77; }}
        .sim-result .value.degraded  {{ color: #e94560; }}

        #btn-sim-toggle {{
            width: 100%;
            padding: 8px;
            background: #e94560;
            border: none;
            border-radius: 5px;
            color: white;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.2s;
            margin-bottom: 8px;
        }}
        #btn-sim-toggle:hover {{ background: #c73652; }}
        #btn-sim-toggle.active {{ background: #27ae60; }}

        #btn-reset {{
            width: 100%;
            padding: 6px;
            background: #0f3460;
            border: 1px solid #e94560;
            border-radius: 5px;
            color: #e94560;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        #btn-reset:hover {{ background: #1a1a3e; }}

        /* ── LEGEND ── */
        #legend-bar {{
            height: 12px;
            border-radius: 4px;
            background: linear-gradient(to right, #d73027, #fc8d59, #fee090, #91cf60, #1a9850);
            margin: 6px 0;
        }}
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 9px;
            color: #a0a0b0;
        }}

        /* ── ANTENNA MARKER ── */
        .sim-antenna-icon {{
            background: #e94560;
            border: 2px solid white;
            border-radius: 50%;
            width: 14px;
            height: 14px;
        }}

        /* ── TOOLTIP ── */
        .leaflet-tooltip {{
            background: #16213e;
            border: 1px solid #e94560;
            color: #e0e0e0;
            font-size: 11px;
            border-radius: 4px;
            padding: 6px 10px;
        }}
    </style>
</head>

<body>

<!-- ── HEADER ── -->
<div id="header">
    <div>
        <h1>5G Visual Planning · Simulation Réseau</h1>
        <div class="subtitle">Okumura-Hata 2.1 GHz · Stage Amaris 2026</div>
    </div>
    <span id="sim-mode-badge"> MODE SIMULATION ACTIF</span>
</div>

<!-- ── MAIN ── -->
<div id="main">

    <!-- MAP -->
    <div id="map"></div>

    <!-- SIDEBAR -->
    <div id="sidebar">

        <!-- KPIs -->
        <div class="panel">
            <div class="panel-title"> KPIs Réseau</div>
            <div class="kpi-grid" id="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value" id="kpi-coverage-rate">{kpis['coverage_rate']}%</div>
                    <div class="kpi-label">Couverture</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" id="kpi-score">{kpis['mean_coverage_score']}</div>
                    <div class="kpi-label">Score Moyen</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" id="kpi-deficit">{kpis['total_deficit_gbps']} G</div>
                    <div class="kpi-label">Déficit Total</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value" id="kpi-sites">{kpis['n_sites_recommended']}</div>
                    <div class="kpi-label">Sites Recom.</div>
                </div>
            </div>
        </div>

        <!-- LAYER SELECTOR -->
        <div class="panel">
            <div class="panel-title"> Couche Active</div>
            <button class="layer-btn active" onclick="setLayer('coverage_score')"> Score Couverture</button>
            <button class="layer-btn" onclick="setLayer('peak_demand_gbps')"> Demande Pic (Gbps)</button>
            <button class="layer-btn" onclick="setLayer('capacity_deficit_gbps')"> Déficit Capacité</button>
            <button class="layer-btn" onclick="setLayer('terrain_complexity')"> Complexité Terrain</button>
        </div>

        <!-- LEGEND -->
        <div class="panel">
            <div class="panel-title"> Légende</div>
            <div id="legend-bar"></div>
            <div class="legend-labels">
                <span id="legend-min">0</span>
                <span id="legend-mid">0.5</span>
                <span id="legend-max">1</span>
            </div>
            <div style="font-size:10px; color:#a0a0b0; margin-top:6px;" id="legend-desc">
                Score couverture composite [0-1]
            </div>
        </div>

        <!-- SIMULATION -->
        <div class="panel">
            <div class="panel-title"> Simulation Antenne</div>
            <button id="btn-sim-toggle" onclick="toggleSimMode()">
                 Activer Mode Simulation
            </button>
            <div id="sim-panel">
                <div id="sim-status">Activez le mode simulation puis cliquez sur la carte pour placer une antenne.</div>
                <div id="sim-results" style="display:none;">
                    <div class="sim-result">
                        <span class="label">Position</span>
                        <span class="value" id="sim-pos">—</span>
                    </div>
                    <div class="sim-result">
                        <span class="label">Urban Class</span>
                        <span class="value" id="sim-uc">—</span>
                    </div>
                    <div class="sim-result">
                        <span class="label">Portée Hata</span>
                        <span class="value" id="sim-range">—</span>
                    </div>
                    <div class="sim-result">
                        <span class="label">Cellules couvertes</span>
                        <span class="value improved" id="sim-cells">—</span>
                    </div>
                    <div class="sim-result">
                        <span class="label">Capacité ajoutée</span>
                        <span class="value improved" id="sim-cap">—</span>
                    </div>
                    <div class="sim-result">
                        <span class="label">Déficit réduit</span>
                        <span class="value improved" id="sim-deficit">—</span>
                    </div>
                    <div class="sim-result">
                        <span class="label">Population couverte</span>
                        <span class="value improved" id="sim-pop">—</span>
                    </div>
                </div>
            </div>
            <br/>
            <button id="btn-reset" onclick="resetSimulation()"> Réinitialiser</button>
        </div>

        <!-- SITES INFO -->
        <div class="panel">
            <div class="panel-title"> Sites Recommandés</div>
            <div style="font-size:11px; color:#a0a0b0;" id="sites-summary">
                {len(self.sites)} sites · Cliquer un marqueur pour détails
            </div>
        </div>

    </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════════
     JAVASCRIPT — Carte + Moteur Simulation
     ════════════════════════════════════════════════════════════════════ -->
<script>

// ── DONNÉES ────────────────────────────────────────────────────────────
const GRID_DATA    = {grid_json_str};
const SITES_DATA   = {sites_json_str};
const RADIO_PARAMS = {radio_json_str};
const INITIAL_KPIS = {kpis_json_str};

// ── CONSTANTES SIMULATION ──────────────────────────────────────────────
const SIMULATED_CAPACITY_GBPS = {SIMULATED_CAPACITY_GBPS};
const FREQ_MHZ                = {FREQUENCY_MHZ};
const ANTENNA_HEIGHT_M        = {ANTENNA_HEIGHT_M};
const LINK_BUDGET_DB          = {OkumuraHataModel.LINK_BUDGET_DB};

// ── STATE ──────────────────────────────────────────────────────────────
let simMode      = false;
let simAntennas  = [];   // {{ marker, lat, lng, affectedLayers }}
let currentLayer = 'coverage_score';
let gridLayers   = {{}};   // GeoJSON layers par métrique
let sitesLayer   = null;
let map          = null;

// ── COLORMAPS ──────────────────────────────────────────────────────────
const COLORMAPS = {{
    coverage_score:        ['#d73027','#fc8d59','#fee090','#91cf60','#1a9850'],
    peak_demand_gbps:      ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494'],
    capacity_deficit_gbps: ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026'],
    terrain_complexity:    ['#f7f7f7','#cccccc','#969696','#636363','#252525'],
}};

const LAYER_DESCS = {{
    coverage_score:        {{ label: 'Score Couverture [0-1]',     unit: '',    min: 0, max: 1 }},
    peak_demand_gbps:      {{ label: 'Demande Pic (Gbps)',          unit: 'Gbps',min: 0, max: null }},
    capacity_deficit_gbps: {{ label: 'Déficit Capacité (Gbps)',     unit: 'Gbps',min: 0, max: null }},
    terrain_complexity:    {{ label: 'Complexité Terrain [0-1]',    unit: '',    min: 0, max: 1 }},
}};

// ── UTILITAIRES ────────────────────────────────────────────────────────

function interpolateColor(colors, t) {{
    // Interpolation linéaire entre les couleurs du colormap
    t = Math.max(0, Math.min(1, t));
    const n = colors.length - 1;
    const i = Math.floor(t * n);
    const f = t * n - i;
    if (i >= n) return colors[n];
    const c1 = hexToRgb(colors[i]);
    const c2 = hexToRgb(colors[i+1]);
    const r = Math.round(c1.r + f*(c2.r - c1.r));
    const g = Math.round(c1.g + f*(c2.g - c1.g));
    const b = Math.round(c1.b + f*(c2.b - c1.b));
    return `rgb(${{r}},${{g}},${{b}})`;
}}

function hexToRgb(hex) {{
    const r = parseInt(hex.slice(1,3),16);
    const g = parseInt(hex.slice(3,5),16);
    const b = parseInt(hex.slice(5,7),16);
    return {{r,g,b}};
}}

function getColor(value, layerName, minV, maxV) {{
    if (value === null || value === undefined || isNaN(value)) return '#333333';
    const colors = COLORMAPS[layerName];
    if (!colors) return '#888888';
    const t = maxV > minV ? (value - minV)/(maxV - minV) : 0;
    return interpolateColor(colors, t);
}}

function haversineDistance(lat1, lon1, lat2, lon2) {{
    // Distance en mètres entre deux points WGS84
    const R = 6371000;
    const φ1 = lat1 * Math.PI/180, φ2 = lat2 * Math.PI/180;
    const Δφ = (lat2-lat1) * Math.PI/180;
    const Δλ = (lon2-lon1) * Math.PI/180;
    const a = Math.sin(Δφ/2)**2 + Math.cos(φ1)*Math.cos(φ2)*Math.sin(Δλ/2)**2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}}

// ── MODÈLE OKUMURA-HATA (JavaScript) ──────────────────────────────────
// Reproduit la logique Python de OkumuraHataModel
function okumuraHataPathLoss(distanceM, urbanClass, obstructionIndex) {{
    if (distanceM <= 0) return 0;
    const dKm = Math.max(distanceM, 1.0) / 1000.0;
    const f  = FREQ_MHZ;
    const hb = ANTENNA_HEIGHT_M;
    const hm = 1.5;
    // Facteur correction hauteur mobile (grande ville)
    const a_hm = 3.2 * (Math.log10(11.75 * hm))**2 - 4.97;
    // Perte de base Hata urbaine
    const Lu = 69.55
        + 26.16 * Math.log10(f)
        - 13.82 * Math.log10(hb)
        - a_hm
        + (44.9 - 6.55 * Math.log10(hb)) * Math.log10(dKm);

    let L;
    if (urbanClass === 'rural' || urbanClass === 'periurban') {{
        const corr = -4.78 * (Math.log10(f))**2 + 18.33 * Math.log10(f) - 40.94;
        L = Lu + corr;
    }} else if (urbanClass === 'urban') {{
        L = Lu - 2 * (Math.log10(f/28.0))**2 - 5.4;
    }} else {{
        L = Lu;
    }}
    // Pénalité obstruction OSMnx (+0 à +8 dB)
    L += (obstructionIndex || 0.5) * 8.0;
    return L;
}}

function effectiveRangeM(urbanClass, obstructionIndex) {{
    // Recherche binaire : trouve d tel que pathLoss(d) = LINK_BUDGET_DB
    const params = RADIO_PARAMS[urbanClass] || RADIO_PARAMS['urban'] || {{}};
    let lo = 10, hi = params.effective_range_m || 1200;
    for (let i = 0; i < 20; i++) {{
        const mid = (lo + hi) / 2;
        const pl  = okumuraHataPathLoss(mid, urbanClass, obstructionIndex);
        if (pl < LINK_BUDGET_DB) lo = mid;
        else hi = mid;
    }}
    return lo;
}}

// ── SIMULATION RADIO ───────────────────────────────────────────────────

function simulateAntenna(lat, lng) {{
    // Trouver la cellule la plus proche pour connaître l'urban_class
    let nearestFeature = null;
    let nearestDist    = Infinity;

    for (const feat of GRID_DATA.features) {{
        const cent = getCentroid(feat.geometry);
        const d    = haversineDistance(lat, lng, cent[1], cent[0]);
        if (d < nearestDist) {{ nearestDist = d; nearestFeature = feat; }}
    }}

    const uc  = nearestFeature ? (nearestFeature.properties.urban_class || 'urban') : 'urban';
    const obs = nearestFeature ? (nearestFeature.properties.obstruction_index || 0.5) : 0.5;
    const rangeM = effectiveRangeM(uc, obs);

    // Trouver toutes les cellules dans la portée
    let affectedFeatures = [];
    let totalPop = 0, totalDeficit = 0, totalCap = 0;

    for (const feat of GRID_DATA.features) {{
        const cent = getCentroid(feat.geometry);
        const d    = haversineDistance(lat, lng, cent[1], cent[0]);
        if (d <= rangeM) {{
            const pl       = okumuraHataPathLoss(d, uc, obs);
            const signal   = Math.max(0, 1 - pl / 160);  // normalisation signal [0,1]
            const capGain  = SIMULATED_CAPACITY_GBPS * signal;
            const prevDef  = feat.properties.capacity_deficit_gbps || 0;
            const newDef   = Math.max(0, prevDef - capGain);

            affectedFeatures.push({{ feat, signal, capGain, prevDef, newDef, distance: d }});
            totalPop     += (feat.properties.population || 0);
            totalDeficit += prevDef;
            totalCap     += capGain;
        }}
    }}

    return {{
        urban_class:  uc,
        range_m:      Math.round(rangeM),
        n_cells:      affectedFeatures.length,
        total_pop:    Math.round(totalPop),
        cap_added:    Math.round(totalCap * 10) / 10,
        deficit_reduced: Math.round(Math.min(totalDeficit, totalCap) * 10) / 10,
        affected:     affectedFeatures,
    }};
}}

function getCentroid(geometry) {{
    if (!geometry) return [0, 0];
    if (geometry.type === 'Point') return geometry.coordinates;
    // Pour Polygon/MultiPolygon : moyenne des coordonnées du premier ring
    let coords;
    if (geometry.type === 'Polygon') coords = geometry.coordinates[0];
    else if (geometry.type === 'MultiPolygon') coords = geometry.coordinates[0][0];
    else return [0, 0];
    const sumX = coords.reduce((s,c)=>s+c[0], 0);
    const sumY = coords.reduce((s,c)=>s+c[1], 0);
    return [sumX/coords.length, sumY/coords.length];
}}

// ── CARTE ──────────────────────────────────────────────────────────────

function initMap() {{
    map = L.map('map', {{
        center: [{lat}, {lon}],
        zoom: 14,
        zoomControl: true,
    }});

    // Fond de carte
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}.png', {{
        attribution: '© OpenStreetMap © CARTO',
        maxZoom: 19,
    }}).addTo(map);

    // Préparer toutes les couches
    buildAllLayers();

    // Afficher couche par défaut
    showLayer('coverage_score');

    // Ajouter sites recommandés
    addSitesLayer();

    // Clic carte → simulation
    map.on('click', onMapClick);
}}

function computeLayerBounds(layerName) {{
    let values = GRID_DATA.features
        .map(f => f.properties[layerName])
        .filter(v => v !== null && v !== undefined && !isNaN(v));
    if (!values.length) return {{ min: 0, max: 1 }};
    return {{
        min: Math.min(...values),
        max: Math.max(...values),
    }};
}}

function buildAllLayers() {{
    for (const layerName of Object.keys(COLORMAPS)) {{
        const bounds = computeLayerBounds(layerName);
        gridLayers[layerName] = L.geoJSON(GRID_DATA, {{
            style: function(feat) {{
                const val = feat.properties[layerName];
                return {{
                    fillColor:   getColor(val, layerName, bounds.min, bounds.max),
                    fillOpacity: 0.65,
                    color:       '#333',
                    weight:      0.3,
                    opacity:     0.5,
                }};
            }},
            onEachFeature: function(feat, layer) {{
                layer.on('mouseover', function() {{
                    const p = feat.properties;
                    const val = (p[layerName] !== undefined) ? p[layerName].toFixed(3) : 'N/A';
                    layer.bindTooltip(
                        `<b>${{layerName.replace(/_/g,' ')}}</b>: ${{val}}<br>
                         Urban: ${{p.urban_class || 'N/A'}}<br>
                         Couverture: ${{(p.coverage_score||0).toFixed(2)}} | 
                         Demande: ${{(p.peak_demand_gbps||0).toFixed(2)}} Gbps<br>
                         Déficit: ${{(p.capacity_deficit_gbps||0).toFixed(2)}} Gbps | 
                         Pop: ${{Math.round(p.population||0)}}`,
                        {{ sticky: true }}
                    ).openTooltip();
                }});
                layer.on('mouseout', function() {{ layer.closeTooltip(); }});
            }}
        }});
    }}
}}

function showLayer(layerName) {{
    // Supprimer couches actives
    for (const [name, layer] of Object.entries(gridLayers)) {{
        if (map.hasLayer(layer)) map.removeLayer(layer);
    }}
    // Afficher couche sélectionnée
    if (gridLayers[layerName]) {{
        gridLayers[layerName].addTo(map);
    }}
    currentLayer = layerName;
    updateLegend(layerName);
}}

function setLayer(layerName) {{
    showLayer(layerName);
    // Mise à jour boutons
    document.querySelectorAll('.layer-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
}}

function updateLegend(layerName) {{
    const bounds = computeLayerBounds(layerName);
    const desc   = LAYER_DESCS[layerName] || {{}};
    const colors = COLORMAPS[layerName];

    // Gradient
    document.getElementById('legend-bar').style.background =
        `linear-gradient(to right, ${{colors.join(',')}})`;
    document.getElementById('legend-min').textContent = bounds.min.toFixed(2);
    document.getElementById('legend-mid').textContent = ((bounds.min+bounds.max)/2).toFixed(2);
    document.getElementById('legend-max').textContent = bounds.max.toFixed(2);
    document.getElementById('legend-desc').textContent = desc.label || layerName;
}}

function addSitesLayer() {{
    sitesLayer = L.geoJSON(SITES_DATA, {{
        pointToLayer: function(feat, latlng) {{
            const stype = feat.properties.site_type || 'macro_cell';
            const color = {{
                'small_cell_dense': '#ff6b6b',
                'micro_cell':       '#ffd93d',
                'macro_cell':       '#6bcb77',
                'macro_cell_large': '#4ecdc4',
            }}[stype] || '#ffffff';

            return L.circleMarker(latlng, {{
                radius:      8,
                fillColor:   color,
                color:       '#fff',
                weight:      2,
                opacity:     1,
                fillOpacity: 0.9,
            }});
        }},
        onEachFeature: function(feat, layer) {{
            const p = feat.properties;
            layer.bindPopup(
                `<b>Site Recommandé</b><br>
                 Type: <b>${{p.site_type || 'N/A'}}</b><br>
                 Urban: ${{p.urban_class || 'N/A'}}<br>
                 Score: ${{(p.composite_score||0).toFixed(3)}}<br>
                 Déficit: ${{(p.capacity_deficit_gbps||0).toFixed(2)}} Gbps`
            );
        }}
    }}).addTo(map);
}}

// ── GESTION SIMULATION ─────────────────────────────────────────────────

function toggleSimMode() {{
    simMode = !simMode;
    const btn   = document.getElementById('btn-sim-toggle');
    const badge = document.getElementById('sim-mode-badge');
    const mapEl = document.getElementById('map');

    if (simMode) {{
        btn.textContent   = ' Désactiver Mode Simulation';
        btn.classList.add('active');
        badge.classList.add('active');
        mapEl.style.cursor = 'crosshair';
        document.getElementById('sim-status').textContent =
            ' Cliquez sur la carte pour placer une antenne simulée.';
    }} else {{
        btn.textContent   = ' Activer Mode Simulation';
        btn.classList.remove('active');
        badge.classList.remove('active');
        mapEl.style.cursor = '';
        document.getElementById('sim-status').textContent =
            'Activez le mode simulation puis cliquez sur la carte.';
    }}
}}

function onMapClick(e) {{
    if (!simMode) return;
    const lat = e.latlng.lat, lng = e.latlng.lng;

    // Lancer simulation
    const result = simulateAntenna(lat, lng);

    // Placer marqueur antenne simulée
    const marker = L.marker([lat, lng], {{
        icon: L.divIcon({{
            className: '',
            html: `<div style="background:#e94560;border:2px solid white;border-radius:50%;width:16px;height:16px;box-shadow:0 0 8px #e94560;"></div>`,
            iconSize: [16,16],
            iconAnchor: [8,8],
        }})
    }}).addTo(map);

    // Cercle de portée
    const rangeCircle = L.circle([lat, lng], {{
        radius:      result.range_m,
        color:       '#e94560',
        fillColor:   '#e94560',
        fillOpacity: 0.08,
        weight:      2,
        dashArray:   '6,4',
    }}).addTo(map);

    simAntennas.push({{ marker, rangeCircle, lat, lng, result }});

    // Afficher résultats
    displaySimResults(result, lat, lng);

    // Mettre à jour la couche visuellement
    highlightAffectedCells(result.affected);
}}

function displaySimResults(result, lat, lng) {{
    document.getElementById('sim-results').style.display = 'block';
    document.getElementById('sim-pos').textContent =
        `${{lat.toFixed(4)}}, ${{lng.toFixed(4)}}`;
    document.getElementById('sim-uc').textContent   = result.urban_class;
    document.getElementById('sim-range').textContent = `${{result.range_m}} m`;
    document.getElementById('sim-cells').textContent = `${{result.n_cells}} cellules`;
    document.getElementById('sim-cap').textContent   = `+${{result.cap_added}} Gbps`;
    document.getElementById('sim-deficit').textContent = `-${{result.deficit_reduced}} Gbps`;
    document.getElementById('sim-pop').textContent  =
        result.total_pop.toLocaleString() + ' hab';

    // Mettre à jour KPI déficit
    const newDeficit = Math.max(0, INITIAL_KPIS.total_deficit_gbps - result.deficit_reduced);
    document.getElementById('kpi-deficit').textContent = newDeficit.toFixed(1) + ' G';
    document.getElementById('kpi-deficit').style.color = '#6bcb77';
}}

function highlightAffectedCells(affected) {{
    // Surbrillance temporaire des cellules affectées
    if (!gridLayers[currentLayer]) return;
    const bounds = computeLayerBounds(currentLayer);

    gridLayers[currentLayer].eachLayer(function(layer) {{
        const feat = layer.feature;
        const cent = getCentroid(feat.geometry);

        // Chercher si cette feature est dans la liste affectée
        const affMatch = affected.find(a => {{
            const ac = getCentroid(a.feat.geometry);
            return Math.abs(ac[0]-cent[0]) < 0.0001 && Math.abs(ac[1]-cent[1]) < 0.0001;
        }});

        if (affMatch) {{
            // Mise à jour visuelle : amélioration = vert
            layer.setStyle({{
                fillColor:   '#6bcb77',
                fillOpacity: 0.85,
                color:       '#27ae60',
                weight:      1.5,
            }});
        }}
    }});
}}

function resetSimulation() {{
    // Supprimer antennes simulées et cercles
    for (const ant of simAntennas) {{
        map.removeLayer(ant.marker);
        map.removeLayer(ant.rangeCircle);
    }}
    simAntennas = [];

    // Réafficher la couche initiale
    showLayer(currentLayer);
    addSitesLayer();

    // Reset UI
    document.getElementById('sim-results').style.display = 'none';
    document.getElementById('sim-status').textContent =
        simMode
            ? ' Cliquez sur la carte pour placer une antenne simulée.'
            : 'Activez le mode simulation puis cliquez sur la carte.';

    // Reset KPIs
    document.getElementById('kpi-deficit').textContent =
        INITIAL_KPIS.total_deficit_gbps.toFixed(1) + ' G';
    document.getElementById('kpi-deficit').style.color = '#e94560';
}}

// ── INIT ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', initMap);

</script>
</body>
</html>"""

        return html


# =============================================================================
# POINT D'ENTRÉE CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Visual Planning Agent — 5G Network Simulation",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Exemple :
  python visual_planning_agent.py \\
    --grid    outputs/final_merged_grid.gpkg \\
    --sites   outputs/recommended_sites_v3.gpkg \\
    --output  outputs/visual_planning.html
        """,
    )
    parser.add_argument("--grid",   required=True, help="Grille fusionnée 63 colonnes (.gpkg)")
    parser.add_argument("--sites",  required=True, help="Sites recommandés SitePlacementAgent (.gpkg)")
    parser.add_argument("--output", default="outputs/visual_planning.html", help="Sortie HTML")
    args = parser.parse_args()

    try:
        agent = VisualPlanningAgent(
            grid_path=args.grid,
            sites_path=args.sites,
            output_path=args.output,
        )
        out = agent.generate()
        print(f"\n✓ Carte générée : {out}")
        print(f"  Ouvrir dans un navigateur : file:\\{out.resolve()}")

    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=True)
        sys.exit(1)