"""
agents/population_demand_agent.py
==================================
Agent d'estimation de la demande réseau 5G — v2

Aligné sur OSMnxUrbanAgent v2 (vectorisé).

Changements majeurs vs v1 :
  - Entrée : GeoDataFrame directement issu de osmnx_agent.run()
              (plus de grid_path .gpkg)
  - Colonnes OSMnx utilisées :
      urban_class, built_density, urban_intensity,
      accessibility_score, obstruction_index,
      propagation_complexity, handover_risk,
      urban_compactness, n_buildings, estimated_height_m
  - "periurban" ajouté aux demand_params (mappé sur la valeur OSMnx réelle)
  - "suburban" supprimé (n'existe pas dans OSMnx urban_class)
  - Modulation demande via urban_intensity  (amplitude urbaine réelle)
  - Modulation demande via accessibility_score (zones enclavées = -demand)
  - Modulation demande via obstruction_index  (densité verticale = +IoT/pro)
  - Fallback population vectorisé (sans raster) basé sur built_density ×
    urban_intensity × cell_area_km2 × densité de référence par classe
  - Population WorldPop toujours présente → zonal_stats inchangé

Utilisation
-----------
    from agents.osmnx_agent import OSMnxUrbanAgent
    from agents.population_demand_agent import PopulationDemandAgent
    from core.grid_generator import generate_grid_from_bounds

    bounds   = (2.345, 48.855, 2.355, 48.865)
    grid     = generate_grid_from_bounds(bounds, cell_size=200)
    osmnx    = OSMnxUrbanAgent(grid, cell_size_m=200)
    osmnx_gdf = osmnx.run()

    pop_agent = PopulationDemandAgent(
        osmnx_grid=osmnx_gdf,
        pop_raster_path="data/fra_pd_2020_1km_UNadj.tif",
    )
    enriched = pop_agent.estimate_demand()
    stats    = pop_agent.get_statistics()
    pop_agent.save("outputs/demand_grid.gpkg")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  COLONNES ATTENDUES DE L'AGENT OSMNX
# ──────────────────────────────────────────────────────────────

_REQUIRED_OSMNX_COLS = {
    "urban_class",
    "built_density",
    "urban_intensity",
    "accessibility_score",
    "obstruction_index",
}

_OPTIONAL_OSMNX_COLS = {
    "propagation_complexity",
    "handover_risk",
    "urban_compactness",
    "n_buildings",
    "estimated_height_m",
    "road_quality_score",
}

# ──────────────────────────────────────────────────────────────
#  PARAMÈTRES DE DEMANDE PAR CLASSE URBAINE OSMnx
#  (valeurs urban_class : rural, periurban, urban,
#                          dense_urban, hyper_dense)
# ──────────────────────────────────────────────────────────────

# Calibrage basé sur benchmarks opérateurs France (Orange/SFR 2023) :
# base_mbps_per_person = débit moyen effectif par habitant WorldPop (résidentiel)
# Intègre : taux pénétration mobile (85%) × taux simultanéité busy hour × débit session
# Références : ITU-T Y.3101, 3GPP TR 38.913, small cell 5G NR FR1 200m
DEMAND_PARAMS: Dict[str, Dict] = {
    "rural": {
        "base_mbps_per_person": 0.3,    # 85% × 3% simult. × 12 Mbps session
        "peak_multiplier":      1.5,
        "ref_density_per_km2":  80,     # habitants/km² de référence (fallback)
    },
    "periurban": {
        "base_mbps_per_person": 0.6,    # 85% × 5% simult. × 14 Mbps session
        "peak_multiplier":      2.0,
        "ref_density_per_km2":  800,
    },
    "urban": {
        "base_mbps_per_person": 1.0,    # 85% × 7% simult. × 17 Mbps session
        "peak_multiplier":      2.2,
        "ref_density_per_km2":  5_000,
    },
    "dense_urban": {
        "base_mbps_per_person": 1.4,    # 85% × 8% simult. × 21 Mbps session
        "peak_multiplier":      2.5,    # → ~500 Mbps base / ~1.25 Gbps pic par cellule 200m
        "ref_density_per_km2":  15_000,
    },
    "hyper_dense": {
        "base_mbps_per_person": 2.2,    # 85% × 12% simult. × 22 Mbps session
        "peak_multiplier":      2.5,    # → ~1 Gbps base / ~2.5 Gbps pic par cellule 200m
        "ref_density_per_km2":  25_000,
    },
}

# ──────────────────────────────────────────────────────────────
#  SEUILS CATÉGORIES DE DEMANDE
# ──────────────────────────────────────────────────────────────

# Seuils calibrés pour cellules 5G small cell 200m (France, 2023)
# low      : < 0.5 Gbps  — zone résidentielle légère / rurale
# medium   : 0.5-2 Gbps  — résidentiel dense / urbain standard
# high     : 2-10 Gbps   — dense_urban / hyper_dense / commercial
# critical : > 10 Gbps   — hubs transport, CBD, événements
DEMAND_CATEGORIES = [
    ("low",      0,    0.5),
    ("medium",   0.5,  2.0),
    ("high",     2.0, 10.0),
    ("critical", 10.0, 1e9),
]


def _categorize_demand(peak_gbps: np.ndarray) -> np.ndarray:
    result = np.full(len(peak_gbps), "low", dtype=object)
    for label, lo, hi in DEMAND_CATEGORIES:
        mask = (peak_gbps >= lo) & (peak_gbps < hi)
        result[mask] = label
    return result


# ──────────────────────────────────────────────────────────────
#  AGENT PRINCIPAL
# ──────────────────────────────────────────────────────────────

class PopulationDemandAgent:
    """
    Estime la demande réseau 5G par cellule à partir :
      - de la grille OSMnx enrichie (GeoDataFrame)
      - du raster WorldPop (GeoTIFF, toujours présent)

    Paramètres
    ----------
    osmnx_grid       : GeoDataFrame issu de OSMnxUrbanAgent.run()
    pop_raster_path  : chemin GeoTIFF WorldPop
    cell_size_m      : taille des cellules en mètres (utilisée pour
                       les fallbacks ; doit correspondre à l'agent OSMnx)
    """

    def __init__(
        self,
        osmnx_grid: gpd.GeoDataFrame,
        pop_raster_path: Union[str, Path],
        cell_size_m: float = 200.0,
    ) -> None:

        # ── Validation grille ────────────────────────────────
        if not isinstance(osmnx_grid, gpd.GeoDataFrame):
            raise TypeError(
                "osmnx_grid doit être un GeoDataFrame (résultat de OSMnxUrbanAgent.run()). "
                "Pour charger depuis un fichier, utilisez : "
                "gpd.read_file('votre_grille.gpkg') avant de passer à l'agent."
            )

        missing = _REQUIRED_OSMNX_COLS - set(osmnx_grid.columns)
        if missing:
            raise ValueError(
                f"Colonnes OSMnx manquantes dans la grille : {missing}\n"
                f"Assurez-vous de passer le GeoDataFrame retourné par "
                f"OSMnxUrbanAgent.run() sans modification de colonnes."
            )

        self.grid        = osmnx_grid.copy()
        self.cell_size_m = cell_size_m
        self.cell_area_m2 = cell_size_m ** 2

        # ── Raster WorldPop ──────────────────────────────────
        self.pop_raster_path = Path(pop_raster_path)
        if not self.pop_raster_path.exists():
            raise FileNotFoundError(
                f"Raster population introuvable : {self.pop_raster_path}"
            )
        self._pop_raster = rasterio.open(self.pop_raster_path)

        # ── Colonnes optionnelles manquantes → 0.0 ──────────
        for col in _OPTIONAL_OSMNX_COLS:
            if col not in self.grid.columns:
                logger.warning(
                    f"Colonne OSMnx optionnelle absente (mise à 0) : '{col}'"
                )
                self.grid[col] = 0.0

        logger.info("✓ PopulationDemandAgent v2 initialisé")
        logger.info(f"  Cellules OSMnx   : {len(self.grid)}")
        logger.info(f"  Raster WorldPop  : {self.pop_raster_path.name}")
        logger.info(
            f"  Classes urbaines : "
            f"{self.grid['urban_class'].value_counts().to_dict()}"
        )

    # ── Extraction population WorldPop ────────────────────────

    def _extract_population(self) -> pd.Series:
        """
        Zonal stats WorldPop → population par cellule.

        Le raster WorldPop encode des habitants/km² à résolution ~1 km.
        Nos cellules (200 m = 0.04 km²) sont bien plus petites que les pixels.

        Stratégie correcte :
          1. `zonal_stats` avec `all_touched=True` pour capturer tous les
             pixels qui touchent la cellule, même partiellement.
          2. On utilise `mean` (densité moyenne en hab/km²) plutôt que `sum`
             pour ne pas cumuler des valeurs de pixels entiers.
          3. On multiplie par la surface réelle de la cellule (km²) pour
             obtenir le nombre d'habitants.

        Cela évite l'artefact classique : une cellule 200 m qui touche un
        pixel 1 km² de 30 000 hab/km² se retrouvait avec 30 000 habitants
        au lieu de 30 000 × 0.04 = 1 200.
        """
        logger.info("Extraction population (zonal_stats WorldPop — mean × area)…")

        grid_for_stats = self.grid
        if self.grid.crs != self._pop_raster.crs:
            logger.info(
                f"Reprojection grille : {self.grid.crs} → {self._pop_raster.crs}"
            )
            grid_for_stats = self.grid.to_crs(self._pop_raster.crs)

        nodata = (
            self._pop_raster.nodata
            if self._pop_raster.nodata is not None
            else -9999
        )

        # mean = densité hab/km² moyenne sur les pixels couverts par la cellule
        stats = zonal_stats(
            grid_for_stats,
            str(self.pop_raster_path),
            stats=["mean"],
            nodata=nodata,
            all_touched=True,          # capture pixels partiellement couverts
        )

        # habitants = densité_hab_km2 × surface_km2
        # cell_area_km2 déjà calculé dans estimate_demand (étape 0) → pas de double calcul
        if "cell_area_km2" in self.grid.columns:
            cell_areas_km2 = self.grid["cell_area_km2"].values
        else:
            cell_areas_km2 = self._cell_areas_km2().values

        pop_values = np.array([
            (s["mean"] if (s["mean"] is not None and s["mean"] > 0) else 0.0)
            for s in stats
        ]) * cell_areas_km2

        pop = pd.Series(
            np.round(np.clip(pop_values, 0, None), 1),
            index=self.grid.index,
            name="population",
        )

        logger.info(
            f"  Population totale : {pop.sum():,.0f} hab  |  "
            f"densité moy. : {(pop.sum() / cell_areas_km2.sum()):,.0f} hab/km²"
        )
        return pop

    # ── Surface métrique par cellule ──────────────────────────

    def _cell_areas_km2(self) -> pd.Series:
        """Surface de chaque cellule en km², dans un CRS métrique."""
        try:
            utm_crs = self.grid.estimate_utm_crs()
            grid_m  = self.grid.to_crs(utm_crs)
            logger.debug(f"CRS métrique : {utm_crs}")
        except Exception:
            logger.warning("Estimation UTM échouée, fallback EPSG:3857")
            grid_m = self.grid.to_crs(epsg=3857)

        areas = grid_m.geometry.area / 1e6
        areas = areas.replace([np.inf, -np.inf], np.nan).fillna(
            self.cell_area_m2 / 1e6
        )
        return areas

    # ── Classification du type d'usage ────────────────────────

    def _classify_usage_vectorized(self) -> pd.Series:
        """
        Classification vectorisée du type d'usage dominant par cellule.

        Règles (priorité décroissante) :
          hyper_dense  + pop_density > 10 000          → commercial
          hyper_dense  + obstruction > 0.6             → mixed
          hyper_dense                                  → mixed
          dense_urban  + built_density > 0.6           → mixed
          dense_urban  + accessibility < 0.3           → residential
          dense_urban                                  → residential
          urban        + obstruction > 0.5             → mixed
          urban                                        → residential
          periurban    + built_density < 0.10          → industrial
          periurban                                    → residential
          rural                                        → residential
        """
        uc  = self.grid["urban_class"].fillna("rural")
        bd  = self.grid["built_density"].fillna(0.0)
        obs = self.grid["obstruction_index"].fillna(0.0)
        acc = self.grid["accessibility_score"].fillna(0.0)
        pd_ = self.grid.get(
            "population_density",
            pd.Series(0.0, index=self.grid.index)
        ).fillna(0.0)

        usage = pd.Series("residential", index=self.grid.index)

        # periurban industriel (peu de bâtiments = zone d'activité légère)
        mask = (uc == "periurban") & (bd < 0.10)
        usage[mask] = "industrial"

        # urban mixte (bâtiments verticaux = commercial + résidentiel)
        mask = (uc == "urban") & (obs > 0.50)
        usage[mask] = "mixed"

        # dense_urban
        mask = (uc == "dense_urban") & (bd > 0.60)
        usage[mask] = "mixed"

        # hyper_dense
        mask = uc == "hyper_dense"
        usage[mask] = "mixed"
        mask = (uc == "hyper_dense") & (obs > 0.60)
        usage[mask] = "mixed"
        mask = (uc == "hyper_dense") & (pd_ > 10_000)
        usage[mask] = "commercial"

        return usage

    # ── Modulation demande par métriques OSMnx ────────────────

    @staticmethod
    def _demand_modulation(
        urban_intensity:   np.ndarray,
        accessibility:     np.ndarray,
        obstruction:       np.ndarray,
        usage_type:        np.ndarray,
    ) -> np.ndarray:
        """
        Calcule un facteur multiplicatif de demande [0.5 … 2.0]
        basé sur les métriques OSMnx.

        Logique :
          +urban_intensity  → plus de densité d'usage → plus de trafic
          +accessibility    → meilleure connexion → plus d'usage mobile
          +obstruction      → plus d'IoT/pro en intérieur → +demande pro

        Le facteur est centré sur 1.0 (neutre) et borné à [0.5, 2.0]
        pour éviter les explosions numériques.
        """
        # Contribution de chaque métrique (delta centré sur 0)
        delta_intensity    = (urban_intensity   - 0.5) * 0.40   # ±0.20
        delta_accessibility = (accessibility    - 0.5) * 0.30   # ±0.15
        delta_obstruction  = (obstruction       - 0.5) * 0.20   # ±0.10

        # Bonus usage commercial (+15%) / industrial (-20%)
        usage_bonus = np.zeros(len(usage_type))
        usage_bonus[usage_type == "commercial"] =  0.15
        usage_bonus[usage_type == "industrial"] = -0.20
        usage_bonus[usage_type == "mixed"]      =  0.08

        factor = (
            1.0
            + delta_intensity
            + delta_accessibility
            + delta_obstruction
            + usage_bonus
        )
        return np.clip(factor, 0.5, 2.0)

    # ── Estimation principale ─────────────────────────────────

    def estimate_demand(self) -> gpd.GeoDataFrame:
        """
        Calcule toutes les métriques de demande réseau 5G.

        Colonnes ajoutées à la grille :
          population            : habitants/cellule (WorldPop)
          cell_area_km2         : surface (km²)
          population_density    : hab/km²
          usage_type            : residential / commercial / mixed / industrial
          demand_modulation     : facteur OSMnx [0.5 … 2.0]
          base_demand_mbps      : demande creuse (Mbps)
          peak_demand_mbps      : demande de pointe (Mbps)
          base_demand_gbps      : idem en Gbps
          peak_demand_gbps      : idem en Gbps
          demand_category       : low / medium / high / critical
          demand_normalized     : [0 … 1] min-max sur peak_demand_gbps

        Returns
        -------
        GeoDataFrame enrichi (même objet que self.grid, modifié sur place
        et retourné pour compatibilité pipeline).
        """
        logger.info("═══ PopulationDemandAgent v2 : estimation demande ═══")

        # ── 0. Surface métrique (UNE SEULE FOIS — réutilisée par _extract_population) ─
        self.grid["cell_area_km2"] = self._cell_areas_km2().values

        # ── 1. Population WorldPop (utilise cell_area_km2 déjà calculé) ──
        self.grid["population"] = self._extract_population().values

        # ── 2. Densité de population ─────────────────────────
        self.grid["population_density"] = np.where(
            self.grid["cell_area_km2"] > 0,
            self.grid["population"] / self.grid["cell_area_km2"],
            0.0,
        )

        logger.info(
            f"  Pop. totale   : {self.grid['population'].sum():,.0f} hab  |  "
            f"densité moy.  : {self.grid['population_density'].mean():,.0f} hab/km²"
        )

        # ── 4. Fallback population (cellules à 0 après WorldPop) ─
        # Pour les cellules sans population WorldPop mais avec bâtiments,
        # on estime la population via la densité de référence de la classe.
        zero_pop_mask = self.grid["population"] <= 0
        if zero_pop_mask.any():
            n_zero = zero_pop_mask.sum()
            logger.info(
                f"  Fallback population pour {n_zero} cellules sans données WorldPop"
            )
            ref_density = self.grid["urban_class"].map(
                {k: v["ref_density_per_km2"] for k, v in DEMAND_PARAMS.items()}
            ).fillna(DEMAND_PARAMS["rural"]["ref_density_per_km2"])

            # Modulation par built_density et urban_intensity
            pop_fallback = (
                ref_density
                * self.grid["cell_area_km2"]
                * self.grid["built_density"].clip(0, 1)
                * (0.5 + 0.5 * self.grid["urban_intensity"].clip(0, 1))
            )
            self.grid.loc[zero_pop_mask, "population"] = (
                pop_fallback[zero_pop_mask].clip(lower=0)
            )
            # Recalcul densité pour les cellules corrigées
            self.grid.loc[zero_pop_mask, "population_density"] = np.where(
                self.grid.loc[zero_pop_mask, "cell_area_km2"] > 0,
                self.grid.loc[zero_pop_mask, "population"]
                / self.grid.loc[zero_pop_mask, "cell_area_km2"],
                0.0,
            )

        # ── 5. Type d'usage ──────────────────────────────────
        logger.info("Classification type d'usage (vectorisée)…")
        self.grid["usage_type"] = self._classify_usage_vectorized().values

        usage_dist = self.grid["usage_type"].value_counts().to_dict()
        logger.info(f"  Usage : {usage_dist}")

        # ── 6. Facteur de modulation OSMnx ──────────────────
        ui  = self.grid["urban_intensity"].values.astype(float)
        acc = self.grid["accessibility_score"].values.astype(float)
        obs = self.grid["obstruction_index"].values.astype(float)
        ut  = self.grid["usage_type"].values

        modulation = self._demand_modulation(ui, acc, obs, ut)
        self.grid["demand_modulation"] = np.round(modulation, 4)

        # ── 7. Calcul vectorisé de la demande ────────────────
        urban_class = self.grid["urban_class"].fillna("rural")
        pop         = self.grid["population"].values.astype(float)

        base_mbps_per_person = urban_class.map(
            {k: v["base_mbps_per_person"] for k, v in DEMAND_PARAMS.items()}
        ).fillna(DEMAND_PARAMS["rural"]["base_mbps_per_person"]).values.astype(float)

        peak_mult = urban_class.map(
            {k: v["peak_multiplier"] for k, v in DEMAND_PARAMS.items()}
        ).fillna(DEMAND_PARAMS["rural"]["peak_multiplier"]).values.astype(float)

        # Ajustement type d'usage
        usage_factor = np.ones(len(self.grid))
        usage_factor[ut == "commercial"] = 2.0
        usage_factor[ut == "industrial"] = 0.5
        usage_factor[ut == "mixed"]      = 1.5

        base_mbps = pop * base_mbps_per_person * usage_factor * modulation
        peak_mbps = base_mbps * peak_mult

        self.grid["base_demand_mbps"] = np.round(base_mbps, 2)
        self.grid["peak_demand_mbps"] = np.round(peak_mbps, 2)
        self.grid["base_demand_gbps"] = np.round(base_mbps / 1000.0, 4)
        self.grid["peak_demand_gbps"] = np.round(peak_mbps / 1000.0, 4)

        logger.info(
            f"  Demande totale — base : {self.grid['base_demand_gbps'].sum():,.1f} Gbps  |  "
            f"pic : {self.grid['peak_demand_gbps'].sum():,.1f} Gbps"
        )

        # ── 8. Catégorie de demande ──────────────────────────
        self.grid["demand_category"] = _categorize_demand(
            self.grid["peak_demand_gbps"].values
        )

        cat_dist = self.grid["demand_category"].value_counts().to_dict()
        logger.info(f"  Catégories : {cat_dist}")

        # ── 9. Normalisation min-max ─────────────────────────
        min_v = self.grid["peak_demand_gbps"].min()
        max_v = self.grid["peak_demand_gbps"].max()
        if (max_v - min_v) > 0:
            self.grid["demand_normalized"] = np.round(
                (self.grid["peak_demand_gbps"] - min_v) / (max_v - min_v), 4
            )
        else:
            self.grid["demand_normalized"] = 0.0

        logger.info("✓ Estimation demande terminée")
        return self.grid

    # ── Statistiques ──────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne un dictionnaire structuré de statistiques.
        Doit être appelé après estimate_demand().
        """
        if "population" not in self.grid.columns:
            raise RuntimeError("Appelez d'abord estimate_demand()")

        df    = self.grid
        total = len(df)

        def _dist(col: str) -> Dict:
            counts = df[col].value_counts().to_dict()
            return {
                k: {
                    "count": int(v),
                    "pct":   round(100 * v / total, 2),
                }
                for k, v in counts.items()
            }

        return {
            "population": {
                "total":              int(df["population"].sum()),
                "mean_density_per_km2": round(float(df["population_density"].mean()), 1),
                "max_density_per_km2":  round(float(df["population_density"].max()), 1),
                "cells_with_population": int((df["population"] > 0).sum()),
            },
            "demand_global": {
                "total_base_gbps":      round(float(df["base_demand_gbps"].sum()), 2),
                "total_peak_gbps":      round(float(df["peak_demand_gbps"].sum()), 2),
                "mean_base_mbps_cell":  round(float(df["base_demand_mbps"].mean()), 1),
                "mean_peak_mbps_cell":  round(float(df["peak_demand_mbps"].mean()), 1),
                "mean_demand_normalized": round(float(df["demand_normalized"].mean()), 4),
            },
            "osmnx_modulation": {
                "mean_modulation_factor":  round(float(df["demand_modulation"].mean()), 4),
                "mean_urban_intensity":    round(float(df["urban_intensity"].mean()), 4),
                "mean_accessibility":      round(float(df["accessibility_score"].mean()), 4),
                "mean_obstruction":        round(float(df["obstruction_index"].mean()), 4),
            },
            "usage_distribution":  _dist("usage_type"),
            "demand_distribution": _dist("demand_category"),
            "urban_class_demand": {
                cls: {
                    "mean_peak_gbps": round(
                        float(df[df["urban_class"] == cls]["peak_demand_gbps"].mean())
                        if (df["urban_class"] == cls).any() else 0.0, 3
                    ),
                    "total_peak_gbps": round(
                        float(df[df["urban_class"] == cls]["peak_demand_gbps"].sum())
                        if (df["urban_class"] == cls).any() else 0.0, 3
                    ),
                }
                for cls in DEMAND_PARAMS
            },
        }

    def summary(self) -> str:
        """Rapport texte lisible, style OSMnxUrbanAgent."""
        s  = self.get_statistics()
        p  = s["population"]
        dg = s["demand_global"]
        om = s["osmnx_modulation"]

        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║   Population & Demand Agent v2 — Rapport         ║",
            "╚══════════════════════════════════════════════════╝",
            "",
            "[ 1 ] Population (WorldPop)",
            f"  Total habitants         : {p['total']:,}",
            f"  Cellules peuplées       : {p['cells_with_population']}",
            f"  Densité moy.            : {p['mean_density_per_km2']:,.0f} hab/km²",
            f"  Densité max.            : {p['max_density_per_km2']:,.0f} hab/km²",
            "",
            "[ 2 ] Demande réseau",
            f"  Demande totale (base)   : {dg['total_base_gbps']:,.1f} Gbps",
            f"  Demande totale (pic)    : {dg['total_peak_gbps']:,.1f} Gbps",
            f"  Moy./cellule (base)     : {dg['mean_base_mbps_cell']:,.0f} Mbps",
            f"  Moy./cellule (pic)      : {dg['mean_peak_mbps_cell']:,.0f} Mbps",
            "",
            "[ 3 ] Modulation OSMnx",
            f"  Facteur modul. moy.     : {om['mean_modulation_factor']:.3f}",
            f"  Urban intensity moy.    : {om['mean_urban_intensity']:.3f}",
            f"  Accessibility moy.      : {om['mean_accessibility']:.3f}",
            f"  Obstruction moy.        : {om['mean_obstruction']:.3f}",
            "",
            "[ 4 ] Types d'usage",
        ]
        for ut, info in s["usage_distribution"].items():
            lines.append(f"    {ut:<15} : {info['count']:>5} cellules ({info['pct']:.1f}%)")

        lines += ["", "[ 5 ] Catégories de demande"]
        for cat, info in s["demand_distribution"].items():
            lines.append(f"    {cat:<15} : {info['count']:>5} cellules ({info['pct']:.1f}%)")

        lines += ["", "[ 6 ] Demande pic par classe urbaine"]
        for cls, d in s["urban_class_demand"].items():
            lines.append(
                f"    {cls:<15} : moy {d['mean_peak_gbps']:.3f} Gbps  "
                f"| total {d['total_peak_gbps']:.2f} Gbps"
            )

        lines.append("")
        return "\n".join(lines)

    # ── Sauvegarde ────────────────────────────────────────────

    def save(self, output_path: Union[str, Path], driver: str = "GPKG") -> None:
        """Sauvegarde la grille enrichie (GPKG ou GeoJSON)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Nettoyage avant écriture
        grid_out = self.grid.copy()
        grid_out = grid_out.replace([np.inf, -np.inf], 0).fillna(0)

        logger.info(f"Sauvegarde → {output_path}")
        grid_out.to_file(output_path, driver=driver)
        logger.info("✓ Sauvegardé")

    def export_statistics(self, output_path: Union[str, Path]) -> None:
        """Exporte un rapport texte lisible."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.summary())

        logger.info(f"✓ Statistiques exportées → {output_path}")

    def __del__(self) -> None:
        if hasattr(self, "_pop_raster"):
            try:
                self._pop_raster.close()
            except Exception:
                pass


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
        description="Population & Demand Agent v2 — pipeline OSMnx",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Exemples :
  # Grille OSMnx pré-calculée sauvegardée en .gpkg
  python population_demand_agent.py \\
    --grid outputs/osmnx_grid.gpkg \\
    --raster data/fra_pd_2020_1km_UNadj.tif \\
    --output outputs/demand_grid.gpkg \\
    --stats  outputs/demand_stats.txt

  # Pipeline complet depuis bounds
  python population_demand_agent.py \\
    --bounds 2.345 48.855 2.355 48.865 \\
    --cell-size 200 \\
    --raster data/fra_pd_2020_1km_UNadj.tif \\
    --output outputs/demand_grid.gpkg
        """,
    )
    parser.add_argument(
        "--grid", type=str,
        help="Grille OSMnx (.gpkg) — résultat de OSMnxUrbanAgent.run() sauvegardé"
    )
    parser.add_argument(
        "--bounds", nargs=4, type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Coordonnées WGS84 pour générer + analyser la grille en une passe"
    )
    parser.add_argument("--cell-size", type=float, default=200)
    parser.add_argument("--raster",    type=str,   required=True,
                        help="Chemin GeoTIFF WorldPop")
    parser.add_argument("--output",    type=str,   default=None)
    parser.add_argument("--stats",     type=str,   default=None)
    args = parser.parse_args()

    if args.grid is None and args.bounds is None:
        parser.print_help()
        print("\n❌  Fournissez --grid ou --bounds")
        sys.exit(1)

    # Chargement / génération grille OSMnx
    if args.grid:
        print(f"Chargement grille OSMnx : {args.grid}")
        osmnx_gdf = gpd.read_file(args.grid)
        print(f"✓ {len(osmnx_gdf)} cellules")
    else:
        import numpy as np
        from shapely.geometry import box as _box

        try:
            from core.generate_grid_from_bounds import generate_grid_from_bounds
        except ImportError:
            def generate_grid_from_bounds(bounds, cell_size=200, **kw):
                gdf = gpd.GeoDataFrame(
                    {"geometry": [_box(*bounds)]}, crs="EPSG:4326"
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

        from agents.osmnx_agent import OSMnxUrbanAgent

        bounds = tuple(args.bounds)
        print(f"Génération grille — bounds: {bounds}, cell_size: {args.cell_size} m")
        raw_grid = generate_grid_from_bounds(bounds, cell_size=args.cell_size)
        osmnx_agent = OSMnxUrbanAgent(raw_grid, cell_size_m=args.cell_size)
        osmnx_gdf   = osmnx_agent.run()
        print(osmnx_agent.summary())

    # Agent demande
    agent    = PopulationDemandAgent(
        osmnx_grid=osmnx_gdf,
        pop_raster_path=args.raster,
        cell_size_m=args.cell_size,
    )
    enriched = agent.estimate_demand()

    print()
    print(agent.summary())

    if args.output:
        agent.save(args.output)
    if args.stats:
        agent.export_statistics(args.stats)

    print(f"\n✓ Terminé — {len(enriched)} cellules, {len(enriched.columns)} colonnes")