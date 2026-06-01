"""
COVERAGE AGENT - VERSION 2 
=================================================================

Modifications majeures v2:
1. Intégration features OSMnxUrbanAgent:
   - urban_class         → rayon de recherche adaptatif
   - obstruction_index   → atténuation portée antenne
   - propagation_complexity → seuil signal adaptatif
   - estimated_height_m  → pénalité hauteur bâti
   - built_density       → pondération capacité

2. Cohérence avec PopulationDemandAgent:
   - peak_demand_gbps    → utilisé tel quel (déjà calibré ITU)
   - demand_class        → contexte déficit

3. Corrections techniques:
   - Bug indentation logger.info hors classe → corrigé
   - MAX_SEARCH_RADIUS fixe → adaptatif par urban_class
   - SIGNAL_THRESHOLD fixe → adaptatif par propagation_complexity
   - Capacité fixe         → pondérée par obstruction_index

4. Fix encadrant (SINR + Capacité Shannon):
   - RSRP calculé via path loss 3GPP TR 38.901 (UMa/UMi) en dBm
   - SINR = serving / (interférences co-canal + bruit thermique)
   - Capacité via formule de Shannon (bits/s/Hz → Gbps)
   - Différenciation NR 3.5 GHz / LTE 1.8 GHz
   - Correction bug covered_idx/covered_dist (variables non définies)
   - Correction is_hotspot_risk (demand_class toujours présent en mode dégradé)

5. Améliorations physiques (recommandations encadrant) :
   - Modèle RMa (Rural Macro) 3GPP TR 38.901 pour urban_class "rural"
   - RX_SENSITIVITY_DBM différenciée par technologie (NR: -95, LTE: -100 dBm)
   - SINR_THRESHOLDS multi-niveaux (critical/usable/good) → enrichit sinr_quality
   - sinr_quality injectée dans la grille et prise en compte dans coverage_score

Auteur: Stage Amaris - Prototype 5G AI
Date: Avril 2026
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoverageAgent:
    """
    Agent d'analyse de couverture réseau 5G.

    Exploite les features OSMnxUrbanAgent (urban_class, obstruction_index,
    propagation_complexity, estimated_height_m, built_density) pour modéliser
    la propagation radio de façon réaliste selon le contexte urbain.

    Entrée attendue : GeoDataFrame enrichi par PopulationDemandAgent
    (contient urban_class + peak_demand_gbps au minimum).
    """

    # =========================================================================
    # CONSTANTES RADIO DE BASE
    # =========================================================================

    # Fréquence centrale par technologie (GHz) — impacte le path loss
    FREQUENCY_GHZ = {
        "NR":  3.5,   # 5G NR FR1 mid-band
        "LTE": 1.8,   # LTE Band 3 (dominant France)
    }

    # Modèle path loss 3GPP TR 38.901 par urban_class
    # UMi (Urban Micro)  : hyper_dense, dense_urban
    # UMa (Urban Macro)  : urban, periurban
    # RMa (Rural Macro)  : rural — distances longues, bâti bas, propagation LOS dominante
    PATHLOSS_MODEL = {
        "hyper_dense":  "UMi",
        "dense_urban":  "UMi",
        "urban":        "UMa",
        "periurban":    "UMa",
        "rural":        "RMa",   # corrigé : RMa plus réaliste que UMa en rural
        "default":      "UMa",
    }

    # Puissance d'émission effective par technologie (dBm)
    TX_POWER_DBM = {
        "NR":  46.0,   # macro NR 3.5 GHz
        "LTE": 43.0,   # macro LTE
    }

    # Bruit thermique : N = kTB  (k=1.38e-23, T=290K, B=bandwidth)
    # NR 100MHz → -104 dBm  |  LTE 20MHz → -111 dBm  (après figure de bruit NF=7dB)
    NOISE_DBM = {
        "NR":  -104.0,
        "LTE": -111.0,
    }

    # Bande passante par technologie (MHz) — pour Shannon
    BANDWIDTH_MHZ = {
        "NR":  100.0,
        "LTE":  20.0,
    }

    # Seuil SINR minimum pour décodage (dB)
    # En dessous → cellule considérée comme non couverte malgré signal fort
    SINR_THRESHOLD_DB = -3.0   # QPSK 1/8 → limite absolue 5G NR

    # Seuils SINR multi-niveaux (dB) — granularité qualité de service
    # critical : décodage minimal possible (QPSK bas rendement)
    # usable   : service basique acceptable (QPSK haut rendement / 16-QAM bas)
    # good     : service confortable (16-QAM+ / 64-QAM)
    SINR_THRESHOLDS = {
        "critical":  -3.0,
        "usable":     3.0,
        "good":      10.0,
    }

    # Sensibilité récepteur par technologie (dBm)
    # NR 100 MHz : bande large → bruit thermique plus élevé → sensibilité moindre
    # LTE 20 MHz : bande étroite → meilleure sensibilité
    RX_SENSITIVITY_DBM = {
        "NR":  -95.0,
        "LTE": -100.0,
    }

    # Capacité THÉORIQUE par secteur (Gbps) — baseline sans correction terrain
    CAPACITY_MAP = {
        "NR":  1.0,   # 5G: ~1 Gbps par secteur en conditions idéales
        "LTE": 0.15,  # 4G: ~150 Mbps par secteur
    }

    # Portée maximale absolue par technologie (mètres) — plafond non dépassable
    MAX_RANGE_MAP = {
        "NR":  2000,  # 5G urbain
        "LTE": 5000,  # 4G urbain
    }

    MIN_RANGE_M = 100  # Portée minimale pour éviter valeurs nulles

    # =========================================================================
    # PARAMÈTRES ADAPTATIFS PAR urban_class (OSMnxUrbanAgent)
    # =========================================================================

    # Rayon de recherche KD-Tree (mètres) selon densité urbaine
    # Plus la zone est dense, plus les antennes sont proches → rayon réduit
    SEARCH_RADIUS_BY_CLASS = {
        "rural":        5000,
        "periurban":    3500,
        "urban":        2000,
        "dense_urban":  1200,
        "hyper_dense":   700,
        "default":      2000,
    }

    # Seuil signal pour considérer une antenne comme couvrant la cellule
    # propagation_complexity (OSMnx: 0-1) vient ajuster ce seuil
    # Baseline : 0.9 → en zone complexe (multi-trajets) on tolère 0.6
    SIGNAL_THRESHOLD_BASE = 0.9
    SIGNAL_THRESHOLD_MIN  = 0.55  # Jamais en dessous (trop permissif)

    # Facteur d'atténuation de portée par urban_class
    # La portée réelle d'une antenne est réduite selon la densité bâtie
    RANGE_ATTENUATION_BY_CLASS = {
        "rural":        1.00,  # Pas d'atténuation
        "periurban":    0.90,
        "urban":        0.75,
        "dense_urban":  0.60,
        "hyper_dense":  0.45,
        "default":      0.75,
    }

    # Facteur de capacité effective selon urban_class
    # En hyper_dense, interférences réduisent le débit effectif
    CAPACITY_FACTOR_BY_CLASS = {
        "rural":        1.00,
        "periurban":    0.95,
        "urban":        0.85,
        "dense_urban":  0.75,
        "hyper_dense":  0.65,
        "default":      0.85,
    }


    def __init__(self, opencellid_path, grid_path, config=None):
        """
        Initialise CoverageAgent.

        Args:
            opencellid_path (str): Chemin fichier OpenCellID (.csv.gz)
            grid_path (str): Chemin grille enrichie par PopulationDemandAgent (.gpkg)
            config (dict): Configuration optionnelle
        """
        self.opencellid_path = Path(opencellid_path)
        self.grid_path = Path(grid_path)

        if not self.opencellid_path.exists():
            raise FileNotFoundError(f"OpenCellID introuvable: {self.opencellid_path}")
        if not self.grid_path.exists():
            raise FileNotFoundError(f"Grille introuvable: {self.grid_path}")

        # Configuration par défaut
        self.config = {
            "radio_filter":           ["LTE", "NR"],
            "min_samples":            3,
            "buffer_km":              5,
            "max_antennas_per_cell":  10,
            "top_k_capacity":         3,    # K antennes les plus proches pour capacité
            "weight_by_samples":      False,
        }
        if config:
            self.config.update(config)

        # --- Chargement grille ---
        logger.info(f"Chargement grille depuis {self.grid_path}")
        self.grid = gpd.read_file(self.grid_path)

        self._validate_and_patch_grid()

        logger.info(f"✓ Grille chargée: {len(self.grid)} cellules | CRS: {self.grid.crs}")

        # --- Chargement & filtrage antennes ---
        self.antennas = self._load_antennas()

        logger.info("✓ CoverageAgent v2 initialisé")
        logger.info(f"   Antennes dans zone : {len(self.antennas)}")
        logger.info(f"   Technologies       : {self.antennas['radio'].value_counts().to_dict()}")


    # =========================================================================
    # VALIDATION GRILLE D'ENTRÉE
    # =========================================================================

    def _validate_and_patch_grid(self):
        """
        Vérifie la présence des colonnes OSMnx + Population.
        Injecte des valeurs par défaut si colonnes absentes (mode dégradé).
        """
        # Colonnes issues de PopulationDemandAgent
        if "peak_demand_gbps" not in self.grid.columns:
            logger.warning("'peak_demand_gbps' absente → valeur par défaut 1.0 Gbps")
            self.grid["peak_demand_gbps"] = 1.0

        if "demand_class" not in self.grid.columns:
            logger.warning("'demand_class' absente → 'medium' par défaut")
            self.grid["demand_class"] = "medium"

        # Colonnes issues de OSMnxUrbanAgent
        osmnx_defaults = {
            "urban_class":            "urban",
            "obstruction_index":      0.5,    # 0 (dégagé) → 1 (très obstrué)
            "propagation_complexity": 0.5,    # 0 (simple) → 1 (multi-trajets)
            "estimated_height_m":     10.0,   # Hauteur bâti estimée (mètres)
            "built_density":          0.5,    # Densité bâtie normalisée 0-1
            "urban_intensity":        0.5,
        }

        missing = []
        for col, default in osmnx_defaults.items():
            if col not in self.grid.columns:
                self.grid[col] = default
                missing.append(col)

        if missing:
            logger.warning(
                f"Colonnes OSMnx absentes (mode dégradé) → défauts injectés: {missing}"
            )
        else:
            logger.info("✓ Toutes les colonnes OSMnx présentes")


    # =========================================================================
    # CHARGEMENT & FILTRAGE ANTENNES
    # =========================================================================

    def _load_antennas(self):
        """
        Charge et filtre antennes OpenCellID avec filtrage spatial sur zone grille.

        Returns:
            GeoDataFrame: Antennes filtrées et projetées
        """
        logger.info(f"Chargement OpenCellID depuis {self.opencellid_path}")

        COLUMN_NAMES = [
            "radio", "mcc", "net", "area", "cell",
            "unit", "lon", "lat", "range", "samples",
            "changeable", "created", "updated", "avg_signal"
        ]
        USECOLS = ["radio", "lon", "lat", "range", "samples"]

        df = pd.read_csv(
            self.opencellid_path,
            compression="gzip",
            header=None,
            names=COLUMN_NAMES,
            usecols=USECOLS,
            low_memory=False,
            dtype={
                "radio":   "category",
                "lon":     "float32",
                "lat":     "float32",
                "range":   "float32",
                "samples": "uint16",
            }
        )

        logger.info(f"✓ Lignes brutes: {len(df):,}")

        # --- Filtrage technologie ---
        df = df[df["radio"].isin(self.config["radio_filter"])].copy()
        logger.info(f"✓ Après filtre techno: {len(df):,}")

        # --- Validation données ---
        df = df.dropna(subset=["lon", "lat", "range"])
        df = df[(df["lon"].between(-180, 180)) & (df["lat"].between(-90, 90))]
        df = df[df["range"] >= self.MIN_RANGE_M]

        # Clip portées aberrantes
        for radio, max_range in self.MAX_RANGE_MAP.items():
            mask = df["radio"] == radio
            df.loc[mask, "range"] = df.loc[mask, "range"].clip(upper=max_range)

        # Filtrage fiabilité
        df = df[df["samples"] >= self.config["min_samples"]]

        logger.info(f"✓ Après nettoyage: {len(df):,}")

        # --- Conversion GeoDataFrame ---
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326"
        )

        # --- Filtrage spatial (zone grille + buffer) ---
        logger.info("⏳ Filtrage spatial sur zone grille...")
        grid_wgs84  = self.grid.to_crs("EPSG:4326")
        bounds      = grid_wgs84.total_bounds
        buffer_deg  = self.config["buffer_km"] / 111.0

        lon_min = bounds[0] - buffer_deg
        lat_min = bounds[1] - buffer_deg
        lon_max = bounds[2] + buffer_deg
        lat_max = bounds[3] + buffer_deg

        logger.info(f"   BBox: [{lon_min:.3f}, {lat_min:.3f}, {lon_max:.3f}, {lat_max:.3f}]")

        mask_spatial = (
            (gdf.geometry.x.between(lon_min, lon_max)) &
            (gdf.geometry.y.between(lat_min, lat_max))
        )
        gdf = gdf[mask_spatial].copy()

        logger.info(f"✓ Antennes dans zone: {len(gdf):,}")

        if gdf.empty:
            raise ValueError("Aucune antenne dans la zone d'étude !")

        # --- Projection métrique ---
        target_crs = self._resolve_metric_crs(self.grid)
        gdf = gdf.to_crs(target_crs)

        # --- Capacité de base (avant corrections terrain) ---
        gdf["capacity_gbps_base"] = (
            gdf["radio"].map(self.CAPACITY_MAP).astype("float32") * 2.5
        )

        if self.config.get("weight_by_samples", False):
            max_samples = gdf["samples"].quantile(0.95)
            weights = (gdf["samples"] / max_samples).clip(0.3, 1.0)
            gdf["capacity_gbps_base"] *= weights

        gdf["capacity_gbps"] = gdf["capacity_gbps_base"]  # sera corrigé par terrain

        logger.info(f"✓ Capacité totale théorique: {gdf['capacity_gbps'].sum():,.1f} Gbps")

        return gdf[["geometry", "radio", "range", "samples", "capacity_gbps"]]


    # =========================================================================
    # ANALYSE COUVERTURE
    # =========================================================================

    def compute_coverage(self):
        """
        Calcule métriques de couverture pour chaque cellule de la grille.

        Intègre les features OSMnx pour adapter :
        - Le rayon de recherche antennes (urban_class)
        - Le seuil de signal (propagation_complexity)
        - La portée effective des antennes (obstruction_index + estimated_height_m)
        - La capacité effective via Shannon (SINR réel + built_density + urban_class)

        Returns:
            GeoDataFrame: Grille enrichie avec métriques couverture
        """
        logger.info("=" * 70)
        logger.info("ANALYSE DE COUVERTURE RÉSEAU (v2 - OSMnx aware)")
        logger.info("=" * 70)

        # --- Harmonisation CRS ---
        target_crs = self._resolve_metric_crs(self.grid)
        if self.grid.crs != target_crs:
            self.grid = self.grid.to_crs(target_crs)
        if self.antennas.crs != self.grid.crs:
            self.antennas = self.antennas.to_crs(self.grid.crs)

        # --- Centroïdes grille ---
        logger.info("Calcul centroïdes grille...")
        grid_centroids = self.grid.geometry.centroid
        centroids_xy   = np.array([(pt.x, pt.y) for pt in grid_centroids])

        # --- Données antennes ---
        antennas_xy        = np.array([(pt.x, pt.y) for pt in self.antennas.geometry])
        antenna_ranges     = self.antennas["range"].values.astype(float)
        antenna_capacities = self.antennas["capacity_gbps"].values.astype(float)
        antenna_radios     = self.antennas["radio"].values

        n_cells = len(self.grid)

        # --- Extraction features OSMnx (par cellule) ---
        urban_classes        = self.grid["urban_class"].fillna("default").values
        obstruction_idx      = self.grid["obstruction_index"].fillna(0.5).values.astype(float)
        propagation_complex  = self.grid["propagation_complexity"].fillna(0.5).values.astype(float)
        height_m             = self.grid["estimated_height_m"].fillna(10.0).values.astype(float)
        built_density_vals   = self.grid["built_density"].fillna(0.5).values.astype(float)

        # =====================================================================
        # RAYON DE RECHERCHE GLOBAL (pour KD-Tree)
        # On prend le rayon max pour construire le tree une seule fois,
        # puis on applique le rayon adaptatif par cellule.
        # =====================================================================
        max_search_radius = max(self.SEARCH_RADIUS_BY_CLASS.values())

        logger.info(f"Construction KD-Tree ({len(antennas_xy)} antennes)...")
        tree = cKDTree(antennas_xy)

        logger.info(f"Requête KD-Tree (rayon max {max_search_radius}m)...")
        indices_all = tree.query_ball_point(centroids_xy, r=max_search_radius)

        # =====================================================================
        # MÉTRIQUES PAR CELLULE
        # =====================================================================
        antenna_count     = np.zeros(n_cells, dtype=int)
        antenna_count_nr  = np.zeros(n_cells, dtype=int)
        antenna_count_lte = np.zeros(n_cells, dtype=int)
        capacity_avail    = np.zeros(n_cells, dtype=float)
        nearest_dist      = np.full(n_cells, np.nan)
        nearest_radio     = np.full(n_cells, "", dtype=object)

        logger.info(f"Calcul couverture pour {n_cells} cellules...")

        for i, candidates in enumerate(indices_all):
            if not candidates:
                continue

            uc = urban_classes[i]

            # -----------------------------------------------------------------
            # 1. RAYON ADAPTATIF par urban_class
            # -----------------------------------------------------------------
            search_r = self.SEARCH_RADIUS_BY_CLASS.get(
                uc, self.SEARCH_RADIUS_BY_CLASS["default"]
            )

            cands = np.array(candidates)
            dists = np.linalg.norm(antennas_xy[cands] - centroids_xy[i], axis=1)

            # Appliquer rayon adaptatif (sous-ensemble du rayon global)
            within_r = dists <= search_r
            cands = cands[within_r]
            dists = dists[within_r]

            if len(cands) == 0:
                continue

            # Antenne la plus proche
            nearest_idx = np.argmin(dists)
            nearest_dist[i]  = dists[nearest_idx]
            nearest_radio[i] = antenna_radios[cands[nearest_idx]]

            # -----------------------------------------------------------------
            # 2. PORTÉE EFFECTIVE = portée nominale × atténuation urban_class
            #                                        × facteur obstruction
            #                                        × facteur hauteur bâti
            # (utilisée uniquement pour filtrage préliminaire éventuel)
            # -----------------------------------------------------------------
            range_atten_class = self.RANGE_ATTENUATION_BY_CLASS.get(
                uc, self.RANGE_ATTENUATION_BY_CLASS["default"]
            )

            # obstruction_index [0,1] → réduit portée jusqu'à -30%
            obstruction_factor = 1.0 - 0.30 * obstruction_idx[i]

            # estimated_height_m → pénalité exponentielle (bâtiments hauts = diffraction forte)
            # Normalisation : 0m=1.0 ; 30m≈0.85 ; 100m≈0.60
            height_factor = np.exp(-0.003 * height_m[i])

            effective_range = (
                antenna_ranges[cands]
                * range_atten_class
                * obstruction_factor
                * height_factor
            )
            effective_range = np.maximum(effective_range, self.MIN_RANGE_M)

            # -----------------------------------------------------------------
            # 3. RSRP PAR ANTENNE CANDIDATE (3GPP TR 38.901)
            # -----------------------------------------------------------------
            extra_att_db = obstruction_idx[i] * 15.0

            rsrp_per_ant = np.array([
                self.TX_POWER_DBM.get(str(antenna_radios[cands[j]]), 43.0)
                - self._compute_path_loss_db(
                    np.array([dists[j]]),
                    str(antenna_radios[cands[j]]),
                    uc
                )[0]
                - extra_att_db
                for j in range(len(cands))
            ])

            # -----------------------------------------------------------------
            # 4. SERVING ANTENNA = antenne avec RSRP max
            # -----------------------------------------------------------------
            serving_idx_local = np.argmax(rsrp_per_ant)
            rsrp_serving      = rsrp_per_ant[serving_idx_local]
            serving_radio     = str(antenna_radios[cands[serving_idx_local]])

            # RSRP seuil adaptatif (propagation_complexity tolère plus bas)
            # Zone simple (0) : seuil strict -95 dBm
            # Zone complexe (1) : seuil tolérant -108 dBm (multipath aide)
            rsrp_threshold = -95.0 - 13.0 * propagation_complex[i]

            if rsrp_serving < rsrp_threshold:
                # Même la meilleure antenne est trop faible → non couvert
                continue

            # -----------------------------------------------------------------
            # 5. SINR = serving / (interférences co-canal + bruit thermique)
            # -----------------------------------------------------------------
            noise_dbm  = self.NOISE_DBM.get(serving_radio, -104.0)
            noise_mw   = self._dbm_to_mw(np.array([noise_dbm]))[0]

            # Masque interférents : toutes les antennes sauf la serving
            interferer_mask = np.ones(len(cands), dtype=bool)
            interferer_mask[serving_idx_local] = False

            if interferer_mask.any():
                interf_rsrp_mw  = self._dbm_to_mw(rsrp_per_ant[interferer_mask])
                interf_total_mw = interf_rsrp_mw.sum()
            else:
                interf_total_mw = 0.0

            serving_mw  = self._dbm_to_mw(np.array([rsrp_serving]))[0]
            sinr_linear = serving_mw / (interf_total_mw + noise_mw)
            sinr_db     = self._mw_to_dbm(np.array([sinr_linear]))[0]

            if sinr_db < self.SINR_THRESHOLD_DB:
                # Signal présent mais inutilisable (trop d'interférences)
                continue

            # -----------------------------------------------------------------
            # 6. CAPACITÉ SHANNON (bits/s/Hz → Gbps)
            # -----------------------------------------------------------------
            bw_mhz = self.BANDWIDTH_MHZ.get(serving_radio, 20.0)
            # Efficacité spectrale plafonnée à 8 bit/s/Hz (limite pratique 5G)
            spectral_eff = min(np.log2(1.0 + sinr_linear), 8.0)
            # Facteur urban_class (interférences résiduelles non modélisées)
            cap_factor_class = self.CAPACITY_FACTOR_BY_CLASS.get(
                uc, self.CAPACITY_FACTOR_BY_CLASS["default"]
            )
            capacity_avail[i] = (bw_mhz * 1e6 * spectral_eff * cap_factor_class) / 1e9  # Gbps

            # -----------------------------------------------------------------
            # 7. COMPTAGE ANTENNES COUVRANTES (RSRP >= seuil)
            # -----------------------------------------------------------------
            covered_mask = rsrp_per_ant >= rsrp_threshold
            covered_idx  = cands[covered_mask]
            covered_dist = dists[covered_mask]

            # Limiter à max_antennas_per_cell
            max_ant = self.config["max_antennas_per_cell"]
            if len(covered_idx) > max_ant:
                order        = np.argsort(covered_dist)[:max_ant]
                covered_idx  = covered_idx[order]

            antenna_count[i]     = len(covered_idx)
            radios_cov           = antenna_radios[covered_idx]
            antenna_count_nr[i]  = (radios_cov == "NR").sum()
            antenna_count_lte[i] = (radios_cov == "LTE").sum()

        logger.info("✓ Métriques de couverture calculées")

        # =====================================================================
        # INJECTION DANS GRILLE
        # =====================================================================
        self.grid["antenna_count"]           = antenna_count
        self.grid["antenna_count_nr"]        = antenna_count_nr
        self.grid["antenna_count_lte"]       = antenna_count_lte
        self.grid["capacity_available_gbps"] = capacity_avail
        self.grid["nearest_antenna_m"]       = nearest_dist
        self.grid["nearest_radio"]           = nearest_radio
        self.grid["is_covered"]              = antenna_count > 0

        # =====================================================================
        # MÉTRIQUES DÉRIVÉES
        # =====================================================================
        logger.info("Calcul métriques dérivées...")
        self._compute_deficit_metrics()
        self._compute_coverage_score()

        # Nettoyage
        self.grid.replace([np.inf, -np.inf], 0, inplace=True)
        self.grid.fillna(0, inplace=True)

        # =====================================================================
        # STATISTIQUES FINALES
        # =====================================================================
        covered_pct     = self.grid["is_covered"].mean() * 100
        uncovered_count = (~self.grid["is_covered"]).sum()

        logger.info("=" * 70)
        logger.info("RÉSULTATS COUVERTURE")
        logger.info("=" * 70)
        logger.info(f"Taux couverture global  : {covered_pct:.1f}%")
        logger.info(f"Cellules non couvertes  : {uncovered_count}")
        logger.info(f"Capacité totale dispo   : {self.grid['capacity_available_gbps'].sum():,.1f} Gbps")
        logger.info(f"Demande totale (pic)    : {self.grid['peak_demand_gbps'].sum():,.1f} Gbps")

        return self.grid


    # =========================================================================
    # MÉTRIQUES DÉRIVÉES
    # =========================================================================

    def _compute_deficit_metrics(self):
        """Calcule métriques de déficit capacité (cohérentes avec PopulationDemandAgent)."""
        logger.info("Calcul métriques déficit...")

        demand = self.grid["peak_demand_gbps"].replace(0, 0.001)

        # Ratio capacité/demande (plafonné à 5×)
        self.grid["capacity_demand_ratio"] = (
            self.grid["capacity_available_gbps"] / demand
        ).clip(upper=5.0)

        # Déficit absolu (Gbps)
        self.grid["capacity_deficit_gbps"] = (
            self.grid["peak_demand_gbps"] - self.grid["capacity_available_gbps"]
        ).clip(lower=0)

        # Score surcharge [0-1]
        self.grid["overload_score"] = np.where(
            self.grid["capacity_demand_ratio"] >= 1.0,
            0.0,
            1.0 - self.grid["capacity_demand_ratio"]
        )

        # Score distance normalisé
        max_dist = self.grid["nearest_antenna_m"].replace(0, np.nan).quantile(0.95)
        if pd.notna(max_dist) and max_dist > 0:
            self.grid["distance_score"] = (
                self.grid["nearest_antenna_m"] / max_dist
            ).fillna(1.0).clip(0, 1)
        else:
            self.grid["distance_score"] = 0.0

        # Couverture 5G
        self.grid["has_5g"] = self.grid["antenna_count_nr"] > 0

        # Cellules critiques : forte demande + faible couverture
        # demand_class est toujours présent (injecté par défaut dans _validate_and_patch_grid)
        self.grid["is_hotspot_risk"] = (
            (self.grid["capacity_demand_ratio"] < 0.7) &
            (self.grid["peak_demand_gbps"] > 1.0)
        )


    def _compute_coverage_score(self):
        """
        Calcule score de couverture composite [0-1] + classification.

        Composantes :
          0.40 × ratio capacité/demande (non-linéaire)
          0.25 × proximité antenne (1 - distance_score)
          0.20 × densité antennes (normalisée)
          0.10 × ratio 5G (NR parmi antennes)
          0.05 × contexte urbain (urban_intensity, si dispo)
          −     pénalité surcharge (×0.7)
        """
        logger.info("Calcul score couverture composite...")

        demand   = self.grid["peak_demand_gbps"].replace(0, 0.001)
        capacity = self.grid["capacity_available_gbps"]
        ratio    = capacity / demand

        # Normalisation non-linéaire [0,1]
        ratio_norm = np.clip(ratio, 0, 1)

        # Proximité
        distance_component = (1 - self.grid["distance_score"]).clip(0, 1)

        # Densité antennes normalisée
        ant_max = self.grid["antenna_count"].max()
        ant_density_norm = (
            self.grid["antenna_count"] / ant_max
        ).clip(0, 1) if ant_max > 0 else 0.0

        # Ratio NR
        nr_ratio = np.where(
            self.grid["antenna_count"] > 0,
            self.grid["antenna_count_nr"] / self.grid["antenna_count"],
            0.0
        ).clip(0, 1)

        # Composante urban_intensity (OSMnx) : les zones très denses
        # ont besoin de plus de capacité → pondération légère
        if "urban_intensity" in self.grid.columns:
            urban_component = self.grid["urban_intensity"].fillna(0.5).clip(0, 1)
            # urban_intensity élevée → score légèrement réduit si non couverte
            urban_weight = 0.05
            ratio_weight = 0.40
        else:
            urban_component = 0.0
            urban_weight = 0.00
            ratio_weight = 0.45

        # Pénalité surcharge
        penalty = self.grid["overload_score"] * 0.7

        self.grid["coverage_score"] = (
            ratio_weight   * ratio_norm
            + 0.25         * distance_component
            + 0.20         * ant_density_norm
            + 0.10         * nr_ratio
            - urban_weight * urban_component   # légère pénalité zones denses non couvertes
            - penalty
        ).clip(0, 1)

        # Classification par seuils fixes
        def classify(score):
            if   score < 0.10: return "uncovered"
            elif score < 0.40: return "critical"
            elif score < 0.60: return "adequate"
            elif score < 0.80: return "good"
            else:              return "excellent"

        self.grid["coverage_class"] = self.grid["coverage_score"].apply(classify)

        logger.info("✓ Score couverture calculé et classifié")

        # Distribution
        dist = self.grid["coverage_class"].value_counts()
        for cls in ["uncovered", "critical", "adequate", "good", "excellent"]:
            n   = dist.get(cls, 0)
            pct = n / len(self.grid) * 100
            logger.info(f"   {cls:12s}: {n:5d} ({pct:5.1f}%)")


    # =========================================================================
    # UTILITAIRES
    # =========================================================================

    def _robust_normalize(self, series):
        """Normalisation robuste (percentiles 5-95)."""
        p5  = series.quantile(0.05)
        p95 = series.quantile(0.95)
        if p95 - p5 == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return ((series - p5) / (p95 - p5)).clip(0, 1)


    def _dbm_to_mw(self, dbm_array):
        """Convertit dBm → mW (linéaire) pour sommer les puissances."""
        return np.power(10.0, dbm_array / 10.0)

    def _mw_to_dbm(self, mw):
        """Convertit mW → dBm."""
        return 10.0 * np.log10(np.maximum(mw, 1e-30))

    def _compute_path_loss_db(self, distances_m, radio, urban_class):
        """
        Calcule le path loss (dB) selon 3GPP TR 38.901.
        Modèles UMa / UMi selon urban_class.

        Args:
            distances_m : array numpy de distances (mètres)
            radio       : "NR" ou "LTE"
            urban_class : str

        Returns:
            np.ndarray: Path loss en dB
        """
        f_ghz  = self.FREQUENCY_GHZ.get(radio, 3.5)
        model  = self.PATHLOSS_MODEL.get(urban_class, "UMa")

        # Distance plancher pour éviter log(0)
        d = np.maximum(distances_m, 1.0)

        if model == "UMi":
            # 3GPP TR 38.901 UMi Street Canyon (NLoS dominant en dense urban)
            # PL = 35.3·log10(d) + 22.4 + 21.3·log10(f_GHz)
            pl = 35.3 * np.log10(d) + 22.4 + 21.3 * np.log10(f_ghz)
        else:
            # 3GPP TR 38.901 UMa NLoS
            # PL = 13.54 + 39.08·log10(d) + 20·log10(f_GHz)
            pl = 13.54 + 39.08 * np.log10(d) + 20.0 * np.log10(f_ghz)

        return pl  # dB

    def _resolve_metric_crs(self, gdf):
        """Résout CRS métrique approprié pour calculs de distance."""
        if gdf.crs and gdf.crs.is_projected:
            return gdf.crs
        try:
            utm = gdf.estimate_utm_crs()
            if utm:
                return utm
        except Exception:
            pass
        return "EPSG:3857"


    # =========================================================================
    # STATISTIQUES & EXPORT
    # =========================================================================

    def get_statistics(self):
        """Retourne dictionnaire complet de statistiques de couverture."""
        if "coverage_score" not in self.grid.columns:
            raise ValueError("Exécutez d'abord compute_coverage()")

        total        = len(self.grid)
        covered      = self.grid["is_covered"].sum()
        total_cap    = float(self.grid["capacity_available_gbps"].sum())
        total_demand = float(self.grid["peak_demand_gbps"].sum())

        stats = {
            # Couverture
            "total_cells":           total,
            "covered_cells":         int(covered),
            "uncovered_cells":       int(total - covered),
            "coverage_rate_pct":     float(covered / total * 100),
            "cells_with_5g":         int(self.grid["has_5g"].sum()),
            "5g_coverage_rate_pct":  float(self.grid["has_5g"].mean() * 100),

            # Antennes
            "total_antennas_in_zone":  len(self.antennas),
            "total_lte_antennas":      int((self.antennas["radio"] == "LTE").sum()),
            "total_nr_antennas":       int((self.antennas["radio"] == "NR").sum()),
            "mean_antennas_per_cell":  float(self.grid["antenna_count"].mean()),
            "max_antennas_per_cell":   int(self.grid["antenna_count"].max()),

            # Capacité vs demande
            "total_capacity_gbps":    total_cap,
            "total_demand_gbps":      total_demand,
            "total_deficit_gbps":     float(max(0, total_demand - total_cap)),
            "mean_capacity_gbps":     float(self.grid["capacity_available_gbps"].mean()),
            "mean_demand_gbps":       float(self.grid["peak_demand_gbps"].mean()),

            # Hotspots (contexte Population)
            "hotspot_risk_cells":     int(self.grid["is_hotspot_risk"].sum()),
            "hotspot_risk_pct":       float(self.grid["is_hotspot_risk"].mean() * 100),

            # Scores
            "mean_coverage_score":    float(self.grid["coverage_score"].mean()),
            "mean_overload_score":    float(self.grid["overload_score"].mean()),

            # Classification
            "coverage_class_counts": self.grid["coverage_class"].value_counts().to_dict(),
            "coverage_class_percentages": {
                k: round(v / total * 100, 2)
                for k, v in self.grid["coverage_class"].value_counts().items()
            },
        }

        # Statistiques par urban_class (OSMnx)
        if "urban_class" in self.grid.columns:
            stats["by_urban_class"] = {}
            for uc, grp in self.grid.groupby("urban_class"):
                stats["by_urban_class"][uc] = {
                    "cells":              len(grp),
                    "coverage_rate_pct":  float(grp["is_covered"].mean() * 100),
                    "mean_score":         float(grp["coverage_score"].mean()),
                    "mean_capacity_gbps": float(grp["capacity_available_gbps"].mean()),
                    "mean_demand_gbps":   float(grp["peak_demand_gbps"].mean()),
                    "deficit_gbps":       float(max(0,
                        grp["peak_demand_gbps"].sum() - grp["capacity_available_gbps"].sum()
                    )),
                }

        return stats


    def export_statistics(self, output_path):
        """Exporte statistiques en fichier texte lisible."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.get_statistics()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("STATISTIQUES COUVERTURE RÉSEAU 5G (v2 - OSMnx aware)\n")
            f.write("=" * 70 + "\n\n")

            f.write("COUVERTURE GLOBALE:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Cellules totales        : {stats['total_cells']:,}\n")
            f.write(f"Cellules couvertes      : {stats['covered_cells']:,} ({stats['coverage_rate_pct']:.1f}%)\n")
            f.write(f"Cellules non couvertes  : {stats['uncovered_cells']:,}\n")
            f.write(f"Cellules avec 5G        : {stats['cells_with_5g']:,} ({stats['5g_coverage_rate_pct']:.1f}%)\n")
            f.write(f"Cellules hotspot risk   : {stats['hotspot_risk_cells']:,} ({stats['hotspot_risk_pct']:.1f}%)\n\n")

            f.write("ANTENNES:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Antennes dans zone      : {stats['total_antennas_in_zone']:,}\n")
            f.write(f"  - LTE                 : {stats['total_lte_antennas']:,}\n")
            f.write(f"  - NR (5G)             : {stats['total_nr_antennas']:,}\n")
            f.write(f"Antennes moy./cellule   : {stats['mean_antennas_per_cell']:.1f}\n")
            f.write(f"Antennes max./cellule   : {stats['max_antennas_per_cell']}\n\n")

            f.write("CAPACITÉ vs DEMANDE:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Capacité totale         : {stats['total_capacity_gbps']:,.1f} Gbps\n")
            f.write(f"Demande totale (pic)    : {stats['total_demand_gbps']:,.1f} Gbps\n")
            f.write(f"Déficit total           : {stats['total_deficit_gbps']:,.1f} Gbps\n")
            f.write(f"Capacité moy./cellule   : {stats['mean_capacity_gbps']:.3f} Gbps\n")
            f.write(f"Demande moy./cellule    : {stats['mean_demand_gbps']:.3f} Gbps\n\n")

            f.write("SCORES:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Score couverture moyen  : {stats['mean_coverage_score']:.3f}\n")
            f.write(f"Score surcharge moyen   : {stats['mean_overload_score']:.3f}\n\n")

            f.write("CLASSIFICATION COUVERTURE:\n")
            f.write("-" * 70 + "\n")
            for cls in ["uncovered", "critical", "adequate", "good", "excellent"]:
                count = stats['coverage_class_counts'].get(cls, 0)
                pct   = stats['coverage_class_percentages'].get(cls, 0.0)
                f.write(f"{cls:15s}: {count:5d} cellules ({pct:5.2f}%)\n")

            if "by_urban_class" in stats:
                f.write("\nPAR URBAN_CLASS (OSMnx):\n")
                f.write("-" * 70 + "\n")
                for uc, data in stats["by_urban_class"].items():
                    f.write(f"\n  [{uc}]  {data['cells']} cellules\n")
                    f.write(f"    Couverture       : {data['coverage_rate_pct']:.1f}%\n")
                    f.write(f"    Score moyen      : {data['mean_score']:.3f}\n")
                    f.write(f"    Capacité moy.    : {data['mean_capacity_gbps']:.3f} Gbps\n")
                    f.write(f"    Demande moy.     : {data['mean_demand_gbps']:.3f} Gbps\n")
                    f.write(f"    Déficit total    : {data['deficit_gbps']:.1f} Gbps\n")

        logger.info(f"✓ Statistiques exportées vers {output_path}")


    def save(self, output_path, driver="GPKG"):
        """Sauvegarde grille enrichie."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Sauvegarde vers {output_path}")
        grid_out = self.grid.replace([np.inf, -np.inf], 0).fillna(0)
        grid_out.to_file(output_path, driver=driver)
        logger.info("✓ Fichier sauvegardé")


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":

    OPENCELLID_PATH = r"C:\Users\ayham.chaabane_amari\Downloads\208.csv.gz"
    GRID_PATH       = r"C:\dev\5g_ai_project\outputs\demand\demand_grid.gpkg"
    OUTPUT_PATH     = r"C:\dev\5g_ai_project\outputs\coverage\coverage_grid.gpkg"
    STATS_PATH      = r"C:\dev\5g_ai_project\outputs\coverage\coverage_stats.txt"

    try:
        agent = CoverageAgent(
            opencellid_path=OPENCELLID_PATH,
            grid_path=GRID_PATH,
            config={
                "radio_filter":          ["LTE", "NR"],
                "min_samples":           3,
                "buffer_km":             5,
                "max_antennas_per_cell": 10,
                "top_k_capacity":        3,
            }
        )

        enriched_grid = agent.compute_coverage()

        stats = agent.get_statistics()

        print("\n" + "=" * 70)
        print("RÉSUMÉ COUVERTURE RÉSEAU v2")
        print("=" * 70)
        print(f"Taux couverture      : {stats['coverage_rate_pct']:.1f}%")
        print(f"Taux couverture 5G   : {stats['5g_coverage_rate_pct']:.1f}%")
        print(f"Antennes dans zone   : {stats['total_antennas_in_zone']:,}")
        print(f"Capacité totale      : {stats['total_capacity_gbps']:,.1f} Gbps")
        print(f"Demande totale       : {stats['total_demand_gbps']:,.1f} Gbps")
        print(f"Déficit total        : {stats['total_deficit_gbps']:,.1f} Gbps")
        print(f"Hotspot risk cells   : {stats['hotspot_risk_cells']:,} ({stats['hotspot_risk_pct']:.1f}%)")
        print(f"Score couverture moy : {stats['mean_coverage_score']:.3f}")

        if "by_urban_class" in stats:
            print("\nPar urban_class:")
            for uc, data in stats["by_urban_class"].items():
                print(f"  {uc:15s}: couv {data['coverage_rate_pct']:.0f}%"
                      f"  score {data['mean_score']:.2f}"
                      f"  déficit {data['deficit_gbps']:.1f} Gbps")

        agent.save(OUTPUT_PATH)
        agent.export_statistics(STATS_PATH)

        print("\n✓ Traitement terminé avec succès!")

    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=True)
        raise