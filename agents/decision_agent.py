import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PARAMÈTRES GLOBAUX
# ─────────────────────────────────────────────

# Distance minimale inter-sites BASE par urban_class (mètres)
# La distance réelle est modulée par LOS et atténuation (voir _greedy_distance_selection)
MIN_DISTANCE_M = {
    "hyper_dense":  200,
    "dense_urban":  350,
    "urban":        500,
    "periurban":    900,
    "rural":       2000,
}

# Densité maximale absolue en sites/km² par urban_class
MAX_DENSITY_PER_KM2 = {
    "hyper_dense":  5.0,
    "dense_urban":  3.0,
    "urban":        2.0,
    "periurban":    0.8,
    "rural":        0.3,
}

# Ratio max de cellules converties en sites (garde-fou secondaire)
MAX_RATIO = {
    "hyper_dense":  0.30,
    "dense_urban":  0.22,
    "urban":        0.35,
    "periurban":    0.60,
    "rural":        0.80,
}

# eps DBSCAN par urban_class (mètres) — conservé pour compatibilité future
DBSCAN_EPS_M = {
    "hyper_dense":  300,
    "dense_urban":  700,
    "urban":        800,
    "periurban":   1200,
    "rural":       2500,
}

# Seuil score composite minimum pour être candidat
SCORE_THRESHOLD = {
    "hyper_dense":  0.40,
    "dense_urban":  0.30,
    "urban":        0.20,
    "periurban":    0.10,
    "rural":        0.05,
}

# Types de sites de base par urban_class (peut être surchargé par les règles radio)
SITE_TYPE_MAP = {
    "hyper_dense":  "small_cell_dense",
    "dense_urban":  "micro_cell",
    "urban":        "macro_cell",
    "periurban":    "macro_cell",
    "rural":        "macro_cell_large",
}

# Poids du score composite v4
# Somme = 1.0 (propagation_score regroupe 3 composantes radio)
SCORE_WEIGHTS = {
    "capacity_deficit":  0.30,   # -5% vs v3 pour faire de la place au radio
    "coverage_gap":      0.25,   # -5% vs v3
    "demand":            0.15,   # -5% vs v3
    "urban_intensity":   0.10,   # -5% vs v3
    "propagation":       0.20,   # NOUVEAU — physique radio
}

# Poids internes du propagation_score
PROPAGATION_WEIGHTS = {
    "los":         0.45,   # LOS = facteur dominant (visibilité directe)
    "attenuation": 0.35,   # Atténuation = impact direct sur signal reçu
    "obstruction": 0.20,   # Obstruction = densité bâti (déjà dans urban_intensity)
}

# Fréquence 5G NR FR1 (3.5 GHz, bande principale déploiements France)
DEFAULT_FREQ_GHZ = 3.5

# Distance nominale pour le calcul path loss par cellule de 200m
NOMINAL_DISTANCE_M = 300.0  # mi-rayon d'une cellule macro typique en urbain


# ─────────────────────────────────────────────
# FONCTION PATH LOSS (module-level, réutilisable)
# ─────────────────────────────────────────────

def estimate_path_loss(
    distance_m: float,
    freq_ghz: float,
    los: float,
    attenuation_db: float
) -> float:
    """
    Modèle de pertes de propagation simplifié pour 5G NR FR1.

    Formule :
        PL = FSPL + attenuation_factor + correction_LOS

    FSPL (Free Space Path Loss) :
        FSPL [dB] = 32.4 + 20·log10(d_km) + 20·log10(f_GHz)
        Source : ITU-R P.525

    Correction LOS :
        - Bonne LOS (los > 0.7) : −5 dB (moins de diffraction)
        - Mauvaise LOS (los < 0.3) : +8 dB (NLOS urbain)
        - Intermédiaire : interpolation linéaire

    Args:
        distance_m    : distance antenne-UE (mètres)
        freq_ghz      : fréquence porteuse (GHz)
        los           : probabilité LOS [0, 1]
        attenuation_db: facteur d'atténuation terrain (dB, issu TerrainAgent)

    Returns:
        Path loss total (dB), borné à [50, 140] dB
    """
    d_km = max(distance_m, 1.0) / 1000.0
    fspl = 32.4 + 20.0 * np.log10(d_km) + 20.0 * np.log10(max(freq_ghz, 0.1))

    # Correction LOS : bonne visibilité = moins de pertes par diffraction
    if los >= 0.7:
        los_correction = -5.0
    elif los <= 0.3:
        los_correction = +8.0
    else:
        # Interpolation linéaire entre 0.3 et 0.7
        los_correction = 8.0 - (los - 0.3) / 0.4 * 13.0

    pl = fspl + attenuation_db + los_correction
    return float(np.clip(pl, 50.0, 140.0))


# ─────────────────────────────────────────────
class SitePlacementAgent:
    """
    Agent de recommandation de sites 5G v4.

    Entrée  : GeoDataFrame issu de CoverageAgent (final_merged_grid.gpkg)
    Sortie  : GeoDataFrame sites recommandés + rapport texte
    """

    def __init__(self, cell_size_m: float = 200.0,
                 freq_ghz: float = DEFAULT_FREQ_GHZ,
                 report_path: str = "output/site_placement_report.txt"):
        self.cell_size_m   = cell_size_m
        self.cell_area_km2 = (cell_size_m / 1000) ** 2  # 0.04 km²
        self.freq_ghz      = freq_ghz
        self._report_path  = report_path

    # ──────────────────────────────────────────
    # POINT D'ENTRÉE
    # ──────────────────────────────────────────
    def run(self, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = grid.copy()
        gdf = gdf.to_crs(epsg=2154)  # Lambert 93 — unités métriques

        print("=" * 70)
        print("  SitePlacementAgent v4 — Recommandation sites 5G (Radio-aware)")
        print("=" * 70)
        print(f"  Cellules en entrée : {len(gdf)}")

        # 0. Normalisation colonnes
        gdf = self._normalize_columns(gdf)

        # 1. Calcul path loss et radio_quality par cellule
        gdf = self._compute_path_loss(gdf)

        # 2. Filtrage préliminaire
        candidates = self._filter_candidates(gdf)

        # 3. Score composite (inclut propagation_score)
        candidates = self._compute_score(candidates)

        # 4. Seuillage par score minimum
        candidates = self._apply_score_threshold(candidates)

        # 5. Sélection greedy par contrainte distance ADAPTATIVE
        sites = self._greedy_distance_selection(candidates)

        # 6. Plafonnement densité absolue
        sites = self._cap_by_density(sites, gdf)

        # 7. Attribution type de site (avec règles radio)
        sites = self._assign_site_type(sites)

        # 8. Rapport
        self._print_report(sites, gdf, report_path=self._report_path)

        return sites

    # ──────────────────────────────────────────
    # 0. NORMALISATION COLONNES
    # ──────────────────────────────────────────
    def _normalize_columns(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        # demand_tier depuis demand_category (PopulationDemandAgent) ou demand_class (legacy)
        if "demand_tier" not in gdf.columns:
            mapping = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            if "demand_category" in gdf.columns:
                gdf["demand_tier"] = gdf["demand_category"].map(mapping).fillna(1).astype(int)
                print("  [FIX] demand_tier mappé depuis demand_category")
            elif "demand_class" in gdf.columns:
                gdf["demand_tier"] = gdf["demand_class"].map(mapping).fillna(1).astype(int)
                print("  [FIX] demand_tier mappé depuis demand_class (legacy)")

        # Valeurs par défaut sécurisées — y compris colonnes radio
        for col, default in [
            ("capacity_deficit_gbps", 0.0),
            ("coverage_score",        0.5),
            ("urban_intensity",       0.5),
            ("demand_tier",           2),
            ("water_ratio",           0.0),
            ("forest_ratio",          0.0),
            ("accessibility_score",   0.5),
            ("coverage_class",        "partial"),
            ("urban_class",           "urban"),
            # Colonnes radio (TerrainEnvironmentAgent)
            ("los_probability",       0.5),
            ("attenuation_factor",    10.0),
            ("obstruction_index",     0.5),
        ]:
            if col not in gdf.columns:
                gdf[col] = default
                print(f"  [DEFAULT] '{col}' = {default}")

        # Clamping sécurité
        gdf["los_probability"]   = gdf["los_probability"].clip(0.0, 1.0)
        gdf["attenuation_factor"]= gdf["attenuation_factor"].clip(0.0, 40.0)
        gdf["obstruction_index"] = gdf["obstruction_index"].clip(0.0, 1.0)

        return gdf

    # ──────────────────────────────────────────
    # 1b. PATH LOSS PAR CELLULE
    # ──────────────────────────────────────────
    def _compute_path_loss(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calcule le path loss estimé pour chaque cellule à distance nominale.

        Utilise estimate_path_loss() avec :
        - distance fixe NOMINAL_DISTANCE_M (300m) — représente la portée typique
          d'une antenne vers le bord de sa cellule en 5G NR FR1
        - LOS et atténuation issues de TerrainEnvironmentAgent

        radio_quality est l'inverse normalisé du path loss :
          radio_quality = 1 - (PL - PL_min) / (PL_max - PL_min)
          → 1 = canal favorable, 0 = canal très dégradé
        """
        los_arr = gdf["los_probability"].values
        att_arr = gdf["attenuation_factor"].values

        pl_arr = np.array([
            estimate_path_loss(NOMINAL_DISTANCE_M, self.freq_ghz, los, att)
            for los, att in zip(los_arr, att_arr)
        ])

        gdf["path_loss_db"] = np.round(pl_arr, 2)

        # Normalisation inverse robuste [0, 1]
        pl_min = np.percentile(pl_arr, 5)
        pl_max = np.percentile(pl_arr, 95)
        if pl_max > pl_min:
            radio_q = 1.0 - np.clip((pl_arr - pl_min) / (pl_max - pl_min), 0.0, 1.0)
        else:
            radio_q = np.full(len(pl_arr), 0.5)

        gdf["radio_quality"] = np.round(radio_q, 4)

        print(f"\n[ PATH LOSS ]")
        print(f"  PL moy.         : {pl_arr.mean():.1f} dB")
        print(f"  PL min/max      : {pl_arr.min():.1f} / {pl_arr.max():.1f} dB")
        print(f"  Radio quality   : {radio_q.mean():.3f} (moy.)")

        return gdf

    # ──────────────────────────────────────────
    # 1. FILTRAGE PRÉLIMINAIRE
    # ──────────────────────────────────────────
    def _filter_candidates(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        print("\n[ FILTRAGE ]")

        mask_water  = gdf["water_ratio"] > 0.7
        mask_access = gdf["accessibility_score"] < 0.05
        mask_forest = gdf["forest_ratio"] > 0.85
        mask_ok     = gdf["coverage_class"].isin(["excellent", "good"])

        excl_water  = mask_water.sum()
        excl_access = mask_access.sum()
        excl_forest = mask_forest.sum()
        excl_ok     = mask_ok.sum()

        excluded   = mask_water | mask_access | mask_forest | mask_ok
        candidates = gdf[~excluded].copy()

        print(f"  Exclus eau              : {excl_water}")
        print(f"  Exclus accès            : {excl_access}")
        print(f"  Exclus forêt            : {excl_forest}")
        print(f"  Exclus couverture OK    : {excl_ok}")
        print(f"  Total exclus            : {excluded.sum()}")
        print(f"  Candidats retenus       : {len(candidates)}")

        return candidates

    # ──────────────────────────────────────────
    # 2. SCORE COMPOSITE (v4 — radio-aware)
    # ──────────────────────────────────────────
    def _compute_score(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Score composite [0, 1] intégrant les métriques radio.

        Structure :
          s_deficit      : urgence capacitaire (CoverageAgent)
          s_coverage     : gap de couverture (CoverageAgent)
          s_demand       : niveau de demande (PopulationDemandAgent)
          s_urban        : intensité urbaine (OSMnxUrbanAgent)
          propagation_score :
            s_los        : bonne LOS → site nécessaire pour couvrir (facteur positif)
            s_attenuation: forte atténuation → site nécessaire (facteur positif)
            s_obstruction: forte obstruction → pénalité (site moins efficace seul)
        """
        df = gdf.copy()

        def norm(series):
            """Normalisation min-max robuste [0, 1]. Retourne 0.5 si constante."""
            mn, mx = series.min(), series.max()
            if mx == mn:
                return pd.Series(0.5, index=series.index)
            return (series - mn) / (mx - mn)

        # ── Composantes infrastructure (identiques v3) ──────────────────────
        s_deficit  = norm(df["capacity_deficit_gbps"].clip(lower=0))
        s_coverage = 1.0 - norm(df["coverage_score"].clip(0, 1))
        s_demand   = norm(df["demand_tier"].astype(float))
        s_urban    = norm(df["urban_intensity"].clip(0, 1))

        # ── Composantes radio (NOUVEAU v4) ──────────────────────────────────

        # LOS : une bonne LOS signifie que le site sera EFFICACE → score élevé
        # (zones avec bonne LOS sont des bons endroits pour placer des antennes)
        s_los = df["los_probability"].clip(0, 1)

        # Atténuation : forte atténuation → besoin d'un site proche → score élevé
        # On inverse la normalisation : att élevée = priorité plus haute
        s_attenuation = norm(df["attenuation_factor"].clip(0, 40))

        # Obstruction : forte obstruction → site moins efficace → pénalité
        # On l'intègre négativement dans propagation_score
        s_obstruction = df["obstruction_index"].clip(0, 1)

        # propagation_score : combinaison pondérée (dans [0, 1])
        # LOS + atténuation augmentent la priorité, obstruction la réduit légèrement
        propagation_score = np.clip(
            PROPAGATION_WEIGHTS["los"]         * s_los
            + PROPAGATION_WEIGHTS["attenuation"] * s_attenuation
            - PROPAGATION_WEIGHTS["obstruction"] * (s_obstruction - 0.5)
            # Centré sur 0 pour la pénalité : obstruction > 0.5 pénalise,
            # obstruction < 0.5 donne un léger bonus (zone dégagée)
            , 0.0, 1.0
        )

        df["propagation_score"] = np.round(propagation_score, 4)

        # ── Score final ──────────────────────────────────────────────────────
        df["composite_score"] = np.clip(
            SCORE_WEIGHTS["capacity_deficit"] * s_deficit
            + SCORE_WEIGHTS["coverage_gap"]   * s_coverage
            + SCORE_WEIGHTS["demand"]          * s_demand
            + SCORE_WEIGHTS["urban_intensity"] * s_urban
            + SCORE_WEIGHTS["propagation"]     * propagation_score
            , 0.0, 1.0
        ).round(4)

        print(f"\n[ SCORE COMPOSITE v4 ]")
        print(f"  Score min        : {df['composite_score'].min():.4f}")
        print(f"  Score max        : {df['composite_score'].max():.4f}")
        print(f"  Score moy.       : {df['composite_score'].mean():.4f}")
        print(f"  Score méd.       : {df['composite_score'].median():.4f}")
        print(f"  Propagation moy. : {df['propagation_score'].mean():.4f}")

        return df

    # ──────────────────────────────────────────
    # 3. SEUILLAGE SCORE MINIMUM
    # ──────────────────────────────────────────
    def _apply_score_threshold(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        print(f"\n[ SEUILLAGE SCORE MINIMUM ]")
        before = len(gdf)

        masks = []
        for uc, threshold in SCORE_THRESHOLD.items():
            mask = (gdf["urban_class"] == uc) & (gdf["composite_score"] >= threshold)
            masks.append(mask)
            n = ((gdf["urban_class"] == uc) & (gdf["composite_score"] < threshold)).sum()
            if n > 0:
                print(f"  {uc:15s}: {n} cellules éliminées (score < {threshold})")

        known      = set(SCORE_THRESHOLD.keys())
        mask_other = (~gdf["urban_class"].isin(known)) & (gdf["composite_score"] >= 0.15)
        masks.append(mask_other)

        combined = pd.concat([gdf[m] for m in masks]).drop_duplicates()
        print(f"  Candidats après seuillage : {len(combined)} (/{before})")
        return combined

    # ──────────────────────────────────────────
    # 4. SÉLECTION GREEDY — DISTANCE ADAPTATIVE
    # ──────────────────────────────────────────
    def _greedy_distance_selection(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Sélection greedy avec distance minimale ADAPTATIVE selon les conditions radio.

        Formule distance adaptative :
            d_adapt = d_base × factor_LOS × factor_attenuation

        factor_LOS :
            Bonne LOS (los ≈ 1) → les antennes portent plus loin → espacement plus grand
            d_adapt *= (0.7 + 0.6 × los)  → facteur dans [0.70, 1.30]

        factor_attenuation :
            Forte atténuation → signal atténué rapidement → espacement réduit
            d_adapt *= (1.0 − 0.3 × att/40)  → facteur dans [0.70, 1.00]

        Résultat :
            Zone LOS parfaite sans atténuation : +30% d'espacement
            Zone très atténuée (40 dB)         : −30% d'espacement
            Combiné worst-case                 : ~0.70 × 0.70 = 0.49× d_base
        """
        print(f"\n[ SÉLECTION GREEDY PAR DISTANCE ADAPTATIVE ]")

        df = gdf.sort_values("composite_score", ascending=False).copy()
        df["cx"] = df.geometry.centroid.x
        df["cy"] = df.geometry.centroid.y

        selected_indices = []
        selected_cx      = []
        selected_cy      = []
        selected_uc      = []
        selected_d_adapt = []   # distance adaptative du site sélectionné (pour cross-class)

        for idx, row in df.iterrows():
            uc       = row["urban_class"]
            d_base   = MIN_DISTANCE_M.get(uc, 500)
            cx, cy   = row["cx"], row["cy"]
            los      = float(row.get("los_probability",   0.5))
            att      = float(row.get("attenuation_factor", 10.0))

            # Distance adaptative pour CE candidat
            factor_los = 0.7 + 0.6 * np.clip(los, 0.0, 1.0)
            factor_att = 1.0 - 0.3 * np.clip(att / 40.0, 0.0, 1.0)
            d_adapt    = d_base * factor_los * factor_att

            too_close = False
            for i, (sx, sy, suc, sd) in enumerate(
                zip(selected_cx, selected_cy, selected_uc, selected_d_adapt)
            ):
                dist = np.hypot(cx - sx, cy - sy)
                if suc == uc:
                    # Même classe : distance minimale = adaptative du candidat courant
                    required = d_adapt
                else:
                    # Cross-class : moyenne des distances adaptatives des deux sites
                    required = (d_adapt + sd) / 3.0
                if dist < required:
                    too_close = True
                    break

            if not too_close:
                selected_indices.append(idx)
                selected_cx.append(cx)
                selected_cy.append(cy)
                selected_uc.append(uc)
                selected_d_adapt.append(d_adapt)

        result = df.loc[selected_indices].copy()

        print(f"  {'urban_class':15s}  {'candidats':>10}  {'sélectionnés':>13}")
        print(f"  {'-'*42}")
        for uc in sorted(df["urban_class"].unique()):
            n_in  = (df["urban_class"] == uc).sum()
            n_out = (result["urban_class"] == uc).sum()
            print(f"  {uc:15s}  {n_in:>10}  {n_out:>13}")
        print(f"  {'TOTAL':15s}  {len(df):>10}  {len(result):>13}")

        return result

    # ──────────────────────────────────────────
    # 5. PLAFONNEMENT PAR DENSITÉ ABSOLUE
    # ──────────────────────────────────────────
    def _cap_by_density(self, sites: gpd.GeoDataFrame, full_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        print(f"\n[ PLAFONNEMENT DENSITÉ ABSOLUE ]")
        print(f"  {'urban_class':15s}  {'sites_greedy':>12}  {'max_autorisé':>13}  {'conservés':>10}")
        print(f"  {'-'*55}")

        kept_frames = []
        for uc in sorted(sites["urban_class"].unique()):
            uc_sites = sites[sites["urban_class"] == uc].copy()
            uc_cells = full_grid[full_grid["urban_class"] == uc]
            n_cells  = len(uc_cells)
            area_km2 = n_cells * self.cell_area_km2

            max_by_density = MAX_DENSITY_PER_KM2.get(uc, 2.0) * area_km2
            max_by_ratio   = MAX_RATIO.get(uc, 0.30) * n_cells
            max_sites      = max(1, int(min(max_by_density, max_by_ratio)))

            n_greedy = len(uc_sites)
            if n_greedy > max_sites:
                uc_sites = uc_sites.nlargest(max_sites, "composite_score")

            kept = len(uc_sites)
            print(f"  {uc:15s}  {n_greedy:>12}  {max_sites:>13}  {kept:>10}  "
                  f"(densité≤{MAX_DENSITY_PER_KM2.get(uc,2):.1f}/km², ratio≤{MAX_RATIO.get(uc,0.3):.0%})")
            kept_frames.append(uc_sites)

        result = gpd.GeoDataFrame(pd.concat(kept_frames), crs=sites.crs)
        print(f"\n  → Sites après plafonnement : {len(result)}")
        return result

    # ──────────────────────────────────────────
    # 6. TYPE DE SITE (v4 — règles radio ajoutées)
    # ──────────────────────────────────────────
    def _assign_site_type(self, sites: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Attribution du type de site avec priorité aux contraintes radio.

        Règles (ordre de priorité décroissant) :
          1. LOS < 0.45  → small_cell_dense
             (visibilité directe mauvaise → small cells obligatoires pour NLOS coverage)
          2. attenuation_factor > 25 dB → small_cell_dense
             (atténuation sévère → small cells pour maintenir le lien)
          3. capacity_deficit > 3 Gbps ET zone dense → small_cell_dense
             (surcharge capacitaire critique)
          4. urban_class mapping de base (SITE_TYPE_MAP)
        """
        sites = sites.copy()

        # Base : mapping urban_class
        sites["site_type"] = sites["urban_class"].map(SITE_TYPE_MAP).fillna("macro_cell")

        # Règle radio 1 : mauvaise LOS → small cell obligatoire
        # Seuil 0.45 : en dessous, les pertes NLOS sont trop importantes
        # pour une macro (3GPP TR 36.814 Urban NLOS model)
        bad_los = sites.get("los_probability", pd.Series(1.0, index=sites.index)) < 0.45
        sites.loc[bad_los, "site_type"] = "small_cell_dense"

        # Règle radio 2 : forte atténuation → small cell obligatoire
        # 25 dB correspond au seuil où le bilan de liaison 5G NR FR1 est compromis
        # pour une macro standard (EIRP typique 43 dBm, sensibilité UE −100 dBm)
        high_att = sites.get("attenuation_factor", pd.Series(0.0, index=sites.index)) > 25.0
        sites.loc[high_att, "site_type"] = "small_cell_dense"

        # Règle capacitaire : déficit > 3 Gbps en zone dense → small cell
        high_deficit = sites["capacity_deficit_gbps"] > 3.0
        is_dense     = sites["urban_class"].isin(["dense_urban", "hyper_dense"])
        sites.loc[high_deficit & is_dense, "site_type"] = "small_cell_dense"

        # Log des surcharges radio
        n_radio_override = (bad_los | high_att).sum()
        if n_radio_override > 0:
            print(f"\n  [RADIO OVERRIDE] {n_radio_override} sites forcés → small_cell_dense")
            print(f"    - LOS < 0.45        : {bad_los.sum()} sites")
            print(f"    - Att > 25 dB       : {high_att.sum()} sites")

        return sites

    # ──────────────────────────────────────────
    # 7. RAPPORT FINAL
    # ──────────────────────────────────────────
    def _print_report(self, sites: gpd.GeoDataFrame, full_grid: gpd.GeoDataFrame,
                      report_path: str = "output/site_placement_report.txt"):
        lines = self._build_report(sites)
        for l in lines:
            print(l)
        import os
        os.makedirs(os.path.dirname(report_path) if os.path.dirname(report_path) else ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n  → Rapport : {report_path}")

    def _build_report(self, sites: gpd.GeoDataFrame) -> list:
        from datetime import datetime

        SEP  = "=" * 70
        SEP2 = "-" * 100
        lines = []

        def L(s=""):
            lines.append(s)

        L(SEP)
        L("   RAPPORT PLACEMENT SITES 5G — SitePlacementAgent v4 (Radio-aware)")
        L(f"   Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        L(SEP)

        L()
        L("[ RÉSUMÉ ]")
        L(f"  Sites recommandés     : {len(sites)}")
        L(f"  Score moyen           : {sites['composite_score'].mean():.4f}")
        L(f"  Score max             : {sites['composite_score'].max():.4f}")
        if "propagation_score" in sites.columns:
            L(f"  Propagation moy.      : {sites['propagation_score'].mean():.4f}")
        if "radio_quality" in sites.columns:
            L(f"  Radio quality moy.    : {sites['radio_quality'].mean():.4f}")
        if "path_loss_db" in sites.columns:
            L(f"  Path loss moy.        : {sites['path_loss_db'].mean():.1f} dB")

        L()
        L("[ DISTRIBUTION PAR TYPE DE SITE ]")
        for st, cnt in sites["site_type"].value_counts().items():
            L(f"  {st:<25s}:  {cnt:>2} sites")

        L()
        L("[ DISTRIBUTION PAR URBAN CLASS ]")
        for uc, grp in sites.groupby("urban_class"):
            sc   = grp["composite_score"].mean()
            def_ = grp["capacity_deficit_gbps"].mean()
            los_ = grp["los_probability"].mean() if "los_probability" in grp else float("nan")
            L(f"  {uc:<15s}:  {len(grp):>2} sites  "
              f"(score moy: {sc:.3f}  déficit moy: {def_:.3f}G  LOS moy: {los_:.2f})")

        L()
        L("[ TOP 10 SITES PAR PRIORITÉ ]")
        hdr = (f"  {'Rank':>4}  {'urban_class':<15}  {'type':<25}  "
               f"{'score':>7}  {'déficit':>9}  {'LOS':>5}  {'PL(dB)':>7}  Justification")
        L(hdr)
        L("  " + "-" * 110)

        top10 = sites.nlargest(10, "composite_score").reset_index(drop=True)
        for i, row in top10.iterrows():
            los_val = row.get("los_probability", float("nan"))
            los_str = f"{los_val*100:.1f}%" if not (isinstance(los_val, float) and np.isnan(los_val)) else "  N/A"
            pl_val  = row.get("path_loss_db", float("nan"))
            pl_str  = f"{pl_val:.1f}" if not (isinstance(pl_val, float) and np.isnan(pl_val)) else "N/A"
            justif  = self._build_justification(row)
            L(f"  {i+1:>4}  {row['urban_class']:<15}  {row['site_type']:<25}"
              f"  {row['composite_score']:>7.4f}  {row['capacity_deficit_gbps']:>9.2f}G"
              f"  {los_str:>5}  {pl_str:>7}  {justif}")

        L()
        L("[ DÉTAIL DE TOUS LES SITES ]")
        sites_sorted = sites.sort_values("composite_score", ascending=False).reset_index(drop=True)

        def _safe_float(v, default=float("nan")):
            try: return float(v) if v is not None else default
            except (TypeError, ValueError): return default

        for i, row in sites_sorted.iterrows():
            cx = row.geometry.centroid.x if row.geometry else float("nan")
            cy = row.geometry.centroid.y if row.geometry else float("nan")

            los_val  = row.get("los_probability", float("nan"))
            los_str  = f"{los_val*100:.1f}%" if not (isinstance(los_val, float) and np.isnan(los_val)) else "N/A"
            pl_val   = row.get("path_loss_db", float("nan"))
            pl_str   = f"{pl_val:.1f} dB" if not (isinstance(pl_val, float) and np.isnan(pl_val)) else "N/A"
            rq_val   = row.get("radio_quality", float("nan"))
            rq_str   = f"{rq_val:.3f}" if not (isinstance(rq_val, float) and np.isnan(rq_val)) else "N/A"
            ps_val   = row.get("propagation_score", float("nan"))
            ps_str   = f"{ps_val:.3f}" if not (isinstance(ps_val, float) and np.isnan(ps_val)) else "N/A"

            pop = row.get("population", None)
            try:
                pop_val = float(pop) if pop is not None else float("nan")
            except (TypeError, ValueError):
                pop_val = float("nan")
            pop_str = f"~{int(pop_val):,} hab" if not np.isnan(pop_val) and pop_val > 0 else "N/A"

            # Rayon de couverture radio réaliste par type de site
            site_type_r = row.get("site_type", "macro_cell")
            los         = _safe_float(row.get("los_probability", float("nan")))
            obstruction = _safe_float(row.get("obstruction_index", 0.5))
            radius_nom  = {
                "small_cell_dense": 80,
                "micro_cell":       220,
                "macro_cell":       550,
                "macro_cell_large": 1500,
            }.get(site_type_r, 300)
            if not np.isnan(los):
                los_factor = 0.7 + 0.6 * los
            else:
                los_factor = 1.0
            obs_factor = 1.0 - 0.3 * obstruction
            radius     = int(radius_nom * los_factor * obs_factor)

            L()
            L(f"  Site #{i+1:02d}")
            L(f"    Coordonnées      : {cx:.6f}, {cy:.6f}")
            L(f"    Urban class      : {row['urban_class']}")
            L(f"    Type             : {row['site_type']}")
            L(f"    Score composite  : {row['composite_score']:.4f}")
            L(f"    Propagation score: {ps_str}")
            L(f"    Radio quality    : {rq_str}")
            L(f"    Path loss estimé : {pl_str}")
            L(f"    Déficit          : {row['capacity_deficit_gbps']:.3f} Gbps")
            L(f"    Pop. couverte    : {pop_str}")
            L(f"    Rayon couv.      : ~{radius} m")
            L(f"    LOS              : {los_str}")
            L(f"    Justification    : {self._build_justification(row)}")

        L()
        L(SEP)
        L("  Fin du rapport")
        L(SEP)

        return lines

    def _build_justification(self, row) -> str:
        """Génère une phrase de justification synthétique incluant le contexte radio."""
        parts = []

        def _fval(key, default=float("nan")):
            v = row.get(key, default)
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        deficit    = _fval("capacity_deficit_gbps", 0.0)
        cov_cls    = row.get("coverage_class", "")
        los        = _fval("los_probability")
        hotspot    = bool(row.get("is_hotspot_risk", False))
        demand_cat = row.get("demand_category", "")
        overload   = _fval("overload_score")
        pop_dens   = _fval("population_density")
        prop_cplx  = _fval("propagation_complexity")
        att_db     = _fval("attenuation_factor")
        pl_db      = _fval("path_loss_db")

        # Déficit capacité
        if deficit >= 3.0:
            parts.append(f"déficit capacité critique ({deficit:.1f} Gbps)")
        elif deficit >= 1.0:
            parts.append(f"déficit capacité significatif ({deficit:.1f} Gbps)")
        elif deficit > 0.1:
            parts.append(f"déficit capacité modéré ({deficit:.2f} Gbps)")

        # Couverture
        if cov_cls in ("none", "critical"):
            parts.append("zone non couverte")
        elif cov_cls == "partial":
            parts.append("couverture partielle")

        # Congestion
        if hotspot:
            if not np.isnan(overload) and overload > 0.7:
                parts.append(f"risque congestion critique (overload={overload:.2f})")
            else:
                parts.append("risque congestion élevé")
        elif demand_cat in ("high", "critical"):
            parts.append(f"demande {demand_cat}")

        # Densité pop
        if not np.isnan(pop_dens) and pop_dens > 20_000:
            parts.append(f"densité pop. élevée ({int(pop_dens):,} hab/km²)")

        # Contexte radio — NOUVEAU v4
        if not np.isnan(los) and los < 0.45:
            parts.append(f"NLOS sévère (LOS={los*100:.0f}%) → small cell recommandé")
        elif not np.isnan(att_db) and att_db > 25.0:
            parts.append(f"atténuation forte ({att_db:.0f} dB) → small cell recommandé")
        elif not np.isnan(pl_db) and pl_db > 110.0:
            parts.append(f"path loss élevé ({pl_db:.0f} dB)")

        # Propagation complexe (OSMnx)
        if not np.isnan(prop_cplx) and prop_cplx > 0.7:
            parts.append(f"propagation complexe ({prop_cplx:.2f})")

        if not parts:
            parts.append("zone à renforcer")

        return " | ".join(parts)


# ─────────────────────────────────────────────
# TEST STANDALONE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os

    print("=== TEST STANDALONE SitePlacementAgent v4 ===")

    gpkg_path = "outputs/final_merged_grid.gpkg"
    if not os.path.exists(gpkg_path):
        gpkg_path = "output/final_merged_grid.gpkg"
    if not os.path.exists(gpkg_path):
        print(f"  ERREUR : fichier introuvable ({gpkg_path})")
        sys.exit(1)

    print(f"  Lecture : {gpkg_path}")
    grid = gpd.read_file(gpkg_path)
    print(f"  {len(grid)} cellules chargées")

    agent = SitePlacementAgent(
        cell_size_m=200.0,
        freq_ghz=3.5,
        report_path="output/site_placement_report_v4.txt"
    )
    sites = agent.run(grid)

    os.makedirs("output", exist_ok=True)
    out_gpkg = "output/recommended_sites_v4.gpkg"
    out_cols = [c for c in sites.columns if c != "geometry"] + ["geometry"]
    sites[out_cols].to_file(out_gpkg, driver="GPKG")
    print(f"  → GeoPackage : {out_gpkg}")