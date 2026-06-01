"""
TERRAIN & ENVIRONMENT AGENT v2
================================

Agent d'analyse du terrain et de l'environnement physique.
Aligné avec le pipeline 5G AI (OSMnxUrbanAgent + PopulationDemandAgent).

Rôle:
- Analyser relief (altitude, pente, rugosité) via DEM SRTM
- Analyser occupation du sol (forêt, eau) via ESA WorldCover
- Calculer facteurs d'impact radio intégrant les features OSMnx

Entrée:
- osmnx_grid (GeoDataFrame) : sortie OSMnxUrbanAgent
  Colonnes attendues : urban_class, obstruction_index, estimated_height_m,
                       built_density, urban_intensity, propagation_complexity
- dem_path (str)        : raster DEM SRTM (.tif)
- landcover_dir (str)   : dossier ESA WorldCover (tuiles .tif)

Sorties (colonnes ajoutées au GeoDataFrame entrant):
  Relief    : elevation_mean, elevation_min, elevation_max,
              elevation_std, elevation_range, slope_mean, slope_norm
  Land Cover: forest_ratio, water_ratio, vegetation_ratio, bare_ratio
  Radio     : attenuation_factor, los_probability, terrain_complexity

Colonnes NOT recalculées (déjà dans OSMnx):
  urban_ratio → remplacé par built_density OSMnx
  obstruction_index, estimated_height_m → utilisés directement

Intégration pipeline:
  OSMnxUrbanAgent ──► TerrainEnvironmentAgent ──► (suite pipeline)
                              ▲
                   PopulationDemandAgent (peut tourner en parallèle)
"""

import geopandas as gpd
import numpy as np
import rasterio
from rasterstats import zonal_stats
from pathlib import Path
import logging
import tempfile
from osgeo import gdal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES
# =============================================================================

# Classes ESA WorldCover 10m v200
LANDCOVER_CLASSES = {
    10: 'tree_cover',
    20: 'shrubland',
    30: 'grassland',
    40: 'cropland',
    50: 'built_up',        # Non utilisé pour atténuation → OSMnx
    60: 'bare_vegetation',
    70: 'snow_ice',
    80: 'water_bodies',
    90: 'herbaceous_wetland',
    95: 'mangroves',
    100: 'moss_lichen'
}

# Colonnes OSMnx requises en entrée
OSMNX_REQUIRED_COLS = [
    'urban_class',
    'obstruction_index',
    'estimated_height_m',
    'built_density',
]

# Colonnes OSMnx optionnelles (dégradation gracieuse si absentes)
OSMNX_OPTIONAL_COLS = [
    'urban_intensity',
    'propagation_complexity',
    'accessibility_score',
]


# =============================================================================
# AGENT
# =============================================================================

class TerrainEnvironmentAgent:
    """
    Agent d'analyse terrain aligné avec le pipeline OSMnx 5G AI.

    Reçoit le GeoDataFrame enrichi par OSMnxUrbanAgent et y ajoute
    les métriques terrain (relief + land cover + impact radio).

    L'atténuation urbaine n'est PAS recalculée depuis ESA WorldCover :
    elle utilise obstruction_index et estimated_height_m d'OSMnx,
    qui sont plus précis (bâtiments réels) que la classe 50 WorldCover.
    """

    def __init__(self, dem_path: str, landcover_dir: str):
        """
        Initialise l'agent avec les rasters uniquement.
        La grille est passée à run() comme les autres agents du pipeline.

        Args:
            dem_path     : Chemin raster DEM SRTM (.tif)
            landcover_dir: Dossier ESA WorldCover contenant les tuiles .tif
        """
        logger.info("=" * 70)
        logger.info("INITIALISATION TerrainEnvironmentAgent v2")
        logger.info("=" * 70)

        self.dem_path = Path(dem_path)
        self.landcover_dir = Path(landcover_dir)

        # Chemins temporaires (créés à run())
        self._vrt_path = None
        self._clipped_lc_path = None
        self._slope_path = None

        # Vérifications existence
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM introuvable: {self.dem_path}")
        if not self.landcover_dir.exists():
            raise FileNotFoundError(f"Dossier Land Cover introuvable: {self.landcover_dir}")

        # Vérification DEM accessible
        with rasterio.open(self.dem_path) as src:
            logger.info(f"DEM chargé: {src.width}×{src.height}px | CRS: {src.crs}")
            self._dem_crs = src.crs

        logger.info("✓ TerrainEnvironmentAgent initialisé (grille attendue dans run())")

    # =========================================================================
    # PIPELINE PRINCIPAL
    # =========================================================================

    def run(self, osmnx_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Pipeline complet : reçoit la grille OSMnx, retourne le GDF enrichi.

        Args:
            osmnx_grid: GeoDataFrame produit par OSMnxUrbanAgent.
                        Doit contenir les colonnes OSMnx (urban_class,
                        obstruction_index, estimated_height_m, built_density).

        Returns:
            GeoDataFrame: même grille + colonnes terrain ajoutées
        """
        logger.info("=" * 70)
        logger.info("ANALYSE TERRAIN & ENVIRONNEMENT")
        logger.info("=" * 70)

        # Copie défensive (ne pas modifier l'original)
        self.grid = osmnx_grid.copy()

        # Validation entrée
        self._validate_osmnx_input()

        # Correction géométries invalides
        self._fix_geometries()

        # Construction land cover VRT + clip sur emprise de la grille
        self._build_landcover(self.grid)

        # Harmonisation CRS
        self._harmonize_crs()

        # ── Extraction ──────────────────────────────────────────────────────
        self._extract_dem_features()
        self._extract_landcover_features()

        # ── Calcul métriques radio ───────────────────────────────────────────
        self._compute_terrain_metrics()

        # Nettoyage valeurs aberrantes
        self.grid.replace([np.inf, -np.inf], 0, inplace=True)
        self.grid.fillna(0, inplace=True)

        # Nettoyage fichiers temporaires
        self._cleanup_temp_files()

        logger.info("=" * 70)
        logger.info("✓ ANALYSE TERRAIN TERMINÉE")
        logger.info(f"  Colonnes ajoutées : {self._new_columns()}")
        logger.info("=" * 70)

        return self.grid

    # =========================================================================
    # VALIDATION ENTRÉE OSMNX
    # =========================================================================

    def _validate_osmnx_input(self):
        """Vérifie que le GDF entrant est bien une sortie OSMnxUrbanAgent."""
        logger.info("Validation entrée OSMnx...")

        if self.grid.empty:
            raise ValueError("GeoDataFrame entrant vide !")

        if self.grid.crs is None:
            raise ValueError("CRS non défini dans le GeoDataFrame entrant !")

        # Colonnes requises
        missing = [c for c in OSMNX_REQUIRED_COLS if c not in self.grid.columns]
        if missing:
            raise ValueError(
                f"Colonnes OSMnx manquantes: {missing}\n"
                f"Colonnes présentes: {list(self.grid.columns)}\n"
                f"Vérifiez que la grille provient bien d'OSMnxUrbanAgent."
            )

        # Colonnes optionnelles — avertissement seulement
        missing_opt = [c for c in OSMNX_OPTIONAL_COLS if c not in self.grid.columns]
        if missing_opt:
            logger.warning(f"Colonnes OSMnx optionnelles absentes (dégradation): {missing_opt}")

        # Valider plages
        if self.grid['obstruction_index'].max() > 1.0 or self.grid['obstruction_index'].min() < 0.0:
            logger.warning("obstruction_index hors [0,1] — valeurs seront clampées")

        logger.info(
            f"✓ Grille valide: {len(self.grid)} cellules | "
            f"CRS: {self.grid.crs} | "
            f"urban_class: {self.grid['urban_class'].unique().tolist()}"
        )

    def _fix_geometries(self):
        """Corrige les géométries invalides."""
        invalid = ~self.grid.geometry.is_valid
        if invalid.any():
            n = invalid.sum()
            logger.warning(f"{n} géométries invalides → correction buffer(0)")
            self.grid.loc[invalid, 'geometry'] = (
                self.grid.loc[invalid, 'geometry'].buffer(0)
            )

    # =========================================================================
    # LAND COVER : VRT + CLIP
    # =========================================================================

    def _build_landcover(self, grid: gpd.GeoDataFrame):
        """
        Construit VRT depuis tuiles ESA WorldCover et le clip sur
        l'emprise de la grille (bounds WGS84).
        """
        logger.info(f"Construction VRT Land Cover depuis: {self.landcover_dir}")

        tif_files = sorted(self.landcover_dir.glob("**/*.tif"))
        if not tif_files:
            raise FileNotFoundError(f"Aucun .tif dans {self.landcover_dir}")
        logger.info(f"  {len(tif_files)} tuile(s) trouvée(s)")

        # VRT virtuel (pas de copie physique)
        self._vrt_path = Path(tempfile.gettempdir()) / "lc_mosaic.vrt"
        vrt = gdal.BuildVRT(
            str(self._vrt_path),
            [str(f) for f in tif_files],
            options=gdal.BuildVRTOptions(resampleAlg="nearest")
        )
        if vrt is None:
            raise RuntimeError("Échec construction VRT")
        vrt.FlushCache()
        vrt = None

        # Clip sur emprise grille (WGS84)
        minx, miny, maxx, maxy = grid.to_crs("EPSG:4326").total_bounds
        self._clipped_lc_path = Path(tempfile.gettempdir()) / "lc_clipped.tif"

        result = gdal.Translate(
            str(self._clipped_lc_path),
            str(self._vrt_path),
            projWin=[minx, maxy, maxx, miny]
        )
        if result is None:
            raise RuntimeError("gdal.Translate clip a échoué")
        result = None

        with rasterio.open(self._clipped_lc_path) as src:
            self._lc_crs = src.crs
            logger.info(
                f"✓ Land Cover clippé: {src.width}×{src.height}px | "
                f"CRS: {src.crs}"
            )

    # =========================================================================
    # HARMONISATION CRS
    # =========================================================================

    def _harmonize_crs(self):
        """Reprojette la grille vers le CRS du land cover si nécessaire."""
        logger.info("Harmonisation CRS...")
        if self.grid.crs != self._lc_crs:
            logger.info(f"Reprojection grille: {self.grid.crs} → {self._lc_crs}")
            self.grid = self.grid.to_crs(self._lc_crs)
        if self._dem_crs != self._lc_crs:
            logger.warning(
                f"DEM CRS ({self._dem_crs}) ≠ Land Cover CRS ({self._lc_crs}) "
                "— attention aux erreurs de projection pour le DEM"
            )
        logger.info("✓ CRS harmonisés")

    # =========================================================================
    # EXTRACTION DEM
    # =========================================================================

    def _extract_dem_features(self):
        """Extrait métriques de relief depuis DEM SRTM."""
        logger.info("Extraction métriques DEM...")

        stats = zonal_stats(
            self.grid,
            str(self.dem_path),
            stats=['mean', 'min', 'max', 'std'],
            nodata=-9999,
            all_touched=True
        )

        self.grid['elevation_mean']  = [s['mean'] or 0.0 for s in stats]
        self.grid['elevation_min']   = [s['min']  or 0.0 for s in stats]
        self.grid['elevation_max']   = [s['max']  or 0.0 for s in stats]
        self.grid['elevation_std']   = [s['std']  or 0.0 for s in stats]
        self.grid['elevation_range'] = self.grid['elevation_max'] - self.grid['elevation_min']

        # Calcul raster de pente réelle (gradient DEM)
        self._slope_path = Path(tempfile.gettempdir()) / "slope.tif"
        with rasterio.open(self.dem_path) as src:
            dem_arr = src.read(1).astype(float)
            gy, gx  = np.gradient(dem_arr, src.res[0], src.res[1])
            slope   = np.sqrt(gx**2 + gy**2)
            transform = src.transform
            crs       = src.crs

        with rasterio.open(
            self._slope_path, 'w',
            driver='GTiff',
            height=slope.shape[0], width=slope.shape[1],
            count=1, dtype=slope.dtype,
            crs=crs, transform=transform
        ) as dst:
            dst.write(slope, 1)

        slope_stats = zonal_stats(
            self.grid,
            str(self._slope_path),
            stats=['mean'],
            nodata=0,
            all_touched=True
        )
        self.grid['slope_mean'] = [s['mean'] or 0.0 for s in slope_stats]
        self.grid['slope_norm'] = self._robust_normalize(self.grid['slope_mean'])

        logger.info(
            f"✓ DEM extrait | "
            f"Alt moy={self.grid['elevation_mean'].mean():.1f}m | "
            f"Pente norm moy={self.grid['slope_norm'].mean():.2f}"
        )

    # =========================================================================
    # EXTRACTION LAND COVER
    # =========================================================================

    def _extract_landcover_features(self):
        """
        Extrait occupation du sol depuis ESA WorldCover.

        NOTE: urban_ratio n'est PAS extrait ici (doublon avec built_density
        d'OSMnxUrbanAgent qui est plus précis). Seules forêt, eau,
        végétation naturelle et sol nu sont calculés.
        """
        logger.info("Extraction occupation sol (ESA WorldCover)...")

        stats = zonal_stats(
            self.grid,
            str(self._clipped_lc_path),
            categorical=True,
            nodata=0,
            all_touched=False
        )

        forest_r = []
        water_r  = []
        veg_r    = []
        bare_r   = []

        for cell in stats:
            if not cell:
                forest_r.append(0.0)
                water_r.append(0.0)
                veg_r.append(0.0)
                bare_r.append(0.0)
                continue

            total = sum(cell.values())
            if total <= 0:
                forest_r.append(0.0)
                water_r.append(0.0)
                veg_r.append(0.0)
                bare_r.append(0.0)
                continue

            forest = cell.get(10, 0) / total
            water  = cell.get(80, 0) / total

            # Végétation naturelle (hors forêt, hors urbain classe 50)
            other_veg = sum([
                cell.get(20, 0),   # Arbustes
                cell.get(30, 0),   # Prairies
                cell.get(40, 0),   # Cultures
                cell.get(90, 0),   # Zones humides
                cell.get(95, 0),   # Mangroves
                cell.get(100, 0),  # Mousses
            ]) / total

            bare = cell.get(60, 0) / total

            forest_r.append(forest)
            water_r.append(water)
            veg_r.append(forest + other_veg)
            bare_r.append(bare)

        self.grid['forest_ratio']    = forest_r
        self.grid['water_ratio']     = water_r
        self.grid['vegetation_ratio']= veg_r
        self.grid['bare_ratio']      = bare_r

        logger.info(
            f"✓ Land Cover extrait | "
            f"Forêt={self.grid['forest_ratio'].mean():.1%} | "
            f"Eau={self.grid['water_ratio'].mean():.1%} | "
            f"Végétation={self.grid['vegetation_ratio'].mean():.1%}"
        )

    # =========================================================================
    # MÉTRIQUES RADIO (intégration OSMnx)
    # =========================================================================

    def _compute_terrain_metrics(self):
        """
        Calcule les métriques d'impact radio en intégrant:
        - Terrain naturel (DEM + land cover)
        - Context urbain OSMnx (obstruction_index, estimated_height_m)
 
        Métriques produites:
        - attenuation_factor : dB total (végétation + relief + bâti)
        - los_probability    : probabilité visibilité directe (0-1)
        - terrain_complexity : indice composite (0-1)
        """
        logger.info("Calcul métriques radio (terrain + OSMnx intégré)...")
 
        # ── Vegetation other than forest ──────────────────────────────────────
        other_veg = (
            self.grid['vegetation_ratio'] - self.grid['forest_ratio']
        ).clip(0, 1)
 
        # ── obstruction_index OSMnx (clamped) ────────────────────────────────
        obstruction = self.grid['obstruction_index'].clip(0, 1)
        heights     = self.grid['estimated_height_m'].clip(0, None)
 
        # ── ATTENUATION FACTOR (dB) ───────────────────────────────────────────
        # Végétation naturelle
        veg_att   = self.grid['forest_ratio'] * 15.0 + other_veg * 5.0
 
        # Relief (pente normalisée → max 10 dB)
        relief_att = self.grid['slope_norm'] * 10.0
 
        # Bâti urbain (OSMnx) — remplace le 0.0 de MorphologyAgent supprimé
        # obstruction_index ∈ [0,1] → [0, 20 dB]
        # estimated_height_m → bonus jusqu'à 5 dB (murs hauts = diffraction)
        height_factor = np.log1p(heights) / np.log1p(50)   # normalise sur ~50m
        urban_att = obstruction * 20.0 + height_factor.clip(0, 1) * 5.0
 
        self.grid['attenuation_factor'] = (
            veg_att + relief_att + urban_att
        ).clip(0, 40)  # plafonné à 40 dB (5G mmWave urban worst case)
 
        # ── LOS PROBABILITY ───────────────────────────────────────────────────
        # Pénalités terrain naturel
        slope_pen = self.grid['slope_norm']
        veg_pen   = self.grid['forest_ratio'] * 1.0 + other_veg * 0.5
 
        # Pénalité bâti OSMnx (obstruction = murs/bâtiments bloquant)
        # Plus le bâti est haut et dense, moins on a de LOS
        # Bonus surfaces ouvertes
        water_bonus = self.grid['water_ratio'] * 0.1
        bare_bonus  = self.grid['bare_ratio']  * 0.05
        urban_pen_nonlinear = (obstruction ** 1.5) * 0.9 + height_factor * 0.2

        self.grid['los_probability'] = (
            1.0
            - 0.20 * slope_pen           # relief (faible impact en urbain)
            - 0.15 * veg_pen             # végétation
            - urban_pen_nonlinear        # bâti : pénalité directe, non pondérée
            + water_bonus
            + bare_bonus
        ).clip(0, 1)
 
        
 
 
        # ── TERRAIN COMPLEXITY ────────────────────────────────────────────────
        # Objectif : refléter la difficulté de déploiement réseau 5G.
        # En zone urbaine dense, le bâti domine largement sur le relief.
        # En zone rurale/périurbaine, relief + végétation prennent le relais.
        #
        # Composantes :
        #   urban_intensity   (OSMnx) → densité bâti, hauteurs, rues : 35%
        #   propagation_complexity (OSMnx) → multipath, obstructions  : 25%
        #   veg_compl         (land cover) → forêt + végétation        : 20%
        #   rugosity          (DEM std)    → variabilité altitude       : 10%
        #   slope_norm        (DEM)        → pente (faible en urbain)   : 10%
        #
        # Dégradation gracieuse si colonnes OSMnx optionnelles absentes.
 
        rugosity  = self._robust_normalize(self.grid['elevation_std'])
        veg_compl = self.grid['forest_ratio'] * 1.0 + other_veg * 0.5
 
        has_urban_intensity    = 'urban_intensity'        in self.grid.columns
        has_propagation_compl  = 'propagation_complexity' in self.grid.columns
 
        if has_urban_intensity and has_propagation_compl:
            # Mode complet — OSMnx fournit les deux features discriminantes
            urban_intensity = self.grid['urban_intensity'].clip(0, 1)
            osmnx_compl     = self.grid['propagation_complexity'].clip(0, 1)
            self.grid['terrain_complexity'] = (
                0.35 * urban_intensity  +   # bâti réel → discriminant principal
                0.25 * osmnx_compl      +   # multipath / obstructions OSMnx
                0.20 * veg_compl        +   # végétation naturelle
                0.10 * rugosity         +   # relief (DEM)
                0.10 * self.grid['slope_norm']
            ).clip(0, 1)
 
        elif has_urban_intensity:
            # Mode dégradé 1 — urban_intensity seul depuis OSMnx
            urban_intensity = self.grid['urban_intensity'].clip(0, 1)
            self.grid['terrain_complexity'] = (
                0.45 * urban_intensity  +
                0.25 * veg_compl        +
                0.15 * rugosity         +
                0.15 * self.grid['slope_norm']
            ).clip(0, 1)
 
        elif has_propagation_compl:
            # Mode dégradé 2 — propagation_complexity seul depuis OSMnx
            osmnx_compl = self.grid['propagation_complexity'].clip(0, 1)
            self.grid['terrain_complexity'] = (
                0.45 * osmnx_compl      +
                0.25 * veg_compl        +
                0.15 * rugosity         +
                0.15 * self.grid['slope_norm']
            ).clip(0, 1)
 
        else:
            # Mode fallback — terrain naturel uniquement (sans OSMnx)
            logger.warning(
                "urban_intensity et propagation_complexity absents — "
                "terrain_complexity basé sur relief+végétation uniquement"
            )
            self.grid['terrain_complexity'] = (
                0.35 * self.grid['slope_norm']  +
                0.35 * veg_compl                +
                0.30 * rugosity
            ).clip(0, 1)
 
        logger.info(
            f"✓ Métriques radio calculées | "
            f"Att moy={self.grid['attenuation_factor'].mean():.1f}dB | "
            f"LOS moy={self.grid['los_probability'].mean():.1%} | "
            f"Complexity moy={self.grid['terrain_complexity'].mean():.2f}"
        )

    # =========================================================================
    # STATISTIQUES & EXPORT
    # =========================================================================

    def get_statistics(self) -> dict:
        """
        Retourne les statistiques terrain du dernier run().
        Doit être appelé après run().
        """
        if 'attenuation_factor' not in self.grid.columns:
            raise RuntimeError("Appelez run() avant get_statistics()")

        return {
            # Relief
            'elevation_mean_m'      : float(self.grid['elevation_mean'].mean()),
            'elevation_min_m'       : float(self.grid['elevation_min'].min()),
            'elevation_max_m'       : float(self.grid['elevation_max'].max()),
            'elevation_std_mean_m'  : float(self.grid['elevation_std'].mean()),
            'slope_mean_norm'       : float(self.grid['slope_norm'].mean()),

            # Land cover naturel
            'forest_ratio_mean'     : float(self.grid['forest_ratio'].mean()),
            'water_ratio_mean'      : float(self.grid['water_ratio'].mean()),
            'vegetation_ratio_mean' : float(self.grid['vegetation_ratio'].mean()),

            # OSMnx (passthrough pour stats)
            'obstruction_index_mean': float(self.grid['obstruction_index'].mean()),
            'height_mean_m'         : float(self.grid['estimated_height_m'].mean()),

            # Impact radio
            'attenuation_mean_db'       : float(self.grid['attenuation_factor'].mean()),
            'attenuation_max_db'        : float(self.grid['attenuation_factor'].max()),
            'los_probability_mean'      : float(self.grid['los_probability'].mean()),
            'terrain_complexity_mean'   : float(self.grid['terrain_complexity'].mean()),

            # Distribution par urban_class
            'by_urban_class': self._stats_by_urban_class(),

            # Meta
            'total_cells'           : len(self.grid),
        }

    def _stats_by_urban_class(self) -> dict:
        """Stats terrain agrégées par urban_class (depuis OSMnx)."""
        result = {}
        for uc, group in self.grid.groupby('urban_class'):
            result[uc] = {
                'n_cells'              : len(group),
                'attenuation_mean_db'  : float(group['attenuation_factor'].mean()),
                'los_probability_mean' : float(group['los_probability'].mean()),
                'terrain_complexity'   : float(group['terrain_complexity'].mean()),
                'forest_ratio_mean'    : float(group['forest_ratio'].mean()),
                'elevation_mean_m'     : float(group['elevation_mean'].mean()),
            }
        return result

    def export_statistics(self, output_path: str):
        """Exporte statistiques dans un fichier texte structuré."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.get_statistics()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("STATISTIQUES TERRAIN & ENVIRONNEMENT v2\n")
            f.write("=" * 70 + "\n\n")

            f.write("RELIEF (DEM SRTM):\n")
            f.write("-" * 70 + "\n")
            f.write(f"Altitude moyenne        : {stats['elevation_mean_m']:.1f} m\n")
            f.write(f"Altitude minimale       : {stats['elevation_min_m']:.1f} m\n")
            f.write(f"Altitude maximale       : {stats['elevation_max_m']:.1f} m\n")
            f.write(f"Rugosité moy. (std)     : {stats['elevation_std_mean_m']:.1f} m\n")
            f.write(f"Pente normalisée moy.   : {stats['slope_mean_norm']:.3f}\n\n")

            f.write("OCCUPATION SOL NATUREL (ESA WorldCover):\n")
            f.write("-" * 70 + "\n")
            f.write(f"Forêt moy.              : {stats['forest_ratio_mean']:.1%}\n")
            f.write(f"Eau moy.                : {stats['water_ratio_mean']:.1%}\n")
            f.write(f"Végétation moy.         : {stats['vegetation_ratio_mean']:.1%}\n\n")

            f.write("CONTEXTE URBAIN (OSMnxUrbanAgent):\n")
            f.write("-" * 70 + "\n")
            f.write(f"Obstruction index moy.  : {stats['obstruction_index_mean']:.3f}\n")
            f.write(f"Hauteur bâti moy.       : {stats['height_mean_m']:.1f} m\n\n")

            f.write("IMPACT RADIO (terrain + bâti intégrés):\n")
            f.write("-" * 70 + "\n")
            f.write(f"Atténuation moy.        : {stats['attenuation_mean_db']:.1f} dB\n")
            f.write(f"Atténuation max.        : {stats['attenuation_max_db']:.1f} dB\n")
            f.write(f"Probabilité LOS moy.    : {stats['los_probability_mean']:.1%}\n")
            f.write(f"Complexité terrain moy. : {stats['terrain_complexity_mean']:.3f}\n\n")

            f.write("PAR URBAN_CLASS (OSMnx):\n")
            f.write("-" * 70 + "\n")
            for uc, s in stats['by_urban_class'].items():
                f.write(f"\n  [{uc}]  {s['n_cells']} cellules\n")
                f.write(f"    Atténuation moy.  : {s['attenuation_mean_db']:.1f} dB\n")
                f.write(f"    LOS prob. moy.    : {s['los_probability_mean']:.1%}\n")
                f.write(f"    Complexité        : {s['terrain_complexity']:.3f}\n")
                f.write(f"    Forêt moy.        : {s['forest_ratio_mean']:.1%}\n")
                f.write(f"    Altitude moy.     : {s['elevation_mean_m']:.1f} m\n")

            f.write(f"\nCellules totales        : {stats['total_cells']}\n")

        logger.info(f"✓ Statistiques exportées: {output_path}")

    def save(self, output_path: str, driver: str = "GPKG"):
        """Sauvegarde le GeoDataFrame enrichi."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.grid.to_file(output_path, driver=driver)
        logger.info(f"✓ Grille sauvegardée: {output_path}")

    # =========================================================================
    # UTILITAIRES
    # =========================================================================

    @staticmethod
    def _robust_normalize(series) -> np.ndarray:
        """Normalisation robuste percentiles 5–95 → [0, 1]."""
        p5  = series.quantile(0.05)
        p95 = series.quantile(0.95)
        if p95 - p5 == 0:
            return np.zeros(len(series))
        return ((series - p5) / (p95 - p5)).clip(0, 1).values

    def _new_columns(self) -> list:
        """Retourne les colonnes ajoutées par cet agent."""
        return [
            'elevation_mean', 'elevation_min', 'elevation_max',
            'elevation_std', 'elevation_range', 'slope_mean', 'slope_norm',
            'forest_ratio', 'water_ratio', 'vegetation_ratio', 'bare_ratio',
            'attenuation_factor', 'los_probability', 'terrain_complexity',
        ]

    def _cleanup_temp_files(self):
        """Supprime les fichiers temporaires créés pendant le run."""
        for attr in ['_vrt_path', '_clipped_lc_path', '_slope_path']:
            path = getattr(self, attr, None)
            if path and Path(path).exists():
                try:
                    Path(path).unlink(missing_ok=True)
                    logger.debug(f"Temp supprimé: {path}")
                except Exception as e:
                    logger.warning(f"Impossible de supprimer {path}: {e}")

    def __del__(self):
        """Nettoyage à la destruction (filet de sécurité)."""
        self._cleanup_temp_files()


# =============================================================================
# EXEMPLE D'UTILISATION (pipeline complet)
# =============================================================================

if __name__ == "__main__":
    import argparse
    import geopandas as gpd

    parser = argparse.ArgumentParser(description="Terrain Environment Agent CLI")

    parser.add_argument("--grid", required=True, help="Path to OSMnx or demand grid (.gpkg)")
    parser.add_argument("--dem", required=True, help="Path to DEM raster (.tif)")
    parser.add_argument("--landcover", required=True, help="Path to ESA WorldCover folder")
    parser.add_argument("--output", required=True, help="Output grid (.gpkg)")
    parser.add_argument("--stats", required=True, help="Output stats (.txt)")

    args = parser.parse_args()

    try:
        # 1. Load grid
        grid = gpd.read_file(args.grid)
        logger.info(f"Grille chargée : {len(grid)} cellules | CRS: {grid.crs}")

        # 2. Run agent
        terrain_agent = TerrainEnvironmentAgent(
            dem_path=args.dem,
            landcover_dir=args.landcover,
        )

        terrain_agent.run(grid)

        # 3. Stats
        stats = terrain_agent.get_statistics()

        print("\n" + "=" * 70)
        print("RÉSUMÉ STATISTIQUES TERRAIN")
        print("=" * 70)
        print(f"Cellules totales     : {stats['total_cells']}")
        print(f"Altitude moy.        : {stats['elevation_mean_m']:.1f} m")
        print(f"Forêt moy.           : {stats['forest_ratio_mean']:.1%}")
        print(f"Atténuation moy.     : {stats['attenuation_mean_db']:.1f} dB")
        print(f"LOS moy.             : {stats['los_probability_mean']:.1%}")
        print(f"Complexité moy.      : {stats['terrain_complexity_mean']:.3f}")

        # 4. Save
        terrain_agent.save(args.output)
        terrain_agent.export_statistics(args.stats)

        print("\n✓ TRAITEMENT TERRAIN TERMINÉ")

    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=True)
        raise