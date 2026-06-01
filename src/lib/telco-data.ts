// Real schema derived from final_merged_grid.gpkg and recommended_sites_v3.gpkg.
// DO NOT add synthetic fields here.

export interface GridProps {
  cell_id: number;
  // urban (osmnx)
  n_buildings: number;
  built_area_m2: number;
  built_density: number;
  building_density_per_km2: number;
  estimated_height_m: number;
  estimated_floors: number;
  urban_intensity: number;
  obstruction_index: number;
  propagation_complexity: number;
  handover_risk: number;
  urban_compactness: number;
  urban_structure: string;
  urban_class: string;
  // roads
  road_length_m: number;
  road_density_m_per_km2: number;
  n_intersections: number;
  intersection_density: number;
  dominant_road_type: string;
  road_quality_score: number;
  accessibility_score: number;
  accessibility_class: string;
  // demand
  cell_area_km2: number;
  population: number;
  population_density: number;
  usage_type: string;
  demand_modulation: number;
  base_demand_mbps: number;
  peak_demand_mbps: number;
  base_demand_gbps: number;
  peak_demand_gbps: number;
  demand_category: string;
  demand_normalized: number;
  demand_class: string;
  // coverage
  antenna_count: number;
  antenna_count_nr: number;
  antenna_count_lte: number;
  capacity_available_gbps: number;
  nearest_antenna_m: number;
  nearest_radio: string;
  is_covered: boolean;
  capacity_demand_ratio: number;
  capacity_deficit_gbps: number;
  overload_score: number;
  distance_score: number;
  has_5g: boolean;
  is_hotspot_risk: boolean;
  coverage_score: number;
  coverage_class: string;
  // terrain
  elevation_mean: number;
  elevation_min: number;
  elevation_max: number;
  elevation_std: number;
  elevation_range: number;
  slope_mean: number;
  slope_norm: number;
  forest_ratio: number;
  water_ratio: number;
  vegetation_ratio: number;
  bare_ratio: number;
  attenuation_factor: number;
  los_probability: number;
  terrain_complexity: number;
}

export interface SiteProps extends GridProps {
  site_id: number;
  demand_tier: number;
  composite_score: number;
  cx: number;
  cy: number;
  site_type: string;
}

export type GridFC = GeoJSON.FeatureCollection<GeoJSON.Polygon, GridProps>;
export type SitesFC = GeoJSON.FeatureCollection<GeoJSON.Point, SiteProps>;

// API base: when running the FastAPI backend (see /backend), set VITE_API_BASE
// to e.g. http://localhost:8000. Otherwise the real GeoJSON exports under
// /data/ (generated from the real .gpkg outputs) are used directly.
const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "");

async function fetchJson<T>(staticPath: string, apiPath: string): Promise<T> {
  const url = API_BASE ? `${API_BASE}${apiPath}` : staticPath;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return (await r.json()) as T;
}

export const fetchGrid = () =>
  fetchJson<GridFC>("/data/grid.geojson", "/api/grid");

export const fetchSites = () =>
  fetchJson<SitesFC>("/data/sites.geojson", "/api/sites");

// ── Derived KPIs (computed from real grid / sites, no hardcoding) ──────────
export interface Kpis {
  cells: number;
  population: number;
  totalPeakDemandGbps: number;
  totalCapacityGbps: number;
  totalDeficitGbps: number;
  coverageRatio: number;
  has5gRatio: number;
  hotspotRiskCount: number;
  meanCoverageScore: number;
  meanLosProbability: number;
  meanAttenuationDb: number;
  antennas: number;
  antennasNr: number;
  antennasLte: number;
}

export function computeKpis(grid: GridFC): Kpis {
  const f = grid.features.map((x) => x.properties);
  const sum = (k: keyof GridProps) => f.reduce((a, b) => a + (Number(b[k]) || 0), 0);
  const mean = (k: keyof GridProps) => (f.length ? sum(k) / f.length : 0);
  const cov = f.filter((c) => c.is_covered).length;
  return {
    cells: f.length,
    population: Math.round(sum("population")),
    totalPeakDemandGbps: sum("peak_demand_gbps"),
    totalCapacityGbps: sum("capacity_available_gbps"),
    totalDeficitGbps: sum("capacity_deficit_gbps"),
    coverageRatio: f.length ? cov / f.length : 0,
    has5gRatio: f.length ? f.filter((c) => c.has_5g).length / f.length : 0,
    hotspotRiskCount: f.filter((c) => c.is_hotspot_risk).length,
    meanCoverageScore: mean("coverage_score"),
    meanLosProbability: mean("los_probability"),
    meanAttenuationDb: mean("attenuation_factor"),
    antennas: Math.round(sum("antenna_count")),
    antennasNr: Math.round(sum("antenna_count_nr")),
    antennasLte: Math.round(sum("antenna_count_lte")),
  };
}

export function distribution<K extends keyof GridProps>(
  grid: GridFC,
  key: K,
): Array<{ label: string; count: number; ratio: number }> {
  const map = new Map<string, number>();
  for (const f of grid.features) {
    const k = String(f.properties[key] ?? "—");
    map.set(k, (map.get(k) ?? 0) + 1);
  }
  const n = grid.features.length || 1;
  return [...map.entries()]
    .map(([label, count]) => ({ label, count, ratio: count / n }))
    .sort((a, b) => b.count - a.count);
}

// Color scales used by map + cards (single source of truth).
export const URBAN_COLORS: Record<string, string> = {
  hyper_dense: "#ef4444",
  dense_urban: "#f97316",
  urban: "#eab308",
  periurban: "#22c55e",
  rural: "#0ea5e9",
};
export const COVERAGE_COLORS: Record<string, string> = {
  uncovered: "#ef4444",
  critical: "#f97316",
  adequate: "#eab308",
  good: "#22c55e",
  excellent: "#0ea5e9",
};
export const DEMAND_COLORS: Record<string, string> = {
  low: "#22c55e",
  medium: "#eab308",
  high: "#f97316",
  very_high: "#ef4444",
};
export const SITE_COLORS: Record<string, string> = {
  macro_cell: "#0ea5e9",
  micro_cell: "#a855f7",
  small_cell_dense: "#ef4444",
};

export function fmt(n: number, digits = 2): string {
  if (!isFinite(n)) return "—";
  if (Math.abs(n) >= 1000) return n.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return n.toLocaleString("en-US", { maximumFractionDigits: digits });
}
