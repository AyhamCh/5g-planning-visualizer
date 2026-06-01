import { useEffect, useMemo, useRef, useState } from "react";
import type {
  GridFC,
  SitesFC,
  GridProps,
} from "@/lib/telco-data";
import {
  COVERAGE_COLORS,
  DEMAND_COLORS,
  SITE_COLORS,
  URBAN_COLORS,
  fmt,
} from "@/lib/telco-data";

export type ColorBy = "coverage_class" | "demand_class" | "urban_class" | "los_probability";

function colorFor(p: GridProps, by: ColorBy): string {
  if (by === "coverage_class") return COVERAGE_COLORS[p.coverage_class] ?? "#64748b";
  if (by === "demand_class") return DEMAND_COLORS[p.demand_class] ?? "#64748b";
  if (by === "urban_class") return URBAN_COLORS[p.urban_class] ?? "#64748b";
  const t = Math.max(0, Math.min(1, p.los_probability));
  return `hsl(${(140 * t).toFixed(0)} 70% 50%)`;
}

function popupHtml(p: GridProps): string {
  return `
  <div style="font-family:Inter,system-ui;font-size:12px;min-width:240px">
    <div style="font-weight:600;margin-bottom:6px">Cell #${p.cell_id} · ${p.urban_class}</div>
    <table style="width:100%;border-collapse:collapse">
      <tr><td style="color:#94a3b8">Population</td><td style="text-align:right">${fmt(p.population)} (${fmt(p.population_density)} /km²)</td></tr>
      <tr><td style="color:#94a3b8">Peak demand</td><td style="text-align:right">${fmt(p.peak_demand_gbps)} Gbps</td></tr>
      <tr><td style="color:#94a3b8">Capacity</td><td style="text-align:right">${fmt(p.capacity_available_gbps)} Gbps</td></tr>
      <tr><td style="color:#94a3b8">Deficit</td><td style="text-align:right">${fmt(p.capacity_deficit_gbps)} Gbps</td></tr>
      <tr><td style="color:#94a3b8">Coverage</td><td style="text-align:right">${p.coverage_class} (${fmt(p.coverage_score)})</td></tr>
      <tr><td style="color:#94a3b8">Antennas</td><td style="text-align:right">${p.antenna_count} (NR ${p.antenna_count_nr} / LTE ${p.antenna_count_lte})</td></tr>
      <tr><td style="color:#94a3b8">LOS prob</td><td style="text-align:right">${(p.los_probability*100).toFixed(0)}%</td></tr>
      <tr><td style="color:#94a3b8">Attenuation</td><td style="text-align:right">${fmt(p.attenuation_factor)} dB</td></tr>
      <tr><td style="color:#94a3b8">Elevation</td><td style="text-align:right">${fmt(p.elevation_mean)} m</td></tr>
    </table>
  </div>`;
}

export function NetworkMap({
  grid,
  sites,
  colorBy,
  showSites = true,
  height = "100%",
}: {
  grid: GridFC;
  sites: SitesFC;
  colorBy: ColorBy;
  showSites?: boolean;
  height?: string | number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  // Hold leaflet module + map; both are browser-only.
  const stateRef = useRef<{
    L: typeof import("leaflet") | null;
    map: import("leaflet").Map | null;
    grid: import("leaflet").GeoJSON | null;
    sites: import("leaflet").LayerGroup | null;
  }>({ L: null, map: null, grid: null, sites: null });
  const [ready, setReady] = useState(false);

  const bbox = useMemo(() => {
    let minLng = Infinity, minLat = Infinity, maxLng = -Infinity, maxLat = -Infinity;
    for (const f of grid.features) {
      for (const ring of f.geometry.coordinates) {
        for (const [lng, lat] of ring as [number, number][]) {
          if (lng < minLng) minLng = lng;
          if (lat < minLat) minLat = lat;
          if (lng > maxLng) maxLng = lng;
          if (lat > maxLat) maxLat = lat;
        }
      }
    }
    return [[minLat, minLng], [maxLat, maxLng]] as [[number, number], [number, number]];
  }, [grid]);

  // Init map (browser only)
  useEffect(() => {
    if (typeof window === "undefined" || !ref.current || stateRef.current.map) return;
    let cancelled = false;
    (async () => {
      const L = (await import("leaflet")).default;
      if (cancelled || !ref.current) return;
      const map = L.map(ref.current, { zoomControl: true });
      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        {
          maxZoom: 19,
          attribution:
            '© <a href="https://www.openstreetmap.org/copyright">OSM</a> · © <a href="https://carto.com/attributions">CARTO</a>',
        },
      ).addTo(map);
      map.fitBounds(bbox, { padding: [24, 24] });
      stateRef.current.L = L;
      stateRef.current.map = map;
      setReady(true);
    })();
    return () => {
      cancelled = true;
      stateRef.current.map?.remove();
      stateRef.current = { L: null, map: null, grid: null, sites: null };
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Grid layer
  useEffect(() => {
    const { L, map } = stateRef.current;
    if (!ready || !L || !map) return;
    stateRef.current.grid?.remove();
    stateRef.current.grid = L.geoJSON(grid as GeoJSON.GeoJsonObject, {
      style: (feat) => {
        const p = (feat?.properties ?? {}) as GridProps;
        return {
          color: "#0f172a",
          weight: 0.5,
          fillColor: colorFor(p, colorBy),
          fillOpacity: 0.65,
        };
      },
      onEachFeature: (feat, layer) => {
        const p = feat.properties as GridProps;
        layer.bindPopup(popupHtml(p));
        layer.bindTooltip(
          `#${p.cell_id} · ${p.urban_class} · cov ${p.coverage_class}`,
          { sticky: true, opacity: 0.9 },
        );
      },
    }).addTo(map);
  }, [grid, colorBy, ready]);

  // Sites layer
  useEffect(() => {
    const { L, map } = stateRef.current;
    if (!ready || !L || !map) return;
    stateRef.current.sites?.remove();
    if (!showSites) return;
    const group = L.layerGroup();
    for (const f of sites.features) {
      const p = f.properties;
      const [lng, lat] = f.geometry.coordinates;
      const r = 8 + p.composite_score * 8;
      const fill = SITE_COLORS[p.site_type] ?? "#a855f7";
      L.circleMarker([lat, lng], {
        radius: r,
        color: "#ffffff",
        weight: 2,
        fillColor: fill,
        fillOpacity: 0.95,
      })
        .bindPopup(
          `<div style="font-family:Inter;font-size:12px;min-width:220px">
            <div style="font-weight:600;margin-bottom:4px">Site #${p.site_id} · ${p.site_type}</div>
            <div>Score: <b>${p.composite_score.toFixed(4)}</b></div>
            <div>Deficit: ${fmt(p.capacity_deficit_gbps)} Gbps</div>
            <div>Urban: ${p.urban_class}</div>
            <div>LOS: ${(p.los_probability*100).toFixed(0)}%</div>
            <div>Pop. density: ${fmt(p.population_density)} /km²</div>
          </div>`,
        )
        .addTo(group);
    }
    stateRef.current.sites = group.addTo(map);
  }, [sites, showSites, ready]);

  return <div ref={ref} style={{ height, width: "100%" }} />;
}
