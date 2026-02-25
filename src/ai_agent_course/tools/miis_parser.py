from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
from lxml import etree
from shapely import wkt
from shapely.geometry import Polygon


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_miis_layers(miis_xml_path: Path | None) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if miis_xml_path is None or not miis_xml_path.exists():
        empty = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:32637")
        return empty.copy(), empty.copy(), empty.copy()

    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    tree = etree.parse(str(miis_xml_path), parser)
    root = tree.getroot()
    crs = root.get("crs", "EPSG:32637")

    utility_rows: list[dict[str, Any]] = []
    for utility in root.findall(".//Utility"):
        geom = utility.findtext("Geometry")
        if not geom:
            continue
        utility_rows.append(
            {
                "id": utility.get("id"),
                "type": utility.get("type", "unknown"),
                "material": utility.get("material", "unknown"),
                "diameter_mm": _safe_float(utility.get("diameter"), 150.0),
                "depth_m": _safe_float(utility.get("depth"), 1.5),
                "geometry": wkt.loads(geom),
            }
        )

    well_rows: list[dict[str, Any]] = []
    for well in root.findall(".//Well"):
        geom = well.findtext("Geometry")
        if not geom:
            continue
        well_rows.append(
            {
                "id": well.get("id"),
                "cadastre": well.get("cadastre"),
                "inventory_year": well.get("inventory_year"),
                "geometry": wkt.loads(geom),
            }
        )

    zone_rows: list[dict[str, Any]] = []
    for zone in root.findall(".//ConstructionZone"):
        geom = zone.findtext("Geometry")
        depth = _safe_float(zone.get("excavation_depth"), 3.0)
        if geom:
            zone_geom = wkt.loads(geom)
        else:
            # В demo-случае, если геометрия не задана, используем большой контур.
            zone_geom = Polygon([(0, 0), (5000, 0), (5000, 5000), (0, 5000)])
        zone_rows.append({"id": zone.get("id"), "excavation_depth_m": depth, "geometry": zone_geom})

    utilities_gdf = gpd.GeoDataFrame(utility_rows, geometry="geometry", crs=crs)
    wells_gdf = gpd.GeoDataFrame(well_rows, geometry="geometry", crs=crs)
    zones_gdf = gpd.GeoDataFrame(zone_rows, geometry="geometry", crs=crs)

    if utilities_gdf.empty:
        utilities_gdf = gpd.GeoDataFrame(
            columns=["id", "type", "material", "diameter_mm", "depth_m", "geometry"],
            geometry="geometry",
            crs=crs,
        )
    if wells_gdf.empty:
        wells_gdf = gpd.GeoDataFrame(columns=["id", "cadastre", "inventory_year", "geometry"], geometry="geometry", crs=crs)
    if zones_gdf.empty:
        zones_gdf = gpd.GeoDataFrame(columns=["id", "excavation_depth_m", "geometry"], geometry="geometry", crs=crs)

    return utilities_gdf, wells_gdf, zones_gdf

