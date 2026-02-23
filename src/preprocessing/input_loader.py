from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from lxml import etree
from shapely import wkt

from config import settings


def load_magnetic_grid(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Файл магнитной сетки не найден: {path}")

    ext = path.suffix.lower()
    if ext == ".npy":
        magnetic_grid = np.load(path)
        if magnetic_grid.ndim != 2:
            raise ValueError(
                f"Ожидался 2D массив магнитной сетки, получено shape={magnetic_grid.shape}"
            )
        return magnetic_grid

    if ext in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            band = src.read(1).astype(float)
            if src.nodata is not None:
                band = np.where(band == src.nodata, np.nan, band)
            return np.nan_to_num(band, nan=float(np.nanmean(band)))

    raise ValueError(
        f"Неподдерживаемый формат магнитной сетки: {path}. Ожидается .npy или .tif/.tiff"
    )


def load_vector_layer(path: Path, dataset_name: str) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл '{dataset_name}' не найден: {path}")

    gdf = gpd.read_file(path)
    if "geometry" not in gdf.columns:
        raise ValueError(f"Файл '{dataset_name}' не содержит геометрию: {path}")
    if gdf.crs is None:
        gdf = gdf.set_crs(settings.CRS_LOCAL)
    return gdf


def _parse_miis_section(parent: etree._Element, section_name: str, row_name: str) -> list[dict[str, Any]]:
    section = parent.find(section_name)
    if section is None:
        return []

    records: list[dict[str, Any]] = []
    for row in section.findall(row_name):
        geom_node = row.find("Geometry")
        if geom_node is None or not geom_node.text:
            continue

        geometry = wkt.loads(geom_node.text)
        item: dict[str, Any] = {
            "id": row.get("id"),
            "type": row.get("type", "unknown"),
            "geometry": geometry,
        }
        risk = row.get("risk")
        if risk is not None:
            item["risk_class"] = risk
        records.append(item)
    return records


def load_miis_xml(path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"MIIS XML не найден: {path}")

    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    tree = etree.parse(str(path), parser)
    root = tree.getroot()
    crs = root.get("crs", settings.CRS_LOCAL)

    anomalies_records = _parse_miis_section(root, "Anomalies", "Anomaly")
    utilities_records = _parse_miis_section(root, "Utilities", "Utility")

    anomalies_gdf = gpd.GeoDataFrame(anomalies_records, geometry="geometry", crs=crs)
    utilities_gdf = gpd.GeoDataFrame(utilities_records, geometry="geometry", crs=crs)

    if anomalies_gdf.empty:
        anomalies_gdf = gpd.GeoDataFrame(columns=["id", "type", "risk_class", "geometry"], geometry="geometry", crs=crs)
    if utilities_gdf.empty:
        utilities_gdf = gpd.GeoDataFrame(columns=["id", "type", "geometry"], geometry="geometry", crs=crs)

    return anomalies_gdf, utilities_gdf


def load_anomalies_and_utilities(
    anomalies_path: Path | None,
    utilities_path: Path | None,
    miis_xml_path: Path | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, str]:
    if miis_xml_path is not None:
        anomalies_gdf, utilities_gdf = load_miis_xml(miis_xml_path)
        return anomalies_gdf, utilities_gdf, "miis_xml"

    xml_path: Path | None = None
    if anomalies_path is not None and anomalies_path.suffix.lower() == ".xml":
        xml_path = anomalies_path
    elif utilities_path is not None and utilities_path.suffix.lower() == ".xml":
        xml_path = utilities_path

    if xml_path is not None:
        anomalies_gdf, utilities_gdf = load_miis_xml(xml_path)
        return anomalies_gdf, utilities_gdf, "miis_xml"

    if anomalies_path is None or utilities_path is None:
        raise ValueError(
            "Для запуска нужны оба слоя (anomalies и utilities) или путь к MIIS XML."
        )

    anomalies_gdf = load_vector_layer(anomalies_path, "аномалии")
    utilities_gdf = load_vector_layer(utilities_path, "коммуникации")
    return anomalies_gdf, utilities_gdf, "vector_layers"
