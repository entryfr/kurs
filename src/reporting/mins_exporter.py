from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
from lxml import etree


def _to_text(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def create_mins_xml(
    anomalies_gdf: gpd.GeoDataFrame,
    utilities_gdf: gpd.GeoDataFrame,
    output_path: str = "mins_exchange.xml",
    profile: str = "normative",
) -> str:
    root = etree.Element(
        "MINSExchange",
        version="1.0",
        authority="Минстрой РФ",
        generatedAt=datetime.now(timezone.utc).isoformat(),
    )
    metadata = etree.SubElement(root, "Metadata")
    etree.SubElement(metadata, "Profile").text = profile
    etree.SubElement(metadata, "AnomaliesCount").text = str(len(anomalies_gdf))
    etree.SubElement(metadata, "UtilitiesCount").text = str(len(utilities_gdf))
    etree.SubElement(metadata, "CRS").text = _to_text(anomalies_gdf.crs, "unknown")

    anomalies_elem = etree.SubElement(root, "UnaccountedUtilities")
    for idx, row in anomalies_gdf.iterrows():
        item = etree.SubElement(
            anomalies_elem,
            "Object",
            id=str(idx),
            riskClass=_to_text(row.get("risk_class"), "LOW"),
            riskProbability=_to_text(row.get("risk_probability"), "0.0"),
            utilityType=_to_text(row.get("utility_type"), "unknown"),
            sourceRule=_to_text(row.get("risk_rule"), "UNKNOWN"),
        )
        geometry = etree.SubElement(item, "Geometry", format="WKT")
        geometry.text = row.geometry.wkt if row.get("geometry") is not None else ""
        etree.SubElement(item, "DepthM").text = _to_text(row.get("depth"), "-")
        etree.SubElement(item, "Recommendation").text = _to_text(
            row.get("recommendation"),
            "Инженерное уточнение",
        )

    utilities_elem = etree.SubElement(root, "RegisteredUtilities")
    for idx, row in utilities_gdf.iterrows():
        utility = etree.SubElement(
            utilities_elem,
            "Utility",
            id=str(idx),
            utilityType=_to_text(row.get("type"), "unknown"),
        )
        geometry = etree.SubElement(utility, "Geometry", format="WKT")
        geometry.text = row.geometry.wkt if row.get("geometry") is not None else ""
        etree.SubElement(utility, "DepthM").text = _to_text(row.get("depth"), "-")
        etree.SubElement(utility, "DiameterMM").text = _to_text(row.get("diameter"), "-")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = etree.ElementTree(root)
    tree.write(str(output), pretty_print=True, xml_declaration=True, encoding="utf-8")
    return str(output)
