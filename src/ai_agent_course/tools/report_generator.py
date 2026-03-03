from __future__ import annotations

from pathlib import Path
from typing import Any

import ezdxf
import geopandas as gpd
import matplotlib
from jinja2 import Environment
from lxml import etree

# В web/worker-потоках на Windows нужен non-GUI backend, иначе падает tkinter/main-thread.
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from shapely.geometry import LineString, MultiLineString


def export_dxf_scheme(
    anomalies_gdf: gpd.GeoDataFrame,
    utilities_gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    for _, row in utilities_gdf.iterrows():
        geom = row.get("geometry")
        if isinstance(geom, LineString):
            msp.add_lwpolyline(list(geom.coords), dxfattribs={"layer": "UTILITIES"})
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                msp.add_lwpolyline(list(part.coords), dxfattribs={"layer": "UTILITIES"})

    for _, row in anomalies_gdf.iterrows():
        geom = row.get("geometry")
        if isinstance(geom, LineString):
            msp.add_lwpolyline(list(geom.coords), dxfattribs={"layer": "UNACCOUNTED", "color": 1})
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                msp.add_lwpolyline(list(part.coords), dxfattribs={"layer": "UNACCOUNTED", "color": 1})

    doc.saveas(str(output_path))
    return str(output_path)


def export_miis_xml(
    anomalies: list[dict[str, Any]],
    output_path: Path,
    crs: str = "EPSG:32637",
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = etree.Element("MIIS", version="2.0", crs=crs)
    an_section = etree.SubElement(root, "Anomalies")

    for item in anomalies:
        node = etree.SubElement(
            an_section,
            "Anomaly",
            id=str(item.get("id")),
            type=str(item.get("utility_type", "unknown")),
            risk=str(item.get("risk_class", "LOW")),
            probability=str(item.get("probability", 0.12)),
        )
        etree.SubElement(node, "Geometry").text = str(item.get("geometry_wkt", ""))
        etree.SubElement(node, "DepthM").text = str(item.get("depth_m", "-"))
        etree.SubElement(node, "LengthM").text = str(item.get("length_m", "-"))
        etree.SubElement(node, "Recommendation").text = str(item.get("recommendation", "-"))

    tree = etree.ElementTree(root)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True, pretty_print=True)
    return str(output_path)


def generate_pdf_act(
    site_name: str,
    anomalies: list[dict[str, Any]],
    validation_issues: list[str],
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = Environment(autoescape=False)
    template = env.from_string(
        """
        АКТ выявленных неучтённых коммуникаций (СП 47.13330.2016, Приложение №5)
        Объект: {{ site_name }}
        Количество аномалий: {{ anomalies|length }}
        """
    )
    header = template.render(site_name=site_name, anomalies=anomalies).strip()

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = [header, "", "Сводка по аномалиям:"]
    for item in anomalies[:25]:
        lines.append(
            f"#{item['id']}: {item['risk_class']} ({item['probability']:.2f}), "
            f"{item['utility_type']}, глубина={item['depth_m']:.2f}м, "
            f"метод={item['recommendation']}"
        )

    if validation_issues:
        lines.append("")
        lines.append("Нормативные замечания:")
        for issue in validation_issues[:20]:
            lines.append(f"- {issue}")

    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=9, family="monospace")
    fig.tight_layout()
    fig.savefig(str(output_path), format="pdf")
    plt.close(fig)
    return str(output_path)

