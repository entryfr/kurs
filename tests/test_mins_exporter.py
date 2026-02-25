from pathlib import Path

import geopandas as gpd
from lxml import etree
from shapely.geometry import LineString

from src.reporting.mins_exporter import create_mins_xml


def test_create_mins_xml_writes_exchange_file(tmp_path: Path) -> None:
    anomalies = gpd.GeoDataFrame(
        {
            "risk_class": ["CRITICAL"],
            "risk_probability": [0.89],
            "utility_type": ["gas"],
            "risk_rule": ["TZ_RULE_CRITICAL_NO_MATCH_IN_CONSTRUCTION"],
            "depth": [2.0],
            "recommendation": ["Шурф №III-247"],
            "geometry": [LineString([(0, 0), (10, 0)])],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )
    utilities = gpd.GeoDataFrame(
        {
            "type": ["water"],
            "depth": [1.5],
            "diameter": [150],
            "geometry": [LineString([(1, 1), (5, 1)])],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )
    output = tmp_path / "mins_exchange.xml"

    written_path = create_mins_xml(
        anomalies_gdf=anomalies,
        utilities_gdf=utilities,
        output_path=str(output),
        profile="normative",
    )

    assert Path(written_path).exists()
    tree = etree.parse(str(output))
    root = tree.getroot()
    assert root.tag == "MINSExchange"
    assert root.findtext("./Metadata/Profile") == "normative"
    assert len(root.findall("./UnaccountedUtilities/Object")) == 1
    assert len(root.findall("./RegisteredUtilities/Utility")) == 1
