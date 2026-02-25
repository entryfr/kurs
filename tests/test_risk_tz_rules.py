import geopandas as gpd
from shapely.geometry import Point, Polygon

from src.pipeline.service import _classify_risk_tz, _compute_zone_flags


def test_classify_risk_tz_critical_when_no_utility_in_construction_zone() -> None:
    utilities = gpd.GeoDataFrame(
        {"depth": [1.0], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:32637",
    )
    row = gpd.GeoSeries(
        {
            "depth": 1.2,
            "utilities_in_buffer": [],
        }
    )
    risk_class, probability, rule, mismatch = _classify_risk_tz(
        row=row,
        utilities_gdf=utilities,
        model_risk_class="LOW",
        model_probability=0.2,
        in_construction_zone=True,
    )
    assert risk_class == "CRITICAL"
    assert probability == 0.89
    assert rule == "TZ_RULE_CRITICAL_NO_MATCH_IN_CONSTRUCTION"
    assert mismatch is None


def test_classify_risk_tz_high_on_depth_mismatch() -> None:
    utilities = gpd.GeoDataFrame(
        {"depth": [1.0], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:32637",
    )
    row = gpd.GeoSeries(
        {
            "depth": 2.0,
            "utilities_in_buffer": [0],
        }
    )
    risk_class, probability, rule, mismatch = _classify_risk_tz(
        row=row,
        utilities_gdf=utilities,
        model_risk_class="LOW",
        model_probability=0.2,
        in_construction_zone=True,
    )
    assert risk_class == "HIGH"
    assert probability == 0.62
    assert rule == "TZ_RULE_DEPTH_MISMATCH"
    assert mismatch is not None and mismatch > 0.5


def test_classify_risk_tz_low_on_depth_match() -> None:
    utilities = gpd.GeoDataFrame(
        {"depth": [1.1], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:32637",
    )
    row = gpd.GeoSeries(
        {
            "depth": 1.2,
            "utilities_in_buffer": [0],
        }
    )
    risk_class, probability, rule, mismatch = _classify_risk_tz(
        row=row,
        utilities_gdf=utilities,
        model_risk_class="HIGH",
        model_probability=0.65,
        in_construction_zone=True,
    )
    assert risk_class == "LOW"
    assert probability == 0.12
    assert rule == "TZ_RULE_DEPTH_MATCH"
    assert mismatch is not None and mismatch <= 0.5


def test_classify_risk_tz_fallback_to_model() -> None:
    utilities = gpd.GeoDataFrame(
        {"depth": [None], "geometry": [Point(0, 0)]},
        geometry="geometry",
        crs="EPSG:32637",
    )
    row = gpd.GeoSeries(
        {
            "depth": None,
            "utilities_in_buffer": [0],
        }
    )
    risk_class, probability, rule, mismatch = _classify_risk_tz(
        row=row,
        utilities_gdf=utilities,
        model_risk_class="HIGH",
        model_probability=0.65,
        in_construction_zone=False,
    )
    assert risk_class == "HIGH"
    assert probability == 0.65
    assert rule == "MODEL_FALLBACK"
    assert mismatch is None


def test_compute_zone_flags_with_excavation_depth() -> None:
    anomalies = gpd.GeoDataFrame(
        {"geometry": [Point(1, 1), Point(20, 20)]},
        geometry="geometry",
        crs="EPSG:32637",
    )
    zones = gpd.GeoDataFrame(
        {
            "excavation_depth": [6.5],
            "geometry": [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )

    in_zone, excavation = _compute_zone_flags(anomalies, zones)
    assert in_zone[0] is True
    assert in_zone[1] is False
    assert excavation[0] == 6.5
    assert excavation[1] is None
