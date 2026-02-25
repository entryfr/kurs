import geopandas as gpd
from shapely.geometry import LineString, Point

from src.pipeline.service import _apply_linear_threshold


def test_apply_linear_threshold_filters_short_lines_in_projected_crs() -> None:
    anomalies = gpd.GeoDataFrame(
        {
            "name": ["short", "long", "point"],
            "geometry": [
                LineString([(0, 0), (5, 0)]),
                LineString([(0, 0), (20, 0)]),
                Point(1, 1),
            ],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )

    filtered, dropped = _apply_linear_threshold(anomalies, min_length_m=10.0)
    assert dropped == 1
    assert len(filtered) == 2
    assert set(filtered["name"].tolist()) == {"long", "point"}


def test_apply_linear_threshold_transforms_from_geographic_crs() -> None:
    anomalies = gpd.GeoDataFrame(
        {
            "name": ["short_deg", "long_deg"],
            "geometry": [
                LineString([(37.6415, 55.6721), (37.64154, 55.6721)]),
                LineString([(37.6415, 55.6721), (37.6417, 55.6721)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    filtered, dropped = _apply_linear_threshold(anomalies, min_length_m=10.0)
    assert dropped == 1
    assert len(filtered) == 1
    assert filtered.iloc[0]["name"] == "long_deg"
