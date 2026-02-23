import geopandas as gpd
from shapely.geometry import LineString, Point

from src.agent.validator import (
    summarize_issues_by_rule,
    validate_anomalies_detailed,
)


def test_validate_anomalies_detailed_returns_rule_codes() -> None:
    anomalies = gpd.GeoDataFrame(
        {
            "risk_class": ["CRITICAL", "CRITICAL", "HIGH", "BAD", "LOW", "LOW"],
            "depth": [0.5, None, 2.0, 1.0, 1.2, 1.2],
            "has_utility": [False, False, False, False, True, True],
            "utility_type": [None, None, None, "water", "gas", "gas"],
            "geometry": [
                Point(0, 0),      # рядом с unknown utility
                Point(100, 100),  # без коммуникаций рядом
                Point(20, 20),    # HIGH без utility
                Point(25, 25),    # invalid risk class
                Point(10, 10),    # duplicate pair A
                Point(10.1, 10.1) # duplicate pair B
            ],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )

    utilities = gpd.GeoDataFrame(
        {
            "type": ["unknown", "gas"],
            "geometry": [
                LineString([(-1, -1), (1, 1)]),
                Point(),  # пустая геометрия для R09
            ],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )

    issues = validate_anomalies_detailed(anomalies, utilities, buffer_dist=2.5)
    rule_ids = {issue.rule_id for issue in issues}

    expected = {"R01", "R02", "R03", "R05", "R06", "R07", "R09", "R10", "R11", "R12"}
    assert expected.issubset(rule_ids)

    text_issues = [issue.to_text() for issue in issues]
    summary = summarize_issues_by_rule(text_issues)
    for rule_id in expected:
        assert summary.get(rule_id, 0) >= 1
