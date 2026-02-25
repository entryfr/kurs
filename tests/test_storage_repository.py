from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString

from src.storage.repository import (
    get_persisted_run,
    list_persisted_runs,
    persist_pipeline_run,
)


def _sample_anomalies() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "risk_class": ["CRITICAL", "HIGH"],
            "risk_probability": [0.89, 0.62],
            "utility_type": ["gas", "water"],
            "depth": [2.2, 1.8],
            "in_blind_zone": [True, False],
            "corrosion_critical": [True, False],
            "geometry": [
                LineString([(0, 0), (15, 0)]),
                LineString([(10, 10), (20, 10)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )


def test_persist_and_read_pipeline_run_sqlite(tmp_path: Path) -> None:
    db_file = tmp_path / "pipeline.db"
    sqlite_url = f"sqlite:///{db_file}"
    summary = {
        "norms_profile": "normative",
        "input_mode": "vector_layers",
        "anomalies_count": 2,
        "validation_issues": 1,
    }

    persisted = persist_pipeline_run(summary, _sample_anomalies(), database_url=sqlite_url)
    assert persisted["enabled"] is True
    assert persisted["status"] == "persisted"
    assert persisted["backend"] == "sqlite"
    run_uid = persisted["run_uid"]

    listing = list_persisted_runs(limit=10, database_url=sqlite_url)
    assert len(listing) == 1
    assert listing[0]["run_uid"] == run_uid
    assert listing[0]["anomalies_count"] == 2

    details = get_persisted_run(run_uid, database_url=sqlite_url)
    assert details is not None
    assert details["run_uid"] == run_uid
    assert len(details["anomalies"]) == 2
    assert details["anomalies"][0]["risk_class"] in {"CRITICAL", "HIGH"}
