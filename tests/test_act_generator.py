from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString

from src.pipeline.service import _build_act_anomaly_rows
from src.reporting.act_generator import generate_act


def test_build_act_anomaly_rows_contains_tz_recommendations() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "risk_class": ["CRITICAL", "HIGH", "LOW"],
            "risk_probability": [0.89, 0.62, 0.12],
            "utility_type": ["gas", "water", "sewage"],
            "depth": [2.5, 4.0, 3.0],
            "linear_length_m": [42.0, 20.0, 12.0],
            "geometry": [
                LineString([(0, 0), (10, 0)]),
                LineString([(0, 1), (20, 1)]),
                LineString([(0, 2), (12, 2)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )

    rows = _build_act_anomaly_rows(gdf)
    assert rows[0]["recommendation"] == "Шурф №III-247"
    assert rows[1]["recommendation"] == "Георадар ЛОМА-5"
    assert rows[2]["recommendation"] == "Видеозондирование"


def test_generate_act_creates_docx_and_optional_pdf(tmp_path: Path) -> None:
    output_docx = tmp_path / "act.docx"
    output_pdf = tmp_path / "act.pdf"

    result = generate_act(
        data={
            "date": "2026-02-19",
            "profile": "normative",
            "anomalies": 1,
            "issues": [],
            "anomaly_rows": [
                {
                    "anomaly_id": 1,
                    "risk_class": "CRITICAL",
                    "risk_probability": "89%",
                    "utility_type": "gas",
                    "depth_m": "2.00",
                    "length_m": "42.00",
                    "coordinates": "(0.0, 0.0) -> (10.0, 0.0)",
                    "recommendation": "Шурф №III-247",
                }
            ],
        },
        output_path=str(output_docx),
        pdf_output_path=str(output_pdf),
    )

    assert output_docx.exists()
    assert result["docx_path"] == str(output_docx)
    if result["pdf_path"] is not None:
        assert Path(result["pdf_path"]).exists()
