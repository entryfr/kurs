import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from src.pipeline.service import _assess_corrosion_for_row


def test_assess_corrosion_for_row_marks_critical_case() -> None:
    utilities = gpd.GeoDataFrame(
        {
            "type": ["gas"],
            "diameter": [50.0],
            "age_infrastructure": [120.0],
            "geometry": [LineString([(0, 0), (1, 0)])],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )
    row = pd.Series(
        {
            "utilities_in_buffer": [0],
            "rho_value": 1.0,
            "soil_moisture": 1.0,
            "ph": 4.0,
        }
    )

    result = _assess_corrosion_for_row(row, utilities)
    assert result["corrosion_critical"] is True
    assert isinstance(result["corrosion_warning"], str)
    assert "срочная замена" in result["corrosion_warning"].lower()


def test_assess_corrosion_for_row_skips_non_metal_types() -> None:
    utilities = gpd.GeoDataFrame(
        {
            "type": ["communication"],
            "diameter": [60.0],
            "geometry": [LineString([(0, 0), (1, 0)])],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )
    row = pd.Series({"utilities_in_buffer": [0]})

    result = _assess_corrosion_for_row(row, utilities)
    assert result["corrosion_critical"] is False
    assert result["corrosion_warning"] is None
