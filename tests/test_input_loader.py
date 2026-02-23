from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.preprocessing.input_loader import (
    load_anomalies_and_utilities,
    load_magnetic_grid,
    load_miis_xml,
)


def test_load_magnetic_grid_from_npy(tmp_path: Path) -> None:
    grid = np.array([[1.0, 2.0], [3.0, 4.0]])
    npy_path = tmp_path / "grid.npy"
    np.save(npy_path, grid)

    loaded = load_magnetic_grid(npy_path)
    np.testing.assert_allclose(loaded, grid)


def test_load_magnetic_grid_from_geotiff(tmp_path: Path) -> None:
    tif_path = tmp_path / "grid.tif"
    data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data, 1)

    loaded = load_magnetic_grid(tif_path)
    np.testing.assert_allclose(loaded, data)


def test_load_miis_xml(tmp_path: Path) -> None:
    xml_path = tmp_path / "miis.xml"
    xml_path.write_text(
        """<?xml version='1.0' encoding='UTF-8'?>
<MIIS version="1.0" crs="EPSG:32637">
  <Anomalies>
    <Anomaly id="1" type="unknown" risk="HIGH">
      <Geometry>POINT (10 20)</Geometry>
    </Anomaly>
  </Anomalies>
  <Utilities>
    <Utility id="u1" type="gas">
      <Geometry>LINESTRING (0 0, 5 5)</Geometry>
    </Utility>
  </Utilities>
</MIIS>
""",
        encoding="utf-8",
    )

    anomalies_gdf, utilities_gdf = load_miis_xml(xml_path)
    assert len(anomalies_gdf) == 1
    assert len(utilities_gdf) == 1
    assert anomalies_gdf.iloc[0]["risk_class"] == "HIGH"
    assert utilities_gdf.iloc[0]["type"] == "gas"


def test_load_anomalies_and_utilities_prefers_miis(tmp_path: Path) -> None:
    xml_path = tmp_path / "miis.xml"
    xml_path.write_text(
        """<?xml version='1.0' encoding='UTF-8'?>
<MIIS version="1.0">
  <Anomalies>
    <Anomaly id="1" type="unknown" risk="LOW">
      <Geometry>POINT (1 1)</Geometry>
    </Anomaly>
  </Anomalies>
  <Utilities>
    <Utility id="1" type="water">
      <Geometry>LINESTRING (0 0, 1 1)</Geometry>
    </Utility>
  </Utilities>
</MIIS>
""",
        encoding="utf-8",
    )

    anomalies_gdf, utilities_gdf, input_mode = load_anomalies_and_utilities(
        anomalies_path=None,
        utilities_path=None,
        miis_xml_path=xml_path,
    )
    assert input_mode == "miis_xml"
    assert len(anomalies_gdf) == 1
    assert len(utilities_gdf) == 1
