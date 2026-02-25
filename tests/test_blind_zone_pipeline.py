import numpy as np

from src.pipeline.service import _detect_blind_zones


def test_detect_blind_zones_returns_polygons_for_high_noise() -> None:
    grid = np.zeros((20, 20), dtype=float)
    grid[5:12, 6:14] = 80.0

    zones = _detect_blind_zones(grid, target_crs="EPSG:32637")
    assert not zones.empty
    assert zones.crs.to_string() == "EPSG:32637"


def test_detect_blind_zones_handles_empty_grid() -> None:
    grid = np.array([])
    zones = _detect_blind_zones(grid, target_crs="EPSG:32637")
    assert zones.empty
