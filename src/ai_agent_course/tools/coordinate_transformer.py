from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from shapely.geometry import LineString, Point


@dataclass(slots=True)
class AffineTransformResult:
    matrix: np.ndarray
    rms_error_m: float


def fit_affine_transform(
    source_points: Iterable[tuple[float, float]],
    target_points: Iterable[tuple[float, float]],
) -> AffineTransformResult:
    src = np.asarray(list(source_points), dtype=float)
    dst = np.asarray(list(target_points), dtype=float)
    if len(src) < 3 or len(dst) < 3 or len(src) != len(dst):
        # Недостаточно опорных точек — используем тождественное преобразование.
        return AffineTransformResult(matrix=np.array([[1, 0, 0], [0, 1, 0]]), rms_error_m=0.0)

    # x' = a*x + b*y + c; y' = d*x + e*y + f
    ones = np.ones((src.shape[0], 1), dtype=float)
    design = np.hstack([src, ones])
    params_x, _, _, _ = np.linalg.lstsq(design, dst[:, 0], rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(design, dst[:, 1], rcond=None)
    matrix = np.vstack([params_x, params_y])

    pred_x = design @ params_x
    pred_y = design @ params_y
    residual = np.sqrt((pred_x - dst[:, 0]) ** 2 + (pred_y - dst[:, 1]) ** 2)
    rms = float(np.sqrt(np.mean(residual**2)))
    return AffineTransformResult(matrix=matrix, rms_error_m=rms)


def transform_xy(x: float, y: float, matrix: np.ndarray) -> tuple[float, float]:
    tx = float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2])
    ty = float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])
    return tx, ty


def transform_linestring(line: LineString, matrix: np.ndarray) -> LineString:
    coords = [transform_xy(float(x), float(y), matrix) for x, y in line.coords]
    return LineString(coords)


def transform_point(point: Point, matrix: np.ndarray) -> Point:
    tx, ty = transform_xy(float(point.x), float(point.y), matrix)
    return Point(tx, ty)

