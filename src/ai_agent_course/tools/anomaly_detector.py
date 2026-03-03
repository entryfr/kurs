from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class DetectedAnomaly:
    x1: float
    y1: float
    x2: float
    y2: float
    length_m: float
    amplitude_nt: float
    midpoint_x_m: float
    midpoint_y_m: float


def _line_intensity(gray: np.ndarray, line: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = line
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=3)
    values = gray[mask > 0]
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def detect_linear_anomalies_from_image(
    image: np.ndarray,
    pixel_size_m: float = 5.0,
    min_length_m: float = 10.0,
) -> list[DetectedAnomaly]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = gray.astype(np.uint8)

    filtered = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(filtered, threshold1=60, threshold2=160)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=max(int(min_length_m / pixel_size_m), 2),
        maxLineGap=4,
    )

    results: list[DetectedAnomaly] = []
    if lines is None:
        return results

    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in line]
        length_px = float(np.hypot(x2 - x1, y2 - y1))
        length_m = length_px * pixel_size_m
        if length_m <= min_length_m:
            continue
        intensity = _line_intensity(gray, (x1, y1, x2, y2))

        # Преобразуем условную яркость изображения в нТл шкалу (0..255 -> 0..140 нТл).
        amplitude_nt = float((intensity / 255.0) * 140.0)
        mid_x_m = ((x1 + x2) / 2.0) * pixel_size_m
        mid_y_m = ((y1 + y2) / 2.0) * pixel_size_m
        results.append(
            DetectedAnomaly(
                x1=float(x1) * pixel_size_m,
                y1=float(y1) * pixel_size_m,
                x2=float(x2) * pixel_size_m,
                y2=float(y2) * pixel_size_m,
                length_m=length_m,
                amplitude_nt=amplitude_nt,
                midpoint_x_m=mid_x_m,
                midpoint_y_m=mid_y_m,
            )
        )

    return results

