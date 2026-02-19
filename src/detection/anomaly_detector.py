import cv2
import numpy as np
from src.preprocessing.noise_filter import apply_filters

def detect_linear_anomalies(magnetic_grid, threshold_nt=30.0, min_length_px=2):
    # Предварительная фильтрация
    filtered = apply_filters(magnetic_grid)
    # Бинаризация по порогу
    mask = (filtered > threshold_nt).astype(np.uint8) * 255
    # Детекция границ Canny
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    # Поиск отрезков методом вероятностного Хафа
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                            threshold=10, minLineLength=min_length_px, maxLineGap=3)
    if lines is None:
        return []
    # Преобразуем в список словарей с координатами
    anomalies = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        anomalies.append({
            'coords': ((x1,y1),(x2,y2)),
            'length': np.hypot(x2-x1, y2-y1),
            'mean_intensity': np.mean(magnetic_grid[y1:y2+1, x1:x2+1])  # пример
        })
    return anomalies