# src/detection/blind_zones.py
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point

def find_blind_zones(magnetic_grid, x_coords, y_coords, threshold_nt=50.0, radius=50.0):
    """
    Определяет области, где магнитное поле > threshold_nt (сильные помехи)
    или где нет данных в радиусе radius (метров).
    magnetic_grid: 2D массив значений магнитного поля
    x_coords, y_coords: координаты (в проекции) для каждой ячейки сетки
    Возвращает GeoDataFrame с полигонами «слепых зон».
    """
    # Бинарная маска: ячейки, где значение превышает порог
    high_noise_mask = magnetic_grid > threshold_nt

    # Здесь можно применить морфологическое расширение (dilation), чтобы объединить близкие точки
    # Для простоты преобразуем маску в полигоны через контуры (opencv)
    import cv2
    mask_uint8 = high_noise_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        # Контур в пикселях -> координаты
        if len(cnt) >= 3:
            # Преобразуем пиксели в реальные координаты (приближённо)
            # В реальности нужно интерполировать по x_coords, y_coords
            # Для упрощения возьмём bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            # Координаты углов в метрах (грубо, используя шаг сетки)
            x_min = x_coords[x]
            x_max = x_coords[x + w] if x + w < len(x_coords) else x_coords[-1]
            y_min = y_coords[y]
            y_max = y_coords[y + h] if y + h < len(y_coords) else y_coords[-1]
            polygons.append(box(x_min, y_min, x_max, y_max))

    # Также можно добавить области отсутствия данных (но пока нет данных)
    return gpd.GeoDataFrame(geometry=polygons, crs='EPSG:28406')