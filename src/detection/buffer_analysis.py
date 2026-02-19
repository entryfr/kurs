# src/detection/buffer_analysis.py
import geopandas as gpd
from shapely.geometry import Point, LineString
from config.settings import BUFFER_DISTANCE

def create_buffer_zones(anomalies_gdf, buffer_dist=BUFFER_DISTANCE):
    """
    Создаёт буферные зоны вокруг аномалий (точек или линий).
    anomalies_gdf: GeoDataFrame с геометрией аномалий
    Возвращает GeoDataFrame с буферными полигонами.
    """
    # Копируем, чтобы не изменять исходный
    zones = anomalies_gdf.copy()
    zones['geometry'] = zones.geometry.buffer(buffer_dist)
    return zones

def check_utilities_in_buffer(anomalies_gdf, utilities_gdf, buffer_dist=BUFFER_DISTANCE):
    """
    Проверяет, попадают ли существующие коммуникации в буферные зоны аномалий.
    Возвращает DataFrame с аномалиями и флагом overlaps, а также список ID коммуникаций.
    """
    # Пространственное соединение: для каждой аномалии ищем коммуникации в буфере
    # Сначала создаём буферы
    buffers = anomalies_gdf.copy()
    buffers['geometry'] = buffers.geometry.buffer(buffer_dist)

    # Выполняем пространственный джойн (intersects)
    joined = gpd.sjoin(buffers, utilities_gdf, how='left', predicate='intersects')

    # Группируем по индексу аномалии, собираем ID коммуникаций
    result = anomalies_gdf.copy()
    result['utilities_in_buffer'] = joined.groupby(joined.index).apply(
        lambda x: list(x['index_right'].dropna().unique())
    )
    result['has_utility'] = result['utilities_in_buffer'].apply(len) > 0
    return result