import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString

# Генерируем синтетическую магнитную сетку (100x100)
np.save('data/raw/magnetic_grid.npy', np.random.rand(100, 100) * 100)

# Генерируем примеры аномалий (для теста)
anomalies = gpd.GeoDataFrame({
    'geometry': [Point(50, 50), Point(20, 30), LineString([(10,10), (20,20)])],
    'risk_class': ['CRITICAL', 'HIGH', 'LOW']
}, crs='EPSG:28406')
anomalies.to_file('data/processed/anomalies.gpkg', driver='GPKG')

# Генерируем примеры существующих коммуникаций
utilities = gpd.GeoDataFrame({
    'geometry': [LineString([(48,48), (52,52)]), LineString([(15,25), (25,35)])],
    'type': ['gas', 'water']
}, crs='EPSG:28406')
utilities.to_file('data/processed/utilities.gpkg', driver='GPKG')

print("Тестовые данные созданы.")