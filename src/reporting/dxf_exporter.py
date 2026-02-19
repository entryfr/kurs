# src/reporting/dxf_exporter.py
import ezdxf
from ezdxf.math import Vec2
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

def export_to_dxf(anomalies_gdf, utilities_gdf, zones_gdf, output_path='scheme.dxf'):
    """
    Создаёт DXF-файл с несколькими слоями:
    - ANOMALIES (точки/линии аномалий)
    - UTILITIES (линии коммуникаций)
    - ZONES (полигоны буферных зон)
    """
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Создаём слои
    doc.layers.new('ANOMALIES', dxfattribs={'color': 1})  # красный
    doc.layers.new('UTILITIES', dxfattribs={'color': 3})  # зелёный
    doc.layers.new('ZONES', dxfattribs={'color': 5})      # синий

    # Функция для добавления геометрии в DXF
    def add_geometry(geoseries, layer_name):
        for geom in geoseries:
            if geom is None:
                continue
            if isinstance(geom, Point):
                msp.add_point(Vec2(geom.x, geom.y), dxfattribs={'layer': layer_name})
            elif isinstance(geom, LineString):
                points = [Vec2(x, y) for x, y in geom.coords]
                msp.add_lwpolyline(points, dxfattribs={'layer': layer_name})
            elif isinstance(geom, Polygon):
                # Внешний контур
                outer = [Vec2(x, y) for x, y in geom.exterior.coords]
                msp.add_lwpolyline(outer, close=True, dxfattribs={'layer': layer_name})
                # Можно добавить отверстия, но для простоты пропустим

    add_geometry(anomalies_gdf.geometry, 'ANOMALIES')
    add_geometry(utilities_gdf.geometry, 'UTILITIES')
    add_geometry(zones_gdf.geometry, 'ZONES')

    doc.saveas(output_path)
    print(f"DXF сохранён в {output_path}")