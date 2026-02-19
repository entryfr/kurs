# src/reporting/miis_exporter.py
from lxml import etree
import geopandas as gpd
from shapely.geometry import mapping
import json

def create_miis_xml(anomalies_gdf, utilities_gdf, output_path='miis_output.xml', schema_path=None):
    """
    Генерирует XML-файл в формате МИИС.
    Структура (упрощённо):
    <MIIS>
        <Anomalies>
            <Anomaly id="..." type="..." risk="...">
                <Geometry>...</Geometry>
            </Anomaly>
        </Anomalies>
        <Utilities>
            <Utility id="..." type="...">
                <Geometry>...</Geometry>
            </Utility>
        </Utilities>
    </MIIS>
    """
    # Корневой элемент
    root = etree.Element('MIIS', version='1.0')

    # Аномалии
    anomalies_elem = etree.SubElement(root, 'Anomalies')
    for idx, row in anomalies_gdf.iterrows():
        anom_elem = etree.SubElement(anomalies_elem, 'Anomaly',
                                     id=str(idx),
                                     type=row.get('type', 'unknown'),
                                     risk=row.get('risk_class', 'LOW'))
        geom_elem = etree.SubElement(anom_elem, 'Geometry')
        # Преобразуем shapely-геометрию в WKT или GML. Используем WKT для простоты
        geom_elem.text = row.geometry.wkt

    # Коммуникации
    utilities_elem = etree.SubElement(root, 'Utilities')
    for idx, row in utilities_gdf.iterrows():
        util_elem = etree.SubElement(utilities_elem, 'Utility',
                                     id=str(idx),
                                     type=row.get('type', 'unknown'))
        geom_elem = etree.SubElement(util_elem, 'Geometry')
        geom_elem.text = row.geometry.wkt

    # Запись в файл
    tree = etree.ElementTree(root)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='utf-8')

    # Валидация по XSD, если указан путь к схеме
    if schema_path:
        with open(schema_path, 'rb') as f:
            schema_doc = etree.parse(f)
            schema = etree.XMLSchema(schema_doc)
            if not schema.validate(root):
                raise ValueError("XML не прошёл валидацию по XSD-схеме")

    print(f"МИИС XML сохранён в {output_path}")