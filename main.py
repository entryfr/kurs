import argparse
import numpy as np
import geopandas as gpd

from src.preprocessing.coordinate_transformer import transform_geometry
from src.detection.anomaly_detector import detect_linear_anomalies
from src.classification.risk_classifier import RiskClassifier
from src.classification.utility_type_identifier import UtilityTypeIdentifier
from src.classification.corrosion_estimator import assess_corrosion
from src.reporting.act_generator import generate_act
from src.reporting.dxf_exporter import export_to_dxf
from src.reporting.miis_exporter import create_miis_xml
from src.agent.validator import validate_anomalies

def main(xml_path, geotiff_path, segy_path):
    print("Загрузка синтетических данных...")
    magnetic_grid = np.load('data/raw/magnetic_grid.npy')
    anomalies_gdf = gpd.read_file('data/processed/anomalies.gpkg')
    utilities_gdf = gpd.read_file('data/processed/utilities.gpkg')

    print("Детекция аномалий...")
    detected = detect_linear_anomalies(magnetic_grid)
    print(f"Найдено {len(detected)} аномалий")

    # Преобразуем в GeoDataFrame (если detected содержит геометрию)
    # Для теста пропустим этот шаг, используем готовые аномалии

    print("Классификация риска...")
    classifier = RiskClassifier()
    anomalies_gdf['risk_class'] = anomalies_gdf.apply(
        lambda row: classifier.predict_risk({'anomaly_depth': 2.0, 'utility_type': 'gas'}), axis=1
    )

    print("Определение типа коммуникации...")
    identifier = UtilityTypeIdentifier()
    anomalies_gdf['utility_type'] = anomalies_gdf.apply(
        lambda row: identifier.predict({
            'amplitude_nt': 50, 'gradient_rho': 0.5, 'depth_vez_m': 3.0,
            'radar_hyperbola_w': 1.2, 'linear_extent_m': 10, 'rho_value': 100,
            'has_thermal_anomaly': 0, 'depth_to_diameter': 5, 'magnetic_gradient': 2,
            'seg_velocity': 0.1, 'near_road': 1, 'area_type': 0, 'age_infrastructure': 20,
            'soil_moisture': 0.3
        }), axis=1
    )

    print("Валидация...")
    errors = validate_anomalies(anomalies_gdf, utilities_gdf)
    if errors:
        print("Найдены ошибки:")
        for e in errors:
            print(f"  - {e}")

    print("Генерация отчётов...")
    export_to_dxf(anomalies_gdf, utilities_gdf, anomalies_gdf, 'output/scheme.dxf')
    create_miis_xml(anomalies_gdf, utilities_gdf, 'output/miis.xml')
    generate_act({'anomalies': len(anomalies_gdf)}, output_path='output/act.docx')

    print("Готово! Результаты в папке output/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', default='dummy.xml')
    parser.add_argument('--tiff', default='dummy.tiff')
    parser.add_argument('--segy', default='dummy.segy')
    args = parser.parse_args()
    main(args.xml, args.tiff, args.segy)