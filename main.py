import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from src.agent.validator import validate_anomalies
from src.classification.risk_classifier import RiskClassifier
from src.classification.utility_type_identifier import UtilityTypeIdentifier
from src.detection.anomaly_detector import detect_linear_anomalies
from src.reporting.act_generator import generate_act
from src.reporting.dxf_exporter import export_to_dxf
from src.reporting.miis_exporter import create_miis_xml

logger = logging.getLogger(__name__)

DEFAULT_IDENTIFIER_FEATURES: dict[str, float | int] = {
    "amplitude_nt": 50.0,
    "gradient_rho": 0.5,
    "depth_vez_m": 3.0,
    "radar_hyperbola_w": 1.2,
    "linear_extent_m": 10.0,
    "rho_value": 100.0,
    "has_thermal_anomaly": 0,
    "depth_to_diameter": 5.0,
    "magnetic_gradient": 2.0,
    "seg_velocity": 0.1,
    "near_road": 1,
    "area_type": 0,
    "age_infrastructure": 20.0,
    "soil_moisture": 0.3,
}


def _load_magnetic_grid(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Файл магнитной сетки не найден: {path}")
    magnetic_grid = np.load(path)
    if magnetic_grid.ndim != 2:
        raise ValueError(
            f"Ожидался 2D массив магнитной сетки, получено shape={magnetic_grid.shape}"
        )
    return magnetic_grid


def _load_geodataframe(path: Path, dataset_name: str) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл '{dataset_name}' не найден: {path}")
    gdf = gpd.read_file(path)
    if "geometry" not in gdf.columns:
        raise ValueError(f"Файл '{dataset_name}' не содержит геометрию: {path}")
    if gdf.crs is None:
        raise ValueError(f"У '{dataset_name}' отсутствует CRS: {path}")
    return gdf


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        if np.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_identifier_features(row: gpd.GeoSeries) -> dict[str, float | int]:
    features = dict(DEFAULT_IDENTIFIER_FEATURES)

    geometry = row.get("geometry")
    if geometry is not None and not geometry.is_empty and hasattr(geometry, "length"):
        features["linear_extent_m"] = max(float(geometry.length), 0.0)

    if "intensity" in row:
        features["amplitude_nt"] = _safe_float(row.get("intensity"), features["amplitude_nt"])

    if "has_thermal_anomaly" in row:
        thermal = row.get("has_thermal_anomaly")
        features["has_thermal_anomaly"] = 1 if thermal else 0

    if "rho_value" in row:
        features["rho_value"] = _safe_float(row.get("rho_value"), features["rho_value"])

    return features


def _build_risk_features(row: gpd.GeoSeries, utility_type: str) -> dict[str, Any]:
    return {
        "excavation_depth": 3.0,
        "anomaly_depth": _safe_float(row.get("depth"), 2.0),
        "utility_type": utility_type,
        "detection_factor": 0.7,
    }


def run_pipeline(
    magnetic_grid_path: Path,
    anomalies_path: Path,
    utilities_path: Path,
    output_dir: Path,
    anomaly_threshold: float,
) -> dict[str, Any]:
    logger.info("Загрузка входных данных...")
    magnetic_grid = _load_magnetic_grid(magnetic_grid_path)
    anomalies_gdf = _load_geodataframe(anomalies_path, "аномалии")
    utilities_gdf = _load_geodataframe(utilities_path, "коммуникации")

    logger.info("Детекция аномалий...")
    detected = detect_linear_anomalies(magnetic_grid, threshold_nt=anomaly_threshold)
    logger.info("Найдено аномалий по магнитной сетке: %s", len(detected))

    classifier = RiskClassifier()
    identifier = UtilityTypeIdentifier()
    result_gdf = anomalies_gdf.copy()

    logger.info("Определение типа коммуникаций...")
    result_gdf["utility_type"] = result_gdf.apply(
        lambda row: identifier.predict(_build_identifier_features(row)), axis=1
    )

    logger.info("Классификация риска...")
    result_gdf["risk_class"] = result_gdf.apply(
        lambda row: classifier.predict_risk(
            _build_risk_features(row, row["utility_type"])
        ),
        axis=1,
    )

    logger.info("Нормативная валидация...")
    errors = validate_anomalies(result_gdf, utilities_gdf)
    if errors:
        for issue in errors:
            logger.warning(issue)

    output_dir.mkdir(parents=True, exist_ok=True)
    dxf_path = output_dir / "scheme.dxf"
    xml_path = output_dir / "miis.xml"
    act_path = output_dir / "act.docx"

    logger.info("Экспорт результатов...")
    export_to_dxf(result_gdf, utilities_gdf, result_gdf, str(dxf_path))
    create_miis_xml(result_gdf, utilities_gdf, str(xml_path))
    generate_act(
        {"date": date.today().isoformat(), "anomalies": len(result_gdf)},
        output_path=str(act_path),
    )

    return {
        "detected_count": len(detected),
        "anomalies_count": len(result_gdf),
        "validation_issues": len(errors),
        "dxf_path": str(dxf_path),
        "xml_path": str(xml_path),
        "act_path": str(act_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Демо-пайплайн анализа геофизических данных подземных коммуникаций."
    )
    parser.add_argument(
        "--magnetic-grid",
        default="data/raw/magnetic_grid.npy",
        help="Путь к файлу .npy с магнитной сеткой.",
    )
    parser.add_argument(
        "--anomalies",
        default="data/processed/anomalies.gpkg",
        help="Путь к GeoPackage с аномалиями.",
    )
    parser.add_argument(
        "--utilities",
        default="data/processed/utilities.gpkg",
        help="Путь к GeoPackage с учтёнными коммуникациями.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Каталог для выходных файлов DXF/XML/DOCX.",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=30.0,
        help="Порог детекции аномалий в нТл.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args()
    summary = run_pipeline(
        magnetic_grid_path=Path(args.magnetic_grid),
        anomalies_path=Path(args.anomalies),
        utilities_path=Path(args.utilities),
        output_dir=Path(args.output_dir),
        anomaly_threshold=args.anomaly_threshold,
    )

    logger.info("Пайплайн завершён: %s", summary)
    print("Готово. Выходные файлы:")
    print(f"  - {summary['dxf_path']}")
    print(f"  - {summary['xml_path']}")
    print(f"  - {summary['act_path']}")


if __name__ == "__main__":
    main()