from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from config import settings
from src.agent.validator import (
    check_compliance,
    generate_validation_report,
    validate_anomalies,
)
from src.classification.risk_classifier import RiskClassifier
from src.classification.utility_type_identifier import UtilityTypeIdentifier
from src.detection.anomaly_detector import detect_linear_anomalies
from src.preprocessing.input_loader import (
    load_anomalies_and_utilities,
    load_magnetic_grid,
)
from src.preprocessing.segy_reader import extract_segy_features
from src.reporting.act_generator import generate_act
from src.reporting.dxf_exporter import export_to_dxf
from src.reporting.miis_exporter import create_miis_xml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        if np.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_identifier_features(
    row: gpd.GeoSeries, segy_features: dict[str, float] | None = None
) -> dict[str, float | int]:
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

    if segy_features:
        features["radar_hyperbola_w"] = _safe_float(
            segy_features.get("radar_hyperbola_w"), features["radar_hyperbola_w"]
        )
        features["seg_velocity"] = _safe_float(
            segy_features.get("seg_velocity"), features["seg_velocity"]
        )

    return features


def _build_risk_features(
    row: gpd.GeoSeries, utility_type: str, segy_features: dict[str, float] | None = None
) -> dict[str, Any]:
    detection_factor = 0.7
    if segy_features and "segy_confidence" in segy_features:
        detection_factor = float(np.clip(segy_features["segy_confidence"], 0.1, 0.99))

    return {
        "excavation_depth": 3.0,
        "anomaly_depth": _safe_float(row.get("depth"), 2.0),
        "utility_type": utility_type,
        "detection_factor": detection_factor,
    }


def _risk_confidence(risk_class: str, row: gpd.GeoSeries) -> float:
    base = {"CRITICAL": 0.9, "HIGH": 0.75, "LOW": 0.6}.get(risk_class, 0.5)
    intensity_boost = 0.0
    if "intensity" in row:
        intensity = _safe_float(row.get("intensity"), 0.0)
        intensity_boost = min(intensity / 500.0, 0.15)
    if row.get("has_utility") is True:
        intensity_boost += 0.05
    return float(np.clip(base + intensity_boost, 0.05, 0.99))


def _build_risk_explanations(result_gdf: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    explanations: list[dict[str, Any]] = []
    for idx, row in result_gdf.iterrows():
        risk_class = row.get("risk_class", "LOW")
        utility_type = row.get("utility_type", "unknown")
        has_utility = bool(row.get("has_utility", False))
        depth = row.get("depth")

        reasons: list[str] = [f"Предсказанный тип коммуникации: {utility_type}."]
        if depth is not None and not np.isnan(depth):
            reasons.append(f"Оценка глубины аномалии: {float(depth):.2f} м.")
        if has_utility:
            reasons.append("В буфере найдена учтённая коммуникация.")
        else:
            reasons.append("В буфере нет подтверждённой коммуникации.")

        explanations.append(
            {
                "anomaly_id": int(idx),
                "risk_class": risk_class,
                "utility_type": utility_type,
                "confidence": _risk_confidence(risk_class, row),
                "reasons": reasons,
            }
        )
    return explanations


def run_pipeline(
    magnetic_grid_path: Path,
    anomalies_path: Path,
    utilities_path: Path,
    output_dir: Path,
    anomaly_threshold: float,
    miis_xml_path: Path | None = None,
    segy_path: Path | None = None,
    norms_profile: str = settings.DEFAULT_NORMS_PROFILE,
) -> dict[str, Any]:
    if norms_profile not in settings.NORMATIVE_PROFILES:
        raise ValueError(
            f"Неизвестный профиль норм: {norms_profile}. "
            f"Доступно: {list(settings.NORMATIVE_PROFILES.keys())}"
        )
    profile = settings.NORMATIVE_PROFILES[norms_profile]

    logger.info("Загрузка входных данных...")
    magnetic_grid = load_magnetic_grid(magnetic_grid_path)
    anomalies_gdf, utilities_gdf, input_mode = load_anomalies_and_utilities(
        anomalies_path=anomalies_path,
        utilities_path=utilities_path,
        miis_xml_path=miis_xml_path,
    )

    segy_features: dict[str, float] = {}
    if segy_path is not None:
        logger.info("Извлечение признаков из SEG-Y...")
        segy_features = extract_segy_features(str(segy_path))
        logger.info("SEG-Y признаки: %s", segy_features)

    logger.info("Детекция аномалий...")
    detected = detect_linear_anomalies(magnetic_grid, threshold_nt=anomaly_threshold)
    logger.info("Найдено аномалий по магнитной сетке: %s", len(detected))

    classifier = RiskClassifier()
    identifier = UtilityTypeIdentifier()
    result_gdf = anomalies_gdf.copy()

    logger.info("Определение типа коммуникаций...")
    result_gdf["utility_type"] = result_gdf.apply(
        lambda row: identifier.predict(_build_identifier_features(row, segy_features)),
        axis=1,
    )

    logger.info("Классификация риска...")
    result_gdf["risk_class"] = result_gdf.apply(
        lambda row: classifier.predict_risk(
            _build_risk_features(row, row["utility_type"], segy_features)
        ),
        axis=1,
    )

    logger.info("Нормативная валидация...")
    validation_errors = validate_anomalies(
        result_gdf,
        utilities_gdf,
        buffer_dist=float(profile["buffer_distance"]),
    )
    compliance_issues = check_compliance(result_gdf, utilities_gdf)
    all_issues = validation_errors + compliance_issues
    validation_report = generate_validation_report(validation_errors, compliance_issues)

    if all_issues:
        for issue in all_issues:
            logger.warning(issue)

    output_dir.mkdir(parents=True, exist_ok=True)
    dxf_path = output_dir / "scheme.dxf"
    xml_path = output_dir / "miis.xml"
    act_path = output_dir / "act.docx"

    logger.info("Экспорт результатов...")
    export_to_dxf(result_gdf, utilities_gdf, result_gdf, str(dxf_path))
    create_miis_xml(result_gdf, utilities_gdf, str(xml_path))
    generate_act(
        {
            "date": date.today().isoformat(),
            "anomalies": len(result_gdf),
            "issues": all_issues,
            "profile": norms_profile,
        },
        output_path=str(act_path),
    )

    risk_details = _build_risk_explanations(result_gdf)

    return {
        "input_mode": input_mode,
        "norms_profile": norms_profile,
        "detected_count": len(detected),
        "anomalies_count": len(result_gdf),
        "validation_issues": len(all_issues),
        "validation_issue_details": all_issues,
        "validation_report": validation_report,
        "risk_details": risk_details,
        "segy_features": segy_features,
        "dxf_path": str(dxf_path),
        "xml_path": str(xml_path),
        "act_path": str(act_path),
    }
