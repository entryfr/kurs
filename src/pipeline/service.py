from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from config import settings
from src.agent.validator import (
    check_compliance,
    generate_validation_report,
    summarize_issues_by_rule,
    validate_anomalies,
)
from src.classification.risk_classifier import RiskClassifier
from src.classification.utility_type_identifier import UtilityTypeIdentifier
from src.classification.corrosion_estimator import assess_corrosion
from src.detection.anomaly_detector import detect_linear_anomalies
from src.detection.blind_zones import find_blind_zones
from src.detection.buffer_analysis import check_utilities_in_buffer
from src.preprocessing.input_loader import (
    load_anomalies_and_utilities,
    load_magnetic_grid,
    load_vector_layer,
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
METALLIC_UTILITY_TYPES = {"gas", "water", "heating", "sewage", "unknown_pipeline"}


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
    row: gpd.GeoSeries,
    utility_type: str,
    segy_features: dict[str, float] | None = None,
    excavation_depth: float = 3.0,
) -> dict[str, Any]:
    detection_factor = 0.7
    if segy_features and "segy_confidence" in segy_features:
        detection_factor = float(np.clip(segy_features["segy_confidence"], 0.1, 0.99))

    return {
        "excavation_depth": excavation_depth,
        "anomaly_depth": _safe_float(row.get("depth"), 2.0),
        "utility_type": utility_type,
        "detection_factor": detection_factor,
    }


def _risk_confidence(risk_class: str, row: gpd.GeoSeries) -> float:
    if "risk_probability" in row and row.get("risk_probability") is not None:
        prob = _safe_float(row.get("risk_probability"), 0.0)
        if prob > 0:
            return float(np.clip(prob, 0.05, 0.99))

    base = {"CRITICAL": 0.9, "HIGH": 0.75, "LOW": 0.6}.get(risk_class, 0.5)
    intensity_boost = 0.0
    if "intensity" in row:
        intensity = _safe_float(row.get("intensity"), 0.0)
        intensity_boost = min(intensity / 500.0, 0.15)
    if row.get("has_utility") is True:
        intensity_boost += 0.05
    return float(np.clip(base + intensity_boost, 0.05, 0.99))


def _build_anomaly_issue_map(issues: list[str]) -> dict[int, list[str]]:
    issue_map: dict[int, list[str]] = {}
    for issue in issues:
        if "Аномалия #" not in issue:
            continue
        match = None
        try:
            import re

            match = re.search(r"Аномалия #(\d+)", issue)
        except re.error:
            match = None
        if match is None:
            continue
        idx = int(match.group(1))
        issue_map.setdefault(idx, []).append(issue)
    return issue_map


def _load_construction_zones(
    construction_zones_path: Path | None,
    target_crs,
) -> gpd.GeoDataFrame:
    if construction_zones_path is None:
        return gpd.GeoDataFrame(
            columns=["geometry"],
            geometry="geometry",
            crs=target_crs,
        )

    zones = load_vector_layer(construction_zones_path, "зоны строительства")
    if zones.crs != target_crs:
        zones = zones.to_crs(target_crs)
    return zones


def _depth_value(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def _construction_depth_column(zones_gdf: gpd.GeoDataFrame) -> str | None:
    candidates = [
        "excavation_depth",
        "construction_depth",
        "depth_excavation",
        "planned_depth",
        "depth",
    ]
    lower_map = {str(col).lower(): col for col in zones_gdf.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _compute_zone_flags(
    anomalies_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
) -> tuple[dict[int, bool], dict[int, float | None]]:
    if zones_gdf.empty:
        return ({int(idx): False for idx in anomalies_gdf.index}, {int(idx): None for idx in anomalies_gdf.index})

    depth_col = _construction_depth_column(zones_gdf)
    in_zone: dict[int, bool] = {}
    excavation_depth: dict[int, float | None] = {}

    for idx, row in anomalies_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            in_zone[int(idx)] = False
            excavation_depth[int(idx)] = None
            continue

        intersections = zones_gdf[zones_gdf.intersects(geom)]
        in_zone[int(idx)] = not intersections.empty
        if intersections.empty or depth_col is None:
            excavation_depth[int(idx)] = None
            continue

        depths = [_depth_value(v) for v in intersections[depth_col].tolist()]
        valid_depths = [d for d in depths if d is not None]
        excavation_depth[int(idx)] = max(valid_depths) if valid_depths else None

    return in_zone, excavation_depth


def _detect_blind_zones(
    magnetic_grid: np.ndarray,
    target_crs,
) -> gpd.GeoDataFrame:
    if magnetic_grid.size == 0:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)

    height, width = magnetic_grid.shape
    x_coords = np.arange(width, dtype=float)
    y_coords = np.arange(height, dtype=float)
    zones = find_blind_zones(
        magnetic_grid=magnetic_grid,
        x_coords=x_coords,
        y_coords=y_coords,
        threshold_nt=settings.BLIND_ZONE_THRESHOLD,
        radius=settings.BLIND_ZONE_RADIUS,
    )

    if zones.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)
    if zones.crs != target_crs:
        zones = zones.to_crs(target_crs)
    return zones


def _extract_nearby_depths(
    utility_indexes: list[int],
    utilities_gdf: gpd.GeoDataFrame,
) -> list[float]:
    if not utility_indexes or "depth" not in utilities_gdf.columns:
        return []
    depths: list[float] = []
    for utility_idx in utility_indexes:
        if utility_idx not in utilities_gdf.index:
            continue
        depth = _depth_value(utilities_gdf.loc[utility_idx].get("depth"))
        if depth is not None:
            depths.append(depth)
    return depths


def _classify_risk_tz(
    row: gpd.GeoSeries,
    utilities_gdf: gpd.GeoDataFrame,
    model_risk_class: str,
    model_probability: float,
    in_construction_zone: bool,
) -> tuple[str, float, str, float | None]:
    utility_indexes = row.get("utilities_in_buffer", [])
    if not isinstance(utility_indexes, list):
        utility_indexes = []
    has_utility = bool(utility_indexes)
    anomaly_depth = _depth_value(row.get("depth"))

    if in_construction_zone and not has_utility:
        return ("CRITICAL", 0.89, "TZ_RULE_CRITICAL_NO_MATCH_IN_CONSTRUCTION", None)

    nearby_depths = _extract_nearby_depths(utility_indexes, utilities_gdf)
    if has_utility and anomaly_depth is not None and nearby_depths:
        min_diff = min(abs(anomaly_depth - d) for d in nearby_depths)
        if min_diff > 0.5:
            return ("HIGH", 0.62, "TZ_RULE_DEPTH_MISMATCH", min_diff)
        return ("LOW", 0.12, "TZ_RULE_DEPTH_MATCH", min_diff)

    return (model_risk_class, model_probability, "MODEL_FALLBACK", None)


def _assess_corrosion_for_row(
    row: gpd.GeoSeries,
    utilities_gdf: gpd.GeoDataFrame,
) -> dict[str, Any]:
    utility_indexes = row.get("utilities_in_buffer", [])
    if not isinstance(utility_indexes, list) or not utility_indexes:
        return {
            "corrosion_rate": None,
            "remaining_life": None,
            "corrosion_critical": False,
            "corrosion_warning": None,
            "corrosion_utility_type": None,
        }

    candidates: list[dict[str, Any]] = []
    for utility_idx in utility_indexes:
        if utility_idx not in utilities_gdf.index:
            continue
        utility = utilities_gdf.loc[utility_idx]
        utility_type = str(utility.get("type", "unknown"))
        if utility_type not in METALLIC_UTILITY_TYPES:
            continue

        initial_diameter = _safe_float(utility.get("diameter"), 200.0)
        if initial_diameter <= 0:
            initial_diameter = 200.0

        corrosion_data = assess_corrosion(
            {
                "soil_rho": max(_safe_float(row.get("rho_value"), 100.0), 1e-3),
                "moisture": float(np.clip(_safe_float(row.get("soil_moisture"), 0.3), 0.0, 1.0)),
                "ph": _safe_float(row.get("ph"), 7.0),
                "initial_diameter": initial_diameter,
                "age": max(
                    _safe_float(
                        utility.get("age_infrastructure"),
                        _safe_float(row.get("age_infrastructure"), 20.0),
                    ),
                    0.0,
                ),
            }
        )
        remaining_life = corrosion_data.get("remaining_life")
        if isinstance(remaining_life, (int, float)) and np.isinf(remaining_life):
            remaining_life = None

        candidates.append(
            {
                "corrosion_rate": corrosion_data.get("corrosion_rate"),
                "remaining_life": remaining_life,
                "corrosion_critical": bool(corrosion_data.get("critical", False)),
                "corrosion_utility_type": utility_type,
            }
        )

    if not candidates:
        return {
            "corrosion_rate": None,
            "remaining_life": None,
            "corrosion_critical": False,
            "corrosion_warning": None,
            "corrosion_utility_type": None,
        }

    # Берём худший сценарий по минимальному остаточному ресурсу.
    def _sort_key(item: dict[str, Any]) -> float:
        life = item.get("remaining_life")
        return life if life is not None else 10_000.0

    selected = sorted(candidates, key=_sort_key)[0]
    warning = None
    if selected["corrosion_critical"]:
        warning = (
            "Требуется срочная замена во избежание аварии "
            f"(T_ост={selected['remaining_life']:.2f} лет)."
            if selected["remaining_life"] is not None
            else "Требуется срочная замена во избежание аварии (T_ост < 5 лет)."
        )

    selected["corrosion_warning"] = warning
    return selected


def _build_risk_explanations(
    result_gdf: gpd.GeoDataFrame, issue_map: dict[int, list[str]] | None = None
) -> list[dict[str, Any]]:
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
        if bool(row.get("in_blind_zone", False)):
            reasons.append("Аномалия попадает в слепую зону геофизики.")
        if bool(row.get("corrosion_critical", False)):
            reasons.append("Обнаружен критический прогноз остаточного ресурса коммуникации.")

        explanations.append(
            {
                "anomaly_id": int(idx),
                "risk_class": risk_class,
                "risk_probability": _safe_float(row.get("risk_probability"), _risk_confidence(risk_class, row)),
                "utility_type": utility_type,
                "confidence": _risk_confidence(risk_class, row),
                "risk_rule": row.get("risk_rule", "UNKNOWN"),
                "reasons": reasons,
                "norm_violations": issue_map.get(int(idx), []) if issue_map else [],
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
    construction_zones_path: Path | None = None,
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
    construction_zones_gdf = _load_construction_zones(
        construction_zones_path=construction_zones_path,
        target_crs=anomalies_gdf.crs,
    )

    if not utilities_gdf.empty and utilities_gdf.crs != anomalies_gdf.crs:
        utilities_gdf = utilities_gdf.to_crs(anomalies_gdf.crs)

    segy_features: dict[str, float] = {}
    if segy_path is not None:
        logger.info("Извлечение признаков из SEG-Y...")
        segy_features = extract_segy_features(str(segy_path))
        logger.info("SEG-Y признаки: %s", segy_features)

    logger.info("Детекция аномалий...")
    detected = detect_linear_anomalies(magnetic_grid, threshold_nt=anomaly_threshold)
    logger.info("Найдено аномалий по магнитной сетке: %s", len(detected))
    logger.info("Выделение слепых зон...")
    blind_zones_gdf = _detect_blind_zones(
        magnetic_grid=magnetic_grid,
        target_crs=anomalies_gdf.crs,
    )
    logger.info("Найдено слепых зон: %s", len(blind_zones_gdf))

    classifier = RiskClassifier()
    identifier = UtilityTypeIdentifier()
    result_gdf = anomalies_gdf.copy()

    logger.info("Буферный анализ коммуникаций...")
    result_gdf = check_utilities_in_buffer(
        result_gdf,
        utilities_gdf,
        buffer_dist=float(profile["buffer_distance"]),
    )
    result_gdf["utilities_in_buffer"] = result_gdf["utilities_in_buffer"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    in_zone_map, excavation_depth_map = _compute_zone_flags(
        anomalies_gdf=result_gdf,
        zones_gdf=construction_zones_gdf,
    )
    result_gdf["in_construction_zone"] = result_gdf.index.map(
        lambda idx: in_zone_map.get(int(idx), False)
    )
    if blind_zones_gdf.empty:
        result_gdf["in_blind_zone"] = False
    else:
        result_gdf["in_blind_zone"] = result_gdf.geometry.apply(
            lambda geom: bool(blind_zones_gdf.intersects(geom).any())
            if geom is not None and not geom.is_empty
            else False
        )

    logger.info("Определение типа коммуникаций...")
    result_gdf["utility_type"] = result_gdf.apply(
        lambda row: identifier.predict(_build_identifier_features(row, segy_features)),
        axis=1,
    )

    logger.info("Классификация риска...")
    result_gdf["risk_class_model"] = result_gdf.apply(
        lambda row: classifier.predict_risk(
            _build_risk_features(
                row,
                row["utility_type"],
                segy_features,
                excavation_depth=excavation_depth_map.get(int(row.name)) or 3.0,
            )
        ),
        axis=1,
    )
    model_probability_map = {"CRITICAL": 0.8, "HIGH": 0.65, "LOW": 0.2}

    tz_results = result_gdf.apply(
        lambda row: _classify_risk_tz(
            row=row,
            utilities_gdf=utilities_gdf,
            model_risk_class=row["risk_class_model"],
            model_probability=model_probability_map.get(row["risk_class_model"], 0.5),
            in_construction_zone=bool(row.get("in_construction_zone", False)),
        ),
        axis=1,
    )
    result_gdf["risk_class"] = tz_results.apply(lambda x: x[0])
    result_gdf["risk_probability"] = tz_results.apply(lambda x: x[1])
    result_gdf["risk_rule"] = tz_results.apply(lambda x: x[2])
    result_gdf["depth_mismatch_m"] = tz_results.apply(lambda x: x[3])

    logger.info("Оценка коррозионного ресурса...")
    corrosion_data = result_gdf.apply(
        lambda row: _assess_corrosion_for_row(row, utilities_gdf),
        axis=1,
    )
    result_gdf["corrosion_rate"] = corrosion_data.apply(lambda x: x.get("corrosion_rate"))
    result_gdf["remaining_life"] = corrosion_data.apply(lambda x: x.get("remaining_life"))
    result_gdf["corrosion_critical"] = corrosion_data.apply(lambda x: x.get("corrosion_critical", False))
    result_gdf["corrosion_warning"] = corrosion_data.apply(lambda x: x.get("corrosion_warning"))
    result_gdf["corrosion_utility_type"] = corrosion_data.apply(lambda x: x.get("corrosion_utility_type"))

    logger.info("Нормативная валидация...")
    validation_errors = validate_anomalies(
        result_gdf,
        utilities_gdf,
        buffer_dist=float(profile["buffer_distance"]),
    )
    compliance_issues = check_compliance(result_gdf, utilities_gdf)
    all_issues = validation_errors + compliance_issues
    validation_report = generate_validation_report(validation_errors, compliance_issues)
    issues_by_rule = summarize_issues_by_rule(all_issues)

    if all_issues:
        for issue in all_issues:
            logger.warning(issue)

    output_dir.mkdir(parents=True, exist_ok=True)
    dxf_path = output_dir / "scheme.dxf"
    xml_path = output_dir / "miis.xml"
    act_path = output_dir / "act.docx"

    logger.info("Экспорт результатов...")
    export_to_dxf(result_gdf, utilities_gdf, blind_zones_gdf, str(dxf_path))
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

    issue_map = _build_anomaly_issue_map(all_issues)
    risk_details = _build_risk_explanations(result_gdf, issue_map)
    corrosion_warnings = [
        f"Аномалия #{idx}: {warning}"
        for idx, warning in result_gdf["corrosion_warning"].items()
        if isinstance(warning, str) and warning.strip()
    ]

    return {
        "input_mode": input_mode,
        "norms_profile": norms_profile,
        "detected_count": len(detected),
        "anomalies_count": len(result_gdf),
        "validation_issues": len(all_issues),
        "validation_issue_details": all_issues,
        "validation_issues_by_rule": issues_by_rule,
        "validation_report": validation_report,
        "risk_details": risk_details,
        "segy_features": segy_features,
        "blind_zones_count": len(blind_zones_gdf),
        "blind_zone_anomalies_count": int(result_gdf["in_blind_zone"].sum()),
        "corrosion_critical_count": int(result_gdf["corrosion_critical"].sum()),
        "corrosion_warnings": corrosion_warnings,
        "dxf_path": str(dxf_path),
        "xml_path": str(xml_path),
        "act_path": str(act_path),
    }
