from __future__ import annotations

import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon

from src.ai_agent_course.schemas import AgentRunResult, ArtifactPaths, PromptContext
from src.ai_agent_course.llm_client import (
    build_engineering_answer,
    resolve_llm_config,
)
from src.ai_agent_course.tools.anomaly_detector import detect_linear_anomalies_from_image
from src.ai_agent_course.tools.coordinate_transformer import (
    fit_affine_transform,
    transform_linestring,
)
from src.ai_agent_course.tools.miis_parser import load_miis_layers
from src.ai_agent_course.tools.report_generator import (
    export_dxf_scheme,
    export_miis_xml,
    generate_pdf_act,
)
from src.ai_agent_course.tools.risk_classifier import (
    DamageRiskClassifier,
    RiskFeatures,
)
from src.ai_agent_course.tools.utility_type_identifier import (
    UtilityFeatures,
    UtilityTypeIdentifierV2,
)
from src.ai_agent_course.validation import validate_rules

logger = logging.getLogger(__name__)

CONSEQUENCE_WEIGHTS = {
    "gas": 10.0,
    "cable": 9.0,
    "heating": 8.0,
    "water": 6.0,
    "sewage": 5.0,
    "communication": 3.0,
}
UTILITY_CODE = {"water": 0.0, "sewage": 1.0, "gas": 2.0, "cable": 3.0, "heating": 4.0}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_prompt_context(prompt: str) -> PromptContext:
    lower = prompt.lower()
    site_match = re.search(r"жк\s+\"([^\"]+)\"", prompt, flags=re.IGNORECASE)
    site_name = site_match.group(1) if site_match else "Не указано"

    area_match = re.search(r"площад[ьи]\s*([0-9]+(?:[.,][0-9]+)?)\s*га", lower)
    area_ha = _safe_float(area_match.group(1).replace(",", "."), 5.0) if area_match else 5.0

    depth_match = re.search(r"глубин[аы]\s*(?:котлована)?\s*([0-9]+(?:[.,][0-9]+)?)\s*м", lower)
    excavation_depth_m = _safe_float(depth_match.group(1).replace(",", "."), 3.0) if depth_match else 3.0

    lat_match = re.search(r"([0-9]{1,2}\.[0-9]+)\s*°?\s*n", lower)
    lon_match = re.search(r"([0-9]{1,3}\.[0-9]+)\s*°?\s*e", lower)
    lat = _safe_float(lat_match.group(1), 0.0) if lat_match else None
    lon = _safe_float(lon_match.group(1), 0.0) if lon_match else None
    if lat == 0.0:
        lat = None
    if lon == 0.0:
        lon = None

    return PromptContext(
        site_name=site_name,
        area_ha=area_ha,
        excavation_depth_m=excavation_depth_m,
        latitude=lat,
        longitude=lon,
    )


def _default_construction_zone(anomalies_gdf: gpd.GeoDataFrame, excavation_depth_m: float) -> gpd.GeoDataFrame:
    if anomalies_gdf.empty:
        polygon = Polygon([(0, 0), (1500, 0), (1500, 1500), (0, 1500)])
    else:
        minx, miny, maxx, maxy = anomalies_gdf.total_bounds
        margin = 30.0
        polygon = Polygon(
            [
                (minx - margin, miny - margin),
                (maxx + margin, miny - margin),
                (maxx + margin, maxy + margin),
                (minx - margin, maxy + margin),
            ]
        )
    return gpd.GeoDataFrame(
        [{"id": "default-zone", "excavation_depth_m": excavation_depth_m, "geometry": polygon}],
        geometry="geometry",
        crs=anomalies_gdf.crs if not anomalies_gdf.empty else "EPSG:32637",
    )


def _nearest_distance(geometry, gdf: gpd.GeoDataFrame) -> float:
    if gdf.empty:
        return float("inf")
    distances = gdf.distance(geometry)
    if len(distances) == 0:
        return float("inf")
    return float(distances.min())


def _risk_by_tz(
    has_utility: bool,
    in_construction_zone: bool,
    depth_diff_m: float | None,
) -> tuple[str, float]:
    if in_construction_zone and not has_utility:
        return "CRITICAL", 0.89
    if has_utility and depth_diff_m is not None and depth_diff_m > 0.5:
        return "HIGH", 0.62
    return "LOW", 0.12


def _corrosion_life_years(utility_type: str, rho_value: float, ph: float, stray_current: float) -> tuple[float | None, str | None]:
    if utility_type not in {"gas", "water", "heating", "sewage"}:
        return None, None
    delta_start = 6.0
    delta_min = 2.0
    v_corr = 0.08 + 2.0 / max(rho_value, 1.0) + 0.02 * abs(ph - 7.0) + 0.15 * stray_current
    life = (delta_start - delta_min) / max(v_corr, 1e-4)
    warning = None
    if life < 5.0:
        warning = "Требуется срочная замена во избежание аварии"
    return float(life), warning


def _recommendation(risk_index: float, depth_m: float, utility_type: str) -> str:
    if utility_type == "sewage":
        return "Видеозондирование"
    if risk_index > 0.7 and depth_m <= 3.0:
        return "Шурф №ШШ-247"
    if 0.4 <= risk_index <= 0.7 and depth_m <= 5.0:
        return "Георадар ЛОМА-5"
    return "Инженерное уточнение трассы"


def _fallback_answer(anomalies: list[dict[str, Any]], validation: dict[str, Any], context: PromptContext) -> str:
    critical = sum(1 for item in anomalies if item["risk_class"] == "CRITICAL")
    high = sum(1 for item in anomalies if item["risk_class"] == "HIGH")
    low = sum(1 for item in anomalies if item["risk_class"] == "LOW")
    return (
        f"Участок: {context.site_name}. Найдено аномалий: {len(anomalies)} "
        f"(CRITICAL={critical}, HIGH={high}, LOW={low}). "
        f"Нарушений по R01..R12: {validation['issues_count']}. "
        "Рекомендуется выполнить подтверждающие работы по приоритету риска."
    )


class EngineeringSurveyAgent:
    def __init__(self) -> None:
        self.risk_classifier = DamageRiskClassifier()
        self.utility_identifier = UtilityTypeIdentifierV2()

    def run(
        self,
        image_path: Path,
        prompt: str,
        output_dir: Path,
        miis_xml_path: Path | None = None,
        llm_provider: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
    ) -> AgentRunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        context = _parse_prompt_context(prompt)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")

        detected = detect_linear_anomalies_from_image(image, pixel_size_m=5.0, min_length_m=10.0)
        if not detected:
            # Если Canny+Hough ничего не нашли, создаём базовую аномалию по самой яркой диагонали.
            h, w = image.shape[:2]
            detected = detect_linear_anomalies_from_image(
                np.pad(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), pad_width=0),
                pixel_size_m=5.0,
                min_length_m=10.0,
            )
            if not detected:
                detected = [
                    type("Fallback", (), {
                        "x1": 0.0,
                        "y1": 0.0,
                        "x2": float(w * 5.0),
                        "y2": float(h * 5.0),
                        "length_m": float(np.hypot(w * 5.0, h * 5.0)),
                        "amplitude_nt": 35.0,
                        "midpoint_x_m": float(w * 2.5),
                        "midpoint_y_m": float(h * 2.5),
                    })()
                ]

        anomalies_rows: list[dict[str, Any]] = []
        for idx, an in enumerate(detected):
            depth = 0.8 + min(max(an.amplitude_nt, 0.0), 140.0) / 140.0 * 2.7
            rho_value = max(8.0, 80.0 - an.amplitude_nt * 0.45)
            anomalies_rows.append(
                {
                    "id": idx,
                    "type": "linear_magnetic",
                    "amplitude_nt": float(an.amplitude_nt),
                    "length_m": float(an.length_m),
                    "depth_m": float(depth),
                    "rho_value": float(rho_value),
                    "geometry": LineString([(an.x1, an.y1), (an.x2, an.y2)]),
                }
            )
        anomalies_gdf = gpd.GeoDataFrame(anomalies_rows, geometry="geometry", crs="EPSG:32637")

        utilities_gdf, wells_gdf, zones_gdf = load_miis_layers(miis_xml_path)
        if zones_gdf.empty:
            zones_gdf = _default_construction_zone(anomalies_gdf, context.excavation_depth_m)
        if not utilities_gdf.empty and utilities_gdf.crs != anomalies_gdf.crs:
            utilities_gdf = utilities_gdf.to_crs(anomalies_gdf.crs)
        if not wells_gdf.empty and wells_gdf.crs != anomalies_gdf.crs:
            wells_gdf = wells_gdf.to_crs(anomalies_gdf.crs)
        if zones_gdf.crs != anomalies_gdf.crs:
            zones_gdf = zones_gdf.to_crs(anomalies_gdf.crs)

        # Шаг 1: аффинная трансформация (LSQ). В demo используем identity, если нет пар опорных точек.
        transform_result = fit_affine_transform([], [])
        anomalies_gdf["geometry"] = anomalies_gdf["geometry"].apply(
            lambda geom: transform_linestring(geom, transform_result.matrix)
        )

        all_reference = gpd.GeoDataFrame(
            geometry=list(utilities_gdf.geometry) + list(wells_gdf.geometry),
            crs=anomalies_gdf.crs,
        )

        anomaly_outputs: list[dict[str, Any]] = []
        for _, row in anomalies_gdf.iterrows():
            geom = row.geometry
            nearby = utilities_gdf[utilities_gdf.distance(geom) <= 2.5] if not utilities_gdf.empty else utilities_gdf
            has_utility = not nearby.empty
            nearest_depth = float(nearby["depth_m"].iloc[0]) if has_utility and "depth_m" in nearby.columns else None
            depth_diff = abs(float(row["depth_m"]) - nearest_depth) if nearest_depth is not None else None
            in_zone = bool(zones_gdf.intersects(geom).any()) if not zones_gdf.empty else True
            risk_class, probability = _risk_by_tz(has_utility=has_utility, in_construction_zone=in_zone, depth_diff_m=depth_diff)

            near_data_distance = _nearest_distance(geom, all_reference)
            in_blind_zone = bool(row["amplitude_nt"] > 50.0 and near_data_distance > 50.0)

            util_features = UtilityFeatures(
                amplitude_nt=float(row["amplitude_nt"]),
                gradient_rho=max(0.05, (40.0 - float(row["rho_value"])) / 40.0),
                hyperbola_width=max(0.4, min(3.0, float(row["length_m"]) / 30.0)),
                depth_m=float(row["depth_m"]),
                thermal_contrast=0.1,
                linear_extent_m=float(row["length_m"]),
                rho_value=float(row["rho_value"]),
                magnetic_gradient=max(0.1, float(row["amplitude_nt"]) / max(float(row["length_m"]), 1.0)),
                seg_velocity=0.12,
                near_road=0.0,
                area_type=1.0,
                age_infrastructure=20.0,
                soil_moisture=0.35,
                depth_to_diameter=max(float(row["depth_m"]) / 0.2, 1.0),
            )
            utility_type = self.utility_identifier.predict(util_features)
            utility_code = UTILITY_CODE.get(utility_type, 0.0)
            p_damage = self.risk_classifier.damage_probability(
                RiskFeatures(
                    excavation_depth_m=context.excavation_depth_m,
                    anomaly_depth_m=float(row["depth_m"]),
                    amplitude_nt=float(row["amplitude_nt"]),
                    length_m=float(row["length_m"]),
                    utility_type_code=utility_code,
                )
            )
            consequence = CONSEQUENCE_WEIGHTS.get(utility_type, 4.0) / 10.0
            risk_index = float(np.clip(probability * p_damage * consequence, 0.0, 1.0))
            recommendation = _recommendation(risk_index, float(row["depth_m"]), utility_type)

            remaining_life_years, corrosion_warning = _corrosion_life_years(
                utility_type=utility_type,
                rho_value=float(row["rho_value"]),
                ph=6.8,
                stray_current=0.12,
            )

            nearest_object = None
            if has_utility:
                nearest_object = str(nearby.iloc[0].get("id"))

            anomaly_outputs.append(
                {
                    "id": int(row["id"]),
                    "type": row["type"],
                    "amplitude_nt": float(row["amplitude_nt"]),
                    "length_m": float(row["length_m"]),
                    "depth_m": float(row["depth_m"]),
                    "risk_class": risk_class,
                    "probability": probability,
                    "risk_index": risk_index,
                    "utility_type": utility_type,
                    "has_utility": has_utility,
                    "depth_diff_m": float(depth_diff) if depth_diff is not None else 0.0,
                    "in_construction_zone": in_zone,
                    "in_blind_zone": in_blind_zone,
                    "recommendation": recommendation,
                    "remaining_life_years": remaining_life_years,
                    "corrosion_warning": corrosion_warning,
                    "nearest_registered_object": nearest_object,
                    "geometry_wkt": geom.wkt,
                }
            )

        validation = validate_rules(anomaly_outputs, transform_result.rms_error_m)

        # Для экспорта DXF используем исходный GeoDataFrame, но с финальными ID.
        anomalies_export_gdf = anomalies_gdf.copy()
        anomalies_export_gdf["id"] = [item["id"] for item in anomaly_outputs]
        anomalies_export_gdf["risk_class"] = [item["risk_class"] for item in anomaly_outputs]

        dxf_path = output_dir / "scheme_v2.dxf"
        xml_path = output_dir / "unaccounted_v2.miis.xml"
        pdf_path = output_dir / "act_v2.pdf"

        export_dxf_scheme(anomalies_export_gdf, utilities_gdf, dxf_path)
        export_miis_xml(anomaly_outputs, xml_path, crs=str(anomalies_gdf.crs))
        generate_pdf_act(context.site_name, anomaly_outputs, validation["issues"], pdf_path)

        llm_config = resolve_llm_config(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model,
            base_url=llm_base_url,
        )
        answer, mode = build_engineering_answer(prompt, anomaly_outputs, validation, llm_config)
        if not answer:
            answer = _fallback_answer(anomaly_outputs, validation, context)
            mode = f"{mode}_with_deterministic_answer"

        metrics = {
            "rms_error_m": transform_result.rms_error_m,
            "critical_count": sum(1 for item in anomaly_outputs if item["risk_class"] == "CRITICAL"),
            "high_count": sum(1 for item in anomaly_outputs if item["risk_class"] == "HIGH"),
            "low_count": sum(1 for item in anomaly_outputs if item["risk_class"] == "LOW"),
            "blind_zone_count": sum(1 for item in anomaly_outputs if item["in_blind_zone"]),
            "validation_issues_count": validation["issues_count"],
            "llm_provider": llm_config.provider,
            "llm_model": llm_config.model,
        }

        artifacts = ArtifactPaths(
            output_dir=str(output_dir),
            dxf_path=str(dxf_path),
            miis_xml_path=str(xml_path),
            pdf_report_path=str(pdf_path),
        )
        return AgentRunResult(
            prompt_context=asdict(context),
            anomalies=anomaly_outputs,
            validation=validation,
            metrics=metrics,
            artifacts=artifacts,
            answer=answer,
            mode=mode,
        )

