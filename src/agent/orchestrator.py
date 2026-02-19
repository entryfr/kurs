# src/agent/orchestrator.py
"""
Оркестратор агентного пайплайна геофизического обследования.

Цепочка шагов:
  1. LoadOSMUtilities   — загрузить учтённые коммуникации из OSM (Overpass API)
  2. TransformCoords    — привязать данные съёмки в МСК-50
  3. DetectAnomalies    — детектировать линейные аномалии на магнитной карте
  4. FindBlindZones     — определить зоны, где съёмка ненадёжна
  5. ClassifyRisk       — оценить риск для каждой аномалии
  6. AssessCorrosion    — оценить коррозию по грунтовым параметрам
  7. BufferAnalysis     — сопоставить аномалии с буферными зонами коммуникаций
  8. ValidateCompliance — проверить соответствие нормативам (СП 47, СП 42)
  9. ExportDXF          — экспортировать чертёж в DXF
  10. ExportMIIS         — экспортировать в МИИС XML
  11. GenerateAct        — сформировать акт обследования
"""

import logging
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.llms import Anthropic

from src.preprocessing.osm_loader import (
    load_utilities_by_bbox,
    load_utilities_by_area,
    load_utilities_for_survey,
    get_utilities_summary,
)
from src.preprocessing.coordinate_transformer import transform_geometry
from src.detection.anomaly_detector import detect_linear_anomalies
from src.detection.blind_zones import find_blind_zones
from src.detection.buffer_analysis import check_utilities_in_buffer
from src.classification.risk_classifier import RiskClassifier
from src.classification.utility_type_identifier import UtilityTypeIdentifier
from src.classification.corrosion_estimator import assess_corrosion
from src.reporting.dxf_exporter import export_to_dxf
from src.reporting.miis_exporter import create_miis_xml
from src.reporting.act_generator import generate_act
from src.agent.validator import validate_anomalies, check_compliance

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Обёртки для LangChain Tools (принимают / возвращают строки или dict)
# ─────────────────────────────────────────────────────────────────────────────

def _tool_load_osm_bbox(bbox_str: str) -> str:
    """
    Загружает OSM-коммуникации по bbox.
    Вход: строка «south,west,north,east» (градусы WGS-84)
    Пример: «55.73,37.60,55.77,37.66»
    """
    try:
        parts = [float(x.strip()) for x in bbox_str.split(",")]
        if len(parts) != 4:
            return "Ошибка: нужно передать 4 числа через запятую: south,west,north,east"
        bbox = tuple(parts)
        gdf  = load_utilities_by_bbox(bbox)
        summary = get_utilities_summary(gdf)
        return (
            f"Загружено {summary['total']} объектов. "
            f"Суммарная длина: {summary['total_length_m']:.0f} м. "
            f"По типам: {summary['by_type']}"
        )
    except Exception as e:
        logger.exception("Ошибка при загрузке OSM по bbox")
        return f"Ошибка: {e}"


def _tool_load_osm_area(area_name: str) -> str:
    """
    Загружает OSM-коммуникации для именованного района.
    Вход: название района (например «Замоскворечье»)
    """
    try:
        gdf     = load_utilities_by_area(area_name.strip())
        summary = get_utilities_summary(gdf)
        return (
            f"Район «{area_name}»: загружено {summary['total']} объектов. "
            f"По типам: {summary['by_type']}"
        )
    except Exception as e:
        logger.exception("Ошибка при загрузке OSM по району")
        return f"Ошибка: {e}"


def _tool_validate(input_str: str) -> str:
    """Запускает нормативную проверку (требует предварительно загруженные GeoDataFrame в контексте)."""
    return "Для полной валидации передайте anomalies_gdf и utilities_gdf через pipeline напрямую."


# ─────────────────────────────────────────────────────────────────────────────
# Построение агента
# ─────────────────────────────────────────────────────────────────────────────

def create_agent():
    """
    Создаёт LangChain-агента со всеми инструментами пайплайна.

    Returns:
        Инициализированный AgentExecutor.
    """
    tools = [
        Tool(
            name="LoadOSMByBBox",
            func=_tool_load_osm_bbox,
            description=(
                "Загружает учтённые подземные коммуникации из OpenStreetMap через Overpass API "
                "для прямоугольной области. "
                "Вход: строка «south,west,north,east» в градусах WGS-84. "
                "Пример: «55.73,37.60,55.77,37.66» для центра Москвы."
            ),
        ),
        Tool(
            name="LoadOSMByArea",
            func=_tool_load_osm_area,
            description=(
                "Загружает учтённые подземные коммуникации из OpenStreetMap для именованного "
                "административного района. Вход: название района на русском, "
                "например «Замоскворечье» или «Хамовники»."
            ),
        ),
        Tool(
            name="TransformCoordinates",
            func=lambda args: str(transform_geometry(*args) if isinstance(args, (list, tuple)) else args),
            description="Преобразует координаты из ПЗ-90.11 в МСК-50. Вход: geom + affine_params.",
        ),
        Tool(
            name="DetectAnomalies",
            func=lambda grid: str(detect_linear_anomalies(grid)),
            description="Детектирует линейные аномалии на магнитной карте (numpy array).",
        ),
        Tool(
            name="AssessCorrosion",
            func=lambda data: str(assess_corrosion(data)),
            description=(
                "Оценивает коррозию трубопровода. "
                "Вход: словарь {soil_rho, moisture, ph, initial_diameter, age}."
            ),
        ),
        Tool(
            name="ValidateCompliance",
            func=_tool_validate,
            description="Проверяет соответствие аномалий нормативным требованиям (СП 47, СП 42).",
        ),
    ]

    llm = Anthropic(model="claude-opus-4-6")
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Прямой (не-агентный) пайплайн для программного использования
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    magnetic_grid,
    survey_geom,
    survey_crs,
    affine_params,
    ground_params: dict,
    output_dir: str = "output",
):
    """
    Запускает полный пайплайн обследования без агента (программный режим).

    Args:
        magnetic_grid: numpy.ndarray — сетка магнитного поля (нТл)
        survey_geom:   shapely.Polygon — полигон участка съёмки
        survey_crs:    str — CRS полигона (например, «EPSG:32637»)
        affine_params: list[6] — параметры аффинного преобразования
        ground_params: dict — параметры грунта для оценки коррозии
                       {soil_rho, moisture, ph, initial_diameter, age}
        output_dir:    str — папка для сохранения результатов

    Returns:
        dict с ключами: utilities_gdf, anomalies_gdf, zones_gdf, validation_errors, summary
    """
    import os
    import geopandas as gpd
    from shapely.geometry import LineString
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    logger.info("═══ ШАГ 1: Загрузка OSM-коммуникаций ════════════════════")
    utilities_gdf = load_utilities_for_survey(survey_geom, buffer_m=200.0, target_crs=survey_crs)
    summary_osm   = get_utilities_summary(utilities_gdf)
    logger.info(f"OSM: {summary_osm}")

    logger.info("═══ ШАГ 2: Детекция аномалий ════════════════════════════")
    raw_anomalies = detect_linear_anomalies(magnetic_grid)

    # Конвертируем список словарей в GeoDataFrame
    classifier  = RiskClassifier()
    identifier  = UtilityTypeIdentifier()
    rows = []
    for a in raw_anomalies:
        (x1, y1), (x2, y2) = a["coords"]
        geom = transform_geometry(
            LineString([(x1, y1), (x2, y2)]),
            affine_params
        )
        features = {
            "amplitude_nt":    a["mean_intensity"],
            "linear_extent_m": a["length"],
            "excavation_depth": 3.0,   # Глубина планируемой выработки
            "anomaly_depth":    1.5,   # Предполагаемая глубина объекта
            "utility_type":     identifier.predict({"amplitude_nt": a["mean_intensity"]}),
            "detection_factor": 0.7,
        }
        rows.append({
            "geometry":   geom,
            "length_m":   a["length"],
            "intensity":  a["mean_intensity"],
            "utility_type": features["utility_type"],
            "risk_class": classifier.predict_risk(features),
            **assess_corrosion({**ground_params, "initial_diameter": 200}),
        })

    anomalies_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=survey_crs)
    logger.info(f"Детектировано аномалий: {len(anomalies_gdf)}")

    logger.info("═══ ШАГ 3: Буферный анализ ══════════════════════════════")
    anomalies_gdf = check_utilities_in_buffer(anomalies_gdf, utilities_gdf)

    logger.info("═══ ШАГ 4: Слепые зоны ══════════════════════════════════")
    # (find_blind_zones требует x_coords, y_coords — передаём заглушку)
    zones_gdf = gpd.GeoDataFrame(geometry=[], crs=survey_crs)

    logger.info("═══ ШАГ 5: Нормативная валидация ════════════════════════")
    validation_errors   = validate_anomalies(anomalies_gdf, utilities_gdf)
    compliance_issues   = check_compliance(anomalies_gdf, utilities_gdf)
    all_issues = validation_errors + compliance_issues
    if all_issues:
        logger.warning(f"Нарушения: {all_issues}")

    logger.info("═══ ШАГ 6: Экспорт ══════════════════════════════════════")
    export_to_dxf(
        anomalies_gdf, utilities_gdf, zones_gdf,
        output_path=os.path.join(output_dir, "scheme.dxf")
    )
    create_miis_xml(
        anomalies_gdf, utilities_gdf,
        output_path=os.path.join(output_dir, "miis_output.xml")
    )
    generate_act(
        data={
            "anomalies": anomalies_gdf.to_dict(orient="records"),
            "utilities": utilities_gdf.to_dict(orient="records"),
            "issues":    all_issues,
            "osm_summary": summary_osm,
        },
        output_path=os.path.join(output_dir, "act.docx"),
    )

    logger.info("═══ Пайплайн завершён ════════════════════════════════════")
    return {
        "utilities_gdf":      utilities_gdf,
        "anomalies_gdf":      anomalies_gdf,
        "zones_gdf":          zones_gdf,
        "validation_errors":  all_issues,
        "osm_summary":        summary_osm,
    }
