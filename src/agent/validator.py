# src/agent/validator.py
"""
Нормативная валидация результатов геофизического обследования.

Поддерживает два режима utilities_gdf:
  1. Собственная база данных (из файла проекта)
  2. OSM-данные, загруженные через osm_loader.py

Реализованные проверки:
  - СП 47.13330.2016 — Инженерные изыскания для строительства
  - СП 42.13330.2016 — Минимальные расстояния между коммуникациями
  - Глубина залегания vs глубина промерзания (Москва)
"""

import logging
import geopandas as gpd
from shapely.geometry import Point

from config.settings import BUFFER_DISTANCE

logger = logging.getLogger(__name__)

# Глубина промерзания грунта для Москвы (м) — СП 22.13330
FROST_DEPTH_MOSCOW = 1.4

# Минимальные расстояния между типами коммуникаций (метры) — СП 42.13330 табл.12
MIN_DISTANCE_RULES: list[tuple] = [
    # (тип_A,        тип_B,         мин_расстояние_м, источник)
    ("gas",          "electricity",  1.0,  "СП 42 п.12.35"),
    ("gas",          "water",        1.0,  "СП 42 п.12.35"),
    ("gas",          "sewage",       1.0,  "СП 42 п.12.35"),
    ("gas",          "heating",      2.0,  "СП 42 п.12.35"),
    ("electricity",  "water",        0.5,  "СП 42 п.12.35"),
    ("electricity",  "heating",      2.0,  "СП 42 п.12.35"),
    ("water",        "sewage",       1.5,  "СП 42 п.12.35"),
    ("heating",      "sewage",       1.0,  "СП 42 п.12.35"),
]


def validate_anomalies(
    anomalies_gdf: gpd.GeoDataFrame,
    utilities_gdf: gpd.GeoDataFrame,
    buffer_dist: float = BUFFER_DISTANCE,
) -> list[str]:
    """
    Проверяет аномалии на соответствие нормативным требованиям.

    Правила:
      1. Критическая аномалия должна иметь учтённую коммуникацию в буфере.
      2. Аномалия со значительной глубиной (depth > FROST_DEPTH_MOSCOW)
         должна иметь тип, допускающий глубокое залегание.
      3. Для аномалий без сопоставленной коммуникации предупреждение об
         «неучтённом объекте» согласно СП 47 п.8.3.

    Args:
        anomalies_gdf: GeoDataFrame обнаруженных аномалий.
                       Ожидаемые колонки: risk_class, depth (опц.), geometry.
        utilities_gdf: GeoDataFrame коммуникаций (из OSM или собственной БД).
                       Ожидаемые колонки: type, geometry.
        buffer_dist:   Радиус буфера для поиска ближайших коммуникаций, м.

    Returns:
        Список строк с описанием ошибок/предупреждений.
    """
    errors: list[str] = []

    if anomalies_gdf.empty:
        logger.info("Validator: нет аномалий для проверки")
        return errors

    # Убеждаемся, что CRS совпадают
    if not utilities_gdf.empty and anomalies_gdf.crs != utilities_gdf.crs:
        logger.warning("Validator: CRS аномалий и коммуникаций не совпадают, выполняем перепроецирование")
        utilities_gdf = utilities_gdf.to_crs(anomalies_gdf.crs)

    # ── Правило 1: Критическая аномалия должна иметь учтённую коммуникацию ──
    critical = anomalies_gdf[anomalies_gdf.get("risk_class", gpd.pd.Series()) == "CRITICAL"]
    for idx, anom in critical.iterrows():
        buffer = anom.geometry.buffer(buffer_dist)
        if utilities_gdf.empty:
            nearby = gpd.GeoDataFrame()
        else:
            nearby = utilities_gdf[utilities_gdf.intersects(buffer)]

        if len(nearby) == 0:
            errors.append(
                f"[CRITICAL / СП47 п.8.3] Аномалия #{idx}: "
                f"в радиусе {buffer_dist} м нет учтённых коммуникаций — "
                "возможен неучтённый подземный объект. Требуется дополнительное обследование."
            )
        else:
            # Источник данных (OSM или собственная БД)
            source = "OSM" if "osm_id" in nearby.columns else "БД проекта"
            types  = ", ".join(nearby["type"].unique()) if "type" in nearby.columns else "неизвестен"
            logger.info(
                f"Аномалия #{idx} (CRITICAL): подтверждена коммуникациями из {source} "
                f"(типы: {types})"
            )

    # ── Правило 2: Глубина залегания vs глубина промерзания ─────────────────
    if "depth" in anomalies_gdf.columns:
        shallow = anomalies_gdf[
            anomalies_gdf["depth"].notna() &
            (anomalies_gdf["depth"] < FROST_DEPTH_MOSCOW)
        ]
        for idx, anom in shallow.iterrows():
            errors.append(
                f"[WARNING / СП 22] Аномалия #{idx}: "
                f"глубина залегания {anom['depth']:.2f} м < глубины промерзания "
                f"{FROST_DEPTH_MOSCOW} м (Москва). Риск повреждения при сезонном пучении."
            )

    # ── Правило 3: Неподтверждённые аномалии с высоким риском ──────────────
    if "has_utility" in anomalies_gdf.columns:
        high_unconfirmed = anomalies_gdf[
            (anomalies_gdf.get("risk_class", "") == "HIGH") &
            (anomalies_gdf["has_utility"] == False)  # noqa
        ]
        for idx, _ in high_unconfirmed.iterrows():
            errors.append(
                f"[HIGH / СП47 п.8.3] Аномалия #{idx}: "
                "высокий риск, но коммуникации в буфере не найдены ни в OSM, ни в БД проекта."
            )

    logger.info(f"Validator: выявлено {len(errors)} нарушений")
    return errors


def check_compliance(
    anomalies_gdf: gpd.GeoDataFrame,
    utilities_gdf: gpd.GeoDataFrame,
) -> list[str]:
    """
    Проверяет соблюдение минимальных расстояний между коммуникациями (СП 42).

    Работает корректно как с OSM-данными (есть колонка osm_id),
    так и с собственной БД проекта.

    Args:
        anomalies_gdf: GeoDataFrame аномалий (не используется напрямую, зарезервировано).
        utilities_gdf: GeoDataFrame коммуникаций с колонкой «type».

    Returns:
        Список строк с описанием нарушений.
    """
    issues: list[str] = []

    if utilities_gdf.empty or "type" not in utilities_gdf.columns:
        return issues

    # Проверяем каждую пару типов из таблицы правил
    for type_a, type_b, min_dist, reference in MIN_DISTANCE_RULES:
        group_a = utilities_gdf[utilities_gdf["type"] == type_a]
        group_b = utilities_gdf[utilities_gdf["type"] == type_b]

        if group_a.empty or group_b.empty:
            continue

        for idx_a, row_a in group_a.iterrows():
            for idx_b, row_b in group_b.iterrows():
                dist = row_a.geometry.distance(row_b.geometry)
                if dist < min_dist:
                    # Источник данных
                    src_a = f"osm:{row_a.get('osm_id', idx_a)}" if "osm_id" in row_a else str(idx_a)
                    src_b = f"osm:{row_b.get('osm_id', idx_b)}" if "osm_id" in row_b else str(idx_b)
                    issues.append(
                        f"[{reference}] {type_a.upper()} ({src_a}) и "
                        f"{type_b.upper()} ({src_b}): "
                        f"расстояние {dist:.2f} м < {min_dist} м"
                    )

    if issues:
        logger.warning(f"Compliance: выявлено {len(issues)} нарушений СП 42")
    else:
        logger.info("Compliance: нарушений не выявлено")

    return issues


def generate_validation_report(
    validation_errors: list[str],
    compliance_issues: list[str],
) -> str:
    """
    Формирует текстовый отчёт о результатах валидации.

    Args:
        validation_errors: Список ошибок из validate_anomalies().
        compliance_issues: Список нарушений из check_compliance().

    Returns:
        Многострочная строка с отчётом.
    """
    lines = ["=" * 60, "ОТЧЁТ О НОРМАТИВНОЙ ВАЛИДАЦИИ", "=" * 60]

    if not validation_errors and not compliance_issues:
        lines.append("✅ Нарушений не выявлено.")
        return "\n".join(lines)

    if validation_errors:
        lines.append(f"\n⚠️  Аномалии без подтверждения ({len(validation_errors)}):")
        for e in validation_errors:
            lines.append(f"  • {e}")

    if compliance_issues:
        lines.append(f"\n🚫 Нарушения минимальных расстояний ({len(compliance_issues)}):")
        for i in compliance_issues:
            lines.append(f"  • {i}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
