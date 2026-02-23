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
import re
from dataclasses import dataclass
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


@dataclass
class NormativeIssue:
    rule_id: str
    severity: str
    reference: str
    message: str

    def to_text(self) -> str:
        return f"[{self.rule_id} / {self.severity} / {self.reference}] {self.message}"


NORMATIVE_RULES = {
    "R01": "CRITICAL аномалия без коммуникаций в буфере",
    "R02": "HIGH аномалия без подтверждения коммуникациями",
    "R03": "Глубина залегания меньше глубины промерзания",
    "R04": "Отсутствует колонка risk_class",
    "R05": "Недопустимое значение risk_class",
    "R06": "Отсутствует utility_type для значимых аномалий",
    "R07": "Нет данных о глубине для HIGH/CRITICAL",
    "R08": "Некорректная геометрия аномалии",
    "R09": "Некорректная геометрия коммуникации",
    "R10": "Коммуникация без заполненного типа",
    "R11": "Критическая аномалия рядом с unknown-типом",
    "R12": "Дублирование аномалий в пределах допуска",
}


def _issue(rule_id: str, severity: str, reference: str, message: str) -> NormativeIssue:
    return NormativeIssue(
        rule_id=rule_id,
        severity=severity,
        reference=reference,
        message=message,
    )


def _is_bad_geometry(geom) -> bool:
    if geom is None:
        return True
    if geom.is_empty:
        return True
    return not geom.is_valid


def validate_anomalies_detailed(
    anomalies_gdf: gpd.GeoDataFrame,
    utilities_gdf: gpd.GeoDataFrame,
    buffer_dist: float = BUFFER_DISTANCE,
    duplicate_tolerance: float = 0.2,
) -> list[NormativeIssue]:
    issues: list[NormativeIssue] = []

    if anomalies_gdf.empty:
        logger.info("Validator: нет аномалий для проверки")
        return issues

    if not utilities_gdf.empty and anomalies_gdf.crs != utilities_gdf.crs:
        logger.warning("Validator: CRS аномалий и коммуникаций не совпадают, выполняем перепроецирование")
        utilities_gdf = utilities_gdf.to_crs(anomalies_gdf.crs)

    # R08: проверка геометрий аномалий
    for idx, row in anomalies_gdf.iterrows():
        if _is_bad_geometry(row.geometry):
            issues.append(
                _issue(
                    "R08",
                    "ERROR",
                    "СП 47 п.8.3",
                    f"Аномалия #{idx}: некорректная или пустая геометрия.",
                )
            )

    # R09/R10: проверка геометрии и типа коммуникаций
    if not utilities_gdf.empty:
        for idx, row in utilities_gdf.iterrows():
            if _is_bad_geometry(row.geometry):
                issues.append(
                    _issue(
                        "R09",
                        "ERROR",
                        "СП 47 п.8.3",
                        f"Коммуникация #{idx}: некорректная или пустая геометрия.",
                    )
                )
            if "type" not in utilities_gdf.columns or row.get("type") in (None, "", "unknown"):
                issues.append(
                    _issue(
                        "R10",
                        "WARNING",
                        "СП 47 п.8.3",
                        f"Коммуникация #{idx}: не заполнен тип коммуникации.",
                    )
                )

    # R04/R05: качество risk_class
    if "risk_class" not in anomalies_gdf.columns:
        issues.append(
            _issue(
                "R04",
                "ERROR",
                "СП 47 п.8.3",
                "В наборе аномалий отсутствует колонка risk_class.",
            )
        )
        risk_series = gpd.pd.Series(["LOW"] * len(anomalies_gdf), index=anomalies_gdf.index)
    else:
        risk_series = anomalies_gdf["risk_class"].fillna("LOW")
        allowed = {"LOW", "HIGH", "CRITICAL"}
        invalid = anomalies_gdf[~risk_series.isin(allowed)]
        for idx, row in invalid.iterrows():
            issues.append(
                _issue(
                    "R05",
                    "ERROR",
                    "СП 47 п.8.3",
                    f"Аномалия #{idx}: недопустимый класс риска '{row.get('risk_class')}'.",
                )
            )

    # R06: utility_type для HIGH/CRITICAL
    significant_mask = risk_series.isin(["HIGH", "CRITICAL"])
    if "utility_type" not in anomalies_gdf.columns:
        if significant_mask.any():
            issues.append(
                _issue(
                    "R06",
                    "WARNING",
                    "СП 47 п.8.3",
                    "Для HIGH/CRITICAL аномалий отсутствует колонка utility_type.",
                )
            )
    else:
        missing_type = anomalies_gdf[significant_mask & anomalies_gdf["utility_type"].isna()]
        for idx, _ in missing_type.iterrows():
            issues.append(
                _issue(
                    "R06",
                    "WARNING",
                    "СП 47 п.8.3",
                    f"Аномалия #{idx}: не определён тип коммуникации.",
                )
            )

    # R07/R03: контроль глубины
    if "depth" not in anomalies_gdf.columns:
        if significant_mask.any():
            issues.append(
                _issue(
                    "R07",
                    "WARNING",
                    "СП 47 п.8.3",
                    "Для HIGH/CRITICAL аномалий отсутствует колонка depth.",
                )
            )
    else:
        missing_depth = anomalies_gdf[significant_mask & anomalies_gdf["depth"].isna()]
        for idx, _ in missing_depth.iterrows():
            issues.append(
                _issue(
                    "R07",
                    "WARNING",
                    "СП 47 п.8.3",
                    f"Аномалия #{idx}: не указана глубина залегания.",
                )
            )

        shallow = anomalies_gdf[
            anomalies_gdf["depth"].notna()
            & (anomalies_gdf["depth"] < FROST_DEPTH_MOSCOW)
        ]
        for idx, anom in shallow.iterrows():
            issues.append(
                _issue(
                    "R03",
                    "WARNING",
                    "СП 22",
                    f"Аномалия #{idx}: глубина залегания {anom['depth']:.2f} м "
                    f"< глубины промерзания {FROST_DEPTH_MOSCOW} м (Москва).",
                )
            )

    # R01/R02/R11: проверки по близости коммуникаций
    critical = anomalies_gdf[risk_series == "CRITICAL"]
    high = anomalies_gdf[risk_series == "HIGH"]

    def _nearby_utilities(geom):
        if utilities_gdf.empty:
            return gpd.GeoDataFrame()
        return utilities_gdf[utilities_gdf.intersects(geom.buffer(buffer_dist))]

    for idx, anom in critical.iterrows():
        nearby = _nearby_utilities(anom.geometry)
        if len(nearby) == 0:
            issues.append(
                _issue(
                    "R01",
                    "ERROR",
                    "СП 47 п.8.3",
                    f"Аномалия #{idx}: в радиусе {buffer_dist} м нет учтённых коммуникаций — "
                    "возможен неучтённый подземный объект. Требуется дополнительное обследование.",
                )
            )
            continue

        if "type" in nearby.columns:
            unknown_nearby = nearby[nearby["type"].isin(["unknown", "unknown_pipeline"])]
            if not unknown_nearby.empty:
                issues.append(
                    _issue(
                        "R11",
                        "WARNING",
                        "СП 47 п.8.3",
                        f"Аномалия #{idx}: рядом есть коммуникации с неопределённым типом.",
                    )
                )

    if "has_utility" in anomalies_gdf.columns:
        high_unconfirmed = anomalies_gdf[
            (risk_series == "HIGH") & (anomalies_gdf["has_utility"] == False)  # noqa
        ]
        for idx, _ in high_unconfirmed.iterrows():
            issues.append(
                _issue(
                    "R02",
                    "WARNING",
                    "СП 47 п.8.3",
                    f"Аномалия #{idx} (HIGH): высокий риск, но коммуникации в буфере не найдены.",
                )
            )
    else:
        for idx, anom in high.iterrows():
            if _nearby_utilities(anom.geometry).empty:
                issues.append(
                    _issue(
                        "R02",
                        "WARNING",
                        "СП 47 п.8.3",
                        f"Аномалия #{idx} (HIGH): высокий риск, но коммуникации в буфере не найдены.",
                    )
                )

    # R12: дубли аномалий
    geometries = list(anomalies_gdf.geometry.items())
    for i in range(len(geometries)):
        idx_a, geom_a = geometries[i]
        if _is_bad_geometry(geom_a):
            continue
        for j in range(i + 1, len(geometries)):
            idx_b, geom_b = geometries[j]
            if _is_bad_geometry(geom_b):
                continue
            if geom_a.distance(geom_b) < duplicate_tolerance:
                issues.append(
                    _issue(
                        "R12",
                        "WARNING",
                        "СП 47 п.8.3",
                        f"Аномалии #{idx_a} и #{idx_b} дублируются (расстояние < {duplicate_tolerance} м).",
                    )
                )

    logger.info(f"Validator: выявлено {len(issues)} нарушений")
    return issues


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
    issues = validate_anomalies_detailed(
        anomalies_gdf=anomalies_gdf,
        utilities_gdf=utilities_gdf,
        buffer_dist=buffer_dist,
    )
    return [issue.to_text() for issue in issues]


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

    all_issues = validation_errors + compliance_issues
    issues_by_rule = summarize_issues_by_rule(all_issues)

    if not all_issues:
        lines.append("✅ Нарушений не выявлено.")
        return "\n".join(lines)

    if issues_by_rule:
        lines.append("\nСводка по правилам:")
        for rule_id, count in sorted(issues_by_rule.items()):
            title = NORMATIVE_RULES.get(rule_id, "Прочее")
            lines.append(f"  • {rule_id}: {count} — {title}")

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


def summarize_issues_by_rule(issues: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        match = re.search(r"\[(R\d{2})", issue)
        if match:
            rule_id = match.group(1)
            counts[rule_id] = counts.get(rule_id, 0) + 1
    return counts
