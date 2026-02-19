# src/preprocessing/osm_loader.py
"""
Модуль загрузки данных об инженерных коммуникациях из OpenStreetMap
через Overpass API.

Основные возможности:
  - Запрос трубопроводов, кабелей, тепловых сетей по bbox или полигону района
  - Кеширование ответов (избегаем повторных запросов к API)
  - Конвертация в GeoDataFrame с нормализованными атрибутами
  - Автоматическое перепроецирование в рабочую CRS проекта
  - Интеграция с validator.py и buffer_analysis.py
"""

import os
import json
import time
import hashlib
import logging
import requests
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon, shape
from typing import Optional, Union

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(query: str) -> Path:
    """Возвращает путь к файлу кеша для данного запроса."""
    key = hashlib.md5(query.encode()).hexdigest()
    cache_dir = Path(settings.OVERPASS_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.json"


def _is_cache_valid(path: Path) -> bool:
    """Проверяет, не истёк ли TTL кеша."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < settings.OVERPASS_CACHE_TTL


def _execute_query(query: str, use_cache: bool = True) -> dict:
    """
    Выполняет запрос к Overpass API с поддержкой кеширования.

    Args:
        query:     Текст Overpass QL запроса.
        use_cache: Использовать кеш (по умолчанию True).

    Returns:
        Словарь с JSON-ответом Overpass API.

    Raises:
        requests.HTTPError: При ошибке HTTP.
        ValueError:         Если ответ не является валидным JSON.
    """
    cache = _cache_path(query)

    # Проверяем кеш
    if use_cache and _is_cache_valid(cache):
        logger.info(f"OSM: загружаем из кеша ({cache})")
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("OSM: отправляем запрос к Overpass API...")
    response = requests.post(
        settings.OVERPASS_URL,
        data={"data": query},
        timeout=settings.OVERPASS_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()

    # Сохраняем в кеш
    if use_cache:
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"OSM: ответ сохранён в кеш ({cache})")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Формирование Overpass QL запросов
# ─────────────────────────────────────────────────────────────────────────────

def _build_bbox_query(bbox: tuple) -> str:
    """
    Строит Overpass QL запрос для прямоугольной области.

    Args:
        bbox: (south, west, north, east) в WGS-84

    Returns:
        Строка запроса Overpass QL
    """
    s, w, n, e = bbox
    bbox_str = f"{s},{w},{n},{e}"

    # Собираем все нужные теги через union
    tag_filters = []
    tag_filters.append(f'way["man_made"="pipeline"]({bbox_str})')
    tag_filters.append(f'way["power"="cable"]({bbox_str})')
    tag_filters.append(f'way["power"="line"]({bbox_str})')
    tag_filters.append(f'way["utility"~"gas|water|sewage|heating|electricity"]({bbox_str})')
    # Также тянем отдельные трубы водопровода/канализации из waterway
    tag_filters.append(f'way["waterway"="drain"]({bbox_str})')
    tag_filters.append(f'way["waterway"="canal"]["usage"="water_supply"]({bbox_str})')

    union_body = "\n  ".join(tag_filters)

    query = f"""
[out:json][timeout:{settings.OVERPASS_TIMEOUT}];
(
  {union_body}
);
out body;
>;
out skel qt;
""".strip()

    return query


def _build_area_query(area_name: str) -> str:
    """
    Строит Overpass QL запрос для именованного административного района.

    Args:
        area_name: Название района (например, «Москва», «Замоскворечье»)

    Returns:
        Строка запроса Overpass QL
    """
    query = f"""
[out:json][timeout:{settings.OVERPASS_TIMEOUT}];
area["name"="{area_name}"]["boundary"="administrative"]->.search_area;
(
  way["man_made"="pipeline"](area.search_area);
  way["power"="cable"](area.search_area);
  way["power"="line"](area.search_area);
  way["utility"~"gas|water|sewage|heating|electricity"](area.search_area);
  way["waterway"="drain"](area.search_area);
);
out body;
>;
out skel qt;
""".strip()

    return query


# ─────────────────────────────────────────────────────────────────────────────
# Разбор ответа Overpass API → GeoDataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _parse_overpass_response(data: dict) -> gpd.GeoDataFrame:
    """
    Преобразует JSON-ответ Overpass API в GeoDataFrame.

    Overpass возвращает nodes + ways. Сначала строим словарь
    node_id -> (lon, lat), затем для каждого way собираем LineString.

    Args:
        data: Словарь с ответом Overpass API (поле «elements»).

    Returns:
        GeoDataFrame с геометриями и нормализованными атрибутами, CRS=WGS84.
    """
    # Индекс: node_id -> координаты
    nodes: dict[int, tuple] = {}
    ways: list[dict] = []

    for el in data.get("elements", []):
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
        elif el["type"] == "way":
            ways.append(el)

    records = []
    for way in ways:
        node_ids = way.get("nodes", [])
        coords = [nodes[nid] for nid in node_ids if nid in nodes]

        if len(coords) < 2:
            # Не хватает узлов для построения линии
            logger.debug(f"Way {way['id']}: недостаточно узлов ({len(coords)}), пропускаем")
            continue

        geometry = LineString(coords)
        tags = way.get("tags", {})

        record = {
            "osm_id":    way["id"],
            "geometry":  geometry,

            # ── Нормализованные атрибуты ──────────────────────────────────
            "type":      _normalize_type(tags),
            "substance": tags.get("substance", None),
            "material":  tags.get("material", None),
            "diameter":  _parse_diameter(tags.get("diameter", None)),
            "depth":     _parse_float(tags.get("depth", None)),
            "location":  tags.get("location", "underground"),
            "operator":  tags.get("operator", None),
            "layer":     _parse_int(tags.get("layer", "0")),
            "pressure":  tags.get("pressure", None),
            "start_date":tags.get("start_date", None),
            "name":      tags.get("name", None),

            # ── Сырые теги (для отладки) ──────────────────────────────────
            "_raw_tags": json.dumps(tags, ensure_ascii=False),
        }
        records.append(record)

    if not records:
        logger.warning("OSM: ни одного объекта не найдено в ответе Overpass")
        return gpd.GeoDataFrame(
            columns=["osm_id", "geometry", "type", "substance", "material",
                     "diameter", "depth", "location", "operator", "layer",
                     "pressure", "start_date", "name", "_raw_tags"],
            geometry="geometry",
            crs=settings.CRS_WGS84,
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=settings.CRS_WGS84)
    logger.info(f"OSM: загружено {len(gdf)} объектов коммуникаций")
    return gdf


def _normalize_type(tags: dict) -> str:
    """
    Определяет внутренний тип коммуникации по OSM-тегам.
    Приоритет: substance > power > man_made/waterway
    """
    substance = tags.get("substance", "").lower()
    if substance in settings.OSM_SUBSTANCE_MAP:
        return settings.OSM_SUBSTANCE_MAP[substance]

    power = tags.get("power", "").lower()
    if power in ("cable", "line"):
        return "electricity"

    utility = tags.get("utility", "").lower()
    if utility in settings.OSM_SUBSTANCE_MAP:
        return settings.OSM_SUBSTANCE_MAP[utility]

    waterway = tags.get("waterway", "").lower()
    if waterway in ("drain", "canal"):
        return "sewage"

    man_made = tags.get("man_made", "").lower()
    if man_made == "pipeline":
        return "unknown_pipeline"

    return "unknown"


def _parse_diameter(value: Optional[str]) -> Optional[float]:
    """Парсит диаметр трубы в мм из строки OSM (форматы: «300», «DN300», «0.3 m»)."""
    if not value:
        return None
    value = value.strip().upper()
    # Убираем префиксы типа DN, D
    for prefix in ("DN", "NW", "D"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    # Если есть «m» (метры), переводим в мм
    if value.endswith(" M") or value.endswith("M"):
        try:
            return float(value.replace("M", "").strip()) * 1000
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    """Безопасный парсинг float."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _parse_int(value: Optional[str]) -> int:
    """Безопасный парсинг int."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Публичный API модуля
# ─────────────────────────────────────────────────────────────────────────────

def load_utilities_by_bbox(
    bbox: tuple,
    target_crs: str = settings.CRS_UTM37N,
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Загружает инженерные коммуникации из OSM для прямоугольной области.

    Args:
        bbox:       (south, west, north, east) в градусах WGS-84.
                    Пример для центра Москвы: (55.73, 37.60, 55.77, 37.66)
        target_crs: CRS результирующего GeoDataFrame.
                    По умолчанию EPSG:32637 (UTM 37N, метрическая).
        use_cache:  Использовать кеш ответов (True по умолчанию).

    Returns:
        GeoDataFrame с коммуникациями в target_crs.

    Example:
        >>> gdf = load_utilities_by_bbox((55.73, 37.60, 55.77, 37.66))
        >>> gdf[['type', 'diameter', 'depth', 'geometry']].head()
    """
    query = _build_bbox_query(bbox)
    data  = _execute_query(query, use_cache=use_cache)
    gdf   = _parse_overpass_response(data)

    if gdf.empty:
        return gdf

    return gdf.to_crs(target_crs)


def load_utilities_by_area(
    area_name: str,
    target_crs: str = settings.CRS_UTM37N,
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Загружает инженерные коммуникации из OSM для именованного района.

    Args:
        area_name:  Название района в OSM (на русском или английском).
                    Примеры: «Замоскворечье», «Хамовники», «Москва»
        target_crs: CRS результирующего GeoDataFrame.
        use_cache:  Использовать кеш.

    Returns:
        GeoDataFrame с коммуникациями в target_crs.

    Example:
        >>> gdf = load_utilities_by_area("Замоскворечье")
        >>> print(gdf.groupby('type').size())
    """
    query = _build_area_query(area_name)
    data  = _execute_query(query, use_cache=use_cache)
    gdf   = _parse_overpass_response(data)

    if gdf.empty:
        return gdf

    return gdf.to_crs(target_crs)


def load_utilities_for_survey(
    survey_geom,
    buffer_m: float = 200.0,
    target_crs: str = settings.CRS_UTM37N,
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Загружает OSM-коммуникации для зоны геофизической съёмки.

    Принимает геометрию участка съёмки (в любой CRS), добавляет буфер,
    конвертирует в WGS-84 и запрашивает Overpass по bbox этой области.

    Args:
        survey_geom: shapely-геометрия (Polygon/LineString/Point) зоны съёмки.
                     Может быть в любой CRS — укажите через survey_crs.
        buffer_m:    Буфер вокруг участка в метрах (загружаем чуть больше, чем надо).
        target_crs:  CRS результата.
        use_cache:   Использовать кеш.

    Returns:
        GeoDataFrame с коммуникациями в target_crs.

    Example:
        >>> from shapely.geometry import box
        >>> # Полигон участка съёмки в EPSG:32637
        >>> poly = box(413000, 6177000, 414500, 6178500)
        >>> gdf = load_utilities_for_survey(poly)
    """
    # Создаём временный GeoDataFrame для конвертации CRS
    tmp = gpd.GeoDataFrame(geometry=[survey_geom], crs=target_crs)
    # Буфер в метрической CRS
    tmp["geometry"] = tmp.geometry.buffer(buffer_m)
    # Конвертируем в WGS84 для Overpass
    tmp_wgs = tmp.to_crs(settings.CRS_WGS84)
    bounds  = tmp_wgs.total_bounds  # (minx, miny, maxx, maxy) = (W, S, E, N)

    bbox = (bounds[1], bounds[0], bounds[3], bounds[2])  # S, W, N, E
    logger.info(f"OSM: запрос для зоны съёмки, bbox={bbox}")

    return load_utilities_by_bbox(bbox, target_crs=target_crs, use_cache=use_cache)


def get_utilities_summary(gdf: gpd.GeoDataFrame) -> dict:
    """
    Возвращает краткую статистику по загруженным коммуникациям.

    Args:
        gdf: GeoDataFrame с коммуникациями (результат load_utilities_*).

    Returns:
        Словарь со статистикой: кол-во по типам, суммарная длина и т.д.
    """
    if gdf.empty:
        return {"total": 0, "by_type": {}, "total_length_m": 0.0}

    # Длина в метрах (требует метрической CRS)
    try:
        total_length = gdf.geometry.length.sum()
    except Exception:
        total_length = None

    by_type = gdf.groupby("type").agg(
        count=("osm_id", "count"),
        length_m=("geometry", lambda x: x.length.sum()),
    ).to_dict(orient="index")

    return {
        "total":          len(gdf),
        "by_type":        by_type,
        "total_length_m": total_length,
        "has_diameter":   gdf["diameter"].notna().sum(),
        "has_depth":      gdf["depth"].notna().sum(),
        "has_material":   gdf["material"].notna().sum(),
    }


def clear_cache() -> int:
    """Удаляет все файлы кеша OSM. Возвращает количество удалённых файлов."""
    cache_dir = Path(settings.OVERPASS_CACHE_DIR)
    count = 0
    if cache_dir.exists():
        for f in cache_dir.glob("*.json"):
            f.unlink()
            count += 1
    logger.info(f"OSM: кеш очищен ({count} файлов)")
    return count
