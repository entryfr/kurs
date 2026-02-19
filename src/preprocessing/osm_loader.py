"""
Совместимый импорт OSM-загрузчика внутри пакета src.preprocessing.

Исторически реализация лежит в корне репозитория (osm_loader.py), а модули
проекта импортируют её как `src.preprocessing.osm_loader`.
Этот модуль экспортирует тот же API, чтобы сохранить обратную совместимость.
"""

from osm_loader import (  # noqa: F401
    _build_area_query,
    _build_bbox_query,
    _cache_path,
    _execute_query,
    _is_cache_valid,
    _normalize_type,
    _parse_diameter,
    _parse_float,
    _parse_int,
    _parse_overpass_response,
    clear_cache,
    get_utilities_summary,
    load_utilities_by_area,
    load_utilities_by_bbox,
    load_utilities_for_survey,
    requests,
)

__all__ = [
    "_build_area_query",
    "_build_bbox_query",
    "_cache_path",
    "_execute_query",
    "_is_cache_valid",
    "_normalize_type",
    "_parse_diameter",
    "_parse_float",
    "_parse_int",
    "_parse_overpass_response",
    "clear_cache",
    "get_utilities_summary",
    "load_utilities_by_area",
    "load_utilities_by_bbox",
    "load_utilities_for_survey",
    "requests",
]
