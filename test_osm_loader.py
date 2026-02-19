# tests/test_osm_loader.py
"""
Тесты для модуля osm_loader.py и обновлённого validator.py.

Запуск:
    pytest tests/test_osm_loader.py -v

Зависимости для тестирования:
    pip install pytest pytest-mock responses shapely geopandas
"""

import json
import pytest
import numpy as np
import geopandas as gpd
from pathlib import Path
from unittest.mock import patch, MagicMock
from shapely.geometry import LineString, Point, box

# ─────────────────────────────────────────────────────────────────────────────
# Фикстуры
# ─────────────────────────────────────────────────────────────────────────────

MOCK_OVERPASS_RESPONSE = {
    "version": 0.6,
    "elements": [
        # Узлы (nodes)
        {"type": "node", "id": 1, "lon": 37.620, "lat": 55.750},
        {"type": "node", "id": 2, "lon": 37.625, "lat": 55.752},
        {"type": "node", "id": 3, "lon": 37.630, "lat": 55.751},
        {"type": "node", "id": 4, "lon": 37.635, "lat": 55.755},
        # Газопровод
        {
            "type": "way", "id": 100,
            "nodes": [1, 2],
            "tags": {
                "man_made": "pipeline",
                "substance": "gas",
                "diameter": "DN300",
                "material": "steel",
                "depth": "1.5",
                "operator": "Мосгаз",
                "location": "underground",
            }
        },
        # Водопровод
        {
            "type": "way", "id": 101,
            "nodes": [2, 3],
            "tags": {
                "man_made": "pipeline",
                "substance": "water",
                "diameter": "200",
                "material": "cast_iron",
                "depth": "2.0",
            }
        },
        # Электрокабель
        {
            "type": "way", "id": 102,
            "nodes": [3, 4],
            "tags": {
                "power": "cable",
                "voltage": "10000",
                "operator": "МОЭСК",
            }
        },
        # Way с недостаточным числом узлов (должен быть пропущен)
        {
            "type": "way", "id": 103,
            "nodes": [99],  # нет в nodes
            "tags": {"man_made": "pipeline"}
        },
        # Теплосеть
        {
            "type": "way", "id": 104,
            "nodes": [1, 3],
            "tags": {
                "man_made": "pipeline",
                "substance": "hot_water",
                "diameter": "0.5 m",  # метры — должен конвертироваться в 500 мм
            }
        },
    ]
}


@pytest.fixture
def mock_overpass(tmp_path, monkeypatch):
    """Подменяет HTTP-запрос к Overpass API мок-ответом."""
    monkeypatch.setattr("config.settings.OVERPASS_CACHE_DIR", str(tmp_path / "osm_cache"))

    with patch("src.preprocessing.osm_loader.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_OVERPASS_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def sample_utilities_gdf():
    """GeoDataFrame с тестовыми коммуникациями в EPSG:32637."""
    return gpd.GeoDataFrame(
        {
            "type":     ["gas",       "electricity", "water",      "heating"],
            "diameter": [300.0,       None,          200.0,        500.0],
            "depth":    [1.5,         1.0,           2.0,          2.5],
            "osm_id":   [100,         102,           101,          104],
            "geometry": [
                LineString([(413000, 6177000), (413500, 6177000)]),
                LineString([(413500, 6177000), (414000, 6177000)]),
                LineString([(413200, 6177200), (413700, 6177200)]),
                LineString([(413100, 6177100), (413600, 6177100)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )


@pytest.fixture
def sample_anomalies_gdf():
    """GeoDataFrame с тестовыми аномалиями."""
    return gpd.GeoDataFrame(
        {
            "risk_class":  ["CRITICAL", "HIGH", "LOW"],
            "depth":       [0.8,         2.5,   3.0],   # 0.8 м < 1.4 м промерзания
            "has_utility": [True,        False, True],
            "geometry": [
                Point(413100, 6177050),
                Point(413800, 6177800),   # вдали от коммуникаций
                Point(413300, 6177200),
            ],
        },
        geometry="geometry",
        crs="EPSG:32637",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Тесты osm_loader.py
# ─────────────────────────────────────────────────────────────────────────────

class TestParseOverpassResponse:
    """Тесты разбора ответа Overpass API."""

    def test_basic_parsing(self):
        """Проверяем, что все объекты с достаточным числом узлов разбираются."""
        from src.preprocessing.osm_loader import _parse_overpass_response
        gdf = _parse_overpass_response(MOCK_OVERPASS_RESPONSE)
        # Way 103 пропускается (нет узла 99 в nodes)
        assert len(gdf) == 4

    def test_geometry_is_linestring(self):
        """Все геометрии должны быть LineString."""
        from src.preprocessing.osm_loader import _parse_overpass_response
        gdf = _parse_overpass_response(MOCK_OVERPASS_RESPONSE)
        for geom in gdf.geometry:
            assert isinstance(geom, LineString), f"Ожидался LineString, получен {type(geom)}"

    def test_crs_is_wgs84(self):
        """CRS результата должна быть WGS-84."""
        from src.preprocessing.osm_loader import _parse_overpass_response
        gdf = _parse_overpass_response(MOCK_OVERPASS_RESPONSE)
        assert gdf.crs.to_epsg() == 4326

    def test_empty_response(self):
        """Пустой ответ должен возвращать пустой GeoDataFrame."""
        from src.preprocessing.osm_loader import _parse_overpass_response
        gdf = _parse_overpass_response({"elements": []})
        assert gdf.empty

    def test_type_normalization(self):
        """Тип коммуникации должен нормализоваться из OSM-тегов."""
        from src.preprocessing.osm_loader import _parse_overpass_response
        gdf = _parse_overpass_response(MOCK_OVERPASS_RESPONSE)
        types = set(gdf["type"].unique())
        assert "gas" in types
        assert "water" in types
        assert "electricity" in types
        assert "heating" in types  # hot_water -> heating


class TestDiameterParsing:
    """Тесты парсинга диаметра трубы."""

    def test_dn_prefix(self):
        from src.preprocessing.osm_loader import _parse_diameter
        assert _parse_diameter("DN300") == 300.0

    def test_plain_number(self):
        from src.preprocessing.osm_loader import _parse_diameter
        assert _parse_diameter("200") == 200.0

    def test_meters(self):
        from src.preprocessing.osm_loader import _parse_diameter
        assert _parse_diameter("0.5 m") == 500.0

    def test_none(self):
        from src.preprocessing.osm_loader import _parse_diameter
        assert _parse_diameter(None) is None

    def test_invalid(self):
        from src.preprocessing.osm_loader import _parse_diameter
        assert _parse_diameter("unknown") is None


class TestLoadByBbox:
    """Тесты загрузки по bbox с мокированием HTTP."""

    def test_returns_geodataframe(self, mock_overpass):
        from src.preprocessing.osm_loader import load_utilities_by_bbox
        bbox = (55.73, 37.60, 55.77, 37.66)
        gdf = load_utilities_by_bbox(bbox, use_cache=False)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 4

    def test_reprojected_to_utm(self, mock_overpass):
        from src.preprocessing.osm_loader import load_utilities_by_bbox
        bbox = (55.73, 37.60, 55.77, 37.66)
        gdf = load_utilities_by_bbox(bbox, target_crs="EPSG:32637", use_cache=False)
        assert gdf.crs.to_epsg() == 32637

    def test_columns_present(self, mock_overpass):
        from src.preprocessing.osm_loader import load_utilities_by_bbox
        bbox = (55.73, 37.60, 55.77, 37.66)
        gdf = load_utilities_by_bbox(bbox, use_cache=False)
        expected_cols = {"type", "diameter", "depth", "material", "osm_id", "operator"}
        assert expected_cols.issubset(set(gdf.columns))

    def test_cache_used_on_second_call(self, mock_overpass, tmp_path, monkeypatch):
        """Второй вызов с теми же параметрами должен использовать кеш (не HTTP)."""
        monkeypatch.setattr("config.settings.OVERPASS_CACHE_DIR", str(tmp_path / "cache"))
        from src.preprocessing.osm_loader import load_utilities_by_bbox
        bbox = (55.73, 37.60, 55.77, 37.66)
        load_utilities_by_bbox(bbox, use_cache=True)
        load_utilities_by_bbox(bbox, use_cache=True)
        # Первый вызов создаёт кеш, второй читает его — HTTP вызван только 1 раз
        assert mock_overpass.call_count == 1


class TestGetUtilitiesSummary:
    """Тесты функции статистики."""

    def test_summary_structure(self, mock_overpass):
        from src.preprocessing.osm_loader import load_utilities_by_bbox, get_utilities_summary
        gdf = load_utilities_by_bbox((55.73, 37.60, 55.77, 37.66), use_cache=False)
        summary = get_utilities_summary(gdf)
        assert "total" in summary
        assert "by_type" in summary
        assert "total_length_m" in summary
        assert summary["total"] == 4

    def test_summary_empty(self):
        from src.preprocessing.osm_loader import get_utilities_summary
        empty = gpd.GeoDataFrame(columns=["type", "geometry"], geometry="geometry", crs="EPSG:32637")
        summary = get_utilities_summary(empty)
        assert summary["total"] == 0


class TestClearCache:
    """Тест очистки кеша."""

    def test_clear_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.settings.OVERPASS_CACHE_DIR", str(tmp_path / "cache"))
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Создаём фиктивные файлы кеша
        for i in range(3):
            (cache_dir / f"test_{i}.json").write_text("{}")
        from src.preprocessing.osm_loader import clear_cache
        count = clear_cache()
        assert count == 3
        assert list(cache_dir.glob("*.json")) == []


# ─────────────────────────────────────────────────────────────────────────────
# Тесты validator.py
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateAnomalies:
    """Тесты нормативной валидации аномалий."""

    def test_critical_with_nearby_utility_no_error(
        self, sample_anomalies_gdf, sample_utilities_gdf
    ):
        """CRITICAL аномалия с коммуникацией рядом — не должна давать ошибку по правилу 1."""
        from src.agent.validator import validate_anomalies
        # Аномалия #0 (CRITICAL) находится на (413100, 6177050) — рядом есть газопровод
        errors = validate_anomalies(sample_anomalies_gdf, sample_utilities_gdf, buffer_dist=100.0)
        rule1_errors = [e for e in errors if "Аномалия #0" in e and "нет учтённых" in e]
        assert len(rule1_errors) == 0

    def test_critical_without_utilities_gives_error(self, sample_anomalies_gdf):
        """CRITICAL аномалия без коммуникаций в буфере — должна давать ошибку."""
        from src.agent.validator import validate_anomalies
        empty_utilities = gpd.GeoDataFrame(
            columns=["type", "geometry"], geometry="geometry", crs="EPSG:32637"
        )
        errors = validate_anomalies(sample_anomalies_gdf, empty_utilities, buffer_dist=10.0)
        assert any("нет учтённых коммуникаций" in e for e in errors)

    def test_shallow_anomaly_warning(self, sample_anomalies_gdf, sample_utilities_gdf):
        """Аномалия с depth < 1.4 м (глубина промерзания) должна давать предупреждение."""
        from src.agent.validator import validate_anomalies
        # Аномалия #0 имеет depth=0.8 < 1.4 м
        errors = validate_anomalies(sample_anomalies_gdf, sample_utilities_gdf)
        assert any("глубина залегания" in e and "0.80" in e for e in errors)

    def test_high_unconfirmed_warning(self, sample_anomalies_gdf, sample_utilities_gdf):
        """HIGH аномалия без сопоставленной коммуникации — предупреждение."""
        from src.agent.validator import validate_anomalies
        errors = validate_anomalies(sample_anomalies_gdf, sample_utilities_gdf, buffer_dist=1.0)
        assert any("HIGH" in e and "#1" in e for e in errors)

    def test_crs_mismatch_handled(self, sample_anomalies_gdf, sample_utilities_gdf):
        """Несовпадение CRS должно обрабатываться автоматически."""
        from src.agent.validator import validate_anomalies
        utilities_wgs = sample_utilities_gdf.to_crs("EPSG:4326")
        # Не должно бросать исключение
        errors = validate_anomalies(sample_anomalies_gdf, utilities_wgs)
        assert isinstance(errors, list)

    def test_empty_anomalies(self, sample_utilities_gdf):
        """Нет аномалий — нет ошибок."""
        from src.agent.validator import validate_anomalies
        empty = gpd.GeoDataFrame(
            columns=["risk_class", "depth", "has_utility", "geometry"],
            geometry="geometry", crs="EPSG:32637"
        )
        errors = validate_anomalies(empty, sample_utilities_gdf)
        assert errors == []


class TestCheckCompliance:
    """Тесты проверки минимальных расстояний СП 42."""

    def test_gas_electricity_too_close(self):
        """Газопровод и кабель ближе 1 м — нарушение."""
        from src.agent.validator import check_compliance
        utilities = gpd.GeoDataFrame(
            {
                "type":     ["gas",        "electricity"],
                "osm_id":   [1,            2],
                "geometry": [
                    LineString([(0, 0), (10, 0)]),
                    LineString([(0, 0.3), (10, 0.3)]),  # 0.3 м < 1 м
                ],
            },
            geometry="geometry", crs="EPSG:32637",
        )
        issues = check_compliance(gpd.GeoDataFrame(), utilities)
        assert any("gas" in i.lower() and "electricity" in i.lower() for i in issues)
        assert any("0.30 м" in i for i in issues)

    def test_compliant_distances(self):
        """Все коммуникации на допустимом расстоянии — нет нарушений."""
        from src.agent.validator import check_compliance
        utilities = gpd.GeoDataFrame(
            {
                "type":     ["gas",          "electricity"],
                "osm_id":   [1,              2],
                "geometry": [
                    LineString([(0, 0), (10, 0)]),
                    LineString([(0, 5), (10, 5)]),  # 5 м > 1 м — OK
                ],
            },
            geometry="geometry", crs="EPSG:32637",
        )
        issues = check_compliance(gpd.GeoDataFrame(), utilities)
        assert issues == []

    def test_empty_utilities(self):
        """Нет коммуникаций — нет нарушений."""
        from src.agent.validator import check_compliance
        empty = gpd.GeoDataFrame(columns=["type", "geometry"], geometry="geometry", crs="EPSG:32637")
        assert check_compliance(empty, empty) == []


class TestGenerateValidationReport:
    """Тесты генерации текстового отчёта."""

    def test_no_errors(self):
        from src.agent.validator import generate_validation_report
        report = generate_validation_report([], [])
        assert "Нарушений не выявлено" in report

    def test_with_errors(self):
        from src.agent.validator import generate_validation_report
        report = generate_validation_report(
            ["[CRITICAL] Аномалия #0: нет коммуникаций"],
            ["[СП 42] gas и electricity: 0.3 м < 1 м"]
        )
        assert "Аномалии без подтверждения" in report
        assert "Нарушения минимальных расстояний" in report
