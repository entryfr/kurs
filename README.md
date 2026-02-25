# Geophysical Utilities Risk Pipeline

Прототип Python-пайплайна для анализа геофизических данных и оценки рисков
повреждения подземных инженерных коммуникаций.

## Что делает проект

1. Загружает входные данные:
   - магнитную сетку (`.npy` или `GeoTIFF`);
   - обнаруженные аномалии и коммуникации (`.gpkg`) либо из `MIIS XML`;
   - опциональные зоны строительства (`.gpkg`);
   - опциональные признаки из `SEG-Y`.
2. Выполняет детекцию линейных аномалий на магнитной карте.
3. Фильтрует короткие линейные аномалии (`<=10 м`) по ТЗ.
4. Определяет вероятный тип коммуникации для каждой аномалии.
5. Классифицирует уровень риска (`LOW` / `HIGH` / `CRITICAL`) c приоритетом ТЗ-правил:
   - `CRITICAL` = `89%` (нет пересечения с учтёнными + в зоне строительства),
   - `HIGH` = `62%` (есть пересечение, но рассогласование глубины `>0.5 м`),
   - `LOW` = `12%` (совпадение по координате/глубине).
6. Выделяет «слепые зоны» по магнитной карте и учитывает их в объяснениях риска.
7. Выполняет оценку остаточного ресурса коммуникации (коррозия) и формирует предупреждения.
8. Проверяет нормативные условия (модуль `validator`).
9. Генерирует артефакты отчётности:
   - `scheme.dxf`
   - `miis.xml`
   - `mins_exchange.xml`
   - `act.docx`
   - `act.pdf` (если доступен `reportlab`)
10. Опционально сохраняет результаты запусков в PostgreSQL/PostGIS
    (таблицы запусков и аномалий + API чтения истории из БД).

## Структура проекта

- `main.py` — CLI запуск демо-пайплайна.
- `src/pipeline` — сервисный слой запуска (`run_pipeline`), общий для CLI и web.
- `src/web` — FastAPI веб-интерфейс.
- `src/preprocessing` — предобработка данных, трансформации координат, OSM loader.
- `src/detection` — детекция аномалий, буферный анализ, слепые зоны.
- `src/classification` — классификация риска/типа коммуникаций, оценка коррозии.
- `src/agent` — оркестратор и нормативная валидация.
- `src/reporting` — экспорт DXF/XML/DOCX.
- `templates/web` — HTML-шаблон веб-интерфейса.
- `data/` — входные демо-данные.
- `output/` — результирующие файлы.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Генерация тестовых данных (опционально)

```bash
python generate_test_data.py
```

### Запуск пайплайна

```bash
python main.py \
  --magnetic-grid data/raw/magnetic_grid.npy \
  --anomalies data/processed/anomalies.gpkg \
  --utilities data/processed/utilities.gpkg \
  --construction-zones data/processed/construction_zones.gpkg \
  --output-dir output \
  --anomaly-threshold 30.0 \
  --norms-profile normative
```

#### Запуск из MIIS XML + GeoTIFF + SEG-Y

```bash
python main.py \
  --magnetic-grid data/raw/magnetic_grid.tif \
  --miis-xml data/raw/input.miis.xml \
  --segy data/raw/profile.segy \
  --output-dir output \
  --anomaly-threshold 30.0 \
  --norms-profile normative
```

### Запуск через веб-интерфейс

```bash
uvicorn src.web.app:app --host 0.0.0.0 --port 8000
```

или

```bash
python web_app.py
```

После старта откройте:
- `http://localhost:8000` — HTML-форма запуска пайплайна;
- `http://localhost:8000/chat` — чат с агентом;
- `http://localhost:8000/docs` — Swagger UI для API;
- `http://localhost:8000/healthz` — health-check.

В web-форме доступны:
- синхронный запуск;
- асинхронный запуск с выдачей `job_id`;
- загрузка входных файлов (`npy/tif/gpkg/xml/segy`) напрямую через браузер;
- история последних запусков.

В чате доступны:
- web-страница диалога с сессиями;
- API endpoint для сообщений (`POST /api/chat`);
- API истории (`GET /api/chat/sessions`, `GET /api/chat/sessions/{session_id}`).
- таймаут и fallback-режим при недоступности LLM-провайдера.

### Опциональное хранение результатов в PostGIS

```bash
export ENABLE_DB_PERSISTENCE=1
export POSTGRES_DSN="postgresql+psycopg2://user:password@localhost:5432/geophysics"
```

После включения persistence:
- каждый запуск сохраняется в таблицы `pipeline_runs` и `anomaly_records`;
- при старте автоматически выполняется попытка включения расширений `postgis`, `pointcloud`, `pointcloud_postgis`;
- доступны API:
  - `GET /api/db/runs?limit=20`
  - `GET /api/db/runs/{run_uid}`

#### API запуск (пример)

```bash
curl -X POST "http://localhost:8000/api/run" \
  -H "Content-Type: application/json" \
  -d '{
    "magnetic_grid": "data/raw/magnetic_grid.npy",
    "anomalies": "data/processed/anomalies.gpkg",
    "utilities": "data/processed/utilities.gpkg",
    "construction_zones": "data/processed/construction_zones.gpkg",
    "miis_xml": "",
    "segy_file": "",
    "output_dir": "output",
    "anomaly_threshold": 30.0,
    "norms_profile": "normative"
  }'
```

#### Асинхронный API запуск и статус

```bash
curl -X POST "http://localhost:8000/api/run/async" \
  -H "Content-Type: application/json" \
  -d '{
    "magnetic_grid": "data/raw/magnetic_grid.npy",
    "anomalies": "data/processed/anomalies.gpkg",
    "utilities": "data/processed/utilities.gpkg",
    "output_dir": "output",
    "anomaly_threshold": 30.0,
    "norms_profile": "normative"
  }'
```

Ответ содержит `job_id` и `status_url`, далее:

```bash
curl "http://localhost:8000/api/run/<job_id>"
curl "http://localhost:8000/api/runs?limit=20"
```

#### API чат (пример)

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Проанализируй риски для участка с котлованом 6.5 м"
  }'
```

## Тесты

```bash
pytest tests/test_coordinate_transformer.py -v
pytest test_osm_loader.py -v
pytest tests/test_utility_type_identifier.py -v
pytest tests/test_web_app.py -v
pytest tests/test_input_loader.py -v
pytest tests/test_validator_rules.py -v
pytest tests/test_risk_tz_rules.py -v
pytest tests/test_blind_zone_pipeline.py -v
pytest tests/test_corrosion_pipeline.py -v
pytest tests/test_linear_threshold.py -v
pytest tests/test_act_generator.py -v
pytest tests/test_mins_exporter.py -v
```

## Текущий статус

Проект находится в стадии прототипа (R&D):
- часть расчётов реализована как приближённые эвристики;
- в `classification` предусмотрен fallback-режим без обученных моделей;
- для production рекомендуется расширить тестовое покрытие и CI/CD.
