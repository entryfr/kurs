# Geophysical Utilities Risk Pipeline

Прототип Python-пайплайна для анализа геофизических данных и оценки рисков
повреждения подземных инженерных коммуникаций.

## Что делает проект

1. Загружает входные данные:
   - магнитную сетку (`.npy`);
   - обнаруженные аномалии (`.gpkg`);
   - учтённые коммуникации (`.gpkg`).
2. Выполняет детекцию линейных аномалий на магнитной карте.
3. Определяет вероятный тип коммуникации для каждой аномалии.
4. Классифицирует уровень риска (`LOW` / `HIGH` / `CRITICAL`).
5. Проверяет нормативные условия (модуль `validator`).
6. Генерирует артефакты отчётности:
   - `scheme.dxf`
   - `miis.xml`
   - `act.docx`

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
  --output-dir output \
  --anomaly-threshold 30.0
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
- `http://localhost:8000/docs` — Swagger UI для API;
- `http://localhost:8000/healthz` — health-check.

#### API запуск (пример)

```bash
curl -X POST "http://localhost:8000/api/run" \
  -H "Content-Type: application/json" \
  -d '{
    "magnetic_grid": "data/raw/magnetic_grid.npy",
    "anomalies": "data/processed/anomalies.gpkg",
    "utilities": "data/processed/utilities.gpkg",
    "output_dir": "output",
    "anomaly_threshold": 30.0
  }'
```

## Тесты

```bash
pytest tests/test_coordinate_transformer.py -v
pytest test_osm_loader.py -v
pytest tests/test_utility_type_identifier.py -v
pytest tests/test_web_app.py -v
```

## Текущий статус

Проект находится в стадии прототипа (R&D):
- часть расчётов реализована как приближённые эвристики;
- в `classification` предусмотрен fallback-режим без обученных моделей;
- для production рекомендуется расширить тестовое покрытие и CI/CD.
