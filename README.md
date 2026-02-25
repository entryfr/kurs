# AI Agent V2 — Курсовой проект

Новый проект с нуля: загрузка **картинки** и **текстового запроса** для выявления
неучтённых подземных коммуникаций по ТЗ.

## Что уже реализовано

- Входной сценарий: `image + prompt (+ optional MIIS XML)`.
- Оркестратор из 6 шагов:
  1. аффинная трансформация (LSQ);
  2. детекция линейных аномалий (Median + Canny + Hough), фильтр `>10м`, буфер 2.5м;
  3. классификация типа коммуникации (RandomForest, 14 признаков);
  4. риск-модель с ТЗ-классами 89% / 62% / 12% + индекс `R`;
  5. прогноз коррозионного ресурса и предупреждение при `T_ост < 5 лет`;
  6. генерация артефактов.
- Выходные артефакты:
  - `scheme_v2.dxf`
  - `unaccounted_v2.miis.xml`
  - `act_v2.pdf`
- Веб UI и API:
  - `GET /`
  - `POST /analyze` (form)
  - `POST /api/analyze` (multipart)

## Структура

- `src/ai_agent_course/` — новый код проекта.
- `templates/agent_v2/` — интерфейс.
- `main.py` — CLI запуск V2.
- `web_app.py` — web запуск V2.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Запуск (Web)

```bash
python web_app.py
```

Открыть:
- `http://localhost:8000`
- `http://localhost:8000/docs`
- `http://localhost:8000/healthz`

### Как работает LLM без ключа

Если ключ не задан, агент всё равно работает в **детерминированном fallback-режиме**:
- выполняет pipeline и расчёты;
- генерирует артефакты;
- текстовый ответ формирует без внешнего LLM.

### Как включить LLM с ключом

Поддерживаются 2 варианта:

1) **Anthropic**
```bash
export ANTHROPIC_API_KEY="..."
export LLM_PROVIDER="anthropic"
export ANTHROPIC_MODEL="claude-3-5-sonnet-latest"
```

2) **OpenAI-compatible endpoint** (в т.ч. кастомные прокси)
```bash
export LLM_PROVIDER="openai_compatible"
export LLM_API_KEY="..."
export LLM_BASE_URL="https://your-endpoint.example/v1"
export LLM_MODEL="gpt-4o-mini"
```

В веб-форме можно передать ключ/провайдер прямо в запросе (ключ не сохраняется в файлы проекта).

## Запуск (CLI)

```bash
python main.py \
  --image data/raw/example.png \
  --prompt "Проанализируй участок строительства ЖК Нагатинский..." \
  --miis-xml data/raw/input.miis.xml \
  --output-dir output_v2 \
  --llm-provider openai_compatible \
  --llm-api-key "..." \
  --llm-base-url "https://your-endpoint.example/v1" \
  --llm-model "gpt-4o-mini"
```

## Пример API

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "prompt=Проанализируй участок строительства ЖК Нагатинский..." \
  -F "image_file=@data/raw/example.png" \
  -F "miis_xml_file=@data/raw/input.miis.xml"
```

## Тесты

```bash
pytest tests/test_agent_v2_web.py -v
pytest tests/test_llm_client.py -v
```
