# Локальный гайд: запуск AI Agent V2

## 1) Что нужно установить

- Python 3.11+ (рекомендуется 3.12)
- Git

Проверка:

```bash
python --version
git --version
```

---

## 2) Клонирование и установка зависимостей

```bash
git clone https://github.com/entryfr/kurs.git
cd kurs
git checkout "cursor/-bc-9fa0f8b0-05e1-41b9-b8c5-6435a6630dd5-d2aa"

python -m venv .venv
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3) Настройка ключа LLM

### Вариант 0 (самый простой)

Открой файл `src/ai_agent_course/local_key.py` и вставь ключ в:

```python
CURSOR_API_KEY = "PASTE_YOUR_CURSOR_API_KEY_HERE"
```

Этого уже достаточно для запуска через Cursor API.

Создай файл:

```bash
cp .env.local.example .env.local
# Windows:
# copy .env.local.example .env.local
```

Заполни один из вариантов.

### Вариант A (Anthropic)

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

### Вариант B (OpenAI-compatible)

```env
LLM_PROVIDER=openai_compatible
LLM_API_KEY=your_key
LLM_BASE_URL=https://your-endpoint.example/v1
LLM_MODEL=gpt-4o-mini
```

### Вариант C (Cursor API)

```env
LLM_PROVIDER=cursor
CURSOR_API_KEY=crsr_xxx_or_key_xxx
CURSOR_BASE_URL=https://api.cursor.com
CURSOR_LLM_PATH=/v1/chat/completions
CURSOR_AUTH_MODE=bearer
CURSOR_MODEL=gpt-4o-mini
```

> `.env.local` не коммитится в git.

---

## 4) Запуск веб-версии

```bash
python web_app.py
```

Открой:
- `http://localhost:8000` — UI
- `http://localhost:8000/docs` — API

---

## 5) Какая картинка нужна для агента

Минимальные требования:

1. Формат: `PNG`/`JPG`/`JPEG`.
2. Разрешение: желательно от `1024x1024` (минимум `512x512`).
3. Контраст: линейные аномалии должны быть видимы (светлые/тёмные полосы).
4. Без сильных артефактов:
   - не размытая;
   - без сильной компрессии;
   - без «водяных знаков» поверх области интереса.
5. Масштаб (желательно): чтобы 1 пиксель был близок к 5 м (или известен масштаб для корректной длины).

Что лучше всего подходит:
- изображение карты магнитной аномальности участка;
- линейные структуры (предполагаемые трассы) должны быть различимы;
- если есть MIIS XML — загрузи его вторым файлом для сопоставления.

---

## 6) Пример запроса в UI

```text
Проанализируй участок строительства ЖК "Нагатинский", координаты 55.6721°N, 37.6415°E, площадь 8.4 га, глубина котлована 6.5 м, аномалия магнитного поля: амплитуда 68 нТл, протяжённость 42 м, глубина по ВЭЗ 1.8 м.
```

---

## 7) Проверка ключа перед запуском анализа

В UI нажми кнопку **«Проверить LLM ключ»**.

API-эквивалент:

```bash
curl -X POST "http://localhost:8000/api/llm/check" \
  -F "llm_provider=cursor" \
  -F "llm_api_key=..." \
  -F "llm_model=gpt-4o-mini" \
  -F "llm_base_url=https://api.cursor.com"
```

---

## 8) Запуск через CLI

```bash
python main.py \
  --image data/raw/example.png \
  --prompt "Проанализируй участок строительства..." \
  --miis-xml data/raw/input.miis.xml \
  --output-dir output_v2
```

---

## 9) Где смотреть результаты

После запуска:
- `.cache/agent_v2/runs/<run_id>/scheme_v2.dxf`
- `.cache/agent_v2/runs/<run_id>/unaccounted_v2.miis.xml`
- `.cache/agent_v2/runs/<run_id>/act_v2.pdf`

История запусков:
- `GET /api/runs`

---

## 10) Если LLM не работает

1. Проверь `.env.local`.
2. Проверь endpoint через `/api/llm/check`.
3. Убедись, что провайдер/модель доступны.
4. Если ключ невалиден — агент всё равно выполнит расчёты в fallback-режиме.
