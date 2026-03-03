"""
Локальная ручная настройка ключа (упрощённый режим).

Если не хочешь использовать .env.local или поля в UI/API,
просто вставь ключ в переменную CURSOR_API_KEY ниже.
"""

# ================== ВСТАВЬ СЮДА СВОЙ CURSOR API KEY ==================
# Пример: "crsr_xxxxxxxxxxxxxxxxx"
CURSOR_API_KEY = "PASTE_YOUR_CURSOR_API_KEY_HERE"
# =====================================================================

# Базовые параметры подключения.
# Если /api/llm/check показывает 404 для CURSOR_LLM_PATH, укажи путь своего
# openai-compatible gateway (например, /v1/chat/completions на прокси-провайдере).
CURSOR_BASE_URL = "https://api.cursor.com"
CURSOR_LLM_PATH = "/v1/chat/completions"
CURSOR_AUTH_MODE = "bearer"  # bearer | basic
CURSOR_MODEL = "gpt-4o-mini"

