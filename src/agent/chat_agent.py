from __future__ import annotations

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

logger = logging.getLogger(__name__)

_agent_lock = threading.Lock()
_cached_agent = None
_MAX_HISTORY_MESSAGES = int(os.getenv("CHAT_AGENT_MAX_HISTORY", "10"))
_MAX_PROMPT_CHARS = int(os.getenv("CHAT_AGENT_MAX_PROMPT_CHARS", "12000"))
_CALL_TIMEOUT_SECONDS = float(os.getenv("CHAT_AGENT_TIMEOUT_SEC", "20"))


def _get_agent():
    global _cached_agent  # noqa: PLW0603
    with _agent_lock:
        if _cached_agent is None:
            from src.agent.orchestrator import create_agent

            _cached_agent = create_agent()
        return _cached_agent


def _compose_prompt(
    message: str,
    session_messages: list[dict[str, Any]] | None = None,
    latest_run: dict[str, Any] | None = None,
) -> str:
    history = []
    for msg in (session_messages or [])[-_MAX_HISTORY_MESSAGES:]:
        role = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()
        if content:
            history.append(f"{role.upper()}: {content}")

    context_lines = [
        "Ты инженерный ИИ-ассистент по выявлению неучтённых подземных коммуникаций.",
        "Отвечай на русском, кратко и прикладно.",
        "Учитывай ТЗ: МИИС/GeoTIFF/SEG-Y, СП 47, СП 11-102-97, буферный анализ, риск-классы.",
    ]
    if latest_run:
        compact_latest_run = dict(latest_run)
        compact_latest_run.pop("risk_details", None)
        compact_latest_run.pop("validation_issue_details", None)
        compact_latest_run.pop("validation_report", None)
        context_lines.append(f"Контекст последнего запуска: {compact_latest_run}")

    prompt_parts = [
        "\n".join(context_lines),
        "",
        "История диалога:",
        "\n".join(history) if history else "(пусто)",
        "",
        f"Новый вопрос пользователя: {message}",
        "Сформируй полезный технический ответ.",
    ]
    prompt = "\n".join(prompt_parts)
    if len(prompt) > _MAX_PROMPT_CHARS:
        # Сохраняем конец (последний вопрос + релевантный контекст), чтобы не терять intent.
        prompt = prompt[-_MAX_PROMPT_CHARS:]
    return prompt


def _invoke_with_timeout(agent: Any, prompt: str, timeout_seconds: float) -> Any:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(agent.invoke, {"input": prompt})
        return future.result(timeout=timeout_seconds)


def _fallback_answer(
    message: str,
    latest_run: dict[str, Any] | None = None,
    reason: str | None = None,
) -> str:
    normalized = message.lower().strip()
    hints: list[str] = []
    reason_hints_count = 0

    if reason == "no_api_key":
        hints.append(
            "LLM не подключён: переменная `ANTHROPIC_API_KEY` не задана, поэтому включён локальный fallback-режим."
        )
        hints.append(
            "Для полноценного ответа LLM задайте ключ и перезапустите приложение: "
            "`$env:ANTHROPIC_API_KEY=\"...\"` (PowerShell) или "
            "`export ANTHROPIC_API_KEY=...` (Linux/macOS)."
        )
        reason_hints_count = len(hints)
    elif reason == "timeout":
        hints.append("LLM не успел ответить в таймаут, поэтому выдан локальный fallback-ответ.")
        reason_hints_count = len(hints)
    elif reason and reason.startswith("error:"):
        hints.append(
            f"LLM временно недоступен ({reason.split(':', 1)[1]}), поэтому выдан локальный fallback-ответ."
        )
        reason_hints_count = len(hints)

    if any(word in normalized for word in ["запуск", "запуст", "run", "старт", "start", "подними"]):
        hints.append(
            "Запуск: `python web_app.py` (веб) или `python main.py --magnetic-grid ... --norms-profile normative` (CLI)."
        )
        hints.append("Проверка: `python -m pytest tests/test_web_app.py -v`.")
    if any(word in normalized for word in ["ошиб", "error", "traceback"]):
        hints.append(
            "При ошибке пришлите полный traceback и команду запуска; сначала проверьте `python -m pip install -r requirements.txt`."
        )
    if any(word in normalized for word in ["чат", "агент", "llm", "anthropic", "api key", "apikey", "ключ"]):
        hints.append(
            "Чат работает в двух режимах: LLM (`mode=llm`) и локальный fallback. "
            "Проверяйте `mode` в сообщении ассистента."
        )
    if any(word in normalized for word in ["postgis", "postgres", "бд", "db", "sql"]):
        hints.append(
            "Для сохранения запусков в БД задайте `ENABLE_DB_PERSISTENCE=1` и `POSTGRES_DSN=...`, "
            "после чего доступны `/api/db/runs` и `/api/db/runs/{run_uid}`."
        )
    if any(word in normalized for word in ["miis", "xml"]):
        hints.append(
            "Можно передать `--miis-xml <path>`: аномалии и коммуникации загрузятся напрямую из MIIS XML."
        )
    if any(word in normalized for word in ["segy", "грл", "радар"]):
        hints.append(
            "SEG-Y подключается через `--segy`; признаки (`radar_hyperbola_w`, `seg_velocity`) учитываются в оценке риска."
        )
    if any(word in normalized for word in ["норм", "сп 47", "валид"]):
        hints.append(
            "Валидация использует правила R01..R12; итог доступен в `validation_report` и в веб-блоке нарушений."
        )
    if any(word in normalized for word in ["риск", "critical", "high", "low", "89", "62", "12"]):
        hints.append(
            "ТЗ-риск: CRITICAL=89% (нет пересечения + в зоне строительства), "
            "HIGH=62% (рассогласование глубины >0.5м), LOW=12% (совпадение по координате/глубине)."
        )
    if any(word in normalized for word in ["корроз", "ресурс", "t_ост", "износ"]):
        hints.append(
            "Коррозия считается по грунтовым параметрам; при `T_ост < 5 лет` формируется предупреждение о срочной замене."
        )
    if any(word in normalized for word in ["слеп", "blind"]):
        hints.append(
            "Слепые зоны выделяются по порогу магнитной аномальности и учитываются в `risk_details`."
        )
    if any(word in normalized for word in ["dxf", "docx", "pdf", "mins", "экспорт", "акт"]):
        hints.append(
            "Экспорт артефактов: `scheme.dxf`, `miis.xml`, `mins_exchange.xml`, `act.docx` и `act.pdf` (если установлен reportlab)."
        )
    if any(word in normalized for word in ["osm", "overpass"]):
        hints.append(
            "Для OSM-запросов используйте bbox/район и кэш; при лимитах Overpass включайте ретраи и мониторинг `/api/status`."
        )

    if latest_run:
        hints.append(
            f"Последний запуск: режим `{latest_run.get('input_mode')}`, профиль `{latest_run.get('norms_profile')}`, "
            f"нарушений `{latest_run.get('validation_issues')}`."
        )

    if len(hints) == reason_hints_count:
        hints.append(
            "Дай уточнение вопроса в формате: цель, входные данные, ожидаемый результат и фактическая ошибка (если есть)."
        )
        hints.append(
            "Пример: «Запускаю `python web_app.py`, передаю MIIS XML + SEG-Y, получаю 400 на /run — помоги разобрать»."
        )

    return "\n".join(f"- {line}" for line in hints)


def ask_agent(
    message: str,
    session_messages: list[dict[str, Any]] | None = None,
    latest_run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not message.strip():
        return {"answer": "Сообщение пустое. Опишите задачу или вопрос.", "mode": "validation"}

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            agent = _get_agent()
            prompt = _compose_prompt(
                message=message,
                session_messages=session_messages,
                latest_run=latest_run,
            )
            result = _invoke_with_timeout(agent, prompt, _CALL_TIMEOUT_SECONDS)
            if isinstance(result, dict):
                output = (
                    result.get("output")
                    or result.get("result")
                    or result.get("final_answer")
                    or str(result)
                )
            else:
                output = str(result)
            output = re.sub(r"\n{3,}", "\n\n", output).strip()
            return {"answer": output, "mode": "llm"}
        except FuturesTimeoutError:
            logger.warning("Таймаут ответа LLM-агента, используем fallback")
            return {
                "answer": _fallback_answer(message, latest_run, reason="timeout"),
                "mode": "fallback_timeout",
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM-агент недоступен, используем fallback")
            return {
                "answer": _fallback_answer(message, latest_run, reason=f"error:{type(exc).__name__}"),
                "mode": f"fallback_after_error:{type(exc).__name__}",
            }

    return {
        "answer": _fallback_answer(message, latest_run, reason="no_api_key"),
        "mode": "fallback_no_api_key",
    }
