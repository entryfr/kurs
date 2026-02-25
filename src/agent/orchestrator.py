from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _extract_text_from_response(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


class _DirectLangChainAgent:
    """Лёгкий адаптер над ChatAnthropic с единым .invoke()."""

    def __init__(self) -> None:
        from langchain_anthropic import ChatAnthropic

        model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        timeout_s = float(os.getenv("CHAT_AGENT_TIMEOUT_SEC", "20"))
        max_tokens = int(os.getenv("CHAT_AGENT_MAX_TOKENS", "1200"))
        temperature = float(os.getenv("CHAT_AGENT_TEMPERATURE", "0.1"))

        self._llm = ChatAnthropic(
            model=model_name,
            timeout=timeout_s,
            max_retries=1,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def invoke(self, payload: dict[str, Any] | str) -> dict[str, str]:
        prompt = payload.get("input") if isinstance(payload, dict) else str(payload)
        response = self._llm.invoke(prompt)
        return {"output": _extract_text_from_response(response)}


class _LocalFallbackAgent:
    """Локальная заглушка на случай недоступности LLM-провайдера."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def invoke(self, payload: dict[str, Any] | str) -> dict[str, str]:
        question = payload.get("input") if isinstance(payload, dict) else str(payload)
        answer = (
            "LLM-провайдер сейчас недоступен, включён локальный fallback.\n"
            f"Причина: {self._reason}\n"
            "Рекомендация: проверьте ANTHROPIC_API_KEY, модель и сетевой доступ.\n\n"
            f"Вопрос: {question[:500]}"
        )
        return {"output": answer}


def create_agent():
    """
    Создаёт устойчивого чат-агента на текущем стеке LangChain.
    Если провайдер недоступен — возвращает локальный fallback-агент.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return _LocalFallbackAgent("ANTHROPIC_API_KEY не задан")

    try:
        return _DirectLangChainAgent()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Не удалось создать LLM-агента, переключаемся на fallback")
        return _LocalFallbackAgent(type(exc).__name__)


def run_full_pipeline(*args, **kwargs) -> dict[str, Any]:
    """
    Совместимость со старым API.
    Для production-использования применяйте src.pipeline.service.run_pipeline.
    """
    logger.warning(
        "run_full_pipeline устарел. Используйте src.pipeline.service.run_pipeline. "
        "Args=%s, kwargs=%s",
        args,
        kwargs,
    )
    return {
        "status": "deprecated",
        "message": "Используйте src.pipeline.service.run_pipeline",
    }
