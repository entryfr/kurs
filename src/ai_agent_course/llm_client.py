from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMConfig:
    provider: str
    api_key: str | None
    model: str
    base_url: str | None = None
    timeout_seconds: float = 25.0


def resolve_llm_config(
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMConfig:
    requested_provider = (provider or os.getenv("LLM_PROVIDER", "auto")).strip().lower()

    anthropic_key = (api_key or os.getenv("ANTHROPIC_API_KEY", "")).strip()
    if requested_provider in {"auto", "anthropic"} and anthropic_key:
        return LLMConfig(
            provider="anthropic",
            api_key=anthropic_key,
            model=(model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")).strip(),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SEC", "25")),
        )

    generic_key = (api_key or os.getenv("LLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or os.getenv("CURSOR_API_KEY", "")).strip()
    generic_base = (
        (base_url or os.getenv("LLM_BASE_URL", "") or os.getenv("OPENAI_BASE_URL", "") or os.getenv("CURSOR_BASE_URL", "")).strip()
    )
    if requested_provider in {"auto", "openai_compatible"} and generic_key and generic_base:
        return LLMConfig(
            provider="openai_compatible",
            api_key=generic_key,
            model=(model or os.getenv("LLM_MODEL", "gpt-4o-mini")).strip(),
            base_url=generic_base.rstrip("/"),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SEC", "25")),
        )

    return LLMConfig(provider="none", api_key=None, model="none", base_url=None)


def _call_openai_compatible(config: LLMConfig, prompt: str) -> str:
    assert config.base_url is not None
    assert config.api_key is not None
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 900,
    }
    body = json.dumps(payload).encode("utf-8")
    endpoint = f"{config.base_url}/chat/completions"

    req = url_request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    with url_request.urlopen(req, timeout=config.timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    decoded = json.loads(raw)
    choices = decoded.get("choices", [])
    if not choices:
        raise ValueError("Пустой ответ от openai-compatible endpoint")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        raise ValueError("Отсутствует content в ответе openai-compatible endpoint")
    return str(content).strip()


def build_engineering_answer(
    prompt: str,
    anomalies: list[dict[str, Any]],
    validation: dict[str, Any],
    config: LLMConfig,
) -> tuple[str, str]:
    if config.provider == "none":
        return "", "fallback_no_key"

    compact = anomalies[:8]
    prompt_text = (
        "Ты инженер-изыскатель. Дай краткий вывод по результатам автоматического анализа.\n"
        f"Запрос: {prompt}\n"
        f"Аномалии: {compact}\n"
        f"Нарушения: {validation.get('issues_by_rule', {})}\n"
        "Сформируй итог на русском: риски, рекомендации, что делать на площадке."
    )

    try:
        if config.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            assert config.api_key is not None
            llm = ChatAnthropic(
                model=config.model,
                anthropic_api_key=config.api_key,
                timeout=config.timeout_seconds,
                max_tokens=900,
                temperature=0.1,
            )
            response = llm.invoke(prompt_text)
            answer = str(getattr(response, "content", response)).strip()
            return answer, "llm_anthropic"

        if config.provider == "openai_compatible":
            answer = _call_openai_compatible(config, prompt_text)
            return answer, "llm_openai_compatible"

        return "", f"fallback_unknown_provider:{config.provider}"
    except (url_error.URLError, TimeoutError, ValueError, Exception) as exc:  # noqa: BLE001
        logger.exception("LLM вызов завершился ошибкой (%s)", config.provider)
        return "", f"fallback_error:{type(exc).__name__}"


def check_llm_connectivity(config: LLMConfig) -> dict[str, Any]:
    started = time.perf_counter()
    if config.provider == "none":
        return {
            "ok": False,
            "provider": "none",
            "model": "none",
            "message": "Ключ/провайдер не настроены. Агент будет работать в fallback-режиме.",
            "latency_ms": 0.0,
        }

    try:
        if config.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            assert config.api_key is not None
            llm = ChatAnthropic(
                model=config.model,
                anthropic_api_key=config.api_key,
                timeout=config.timeout_seconds,
                max_tokens=16,
                temperature=0.0,
            )
            response = llm.invoke("Ответь строго одним словом: OK")
            content = str(getattr(response, "content", response)).strip()
            ok = "ok" in content.lower()
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return {
                "ok": ok,
                "provider": config.provider,
                "model": config.model,
                "message": "Подключение к Anthropic проверено." if ok else f"Неожиданный ответ: {content}",
                "latency_ms": latency_ms,
            }

        if config.provider == "openai_compatible":
            content = _call_openai_compatible(config, "Ответь строго одним словом: OK")
            ok = "ok" in content.lower()
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return {
                "ok": ok,
                "provider": config.provider,
                "model": config.model,
                "base_url": config.base_url,
                "message": (
                    "Подключение к openai-compatible endpoint проверено."
                    if ok
                    else f"Неожиданный ответ: {content}"
                ),
                "latency_ms": latency_ms,
            }

        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return {
            "ok": False,
            "provider": config.provider,
            "model": config.model,
            "message": f"Неизвестный provider: {config.provider}",
            "latency_ms": latency_ms,
        }
    except (url_error.URLError, TimeoutError, ValueError, Exception) as exc:  # noqa: BLE001
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return {
            "ok": False,
            "provider": config.provider,
            "model": config.model,
            "base_url": config.base_url,
            "message": f"Ошибка подключения: {type(exc).__name__}: {exc}",
            "latency_ms": latency_ms,
        }

