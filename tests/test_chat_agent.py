import time

from src.agent import chat_agent


def test_ask_agent_empty_message_returns_validation() -> None:
    result = chat_agent.ask_agent("   ")
    assert result["mode"] == "validation"
    assert "пуст" in result["answer"].lower()


def test_ask_agent_fallback_when_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(chat_agent, "_cached_agent", None)

    result = chat_agent.ask_agent("Как запустить проект?")
    assert result["mode"] == "fallback_no_api_key"
    assert "ANTHROPIC_API_KEY" in result["answer"]
    assert "запуск" in result["answer"].lower() or "python web_app.py" in result["answer"]


def test_ask_agent_llm_mode_with_mock_agent(monkeypatch) -> None:
    class _MockAgent:
        def invoke(self, payload):
            assert "input" in payload
            return {"output": "Готово.\n\n\nОтвет."}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(chat_agent, "_cached_agent", None)
    monkeypatch.setattr(chat_agent, "_get_agent", lambda: _MockAgent())

    result = chat_agent.ask_agent("Дай краткий отчёт")
    assert result["mode"] == "llm"
    assert result["answer"] == "Готово.\n\nОтвет."


def test_ask_agent_timeout_switches_to_fallback(monkeypatch) -> None:
    class _SlowAgent:
        def invoke(self, payload):
            time.sleep(0.2)
            return {"output": "late"}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(chat_agent, "_cached_agent", None)
    monkeypatch.setattr(chat_agent, "_get_agent", lambda: _SlowAgent())
    monkeypatch.setattr(chat_agent, "_CALL_TIMEOUT_SECONDS", 0.01)

    result = chat_agent.ask_agent("Проверь таймаут")
    assert result["mode"] == "fallback_timeout"
    assert "таймаут" in result["answer"].lower()


def test_ask_agent_unknown_question_requests_clarification(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(chat_agent, "_cached_agent", None)

    result = chat_agent.ask_agent("Расскажи что-нибудь")
    assert result["mode"] == "fallback_no_api_key"
    assert "уточнение" in result["answer"].lower() or "пример" in result["answer"].lower()
