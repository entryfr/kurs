import json

from src.ai_agent_course import llm_client


def test_resolve_llm_config_returns_none_without_keys(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("CURSOR_BASE_URL", raising=False)

    cfg = llm_client.resolve_llm_config()
    assert cfg.provider == "none"
    assert cfg.api_key is None


def test_resolve_llm_config_prefers_anthropic(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    cfg = llm_client.resolve_llm_config(provider="anthropic")
    assert cfg.provider == "anthropic"
    assert cfg.api_key == "anthropic-key"


def test_resolve_llm_config_openai_compatible(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.local/v1")
    cfg = llm_client.resolve_llm_config(provider="openai_compatible")
    assert cfg.provider == "openai_compatible"
    assert cfg.base_url == "https://example.local/v1"


def test_resolve_llm_config_cursor_auto(monkeypatch) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "crsr_test_key")
    cfg = llm_client.resolve_llm_config(provider="auto")
    assert cfg.provider == "cursor"
    assert cfg.base_url == "https://api.cursor.com"


def test_resolve_llm_config_cursor_from_local_key(monkeypatch) -> None:
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    monkeypatch.setattr(llm_client.local_key, "CURSOR_API_KEY", "crsr_local_key", raising=False)
    cfg = llm_client.resolve_llm_config(provider="cursor")
    assert cfg.provider == "cursor"
    assert cfg.api_key == "crsr_local_key"


def test_build_engineering_answer_openai_compatible(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, payload: dict):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(req, timeout=0):
        _ = req
        _ = timeout
        return _FakeResponse(
            {"choices": [{"message": {"content": "LLM ответ по инженерному анализу"}}]}
        )

    monkeypatch.setattr(llm_client.url_request, "urlopen", _fake_urlopen)
    cfg = llm_client.LLMConfig(
        provider="openai_compatible",
        api_key="key",
        model="demo-model",
        base_url="https://example.local/v1",
        timeout_seconds=5.0,
    )
    answer, mode = llm_client.build_engineering_answer(
        prompt="Проанализируй участок",
        anomalies=[{"id": 1, "risk_class": "CRITICAL"}],
        validation={"issues_by_rule": {"R01": 1}},
        config=cfg,
    )
    assert mode == "llm_openai_compatible"
    assert "инженерному анализу" in answer.lower()


def test_check_llm_connectivity_none_provider() -> None:
    cfg = llm_client.LLMConfig(provider="none", api_key=None, model="none")
    status = llm_client.check_llm_connectivity(cfg)
    assert status["ok"] is False
    assert status["provider"] == "none"


def test_check_llm_connectivity_openai_compatible(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, payload: dict):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(req, timeout=0):
        _ = req
        _ = timeout
        return _FakeResponse({"choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr(llm_client.url_request, "urlopen", _fake_urlopen)
    cfg = llm_client.LLMConfig(
        provider="openai_compatible",
        api_key="key",
        model="demo-model",
        base_url="https://example.local/v1",
        timeout_seconds=5.0,
    )
    status = llm_client.check_llm_connectivity(cfg)
    assert status["ok"] is True
    assert status["provider"] == "openai_compatible"


def test_build_engineering_answer_cursor(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, payload: dict):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(req, timeout=0):
        _ = req
        _ = timeout
        return _FakeResponse({"choices": [{"message": {"content": "Cursor LLM OK"}}]})

    monkeypatch.setattr(llm_client.url_request, "urlopen", _fake_urlopen)
    cfg = llm_client.LLMConfig(
        provider="cursor",
        api_key="crsr_key",
        model="gpt-4o-mini",
        base_url="https://api.cursor.com",
        endpoint_path="/v1/chat/completions",
        auth_mode="bearer",
    )
    answer, mode = llm_client.build_engineering_answer(
        prompt="Проанализируй",
        anomalies=[],
        validation={"issues_by_rule": {}},
        config=cfg,
    )
    assert mode == "llm_cursor"
    assert "cursor" in answer.lower()
