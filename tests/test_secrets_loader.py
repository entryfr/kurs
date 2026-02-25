from pathlib import Path
import os

from src.ai_agent_course.secrets_loader import load_local_env


def test_load_local_env_reads_key_values(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "LLM_PROVIDER=openai_compatible",
                "LLM_API_KEY='secret-key'",
                'LLM_BASE_URL="https://example.local/v1"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    loaded = load_local_env(env_file)
    assert loaded["LLM_PROVIDER"] == "openai_compatible"
    assert loaded["LLM_API_KEY"] == "secret-key"
    assert loaded["LLM_BASE_URL"] == "https://example.local/v1"


def test_load_local_env_does_not_override_by_default(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text("LLM_PROVIDER=openai_compatible\n", encoding="utf-8")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    load_local_env(env_file, override=False)
    assert os.getenv("LLM_PROVIDER") == "anthropic"

    load_local_env(env_file, override=True)
    assert os.getenv("LLM_PROVIDER") == "openai_compatible"

